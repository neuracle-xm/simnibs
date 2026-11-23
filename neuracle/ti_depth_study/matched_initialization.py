"""在 16 的研究进程内替换初始种群的一个个体。"""

from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from inspect import getsource
from pathlib import Path
from time import perf_counter
from types import FunctionType
from typing import Any, Callable

import numpy as np

from neuracle.ti_depth_study import restart_budget
from neuracle.ti_depth_study.restart_budget import BudgetReached, FloatArray, Trace


GA_VERSION = "1.0.2"
GA_RUN_SHA256 = "de0b0f6c7d61d35934cd9a146571b93a70208a39758e06e5da9dfacc7219cfc3"
SEED_ATOL = 1e-14


def _bind(function: Callable[..., Any], name: str, replacement: Any) -> Callable[..., Any]:
    """在函数的私有名称表中替换一个依赖，保留原算法及共享模块。"""
    if name not in function.__code__.co_names:
        raise RuntimeError(f"已登记的初始化接点缺失: {name}")
    namespace = function.__globals__ | {name: replacement}
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def guarded_ga() -> type:
    """只允许已实测初始染色体保存顺序的 GA 1.0.2 实现。"""
    ga = restart_budget._require_genetic_algorithm()
    if version("geneticalgorithm") != GA_VERSION:
        raise RuntimeError(f"16 的单个体注入只验收 geneticalgorithm {GA_VERSION}")
    source = getsource(ga.run).replace("\r\n", "\n").encode()
    if sha256(source).hexdigest() != GA_RUN_SHA256:
        raise RuntimeError("GA 初始化源码散列发生变化，暂停互供试验")
    return ga


class _FirstRowRandom:
    """消耗原随机流并只替换第一组整数基因。"""

    def __init__(self, row: list[int]) -> None:
        """保存待替换的单个染色体及本地注入位置。"""
        self.row = row
        self.used = 0

    def __getattr__(self, name: str) -> Any:
        """将其他随机操作转发到 GA 原有的 NumPy 随机流。"""
        return getattr(np.random, name)

    def randint(self, *args: Any, **kwargs: Any) -> Any:
        """执行原有抽样，依次替换第一行内已核验边界的标量。"""
        sampled = np.random.randint(*args, **kwargs)
        if self.used >= len(self.row):
            return sampled
        if len(args) != 2 or kwargs or not isinstance(sampled, (int, np.integer)):
            raise RuntimeError("GA 第一个体的抽样接口发生变化")
        value = self.row[self.used]
        if not args[0] <= value < args[1]:
            raise ValueError("供体整数基因超出接收方边界")
        self.used += 1
        return value


class _LocalNumpy:
    """只对一份绑定后的 GA.run 暴露独立注入对象。"""

    def __init__(self, random: _FirstRowRandom) -> None:
        """保存本次作用域的随机接口。"""
        self.random = random

    def __getattr__(self, name: str) -> Any:
        """转发 NumPy 数组及数值运算。"""
        return getattr(np, name)


def seeded_ga(candidate: FloatArray) -> type:
    """为当前染色体生成一个保留原种群随机序列的 GA 类。"""
    base = guarded_ga()
    if candidate.shape != (6,) or not np.array_equal(candidate, np.rint(candidate)):
        raise ValueError("based 供体必须包含六个整数基因")
    row = [int(value) for value in candidate]

    class SeededGA(base):
        """在依赖内部原生写入种群的位置注入整行基因。"""

        def run(self) -> None:
            """执行原 GA，并核验注入恰好消费一个个体的整数抽样。"""
            if self.dim != len(row) or self.pop_s != restart_budget.BASED_INITIAL_REQUESTS or not np.all(self.var_type == "int"):
                raise RuntimeError("based 原生种群维度、类型或规模变化")
            random = _FirstRowRandom(row)
            bound = _bind(base.run, "np", _LocalNumpy(random))
            try:
                bound(self)
            finally:
                if random.used != len(row):
                    raise RuntimeError("based 初值未完整注入种群")

    return SeededGA


def with_free_seed(candidate: FloatArray) -> Callable[..., Any]:
    """使用 SciPy 自带 x0 替换差分进化初始种群的首行。"""
    if candidate.shape != (restart_budget.FREE_N_DIMENSIONS,) or not np.isfinite(candidate).all():
        raise ValueError("free 供体连续参数形状或值不合法")
    solver = restart_budget.differential_evolution

    def start(objective: Callable[..., Any], **options: Any) -> Any:
        """核对 15 的 LHS 入口，沿用原随机种子、规模及搜索设置。"""
        if ("x0" not in options or "seed" not in options or options.get("init", "latinhypercube") != "latinhypercube"
                or options["popsize"] != restart_budget.FREE_POPSIZE):
            raise RuntimeError("free 原生初始化接口或规模变化")
        return solver(objective, **(options | {"x0": candidate.copy()}))

    return start


@dataclass
class MatchedTrace(Trace):
    """逐项核对保护的种群成员，并复用 15 的有效评价计数。"""

    expected_first: FloatArray | None = None
    expected_rest: FloatArray | None = None
    first_atol: float = 0.0
    duration_s: float | None = None
    started: float = field(default_factory=perf_counter)
    duration_hit: bool = False

    def check_budget(self) -> None:
        """在独立重启达到附加时长后停机，并保留实际评价次数。"""
        if self.duration_s is not None and perf_counter() - self.started >= self.duration_s:
            self.duration_hit = True
            raise BudgetReached
        super().check_budget()

    def record(self, parameters: FloatArray, key: str | None, score: float, start: float) -> float:
        """记录原目标，同时校验注入的首行和未替换的初始行。"""
        index = len(self.rows)
        value = super().record(parameters, key, score, start)
        if self.expected_first is None or index >= self.minimum_requests:
            return value
        if index == 0:
            if not np.allclose(parameters, self.expected_first, rtol=0, atol=self.first_atol):
                raise RuntimeError("原生优化器实际首个体与登记供体不一致")
            if key is None:
                raise ValueError("供体注入后未通过接收路线的原生可行性检查")
        elif self.expected_rest is None or not np.array_equal(parameters, self.expected_rest[index - 1]):
            raise RuntimeError(f"第 {index + 1} 个初始个体与同种子随机臂不一致")
        return value


def search(
    prepared: dict[str, Any], trace: MatchedTrace, method: str, seed: int, output: Path,
) -> list[str]:
    """运行 15 原搜索器，通过每次函数的私有作用域接入研究初值。"""
    candidate = trace.expected_first
    if method == "leadfield_based":
        based = restart_budget.search_based
        if candidate is not None:
            ga = seeded_ga(candidate)
            based = _bind(based, "_require_genetic_algorithm", lambda: ga)
        return based(prepared, trace, seed)
    free = restart_budget.search_free
    if candidate is not None:
        free = _bind(free, "differential_evolution", with_free_seed(candidate))
    return free(prepared, trace, seed, output)
