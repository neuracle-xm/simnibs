"""
``geneticalgorithm`` 六基因适配层。

染色体同时搜索四个固定 montage 电极和两路独立电流 tick，避免在每个电极候选
内部枚举电流笛卡尔积。等价的电极方向和通道交换使用同一个 objective 缓存键。
"""

import logging
from typing import Any

import numpy as np

from neuracle.ti_leadfield_optimization.fitness import LeadfieldFitnessEvaluator
from neuracle.ti_leadfield_optimization.models import (
    CurrentPair,
    ElectrodeChromosome,
    GASettings,
    LeadfieldGAResult,
)

try:
    from geneticalgorithm import geneticalgorithm as ga
except ImportError as import_error:
    ga = None
    GA_IMPORT_ERROR: ImportError | None = import_error
else:
    GA_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _current_tick_bounds(settings: GASettings) -> tuple[int, int]:
    """将两路独立电流范围转换为 GA 整数 tick 边界。

    Parameters
    ----------
    settings : GASettings
        电流上下限和离散步长。

    Returns
    -------
    tuple[int, int]
        闭区间 ``(minimum_tick, maximum_tick)``。

    Raises
    ------
    ValueError
        电流范围非法或不能由步长精确表示时抛出。
    """
    if settings.current_step_ma <= 0 or settings.current_min_ma < 0:
        raise ValueError("电流下限不能为负数，步长必须为正数")
    if settings.current_max_ma < settings.current_min_ma:
        raise ValueError("电流上限不能低于下限")
    minimum_tick = round(settings.current_min_ma / settings.current_step_ma)
    maximum_tick = round(settings.current_max_ma / settings.current_step_ma)
    if not np.isclose(settings.current_min_ma, minimum_tick * settings.current_step_ma):
        raise ValueError("电流下限必须能被步长精确表示")
    if not np.isclose(settings.current_max_ma, maximum_tick * settings.current_step_ma):
        raise ValueError("电流上限必须能被步长精确表示")
    return minimum_tick, maximum_tick


def _decode_candidate(
    values: np.ndarray,
    settings: GASettings,
) -> tuple[ElectrodeChromosome, CurrentPair]:
    """将六个整数基因解码为四电极和两路电流。

    Parameters
    ----------
    values : numpy.ndarray
        ``(A+, A-, B+, B-, current_A_tick, current_B_tick)``。
    settings : GASettings
        电流步长配置。

    Returns
    -------
    tuple[ElectrodeChromosome, CurrentPair]
        解码后的四电极染色体和独立电流。

    Raises
    ------
    ValueError
        基因数量不是六个时抛出。
    """
    genes = tuple(int(value) for value in np.rint(values))
    if len(genes) != 6:
        raise ValueError(f"GA 候选必须包含六个整数基因，实际 {len(genes)}")
    chromosome = ElectrodeChromosome(genes[:4])
    currents = CurrentPair(
        current_a_ma=round(genes[4] * settings.current_step_ma, 10),
        current_b_ma=round(genes[5] * settings.current_step_ma, 10),
    )
    return chromosome, currents


class _GAObjective:
    """解码六基因候选、处理重复电极并缓存等价 focality。"""

    def __init__(
        self,
        evaluator: LeadfieldFitnessEvaluator,
        settings: GASettings,
    ) -> None:
        """保存适应度计算器、电流步长和 objective 缓存。

        Parameters
        ----------
        evaluator : LeadfieldFitnessEvaluator
            负责 SimNIBS focality 计算的 evaluator。
        settings : GASettings
            电流步长和重复电极惩罚配置。
        """
        self.evaluator = evaluator
        self.settings = settings
        self.objective_cache: dict[tuple[int, int, int, int, int, int], float] = {}
        self.cache_hits = 0

    @staticmethod
    def _canonical_key(genes: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
        """把方向交换和 A/B 通道交换的等价解归一到同一缓存键。

        Parameters
        ----------
        genes : tuple[int, ...]
            六个整数基因。

        Returns
        -------
        tuple[int, int, int, int, int, int]
            规范化后的两组 ``(低索引, 高索引, 电流 tick)``。

        Notes
        -----
        max_TI 对单个电极对的正负方向反转不变；同时交换 A/B 电场及对应电流也
        不改变结果，因此这些候选可以安全复用 objective。
        """
        channel_a = (min(genes[0], genes[1]), max(genes[0], genes[1]), genes[4])
        channel_b = (min(genes[2], genes[3]), max(genes[2], genes[3]), genes[5])
        first, second = sorted((channel_a, channel_b))
        return first[0], first[1], second[0], second[1], first[2], second[2]

    def __call__(self, values: np.ndarray) -> float:
        """返回 ``geneticalgorithm`` 需要最小化的 focality objective。

        Parameters
        ----------
        values : numpy.ndarray
            GA 产生的六个整数基因。

        Returns
        -------
        float
            重复电极惩罚或 SimNIBS focality objective。
        """
        genes = tuple(int(value) for value in np.rint(values))
        electrode_indices = genes[:4]
        unique_count = len(set(electrode_indices))
        if unique_count != 4:
            return (
                self.settings.duplicate_electrode_penalty
                + 100.0 * abs(unique_count - 4) ** 2
            )
        key = self._canonical_key(genes)
        cached = self.objective_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        chromosome, currents = _decode_candidate(np.asarray(genes), self.settings)
        objective = self.evaluator.evaluate_objective(chromosome, currents)
        self.objective_cache[key] = objective
        return objective


def _require_genetic_algorithm() -> Any:
    """获取 GA 类，缺少依赖时给出明确安装指引。

    Returns
    -------
    Any
        ``geneticalgorithm.geneticalgorithm`` 类。

    Raises
    ------
    RuntimeError
        当 ``geneticalgorithm==1.0.2`` 未安装时抛出。
    """
    if ga is None:
        raise RuntimeError(
            "缺少 demo 依赖 geneticalgorithm==1.0.2，请先在 simnibs_env 中安装"
        ) from GA_IMPORT_ERROR
    return ga


def run_genetic_optimization(
    evaluator: LeadfieldFitnessEvaluator,
    settings: GASettings,
) -> LeadfieldGAResult:
    """用六基因 GA 同时优化四个电极和两路独立电流。

    Parameters
    ----------
    evaluator : LeadfieldFitnessEvaluator
        已绑定 leadfield、ROI 和 SimNIBS focality 阈值的 evaluator。
    settings : GASettings
        GA 参数、独立电流范围、步长和随机种子。

    Returns
    -------
    LeadfieldGAResult
        最优四电极、两路电流、完整 focality 指标和收敛曲线。

    Raises
    ------
    ValueError
        GA、电流或 montage 参数非法时抛出。
    RuntimeError
        GA 返回非法六基因候选时抛出。
    """
    genetic_algorithm = _require_genetic_algorithm()
    electrode_count = len(evaluator.leadfield.electrode_names)
    if electrode_count < 4:
        raise ValueError("固定 montage 至少需要四个唯一电极")
    if settings.max_num_iteration < 1 or settings.population_size < 4:
        raise ValueError("GA 迭代数必须大于 0，种群数必须至少为 4")
    probabilities = (
        settings.mutation_probability,
        settings.elit_ratio,
        settings.crossover_probability,
        settings.parents_portion,
    )
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("GA 概率和比例参数必须在 [0, 1] 范围内")
    if settings.function_timeout_seconds <= 0:
        raise ValueError("GA function_timeout_seconds 必须大于 0")
    minimum_tick, maximum_tick = _current_tick_bounds(settings)
    algorithm_parameters = {
        "max_num_iteration": settings.max_num_iteration,
        "population_size": settings.population_size,
        "mutation_probability": settings.mutation_probability,
        "elit_ratio": settings.elit_ratio,
        "crossover_probability": settings.crossover_probability,
        "parents_portion": settings.parents_portion,
        "crossover_type": settings.crossover_type,
        "max_iteration_without_improv": settings.max_iteration_without_improv,
    }
    objective = _GAObjective(evaluator, settings)
    boundaries = np.array(
        [[0, electrode_count - 1]] * 4
        + [[minimum_tick, maximum_tick], [minimum_tick, maximum_tick]],
        dtype=np.int64,
    )
    random_state = np.random.get_state()
    np.random.seed(settings.random_seed)
    try:
        optimizer = genetic_algorithm(
            function=objective,
            dimension=6,
            variable_type="int",
            variable_boundaries=boundaries,
            algorithm_parameters=algorithm_parameters,
            function_timeout=settings.function_timeout_seconds,
            convergence_curve=False,
            progress_bar=True,
        )
        optimizer.run()
    finally:
        np.random.set_state(random_state)
    chromosome, currents = _decode_candidate(
        optimizer.output_dict["variable"], settings
    )
    indices = chromosome.electrode_indices
    if len(indices) != 4 or len(set(indices)) != 4:
        raise RuntimeError(f"GA 返回非法四电极候选: {indices}")
    minimum_current = minimum_tick * settings.current_step_ma
    maximum_current = maximum_tick * settings.current_step_ma
    if any(
        current < minimum_current or current > maximum_current
        for current in (currents.current_a_ma, currents.current_b_ma)
    ):
        raise RuntimeError(f"GA 返回超出范围的电流: {currents}")
    metrics = evaluator.evaluate_current_pair(chromosome, currents)
    names = evaluator.electrode_names(chromosome)
    convergence = tuple(float(value) for value in optimizer.report)
    logger.info(
        "GA 优化完成: electrodes=%s, currents=(%s, %s), focality_score=%s, "
        "unique_evaluations=%s, cache_hits=%s",
        names,
        currents.current_a_ma,
        currents.current_b_ma,
        metrics.score,
        len(objective.objective_cache),
        objective.cache_hits,
    )
    return LeadfieldGAResult(
        chromosome=chromosome,
        electrode_names=names,
        metrics=metrics,
        convergence=convergence,
        random_seed=settings.random_seed,
    )
