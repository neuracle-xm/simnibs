"""
``geneticalgorithm`` 适配层。

该模块保持论文的四整数基因和 GA 参数，重复电极组合直接返回
大惩罚，合法染色体交给 ``LeadfieldFitnessEvaluator`` 枚举电流。
"""

import logging
from typing import Any

import numpy as np

from neuracle.ti_leadfield_optimization.fitness import LeadfieldFitnessEvaluator
from neuracle.ti_leadfield_optimization.models import (
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


class _GAObjective:
    """将 NumPy 染色体转换为 leadfield 适应度并处理重复电极惩罚。"""

    def __init__(
        self,
        evaluator: LeadfieldFitnessEvaluator,
        duplicate_penalty: float,
    ) -> None:
        """保存适应度计算器和论文重复电极惩罚基值。

        Parameters
        ----------
        evaluator : LeadfieldFitnessEvaluator
            负责合法染色体电流枚举的计算器。
        duplicate_penalty : float
            四电极不唯一时的惩罚基值。
        """
        self.evaluator = evaluator
        self.duplicate_penalty = duplicate_penalty

    def __call__(self, values: np.ndarray) -> float:
        """返回 ``geneticalgorithm`` 需要最小化的标量 objective。

        Parameters
        ----------
        values : numpy.ndarray
            GA 产生的四个 montage 整数索引。

        Returns
        -------
        float
            重复电极惩罚或合法染色体的负 score。
        """
        indices = tuple(int(value) for value in np.rint(values))
        unique_count = len(set(indices))
        if unique_count != 4:
            return self.duplicate_penalty + 100.0 * abs(unique_count - 4) ** 2
        chromosome = ElectrodeChromosome(indices)
        return self.evaluator.evaluate(chromosome).objective


def _require_genetic_algorithm() -> Any:
    """获取 GA 类，缺少 demo 依赖时给出明确安装指引。

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
    """使用论文参数运行四电极 ``geneticalgorithm`` 优化。

    Parameters
    ----------
    evaluator : LeadfieldFitnessEvaluator
        已绑定 leadfield、ROI 和合法电流的适应度计算器。
    settings : GASettings
        论文 GA 参数、随机种子和惩罚设置。

    Returns
    -------
    LeadfieldGAResult
        最优四电极、内部枚举得到的电流和收敛曲线。

    Raises
    ------
    RuntimeError
        GA 返回非法染色体或无法在缓存中找到最优指标时抛出。
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
    objective = _GAObjective(evaluator, settings.duplicate_electrode_penalty)
    random_state = np.random.get_state()
    np.random.seed(settings.random_seed)
    try:
        optimizer = genetic_algorithm(
            function=objective,
            dimension=4,
            variable_type="int",
            variable_boundaries=np.array([[0, electrode_count - 1]] * 4),
            algorithm_parameters=algorithm_parameters,
            function_timeout=settings.function_timeout_seconds,
            convergence_curve=False,
            progress_bar=True,
        )
        optimizer.run()
    finally:
        np.random.set_state(random_state)
    output = optimizer.output_dict
    indices = tuple(int(value) for value in np.rint(output["variable"]))
    if len(indices) != 4 or len(set(indices)) != 4:
        raise RuntimeError(f"GA 返回非法四电极染色体: {indices}")
    chromosome = ElectrodeChromosome(indices)
    metrics = evaluator.evaluation_cache.get(indices)
    if metrics is None:
        metrics = evaluator.evaluate(chromosome)
    names = evaluator.electrode_names(chromosome)
    convergence = tuple(float(value) for value in optimizer.report)
    logger.info(
        "GA 优化完成: electrodes=%s, currents=(%s, %s), score=%s",
        names,
        metrics.currents.current_a_ma,
        metrics.currents.current_b_ma,
        metrics.score,
    )
    return LeadfieldGAResult(
        chromosome=chromosome,
        electrode_names=names,
        metrics=metrics,
        convergence=convergence,
        random_seed=settings.random_seed,
    )
