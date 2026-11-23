"""方案 05 的覆盖、软覆盖与两类外溢约束。"""

import numpy as np

from neuracle.ti_depth_study.metrics import BoolArray, FloatArray


def coverage_terms(
    field: FloatArray, volume: FloatArray, target: BoolArray, cortex: BoolArray,
    target_threshold: float = 0.10, outside_threshold: float = 0.10,
) -> dict[str, float]:
    """按体积计算硬覆盖、软覆盖及靶外和皮层超阈绝对量。

    Parameters
    ----------
    field, volume : ndarray
        同序 TI 场强（V/m）和逐四面体体积（mm³）。
    target, cortex : ndarray of bool
        全脑掩膜内的靶与皮层；靶外皮层由差集计算。
    target_threshold, outside_threshold : float, optional
        靶内和靶外的阈值，单位 V/m。

    Returns
    -------
    dict
        K、F、S 和两类外溢体积。场无效或区域为空时抛出异常。
    """
    if not target.any() or not (~target).any() or not np.isfinite(field).all() or not np.any(field > 0):
        raise ValueError("覆盖损失需要有限非零场、非空靶区及靶外区")
    roi = field[target]
    weights = volume[target]
    hard = float(np.average(roi >= target_threshold, weights=weights))
    soft = float(np.average(np.minimum(roi / target_threshold, 1.0), weights=weights))
    outside = ~target & (field >= outside_threshold)
    return dict(K=hard, F=soft, S=(hard + soft) / 2, W_R_mm3=float(volume[outside].sum()), W_C_mm3=float(volume[outside & cortex].sum()))


def spill_loss(terms: dict[str, float], budget: tuple[float, float], scale: float) -> dict[str, float | bool]:
    """按严格体积预算将可行和不可行布局映射到不同损失区间。

    Parameters
    ----------
    terms : dict
        当前场的 S、W_R_mm3、W_C_mm3。
    budget : tuple of float
        同一计算层、同一靶的 ROC0 全脑靶外与靶外皮层体积。
    scale : float
        两类预算共同乘数，主试验为 1，收紧条件为 0.75。

    Returns
    -------
    dict
        可行性、超额体积及用于排序的有界标量损失。
    """
    outside, cortical = budget
    excess_outside = max(0.0, terms["W_R_mm3"] - scale * outside)
    excess_cortex = max(0.0, terms["W_C_mm3"] - scale * cortical)
    violation = excess_outside / max(outside, 1.0) + excess_cortex / max(cortical, 1.0)
    feasible = excess_outside == excess_cortex == 0.0
    loss = 0.9 * (1 - terms["S"]) if feasible else 1 + 0.5 * violation / (1 + violation)
    return dict(feasible=feasible, violation=violation, returned_loss=loss,
                excess_R_mm3=excess_outside, excess_C_mm3=excess_cortex)


def pareto_mask(terms: list[dict[str, float]]) -> BoolArray:
    """标记覆盖最大化且两类外溢最小化的非支配候选。

    Parameters
    ----------
    terms : list of dict
        同个体、靶和计算层的 K、W_R_mm3、W_C_mm3。

    Returns
    -------
    ndarray of bool
        每个候选是否处于非支配边界，等指标候选均保留。
    """
    values = np.array([(-row["K"], row["W_R_mm3"], row["W_C_mm3"]) for row in terms])
    return np.array([not np.any(np.all(values <= item, axis=1) & np.any(values < item, axis=1)) for item in values], dtype=bool)
