"""
Leadfield-based TI 电场重建、电流枚举和适应度计算。

论文的 GA 染色体只包含四个电极索引。本模块对每个染色体枚举
21 组合法电流，重建两路电场并计算 ROI/Rest 体积加权指标。
"""

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from neuracle.ti_leadfield_optimization.models import (
    CurrentPair,
    ElectrodeChromosome,
    FitnessMetrics,
    LeadfieldData,
    RegionMasks,
)
from simnibs.utils import TI_utils as TI

logger = logging.getLogger(__name__)


def generate_current_pairs(
    minimum_ma: float = 0.5,
    maximum_ma: float = 1.5,
    step_ma: float = 0.05,
    total_ma: float = 2.0,
) -> tuple[CurrentPair, ...]:
    """生成论文约束下的全部离散两路电流组合。

    Parameters
    ----------
    minimum_ma : float, optional
        单路电流下限，单位 mA。
    maximum_ma : float, optional
        单路电流上限，单位 mA。
    step_ma : float, optional
        离散电流步长，单位 mA。
    total_ma : float, optional
        两路电流幅值之和，单位 mA。

    Returns
    -------
    tuple[CurrentPair, ...]
        所有满足范围、步长和总和约束的电流对。

    Raises
    ------
    ValueError
        参数不能由指定步长精确表示，或没有合法组合时抛出。
    """
    if step_ma <= 0 or minimum_ma <= 0 or maximum_ma < minimum_ma:
        raise ValueError("电流范围和步长必须为有效正数")
    minimum_tick = round(minimum_ma / step_ma)
    maximum_tick = round(maximum_ma / step_ma)
    total_tick = round(total_ma / step_ma)
    values = (
        (minimum_ma, minimum_tick),
        (maximum_ma, maximum_tick),
        (total_ma, total_tick),
    )
    if any(not np.isclose(value, tick * step_ma) for value, tick in values):
        raise ValueError("电流上下限和总和必须能被步长精确表示")
    pairs = tuple(
        CurrentPair(
            current_a_ma=round(current_a_tick * step_ma, 10),
            current_b_ma=round((total_tick - current_a_tick) * step_ma, 10),
        )
        for current_a_tick in range(minimum_tick, maximum_tick + 1)
        if minimum_tick <= total_tick - current_a_tick <= maximum_tick
    )
    if not pairs:
        raise ValueError("电流约束下没有合法组合")
    return pairs


def calculate_region_metrics(
    max_ti: npt.NDArray[np.float64],
    region_masks: RegionMasks,
) -> tuple[float, float, float, float]:
    """计算 max_TI 在 ROI 和 Rest 中的体积加权指标。

    Parameters
    ----------
    max_ti : numpy.ndarray
        与 mesh element 顺序对齐的 max_TI，单位 V/m。
    region_masks : RegionMasks
        ROI、Rest mask 和 element 体积。

    Returns
    -------
    tuple[float, float, float, float]
        ``(roi_mean, rest_mean, roi_rest_ratio, roi_max)``。

    Raises
    ------
    ValueError
        max_TI 形状不匹配或包含非有限值时抛出。
    """
    values = np.asarray(max_ti, dtype=np.float64)
    if values.shape != region_masks.roi_mask.shape:
        raise ValueError(
            f"max_TI 形状与 ROI mask 不一致: {values.shape} != {region_masks.roi_mask.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("max_TI 包含 NaN 或 Inf")
    roi_mean = float(
        np.sum(
            values[region_masks.roi_mask]
            * region_masks.element_volumes[region_masks.roi_mask]
        )
        / region_masks.roi_volume
    )
    rest_mean = float(
        np.sum(
            values[region_masks.rest_mask]
            * region_masks.element_volumes[region_masks.rest_mask]
        )
        / region_masks.rest_volume
    )
    denominator = max(rest_mean, np.finfo(np.float64).eps)
    if rest_mean <= np.finfo(np.float64).eps:
        logger.warning("Rest 平均 max_TI 接近 0，使用 float64 epsilon 计算 ratio")
    ratio = roi_mean / denominator
    roi_max = float(np.max(values[region_masks.roi_mask]))
    return roi_mean, rest_mean, ratio, roi_max


class LeadfieldFitnessEvaluator:
    """对四电极染色体重建 TI 电场并完成论文电流枚举。"""

    def __init__(
        self,
        leadfield: LeadfieldData,
        region_masks: RegionMasks,
        current_pairs: Sequence[CurrentPair],
        target_threshold_v_per_m: float,
    ) -> None:
        """初始化适应度计算器和只保存标量指标的缓存。

        Parameters
        ----------
        leadfield : LeadfieldData
            已全量加载的单位电流电场。
        region_masks : RegionMasks
            与 leadfield element 顺序对齐的 ROI/Rest 数据。
        current_pairs : sequence of CurrentPair
            每个染色体内部要枚举的合法电流。
        target_threshold_v_per_m : float
            ROI 内最大 max_TI 阈值，单位 V/m。
        """
        if leadfield.values.shape[1] != region_masks.roi_mask.size:
            raise ValueError("leadfield element 数与 ROI/Rest mask 不一致")
        if not current_pairs:
            raise ValueError("电流候选列表不能为空")
        if target_threshold_v_per_m < 0:
            raise ValueError("ROI max_TI 阈值不能为负数")
        self.leadfield = leadfield
        self.region_masks = region_masks
        self.current_pairs = tuple(current_pairs)
        self.target_threshold_v_per_m = target_threshold_v_per_m
        self.evaluation_cache: dict[tuple[int, int, int, int], FitnessMetrics] = {}

    def _unit_pair_field(
        self, positive_name: str, negative_name: str
    ) -> npt.NDArray[np.float64]:
        """根据参考电极语义计算单位 1 A 电极对的电场。

        Parameters
        ----------
        positive_name : str
            注入正电流的 montage 电极名称。
        negative_name : str
            注入负电流的 montage 电极名称。

        Returns
        -------
        numpy.ndarray
            形状为 ``(N_elements, 3)`` 的单位电流电场。
        """
        positive_row = self.leadfield.electrode_rows[positive_name]
        negative_row = self.leadfield.electrode_rows[negative_name]
        if positive_row is None:
            return -self.leadfield.values[negative_row]
        if negative_row is None:
            return self.leadfield.values[positive_row]
        return self.leadfield.values[positive_row] - self.leadfield.values[negative_row]

    def _evaluate_current(
        self,
        geometry: tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        currents: CurrentPair,
    ) -> FitnessMetrics:
        """对固定电极和单组电流计算 max_TI 与 ROI/Rest 指标。

        Parameters
        ----------
        geometry : tuple[numpy.ndarray, ...]
            两路单位电场的模、绝对点积和叉积模，当前染色体只计算一次。
        currents : CurrentPair
            需要评估的两路电流幅值。

        Returns
        -------
        FitnessMetrics
            未施加跨电流组惩罚的单组电流指标。
        """
        max_ti = self._scaled_max_ti(geometry, currents)
        roi_mean, rest_mean, ratio, roi_max = calculate_region_metrics(
            max_ti,
            self.region_masks,
        )
        raw_score = 10000.0 * ratio
        threshold_satisfied = roi_max >= self.target_threshold_v_per_m
        return FitnessMetrics(
            objective=-round(raw_score, 2),
            score=round(raw_score, 2),
            roi_rest_ratio=ratio,
            roi_mean_v_per_m=roi_mean,
            rest_mean_v_per_m=rest_mean,
            roi_max_v_per_m=roi_max,
            penalty=0.0,
            currents=currents,
            threshold_satisfied=threshold_satisfied,
        )

    def _pair_geometry(
        self,
        unit_field_a: npt.NDArray[np.float64],
        unit_field_b: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """预计算 21 组电流共用的两路电场几何量。

        Parameters
        ----------
        unit_field_a : numpy.ndarray
            A 电极对在 1 A 下的电场。
        unit_field_b : numpy.ndarray
            B 电极对在 1 A 下的电场。

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(|D_A|, |D_B|, |D_A·D_B|, |D_A×D_B|)`` element 数组。

        Notes
        -----
        max_TI 对正电流缩放具有解析形式，因此枚举电流时无需重复创建两个
        ``(N_elements, 3)`` 电场并计算点积和叉积。
        """
        norm_a = np.linalg.norm(unit_field_a, axis=1)
        norm_b = np.linalg.norm(unit_field_b, axis=1)
        absolute_dot = np.abs(np.einsum("ij,ij->i", unit_field_a, unit_field_b))
        cross_norm = np.linalg.norm(np.cross(unit_field_a, unit_field_b), axis=1)
        return norm_a, norm_b, absolute_dot, cross_norm

    def _scaled_max_ti(
        self,
        geometry: tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        currents: CurrentPair,
    ) -> npt.NDArray[np.float64]:
        """使用预计算几何量快速计算某组电流的 max_TI。

        Parameters
        ----------
        geometry : tuple[numpy.ndarray, ...]
            ``_pair_geometry`` 返回的单位电场几何量。
        currents : CurrentPair
            两路正电流幅值，单位 mA。

        Returns
        -------
        numpy.ndarray
            与 ``TI_utils.get_maxTI`` 等价的 element-wise max_TI。
        """
        norm_a, norm_b, absolute_dot, cross_norm = geometry
        scale_a = currents.current_a_ma / 1000.0
        scale_b = currents.current_b_ma / 1000.0
        scaled_norm_a = scale_a * norm_a
        scaled_norm_b = scale_b * norm_b
        strong_norm = np.maximum(scaled_norm_a, scaled_norm_b)
        weak_norm = np.minimum(scaled_norm_a, scaled_norm_b)
        norm_product = norm_a * norm_b
        cosine = np.divide(
            absolute_dot,
            norm_product,
            out=np.zeros_like(absolute_dot),
            where=norm_product > np.finfo(np.float64).eps,
        )
        denominator_squared = (
            scaled_norm_a**2 + scaled_norm_b**2 - 2.0 * scale_a * scale_b * absolute_dot
        )
        denominator = np.sqrt(np.maximum(denominator_squared, 0.0))
        general_case = np.divide(
            2.0 * scale_a * scale_b * cross_norm,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > np.finfo(np.float64).eps,
        )
        max_ti = np.where(
            weak_norm <= strong_norm * cosine,
            2.0 * weak_norm,
            general_case,
        )
        return np.nan_to_num(max_ti, nan=0.0, posinf=0.0, neginf=0.0)

    def evaluate_current_pair(
        self,
        chromosome: ElectrodeChromosome,
        currents: CurrentPair,
    ) -> FitnessMetrics:
        """评估指定四电极和指定两路电流，用于 baseline 对照。

        Parameters
        ----------
        chromosome : ElectrodeChromosome
            A+、A-、B+、B- 四个 montage 索引。
        currents : CurrentPair
            两路固定电流幅值。

        Returns
        -------
        FitnessMetrics
            Baseline 的 ROI/Rest 指标。
        """
        names = self.electrode_names(chromosome)
        unit_field_a = self._unit_pair_field(names[0], names[1])
        unit_field_b = self._unit_pair_field(names[2], names[3])
        geometry = self._pair_geometry(unit_field_a, unit_field_b)
        return self._evaluate_current(geometry, currents)

    def evaluate(self, chromosome: ElectrodeChromosome) -> FitnessMetrics:
        """枚举当前染色体的全部合法电流并返回论文适应度。

        Parameters
        ----------
        chromosome : ElectrodeChromosome
            四个互不重复的 montage 电极索引。

        Returns
        -------
        FitnessMetrics
            达标电流中 ratio 最高的结果；全部未达标时按论文施加惩罚。
        """
        key = chromosome.electrode_indices
        if key in self.evaluation_cache:
            return self.evaluation_cache[key]
        if len(set(key)) != 4:
            raise ValueError("适应度计算只接受四个互不重复的电极")
        names = self.electrode_names(chromosome)
        unit_field_a = self._unit_pair_field(names[0], names[1])
        unit_field_b = self._unit_pair_field(names[2], names[3])
        geometry = self._pair_geometry(unit_field_a, unit_field_b)
        candidates = [
            self._evaluate_current(geometry, currents)
            for currents in self.current_pairs
        ]
        feasible = [
            candidate for candidate in candidates if candidate.threshold_satisfied
        ]
        if feasible:
            selected = max(feasible, key=lambda candidate: candidate.score)
        else:
            base = min(candidates, key=lambda candidate: candidate.score)
            mean_roi_max = float(
                np.mean([candidate.roi_max_v_per_m for candidate in candidates])
            )
            penalty = (
                100.0 * (self.target_threshold_v_per_m - mean_roi_max) ** 2 + 1000.0
            )
            score = round(base.score - penalty, 2)
            selected = FitnessMetrics(
                objective=-score,
                score=score,
                roi_rest_ratio=base.roi_rest_ratio,
                roi_mean_v_per_m=base.roi_mean_v_per_m,
                rest_mean_v_per_m=base.rest_mean_v_per_m,
                roi_max_v_per_m=base.roi_max_v_per_m,
                penalty=penalty,
                currents=base.currents,
                threshold_satisfied=False,
            )
        self.evaluation_cache[key] = selected
        return selected

    def electrode_names(
        self, chromosome: ElectrodeChromosome
    ) -> tuple[str, str, str, str]:
        """将四个染色体索引转换为 montage 电极名称。

        Parameters
        ----------
        chromosome : ElectrodeChromosome
            需要解码的四个电极索引。

        Returns
        -------
        tuple[str, str, str, str]
            A+、A-、B+、B- 电极名称。

        Raises
        ------
        ValueError
            染色体长度不是 4 或索引越界时抛出。
        """
        if len(chromosome.electrode_indices) != 4:
            raise ValueError("TI 电极染色体必须包含四个索引")
        electrode_count = len(self.leadfield.electrode_names)
        if any(
            index < 0 or index >= electrode_count
            for index in chromosome.electrode_indices
        ):
            raise ValueError("染色体包含超出 montage 范围的电极索引")
        return tuple(
            self.leadfield.electrode_names[index]
            for index in chromosome.electrode_indices
        )

    def reconstruct_fields(
        self,
        chromosome: ElectrodeChromosome,
        currents: CurrentPair,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """为最终方案重建两路完整电场和 max_TI 供结果导出。

        Parameters
        ----------
        chromosome : ElectrodeChromosome
            最终四电极索引。
        currents : CurrentPair
            最终两路电流幅值。

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(E_A, E_B, max_TI)``，电场单位为 V/m。
        """
        names = self.electrode_names(chromosome)
        field_a = (currents.current_a_ma / 1000.0) * self._unit_pair_field(
            names[0], names[1]
        )
        field_b = (currents.current_b_ma / 1000.0) * self._unit_pair_field(
            names[2], names[3]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            max_ti = TI.get_maxTI(field_a, field_b)
        max_ti = np.nan_to_num(max_ti, nan=0.0, posinf=0.0, neginf=0.0)
        return field_a, field_b, max_ti
