"""
Leadfield-based TI 电场重建和 SimNIBS focality 适应度计算。

GA 染色体同时包含四个电极索引和两路独立电流。适应度直接使用
SimNIBS TES 逆向优化的 ROC focality，ROI/Rest 均值只作为结果诊断。
"""

import logging

import numpy as np
import numpy.typing as npt

from neuracle.ti_leadfield_optimization.models import (
    CurrentPair,
    ElectrodeChromosome,
    FitnessMetrics,
    LeadfieldData,
    RegionMasks,
)
from simnibs.optimization.tes_flex_optimization.measures import ROC
from simnibs.utils import TI_utils as TI

logger = logging.getLogger(__name__)


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


def calculate_focality_objective(
    max_ti: npt.NDArray[np.float64],
    region_masks: RegionMasks,
    non_roi_threshold_v_per_m: float,
    roi_threshold_v_per_m: float,
) -> tuple[float, float, float]:
    """按 SimNIBS TES 逆向优化定义计算最小化目标。

    Parameters
    ----------
    max_ti : numpy.ndarray
        与 mesh element 顺序对齐的 max_TI，单位 V/m。
    region_masks : RegionMasks
        ROI 和非 ROI element mask。
    non_roi_threshold_v_per_m : float
        非 ROI 不应超过的场强阈值，单位 V/m。
    roi_threshold_v_per_m : float
        ROI 应达到的场强阈值，单位 V/m。

    Returns
    -------
    tuple[float, float, float]
        ``(objective, score, roc_distance)``。objective 与 SimNIBS
        ``goal="focality"`` 完全一致。

    Raises
    ------
    ValueError
        输入形状、数值或阈值不合法时抛出。

    """
    values = np.asarray(max_ti, dtype=np.float64)
    if values.shape != region_masks.roi_mask.shape:
        raise ValueError(
            f"max_TI 形状与 ROI mask 不一致: {values.shape} != {region_masks.roi_mask.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("max_TI 包含 NaN 或 Inf")
    if non_roi_threshold_v_per_m < 0 or roi_threshold_v_per_m < 0:
        raise ValueError("focality 阈值不能为负数")
    if non_roi_threshold_v_per_m > roi_threshold_v_per_m:
        raise ValueError("非 ROI 阈值不能高于 ROI 阈值")
    roi_values = values[region_masks.roi_mask]
    non_roi_values = values[region_masks.rest_mask]
    thresholds = [non_roi_threshold_v_per_m, roi_threshold_v_per_m]
    roc_distance = float(
        ROC(
            e1=roi_values,
            e2=non_roi_values,
            threshold=thresholds,
            focal=True,
        )
    )
    score = 100.0 * (np.sqrt(2.0) - roc_distance)
    return -score, score, roc_distance


def calculate_focality_metrics(
    max_ti: npt.NDArray[np.float64],
    region_masks: RegionMasks,
    non_roi_threshold_v_per_m: float,
    roi_threshold_v_per_m: float,
) -> tuple[float, float, float, float, float]:
    """计算 SimNIBS focality 目标及 sensitivity/false-positive 诊断。

    Parameters
    ----------
    max_ti : numpy.ndarray
        与 mesh element 顺序对齐的 max_TI，单位 V/m。
    region_masks : RegionMasks
        ROI 和非 ROI element mask。
    non_roi_threshold_v_per_m : float
        非 ROI 不应超过的场强阈值，单位 V/m。
    roi_threshold_v_per_m : float
        ROI 应达到的场强阈值，单位 V/m。

    Returns
    -------
    tuple[float, float, float, float, float]
        ``(objective, score, roc_distance, roi_sensitivity,
        non_roi_false_positive_rate)``。

    Notes
    -----
    SimNIBS focality 对 element 计数，不使用 element 体积加权。体积加权
    ROI/Rest mean 由 ``calculate_region_metrics`` 单独提供，仅用于结果诊断。
    """
    objective, score, roc_distance = calculate_focality_objective(
        max_ti,
        region_masks,
        non_roi_threshold_v_per_m,
        roi_threshold_v_per_m,
    )
    values = np.asarray(max_ti, dtype=np.float64)
    roi_sensitivity = float(
        np.mean(values[region_masks.roi_mask] >= roi_threshold_v_per_m)
    )
    non_roi_false_positive_rate = float(
        np.mean(values[region_masks.rest_mask] >= non_roi_threshold_v_per_m)
    )
    return objective, score, roc_distance, roi_sensitivity, non_roi_false_positive_rate


class LeadfieldFitnessEvaluator:
    """对四电极和两路独立电流计算 SimNIBS focality。"""

    def __init__(
        self,
        leadfield: LeadfieldData,
        region_masks: RegionMasks,
        non_roi_threshold_v_per_m: float,
        roi_threshold_v_per_m: float,
    ) -> None:
        """初始化 leadfield、ROI 及 SimNIBS focality 阈值。

        Parameters
        ----------
        leadfield : LeadfieldData
            已全量加载的单位电流电场。
        region_masks : RegionMasks
            与 leadfield element 顺序对齐的 ROI/Rest 数据。
        non_roi_threshold_v_per_m : float
            非 ROI 不应超过的 max_TI 阈值，单位 V/m。
        roi_threshold_v_per_m : float
            ROI 应达到的 max_TI 阈值，单位 V/m。
        """
        if leadfield.values.shape[1] != region_masks.roi_mask.size:
            raise ValueError("leadfield element 数与 ROI/Rest mask 不一致")
        if non_roi_threshold_v_per_m < 0 or roi_threshold_v_per_m < 0:
            raise ValueError("focality 阈值不能为负数")
        if non_roi_threshold_v_per_m > roi_threshold_v_per_m:
            raise ValueError("非 ROI 阈值不能高于 ROI 阈值")
        self.leadfield = leadfield
        self.region_masks = region_masks
        self.non_roi_threshold_v_per_m = non_roi_threshold_v_per_m
        self.roi_threshold_v_per_m = roi_threshold_v_per_m

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
        """对固定电极和单组电流计算 focality 及完整诊断指标。

        Parameters
        ----------
        geometry : tuple[numpy.ndarray, ...]
            两路单位电场的模、绝对点积和叉积模，当前候选只计算一次。
        currents : CurrentPair
            需要评估的两路电流幅值。

        Returns
        -------
        FitnessMetrics
            SimNIBS focality、ROI/Rest 统计及电流。
        """
        max_ti = self._scaled_max_ti(geometry, currents)
        roi_mean, rest_mean, ratio, roi_max = calculate_region_metrics(
            max_ti,
            self.region_masks,
        )
        objective, score, roc_distance, roi_sensitivity, false_positive_rate = (
            calculate_focality_metrics(
                max_ti,
                self.region_masks,
                self.non_roi_threshold_v_per_m,
                self.roi_threshold_v_per_m,
            )
        )
        return FitnessMetrics(
            objective=objective,
            score=score,
            roc_distance=roc_distance,
            roi_sensitivity=roi_sensitivity,
            non_roi_false_positive_rate=false_positive_rate,
            roi_rest_ratio=ratio,
            roi_mean_v_per_m=roi_mean,
            rest_mean_v_per_m=rest_mean,
            roi_max_v_per_m=roi_max,
            currents=currents,
        )

    def _evaluate_objective(
        self,
        geometry: tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        currents: CurrentPair,
    ) -> float:
        """仅计算 GA 需要的 SimNIBS focality objective。

        Parameters
        ----------
        geometry : tuple[numpy.ndarray, ...]
            两路单位电场的预计算几何量。
        currents : CurrentPair
            两路独立电流幅值，单位 mA。

        Returns
        -------
        float
            与 SimNIBS ``goal="focality"`` 一致的最小化目标值。

        Notes
        -----
        GA 循环不计算体积加权 mean、ratio 和 max，避免为每个候选执行额外的
        全 mesh 聚合；这些诊断指标只在最终候选确定后计算。
        """
        max_ti = self._scaled_max_ti(geometry, currents)
        objective, _, _ = calculate_focality_objective(
            max_ti,
            self.region_masks,
            self.non_roi_threshold_v_per_m,
            self.roi_threshold_v_per_m,
        )
        return objective

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
        """预计算当前电极和电流共用的两路电场几何量。

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
        max_TI 对正电流缩放具有解析形式，因此无需创建两个缩放后的
        ``(N_elements, 3)`` 电场再计算点积和叉积。
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
        """评估指定四电极和两路电流并返回完整结果指标。

        Parameters
        ----------
        chromosome : ElectrodeChromosome
            A+、A-、B+、B- 四个 montage 索引。
        currents : CurrentPair
            两路固定电流幅值。

        Returns
        -------
        FitnessMetrics
            SimNIBS focality 和 ROI/Rest 诊断指标。
        """
        names = self.electrode_names(chromosome)
        unit_field_a = self._unit_pair_field(names[0], names[1])
        unit_field_b = self._unit_pair_field(names[2], names[3])
        geometry = self._pair_geometry(unit_field_a, unit_field_b)
        return self._evaluate_current(geometry, currents)

    def evaluate_objective(
        self,
        chromosome: ElectrodeChromosome,
        currents: CurrentPair,
    ) -> float:
        """为 GA 候选计算一次 SimNIBS focality objective。

        Parameters
        ----------
        chromosome : ElectrodeChromosome
            四个互不重复的 montage 电极索引。
        currents : CurrentPair
            GA 当前候选的两路独立电流。

        Returns
        -------
        float
            SimNIBS ``goal="focality"`` 的最小化目标值。

        Raises
        ------
        ValueError
            电极重复或电流不是有限正数时抛出。
        """
        if len(set(chromosome.electrode_indices)) != 4:
            raise ValueError("适应度计算只接受四个互不重复的电极")
        current_values = (currents.current_a_ma, currents.current_b_ma)
        if any(not np.isfinite(value) or value <= 0 for value in current_values):
            raise ValueError("两路电流必须为有限正数")
        names = self.electrode_names(chromosome)
        unit_field_a = self._unit_pair_field(names[0], names[1])
        unit_field_b = self._unit_pair_field(names[2], names[3])
        geometry = self._pair_geometry(unit_field_a, unit_field_b)
        return self._evaluate_objective(geometry, currents)

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
