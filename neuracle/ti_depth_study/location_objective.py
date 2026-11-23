"""方案 04 的体积短缺、强场偏离和皮层热点损失。"""

import numpy as np

from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry, _face_indices, weighted_quantile
from simnibs.mesh_tools.cgal import pyAABBTree


def volume_tail(field: FloatArray, volume: FloatArray, amount: float, high: bool = True) -> FloatArray:
    """提取固定体积的强场或弱场尾部，平分截断值处的体积权重。

    Parameters
    ----------
    field, volume : ndarray
        同序的有效包络场和四面体体积。
    amount : float
        选取体积，超过总量时取整个区域。
    high : bool, optional
        True 从强场开始，False 从弱场开始。

    Returns
    -------
    ndarray
        各单元贡献的体积。同场强的截断组同比例纳入。
    """
    if amount <= 0 or not len(field):
        raise ValueError("尾部体积和区域必须非空")
    order = np.argsort(-field if high else field, kind="stable")
    accumulated = np.cumsum(volume[order])
    if amount >= accumulated[-1]:
        return volume.copy()
    cutoff = field[order[np.searchsorted(accumulated, amount, side="left")]]
    full = field > cutoff if high else field < cutoff
    boundary = field == cutoff
    remainder = amount - volume[full].sum()
    result = np.zeros_like(volume)
    result[full] = volume[full]
    result[boundary] = volume[boundary] * np.clip(remainder / volume[boundary].sum(), 0.0, 1.0)
    return result


def centroid_target_distances(geometry: Geometry, target: BoolArray) -> FloatArray:
    """预计算共同脑域的各四面体重心至目标实体的距离。

    Parameters
    ----------
    geometry : Geometry
        具有节点坐标和连通关系的参考四面体。
    target : ndarray of bool
        目标四面体掩膜。

    Returns
    -------
    ndarray
        靶内为零，靶外通过 CGAL 树求到目标表面三角形的最近距离，单位 mm。
    """
    if not target.any():
        raise ValueError("目标没有参考网格四面体")
    faces, _, counts = _face_indices(geometry.connectivity[target])
    if np.any(counts > 2):
        raise ValueError("目标中存在三重共享三角面")
    surface = faces[counts == 1]
    nodes, indices = np.unique(surface, return_inverse=True)
    if not len(surface):
        raise ValueError("目标实体没有边界面")
    tree = pyAABBTree()
    tree.set_data(geometry.node_mm[nodes], indices.reshape(-1, 3))
    result = np.zeros(len(target), dtype=np.float64)
    if np.any(~target):
        result[~target] = np.sqrt(np.asarray(tree.min_sqdist(geometry.centroid_mm[~target]), dtype=np.float64))
    return result


def location_terms(
    field: FloatArray, volume: FloatArray, target: BoolArray, cortex: BoolArray,
    distance: FloatArray, threshold: float = 0.1, tail_mm3: float = 100.0, distance_scale: float = 10.0,
) -> dict[str, float]:
    """计算方案 04 的三项损失，供各消融臂和最终复评共用。

    Parameters
    ----------
    field, volume : ndarray
        同序脑实质包络（V/m）及体积（mm³）。
    target, cortex : ndarray of bool
        目标与皮层代理区，计算时从皮层扣除靶区。
    distance : ndarray
        每个参考重心至目标四面体并集的距离，单位 mm。
    threshold, tail_mm3, distance_scale : float, optional
        目标场阈值、强场体积、偏离距离归一化尺度。

    Returns
    -------
    dict
        靶内不足 A、偏离 D、靶外皮层热点 P、皮层强场均值 H_C 和靶内中位数 m_T。
    """
    cortical = cortex & ~target
    if not target.any() or not cortical.any() or not np.isfinite(field).all() or not np.any(field > 0):
        raise ValueError("定位损失需要有限的非零场、非空靶区和靶外皮层")
    shortfall = np.average(np.maximum(0.0, 1 - field[target] / threshold), weights=volume[target])
    head_tail = volume_tail(field, volume, tail_mm3)
    cortical_tail = volume_tail(field[cortical], volume[cortical], tail_mm3)
    hot_cortex = float(np.average(field[cortical], weights=cortical_tail))
    median = weighted_quantile(field[target], volume[target], 0.5)
    offset = float(np.average(distance / (distance + distance_scale), weights=head_tail))
    return dict(A=float(shortfall), D=offset, P=max(0.0, hot_cortex - median) / threshold,
                H_C=hot_cortex, m_T=median, hot_tail_mm3=float(head_tail.sum()), cortex_tail_mm3=float(cortical_tail.sum()))
