"""在共同脑实质四面体网格上计算 TI 场的研究指标。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from simnibs.utils.TI_utils import get_maxTI


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


@dataclass(frozen=True)
class Geometry:
    """以固定单元 ID 行序保存脑实质网格。

    Attributes
    ----------
    element_id : ndarray
        每个四面体在参考 MSH 中的唯一 ID。
    node_mm : ndarray
        个体物理空间的节点坐标，单位 mm。
    connectivity : ndarray
        每个四面体的四个节点下标，从 0 开始。
    volume_mm3 : ndarray
        四面体体积，单位 mm³。
    centroid_mm : ndarray
        四面体几何重心，单位 mm。
    cortex_mask : ndarray
        图谱皮层标签与灰质的交集。
    """

    element_id: IntArray
    node_mm: FloatArray
    connectivity: IntArray
    volume_mm3: FloatArray
    centroid_mm: FloatArray
    cortex_mask: BoolArray


def ti_envelope(e_a: FloatArray, e_b: FloatArray) -> FloatArray:
    """计算两路同序单频场的方向无约束包络。

    Parameters
    ----------
    e_a, e_b : ndarray
        两路 (N, 3) 向量场，单位 V/m。

    Returns
    -------
    ndarray
        同序包络。单路为零时包络为零，非有限输入仍为 NaN。
    """
    if e_a.shape != e_b.shape or e_a.ndim != 2 or e_a.shape[1] != 3:
        raise ValueError("两路单频场须使用同序 (N, 3) 向量")
    valid = np.isfinite(e_a).all(axis=1) & np.isfinite(e_b).all(axis=1)
    active = valid & np.any(e_a != 0, axis=1) & np.any(e_b != 0, axis=1)
    result = np.full(len(e_a), np.nan)
    result[valid] = 0.0
    coincident = active & (np.all(e_a == e_b, axis=1) | np.all(e_a == -e_b, axis=1))
    result[coincident] = 2 * np.linalg.norm(e_a[coincident], axis=1)
    active &= ~coincident
    with np.errstate(divide="ignore", invalid="ignore"):
        result[active] = get_maxTI(e_a[active], e_b[active])
    return result


def weighted_quantile(values: FloatArray, volumes: FloatArray, q: float) -> float:
    """按体积求离散累计分布的逆。

    Parameters
    ----------
    values, volumes : ndarray
        已验证为有限的读数和同序正体积。
    q : float
        请求的累计分位，取值 [0, 1]。

    Returns
    -------
    float
        首个累计体积达到分位数的读数。区域为空时返回 NaN。
    """
    if not len(values):
        return np.nan
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(volumes[order])
    index = np.searchsorted(cumulative, q * cumulative[-1], side="left")
    return float(values[order[index]])


def coverage(field: FloatArray, volumes: FloatArray, threshold: float) -> float:
    """求给定区域内场强达到阈值的体积比例。

    Parameters
    ----------
    field, volumes : ndarray
        同序的区域场强和单元体积。
    threshold : float
        场强阈值，单位 V/m，包含等于阈值的单元。

    Returns
    -------
    float
        覆盖体积除以该区域总体积。区域为空时返回 NaN。
    """
    return float(volumes[field >= threshold].sum() / volumes.sum()) if len(volumes) else np.nan


def _face_indices(tetrahedra: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """按排序节点三元组为各个四面体共享面建立相同编号。

    Parameters
    ----------
    tetrahedra : ndarray
        四面体节点下标。

    Returns
    -------
    faces, inverse, counts : tuple of ndarray
        唯一三角面、各四面体的四个面编号及各面出现次数。
    """
    faces = np.sort(tetrahedra[:, _FACES].reshape(-1, 3), axis=1)
    unique, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse.reshape(-1, 4), counts


def distance_to_target(point: FloatArray, geometry: Geometry, target: BoolArray) -> float:
    """计算点至目标四面体实体并集的最短欧氏距离。

    Parameters
    ----------
    point : ndarray
        个体空间中的三维点，单位 mm。
    geometry : Geometry
        与目标掩膜同序的四面体几何。
    target : ndarray of bool
        所有属于目标的四面体。

    Returns
    -------
    float
        点在目标中时为零。外部点到边界三角形最近点的距离，单位 mm。
    """
    connectivity = geometry.connectivity[target]
    if not len(connectivity):
        return np.nan
    nodes = geometry.node_mm[connectivity]
    edges = nodes[:, 1:] - nodes[:, :1]
    local = np.linalg.solve(edges.swapaxes(1, 2), (point - nodes[:, 0])[..., None])[..., 0]
    tolerance = 1e-10
    if np.any(np.all(local >= -tolerance, axis=1) & (local.sum(axis=1) <= 1 + tolerance)):
        return 0.0
    faces, _, counts = _face_indices(connectivity)
    triangles = geometry.node_mm[faces[counts == 1]]
    starts = triangles
    sides = np.roll(triangles, -1, axis=1) - starts
    relative = point - starts
    projection = np.einsum("nki,nki->nk", relative, sides) / np.einsum("nki,nki->nk", sides, sides)
    edge_distance = np.linalg.norm(relative - np.clip(projection, 0, 1)[..., None] * sides, axis=2).min()
    normal = np.cross(sides[:, 0], sides[:, 1])
    within_face = np.all(np.einsum("nki,ni->nk", np.cross(sides, relative), normal) >= 0, axis=1)
    planar_distance = np.abs(np.einsum("ni,ni->n", relative[:, 0], normal)) / np.linalg.norm(normal, axis=1)
    return float(min(edge_distance, planar_distance[within_face].min(initial=np.inf)))


def _hotspot(
    field: FloatArray, geometry: Geometry, target: BoolArray, requested_mm3: float,
) -> dict[str, object]:
    """选取最高场体积并按共享三角面分连通簇。

    Parameters
    ----------
    field : ndarray
        共同脑实质网格上的有效包络，单位 V/m。
    geometry : Geometry
        同序的共同网格。
    target : ndarray of bool
        解剖目标的单元掩膜。
    requested_mm3 : float
        从高到低累积的目标体积，最后一个单元完整计入。

    Returns
    -------
    dict
        主簇中心、距靶距离、各簇体积及热点与目标重叠。
    """
    order = np.lexsort((geometry.element_id, -field))
    count = np.searchsorted(np.cumsum(geometry.volume_mm3[order]), requested_mm3) + 1
    hot = order[:count]
    _, inverse, face_counts = _face_indices(geometry.connectivity[hot])
    if np.any(face_counts > 2):
        raise ValueError("热点中存在三个或更多四面体共用的三角面")
    shared = np.argsort(inverse.ravel())[np.repeat(face_counts == 2, face_counts)].reshape(-1, 2) // 4
    adjacency = coo_matrix((np.ones(len(shared)), (shared[:, 0], shared[:, 1])), shape=(len(hot), len(hot)))
    n_clusters, labels = connected_components(adjacency.tocsr(), directed=False)
    volumes = geometry.volume_mm3[hot]
    cluster_volumes = np.bincount(labels, weights=volumes)
    integrals = np.bincount(labels, weights=volumes * field[hot])
    min_ids = np.full(n_clusters, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(min_ids, labels, geometry.element_id[hot])
    by_priority = np.lexsort((min_ids, -integrals, -cluster_volumes))
    main = labels == by_priority[0]
    center = np.average(geometry.centroid_mm[hot[main]], axis=0, weights=volumes[main])
    distance = distance_to_target(center, geometry, target)
    overlap = float(volumes[main & target[hot]].sum())
    return {
        "D_hot_mm": distance, "hot_center_mm": center.tolist(), "hot_mm3": float(volumes.sum()),
        "cluster_mm3": cluster_volumes[by_priority].tolist(),
        "hot_overlap": float(volumes[target[hot]].sum() / volumes.sum()),
        "main_overlap_mm3": overlap, "main_overlap": overlap / float(volumes[main].sum()),
        "absolute_hit": distance == 0.0 and overlap > 0,
    }


def measure_field(
    geometry: Geometry, target: BoolArray, e_a: FloatArray, e_b: FloatArray,
    valid_mask: BoolArray | None = None, hotspot_mm3: float = 100.0,
) -> dict[str, object]:
    """计算共同正向场的覆盖、外溢与绝对定位。

    Parameters
    ----------
    geometry : Geometry
        只包含 WM/GM 四面体的固定参考网格。
    target : ndarray of bool
        参考网格中的目标掩膜。
    e_a, e_b : ndarray
        与参考网格同序的个体物理坐标单频向量，单位 V/m。
    valid_mask : ndarray of bool, optional
        两路均可对齐的参考单元。缺失体积单独记录，定位不予判定。
    hotspot_mm3 : float, optional
        等体积热点的请求体积。共同读数为 100 mm³。

    Returns
    -------
    dict
        绝对读数和定位状态。空目标、缺场、全零包络均给出原因。
    """
    field = ti_envelope(e_a, e_b)
    n = len(geometry.element_id)
    volumes = geometry.volume_mm3
    if field.shape != (n,) or target.shape != (n,) or (valid_mask is not None and valid_mask.shape != (n,)):
        raise ValueError("场、目标和有效掩膜须与参考单元逐行对应")
    if not np.all(np.isfinite(volumes) & (volumes > 0)) or not np.isfinite(hotspot_mm3) or hotspot_mm3 <= 0:
        raise ValueError("网格与热点体积须为有限正值")
    valid = np.isfinite(field)
    if valid_mask is not None:
        valid &= valid_mask
    selected = target & valid
    outside = ~target & valid
    cortical = geometry.cortex_mask & outside
    missing = float(volumes[~valid].sum())
    result: dict[str, object] = {
        "roi_mm3": float(volumes[target].sum()), "missing_mm3": missing,
        "missing_target_mm3": float(volumes[target & ~valid].sum()),
        "Q10_T": weighted_quantile(field[selected], volumes[selected], 0.10),
        "Q50_T": weighted_quantile(field[selected], volumes[selected], 0.50),
        "Q99_Cout": weighted_quantile(field[cortical], volumes[cortical], 0.99),
        "Q99_A_Cout": weighted_quantile(np.linalg.norm(e_a[cortical], axis=1), volumes[cortical], 0.99),
        "Q99_B_Cout": weighted_quantile(np.linalg.norm(e_b[cortical], axis=1), volumes[cortical], 0.99),
    }
    for suffix, threshold in (("010", 0.10), ("020", 0.20)):
        above = field >= threshold
        result[f"Cov{suffix}"] = coverage(field[selected], volumes[selected], threshold)
        result[f"covered{suffix}_mm3"] = float(volumes[selected & above].sum())
        result[f"Vout{suffix}"] = float(volumes[outside & above].sum())
        result[f"Vcout{suffix}"] = float(volumes[cortical & above].sum())
    if not np.any(target):
        return result | {"status": "empty_target"}
    if missing > 0:
        return result | {"status": "incomplete_field"}
    if not np.any(field > 0):
        return result | {"status": "zero_envelope"}
    result.update(_hotspot(field, geometry, target, hotspot_mm3))
    peak = np.lexsort((geometry.element_id, -field))[0]
    result["D_peak_mm"] = distance_to_target(geometry.centroid_mm[peak], geometry, target)
    target_center = np.average(geometry.centroid_mm[target], axis=0, weights=volumes[target])
    result["D_centers_mm"] = float(np.linalg.norm(np.asarray(result["hot_center_mm"]) - target_center))
    return result | {"status": "ok"}
