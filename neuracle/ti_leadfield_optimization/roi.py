"""
Atlas 或 MNI 球形 ROI、non-ROI 与 Rest 的 element mask 和体积权重构建。

该模块支持把 MNI atlas mask 映射到 SimNIBS subject mesh，也支持把 MNI 球心
变换到 subject 空间后按半径筛选，并严格将统计范围限定为 WM、GM 四面体。
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neuracle.ti_leadfield_optimization.models import RegionMasks
from simnibs import ElementTags
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.utils.region_of_interest import RegionOfInterest

logger = logging.getLogger(__name__)


def _map_atlas_masks_to_tissue(
    mesh: Msh,
    subject_dir: str | Path,
    atlas_mask_paths: Sequence[str | Path],
    region_name: str,
) -> np.ndarray:
    """把多个 MNI atlas mask 的并集映射到 mesh 的 WM/GM 四面体。

    Parameters
    ----------
    mesh : simnibs.mesh_tools.mesh_io.Msh
        目标 subject mesh。
    subject_dir : str or pathlib.Path
        包含 MNI 到 subject 变换的 ``m2m_*`` 目录。
    atlas_mask_paths : sequence of str or pathlib.Path
        一个或多个二值 MNI atlas mask。
    region_name : str
        用于异常消息的区域名称。

    Returns
    -------
    numpy.ndarray
        与 mesh element 对齐的布尔 mask。

    Raises
    ------
    ValueError
        Mask 列表为空或映射后未命中 WM/GM 四面体时抛出。
    FileNotFoundError
        Atlas mask 文件不存在时抛出。
    """
    mask_paths = [Path(path) for path in atlas_mask_paths]
    if not mask_paths:
        raise ValueError(f"至少需要一个 atlas {region_name} mask")
    for mask_path in mask_paths:
        if not mask_path.is_file():
            raise FileNotFoundError(f"atlas {region_name} mask 不存在: {mask_path}")
    region = RegionOfInterest()
    region.method = "volume"
    region.mesh = mesh
    region.subpath = str(subject_dir)
    region.tissues = [ElementTags.WM, ElementTags.GM]
    region.mask_path = [str(path) for path in mask_paths]
    region.mask_space = ["mni"] * len(mask_paths)
    region.mask_value = [1] * len(mask_paths)
    region.mask_operator = ["intersection"] + ["union"] * (len(mask_paths) - 1)
    region._prepare()
    region_mask = np.asarray(region._mask, dtype=np.bool_).copy()
    tissue_mask = np.isin(mesh.elm.tag1, [ElementTags.WM, ElementTags.GM]) & (
        mesh.elm.elm_type == 4
    )
    region_mask &= tissue_mask
    if region_mask.shape != (mesh.elm.nr,):
        raise ValueError(f"{region_name} mask 形状与 mesh element 数不一致")
    if not np.any(region_mask):
        raise ValueError(f"atlas {region_name} 映射后未命中任何 WM/GM 四面体")
    return region_mask


def build_atlas_region_masks(
    mesh: Msh,
    subject_dir: str | Path,
    atlas_mask_paths: Sequence[str | Path],
) -> RegionMasks:
    """将多个 atlas 子区取并集，构建 ROI/Rest mask 和 element 体积。

    Parameters
    ----------
    mesh : simnibs.mesh_tools.mesh_io.Msh
        Leadfield 内嵌 mesh 或直接 FEM 结果 mesh。
    subject_dir : str or pathlib.Path
        包含 MNI 到 subject 变换的 ``m2m_*`` 目录。
    atlas_mask_paths : sequence of str or pathlib.Path
        一个或多个二值 MNI atlas mask，多个 mask 取并集。

    Returns
    -------
    RegionMasks
        与 mesh element 顺序对齐的 ROI、Rest mask 和体积权重。

    Raises
    ------
    ValueError
        Mask 列表为空、ROI/Rest 为空或体积非法时抛出。
    FileNotFoundError
        Atlas mask 文件不存在时抛出。

    Notes
    -----
    第一个 mask 与 WM/GM 范围做 intersection，后续 mask 用 union 合并；
    Rest 是同一 mesh 中所有 WM/GM 四面体扣除 ROI。
    """
    mask_paths = [Path(path) for path in atlas_mask_paths]
    if not mask_paths:
        raise ValueError("至少需要一个 atlas ROI mask")
    for mask_path in mask_paths:
        if not mask_path.is_file():
            raise FileNotFoundError(f"atlas ROI mask 不存在: {mask_path}")
    roi = RegionOfInterest()
    roi.method = "volume"
    roi.mesh = mesh
    roi.subpath = str(subject_dir)
    roi.tissues = [ElementTags.WM, ElementTags.GM]
    roi.mask_path = [str(path) for path in mask_paths]
    roi.mask_space = ["mni"] * len(mask_paths)
    roi.mask_value = [1] * len(mask_paths)
    roi.mask_operator = ["intersection"] + ["union"] * (len(mask_paths) - 1)
    roi._prepare()
    roi_mask = np.asarray(roi._mask, dtype=np.bool_).copy()
    tissue_mask = np.isin(mesh.elm.tag1, [ElementTags.WM, ElementTags.GM]) & (
        mesh.elm.elm_type == 4
    )
    roi_mask &= tissue_mask
    rest_mask = tissue_mask & ~roi_mask
    element_volumes = np.asarray(
        mesh.elements_volumes_and_areas().value,
        dtype=np.float64,
    )
    if roi_mask.shape != (mesh.elm.nr,) or rest_mask.shape != (mesh.elm.nr,):
        raise ValueError("ROI/Rest mask 形状与 mesh element 数不一致")
    if not np.any(roi_mask):
        raise ValueError("atlas ROI 映射后未命中任何 WM/GM 四面体")
    if not np.any(rest_mask):
        raise ValueError("Rest 区域为空")
    if not np.all(np.isfinite(element_volumes)) or np.any(element_volumes <= 0):
        raise ValueError("mesh element 体积必须是有限正数")
    roi_volume = float(np.sum(element_volumes[roi_mask]))
    rest_volume = float(np.sum(element_volumes[rest_mask]))
    if roi_volume <= 0 or rest_volume <= 0:
        raise ValueError("ROI 和 Rest 的总体积必须大于 0")
    logger.info(
        "Atlas ROI 构建完成: roi_elements=%s, rest_elements=%s, roi_volume=%s, rest_volume=%s",
        int(np.count_nonzero(roi_mask)),
        int(np.count_nonzero(rest_mask)),
        roi_volume,
        rest_volume,
    )
    return RegionMasks(
        roi_mask=roi_mask,
        rest_mask=rest_mask,
        element_volumes=element_volumes,
        roi_volume=roi_volume,
        rest_volume=rest_volume,
    )


def build_atlas_roi_non_roi_masks(
    mesh: Msh,
    subject_dir: str | Path,
    roi_mask_paths: Sequence[str | Path],
    non_roi_mask_paths: Sequence[str | Path],
) -> RegionMasks:
    """分别映射 atlas ROI 与指定 non-ROI 并构建体积权重。

    Parameters
    ----------
    mesh : simnibs.mesh_tools.mesh_io.Msh
        Leadfield 内嵌 mesh 或直接 FEM 结果 mesh。
    subject_dir : str or pathlib.Path
        包含 MNI 到 subject 变换的 ``m2m_*`` 目录。
    roi_mask_paths : sequence of str or pathlib.Path
        组成目标 ROI 的一个或多个二值 MNI atlas mask。
    non_roi_mask_paths : sequence of str or pathlib.Path
        组成指定 non-ROI 的一个或多个二值 MNI atlas mask。

    Returns
    -------
    RegionMasks
        与 mesh element 顺序对齐的 ROI、指定 non-ROI mask 和体积权重。

    Raises
    ------
    ValueError
        区域为空、发生重叠或 element 体积非法时抛出。
    FileNotFoundError
        任一 atlas mask 文件不存在时抛出。

    Notes
    -----
    ``RegionMasks.rest_mask`` 和 ``rest_volume`` 是兼容既有计算接口的字段名；
    在本函数返回值中，它们仅保存显式指定的 non-ROI，而不是其余全部 WM/GM。
    """
    roi_mask = _map_atlas_masks_to_tissue(
        mesh,
        subject_dir,
        roi_mask_paths,
        "ROI",
    )
    non_roi_mask = _map_atlas_masks_to_tissue(
        mesh,
        subject_dir,
        non_roi_mask_paths,
        "non-ROI",
    )
    overlap_count = int(np.count_nonzero(roi_mask & non_roi_mask))
    if overlap_count:
        raise ValueError(f"atlas ROI 与 non-ROI 重叠 {overlap_count} 个 mesh element")
    element_volumes = np.asarray(
        mesh.elements_volumes_and_areas().value,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(element_volumes)) or np.any(element_volumes <= 0):
        raise ValueError("mesh element 体积必须是有限正数")
    roi_volume = float(np.sum(element_volumes[roi_mask]))
    non_roi_volume = float(np.sum(element_volumes[non_roi_mask]))
    if roi_volume <= 0 or non_roi_volume <= 0:
        raise ValueError("ROI 和 non-ROI 的总体积必须大于 0")
    logger.info(
        "Atlas ROI/non-ROI 构建完成: roi_elements=%s, non_roi_elements=%s, "
        "roi_volume=%s, non_roi_volume=%s",
        int(np.count_nonzero(roi_mask)),
        int(np.count_nonzero(non_roi_mask)),
        roi_volume,
        non_roi_volume,
    )
    return RegionMasks(
        roi_mask=roi_mask,
        rest_mask=non_roi_mask,
        element_volumes=element_volumes,
        roi_volume=roi_volume,
        rest_volume=non_roi_volume,
    )


def build_mni_sphere_region_masks(
    mesh: Msh,
    subject_dir: str | Path,
    center_mni: Sequence[float],
    radius_mm: float,
) -> RegionMasks:
    """使用 MNI 球心和半径构建 WM/GM ROI 与 Rest mask。

    Parameters
    ----------
    mesh : simnibs.mesh_tools.mesh_io.Msh
        Leadfield 内嵌 mesh 或直接 FEM 结果 mesh。
    subject_dir : str or pathlib.Path
        包含 MNI 到 subject 非线性变换的 ``m2m_*`` 目录。
    center_mni : sequence of float
        球心的 MNI ``(x, y, z)`` 坐标，单位 mm。
    radius_mm : float
        subject 空间中的球形筛选半径，单位 mm。

    Returns
    -------
    RegionMasks
        与 mesh element 顺序对齐的 ROI、Rest mask 和体积权重。

    Raises
    ------
    ValueError
        球心、半径、映射结果或 element 体积非法时抛出。

    Notes
    -----
    SimNIBS 先把 MNI 球心非线性变换到 subject 空间，再用 KD-tree 查找半径内的
    element center；最终 ROI 严格限制为 WM/GM 四面体，不读取 atlas mask。
    """
    center_values = np.asarray(center_mni, dtype=np.float64)
    radius_value = float(radius_mm)
    if center_values.shape != (3,) or not np.all(np.isfinite(center_values)):
        raise ValueError("MNI 球心必须是三个有限坐标")
    if not np.isfinite(radius_value) or radius_value <= 0:
        raise ValueError("MNI 球形 ROI 半径必须是有限正数")
    roi = RegionOfInterest()
    roi.method = "volume"
    roi.mesh = mesh
    roi.subpath = str(subject_dir)
    roi.tissues = [ElementTags.WM, ElementTags.GM]
    roi.roi_sphere_center = center_values.tolist()
    roi.roi_sphere_radius = radius_value
    roi.roi_sphere_center_space = "mni"
    roi.roi_sphere_operator = "intersection"
    roi._prepare()
    roi_mask = np.asarray(roi._mask, dtype=np.bool_).copy()
    tissue_mask = np.isin(mesh.elm.tag1, [ElementTags.WM, ElementTags.GM]) & (
        mesh.elm.elm_type == 4
    )
    roi_mask &= tissue_mask
    rest_mask = tissue_mask & ~roi_mask
    element_volumes = np.asarray(
        mesh.elements_volumes_and_areas().value,
        dtype=np.float64,
    )
    if roi_mask.shape != (mesh.elm.nr,) or rest_mask.shape != (mesh.elm.nr,):
        raise ValueError("ROI/Rest mask 形状与 mesh element 数不一致")
    if not np.any(roi_mask):
        raise ValueError("MNI 球形 ROI 映射后未命中任何 WM/GM 四面体")
    if not np.any(rest_mask):
        raise ValueError("Rest 区域为空")
    if not np.all(np.isfinite(element_volumes)) or np.any(element_volumes <= 0):
        raise ValueError("mesh element 体积必须是有限正数")
    roi_volume = float(np.sum(element_volumes[roi_mask]))
    rest_volume = float(np.sum(element_volumes[rest_mask]))
    if roi_volume <= 0 or rest_volume <= 0:
        raise ValueError("ROI 和 Rest 的总体积必须大于 0")
    logger.info(
        "MNI 球形 ROI 构建完成: center_mni=%s, radius_mm=%s, roi_elements=%s, "
        "rest_elements=%s, roi_volume=%s, rest_volume=%s",
        center_values.tolist(),
        radius_value,
        int(np.count_nonzero(roi_mask)),
        int(np.count_nonzero(rest_mask)),
        roi_volume,
        rest_volume,
    )
    return RegionMasks(
        roi_mask=roi_mask,
        rest_mask=rest_mask,
        element_volumes=element_volumes,
        roi_volume=roi_volume,
        rest_volume=rest_volume,
    )
