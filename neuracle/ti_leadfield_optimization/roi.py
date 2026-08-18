"""
Atlas ROI 与 Rest 的 element mask 和体积权重构建。

该模块把一个或多个 MNI atlas NIfTI mask 映射到 SimNIBS subject mesh，
并严格将统计范围限定为 WM、GM 四面体。
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
