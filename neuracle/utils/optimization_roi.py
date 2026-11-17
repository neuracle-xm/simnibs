"""共享 dev 的 ROI 配置，分别创建独立 RegionOfInterest 对象。"""

import logging
from pathlib import Path
from typing import Any

from neuracle.atlas.julich_vpm_merge import (
    build_julich_vpm_rh_merged_mask,
    get_julich_vpm_rh_merged_mask_path,
    is_julich_vpm_rh,
)
from neuracle.atlas.standardized import get_standardized_roi_path
from neuracle.parameters.schemas import ROIParam
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.utils.mesh_element_properties import ElementTags
from simnibs.utils.region_of_interest import RegionOfInterest

logger = logging.getLogger(__name__)


def resolve_atlas_roi_mask_path(
    atlas_name: str,
    area_name: str,
) -> Path:
    """解析 TI 逆向优化实际使用的 atlas ROI mask。

    Parameters
    ----------
    atlas_name : str
        atlas 名称。
    area_name : str
        atlas 脑区名称。
    Returns
    -------
    Path
        原始标准化 ROI 或 VPM 右侧合并 ROI 的路径。

    Notes
    -----
    只有 Julich VPM 右侧同时精确命中 atlas 和脑区名称时才使用离线合并
    mask 的固定路径；文件缺失时自动生成一次。其他选择继续使用现有
    标准化单 ROI 解析逻辑。
    """
    if is_julich_vpm_rh(atlas_name, area_name):
        merged_path = get_julich_vpm_rh_merged_mask_path()
        if not merged_path.exists():
            logger.info("Julich VPM 右侧离线合并 ROI 不存在，开始生成: %s", merged_path)
            merged_path = build_julich_vpm_rh_merged_mask(merged_path)
        logger.info("逆向仿真使用 Julich VPM 右侧合并 ROI: %s", merged_path)
        return merged_path
    return get_standardized_roi_path(atlas_name, area_name)


def resolve_roi_settings(roi_type: str, roi_param: ROIParam) -> dict[str, Any]:
    """将正式 atlas/MNI 参数解析为共享区域配置，保留 VPM 合并入口。

    Parameters
    ----------
    roi_type : str
        atlas 或 mni_pos。
    roi_param : ROIParam
        经业务校验的 ROI 参数。

    Returns
    -------
    dict[str, Any]
        供 build_optimization_rois 使用的配置。
    """
    if roi_type == "atlas" and roi_param.atlas_param is not None:
        mask_path = resolve_atlas_roi_mask_path(
            roi_param.atlas_param.name, roi_param.atlas_param.area
        )
        if not mask_path.is_file():
            raise FileNotFoundError(f"标准化 ROI 不存在: {mask_path}")
        return {"roi_mask_path": str(mask_path), "roi_mask_space": "mni"}
    if roi_type == "mni_pos" and roi_param.mni_param is not None:
        return {
            "roi_center": roi_param.mni_param.center,
            "roi_radius": roi_param.mni_param.radius,
            "roi_center_space": "mni",
        }
    raise ValueError("ROI 类型与参数不匹配")


def build_optimization_rois(
    subject_dir: str,
    mesh: str | Msh,
    goal: str = "focality",
    roi_center: list[float] | None = None,
    roi_radius: float | None = None,
    roi_center_space: str = "subject",
    roi_mask_path: str | None = None,
    roi_mask_space: str | None = None,
    non_roi_center: list[float] | None = None,
    non_roi_radius: float | None = None,
) -> list[RegionOfInterest]:
    """创建独立区域对象，沿用 dev 的 WM/GM、atlas 差集及 25 mm 排除球。

    Parameters
    ----------
    subject_dir : str
        含 MNI 变换的 m2m 目录。
    mesh : str or Msh
        当前求解网格，based 必须传入 leadfield 内嵌网格。
    goal : str
        focality/focality_inv 额外创建 non-ROI。
    roi_center, roi_radius, roi_center_space : optional
        目标球配置，默认中心 [-41, -13, 66]、半径 20 mm。
    roi_mask_path, roi_mask_space : optional
        mask 优先于球配置，默认 MNI 空间。
    non_roi_center, non_roi_radius : optional
        排除球默认与目标同中心，半径固定默认 25 mm。

    Returns
    -------
    list[RegionOfInterest]
        尚未准备的 ROI、non-ROI；每次调用均新建。
    """
    if roi_center is None:
        roi_center = [-41.0, -13.0, 66.0]
    if roi_radius is None:
        roi_radius = 20.0
    # 配置 ROI
    logger.info("配置 ROI")
    roi = RegionOfInterest()
    regions = [roi]
    roi.method = "volume"
    roi.mesh = mesh
    roi.subpath = subject_dir
    roi.tissues = [ElementTags.WM, ElementTags.GM]  # 只保留白质和灰质
    if roi_mask_path:
        roi.mask_path = roi_mask_path
        roi.mask_space = roi_mask_space or "mni"
        roi.mask_value = 1
        logger.info("使用 atlas ROI mask: %s (space=%s)", roi_mask_path, roi.mask_space)
    else:
        roi.roi_sphere_center_space = roi_center_space
        roi.roi_sphere_center = roi_center
        roi.roi_sphere_radius = roi_radius
        logger.info(
            "使用体积球形 ROI: 中心=%s, 半径=%s, space=%s",
            roi_center,
            roi_radius,
            roi_center_space,
        )
    # focality 目标的第二个 ROI 表示"除目标 ROI 外的其余体积"
    if goal in ["focality", "focality_inv"]:
        non_roi = RegionOfInterest()
        regions.append(non_roi)
        non_roi.method = "volume"
        non_roi.mesh = mesh
        non_roi.subpath = subject_dir
        non_roi.tissues = [ElementTags.WM, ElementTags.GM]  # 只保留白质和灰质
        if roi_mask_path:
            non_roi.mask_path = roi_mask_path
            non_roi.mask_space = roi_mask_space or "mni"
            non_roi.mask_value = 1
            non_roi.mask_operator = ["difference"]
            logger.info(
                "配置 Non-ROI: 使用 atlas ROI 差集 (mask=%s, space=%s)",
                roi_mask_path,
                non_roi.mask_space,
            )
        else:
            if non_roi_center is None:
                non_roi_center = roi_center
            if non_roi_radius is None:
                non_roi_radius = 25.0
            non_roi.roi_sphere_center_space = roi_center_space
            non_roi.roi_sphere_center = non_roi_center
            non_roi.roi_sphere_radius = non_roi_radius
            non_roi.roi_sphere_operator = ["difference"]
            logger.info(
                "配置 Non-ROI: 中心=%s, 半径=%s, space=%s",
                non_roi_center,
                non_roi_radius,
                roi_center_space,
            )
    return regions
