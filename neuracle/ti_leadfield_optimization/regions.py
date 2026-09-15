"""将 SimNIBS RegionOfInterest 的区域结果适配到 leadfield 元素顺序。"""

import numpy as np

from neuracle.ti_leadfield_optimization.models import RegionMasks
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.utils.region_of_interest import RegionOfInterest


def prepare_region_masks(mesh: Msh, regions: list[RegionOfInterest]) -> RegionMasks:
    """在同一个内嵌 mesh 上准备区域；内部 mask 访问集中在此处。

    Parameters
    ----------
    mesh : Msh
        与 leadfield dataset 对齐的内嵌网格。
    regions : list[RegionOfInterest]
        共享配置工厂创建的 ROI 和 non-ROI。

    Returns
    -------
    RegionMasks
        元素布尔 mask 和以 mm³ 表示的体积；不初始化 OnlineFEM。
    """
    if len(regions) != 2:
        raise ValueError("focality 必须包含 ROI 和 non-ROI")
    masks = []
    for region in regions:
        if region.mesh is not mesh:
            raise ValueError("区域必须使用 leadfield 的同一个内嵌 mesh")
        region.get_nodes()
        if region._mesh is not mesh or region._mask_type != "elm_center":
            raise ValueError("RegionOfInterest 未返回原网格上的元素 mask")
        mask = np.asarray(region._mask, dtype=np.bool_).copy()
        if mask.shape != (mesh.elm.nr,) or not np.any(mask):
            raise ValueError("ROI/non-ROI 为空或与 leadfield 元素数量不符")
        masks.append(mask)
    volumes = np.asarray(mesh.elements_volumes_and_areas().value, dtype=np.float64)
    if (
        volumes.shape != (mesh.elm.nr,)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0)
    ):
        raise ValueError("leadfield 元素体积必须是有限正数")
    return RegionMasks(
        masks[0],
        masks[1],
        volumes,
        float(volumes[masks[0]].sum()),
        float(volumes[masks[1]].sum()),
    )
