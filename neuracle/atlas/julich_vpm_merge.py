"""Julich VPM 右侧 ROI 合并工具。

本模块仅为 VPM 右侧脑区解析 Julich-Brain v3.1 官方层级，并将最近
同侧多 ROI 祖先下的本地 mask 做逐体素逻辑并集。其他脑区不使用本模块。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

from neuracle.atlas.loader import get_atlas_spec
from neuracle.atlas.registry import ATLAS_ROOT

logger = logging.getLogger(__name__)

JULICH_ATLAS_NAME = "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
JULICH_VPM_RH_AREA = "VPM (Thalamus, ventral posterior medial Nucleus)_rh"
JULICH_HIERARCHY_PATH = (
    ATLAS_ROOT / "merge_metadata" / "julich_brain_v3_1_hierarchy.json"
)
JULICH_EXPECTED_VERSION = "3.1.0"
JULICH_VPM_RH_MERGED_FILENAME = "julich_vpm_rh_ventral_group_merged.nii.gz"


def is_julich_vpm_rh(atlas_name: str, area_name: str) -> bool:
    """判断 atlas 选择是否为唯一允许合并的 Julich VPM 右侧脑区。

    Parameters
    ----------
    atlas_name : str
        atlas 名称。
    area_name : str
        atlas 脑区英文名称。

    Returns
    -------
    bool
        atlas 和脑区名称均精确命中目标时返回 ``True``。

    Notes
    -----
    使用精确匹配可以保证其他 Julich 脑区继续使用原始单 ROI mask。
    """
    return atlas_name == JULICH_ATLAS_NAME and area_name == JULICH_VPM_RH_AREA


def get_julich_vpm_rh_merged_mask_path(
    spec: dict[str, Any] | None = None,
) -> Path:
    """返回离线生成的 Julich VPM 右侧合并 mask 固定路径。

    Parameters
    ----------
    spec : dict[str, Any] | None, optional
        已解析路径的 Julich atlas 规范；默认从 registry 加载。

    Returns
    -------
    Path
        标准化 Julich atlas 下的合并 ROI 绝对路径。

    Raises
    ------
    ValueError
        atlas 规范不是目标 Julich atlas。

    Notes
    -----
    本函数只解析路径，不读取层级、不合并 mask，也不创建文件，供在线逆向
    仿真直接定位离线产物。
    """
    atlas_spec = spec or get_atlas_spec(JULICH_ATLAS_NAME)
    if atlas_spec.get("name") != JULICH_ATLAS_NAME:
        raise ValueError(f"VPM 合并不支持 atlas: {atlas_spec.get('name')}")
    return (
        Path(atlas_spec["standardized_dir"])
        / "merged_rois"
        / JULICH_VPM_RH_MERGED_FILENAME
    ).resolve()


def _normalize_name(value: str) -> str:
    """生成 Julich 官方名称的精确匹配键。

    Parameters
    ----------
    value : str
        原始区域名称。

    Returns
    -------
    str
        仅统一连续空白和大小写后的名称。
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


def _julich_official_name(local_name: str) -> tuple[str, str | None]:
    """把本地 Julich 侧别后缀转换为官方区域树名称。

    Parameters
    ----------
    local_name : str
        本地 Julich 英文标签。

    Returns
    -------
    tuple[str, str | None]
        官方完整名称和侧别；无侧别后缀时侧别为 ``None``。
    """
    if local_name.endswith("_lh"):
        return f"{local_name[:-3]} - left hemisphere", "left"
    if local_name.endswith("_rh"):
        return f"{local_name[:-3]} - right hemisphere", "right"
    return local_name, None


def _load_julich_hierarchy(
    path: Path = JULICH_HIERARCHY_PATH,
) -> dict[str, Any]:
    """加载并校验 Julich-Brain v3.1 官方层级快照。

    Parameters
    ----------
    path : Path, optional
        层级快照路径。

    Returns
    -------
    dict[str, Any]
        已通过 atlas 版本和节点 SHA-256 校验的快照。

    Raises
    ------
    TypeError
        层级节点不是列表。
    ValueError
        层级版本或节点哈希不符合预期。
    """
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot.get("nodes"), list):
        raise TypeError("Julich 层级快照缺少节点列表")
    nodes_payload = json.dumps(
        snapshot["nodes"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual_hash = hashlib.sha256(nodes_payload).hexdigest()
    if snapshot.get("atlas_version") != JULICH_EXPECTED_VERSION:
        raise ValueError(f"Julich 层级版本不匹配: {snapshot.get('atlas_version')}")
    if actual_hash != snapshot.get("nodes_sha256"):
        raise ValueError("Julich 层级节点哈希校验失败")
    return snapshot


def _ancestor_ids(
    node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """返回 Julich 节点自身到根节点的 ID 链。

    Parameters
    ----------
    node_id : str
        起始节点 ID。
    nodes_by_id : dict[str, dict[str, Any]]
        官方节点 ID 索引。

    Returns
    -------
    list[str]
        从当前节点到根节点的有序 ID 列表。

    Raises
    ------
    ValueError
        层级包含缺失父节点或环时。
    """
    ancestors: list[str] = []
    current_id: str | None = node_id
    while current_id is not None:
        if current_id in ancestors:
            raise ValueError(f"Julich 层级存在环: {current_id}")
        if current_id not in nodes_by_id:
            raise ValueError(f"Julich 层级缺少节点: {current_id}")
        ancestors.append(current_id)
        parent_id = nodes_by_id[current_id].get("parent_id")
        current_id = str(parent_id) if parent_id is not None else None
    return ancestors


def resolve_julich_vpm_rh_members(
    spec: dict[str, Any] | None = None,
) -> tuple[str, list[int]]:
    """解析 VPM 右侧最近官方父组及其本地 ROI 成员。

    Parameters
    ----------
    spec : dict[str, Any] | None, optional
        已解析路径的 Julich atlas 规范；默认从 registry 加载。

    Returns
    -------
    tuple[str, list[int]]
        官方父组名称和按 index 排序的同侧本地 ROI 成员。

    Raises
    ------
    ValueError
        atlas 不匹配、目标节点无法精确匹配或没有同侧多成员祖先。

    Notes
    -----
    本函数复用 ``merge-roi`` 的 Julich 方法：本地名称精确对应官方叶节点，
    再沿 ``parent_id`` 选择最近的同侧多 ROI 祖先，不依赖关键词推断。
    """
    atlas_spec = spec or get_atlas_spec(JULICH_ATLAS_NAME)
    if atlas_spec.get("name") != JULICH_ATLAS_NAME:
        raise ValueError(f"VPM 合并不支持 atlas: {atlas_spec.get('name')}")
    snapshot = _load_julich_hierarchy()
    nodes_by_id = {str(node["id"]): node for node in snapshot["nodes"]}
    ids_by_name: dict[str, list[str]] = {}
    for node in snapshot["nodes"]:
        ids_by_name.setdefault(_normalize_name(str(node["name"])), []).append(
            str(node["id"])
        )
    matched_nodes: dict[int, str] = {}
    hemispheres: dict[int, str] = {}
    target_index: int | None = None
    for area in atlas_spec["areas"]:
        index = int(area["index"])
        local_name = str(area["label_en"])
        official_name, hemisphere = _julich_official_name(local_name)
        matches = ids_by_name.get(_normalize_name(official_name), [])
        if hemisphere is not None and len(matches) == 1:
            matched_nodes[index] = matches[0]
            hemispheres[index] = hemisphere
        if local_name == JULICH_VPM_RH_AREA:
            target_index = index
    if target_index is None:
        raise ValueError(f"Julich atlas 缺少目标脑区: {JULICH_VPM_RH_AREA}")
    if target_index not in matched_nodes:
        raise ValueError("Julich VPM 右侧无法与官方层级节点精确匹配")
    ancestors_by_index = {
        index: _ancestor_ids(node_id, nodes_by_id)
        for index, node_id in matched_nodes.items()
    }
    target_hemisphere = hemispheres[target_index]
    for candidate_id in ancestors_by_index[target_index][1:]:
        member_indices = sorted(
            index
            for index, ancestor_ids in ancestors_by_index.items()
            if hemispheres[index] == target_hemisphere and candidate_id in ancestor_ids
        )
        if len(member_indices) >= 2:
            parent_name = str(nodes_by_id[candidate_id]["name"])
            logger.info(
                "Julich VPM 右侧命中父组 %s，合并成员=%s",
                parent_name,
                member_indices,
            )
            return parent_name, member_indices
    raise ValueError("Julich VPM 右侧没有同侧多 ROI 祖先")


def build_julich_vpm_rh_merged_mask(
    output_path: Path,
    spec: dict[str, Any] | None = None,
) -> Path:
    """生成 Julich VPM 右侧父组的二值并集 mask。

    Parameters
    ----------
    output_path : Path
        合并 mask 输出路径，必须以 ``.nii.gz`` 结尾。
    spec : dict[str, Any] | None, optional
        已解析路径的 Julich atlas 规范；默认从 registry 加载。

    Returns
    -------
    Path
        已写出的合并 mask 绝对路径。

    Raises
    ------
    FileNotFoundError
        任一成员 ROI mask 不存在。
    ValueError
        输出后缀、成员 mask 二值语义或 NIfTI 空间不符合要求。

    Notes
    -----
    本函数主要用于离线生成。合并只使用逐体素逻辑 OR，原始 atlas 和单
    ROI 文件不会被修改；在线逆向仿真仅在固定结果缺失时调用一次。
    """
    if not output_path.name.endswith(".nii.gz"):
        raise ValueError("Julich VPM 合并 mask 必须使用 .nii.gz 后缀")
    atlas_spec = spec or get_atlas_spec(JULICH_ATLAS_NAME)
    parent_name, member_indices = resolve_julich_vpm_rh_members(atlas_spec)
    area_by_index = {int(area["index"]): area for area in atlas_spec["areas"]}
    first_image: nib.Nifti1Image | None = None
    merged_mask: npt.NDArray[np.bool_] | None = None
    for index in member_indices:
        source_path = Path(area_by_index[index]["roi_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Julich 合并成员 ROI 不存在: {source_path}")
        image = nib.load(str(source_path))
        source_data = np.asanyarray(image.dataobj)
        if not np.all(np.isin(np.unique(source_data), [0, 1])):
            raise ValueError(f"Julich 合并成员 ROI 不是二值 mask: {source_path}")
        source_mask = source_data > 0
        if first_image is None:
            first_image = image
            merged_mask = source_mask.copy()
        else:
            if image.shape != first_image.shape or not np.allclose(
                image.affine, first_image.affine
            ):
                raise ValueError(f"Julich 合并成员 ROI 空间不一致: {source_path}")
            merged_mask |= source_mask
    if first_image is None or merged_mask is None:
        raise ValueError("Julich VPM 右侧没有可合并的成员 mask")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.nii.gz"
    )
    header = first_image.header.copy()
    header.set_data_dtype(np.uint8)
    try:
        nib.save(
            nib.Nifti1Image(
                merged_mask.astype(np.uint8),
                first_image.affine,
                header,
            ),
            str(temporary_path),
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info(
        "Julich VPM 右侧父组 %s 合并 mask 已生成: %s",
        parent_name,
        output_path,
    )
    return output_path
