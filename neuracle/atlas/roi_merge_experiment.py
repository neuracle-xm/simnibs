"""
Atlas ROI 外置合并试验工具。

本模块只为合并 ROI demo 提供内部函数，不修改 atlas registry、标准化
ROI 或 TI 逆向优化业务入口。BNA 使用官方 Gyrus 映射，Julich 使用
官方 v3.1 区域树，DiFuMo128～1024 使用相邻分辨率的实际体素重叠，
DiFuMo64 固定作为 keep 对照。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import numpy.typing as npt

from neuracle.atlas.loader import iter_atlas_specs
from neuracle.atlas.registry import ATLAS_ROOT, load_atlas_registry

logger = logging.getLogger(__name__)


BNA_ATLAS_NAME = "BN_Atlas_246_1mm"
JULICH_ATLAS_NAME = "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
DIFUMO64_ATLAS_NAME = "DiFuMo64"
ATLAS_NAMES = (
    BNA_ATLAS_NAME,
    JULICH_ATLAS_NAME,
    DIFUMO64_ATLAS_NAME,
    "DiFuMo128",
    "DiFuMo256",
    "DiFuMo512",
    "DiFuMo1024",
)
DIFUMO_PARENT_ATLAS = {
    "DiFuMo128": "DiFuMo64",
    "DiFuMo256": "DiFuMo128",
    "DiFuMo512": "DiFuMo256",
    "DiFuMo1024": "DiFuMo512",
}
JULICH_HIERARCHY_PATH = (
    ATLAS_ROOT / "merge_metadata" / "julich_brain_v3_1_hierarchy.json"
)
JULICH_EXPECTED_VERSION = "3.1.0"
XLSX_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _normalize_name(value: str) -> str:
    """
    生成名称精确匹配键。

    Parameters
    ----------
    value : str
        原始名称。

    Returns
    -------
    str
        仅统一空白和大小写后的名称。
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


def _julich_official_name(local_name: str) -> tuple[str, str | None]:
    """
    把本地 Julich 侧别后缀转换为官方名称格式。

    Parameters
    ----------
    local_name : str
        本地 Julich 标签名。

    Returns
    -------
    tuple[str, str | None]
        官方完整名称和侧别；无法识别侧别时返回 None。
    """
    if local_name.endswith("_lh"):
        return f"{local_name[:-3]} - left hemisphere", "left"
    if local_name.endswith("_rh"):
        return f"{local_name[:-3]} - right hemisphere", "right"
    return local_name, None


def _load_discrete_atlas(
    spec: dict[str, Any],
) -> tuple[nib.Nifti1Image, npt.NDArray[np.int32], float]:
    """
    加载标准化离散 atlas。

    Parameters
    ----------
    spec : dict[str, Any]
        已解析路径的 atlas 规范。

    Returns
    -------
    tuple[nib.Nifti1Image, numpy.ndarray, float]
        NIfTI、整数标签数组和单体素体积 mm³。
    """
    image = nib.load(str(spec["standardized_atlas"]))
    data = np.rint(np.asanyarray(image.dataobj)).astype(np.int32)
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    return image, data, voxel_volume


def measure_atlas_volumes(spec: dict[str, Any]) -> dict[int, float]:
    """
    统计 atlas 中每个注册 ROI 的实际体积。

    Parameters
    ----------
    spec : dict[str, Any]
        已解析路径的 atlas 规范。

    Returns
    -------
    dict[int, float]
        ROI index 到体积 mm³ 的映射。
    """
    _, data, voxel_volume = _load_discrete_atlas(spec)
    max_index = max(int(area["index"]) for area in spec["areas"])
    counts = np.bincount(data.ravel(), minlength=max_index + 1)
    return {
        int(area["index"]): float(counts[int(area["index"])] * voxel_volume)
        for area in spec["areas"]
    }


def _xlsx_column(cell_reference: str) -> str:
    """
    从 XLSX 单元格引用中提取列名。

    Parameters
    ----------
    cell_reference : str
        例如 ``A1`` 的单元格引用。

    Returns
    -------
    str
        列名。
    """
    match = re.match(r"[A-Z]+", cell_reference)
    return match.group(0) if match else ""


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """
    使用标准库读取简单 XLSX 工作表。

    Parameters
    ----------
    path : Path
        XLSX 文件路径。

    Returns
    -------
    list[dict[str, str]]
        以首行为字段名的行字典。
    """
    namespace = {"m": XLSX_NAMESPACE}
    with ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(text.text or "" for text in item.findall(".//m:t", namespace))
            for item in shared_root.findall("m:si", namespace)
        ]
        sheet_root = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet1.xml")
        )
    raw_rows: list[dict[str, str]] = []
    for row in sheet_root.findall(".//m:sheetData/m:row", namespace):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", namespace):
            column = _xlsx_column(cell.attrib.get("r", ""))
            value_node = cell.find("m:v", namespace)
            value = (
                ""
                if value_node is None or value_node.text is None
                else value_node.text
            )
            if cell.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            values[column] = value
        raw_rows.append(values)
    if not raw_rows:
        return []
    headers = raw_rows[0]
    return [
        {header: row.get(column, "") for column, header in headers.items()}
        for row in raw_rows[1:]
    ]


def _decision_template(source_index: int) -> dict[str, Any]:
    """
    创建统一的自动决策字段。

    Parameters
    ----------
    source_index : int
        源 ROI index。

    Returns
    -------
    dict[str, Any]
        默认 unmatched 决策。
    """
    return {
        "source_index": source_index,
        "parent_source": None,
        "parent_atlas_name": None,
        "parent_index": None,
        "parent_name": None,
        "hierarchy_parent_id": None,
        "coverage": None,
        "dice": None,
        "overlap_volume_mm3": None,
        "second_best_coverage": None,
        "candidate_rankings": [],
        "matched_node_id": None,
        "matched_node_name": None,
        "hemisphere": None,
        "member_indices": [],
        "decision": "unmatched",
        "mapping_issue": None,
    }


def _build_bna_decisions(spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """
    根据 BNA 官方 Gyrus 表构建同侧合并组。

    Parameters
    ----------
    spec : dict[str, Any]
        BNA atlas 规范。

    Returns
    -------
    dict[int, dict[str, Any]]
        每个 ROI 的自动合并决策。
    """
    xlsx_path = Path(spec["raw_atlas"]).parent / "BNA_subregions.xlsx"
    rows = _read_xlsx_rows(xlsx_path)
    groups: dict[tuple[str, str], list[int]] = {}
    parent_names: dict[tuple[str, str], str] = {}
    current_gyrus = ""
    for row in rows:
        if row.get("Gyrus", "").strip():
            current_gyrus = row["Gyrus"].strip()
        if not current_gyrus:
            continue
        for column, hemisphere in (("Label ID.L", "left"), ("Label ID.R", "right")):
            raw_index = row.get(column, "").strip()
            if not raw_index:
                continue
            index = int(float(raw_index))
            key = (current_gyrus, hemisphere)
            groups.setdefault(key, []).append(index)
            parent_names[key] = current_gyrus
    decisions = {
        int(area["index"]): _decision_template(int(area["index"]))
        for area in spec["areas"]
    }
    for (gyrus, hemisphere), members in groups.items():
        sorted_members = sorted(members)
        gyrus_code = gyrus.split(",", 1)[0].strip()
        for index in sorted_members:
            decisions[index].update(
                {
                    "parent_source": "official_bna_gyrus",
                    "parent_atlas_name": BNA_ATLAS_NAME,
                    "parent_name": parent_names[(gyrus, hemisphere)],
                    "hierarchy_parent_id": f"bna:gyrus:{gyrus_code}:{hemisphere}",
                    "hemisphere": hemisphere,
                    "member_indices": sorted_members,
                    "decision": "merged" if len(sorted_members) >= 2 else "keep",
                }
            )
    for index, decision in decisions.items():
        if decision["decision"] == "unmatched":
            decision["mapping_issue"] = "bna_index_not_found_in_official_gyrus_table"
            logger.warning("BNA ROI %s 未在官方 Gyrus 表中找到", index)
    return decisions


def _load_julich_hierarchy(path: Path = JULICH_HIERARCHY_PATH) -> dict[str, Any]:
    """
    加载并校验 Julich-Brain v3.1 层级快照。

    Parameters
    ----------
    path : Path, optional
        层级快照路径。

    Returns
    -------
    dict[str, Any]
        已通过版本和节点哈希校验的快照。

    Raises
    ------
    ValueError
        快照版本或节点哈希不符合预期。
    """
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    nodes_payload = json.dumps(
        snapshot["nodes"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual_hash = hashlib.sha256(nodes_payload).hexdigest()
    if snapshot.get("atlas_version") != JULICH_EXPECTED_VERSION:
        raise ValueError(
            f"Julich 层级版本不匹配: {snapshot.get('atlas_version')}"
        )
    if actual_hash != snapshot.get("nodes_sha256"):
        raise ValueError("Julich 层级节点哈希校验失败")
    return snapshot


def _ancestor_ids(node_id: str, nodes_by_id: dict[str, dict[str, Any]]) -> list[str]:
    """
    返回节点自身到根节点的 ID 链。

    Parameters
    ----------
    node_id : str
        起始节点 ID。
    nodes_by_id : dict[str, dict[str, Any]]
        官方节点索引。

    Returns
    -------
    list[str]
        从自身到根节点的 ID 列表。
    """
    ancestors: list[str] = []
    current_id: str | None = node_id
    while current_id is not None:
        ancestors.append(current_id)
        current_id = nodes_by_id[current_id]["parent_id"]
    return ancestors


def _build_julich_decisions(spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """
    使用 Julich-Brain v3.1 官方区域树构建合并组。

    Parameters
    ----------
    spec : dict[str, Any]
        Julich atlas 规范。

    Returns
    -------
    dict[int, dict[str, Any]]
        每个 ROI 的自动合并决策。
    """
    snapshot = _load_julich_hierarchy()
    nodes_by_id = {str(node["id"]): node for node in snapshot["nodes"]}
    ids_by_name: dict[str, list[str]] = {}
    for node in snapshot["nodes"]:
        ids_by_name.setdefault(_normalize_name(str(node["name"])), []).append(
            str(node["id"])
        )
    matched_nodes: dict[int, str] = {}
    hemispheres: dict[int, str] = {}
    match_issues: dict[int, str] = {}
    for area in spec["areas"]:
        index = int(area["index"])
        official_name, hemisphere = _julich_official_name(str(area["label_en"]))
        matches = ids_by_name.get(_normalize_name(official_name), [])
        if hemisphere is None:
            match_issues[index] = "julich_local_label_has_no_hemisphere_suffix"
        elif len(matches) == 0:
            match_issues[index] = "julich_official_name_not_found"
        elif len(matches) > 1:
            match_issues[index] = "julich_official_name_is_ambiguous"
        else:
            matched_nodes[index] = matches[0]
            hemispheres[index] = hemisphere
    ancestors_by_index = {
        index: _ancestor_ids(node_id, nodes_by_id)
        for index, node_id in matched_nodes.items()
    }
    decisions = {
        int(area["index"]): _decision_template(int(area["index"]))
        for area in spec["areas"]
    }
    for index, node_id in matched_nodes.items():
        hemisphere = hemispheres[index]
        selected_parent: str | None = None
        selected_members: list[int] = []
        decisions[index].update(
            {
                "matched_node_id": node_id,
                "matched_node_name": nodes_by_id[node_id]["name"],
                "hemisphere": hemisphere,
            }
        )
        for candidate_id in ancestors_by_index[index][1:]:
            members = sorted(
                member_index
                for member_index, ancestor_ids in ancestors_by_index.items()
                if hemispheres[member_index] == hemisphere
                and candidate_id in ancestor_ids
            )
            if len(members) >= 2:
                selected_parent = candidate_id
                selected_members = members
                break
        if selected_parent is None:
            decisions[index]["mapping_issue"] = "julich_no_same_side_multi_roi_ancestor"
            continue
        decisions[index].update(
            {
                "parent_source": "official_julich_v3_1_hierarchy",
                "parent_atlas_name": snapshot["atlas_name"],
                "parent_name": nodes_by_id[selected_parent]["name"],
                "hierarchy_parent_id": f"{selected_parent}:{hemisphere}",
                "member_indices": selected_members,
                "decision": "merged",
            }
        )
    for index, issue in match_issues.items():
        decisions[index]["mapping_issue"] = issue
        logger.warning("Julich ROI %s 自动匹配失败: %s", index, issue)
    return decisions


def _build_keep_decisions(spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """
    为对照 atlas 构建固定 keep 决策。

    Parameters
    ----------
    spec : dict[str, Any]
        atlas 规范。

    Returns
    -------
    dict[int, dict[str, Any]]
        每个 ROI 的 keep 决策。
    """
    decisions: dict[int, dict[str, Any]] = {}
    for area in spec["areas"]:
        index = int(area["index"])
        decision = _decision_template(index)
        decision.update(
            {
                "parent_source": "keep_baseline",
                "member_indices": [index],
                "decision": "keep",
            }
        )
        decisions[index] = decision
    return decisions


def _build_overlap_decisions(
    source_spec: dict[str, Any],
    parent_spec: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """
    根据两个离散 atlas 的实际体素重叠构建父区分组。

    Parameters
    ----------
    source_spec : dict[str, Any]
        较细源 atlas 规范。
    parent_spec : dict[str, Any]
        指定的较粗父 atlas 规范。

    Returns
    -------
    dict[int, dict[str, Any]]
        每个源 ROI 的匹配指标和自动决策。
    """
    source_image, source_data, voxel_volume = _load_discrete_atlas(source_spec)
    parent_image, parent_data, parent_voxel_volume = _load_discrete_atlas(parent_spec)
    if source_data.shape != parent_data.shape or not np.allclose(
        source_image.affine, parent_image.affine
    ):
        raise ValueError(
            f"{source_spec['name']} 与 {parent_spec['name']} 不在同一标准化网格"
        )
    if not np.isclose(voxel_volume, parent_voxel_volume):
        raise ValueError("源 atlas 与父 atlas 的体素体积不一致")
    source_max = max(int(area["index"]) for area in source_spec["areas"])
    parent_max = max(int(area["index"]) for area in parent_spec["areas"])
    source_counts = np.bincount(source_data.ravel(), minlength=source_max + 1)
    parent_counts = np.bincount(parent_data.ravel(), minlength=parent_max + 1)
    valid = (source_data > 0) & (parent_data > 0)
    pair_codes = source_data[valid] * (parent_max + 1) + parent_data[valid]
    overlaps = np.bincount(
        pair_codes,
        minlength=(source_max + 1) * (parent_max + 1),
    ).reshape(source_max + 1, parent_max + 1)
    parent_area_by_index = {
        int(area["index"]): area for area in parent_spec["areas"]
    }
    decisions: dict[int, dict[str, Any]] = {}
    selected_parent_by_source: dict[int, int] = {}
    for area in source_spec["areas"]:
        source_index = int(area["index"])
        decision = _decision_template(source_index)
        candidates: list[tuple[float, float, float, int]] = []
        for parent_index in parent_area_by_index:
            overlap_count = int(overlaps[source_index, parent_index])
            if overlap_count == 0:
                continue
            coverage = overlap_count / int(source_counts[source_index])
            dice = 2.0 * overlap_count / (
                int(source_counts[source_index]) + int(parent_counts[parent_index])
            )
            candidates.append(
                (coverage, dice, overlap_count * voxel_volume, parent_index)
            )
        candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        if candidates:
            coverage, dice, overlap_volume, parent_index = candidates[0]
            selected_parent_by_source[source_index] = parent_index
            decision.update(
                {
                    "parent_source": "spatial_overlap",
                    "parent_atlas_name": parent_spec["name"],
                    "parent_index": parent_index,
                    "parent_name": parent_area_by_index[parent_index]["label_en"],
                    "coverage": float(coverage),
                    "dice": float(dice),
                    "overlap_volume_mm3": float(overlap_volume),
                    "second_best_coverage": (
                        float(candidates[1][0]) if len(candidates) > 1 else None
                    ),
                    "candidate_rankings": [
                        {
                            "parent_index": candidate[3],
                            "parent_name": parent_area_by_index[candidate[3]][
                                "label_en"
                            ],
                            "coverage": float(candidate[0]),
                            "dice": float(candidate[1]),
                            "overlap_volume_mm3": float(candidate[2]),
                        }
                        for candidate in candidates
                    ],
                }
            )
        else:
            decision["mapping_issue"] = "no_positive_voxel_overlap"
        decisions[source_index] = decision
    groups: dict[int, list[int]] = {}
    for source_index, parent_index in selected_parent_by_source.items():
        groups.setdefault(parent_index, []).append(source_index)
    for source_index, parent_index in selected_parent_by_source.items():
        members = sorted(groups[parent_index])
        decisions[source_index]["member_indices"] = members
        decisions[source_index]["decision"] = (
            "merged" if len(members) >= 2 else "keep"
        )
    return decisions


def build_merge_decisions(
    specs: dict[str, dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    为七个 atlas 构建全部确定性自动决策。

    Parameters
    ----------
    specs : dict[str, dict[str, Any]]
        atlas 名称到已解析规范的映射。

    Returns
    -------
    dict[str, dict[int, dict[str, Any]]]
        atlas 和 ROI index 两级决策映射。
    """
    decisions = {
        BNA_ATLAS_NAME: _build_bna_decisions(specs[BNA_ATLAS_NAME]),
        JULICH_ATLAS_NAME: _build_julich_decisions(specs[JULICH_ATLAS_NAME]),
        DIFUMO64_ATLAS_NAME: _build_keep_decisions(specs[DIFUMO64_ATLAS_NAME]),
    }
    for source_name, parent_name in DIFUMO_PARENT_ATLAS.items():
        logger.info("计算 %s 到 %s 的实际体素重叠", source_name, parent_name)
        decisions[source_name] = _build_overlap_decisions(
            specs[source_name], specs[parent_name]
        )
    return decisions


def build_experiment_samples(samples_per_atlas: int = 5) -> list[dict[str, Any]]:
    """
    构建每个 atlas 体积最小的固定试验样本。

    Parameters
    ----------
    samples_per_atlas : int, optional
        每个 atlas 的样本数，默认 5。

    Returns
    -------
    list[dict[str, Any]]
        包含源 ROI、体积和自动决策的样本列表。

    Raises
    ------
    ValueError
        样本数小于 1 或 registry 缺少约定 atlas。
    """
    if samples_per_atlas < 1:
        raise ValueError("samples_per_atlas 必须大于 0")
    registry = load_atlas_registry()
    specs = {spec["name"]: spec for spec in iter_atlas_specs(registry)}
    missing = [name for name in ATLAS_NAMES if name not in specs]
    if missing:
        raise ValueError(f"atlas registry 缺少: {', '.join(missing)}")
    volumes = {name: measure_atlas_volumes(specs[name]) for name in ATLAS_NAMES}
    decisions = build_merge_decisions(specs)
    samples: list[dict[str, Any]] = []
    for atlas_name in ATLAS_NAMES:
        spec = specs[atlas_name]
        area_by_index = {int(area["index"]): area for area in spec["areas"]}
        selected_indices = sorted(
            area_by_index,
            key=lambda index: (volumes[atlas_name][index], index),
        )[:samples_per_atlas]
        for index in selected_indices:
            area = area_by_index[index]
            decision = dict(decisions[atlas_name][index])
            member_indices = list(decision["member_indices"])
            merged_volume = sum(
                volumes[atlas_name][member_index]
                for member_index in member_indices
            )
            source_volume = volumes[atlas_name][index]
            samples.append(
                {
                    "atlas_name": atlas_name,
                    "source_index": index,
                    "source_label_en": area["label_en"],
                    "source_label_zh": area["label_zh"],
                    "source_roi_path": area["roi_path"],
                    "source_volume_mm3": source_volume,
                    "merged_volume_mm3": merged_volume or None,
                    "volume_ratio": (
                        merged_volume / source_volume if merged_volume else None
                    ),
                    **decision,
                }
            )
    return samples


def write_merged_mask(
    spec: dict[str, Any],
    member_indices: list[int],
    output_path: Path,
) -> dict[str, Any]:
    """
    将同一源 atlas 的成员 mask 做逻辑并集并写出。

    Parameters
    ----------
    spec : dict[str, Any]
        源 atlas 规范。
    member_indices : list[int]
        要合并的源 ROI index。
    output_path : Path
        合并 NIfTI 输出路径。

    Returns
    -------
    dict[str, Any]
        输出路径、体积和空间校验结果。

    Raises
    ------
    ValueError
        成员为空或成员 NIfTI 空间不一致。
    """
    if not member_indices:
        raise ValueError("合并成员不能为空")
    area_by_index = {int(area["index"]): area for area in spec["areas"]}
    first_image: nib.Nifti1Image | None = None
    merged_mask: npt.NDArray[np.bool_] | None = None
    for index in member_indices:
        image = nib.load(str(area_by_index[index]["roi_path"]))
        mask = np.asanyarray(image.dataobj) > 0
        if first_image is None:
            first_image = image
            merged_mask = mask.copy()
        else:
            if image.shape != first_image.shape or not np.allclose(
                image.affine, first_image.affine
            ):
                raise ValueError(f"成员 ROI {index} 的 NIfTI 空间不一致")
            merged_mask |= mask
    if first_image is None or merged_mask is None:
        raise ValueError("没有可写出的合并 mask")
    header = first_image.header.copy()
    header.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(merged_mask.astype(np.uint8), first_image.affine, header),
        str(output_path),
    )
    saved_image = nib.load(str(output_path))
    saved_data = np.asanyarray(saved_image.dataobj)
    saved_binary = saved_data > 0
    voxel_volume = float(abs(np.linalg.det(first_image.affine[:3, :3])))
    return {
        "result_mask_path": str(output_path.resolve()),
        "actual_merged_volume_mm3": float(saved_binary.sum() * voxel_volume),
        "result_shape": list(saved_image.shape),
        "result_dtype": str(saved_image.get_data_dtype()),
        "result_is_binary": bool(np.all(np.isin(np.unique(saved_data), [0, 1]))),
        "union_matches": bool(np.array_equal(saved_binary, merged_mask)),
        "spatial_matches": bool(
            saved_image.shape == first_image.shape
            and np.allclose(saved_image.affine, first_image.affine)
        ),
    }


def atlas_specs_by_name() -> dict[str, dict[str, Any]]:
    """
    返回当前 registry 中已解析的 atlas 规范。

    Returns
    -------
    dict[str, dict[str, Any]]
        atlas 名称到规范的映射。
    """
    registry = load_atlas_registry()
    return {spec["name"]: spec for spec in iter_atlas_specs(registry)}
