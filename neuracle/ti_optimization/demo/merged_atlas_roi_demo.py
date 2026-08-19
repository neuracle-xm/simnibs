"""
合并 Atlas ROI 试验 Demo。

Demo 固定选择七个 atlas 中体积最小的若干 ROI，执行外置自动分组、
MNI mask 并集校验和可选的个体 WM/GM mesh 匹配。验收不通过时仅记录，
不会换样或修改合并规则。本文件不执行完整 TI 优化。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import nibabel as nib
import numpy as np
import numpy.typing as npt
from PIL import Image

from neuracle.atlas.registry import SIMNIBS_MNI_TEMPLATE
from neuracle.atlas.roi_merge_experiment import (
    DIFUMO64_ATLAS_NAME,
    JULICH_HIERARCHY_PATH,
    atlas_specs_by_name,
    build_experiment_samples,
    write_merged_mask,
)
from neuracle.utils.constants import DATA_ROOT
from simnibs.mesh_tools import mesh_io
from simnibs.utils import file_finder
from simnibs.utils.mesh_element_properties import ElementTags
from simnibs.utils.region_of_interest import RegionOfInterest

logger = logging.getLogger(__name__)


EXPECTED_VOLUME_MM3 = {
    ("BN_Atlas_246_1mm", 213): 661.0,
    ("BN_Atlas_246_1mm", 117): 785.0,
    ("BN_Atlas_246_1mm", 214): 1018.0,
    ("BN_Atlas_246_1mm", 235): 1054.0,
    ("BN_Atlas_246_1mm", 116): 1077.0,
    ("JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152", 17): 5.0,
    ("JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152", 385): 7.0,
    ("JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152", 178): 9.0,
    ("JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152", 130): 18.0,
    ("JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152", 61): 20.0,
    ("DiFuMo64", 4): 8917.0,
    ("DiFuMo64", 60): 9661.0,
    ("DiFuMo64", 18): 10320.0,
    ("DiFuMo64", 3): 12035.0,
    ("DiFuMo64", 54): 12111.0,
    ("DiFuMo128", 52): 5874.0,
    ("DiFuMo128", 72): 6278.0,
    ("DiFuMo128", 121): 6691.0,
    ("DiFuMo128", 16): 7047.0,
    ("DiFuMo128", 33): 7097.0,
    ("DiFuMo256", 78): 3631.0,
    ("DiFuMo256", 195): 3700.0,
    ("DiFuMo256", 218): 3730.0,
    ("DiFuMo256", 15): 3765.0,
    ("DiFuMo256", 151): 4059.0,
    ("DiFuMo512", 216): 1653.0,
    ("DiFuMo512", 331): 1814.0,
    ("DiFuMo512", 47): 1846.0,
    ("DiFuMo512", 432): 1854.0,
    ("DiFuMo512", 413): 1900.0,
    ("DiFuMo1024", 288): 863.0,
    ("DiFuMo1024", 621): 913.0,
    ("DiFuMo1024", 885): 1011.0,
    ("DiFuMo1024", 998): 1013.0,
    ("DiFuMo1024", 20): 1014.0,
}


def _build_parser() -> argparse.ArgumentParser:
    """
    构建 Demo 命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        命令行解析器。
    """
    parser = argparse.ArgumentParser(
        description="验证 Atlas 小 ROI 的外置自动合并和个体 WM/GM 匹配"
    )
    parser.add_argument(
        "--subject-dir",
        type=Path,
        default=DATA_ROOT / "m2m_ernie",
        help="个体 m2m 目录，默认 data/m2m_ernie",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "roi_merge_demo_outputs",
        help="各次 run_id 输出目录的父目录",
    )
    parser.add_argument(
        "--samples-per-atlas",
        type=int,
        default=5,
        help="每个 atlas 按体积选择的最小 ROI 数量，默认 5",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="只执行 MNI 合并与校验，不读取个体头模",
    )
    return parser


def _safe_token(value: str) -> str:
    """
    将字符串转换为安全的输出目录名称。

    Parameters
    ----------
    value : str
        原始字符串。

    Returns
    -------
    str
        文件名安全 token。
    """
    token = re_sub_non_filename(value)
    return token or "atlas"


def re_sub_non_filename(value: str) -> str:
    """
    替换文件名中的非安全字符。

    Parameters
    ----------
    value : str
        原始字符串。

    Returns
    -------
    str
        替换后的字符串。
    """
    result = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value.strip()
    )
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def _mask_volume(image: nib.Nifti1Image) -> float:
    """
    计算二值 NIfTI 的非零体积。

    Parameters
    ----------
    image : nib.Nifti1Image
        ROI NIfTI。

    Returns
    -------
    float
        非零体积 mm³。
    """
    mask = np.asanyarray(image.dataobj) > 0
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    return float(mask.sum() * voxel_volume)


def _render_panel(
    background: npt.NDArray[np.float32],
    source_mask: npt.NDArray[np.bool_],
    result_mask: npt.NDArray[np.bool_],
) -> Image.Image:
    """
    渲染单个切面，红色为源 ROI，绿色为结果 ROI。

    Parameters
    ----------
    background : numpy.ndarray
        灰度背景切面。
    source_mask : numpy.ndarray
        源 ROI 切面。
    result_mask : numpy.ndarray
        结果 ROI 切面。

    Returns
    -------
    PIL.Image.Image
        256×256 RGB 图像。
    """
    normalized = background.astype(np.float32)
    normalized -= float(normalized.min())
    maximum = float(normalized.max())
    if maximum > 0:
        normalized /= maximum
    rgb = np.repeat((normalized * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[result_mask] = (
        0.45 * rgb[result_mask] + 0.55 * np.array([40, 220, 80])
    ).astype(np.uint8)
    rgb[source_mask] = (
        0.20 * rgb[source_mask] + 0.80 * np.array([255, 60, 40])
    ).astype(np.uint8)
    rotated = np.rot90(rgb)
    return Image.fromarray(rotated).resize((256, 256), Image.Resampling.BILINEAR)


def _save_overlay(
    source_path: Path,
    result_path: Path | None,
    template_data: npt.NDArray[np.float32],
    output_path: Path,
) -> None:
    """
    保存源 ROI 和结果 ROI 的三正交切面叠加图。

    Parameters
    ----------
    source_path : Path
        源 ROI NIfTI。
    result_path : Path | None
        合并或 keep 后的结果 NIfTI；无结果时为 None。
    template_data : numpy.ndarray
        SimNIBS MNI 模板数据。
    output_path : Path
        PNG 输出路径。
    """
    source_image = nib.load(str(source_path))
    source_mask = np.asanyarray(source_image.dataobj) > 0
    result_mask = (
        np.asanyarray(nib.load(str(result_path)).dataobj) > 0
        if result_path is not None
        else np.zeros_like(source_mask)
    )
    if template_data.shape != source_mask.shape:
        raise ValueError("MNI 模板与 ROI shape 不一致，无法生成 overlay")
    combined = source_mask | result_mask
    coordinates = np.argwhere(combined)
    center = (
        np.rint(coordinates.mean(axis=0)).astype(int)
        if coordinates.size
        else np.array(source_mask.shape) // 2
    )
    x, y, z = (int(value) for value in center)
    panels = [
        _render_panel(
            template_data[:, :, z], source_mask[:, :, z], result_mask[:, :, z]
        ),
        _render_panel(
            template_data[:, y, :], source_mask[:, y, :], result_mask[:, y, :]
        ),
        _render_panel(
            template_data[x, :, :], source_mask[x, :, :], result_mask[x, :, :]
        ),
    ]
    canvas = Image.new("RGB", (256 * len(panels), 256), "white")
    for panel_index, panel in enumerate(panels):
        canvas.paste(panel, (panel_index * 256, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _append_issue(record: dict[str, Any], issue: str) -> None:
    """
    向样本追加不重复的验收问题。

    Parameters
    ----------
    record : dict[str, Any]
        样本报告。
    issue : str
        问题代码。
    """
    if issue not in record["acceptance_issues"]:
        record["acceptance_issues"].append(issue)


def _prepare_offline_record(
    sample: dict[str, Any],
    spec: dict[str, Any],
    run_dir: Path,
    template_data: npt.NDArray[np.float32],
) -> dict[str, Any]:
    """
    生成并验证单个样本的 MNI 结果 mask。

    Parameters
    ----------
    sample : dict[str, Any]
        自动选样和决策记录。
    spec : dict[str, Any]
        源 atlas 规范。
    run_dir : Path
        本次 demo 输出目录。
    template_data : numpy.ndarray
        SimNIBS MNI 模板数据。

    Returns
    -------
    dict[str, Any]
        包含验收问题、结果路径和 overlay 的样本报告。
    """
    record = dict(sample)
    record["acceptance_issues"] = []
    record["offline_processing_status"] = "completed"
    record["subject_matching_status"] = "not_run"
    record["match_status"] = None
    expected_volume = EXPECTED_VOLUME_MM3.get(
        (str(record["atlas_name"]), int(record["source_index"]))
    )
    record["expected_source_volume_mm3"] = expected_volume
    if expected_volume is None:
        _append_issue(record, "missing_documented_volume_baseline")
    elif not np.isclose(float(record["source_volume_mm3"]), expected_volume):
        _append_issue(record, "source_volume_baseline_drift")
    expected_decision = (
        "keep" if record["atlas_name"] == DIFUMO64_ATLAS_NAME else "merged"
    )
    record["expected_decision"] = expected_decision
    if record["decision"] != expected_decision:
        _append_issue(record, f"expected_{expected_decision}_got_{record['decision']}")
    if record.get("mapping_issue"):
        _append_issue(record, str(record["mapping_issue"]))
    source_path = Path(str(record["source_roi_path"]))
    result_path: Path | None = None
    try:
        source_image = nib.load(str(source_path))
        source_mask_volume = _mask_volume(source_image)
        record["actual_source_mask_volume_mm3"] = source_mask_volume
        if not np.isclose(source_mask_volume, float(record["source_volume_mm3"])):
            _append_issue(record, "source_roi_volume_disagrees_with_atlas")
        if source_mask_volume <= 0:
            _append_issue(record, "source_mask_is_empty")
        if record["decision"] == "merged":
            member_token = "_".join(str(index) for index in record["member_indices"])
            result_path = (
                run_dir
                / "merged_masks"
                / _safe_token(str(record["atlas_name"]))
                / f"merged_{member_token}.nii.gz"
            )
            validation = write_merged_mask(
                spec,
                [int(index) for index in record["member_indices"]],
                result_path,
            )
            record.update(validation)
            if not validation["union_matches"]:
                _append_issue(record, "merged_mask_is_not_member_union")
            if not validation["spatial_matches"]:
                _append_issue(record, "merged_mask_space_mismatch")
            if not validation["result_is_binary"]:
                _append_issue(record, "merged_mask_is_not_binary")
            if validation["result_dtype"] != "uint8":
                _append_issue(record, "merged_mask_dtype_is_not_uint8")
            if validation["actual_merged_volume_mm3"] <= source_mask_volume:
                _append_issue(record, "merged_mask_has_no_volume_gain")
        elif record["decision"] == "keep":
            result_path = source_path
            record.update(
                {
                    "result_mask_path": str(source_path.resolve()),
                    "actual_merged_volume_mm3": source_mask_volume,
                    "result_shape": list(source_image.shape),
                    "result_dtype": str(source_image.get_data_dtype()),
                    "result_is_binary": bool(
                        np.all(
                            np.isin(
                                np.unique(np.asanyarray(source_image.dataobj)),
                                [0, 1],
                            )
                        )
                    ),
                    "union_matches": True,
                    "spatial_matches": True,
                }
            )
        else:
            record["result_mask_path"] = None
            _append_issue(record, "no_valid_result_mask")
        overlay_path = (
            run_dir
            / "overlays"
            / _safe_token(str(record["atlas_name"]))
            / f"{int(record['source_index']):04d}_overlay.png"
        )
        _save_overlay(source_path, result_path, template_data, overlay_path)
        record["overlay_path"] = str(overlay_path.resolve())
    except Exception as exc:
        logger.exception(
            "样本离线处理失败: atlas=%s, index=%s",
            record["atlas_name"],
            record["source_index"],
        )
        record["offline_processing_status"] = "recorded_error"
        record["offline_error"] = f"{type(exc).__name__}: {exc}"
        record["result_mask_path"] = None
        _append_issue(record, "offline_processing_error")
    record["offline_acceptance_status"] = (
        "passed" if not record["acceptance_issues"] else "recorded_issues"
    )
    return record


def _count_wm_gm_elements(
    mask_path: Path,
    subject_dir: Path,
    mesh: Any,
) -> int:
    """
    统计 MNI mask 在个体头模中的 WM/GM 四面体元素数。

    Parameters
    ----------
    mask_path : Path
        MNI 空间二值 ROI。
    subject_dir : Path
        个体 m2m 目录。
    mesh : Any
        已加载的 SimNIBS 头模。

    Returns
    -------
    int
        与 mask 相交的 WM/GM 四面体元素数。
    """
    roi = RegionOfInterest()
    roi.method = "volume"
    roi.mesh = mesh
    roi.subpath = str(subject_dir)
    roi.tissues = [ElementTags.WM, ElementTags.GM]
    roi.mask_path = str(mask_path)
    roi.mask_space = "mni"
    roi.mask_value = 1
    return int(len(roi.get_nodes()))


def _apply_subject_matching(
    records: list[dict[str, Any]],
    subject_dir: Path,
) -> None:
    """
    为所有有效结果执行个体 WM/GM 实际匹配。

    Parameters
    ----------
    records : list[dict[str, Any]]
        离线样本报告；本函数原位追加个体字段。
    subject_dir : Path
        个体 m2m 目录。
    """
    subject_files = file_finder.SubjectFiles(subpath=str(subject_dir))
    mesh_path = Path(subject_files.fnamehead)
    if not mesh_path.exists():
        raise FileNotFoundError(f"未找到个体头模: {mesh_path}")
    mesh = mesh_io.read_msh(str(mesh_path))
    element_count_cache: dict[str, int] = {}
    for record in records:
        result_path_value = record.get("result_mask_path")
        if not result_path_value:
            record["subject_matching_status"] = "skipped_no_valid_result_mask"
            continue
        source_path = Path(str(record["source_roi_path"])).resolve()
        result_path = Path(str(result_path_value)).resolve()
        try:
            source_key = str(source_path)
            result_key = str(result_path)
            if source_key not in element_count_cache:
                element_count_cache[source_key] = _count_wm_gm_elements(
                    source_path, subject_dir, mesh
                )
            if result_key not in element_count_cache:
                element_count_cache[result_key] = _count_wm_gm_elements(
                    result_path, subject_dir, mesh
                )
            source_count = element_count_cache[source_key]
            result_count = element_count_cache[result_key]
            record["subject_dir"] = str(subject_dir.resolve())
            record["subject_mesh_path"] = str(mesh_path.resolve())
            record["source_wm_gm_element_count"] = source_count
            record["result_wm_gm_element_count"] = result_count
            record["wm_gm_element_gain"] = result_count - source_count
            record["wm_gm_element_ratio"] = (
                result_count / source_count if source_count > 0 else None
            )
            if result_count == 0:
                record["match_status"] = "empty"
            elif result_count <= source_count:
                record["match_status"] = "no_gain"
            else:
                record["match_status"] = "expanded"
            record["subject_matching_status"] = "completed"
        except Exception as exc:
            logger.exception(
                "样本个体匹配失败: atlas=%s, index=%s",
                record["atlas_name"],
                record["source_index"],
            )
            record["subject_matching_status"] = "recorded_error"
            record["subject_matching_error"] = f"{type(exc).__name__}: {exc}"
            _append_issue(record, "subject_matching_error")


def _summarize(
    records: list[dict[str, Any]],
    offline_only: bool,
) -> dict[str, Any]:
    """
    汇总决策、验收和个体匹配状态。

    Parameters
    ----------
    records : list[dict[str, Any]]
        全部样本报告。
    offline_only : bool
        是否跳过了个体阶段。

    Returns
    -------
    dict[str, Any]
        汇总统计。
    """
    decision_counts = Counter(str(record["decision"]) for record in records)
    issue_records = [record for record in records if record["acceptance_issues"]]
    matching_counts = Counter(
        str(record["subject_matching_status"]) for record in records
    )
    return {
        "status": "completed_with_recorded_issues" if issue_records else "passed",
        "offline_only": offline_only,
        "sample_count": len(records),
        "atlas_count": len({record["atlas_name"] for record in records}),
        "decision_counts": dict(decision_counts),
        "samples_passing_all_acceptance_items": len(records) - len(issue_records),
        "samples_with_recorded_issues": len(issue_records),
        "subject_matching_status_counts": dict(matching_counts),
    }


def _write_reports(
    run_dir: Path,
    report: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """
    写出 JSON 和 CSV 报告。

    Parameters
    ----------
    run_dir : Path
        本次输出目录。
    report : dict[str, Any]
        JSON 总报告。
    records : list[dict[str, Any]]
        CSV 样本行。

    Returns
    -------
    tuple[Path, Path]
        JSON、CSV 路径。
    """
    json_path = run_dir / "roi_merge_report.json"
    csv_path = run_dir / "roi_merge_report.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in record.items()
                }
            )
    return json_path, csv_path


def main() -> None:
    """执行合并 ROI demo，但不运行完整 TI 优化。"""
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = args.output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("构建每个 atlas 的最小 ROI 自动合并决策")
    samples = build_experiment_samples(args.samples_per_atlas)
    specs = atlas_specs_by_name()
    template_image = nib.load(str(SIMNIBS_MNI_TEMPLATE))
    template_data = np.asanyarray(template_image.dataobj, dtype=np.float32)
    records = [
        _prepare_offline_record(
            sample,
            specs[str(sample["atlas_name"])],
            run_dir,
            template_data,
        )
        for sample in samples
    ]
    if args.offline_only:
        for record in records:
            record["subject_matching_status"] = "not_run_offline_only"
    else:
        logger.info("开始个体 WM/GM 实际匹配: %s", args.subject_dir)
        _apply_subject_matching(records, args.subject_dir.resolve())
    summary = _summarize(records, args.offline_only)
    hierarchy = json.loads(JULICH_HIERARCHY_PATH.read_text(encoding="utf-8"))
    report = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "phase_five_optimization_run": False,
        "configuration": {
            "samples_per_atlas": args.samples_per_atlas,
            "subject_dir": str(args.subject_dir.resolve()),
            "output_dir": str(run_dir),
            "offline_only": args.offline_only,
        },
        "julich_hierarchy_source": {
            "atlas_version": hierarchy["atlas_version"],
            "source": hierarchy["source"],
            "nodes_sha256": hierarchy["nodes_sha256"],
        },
        "summary": summary,
        "samples": records,
    }
    json_path, csv_path = _write_reports(run_dir, report, records)
    print(f"run_id: {run_id}")
    print(f"status: {summary['status']}")
    print(f"samples: {summary['sample_count']}")
    print(f"recorded issues: {summary['samples_with_recorded_issues']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
