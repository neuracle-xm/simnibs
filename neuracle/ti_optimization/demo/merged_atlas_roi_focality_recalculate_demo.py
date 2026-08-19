"""
使用合并前 ROI 定义重算 merged 结果的 focality。

本脚本只读取已经完成的合并前后优化结果，不重新运行优化、FEM 或
电极映射。original 和 merged focality 使用队列保存的优化目标值；
重算值使用 merged 根目录 mesh 的 ``max_TI`` 场，分别结合 original 的
``ROI``/``non-ROI``，以及 original 的 ``ROI``/merged 的 ``non-ROI``。
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT
from simnibs import mesh_io
from simnibs.optimization.tes_flex_optimization.measures import ROC

logger = logging.getLogger(__name__)


BATCH_OUTPUT_DIR = (
    DATA_ROOT / "roi_merge_focality_comparisons" / "small_roi_25_focality_3ma"
)
QUEUE_STATE_PATH = BATCH_OUTPUT_DIR / "queue_state.json"
CSV_OUTPUT_PATH = BATCH_OUTPUT_DIR / "focality_recalculated_comparison.csv"
JSON_OUTPUT_PATH = BATCH_OUTPUT_DIR / "focality_recalculated_comparison.json"
NON_ROI_THRESHOLD_V_PER_M = 0.1
ROI_THRESHOLD_V_PER_M = 0.2
FIELD_VALUE_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    """
    读取 JSON 对象。

    Parameters
    ----------
    path : Path
        JSON 文件路径。

    Returns
    -------
    dict[str, Any]
        JSON 顶层对象。
    """
    if not path.is_file():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """
    原子写入 JSON 报告。

    Parameters
    ----------
    path : Path
        输出路径。
    value : dict[str, Any]
        待写入的 JSON 对象。

    Returns
    -------
    None
        无返回值。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    原子写入逐 ROI CSV 结果。

    Parameters
    ----------
    path : Path
        输出路径。
    rows : list[dict[str, Any]]
        逐 ROI 结果。

    Returns
    -------
    None
        无返回值。
    """
    fieldnames = [
        "atlas_name",
        "source_index",
        "source_label_en",
        "original_status",
        "merged_status",
        "original_focality",
        "merged_focality",
        "merged_on_original_roi_focality",
        "merged_on_original_roi_merged_non_roi_focality",
        "merged_minus_original_focality",
        "recalculated_minus_original_focality",
        "recalculated_vs_original",
        "comparison_status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _find_optimization_mesh(job: dict[str, Any]) -> Path:
    """
    查找任务根目录中唯一的未映射优化 mesh。

    Parameters
    ----------
    job : dict[str, Any]
        队列任务记录。

    Returns
    -------
    Path
        未映射电极的优化结果 mesh。
    """
    result = job.get("result") or {}
    output_value = result.get("optimization_output_folder") or result.get("output_dir")
    if not output_value:
        raise ValueError(f"任务缺少优化输出目录: {job.get('job_id')}")
    output_dir = Path(str(output_value))
    candidates = sorted(output_dir.glob("*_tes_flex_opt_head_mesh.msh"))
    if len(candidates) != 1:
        raise ValueError(
            f"任务根目录应有且仅有一个未映射优化 mesh: {job.get('job_id')}, "
            f"实际 {len(candidates)} 个"
        )
    return candidates[0]


def _field_value(mesh: mesh_io.Msh, field_name: str) -> np.ndarray:
    """
    读取 mesh 中唯一的元素字段。

    Parameters
    ----------
    mesh : mesh_io.Msh
        SimNIBS mesh。
    field_name : str
        元素字段名称。

    Returns
    -------
    numpy.ndarray
        元素字段数组。
    """
    matches = [field.value for field in mesh.elmdata if field.field_name == field_name]
    if len(matches) != 1:
        raise ValueError(f"mesh 字段 {field_name} 应有且仅有一个，实际 {len(matches)} 个")
    value = np.asarray(matches[0])
    if value.ndim != 1 or value.shape[0] != mesh.elm.nr:
        raise ValueError(
            f"mesh 字段 {field_name} 维度不正确: {value.shape}, "
            f"元素数 {mesh.elm.nr}"
        )
    return value


def _validate_mesh_alignment(
    original_mesh: mesh_io.Msh,
    merged_mesh: mesh_io.Msh,
) -> None:
    """
    校验 original 和 merged mesh 的元素顺序一致。

    Parameters
    ----------
    original_mesh : mesh_io.Msh
        original 优化结果 mesh。
    merged_mesh : mesh_io.Msh
        merged 优化结果 mesh。

    Returns
    -------
    None
        无返回值。
    """
    if original_mesh.elm.nr != merged_mesh.elm.nr:
        raise ValueError(
            "original 和 merged mesh 元素数不一致: "
            f"{original_mesh.elm.nr} != {merged_mesh.elm.nr}"
        )
    checks = (
        ("元素编号", original_mesh.elm.elm_number, merged_mesh.elm.elm_number),
        ("元素类型", original_mesh.elm.elm_type, merged_mesh.elm.elm_type),
        ("组织标签 tag1", original_mesh.elm.tag1, merged_mesh.elm.tag1),
        ("组织标签 tag2", original_mesh.elm.tag2, merged_mesh.elm.tag2),
    )
    for name, original_value, merged_value in checks:
        if not np.array_equal(original_value, merged_value):
            raise ValueError(f"original 和 merged mesh 的{name}顺序不一致")


def _focality_goal_value(
    field: np.ndarray,
    roi_mask: np.ndarray,
    non_roi_mask: np.ndarray,
) -> float:
    """
    按 SimNIBS focality ROC 公式计算目标值。

    Parameters
    ----------
    field : numpy.ndarray
        每个 mesh 元素的 ``max_TI`` 场。
    roi_mask : numpy.ndarray
        合并前 ROI 元素掩膜。
    non_roi_mask : numpy.ndarray
        本次重算使用的 non-ROI 元素掩膜。

    Returns
    -------
    float
        与优化器同方向的 focality 目标值，越小越好。
    """
    roi_field = field[roi_mask]
    non_roi_field = field[non_roi_mask]
    if roi_field.size == 0:
        raise ValueError("合并前 ROI 在个体 mesh 中没有有效元素")
    if non_roi_field.size == 0:
        raise ValueError("合并前 non-ROI 在个体 mesh 中没有有效元素")
    if not np.all(np.isfinite(roi_field)) or not np.all(np.isfinite(non_roi_field)):
        raise ValueError("重算 focality 的 max_TI 场包含非有限值")
    roc_distance = ROC(
        e1=roi_field,
        e2=non_roi_field,
        threshold=[NON_ROI_THRESHOLD_V_PER_M, ROI_THRESHOLD_V_PER_M],
        focal=True,
    )
    return float(-100.0 * (np.sqrt(2.0) - roc_distance))


def _comparison_direction(recalculated: float, original: float) -> str:
    """
    判断重算 focality 相对 original 的方向。

    Parameters
    ----------
    recalculated : float
        merged 场在 original ROI 定义上的重算 focality。
    original : float
        original 优化记录的 focality。

    Returns
    -------
    str
        ``recalculated_better``、``original_better`` 或 ``equal``。
    """
    if np.isclose(recalculated, original, atol=FIELD_VALUE_TOLERANCE, rtol=0.0):
        return "equal"
    if recalculated < original:
        return "recalculated_better"
    return "original_better"


def _process_sample(
    sample: dict[str, Any],
    jobs_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    处理一个 original/merged 配对样本。

    Parameters
    ----------
    sample : dict[str, Any]
        队列中的 ROI 样本。
    jobs_by_id : dict[str, dict[str, Any]]
        任务 ID 到任务记录的映射。

    Returns
    -------
    dict[str, Any]
        三类 focality 和处理状态。
    """
    original_job = jobs_by_id[sample["condition_jobs"]["original"]]
    merged_job = jobs_by_id[sample["condition_jobs"]["merged"]]
    original_result = original_job.get("result") or {}
    merged_result = merged_job.get("result") or {}
    original_focality = original_result.get("optimization_goal_value")
    merged_focality = merged_result.get("optimization_goal_value")
    row: dict[str, Any] = {
        "atlas_name": sample["atlas_name"],
        "source_index": sample["source_index"],
        "source_label_en": sample["source_label_en"],
        "original_status": original_job.get("status"),
        "merged_status": merged_job.get("status"),
        "original_focality": original_focality,
        "merged_focality": merged_focality,
        "merged_on_original_roi_focality": None,
        "merged_on_original_roi_merged_non_roi_focality": None,
        "merged_minus_original_focality": None,
        "recalculated_minus_original_focality": None,
        "recalculated_vs_original": None,
        "comparison_status": "failed",
        "error": None,
    }
    if original_focality is not None and merged_focality is not None:
        row["merged_minus_original_focality"] = float(
            merged_focality - original_focality
        )
    if original_job.get("status") != "completed":
        error = original_result.get("optimization_error") or "original 任务未完成"
        row["error"] = str(error)
        return row
    if merged_job.get("status") != "completed":
        error = merged_result.get("optimization_error") or "merged 任务未完成"
        row["error"] = str(error)
        return row
    try:
        original_mesh_path = _find_optimization_mesh(original_job)
        merged_mesh_path = _find_optimization_mesh(merged_job)
        original_mesh = mesh_io.read_msh(str(original_mesh_path))
        merged_mesh = mesh_io.read_msh(str(merged_mesh_path))
        _validate_mesh_alignment(original_mesh, merged_mesh)
        original_roi = _field_value(original_mesh, "ROI") > 0
        original_non_roi = _field_value(original_mesh, "non-ROI") > 0
        merged_non_roi = _field_value(merged_mesh, "non-ROI") > 0
        merged_field = _field_value(merged_mesh, "max_TI")
        recalculated = _focality_goal_value(
            merged_field,
            original_roi,
            original_non_roi,
        )
        recalculated_with_merged_non_roi = _focality_goal_value(
            merged_field,
            original_roi,
            merged_non_roi,
        )
        row["merged_on_original_roi_focality"] = recalculated
        row["merged_on_original_roi_merged_non_roi_focality"] = (
            recalculated_with_merged_non_roi
        )
        row["recalculated_minus_original_focality"] = float(
            recalculated - float(original_focality)
        )
        row["recalculated_vs_original"] = _comparison_direction(
            recalculated,
            float(original_focality),
        )
        row["comparison_status"] = "completed"
    except Exception as exc:
        logger.exception(
            "ROI focality 重算失败: %s, %s",
            sample["atlas_name"],
            sample["source_index"],
        )
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _comparison_statistics(
    rows: list[dict[str, Any]],
    value_field: str,
) -> dict[str, int]:
    """
    统计一种 focality 相对合并前结果的比较数量。

    Parameters
    ----------
    rows : list[dict[str, Any]]
        逐 ROI 重算结果。
    value_field : str
        需要与 ``original_focality`` 比较的字段名。

    Returns
    -------
    dict[str, int]
        可比较、当前方法更优、合并前更优、相同和无法比较数量。
    """
    statistics = {
        "comparable_count": 0,
        "method_better_count": 0,
        "original_better_count": 0,
        "equal_count": 0,
        "unavailable_count": 0,
    }
    for row in rows:
        original_value = row.get("original_focality")
        method_value = row.get(value_field)
        if original_value is None or method_value is None:
            statistics["unavailable_count"] += 1
            continue
        statistics["comparable_count"] += 1
        direction = _comparison_direction(float(method_value), float(original_value))
        if direction == "recalculated_better":
            statistics["method_better_count"] += 1
        elif direction == "original_better":
            statistics["original_better_count"] += 1
        else:
            statistics["equal_count"] += 1
    return statistics


def _build_report(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    构造 focality 重算 JSON 报告。

    Parameters
    ----------
    state : dict[str, Any]
        原批量队列状态。
    rows : list[dict[str, Any]]
        逐 ROI 重算结果。

    Returns
    -------
    dict[str, Any]
        包含配置、汇总和逐 ROI 结果的报告。
    """
    completed_rows = [row for row in rows if row["comparison_status"] == "completed"]
    recalculated_better_count = sum(
        row["recalculated_vs_original"] == "recalculated_better"
        for row in completed_rows
    )
    original_better_count = sum(
        row["recalculated_vs_original"] == "original_better"
        for row in completed_rows
    )
    equal_count = sum(
        row["recalculated_vs_original"] == "equal" for row in completed_rows
    )
    comparison_statistics = {
        "merged_focality_vs_original": _comparison_statistics(
            rows,
            "merged_focality",
        ),
        "merged_on_original_roi_focality_vs_original": _comparison_statistics(
            rows,
            "merged_on_original_roi_focality",
        ),
        "merged_on_original_roi_merged_non_roi_focality_vs_original": (
            _comparison_statistics(
                rows,
                "merged_on_original_roi_merged_non_roi_focality",
            )
        ),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_queue_state": str(QUEUE_STATE_PATH.resolve()),
        "configuration_fingerprint": state.get("configuration_fingerprint"),
        "threshold": [NON_ROI_THRESHOLD_V_PER_M, ROI_THRESHOLD_V_PER_M],
        "focality_lower_is_better": True,
        "recalculation_definition": (
            "merged max_TI on original ROI and original non-ROI"
        ),
        "merged_non_roi_recalculation_definition": (
            "merged max_TI on original ROI and merged non-ROI"
        ),
        "sample_count": len(rows),
        "completed_count": len(completed_rows),
        "failed_count": len(rows) - len(completed_rows),
        "recalculated_better_count": recalculated_better_count,
        "original_better_count": original_better_count,
        "equal_count": equal_count,
        "comparison_statistics": comparison_statistics,
        "comparisons": rows,
    }


def main() -> None:
    """
    重算全部 merged 结果在 original ROI 定义上的 focality。

    Returns
    -------
    None
        无返回值。
    """
    setup_logging(str(PROJECT_ROOT / "log" / "roi_merge_focality_recalculation"))
    state = _read_json(QUEUE_STATE_PATH)
    samples = state.get("samples")
    jobs = state.get("jobs")
    if not isinstance(samples, list) or not isinstance(jobs, list):
        raise TypeError("queue_state.json 缺少 samples 或 jobs 数组")
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    rows: list[dict[str, Any]] = []
    for position, sample in enumerate(samples, start=1):
        logger.info(
            "重算 focality [%d/%d]: %s, %s",
            position,
            len(samples),
            sample["atlas_name"],
            sample["source_index"],
        )
        rows.append(_process_sample(sample, jobs_by_id))
    report = _build_report(state, rows)
    _write_csv_atomic(CSV_OUTPUT_PATH, rows)
    _write_json_atomic(JSON_OUTPUT_PATH, report)
    print(f"samples: {report['sample_count']}")
    print(f"completed: {report['completed_count']}")
    print(f"failed: {report['failed_count']}")
    print(f"recalculated_better: {report['recalculated_better_count']}")
    print(f"original_better: {report['original_better_count']}")
    print(f"equal: {report['equal_count']}")
    print(f"csv: {CSV_OUTPUT_PATH}")
    print(f"json: {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
