"""
25 个已合并小 ROI 的 focality 逆向优化批量对比 Demo。

本 Demo 不接收命令行参数。它读取固定的合并报告，仅把实际判定为
``merged`` 的 25 个小 ROI 放入可恢复队列，最多同时运行 4 个逆向优化。
每个样本都运行 original 和 merged 两个条件，两条件除 ROI mask 外使用
完全相同的参数。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuracle.logger import setup_logging
from neuracle.ti_optimization import (
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils import cond_dict_to_list
from neuracle.utils.constants import (
    DATA_ROOT,
    ELECTRODE_RADIUS,
    PROJECT_ROOT,
    STANDARD_COND,
)
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


MERGE_REPORT_PATH = (
    DATA_ROOT
    / "roi_merge_demo_outputs"
    / "20260817_110807_8c9b0ef7"
    / "roi_merge_report.json"
)
SUBJECT_DIR = DATA_ROOT / "m2m_ernie"
BATCH_OUTPUT_DIR = (
    DATA_ROOT / "roi_merge_focality_comparisons" / "small_roi_25_focality_3ma"
)
NON_ROI_THRESHOLD_V_PER_M = 0.1
ROI_THRESHOLD_V_PER_M = 0.2
ELECTRODE_CURRENT_A = 0.003
OPTIMIZER_SEED = 42
MAX_CONCURRENT_OPTIMIZATIONS = 4
N_WORKERS_PER_OPTIMIZATION = 4
ANISOTROPY_TYPE = "vn"


FOCUS_ROIS = (
    ("BN_Atlas_246_1mm", 213, "lAmyg_L"),
    ("BN_Atlas_246_1mm", 117, "TI_L"),
    ("BN_Atlas_246_1mm", 214, "lAmyg_R"),
    ("BN_Atlas_246_1mm", 235, "Stha_L"),
    ("BN_Atlas_246_1mm", 116, "A28/34_R"),
    (
        "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152",
        17,
        "Area PirTB (PiriformCortexMesial, temporobasal)_lh",
    ),
    (
        "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152",
        385,
        "VPM (Thalamus, ventral posterior medial Nucleus)_rh",
    ),
    (
        "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152",
        178,
        "VPM (Thalamus, ventral posterior medial Nucleus)_lh",
    ),
    (
        "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152",
        130,
        "Area PirT (PiriformCortexMesial, temporal)_lh",
    ),
    (
        "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152",
        61,
        "Pv (Thalamus, paraventricular Nucleus)_lh",
    ),
    ("DiFuMo128", 121, "Precuneus posterior"),
    ("DiFuMo128", 33, "Posterior cingulate cortex"),
    ("DiFuMo256", 78, "Lingual gyrus mid-posterior"),
    ("DiFuMo256", 195, "Lingual gyrus medial"),
    ("DiFuMo256", 218, "Precuneus inferior"),
    ("DiFuMo256", 15, "Posterior cingulate cortex posterior"),
    ("DiFuMo256", 151, "Parieto-occipital sulcus anterior"),
    ("DiFuMo512", 216, "Precuneus postero-inferior RH"),
    ("DiFuMo512", 331, "Parieto-occipital sulcus middle"),
    ("DiFuMo512", 432, "Subparietal sulcus inferior LH"),
    ("DiFuMo512", 413, "Precuneus RH"),
    ("DiFuMo1024", 288, "Descending occipital gyrus superior"),
    ("DiFuMo1024", 885, "Middle temporal gyrus posterior inferior LH"),
    ("DiFuMo1024", 998, "Superior parietal lobule posterior LH"),
    ("DiFuMo1024", 20, "Parieto-occipital sulcus middle"),
)


def _safe_token(value: str) -> str:
    """
    将 atlas 名称转换为安全目录名。

    Parameters
    ----------
    value : str
        原始 atlas 名称。

    Returns
    -------
    str
        不包含路径分隔符的目录 token。
    """
    token = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value.strip()
    )
    return token.strip("_") or "atlas"


def _json_safe(value: Any) -> Any:
    """
    将 NumPy 和 Path 值转换为 JSON 可序列化对象。

    Parameters
    ----------
    value : Any
        原始值。

    Returns
    -------
    Any
        JSON 可序列化值。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """
    原子式写出 JSON 文件。

    Parameters
    ----------
    path : Path
        输出路径。
    value : dict[str, Any]
        JSON 内容。
    """
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _resolve_record_path(value: str, report_path: Path) -> Path:
    """
    解析合并报告中的文件路径。

    Parameters
    ----------
    value : str
        报告中的绝对或相对路径。
    report_path : Path
        合并报告路径。

    Returns
    -------
    Path
        解析后的绝对路径。
    """
    path = Path(value)
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def _mask_volume_mm3(path: Path) -> float:
    """
    计算二值 ROI 的非零体积。

    Parameters
    ----------
    path : Path
        ROI NIfTI 路径。

    Returns
    -------
    float
        ROI 体积 mm³。
    """
    image = nib.load(str(path))
    mask = np.asanyarray(image.dataobj) > 0
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    return float(mask.sum() * voxel_volume)


def _configuration() -> dict[str, Any]:
    """
    返回全部硬编码实验参数。

    Returns
    -------
    dict[str, Any]
        可写入队列状态的固定参数。
    """
    return {
        "merge_report_path": str(MERGE_REPORT_PATH.resolve()),
        "subject_dir": str(SUBJECT_DIR.resolve()),
        "batch_output_dir": str(BATCH_OUTPUT_DIR.resolve()),
        "goal": "focality",
        "non_roi_threshold_v_per_m": NON_ROI_THRESHOLD_V_PER_M,
        "roi_threshold_v_per_m": ROI_THRESHOLD_V_PER_M,
        "electrode_current_a": ELECTRODE_CURRENT_A,
        "electrode_radius_mm": ELECTRODE_RADIUS,
        "optimizer_seed": OPTIMIZER_SEED,
        "max_concurrent_optimizations": MAX_CONCURRENT_OPTIMIZATIONS,
        "workers_per_optimization": N_WORKERS_PER_OPTIMIZATION,
        "anisotropy_type": ANISOTROPY_TYPE,
        "conductivity_overrides_s_per_m": {"WM": 0.14, "GM": 0.30},
        "focus_rois": [list(item) for item in FOCUS_ROIS],
    }


def _configuration_fingerprint(configuration: dict[str, Any]) -> str:
    """
    计算固定参数和合并报告内容的联合指纹。

    Parameters
    ----------
    configuration : dict[str, Any]
        固定实验参数。

    Returns
    -------
    str
        SHA-256 十六进制指纹。
    """
    report_bytes = MERGE_REPORT_PATH.read_bytes()
    config_bytes = json.dumps(
        configuration,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(config_bytes + report_bytes).hexdigest()


def _validate_masks(
    source_path: Path,
    result_path: Path,
) -> dict[str, float | str | None]:
    """
    校验合并前后 mask 并统计体积。

    Parameters
    ----------
    source_path : Path
        原始 ROI。
    result_path : Path
        合并后的 ROI。

    Returns
    -------
    dict[str, float | str | None]
        路径和体积信息。
    """
    if not source_path.is_file():
        raise ValueError(f"原始 ROI 不存在: {source_path}")
    source_image = nib.load(str(source_path))
    source_mask = np.asanyarray(source_image.dataobj) > 0
    source_volume = _mask_volume_mm3(source_path)
    if not result_path.is_file():
        raise ValueError(f"结果 ROI 不存在: {result_path}")
    result_image = nib.load(str(result_path))
    if source_image.shape != result_image.shape or not np.allclose(
        source_image.affine, result_image.affine
    ):
        raise ValueError("原始 ROI 与结果 ROI 的 NIfTI 空间不一致")
    result_mask = np.asanyarray(result_image.dataobj) > 0
    if np.any(source_mask & ~result_mask):
        raise ValueError("结果 ROI 没有完整包含原始 ROI")
    result_volume = _mask_volume_mm3(result_path)
    if result_volume <= source_volume:
        raise ValueError("merged 样本的结果体积没有增大")
    return {
        "source_roi_path": str(source_path),
        "result_roi_path": str(result_path),
        "source_volume_mm3": source_volume,
        "result_volume_mm3": result_volume,
        "volume_ratio": result_volume / source_volume,
    }


def _load_focus_samples() -> list[dict[str, Any]]:
    """
    读取固定 25 个 merged ROI 并校验名称、决策和 mask。

    Returns
    -------
    list[dict[str, Any]]
        固定顺序的样本记录。
    """
    report_path = MERGE_REPORT_PATH.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_samples = {
        (str(sample["atlas_name"]), int(sample["source_index"])): sample
        for sample in report.get("samples", [])
    }
    samples: list[dict[str, Any]] = []
    for atlas_name, source_index, expected_label in FOCUS_ROIS:
        key = (atlas_name, source_index)
        if key not in report_samples:
            raise ValueError(f"合并报告缺少固定 ROI: {atlas_name}, {source_index}")
        source = report_samples[key]
        actual_label = str(source.get("source_label_en", ""))
        if actual_label != expected_label:
            raise ValueError(
                f"ROI 名称不匹配: {atlas_name}, {source_index}, "
                f"expected={expected_label}, actual={actual_label}"
            )
        decision = str(source.get("decision"))
        if decision != "merged":
            raise ValueError(
                f"比较队列只允许 merged ROI: {atlas_name}, "
                f"{source_index}, decision={decision}"
            )
        source_path = _resolve_record_path(
            str(source["source_roi_path"]), report_path
        )
        result_value = source.get("result_mask_path")
        if not result_value:
            raise ValueError(f"merged ROI 缺少结果 mask: {atlas_name}, {source_index}")
        result_path = _resolve_record_path(str(result_value), report_path)
        mask_info = _validate_masks(source_path, result_path)
        samples.append(
            {
                "atlas_name": atlas_name,
                "source_index": source_index,
                "source_label_en": actual_label,
                "decision": decision,
                "member_indices": source.get("member_indices", []),
                "parent_name": source.get("parent_name"),
                **mask_info,
            }
        )
    return samples


def _job_id(atlas_name: str, source_index: int, condition: str) -> str:
    """
    构建稳定的队列任务 ID。

    Parameters
    ----------
    atlas_name : str
        atlas 名称。
    source_index : int
        ROI index。
    condition : str
        original 或 merged。

    Returns
    -------
    str
        稳定任务 ID。
    """
    return f"{atlas_name}:{source_index}:{condition}"


def _build_initial_state(
    configuration: dict[str, Any],
    fingerprint: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    为固定样本生成首次队列状态。

    Parameters
    ----------
    configuration : dict[str, Any]
        固定实验参数。
    fingerprint : str
        参数和输入报告联合指纹。
    samples : list[dict[str, Any]]
        固定 ROI 样本。

    Returns
    -------
    dict[str, Any]
        可恢复队列状态。
    """
    jobs: list[dict[str, Any]] = []
    sample_states: list[dict[str, Any]] = []
    for sample in samples:
        atlas_name = str(sample["atlas_name"])
        source_index = int(sample["source_index"])
        original_id = _job_id(atlas_name, source_index, "original")
        jobs.append(
            {
                "job_id": original_id,
                "atlas_name": atlas_name,
                "source_index": source_index,
                "source_label_en": sample["source_label_en"],
                "condition": "original",
                "mask_path": sample["source_roi_path"],
                "status": "pending",
                "attempt_count": 0,
                "result": None,
            }
        )
        merged_id = _job_id(atlas_name, source_index, "merged")
        jobs.append(
            {
                "job_id": merged_id,
                "atlas_name": atlas_name,
                "source_index": source_index,
                "source_label_en": sample["source_label_en"],
                "condition": "merged",
                "mask_path": sample["result_roi_path"],
                "status": "pending",
                "attempt_count": 0,
                "result": None,
            }
        )
        sample_states.append(
            {
                **sample,
                "condition_jobs": {
                    "original": original_id,
                    "merged": merged_id,
                },
            }
        )
    now = datetime.now().astimezone().isoformat()
    return {
        "schema_version": 1,
        "configuration_fingerprint": fingerprint,
        "created_at": now,
        "updated_at": now,
        "configuration": configuration,
        "samples": sample_states,
        "jobs": jobs,
    }


def _load_or_create_state(
    state_path: Path,
    configuration: dict[str, Any],
    fingerprint: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    加载已有队列或创建新队列，并恢复中断任务。

    Parameters
    ----------
    state_path : Path
        队列状态文件。
    configuration : dict[str, Any]
        固定实验参数。
    fingerprint : str
        参数和输入报告联合指纹。
    samples : list[dict[str, Any]]
        固定 ROI 样本。

    Returns
    -------
    dict[str, Any]
        当前可执行队列状态。
    """
    if not state_path.exists():
        return _build_initial_state(configuration, fingerprint, samples)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("configuration_fingerprint") != fingerprint:
        raise ValueError(
            "已有队列与当前硬编码参数或合并报告不一致，请保留旧结果后"
            "手动移走 batch 输出目录再开始新实验"
        )
    for job in state["jobs"]:
        if job["status"] == "running":
            job["status"] = "pending"
            job["recovered_from_interruption"] = True
    state["updated_at"] = datetime.now().astimezone().isoformat()
    return state


def _find_result_mesh(output_folder: Path, subid: str) -> Path | None:
    """
    查找最终优化结果 mesh，优先使用映射电极后的结果。

    Parameters
    ----------
    output_folder : Path
        SimNIBS 优化输出目录。
    subid : str
        个体 ID。

    Returns
    -------
    Path | None
        可导出的结果 mesh；不存在时返回 None。
    """
    mapped_path = (
        output_folder
        / "mapped_electrodes_simulation"
        / f"{subid}_tes_mapped_opt_head_mesh.msh"
    )
    if mapped_path.is_file():
        return mapped_path
    candidates = sorted(output_folder.glob("*_tes_flex_opt_head_mesh.msh"))
    return candidates[0] if candidates else None


def _run_condition(job: dict[str, Any]) -> dict[str, Any]:
    """
    在独立进程中执行一个固定参数 focality 任务。

    Parameters
    ----------
    job : dict[str, Any]
        已分配 attempt 输出目录的队列任务。

    Returns
    -------
    dict[str, Any]
        优化状态、目标函数值和结果路径。
    """
    condition = str(job["condition"])
    mask_path = Path(str(job["mask_path"]))
    output_dir = Path(str(job["attempt_output_dir"]))
    subject_dir = SUBJECT_DIR.resolve()
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    result: dict[str, Any] = {
        "condition": condition,
        "roi_mask_path": str(mask_path),
        "output_dir": str(output_dir),
        "started_at": started_at.isoformat(),
        "optimization_success": False,
        "export_success": False,
    }
    try:
        subject_files = file_finder.SubjectFiles(subpath=str(subject_dir))
        conductivity_config = {**STANDARD_COND, "WM": 0.14, "GM": 0.30}
        opt = init_optimization(
            subject_dir=str(subject_dir),
            output_dir=str(output_dir),
            msh_file_path=subject_files.fnamehead,
            anisotropy_type=ANISOTROPY_TYPE,
            cond=cond_dict_to_list(conductivity_config),
        )
        setup_goal(
            opt=opt,
            goal="focality",
            focality_threshold=[
                NON_ROI_THRESHOLD_V_PER_M,
                ROI_THRESHOLD_V_PER_M,
            ],
        )
        opt.seed = OPTIMIZER_SEED
        setup_electrodes_and_roi(
            opt=opt,
            goal="focality",
            mesh_file_path=subject_files.fnamehead,
            electrode_pair1_center=[[0, 0]],
            electrode_pair2_center=[[0, 0]],
            electrode_radius=[ELECTRODE_RADIUS],
            electrode_current1=[ELECTRODE_CURRENT_A, -ELECTRODE_CURRENT_A],
            electrode_current2=[ELECTRODE_CURRENT_A, -ELECTRODE_CURRENT_A],
            roi_mask_path=str(mask_path),
            roi_mask_space="mni",
        )
        output_folder = Path(
            run_optimization(
                opt=opt,
                n_workers=N_WORKERS_PER_OPTIMIZATION,
            )
        )
        result.update(
            {
                "optimization_success": True,
                "optimization_output_folder": str(output_folder.resolve()),
                "optimization_goal_value": _json_safe(opt.optim_funvalue),
                "function_evaluation_count": opt.n_test,
                "fem_evaluation_count": opt.n_sim,
                "electrode_positions": _json_safe(opt.electrode_pos_opt),
            }
        )
        result_mesh = _find_result_mesh(output_folder, subject_files.subid)
        result["result_mesh_path"] = (
            str(result_mesh.resolve()) if result_mesh is not None else None
        )
        if result_mesh is None:
            result["export_error"] = "未找到可导出的优化结果 mesh"
        else:
            try:
                nifti_path = export_ti_to_nifti(
                    msh_path=str(result_mesh),
                    output_dir=str(output_dir),
                    reference=str(subject_dir / "T1.nii.gz"),
                    field_name="max_TI",
                    prefix=f"{subject_files.subid}_{condition}_optimization",
                )
                result["export_success"] = True
                result["ti_nifti_path"] = str(Path(nifti_path).resolve())
            except Exception as exc:
                logger.exception("%s 条件的 NIfTI 导出失败", condition)
                result["export_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        logger.exception("%s 条件的 focality 优化失败", condition)
        result["optimization_error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = time.perf_counter() - start_time
    result["finished_at"] = datetime.now().astimezone().isoformat()
    return result


def _execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """
    ProcessPool worker 入口。

    Parameters
    ----------
    job : dict[str, Any]
        可序列化队列任务。

    Returns
    -------
    dict[str, Any]
        条件运行结果。
    """
    return _run_condition(job)


def _comparison_for_sample(
    sample: dict[str, Any],
    jobs_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    汇总一个 ROI 的 original/merged 比较状态。

    Parameters
    ----------
    sample : dict[str, Any]
        样本队列记录。
    jobs_by_id : dict[str, dict[str, Any]]
        任务 ID 到状态的映射。

    Returns
    -------
    dict[str, Any]
        样本级比较结果。
    """
    original_job = jobs_by_id[sample["condition_jobs"]["original"]]
    merged_id = sample["condition_jobs"]["merged"]
    result: dict[str, Any] = {
        "atlas_name": sample["atlas_name"],
        "source_index": sample["source_index"],
        "source_label_en": sample["source_label_en"],
        "decision": sample["decision"],
        "original_job_status": original_job["status"],
        "merged_job_status": jobs_by_id[merged_id]["status"],
        "both_optimizations_succeeded": False,
        "goal_value_lower_is_better": True,
    }
    merged_job = jobs_by_id[merged_id]
    original_result = original_job.get("result") or {}
    merged_result = merged_job.get("result") or {}
    both_succeeded = bool(
        original_result.get("optimization_success")
        and merged_result.get("optimization_success")
    )
    result["both_optimizations_succeeded"] = both_succeeded
    if not both_succeeded:
        result["comparison_status"] = "waiting_or_failed"
        return result
    original_goal = float(original_result["optimization_goal_value"])
    merged_goal = float(merged_result["optimization_goal_value"])
    result.update(
        {
            "comparison_status": "completed",
            "original_goal_value": original_goal,
            "merged_goal_value": merged_goal,
            "merged_minus_original_goal_value": merged_goal - original_goal,
            "better_goal_condition": (
                "equal"
                if np.isclose(merged_goal, original_goal)
                else "merged" if merged_goal < original_goal else "original"
            ),
        }
    )
    return result


def _build_batch_report(state: dict[str, Any]) -> dict[str, Any]:
    """
    根据当前队列状态生成可读汇总报告。

    Parameters
    ----------
    state : dict[str, Any]
        当前队列状态。

    Returns
    -------
    dict[str, Any]
        任务计数和 25 个样本的比较结果。
    """
    jobs_by_id = {job["job_id"]: job for job in state["jobs"]}
    status_counts: dict[str, int] = {}
    for job in state["jobs"]:
        status = str(job["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    comparisons = [
        _comparison_for_sample(sample, jobs_by_id)
        for sample in state["samples"]
    ]
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "configuration_fingerprint": state["configuration_fingerprint"],
        "configuration": state["configuration"],
        "sample_count": len(state["samples"]),
        "job_count": len(state["jobs"]),
        "job_status_counts": status_counts,
        "comparisons": comparisons,
    }


def _write_queue_artifacts(state: dict[str, Any]) -> None:
    """
    增量写出队列状态、批量报告和任务 CSV。

    Parameters
    ----------
    state : dict[str, Any]
        当前队列状态。
    """
    state["updated_at"] = datetime.now().astimezone().isoformat()
    state_path = BATCH_OUTPUT_DIR / "queue_state.json"
    report_path = BATCH_OUTPUT_DIR / "focality_batch_report.json"
    csv_path = BATCH_OUTPUT_DIR / "focality_jobs.csv"
    temporary_csv_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    _write_json_atomic(state_path, state)
    _write_json_atomic(report_path, _build_batch_report(state))
    fieldnames = sorted({key for job in state["jobs"] for key in job})
    with temporary_csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for job in state["jobs"]:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else _json_safe(value)
                    )
                    for key, value in job.items()
                }
            )
    temporary_csv_path.replace(csv_path)


def _prepare_job_attempt(job: dict[str, Any]) -> dict[str, Any]:
    """
    为即将运行的任务分配独立 attempt 输出目录。

    Parameters
    ----------
    job : dict[str, Any]
        队列中的可变任务记录。

    Returns
    -------
    dict[str, Any]
        传给 worker 的任务副本。
    """
    job["attempt_count"] = int(job["attempt_count"]) + 1
    atlas_token = _safe_token(str(job["atlas_name"]))
    sample_dir = f"{atlas_token}_{int(job['source_index']):04d}"
    attempt_output_dir = (
        BATCH_OUTPUT_DIR
        / "jobs"
        / sample_dir
        / str(job["condition"])
        / f"attempt_{job['attempt_count']}"
    )
    job["attempt_output_dir"] = str(attempt_output_dir.resolve())
    job["status"] = "running"
    job["started_at"] = datetime.now().astimezone().isoformat()
    job["result"] = None
    return dict(job)


def _run_queue(state: dict[str, Any]) -> None:
    """
    以最多四个并发进程消费全部 pending 任务。

    Parameters
    ----------
    state : dict[str, Any]
        可恢复队列状态；本函数原位更新并持续落盘。
    """
    jobs_by_id = {job["job_id"]: job for job in state["jobs"]}
    pending_ids = deque(
        job["job_id"] for job in state["jobs"] if job["status"] == "pending"
    )
    active: dict[Future[dict[str, Any]], str] = {}
    with ProcessPoolExecutor(
        max_workers=MAX_CONCURRENT_OPTIMIZATIONS
    ) as executor:
        while pending_ids or active:
            while pending_ids and len(active) < MAX_CONCURRENT_OPTIMIZATIONS:
                job_id = pending_ids.popleft()
                worker_job = _prepare_job_attempt(jobs_by_id[job_id])
                future = executor.submit(_execute_job, worker_job)
                active[future] = job_id
                _write_queue_artifacts(state)
                logger.info("任务进入运行队列: %s", job_id)
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                job_id = active.pop(future)
                job = jobs_by_id[job_id]
                try:
                    result = future.result()
                    job["result"] = result
                    job["status"] = (
                        "completed"
                        if result.get("optimization_success")
                        else "failed"
                    )
                except Exception as exc:
                    logger.exception("队列 worker 异常: %s", job_id)
                    job["status"] = "failed"
                    job["result"] = {
                        "optimization_success": False,
                        "worker_error": f"{type(exc).__name__}: {exc}",
                    }
                job["finished_at"] = datetime.now().astimezone().isoformat()
                _write_queue_artifacts(state)
                logger.info("任务结束: %s, status=%s", job_id, job["status"])


def main() -> None:
    """
    校验固定参数并消费 25 个 merged ROI 的可恢复优化队列。

    本入口不解析命令行参数。直接运行脚本即可从固定报告构造队列，最多
    并行四个 focality 优化；重复运行时依据落盘状态继续未完成任务。
    """
    setup_logging(str(PROJECT_ROOT / "log" / "roi_merge_focality_batch"))
    if len(FOCUS_ROIS) != 25:
        raise ValueError(f"固定 ROI 数量应为 25，实际为 {len(FOCUS_ROIS)}")
    if not MERGE_REPORT_PATH.is_file():
        raise ValueError(f"合并报告不存在: {MERGE_REPORT_PATH}")
    if not SUBJECT_DIR.is_dir():
        raise ValueError(f"个体目录不存在: {SUBJECT_DIR}")
    if MAX_CONCURRENT_OPTIMIZATIONS != 4:
        raise ValueError("并发逆向优化数量必须固定为 4")
    configuration = _configuration()
    fingerprint = _configuration_fingerprint(configuration)
    samples = _load_focus_samples()
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = BATCH_OUTPUT_DIR / "queue_state.json"
    state = _load_or_create_state(
        state_path,
        configuration,
        fingerprint,
        samples,
    )
    _write_queue_artifacts(state)
    _run_queue(state)
    report = _build_batch_report(state)
    print(f"samples: {report['sample_count']}")
    print(f"jobs: {report['job_count']}")
    print(f"job status: {report['job_status_counts']}")
    print(f"report: {BATCH_OUTPUT_DIR / 'focality_batch_report.json'}")


if __name__ == "__main__":
    main()
