"""
Leadfield GA 优化结果的 JSON、CSV、MSH 和 NIfTI 导出。

输出文件同时保留可机读指标和可视化电场，便于与 baseline 及直接 FEM
结果做一致性对照。
"""

import csv
import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neuracle.ti_leadfield_optimization.models import (
    FitnessMetrics,
    GASettings,
    LeadfieldData,
    LeadfieldGAResult,
    RegionMasks,
)
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs import ElementTags, mesh_io

logger = logging.getLogger(__name__)


def fitness_metrics_to_dict(metrics: FitnessMetrics) -> dict[str, Any]:
    """将适应度 dataclass 转换为稳定的 JSON 字段。

    Parameters
    ----------
    metrics : FitnessMetrics
        需要序列化的指标。

    Returns
    -------
    dict[str, Any]
        包含 objective、ROI/Rest 指标、惩罚和电流的字典。
    """
    return {
        "objective": metrics.objective,
        "score": metrics.score,
        "roi_rest_ratio": metrics.roi_rest_ratio,
        "roi_mean_v_per_m": metrics.roi_mean_v_per_m,
        "rest_mean_v_per_m": metrics.rest_mean_v_per_m,
        "roi_max_v_per_m": metrics.roi_max_v_per_m,
        "penalty": metrics.penalty,
        "threshold_satisfied": metrics.threshold_satisfied,
        "currents_mA": {
            "pair_A": metrics.currents.current_a_ma,
            "pair_B": metrics.currents.current_b_ma,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """使用 UTF-8 和稳定缩进写入 JSON 结果。

    Parameters
    ----------
    path : pathlib.Path
        输出 JSON 路径。
    payload : dict[str, Any]
        可序列化内容。

    Returns
    -------
    pathlib.Path
        已写入的 JSON 路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path


def write_baseline_metrics(
    path: str | Path,
    electrode_names: tuple[str, str, str, str],
    metrics: FitnessMetrics,
) -> Path:
    """写入 GA 运行前的论文 baseline 电极、电流和指标。

    Parameters
    ----------
    path : str or pathlib.Path
        ``baseline_metrics.json`` 路径。
    electrode_names : tuple[str, str, str, str]
        Baseline A+、A-、B+、B- 电极名称。
    metrics : FitnessMetrics
        Baseline leadfield 指标。

    Returns
    -------
    pathlib.Path
        已写入的 JSON 路径。
    """
    payload = {
        "label": "paper_baseline",
        "electrode_A": [
            {"name": electrode_names[0], "current_mA": metrics.currents.current_a_ma},
            {"name": electrode_names[1], "current_mA": -metrics.currents.current_a_ma},
        ],
        "electrode_B": [
            {"name": electrode_names[2], "current_mA": metrics.currents.current_b_ma},
            {"name": electrode_names[3], "current_mA": -metrics.currents.current_b_ma},
        ],
        "metrics": fitness_metrics_to_dict(metrics),
    }
    return _write_json(Path(path), payload)


def _sha256_file(path: Path) -> str:
    """计算结果 JSON 中用于追溯 leadfield 的 SHA-256。

    Parameters
    ----------
    path : pathlib.Path
        需要计算摘要的文件。

    Returns
    -------
    str
        十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_optimization_result(
    path: str | Path,
    result: LeadfieldGAResult,
    leadfield: LeadfieldData,
    settings: GASettings,
    montage_path: str | Path,
) -> Path:
    """写入最优电极、电流、GA 参数、指标和 leadfield 追溯信息。

    Parameters
    ----------
    path : str or pathlib.Path
        ``optimization_result.json`` 路径。
    result : LeadfieldGAResult
        GA 最优结果。
    leadfield : LeadfieldData
        优化使用的 leadfield。
    settings : GASettings
        GA 和电流约束。
    montage_path : str or pathlib.Path
        固定 montage CSV 路径。

    Returns
    -------
    pathlib.Path
        已写入的 JSON 路径。
    """
    names = result.electrode_names
    currents = result.metrics.currents
    payload = {
        "algorithm": "leadfield_ga",
        "ga_library": "geneticalgorithm==1.0.2",
        "montage": str(Path(montage_path).resolve()),
        "electrode_A": [
            {"name": names[0], "current_mA": currents.current_a_ma},
            {"name": names[1], "current_mA": -currents.current_a_ma},
        ],
        "electrode_B": [
            {"name": names[2], "current_mA": currents.current_b_ma},
            {"name": names[3], "current_mA": -currents.current_b_ma},
        ],
        "current_constraints": {
            "minimum_per_pair_mA": settings.current_min_ma,
            "maximum_per_pair_mA": settings.current_max_ma,
            "step_mA": settings.current_step_ma,
            "sum_mA": settings.current_sum_ma,
        },
        "metrics": fitness_metrics_to_dict(result.metrics),
        "ga": {
            "random_seed": result.random_seed,
            "generations": settings.max_num_iteration,
            "population_size": settings.population_size,
            "mutation_probability": settings.mutation_probability,
            "elit_ratio": settings.elit_ratio,
            "crossover_probability": settings.crossover_probability,
            "parents_portion": settings.parents_portion,
            "crossover_type": settings.crossover_type,
        },
        "leadfield": {
            "path": str(leadfield.path.resolve()),
            "manifest_path": str(
                (leadfield.path.parent / "leadfield_manifest.json").resolve()
            ),
            "manifest_sha256": _sha256_file(
                leadfield.path.parent / "leadfield_manifest.json"
            ),
            "shape": list(leadfield.values.shape),
            "dtype": str(leadfield.values.dtype),
        },
    }
    return _write_json(Path(path), payload)


def write_convergence_csv(path: str | Path, convergence: tuple[float, ...]) -> Path:
    """将 ``geneticalgorithm.report`` 写为代数与最佳 objective CSV。

    Parameters
    ----------
    path : str or pathlib.Path
        ``convergence.csv`` 路径。
    convergence : tuple[float, ...]
        每代的最佳 objective。

    Returns
    -------
    pathlib.Path
        已写入的 CSV 路径。
    """
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["generation", "best_objective"])
        for generation, objective in enumerate(convergence):
            writer.writerow([generation, objective])
    return csv_path


def write_result_mesh(
    path: str | Path,
    leadfield: LeadfieldData,
    region_masks: RegionMasks,
    field_a: npt.NDArray[np.float64],
    field_b: npt.NDArray[np.float64],
    max_ti: npt.NDArray[np.float64],
) -> Path:
    """将最优两路电场、max_TI 和 ROI/Rest mask 写入 element-data mesh。

    Parameters
    ----------
    path : str or pathlib.Path
        输出 MSH 路径。
    leadfield : LeadfieldData
        提供对齐后的 WM/GM mesh。
    region_masks : RegionMasks
        ROI 和 Rest element mask。
    field_a : numpy.ndarray
        A 电极对电场向量。
    field_b : numpy.ndarray
        B 电极对电场向量。
    max_ti : numpy.ndarray
        TI 最大调制包络。

    Returns
    -------
    pathlib.Path
        已写入的 MSH 路径。
    """
    mesh_path = Path(path)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = deepcopy(leadfield.mesh)
    mesh.elmdata = []
    mesh.nodedata = []
    mesh.add_element_field(np.linalg.norm(field_a, axis=1), "E_magn_A")
    mesh.add_element_field(np.linalg.norm(field_b, axis=1), "E_magn_B")
    mesh.add_element_field(max_ti, "max_TI")
    mesh.add_element_field(region_masks.roi_mask.astype(np.float64), "ROI_mask")
    mesh.add_element_field(region_masks.rest_mask.astype(np.float64), "Rest_mask")
    mesh_io.write_msh(mesh, str(mesh_path))
    view = mesh.view(
        visible_tags=[ElementTags.WM, ElementTags.GM],
        visible_fields="max_TI",
    )
    view.write_opt(str(mesh_path))
    logger.info("最优 TI mesh 已写入: %s", mesh_path)
    return mesh_path


def export_result_nifti(
    mesh_path: str | Path,
    output_dir: str | Path,
    reference_path: str | Path,
    prefix: str,
) -> Path:
    """将最优 mesh 中的 max_TI 插值到 T1 体素空间。

    Parameters
    ----------
    mesh_path : str or pathlib.Path
        包含 ``max_TI`` element field 的 MSH。
    output_dir : str or pathlib.Path
        NIfTI 输出目录。
    reference_path : str or pathlib.Path
        提供空间和仿射矩阵的 T1 NIfTI。
    prefix : str
        NIfTI 文件前缀。

    Returns
    -------
    pathlib.Path
        实际生成的 max_TI NIfTI 路径。
    """
    nifti_path = export_ti_to_nifti(
        msh_path=str(mesh_path),
        output_dir=str(output_dir),
        reference=str(reference_path),
        field_name="max_TI",
        prefix=prefix,
    )
    return Path(nifti_path)


def write_comparison(path: str | Path, payload: dict[str, Any]) -> Path:
    """写入 baseline、leadfield 最优解和直接 FEM 的指标对照。

    Parameters
    ----------
    path : str or pathlib.Path
        ``comparison.json`` 路径。
    payload : dict[str, Any]
        已组装的对照指标和验收结果。

    Returns
    -------
    pathlib.Path
        已写入的 JSON 路径。
    """
    return _write_json(Path(path), payload)
