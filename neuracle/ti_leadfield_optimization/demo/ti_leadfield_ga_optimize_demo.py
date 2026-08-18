"""
使用现有 ernie 头模复现论文的 leadfield-based TI 遗传算法优化。

运行方式：
``conda run -n simnibs_env python -m``
``neuracle.ti_leadfield_optimization.demo.ti_leadfield_ga_optimize_demo``

Demo 会首先生成或复用 WM/GM volumetric TDCS leadfield，计算论文 baseline，
再使用 ``geneticalgorithm==1.0.2`` 优化四个 montage 电极和两路电流。
最后导出 JSON/CSV/MSH/NIfTI，并对 baseline 和最优解各运行一次直接 FEM。
运行前需在 ``simnibs_env`` 安装 ``geneticalgorithm==1.0.2``。
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from neuracle.logger import setup_logging
from neuracle.parameters.schemas import AnisotropyType, ElectrodeWithCurrent
from neuracle.ti_forward import run_ti_forward
from neuracle.ti_leadfield_optimization import (
    ElectrodeChromosome,
    LeadfieldFitnessEvaluator,
    LeadfieldGADemoConfig,
    build_atlas_region_masks,
    calculate_region_metrics,
    ensure_leadfield,
    generate_current_pairs,
    load_leadfield,
    run_genetic_optimization,
)
from neuracle.ti_leadfield_optimization.models import FitnessMetrics
from neuracle.ti_leadfield_optimization.result import (
    export_result_nifti,
    fitness_metrics_to_dict,
    write_baseline_metrics,
    write_comparison,
    write_convergence_csv,
    write_optimization_result,
    write_result_mesh,
)
from neuracle.utils.constants import (
    DATA_ROOT,
    ELECTRODE_RADIUS,
    PROJECT_ROOT,
    STANDARD_COND,
)
from simnibs import mesh_io

logger = logging.getLogger(__name__)


def _build_demo_config() -> LeadfieldGADemoConfig:
    """使用仓库现有 ernie、10-10 montage 和 Brainnetome 右海马构建 demo 配置。

    Returns
    -------
    LeadfieldGADemoConfig
        可直接用于 leadfield 生成、GA 和直接 FEM 验证的配置。

    Notes
    -----
    ROI 是 ``rHipp_R`` 和 ``cHipp_R`` 的并集，即 Brainnetome atlas 中的完整右海马。
    """
    subject_dir = DATA_ROOT / "m2m_ernie"
    output_root = DATA_ROOT / "ti_leadfield_ga_ernie"
    atlas_root = (
        PROJECT_ROOT
        / "neuracle"
        / "atlas"
        / "standardized"
        / "BN_Atlas_246_1mm"
        / "rois"
    )
    return LeadfieldGADemoConfig(
        head_model_id="ernie",
        head_model_dir=subject_dir,
        mesh_path=subject_dir / "ernie.msh",
        t1_path=subject_dir / "T1.nii.gz",
        montage_path=(subject_dir / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv"),
        atlas_mask_paths=(
            atlas_root / "0216_rHipp_R.nii.gz",
            atlas_root / "0218_cHipp_R.nii.gz",
        ),
        leadfield_dir=output_root / "leadfield",
        result_dir=output_root / "results",
        conductivity_config=dict(STANDARD_COND),
        anisotropy_type="scalar",
        dti_path=None,
        electrode_radius_mm=ELECTRODE_RADIUS,
        electrode_thickness_mm=2.0,
        n_workers=8,
        run_direct_fem_validation=True,
    )


def _validate_demo_config(config: LeadfieldGADemoConfig) -> None:
    """在耗时 FEM 之前验证 demo 的现有输入和基本数值约束。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        需要验证的 demo 配置。

    Returns
    -------
    None
        验证通过时不返回额外数据。

    Raises
    ------
    FileNotFoundError
        头模、T1、montage 或 atlas mask 不存在时抛出。
    ValueError
        电极几何、并行数或 baseline 电流不合法时抛出。
    """
    required_paths = [
        config.head_model_dir,
        config.mesh_path,
        config.t1_path,
        config.montage_path,
        *config.atlas_mask_paths,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"demo 输入不存在: {path}")
    if config.electrode_radius_mm <= 0 or config.electrode_thickness_mm <= 0:
        raise ValueError("电极半径和厚度必须大于 0")
    if config.n_workers < 1:
        raise ValueError("n_workers 必须大于等于 1")
    baseline_total = (
        config.baseline_currents.current_a_ma + config.baseline_currents.current_b_ma
    )
    if not np.isclose(baseline_total, config.ga.current_sum_ma):
        raise ValueError("baseline 两路电流之和必须为 2 mA")


def _chromosome_from_names(
    electrode_names: tuple[str, ...],
    selected_names: tuple[str, str, str, str],
) -> ElectrodeChromosome:
    """将 baseline montage 名称转换为与 leadfield 顺序对齐的染色体。

    Parameters
    ----------
    electrode_names : tuple[str, ...]
        Leadfield 中的完整 montage 电极顺序。
    selected_names : tuple[str, str, str, str]
        A+、A-、B+、B- 名称。

    Returns
    -------
    ElectrodeChromosome
        四个整数 montage 索引。

    Raises
    ------
    ValueError
        Baseline 电极重复或不在 montage 中时抛出。
    """
    if len(set(selected_names)) != 4:
        raise ValueError("baseline 必须使用四个互不重复的电极")
    missing = [name for name in selected_names if name not in electrode_names]
    if missing:
        raise ValueError(f"baseline 电极不在固定 montage 中: {missing}")
    indices = tuple(electrode_names.index(name) for name in selected_names)
    return ElectrodeChromosome(indices)


def _find_ti_mesh(output_dir: Path) -> Path | None:
    """在单个直接 FEM 输出目录中定位已生成的 TI mesh。

    Parameters
    ----------
    output_dir : pathlib.Path
        Baseline 或 optimized 直接 FEM 目录。

    Returns
    -------
    pathlib.Path or None
        唯一 ``*_TI.msh`` 路径；尚未生成时返回 None。

    Raises
    ------
    RuntimeError
        目录中存在多个 TI mesh 时抛出。
    """
    candidates = sorted(output_dir.rglob("*_TI.msh")) if output_dir.exists() else []
    if len(candidates) > 1:
        raise RuntimeError(f"直接 FEM 输出中存在多个 TI mesh: {output_dir}")
    return candidates[0] if candidates else None


def _run_direct_fem_case(
    config: LeadfieldGADemoConfig,
    output_dir: Path,
    electrode_names: tuple[str, str, str, str],
    metrics: FitnessMetrics,
) -> Path:
    """对一组四电极和两路电流运行或复用直接 TI FEM。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        头模、montage、电导率和并行配置。
    output_dir : pathlib.Path
        当前对照方案的独立输出目录。
    electrode_names : tuple[str, str, str, str]
        A+、A-、B+、B- 电极名称。
    metrics : FitnessMetrics
        包含该方案两路电流幅值的指标。

    Returns
    -------
    pathlib.Path
        直接 FEM 生成的 TI mesh。
    """
    existing_mesh = _find_ti_mesh(output_dir)
    case_path = output_dir / "direct_case.json"
    currents = metrics.currents
    case_payload = {
        "electrodes": list(electrode_names),
        "currents_mA": [currents.current_a_ma, currents.current_b_ma],
        "montage": str(config.montage_path.resolve()),
        "conductivity": config.conductivity_config,
        "anisotropy_type": config.anisotropy_type,
        "electrode_radius_mm": config.electrode_radius_mm,
        "electrode_thickness_mm": config.electrode_thickness_mm,
    }
    if existing_mesh is not None:
        if not case_path.is_file():
            raise RuntimeError(f"直接 FEM 缓存缺少配置摘要: {case_path}")
        with case_path.open("r", encoding="utf-8") as stream:
            cached_payload = json.load(stream)
        if cached_payload != case_payload:
            raise RuntimeError(f"直接 FEM 缓存与当前配置不一致: {output_dir}")
        logger.info("复用已有直接 FEM 结果: %s", existing_mesh)
        return existing_mesh
    if case_path.exists():
        raise RuntimeError(f"直接 FEM 缓存不完整: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"直接 FEM 目录包含未完成结果: {output_dir}")
    run_ti_forward(
        head_model_id=config.head_model_id,
        head_model_dir=str(config.head_model_dir),
        output_dir=str(output_dir),
        montage=str(config.montage_path),
        electrode_A=[
            ElectrodeWithCurrent(electrode_names[0], currents.current_a_ma),
            ElectrodeWithCurrent(electrode_names[1], -currents.current_a_ma),
        ],
        electrode_B=[
            ElectrodeWithCurrent(electrode_names[2], currents.current_b_ma),
            ElectrodeWithCurrent(electrode_names[3], -currents.current_b_ma),
        ],
        conductivity_config=config.conductivity_config,
        anisotropy=AnisotropyType(config.anisotropy_type),
        n_workers=config.n_workers,
        electrode_radius=config.electrode_radius_mm,
    )
    generated_mesh = _find_ti_mesh(output_dir)
    if generated_mesh is None:
        raise RuntimeError(f"直接 FEM 未生成 TI mesh: {output_dir}")
    write_comparison(case_path, case_payload)
    return generated_mesh


def _direct_fem_metrics(
    config: LeadfieldGADemoConfig,
    ti_mesh_path: Path,
) -> dict[str, float]:
    """在与 leadfield 相同的 atlas ROI/Rest 定义下计算直接 FEM 指标。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        Subject 目录和 atlas mask 配置。
    ti_mesh_path : pathlib.Path
        包含 ``max_TI`` element field 的直接 FEM mesh。

    Returns
    -------
    dict[str, float]
        ROI 均值、Rest 均值、ratio 和 ROI 最大值。
    """
    mesh = mesh_io.read_msh(str(ti_mesh_path))
    if "max_TI" not in mesh.field:
        raise ValueError(f"直接 FEM mesh 缺少 max_TI 字段: {ti_mesh_path}")
    max_ti = np.asarray(mesh.field["max_TI"].value, dtype=np.float64)
    region_masks = build_atlas_region_masks(
        mesh,
        config.head_model_dir,
        config.atlas_mask_paths,
    )
    roi_mean, rest_mean, ratio, roi_max = calculate_region_metrics(
        max_ti,
        region_masks,
    )
    return {
        "roi_mean_v_per_m": roi_mean,
        "rest_mean_v_per_m": rest_mean,
        "roi_rest_ratio": ratio,
        "roi_max_v_per_m": roi_max,
    }


def _relative_error(reference: float, actual: float) -> float:
    """计算 leadfield 相对于直接 FEM 指标的对称安全相对误差。

    Parameters
    ----------
    reference : float
        Leadfield 指标。
    actual : float
        直接 FEM 指标。

    Returns
    -------
    float
        以较大绝对值为分母的相对误差。
    """
    denominator = max(abs(reference), abs(actual), np.finfo(np.float64).eps)
    return abs(reference - actual) / denominator


def _comparison_payload(
    config: LeadfieldGADemoConfig,
    baseline_metrics: FitnessMetrics,
    optimized_metrics: FitnessMetrics,
    direct_baseline: dict[str, float] | None,
    direct_optimized: dict[str, float] | None,
) -> dict[str, Any]:
    """组装 leadfield 效果提升、阈值和直接 FEM 一致性验收结果。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        阈值及是否执行直接 FEM 的配置。
    baseline_metrics : FitnessMetrics
        论文 baseline 的 leadfield 指标。
    optimized_metrics : FitnessMetrics
        GA 最优解的 leadfield 指标。
    direct_baseline : dict[str, float] or None
        Baseline 直接 FEM 指标。
    direct_optimized : dict[str, float] or None
        最优解直接 FEM 指标。

    Returns
    -------
    dict[str, Any]
        可直接写入 ``comparison.json`` 的对照内容。
    """
    optimized_improves_leadfield = optimized_metrics.score > baseline_metrics.score
    threshold_satisfied = (
        optimized_metrics.roi_max_v_per_m >= config.ga.target_threshold_v_per_m
    )
    validation = {
        "optimized_score_above_baseline": optimized_improves_leadfield,
        "optimized_threshold_satisfied": threshold_satisfied,
    }
    payload: dict[str, Any] = {
        "leadfield": {
            "baseline": fitness_metrics_to_dict(baseline_metrics),
            "optimized": fitness_metrics_to_dict(optimized_metrics),
        },
        "direct_fem": None,
        "validation": validation,
    }
    if direct_baseline is not None and direct_optimized is not None:
        metric_names = (
            "roi_mean_v_per_m",
            "rest_mean_v_per_m",
            "roi_rest_ratio",
            "roi_max_v_per_m",
        )
        baseline_leadfield = fitness_metrics_to_dict(baseline_metrics)
        optimized_leadfield = fitness_metrics_to_dict(optimized_metrics)
        errors = {
            "baseline": {
                name: _relative_error(baseline_leadfield[name], direct_baseline[name])
                for name in metric_names
            },
            "optimized": {
                name: _relative_error(optimized_leadfield[name], direct_optimized[name])
                for name in metric_names
            },
        }
        max_relative_error = max(
            value for case_errors in errors.values() for value in case_errors.values()
        )
        direct_improves = (
            direct_optimized["roi_rest_ratio"] > direct_baseline["roi_rest_ratio"]
        )
        validation["direct_fem_ratio_above_baseline"] = direct_improves
        validation["leadfield_fem_relative_error_within_5_percent"] = (
            max_relative_error <= 0.05
        )
        payload["direct_fem"] = {
            "baseline": direct_baseline,
            "optimized": direct_optimized,
            "relative_error": errors,
            "maximum_relative_error": max_relative_error,
        }
    validation["passed"] = all(validation.values())
    return payload


def main() -> None:
    """执行 ernie leadfield GA demo 并以 baseline 和直接 FEM 验证效果。

    Returns
    -------
    None
        结果写入 ``data/ti_leadfield_ga_ernie``，验收失败时抛出异常。

    Notes
    -----
    该函数是新算法的唯一运行入口；不修改或调用现有 ``ti_inverse``。
    """
    setup_logging(str(PROJECT_ROOT / "log" / "ti_leadfield_ga_optimize_demo"))
    config = _build_demo_config()
    _validate_demo_config(config)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[1/8] 生成或复用 volumetric TDCS leadfield")
    leadfield_path = ensure_leadfield(config)
    logger.info("[2/8] 一次性加载完整 float64 leadfield")
    leadfield = load_leadfield(leadfield_path)
    logger.info("[3/8] 构建完整右海马 ROI 与 Rest")
    region_masks = build_atlas_region_masks(
        leadfield.mesh,
        config.head_model_dir,
        config.atlas_mask_paths,
    )
    current_pairs = generate_current_pairs(
        minimum_ma=config.ga.current_min_ma,
        maximum_ma=config.ga.current_max_ma,
        step_ma=config.ga.current_step_ma,
        total_ma=config.ga.current_sum_ma,
    )
    if len(current_pairs) != 21:
        raise ValueError(f"论文电流组合应为 21 组，实际 {len(current_pairs)}")
    evaluator = LeadfieldFitnessEvaluator(
        leadfield,
        region_masks,
        current_pairs,
        config.ga.target_threshold_v_per_m,
    )
    logger.info("[4/8] 计算论文 baseline，不参与 GA 初始化")
    baseline_chromosome = _chromosome_from_names(
        leadfield.electrode_names,
        config.baseline_electrodes,
    )
    baseline_metrics = evaluator.evaluate_current_pair(
        baseline_chromosome,
        config.baseline_currents,
    )
    write_baseline_metrics(
        config.result_dir / "baseline_metrics.json",
        config.baseline_electrodes,
        baseline_metrics,
    )
    logger.info("[5/8] 运行 geneticalgorithm 四电极优化")
    optimized = run_genetic_optimization(evaluator, config.ga)
    write_optimization_result(
        config.result_dir / "optimization_result.json",
        optimized,
        leadfield,
        config.ga,
        config.montage_path,
    )
    write_convergence_csv(
        config.result_dir / "convergence.csv",
        optimized.convergence,
    )
    logger.info("[6/8] 重建并导出最优 TI 电场")
    field_a, field_b, max_ti = evaluator.reconstruct_fields(
        optimized.chromosome,
        optimized.metrics.currents,
    )
    result_mesh_path = write_result_mesh(
        config.result_dir / "ti_leadfield_ga_result.msh",
        leadfield,
        region_masks,
        field_a,
        field_b,
        max_ti,
    )
    export_result_nifti(
        result_mesh_path,
        config.result_dir,
        config.t1_path,
        "ernie_leadfield_ga",
    )
    direct_baseline = None
    direct_optimized = None
    if config.run_direct_fem_validation:
        logger.info("[7/8] 对 baseline 和最优解运行直接 FEM 对照")
        validation_dir = config.result_dir / "final_validation"
        baseline_mesh = _run_direct_fem_case(
            config,
            validation_dir / "baseline",
            config.baseline_electrodes,
            baseline_metrics,
        )
        optimized_mesh = _run_direct_fem_case(
            config,
            validation_dir / "optimized",
            optimized.electrode_names,
            optimized.metrics,
        )
        direct_baseline = _direct_fem_metrics(config, baseline_mesh)
        direct_optimized = _direct_fem_metrics(config, optimized_mesh)
    logger.info("[8/8] 写入效果对照并执行验收")
    comparison = _comparison_payload(
        config,
        baseline_metrics,
        optimized.metrics,
        direct_baseline,
        direct_optimized,
    )
    comparison_path = write_comparison(
        config.result_dir / "final_validation" / "comparison.json",
        comparison,
    )
    if not comparison["validation"]["passed"]:
        raise RuntimeError(f"leadfield GA demo 效果验证未通过: {comparison_path}")
    logger.info("leadfield GA demo 效果验证通过: %s", comparison_path)


if __name__ == "__main__":
    main()
