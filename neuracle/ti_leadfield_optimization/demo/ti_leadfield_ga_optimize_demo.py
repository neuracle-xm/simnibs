"""
使用现有 ernie 头模运行工程化 leadfield-based TI 遗传算法优化。

运行方式：
``conda run -n simnibs_env python -m``
``neuracle.ti_leadfield_optimization.demo.ti_leadfield_ga_optimize_demo``

Demo 会首先生成或复用 NSN 10-10 montage 的 WM/GM volumetric TDCS leadfield，
再使用 ``geneticalgorithm==1.0.2`` 同时优化四个电极和两路独立电流。
目标函数使用 SimNIBS TES 逆向优化的 focality，最后导出 JSON/CSV/MSH/NIfTI，
并对最优解运行一次直接 FEM。
运行前需在 ``simnibs_env`` 安装 ``geneticalgorithm==1.0.2``。
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from neuracle.logger import setup_logging
from neuracle.parameters.schemas import AnisotropyType, ElectrodeWithCurrent
from neuracle.ti_forward import run_ti_forward
from neuracle.ti_leadfield_optimization import (
    LeadfieldFitnessEvaluator,
    LeadfieldGADemoConfig,
    build_atlas_region_masks,
    calculate_focality_metrics,
    calculate_region_metrics,
    ensure_leadfield,
    load_leadfield,
    run_genetic_optimization,
)
from neuracle.ti_leadfield_optimization.models import FitnessMetrics
from neuracle.ti_leadfield_optimization.result import (
    export_result_nifti,
    fitness_metrics_to_dict,
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
LOG_DIRECTORY_NAME = "ti_leadfield_ga_nsn_10_10_optimize_demo"


def _format_elapsed(seconds: float) -> str:
    """把单调时钟测得的耗时格式化为时分秒和总秒数。

    Parameters
    ----------
    seconds : float
        ``time.perf_counter`` 测得的秒数。

    Returns
    -------
    str
        ``HH:MM:SS.mmm (N.NNN s)`` 格式的耗时。
    """
    milliseconds = round(max(seconds, 0.0) * 1000.0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d} "
        f"({seconds:.3f} s)"
    )


def _log_stage_elapsed(
    stage: str,
    stage_started_at: float,
    run_started_at: float,
) -> None:
    """记录阶段耗时和从本次运行开始计算的累计耗时。

    Parameters
    ----------
    stage : str
        阶段名称，例如 ``[1/7] leadfield``。
    stage_started_at : float
        阶段开始时的 ``time.perf_counter`` 值。
    run_started_at : float
        整次运行开始时的 ``time.perf_counter`` 值。

    Returns
    -------
    None
        耗时写入模块 logger。
    """
    completed_at = time.perf_counter()
    logger.info(
        "%s 完成: stage_elapsed=%s, total_elapsed=%s",
        stage,
        _format_elapsed(completed_at - stage_started_at),
        _format_elapsed(completed_at - run_started_at),
    )


def _setup_demo_logging() -> None:
    """配置 NSN demo 日志并处理 ``python -m`` 的 ``__main__`` logger。

    Returns
    -------
    None
        日志写入 NSN demo 专用目录。

    Notes
    -----
    使用 ``python -m`` 时模块 logger 名称会变成 ``__main__``，不会继承
    ``setup_logging`` 配置的 ``neuracle`` handlers。此处将入口 logger 重新绑定
    为已配置的子 logger，确保阶段耗时、总耗时和失败前日志都能写入文件。
    """
    global logger
    setup_logging(str(PROJECT_ROOT / "log" / LOG_DIRECTORY_NAME))
    if logger.name == "__main__":
        logger = logging.getLogger("neuracle").getChild(Path(__file__).stem)


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
    output_root = DATA_ROOT / "ti_leadfield_ga_ernie_nsn_10_10"
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
        montage_path=(subject_dir / "eeg_positions" / "EEG10-10_NSN.csv"),
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
        电极几何、并行数、电流范围或 focality 阈值不合法时抛出。
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
    ga = config.ga
    if ga.current_step_ma <= 0 or ga.current_min_ma <= 0:
        raise ValueError("两路电流下限和步长必须为正数")
    if ga.current_max_ma < ga.current_min_ma:
        raise ValueError("两路电流上限不能低于下限")
    current_bounds = (ga.current_min_ma, ga.current_max_ma)
    if any(
        not np.isclose(value, round(value / ga.current_step_ma) * ga.current_step_ma)
        for value in current_bounds
    ):
        raise ValueError("电流上下限必须能被步长精确表示")
    if ga.non_roi_threshold_v_per_m > ga.roi_threshold_v_per_m:
        raise ValueError("focality 非 ROI 阈值不能高于 ROI 阈值")


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
        SimNIBS focality、ROI 均值、Rest 均值、ratio 和 ROI 最大值。
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
    objective, score, roc_distance, roi_sensitivity, false_positive_rate = (
        calculate_focality_metrics(
            max_ti,
            region_masks,
            config.ga.non_roi_threshold_v_per_m,
            config.ga.roi_threshold_v_per_m,
        )
    )
    return {
        "objective": objective,
        "score": score,
        "roc_distance": roc_distance,
        "roi_sensitivity": roi_sensitivity,
        "non_roi_false_positive_rate": false_positive_rate,
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
    optimized_metrics: FitnessMetrics,
    direct_optimized: dict[str, float] | None,
) -> dict[str, Any]:
    """组装最优解的 leadfield 与直接 FEM focality 一致性结果。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        Focality 阈值及是否执行直接 FEM 的配置。
    optimized_metrics : FitnessMetrics
        GA 最优解的 leadfield 指标。
    direct_optimized : dict[str, float] or None
        最优解直接 FEM 指标。

    Returns
    -------
    dict[str, Any]
        可直接写入 ``comparison.json`` 的对照内容。
    """
    validation: dict[str, bool] = {}
    payload: dict[str, Any] = {
        "objective": {
            "name": "simnibs_focality",
            "non_roi_threshold_v_per_m": config.ga.non_roi_threshold_v_per_m,
            "roi_threshold_v_per_m": config.ga.roi_threshold_v_per_m,
        },
        "leadfield": {"optimized": fitness_metrics_to_dict(optimized_metrics)},
        "direct_fem": None,
        "validation": validation,
    }
    if direct_optimized is not None:
        metric_names = (
            "score",
            "roc_distance",
            "roi_sensitivity",
            "non_roi_false_positive_rate",
            "roi_mean_v_per_m",
            "rest_mean_v_per_m",
            "roi_rest_ratio",
            "roi_max_v_per_m",
        )
        optimized_leadfield = fitness_metrics_to_dict(optimized_metrics)
        errors = {
            name: _relative_error(optimized_leadfield[name], direct_optimized[name])
            for name in metric_names
        }
        focality_score_error = errors["score"]
        validation["leadfield_fem_focality_score_relative_error_within_5_percent"] = (
            bool(focality_score_error <= 0.05)
        )
        payload["direct_fem"] = {
            "optimized": direct_optimized,
            "relative_error": errors,
            "focality_score_relative_error": focality_score_error,
        }
    validation["passed"] = all(validation.values()) if validation else True
    return payload


def _run_demo(config: LeadfieldGADemoConfig, run_started_at: float) -> Path:
    """执行各计算阶段并记录阶段实际耗时。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        已验证的 NSN 10-10 leadfield GA 配置。
    run_started_at : float
        整次运行开始时的 ``time.perf_counter`` 值。

    Returns
    -------
    pathlib.Path
        最终 ``comparison.json`` 路径。

    Raises
    ------
    RuntimeError
        Leadfield、GA、直接 FEM 或最终验证失败时抛出。
    """
    config.result_dir.mkdir(parents=True, exist_ok=True)
    stage_started_at = time.perf_counter()
    logger.info("[1/7] 生成或复用 NSN 10-10 volumetric TDCS leadfield")
    leadfield_path = ensure_leadfield(config)
    _log_stage_elapsed("[1/7] leadfield", stage_started_at, run_started_at)
    stage_started_at = time.perf_counter()
    logger.info("[2/7] 一次性加载完整 float64 leadfield")
    leadfield = load_leadfield(leadfield_path)
    _log_stage_elapsed("[2/7] 加载 leadfield", stage_started_at, run_started_at)
    stage_started_at = time.perf_counter()
    logger.info("[3/7] 构建完整右海马 ROI 与非 ROI")
    region_masks = build_atlas_region_masks(
        leadfield.mesh,
        config.head_model_dir,
        config.atlas_mask_paths,
    )
    evaluator = LeadfieldFitnessEvaluator(
        leadfield,
        region_masks,
        config.ga.non_roi_threshold_v_per_m,
        config.ga.roi_threshold_v_per_m,
    )
    _log_stage_elapsed("[3/7] 构建 ROI", stage_started_at, run_started_at)
    stage_started_at = time.perf_counter()
    logger.info(
        "[4/7] 运行六基因 GA: currents=%s-%s mA, step=%s mA, no_sum_constraint",
        config.ga.current_min_ma,
        config.ga.current_max_ma,
        config.ga.current_step_ma,
    )
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
    _log_stage_elapsed("[4/7] 六基因 GA", stage_started_at, run_started_at)
    stage_started_at = time.perf_counter()
    logger.info("[5/7] 重建并导出最优 TI 电场")
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
        "ernie_nsn_10_10_leadfield_ga",
    )
    _log_stage_elapsed("[5/7] 重建与导出", stage_started_at, run_started_at)
    stage_started_at = time.perf_counter()
    direct_optimized = None
    if config.run_direct_fem_validation:
        logger.info("[6/7] 对最优解运行直接 FEM 对照")
        validation_dir = config.result_dir / "final_validation"
        optimized_mesh = _run_direct_fem_case(
            config,
            validation_dir / "optimized",
            optimized.electrode_names,
            optimized.metrics,
        )
        direct_optimized = _direct_fem_metrics(config, optimized_mesh)
    else:
        logger.info("[6/7] 已禁用直接 FEM 对照")
    _log_stage_elapsed("[6/7] 直接 FEM 对照", stage_started_at, run_started_at)
    stage_started_at = time.perf_counter()
    logger.info("[7/7] 写入 focality 对照并执行验收")
    comparison = _comparison_payload(
        config,
        optimized.metrics,
        direct_optimized,
    )
    comparison_path = write_comparison(
        config.result_dir / "final_validation" / "comparison.json",
        comparison,
    )
    _log_stage_elapsed("[7/7] 结果验收", stage_started_at, run_started_at)
    if not comparison["validation"]["passed"]:
        raise RuntimeError(f"leadfield GA demo 效果验证未通过: {comparison_path}")
    logger.info("leadfield GA demo 效果验证通过: %s", comparison_path)
    return comparison_path


def main() -> None:
    """执行 NSN 10-10 leadfield GA demo 并记录总实际耗时。

    Returns
    -------
    None
        结果写入 ``data/ti_leadfield_ga_ernie_nsn_10_10``，验收失败时抛出异常。

    Notes
    -----
    该函数是新算法的唯一运行入口；不修改或调用现有 ``ti_inverse``。
    """
    _setup_demo_logging()
    run_started_at = time.perf_counter()
    logger.info("本次 NSN 10-10 leadfield GA 运行开始")
    try:
        config = _build_demo_config()
        _validate_demo_config(config)
        _run_demo(config, run_started_at)
    finally:
        total_seconds = time.perf_counter() - run_started_at
        logger.info(
            "本次 NSN 10-10 leadfield GA 总运行耗时: total_elapsed=%s, "
            "total_elapsed_seconds=%.3f",
            _format_elapsed(total_seconds),
            total_seconds,
        )


if __name__ == "__main__":
    main()
