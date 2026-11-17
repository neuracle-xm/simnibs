"""正式 leadfield-based 入口：共享缓存、业务 ROI、固定 GA 和统一输出。"""

import logging
import time
from pathlib import Path

from neuracle.parameters.schemas import AnisotropyType, ROIParam
from neuracle.ti_leadfield_optimization.fitness import LeadfieldFitnessEvaluator
from neuracle.ti_leadfield_optimization.leadfield import (
    _read_manifest,
    _sha256_file,
    _write_json_atomic,
    ensure_leadfield,
    load_leadfield,
)
from neuracle.ti_leadfield_optimization.models import GASettings, LeadfieldConfig
from neuracle.ti_leadfield_optimization.optimizer import (
    _require_genetic_algorithm,
    run_genetic_optimization,
)
from neuracle.ti_leadfield_optimization.regions import prepare_region_masks
from neuracle.ti_leadfield_optimization.result import (
    export_result_nifti,
    write_convergence_csv,
    write_optimization_result,
    write_result_mesh,
)
from neuracle.utils import find_montage_file
from neuracle.utils.constants import NON_ROI_THRESHOLD
from neuracle.utils.find_nifty import find_optional_nifti_file
from neuracle.utils.optimization_result import write_electrode_mapping
from neuracle.utils.optimization_roi import (
    build_optimization_rois,
    resolve_roi_settings,
)

logger = logging.getLogger(__name__)


def run_ti_leadfield_inverse(
    head_model_id: str,
    head_model_dir: str,
    output_dir: str,
    montage: str,
    roi_type: str,
    roi_param: ROIParam,
    target_threshold: float,
    conductivity_config: dict[str, float],
    anisotropy: AnisotropyType,
    data_root: str,
    n_workers: int = 8,
) -> None:
    """按正式业务输入运行优化，GA 与电极几何固定在计算层。

    Parameters
    ----------
    head_model_id : str
        业务头模 ID，仅用于缓存生命周期隔离。
    head_model_dir : str
        实际 m2m 目录，物理文件前缀按目录解析。
    output_dir : str
        单次仿真输出目录。
    montage : str
        当前头模 montage 名称。
    roi_type, roi_param : str, ROIParam
        正式 atlas/MNI 区域输入。
    target_threshold : float
        ROI 阈值，单位 V/m。
    conductivity_config : dict[str, float]
        后端快照中完整的组织电导率。
    anisotropy : AnisotropyType
        实际各向异性模式，非 scalar 必须找到 tensor。
    data_root : str
        缓存位于 leadfields/<head_model_id>，独立于任务。
    n_workers : int
        FEM 原生计算线程预算，固定单进程复用 PARDISO，不改变 GA 设置。
    """
    _require_genetic_algorithm()
    subject = Path(head_model_dir)
    output = Path(output_dir)
    if (
        not head_model_id
        or any(char in head_model_id for char in ("/", "\\", ":"))
        or head_model_id in (".", "..")
    ):
        raise ValueError("非法头模 ID")
    prefix = subject.resolve().name.removeprefix("m2m_")
    mesh_path = subject / (prefix + ".msh")
    t1_path = find_optional_nifti_file(subject, ("T1.nii.gz", "T1.nii"))
    if not mesh_path.is_file() or t1_path is None:
        raise FileNotFoundError(f"头模 mesh 或 T1 缺失: {subject}")
    dti_path = find_optional_nifti_file(
        subject, ("DTI_coregT1_tensor.nii.gz", "DTI_coregT1_tensor.nii")
    )
    roi_settings = resolve_roi_settings(roi_type, roi_param)
    config = LeadfieldConfig(
        head_model_dir=subject,
        mesh_path=mesh_path,
        montage_path=Path(find_montage_file(str(subject), montage)),
        leadfield_dir=Path(data_root) / "leadfields" / head_model_id,
        conductivity_config=conductivity_config,
        anisotropy_type=AnisotropyType(anisotropy).value,
        dti_path=Path(dti_path) if dti_path is not None else None,
        n_workers=n_workers,
    )
    settings = GASettings(
        roi_threshold_v_per_m=target_threshold,
        non_roi_threshold_v_per_m=min(target_threshold, NON_ROI_THRESHOLD),
    )
    path = ensure_leadfield(config, output)
    load_started = time.perf_counter()
    leadfield = load_leadfield(path)
    reference = _read_manifest(output / "leadfield_reference.json")
    reference["load_seconds"] = time.perf_counter() - load_started
    evidence_paths = [Path(t1_path)] + sorted((subject / "toMNI").glob("*"))
    if "roi_mask_path" in roi_settings:
        evidence_paths.append(Path(roi_settings["roi_mask_path"]))
    reference["roi_export_inputs"] = [
        {"path": str(item), "sha256": _sha256_file(item)}
        for item in evidence_paths
        if item.is_file()
    ]
    _write_json_atomic(output / "leadfield_reference.json", reference)
    regions = build_optimization_rois(str(subject), leadfield.mesh, **roi_settings)
    masks = prepare_region_masks(leadfield.mesh, regions)
    evaluator = LeadfieldFitnessEvaluator(
        leadfield,
        masks,
        settings.non_roi_threshold_v_per_m,
        settings.roi_threshold_v_per_m,
    )
    result = run_genetic_optimization(evaluator, settings)
    field_a, field_b, max_ti = evaluator.reconstruct_fields(
        result.chromosome, result.metrics.currents
    )
    mesh_result = write_result_mesh(
        output / (prefix + "_optimization.msh"),
        leadfield,
        masks,
        field_a,
        field_b,
        max_ti,
    )
    export_result_nifti(mesh_result, output, t1_path, prefix + "_optimization")
    write_optimization_result(
        output / "optimization_result.json",
        result,
        leadfield,
        settings,
        config.montage_path,
    )
    write_convergence_csv(output / "convergence.csv", result.convergence)
    currents = result.metrics.currents
    write_electrode_mapping(
        output,
        "leadfield_based",
        list(result.electrode_names),
        [currents.current_a_ma, -currents.current_a_ma],
        [currents.current_b_ma, -currents.current_b_ma],
    )
    logger.info("leadfield-based TI 逆向优化完成: %s", output)
