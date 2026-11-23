"""从登记的独立头模执行原生基线、共同正向和候选库复算。"""

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any, TypeVar

from filelock import FileLock
import numpy as np

from neuracle.atlas.registry import ATLAS_STANDARDIZED_DIR
from neuracle.demo.ti_depth.exp02 import BN
from neuracle.ti_depth_study import baseline, fields, figures, metrics, records
from neuracle.ti_depth_study.baseline import FIELD_MODEL, PHYSICS, candidate_layouts, content_id, forward_config, inverse_config, run_task
from neuracle.ti_depth_study.fields import load_geometry, prepare_field, prepare_geometry
from neuracle.ti_depth_study.figures import plot_roi_slices
from neuracle.ti_depth_study.metrics import measure_field
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json
from simnibs.utils.mesh_element_properties import ElementTags


INVERSE_OUTPUTS = {
    "leadfield_based": ("electrode_mapping.json", "*_optimization.msh", "*_optimization_max_TI.nii.gz"),
    "leadfield_free": ("electrode_mapping.json", "mapped_electrodes_simulation/*_tes_mapped_opt_head_mesh.msh", "*_optimization_max_TI.nii.gz"),
}
FORWARD_OUTPUTS = ("*_TDCS_1_scalar.msh", "*_TDCS_2_scalar.msh", "*_TI.msh", "*_max_TI.nii.gz")
CASE_COLUMNS = ("case_id", "subject", "target", "method", "arm", "seed", "condition", "task_id", "status", "reason", "wall_s")
BASE_COLUMNS = ("case_id", "subject", "target", "method", "inverse_task", "forward_task", "IA_mA", "IB_mA", "mesh_hash", "field_id", "D_hot_mm", "Cov010", "Q50_T", "Q99_Cout", "Vout010", "status")
T = TypeVar("T")
SOLVER_ERRORS = (OSError, RuntimeError, ValueError, KeyError)


def write_report(study: Path, out: Path, run: dict[str, Any], cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """按案例更新运行进度和基线表，保留失败与待开始的格子。

    Parameters
    ----------
    study, out : Path
        本地研究根及基线运行目录。
    run : dict
        当前阶段运行清单。
    cases, rows : list of dict
        按案例 ID 登记的流程状态与绝对场读数。
    """
    write_csv(out / "cases.csv", cases, CASE_COLUMNS)
    columns = tuple(dict.fromkeys([*BASE_COLUMNS, *(key for row in rows for key in row)]))
    write_csv(out / "baseline_fields.csv", rows, columns)
    write_json(out / "run.json", run)
    native = [row for row in cases if row["arm"] == "native_roc"]
    counts = Counter(row["status"] for row in native)
    summary = [f"# 03 原生基线与共同正向场\n\n阶段 `{run['stage']}`，运行状态 `{run['status']}`。\n",
               "| 路线 | 具名正向场及指标 | 失败 | 待完成 | 总数 |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in ("leadfield_based", "leadfield_free"):
        statuses = Counter(row["status"] for row in native if row["method"] == method)
        summary.append(f"| {method} | {statuses['ok']} | {statuses['failed']} | {statuses['pending'] + statuses['running']} | {statuses.total()} |")
    summary.append("\n[案例与失败原因](cases.csv) · [基线读数](baseline_fields.csv) · [共享输入](shared_inputs.csv) · [实际命令](commands.ps1)")
    pool = Counter(row["status"] for row in cases if row["arm"] == "candidate_pool")
    if pool:
        summary.insert(-1, f"\n候选共同场已完成 {pool['ok']} 格，失败 {pool['failed']} 格，运行中 {pool['running']} 格。")
    report = out / "report.md"
    report.write_text("\n".join(summary) + "\n", encoding="utf-8")
    update_index(study, dict(experiment="03", run_id=run["run_id"], stage=run["stage"], run_status=run["status"],
                 scientific_decision="基线", ok_cases=counts["ok"], failed_cases=counts["failed"],
                 pending_cases=counts["pending"] + counts["running"], report_path=report.relative_to(study).as_posix()))


def build_reference(study: Path, subject: str, target_table: dict[str, dict[str, str]], mesh_hash: str) -> tuple[Path, list[dict[str, str]]]:
    """以头模、变换和图谱散列定位受试者参考网格，并验收几何侧别。

    Parameters
    ----------
    study : Path
        研究根目录。
    subject : str
        当前头模 ID。
    target_table : dict
        已登记的十靶及标准化掩膜。
    mesh_hash : str
        当前参考 MSH 散列。

    Returns
    -------
    geometry_path, quality_rows : tuple
        共同几何和每靶质控读数。各格按本靶的质量状态决定能否运行。
    """
    head = study / "head_models" / f"m2m_{subject}"
    atlas_path = ATLAS_STANDARDIZED_DIR / BN / "atlas_simnibs_mni.nii.gz"
    masks = {key: Path(row["mask_path"]) for key, row in target_table.items()}
    configuration = {"mesh_hash": mesh_hash, "cortex_atlas": digest(atlas_path), "roi_inputs": {key: digest(path) for key, path in masks.items()},
                     "registration": digest(head / "toMNI/Conform2MNI_nonl.nii.gz"),
                     "inverse_registration": digest(head / "toMNI/MNI2Conform_nonl.nii.gz"),
                     "implementation": digest(Path(fields.__file__))}
    geometry_id = content_id(configuration)[:20]
    shared = study / "shared" / subject / geometry_id
    geometry_path = shared / "geometry.npz"
    qc_path = shared / "geometry_qc.csv"
    if not (geometry_path.is_file() and qc_path.is_file() and (shared / "geometry_inputs.json").is_file()):
        checks = prepare_geometry(head, atlas_path, masks, geometry_path)
        write_csv(qc_path, checks)
        geometry, tags, _ = load_geometry(geometry_path)
        grey = tags == ElementTags.GM
        write_json(shared / "geometry_inputs.json", configuration | {
            "mesh": str(head / f"{subject}.msh"), "geometry_sha256": digest(geometry_path),
            "cortex_fraction_of_gm": float(geometry.volume_mm3[geometry.cortex_mask].sum() / geometry.volume_mm3[grey].sum()),
            "gm_outside_cortex_mm3": float(geometry.volume_mm3[grey & ~geometry.cortex_mask].sum()),
        })
    registered = json.loads((shared / "geometry_inputs.json").read_text(encoding="utf-8"))
    if digest(geometry_path) != registered["geometry_sha256"]:
        raise ValueError(f"共同参考网格散列已改变: {geometry_path}")
    return geometry_path, read_csv(qc_path)


def align_common_forward(directory: Path, geometry: Path, field_dir: Path) -> tuple[Path, dict[str, float]]:
    """按头模几何和载波输入散列索引共同场，可复用已对齐缓存。

    Parameters
    ----------
    directory : Path
        已核验退出码的共同正向 CLI 目录。
    geometry, field_dir : Path
        固定几何及对齐载波的输出目录。

    Returns
    -------
    field_path, alignment : tuple
        对齐后的 NPZ 与精确、取样、缺失体积。
    """
    forward = json.loads((directory / "study_execution.json").read_text(encoding="utf-8"))
    config = {"geometry": digest(geometry), "outputs": forward["sha256"], "model": FIELD_MODEL}
    field_id = content_id(config)[:24]
    output = field_dir / f"{field_id}.npz"
    manifest = output.with_suffix(".json")
    if output.is_file() and manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata["sha256"] == digest(output):
            return output, metadata["alignment"]
    mesh_a = next(directory.glob(FORWARD_OUTPUTS[0]))
    mesh_b = next(directory.glob(FORWARD_OUTPUTS[1]))
    alignment = prepare_field(mesh_a, mesh_b, geometry, output)
    write_json(manifest, {"source": str(directory), "sha256": digest(output), "alignment": alignment, "inputs": config})
    return output, alignment


def real_metric_check(study: Path, protocol: Path, record: dict[str, Any], geometry_path: Path, field_path: Path) -> None:
    """用参考四面体顶点的行列式独立核算首个正向场的覆盖率。

    Parameters
    ----------
    study, protocol : Path
        研究根和实验 02 的运行目录。
    record : dict
        已生成的共同场指标和案例键。
    geometry_path, field_path : Path
        参考几何及对齐场，其散列由 03 登记。
    """
    checks = read_csv(protocol / "metric_checks.csv")
    item = next(row for row in checks if row["check"] == "common_forward")
    if item["status"] == "ok":
        return
    geometry, _, targets = load_geometry(geometry_path)
    chosen = targets[record["target"]]
    nodes = geometry.node_mm[geometry.connectivity[chosen]]
    volume = np.abs(np.linalg.det(nodes[:, 1:] - nodes[:, :1])) / 6
    with np.load(field_path) as data:
        above = data["max_TI"][chosen] >= 0.10
    numerator, denominator = float(volume[above].sum()), float(volume.sum())
    manual = numerator / denominator
    center = nodes[0].sum(axis=0) / 4
    match = np.isclose(manual, record["Cov010"], atol=1e-12) and np.allclose(center, geometry.centroid_mm[chosen][0], atol=1e-8)
    item.update(expected=str(manual), observed=str(record["Cov010"]), status="ok" if match else "failed",
                evidence=f"{record['case_id']} 覆盖 {numerator} / {denominator} mm³，元素 {geometry.element_id[chosen][0]} 手算重心 {center.tolist()}")
    write_csv(protocol / "metric_checks.csv", checks)
    protocol_cases = read_csv(protocol / "cases.csv")
    next(row for row in protocol_cases if row["case_id"] == "common_forward").update(status=item["status"], reason=item["evidence"])
    write_csv(protocol / "cases.csv", protocol_cases)
    protocol_report = protocol / "report.md"
    protocol_report.write_text(protocol_report.read_text(encoding="utf-8").replace("真实共同场复核状态为 pending。", f"真实共同场复核状态为 {item['status']}。"), encoding="utf-8")
    protocol_artifacts = read_csv(protocol / "artifacts.csv")
    touched = { (protocol / name).relative_to(study).as_posix() for name in ("cases.csv", "metric_checks.csv", "report.md") }
    for row in protocol_artifacts:
        if row["path"] in touched:
            row["sha256"] = digest(study / row["path"])
    protocol_artifacts.extend([artifact(study, "common_forward", "real_metric_input", path) for path in (geometry_path, field_path)])
    write_csv(protocol / "artifacts.csv", protocol_artifacts)
    index = next(row for row in read_csv(study / "results/index.csv") if row["experiment"] == "02" and row["run_id"] == protocol.name)
    index.update(ok_cases=sum(row["status"] == "ok" for row in checks), failed_cases=sum(row["status"] == "failed" for row in checks), pending_cases=0)
    update_index(study, index)
    if not match:
        raise ValueError("真实共同场的覆盖体积或坐标复核失败")


def head_ready(head: Path, subject: str, source_hash: str) -> bool:
    """用 CHARM 的最终产物和原始扫描散列验收现有头模。

    Parameters
    ----------
    head, subject : Path and str
        CHARM 的 m2m 输出目录和头模 ID。
    source_hash : str
        01 审计登记的原始 T1 SHA-256。

    Returns
    -------
    bool
        原扫描一致、必需产物齐全且 CHARM 已完成。正常成功后 CHARM 会删除续跑标记。
    """
    required = (f"{subject}.msh", "T1.nii.gz", "final_tissues.nii.gz", "surface.gii.gz",
                "eeg_positions/EEG10-10_NSN.csv", "toMNI/Conform2MNI_nonl.nii.gz", "toMNI/MNI2Conform_nonl.nii.gz")
    if not all((head / name).is_file() and (head / name).stat().st_size for name in required):
        return False
    marker = head / "study_source_sha256.txt"
    progress = head / ".progress.txt"
    return (marker.is_file() and marker.read_text(encoding="utf-8-sig").strip() == source_hash
            and (not progress.exists() or progress.read_text().strip() == "100"))


def solve_cases(
    cases: list[dict[str, Any]], jobs: int, commands: Path,
    solve: Callable[[dict[str, Any], Path], T], publish: Callable[[], None],
) -> Iterator[tuple[dict[str, Any], Future[T], Path, float]]:
    """按容量提交独立原生求解，主进程收到结果后继续调度。

    Parameters
    ----------
    cases : list of dict
        已赋予稳定任务 ID 的病例或候选组。
    jobs : int
        同时运行的求解子进程上限。
    commands : Path
        总命令记录。各任务的调用另记于同级 task_commands。
    solve, publish : callable
        工作线程的求解入口和主进程的状态发布入口。

    Yields
    ------
    case, future, command_file, started : tuple
        已完成的任务、可提取结果的 Future、命令文件及其开始时间。
    """
    command_dir = commands.parent / "task_commands"
    command_dir.mkdir(exist_ok=True)
    pending = iter(cases)
    active: dict[Future[T], tuple[dict[str, Any], Path, float]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        while True:
            while len(active) < jobs and (case := next(pending, None)) is not None:
                script = command_dir / f"{case['task_id']}.ps1"
                if not script.exists():
                    script.write_text("conda activate simnibs_env\n", encoding="utf-8")
                with commands.open("a", encoding="utf-8") as stream:
                    stream.write(f'\n& "$PSScriptRoot/task_commands/{script.name}"\n')
                case["status"] = "running"
                started = monotonic()
                active[executor.submit(solve, case, script)] = (case, script, started)
                publish()
            if not active:
                return
            future = next(as_completed(active))
            case, script, started = active.pop(future)
            yield case, future, script, started


def solve_native(
    study: Path, target_table: dict[str, dict[str, str]], qc: dict[str, str], case: dict[str, Any], commands: Path,
) -> tuple[Path, dict[str, Any], Path | None, Exception | None]:
    """求解一个原生病例，前向失败时保留已完成的逆向布局。

    Parameters
    ----------
    study, commands : Path
        研究根和此病例的独立命令文件。
    target_table, qc, case : dict
        登记的 atlas 靶区表、个体空间质控状态及病例配置。

    Returns
    -------
    inverse, mapping, forward, error : tuple
        已通过验收的逆向目录与电极映射，以及正向目录或失败原因。
    """
    subject, target, method, task = (case[key] for key in ("subject", "target", "method", "task_id"))
    if qc[target] != "ok":
        raise ValueError(f"个体空间靶区质控: {qc[target]}")
    inverse = run_task(study, "ti_inverse", subject, task, inverse_config(target_table[target], method), commands, INVERSE_OUTPUTS[method])
    mapping = json.loads((inverse / "electrode_mapping.json").read_text(encoding="utf-8"))
    try:
        config = forward_config(mapping["mapped_labels"], (mapping["current_A"], mapping["current_B"]))
        forward = run_task(study, "ti_forward", subject, task, config, commands, FORWARD_OUTPUTS)
    except SOLVER_ERRORS as error:
        return inverse, mapping, None, error
    return inverse, mapping, forward, None


def solve_candidate(study: Path, layouts: dict[str, tuple[str, ...]], case: dict[str, Any], commands: Path) -> Path:
    """按候选登记的帽位和统一电流完成一格正向求解。

    Parameters
    ----------
    study, commands : Path
        研究根和此候选的独立命令文件。
    layouts, case : dict
        候选键到具名布局的映射及该候选的求解状态。

    Returns
    -------
    Path
        已验收的共同正向任务目录。
    """
    labels = layouts[case["case_id"]]
    config = forward_config(list(labels), ([1., -1.], [1., -1.]))
    return run_task(study, "ti_forward", case["subject"], case["task_id"], config, commands, FORWARD_OUTPUTS)


def run(study: Path, run_id: str, stage: str, jobs: int = 1) -> None:
    """运行已登记阶段的基线和两路各 1 mA 的候选共同正向场。

    Parameters
    ----------
    study : Path
        含审计、协议及独立头模的研究目录。
    run_id, stage : str
        03 运行号和 pilot 或 development 阶段。
    jobs : int, optional
        同一头模的并发求解数，默认 1。
    """
    protocol = study / "results/02/run-001"
    target_table = {row["target"]: row for row in read_csv(protocol / "targets.csv")}
    matrix = [row for row in read_csv(protocol / "matrix.csv")
              if row["planned_stage"] == "pilot" or (stage == "development" and row["split"] == "development")]
    # based 先生成并复用同一头模的 leadfield；free 仍经原生 CLI 求解。
    matrix.sort(key=lambda row: (row["subject"], row["target"] != "stn295", row["target"], row["method"]))
    out = study / "results/03" / run_id
    out.mkdir(parents=True, exist_ok=True)
    commands = out / "commands.ps1"
    if not commands.exists():
        commands.write_text("conda activate simnibs_env\n", encoding="utf-8")
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {timestamp()} {Path.cwd()}\n# python -m neuracle.demo.ti_depth.exp03 --study-root '{study}' --run-id {run_id} --stage {stage} --jobs {jobs}\n")
    config = PHYSICS | {"threshold_vpm": 0.10, "candidate_current_mA": [1.0, 1.0], "jobs": jobs}
    run_info: dict[str, Any] = dict(experiment="03", run_id=run_id, stage=stage, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                   environment=sys.executable, upstream_runs=["01/run-001", "02/run-001"], data_id={},
                                   config=config, started_at=timestamp(), finished_at=None, status="running")
    cases = [dict(case_id=row["case_id"], subject=row["subject"], target=row["target"], method=row["method"], arm="native_roc",
                  seed="NA", condition="t010", task_id="NA", status="pending", reason="NA", wall_s="NA") for row in matrix]
    rows: list[dict[str, Any]] = []
    shared_inputs = []
    artifacts = [artifact(study, "run", "implementation", Path(source))
                 for source in (__file__, baseline.__file__, fields.__file__, figures.__file__, metrics.__file__, records.__file__)]
    publish = partial(write_report, study, out, run_info, cases, rows)
    publish()
    for subject in dict.fromkeys(row["subject"] for row in matrix):
        selected = [row for row in cases if row["subject"] == subject]
        head = study / "head_models" / f"m2m_{subject}"
        main_mesh = head / f"{subject}.msh"
        expected_hash = next(row["raw_T1_sha256"] for row in matrix if row["subject"] == subject)
        if not head_ready(head, subject, expected_hash):
            for case in selected:
                case.update(reason="等待当前原始 T1 散列对应的独立头模")
            continue
        mesh_hash = digest(main_mesh)
        print(f"{subject} 共同评估网格和靶区", flush=True)
        try:
            geometry_path, quality = build_reference(study, subject, target_table, mesh_hash)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            for case in selected:
                case.update(status="failed", reason=f"共同评估网格: {error}")
                rows.append({key: case[key] for key in ("case_id", "subject", "target", "method", "status", "reason")})
            print(str(error), flush=True)
            write_report(study, out, run_info, cases, rows)
            continue
        geo, _, targets = load_geometry(geometry_path)
        qc = {row["target"]: row["status"] for row in quality}
        source_pool: dict[tuple[str, ...], set[str]] = {}
        artifacts.extend(artifact(study, subject, role, path) for role, path in (
            ("reference_mesh", main_mesh), ("reference_geometry", geometry_path), ("geometry_qc", geometry_path.with_name("geometry_qc.csv")),
            ("cap", head / "eeg_positions/EEG10-10_NSN.csv"), ("reference_T1", head / "T1.nii.gz")))
        artifacts.extend(artifact(study, subject, "subject_roi", Path(row["roi_path"])) for row in quality)
        for key in dict.fromkeys(row["target"] for row in selected if qc[row["target"]] == "ok"):
            figure = out / "figures" / f"{subject}_{key}_roi.png"
            if not figure.is_file():
                plot_roi_slices(head / "T1.nii.gz", geometry_path.parent / "roi" / f"{key}.nii.gz", figure, {})
            artifacts.append(artifact(study, f"{subject}_{key}", "roi_overlay", figure))
        stn_fields: dict[str, Path] = {}
        for case in selected:
            case["task_id"] = f"03_{run_id}_{case['case_id']}_t010_{mesh_hash[:12]}"
        native_solver = partial(solve_native, study, target_table, qc)
        for case, future, command_file, started in solve_cases(selected, jobs, commands, native_solver, publish):
            case_id, target, method = (case[key] for key in ("case_id", "target", "method"))
            record: dict[str, Any] = dict(case_id=case_id, subject=subject, target=target, method=method, mesh_hash=mesh_hash)
            rows.append(record)
            artifacts.append(artifact(study, case_id, "commands", command_file))
            try:
                inverse, mapping, forward, error = future.result()
                record["inverse_task"] = inverse.name
                labels = mapping["mapped_labels"]
                currents = (mapping["current_A"], mapping["current_B"])
                record.update(IA_mA=currents[0][0], IB_mA=currents[1][0], electrode_names=";".join(labels))
                source_pool.setdefault(tuple(labels), set()).add(f"{method}:{case_id}")
                for name in ("params.json", "electrode_mapping.json", "study_console.log", "optimization_result.json", "leadfield_reference.json", "convergence.csv"):
                    if (inverse / name).is_file():
                        artifacts.append(artifact(study, case_id, "inverse", inverse / name))
                artifacts.extend(artifact(study, case_id, "inverse_field", path) for path in inverse.rglob("*.msh"))
                if error is not None:
                    raise error
                assert forward is not None
                record["forward_task"] = forward.name
                field, alignment = align_common_forward(forward, geometry_path, geometry_path.parent / "fields")
                with np.load(field) as data:
                    measured = measure_field(geo, targets[target], data["E_A"], data["E_B"], data["valid_mask"])
                record.update(measured, field_id=field.stem, field_path=field.relative_to(study).as_posix(), **alignment)
                artifacts.extend(artifact(study, case_id, "common_forward", path) for pattern in FORWARD_OUTPUTS for path in forward.glob(pattern))
                artifacts.extend(artifact(study, case_id, "common_forward", forward / name) for name in ("params.json", "study_console.log"))
                artifacts.append(artifact(study, case_id, "aligned", field))
                if measured["status"] != "ok":
                    raise ValueError(f"共同场指标: {measured['status']}")
                real_metric_check(study, protocol, record, geometry_path, field)
                if target == "stn295":
                    stn_fields[method] = next(forward.glob(FORWARD_OUTPUTS[-1]))
                case.update(status="ok", reason="NA")
            except SOLVER_ERRORS as error:
                case.update(status="failed", reason=str(error))
                print(str(error), flush=True)
            record.update(status=case["status"], reason=case["reason"])
            case["wall_s"] = round(monotonic() - started, 3)
            write_csv(out / "artifacts.csv", artifacts)
            publish()
        if stn_fields:
            figure = out / "figures" / f"{subject}_stn295_common_forward.png"
            plot_roi_slices(head / "T1.nii.gz", geometry_path.parent / "roi/stn295.nii.gz", figure, stn_fields)
            artifacts.append(artifact(study, subject, "common_forward_overlay", figure))
        if not source_pool:
            continue
        cap = head / "eeg_positions/EEG10-10_NSN.csv"
        layouts = candidate_layouts(cap, source_pool)
        geometry_hash = digest(geometry_path)
        pool_config = {"geometry": geometry_hash, "cap": digest(cap), "field_model": FIELD_MODEL, "physics": PHYSICS,
                       "current_mA": [1.0, 1.0], "layouts": [(list(labels), sorted(origins)) for labels, origins in sorted(layouts.items())]}
        data_id = content_id(pool_config)[:20]
        pool_dir = study / "shared" / subject / data_id
        pool_dir.mkdir(exist_ok=True)
        geometry_link = pool_dir / "geometry.npz"
        if not geometry_link.exists():
            geometry_link.hardlink_to(geometry_path)
        index = pool_dir / "candidates.csv"
        candidates = []
        pool_cases = []
        pool_layouts = {}
        for labels, origins in sorted(layouts.items()):
            identifier = content_id({"subject": subject, "mesh": mesh_hash, "cap": pool_config["cap"], "layout": labels,
                                     "field_model": FIELD_MODEL, "physics": PHYSICS, "current": 1.0})[:20]
            case = dict(case_id=identifier, subject=subject, target="NA", method="shared_pool", arm="candidate_pool", seed="NA", condition="1+1mA",
                        task_id=f"03_{run_id}_{subject}_pool_{identifier}", status="pending", reason="NA", wall_s="NA")
            pool_cases.append(case)
            pool_layouts[identifier] = labels
        cases.extend(pool_cases)
        candidate_solver = partial(solve_candidate, study, pool_layouts)
        for case, future, command_file, started in solve_cases(pool_cases, jobs, commands, candidate_solver, publish):
            identifier = case["case_id"]
            labels = pool_layouts[identifier]
            origins = layouts[labels]
            artifacts.append(artifact(study, identifier, "commands", command_file))
            row: dict[str, Any] = dict(candidate_id=identifier, origin_methods=";".join(sorted(origins)), montage=PHYSICS["montage"],
                        A_plus=labels[0], A_minus=labels[1], B_plus=labels[2], B_minus=labels[3], IA_mA=1.0, IB_mA=1.0,
                        field_id="NA", geometry_hash=geometry_hash, field_model=FIELD_MODEL, field_path="NA", status="pending", reason="NA")
            try:
                forward = future.result()
                field, alignment = align_common_forward(forward, geometry_path, geometry_path.parent / "fields")
                row.update(field_id=field.stem, field_path=field.relative_to(study).as_posix())
                if "pool_disk_estimate_bytes" not in run_info:
                    run_info["pool_disk_estimate_bytes"] = field.stat().st_size * len(layouts)
                if alignment["missing_mm3"] > 0:
                    raise ValueError(f"候选场缺失 {alignment['missing_mm3']} mm³")
                case.update(status="ok")
                artifacts.append(artifact(study, identifier, "candidate_field", field))
            except SOLVER_ERRORS as error:
                case.update(status="failed", reason=str(error))
                print(str(error), flush=True)
            case["wall_s"] = round(monotonic() - started, 3)
            row.update(status=case["status"], reason=case["reason"])
            candidates.append(row)
            write_csv(index, sorted(candidates, key=lambda item: item["candidate_id"]))
            publish()
        run_info["data_id"][subject] = data_id
        for role, path in (("geometry", geometry_link), ("candidate_index", index)):
            shared_inputs.append(dict(subject=subject, data_id=data_id, role=role, path=path.relative_to(study).as_posix(), sha256=digest(path)))
        write_csv(out / "shared_inputs.csv", shared_inputs)
    write_csv(out / "artifacts.csv", artifacts)
    run_info.update(finished_at=timestamp(), status="complete" if all(row["status"] == "ok" for row in cases) else "partial")
    write_report(study, out, run_info, cases, rows)


def main() -> None:
    """从仓库根解析 03 的本地研究入口参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须至少为 1")
    study = args.study_root.resolve()
    out = study / "results/03" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".scheduler.lock"), timeout=0):
        run(study, args.run_id, args.stage, args.jobs)


if __name__ == "__main__":
    main()
