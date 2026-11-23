"""对固定具名布局施加单电极帽位偏差并复算共同正向场。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

from filelock import FileLock
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial.distance import pdist

from neuracle.demo.ti_depth.exp02 import METHODS, PILOT_TARGETS, TARGETS
from neuracle.demo.ti_depth.exp03 import CASE_COLUMNS, FORWARD_OUTPUTS, SOLVER_ERRORS, align_common_forward, solve_cases
from neuracle.ti_depth_study.baseline import content_id, run_task
from neuracle.ti_depth_study.diagnostics import load_case
from neuracle.ti_depth_study.metrics import FloatArray, measure_field
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json
from neuracle.utils import find_montage_file
from simnibs.mesh_tools.mesh_io import Msh, read_msh
from simnibs.utils.csv_reader import read_csv_positions, write_csv_positions
from simnibs.utils.mesh_element_properties import ElementTags


ROLES = ("A+", "A−", "B+", "B−")
OFFSETS = tuple((role, tangent, sign) for role in ROLES for tangent in ("t1", "t2") for sign in (-1, 1))
SHIFT_MM = 2.0
CLEARANCE_MM = 5.0
ENDPOINTS = ("D_hot_mm", "Cov010", "Q50_T", "Q99_Cout", "Vout010", "absolute_hit", "hot_overlap")
OFFSET_COLUMNS = ("layout_id", "electrode_role", "tangent", "sign", "requested_mm", "actual_mm", "cap_file", "geometry_valid",
                  "forward_task", "D_hot_mm", "Cov010", "Q50_T", "Q99_Cout", "Vout010", "absolute_hit", "hot_overlap",
                  "cap_sha256", "field_path", "field_sha256", "electrode", "nominal_xyz_mm", "shifted_xyz_mm", "min_center_mm")
GROUPS = {row[0]: row[3] for row in TARGETS}


def shifted_positions(mesh: Msh, centers: FloatArray) -> FloatArray:
    """按局部投影三角面生成两条正交切向并重新投影。

    Parameters
    ----------
    mesh : Msh
        包含个体头皮表面的原始头模。
    centers : ndarray of shape (4, 3)
        A+、A−、B+、B− 的原帽位坐标，单位 mm。

    Returns
    -------
    ndarray of shape (4, 4, 3)
        每个角色的 t1−、t1+、t2−、t2+ 四个偏移坐标。
    """
    projected, face_ids = mesh.project_points_on_surface(centers, ElementTags.SCALP_TH_SURFACE, return_element_index=True)
    triangle = mesh.nodes.node_coord[mesh.elm.node_number_list[face_ids - 1, :3] - 1]
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    length = np.linalg.norm(normal, axis=1)
    if np.any(length == 0) or not np.isfinite(normal).all():
        raise ValueError("名义帽位的头皮三角面无法定义法向")
    normal /= length[:, None]
    axis = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
    tangent_1 = np.cross(normal, axis)
    tangent_1 /= np.linalg.norm(tangent_1, axis=1)[:, None]
    tangent_2 = np.cross(normal, tangent_1)
    offsets = np.stack((-tangent_1, tangent_1, -tangent_2, tangent_2), axis=1)
    requests = (projected[:, None, :] + SHIFT_MM * offsets).reshape(-1, 3)
    return mesh.project_points_on_surface(requests, ElementTags.SCALP_TH_SURFACE).reshape(4, 4, 3)


def plan_offsets(
    study: Path, out: Path, mesh: Msh, cap: Path, cap_hash: str, source: dict[str, str], config: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """登记单电极扰动，按内容索引绝对帽位路径和物理任务 ID。

    Parameters
    ----------
    study, out : Path
        研究根和当前实验输出目录。
    mesh, cap, cap_hash : Msh, Path, str
        经散列核对的个体头皮、帽位及其指纹。
    source, config : dict
        03 共同场证据行和该正向任务原始参数。
    scenarios : list of dict
        当前布局的十六个预登记扰动行。

    Returns
    -------
    list of tuple
        几何有效的扰动行及其正向输入，原无效行仍保留。
    """
    kinds, coordinates, extra, names, extra_cols, header = read_csv_positions(str(cap))
    labels = [item["name"] for key in ("electrode_A", "electrode_B") for item in config[key]]
    indexed = {name: i for i, (kind, name) in enumerate(zip(kinds, names)) if kind in {"Electrode", "ReferenceElectrode"}}
    if len(labels) != 4 or len(set(labels)) != 4 or ";".join(labels) != source["electrode_names"]:
        raise ValueError("正向配置与 03 不匹配，或缺四个互异的具名电极")
    indices = [indexed[label] for label in labels]
    centers = coordinates[indices]
    positions = shifted_positions(mesh, centers).reshape(-1, 3)
    spacing = 2 * float(config["electrode_radius"]) + CLEARANCE_MM
    planned = []
    for number, (row, point) in enumerate(zip(scenarios, positions)):
        index = number // 4
        if not np.isfinite(point).all():
            row.update(geometry_valid=False, status="failed", reason="头皮投影返回非有限坐标")
            continue
        moved = centers.copy()
        moved[index] = point
        actual_mm = float(np.linalg.norm(centers[index] - point))
        minimum_mm = float(pdist(moved).min())
        geometry_valid = minimum_mm >= spacing and actual_mm > 0
        row.update(electrode=labels[index], actual_mm=actual_mm, nominal_xyz_mm=json.dumps(centers[index].tolist()),
                   shifted_xyz_mm=json.dumps(point.tolist()), min_center_mm=minimum_mm, geometry_valid=geometry_valid)
        if not geometry_valid:
            row.update(status="failed", reason="位移为零或电极圆片间距不足")
            continue
        geometry_id = content_id({"cap": cap_hash, "electrode": labels[index], "point": point.tolist()})[:24]
        shifted_cap = out / "geometry" / f"{source['subject']}_{geometry_id}.csv"
        points = coordinates.copy()
        points[indices[index]] = point
        write_csv_positions(shifted_cap, kinds, points, names, extra=extra, extra_cols=extra_cols, header=header)
        shifted_hash = digest(shifted_cap)
        params = config | {"montage": str(shifted_cap)}
        task_config = {"mesh": source["mesh_hash"], "params": config | {"montage": shifted_hash}}
        task_id = f"18_{out.name}_{source['subject']}_offset_{content_id(task_config)[:24]}"
        row.update(cap_file=shifted_cap.relative_to(study).as_posix(), cap_sha256=shifted_hash, task_id=task_id,
                   status="pending", reason="等待共同正向复算")
        planned.append((row, params))
    return planned


def plot_offsets(path: Path, details: list[dict[str, Any]]) -> None:
    """绘制实际位移和同布局的最差正向定位。

    Parameters
    ----------
    path : Path
        图像文件路径。
    details : list of dict
        含实际位移、布局标识和共同正向指标的明细。
    """
    figure = Figure(figsize=(11, 3.9), constrained_layout=True)
    FigureCanvasAgg(figure)
    geometry, location = figure.subplots(1, 2)
    for index, role in enumerate(ROLES):
        distances = [row["actual_mm"] for row in details if row["electrode_role"] == role and "actual_mm" in row]
        geometry.scatter([index] * len(distances), distances, s=12, alpha=0.55)
    geometry.axhline(SHIFT_MM, color="0.5", linestyle="--")
    geometry.set(xticks=range(len(ROLES)), xticklabels=ROLES, ylabel="Actual displacement (mm)")
    by_layout = {row["layout_id"]: [] for row in details if row["electrode_role"] == "NA"}
    for row in details:
        if row["status"] == "ok" and row["electrode_role"] != "NA":
            by_layout[row["layout_id"]].append(row["D_hot_mm"])
    measured = {layout: max(values) for layout, values in by_layout.items() if values}
    if measured:
        labels = [label.removesuffix("_native").replace("_leadfield_", " ") for label in measured]
        location.barh(labels, list(measured.values()))
    else:
        location.text(0.5, 0.5, "Forward results pending", ha="center", va="center", transform=location.transAxes)
    location.set_xlabel("Worst perturbed hotspot distance (mm)")
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=140)


def publish_state(study: Path, out: Path, info: dict[str, Any], rows: list[dict[str, Any]], inputs: list[dict[str, str]]) -> None:
    """发布扰动分母、逐格门槛、报告和文件散列。

    Parameters
    ----------
    study, out : Path
        研究根和当前输出目录。
    info, rows, inputs : dict and lists of dict
        当前清单、逐场景记录和输入产物指纹。
    """
    cases = out / "cases.csv"
    offsets = out / "position_offsets.csv"
    report = out / "report.md"
    paired = out / "paired.csv"
    columns = CASE_COLUMNS + OFFSET_COLUMNS
    write_csv(cases, [{key: row[key] for key in CASE_COLUMNS} for row in rows], CASE_COLUMNS)
    write_csv(offsets, [{key: row[key] for key in columns if key in row} for row in rows], columns)
    references = {row["layout_id"]: row for row in rows if row["electrode_role"] == "NA"}
    comparisons = []
    for row in rows:
        if row["electrode_role"] != "NA":
            reference = references[row["layout_id"]]
            identity = {key: row[key] for key in ("subject", "target", "method", "arm", "seed", "condition", "case_id")}
            result = compare_endpoints(row, reference, GROUPS[row["target"]])
            reason = row["reason"] if row["status"] != "ok" else result["reason"]
            comparisons.append(identity | dict(stage=info["stage"], field_source="common_forward", reference_arm=reference["arm"],
                               reference_case_id=reference["case_id"]) | result | {"reason": reason})
    write_csv(paired, comparisons, tuple(dict.fromkeys(key for row in comparisons for key in row)))
    good = sum(row["status"] == "ok" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    pending = len(rows) - good - failed
    valid = sum(row.get("geometry_valid") is True for row in rows if row["electrode_role"] != "NA")
    content = ["# 18 帽位偏差下的布局稳定性", "", f"{info['stage']} 阶段，状态 `{info['status']}`。"
               f"正向指标已验收 {good}/{len(rows)} 格，失败 {failed} 格，待复算 {pending} 格。"
               f"投影后几何有效的扰动 {valid}/{len(comparisons)} 格。", "",
               "本次固定 03 原生解及实际有符号电流。新目标布局及稳定性配对待上游闭环正向验收。", ""]
    for layout, reference in references.items():
        subset = [row for row in rows if row["layout_id"] == layout and row["electrode_role"] != "NA"]
        measured = [row for row in subset if row["status"] == "ok"]
        passed = sum(row["reference_case_id"] == reference["case_id"] and row["decision"] not in {"failed", "undetermined", "zero_coverage"}
                     for row in comparisons)
        valid_count = sum(row.get("geometry_valid") is True for row in subset)
        geometry_count = sum("geometry_valid" in row for row in subset)
        content.append(f"- `{layout}` 有效几何 {valid_count}/{geometry_count}，"
                       f"扰动定位与覆盖门槛通过 {passed}/{valid_count}，正向已测 {len(measured)} 格。")
        if measured:
            for name, worst, unit in (("D_hot_mm", max, "mm"), ("Cov010", min, "比例"), ("Q99_Cout", max, "V/m")):
                values = [row[name] for row in measured]
                content.append(f"  - {name} 名义 {reference[name]:.4f}，扰动最差 {worst(values):.4f}，"
                               f"中位 {np.median(values):.4f}（{unit}）。")
    report.write_text("\n".join([*content, "", "[扰动明细](position_offsets.csv) · [相对名义条件](paired.csv) · "
                                "[病例](cases.csv) · [命令](commands.ps1) · [图](figures/offsets.png)", ""]), encoding="utf-8")
    figure = out / "figures/offsets.png"
    outputs = [cases, offsets, paired, report, out / "commands.ps1", figure]
    write_csv(out / "artifacts.csv", [*inputs, *(artifact(study, "run", "output", path) for path in outputs if path.is_file())])
    write_json(out / "run.json", info)
    update_index(study, dict(experiment="18", run_id=info["run_id"], stage=info["stage"], run_status=info["status"],
                             scientific_decision="原生布局帽位稳定性" if good > len(references) else "等待扰动场",
                             ok_cases=good, failed_cases=failed, pending_cases=pending, report_path=report.relative_to(study).as_posix()))


def run(study: Path, out: Path, args: argparse.Namespace) -> None:
    """运行固定原生布局帽位扰动，可只生成几何用于独立验收。

    Parameters
    ----------
    study, out : Path
        研究根和当前输出目录。
    args : Namespace
        已验证的阶段、上游运行号和求解控制。
    """
    baseline = study / "results/03" / args.baseline_id
    native = baseline / "baseline_fields.csv"
    source_rows = {(row["subject"], row["target"], row["method"]): row for row in read_csv(native)} if native.is_file() else {}
    fingerprints = baseline / "head_model_hashes.csv"
    head_hashes = {row["subject"]: row for row in read_csv(fingerprints)} if fingerprints.is_file() else {}
    subjects = ("sub1",) if args.stage == "pilot" else ("sub1", "sub2")
    targets = ("stn295", "a4ul58") if args.stage == "pilot" else PILOT_TARGETS
    commands = out / "commands.ps1"
    invocation = " ".join("'" + part.replace("'", "''") + "'" for part in sys.argv[1:])
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp18 {invocation}\n")
    (out / "geometry").mkdir(exist_ok=True)
    info = dict(experiment="18", run_id=args.run_id, stage=args.stage, environment=sys.executable,
                upstream_runs=[f"03/{args.baseline_id}"], git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                data_id={}, started_at=timestamp(), finished_at=None, status="running",
                config=dict(offset_mm=SHIFT_MM, circle_clearance_mm=CLEARANCE_MM, arm="native_roc", geometry_only=args.geometry_only))
    inputs = [artifact(study, "run", "implementation", Path(__file__))]
    inputs.extend(artifact(study, "run", "upstream", path) for path in (native, fingerprints) if path.is_file())
    layouts = {}
    rows = []
    for subject in subjects:
        for target in targets:
            for method in METHODS:
                layout = f"{subject}_{target}_{method}_native"
                statuses = []
                for number, (role, tangent, sign) in enumerate((("NA", "NA", 0), *OFFSETS)):
                    statuses.append(dict(layout_id=layout, case_id=f"{layout}:offset{number:02}", subject=subject, target=target,
                                         method=method, arm="nominal" if number == 0 else "offset", seed="NA",
                                         condition=f"offset{number:02}", task_id="NA", status="pending", reason="等待 03 共同正向",
                                         wall_s="NA", electrode_role=role, tangent=tangent, sign=sign, requested_mm=SHIFT_MM if number else 0.0))
                layouts[(subject, target, method)] = statuses
                rows.extend(statuses)
    publish_state(study, out, info, rows, inputs)
    checked_heads: dict[str, tuple[Msh, str]] = {}
    configs: dict[str, dict[str, Any]] = {}
    references: dict[str, Path] = {}
    scenarios: dict[str, list[dict[str, Any]]] = {}
    for key, statuses in layouts.items():
        source = source_rows.get(key)
        if source is None or source["status"] != "ok":
            continue
        original, *offsets = statuses
        subject = source["subject"]
        directory = study / "simulations" / source["forward_task"]
        head = study / "head_models" / f"m2m_{subject}"
        started = monotonic()
        try:
            field = load_case(study, source)
            config = json.loads((directory / "params.json").read_text(encoding="utf-8"))
            cap = Path(find_montage_file(str(head), config["montage"]))
            if subject not in checked_heads:
                msh_path = head / f"{subject}.msh"
                checked_heads[subject] = (read_msh(msh_path), digest(msh_path))
            mesh, mesh_hash = checked_heads[subject]
            cap_hash = digest(cap)
            if mesh_hash != source["mesh_hash"] or cap_hash != head_hashes[subject]["cap_sha256"]:
                raise ValueError("个体头模或原始帽位指纹不匹配")
            currents = (config["electrode_A"][0]["current_mA"], config["electrode_B"][0]["current_mA"])
            if config["anisotropy_type"] != "scalar" or currents != (float(source["IA_mA"]), float(source["IB_mA"])):
                raise ValueError("正向模型或电流与 03 不匹配")
            measured = measure_field(field.geometry, field.target, field.e_a, field.e_b, field.valid)
            if measured["status"] != "ok":
                raise ValueError(f"原位共同场 {measured['status']}")
            original.update({key: measured[key] for key in ENDPOINTS}, actual_mm=0.0, geometry_valid=True, status="ok", reason="NA",
                            forward_task=directory.name, task_id=directory.name.removeprefix("ti_forward_"), field_path=source["field_path"],
                            field_sha256=digest(field.field_path), cap_file=cap.relative_to(study).as_posix(), cap_sha256=cap_hash,
                            wall_s=round(monotonic() - started, 3))
            references[original["layout_id"]] = field.geometry_path
            info["data_id"][subject] = field.geometry_path.parent.name
            inputs.extend(artifact(study, original["case_id"], role, path) for role, path in (("reference_geometry", field.geometry_path),
                          ("reference_field", field.field_path), ("nominal_cap", cap), ("nominal_params", directory / "params.json")))
            for row, params in plan_offsets(study, out, mesh, cap, cap_hash, source, config, offsets):
                configs[row["task_id"]] = params
                scenarios.setdefault(row["task_id"], []).append(row)
                inputs.append(artifact(study, row["case_id"], "offset_cap", study / row["cap_file"]))
        except SOLVER_ERRORS as error:
            for row in statuses:
                if row["status"] == "pending":
                    row.update(status="failed", reason=str(error))
    plot_offsets(out / "figures/offsets.png", rows)
    publish_state(study, out, info, rows, inputs)
    if not args.geometry_only:
        def publish() -> None:
            """保存当前正在执行的场景状态。"""
            publish_state(study, out, info, rows, inputs)

        def solve(row: dict[str, Any], command_file: Path) -> Path:
            """以场景帽位和原电流调用原生正向仿真。"""
            return run_task(study, "ti_forward", row["subject"], row["task_id"], configs[row["task_id"]], command_file, FORWARD_OUTPUTS)

        for case, future, command_file, started in solve_cases([group[0] for group in scenarios.values()], args.jobs, commands, solve, publish):
            for row in scenarios[case["task_id"]]:
                try:
                    directory = future.result()
                    reference = references[row["layout_id"]]
                    field, _ = align_common_forward(directory, reference, reference.parent / "fields")
                    evaluated = load_case(study, row | {"field_path": field.relative_to(study).as_posix()})
                    measured = measure_field(evaluated.geometry, evaluated.target, evaluated.e_a, evaluated.e_b, evaluated.valid)
                    if measured["status"] != "ok":
                        raise ValueError(f"扰动场指标 {measured['status']}")
                    row.update({key: measured[key] for key in ENDPOINTS}, status="ok", reason="NA", forward_task=directory.name,
                               field_path=field.relative_to(study).as_posix(), field_sha256=digest(field))
                    inputs.extend(artifact(study, row["case_id"], "forward_output", path) for pattern in FORWARD_OUTPUTS for path in directory.glob(pattern))
                    inputs.extend(artifact(study, row["case_id"], role, path) for role, path in (("aligned_field", field), ("command", command_file)))
                except SOLVER_ERRORS as error:
                    row.update(status="failed", reason=str(error))
                    if command_file.is_file():
                        inputs.append(artifact(study, row["case_id"], "command", command_file))
                row["wall_s"] = round(monotonic() - started, 3)
            publish()
    pending = any(row["status"] == "pending" for row in rows)
    failed = any(row["status"] == "failed" for row in rows)
    info.update(status="geometry_ready" if args.geometry_only and scenarios else "partial" if pending or failed else "complete", finished_at=timestamp())
    plot_offsets(out / "figures/offsets.png", rows)
    publish_state(study, out, info, rows, inputs)


def main() -> None:
    """解析帽位偏差实验入口参数，同次运行只允许一个协调进程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须至少为 1")
    study = args.study_root.resolve()
    out = study / "results/18" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".scheduler.lock"), timeout=0):
        run(study, out, args)


if __name__ == "__main__":
    main()
