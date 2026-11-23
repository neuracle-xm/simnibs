"""对 23 当前域内的有限见证布局独立作共同正向复算。"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

from filelock import FileLock
import numpy as np

from neuracle.demo.ti_depth.exp03 import CASE_COLUMNS, FORWARD_OUTPUTS, SOLVER_ERRORS, align_common_forward, solve_cases
from neuracle.demo.ti_depth.exp23 import MODEL
from neuracle.ti_depth_study.baseline import FIELD_MODEL, content_id, forward_config, run_task
from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.metrics import measure_field
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, write_csv, write_json
from simnibs.utils.csv_reader import _get_eeg_positions


ENDPOINTS = ("Cov010", "Cov020", "D_hot_mm", "Q50_T", "Q99_Cout", "Vout010", "absolute_hit")
PLAN_COLUMNS = (*CASE_COLUMNS, "domain_hash", "witness_layout", "candidate_source", "source_model_id", "model_id", "mesh_hash",
                "cap_hash", "geometry_sha256", "physical_id", "IA_mA", "IB_mA")
RESULT_COLUMNS = (*PLAN_COLUMNS, "forward_task", "field_path", "field_sha256", *ENDPOINTS, *(f"HDF5_{key}" for key in ENDPOINTS))


@dataclass(frozen=True)
class Replay:
    """一个已登记的有限布局及独立共同正向所需的全部物理输入。"""

    row: dict[str, Any]
    config: dict[str, Any]
    geometry: Path


def plan(study: Path, source: Path) -> tuple[list[Replay], list[dict[str, str]], str]:
    """只为当前建界域的活跃见证登记直接 FEM 任务。

    Parameters
    ----------
    study, source : Path
        本地研究根和 23 的上界运行目录。

    Returns
    -------
    runs, inputs, stage : tuple
        具名物理计划、来源散列及上游阶段。
    """
    run_file = source / "run.json"
    manifest = source / "artifacts.csv"
    bounds_file = source / "coverage_bounds.csv"
    witness_file = source / "witnesses.csv"
    recorded = read_csv(manifest)
    hashes = {row["path"]: row["sha256"] for row in recorded}
    for path in (bounds_file, witness_file):
        if digest(path) != hashes[path.relative_to(study).as_posix()]:
            raise ValueError(f"23 当前运行的证据散列不符 {path.name}")
    prior = json.loads(run_file.read_text(encoding="utf-8"))
    baseline = study / "results" / next(run for run in prior["upstream_runs"] if run.startswith("03/"))
    geometry_rows = {row["case_id"]: row for row in recorded if row["role"] == "geometry"}
    head_model_file = baseline / "head_model_hashes.csv"
    heads = {row["subject"]: row for row in read_csv(head_model_file)}
    active = {row["domain_hash"]: row for row in read_csv(bounds_file) if row["model_id"] == MODEL and row["exhaustive"] == "True"}
    inputs = [artifact(study, "run", "upstream", path) for path in (run_file, manifest, bounds_file, witness_file, head_model_file)]
    scenarios = []
    seen = set()
    for witness in read_csv(witness_file):
        domain = witness["domain_hash"]
        if domain not in active or witness.get("field_source") != MODEL or witness.get("whole_brain_rows", "NA") == "NA":
            continue
        layout = witness["layout"]
        key = domain, layout
        if key in seen:
            continue
        seen.add(key)
        row = active[domain]
        subject, target = row["subject"], row["target"]
        labels = layout.split(";")
        config = forward_config(labels, ([1.0, -1.0], [1.0, -1.0]))
        cap = study / "head_models" / f"m2m_{subject}" / "eeg_positions" / f"{config['montage']}.csv"
        if not set(labels) <= _get_eeg_positions(str(cap)).keys():
            raise ValueError(f"见证布局的标签不在当前帽位 {layout}")
        geom = geometry_rows[row["case_id"]]
        geometry_path = study / geom["path"]
        mesh_path = study / "head_models" / f"m2m_{subject}" / f"{subject}.msh"
        source_mesh = geometry_path.with_name("geometry_inputs.json")
        geometry_inputs = json.loads(source_mesh.read_text(encoding="utf-8"))
        mesh_hash, cap_hash = heads[subject]["mesh_sha256"], heads[subject]["cap_sha256"]
        if digest(geometry_path) != geom["sha256"] or digest(mesh_path) != mesh_hash or digest(cap) != cap_hash or geometry_inputs["mesh_hash"] != mesh_hash:
            raise ValueError(f"23 的见证参考几何与当前头模或帽位不符 {subject}")
        physical_id = content_id(dict(subject=subject, mesh=mesh_hash, cap=cap_hash, model=FIELD_MODEL, params=config))
        case_id = f"{subject}_{target}_{physical_id[:20]}"
        task_id = f"23_{source.name}_{subject}_witness_{physical_id[:20]}"
        data = dict(case_id=case_id, subject=subject, target=target, method="leadfield_based", arm="common_witness_replay",
                    seed="NA", condition="1+1mA", task_id=task_id, status="pending", reason="等待共同正向求解", wall_s="NA",
                    domain_hash=domain, witness_layout=layout, candidate_source=witness["candidate_source"], source_model_id=MODEL,
                    model_id=FIELD_MODEL, mesh_hash=mesh_hash, cap_hash=cap_hash, geometry_sha256=geom["sha256"],
                    physical_id=physical_id, IA_mA=1.0, IB_mA=1.0)
        data.update({f"HDF5_{key}": witness.get(key, "NA") for key in ENDPOINTS})
        scenarios.append(Replay(data, config, geometry_path))
        inputs.extend(artifact(study, case_id, role, path) for role, path in (("reference_geometry", geometry_path), ("cap", cap), ("mesh_source", source_mesh)))
    if not scenarios:
        raise ValueError("23 当前域无已验收的全脑具名见证")
    return scenarios, inputs, prior["stage"]


def publish(study: Path, out: Path, info: dict[str, Any], replays: list[Replay], inputs: list[dict[str, str]]) -> None:
    """持续写入单独模型的正向状态、可核对的指标和产物散列。

    Parameters
    ----------
    study, out : Path
        研究根和共同正向独立结果目录。
    info, replays, inputs : dict and lists
        执行清单、逐病例状态和已验收的文件产物。
    """
    cases, result, report = (out / name for name in ("cases.csv", "witness_forward.csv", "report.md"))
    rows = [item.row for item in replays]
    write_csv(cases, [{key: row[key] for key in CASE_COLUMNS} for row in rows], CASE_COLUMNS)
    write_csv(result, rows, RESULT_COLUMNS)
    good = sum(row["status"] == "ok" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    pending = len(rows) - good - failed
    lines = ["# 23 有限见证布局的共同正向复算", "", f"{info['stage']} 阶段，状态 `{info['status']}`。已验收 {good}/{len(rows)} 格，失败 {failed} 格，待完成 {pending} 格。", "",
             "每份原布局保持 A+、A−、B+、B− 的帽位和两路各 1 mA。共同 scalar FEM 指标独立登记，HDF5 指标按布局引用。", ""]
    for row in rows:
        if row["status"] == "ok":
            lines.append(f"- {row['subject']}/{row['target']} `{row['witness_layout']}`，共同正向距靶 {row['D_hot_mm']:.4f} mm，"
                         f"0.10 V/m 覆盖 {row['Cov010']:.4f}，0.20 V/m 覆盖 {row['Cov020']:.4f}。")
    lines.extend(("", "[逐布局结果](witness_forward.csv) · [病例](cases.csv) · [命令](commands.ps1) · [同模型上界](../report.md)", ""))
    report.write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "run.json", info)
    outputs = (cases, result, report, out / "commands.ps1", out / "plan.csv")
    write_csv(out / "artifacts.csv", [*inputs, *(artifact(study, "run", "output", path) for path in outputs)])


def run(study: Path, source: Path, out: Path, args: argparse.Namespace) -> None:
    """调度互异共同正向任务，使用参考网格分别测量每个靶。

    Parameters
    ----------
    study, source, out : Path
        研究根、23 运行目录和共同正向结果目录。
    args : Namespace
        计划预核验及并行求解控制。
    """
    with FileLock(str(source / ".scheduler.lock")):
        replays, inputs, stage = plan(study, source)
    write_csv(out / "plan.csv", [{key: replay.row[key] for key in PLAN_COLUMNS} for replay in replays], PLAN_COLUMNS)
    if args.plan_only:
        print(f"23 已登记 {len(replays)} 个布局和 {len({replay.row['task_id'] for replay in replays})} 个共同正向任务。{out / 'plan.csv'}")
        return
    inputs.append(artifact(study, "run", "implementation", Path(__file__)))
    commands = out / "commands.ps1"
    invocation = " ".join("'" + part.replace("'", "''") + "'" for part in sys.argv[1:])
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp23_forward {invocation}\n")
    info = dict(experiment="23", run_id=source.name, stage=stage, upstream_runs=[f"23/{source.name}"], status="running",
                data_id=sorted({replay.row["physical_id"] for replay in replays}), started_at=timestamp(), finished_at=None,
                environment=sys.executable, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                config=dict(arm="common_witness_replay", currents_mA=[1.0, 1.0], model=FIELD_MODEL, jobs=args.jobs))
    groups: dict[str, list[Replay]] = {}
    for replay in replays:
        groups.setdefault(replay.row["task_id"], []).append(replay)
    def update() -> None:
        """发布主进程当前共同正向状态。"""
        publish(study, out, info, replays, inputs)
    def solve(row: dict[str, Any], script: Path) -> Path:
        """调用同条件直接 FEM，指纹相符的成功产物按任务复用。"""
        task_id = row["task_id"]
        return run_task(study, "ti_forward", row["subject"], task_id, groups[task_id][0].config, script, FORWARD_OUTPUTS)
    update()
    for scheduled, future, script, started in solve_cases([group[0].row for group in groups.values()], args.jobs, commands, solve, update):
        for replay in groups[scheduled["task_id"]]:
            row = replay.row
            try:
                directory = future.result()
                field, _ = align_common_forward(directory, replay.geometry, replay.geometry.parent / "fields")
                geometry, _, targets = load_geometry(replay.geometry)
                with np.load(field) as data:
                    measured = measure_field(geometry, targets[row["target"]], data["E_A"], data["E_B"], data["valid_mask"])
                if measured["status"] != "ok":
                    raise ValueError(f"共同正向场 {measured['status']}")
                row.update({key: measured[key] for key in ENDPOINTS}, status="ok", reason="NA", forward_task=directory.name,
                           field_path=field.relative_to(study).as_posix(), field_sha256=digest(field))
                inputs.extend(artifact(study, row["case_id"], "common_forward", path) for pattern in FORWARD_OUTPUTS for path in directory.glob(pattern))
                inputs.extend(artifact(study, row["case_id"], role, path) for role, path in (("aligned_field", field), ("parameters", directory / "params.json")))
            except SOLVER_ERRORS as error:
                row.update(status="failed", reason=str(error))
            row["wall_s"] = round(monotonic() - started, 3)
            inputs.append(artifact(study, row["case_id"], "task_command", script))
        update()
    info.update(status="complete" if all(item.row["status"] == "ok" for item in replays) else "partial", finished_at=timestamp())
    update()
    if info["status"] != "complete":
        raise SystemExit(1)


def main() -> None:
    """解析共同正向复算入口，同一复算运行由一处发布。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须至少为 1")
    if not args.run_id or args.run_id in {".", ".."} or any(sep in args.run_id for sep in ("/", "\\", ":")):
        parser.error("运行号必须为单个目录名")
    study = args.study_root.resolve()
    source = study / "results/23" / args.run_id
    out = source / "common_forward"
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".scheduler.lock"), timeout=0):
        run(study, source, out, args)


if __name__ == "__main__":
    main()
