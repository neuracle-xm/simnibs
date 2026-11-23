"""在固定具名布局上复算七组 scalar 电导率。"""

import argparse
from functools import partial
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

import numpy as np

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.demo.ti_depth.exp03 import FORWARD_OUTPUTS, align_common_forward, solve_cases
from neuracle.ti_depth_study.baseline import content_id, run_task
from neuracle.ti_depth_study.diagnostics import FieldCase, load_case
from neuracle.ti_depth_study.metrics import measure_field
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json
from neuracle.utils.constants import CONDUCTIVITY_TISSUE_NAMES, STANDARD_COND
from simnibs.mesh_tools.mesh_io import read_msh


TAGS = dict(zip(CONDUCTIVITY_TISSUE_NAMES, range(1, 11)))
GROUPS = {"WM": ("WM",), "GM": ("GM",), "skull": ("Bone", "CompactBone", "SpongyBone")}
SCENARIOS = (("nominal", 1.0), *((group, factor) for group in GROUPS for factor in (0.8, 1.2)))
ENDPOINTS = ("D_hot_mm", "Cov010", "Q99_Cout", "Vout010", "absolute_hit")
CASE_COLUMNS = ("case_id", "subject", "target", "method", "arm", "seed", "condition", "task_id", "status", "reason", "wall_s")
SWEEP_COLUMNS = ("layout_id", "config_hash", "tissue_group", "factor", "applied_tags", "mode", "forward_task", *ENDPOINTS)
RTOL = 1e-5
ATOL = 1e-7


def tissue_volumes(mesh_path: Path) -> dict[str, float]:
    """汇总实际网格中的十种组织体积，忽略三角表面标签。

    Parameters
    ----------
    mesh_path : Path
        与上游共同正向任务一致的 CHARM 体网格。

    Returns
    -------
    dict
        组织名到 mm³ 的映射，缺失组织体积为零。
    """
    mesh = read_msh(mesh_path, skip_data=True)
    volumes = mesh.elements_volumes_and_areas().value
    tetrahedra = mesh.elm.elm_type == 4
    return {name: float(volumes[tetrahedra & (mesh.elm.tag1 == tag)].sum()) for name, tag in TAGS.items()}


def scenario_config(nominal: dict[str, float], volumes: dict[str, float], group: str, factor: float) -> tuple[dict[str, float], tuple[str, ...]]:
    """仅缩放当前网格实际存在的指定组织，保留完整十项配置。

    Parameters
    ----------
    nominal, volumes : dict
        名义电导率（S/m）与相应组织体积（mm³）。
    group, factor : str and float
        预登记的组织组及乘数。

    Returns
    -------
    tuple
        完整电导率和发生改变的组织名。
    """
    affected = tuple(name for name in GROUPS.get(group, ()) if volumes[name] > 0)
    config = {name: nominal[name] * factor if name in affected else nominal[name] for name in TAGS}
    return config, affected


def solve_scenario(study: Path, case: FieldCase, row: dict[str, Any], command: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """通过原正向 CLI 复算参数组并测量同一参考网格上的终点。

    Parameters
    ----------
    study, case : Path and FieldCase
        研究根与已核验的原布局、电流、共同几何及名义场。
    row, command : dict and Path
        单一情形的正向配置和命令记录路径。

    Returns
    -------
    tuple
        指标和本次正向输入、执行清单及对齐场的证据。
    """
    task = run_task(study, "ti_forward", row["subject"], row["task_id"], row["config"], command, FORWARD_OUTPUTS)
    field, alignment = align_common_forward(task, case.geometry_path, case.geometry_path.parent / "fields")
    if alignment["missing_mm3"] > 0 or not case.valid.all():
        raise ValueError("原场或本次复算场缺少参考网格元素")
    with np.load(field) as values:
        if not values["valid_mask"].all():
            raise ValueError("电导率复算未覆盖参考网格")
        e_a, e_b = values["E_A"], values["E_B"]
        direct_check = {}
        if row["condition"] == "nominal":
            rms = {name: float(np.sqrt(np.mean((new - old) ** 2)))
                   for name, new, old in (("A", e_a, case.e_a), ("B", e_b, case.e_b), ("TI", values["max_TI"], case.field))}
            if not all(np.allclose(new, old, rtol=RTOL, atol=ATOL) for new, old in
                       ((e_a, case.e_a), (e_b, case.e_b), (values["max_TI"], case.field))):
                raise ValueError(f"名义电导率独立复算与 03 共同场不一致，RMS (V/m) {rms}")
            direct_check = {f"nominal_rms_{name}_vpm": rms[name] for name in rms}
        measured = measure_field(case.geometry, case.target, e_a, e_b, values["valid_mask"])
        if measured["status"] != "ok":
            raise ValueError(f"复算场指标未通过验收，{measured['status']}")
    source = [(role, path) for role, path in (("forward_config", task / "params.json"),
              ("forward_execution", task / "study_execution.json"), ("aligned_field", field))]
    return dict(measured, **direct_check, forward_task=task.name,
                field_path=field.relative_to(study).as_posix()), [artifact(study, row["case_id"], role, path) for role, path in source]


def publish(study: Path, out: Path, run: dict[str, Any], cases: list[dict[str, Any]], conductivity: list[dict[str, Any]], artifacts: list[dict[str, str]]) -> None:
    """写入真实复算状态、组织输入及每一布局内对名义值的差。

    Parameters
    ----------
    study, out : Path
        研究根与 19 当次运行目录。
    run, cases, conductivity, artifacts : dict and lists of dict
        运行清单、情形状态、组织物性明细和已核验证据。
    """
    write_csv(out / "cases.csv", [{key: row[key] for key in CASE_COLUMNS} for row in cases], CASE_COLUMNS)
    write_csv(out / "conductivities.csv", conductivity,
              ("subject", "mesh_sha256", "config_hash", "tissue_group", "factor", "tissue", "tag", "volume_mm3", "conductivity_s_m", "changed"))
    sweep = [{key: value for key, value in row.items() if key != "config"} for row in cases]
    columns = tuple(dict.fromkeys((*CASE_COLUMNS, *SWEEP_COLUMNS, *(key for row in sweep for key in row))))
    write_csv(out / "conductivity_sweep.csv", sweep, columns)
    nominal = {row["layout_id"]: row for row in sweep if row["condition"] == "nominal" and row["status"] == "ok"}
    paired = []
    for row in sweep:
        reference = nominal.get(row["layout_id"])
        if row["status"] != "ok" or row["condition"] == "nominal" or reference is None:
            continue
        paired.append(dict(case_id=row["case_id"], reference_case_id=reference["case_id"], layout_id=row["layout_id"],
                           tissue_group=row["tissue_group"], factor=row["factor"],
                           **{f"delta_{key}": row[key] - reference[key] for key in ENDPOINTS[:-1]},
                           absolute_hit_changed=row["absolute_hit"] != reference["absolute_hit"]))
    write_csv(out / "paired.csv", paired, ("case_id", "reference_case_id", "layout_id", "tissue_group", "factor",
              *(f"delta_{key}" for key in ENDPOINTS[:-1]), "absolute_hit_changed"))
    ok = sum(row["status"] == "ok" for row in cases)
    failed = sum(row["status"] == "failed" for row in cases)
    pending = len(cases) - ok - failed
    report = out / "report.md"
    report.write_text(f"# 19 固定布局电导率敏感性\n\n{run['stage']} 阶段，原布局已完成 {ok}/{len(cases)} 格，失败 {failed} 格，待完成 {pending} 格。\n\n"
                      f"名义场复算通过 {len(nominal)} 个布局。扰动相对该布局名义值的绝对命中翻转 {sum(row['absolute_hit_changed'] for row in paired)}/{len(paired)} 格。\n\n"
                      "物性范围为 WM、GM 及头模中实际存在的颅骨组织各 ±20%。新旧布局配对等待锁定的新解。\n\n"
                      "[七种情形](conductivity_sweep.csv) · [组织体积与取值](conductivities.csv) · [相对名义差值](paired.csv)\n", encoding="utf-8")
    panels = ("cases.csv", "conductivities.csv", "conductivity_sweep.csv", "paired.csv", "report.md", "commands.ps1")
    write_csv(out / "artifacts.csv", [*artifacts, *(artifact(study, "run", "output", out / name) for name in panels)])
    write_json(out / "run.json", run)
    update_index(study, dict(experiment="19", run_id=run["run_id"], stage=run["stage"], run_status=run["status"],
                             scientific_decision="原布局物性敏感性，新旧配对待锁定" if ok else "等待启动条件",
                             ok_cases=ok, failed_cases=failed, pending_cases=pending, report_path=report.relative_to(study).as_posix()))


def run(study: Path, run_id: str, baseline_id: str, stage: str, jobs: int) -> None:
    """先复算名义场并验证，再并行安排通过核验的布局扰动。

    Parameters
    ----------
    study, run_id, baseline_id, stage : Path and str
        研究目录、19 与 03 运行号和研究阶段。
    jobs : int
        可同时求解的正向任务数。
    """
    out = study / "results/19" / run_id
    out.mkdir(parents=True, exist_ok=True)
    commands = out / "commands.ps1"
    with commands.open("a", encoding="utf-8") as script:
        script.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\n"
                     f"# python -m neuracle.demo.ti_depth.exp19 --study-root '{study}' --run-id {run_id} --baseline-id {baseline_id} --stage {stage} --jobs {jobs}\n")
    input_path = study / "results/03" / baseline_id / "baseline_fields.csv"
    raw = read_csv(input_path) if input_path.is_file() else []
    rows = {(row["subject"], row["target"], row["method"]): row for row in raw if row["status"] == "ok"}
    artifacts = [artifact(study, "run", "implementation", Path(__file__))]
    if input_path.is_file():
        artifacts.append(artifact(study, "run", "upstream_baselines", input_path))
    info = dict(experiment="19", run_id=run_id, stage=stage, environment=sys.executable, upstream_runs=[f"03/{baseline_id}"],
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                data_id={}, config=dict(scenarios=SCENARIOS, groups=GROUPS, nominal_rtol=RTOL, nominal_atol_vpm=ATOL, jobs=jobs),
                started_at=timestamp(), finished_at=None, status="running")
    subjects, targets = (("sub1",), ("stn295",)) if stage == "pilot" else (("sub1", "sub2"), PILOT_TARGETS)
    cases: list[dict[str, Any]] = []
    conductivity: list[dict[str, Any]] = []
    registered = set()
    for subject, target, method in product(subjects, targets, ("leadfield_based", "leadfield_free")):
        native = rows.get((subject, target, method))
        layout_id = f"{subject}_{target}_{method}"
        layouts = [dict(layout_id=layout_id, case_id=f"{layout_id}_{group}_{factor:g}",
                        subject=subject, target=target, method=method, arm="original", seed="NA", mode="fixed_layout",
                        condition="nominal" if group == "nominal" else f"{group}_{factor:g}",
                        tissue_group=group, factor=factor, task_id="NA", status="pending",
                        reason="等待 03 共同正向" if native is None else "等待名义场核验", wall_s="NA")
                   for group, factor in SCENARIOS]
        cases.extend(layouts)
        if native is None:
            continue
        try:
            mesh_path = study / "head_models" / f"m2m_{subject}" / f"{subject}.msh"
            if subject not in info["data_id"]:
                info["data_id"][subject] = dict(mesh_sha256=digest(mesh_path), volumes_mm3=tissue_volumes(mesh_path))
                artifacts.append(artifact(study, subject, "mesh", mesh_path))
            data = info["data_id"][subject]
            if data["mesh_sha256"] != native["mesh_hash"]:
                raise ValueError("当前头模散列与 03 共同正向不符")
            params = study / "simulations" / native["forward_task"] / "params.json"
            config = json.loads(params.read_text(encoding="utf-8"))
            if config["conductivity_config"] != STANDARD_COND or config["anisotropy_type"] != "scalar":
                raise ValueError("03 具名布局未使用标准 scalar 物性")
            artifacts.append(artifact(study, layout_id, "baseline_forward_input", params))
            for row in layouts:
                group, factor = row["tissue_group"], row["factor"]
                values, affected = scenario_config(config["conductivity_config"], data["volumes_mm3"], group, factor)
                updated = config | {"conductivity_config": values}
                physical_id = content_id({"mesh": data["mesh_sha256"], "config": updated})
                config_hash = content_id(values)
                row.update(config=updated, config_hash=config_hash, applied_tags=";".join(str(TAGS[name]) for name in affected) or "NA",
                           task_id=f"19_{run_id}_{subject}_{physical_id[:20]}")
                key = subject, group, factor, config_hash
                if key not in registered:
                    registered.add(key)
                    conductivity.extend(dict(subject=subject, mesh_sha256=data["mesh_sha256"], config_hash=config_hash, tissue_group=group,
                                             factor=factor, tissue=name, tag=TAGS[name], volume_mm3=data["volumes_mm3"][name],
                                             conductivity_s_m=value, changed=name in affected) for name, value in values.items())
                if group != "nominal" and not affected:
                    row.update(status="failed", reason="网格中缺少该组织组")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            for row in layouts:
                row.update(status="failed", reason=str(error))
    write_state = partial(publish, study, out, info, cases, conductivity, artifacts)
    write_state()
    for layout_id in dict.fromkeys(row["layout_id"] for row in cases):
        selected = [row for row in cases if row["layout_id"] == layout_id and row["status"] == "pending" and "config" in row]
        if not selected:
            continue
        nominal = selected[0]
        try:
            case = load_case(study, rows[(nominal["subject"], nominal["target"], nominal["method"])])
            artifacts.extend(artifact(study, layout_id, role, path) for role, path in
                             (("baseline_field", case.field_path), ("baseline_geometry", case.geometry_path)))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            for row in selected:
                row.update(status="failed", reason=str(error))
            write_state()
            continue
        worker = partial(solve_scenario, study, case)
        for group in ([nominal], selected[1:]):
            if group and group[0] is not nominal and nominal["status"] != "ok":
                break
            for row, future, command, start in solve_cases(group, jobs, commands, worker, write_state):
                try:
                    measured, files = future.result()
                    row.update(measured, status="ok", reason="NA")
                    artifacts.extend(files)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    row.update(status="failed", reason=str(error))
                row["wall_s"] = round(monotonic() - start, 3)
                artifacts.append(artifact(study, row["case_id"], "command", command))
                write_state()
    good = sum(row["status"] == "ok" for row in cases)
    info.update(status="complete" if good == len(cases) else "partial" if good else "blocked", finished_at=timestamp())
    write_state()


def main() -> None:
    """从仓库根执行原布局的物性先导或开发复算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 至少为 1")
    run(args.study_root.resolve(), args.run_id, args.baseline_id, args.stage, args.jobs)


if __name__ == "__main__":
    main()
