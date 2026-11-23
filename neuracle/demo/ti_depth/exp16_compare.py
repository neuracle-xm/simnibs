"""执行 16 的 sub1/STN 同种群、等有效预算互供先导对照。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

from filelock import FileLock
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy import __version__ as SCIPY_VERSION

from neuracle.demo.ti_depth.exp15 import SEARCH_ERRORS, TRACES, forward_winner, input_id, registered_case
from neuracle.demo.ti_depth.exp16 import FixedDonor, leadfield_names, read_fixed_donor
from neuracle.ti_depth_study import matched_initialization
from neuracle.ti_depth_study.baseline import content_id
from neuracle.ti_depth_study.diagnostics import FieldCase, load_case
from neuracle.ti_depth_study.matched_initialization import SEED_ATOL, MatchedTrace, guarded_ga, search
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, write_csv, write_json
from neuracle.ti_depth_study.restart_budget import BASED_SETTINGS, CURRENT_MA, INITIAL_REQUESTS, METHODS, THRESHOLD, FloatArray, prepare_case


SUBJECT = "sub1"
TARGET = "stn295"
SEED = 20220824
RESTART_SEED = 20220825
ARMS = ("random", "own", "cross")
CASE_COLUMNS = ("case_id", "subject", "target", "receiver", "arm", "status", "reason", "donor_case_id", "valid_encoding")
TABLE_COLUMNS = (*CASE_COLUMNS, "B_planned", "n_valid_unique", "n_requests", "cache_hits", "invalid", "max_roundtrip_mm",
                 "init_score", "injected_score", "final_score", "donor_s", "receiver_s", "total_s", "field_s", "labels",
                 "D_hot_mm", "Cov010", "field_path", "forward_task")


@dataclass(frozen=True)
class Inputs:
    """两份真实同条件供体和登记预核验结果。"""

    donors: dict[str, FixedDonor]
    encodings: dict[str, dict[str, str]]
    budget_id: str
    signature: str
    evidence: list[dict[str, str]]


def read_inputs(study: Path, matched: Path, budget_id: str) -> Inputs:
    """核对 16 编码收据与当前 15 供体，避免跨头模或过期结果复用。"""
    report = matched / "run.json"
    receipt = json.loads(report.read_text(encoding="utf-8"))
    if f"15/{budget_id}" not in receipt["upstream_runs"] or receipt["verified_donors"] != 2:
        raise ValueError("等待 16 使用指定 15 运行完成两个匹配供体的编码核验")
    checks = {row["path"]: row["sha256"] for row in read_csv(matched / "artifacts.csv")}
    table = matched / "warm_start.csv"
    natural = study / "results/15" / budget_id / "search/natural"
    head = study / "head_models" / f"m2m_{SUBJECT}"
    mesh, cap = head / f"{SUBJECT}.msh", head / "eeg_positions/EEG10-10_NSN.csv"
    target = next((row for row in read_csv(study / "results/02/run-001/targets.csv") if row["target"] == TARGET), None)
    if target is None:
        raise ValueError(f"02 尚未登记靶区 {TARGET}")
    mesh_hash, cap_hash = digest(mesh), digest(cap)
    signature = input_id(dict(head=head, mesh=mesh, cap=cap), dict(mesh_hash=mesh_hash), target)
    names, cache, references = leadfield_names(natural / METHODS[0], mesh_hash, cap_hash)
    if receipt["data_id"] != dict(mesh=mesh_hash, cap=cap_hash, leadfield=cache):
        raise ValueError("当前头模、帽位或导联场与 16 编码收据不一致")
    donors = {method: read_fixed_donor(natural / method, method, signature, names) for method in METHODS}
    donor_files = [path for donor in donors.values() for path in donor.evidence]
    donor_files.append(natural / METHODS[1] / "electrode_mapping.json")
    if any(checks.get(path.relative_to(study).as_posix()) != digest(path) for path in [table, *donor_files]):
        raise ValueError("16 编码收据与当前供体产物的散列不一致")
    encodings = {row["receiver"]: row for row in read_csv(table) if row["condition_matched"] == "True"}
    if set(encodings) != set(METHODS):
        raise ValueError("16 缺少匹配供体的双向编码状态")
    for receiver, donor in (METHODS, METHODS[::-1]):
        if encodings[receiver]["donor_case_id"] != donor_id(budget_id, donor):
            raise ValueError("16 供体编号与接收路线不一致")
    evidence = [artifact(study, "run", "matched_input", path) for path in (report, table, *donor_files, *references)]
    return Inputs(donors, encodings, budget_id, signature, evidence)


def donor_id(budget_id: str, method: str) -> str:
    """从 15 运行号及路线生成唯一的本先导供体标识。"""
    return f"15/{budget_id}/natural/{SUBJECT}/{TARGET}/{method}"


def declared_arms(receiver: str, inputs: Inputs | None) -> list[dict[str, Any]]:
    """冻结每个接收方的共同预算、首个个体和编码失败状态。"""
    peer = next(method for method in METHODS if method != receiver)
    rows = []
    for arm, donor_method in zip(ARMS, (None, receiver, peer)):
        row = dict(case_id=f"{SUBJECT}_{TARGET}_{receiver}_{arm}", subject=SUBJECT, target=TARGET, receiver=receiver,
                   arm=arm, status="pending", reason="等待两个同条件供体及 03 共评场", donor_case_id="NA", valid_encoding="NA",
                   donor_s=0.0, B_planned="NA", seed=SEED, candidate=None)
        if inputs is not None:
            row["B_planned"] = inputs.donors[receiver].unique // 3 * 3
            if donor_method is not None:
                donor = inputs.donors[donor_method]
                row.update(donor_case_id=donor_id(inputs.budget_id, donor_method), donor_s=donor.wall_s,
                           donor_n_valid_unique=donor.unique, donor_n_requests=donor.requests, valid_encoding=True)
                if arm == "own":
                    row["candidate"] = donor.parameters.tolist()
                else:
                    encoded = inputs.encodings[receiver]
                    accepted = encoded["valid_encoding"] == "True"
                    row.update(valid_encoding=accepted, max_roundtrip_mm=encoded["max_roundtrip_mm"])
                    if not accepted:
                        row.update(status="failed", reason=encoded["reason"])
                    else:
                        params = np.asarray(json.loads(encoded["encoded"]), dtype=np.float64).ravel().tolist()
                        if receiver == METHODS[0]:
                            ticks = round(CURRENT_MA / BASED_SETTINGS.current_step_ma)
                            params.extend((ticks, ticks))
                        row["candidate"] = params
        rows.append(row)
    return rows


def run_search(
    prepared: dict[str, Any], output: Path, row: dict[str, Any], signature: str,
    expected_rest: FloatArray | None = None, duration_s: float | None = None,
) -> tuple[dict[str, Any], Path]:
    """重放或执行一个起点，记录实际首个体、整批初始种群和预算。"""
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "evaluations.csv"
    result_path = output / "result.json"
    candidate = row["candidate"]
    method, budget = row["receiver"], row["B_planned"]
    configuration = dict(input=signature, method=method, seed=row["seed"], budget=budget, candidate=candidate,
                         expected_rest=content_id({"rows": expected_rest.tolist()}) if expected_rest is not None else None,
                         duration_s=duration_s)
    if result_path.is_file() and trace_path.is_file():
        recorded = json.loads(result_path.read_text(encoding="utf-8"))
        if (recorded["configuration"] == configuration and recorded["trace_sha256"] == digest(trace_path)
                and recorded["status"] in ("ok", "cost_unmatched")):
            return recorded, trace_path
    trace = MatchedTrace(budget, INITIAL_REQUESTS[method], expected_first=np.asarray(candidate) if candidate is not None else None,
                         expected_rest=expected_rest, first_atol=SEED_ATOL if method == METHODS[1] else 0.0, duration_s=duration_s)
    initial = trace.minimum_requests
    start = perf_counter()
    print(f"16 {row['case_id']} B={budget} seed={row['seed']}", flush=True)
    status, reason, labels = "ok", "NA", []
    try:
        labels = search(prepared, trace, method, row["seed"], output)
        if len(trace.rows) < initial or len(labels) != 4 or len(set(labels)) != 4:
            raise ValueError("搜索未完成原生初始化或无四个互异胜者帽位")
        if duration_s is None and trace.new_evals != budget:
            status, reason = "cost_unmatched", f"有效评价 {trace.new_evals} / {budget}"
    except SEARCH_ERRORS as error:
        status, reason = "failed", str(error)
    initial_scores = [item["objective"] for item in trace.rows[:initial] if item["status"] != "invalid"]
    write_csv(trace_path, trace.rows, TRACES)
    result = dict(configuration=configuration, status=status, reason=reason, labels=";".join(labels),
                  trace_sha256=digest(trace_path), n_valid_unique=trace.new_evals, n_requests=len(trace.rows),
                  cache_hits=trace.cache_hits, invalid=trace.invalid, receiver_s=perf_counter() - start, duration_hit=trace.duration_hit,
                  initial_requests=min(initial, len(trace.rows)), init_score=min(initial_scores) if initial_scores else "NA",
                  injected_score=trace.rows[0]["objective"] if candidate is not None and trace.rows and trace.rows[0]["status"] != "invalid" else "NA",
                  final_score=trace.best_score if trace.best_parameters is not None else "NA")
    write_json(result_path, result)
    return result, trace_path


def initial_rows(path: Path, count: int) -> FloatArray:
    """从已核验的逐请求记录读取随机臂的完整初始种群。"""
    records = read_csv(path)[:count]
    if len(records) != count:
        raise ValueError("随机臂初始化未完成")
    return np.asarray([json.loads(row["parameters"]) for row in records], dtype=np.float64)


def simulate_winner(
    study: Path, out: Path, common: FieldCase, result: dict[str, Any], witness_plan: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """按冻结搜索分选择完成后，使用已有共同正向入口验收胜者。"""
    start = perf_counter()
    try:
        field = forward_winner(study, out, common, result, witness_plan)
        artifacts = [artifact(study, result["case_id"], "common_forward", study / field["field_path"])]
    except SEARCH_ERRORS as error:
        field, artifacts = dict(status="failed", reason=str(error)), []
    return field | dict(field_s=perf_counter() - start), artifacts


def run_zero_cost(
    study: Path, out: Path, common: FieldCase, target: dict[str, str], prepared: dict[str, Any], random: dict[str, Any],
    warm: dict[str, Any], fields: dict[str, dict[str, Any]], signature: str, witness_plan: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, str]]]:
    """给独立重启分配余下的实测总时长，在同等 ROC0 内比较获胜者。"""
    method = warm["receiver"]
    remaining = warm["total_s"] - random["receiver_s"]
    identity = {key: warm[key] for key in ("subject", "target", "receiver", "arm")}
    row = identity | dict(case_id=f"{warm['case_id']}_restart_control", seed=RESTART_SEED, candidate=None,
                          B_planned=(1000 + 1) * INITIAL_REQUESTS[method], planned_total_s=warm["total_s"],
                          duration_s=remaining, status="cost_unmatched")
    if remaining <= 0:
        return row | dict(reason="基准随机搜索已用完整个从零预算", total_s=random["receiver_s"],
                          delta_s=-remaining), None, []
    restarted, trace = run_search(prepared, out / row["case_id"], row, signature, duration_s=remaining)
    evidence = [artifact(study, row["case_id"], "cold_cost_restart", path) for path in (trace, trace.with_name("result.json"))]
    total = random["receiver_s"] + restarted["receiver_s"]
    row.update({key: value for key, value in restarted.items() if key not in ("configuration", "trace_sha256")})
    row.update(total_s=total, delta_s=total - warm["total_s"], source="random")
    if restarted["status"] != "ok" or not restarted["duration_hit"]:
        return row | dict(status="cost_unmatched", reason=restarted["reason"] if restarted["status"] != "ok" else "未达到预登记的时间边界"), None, evidence
    field = fields["random"]
    if restarted["final_score"] < random["final_score"]:
        field, paths = simulate_winner(study, out, common, row, witness_plan)
        row["source"] = "independent_restart"
        evidence.extend(paths)
    row.update({key: field.get(key, "NA") for key in ("D_hot_mm", "Cov010", "field_path", "forward_task")})
    if field["status"] != "ok":
        row.update(status="failed", reason=field.get("reason", field["status"]))
    pair = compare_endpoints(fields[warm["arm"]], field, target["group"])
    return row, identity | dict(case_id=warm["case_id"], comparison="from_zero_time_boundary", **pair), evidence


def run_receiver(
    study: Path, out: Path, baseline_id: str, budget_id: str, arms: list[dict[str, Any]], signature: str, zero_cost: bool,
) -> dict[str, Any]:
    """为同一接收路线运行完随机、同路及互供的独立初始化。"""
    method = arms[0]["receiver"]
    directory = out / "search" / method
    directory.mkdir(parents=True, exist_ok=True)
    evidence, fields, comparisons, controls = [], {}, [], []
    if arms[0]["B_planned"] <= 0:
        for row in arms:
            if row["status"] == "pending":
                row.update(reason="有效自然预算不足以保证完整初始化")
        return dict(arms=arms, comparisons=comparisons, controls=controls, evidence=evidence)
    try:
        native, target = registered_case(study, baseline_id, SUBJECT, TARGET, method)
        common = load_case(study, native)
        prepared = prepare_case(study, SUBJECT, target, method, directory)
        if input_id(prepared, native, target) != signature:
            raise ValueError("接收者与 15 供体的输入身份不一致")
    except SEARCH_ERRORS as error:
        for row in arms:
            if row["status"] == "pending":
                row.update(reason=str(error))
        return dict(arms=arms, comparisons=comparisons, controls=controls, evidence=evidence)
    search_id = content_id(dict(study_case=signature, initialization=digest(Path(matched_initialization.__file__)),
                                implementation=digest(Path(__file__)), scipy=SCIPY_VERSION))
    evidence.append(artifact(study, method, "geometry", common.geometry_path))
    population = None
    witness_plan = study / "results/23" / budget_id / "witness_forward_plan.csv"
    for row in arms:
        arm = row["arm"]
        if row["status"] == "failed":
            continue
        if arm != "random" and population is None:
            row.update(reason="等待同种子随机臂的完整初始种群")
            continue
        result, trace = run_search(prepared, directory / arm, row, search_id, None if population is None else population[1:])
        evidence.extend(artifact(study, row["case_id"], "search", path) for path in (trace, trace.with_name("result.json")))
        row.update({key: value for key, value in result.items() if key not in ("configuration", "trace_sha256")})
        row["total_s"] = row["receiver_s"] + row["donor_s"]
        if arm == "random" and row["initial_requests"] == INITIAL_REQUESTS[method]:
            population = initial_rows(trace, INITIAL_REQUESTS[method])
        if row["status"] not in ("ok", "cost_unmatched"):
            continue
        field, paths = simulate_winner(study, directory, common, row, witness_plan)
        evidence.extend(paths)
        fields[arm] = field
        row.update({key: field.get(key, "NA") for key in ("D_hot_mm", "Cov010", "field_path", "forward_task", "field_s")})
        if field["status"] != "ok":
            row.update(status="failed", reason=field.get("reason", field["status"]))
    random = arms[0]
    for row in arms[1:]:
        if row["status"] != "ok" or random["status"] != "ok":
            continue
        pair = compare_endpoints(fields[row["arm"]], fields["random"], target["group"])
        comparisons.append(dict(case_id=row["case_id"], receiver=method, arm=row["arm"], comparison="cached_donor", **pair))
        if zero_cost:
            control, comparison, paths = run_zero_cost(study, directory, common, target, prepared, random, row,
                                                       fields, search_id, witness_plan)
            controls.append(control)
            evidence.extend(paths)
            if comparison is not None:
                comparisons.append(comparison)
    return dict(arms=arms, comparisons=comparisons, controls=controls, evidence=evidence)


def plot_cost(out: Path, arms: list[dict[str, Any]], controls: list[dict[str, Any]]) -> Path | None:
    """显示现有供体与从零成本下，完成共评的各臂定位读数。"""
    valid = [row for row in arms if row["status"] == "ok"]
    if not valid:
        return None
    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    colours = dict(zip(METHODS, ("#1169a0", "#aa5524")))
    styles = dict(random="o", own="s", cross="^", restart="x")
    for row in valid:
        colour, symbol = colours[row["receiver"]], styles[row["arm"]]
        left.scatter(row["receiver_s"], row["D_hot_mm"], color=colour, marker=symbol)
        right.scatter(row["total_s"], row["D_hot_mm"], color=colour, marker=symbol,
                      label=f"{row['receiver']} {row['arm']}")
    for row in controls:
        if row.get("D_hot_mm", "NA") != "NA":
            right.scatter(row["total_s"], row["D_hot_mm"], color=colours[row["receiver"]], marker=styles["restart"],
                          label=f"{row['receiver']} {row['arm']} restart")
    left.set(title="Cached donor", xlabel="Receiver search (s)", ylabel="Hotspot distance (mm)")
    right.set(title="From zero", xlabel="Donor and receiver search (s)", ylabel="Hotspot distance (mm)")
    right.legend(fontsize=7)
    path = out / "figures/warm_start_cost.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145)
    return path


def output_table(path: Path, rows: list[dict[str, Any]], required: tuple[str, ...]) -> None:
    """发布共同协议的固定列及本次已生成的有效字段。"""
    columns = tuple(dict.fromkeys([*required, *(key for row in rows for key in row)]))
    write_csv(path, rows, columns)


def write_report(
    study: Path, out: Path, info: dict[str, Any], arms: list[dict[str, Any]], results: list[dict[str, Any]], evidence: list[dict[str, str]],
) -> None:
    """发布初始化判定、同成本配对、独立重启详情与成本图。"""
    pairs = [row for result in results for row in result["comparisons"]]
    controls = [row for result in results for row in result["controls"]]
    plot = plot_cost(out, arms, controls)
    files = {
        "cases.csv": ([{key: row[key] for key in CASE_COLUMNS} for row in arms], CASE_COLUMNS),
        "warm_start.csv": ([{key: value for key, value in row.items() if key != "candidate"} for row in arms], TABLE_COLUMNS),
        "paired.csv": (pairs, ("case_id", "receiver", "arm", "comparison", "decision")),
        "cold_restart.csv": (controls, ("case_id", "receiver", "arm", "planned_total_s", "duration_s", "total_s", "delta_s", "status", "reason")),
    }
    for filename, (rows, columns) in files.items():
        output_table(out / filename, rows, columns)
    good = sum(row["status"] == "ok" for row in arms)
    failed = sum(row["status"] in ("failed", "cost_unmatched") for row in arms)
    pending = len(arms) - good - failed
    lines = ["# 16 同条件互供先导对照", "", f"sub1/STN 的共同正向合格臂 {good}/{len(arms)}，失败或成本不匹配 {failed}，待处理 {pending}。",
             "", "每条接收路线的各臂使用同一个体、靶位、ROC0、等幅电流和有效评价预算。"
             "随机臂的初始种群为参照，两个供体臂各替换一个个体。", "",
             "已有供体的成本取接收方搜索时间。从零成本包含供体实际搜索和接收方搜索。"
             "可选独立重启在达到接收方时间边界的下一次请求前停止，记录实际总耗时、超差和有效次数。"
             "共评耗时单列。", ""]
    for row in arms:
        lines.append(f"- `{row['case_id']}` 为 `{row['status']}`，原因 `{row['reason']}`。")
    lines.extend(["", "[搜索与共同场](warm_start.csv) · [配对](paired.csv) · [独立重启](cold_restart.csv)"])
    if plot is not None:
        lines.append("[成本和定位图](figures/warm_start_cost.png)")
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = [report, out / "plan.json", out / "commands.ps1", *(out / filename for filename in files)]
    if plot is not None:
        outputs.append(plot)
    write_csv(out / "artifacts.csv", [*evidence, *(row for result in results for row in result["evidence"]),
                                      *(artifact(study, "run", "output", path) for path in outputs)])
    write_json(out / "run.json", info | dict(passed=good, failed=failed, pending=pending))


def run(study: Path, out: Path, baseline_id: str, budget_id: str, run_id: str, execute: bool, zero_cost: bool, jobs: int) -> None:
    """冻结先导对照计划，并在显式执行时运行已具备条件的接收路线。"""
    implementation = [artifact(study, "run", "implementation", Path(path)) for path in (__file__, matched_initialization.__file__)]
    reason, inputs = "NA", None
    try:
        guarded_ga()
        inputs = read_inputs(study, out.parent / "matched", budget_id)
    except SEARCH_ERRORS as error:
        reason = str(error)
    arms = [row for method in METHODS for row in declared_arms(method, inputs)]
    ready = []
    if inputs is None:
        for row in arms:
            row["reason"] = reason
    else:
        for method in METHODS:
            selected = [row for row in arms if row["receiver"] == method and row["status"] == "pending"]
            try:
                registered_case(study, baseline_id, SUBJECT, TARGET, method)
            except SEARCH_ERRORS as error:
                reason = str(error)
                for row in selected:
                    row["reason"] = reason
            else:
                ready.append(method)
                for row in selected:
                    row["reason"] = "待执行"
    info = dict(experiment="16", phase="matched_comparison", stage="pilot", run_id=run_id,
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), environment=sys.executable,
                upstream_runs=[f"03/{baseline_id}", f"15/{budget_id}", f"16/{run_id}/matched"], started_at=timestamp(), finished_at=None,
                config=dict(seed=SEED, restart_seed=RESTART_SEED, current_mA=[CURRENT_MA, CURRENT_MA], threshold_vpm=THRESHOLD,
                            initial_population=INITIAL_REQUESTS, zero_cost_control=zero_cost, jobs=jobs), status="ready" if ready else "blocked")
    write_json(out / "plan.json", info | dict(arms=arms))
    print(f"16 同预算对照 {info['status']}，计划 {out / 'plan.json'}，原因 {reason}", flush=True)
    if not execute:
        return
    command = " ".join("'" + arg.replace("'", "''") + "'" for arg in sys.argv[1:])
    with (out / "commands.ps1").open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp16_compare {command}\n")
    evidence, results = implementation.copy(), []
    if inputs is not None:
        evidence.extend(inputs.evidence)
    if ready:
        info["status"] = "running"
        write_report(study, out, info, arms, results, evidence)
        with ProcessPoolExecutor(max_workers=min(jobs, len(ready))) as executor:
            pending = {}
            for method in ready:
                rows = [row for row in arms if row["receiver"] == method]
                pending[executor.submit(run_receiver, study, out, baseline_id, budget_id, rows, inputs.signature, zero_cost)] = rows
            for future in as_completed(pending):
                try:
                    result = future.result()
                    for original, updated in zip(pending[future], result["arms"]):
                        original.update(updated)
                    results.append(result)
                except SEARCH_ERRORS as error:
                    for original in pending[future]:
                        if original["status"] == "pending":
                            original.update(status="failed", reason=str(error))
                write_report(study, out, info, arms, results, evidence)
        info["status"] = "compared" if all(row["status"] == "ok" for row in arms) else "partial"
    info["finished_at"] = timestamp()
    write_report(study, out, info, arms, results, evidence)


def main() -> None:
    """解析 16 先导比较和可选从零时间对照的运行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--budget-id", default="run-001")
    parser.add_argument("--execute", action="store_true", help="运行实际搜索和共同正向，缺省仅生成计划")
    parser.add_argument("--zero-cost-control", action="store_true", help="执行从零总时长的独立重启对照")
    parser.add_argument("--jobs", type=int, default=1, help="并行接收路线数")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 须为正整数")
    for run_id in (args.run_id, args.baseline_id, args.budget_id):
        if not run_id or run_id in (".", "..") or any(char in run_id for char in ("/", "\\", ":")):
            parser.error("运行号必须为单个目录名")
    study = args.study_root.resolve()
    out = study / "results/16" / args.run_id / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".scheduler.lock"), timeout=0):
        run(study, out, args.baseline_id, args.budget_id, args.run_id, args.execute, args.zero_cost_control, args.jobs)


if __name__ == "__main__":
    main()
