"""在两路各 1 mA 的 ROC0 下比较等预算的单起点与三起点。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.demo.ti_depth.exp03 import FORWARD_OUTPUTS, align_common_forward
from neuracle.ti_depth_study import restart_budget
from neuracle.ti_depth_study.baseline import content_id, forward_config, run_task
from neuracle.ti_depth_study.diagnostics import FieldCase, load_case
from neuracle.ti_depth_study.metrics import measure_field
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json
from neuracle.ti_depth_study.restart_budget import BASED_SETTINGS, CURRENT_MA, INITIAL_REQUESTS, METHODS, THRESHOLD, Trace
from neuracle.ti_depth_study.restart_budget import prepare_case, search_based, search_free


SEED_GROUPS = ((20220815, 20220816, 20220817), (20220818, 20220819, 20220820), (20220821, 20220822, 20220823))
TRACES = ("request", "status", "configuration_id", "n_valid_unique", "arm_n_valid_unique", "objective",
          "best_objective", "parameters", "wall_s")
RESTART_COLUMNS = ("method", "seed_group", "arm", "restart_id", "B_planned", "n_valid_unique", "n_requests",
                   "cache_hits", "invalid", "selected", "wall_s", "case_id", "subject", "target", "seed",
                   "best_objective", "status", "reason", "labels", "D_hot_mm", "Cov010")
SEARCH_ERRORS = (OSError, RuntimeError, ValueError, KeyError, TypeError, ArithmeticError)


def registered_case(study: Path, baseline_id: str, subject: str, target: str, method: str) -> tuple[dict[str, str], dict[str, str]]:
    """读取已质控的 03 共评场与已登记的 02 靶目标。"""
    table = study / "results/03" / baseline_id / "baseline_fields.csv"
    if not table.exists():
        raise FileNotFoundError(f"等待 03 共评场表: {table}")
    native = next((row for row in read_csv(table) if (row["subject"], row["target"], row["method"]) == (subject, target, method)), None)
    if native is None or native["status"] != "ok":
        raise ValueError(f"等待 03 共评场验收: {subject}/{target}/{method}")
    targets = read_csv(study / "results/02/run-001/targets.csv")
    target_row = next((row for row in targets if row["target"] == target), None)
    if target_row is None:
        raise ValueError(f"02 尚未登记靶区 {target}")
    return native, target_row


def input_id(prepared: dict[str, Any], native: dict[str, str], target: dict[str, str]) -> str:
    """以同一头模、帽位、配准、靶源和研究实现标识搜索输入。"""
    head = prepared["head"]
    mesh_hash = digest(prepared["mesh"])
    if mesh_hash != native["mesh_hash"]:
        raise ValueError("搜索头模与 03 共同评估网格来源不同")
    return content_id(dict(mesh=mesh_hash, cap=digest(prepared["cap"]), atlas=digest(Path(target["mask_path"])),
                           registration={item.name: digest(item) for item in sorted((head / "toMNI").glob("*")) if item.is_file()},
                           code=digest(Path(restart_budget.__file__)), current_ma=CURRENT_MA, threshold_vpm=THRESHOLD))


def run_search(
    prepared: dict[str, Any], output: Path, method: str, seed: int, budget: int | None, config_id: str,
    scores: dict[str, float],
) -> tuple[dict[str, Any], Path]:
    """执行或核验复用一个起点，保持共享缓存与逐请求轨迹累计一致。

    Parameters
    ----------
    prepared, output : dict and Path
        研究头模、帽位、靶区等输入及单起点目录。
    method, seed, budget, config_id : str, int, int or None and str
        路线、起点种子、有效评价上限和完整输入校验标识。
    scores : dict
        同一比较臂此前起点已有的规范配置及其目标。

    Returns
    -------
    result, trace_path : tuple
        单起点摘要及 evaluation 表路径。
    """
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "evaluations.csv"
    result_path = output / "result.json"
    configuration = dict(method=method, seed=seed, budget=budget, input=config_id,
                         cache_input=content_id({"scores": scores}))
    if result_path.is_file() and trace_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (previous["configuration"] == configuration and previous["trace_sha256"] == digest(trace_path)
                and previous["status"] in ("ok", "cost_unmatched")):
            scores.update({row["configuration_id"]: float(row["objective"]) for row in read_csv(trace_path) if row["status"] == "new"})
            return previous, trace_path
    trace = Trace(budget, INITIAL_REQUESTS[method], scores.copy())
    started = perf_counter()
    print(f"15 {output.name} {method} {seed} B={budget}", flush=True)
    labels = []
    reason = "NA"
    status = "ok"
    try:
        labels = search_based(prepared, trace, seed) if method == "leadfield_based" else search_free(prepared, trace, seed, output)
        if len(labels) != 4 or len(set(labels)) != 4 or len(trace.rows) < trace.minimum_requests:
            raise ValueError("未验收四个互异帽位及完整初始化")
        if budget is not None and trace.new_evals != budget:
            status, reason = "cost_unmatched", f"有效评价 {trace.new_evals} / {budget}"
    except SEARCH_ERRORS as error:
        status, reason = "failed", str(error)
    write_csv(trace_path, trace.rows, TRACES)
    result = dict(configuration=configuration, n_valid_unique=trace.new_evals, n_requests=len(trace.rows),
                  cache_hits=trace.cache_hits, invalid=trace.invalid, wall_s=perf_counter() - started, labels=labels,
                  best_objective=trace.best_score if trace.best_parameters is not None else "NA", status=status, reason=reason,
                  trace_sha256=digest(trace_path))
    write_json(result_path, result)
    scores.update(trace.scores)
    print(f"15 {method} 请求 {len(trace.rows)} 有效 {trace.new_evals} 状态 {status} {reason}", flush=True)
    return result, trace_path


def natural(study: Path, out: Path, baseline_id: str, method: str) -> dict[str, Any]:
    """测量 sub1/STN 的固定电流原生搜索预算，每条路线独立获锁复算。"""
    subject, target = "sub1", "stn295"
    native, target_row = registered_case(study, baseline_id, subject, target, method)
    directory = out / "search/natural" / method
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".search.lock")):
        prepared = prepare_case(study, subject, target_row, method, directory)
        result, trace = run_search(prepared, directory, method, SEED_GROUPS[0][0], None,
                                   input_id(prepared, native, target_row), {})
    initial = sum(row["status"] == "new" for row in read_csv(trace)[:INITIAL_REQUESTS[method]])
    budget = result["n_valid_unique"] // 3 * 3
    return {key: result[key] for key in ("n_valid_unique", "n_requests", "cache_hits", "invalid", "wall_s", "status", "reason")} | dict(
        subject=subject, target=target, method=method, condition="fixed_1mA_per_channel_ROC0", trace=trace.relative_to(out).as_posix(),
        n_init_valid_unique=initial, B=budget if result["status"] == "ok" and budget >= 3 and budget // 3 >= initial else "NA")


def forward_winner(study: Path, out: Path, common: FieldCase, winner: dict[str, Any]) -> dict[str, Any]:
    """只对本臂原目标最优起点执行 03 同口径复算，并回填物理读数。

    Parameters
    ----------
    study, out : Path
        数据根和本病例的独立运行目录。
    common : FieldCase
        03 通过核验的参考几何和目标。
    winner : dict
        已选起点，含按载波排列的帽位及配对标识。

    Returns
    -------
    dict
        同协议场指标、任务和对齐产物路径。
    """
    subject = winner["subject"]
    pair = [CURRENT_MA, -CURRENT_MA]
    config = forward_config(winner["labels"].split(";"), (pair, pair))
    task = f"15_{subject}_{content_id(dict(config=config, mesh=common.native['mesh_hash']))[:20]}"
    directory = run_task(study, "ti_forward", subject, task, config, out / "commands.ps1", FORWARD_OUTPUTS)
    field_dir = out / "fields"
    field_dir.mkdir(exist_ok=True)
    field, alignment = align_common_forward(directory, common.geometry_path, field_dir)
    with np.load(field) as data:
        metrics = measure_field(common.geometry, common.target, data["E_A"], data["E_B"], data["valid_mask"])
    return metrics | alignment | dict(forward_task=directory.relative_to(study).as_posix(),
                                       field_path=field.relative_to(study).as_posix())


def compare_case(study: Path, out: Path, baseline_id: str, case: dict[str, Any], budget: int) -> dict[str, Any]:
    """执行一格三组等预算配对，逐起点保留复跑证据并复算获胜者。

    Parameters
    ----------
    study, out, baseline_id : Path and str
        数据根、运行目录及 03 上游运行号。
    case, budget : dict and int
        受试者、靶点和路线的登记信息及冻结有效预算。

    Returns
    -------
    dict
        病例状态、起点及配对读数、逐请求轨迹和文件证据。
    """
    subject, target, method, case_id = (case[key] for key in ("subject", "target", "method", "case_id"))
    directory = out / "search/compare" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    restarts, traces, pairs, fields, evidence = [], [], [], [], []
    result = dict(case=case, restarts=restarts, traces=traces, pairs=pairs, fields=fields, evidence=evidence)
    try:
        native, target_row = registered_case(study, baseline_id, subject, target, method)
        common = load_case(study, native)
        prepared = prepare_case(study, subject, target_row, method, directory)
        signature = input_id(prepared, native, target_row)
    except SEARCH_ERRORS as error:
        case.update(status="failed", reason=str(error), wall_s=perf_counter() - started)
        return result
    evidence.append(artifact(study, case_id, "共同网格", common.geometry_path))
    for group, seeds in enumerate(SEED_GROUPS, 1):
        conditions = dict(case_id=case_id, subject=subject, target=target, method=method, seed_group=group)
        selected_fields = {}
        counts = {}
        for arm, planned, group_seeds in (("single", budget, seeds[:1]), ("triple", budget // 3, seeds)):
            scores: dict[str, float] = {}
            arm_rows = []
            for restart_id, seed in enumerate(group_seeds, 1):
                part = directory / f"g{group}_{arm}_{restart_id}"
                record, trace = run_search(prepared, part, method, seed, planned, signature, scores)
                identity = conditions | dict(arm=arm, restart_id=restart_id)
                row = identity | dict(B_planned=planned, seed=seed, selected=False, labels=";".join(record["labels"])) | {
                    key: record[key] for key in ("n_valid_unique", "n_requests", "cache_hits", "invalid", "wall_s", "best_objective", "status", "reason")}
                arm_rows.append(row)
                traces.append((identity, trace))
                evidence.append(artifact(study, case_id, "搜索轨迹", trace))
                evidence.append(artifact(study, case_id, "搜索摘要", part / "result.json"))
                mapping = part / "electrode_mapping.json"
                if mapping.is_file():
                    evidence.append(artifact(study, case_id, "原生帽位映射", mapping))
            restarts.extend(arm_rows)
            valid = [row for row in arm_rows if row["status"] in ("ok", "cost_unmatched") and row["labels"]]
            counts[arm] = sum(row["n_valid_unique"] for row in arm_rows)
            if not valid:
                continue
            winner = min(valid, key=lambda row: (row["best_objective"], row["restart_id"]))
            winner["selected"] = True
            identity = conditions | dict(arm=arm, restart_id=winner["restart_id"], labels=winner["labels"])
            try:
                field = forward_winner(study, directory, common, winner)
                evidence.append(artifact(study, case_id, "对齐共同场", study / field["field_path"]))
                evidence.append(artifact(study, case_id, "正向执行清单", study / field["forward_task"] / "study_execution.json"))
                winner.update({key: field.get(key, "NA") for key in ("D_hot_mm", "Cov010")})
            except SEARCH_ERRORS as error:
                field = dict(status="failed", reason=str(error))
            selected_fields[arm] = field
            fields.append(identity | field)
        complete = all(row["status"] == "ok" for row in restarts if row["seed_group"] == group)
        cost_status = "matched" if complete and counts["single"] == counts["triple"] == budget else "cost_unmatched"
        pair = conditions | dict(arm="triple", reference_arm="single", B_planned=budget,
                                 n_single=counts["single"], n_triple=counts["triple"], cost_status=cost_status)
        assessment = compare_endpoints(selected_fields.get("triple", {}), selected_fields.get("single", {}), target_row["group"])
        if cost_status != "matched":
            assessment.update(field_decision=assessment["decision"], decision="undetermined", reason="未达到相同有效评价预算")
        pairs.append(pair | assessment)
    bad = sum(pair["cost_status"] != "matched" or pair["decision"] == "undetermined" for pair in pairs)
    case.update(status="ok" if not bad else "failed", reason="NA" if not bad else f"{bad} 组成本不匹配或共评场缺失",
                wall_s=perf_counter() - started)
    return result


def plot_best(out: Path, evaluations: list[dict[str, Any]]) -> Path:
    """按路线分栏绘出两臂随累计新有效评价变化的最佳搜索分。"""
    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(METHODS))
    series = {}
    for row in evaluations:
        if row["status"] == "new":
            key = tuple(row[name] for name in ("method", "case_id", "seed_group", "arm"))
            series.setdefault(key, []).append((int(row["arm_n_valid_unique"]), float(row["objective"])))
    styles = {"single": ("#0b63a8", "Single start"), "triple": ("#c25020", "Three starts"), "natural": ("#797979", "Natural")}
    for method, axis in zip(METHODS, axes):
        shown = set()
        for (route, _, _, arm), samples in series.items():
            if route != method:
                continue
            xs, ys = zip(*sorted(samples))
            color, label = styles[arm]
            axis.step(xs, -np.minimum.accumulate(ys), where="post", color=color, alpha=0.55, label=label if arm not in shown else None)
            shown.add(arm)
        if shown:
            axis.legend()
        axis.set(xlabel="Cumulative unique valid evaluations", ylabel="Best search ROC0 score", title=method)
    output = out / "figures/best_by_evals.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=145)
    return output


def trajectory_prefix(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """以相同新评价次数截断每组轨迹，供预算未匹配时核对搜索分。"""
    series = {}
    for row in evaluations:
        if row["status"] == "new" and row["arm"] in ("single", "triple"):
            key = (row["case_id"], row["seed_group"])
            by_arm = series.setdefault(key, {})
            by_arm.setdefault(row["arm"], []).append((int(row["arm_n_valid_unique"]), float(row["objective"])))
    rows = []
    for (case_id, group), arms in series.items():
        if not all(arms.get(key) for key in ("single", "triple")):
            continue
        cutoff = min(len(samples) for samples in arms.values())
        values = {f"best_{arm}": min(value for count, value in samples if count <= cutoff) for arm, samples in arms.items()}
        rows.append(dict(case_id=case_id, seed_group=group, n_valid_unique=cutoff, **values))
    return rows


def save_state(
    study: Path, out: Path, run_info: dict[str, Any], cases: list[dict[str, Any]],
    budgets: list[dict[str, Any]], results: list[dict[str, Any]],
) -> None:
    """发布可重放的 15 记录、等预算验收判定和研究索引。"""
    evidence = [artifact(study, "run", "研究实现", Path(path)) for path in (__file__, restart_budget.__file__)]
    evaluations = []
    for row in budgets:
        if "trace" not in row:
            continue
        path = out / row["trace"]
        identity = {key: row[key] for key in ("subject", "target", "method")}
        identity |= dict(case_id=f"natural_{row['method']}", seed_group="natural", arm="natural", restart_id=1)
        evaluations.extend(identity | item for item in read_csv(path))
        evidence.append(artifact(study, identity["case_id"], "自然预算轨迹", path))
        evidence.append(artifact(study, identity["case_id"], "自然预算摘要", path.with_name("result.json")))
    results = sorted(results, key=lambda result: result["case"]["case_id"])
    for result in results:
        evidence.extend(result["evidence"])
        for identity, path in result["traces"]:
            evaluations.extend(identity | item for item in read_csv(path))
    if evaluations:
        evidence.append(artifact(study, "run", "累计有效评价曲线", plot_best(out, evaluations)))
    tables = {
        "fixed_budget_observations.csv": (budgets, ("subject", "target", "method", "condition", "B", "n_valid_unique", "n_init_valid_unique",
                                                   "n_requests", "cache_hits", "invalid", "wall_s", "status", "reason", "trace")),
        "restart_budget.csv": ([row for result in results for row in result["restarts"]], RESTART_COLUMNS),
        "evaluations.csv": (evaluations, ("case_id", "subject", "target", "method", "seed_group", "arm", "restart_id", *TRACES)),
        "winner_fields.csv": ([row for result in results for row in result["fields"]], ("case_id", "subject", "target", "method",
                                                                                        "seed_group", "arm", "status")),
        "paired.csv": ([row for result in results for row in result["pairs"]], ("case_id", "subject", "target", "method", "seed_group",
                                                                               "arm", "reference_arm", "cost_status", "decision")),
        "trajectory_pairs.csv": (trajectory_prefix(evaluations), ("case_id", "seed_group", "n_valid_unique", "best_single", "best_triple")),
        "cases.csv": (cases, ("case_id", "subject", "target", "method", "status", "reason", "wall_s")),
    }
    for name, (rows, required) in tables.items():
        path = out / name
        columns = tuple(dict.fromkeys([*required, *(key for row in rows for key in row)]))
        write_csv(path, rows, columns)
        evidence.append(artifact(study, "run", "结果表", path))
    write_csv(out / "artifacts.csv", evidence)
    passed = sum(row["status"] == "ok" for row in cases)
    failed = sum(row["status"] == "failed" for row in cases)
    pending = len(cases) - passed - failed
    pairs = tables["paired.csv"][0]
    matched = [row for row in pairs if row["cost_status"] == "matched" and row["decision"] != "undetermined"]
    report = ["# 15 固定预算的多起点搜索", "", f"阶段 {run_info['stage']}，病例通过 {passed}/{len(cases)}，失败 {failed}，待处理 {pending}。",
              "", "## 固定电流自然计时", "", "| 路线 | 请求 | 有效新布局 | 初始化有效 | 缓存 | 无效 | 冻结 B | 秒 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in budgets:
        report.append(f"| {row['method']} | {row.get('n_requests', 'NA')} | {row.get('n_valid_unique', 'NA')} | "
                      f"{row.get('n_init_valid_unique', 'NA')} | {row.get('cache_hits', 'NA')} | {row.get('invalid', 'NA')} | "
                      f"{row['B']} | {row.get('wall_s', 'NA')} |")
    report.extend(["", "种子组和预算记录见 [run.json](run.json)。同臂三个起点使用独立种群和种子，跨起点命中缓存也不计为新评价。",
                   "共同正向指标来自每臂按搜索 ROC0 选出的一个布局。各起点原目标及选中标记见 [restart_budget.csv](restart_budget.csv)。",
                   "based 计时从 leadfield 及 ROI 准备完毕起算。free 计时包括连续搜索器的头模和在线 FEM 准备。",
                   "", f"共评且预算相同 {len(matched)}/{len(pairs)} 组。以下差值为三起点减单起点。"])
    for route in METHODS:
        route_rows = [row for row in matched if row["method"] == route]
        if route_rows:
            distances = [row["delta_D_hot_mm"] for row in route_rows]
            coverages = [row["delta_Cov010_pp"] for row in route_rows]
            report.append(f"- {route} 的 {len(route_rows)} 组距离差中位数 {np.median(distances):.3f} mm，"
                          f"范围 {min(distances):.3f} 至 {max(distances):.3f} mm。覆盖差中位数 {np.median(coverages):.3f} 个百分点。")
    report.extend(["", "成本不匹配时，同一有效评价次数的原目标读数见 [轨迹截断](trajectory_pairs.csv)。",
                   "开发阶段先评估三个先导靶。后续保留头模的规则冻结取决于共同协议的皮层回归验收。",
                   "", "[逐请求记录](evaluations.csv) · [胜者共同场](winner_fields.csv) · [配对分布](paired.csv)"])
    if evaluations:
        report.append("[有效评价曲线](figures/best_by_evals.png)")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "run.json", run_info)
    update_index(study, dict(experiment="15", run_id=run_info["run_id"], stage=run_info["stage"], run_status=run_info["status"],
                             scientific_decision="等待开发验收" if matched else "等待等成本共评", ok_cases=passed,
                             failed_cases=failed, pending_cases=pending, report_path=(out / "report.md").relative_to(study).as_posix()))


def run(study: Path, out: Path, run_id: str, baseline_id: str, stage: str, jobs: int) -> None:
    """先登记自然预算和种子，再执行研究所需的先导或开发病例。"""
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".run.lock")):
        arguments = (str(study), run_id, baseline_id)
        study_arg, run_arg, baseline_arg = ("'" + value.replace("'", "''") + "'" for value in arguments)
        command = (f"python -m neuracle.demo.ti_depth.exp15 --study-root {study_arg} --run-id {run_arg} "
                   f"--baseline-id {baseline_arg} --stage {stage} --jobs {jobs}")
        with (out / "commands.ps1").open("a", encoding="utf-8") as stream:
            stream.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\n{command}\n")
        budgets = []
        for method in METHODS:
            try:
                budgets.append(natural(study, out, baseline_id, method))
            except SEARCH_ERRORS as error:
                budgets.append(dict(method=method, B="NA", reason=str(error), status="pending"))
        frozen = {row["method"]: row["B"] for row in budgets if row["B"] != "NA"}
        manifest = out / "run.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            recorded = previous.get("config", {}).get("budgets", {})
            if any(frozen.get(route) not in (None, value) for route, value in recorded.items()):
                raise ValueError("同一运行号的自然预算与预登记预算不一致，请新建运行号")
        subjects = ("sub1",) if stage == "pilot" else ("sub1", "sub2")
        targets = ("stn295",) if stage == "pilot" else PILOT_TARGETS
        cases = [dict(case_id=f"{subject}_{target}_{method}", subject=subject, target=target, method=method,
                      status="pending", reason="等待等预算实测", wall_s="NA")
                 for subject in subjects for target in targets for method in METHODS]
        run_info = dict(experiment="15", run_id=run_id, stage=stage, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        environment=sys.executable, upstream_runs=[f"03/{baseline_id}", "02/run-001"], started_at=timestamp(), finished_at=None,
                        status="running", config=dict(current_mA=[CURRENT_MA, CURRENT_MA], threshold_vpm=THRESHOLD, budgets=frozen,
                                                      seed_groups=SEED_GROUPS, campaign_jobs=jobs, based_settings=asdict(BASED_SETTINGS),
                                                      minimum_initialization_requests=INITIAL_REQUESTS))
        ready = []
        for case in cases:
            if case["method"] not in frozen:
                case["reason"] = "等待足以完成各起点初始化的自然预算"
                continue
            try:
                registered_case(study, baseline_id, case["subject"], case["target"], case["method"])
            except (OSError, ValueError) as error:
                case["reason"] = str(error)
                continue
            ready.append(case)
        results = []
        save_state(study, out, run_info, cases, budgets, results)
        if ready:
            with ProcessPoolExecutor(max_workers=min(jobs, len(ready))) as executor:
                pending = {executor.submit(compare_case, study, out, baseline_id, case.copy(), frozen[case["method"]]): case for case in ready}
                for future in as_completed(pending):
                    case = pending[future]
                    try:
                        result = future.result()
                    except SEARCH_ERRORS as error:
                        case.update(status="failed", reason=str(error))
                    else:
                        case.update(result["case"])
                        results.append(result)
                    save_state(study, out, run_info, cases, budgets, results)
        run_info.update(status="complete" if all(case["status"] == "ok" for case in cases) else "partial" if ready else "blocked", finished_at=timestamp())
        save_state(study, out, run_info, cases, budgets, results)


def main() -> None:
    """核验固定电流自然预算或执行完整 15 研究阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    parser.add_argument("--measure-method", choices=METHODS, help="仅测量本路线的 sub1/STN 固定电流自然预算")
    parser.add_argument("--jobs", type=int, default=1, help="并行起点比较病例数")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 须为正整数")
    for name in (args.run_id, args.baseline_id):
        if not name or name in (".", "..") or any(char in name for char in ("/", "\\", ":")):
            parser.error("运行号必须为单个目录名")
    study = args.study_root.resolve()
    out = study / "results/15" / args.run_id
    if args.measure_method:
        result = natural(study, out, args.baseline_id, args.measure_method)
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "ok":
            raise SystemExit(1)
        return
    run(study, out, args.run_id, args.baseline_id, args.stage, args.jobs)


if __name__ == "__main__":
    main()
