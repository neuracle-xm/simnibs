"""从同一版共同正向候选库做可追溯的离线重排。"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np

from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry, measure_field
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json
from simnibs.optimization.tes_flex_optimization.measures import ROC


@dataclass(frozen=True)
class Candidate:
    """某候选的共同正向场、ROI 读数与原口径 ROC。"""

    identifier: str
    path: Path
    absolute: dict[str, Any]
    roc: float


ScoreFunction = Callable[[Geometry, BoolArray, list[Candidate]], list[dict[str, Any]]]
_ENDPOINTS = ("D_hot_mm", "Cov010", "Q50_T", "Q99_Cout", "Vout010", "absolute_hit", "hot_overlap")


def compare_endpoints(new: dict[str, Any], reference: dict[str, Any], group: str) -> dict[str, Any]:
    """用 02 的共同容限比较候选与同层参照。

    Parameters
    ----------
    new, reference : dict
        同一个头模、靶和电流层上的共同 FEM 指标。
    group : str
        02 中登记的解剖分组。皮层单列绝对定位和覆盖。

    Returns
    -------
    dict
        配对绝对值、差值、强度与外溢比及逐格判定。
    """
    comparison = {key: value for endpoint in _ENDPOINTS for key, value in (
        (endpoint, new.get(endpoint, "NA")), (f"reference_{endpoint}", reference.get(endpoint, "NA")))}
    if not all(row.get("status") == "ok" and all(np.isfinite(float(row[metric])) for metric in _ENDPOINTS[:5]) for row in (new, reference)):
        return comparison | {"decision": "undetermined", "reason": "共同场或关键指标缺失"}
    delta_distance = new["D_hot_mm"] - reference["D_hot_mm"]
    delta_coverage = 100 * (new["Cov010"] - reference["Cov010"])
    ratios = {output: new[metric] / reference[metric] if reference[metric] > 0 else "NA" for output, metric in (
        ("Q50_ratio", "Q50_T"), ("Q99_Cout_ratio", "Q99_Cout"), ("Vout010_ratio", "Vout010"))}
    failures = []
    if delta_distance > 2.0:
        failures.append("距靶退化超过 2 mm")
    if delta_coverage < -5.0 - 1e-10 or (reference["Cov010"] > 0 and new["Cov010"] == 0):
        failures.append("覆盖越过保持线")
    if new["Q50_T"] < 0.95 * reference["Q50_T"]:
        failures.append("靶内中位场保留不足 95%")
    if new["Q99_Cout"] > 1.05 * reference["Q99_Cout"] or new["Vout010"] > 1.05 * reference["Vout010"]:
        failures.append("靶外读数超过 105% 上限")
    eligible = reference["D_hot_mm"] >= 2.0
    if group == "subcortical" and not eligible and new["D_hot_mm"] > 2.0:
        failures.append("距靶原在下限以内，候选超过 2 mm")
    if failures:
        decision = "failed"
    elif group != "subcortical":
        decision = "maintained_hit" if new["absolute_hit"] and new["Cov010"] > 0 else "zero_coverage" if new["Cov010"] == reference["Cov010"] == 0 else "maintained_level"
    else:
        decision = "improved" if eligible and delta_distance <= -2.0 and new["Cov010"] > 0 else "maintained"
    return comparison | ratios | {"delta_D_hot_mm": delta_distance, "delta_Cov010_pp": delta_coverage,
                                  "eligible_improvement": eligible, "decision": decision, "reason": "；".join(failures) or "NA"}


def _measure_candidates(study: Path, geometry: Geometry, target: BoolArray, rows: list[dict[str, str]], geo_hash: str, threshold: float) -> list[Candidate]:
    """验证库实体散列，以同一四面体口径计算各布局的绝对读数。

    Parameters
    ----------
    study : Path
        数据根；候选索引路径以此为基准。
    geometry, target : Geometry and ndarray
        固定网格及当前解剖目标。
    rows : list of dict
        候选索引中的成功布局。
    geo_hash : str
        已验证的共同网格 SHA-256。
    threshold : float
        原 ROC 的靶阈值。靶外固定 0.10 V/m。

    Returns
    -------
    list of Candidate
        包络有效且可作定位判定的布局。
    """
    candidates = []
    for row in rows:
        if row["geometry_hash"] != geo_hash or float(row["IA_mA"]) != 1.0 or float(row["IB_mA"]) != 1.0:
            raise ValueError(f"候选输入超出共同网格或电流约定: {row['candidate_id']}")
        path = study / row["field_path"]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if digest(path) != meta["sha256"]:
            raise ValueError(f"场实体散列已改变: {path}")
        with np.load(path) as data:
            field = data["max_TI"]
            measured = measure_field(geometry, target, data["E_A"], data["E_B"], data["valid_mask"])
        if measured["status"] != "ok" or not np.isfinite(field).all():
            continue
        if not target.all():
            roc = float(ROC(e1=field[target], e2=field[~target], threshold=[0.10, threshold], focal=True))
            candidates.append(Candidate(row["candidate_id"], path, measured, roc))
    return candidates


def run_pool_experiment(
    study: Path, experiment: str, run_id: str, baselines: str, stage: str, score: ScoreFunction,
    table: str, columns: tuple[str, ...], config: dict[str, Any],
    table_transform: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> None:
    """按 02 的约定执行同池评分、排名、配对及状态交接。

    Parameters
    ----------
    study : Path
        含 02、03 产物的本地研究目录。
    experiment, run_id, baselines, stage : str
        实验编号、本次运行号、03 运行号及 pilot/development。
    score : callable
        在同一头模、靶、候选上输出臂、损失和专属诊断。参照臂为 roc0。
    table, columns : str and tuple
        执行单要求的结果文件和专属字段。
    config : dict
        本次评分规则、阈值、尺度和实证阶段配置。
    table_transform : callable, optional
        将同一候选的多臂行合为专属表需要的单行结构。
    """
    started = timestamp()
    upstream = study / "results/03" / baselines
    out = study / "results" / experiment / run_id
    out.mkdir(parents=True, exist_ok=True)
    paths = upstream / "shared_inputs.csv"
    sources = read_csv(paths) if paths.is_file() else []
    groups = {row["target"]: row["group"] for row in read_csv(study / "results/02/run-001/targets.csv")}
    cases, details, paired, artifacts = [], [], [], []
    data_ids = {}
    expected_subjects = ("sub1",) if stage == "pilot" else ("sub1", "sub2")
    expected_targets = tuple(config.get("pilot_targets", ("stn295",))) if stage == "pilot" else tuple(config.get("development_targets", groups))
    for subject in expected_subjects:
        inputs = {row["role"]: row for row in sources if row["subject"] == subject}
        if inputs and inputs["geometry"]["data_id"] != inputs["candidate_index"]["data_id"]:
            raise ValueError(f"{subject} 的网格和候选库属于不同的冻结版本")
        if inputs:
            for item in inputs.values():
                if digest(study / item["path"]) != item["sha256"]:
                    raise ValueError(f"共享输入散列已改变: {item['path']}")
                artifacts.append(artifact(study, subject, item["role"], study / item["path"]))
            geo, _, targets = load_geometry(study / inputs["geometry"]["path"])
            pool_rows = read_csv(study / inputs["candidate_index"]["path"])
            valid_rows = [row for row in pool_rows if row["status"] == "ok"]
            data_ids[subject] = inputs["geometry"]["data_id"]
        for key in expected_targets:
            unique = f"{subject}_{key}"
            case: dict[str, Any] = dict(case_id=unique, subject=subject, target=key, method="shared_pool", arm="screening", seed="NA",
                        condition="1+1mA", task_id="NA", status="pending", reason="等待共同候选库", wall_s="NA")
            cases.append(case)
            if not inputs:
                continue
            try:
                candidates = _measure_candidates(study, geo, targets[key], valid_rows, inputs["geometry"]["sha256"], 0.10)
                if not candidates:
                    raise ValueError("共同候选库没有可作绝对定位的有限非零场")
                by_id = {candidate.identifier: candidate for candidate in candidates}
                scored = score(geo, targets[key], candidates)
                arms = dict.fromkeys(row["arm"] for row in scored)
                for arm in arms:
                    ranks = sorted((row for row in scored if row["arm"] == arm),
                                   key=lambda row: (row["returned_loss"], row["candidate_id"]))
                    rank = 0
                    for row in ranks:
                        if row.get("feasible", True):
                            rank += 1
                            row["rank"] = rank
                        else:
                            row["rank"] = "NA"
                        candidate = by_id[row["candidate_id"]]
                        row.update(case_id=unique, subject=subject, target=key, field_source="common_forward", D_hot_mm=candidate.absolute["D_hot_mm"],
                                   Cov010=candidate.absolute["Cov010"], Q99_Cout=candidate.absolute["Q99_Cout"], Vout010=candidate.absolute["Vout010"])
                winners = {row["arm"]: by_id[row["candidate_id"]] for row in scored if row["rank"] == 1}
                reference_arm = config.get("comparison_arm", "roc0")
                reference = winners[reference_arm]
                for arm, winner in winners.items():
                    if arm == reference_arm:
                        continue
                    result = compare_endpoints(winner.absolute, reference.absolute, groups[key])
                    paired.append(dict(case_id=f"{unique}:{winner.identifier}", reference_case_id=f"{unique}:{reference.identifier}",
                                       subject=subject, target=key, method="shared_pool", arm=arm, reference_arm=reference_arm, seed="NA",
                                       condition="1+1mA", stage=stage, field_source="common_forward", **result))
                details.extend(scored)
                case.update(status="ok", reason=f"有效场 {len(candidates)} / 共同池 {len(pool_rows)}")
            except (OSError, ValueError, KeyError) as error:
                case.update(status="failed", reason=str(error))
    known = ["case_id", "subject", "target", "arm", "candidate_id", "returned_loss", "rank", "field_source", "D_hot_mm", "Cov010", "Q99_Cout", "Vout010"]
    if table_transform is not None:
        details = table_transform(details)
        known = ["case_id", "subject", "target", "candidate_id", "field_source"]
    detail_keys = tuple(dict.fromkeys([*known, *columns, *(key for row in details for key in row)]))
    pair_keys = tuple(dict.fromkeys(["case_id", "reference_case_id", "subject", "target", "method", "arm", "reference_arm", "seed",
                                   "condition", "stage", "field_source", "delta_D_hot_mm", "delta_Cov010_pp", "decision", "reason",
                                   *(key for row in paired for key in row)]))
    write_csv(out / table, details, detail_keys)
    write_csv(out / "paired.csv", paired, pair_keys)
    write_csv(out / "cases.csv", cases)
    commands = out / "commands.ps1"
    commands.write_text(f"conda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp{experiment} --study-root '{study}' --run-id {run_id} --baseline-id {baselines} --stage {stage}\n", encoding="utf-8")
    report = out / "report.md"
    done = sum(case["status"] == "ok" for case in cases)
    failed = sum(case["status"] == "failed" for case in cases)
    pending = len(cases) - done - failed
    run_status = "complete" if done == len(cases) else "partial" if done else "blocked"
    decision = "仅筛查" if done else "无法判定"
    report.write_text(f"# {experiment} 同池评分\n\n{stage} 阶段，共同 FEM 复算后的固定库有效 {done} 格，失败 {failed} 格，待上游 {pending} 格。结论为{decision}。"
                      "两路线仅贡献具名候选；独立等预算闭环和 sub3 保留验证待开发门槛确认。\n\n"
                      f"[评分明细]({table}) · [成对差值](paired.csv) · [案例状态](cases.csv)\n", encoding="utf-8")
    write_csv(out / "artifacts.csv", artifacts + [artifact(study, "run", "output", path) for path in (out / table, out / "cases.csv", out / "paired.csv", report, commands)])
    write_json(out / "run.json", dict(experiment=experiment, run_id=run_id, stage=stage,
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), environment=sys.executable,
               upstream_runs=[f"03/{baselines}"], data_id=data_ids, config=config, started_at=started, finished_at=timestamp(), status=run_status))
    update_index(study, dict(experiment=experiment, run_id=run_id, stage=stage, run_status=run_status, scientific_decision=decision,
                 ok_cases=done, failed_cases=failed, pending_cases=pending, report_path=report.relative_to(study).as_posix()))
