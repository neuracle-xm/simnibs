"""按登记的解剖靶区核对原 ROC 与待锁定的新目标分支。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from neuracle.demo.ti_depth.exp03 import CASE_COLUMNS, head_ready
from neuracle.ti_depth_study.baseline import forward_config, inverse_config
from neuracle.ti_depth_study.diagnostics import load_case
from neuracle.ti_depth_study.metrics import measure_field
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json


ROUTING_COLUMNS = ("method", "target", "old_case_id", "new_case_id", "chosen_case_id", "rule", "chosen_source",
                   "fallback_reason", "old_hit", "new_hit", "chosen_hit", "compute_s")
ENDPOINTS = ("D_hot_mm", "Cov010", "Q50_T", "Q99_Cout", "Vout010", "hot_overlap", "main_overlap_mm3", "absolute_hit")


def route(atlas: str, label: str, registry: dict[tuple[str, str], dict[str, str]]) -> str:
    """对精确图谱版本及标签执行研究分流。

    Parameters
    ----------
    atlas, label : str
        完整图谱键和标签；未注册的键或标签走原 ROC。
    registry : dict
        02 以图谱和标签为联合键的解剖分组表。

    Returns
    -------
    str
        登记的深部候选分支、皮层原分支或尚未验证的原分支。
    """
    target = registry.get((atlas, label))
    if target is None:
        return "unregistered_original_roc"
    return "locked_deep_or_original_roc" if target["group"] == "subcortical" else "cortical_original_roc"


def original(study: Path, native: dict[str, str], planned: dict[str, str], target: dict[str, str],
             mesh_hash: str) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    """从已验收的原生布局重取实际电流和共同 FEM 的绝对终点。

    Parameters
    ----------
    study : Path
        研究根目录。
    native, planned, target : dict
        03 确认成功的行、02 病例计划及当前解剖条目。
    mesh_hash : str
        当前独立头模的已验证 SHA-256。

    Returns
    -------
    tuple
        原布局的绝对指标、实际配置及其证据路径。
    """
    if native["mesh_hash"] != mesh_hash or native["case_id"] != planned["case_id"]:
        raise ValueError("原场与当前个体头模或 02 病例不同")
    inverse = study / "simulations" / native["inverse_task"]
    forward = study / "simulations" / native["forward_task"]
    inverse_params, forward_params = inverse / "params.json", forward / "params.json"
    mapping_path = inverse / "electrode_mapping.json"
    execution = inverse / "study_execution.json"
    receipt = json.loads(execution.read_text(encoding="utf-8"))
    if receipt["exit_code"] != 0 or receipt["sha256"].get(mapping_path.name) != digest(mapping_path):
        raise ValueError("原生布局与 03 求解回执不符")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    labels = mapping["mapped_labels"]
    currents = (mapping["current_A"], mapping["current_B"])
    if labels != native["electrode_names"].split(";") or any(
            float(native[f"I{key}_mA"]) != current[0] for key, current in zip(("A", "B"), currents)):
        raise ValueError("原生布局或实际电流与 03 共同场索引不符")
    if json.loads(inverse_params.read_text(encoding="utf-8")) != inverse_config(target, native["method"]):
        raise ValueError("原 ROI、ROC 阈值或搜索电流约定已变化")
    if json.loads(forward_params.read_text(encoding="utf-8")) != forward_config(labels, currents):
        raise ValueError("正向场的帽位、物性或电流与原布局不一致")
    case = load_case(study, native)
    field_receipt = json.loads(case.field_path.with_suffix(".json").read_text(encoding="utf-8"))
    forward_execution = forward / "study_execution.json"
    forward_receipt = json.loads(forward_execution.read_text(encoding="utf-8"))
    if (forward_receipt["exit_code"] != 0 or Path(field_receipt["source"]).resolve() != forward.resolve()
            or field_receipt["inputs"]["outputs"] != forward_receipt["sha256"]):
        raise ValueError("共同场来源与正向执行回执不符")
    values = measure_field(case.geometry, case.target, case.e_a, case.e_b, case.valid)
    if values["status"] != "ok":
        raise ValueError(f"原共同正向场不可判，{values['status']}")
    evidence = [("inverse_input", inverse_params), ("inverse_execution", execution), ("mapping", mapping_path),
                ("forward_input", forward_params), ("forward_execution", forward_execution),
                ("geometry", case.geometry_path), ("field", case.field_path)]
    return {key: values[key] for key in ENDPOINTS} | dict(IA_mA=currents[0][0], IB_mA=currents[1][0],
                electrodes=";".join(labels), field_id=case.field_path.stem), evidence


def run(study: Path, run_id: str, baseline_id: str, stage: str) -> None:
    """保存逐病例分流、皮层绝对命中和深部原解待比较状态。

    Parameters
    ----------
    study, run_id, baseline_id, stage : Path and str
        研究根、13 和 03 运行号及先导或开发阶段。
    """
    protocol = study / "results/02/run-001"
    out = study / "results/13" / run_id
    out.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    targets = {row["target"]: row for row in read_csv(protocol / "targets.csv")}
    registry = {(row["atlas_name"], row["label"]): row for row in targets.values()}
    if len(registry) != len(targets) or len(targets) != 10:
        raise ValueError("需要 02 的十个唯一图谱及标签组合")
    route_rows = [dict(target=row["target"], atlas_name=atlas, label=label, group=row["group"],
                       rule=route(atlas, label, registry)) for (atlas, label), row in registry.items()]
    write_csv(out / "routing_rules.csv", route_rows)
    matrix_path = protocol / "matrix.csv"
    matrix = [item for item in read_csv(matrix_path) if item["split"] == "development"
              and (stage == "development" or item["subject"] == "sub1" and item["target"] in ("stn295", "a4ul58"))]
    upstream = study / "results/03" / baseline_id / "baseline_fields.csv"
    available = {item["case_id"]: item for item in read_csv(upstream)} if upstream.is_file() else {}
    artifacts = [artifact(study, "run", "implementation", Path(__file__)), artifact(study, "run", "matrix", matrix_path),
                 artifact(study, "run", "target_registry", protocol / "targets.csv")]
    if upstream.is_file():
        artifacts.append(artifact(study, "run", "upstream_baselines", upstream))
    cases, routing = [], []
    mesh_hashes: dict[str, str] = {}
    for planned in matrix:
        case_id, subject, target_name, method = (planned[key] for key in ("case_id", "subject", "target", "method"))
        target = targets[target_name]
        rule = route(target["atlas_name"], target["label"], registry)
        deep = rule == "locked_deep_or_original_roc"
        row = dict(case_id=case_id, subject=subject, target=target_name, method=method, arm="anatomy_routing", seed="NA",
                   condition="native_roc", task_id="NA", status="pending", reason="等待 03 原生共同场", wall_s="NA")
        routed = dict(method=method, target=target_name, subject=subject, old_case_id="NA", new_case_id="NA",
                      chosen_case_id="NA", rule=rule, chosen_source="NA", fallback_reason="NA",
                      old_hit="NA", new_hit="NA", chosen_hit="NA", compute_s="NA", old_evidence="pending")
        cases.append(row)
        routing.append(routed)
        native = available.get(case_id)
        if native is None or native["status"] != "ok":
            if native is not None:
                row.update(status=native["status"], reason=native["reason"])
            routed["fallback_reason"] = row["reason"]
            continue
        try:
            head = study / "head_models" / f"m2m_{subject}"
            if not head_ready(head, subject, planned["raw_T1_sha256"]):
                raise ValueError("原始扫描与登记的独立 CHARM 头模不一致")
            if subject not in mesh_hashes:
                mesh_hashes[subject] = digest(head / f"{subject}.msh")
                artifacts.append(artifact(study, subject, "cap", head / "eeg_positions/EEG10-10_NSN.csv"))
            metrics, evidence = original(study, native, planned, target, mesh_hashes[subject])
            artifacts.extend(artifact(study, case_id, role, path) for role, path in evidence)
            hit = metrics["absolute_hit"] and metrics["Cov010"] > 0
            routed.update(old_case_id=case_id, old_hit=hit, old_evidence="verified",
                          **{f"old_{key}": value for key, value in metrics.items()})
            row["task_id"] = native["forward_task"]
            if deep:
                row["reason"] = "已核验原 ROC，等待锁定新目标和同条件共同正向"
                routed["fallback_reason"] = row["reason"]
            else:
                row.update(status="ok", reason="NA")
                routed.update(chosen_case_id=case_id, chosen_source="original_roc", chosen_hit=hit, compute_s=0)
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            row.update(status="failed", reason=str(error))
            routed.update(fallback_reason=str(error), old_evidence="failed")
    done = sum(row["status"] == "ok" for row in cases)
    failed = sum(row["status"] == "failed" for row in cases)
    pending = len(cases) - done - failed
    deep_ready = sum(row["rule"] == "locked_deep_or_original_roc" and row["old_evidence"] == "verified" for row in routing)
    count_hits = sum(row["chosen_hit"] is True for row in routing)
    status = "partial" if done or deep_ready else "blocked"
    write_csv(out / "cases.csv", cases, CASE_COLUMNS)
    write_csv(out / "routing.csv", routing, tuple(dict.fromkeys((*ROUTING_COLUMNS, *(key for row in routing for key in row)))))
    commands = out / "commands.ps1"
    commands.write_text(f"conda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp13 --study-root '{study}' "
                        f"--run-id {run_id} --baseline-id {baseline_id} --stage {stage}\n", encoding="utf-8")
    report = out / "report.md"
    report.write_text(f"# 13 解剖分流先决核验\n\n{stage} 阶段，皮层复现 {done} 格，其中绝对命中且正覆盖 {count_hits} 格。"
                      f"已核验 {deep_ready} 格深部原场。全部病例 {len(cases)} 格，失败 {failed} 格，待完成 {pending} 格。\n\n"
                      "深部同条件新解、双 ROC 等成本对照及保留例等待 04/05 开发锁定。\n\n"
                      "[图谱规则](routing_rules.csv) · [逐病例分流](routing.csv) · [验收状态](cases.csv)\n", encoding="utf-8")
    write_csv(out / "artifacts.csv", [*artifacts, *(artifact(study, "run", "output", out / name) for name in
              ("routing_rules.csv", "routing.csv", "cases.csv", "commands.ps1", "report.md"))])
    write_json(out / "run.json", dict(experiment="13", run_id=run_id, stage=stage, status=status, environment=sys.executable,
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), data_id=mesh_hashes,
               upstream_runs=["02/run-001", f"03/{baseline_id}"], config=dict(new_objective_status="unlocked"),
               started_at=started, finished_at=timestamp()))
    update_index(study, dict(experiment="13", run_id=run_id, stage=stage, run_status=status,
                             scientific_decision="原 ROC 分流核验，深部对照待锁定", ok_cases=done, failed_cases=failed,
                             pending_cases=pending, report_path=report.relative_to(study).as_posix()))


def main() -> None:
    """从仓库根执行登记靶的分流先决核验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    run(args.study_root.resolve(), args.run_id, args.baseline_id, args.stage)


if __name__ == "__main__":
    main()
