"""在保留的原生具名布局和共同正向场上运行物理诊断。"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any, Callable

import numpy as np

from neuracle.ti_depth_study.baseline import FIELD_MODEL
from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json


@dataclass(frozen=True)
class FieldCase:
    """已核验的个体头模、参考 ROI 和同序共同正向载波。"""

    study: Path
    native: dict[str, str]
    geometry_path: Path
    field_path: Path
    geometry: Geometry
    target: BoolArray
    e_a: FloatArray
    e_b: FloatArray
    field: FloatArray
    valid: BoolArray


ResultRows = tuple[list[dict[str, Any]], list[dict[str, Any]]]
Diagnostic = Callable[[FieldCase, Path, list[dict[str, str]]], ResultRows]


def load_case(study: Path, row: dict[str, str]) -> FieldCase:
    """加载有实际电流和几何证据的共同正向病例。

    Parameters
    ----------
    study : Path
        研究根，基线表的场路径相对于它。
    row : dict
        03 中通过共同正向和指标验收的一行。

    Returns
    -------
    FieldCase
        已核对模型、参考网格和逐场文件内容的诊断输入。
    """
    field_path = study / row["field_path"]
    geometry_path = field_path.parent.parent / "geometry.npz"
    metadata = json.loads(field_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["inputs"]["model"] != FIELD_MODEL or metadata["inputs"]["geometry"] != digest(geometry_path) or metadata["sha256"] != digest(field_path):
        raise ValueError("共同 FEM 场或参考网格的来源、模型、散列不符")
    geometry, _, targets = load_geometry(geometry_path)
    with np.load(field_path) as data:
        return FieldCase(study, row, geometry_path, field_path, geometry, targets[row["target"]],
                         data["E_A"], data["E_B"], data["max_TI"], data["valid_mask"])


def write_diagnostic_state(
    study: Path, out: Path, info: dict[str, Any], cases: list[dict[str, Any]], rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]], table: str, columns: tuple[str, ...], artifacts: list[dict[str, str]],
) -> None:
    """按共同协议保存已实际执行的病例、指标及待完成状态。

    Parameters
    ----------
    study, out : Path
        研究根与当前运行目录。
    info, cases, rows, pairs : dict and lists of dict
        运行清单、案例分母、当前绝对读数和逐配置差值。
    table, columns : str and tuple
        本篇专属 CSV 名称及必要字段。
    artifacts : list of dict
        已核验散列的脚本、输入或额外诊断产物。
    """
    cases_path = out / "cases.csv"
    write_csv(cases_path, cases)
    result_path = out / table
    keys = tuple(dict.fromkeys(["case_id", "subject", "target", "method", *columns, *(key for row in rows for key in row)]))
    write_csv(result_path, rows, keys)
    pair_keys = tuple(dict.fromkeys(["case_id", "reference_case_id", "subject", "target", "method", "arm", "reference_arm",
                                    "field_source", "stage", *(key for row in pairs for key in row)]))
    write_csv(out / "paired.csv", pairs, pair_keys)
    good = sum(row["status"] == "ok" for row in cases)
    failed = sum(row["status"] == "failed" for row in cases)
    pending = len(cases) - good - failed
    methods = {method: [row for row in cases if row["method"] == method] for method in dict.fromkeys(row["method"] for row in cases)}
    summary = "\n".join(f"- {method} {sum(row['status'] == 'ok' for row in subset)}/{len(subset)} 格有诊断读数。" for method, subset in methods.items())
    decision = "固定布局诊断" if good else "无法判定"
    report = out / "report.md"
    report.write_text(f"# {info['experiment']} 共同正向诊断\n\n{info['stage']} 阶段，状态 {info['status']}，完成 {good} 格，失败 {failed} 格，待完成 {pending} 格。\n\n"
                      f"{summary}\n\n本次比较保留 03 具名布局，本次结果对应 {decision}。\n\n"
                      f"[诊断明细]({table}) · [配对差值](paired.csv) · [案例](cases.csv)\n", encoding="utf-8")
    panels = (cases_path, result_path, out / "paired.csv", report, out / "commands.ps1")
    write_csv(out / "artifacts.csv", [*artifacts, *(artifact(study, "run", "output", path) for path in panels)])
    write_json(out / "run.json", info)
    update_index(study, dict(experiment=info["experiment"], run_id=info["run_id"], stage=info["stage"], run_status=info["status"],
                             scientific_decision=decision, ok_cases=good, failed_cases=failed, pending_cases=pending,
                             report_path=report.relative_to(study).as_posix()))


def run_diagnostic(
    study: Path, experiment: str, run_id: str, baseline_id: str, stage: str, targets: tuple[str, ...],
    process: Diagnostic, table: str, columns: tuple[str, ...], config: dict[str, Any],
) -> None:
    """逐头模靶点和路线重放冻结布局上的诊断，并持续写入进度。

    Parameters
    ----------
    study : Path
        研究根目录。
    experiment, run_id, baseline_id, stage : str
        实验与运行编号，03 上游运行号及开发阶段。
    targets : tuple of str
        当前阶段预登记的靶键。
    process : callable
        接收一格的固定输入、输出路径和产物登记，返回该格的绝对明细与配对。
    table, columns, config : str, tuple and dict
        执行单专属表以及当前配置。
    """
    up = study / "results/03" / baseline_id
    out = study / "results" / experiment / run_id
    out.mkdir(parents=True, exist_ok=True)
    commands = out / "commands.ps1"
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\n"
                     f"python -m neuracle.demo.ti_depth.exp{experiment} --study-root '{study}' --run-id {run_id} --baseline-id {baseline_id} --stage {stage}\n")
    native_path = up / "baseline_fields.csv"
    source_rows = read_csv(native_path) if native_path.is_file() else []
    by_key = {(row["subject"], row["target"], row["method"]): row for row in source_rows}
    subjects = ("sub1",) if stage == "pilot" else ("sub1", "sub2")
    planned = [dict(case_id=f"{subject}_{target}_{method}", subject=subject, target=target, method=method, arm="fixed_layout",
                    seed="NA", condition="diagnostic", task_id="NA", status="pending", reason="等待 03 共同正向场", wall_s="NA")
               for subject in subjects for target in targets for method in ("leadfield_based", "leadfield_free")]
    info = dict(experiment=experiment, run_id=run_id, stage=stage, environment=sys.executable,
                upstream_runs=[f"03/{baseline_id}"], data_id={}, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                config=config, started_at=timestamp(), finished_at=None, status="running")
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    artifacts = [artifact(study, "run", "implementation", Path(__file__))]
    if native_path.is_file():
        artifacts.append(artifact(study, "run", "upstream_baselines", native_path))
    write_diagnostic_state(study, out, info, planned, rows, pairs, table, columns, artifacts)
    for row in planned:
        key = (row["subject"], row["target"], row["method"])
        baseline = by_key.get(key)
        if baseline is None or baseline["status"] != "ok":
            if baseline is not None:
                row["reason"] = baseline.get("reason", row["reason"])
            continue
        started = monotonic()
        try:
            case = load_case(study, baseline)
            artifacts.extend(artifact(study, row["case_id"], role, path) for role, path in (("geometry", case.geometry_path), ("field", case.field_path)))
            info["data_id"][row["subject"]] = case.geometry_path.parent.name
            measured, compared = process(case, out, artifacts)
            identity = {key: row[key] for key in ("case_id", "subject", "target", "method")}
            rows.extend(identity | item for item in measured)
            pairs.extend(identity | item | {"stage": stage} for item in compared)
            row.update(status="ok", reason="NA", task_id=baseline["forward_task"])
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            row.update(status="failed", reason=str(error))
        row["wall_s"] = round(monotonic() - started, 3)
        write_diagnostic_state(study, out, info, planned, rows, pairs, table, columns, artifacts)
    good = sum(row["status"] == "ok" for row in planned)
    info.update(status="complete" if good == len(planned) else "partial" if good else "blocked", finished_at=timestamp())
    write_diagnostic_state(study, out, info, planned, rows, pairs, table, columns, artifacts)
