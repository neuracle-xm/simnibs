"""登记共同实验矩阵和执行先导指标核验。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from neuracle.atlas.loader import get_atlas_spec
from neuracle.atlas.registry import load_atlas_registry
from neuracle.ti_depth_study import metrics, records
from neuracle.ti_depth_study.metrics import Geometry, measure_field
from neuracle.ti_depth_study.records import artifact, read_csv, timestamp, update_index, write_csv, write_json


BN = "BN_Atlas_246_1mm"
JULICH = "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
TARGETS = (
    ("a9d16", BN, "16", "A9/46d_R", "cortical"),
    ("a4ul58", BN, "58", "A4ul_R", "cortical"),
    ("opc204", BN, "204", "OPC_R", "cortical"),
    ("a24rv178", BN, "178", "A24rv_R", "medial_cortical"),
    ("lamyg214", BN, "214", "lAmyg_R", "subcortical"),
    ("rhipp216", BN, "216", "rHipp_R", "subcortical"),
    ("gp222", BN, "222", "GP_R", "subcortical"),
    ("dca228", BN, "228", "dCa_R", "subcortical"),
    ("stha236", BN, "236", "Stha_R", "subcortical"),
    ("stn295", JULICH, "295", "STN (Subthalamus)_rh", "subcortical"),
)
PILOT_TARGETS = ("stn295", "rhipp216", "a4ul58")
METHODS = ("leadfield_based", "leadfield_free")


def register_targets() -> list[dict[str, str]]:
    """与本地标准化 atlas 的整数标签和英文名逐项核对。

    Returns
    -------
    list of dict
        含本轮靶键、解剖组别和标准化 mask 路径的登记行。
    """
    registry = load_atlas_registry()
    specs = {key: get_atlas_spec(key, registry) for key in (BN, JULICH)}
    targets = []
    for key, atlas, label, name, group in TARGETS:
        matches = [area for area in specs[atlas]["areas"] if str(area["index"]) == label]
        if len(matches) != 1 or matches[0]["label_en"].casefold() != name.casefold():
            raise ValueError(f"图谱标签不匹配: {atlas} {label} {name}")
        mask = Path(matches[0]["roi_path"])
        if not mask.is_file():
            raise FileNotFoundError(mask)
        targets.append(dict(target=key, atlas_name=atlas, label=label, group=group, label_en=matches[0]["label_en"], mask_path=str(mask)))
    return targets


def check_metrics() -> list[dict[str, object]]:
    """复算协议登记的两单元先导算例。

    Returns
    -------
    list of dict
        在体积比和热点绝对定位上的人工期望与实际读数。
    """
    nodes = np.array([[0, 0, 0], [6, 0, 0], [0, 1, 0], [0, 0, 1],
                      [10, 0, 0], [28, 0, 0], [10, 1, 0], [10, 0, 1]], dtype=float)
    cells = np.arange(8).reshape(2, 4)
    volume = np.array([1., 3.])
    geometry = Geometry(np.array([1, 2]), nodes, cells, volume, nodes[cells].mean(axis=1), np.array([False, False]))
    field_a = np.array([[0.05, 0, 0], [0.01, 0, 0]])
    result = measure_field(geometry, np.array([True, True]), field_a, field_a, hotspot_mm3=1.0)
    return [
        dict(check=check, expected=expected, observed=result[check], status="ok" if result[check] == expected else "failed", evidence=evidence)
        for check, expected, evidence in (
            ("Cov010", 0.25, "单元 1/2 体积分别为 1/3 mm³，分子 1 mm³，分母 4 mm³"),
            ("D_hot_mm", 0.0, "主热点为单元 1，其体积质心 (1.5,0.25,0.25) 在目标内"),
        )
    ]


def run(study: Path, run_id: str, audit_id: str) -> None:
    """建立十靶、三头模、两路线的登记，并写入本地交接表。

    Parameters
    ----------
    study : Path
        含历史审计输入的研究根目录。
    run_id, audit_id : str
        本次 02 与上游 01 的运行编号。
    """
    started = timestamp()
    audit = study / "results/01" / audit_id
    if json.loads((audit / "run.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise ValueError("历史审计尚未完成")
    source_rows = [row for row in read_csv(audit / "mri_files.csv") if row["role"] == "source"]
    sources = {row["subject"]: row for row in source_rows}
    if len(source_rows) != 3 or set(sources) != {"sub1", "sub2", "sub3"} or len({row["voxel_sha256"] for row in source_rows}) != 3:
        raise ValueError("历史审计未登记三个独立的原始 T1 体素载荷")
    targets = register_targets()
    matrix = [
        dict(case_id=f"{subject}_{item['target']}_{method}_native", subject=subject, target=item["target"],
             group=item["group"], method=method, arm="native_roc", split="holdout" if subject == "sub3" else "development",
             planned_stage="validation" if subject == "sub3" else "pilot" if subject == "sub1" and item["target"] in PILOT_TARGETS else "development",
             reference_arm="NA", raw_T1_sha256=sources[subject]["sha256"])
        for subject in sorted(sources) for item in targets for method in METHODS
    ]
    config = {
        "roc0_current_mA": [1.0, 1.0], "seeds": [101, 202, 303], "thresholds_vpm": [0.10, 0.20],
        "hotspot_mm3": 100, "hotspot_sensitivity_mm3": [50, 200], "montage": "EEG10-10_NSN",
        "min_improved_eligible_fraction": "2/3", "eligible_D_hot_mm": 2.0,
        "min_D_hot_improvement_mm": 2.0, "max_D_hot_regression_mm": 2.0,
        "max_Cov010_regression_pp": 5.0, "min_Q50_ratio": 0.95, "max_Q99_Cout_ratio": 1.05,
        "max_Vout010_ratio": 1.05, "baseline_zero_outside_requires_zero": True,
        "baseline_positive_coverage_requires_positive": True, "cortex_absolute_hit_requires_coverage": True,
    }
    out = study / "results/02" / run_id
    out.mkdir(parents=True, exist_ok=True)
    checks = check_metrics()
    prior_checks = out / "metric_checks.csv"
    prior_real = next((row for row in read_csv(prior_checks) if row["check"] == "common_forward"), None) if prior_checks.exists() else None
    checks.append(prior_real or dict(check="common_forward", expected="独立比对体积和坐标", observed="NA", status="pending", evidence="等待 03 首个共同正向场"))
    cases = [dict(case_id=row["check"], subject="NA", target="NA", method="NA", arm="NA", seed="NA",
                  condition="metric_check", task_id="NA", status=row["status"], reason=row["evidence"], wall_s="NA") for row in checks]
    write_csv(out / "targets.csv", targets)
    write_csv(out / "matrix.csv", matrix)
    write_csv(out / "metric_checks.csv", checks)
    write_csv(out / "cases.csv", cases)
    (out / "commands.ps1").write_text(f'conda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp02 --study-root "{study}" --run-id {run_id} --audit-id {audit_id}\n', encoding="utf-8")
    report = out / "report.md"
    report.write_text(
        "# 02 共同实验协议登记\n\n十个靶已逐个核对 atlas 整数标签及标准化 ROI。60 格原生矩阵由三个原始 T1、十靶和两条路线组成。"
        "sub1/sub2 的 40 格用于开发，sub3 的 20 格留作验证。sub1 的 STN、rHipp、A4ul 共六格为 03 先导输入。\n\n"
        f"两单元算例的覆盖为 {checks[0]['observed']}，主热点距靶为 {checks[1]['observed']} mm。"
        f"真实共同场复核状态为 {checks[2]['status']}。研究配置与通过门槛见 run.json。当前没有可配对的新旧场，paired.csv 不适用。\n\n"
        "[靶点](targets.csv) · [矩阵](matrix.csv) · [指标复核](metric_checks.csv)\n", encoding="utf-8",
    )
    inputs = [artifact(study, "protocol", "audit", audit / name) for name in ("inventory.csv", "mri_files.csv")]
    masks = [artifact(study, row["target"], "atlas_roi", Path(row["mask_path"])) for row in targets]
    scripts = [artifact(study, "protocol", "implementation", Path(source)) for source in (__file__, metrics.__file__, records.__file__)]
    files = [artifact(study, "protocol", "output", out / name) for name in ("targets.csv", "matrix.csv", "metric_checks.csv", "cases.csv", "commands.ps1", "report.md")]
    write_csv(out / "artifacts.csv", inputs + masks + scripts + files)
    status = "partial" if any(row["status"] == "failed" for row in checks) else "complete"
    write_json(out / "run.json", dict(experiment="02", run_id=run_id, stage="pilot",
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), environment=sys.executable,
               upstream_runs=[f"01/{audit_id}"], data_id="NA", config=config, started_at=started, finished_at=timestamp(), status=status))
    update_index(study, dict(experiment="02", run_id=run_id, stage="pilot", run_status=status, scientific_decision="准备工作",
                 ok_cases=sum(row["status"] == "ok" for row in checks), failed_cases=sum(row["status"] == "failed" for row in checks),
                 pending_cases=sum(row["status"] == "pending" for row in checks), report_path=report.relative_to(study).as_posix()))


def main() -> None:
    """解析运行编号和研究根，从仓库根调用实验 02。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--audit-id", default="run-001")
    args = parser.parse_args()
    run(args.study_root.resolve(), args.run_id, args.audit_id)


if __name__ == "__main__":
    main()
