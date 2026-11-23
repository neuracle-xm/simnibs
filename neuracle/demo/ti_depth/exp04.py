"""对共同正向库执行方案 04 的目标函数消融。"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from neuracle.ti_depth_study.location_objective import centroid_target_distances, location_terms
from neuracle.ti_depth_study.figures import plot_roi_slices
from neuracle.ti_depth_study.metrics import BoolArray, Geometry, distance_to_target
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


ARMS = {"shortfall": (0.0, 0.0), "shortfall_distance": (1.0, 0.0), "full": (1.0, 1.0)}


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """对相同候选场输出原 ROC 和三个定位评分实验臂。

    Parameters
    ----------
    geometry : Geometry
        含逐单元体积、靶区距离原始几何与皮层代理区的参考网格。
    target : ndarray of bool
        当前解剖靶区。
    candidates : list of Candidate
        同一物理候选池中的有效共同 FEM 正向场。

    Returns
    -------
    list of dict
        每个候选的四臂损失和三项分量，并保存单因素尺度敏感性。
    """
    if target.all():
        raise ValueError("靶区覆盖整个共同评价域，无法核对靶外距离")
    distance = centroid_target_distances(geometry, target)
    example = np.flatnonzero(~target)[0]
    manual = distance_to_target(geometry.centroid_mm[example], geometry, target)
    if not np.all(distance[target] == 0) or not np.isclose(distance[example], manual, atol=1e-4):
        raise ValueError("靶内零距离或首个靶外元素的边界三角面距离复核失败")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        common = dict(candidate_id=candidate.identifier, roc=candidate.roc, candidate_field_path=str(candidate.path))
        rows.append(common | dict(arm="roc0", J=candidate.roc, returned_loss=candidate.roc))
        with np.load(candidate.path) as data:
            field = data["max_TI"]
        arguments = (field, geometry.volume_mm3, target, geometry.cortex_mask, distance)
        terms = location_terms(*arguments)
        head_cap = min(100.0, float(geometry.volume_mm3.sum()))
        cortex_cap = min(100.0, float(geometry.volume_mm3[geometry.cortex_mask & ~target].sum()))
        if not np.isclose(terms["hot_tail_mm3"], head_cap) or not np.isclose(terms["cortex_tail_mm3"], cortex_cap):
            raise ValueError(f"候选 {candidate.identifier} 的强场截断体积复核失败")
        sensitivity = {}
        for label, changes in (("h50", {"tail_mm3": 50.0}), ("h200", {"tail_mm3": 200.0}),
                               ("d5", {"distance_scale": 5.0}), ("d20", {"distance_scale": 20.0}),
                               ("t020", {"threshold": 0.20})):
            alternate = location_terms(*arguments, **changes)
            sensitivity[label] = alternate
        for arm, (lambda_distance, mu_cortex) in ARMS.items():
            result = terms["A"] + lambda_distance * terms["D"] + mu_cortex * terms["P"]
            extra = {f"J_{label}": item["A"] + lambda_distance * item["D"] + mu_cortex * item["P"] for label, item in sensitivity.items()}
            rows.append(common | terms | extra | dict(arm=arm, J=result, returned_loss=result / (1 + result),
                        distance_check_element=int(geometry.element_id[example]), distance_tree_mm=distance[example], distance_direct_mm=manual))
    return rows


def plot_ablations(study: Path, out: Path) -> None:
    """把入选和参照场在同一 T1、切面和色标上比较。

    Parameters
    ----------
    study, out : Path
        研究根和含候选消融读数的运行目录。
    """
    pairs = [row for row in read_csv(out / "paired.csv") if row["delta_D_hot_mm"] != "NA"]
    if not pairs:
        return
    details = {(row["case_id"], row["candidate_id"]): row for row in read_csv(out / "term_ablation.csv")}
    by_change = sorted(pairs, key=lambda row: (float(row["delta_D_hot_mm"]), row["case_id"], row["arm"]))
    selected = [("hotspot_shift", by_change[0])]
    if by_change[-1]["delta_D_hot_mm"] != by_change[0]["delta_D_hot_mm"]:
        selected.append(("hotspot_shift_largest_regression", by_change[-1]))
    manifest = out / "artifacts.csv"
    artifacts = read_csv(manifest)
    for name, pairing in selected:
        case_id, arm = pairing["case_id"], pairing["arm"]
        case_key, candidate_id = case_id.split(":", 1)
        reference_id = pairing["reference_case_id"].split(":", 1)[1]
        fields = {}
        hotspots = {}
        for label, identifier in (("ROC0", reference_id), (arm, candidate_id)):
            detail = details[(case_key, identifier)]
            source = Path(detail["candidate_field_path"])
            meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            if "source" in meta:
                niftis = sorted(Path(meta["source"]).glob("*_max_TI.nii.gz"))
                if len(niftis) == 1:
                    caption = f"{label}, D_hot {float(detail['D_hot_mm']):.2f} mm"
                    fields[caption] = niftis[0]
                    hotspots[caption] = json.loads(detail["hot_center_mm"])
        if len(fields) == 2:
            subject_dir = study / "head_models" / f"m2m_{pairing['subject']}"
            roi = source.parent.parent / "roi" / f"{pairing['target']}.nii.gz"
            figure = out / "figures" / f"{name}.png"
            plot_roi_slices(subject_dir / "T1.nii.gz", roi, figure, fields, hotspots)
            artifacts.append(artifact(study, case_id, "field_overlay", figure))
    write_csv(manifest, artifacts)


def main() -> None:
    """读取实验目录和阶段，执行 04 的同池筛查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    config = {"thresholds_vpm": [0.10, 0.20], "tail_mm3": 100, "distance_scale_mm": 10, "weights": ARMS,
              "sensitivity_tail_mm3": [50, 200], "sensitivity_distance_scale_mm": [5, 20]}
    study = args.study_root.resolve()
    run_pool_experiment(study, "04", args.run_id, args.baseline_id, args.stage, score_candidates,
                        "term_ablation.csv", ("A", "D", "P", "J", "H_C", "m_T"), config)
    plot_ablations(study, study / "results/04" / args.run_id)


if __name__ == "__main__":
    main()
