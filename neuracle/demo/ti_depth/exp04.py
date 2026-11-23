"""对共同正向库执行方案 04 的目标函数消融。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from neuracle.ti_depth_study.location_objective import centroid_target_distances, location_terms
from neuracle.ti_depth_study.metrics import BoolArray, Geometry, distance_to_target
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment


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
        common = dict(candidate_id=candidate.identifier, roc=candidate.roc)
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
    run_pool_experiment(args.study_root.resolve(), "04", args.run_id, args.baseline_id, args.stage, score_candidates,
                        "term_ablation.csv", ("A", "D", "P", "J", "H_C", "m_T"), config)


if __name__ == "__main__":
    main()
