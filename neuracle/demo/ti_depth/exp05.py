"""在固定共同 FEM 候选库内比较靶区覆盖和外溢约束。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.ti_depth_study.coverage_objective import coverage_terms, pareto_mask, spill_loss
from neuracle.ti_depth_study.metrics import BoolArray, Geometry
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv
from simnibs.optimization.tes_flex_optimization.measures import ROC


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """比较原 ROC、无约束覆盖和双外溢约束的两档预算。

    Parameters
    ----------
    geometry : Geometry
        含固定体积及皮层掩膜的参考网格。
    target : ndarray of bool
        当前个体空间目标。
    candidates : list of Candidate
        两路各 1 mA 的有效共同库场。

    Returns
    -------
    list of dict
        各臂覆盖、外溢、违约量、可行性及 Pareto 边界明细。
    """
    reference = min(candidates, key=lambda candidate: (candidate.roc, candidate.identifier))
    terms: dict[str, dict[str, float]] = {}
    sensitivity: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        with np.load(candidate.path) as data:
            field = data["max_TI"]
        args = (field, geometry.volume_mm3, target, geometry.cortex_mask)
        terms[candidate.identifier] = coverage_terms(*args)
        other = coverage_terms(*args, target_threshold=0.20)
        roc = float(ROC(e1=field[target], e2=field[~target], threshold=[0.10, 0.20], focal=True))
        sensitivity[candidate.identifier] = {"K_020": other["K"], "F_020": other["F"], "S_020": other["S"], "ROC_020": roc}
    ref = terms[reference.identifier]
    budget = (ref["W_R_mm3"], ref["W_C_mm3"])
    if not spill_loss(ref, budget, 1.0)["feasible"]:
        raise ValueError("共同 FEM 的 ROC0 首选未能重现 k=1 可行性")
    frontier = pareto_mask(list(terms.values()))
    rows: list[dict[str, Any]] = []
    for candidate, nondominated in zip(candidates, frontier):
        key = candidate.identifier
        values = terms[key]
        common = dict(candidate_id=key, reference_case_id=reference.identifier, layer="common_forward",
                      B_R_mm3=budget[0], B_C_mm3=budget[1], pareto=bool(nondominated), **values, **sensitivity[key])
        rows.append(common | dict(arm="roc0", returned_loss=candidate.roc, feasible=True))
        rows.append(common | dict(arm="coverage_only", returned_loss=0.9 * (1 - values["S"]), feasible=True))
        for arm, k in (("k100", 1.0), ("k075", 0.75)):
            rows.append(common | dict(arm=arm, budget_scale=k) | spill_loss(values, budget, k))
    for arm in ("k100", "k075"):
        subset = [row for row in rows if row["arm"] == arm]
        n_feasible = sum(row["feasible"] for row in subset)
        minimum_violation = min(row["violation"] for row in subset)
        for row in subset:
            row.update(feasible_count=n_feasible, total_candidates=len(subset), minimum_violation=minimum_violation)
    return rows


def write_figures(study: Path, out: Path) -> None:
    """绘制每个病例的覆盖、Pareto 边界和两档预算结果。

    Parameters
    ----------
    study, out : Path
        研究根和 05 运行目录，后者已写入候选评分表。
    """
    details = read_csv(out / "spill_budget.csv")
    if not details:
        return
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    artifacts = read_csv(out / "artifacts.csv")
    for index, case in enumerate(dict.fromkeys(row["case_id"] for row in details)):
        rows = [row for row in details if row["case_id"] == case and row["arm"] == "k100"]
        fig = Figure(figsize=(7, 4.5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for status, marker, name in (("True", "o", "Within both budgets"), ("False", "x", "Exceeds a budget")):
            group = [row for row in rows if row["feasible"] == status]
            if group:
                ax.scatter([float(row["W_R_mm3"]) / 1000 for row in group], [float(row["K"]) for row in group],
                           marker=marker, label=name, alpha=0.75)
        frontier = sorted((row for row in rows if row["pareto"] == "True"), key=lambda row: float(row["W_R_mm3"]))
        ax.scatter([float(row["W_R_mm3"]) / 1000 for row in frontier], [float(row["K"]) for row in frontier],
                   facecolors="none", edgecolors="#252525", s=74, label="Pareto (coverage and two spill volumes)")
        for arm, marker in (("roc0", "s"), ("k100", "*"), ("k075", "D")):
            winner = next((row for row in details if row["case_id"] == case and row["arm"] == arm and row["rank"] == "1"), None)
            if winner:
                ax.scatter(float(winner["W_R_mm3"]) / 1000, float(winner["K"]), marker=marker, s=99, label=arm)
        budget = float(rows[0]["B_R_mm3"]) / 1000
        cortex_budget = float(rows[0]["B_C_mm3"]) / 1000
        ax.axvline(budget, color="#606060", linestyle="--", label="Whole-brain outside-ROI budget")
        ax.set(xlabel="Outside-ROI volume at 0.10 V/m (mL)", ylabel="Target coverage at 0.10 V/m",
               title=f"{case} | Cortical budget {cortex_budget:.2f} mL", ylim=(-0.03, 1.03))
        ax.legend(loc="best", fontsize=7)
        output = figures / ("coverage_vs_spill.png" if index == 0 else f"{case}_coverage_vs_spill.png")
        fig.savefig(output, dpi=140)
        artifacts.append(artifact(study, case, "coverage_plot", output))
    write_csv(out / "artifacts.csv", artifacts)


def main() -> None:
    """从仓库根执行 05 的先导或开发同池筛查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    config = {"target_thresholds_vpm": [0.10, 0.20], "outside_threshold_vpm": 0.10, "budget_scales": [1.0, 0.75]}
    study = args.study_root.resolve()
    run_pool_experiment(study, "05", args.run_id, args.baseline_id, args.stage, score_candidates,
                        "spill_budget.csv", ("reference_case_id", "layer", "budget_scale", "K", "F", "S", "W_R_mm3", "W_C_mm3", "B_R_mm3", "B_C_mm3", "violation", "feasible", "pareto"), config)
    write_figures(study, study / "results/05" / args.run_id)


if __name__ == "__main__":
    main()
