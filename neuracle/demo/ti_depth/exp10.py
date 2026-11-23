"""比较 TI 体积 ROC 的单阈值与三阈值平均排序。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry, coverage
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


THRESHOLDS = (0.10, 0.15, 0.20)


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """逐布局计算三档覆盖，再比较主阈值和三档平均的选择。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        共同四面体体积及同序解剖目标。
    candidates : list of Candidate
        固定 1+1 mA 的有效场。

    Returns
    -------
    list of dict
        各候选两臂的损失、三档读数及与其发生严格覆盖交叉的布局。
    """
    curves: list[FloatArray] = []
    losses: list[FloatArray] = []
    false_positive = []
    volume = geometry.volume_mm3
    for candidate in candidates:
        with np.load(candidate.path) as data:
            field = data["max_TI"]
        inside = field[target]
        curve = np.array([coverage(inside, volume[target], threshold) for threshold in THRESHOLDS])
        if np.any(np.diff(curve) > 0):
            raise ValueError(f"候选 {candidate.identifier} 的覆盖曲线非单调")
        outside = coverage(field[~target], volume[~target], 0.10)
        curves.append(curve)
        losses.append(np.hypot(1 - curve, outside))
        false_positive.append(outside)
    matrix = np.stack(curves)
    rows = []
    for index, candidate in enumerate(candidates):
        delta = matrix - matrix[index]
        crossed = np.any(delta > 0, axis=1) & np.any(delta < 0, axis=1)
        crossings = ";".join(other.identifier for other, cross in zip(candidates, crossed) if cross)
        loss = losses[index]
        common = dict(candidate_id=candidate.identifier, coverage_curve=curves[index].tolist(), distance_curve=loss.tolist(),
                      mean_distance=float(loss.mean()), outside_coverage=false_positive[index], crosses_candidates=crossings or "NA")
        rows.append(common | dict(arm="threshold_010", returned_loss=float(loss[0])))
        rows.append(common | dict(arm="mean_thresholds", returned_loss=float(loss.mean())))
    return rows


def threshold_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并两臂名次并按阈值展开逐候选明细。

    Parameters
    ----------
    rows : list of dict
        由共同筛查器以两种损失分别排名的行。

    Returns
    -------
    list of dict
        每个病例、候选及阈值一行。
    """
    mean_ranks = {(row["case_id"], row["candidate_id"]): row["rank"] for row in rows if row["arm"] == "mean_thresholds"}
    table = []
    for row in rows:
        if row["arm"] == "threshold_010":
            clean = {key: value for key, value in row.items() if key not in {"arm", "rank", "returned_loss", "coverage_curve", "distance_curve"}}
            for threshold, cov, distance in zip(THRESHOLDS, row["coverage_curve"], row["distance_curve"]):
                table.append(clean | dict(t_roi_vpm=threshold, t_rest_vpm=0.10, Cov=cov, ROC_distance=distance,
                                         rank_at_010=row["rank"], rank_mean=mean_ranks[row["case_id"], row["candidate_id"]]))
    return table


def plot_thresholds(study: Path, out: Path) -> None:
    """把每个候选覆盖的三点连线，并突出两个独立首选。

    Parameters
    ----------
    study, out : Path
        研究根及 10 的运行目录。
    """
    rows = read_csv(out / "threshold_curves.csv")
    if not rows:
        return
    case = rows[0]["case_id"]
    group = [row for row in rows if row["case_id"] == case]
    fig = Figure(figsize=(7, 4.6), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for name in dict.fromkeys(row["candidate_id"] for row in group):
        points = sorted((row for row in group if row["candidate_id"] == name), key=lambda row: float(row["t_roi_vpm"]))
        selected_a = points[0]["rank_at_010"] == "1"
        selected_b = points[0]["rank_mean"] == "1"
        label = "Both criteria" if selected_a and selected_b else "0.10 V/m ROC" if selected_a else "Three-threshold mean" if selected_b else None
        opts = dict(marker="o", linewidth=2.3, label=label, zorder=4) if label else dict(color="#8390a0", alpha=0.25, linewidth=0.8)
        ax.plot([float(row["t_roi_vpm"]) for row in points], [float(row["Cov"]) for row in points], **opts)
    ax.axvline(0.10, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Target threshold (V/m)", ylabel="Volume coverage", title=case, xticks=THRESHOLDS, ylim=(-0.03, 1.03))
    ax.legend(loc="best", fontsize=8)
    path = out / "figures/coverage_curve.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=145)
    manifest = out / "artifacts.csv"
    write_csv(manifest, [*read_csv(manifest), artifact(study, case, "threshold_plot", path)])


def main() -> None:
    """从仓库根启动 10 的同场跨阈值筛查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    config = {"target_thresholds_vpm": THRESHOLDS, "rest_threshold_vpm": 0.10,
              "development_targets": PILOT_TARGETS, "comparison_arm": "threshold_010"}
    columns = ("t_roi_vpm", "t_rest_vpm", "Cov", "ROC_distance", "mean_distance", "rank_at_010", "rank_mean")
    run_pool_experiment(study, "10", args.run_id, args.baseline_id, args.stage, score_candidates,
                        "threshold_curves.csv", columns, config, threshold_table)
    plot_thresholds(study, study / "results/10" / args.run_id)


if __name__ == "__main__":
    main()
