"""检查 TI 硬阈值并列，并用两档平滑阈值重排同一批场。"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


WIDTHS = {"soft_005": 0.005, "soft_020": 0.020}


def soft_roc(field: FloatArray, target: BoolArray, threshold: float, width: float) -> tuple[float, float, float]:
    """保留计数口径，仅以 sigmoid 分别替换靶内和靶外的硬阈值。

    Parameters
    ----------
    field, target : ndarray
        同序全脑场和靶掩膜。
    threshold, width : float
        共同阈值及软化尺度，单位 V/m。

    Returns
    -------
    distance, sensitivity, false_positive : tuple of float
        ROC 距离及平滑的计数平均通过率。
    """
    if width <= 0 or not target.any() or target.all():
        raise ValueError("软 ROC 需要正宽度与非空靶内外区域")
    softened = expit((field - threshold) / width)
    tpr, fpr = float(softened[target].mean()), float(softened[~target].mean())
    return float(np.hypot(1 - tpr, fpr)), tpr, fpr


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """用整数超阈数定硬分组，再计算两档独立的组内软名次。

    Parameters
    ----------
    geometry : Geometry
        同池参考网格。原 ROC 分组沿用单元计数。
    target : ndarray of bool
        目标及其靶外补集。
    candidates : list of Candidate
        冻结电流下的共同正向布局。

    Returns
    -------
    list of dict
        每候选三臂的软硬评分、硬覆盖及并列组统计。
    """
    del geometry
    threshold = 0.10
    if expit(0.0) != 0.5:
        raise ValueError("阈值处软指示必须为 0.5")
    scores = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            field = data["max_TI"]
        above = field >= threshold
        roi_count, rest_count = int(np.count_nonzero(above & target)), int(np.count_nonzero(above & ~target))
        common = dict(candidate_id=candidate.identifier, roi_count=roi_count, rest_count=rest_count,
                      tie_group=f"{roi_count}:{rest_count}", hard_score=candidate.roc, hard_Cov010=candidate.absolute["Cov010"])
        scores.append(common | dict(arm="roc0", returned_loss=candidate.roc))
        for name, width in WIDTHS.items():
            distance, tpr, fpr = soft_roc(field, target, threshold, width)
            scores.append(common | dict(arm=name, w_vpm=width, soft_score=distance, soft_tpr=tpr, soft_fpr=fpr, returned_loss=distance))
    size = Counter(row["tie_group"] for row in scores if row["arm"] == "roc0")
    for row in scores:
        row["tie_group_size"] = size[row["tie_group"]]
    for name in WIDTHS:
        for key, n in size.items():
            subset = sorted((row for row in scores if row["arm"] == name and row["tie_group"] == key),
                            key=lambda row: (row["soft_score"], row["candidate_id"]))
            distinct = len({row["soft_score"] for row in subset})
            for rank, row in enumerate(subset, start=1):
                row.update(soft_rank_in_tie=rank, soft_distinct_in_tie=distinct, plateau_split=n > 1 and distinct > 1)
    return scores


def plot_plateau(study: Path, output: Path) -> None:
    """将最大的实际硬分并列组按硬分、软分和硬覆盖分别绘图。

    Parameters
    ----------
    study, output : Path
        研究根及已生成 07 明细的运行目录。
    """
    rows = read_csv(output / "threshold_plateau.csv")
    hard = [row for row in rows if row["arm"] == "roc0" and int(row["tie_group_size"]) > 1]
    if not hard:
        return
    largest = max(hard, key=lambda row: int(row["tie_group_size"]))
    subset = [row for row in rows if (row["case_id"], row["tie_group"]) == (largest["case_id"], largest["tie_group"])]
    candidates = sorted(row["candidate_id"] for row in subset if row["arm"] == "roc0")
    order = {identifier: index for index, identifier in enumerate(candidates)}
    fig = Figure(figsize=(10, 3.6), layout="constrained")
    FigureCanvasAgg(fig)
    hard_ax, soft_ax, cov_ax = fig.subplots(1, 3, sharex=True)
    baseline = [row for row in subset if row["arm"] == "roc0"]
    idx = [order[row["candidate_id"]] for row in baseline]
    hard_ax.scatter(idx, [float(row["hard_score"]) for row in baseline], s=12)
    cov_ax.scatter(idx, [float(row["hard_Cov010"]) for row in baseline], s=12)
    for name, width in WIDTHS.items():
        smoothed = [row for row in subset if row["arm"] == name]
        soft_ax.scatter([order[row["candidate_id"]] for row in smoothed], [float(row["soft_score"]) for row in smoothed], s=12, label=f"{width:.3f} V/m")
    hard_ax.set(ylabel="Hard ROC distance", xlabel="Candidate index")
    soft_ax.set(ylabel="Soft ROC distance", xlabel="Candidate index")
    cov_ax.set(ylabel="Volume coverage at 0.10 V/m", xlabel="Candidate index", ylim=(-0.03, 1.03))
    soft_ax.legend(fontsize=8)
    fig.suptitle(f"{largest['case_id']} | Equal integer counts {largest['tie_group']}")
    path = output / "figures/plateau.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    table = output / "artifacts.csv"
    write_csv(table, [*read_csv(table), artifact(study, largest["case_id"], "plateau_plot", path)])


def main() -> None:
    """从仓库根启动 07 的硬软评分同池比较。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    config = {"threshold_vpm": 0.10, "width_vpm": WIDTHS, "development_targets": PILOT_TARGETS}
    columns = ("roi_count", "rest_count", "tie_group", "w_vpm", "soft_score", "hard_score", "hard_Cov010")
    run_pool_experiment(study, "07", args.run_id, args.baseline_id, args.stage, score_candidates, "threshold_plateau.csv", columns, config)
    plot_plateau(study, study / "results/07" / args.run_id)


if __name__ == "__main__":
    main()
