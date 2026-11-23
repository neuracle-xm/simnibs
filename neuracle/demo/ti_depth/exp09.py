"""在共同场上量化近靶壳层外溢及其迁移。"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.location_objective import centroid_target_distances
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


ARMS = {"volume_base": 0.0, "shell025": 0.25, "shell100": 1.0}
SHELLS = ("target", "near_5", "mid_5_15", "far_15")


def shell_masks(target: BoolArray, distance: FloatArray) -> dict[str, BoolArray]:
    """用目标实体的距离划分脑域，保留靶外零距离单元。

    Parameters
    ----------
    target, distance : ndarray
        同序目标掩膜及至目标实体距离，单位 mm。

    Returns
    -------
    dict
        靶内、近壳层、远壳层和剩余域四个互不重叠的布尔掩膜。
    """
    near = ~target & (distance <= 5)
    mid = ~target & (distance > 5) & (distance <= 15)
    far = ~target & (distance > 15)
    partition = dict(zip(SHELLS, (target, near, mid, far)))
    if not np.all(np.count_nonzero(list(partition.values()), axis=0) == 1):
        raise ValueError("目标及壳层必须恰好分割共同脑域")
    return partition


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """用共同体积 ROC 和近壳层覆盖惩罚分别排序。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        同一次配准的参考脑域、体积与靶区。
    candidates : list of Candidate
        固定电流和正向模型下生成的候选。

    Returns
    -------
    list of dict
        各实验臂的损失及四个空间分区绝对读数。
    """
    masks = shell_masks(target, centroid_target_distances(geometry, target))
    volume = geometry.volume_mm3
    domain = {key: float(volume[mask].sum()) for key, mask in masks.items()}
    if not np.isclose(sum(domain.values()), volume.sum()):
        raise ValueError("壳层体积之和须等于评估脑域体积")
    rows = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            above = data["max_TI"] >= 0.10
        covered = {key: float(volume[mask & above].sum()) for key, mask in masks.items()}
        rates = {key: covered[key] / size if size else "NA" for key, size in domain.items()}
        fpr = sum(covered[key] for key in SHELLS[1:]) / sum(domain[key] for key in SHELLS[1:])
        base = float(np.hypot(1 - rates["target"], fpr))
        for arm, weight in ARMS.items():
            applicable = bool(domain["near_5"] or not weight)
            loss = base + weight * float(rates["near_5"]) if weight and applicable else base
            rows.append(dict(candidate_id=candidate.identifier, arm=arm, returned_loss=loss, feasible=applicable,
                             domains=domain, above=covered, rates=rates, base_weighted_roc=base,
                             hot_center_mm=candidate.absolute["hot_center_mm"], **{"lambda": weight}))
    return rows


def shell_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """展开每组臂的各分区，给空层保留零体积和 NA 比例。

    Parameters
    ----------
    rows : list of dict
        已排好各实验臂名次的候选记录。

    Returns
    -------
    list of dict
        每个候选、实验臂和分区各一行。
    """
    return [{key: value for key, value in row.items() if key not in {"domains", "above", "rates"}}
            | dict(shell=shell, domain_mm3=row["domains"][shell], above_mm3=row["above"][shell], Cov010=row["rates"][shell])
            for row in rows for shell in SHELLS]


def plot_shells(study: Path, out: Path, baseline_id: str) -> None:
    """在通过靶区的个体轴向切面绘制壳层等值线及各臂主热点投影。

    Parameters
    ----------
    study, out : Path
        本地研究根与 09 的运行目录。
    baseline_id : str
        供绘图读取共同几何的 03 运行号。
    """
    scores = read_csv(out / "shell_spill.csv")
    winners = [row for row in scores if row["shell"] == "near_5" and row["rank"] == "1"]
    if not winners:
        return
    first = winners[0]
    item = next(row for row in read_csv(study / "results/03" / baseline_id / "shared_inputs.csv")
                if row["subject"] == first["subject"] and row["role"] == "geometry")
    geometry, _, targets = load_geometry(study / item["path"])
    target = targets[first["target"]]
    distance = centroid_target_distances(geometry, target)
    points = geometry.centroid_mm
    roi = points[target]
    zz = float(np.average(roi[:, 2], weights=geometry.volume_mm3[target]))
    x, y = np.meshgrid(np.linspace(roi[:, 0].min() - 20, roi[:, 0].max() + 20, 200),
                       np.linspace(roi[:, 1].min() - 20, roi[:, 1].max() + 20, 200))
    grid = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, zz)))
    nearest_dist, nearest = cKDTree(points).query(grid)
    sampled = np.ma.masked_where(nearest_dist.reshape(x.shape) > 4, distance[nearest].reshape(x.shape))
    figure = Figure(figsize=(7, 5.4), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    if sampled.count() and sampled.max() >= 5:
        contours = ax.contour(x, y, sampled, levels=[5, 15], colors=["#ab562d", "#725a94"])
        ax.clabel(contours, fmt="%g mm")
    roi_grid = target[nearest].reshape(x.shape)
    if roi_grid.any() and not roi_grid.all():
        ax.contour(x, y, roi_grid, levels=[0.5], colors=["#007c70"])
    for row in winners:
        if row["case_id"] == first["case_id"]:
            center = json.loads(row["hot_center_mm"])
            ax.scatter(*center[:2], label=row["arm"], s=35, clip_on=False)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title=f"{first['case_id']} | axial slice at Z={zz:.1f} mm")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    output = out / "figures/shell_overlay.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=145)
    table = out / "artifacts.csv"
    write_csv(table, [*read_csv(table), artifact(study, first["case_id"], "shell_plot", output)])


def main() -> None:
    """从仓库根运行 09 的同池壳层对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    config = {"threshold_vpm": 0.10, "limits_mm": [5, 15], "lambda": ARMS,
              "development_targets": ["stn295", "gp222", "a4ul58"], "comparison_arm": "volume_base"}
    columns = ("arm", "lambda", "shell", "domain_mm3", "above_mm3", "Cov010", "rank", "D_hot_mm")
    run_pool_experiment(study, "09", args.run_id, args.baseline_id, args.stage, score_candidates, "shell_spill.csv", columns, config, shell_table)
    plot_shells(study, study / "results/09" / args.run_id, args.baseline_id)


if __name__ == "__main__":
    main()
