"""在同一网格和共同场上比较单元计数与体积加权 ROC。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


def roc_inputs(field: FloatArray, volume: FloatArray, target: BoolArray, threshold: float) -> dict[str, Any]:
    """在相同超阈布尔条件下给出计数及体积 ROC 的可复算分量。

    Parameters
    ----------
    field, volume : ndarray
        同序的逐单元场和体积。
    target : ndarray of bool
        靶区掩膜，补集为原 Atlas ROC 靶外域。
    threshold : float
        两区域共用的 V/m 阈值，包含等于阈值的单元。

    Returns
    -------
    dict
        两区域超阈分子、分母、ROC 距离和体积四分组贡献。
    """
    above = field >= threshold
    region = {"roi": target, "rest": ~target}
    boundaries = np.quantile(volume, [0.25, 0.50, 0.75])
    quartiles = np.searchsorted(boundaries, volume, side="left")
    output: dict[str, Any] = {"volume_quartile_edges_mm3": boundaries.tolist()}
    for label, mask in region.items():
        covered = above & mask
        elements = int(mask.sum())
        size = float(volume[mask].sum())
        if elements == 0 or size <= 0:
            raise ValueError(f"{label} 无有效四面体")
        count = int(covered.sum())
        weight = float(volume[covered].sum())
        count_numerator = np.bincount(quartiles[covered], minlength=4)
        volume_numerator = np.bincount(quartiles[covered], weights=volume[covered], minlength=4)
        if count_numerator.sum() != count or not np.isclose(volume_numerator.sum(), weight):
            raise ValueError(f"{label} 的超阈四分组复核失败")
        output.update({f"{label}_elements": elements, f"{label}_mm3": size, f"{label}_above_elements": count,
                       f"{label}_above_mm3": weight, f"{label}_quartile_count_contribution": (count_numerator / elements).tolist(),
                       f"{label}_quartile_volume_contribution": (volume_numerator / size).tolist()})
    count_tpr = output["roi_above_elements"] / output["roi_elements"]
    count_fpr = output["rest_above_elements"] / output["rest_elements"]
    vol_tpr = output["roi_above_mm3"] / output["roi_mm3"]
    vol_fpr = output["rest_above_mm3"] / output["rest_mm3"]
    return output | dict(count_tpr=count_tpr, count_fpr=count_fpr, vol_tpr=vol_tpr, vol_fpr=vol_fpr,
                         roc_count=float(np.hypot(1 - count_tpr, count_fpr)), roc_volume=float(np.hypot(1 - vol_tpr, vol_fpr)))


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """对两种权重分别打分并核对计数臂与原生 ROC 一致。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        共同脑域的体积及同序靶掩膜。
    candidates : list of Candidate
        合法、同电流的共同正向场。

    Returns
    -------
    list of dict
        每个候选的计数参照臂及体积加权臂。
    """
    rows = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            info = roc_inputs(data["max_TI"], geometry.volume_mm3, target, 0.10)
        if not np.isclose(info["roc_count"], candidate.roc, atol=1e-12, rtol=0):
            raise ValueError(f"计数距离与 SimNIBS ROC 不一致: {candidate.identifier}")
        for arm, loss in (("roc0", "roc_count"), ("volume_roc", "roc_volume")):
            rows.append(info | dict(candidate_id=candidate.identifier, arm=arm, returned_loss=info[loss]))
    return rows


def merge_arms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为同病例同候选合并两套排名，保留首选是否翻转。

    Parameters
    ----------
    rows : list of dict
        由同池运行器按实验臂分别排名的明细。

    Returns
    -------
    list of dict
        每个候选一行，含两种排名及本靶获选是否改变。
    """
    volume_by_key = {(row["case_id"], row["candidate_id"]): row["rank"] for row in rows if row["arm"] == "volume_roc"}
    result = []
    for row in rows:
        if row["arm"] == "roc0":
            case_id, candidate_id = row["case_id"], row["candidate_id"]
            pivot = {key: value for key, value in row.items() if key not in {"arm", "returned_loss", "rank"}}
            result.append(pivot | dict(rank_count=row["rank"], rank_volume=volume_by_key[case_id, candidate_id]))
    flipped = {row["case_id"] for row in result if row["rank_count"] == 1 and row["rank_volume"] != 1}
    return [row | {"winner_changed": row["case_id"] in flipped} for row in result]


def plot_ranks(study: Path, out: Path) -> None:
    """绘制病例的计数与体积名次，并标记两者各自首选。

    Parameters
    ----------
    study, out : Path
        研究根及含专属表的 06 运行目录。
    """
    rows = read_csv(out / "roc_weights.csv")
    if not rows:
        return
    figure = Figure(figsize=(7, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    n = max(int(row["rank_count"]) for row in rows)
    ax.plot([1, n], [1, n], color="#888888", linestyle="--", linewidth=0.8)
    for case in dict.fromkeys(row["case_id"] for row in rows):
        subset = [row for row in rows if row["case_id"] == case]
        ax.scatter([int(row["rank_count"]) for row in subset], [int(row["rank_volume"]) for row in subset], alpha=0.65, label=case)
    winners = [row for row in rows if row["rank_count"] == "1" or row["rank_volume"] == "1"]
    ax.scatter([int(row["rank_count"]) for row in winners], [int(row["rank_volume"]) for row in winners],
               facecolors="none", edgecolors="#151515", s=95, label="Selected by either ROC")
    ax.set(xlabel="Element-count ROC rank", ylabel="Volume-weighted ROC rank")
    ax.legend(loc="best", fontsize=7)
    target = out / "figures/roc_rank.png"
    target.parent.mkdir(exist_ok=True)
    figure.savefig(target, dpi=140)
    table = out / "artifacts.csv"
    write_csv(table, [*read_csv(table), artifact(study, "run", "rank_plot", target)])


def main() -> None:
    """从仓库根启动计数与体积的同池离线对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    config = {"threshold_vpm": 0.10, "development_targets": PILOT_TARGETS}
    columns = ("count_tpr", "vol_tpr", "count_fpr", "vol_fpr", "rank_count", "rank_volume", "roi_mm3", "rest_mm3")
    run_pool_experiment(study, "06", args.run_id, args.baseline_id, args.stage, score_candidates,
                        "roc_weights.csv", columns, config, merge_arms)
    plot_ranks(study, study / "results/06" / args.run_id)


if __name__ == "__main__":
    main()
