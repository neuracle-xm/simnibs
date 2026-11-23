"""在同一外溢预算内比较全靶缺口与弱场尾部缺口。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.location_objective import volume_tail
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


FRACTIONS = {"whole_target": 1.0, "low20": 0.20, "low10": 0.10, "low30": 0.30}


def tail_shortfall(field: FloatArray, volume: FloatArray, fraction: float) -> tuple[FloatArray, float]:
    """以精确体积截断选择弱场尾部，并计算归一化不足。

    Parameters
    ----------
    field, volume : ndarray
        靶区内同序的包络场及体积。
    fraction : float
        请求累积的靶区体积分数，1 表示全靶。

    Returns
    -------
    selected_mm3, shortfall : tuple
        每单元纳入体积和在 0.10 V/m 处的截断平均缺口。
    """
    amount = fraction * float(volume.sum())
    weights = volume_tail(field, volume, amount, high=False)
    if not np.isclose(weights.sum(), amount):
        raise ValueError("弱场尾部选取体积与请求体积不一致")
    return weights, float(np.average(np.maximum(0, 1 - field / 0.10), weights=weights))


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """固定同池 ROC0 外溢预算，输出全靶与不同尾部比例的排序分量。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        共同脑域、体积和靶掩膜。
    candidates : list of Candidate
        1+1 mA 的有效场及绝对指标。

    Returns
    -------
    list of dict
        包含严格可行性、边界分配、尾部独立元素数及主导程度的各臂行。
    """
    baseline = min(candidates, key=lambda candidate: (candidate.roc, candidate.identifier))
    keys = ("Vout010", "Vcout010")
    budget = [baseline.absolute[key] for key in keys]
    volume = geometry.volume_mm3[target]
    rows = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            field = data["max_TI"][target]
        feasible = all(candidate.absolute[key] <= cap for key, cap in zip(keys, budget))
        for arm, fraction in FRACTIONS.items():
            weights, loss = tail_shortfall(field, volume, fraction)
            selected = weights > 0
            boundary = field == field[selected].max()
            rows.append(dict(candidate_id=candidate.identifier, arm=arm, returned_loss=loss, feasible=feasible,
                             Q10_T=candidate.absolute["Q10_T"], Q50_T=candidate.absolute["Q50_T"], tail_fraction=fraction,
                             tail_mm3=float(weights.sum()), n_elements=int(selected.sum()),
                             boundary_fraction=float(weights[boundary].sum() / volume[boundary].sum()),
                             max_element_share=float(weights.max() / weights.sum()), shortfall=loss,
                             budget_candidate_id=baseline.identifier, B_R_mm3=budget[0], B_C_mm3=budget[1],
                             W_C_mm3=candidate.absolute["Vcout010"], candidate_field_path=str(candidate.path)))
    return rows


def plot_tail(study: Path, out: Path, baseline_id: str) -> None:
    """以个体坐标并排显示全靶与 20% 弱场方案获选场的尾部位置。

    Parameters
    ----------
    study, out : Path
        研究根及含 08 明细的运行目录。
    baseline_id : str
        共享几何所属的 03 运行号。
    """
    rows = read_csv(out / "tail_coverage.csv")
    good = [row for row in rows if row["rank"] == "1" and row["arm"] == "low20"]
    if not good:
        return
    chosen = good[0]
    other = next(row for row in rows if row["case_id"] == chosen["case_id"] and row["rank"] == "1" and row["arm"] == "whole_target")
    path_row = next(row for row in read_csv(study / "results/03" / baseline_id / "shared_inputs.csv")
                    if row["subject"] == chosen["subject"] and row["role"] == "geometry")
    geometry, _, targets = load_geometry(study / path_row["path"])
    roi = targets[chosen["target"]]
    points = geometry.centroid_mm[roi]
    volumes = geometry.volume_mm3[roi]
    fig = Figure(figsize=(10, 5.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3, sharex="col", sharey="col")
    for index, selected in enumerate((other, chosen)):
        with np.load(selected["candidate_field_path"]) as data:
            weight, _ = tail_shortfall(data["max_TI"][roi], volumes, 0.20)
        keep = weight > 0
        for ax, (x, y) in zip(axes[index], ((0, 1), (0, 2), (1, 2))):
            ax.scatter(points[:, x], points[:, y], color="#ced8e1", s=7, label="Target")
            ax.scatter(points[keep, x], points[keep, y], color="#ac441b", s=7 + 24 * weight[keep] / volumes[keep], label="Weakest 20% by volume")
            ax.set(xlabel=f"{'XYZ'[x]} (mm)", ylabel=f"{'XYZ'[y]} (mm)", title=selected["arm"])
            ax.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(chosen["case_id"])
    output = out / "figures/tail_overlay.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=145)
    table = out / "artifacts.csv"
    write_csv(table, [*read_csv(table), artifact(study, chosen["case_id"], "tail_plot", output)])


def main() -> None:
    """从仓库根执行 08 的先导或开发同池试验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    config = {"threshold_vpm": 0.10, "tail_fractions": FRACTIONS, "pilot_targets": ["rhipp216"],
              "development_targets": PILOT_TARGETS, "comparison_arm": "whole_target"}
    columns = ("tail_fraction", "tail_mm3", "n_elements", "boundary_fraction", "shortfall", "Q10_T", "Cov010", "feasible")
    run_pool_experiment(study, "08", args.run_id, args.baseline_id, args.stage, score_candidates, "tail_coverage.csv", columns, config)
    plot_tail(study, study / "results/08" / args.run_id, args.baseline_id)


if __name__ == "__main__":
    main()
