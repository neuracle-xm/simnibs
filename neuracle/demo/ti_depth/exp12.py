"""对合格的 TI 候选比较与解剖靶等体积热点的实际交集。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.location_objective import volume_tail
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry, tetrahedron_components
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


def hotspot_overlap(geometry: Geometry, target: BoolArray, field: FloatArray) -> tuple[FloatArray, dict[str, Any]]:
    """按精确靶体积取最高场，核对交集及所选单元共享面连通性。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        脑实质四面体体积、连通和目标掩膜。
    field : ndarray
        与网格同序的有效非零包络。

    Returns
    -------
    weights, metrics : tuple
        各单元纳入等体积热点的 mm³，以及交集和分簇读数。
    """
    if not np.any(field > 0):
        raise ValueError("零场没有可判定的等体积热点")
    target_volume = float(geometry.volume_mm3[target].sum())
    weights = volume_tail(field, geometry.volume_mm3, target_volume)
    hot_volume = float(weights.sum())
    if not np.isclose(hot_volume, target_volume):
        raise ValueError("等体积热点累计体积和目标体积不一致")
    overlap = float(weights[target].sum())
    coefficient = overlap / target_volume
    if not -1e-12 <= coefficient <= 1 + 1e-12:
        raise ValueError("等体积热点重叠系数超出 [0,1]")
    positive = weights > 0
    number, components = tetrahedron_components(geometry.connectivity[positive])
    volumes = np.bincount(components, weights=weights[positive], minlength=number)
    intersection = np.bincount(components, weights=weights[positive] * target[positive], minlength=number)
    order = np.argsort(-volumes, kind="stable")
    return weights, dict(V_T_mm3=target_volume, V_H_mm3=hot_volume, intersection_mm3=overlap, O=float(np.clip(coefficient, 0, 1)),
                         component_count=number, largest_component_mm3=float(volumes.max()), component_mm3=volumes[order].tolist(),
                         component_overlap_mm3=intersection[order].tolist(), max_element_share=float(weights.max() / hot_volume))


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """用原 ROC 的覆盖及外溢门槛筛选，再比较重叠和 ROC 名次。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        当前病例的共同脑域和解剖靶。
    candidates : list of Candidate
        共同 FEM 的同电流有效候选。

    Returns
    -------
    list of dict
        同一可行域内的原 ROC 与等体积重叠两套排序明细。
    """
    reference = min(candidates, key=lambda candidate: (candidate.roc, candidate.identifier))
    baseline = reference.absolute
    if not np.isfinite(baseline["Q99_Cout"]):
        raise ValueError("同池 ROC 参照缺少有效靶外皮层分位数")
    rows = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            _, geometry_metrics = hotspot_overlap(geometry, target, data["max_TI"])
        absolute = candidate.absolute
        coverage_ok = (absolute["Cov010"] + 1e-12 >= baseline["Cov010"] - 0.05
                       and (baseline["Cov010"] <= 0 or absolute["Cov010"] > 0))
        feasible = coverage_ok and absolute["Vout010"] <= 1.05 * baseline["Vout010"] and absolute["Q99_Cout"] <= 1.05 * baseline["Q99_Cout"]
        common = geometry_metrics | dict(candidate_id=candidate.identifier, feasible=feasible, reference_candidate_id=reference.identifier,
                         overlap_100mm3=absolute["hot_overlap"], main_cluster_overlap=absolute["main_overlap"],
                         main_100mm3_cluster_volumes=absolute["cluster_mm3"], candidate_field_path=str(candidate.path))
        rows.append(common | dict(arm="roc0", returned_loss=candidate.roc))
        rows.append(common | dict(arm="equal_volume_overlap", returned_loss=1 - geometry_metrics["O"]))
    return rows


def plot_overlap(study: Path, out: Path, baseline_id: str) -> None:
    """分色标记两种获选场的 H∩T、H∖T 和 T∖H 在个体空间的位置。

    Parameters
    ----------
    study, out : Path
        研究根及含 12 明细的运行目录。
    baseline_id : str
        提供几何的 03 运行号。
    """
    rows = read_csv(out / "hot_overlap.csv")
    selected = [row for row in rows if row["arm"] == "equal_volume_overlap" and row["rank"] == "1"]
    if not selected:
        return
    first = selected[0]
    reference = next(row for row in rows if row["case_id"] == first["case_id"] and row["arm"] == "roc0" and row["rank"] == "1")
    item = next(row for row in read_csv(study / "results/03" / baseline_id / "shared_inputs.csv")
                if row["subject"] == first["subject"] and row["role"] == "geometry")
    geometry, _, targets = load_geometry(study / item["path"])
    target = targets[first["target"]]
    points = geometry.centroid_mm
    figure = Figure(figsize=(10, 5.4), layout="constrained")
    FigureCanvasAgg(figure)
    for axes, row in zip(figure.subplots(2, 3, sharex="col", sharey="col"), (reference, first)):
        with np.load(row["candidate_field_path"]) as data:
            weights, _ = hotspot_overlap(geometry, target, data["max_TI"])
        domains = ((target & (weights < geometry.volume_mm3), "Target outside hotspot", "#6f91b5"),
                   (~target & (weights > 0), "Hotspot outside target", "#d67e38"),
                   (target & (weights > 0), "Intersection", "#358d80"))
        for ax, (x, y) in zip(axes, ((0, 1), (0, 2), (1, 2))):
            for mask, label, color in domains:
                ax.scatter(points[mask, x], points[mask, y], s=9, alpha=0.62, color=color, label=label)
            ax.set(xlabel=f"{'XYZ'[x]} (mm)", ylabel=f"{'XYZ'[y]} (mm)", title=row["arm"])
            ax.set_aspect("equal", adjustable="box")
    figure.axes[0].legend(fontsize=7)
    figure.suptitle(f"{first['case_id']} | hotspot volume {float(first['V_T_mm3']):.1f} mm³")
    output = out / "figures/overlap.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=145)
    index = out / "artifacts.csv"
    write_csv(index, [*read_csv(index), artifact(study, first["case_id"], "overlap_plot", output)])


def main() -> None:
    """从仓库根运行 12 的先导或开发同池比较。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    config = {"development_targets": ["stn295", "rhipp216", "a4ul58", "a9d16", "opc204", "a24rv178"], "threshold_vpm": 0.10}
    columns = ("V_T_mm3", "V_H_mm3", "intersection_mm3", "O", "component_count", "largest_component_mm3", "feasible")
    run_pool_experiment(study, "12", args.run_id, args.baseline_id, args.stage, score_candidates, "hot_overlap.csv", columns, config)
    plot_overlap(study, study / "results/12" / args.run_id, args.baseline_id)


if __name__ == "__main__":
    main()
