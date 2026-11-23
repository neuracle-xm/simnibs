"""在固定 TI 向量场上量化个体物理坐标轴的包络敏感性。"""

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.diagnostics import FieldCase, ResultRows, load_case, run_diagnostic
from neuracle.ti_depth_study.metrics import FloatArray, measure_field
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, digest, read_csv, write_csv


AXIS_NAMES = ("LR", "AP", "IS")
WORLD_AXES = np.eye(3, dtype=np.float64)
ATOL = 1e-8
RTOL = 1e-6


def direction_fields(e_a: FloatArray, e_b: FloatArray, axes: FloatArray) -> FloatArray:
    """在各单位物理轴求解两路向量投影的调制包络。

    Parameters
    ----------
    e_a, e_b : ndarray
        (N, 3) 的个体世界空间载波向量，单位 V/m。
    axes : ndarray
        (K, 3) 的单位方向，顺序与输出列相同。

    Returns
    -------
    ndarray
        (N, K) 的方向包络 2 min(|n·E_A|, |n·E_B|)。
    """
    if not np.allclose(np.linalg.norm(axes, axis=1), 1):
        raise ValueError("方向必须由个体物理坐标中的单位向量定义")
    return 2 * np.minimum(np.abs(e_a @ axes.T), np.abs(e_b @ axes.T))


def evaluate(case: FieldCase, out: Path, artifacts: list[dict[str, str]]) -> ResultRows:
    """回算三条 RAS 固定轴和无方向场，逐元素核验方向上界及换号不变性。

    Parameters
    ----------
    case, out, artifacts : FieldCase, Path and list of dict
        冻结共同场、运行目录和增量证据索引。

    Returns
    -------
    rows, pairs : tuple of lists
        四种场的指标及相对 max_TI 的逐方向差值。
    """
    del out
    if not case.valid.all():
        raise ValueError("方向试验需要两路完整且已对齐的单频向量")
    native = case.native
    t1 = case.study / "head_models" / f"m2m_{native['subject']}" / "T1.nii.gz"
    affine = nib.load(t1).affine
    voxel_rotation = np.linalg.inv(affine[:3, :3])
    preferred = direction_fields(case.e_a, case.e_b, WORLD_AXES)
    flipped = direction_fields(case.e_a, case.e_b, -WORLD_AXES)
    if not np.array_equal(preferred, flipped):
        raise ValueError("方向换号后包络变化")
    difference = preferred - case.field[:, None]
    violations = difference > ATOL + RTOL * np.abs(case.field[:, None])
    if violations.any():
        raise ValueError(f"方向包络超过 max_TI，最大超额 {difference.max()} V/m")
    artifacts.append(artifact(case.study, native["case_id"], "field_frame_T1", t1))
    common = dict(candidate_id=native["case_id"], field_frame="subject_RAS_mm", affine_sha256=digest(t1),
                  field_source="common_forward", bound_atol_vpm=ATOL, bound_rtol=RTOL)
    base = measure_field(case.geometry, case.target, case.e_a, case.e_b, case.valid)
    rows = [common | base | dict(axis="max_TI", upper_bound_excess_vpm=0.0)]
    group = "cortical" if native["target"] == "a4ul58" else "subcortical"
    pairs = []
    for axis, vector, excess in zip(AXIS_NAMES, WORLD_AXES, difference.T):
        e_a = (case.e_a @ vector)[:, None] * vector
        e_b = (case.e_b @ vector)[:, None] * vector
        result = measure_field(case.geometry, case.target, e_a, e_b, case.valid)
        vx = voxel_rotation @ vector
        row = common | result | dict(axis=axis, nx=vector[0], ny=vector[1], nz=vector[2],
                                     voxel_step_per_mm=vx.tolist(), upper_bound_excess_vpm=float(max(0.0, excess.max())))
        rows.append(row)
        pairs.append(dict(case_id=f"{native['case_id']}:{axis}", reference_case_id=f"{native['case_id']}:max_TI",
                          arm=axis, reference_arm="max_TI", field_source="common_forward", **compare_endpoints(row, base, group)))
    return rows, pairs


def finish_diagnostic(study: Path, out: Path, baseline_id: str) -> None:
    """比较同病例的候选名次，并绘制四种场共享色标的轴向切面。

    Parameters
    ----------
    study, out : Path
        本地研究根及方向试验运行目录。
    baseline_id : str
        本篇使用的原生基线运行编号。
    """
    table = out / "direction_axes.csv"
    rows = read_csv(table)
    if not rows:
        return
    groups = {(row["subject"], row["target"], row["axis"]) for row in rows}
    for key in groups:
        ranked = sorted((row for row in rows if row["status"] == "ok" and (row["subject"], row["target"], row["axis"]) == key),
                        key=lambda row: (-float(row["Cov010"]), float(row["D_hot_mm"]), row["candidate_id"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = str(rank)
    write_csv(table, rows)
    chosen = next(row for row in read_csv(study / "results/03" / baseline_id / "baseline_fields.csv") if row["case_id"] == rows[0]["candidate_id"])
    case = load_case(study, chosen)
    points = case.geometry.centroid_mm
    roi = points[case.target]
    center_z = float(roi[:, 2].mean())
    x, y = np.meshgrid(np.linspace(points[:, 0].min(), points[:, 0].max(), 190), np.linspace(points[:, 1].min(), points[:, 1].max(), 190))
    grid = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, center_z)))
    gap, idx = cKDTree(points).query(grid)
    directional = direction_fields(case.e_a, case.e_b, WORLD_AXES)
    all_fields = (*directional.T, case.field)
    vmax = float(case.field.max())
    figure = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for label, field, ax in zip((*AXIS_NAMES, "max_TI"), all_fields, axes.ravel()):
        data = np.ma.masked_where(gap.reshape(x.shape) > 4.0, field[idx].reshape(x.shape))
        panel = ax.pcolormesh(x, y, data, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        section = case.target[idx].reshape(x.shape) & (gap.reshape(x.shape) <= 4.0)
        if section.any() and not section.all():
            ax.contour(x, y, section, levels=[0.5], colors=["#41b9d0"])
        hot_row = next(row for row in rows if row["candidate_id"] == chosen["case_id"] and row["axis"] == label)
        if hot_row["status"] == "ok":
            ax.scatter(*json.loads(hot_row["hot_center_mm"])[:2], marker="x", color="#21c5b3")
        ax.set(xlabel="X (mm)", ylabel="Y (mm)", title=label)
        ax.set_aspect("equal", adjustable="box")
    figure.colorbar(panel, ax=axes, label="TI envelope (V/m)", shrink=0.72)
    figure.suptitle(f"{chosen['case_id']} | axial slice at Z={center_z:.1f} mm")
    output = out / "figures/axes.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=145)
    manifest = out / "artifacts.csv"
    artifacts = read_csv(manifest)
    for row in artifacts:
        if row["path"] == table.relative_to(study).as_posix():
            row["sha256"] = digest(table)
    write_csv(manifest, [*artifacts, artifact(study, chosen["case_id"], "direction_plot", output)])


def main() -> None:
    """从仓库根执行共同 FEM 场的方向敏感性试验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    targets = ("rhipp216", "a4ul58") if args.stage == "pilot" else PILOT_TARGETS
    config = {"field_frame": "subject_RAS_mm", "world_axes": dict(zip(AXIS_NAMES, WORLD_AXES.tolist())), "bound_atol_vpm": ATOL, "bound_rtol": RTOL}
    columns = ("candidate_id", "axis", "nx", "ny", "nz", "field_frame", "upper_bound_excess_vpm", "D_hot_mm", "Cov010", "rank")
    run_diagnostic(study, "17", args.run_id, args.baseline_id, args.stage, targets, evaluate, "direction_axes.csv", columns, config)
    finish_diagnostic(study, study / "results/17" / args.run_id, args.baseline_id)


if __name__ == "__main__":
    main()
