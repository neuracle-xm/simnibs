"""核对双侧同名区，并在固定外溢预算内比较对侧覆盖代价。"""

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.atlas.standardized import get_standardized_roi_path
from neuracle.demo.ti_depth.exp02 import BN, JULICH, PILOT_TARGETS
from neuracle.ti_depth_study.fields import brain_rows, load_geometry
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry, coverage, weighted_quantile
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, digest, read_csv, write_csv
from simnibs.mesh_tools.mesh_io import read_msh
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.region_of_interest import mask_image_to_index_mask, mni_mask_to_sub
from simnibs.utils.transformations import subject2mni_coords


PAIRS = {"stn295": (JULICH, "295", "88"), "rhipp216": (BN, "216", "215"), "a4ul58": (BN, "58", "57")}
ARMS = {"side_baseline": 0.0, "contra025": 0.25, "contra100": 1.0}


def prepare_bilateral(head: Path, key: str, geometry: Geometry, target: BoolArray, out: Path) -> tuple[BoolArray, dict[str, Any]]:
    """把同图谱左侧标签取样到与右靶同一参考网格并核验侧别。

    Parameters
    ----------
    head, out : Path
        独立 CHARM 头模及本篇保存配准证据的目录。
    key : str
        共同协议中的右侧靶键。
    geometry, target : Geometry and ndarray
        03 的固定元素顺序及已使用的右侧掩膜。

    Returns
    -------
    left, evidence : tuple
        左侧掩膜及两侧名称、世界坐标、体积和散列记录。
    """
    atlas, right_id, left_id = PAIRS[key]
    original_left = get_standardized_roi_path(atlas, left_id)
    original_right = get_standardized_roi_path(atlas, right_id)
    subject = SubjectFiles(subpath=str(head))
    mesh = read_msh(subject.fnamehead, skip_data=True)
    selected = brain_rows(mesh)
    if (not np.array_equal(mesh.elm.elm_number[selected], geometry.element_id)
            or not np.array_equal(mesh.elm.node_number_list[selected] - 1, geometry.connectivity)
            or not np.array_equal(mesh.nodes.node_coord, geometry.node_mm)):
        raise ValueError("左侧 ROI 的头模网格与共同评价域不一致")
    left_image = mni_mask_to_sub(nib.load(original_left), subject)
    left = np.isin(selected, mask_image_to_index_mask("elm_center", left_image, mesh, 1))
    if not left.any() or not target.any() or np.any(left & target):
        raise ValueError("左右同名区为空或彼此重叠")
    volume = geometry.volume_mm3
    center = np.stack([np.average(geometry.centroid_mm[roi], axis=0, weights=volume[roi]) for roi in (target, left)])
    center_mni = subject2mni_coords(center, str(head))
    if not (center_mni[0, 0] > 0 > center_mni[1, 0]):
        raise ValueError("返回 MNI 的双侧重心与左右约定不符")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{subject.subid}_{key}_left.nii.gz"
    mask_path = out / f"{subject.subid}_{key}_left.npy"
    nib.save(left_image, path)
    np.save(mask_path, left)
    return left, dict(atlas_name=atlas, target_label=right_id, contra_label=left_id, left_subject_path=str(path),
                      left_mask_path=str(mask_path), left_mask_sha256=digest(mask_path), left_subject_sha256=digest(path),
                      template_left_sha256=digest(original_left), template_right_sha256=digest(original_right),
                      right_center_mm=center[0].tolist(), left_center_mm=center[1].tolist(),
                      right_center_mni_mm=center_mni[0].tolist(), left_center_mni_mm=center_mni[1].tolist(),
                      right_mm3=float(volume[target].sum()), left_mm3=float(volume[left].sum()), overlap_mm3=0.0)


def laterality_terms(field: FloatArray, volume: FloatArray, right: BoolArray, left: BoolArray) -> dict[str, float]:
    """计算各侧的实际覆盖、中位场及右减左覆盖百分点。

    Parameters
    ----------
    field, volume, right, left : ndarray
        同一参考网格的场、体积及已核验的双侧掩膜。

    Returns
    -------
    dict
        两侧独立读数，差值只作补充诊断。
    """
    cov_r = coverage(field[right], volume[right], 0.10)
    cov_l = coverage(field[left], volume[left], 0.10)
    return dict(Cov_right=cov_r, Cov_left=cov_l, delta_RL_pp=100 * (cov_r - cov_l),
                Q50_R=weighted_quantile(field[right], volume[right], 0.5), Q50_L=weighted_quantile(field[left], volume[left], 0.5))


def score_candidates(geometry: Geometry, target: BoolArray, candidates: list[Candidate], study: Path, out: Path, subject: str, key: str) -> list[dict[str, Any]]:
    """在同池双预算内比较是否加入对侧强场惩罚。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        固定网格和右侧目标。
    candidates : list of Candidate
        共同、同电流的有效候选场。
    study, out : Path
        研究根及本篇运行目录。
    subject, key : str
        当前头模和靶。

    Returns
    -------
    list of dict
        左右读数、皮层和全脑可行性及各档损失。
    """
    left, evidence = prepare_bilateral(study / "head_models" / f"m2m_{subject}", key, geometry, target, out / "roi")
    roc = min(candidates, key=lambda candidate: (candidate.roc, candidate.identifier))
    budget = [roc.absolute[key_name] for key_name in ("Vout010", "Vcout010")]
    rows = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            field = data["max_TI"]
        terms = laterality_terms(field, geometry.volume_mm3, target, left)
        peak = int(np.lexsort((geometry.element_id, -field))[0])
        region = "right" if target[peak] else "left" if left[peak] else "other"
        feasible = candidate.absolute["Vout010"] <= budget[0] and candidate.absolute["Vcout010"] <= budget[1]
        for arm, weight in ARMS.items():
            loss = 1 - terms["Cov_right"] + weight * terms["Cov_left"]
            rows.append(evidence | terms | dict(candidate_id=candidate.identifier, arm=arm, feasible=feasible, returned_loss=loss,
                        B_R_mm3=budget[0], B_C_mm3=budget[1], W_C_mm3=candidate.absolute["Vcout010"], budget_candidate_id=roc.identifier,
                        main_hot_center_mm=candidate.absolute["hot_center_mm"], peak_location=region,
                        peak_mm=geometry.centroid_mm[peak].tolist(), **{"lambda": weight}))
    return rows


def plot_bilateral(study: Path, out: Path, baseline_id: str) -> None:
    """绘制左右同名区及两条评分规则选出的主热点。

    Parameters
    ----------
    study, out : Path
        研究根和当前实验目录。
    baseline_id : str
        用于定位共同参考几何的上游运行号。
    """
    scores = read_csv(out / "laterality.csv")
    winners = [row for row in scores if row["rank"] == "1"]
    if not winners:
        return
    sample = winners[0]
    geometry_info = next(row for row in read_csv(study / "results/03" / baseline_id / "shared_inputs.csv")
                         if row["subject"] == sample["subject"] and row["role"] == "geometry")
    geometry, _, targets = load_geometry(study / geometry_info["path"])
    left = np.load(sample["left_mask_path"])
    right = targets[sample["target"]]
    positions = geometry.centroid_mm
    figure = Figure(figsize=(10, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    for ax, (x, y) in zip(figure.subplots(1, 3), ((0, 1), (0, 2), (1, 2))):
        ax.scatter(positions[left, x], positions[left, y], s=8, color="#008b89", alpha=0.6, label="Left ROI")
        ax.scatter(positions[right, x], positions[right, y], s=8, color="#ba522a", alpha=0.6, label="Right ROI")
        for row in winners:
            if row["case_id"] == sample["case_id"]:
                point = json.loads(row["main_hot_center_mm"])
                ax.scatter(point[x], point[y], marker="x", s=35, label=row["arm"])
        ax.set(xlabel=f"{'XYZ'[x]} (mm)", ylabel=f"{'XYZ'[y]} (mm)")
        ax.set_aspect("equal", adjustable="box")
    figure.axes[0].legend(fontsize=7)
    figure.suptitle(sample["case_id"])
    path = out / "figures/bilateral.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145)
    table = out / "artifacts.csv"
    artifacts = read_csv(table)
    for row in {item["case_id"]: item for item in scores}.values():
        artifacts.extend(artifact(study, row["case_id"], "contralateral_roi", Path(row[key])) for key in ("left_mask_path", "left_subject_path"))
    write_csv(table, [*artifacts, artifact(study, sample["case_id"], "bilateral_plot", path)])


def main() -> None:
    """从仓库根运行 11 的同池对侧选择性实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    out = study / "results/11" / args.run_id
    config = {"threshold_vpm": 0.10, "labels": PAIRS, "lambda": ARMS,
              "development_targets": PILOT_TARGETS, "comparison_arm": "side_baseline"}
    columns = ("target_label", "contra_label", "Cov_right", "Cov_left", "delta_RL_pp", "Q50_R", "Q50_L", "lambda", "feasible")
    run_pool_experiment(study, "11", args.run_id, args.baseline_id, args.stage, None, "laterality.csv", columns, config,
                        score_factory=lambda subject, key: partial(score_candidates, study=study, out=out, subject=subject, key=key))
    plot_bilateral(study, out, args.baseline_id)


if __name__ == "__main__":
    main()
