"""核对个体靶区位移，并比较固定共同 FEM 候选的名义和稳健排序。"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

from filelock import FileLock
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import nibabel as nib
import numpy as np

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.demo.ti_depth.exp21 import SHIFTS, sample_shifted, weighted_dice
from neuracle.ti_depth_study.coverage_objective import coverage_terms, spill_loss
from neuracle.ti_depth_study.diagnostics import load_case
from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.metrics import BoolArray, Geometry, distance_to_target
from neuracle.ti_depth_study.pool import Candidate, run_pool_experiment
from neuracle.ti_depth_study.records import artifact, digest, read_csv, update_index, write_csv, write_json
from simnibs.utils.transformations import subject2mni_coords


MASK_IDS = tuple(f"shift{index:02}" for index in range(len(SHIFTS)))
SPLITS = tuple("holdout" if 2.0 in map(abs, shift) else "training" if any(shift) else "nominal" for shift in SHIFTS)
TRAINING_SIZE = SPLITS.count("nominal") + SPLITS.count("training")
SHIFT_COLUMNS = ("subject", "target", "atlas_name", "label", "mask_id", "split", "dx_mm", "dy_mm", "dz_mm", "roi_mm3", "n_elements",
                 "mni_centroid_x_mm", "center_mm", "centroid_along_shift_mm", "Dice_nominal", "mask_sha256", "geometry_sha256", "ROI_sha256",
                 "source_voxel_mm3", "source_roi_mm3", "source_axes", "status", "reason")
ROBUST_COLUMNS = ("mask_id", "split", "dx_mm", "dy_mm", "dz_mm", "roi_mm3", "shortfall", "Cov010", "D_hot_mm", "feasible",
                  "rank", "returned_loss", "Vout010", "Vcout010", "B_R_mm3", "B_C_mm3", "reference_roc0_candidate_id")


def find_geometry(study: Path, upstream: Path, subjects: tuple[str, ...]) -> dict[str, Path]:
    """定位 03 几何的 ROI 来源，并与冻结的候选网格复核散列。

    Parameters
    ----------
    study, upstream : Path
        研究根与上游 03 运行目录。
    subjects : tuple of str
        此阶段预登记的开发头模。

    Returns
    -------
    dict
        有散列或原生场来源证据的已知头模参考网格。
    """
    fixed = upstream / "shared_inputs.csv"
    native = upstream / "baseline_fields.csv"
    paths: dict[str, Path] = {}
    frozen: dict[str, str] = {}
    if fixed.is_file():
        for row in read_csv(fixed):
            if row["role"] == "geometry" and row["subject"] in subjects:
                source = study / row["path"]
                if digest(source) != row["sha256"]:
                    raise ValueError(f"03 冻结几何散列已改变: {source}")
                # 候选包只复制 geometry.npz，位移还需原参考目录的个体 ROI 和变换清单。
                frozen[row["subject"]] = row["sha256"]
                if source.with_name("geometry_inputs.json").is_file():
                    paths[row["subject"]] = source
    if native.is_file():
        for row in read_csv(native):
            subject = row["subject"]
            if subject in subjects and subject not in paths and row["status"] == "ok":
                source = load_case(study, row).geometry_path
                if subject in frozen and digest(source) != frozen[subject]:
                    raise ValueError(f"{subject} 的原生 ROI 几何与冻结候选网格不一致")
                paths[subject] = source
    return paths


def prepare_masks(
    study: Path, out: Path, subject: str, registered: dict[str, str],
    geometry_path: Path, geometry: Geometry, nominal: BoolArray,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, str]]]:
    """按原仿射最近邻生成掩膜，并核验原位一致、非空及回到 MNI 后的侧别。

    Parameters
    ----------
    study, out : Path
        研究根和本篇结果目录。
    subject : str
        头模 ID。
    registered : dict
        02 登记的右侧解剖靶及原始图谱掩膜。
    geometry_path, geometry, nominal : Path, Geometry and ndarray
        03 的已验收参考源及对应靶区。

    Returns
    -------
    Path, list of dict, list of dict
        13 组同序位移掩膜的登记、逐个几何质控和可追溯文件。
    """
    key = registered["target"]
    case_id = f"{subject}_{key}"
    source = geometry_path.parent / "roi" / f"{key}.nii.gz"
    head = study / "head_models" / f"m2m_{subject}"
    chain_path = geometry_path.with_name("geometry_inputs.json")
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    registered_mask = Path(registered["mask_path"])
    if digest(registered_mask) != chain["roi_inputs"][key]:
        raise ValueError(f"{case_id} 的 02 图谱源掩膜散列已改变")
    registration = head / "toMNI/Conform2MNI_nonl.nii.gz"
    if digest(registration) != chain["registration"]:
        raise ValueError(f"{case_id} 的个体到 MNI 变换散列已改变")
    image = nib.load(source)
    values = np.asanyarray(image.dataobj)
    voxel_mm3 = float(abs(np.linalg.det(image.affine[:3, :3])))
    if values.ndim != 3 or not np.isfinite(voxel_mm3) or voxel_mm3 <= 0 or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{case_id} 的个体掩膜需要有效仿射和二值前景")
    source_roi_mm3 = float(np.count_nonzero(values) * voxel_mm3)
    image = nib.Nifti1Image(values.astype(np.uint8), image.affine)
    masks = [sample_shifted(image, geometry.centroid_mm, vector) for vector in SHIFTS]
    centers = {index: np.average(geometry.centroid_mm[mask], axis=0, weights=geometry.volume_mm3[mask])
               for index, mask in enumerate(masks) if mask.any()}
    mni_x = dict(zip(centers, subject2mni_coords(np.asarray(list(centers.values())), str(head))[:, 0])) if centers else {}
    source_hash = digest(source)
    rows: list[dict[str, Any]] = []
    for index, (mask_id, split, (dx, dy, dz), mask) in enumerate(zip(MASK_IDS, SPLITS, SHIFTS, masks)):
        shift = np.array([dx, dy, dz])
        along: float | str = "NA"
        if index and index in centers and 0 in centers:
            along = float(np.dot(centers[index] - centers[0], shift) / np.linalg.norm(shift))
        if index not in centers:
            reason = "位移掩膜与脑实质交集为空"
        elif not np.isfinite(mni_x[index]) or mni_x[index] <= 0:
            reason = "个体掩膜重心回到 MNI 后未落于右侧"
        elif index == 0 and not np.array_equal(mask, nominal):
            reason = "原位最近邻掩膜与 03 网格靶区不一致"
        elif isinstance(along, float) and along <= 0:
            reason = "脑实质交集重心未沿指定方向移动"
        else:
            reason = "NA"
        dice = weighted_dice(nominal, mask, geometry.volume_mm3)
        rows.append(dict(subject=subject, target=key, atlas_name=registered["atlas_name"], label=registered["label"],
                         mask_id=mask_id, split=split, dx_mm=dx, dy_mm=dy, dz_mm=dz,
                         roi_mm3=float(geometry.volume_mm3[mask].sum()), n_elements=int(mask.sum()),
                         mni_centroid_x_mm=mni_x.get(index, "NA"), center_mm=centers[index].tolist() if index in centers else "NA",
                         centroid_along_shift_mm=along,
                         Dice_nominal=dice if np.isfinite(dice) else "NA", mask_sha256=hashlib.sha256(np.packbits(mask)).hexdigest(),
                         geometry_sha256=chain["geometry_sha256"], ROI_sha256=source_hash, source_voxel_mm3=voxel_mm3,
                         source_roi_mm3=source_roi_mm3, source_axes="".join(nib.aff2axcodes(image.affine)),
                         status="ok" if reason == "NA" else "failed", reason=reason))
    path = out / "masks" / subject / f"{key}_shifts.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, element_id=geometry.element_id, offsets_mm=np.asarray(SHIFTS), **dict(zip(MASK_IDS, masks)))
    temporary.replace(path)
    evidence = [artifact(study, case_id, role, entry) for role, entry in (
        ("mask_geometry", geometry_path), ("individual_ROI", source), ("anatomical_chain", chain_path),
        ("registered_ROI", registered_mask),
        ("world_transform", registration), ("shift_masks", path))]
    return path, rows, evidence


def score_candidates(
    geometry: Geometry, target: BoolArray, candidates: list[Candidate], masks: Path | None,
) -> list[dict[str, Any]]:
    """按相同名义预算比较名义不足量与七个位移中最大的不足量。

    Parameters
    ----------
    geometry, target : Geometry and ndarray
        冻结候选库的共同网格及同序名义靶区。
    candidates : list of Candidate
        两路各 1 mA，已核验散列及完整共同场的等条件布局。
    masks : Path or None
        同一网格上已核验的位移掩膜集。若尚未核验，病例不排名。

    Returns
    -------
    list of dict
        每布局的两臂损失及逐位移明细。预留位移只写入诊断。
    """
    if masks is None:
        raise ValueError("13 组掩膜未完成几何核验")
    with np.load(masks) as data:
        if not np.array_equal(data["element_id"], geometry.element_id) or not np.array_equal(data[MASK_IDS[0]], target):
            raise ValueError("位移掩膜和冻结候选不是同一个网格及名义靶区")
        targets = [data[key] for key in MASK_IDS]
    reference = min(candidates, key=lambda candidate: (candidate.roc, candidate.identifier))
    volume = geometry.volume_mm3
    cortex = geometry.cortex_mask
    with np.load(reference.path) as data:
        reference_terms = coverage_terms(data["max_TI"], volume, target, cortex)
    budget = (reference_terms["W_R_mm3"], reference_terms["W_C_mm3"])
    rows = []
    for candidate in candidates:
        with np.load(candidate.path) as data:
            field = data["max_TI"]
        shortfall = np.maximum(0.0, 1.0 - field / 0.10)
        training = [float(np.average(shortfall[mask], weights=volume[mask])) for mask in targets[:TRAINING_SIZE]]
        losses = {"nominal": training[0], "robust": max(training)}
        center = np.asarray(candidate.absolute["hot_center_mm"])
        terms = [coverage_terms(field, volume, mask, cortex) for mask in targets]
        feasible = spill_loss(terms[0], budget, 1.0)["feasible"]
        readings = [dict(mask_id=key, split=split, dx_mm=dx, dy_mm=dy, dz_mm=dz, roi_mm3=float(volume[mask].sum()),
                         shortfall=float(np.average(shortfall[mask], weights=volume[mask])), Cov010=values["K"],
                         D_hot_mm=distance_to_target(center, geometry, mask), Vout010=values["W_R_mm3"], Vcout010=values["W_C_mm3"])
                    for key, split, (dx, dy, dz), mask, values in zip(MASK_IDS, SPLITS, SHIFTS, targets, terms)]
        rows.extend(dict(arm=arm, candidate_id=candidate.identifier, returned_loss=loss, feasible=feasible, readings=readings,
                         B_R_mm3=budget[0], B_C_mm3=budget[1], reference_roc0_candidate_id=reference.identifier)
                    for arm, loss in losses.items())
    return rows


def expand_readings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将同池排序的候选行展开为逐位移明细。

    Parameters
    ----------
    rows : list of dict
        两臂已完成可行性排序的候选行，含全部诊断读数。

    Returns
    -------
    list of dict
        保留同一 rank 的各个名义、训练和预留位置。
    """
    return [{key: value for key, value in row.items() if key not in ("readings", "Q99_Cout", "hot_center_mm")} | reading
            for row in rows for reading in row["readings"]]


def summarize_winners(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """分别汇总两臂胜者的名义、训练和预留读数。

    Parameters
    ----------
    rows : list of dict
        保留了臂、排名与位移组别的专属表。

    Returns
    -------
    list of dict
        每组覆盖范围、最大距离及最大靶外体积。
    """
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row["rank"] == "1":
            key = tuple(row[column] for column in ("case_id", "arm", "candidate_id", "split"))
            groups.setdefault(key, []).append(row)
    return [dict(case_id=case, arm=arm, candidate_id=candidate, split=group,
                 min_coverage=min(float(row["Cov010"]) for row in subset),
                 max_coverage=max(float(row["Cov010"]) for row in subset),
                 max_distance=max(float(row["D_hot_mm"]) for row in subset),
                 max_outside=max(float(row["Vout010"]) for row in subset),
                 max_cortical=max(float(row["Vcout010"]) for row in subset))
            for (case, arm, candidate, group), subset in groups.items()]


def plot_robustness(out: Path, rows: list[dict[str, Any]]) -> Path:
    """按病例比较两臂胜者在三个位置组的覆盖范围。

    Parameters
    ----------
    out : Path
        当前研究输出目录。
    rows : list of dict
        两臂胜者在各位移组的汇总读数。

    Returns
    -------
    Path
        诊断图路径。
    """
    winners = list(dict.fromkeys((row["case_id"], row["arm"]) for row in rows))
    coverage = {(row["case_id"], row["arm"], row["split"]): (row["min_coverage"], row["max_coverage"]) for row in rows}
    fig = Figure(figsize=(max(8, 1.12 * len(winners)), 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for index, (group, color) in enumerate((("nominal", "#3d5375"), ("training", "#3e997d"), ("holdout", "#bf7144"))):
        values = np.asarray([coverage[case, arm, group] for case, arm in winners])
        ax.errorbar(np.arange(len(winners)) + (index - 1) * 0.22, values.mean(axis=1),
                    yerr=(values[:, 1] - values[:, 0]) / 2, fmt="o", capsize=4, color=color, label=group)
    ax.set_xticks(np.arange(len(winners)), [f"{case}\n{arm}" for case, arm in winners], fontsize=8)
    ax.set(ylabel="Coverage range at 0.10 V/m", ylim=(-0.03, 1.05))
    ax.legend()
    path = out / "figures" / "robustness.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    return path


def run(study: Path, out: Path, run_id: str, baseline_id: str, stage: str) -> None:
    """先复核掩膜，然后按可用的冻结共同候选计算筛查结果。

    Parameters
    ----------
    study, out : Path
        研究根与当前 20 运行目录。
    run_id, baseline_id, stage : str
        20 运行号、03 上游运行号和开发阶段。
    """
    upstream = study / "results/03" / baseline_id
    subjects, targets = (("sub1",), ("stn295",)) if stage == "pilot" else (("sub1", "sub2"), PILOT_TARGETS)
    sources = find_geometry(study, upstream, subjects)
    registered = {row["target"]: row for row in read_csv(study / "results/02/run-001/targets.csv")}
    library: dict[tuple[str, str], Path] = {}
    checks: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = [artifact(study, "run", "implementation", Path(__file__))]
    staged_cases = []
    for subject in subjects:
        if subject in sources:
            geometry, _, original = load_geometry(sources[subject])
        for key in targets:
            case = dict(case_id=f"{subject}_{key}_masks", subject=subject, target=key, method="shared_geometry", arm="geometry_qc",
                        seed="NA", condition="world_RAS_mm", task_id="NA", status="pending", reason="等待 03 参考几何", wall_s="NA")
            staged_cases.append(case)
            if subject not in sources:
                continue
            try:
                path, rows, evidence = prepare_masks(study, out, subject, registered[key], sources[subject], geometry, original[key])
                checks.extend(rows)
                artifacts.extend(evidence)
                valid = all(row["status"] == "ok" for row in rows)
                if valid:
                    library[subject, key] = path
                case.update(status="ok" if valid else "failed",
                            reason="13 个位置均通过几何 QC" if valid else "位移掩膜未通过几何 QC，详见 shift_masks.csv")
            except (OSError, ValueError, RuntimeError, KeyError, nib.filebasedimages.ImageFileError) as error:
                case.update(status="failed", reason=str(error))
    shifts = out / "shift_masks.csv"
    write_csv(shifts, checks, SHIFT_COLUMNS)
    config = {"threshold_vpm": 0.10, "development_targets": PILOT_TARGETS, "pilot_targets": ("stn295",),
              "comparison_arm": "nominal", "selection_offsets_mm": SHIFTS[:TRAINING_SIZE],
              "holdout_offsets_mm": SHIFTS[TRAINING_SIZE:], "budget_source": "共同正向候选内的 ROC0 最优布局",
              "interpolation": "individual_world_RAS_nearest"}
    run_pool_experiment(study, "20", run_id, baseline_id, stage, None, "robust_masks.csv", ROBUST_COLUMNS, config,
                        table_transform=expand_readings,
                        score_factory=lambda subject, key: partial(score_candidates, masks=library.get((subject, key))))
    cases_path = out / "cases.csv"
    cases = staged_cases + read_csv(cases_path)
    for subject in subjects:
        for key in targets:
            cases.extend(dict(case_id=f"{subject}_{key}_{method}_search", subject=subject, target=key, method=method, arm="independent_search",
                              seed="NA", condition="1+1mA", task_id="NA", status="pending",
                              reason="等待逐路线稳健实际重搜及胜者共同正向复算", wall_s="NA")
                         for method in ("leadfield_based", "leadfield_free"))
    write_csv(cases_path, cases)
    report = out / "report.md"
    counts = {value: sum(row["status"] == value for row in cases) for value in ("ok", "failed", "pending")}
    mask_ok = sum(case["status"] == "ok" for case in staged_cases)
    selected = read_csv(out / "robust_masks.csv")
    winners = summarize_winners(selected)
    decision = "仅筛查" if selected else "几何准备" if mask_ok else "无法判定"
    report_text = (f"# 20 靶区定位不确定性\n\n{stage} 阶段完成几何核验 {mask_ok}/{len(staged_cases)} 格。"
                   f"全部任务完成 {counts['ok']} 格，失败 {counts['failed']} 格，待完成 {counts['pending']} 格。当前结论为{decision}。\n\n"
                   "固定场筛查对两臂施加相同的名义全脑及皮层外溢约束。稳健臂按原位与六个 ±1 mm 位移中的最大不足量排序。"
                   "六个 ±2 mm 预留位移只用于诊断。逐路线实际重搜及其胜者复算待执行。\n\n"
                   "[掩膜几何](shift_masks.csv) · [候选读数](robust_masks.csv) · [名义配对](paired.csv) · [案例状态](cases.csv)\n")
    if winners:
        summary_table = "\n".join(f"|{row['case_id']}|{row['arm']}|`{row['candidate_id']}`|{row['split']}|{row['min_coverage']:.2%}|"
                                  f"{row['max_distance']:.3f}|{row['max_outside'] / 1000:.3f}|{row['max_cortical'] / 1000:.3f}|"
                                  for row in winners)
        report_text += ("\n[胜者覆盖范围](figures/robustness.png)\n\n"
                        "|病例|试验臂|胜者|位置组|最低覆盖|最大距靶 mm|最大全脑外溢 mL|最大皮层外溢 mL|\n"
                        "|---|---|---|---|---:|---:|---:|---:|\n" + summary_table + "\n")
    report.write_text(report_text, encoding="utf-8")
    artifacts_path = out / "artifacts.csv"
    outputs = [row for row in read_csv(artifacts_path) if row["path"] not in {report.relative_to(study).as_posix(), cases_path.relative_to(study).as_posix()}]
    produced = [("geometry_checks", shifts), ("status", cases_path), ("report", report)]
    if winners:
        produced.append(("coverage_plot", plot_robustness(out, winners)))
    write_csv(artifacts_path, [*outputs, *artifacts, *(artifact(study, "run", role, path) for role, path in produced)])
    info_path = out / "run.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    status = "partial" if counts["ok"] else "blocked"
    info.update(status=status, scientific_decision=decision)
    info["data_id"].update({subject: path.parent.name for subject, path in sources.items() if subject not in info["data_id"]})
    write_json(info_path, info)
    update_index(study, dict(experiment="20", run_id=run_id, stage=stage, run_status=status, scientific_decision=decision,
                            ok_cases=counts["ok"], failed_cases=counts["failed"], pending_cases=counts["pending"],
                            report_path=report.relative_to(study).as_posix()))


def main() -> None:
    """从仓库根运行先导或开发阶段，并串行化同一运行号的输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    out = study / "results/20" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".run.lock")):
        run(study, out, args.run_id, args.baseline_id, args.stage)


if __name__ == "__main__":
    main()
