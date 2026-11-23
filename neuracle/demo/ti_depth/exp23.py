"""枚举同一固定帽位基场域的 TI 靶内弱载波覆盖上界。"""

import argparse
from itertools import combinations
import json
from pathlib import Path
from time import monotonic
from typing import Any

import h5py
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.baseline import content_id
from neuracle.ti_depth_study.diagnostics import FieldCase, ResultRows, run_diagnostic
from neuracle.ti_depth_study.fields import brain_rows
from neuracle.ti_depth_study.metrics import FloatArray, coverage, ti_envelope
from neuracle.ti_depth_study.records import artifact, digest, read_csv, write_csv
from neuracle.ti_leadfield_optimization.leadfield import LEADFIELD_DATASET, LEADFIELD_MESH_PATH, _decode_name
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.utils.region_of_interest import mask_image_to_index_mask


THRESHOLDS = (0.10, 0.20)
REQUIREMENTS = (0.25, 0.50)
ATOL_VPM = 1e-8
BATCH_SIZE = 32


def target_basis(case: FieldCase, artifacts: list[dict[str, str]]) -> tuple[tuple[str, ...], FloatArray, FloatArray, str]:
    """提取 HDF5 实际目标四面体上各帽位的 1 mA 单位场。

    Parameters
    ----------
    case, artifacts : FieldCase and list of dict
        已知个体靶区和 based 原生任务以及证据索引。

    Returns
    -------
    names, vectors, volumes, cache_hash : tuple
        完整帽位、含零参考行的电场、靶体积及基场摘要。
    """
    row = case.native
    task = case.study / "simulations" / row["inverse_task"]
    ref = task / "leadfield_reference.json"
    manifest = Path(json.loads(ref.read_text(encoding="utf-8"))["manifest_path"])
    data = json.loads(manifest.read_text(encoding="utf-8"))
    source = manifest.parent / data["leadfield"]["filename"]
    roi_path = case.geometry_path.parent / "roi" / f"{row['target']}.nii.gz"
    mesh = Msh.read_hdf5(str(source), LEADFIELD_MESH_PATH)
    selected = np.intersect1d(brain_rows(mesh), mask_image_to_index_mask("elm_center", nib.load(roi_path), mesh, 1))
    if not selected.size:
        raise ValueError("基场网格上的目标实体为空")
    volumes = mesh.elements_volumes_and_areas().value[selected]
    with h5py.File(source, "r") as handle:
        values = handle[LEADFIELD_DATASET]
        attrs = values.attrs
        if any(_decode_name(attrs[key]) != value for key, value in (("current", "1A"), ("field", "E"), ("units", "V/m"), ("d_type", "element_data"))):
            raise ValueError("基场需为 V/m、1 A 的逐元素 E 向量")
        names = tuple(_decode_name(name) for name in attrs["electrode_names"])
        reference = _decode_name(attrs["reference_electrode"])
        if len(names) < 4 or len(set(names)) != len(names) or reference not in names or values.shape != (len(names) - 1, mesh.elm.nr, 3):
            raise ValueError("基场帽位身份、元素排序或头维不合法")
        vectors = np.zeros((len(names), selected.size, 3))
        for index, name in enumerate(item for item in names if item != reference):
            vectors[names.index(name)] = values[index, selected, :] * 0.001
    if not np.isfinite(vectors).all() or not np.all(np.isfinite(volumes) & (volumes > 0)):
        raise ValueError("基场目标域包含非有限场或无效体积")
    evidence = [artifact(case.study, row["case_id"], role, path) for role, path in (("basis", source), ("basis_manifest", manifest), ("basis_roi", roi_path))]
    if evidence[0]["sha256"] != data["leadfield"]["sha256"]:
        raise ValueError("基场文件散列与缓存清单不一致")
    artifacts.extend(evidence)
    return names, vectors, volumes, content_id({"basis_sha256": evidence[0]["sha256"], "roi_sha256": evidence[-1]["sha256"], "elements": selected.tolist()})


def enumerate_upper(names: tuple[str, ...], vectors: FloatArray, volumes: FloatArray, domain: str, batches: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按位集的加权交集计算全部互不共享电极的布局上界。

    Parameters
    ----------
    names, vectors, volumes : tuple and ndarray
        帽位名称、目标网格中的 1 mA 单极基场及个体靶体积。
    domain : str
        已登记的模型、帽位、网格与阈值容差构成的摘要。
    batches : Path
        可在同一 domain 下继续的批次记录。

    Returns
    -------
    summaries, witnesses : tuple of lists
        两档场强与各覆盖要求的完整域上界，及上界布局的精确靶区复算。
    """
    pairs = np.asarray(list(combinations(range(len(names)), 2)), dtype=np.int64)
    count = len(pairs)
    records = read_csv(batches) if batches.exists() else []
    checkpoint = {(float(row["t_vpm"]), int(row["offset"])): row for row in records if row["domain_hash"] == domain}
    summaries = []
    witnesses = []
    for threshold in THRESHOLDS:
        hits = np.array([2 * np.linalg.norm(vectors[a] - vectors[b], axis=1) >= threshold - ATOL_VPM for a, b in pairs], dtype=float)
        stats = []
        for start in range(0, count, BATCH_SIZE):
            previous = checkpoint.get((threshold, start))
            if previous is not None:
                stats.append(previous)
                continue
            tick = monotonic()
            end = min(start + BATCH_SIZE, count)
            left = pairs[start:end]
            legal = (np.arange(count)[None, :] > np.arange(start, end)[:, None]) & (left[:, None, :, None] != pairs[None, :, None, :]).all(axis=(2, 3))
            if not legal.any():
                continue
            bounds = (hits[start:end] * volumes) @ hits.T / volumes.sum()
            bounds = np.clip(bounds, 0, 1)
            selected = np.where(legal, bounds, -1.0)
            i, j = np.unravel_index(selected.argmax(), selected.shape)
            row = dict(domain_hash=domain, t_vpm=threshold, offset=start, stop=end, n_legal=int(legal.sum()),
                       n_pruned_025=int((legal & (bounds < 0.25 - 1e-12)).sum()), n_pruned_050=int((legal & (bounds < 0.50 - 1e-12)).sum()),
                       U=float(bounds[i, j]), first_pair=start + int(i), second_pair=int(j), wall_s=round(monotonic() - tick, 4))
            stats.append(row)
            records.append(row)
            write_csv(batches, records)
        winner = max(stats, key=lambda row: (float(row["U"]), -int(row["first_pair"]), -int(row["second_pair"])))
        a, b = pairs[int(winner["first_pair"])]
        c, d = pairs[int(winner["second_pair"])]
        layout = [names[index] for index in (a, b, c, d)]
        exact = coverage(ti_envelope(vectors[a] - vectors[b], vectors[c] - vectors[d]), volumes, threshold)
        upper = float(winner["U"])
        if exact > upper + 1e-12:
            raise ValueError("上界获选布局的精确靶内场违反弱载波上界")
        total = sum(int(row["n_legal"]) for row in stats)
        expected = 3 * len(names) * (len(names)-1) * (len(names)-2) * (len(names)-3) // 24
        if total != expected:
            raise ValueError(f"合法枚举数 {total} 与理论 {expected} 不一致")
        witnesses.append(dict(domain_hash=domain, t_vpm=threshold, layout=";".join(layout), U_layout=upper,
                              exact_target_coverage=exact, field_source="based_target_only", whole_brain_feasibility="pending"))
        for requirement, pruned in zip(REQUIREMENTS, ("n_pruned_025", "n_pruned_050")):
            summaries.append(dict(domain_hash=domain, t_vpm=threshold, c_required=requirement, n_legal=total, n_evaluated=total,
                                  n_pruned=sum(int(row[pruned]) for row in stats), n_exact_evaluated=1, exhaustive=True, U=upper, L="NA",
                                  upper_layout=";".join(layout), witness_layout="NA", model_id="HDF5_fixed_cap_1mA_1mA",
                                  decision="excluded_in_model" if upper < requirement - 1e-12 else "unresolved_whole_brain"))
    return summaries, witnesses


def evaluate(case: FieldCase, out: Path, artifacts: list[dict[str, str]]) -> ResultRows:
    """登记完整域上界；保持精确靶区试算与全脑可行见证的区别。

    Parameters
    ----------
    case, out, artifacts : FieldCase, Path and list of dict
        03 可追溯基线和运行目录中的证据索引。

    Returns
    -------
    rows, pairs : tuple of lists
        上界是绝对结果，没有跨物理模型配对时第二项为空。
    """
    if case.native["method"] != "leadfield_based":
        return [dict(model_id="common_forward", exhaustive=False, U="NA", L="NA", decision="separate_model",
                     fixed_cap_layout=case.native.get("electrode_names", "NA"), actual_coverage_010=case.native.get("Cov010", "NA"),
                     actual_currents_mA=[case.native.get("IA_mA", "NA"), case.native.get("IB_mA", "NA")])], []
    names, vectors, volumes, basis_id = target_basis(case, artifacts)
    domain = content_id({"basis": basis_id, "names": names, "thresholds": THRESHOLDS, "requirements": REQUIREMENTS,
                         "geometry_rule": "four_distinct_HDF5_electrodes", "tolerance_vpm": ATOL_VPM, "source": digest(Path(__file__))})
    results, witnesses = enumerate_upper(names, vectors, volumes, domain, out / "batches.csv")
    witness_path = out / "witnesses.csv"
    prior = read_csv(witness_path) if witness_path.exists() else []
    write_csv(witness_path, [*(row for row in prior if row["domain_hash"] != domain), *witnesses])
    return results, []


def write_bounds_figure(study: Path, out: Path) -> None:
    """显示各完整域上界与预登记覆盖要求，缺下界保持空白。

    Parameters
    ----------
    study, out : Path
        研究根及含覆盖上界表的当前目录。
    """
    rows = [row for row in read_csv(out / "coverage_bounds.csv") if row["exhaustive"] == "True"]
    if not rows:
        return
    figure = Figure(figsize=(8, 4.7), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    keys = list(dict.fromkeys((row["subject"], row["target"], row["t_vpm"]) for row in rows))
    values = [float(next(row["U"] for row in rows if (row["subject"], row["target"], row["t_vpm"]) == key)) for key in keys]
    ax.scatter(values, range(len(values)), label="Exhaustive fixed-cap model upper bound", color="#186e91")
    for coverage_requirement in REQUIREMENTS:
        ax.axvline(coverage_requirement, linestyle="--", color="#737373", linewidth=0.8)
    ax.set(xlabel="Target coverage fraction", xlim=(-0.03, 1.03), yticks=range(len(keys)),
           yticklabels=[f"{subject} {target} / {t} V/m" for subject, target, t in keys])
    ax.legend(fontsize=8)
    path = out / "figures/bounds.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145)
    manifest = out / "artifacts.csv"
    write_csv(manifest, [*read_csv(manifest), *(artifact(study, "run", "bound_output", item) for item in (path, out / "batches.csv", out / "witnesses.csv"))])


def main() -> None:
    """从仓库根运行 23 的完整帽位域上界扫描。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    targets = ("stn295",) if args.stage == "pilot" else PILOT_TARGETS
    columns = ("domain_hash", "t_vpm", "c_required", "n_legal", "n_evaluated", "n_pruned", "exhaustive", "U", "L", "witness_layout", "model_id")
    config = {"current_mA": [1.0, 1.0], "thresholds_vpm": THRESHOLDS, "coverage_requirements": REQUIREMENTS,
              "amplitude_tolerance_vpm": ATOL_VPM, "target_only_exact_witnesses_per_threshold": 1}
    run_diagnostic(study, "23", args.run_id, args.baseline_id, args.stage, targets, evaluate, "coverage_bounds.csv", columns, config)
    write_bounds_figure(study, study / "results/23" / args.run_id)


if __name__ == "__main__":
    main()
