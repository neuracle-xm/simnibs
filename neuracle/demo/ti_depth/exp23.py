"""枚举同一固定帽位基场域的 TI 靶内弱载波覆盖上界。"""

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from time import monotonic
from typing import Any

from filelock import FileLock
import h5py
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS, TARGETS
from neuracle.demo.ti_depth.exp15 import input_id
from neuracle.ti_depth_study.baseline import content_id
from neuracle.ti_depth_study.diagnostics import FieldCase, ResultRows, run_diagnostic
from neuracle.ti_depth_study.fields import align_element_values, brain_rows, load_geometry
from neuracle.ti_depth_study.metrics import FloatArray, IntArray, coverage, measure_field, ti_envelope
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, digest, read_csv, write_csv
from neuracle.ti_depth_study.restart_budget import BASED_SETTINGS, CURRENT_MA
from neuracle.ti_leadfield_optimization.leadfield import LEADFIELD_DATASET, LEADFIELD_MESH_PATH, _decode_name
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.utils.region_of_interest import mask_image_to_index_mask


THRESHOLDS = (0.10, 0.20)
REQUIREMENTS = (0.25, 0.50)
ATOL_VPM = 1e-8
BATCH_SIZE = 32
MODEL = "HDF5_fixed_cap_1mA_1mA"
TARGET_GROUPS = {key: group for key, _, _, _, group in TARGETS}


@dataclass(frozen=True)
class Basis:
    """固定电流枚举和全脑见证共用的 HDF5 基场与具体 ROI。"""

    names: tuple[str, ...]
    reference: str
    path: Path
    manifest: Path
    cache_key: str
    mesh: Msh
    selected: IntArray
    vectors: FloatArray
    volumes: FloatArray
    identifier: str


def target_basis(case: FieldCase, artifacts: list[dict[str, str]]) -> Basis:
    """提取 HDF5 实际目标四面体上各帽位的 1 mA 单位场。

    Parameters
    ----------
    case, artifacts : FieldCase and list of dict
        已知个体靶区和 based 原生任务以及证据索引。

    Returns
    -------
    Basis
        完整帽位、同一 HDF5 的靶内场和用于全脑复核的网格身份。
    """
    row = case.native
    task = case.study / "simulations" / row["inverse_task"]
    ref = task / "leadfield_reference.json"
    reference_data = json.loads(ref.read_text(encoding="utf-8"))
    manifest = Path(reference_data["manifest_path"])
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["cache_key"] != reference_data["cache_key"] or data["configuration"]["mesh"]["sha256"] != row["mesh_hash"]:
        raise ValueError("HDF5 缓存与 03 的基场或头模身份不符")
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
    identifier = content_id({"basis_sha256": evidence[0]["sha256"], "roi_sha256": evidence[-1]["sha256"], "elements": selected.tolist()})
    return Basis(names, reference, source, manifest, data["cache_key"], mesh, selected, vectors, volumes, identifier)


def verify_whole_brain(case: FieldCase, basis: Basis) -> IntArray:
    """证明参考 WM/GM 与 HDF5 元素逐个匹配，并核对靶掩膜与体积。

    Parameters
    ----------
    case, basis : FieldCase and Basis
        03 的参考评估域与建界的基场。

    Returns
    -------
    ndarray
        按参考几何顺序索引 HDF5 元素的零基行号。
    """
    geometry, tissue, _ = load_geometry(case.geometry_path)
    mesh = basis.mesh
    mapped, exact = align_element_values(mesh, np.arange(mesh.elm.nr, dtype=float), geometry, tissue)
    if not exact.all() or len(mapped) != len(brain_rows(mesh)) or len(np.unique(mapped)) != len(mapped):
        raise ValueError("基场 WM/GM 与共同网格不能建立无插值的一对一映射")
    rows = mapped.astype(np.int64)
    volumes = mesh.elements_volumes_and_areas().value[rows]
    if (not np.array_equal(mesh.elm.tag1[rows], tissue) or not np.array_equal(np.isin(rows, basis.selected), case.target)
            or not np.allclose(volumes, geometry.volume_mm3, rtol=1e-12, atol=1e-10)):
        raise ValueError("基场与共同网格的组织、靶内单元或体积不符")
    return rows


def fixed_roc0(case: FieldCase, basis: Basis, out: Path, artifacts: list[dict[str, str]]) -> list[str]:
    """验收 15 的原生固定电流布局、其最佳评价和同一缓存指纹。

    Parameters
    ----------
    case, basis, out, artifacts : FieldCase, Basis, Path and list of dict
        当前病例、建界模型、运行目录及来源指纹索引。

    Returns
    -------
    list of str
        sub1/STN 同条件 15 ROC0 的四个有序帽位。
    """
    if (case.native["subject"], case.native["target"]) != ("sub1", "stn295"):
        raise ValueError("15 尚未登记该头模和靶的固定电流 ROC0 布局")
    donor = case.study / "results/15" / out.name / "search/natural/leadfield_based"
    result_path, trace_path, reference_path = (donor / name for name in ("result.json", "evaluations.csv", "leadfield_reference.json"))
    if not all(path.is_file() for path in (result_path, trace_path, reference_path)):
        raise ValueError("等待 15 的固定电流 based ROC0 布局与评价轨迹")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    configuration = result["configuration"]
    if (result["status"] != "ok" or configuration["method"] != "leadfield_based" or configuration["budget"] is not None
            or result["trace_sha256"] != digest(trace_path)
            or reference["cache_key"] != basis.cache_key or Path(reference["manifest_path"]).resolve() != basis.manifest.resolve()):
        raise ValueError("15 的结果状态、轨迹或基场身份不匹配")
    roi_row = next(row for row in read_csv(case.study / "results/02/run-001/targets.csv") if row["target"] == case.native["target"])
    head = case.study / "head_models" / f"m2m_{case.native['subject']}"
    prepared = {"head": head, "mesh": head / f"{case.native['subject']}.msh", "cap": head / "eeg_positions/EEG10-10_NSN.csv"}
    if configuration["input"] != input_id(prepared, case.native, roi_row):
        raise ValueError("15 的头模、帽位、靶区或研究配置指纹不匹配")
    trials = [row for row in read_csv(trace_path) if row["status"] == "new"]
    best = min(trials, key=lambda row: float(row["objective"]))
    parameters = json.loads(best["parameters"])
    genes = [int(value) for value in parameters]
    if (genes != parameters or len(genes) != 6 or len(set(genes[:4])) != 4 or not all(0 <= gene < len(basis.names) for gene in genes[:4])
            or not all(np.isclose(tick * BASED_SETTINGS.current_step_ma, CURRENT_MA, rtol=0, atol=1e-12) for tick in genes[4:])):
        raise ValueError("15 的最佳评价缺少四个互异帽位和两路各 1 mA 条件")
    labels = [basis.names[gene] for gene in genes[:4]]
    if labels != result["labels"] or float(best["objective"]) != result["best_objective"]:
        raise ValueError("15 的输出布局与最佳固定电流评价不一致")
    artifacts.extend(artifact(case.study, case.native["case_id"], "same_model_roc0", path) for path in (result_path, trace_path, reference_path))
    return labels


def measure_basis_layout(case: FieldCase, basis: Basis, rows: IntArray, labels: list[str]) -> dict[str, Any]:
    """读取四个帽位基场，以 1 mA/路重建并测量同模型全脑场。

    Parameters
    ----------
    case, basis : FieldCase and Basis
        用于定位与外溢的参考几何，以及给出上界的基场。
    rows, labels : ndarray and list of str
        参考网格到 HDF5 的精确元素映射，和四个有序帽位。

    Returns
    -------
    dict
        02 口径的共同几何绝对指标，物理场来自 HDF5。
    """
    if len(labels) != 4 or len(set(labels)) != 4:
        raise ValueError("见证需要四个互异且按通道极性排列的帽位")
    non_reference = {name: index for index, name in enumerate(name for name in basis.names if name != basis.reference)}
    with h5py.File(basis.path, "r") as file:
        data = file[LEADFIELD_DATASET]
        vectors = [np.zeros((len(rows), 3)) if label == basis.reference else data[non_reference[label], :, :][rows] * 0.001 for label in labels]
    return measure_field(case.geometry, case.target, vectors[0] - vectors[1], vectors[2] - vectors[3])


def finite_witnesses(
    case: FieldCase, basis: Basis, out: Path, artifacts: list[dict[str, str]],
    bounds: list[dict[str, Any]], witnesses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只在同一 HDF5 全脑模型里验收有限候选并更新已知可行下界。

    Parameters
    ----------
    case, basis, out, artifacts : FieldCase, Basis, Path and list of dict
        固定病例、建界基场、实验产物目录及可追溯索引。
    bounds, witnesses : list of dict
        完整域上界与精确靶内见证，原地附加全脑证据。

    Returns
    -------
    list of dict
        全脑候选与同层 ROC0 的成对差值及门槛。
    """
    rows = verify_whole_brain(case, basis)
    candidates = {witness["layout"]: dict(candidate_source="upper_layout", labels=witness["layout"].split(";")) for witness in witnesses}
    reason = "NA"
    reference_layout = None
    try:
        labels = fixed_roc0(case, basis, out, artifacts)
        reference_layout = ";".join(labels)
        candidates[reference_layout] = dict(candidate_source="ROC0" if reference_layout not in candidates else "ROC0;upper_layout", labels=labels)
    except (OSError, RuntimeError, ValueError, KeyError, StopIteration) as error:
        reason = str(error)
    for candidate in candidates.values():
        candidate["metrics"] = measure_basis_layout(case, basis, rows, candidate["labels"])
    reference = candidates[reference_layout]["metrics"] if reference_layout is not None else None
    pairs = []
    for layout, candidate in candidates.items():
        metrics = candidate["metrics"]
        comparison = compare_endpoints(metrics, reference, TARGET_GROUPS[case.native["target"]]) if reference is not None else {}
        gate = comparison.get("decision", "undetermined")
        candidate["feasible"] = gate not in {"undetermined", "failed", "zero_coverage"} and metrics["Cov010"] > 0
        if reference is not None and layout != reference_layout:
            case_id = case.native["case_id"]
            pairs.append(dict(case_id=f"{case_id}:{layout}", reference_case_id=f"{case_id}:{reference_layout}",
                              arm=candidate["candidate_source"], reference_arm="ROC0", seed="NA", condition="1+1mA", field_source=MODEL) | comparison)
        info = dict(candidate_source=candidate["candidate_source"], whole_brain_feasibility=gate, gate_reason=comparison.get("reason", reason),
                    feasible=candidate["feasible"], whole_brain_rows=len(rows), reference_layout=reference_layout or "NA",
                    localization_rule="02_relative_to_fixed_ROC0", **{key: metrics.get(key, "NA") for key in (
                        "D_hot_mm", "Q50_T", "Cov010", "Cov020", "Q99_Cout", "Vout010", "absolute_hit")})
        related = [witness for witness in witnesses if witness["layout"] == layout]
        if not related:
            related = [dict(domain_hash=bounds[0]["domain_hash"], t_vpm="NA", layout=layout, field_source=MODEL)]
            witnesses.extend(related)
        for witness in related:
            if witness["t_vpm"] != "NA":
                exact = metrics["Cov010" if witness["t_vpm"] == 0.10 else "Cov020"]
                if not np.isclose(exact, witness["exact_target_coverage"], rtol=0, atol=1e-12):
                    raise ValueError("同一 HDF5 布局的靶内和全脑覆盖不一致")
            witness.update(info, field_source=MODEL)
    for threshold, metric in zip(THRESHOLDS, ("Cov010", "Cov020")):
        rows_at_threshold = [row for row in bounds if row["t_vpm"] == threshold]
        if reference is not None and reference["status"] == "ok":
            rows_at_threshold.append(rows_at_threshold[0] | dict(c_required=max(0., reference[metric] - 0.05), requirement="ROC0_retention", n_pruned="NA"))
            bounds.append(rows_at_threshold[-1])
        best = max(((candidate["metrics"][metric], layout) for layout, candidate in candidates.items() if candidate["feasible"]), default=None)
        for row in rows_at_threshold:
            row.update(requirement=row.get("requirement", "registered_absolute"), finite_whole_brain_candidates=len(candidates),
                       feasible_whole_brain_candidates=sum(item["feasible"] for item in candidates.values()),
                       gating_reference=reference_layout or "NA", lower_bound_reason=reason if reference is None else "有限候选未满足全脑门槛")
            if best is not None:
                if best[0] > row["U"] + 1e-12:
                    raise ValueError("同模型可行见证超过完整域上界")
                row.update(L=best[0], witness_layout=best[1], lower_bound_reason="NA")
            if row["U"] < row["c_required"] - 1e-12:
                row["decision"] = "excluded_in_model"
            elif best is not None and best[0] >= row["c_required"] - 1e-12:
                row["decision"] = "witnessed_in_model"
            else:
                row["decision"] = "unresolved_finite_witnesses" if reference is not None else "unresolved_missing_ROC0"
    return pairs


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
        if upper > 1.0 - 1e-12:
            upper = 1.0
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
                                  upper_layout=";".join(layout), witness_layout="NA", model_id=MODEL,
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
    native = case.native
    common = dict(model_id="common_forward", exhaustive=False, U="NA", L="NA", decision="separate_model",
                  fixed_cap_layout=native.get("electrode_names", "NA"), actual_coverage_010=native.get("Cov010", "NA"),
                  actual_coverage_020=native.get("Cov020", "NA"), actual_currents_mA=[native.get("IA_mA", "NA"), native.get("IB_mA", "NA")])
    if native["method"] != "leadfield_based":
        return [common], []
    basis = target_basis(case, artifacts)
    domain = content_id({"basis": basis.identifier, "names": basis.names, "thresholds": THRESHOLDS, "requirements": REQUIREMENTS,
                         "geometry_rule": "four_distinct_HDF5_electrodes", "tolerance_vpm": ATOL_VPM, "source": digest(Path(__file__))})
    results, witnesses = enumerate_upper(basis.names, basis.vectors, basis.volumes, domain, out / "batches.csv")
    pairs = []
    try:
        pairs = finite_witnesses(case, basis, out, artifacts, results, witnesses)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        for row in results:
            row.update(L="NA", witness_layout="NA", lower_bound_reason=str(error),
                       decision="excluded_in_model" if row["U"] < row["c_required"] - 1e-12 else "unresolved_whole_brain")
    witness_path = out / "witnesses.csv"
    prior = read_csv(witness_path) if witness_path.exists() else []
    data = [*(row for row in prior if row["domain_hash"] != domain), *witnesses]
    write_csv(witness_path, data, tuple(dict.fromkeys(key for row in data for key in row)))
    return [*results, common], pairs


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
    displayed = [next(row for row in rows if (row["subject"], row["target"], row["t_vpm"]) == key) for key in keys]
    values = [float(row["U"]) for row in displayed]
    ax.scatter(values, range(len(values)), label="Exhaustive fixed-cap model upper bound", color="#186e91")
    lower = [(index, float(row["L"])) for index, row in enumerate(displayed) if row["L"] != "NA"]
    if lower:
        ax.hlines([index for index, _ in lower], [value for _, value in lower], [values[index] for index, _ in lower], color="#627e87")
        ax.scatter([value for _, value in lower], [index for index, _ in lower], label="Feasible same-model witness", color="#167366")
    for coverage_requirement in REQUIREMENTS:
        ax.axvline(coverage_requirement, linestyle="--", color="#737373", linewidth=0.8)
    ax.set(xlabel="Target coverage fraction", xlim=(-0.03, 1.03), yticks=range(len(keys)),
           yticklabels=[f"{subject} {target} / {t} V/m" for subject, target, t in keys])
    ax.legend(fontsize=8)
    path = out / "figures/bounds.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145)
    report = out / "report.md"
    content = ["\n## 固定帽位 HDF5 全脑见证\n", "上界 U 来自四个互异帽位的完整枚举。已知可行下界 L 从有限候选取得，"
               "按同模型固定电流 ROC0 的共同协议相对保持线、外溢及正覆盖判断。绝对命中单列。两条路线的共同 FEM 读数使用各自表中登记的模型和电流。\n"]
    for row in displayed:
        content.append(f"- {row['subject']}/{row['target']}，{row['t_vpm']} V/m，L {row['L']}，U {row['U']}。"
                       f"见证布局 `{row['witness_layout']}`。")
    content.extend(("", "[全脑见证](witnesses.csv) · [上界图](figures/bounds.png)", ""))
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(content))
    manifest = out / "artifacts.csv"
    write_csv(manifest, [*(row for row in read_csv(manifest) if row["path"] != report.relative_to(study).as_posix()),
                        *(artifact(study, "run", "bound_output", item) for item in (report, path, out / "batches.csv", out / "witnesses.csv"))])


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
              "amplitude_tolerance_vpm": ATOL_VPM, "upper_witnesses_per_threshold": 1,
              "same_model_reference": f"15/{args.run_id}/search/natural/leadfield_based", "full_brain_gate": "02_relative_to_fixed_ROC0"}
    out = study / "results/23" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".scheduler.lock")):
        run_diagnostic(study, "23", args.run_id, args.baseline_id, args.stage, targets, evaluate, "coverage_bounds.csv", columns, config)
        write_bounds_figure(study, out)


if __name__ == "__main__":
    main()
