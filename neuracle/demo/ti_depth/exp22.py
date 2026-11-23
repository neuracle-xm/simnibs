"""分开比较 TI 连续映射、电极模型和基场重建的误差。"""

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.ti_depth_study.diagnostics import FieldCase, ResultRows, run_diagnostic
from neuracle.ti_depth_study.fields import align_carrier, align_element_values, load_geometry
from neuracle.ti_depth_study.metrics import FloatArray, IntArray, measure_field, ti_envelope
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv
from neuracle.ti_leadfield_optimization.leadfield import LEADFIELD_DATASET, LEADFIELD_MESH_PATH, _decode_name
from simnibs.mesh_tools.mesh_io import Msh, read_msh


Vectors = tuple[FloatArray, FloatArray]
PROVENANCE = {"F1": ("free_continuous_double_layer", "free_mapped_double_layer"),
              "F2": ("free_mapped_double_layer", "common_mapped_single_layer"),
              "B1": ("based_reconstruction_single_layer", "common_mapped_single_layer")}


def residual(field: FloatArray, reference: FloatArray, volumes: FloatArray) -> tuple[float, float | str]:
    """计算逐元素体积加权绝对 RMS 及以参照场能量归一的相对 RMS。

    Parameters
    ----------
    field, reference : ndarray
        同一空间域内的标量包络或单频向量。
    volumes : ndarray
        对应四面体的正体积。仅包括两份场均匹配的单元。

    Returns
    -------
    abs_rms, relative_rms : tuple
        绝对值单位 V/m。参照为零能量时相对值为 NA。
    """
    if not len(volumes):
        return np.nan, "NA"
    squared_error = np.square((field - reference).reshape(len(field), -1)).sum(axis=1)
    squared_reference = np.square(reference.reshape(len(reference), -1)).sum(axis=1)
    mse = float(np.average(squared_error, weights=volumes))
    energy = float(np.average(squared_reference, weights=volumes))
    return float(np.sqrt(mse)), float(np.sqrt(mse / energy)) if energy else "NA"


def based_carriers(case: FieldCase, task: Path, mapping: dict[str, Any], tags: IntArray, artifacts: list[dict[str, str]]) -> Vectors:
    """只读载入 HDF5 内所选具名电极的 1 A 行，并按各自 mA 重建。

    Parameters
    ----------
    case, task : FieldCase and Path
        共同参考网格和 based 的原生任务。
    mapping, tags, artifacts : dict, ndarray and list
        具名电极电流、参考组织编号及产物索引。

    Returns
    -------
    tuple of ndarray
        对齐到共同网格的两路有符号 V/m 向量。
    """
    pointer = task / "leadfield_reference.json"
    ref = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = Path(ref["manifest_path"])
    cache = json.loads(manifest.read_text(encoding="utf-8"))
    binary = manifest.parent / cache["leadfield"]["filename"]
    mesh = Msh.read_hdf5(str(binary), LEADFIELD_MESH_PATH)
    labels = mapping["mapped_labels"]
    with h5py.File(binary, "r") as handle:
        ds = handle[LEADFIELD_DATASET]
        attrs = ds.attrs
        if any(_decode_name(attrs[key]) != value for key, value in (("current", "1A"), ("field", "E"), ("d_type", "element_data"), ("units", "V/m"))):
            raise ValueError("HDF5 载波属性与 1 A 逐元素三维向量约定不符")
        names = [_decode_name(name) for name in attrs["electrode_names"]]
        reference = _decode_name(attrs["reference_electrode"])
        if reference not in names or len(set(names)) != len(names) or ds.shape != (len(names) - 1, mesh.elm.nr, 3):
            raise ValueError("HDF5 电极身份或元素顺序与缓存内容不符")
        indices = {name: index for index, name in enumerate(item for item in names if item != reference)}
        unit = {name: (np.asarray(ds[indices[name]], dtype=np.float64) if name != reference else np.zeros(ds.shape[1:])) for name in labels}
    vectors = []
    for first, second, current in ((0, 1, mapping["current_A"][0]), (2, 3, mapping["current_B"][0])):
        value = (unit[labels[first]] - unit[labels[second]]) * (current / 1000)
        vectors.append(align_element_values(mesh, value, case.geometry, tags)[0])
    artifacts.extend(artifact(case.study, case.native["case_id"], "leadfield_source", path) for path in (pointer, manifest, binary))
    return vectors[0], vectors[1]


def free_carriers(case: FieldCase, directory: Path, folders: tuple[str, str], tags: IntArray, artifacts: list[dict[str, str]]) -> Vectors:
    """按通道子目录读取并对齐 free 终算的原始单频 MSH。

    Parameters
    ----------
    case, directory : FieldCase and Path
        共同网格和 free 任务根。
    folders, tags, artifacts : tuple, ndarray and list
        A/B 子目录、参考组织行及产物目录。

    Returns
    -------
    tuple of ndarray
        共同元素顺序的有符号单频向量，有缺失时保留 NaN。
    """
    aligned = []
    for folder in folders:
        source = list((directory / folder).glob("*_TDCS_1_scalar.msh"))
        if len(source) != 1:
            raise FileNotFoundError(f"{directory / folder} 单频终算网格数量为 {len(source)}")
        artifacts.append(artifact(case.study, case.native["case_id"], "free_carrier", source[0]))
        aligned.append(align_carrier(read_msh(source[0]), case.geometry, tags)[0])
    return aligned[0], aligned[1]


def compare(case: FieldCase, pair: str, source: Vectors, reference: Vectors) -> ResultRows:
    """按三个区域分别比较两个载波和 max_TI，保留共同有效体积。

    Parameters
    ----------
    case : FieldCase
        固定参考脑域和解剖目标。
    pair : str
        F1、F2 或 B1。
    source, reference : tuple of ndarray
        配对起始场与该对指定的参照场，均已对齐。

    Returns
    -------
    rows, endpoints : tuple
        各区域各字段的 RMS 及沿配对方向的绝对终点差值。
    """
    geometry = case.geometry
    source_f, reference_f = ti_envelope(*source), ti_envelope(*reference)
    valid = np.isfinite(source_f) & np.isfinite(reference_f)
    first, second = PROVENANCE[pair]
    initial = measure_field(geometry, case.target, *source, valid)
    final = measure_field(geometry, case.target, *reference, valid)
    case_id = case.native["case_id"]
    group = "cortical" if case.native["target"] == "a4ul58" else "subcortical"
    change = compare_endpoints(final, initial, group)
    ranges = {"target": case.target, "outside_cortex": geometry.cortex_mask & ~case.target, "whole_brain": np.ones(len(valid), dtype=bool)}
    rows = []
    for region, region_mask in ranges.items():
        matched = region_mask & valid
        volume = geometry.volume_mm3[matched]
        missing = float(geometry.volume_mm3[region_mask & ~valid].sum())
        for label, left, right in zip(("E_A", "E_B", "max_TI"), (*source, source_f), (*reference, reference_f)):
            absolute, relative = residual(left[matched], right[matched], volume)
            rows.append(dict(pair=pair, field_name=label, region=region, source_model=first, reference_model=second,
                             source_case_id=f"{case_id}:{first}", reference_case_id=f"{case_id}:{second}",
                             abs_rms_vpm=absolute, relative_rms=relative, matched_mm3=float(volume.sum()), missing_mm3=missing,
                             delta_D_hot_mm=change.get("delta_D_hot_mm", "NA"), delta_Cov010_pp=change.get("delta_Cov010_pp", "NA"),
                             rms_over_10pct=isinstance(relative, float) and relative > 0.1, decision=change["decision"]))
    endpoints = [dict(case_id=f"{case_id}:{second}", reference_case_id=f"{case_id}:{first}", arm=pair, reference_arm=first,
                      field_source=second, **change)]
    return rows, endpoints


def plot_pair(case: FieldCase, out: Path, pair: str, source: Vectors, reference: Vectors, artifacts: list[dict[str, str]]) -> None:
    """以共享色标绘制相同个体轴向位置的包络及带符号差场。

    Parameters
    ----------
    case, out, pair : FieldCase, Path and str
        参考病例、当前运行目录及配对类型。
    source, reference, artifacts : tuple, tuple and list
        两份载波场及证据索引。
    """
    points = case.geometry.centroid_mm
    z = np.average(points[case.target, 2], weights=case.geometry.volume_mm3[case.target])
    section = np.abs(points[:, 2] - z) <= 1
    initial, final = (ti_envelope(*vectors)[section] for vectors in (source, reference))
    span = float(np.nanmax([initial, final], initial=0.0))
    change = final - initial
    delta = max(1e-12, float(np.nanmax(np.abs(change), initial=0.0)))
    figure = Figure(figsize=(11, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharex=True, sharey=True)
    for axis, values, title, cmap, low, high in zip(axes, (initial, final, change), (*PROVENANCE[pair], "Reference − source"),
                                                    ("inferno", "inferno", "coolwarm"), (0, 0, -delta), (span, span, delta)):
        values = np.ma.masked_invalid(values)
        panel = axis.scatter(points[section, 0], points[section, 1], c=values, s=6, cmap=cmap, vmin=low, vmax=high)
        target = section & case.target
        axis.scatter(points[target, 0], points[target, 1], s=15, facecolors="none", edgecolors="#38b6af", linewidths=0.55)
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", title=title.replace("_", " "))
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(panel, ax=axis, label="V/m", shrink=0.7)
    path = out / "figures" / f"{case.native['subject']}_{case.native['target']}_{pair}_paired_fields.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=140)
    artifacts.append(artifact(case.study, case.native["case_id"], "paired_fields", path))


def evaluate(case: FieldCase, out: Path, artifacts: list[dict[str, str]]) -> ResultRows:
    """采集 free 的 F1/F2 或 based 的 B1 共同网格诊断。

    Parameters
    ----------
    case, out, artifacts : FieldCase, Path and list
        已验收的共同正向数据、当前运行目录及产物记录。

    Returns
    -------
    rows, pairs : tuple of lists
        分区域残差和模拟端点差值。
    """
    row = case.native
    directory = case.study / "simulations" / row["inverse_task"]
    path = directory / "electrode_mapping.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping["mapped_labels"] != row["electrode_names"].split(";") or any(mapping[key] != [expected, -expected] for key, expected in
      (("current_A", float(row["IA_mA"])), ("current_B", float(row["IB_mA"])))):
        raise ValueError("具名正向电极或通道电流与逆向映射不一致")
    artifacts.append(artifact(case.study, row["case_id"], "electrode_mapping", path))
    _, tags, _ = load_geometry(case.geometry_path)
    common = case.e_a, case.e_b
    if row["method"] == "leadfield_based":
        inputs = [("B1", based_carriers(case, directory, mapping, tags, artifacts), common)]
    else:
        continuous = free_carriers(case, directory, ("final_sim_0", "final_sim_1"), tags, artifacts)
        mapped = free_carriers(case, directory, ("mapped_electrodes_simulation/mapped_sim_0", "mapped_electrodes_simulation/mapped_sim_1"), tags, artifacts)
        inputs = [("F1", continuous, mapped), ("F2", mapped, common)]
        positions = np.asarray(mapping["optimized_positions"])
        caps = np.asarray(mapping["mapped_positions"])
        distances = np.linalg.norm(caps - positions, axis=1)
        if positions.shape != (4, 3) or caps.shape != (4, 3) or not np.allclose(distances, mapping["distances"], atol=1e-6):
            raise ValueError("free 帽位映射的四个坐标或记录位移不一致")
        offset_info = [dict(case_id=row["case_id"], role=role, label=label, continuous_mm=start.tolist(), mapped_mm=end.tolist(), distance_mm=float(distance))
                       for role, label, start, end, distance in zip(("A+", "A-", "B+", "B-"), mapping["mapped_labels"], positions, caps, distances)]
        mapping_csv = out / "mapping_distances.csv"
        previous = read_csv(mapping_csv) if mapping_csv.exists() else []
        write_csv(mapping_csv, [*(item for item in previous if item["case_id"] != row["case_id"]), *offset_info])
    details, pairs = [], []
    for name, source, reference in inputs:
        data, matched = compare(case, name, source, reference)
        details.extend(data)
        pairs.extend(matched)
        plot_pair(case, out, name, source, reference, artifacts)
    return details, pairs


def main() -> None:
    """从仓库根运行三段场一致性诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    targets = ("stn295",) if args.stage == "pilot" else ("stn295", "a4ul58")
    columns = ("pair", "reference_case_id", "field_name", "region", "matched_mm3", "abs_rms_vpm", "relative_rms", "delta_D_hot_mm", "delta_Cov010_pp")
    run_diagnostic(study, "22", args.run_id, args.baseline_id, args.stage, targets, evaluate, "forward_agreement.csv", columns, {"models": PROVENANCE, "relative_rms_trigger": 0.10})
    out = study / "results/22" / args.run_id
    offsets = out / "mapping_distances.csv"
    if offsets.is_file():
        index = out / "artifacts.csv"
        write_csv(index, [*read_csv(index), artifact(study, "run", "free_mapping_distances", offsets)])


if __name__ == "__main__":
    main()
