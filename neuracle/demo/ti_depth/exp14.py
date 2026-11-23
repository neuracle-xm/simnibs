"""用共同单层 FEM 的向量场分离电流配比与换帽位效应。"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.demo.ti_depth.exp03 import FORWARD_OUTPUTS, align_common_forward
from neuracle.ti_depth_study.baseline import FIELD_MODEL, content_id, forward_config, run_task
from neuracle.ti_depth_study.diagnostics import FieldCase, ResultRows, run_diagnostic
from neuracle.ti_depth_study.metrics import FloatArray, measure_field, ti_envelope
from neuracle.ti_depth_study.pool import compare_endpoints
from neuracle.ti_depth_study.records import artifact, read_csv, write_csv


RATIOS = ((0.5, 1.5), (1.0, 1.0), (1.5, 0.5), (1.0, 3.0), (2.0, 2.0), (3.0, 1.0))
DIRECT = (0.5, 1.5)


def scale_carriers(e_a: FloatArray, e_b: FloatArray, reference: tuple[float, float], requested: tuple[float, float]) -> tuple[FloatArray, FloatArray]:
    """分别换算同几何、同物性和同边界条件下的两路载波。

    Parameters
    ----------
    e_a, e_b : ndarray
        已知非零幅值时的两路单频场 E，单位 V/m。
    reference, requested : tuple of float
        每路的源参考电流和新的仿真幅值，单位 mA。

    Returns
    -------
    tuple of ndarray
        按各自电流比例缩放后的 E_A、E_B。包络需要从两路重算。
    """
    if not all(np.isfinite(current) and current > 0 for current in (*reference, *requested)):
        raise ValueError("线性重标定需要两路正且有限的参考与目标电流")
    return e_a * (requested[0] / reference[0]), e_b * (requested[1] / reference[1])


def verify_direct(case: FieldCase, out: Path, layout_id: str, current: tuple[float, float], vectors: tuple[FloatArray, FloatArray], artifacts: list[dict[str, str]]) -> dict[str, Any]:
    """独立重算抽查配比，比较载波及包络的实际误差。

    Parameters
    ----------
    case : FieldCase
        原生路线的具名解、共同网格及场。
    out : Path
        14 运行目录，命令将被登记在其中。
    layout_id, current : str and tuple
        物理配置标识和待独立复算的 A/B 幅值。
    vectors : tuple of ndarray
        由原场按电流重标定的两路向量。
    artifacts : list of dict
        本次运行的产物表。

    Returns
    -------
    dict
        两路向量和包络误差，及用于恢复直接 FEM 的任务引用。
    """
    ia, ib = current
    config = forward_config(case.native["electrode_names"].split(";"), ([ia, -ia], [ib, -ib]))
    task = run_task(case.study, "ti_forward", case.native["subject"], f"14_{out.name}_{layout_id}_asym",
                    config, out / "commands.ps1", FORWARD_OUTPUTS)
    field_path, alignment = align_common_forward(task, case.geometry_path, case.geometry_path.parent / "fields")
    if alignment["missing_mm3"] > 0:
        raise ValueError("非对称配比独立正向场存在无法对应的参考元素")
    with np.load(field_path) as direct:
        reference = ti_envelope(*vectors)
        error = direct["max_TI"] - reference
        volume = case.geometry.volume_mm3
        rms = float(np.sqrt(np.average(error**2, weights=volume)))
        rms_a = float(np.sqrt(np.average(np.sum((direct["E_A"] - vectors[0])**2, axis=1), weights=volume)))
        rms_b = float(np.sqrt(np.average(np.sum((direct["E_B"] - vectors[1])**2, axis=1), weights=volume)))
        peak = float(np.max(np.abs(error)))
    artifacts.append(artifact(case.study, case.native["case_id"], "direct_ratio_field", field_path))
    artifacts.append(artifact(case.study, case.native["case_id"], "direct_ratio_command", task / "study_execution.json"))
    return dict(check_rms_vpm=rms, check_peak_error_vpm=peak, check_carrier_A_rms_vpm=rms_a,
                check_carrier_B_rms_vpm=rms_b, direct_task=task.name)


def evaluate(case: FieldCase, out: Path, artifacts: list[dict[str, str]]) -> ResultRows:
    """在冻结帽位上扫两档总电流与各自三种 A/B 配比。

    Parameters
    ----------
    case, out, artifacts : FieldCase, Path and list of dict
        同源向量场、当前运行目录及证据索引。

    Returns
    -------
    rows, pairs : tuple of lists
        六个条件的绝对值和各固定总量内相对 1:1 的差值。
    """
    source = case.native
    reference = (float(source["IA_mA"]), float(source["IB_mA"]))
    geometry = case.geometry
    if not case.valid.all():
        raise ValueError("共同场对齐仍有缺失，不能比较头模内部热点位移")
    case_id = source["case_id"]
    layout_id = content_id({"subject": source["subject"], "mesh": source["mesh_hash"], "layout": source["electrode_names"], "model": FIELD_MODEL})[:20]
    common = dict(layout_id=layout_id, method_origin=source["method"], electrodes=source["electrode_names"], field_source="scaled_common_forward",
                  source_field_id=case.field_path.stem, source_IA_mA=reference[0], source_IB_mA=reference[1])
    rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        vectors = scale_carriers(case.e_a, case.e_b, reference, ratio)
        measured = measure_field(geometry, case.target, *vectors, case.valid)
        a, b = ratio
        coord = measured.get("hot_center_mm", ["NA", "NA", "NA"])
        condition = f"a{a:g}_b{b:g}"
        row = common | measured | dict(IA_mA=a, IB_mA=b, total_mA=a+b, condition=condition, ratio_case_id=f"{case_id}:{condition}",
                                       x_hot_mm=coord[0], y_hot_mm=coord[1], z_hot_mm=coord[2], check_rms_vpm="NA")
        if ratio == DIRECT and source["target"] == "stn295":
            row.update(verify_direct(case, out, layout_id, ratio, vectors, artifacts))
        rows.append(row)
    for a, b in (((1.0, 1.0), (2.0, 2.0)), ((0.5, 1.5), (1.0, 3.0)), ((1.5, 0.5), (3.0, 1.0))):
        small = ti_envelope(*scale_carriers(case.e_a, case.e_b, reference, a))
        large = ti_envelope(*scale_carriers(case.e_a, case.e_b, reference, b))
        if not np.allclose(large, 2 * small, rtol=1e-10, atol=1e-10):
            raise ValueError("两路同时成比例时的包络缩放复核失败")
    group = "cortical" if source["target"] == "a4ul58" else "subcortical"
    paired = []
    for row in rows:
        even = next(item for item in rows if item["total_mA"] == row["total_mA"] and item["IA_mA"] == item["IB_mA"])
        if even is row:
            continue
        compared = compare_endpoints(row, even, group)
        shift = np.linalg.norm(np.asarray(row["hot_center_mm"]) - np.asarray(even["hot_center_mm"])) if row["status"] == even["status"] == "ok" else "NA"
        paired.append(dict(case_id=row["ratio_case_id"], reference_case_id=even["ratio_case_id"], arm=row["condition"], reference_arm=even["condition"],
                           field_source=row["field_source"], total_mA=row["total_mA"], layout_id=layout_id,
                           hot_center_shift_mm=shift, **compared))
    return rows, paired


def plot_ratios(study: Path, out: Path) -> None:
    """在个体坐标中按总量分开绘制两路布局的主热点位置。

    Parameters
    ----------
    study, out : Path
        研究根和 14 的运行目录。
    """
    rows = [row for row in read_csv(out / "ratio_sweep.csv") if row["status"] == "ok"]
    if not rows:
        return
    key = (rows[0]["subject"], rows[0]["target"])
    selected = [row for row in rows if (row["subject"], row["target"]) == key]
    figure = Figure(figsize=(10, 4.3), layout="constrained")
    FigureCanvasAgg(figure)
    for ax, total in zip(figure.subplots(1, 2, sharex=True, sharey=True), (2.0, 4.0)):
        for method in dict.fromkeys(row["method"] for row in selected):
            subset = [row for row in selected if row["method"] == method and float(row["total_mA"]) == total]
            x = [float(row["x_hot_mm"]) for row in subset]
            y = [float(row["y_hot_mm"]) for row in subset]
            ax.plot(x, y, marker="o", label=method)
            for x1, y1, row in zip(x, y, subset):
                ax.annotate(f"{row['IA_mA']}:{row['IB_mA']}", (x1, y1), fontsize=7, xytext=(3, 4), textcoords="offset points")
        ax.set(xlabel="X (mm)", ylabel="Y (mm)", title=f"Total {total:g} mA | axial projection")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7)
    figure.suptitle(" / ".join(key))
    path = out / "figures/ratio_track.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145)
    table = out / "artifacts.csv"
    write_csv(table, [*read_csv(table), artifact(study, "run", "ratio_plot", path)])


def main() -> None:
    """从仓库根运行 14 的固定布局配比诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    keys = ("layout_id", "method_origin", "IA_mA", "IB_mA", "total_mA", "field_source", "x_hot_mm", "y_hot_mm", "z_hot_mm", "D_hot_mm", "Cov010", "check_rms_vpm")
    targets = ("stn295",) if args.stage == "pilot" else PILOT_TARGETS
    run_diagnostic(study, "14", args.run_id, args.baseline_id, args.stage, targets, evaluate,
                   "ratio_sweep.csv", keys, {"ratios_mA": RATIOS, "direct_check_mA": DIRECT, "field_model": FIELD_MODEL})
    plot_ratios(study, study / "results/14" / args.run_id)


if __name__ == "__main__":
    main()
