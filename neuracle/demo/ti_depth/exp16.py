"""用同一头模核对两条 TI 搜索路线的具名布局编码。"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

from filelock import FileLock
import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist

from neuracle.demo.ti_depth.exp02 import METHODS
from neuracle.demo.ti_depth.exp03 import CASE_COLUMNS, SOLVER_ERRORS
from neuracle.ti_depth_study.metrics import FloatArray
from neuracle.ti_depth_study.records import artifact, digest, read_csv, timestamp, update_index, write_csv, write_json
from neuracle.ti_leadfield_optimization.leadfield import LEADFIELD_DATASET
from simnibs.mesh_tools.mesh_io import read_msh
from simnibs.optimization.tes_flex_optimization.electrode_layout import valid_skin_region
from simnibs.optimization.tes_flex_optimization.ellipsoid import Ellipsoid, ellipsoid2subject
from simnibs.optimization.tes_flex_optimization.tes_flex_optimization import TesFlexOptimization
from simnibs.utils.csv_reader import _get_eeg_positions
from simnibs.utils.mesh_element_properties import ElementTags


ROLES = ("A+", "A−", "B+", "B−")
ROUNDTRIP_MM = 0.001
CLEARANCE_MM = 5.0
SUMMARY_COLUMNS = ("donor_case_id", "receiver", "arm", "valid_encoding", "max_roundtrip_mm", "init_score", "final_score",
                   "donor_s", "receiver_s", "total_s", "case_id", "donor_IA_mA", "donor_IB_mA", "condition_matched",
                   "original_min_center_mm", "encoding_s", "encoded", "reason")
ELECTRODE_COLUMNS = ("case_id", "electrode", "role", "donor_current_mA", "original_xyz_mm", "roundtrip_xyz_mm", "roundtrip_mm",
                     "valid_encoding", "encoded", "reason")


@dataclass(frozen=True)
class Encoding:
    """接收方原生参数、逐电极反算坐标和布局级合法性。"""

    parameters: list[list[float]] | list[int]
    roundtrip: list[FloatArray | None]
    reason: str


def encode_based(
    labels: list[str], positions: FloatArray, names: list[str], cap: dict[str, FloatArray], radius: float,
) -> Encoding:
    """将 A+、A−、B+、B− 转为当前 HDF5 的 GA 基因索引。

    Parameters
    ----------
    labels, positions : list of str, ndarray
        供体映射标签和对应世界坐标。顺序保留极性。
    names, cap : list of str, dict
        实际 HDF5 electrode_names 属性和同一头模的帽位表。
    radius : float
        接收方的圆片半径，单位 mm。

    Returns
    -------
    Encoding
        前四个 GA 基因及反查坐标，其余两电流基因由接收方注册预算指定。
    """
    if len(set(labels)) != 4 or len(set(names)) != len(names):
        raise ValueError("GA 编码要求四个不同标签与无重复的 HDF5 标签索引")
    genes = [names.index(label) for label in labels]
    points = [cap[names[index]] for index in genes]
    reason = "NA"
    if not np.allclose(points, positions, rtol=0, atol=ROUNDTRIP_MM):
        reason = "供体映射坐标与接收帽位不符"
    elif float(pdist(np.asarray(points)).min()) < 2 * radius + CLEARANCE_MM:
        reason = "圆片间距不满足共同几何约束"
    return Encoding(genes, points, reason)


def prepare_free_geometry(mesh_path: Path, radius: float, currents: tuple[list[float], list[float]]) -> TesFlexOptimization:
    """复用原生 free 的可放置头皮、拟合椭球及电极阵列配置。

    Parameters
    ----------
    mesh_path : Path
        接收方的个体头模。
    radius, currents : float, tuple of list
        两组圆片的半径和已登记原生电流，单位 mm 和 mA。

    Returns
    -------
    TesFlexOptimization
        可以调用原生位置解码和合法性检查的几何状态。
    """
    optimizer = TesFlexOptimization()
    mesh = read_msh(mesh_path)
    surface_mesh = mesh if mesh.elm.get_tags(ElementTags.INTERNAL_AIR_TH_SURFACE).any() else mesh.relabel_internal_air()
    optimizer._skin_surface = valid_skin_region(surface_mesh.crop_mesh(tags=ElementTags.SCALP_TH_SURFACE), mesh=surface_mesh,
                                               fn_electrode_mask=optimizer._fn_electrode_mask, additional_distance=0)
    optimizer._node_idx_msh = np.flatnonzero(np.isin(mesh.nodes.node_coord, optimizer._skin_surface.nodes.node_coord).all(axis=1))
    if not np.array_equal(mesh.nodes.node_coord[optimizer._node_idx_msh], optimizer._skin_surface.nodes.node_coord):
        raise ValueError("原生可放置头皮与头模的节点映射不一致")
    optimizer._ellipsoid.fit(points=optimizer._skin_surface.nodes.node_coord)
    for pair in currents:
        layout = optimizer.add_electrode_layout("ElectrodeArrayPair")
        layout.center = [[0, 0]]
        layout.radius = [radius]
        layout.current = [value / 1000 for value in pair]
        layout._prepare()
    optimizer.n_channel_stim = len(optimizer.electrode)
    optimizer.n_ele_free = [len(layout._electrode_arrays) for layout in optimizer.electrode]
    optimizer.electrode_pos = [[np.zeros(2) for _ in range(count)] for count in optimizer.n_ele_free]
    return optimizer


def scalp_to_jacobi(ellipsoid: Ellipsoid, positions: FloatArray) -> FloatArray:
    """使帽位与椭球正交投影同线，按 Jacobi 参数的解析式反解角度。

    Parameters
    ----------
    ellipsoid : Ellipsoid
        接收方拟合的头皮椭球。
    positions : ndarray
        源帽位的世界坐标。

    Returns
    -------
    ndarray
        每个电极的 beta、lambda，单位 rad。
    """
    radii2 = ellipsoid.radii2
    angles = []
    for point in ellipsoid.rot_org2norm(positions):
        def distance_equation(multiplier: float) -> float:
            """计算法向约束下椭球面的隐式残差。"""
            return float(np.sum(radii2 * (point / (radii2 + multiplier)) ** 2) - 1)
        multiplier = brentq(distance_equation, -float(radii2.min()) * (1 - 1e-9), float(np.linalg.norm(ellipsoid.radii * point)))
        x, y, z = point / (radii2 + multiplier)
        angles.append((np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x)))
    return np.asarray(angles)


def encode_free(optimizer: TesFlexOptimization, positions: FloatArray) -> Encoding:
    """编码帽位并调用原生 free 投影和电极合法性检查。

    Parameters
    ----------
    optimizer : TesFlexOptimization
        已准备接收方个体头皮、电极与椭球几何的实例。
    positions : ndarray
        A+、A−、B+、B− 的同头模帽位坐标。

    Returns
    -------
    Encoding
        每极 Jacobi 参数、反算坐标及原生检查结果。只有合法布局可作为初值。
    """
    ellipsoid = optimizer._ellipsoid
    angles = scalp_to_jacobi(ellipsoid, positions)
    spherical = ellipsoid.cartesian2ellipsoid(ellipsoid.jacobi2cartesian(angles))
    points = []
    for angle in spherical:
        _, hits = ellipsoid2subject(angle, ellipsoid, optimizer._skin_surface)
        points.append(hits[0] if len(hits) == 1 else None)
    native = optimizer.get_nodes_electrode(optimizer.get_electrode_pos_from_array(angles.ravel()))
    if isinstance(native, tuple):
        reason = native[0]
    elif any(point is None for point in points):
        reason = "头皮投影交点缺失或不唯一"
    elif not np.allclose(np.linalg.norm(np.asarray(points) - positions, axis=1), 0, rtol=0, atol=ROUNDTRIP_MM):
        reason = "反算帽位坐标超过数值误差容限"
    else:
        reason = "NA"
    return Encoding(angles.tolist(), points, reason)


def run(study: Path, out: Path, baseline_id: str, run_id: str) -> None:
    """用 sub1/STN 原生两路产物实测双向编码，冻结几何证据。

    Parameters
    ----------
    study, out : Path
        研究根和本篇输出目录。
    baseline_id, run_id : str
        上游 03 和当前运行编号。
    """
    commands = out / "commands.ps1"
    invocation = " ".join("'" + part.replace("'", "''") + "'" for part in sys.argv[1:])
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"\n# {timestamp()} {Path.cwd()}\nconda activate simnibs_env\npython -m neuracle.demo.ti_depth.exp16 {invocation}\n")
    baseline = study / "results/03" / baseline_id
    source = baseline / "baseline_fields.csv"
    subject, target = "sub1", "stn295"
    cases = [dict(case_id=f"{subject}_{target}_{donor}_to_{receiver}", subject=subject, target=target, method=receiver,
                  arm="encoding_preflight", seed="NA", condition="native_geometry", task_id="NA", status="pending",
                  reason="等待双向编码", wall_s="NA") for donor, receiver in (METHODS, METHODS[::-1])]
    write_csv(out / "cases.csv", cases, CASE_COLUMNS)
    info = dict(experiment="16", stage="pilot", run_id=run_id, environment=sys.executable, upstream_runs=[f"03/{baseline_id}"],
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), data_id={},
                config=dict(arm="encoding_preflight", recipient_roundtrip_mm=ROUNDTRIP_MM, based_clearance_mm=CLEARANCE_MM,
                            required_comparison_current_mA=[1.0, 1.0]), started_at=timestamp(), finished_at=None, status="running")
    write_json(out / "run.json", info)
    registered = {row["method"]: row for row in read_csv(source) if (row["subject"], row["target"]) == (subject, target) and row["status"] == "ok"} if source.is_file() else {}
    summaries = []
    electrodes = []
    inputs = [artifact(study, "run", "implementation", Path(__file__))]
    if source.is_file():
        inputs.append(artifact(study, "run", "upstream_baselines", source))
    if len(registered) == 2:
        try:
            head = study / "head_models" / f"m2m_{subject}"
            mesh_path = head / f"{subject}.msh"
            cap_path = head / "eeg_positions/EEG10-10_NSN.csv"
            cap = _get_eeg_positions(str(cap_path))
            mesh_hash, cap_hash = digest(mesh_path), digest(cap_path)
            directories = {method: study / "simulations" / row["inverse_task"] for method, row in registered.items()}
            mappings = {}
            configs = {}
            for method, directory in directories.items():
                mapping_path = directory / "electrode_mapping.json"
                execution = json.loads((directory / "study_execution.json").read_text(encoding="utf-8"))
                if execution["exit_code"] != 0 or execution["sha256"][mapping_path.name] != digest(mapping_path):
                    raise ValueError(f"03 布局退出码或散列不匹配 {method}")
                mappings[method] = json.loads(mapping_path.read_text(encoding="utf-8"))
                configs[method] = json.loads((directory / "params.json").read_text(encoding="utf-8"))
                inputs.extend(artifact(study, method, "donor", path) for path in (mapping_path, directory / "params.json"))
                if registered[method]["mesh_hash"] != mesh_hash or configs[method]["montage"] != "EEG10-10_NSN":
                    raise ValueError(f"03 头模或帽位身份不符 {method}")
            if configs[METHODS[0]]["roi_param"] != configs[METHODS[1]]["roi_param"] or configs[METHODS[0]]["target_threshold"] != configs[METHODS[1]]["target_threshold"]:
                raise ValueError("两份先导布局的靶和阈值不一致")
            based, free = METHODS
            reference_path = directories[based] / "leadfield_reference.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            manifest_path = Path(reference["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            configuration = manifest["configuration"]
            if manifest["state"] != "complete" or manifest["cache_key"] != reference["cache_key"] or configuration["mesh"]["sha256"] != mesh_hash or configuration["montage"]["sha256"] != cap_hash:
                raise ValueError("实际 based 导联场与帽位或头模不匹配")
            leadfield = manifest_path.parent / manifest["leadfield"]["filename"]
            with h5py.File(leadfield, "r") as h5:
                names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5[LEADFIELD_DATASET].attrs["electrode_names"]]
            if names != configuration["electrode_names"]:
                raise ValueError("HDF5 头属性与登记的电极基因顺序不一致")
            info["data_id"] = dict(mesh=mesh_hash, cap=cap_hash, leadfield=reference["cache_key"])
            inputs.extend(artifact(study, "run", "encoding_input", path) for path in (mesh_path, cap_path, reference_path, manifest_path))
            for case, donor in zip(cases, METHODS):
                started = monotonic()
                receiver = case["method"]
                mapping = mappings[donor]
                labels = mapping["mapped_labels"]
                currents = mapping["current_A"] + mapping["current_B"]
                positions = np.asarray(mapping["mapped_positions"] if "mapped_positions" in mapping else [cap[label] for label in labels])
                try:
                    if ";".join(labels) != registered[donor]["electrode_names"] or not np.allclose(positions, [cap[label] for label in labels], rtol=0, atol=ROUNDTRIP_MM):
                        raise ValueError("供体的具名电极及映射坐标与 03 不符")
                    if np.sign(currents).tolist() != [1, -1, 1, -1] or not np.allclose(np.asarray(currents).reshape(2, 2).sum(axis=1), 0):
                        raise ValueError("供体的通道或极性顺序不符")
                    radius = float(configs[receiver]["electrode_radius"])
                    if receiver == free:
                        optimizer = prepare_free_geometry(mesh_path, radius, (mappings[free]["current_A"], mappings[free]["current_B"]))
                        encoded = encode_free(optimizer, positions)
                    else:
                        encoded = encode_based(labels, positions, names, cap, radius)
                except SOLVER_ERRORS as error:
                    encoded = Encoding([], [None] * 4, str(error))
                elapsed = round(monotonic() - started, 3)
                errors = [float(np.linalg.norm(back - original)) if back is not None else None for back, original in zip(encoded.roundtrip, positions)]
                good = encoded.reason == "NA"
                case.update(status="ok" if good else "failed", reason=encoded.reason, wall_s=elapsed)
                summaries.append(dict(donor_case_id=registered[donor]["case_id"], receiver=receiver, arm=case["arm"], valid_encoding=good,
                                      max_roundtrip_mm=max(errors) if all(error is not None for error in errors) else "NA", case_id=case["case_id"],
                                      donor_IA_mA=currents[0], donor_IB_mA=currents[2], condition_matched=False, encoding_s=elapsed,
                                      original_min_center_mm=float(pdist(positions).min()), encoded=json.dumps(encoded.parameters), reason=encoded.reason))
                for i, (back, error) in enumerate(zip(encoded.roundtrip, errors)):
                    valid = error is not None and error <= ROUNDTRIP_MM
                    electrodes.append(dict(case_id=case["case_id"], electrode=labels[i], role=ROLES[i], donor_current_mA=currents[i],
                                           original_xyz_mm=json.dumps(positions[i].tolist()), roundtrip_xyz_mm=json.dumps(back.tolist()) if back is not None else "NA",
                                           roundtrip_mm=error if error is not None else "NA", valid_encoding=valid,
                                           encoded=json.dumps(encoded.parameters[i]) if encoded.parameters else "NA",
                                           reason="NA" if valid else "可放置头皮无唯一有效反算位置" if back is None else "反算偏差超限"))
        except SOLVER_ERRORS as error:
            for case in cases:
                if case["status"] == "pending":
                    case.update(status="failed", reason=str(error))
    good = sum(case["status"] == "ok" for case in cases)
    failed = sum(case["status"] == "failed" for case in cases)
    info.update(status="encoding_checked" if len(summaries) == 2 else "partial", finished_at=timestamp())
    write_csv(out / "cases.csv", cases, CASE_COLUMNS)
    write_csv(out / "warm_start.csv", summaries, SUMMARY_COLUMNS)
    write_csv(out / "encoding_electrodes.csv", electrodes, ELECTRODE_COLUMNS)
    report = out / "report.md"
    lines = ["# 16 两路初值编码预核验", "", f"sub1/STN 已完成几何编码 {good + failed}/2 个方向，合法 {good} 个，失败 {failed} 个。", "",
             "供体取自 03 原生搜索，两路实际电流各异。原生布局只用于验证表示和可放置几何。"
             "等预算比较仍需同头模、同靶、同目标与 1+1 mA 的两路供体，随后对照随机、同路与跨路初始化。", ""]
    for row in summaries:
        lines.append(f"- `{row['donor_case_id']}` 至 `{row['receiver']}`，最大反算误差 {row['max_roundtrip_mm']} mm，原因 `{row['reason']}`。")
    lines.extend(("", "[双向明细](warm_start.csv) · [逐电极反算](encoding_electrodes.csv) · [病例](cases.csv) · [命令](commands.ps1)", ""))
    report.write_text("\n".join(lines), encoding="utf-8")
    write_csv(out / "artifacts.csv", [*inputs, *(artifact(study, "run", "output", out / file) for file in (
              "cases.csv", "warm_start.csv", "encoding_electrodes.csv", "report.md", "commands.ps1"))])
    write_json(out / "run.json", info)
    update_index(study, dict(experiment="16", run_id=run_id, stage="pilot", run_status=info["status"], scientific_decision="几何编码预核验",
                             ok_cases=good, failed_cases=failed, pending_cases=len(cases) - good - failed, report_path=report.relative_to(study).as_posix()))


def main() -> None:
    """解析预核验参数并执行真实个体布局的双向编码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot",), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    out = study / "results/16" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".scheduler.lock"), timeout=0):
        run(study, out, args.baseline_id, args.run_id)


if __name__ == "__main__":
    main()
