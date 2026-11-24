"""从本轮已落盘的共同场生成彩色三视图汇总。"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from neuracle.demo.ti_depth.exp14 import scale_carriers
from neuracle.demo.ti_depth.exp22 import based_carriers, free_carriers
from neuracle.ti_depth_study.baseline import FIELD_MODEL, content_id, forward_config
from neuracle.ti_depth_study.diagnostics import load_case
from neuracle.ti_depth_study.fields import load_geometry
from neuracle.ti_depth_study.metrics import _hotspot, measure_field, ti_envelope
from neuracle.ti_depth_study.records import digest
from simnibs.mesh_tools.mesh_io import Msh, Nodes, Elements

from .target_gallery import publish_targeted


ROOT = Path("neuracle/TI-study-v1").resolve()
RESULTS = ROOT / "results"
OUT = RESULTS / "彩色TI场强三视图"
PLOTS = OUT / "figures"
FONT = Path(os.environ.get("SystemRoot", "C:/Windows")) / "Fonts/msyh.ttc"
SCHEME = "mesh-ras-0.8-mm-turbo-0.40-v2"
MAX_FIELD = 0.40
STEP = 0.8
TARGETS = {
    "stn295": "丘脑底核 STN", "gp222": "苍白球 GP", "dca228": "背侧尾状核 dCa",
    "lamyg214": "杏仁核 lAmyg", "rhipp216": "海马 rHipp", "stha236": "Stha",
    "a4ul58": "运动皮层 A4ul", "a9d16": "前额叶 A9/46d", "a24rv178": "内侧皮层 A24rv", "opc204": "OPC",
}
EXPERIMENTS = {
    "03": "原生基线", "04": "定位与皮层热点", "05": "覆盖与外溢约束", "06": "体积加权 ROC", "07": "平滑阈值",
    "08": "低场尾部", "09": "邻近壳层", "10": "跨阈值覆盖", "11": "双侧选择性", "12": "热点重叠",
    "13": "解剖分流", "14": "电流配比", "15": "多起点", "16": "互供初值", "17": "包络方向",
    "18": "帽位扰动", "19": "电导率扰动", "20": "稳健寻优", "21": "靶区位移", "22": "电极映射与复算",
    "23": "固定帽位见证", "pool": "固定候选库（全量）",
}
ARMS = {
    "roc0": "ROC0 参照", "shortfall": "仅靶内不足 A", "shortfall_distance": "不足＋距离 A+D", "full": "完整 A+D+P",
    "coverage_only": "仅靶区覆盖", "k100": "原外溢预算", "k075": "75% 外溢预算", "random": "随机初值",
    "own": "同路供体", "cross": "跨路供体", "cross_repaired": "T7/T8 修复后跨路供体",
    "single": "单起点", "triple": "三起点", "nominal": "名义靶区", "robust": "稳健评分",
    "leadfield_based": "leadfield-based", "leadfield_free": "leadfield-free",
}
VIEWS = (
    (0, 1, 2, "矢状面", "P", "A", "I", "S"), (1, 0, 2, "冠状面", "L", "R", "I", "S"),
    (2, 0, 1, "轴位", "L", "R", "P", "A"),
)


def rows(path: Path) -> list[dict[str, str]]:
    """读取研究 CSV 并保留原始字段值。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    """读取落盘的 JSON 收据。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value: Any) -> float | None:
    """将公开指标规范化成有限数或空值。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def point(value: Any) -> list[float] | None:
    """转录已登记的三维热点，保留缺失状态。"""
    if isinstance(value, str):
        value = json.loads(value) if value.startswith("[") else None
    if value is not None and len(value) == 3 and all(number(x) is not None for x in value):
        return [float(x) for x in value]
    return None


def root_rel(path: Path) -> str:
    """将证据路径统一为研究根目录相对路径。"""
    return path.resolve().relative_to(ROOT).as_posix()


class Catalog:
    """建立共享图像和各实验证据之间的对应关系。"""

    def __init__(self) -> None:
        """初始化目录及图像去重表。"""
        self.entries: list[dict[str, Any]] = []
        self.panels: dict[str, dict[str, Any]] = {}
        self.derived: dict[str, Any] = {}
        self.masks: dict[str, tuple[Path, str]] = {}
        self.geometries: dict[str, Path] = {}
        self.notes: list[str] = []

    def add(
        self, experiment: str, label: str, reading: dict[str, Any], source: str, table: Path, *,
        kind: str = "共同正向 FEM", mask: str = "", reference: bool = False, token: str = "",
    ) -> None:
        """登记一个可浏览结果并按场、靶区和热点复用图片。"""
        subject, target = reading["subject"], reading["target"]
        hot = point(reading.get("hot_center_mm"))
        material = str(Path(source).resolve()) if not source.startswith("derived:") else source
        if not material.startswith("derived:") and not Path(material).is_file():
            raise FileNotFoundError(material)
        digest_in = str((SCHEME, subject, target, material, mask, hot))
        key = hashlib.sha256(digest_in.encode()).hexdigest()[:20]
        self.panels.setdefault(key, dict(subject=subject, target=target, source=material, mask=mask, hot=hot))
        self.entries.append(dict(experiment=experiment, figure=f"figures/{key}.png", title=label,
                                 subject=subject, target=target, kind=kind, reference=reference, token=token,
                                 distance=number(reading.get("D_hot_mm")), coverage=number(reading.get("Cov010")),
                                 spill=number(reading.get("Vout010")), table=root_rel(table),
                                 source=root_rel(Path(material)) if not material.startswith("derived:") else material))


def add_baselines(cat: Catalog) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    """加入 03 的全部具名场，建立历史及当前病例索引。"""
    directory = RESULTS / "03/run-001"
    table = directory / "baseline_fields.csv"
    by_case = {}
    by_method = {}
    for reading in rows(table):
        sid = reading["case_id"].removesuffix("_native")
        by_case[sid] = reading
        by_method[reading["subject"], reading["target"], reading["method"]] = reading
        cat.add("03", ARMS[reading["method"]], reading, str(ROOT / reading["field_path"]), table, reference=True, token=sid)
    for item in rows(directory / "shared_inputs.csv"):
        if item["role"] == "geometry":
            cat.geometries[item["subject"]] = ROOT / item["path"]
    return by_case, by_method


def add_candidate_groups(cat: Catalog) -> dict[tuple[str, str], dict[str, str]]:
    """登记 195 个固定候选及 04–12 中各对照臂与首选。"""
    table = RESULTS / "04/run-001/term_ablation.csv"
    content = rows(table)
    pool_paths = {(entry["subject"], row["candidate_id"]): ROOT / row["field_path"]
                  for entry in rows(RESULTS / "03/run-001/shared_inputs.csv") if entry["role"] == "candidate_index"
                  for row in rows(ROOT / entry["path"]) if row["status"] == "ok"}
    for reading in content:
        reading["candidate_field_path"] = str(pool_paths[reading["subject"], reading["candidate_id"]])
    raw = {(r["case_id"], r["candidate_id"]): r for r in content if r["arm"] == "roc0"}
    for (case, identifier), reading in raw.items():
        cat.add("pool", f"ROC0 排名 {reading['rank']}", reading, reading["candidate_field_path"], table,
                token=identifier, reference=reading["rank"] == "1")
    for reading in content:
        if reading["rank"] == "1":
            arm = reading["arm"]
            cat.add("04", ARMS[arm], reading, reading["candidate_field_path"], table,
                    token=reading["candidate_id"], reference=arm == "roc0")
    for case, identifier in (("sub2_gp222", "ae506c6b3643fbaa8b50"),
                             ("sub2_dca228", "4878c9ad8ebce0c8852d"),
                             ("sub2_dca228", "65167ca2671e5ea9b611")):
        reading = raw[case, identifier]
        cat.add("04", "定位线索 · 固定库候选", reading, reading["candidate_field_path"], table, token=identifier)
    for code in (f"{n:02d}" for n in range(5, 13)):
        table = RESULTS / code / "run-001/paired.csv"
        seen = set()
        for row in rows(table):
            for prefix in ("reference_", ""):
                case, identifier = row[prefix + "case_id"].rsplit(":", 1)
                arm = row[prefix + "arm"]
                key = (case, identifier, arm)
                if key in seen:
                    continue
                seen.add(key)
                reading = raw[case, identifier]
                cat.add(code, ARMS.get(arm, arm), reading, reading["candidate_field_path"], table,
                        token=identifier, reference=bool(prefix))
    return raw


def add_routing_and_ratios(cat: Catalog, baselines: dict[str, dict[str, str]]) -> None:
    """登记 13 原布局和 14 的载波缩放场。"""
    table = RESULTS / "13/run-001/routing.csv"
    for row in rows(table):
        if row["old_evidence"] != "verified":
            continue
        original = baselines[row["old_case_id"].removesuffix("_native")]
        cat.add("13", "原生场核验 · " + ARMS[row["method"]], original,
                str(ROOT / original["field_path"]), table, reference=True)
    table = RESULTS / "14/run-001/ratio_sweep.csv"
    for row in rows(table):
        source = ROOT / baselines[row["case_id"]]["field_path"]
        current = (float(row["IA_mA"]), float(row["IB_mA"]))
        original = (float(row["source_IA_mA"]), float(row["source_IB_mA"]))
        tag = f"derived:14:{row['ratio_case_id']}"
        cat.derived[tag] = ("ratio", source, original, current)
        cat.add("14", f"{ARMS[row['method']]} · {current[0]:g}+{current[1]:g} mA", row, tag, table,
                kind="共同载波按电流缩放", reference=current[0] == current[1])


def add_search_results(cat: Catalog, base: dict[tuple[str, str, str], dict[str, str]]) -> None:
    """加入 15 和 16 已落盘的正向胜者及已发布的计时参照。"""
    table = RESULTS / "15/run-001/winner_fields.csv"
    for row in rows(table):
        cat.add("15", f"{ARMS[row['method']]} · 种子组 {row['seed_group']} · {ARMS[row['arm']]}", row,
                str(ROOT / row["field_path"]), table, reference=row["arm"] == "single", token=row["labels"])
    directory = RESULTS / "16/run-002/comparison"
    issued = set()
    for name in ("warm_start.csv", "cold_restart.csv"):
        table = directory / name
        for row in rows(table):
            if row["status"] != "ok" or row["field_path"] in ("NA", ""):
                continue
            method, arm = row["receiver"], row["arm"]
            label = f"{ARMS[method]} · {ARMS[arm]}" + (" · 从零计时对照" if name == "cold_restart.csv" else "")
            cat.add("16", label, row, str(ROOT / row["field_path"]), table,
                    token=row["labels"], reference=arm == "random" or name == "cold_restart.csv")
            if name == "warm_start.csv":
                issued.add((method, arm))
    for method in ("leadfield_based", "leadfield_free"):
        models = directory / "search" / method
        source_fields = {Path(read_json(path)["source"]).name: path.with_suffix(".npz")
                         for path in (models / "fields").glob("*.json")}
        common = None
        geometry_hash = None
        baseline = base["sub1", "stn295", method]
        for arm in ("random", "own", "cross", "cross_repaired"):
            record_path = models / arm / "result.json"
            if (method, arm) in issued or not record_path.is_file():
                continue
            record = read_json(record_path)
            config = forward_config(record["labels"].split(";"), ([1.0, -1.0], [1.0, -1.0]))
            task = "ti_forward_15_sub1_" + content_id(dict(config=config, mesh=baseline["mesh_hash"]))[:20]
            field = source_fields.get(task)
            if field is None:
                continue
            if common is None:
                common = load_case(ROOT, baseline)
                geometry_hash = digest(common.geometry_path)
            receipt = read_json(field.with_suffix(".json"))
            if receipt["inputs"]["geometry"] != geometry_hash or receipt["inputs"]["model"] != FIELD_MODEL or digest(field) != receipt["sha256"]:
                raise ValueError(f"16 缓存场来源不一致 {field}")
            with np.load(field) as array:
                values = measure_field(common.geometry, common.target, array["E_A"], array["E_B"], array["valid_mask"])
            values.update(subject="sub1", target="stn295")
            cat.add("16", f"{ARMS[method]} · {ARMS[arm]}", values, str(field), record_path,
                    kind="共同正向 FEM（配对汇总前）", token=record["labels"], reference=arm == "random")
    if not (directory / "pilot.exit").is_file():
        cat.notes.append("16 的计时对照仍可能新增正向场，本汇总包含快照时已落盘的图。")


def add_diagnostics(
    cat: Catalog, baselines: dict[str, dict[str, str]], raw: dict[tuple[str, str], dict[str, str]],
) -> None:
    """登记 17–23 可绘制的场、方向和诊断掩膜。"""
    table = RESULTS / "17/run-001/direction_axes.csv"
    for row in rows(table):
        original = ROOT / baselines[row["case_id"]]["field_path"]
        tag = str(original)
        if row["axis"] != "max_TI":
            tag = f"derived:17:{row['case_id']}:{row['axis']}"
            cat.derived[tag] = ("axis", original, [float(row[key]) for key in ("nx", "ny", "nz")])
        cat.add("17", f"{ARMS[row['method']]} · {row['axis']}", row, tag, table,
                kind="固定方向包络" if row["axis"] != "max_TI" else "共同正向 FEM",
                reference=row["axis"] == "max_TI")
    for code, filename, nominal in (("18", "position_offsets.csv", "offset00"), ("19", "conductivity_sweep.csv", "nominal")):
        table = RESULTS / code / "run-001" / filename
        for row in rows(table):
            field = row.get("field_path", "NA")
            if row["status"] != "ok" or field in ("NA", ""):
                continue
            cat.add(code, f"{ARMS[row['method']]} · {row['condition']}", row, str(ROOT / field), table,
                    reference=row["condition"] == nominal, token=row["case_id"])
    table = RESULTS / "20/run-001/robust_masks.csv"
    mask_file = RESULTS / "20/run-001/masks/sub1/stn295_shifts.npz"
    for row in sorted(rows(table), key=lambda item: (item["mask_id"], number(item["rank"]) or math.inf, item["arm"])):
        reading = raw[row["case_id"], row["candidate_id"]]
        key = row["mask_id"]
        mask = f"20:{key}"
        cat.masks[mask] = (mask_file, key)
        reading = reading | {"D_hot_mm": row["D_hot_mm"], "Cov010": row["Cov010"], "Vout010": row["Vout010"]}
        rank = f"排名 {row['rank']}" if number(row["rank"]) is not None else "未入排序"
        cat.add("20", f"{ARMS[row['arm']]} · {key} · {rank}", reading, reading["candidate_field_path"], table,
                kind="共同正向 FEM · 变换靶区", mask=mask, token=row["candidate_id"], reference=key == "shift00")
    table = RESULTS / "21/run-001/roi_shift.csv"
    mask_file = RESULTS / "21/run-001/masks/sub1/stn295_tet_masks.npz"
    for row in rows(table):
        if row["status"] != "ok":
            continue
        key = row["mask_id"]
        mask = f"21:{key}"
        cat.masks[mask] = (mask_file, key)
        source = str(ROOT / baselines[row["case_id"]]["field_path"])
        cat.add("21", f"{ARMS[row['method']]} · {key}", row, source, table,
                kind="同一电场 · 变换靶区", mask=mask, reference=key == "shift00")
    table = RESULTS / "22/run-001/paired.csv"
    registered = set()
    for pair in rows(table):
        for side in ("reference_", ""):
            identifier = pair[side + "case_id"]
            if identifier in registered:
                continue
            registered.add(identifier)
            case_id, name = identifier.split(":", 1)
            original = baselines[case_id.removesuffix("_native")]
            source = str(ROOT / original["field_path"])
            if name != "common_mapped_single_layer":
                source = f"derived:22:{identifier}"
                cat.derived[source] = ("mapping", original, name)
            reading = original | {key: pair[side + key] for key in ("D_hot_mm", "Cov010", "Vout010")}
            reading.pop("hot_center_mm", None)
            if name == "common_mapped_single_layer":
                reading["hot_center_mm"] = original["hot_center_mm"]
            names = {"common_mapped_single_layer": "共同正向单层",
                     "based_reconstruction_single_layer": "导联场重建单层",
                     "free_continuous_double_layer": "free 连续位置双层",
                     "free_mapped_double_layer": "free 帽位映射双层"}
            cat.add("22", f"{ARMS[original['method']]} · {names[name]}", reading, source, table,
                    kind=names[name], reference=bool(side), token=identifier)
    table = RESULTS / "23/run-001/common_forward/witness_forward.csv"
    for row in rows(table):
        cat.add("23", "共同正向见证", row, str(ROOT / row["field_path"]), table, token=row["witness_layout"])
    cat.notes.extend(("01–02 为历史审计与协议登记。sub3 的本轮保留例没有落盘仿真场。",
                      "23 的全枚举覆盖上界是数值诊断。三视图展示已复算的三个见证布局。",
                      "没有通过放置检查或尚未输出电场的病例，请查看对应实验的原报告。"))


def columns_for(view: tuple[Any, ...], axes: list[np.ndarray], centre: np.ndarray) -> np.ndarray:
    """构造定距世界坐标三视图像素中心，首行对应图像上方。"""
    fixed, horiz, vert = view[:3]
    xx, yy = np.meshgrid(axes[horiz], axes[vert][::-1], indexing="xy")
    xyz = np.empty(xx.shape + (3,), dtype=np.float64)
    xyz[..., fixed], xyz[..., horiz], xyz[..., vert] = centre[fixed], xx, yy
    return xyz


class Renderer:
    """为同一头模预计算三视图像素与脑实质四面体的对应。"""

    def __init__(self, catalog: Catalog, subject: str, targets: set[str]) -> None:
        """加载参考网格并固定该头模的切面、取样和可用掩膜。"""
        self.catalog = catalog
        self.subject = subject
        self.geometry_path = catalog.geometries[subject]
        self.geometry, self.tags, self.targets = load_geometry(self.geometry_path)
        self.geo_hash = digest(self.geometry_path)
        self.bounds = [(int(np.floor(low))-4, int(np.ceil(high))+4)
                       for low, high in zip(self.geometry.centroid_mm.min(0), self.geometry.centroid_mm.max(0))]
        self.axes = [np.arange(low, high + STEP / 2, STEP) for low, high in self.bounds]
        self.centres = {key: np.average(self.geometry.centroid_mm[self.targets[key]], weights=self.geometry.volume_mm3[self.targets[key]],
                                       axis=0) for key in targets}
        # 首个超界值作为图像背景，元素行号统一为零基。
        self.lookup = {}
        panels = [(key, dim, columns_for(view, self.axes, self.centres[key]))
                  for key in sorted(targets) for dim, view in enumerate(VIEWS)]
        query = np.concatenate([xyz.reshape(-1, 3) for _, _, xyz in panels])
        mesh = Msh(nodes=Nodes(self.geometry.node_mm), elements=Elements(tetrahedra=self.geometry.connectivity + 1))
        result = mesh.find_tetrahedron_with_points(query, compute_baricentric=False)
        result = np.where(result > 0, result - 1, -1).astype(np.int32)
        offset = 0
        for key, dim, xyz in panels:
            width = xyz.shape[0] * xyz.shape[1]
            self.lookup[key, dim] = result[offset:offset + width].reshape(xyz.shape[:2])
            offset += width
        self.outlines = {}
        self.colours = np.asarray(mpl.colormaps["turbo"](np.arange(256) / 255, bytes=True)[:, :3])
        self.body = ImageFont.truetype(str(FONT), 14)
        self.heading = ImageFont.truetype(str(FONT), 19)
        self.cached_masks = {}
        self.carriers = {}
        print(f"{subject} 切面映射完成，{len(query):,} 像素，{np.count_nonzero(result >= 0):,} 个脑内像素", flush=True)

    def mask(self, target: str, override: str) -> np.ndarray:
        """读取名义或登记位移后的参考网格靶掩膜。"""
        if not override:
            return self.targets[target]
        if override not in self.cached_masks:
            path, name = self.catalog.masks[override]
            with np.load(path, allow_pickle=False) as data:
                if "element_id" in data:
                    if not np.array_equal(data["element_id"], self.geometry.element_id):
                        raise ValueError(f"靶区掩膜的参考网格不一致 {path}")
                mask = data[name].astype(bool)
            if mask.shape != self.targets[target].shape:
                raise ValueError(f"掩膜维数不匹配 {path}")
            self.cached_masks[override] = mask
        return self.cached_masks[override]

    def field(self, source: str) -> np.ndarray:
        """读取真场或按原方案公式从载波还原诊断标量。"""
        if not source.startswith("derived:"):
            path = Path(source)
            receipt = read_json(path.with_suffix(".json"))
            if receipt["inputs"]["geometry"] != self.geo_hash:
                raise ValueError(f"场和切面网格不匹配 {path}")
            with np.load(path, allow_pickle=False) as data:
                field = data["max_TI"]
            return field
        kind, source, *parameters = self.catalog.derived[source]
        if kind == "mapping":
            baseline = source
            name, = parameters
            case = load_case(ROOT, baseline)
            folder = ROOT / "simulations" / baseline["inverse_task"]
            if name == "based_reconstruction_single_layer":
                mapping = read_json(folder / "electrode_mapping.json")
                vectors = based_carriers(case, folder, mapping, self.tags, [])
            else:
                folders = (("final_sim_0", "final_sim_1") if name == "free_continuous_double_layer" else
                           ("mapped_electrodes_simulation/mapped_sim_0", "mapped_electrodes_simulation/mapped_sim_1"))
                vectors = free_carriers(case, folder, folders, self.tags, [])
            return ti_envelope(*vectors)
        if source not in self.carriers:
            with np.load(source, allow_pickle=False) as data:
                self.carriers[source] = (data["E_A"], data["E_B"])
        first, second = self.carriers[source]
        if kind == "ratio":
            original, current = parameters
            return ti_envelope(*scale_carriers(first, second, original, current))
        vector = np.asarray(parameters[0])
        return 2 * np.minimum(np.abs(first @ vector), np.abs(second @ vector))

    def outline(self, target: str, mask: str, dim: int) -> np.ndarray:
        """预计算指定掩膜在同一世界切面上的青色边缘。"""
        key = (target, mask, dim)
        if key not in self.outlines:
            pixels = self.lookup[target, dim]
            roi = (pixels >= 0) & self.mask(target, mask)[np.maximum(0, pixels)]
            self.outlines[key] = roi & ~binary_erosion(roi)
        return self.outlines[key]

    def plot(self, destination: Path, panel: dict[str, Any], field: np.ndarray) -> None:
        """绘制统一布局的三切面、标注与色标。"""
        target, mask = panel["target"], panel["mask"]
        hotspot = panel["hot"]
        if hotspot is None and np.any(np.isfinite(field) & (field > 0)):
            computed = np.nan_to_num(field, nan=-1.0)
            hotspot = _hotspot(computed, self.geometry, self.mask(target, mask), 100.0)["hot_center_mm"]
        centre = self.centres[target]
        if field.shape != self.geometry.element_id.shape:
            raise ValueError(f"场数组逐元素长度不符 {panel['source']}")
        canvas = Image.new("RGB", (928, 303), "#F1F4F8")
        painter = ImageDraw.Draw(canvas)
        for dim, (fixed, horiz, vert, name, left, right, bottom, top) in enumerate(VIEWS):
            indices = self.lookup[target, dim]
            values = field[np.maximum(0, indices)]
            valid = (indices >= 0) & np.isfinite(values) & (values > 0)
            bins = (np.clip(np.where(valid, values, 0.0), 0, MAX_FIELD) * (255 / MAX_FIELD)).astype(np.uint8)
            rgb = self.colours[bins]
            rgb[~valid] = (8, 10, 12)
            outline = self.outline(target, mask, dim)
            rgb[binary_dilation(outline) & ~outline] = (0, 48, 54)
            rgb[outline] = (43, 255, 236)
            image = Image.fromarray(rgb)
            image.thumbnail((272, 247), Image.Resampling.NEAREST)
            slot = 12 + dim * 282
            x0, y0 = slot + (272 - image.width) // 2, 35 + (247 - image.height) // 2
            painter.rectangle((slot, 35, slot+271, 281), fill="#080a0c")
            canvas.paste(image, (x0, y0))
            painter.text((slot + 136, 19), name, font=self.heading, fill="#23354B", anchor="mm")
            def local(pos: np.ndarray | list[float]) -> tuple[float, float]:
                """将 RAS 点的正交投影变换到裁切后的像素。"""
                hh = (pos[horiz] - self.axes[horiz][0]) / STEP + .5
                vv = (self.axes[vert][-1] - pos[vert]) / STEP + .5
                return x0 + hh * image.width / indices.shape[1], y0 + vv * image.height / indices.shape[0]
            cx, cy = local(centre)
            for xx in range(slot+3, slot+269, 5):
                painter.point((xx, cy), fill="#A0ADB7")
            for yy in range(38, 279, 5):
                painter.point((cx, yy), fill="#A0ADB7")
            painter.text((slot+9, 153), left, font=self.body, fill="#D8E2EA", anchor="lm")
            painter.text((slot+263, 153), right, font=self.body, fill="#D8E2EA", anchor="rm")
            painter.text((slot+135, 39), top, font=self.body, fill="#D8E2EA", anchor="mt")
            painter.text((slot+135, 279), bottom, font=self.body, fill="#D8E2EA", anchor="mb")
            if hotspot is not None:
                hx, hy = local(hotspot)
                for sign in (-1, 1):
                    line = (hx-5, hy-5*sign, hx+5, hy+5*sign)
                    painter.line(line, fill="#080A0C", width=4)
                    painter.line(line, fill="white", width=2)
                distance = abs(hotspot[fixed]-centre[fixed])
                painter.text((slot+262, 258), f"离切面 {distance:.1f} mm", font=self.body, fill="white",
                             anchor="ra", stroke_width=2, stroke_fill="#080A0C")
        grad = self.colours[np.rint(np.linspace(255, 0, 190)).astype(np.uint8)]
        canvas.paste(Image.fromarray(np.repeat(grad[:, None, :], 13, axis=1)), (862, 57))
        painter.text((857, 36), "V/m", font=self.body, fill="#23354B")
        for tick in (0, .1, .2, .3, .4):
            painter.text((881, round(57 + 189 * (1 - tick / MAX_FIELD))), f"{tick:.1f}",
                         font=self.body, fill="#23354B", anchor="lm")
        temp = destination.with_suffix(".tmp")
        canvas.save(temp, "PNG", compress_level=2)
        temp.replace(destination)


def render(catalog: Catalog, selected: set[str]) -> None:
    """按源场分组；每个实际场按需读取一次并复用切面。"""
    work = {key: panel for key, panel in catalog.panels.items() if key in selected and not (PLOTS / f"{key}.png").is_file()}
    subjects = sorted({panel["subject"] for panel in work.values()})
    for subject in subjects:
        groups = defaultdict(list)
        for key, panel in work.items():
            if panel["subject"] == subject:
                groups[panel["source"]].append((key, panel))
        targets = {panel["target"] for group in groups.values() for _, panel in group}
        renderer = Renderer(catalog, subject, targets)
        total = sum(map(len, groups.values()))
        count = 0
        for source, group in groups.items():
            field = renderer.field(source)
            for key, panel in group:
                renderer.plot(PLOTS / f"{key}.png", panel, field)
                count += 1
            if count % 100 < len(group) or count == total:
                print(f"{subject} 已绘制 {count}/{total}", flush=True)


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TI 场强三视图</title>
<style>
:root{color-scheme:light;font:16px/1.55 "Microsoft YaHei",system-ui,sans-serif;color:#14253b;background:#edf2f7}
*{box-sizing:border-box}body{margin:0}main{max-width:1200px;margin:auto;padding:26px 24px 60px}
header{padding:8px 0 22px}h1{font-size:clamp(1.7rem,4vw,2.65rem);margin:0 0 8px;letter-spacing:-.025em}
p{margin:.4em 0}.controls,article,details{background:#fff;border:1px solid #d9e3ed;border-radius:14px}
.controls{display:flex;gap:12px 18px;align-items:end;flex-wrap:wrap;padding:16px}label{display:grid;gap:4px;font-size:.85rem}
button,select{font:inherit;padding:7px 10px;border:1px solid #b8c9d9;border-radius:8px;background:white;color:#14253b}
button{cursor:pointer}button:hover,a:hover{color:#007e89}button[disabled]{opacity:.45;cursor:default}
.quick{display:flex;gap:8px;flex-wrap:wrap}a{color:#135876}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin:14px 0}
.cards{display:grid;gap:12px}article{padding:14px 16px 12px}article img{display:block;width:100%;max-width:928px;height:auto;margin:10px auto 4px}
.heading,.meta{display:flex;gap:7px 16px;flex-wrap:wrap;align-items:baseline}h2{font-size:1.05rem;margin:0}
.meta{font-size:.85rem;color:#3c5168}.metric b{font-variant-numeric:tabular-nums;color:#143752}.code{font:11px/1.45 monospace;color:#52677b}
details{padding:10px 16px;margin:14px 0}summary{cursor:pointer}.nav{display:flex;gap:8px;align-items:center}.notes{color:#52677b;font-size:.875rem}
@media(max-width:740px){main{padding:14px 10px 36px}.controls{align-items:stretch}.controls label{width:min(100%,300px)}}
</style></head><body><main>
<header><h1>TI 场强三视图</h1><p>同一头模、同一靶点使用相同切面。青色是靶区，白色 × 是主热点在切面上的投影。</p>
<p class="notes">统一显示 0–0.40 V/m，超过上限使用最高色。覆盖和外溢按 0.10 V/m 计算。</p></header>
<section class="controls" aria-label="筛选">
<label>实验<select id="experiment"></select></label><label>头模<select id="subject"></select></label>
<label>靶点<select id="target"></select></label><label>卡片顺序<select id="sort"><option value="report">实验顺序</option><option value="distance">基线优先，其余按距离</option></select></label>
<div class="quick"><button data-target="gp222" type="button">看苍白球</button><button data-target="dca228" type="button">看尾状核</button><button data-target="stn295" type="button">看 STN</button></div>
</section>
<div class="toolbar"><span id="summary" role="status"></span><div class="nav"><button type="button" id="prev">上一页</button><span id="page"></span><button type="button" id="next">下一页</button></div></div>
<div id="cards" class="cards"></div>
<details><summary>图的范围与来源</summary><ul id="notes"></ul><p class="notes" id="snapshot"></p><a href="full_audit.csv">查看全量图像证据清单</a> · <a href="index.html">返回同靶对照</a></details>
</main>
<script type="application/json" id="gallery-data">@@DATA@@</script>
<script>
const data=JSON.parse(document.querySelector('#gallery-data').textContent);
const q=id=>document.getElementById(id);
const exp=q('experiment'),subject=q('subject'),target=q('target'),sort=q('sort');
const PAGE=18;
let page=0;
function choices(node,items,preferred){
  node.replaceChildren(...items.map(([key,label])=>new Option(label,key)));
  node.value=items.some(([key])=>key===preferred)?preferred:items[0][0];
}
function matchesExperiment(row){return exp.value==='all'?row.experiment!=='pool':row.experiment===exp.value;}
function refresh(preferredSubject,preferredTarget){
  const records=data.entries.filter(matchesExperiment);
  const subjects=[...new Set(records.map(x=>x.subject))].sort();
  choices(subject,[['all','所有头模'],...subjects.map(x=>[x,x])],preferredSubject);
  const targets=new Set(records.filter(x=>subject.value==='all'||x.subject===subject.value).map(x=>x.target));
  choices(target,[['all','所有靶点'],...Object.entries(data.targets).filter(([key])=>targets.has(key))],preferredTarget);
}
function add(parent,tag,content,cls){
  const item=document.createElement(tag);
  if(content!==undefined)item.textContent=content;
  if(cls)item.className=cls;
  parent.append(item);
  return item;
}
function metric(parent,label,value,suffix,scale=1){
  if(value===null)return;
  const slot=add(parent,'span',label+' ','metric');
  add(slot,'b',(value*scale).toFixed(2)+suffix);
}
function relative(path){return '../../'+path.split('/').map(encodeURIComponent).join('/');}
function display(){
  function selected(row){
    return (subject.value==='all'||row.subject===subject.value)&&(target.value==='all'||row.target===target.value);
  }
  let cards=data.entries.filter(x=>matchesExperiment(x)&&selected(x));
  if(exp.value==='04'&&target.value!=='all')cards=[...data.entries.filter(x=>x.experiment==='03'&&selected(x)),...cards];
  if(sort.value==='distance')cards.sort((a,b)=>Number(b.reference)-Number(a.reference)||(a.distance??Infinity)-(b.distance??Infinity));
  const pages=Math.max(1,Math.ceil(cards.length/PAGE));
  page=Math.max(0,Math.min(page,pages-1));
  q('summary').textContent=cards.length+' 项结果';
  q('page').textContent=(page+1)+' / '+pages;
  q('prev').disabled=page===0;
  q('next').disabled=page+1>=pages;
  q('cards').replaceChildren();
  for(const row of cards.slice(page*PAGE,(page+1)*PAGE)){
    const card=add(q('cards'),'article');
    const head=add(card,'div',undefined,'heading');
    const prefix=exp.value==='all'?row.experiment+' · ':(exp.value==='04'&&row.experiment==='03'?'原生基线 · ':'');
    add(head,'h2',prefix+row.title);
    add(head,'span',row.subject+' · '+data.targets[row.target]+' · '+row.kind,'meta');
    const stat=add(card,'div',undefined,'meta');
    metric(stat,'热点距靶',row.distance,' mm');
    metric(stat,'靶内覆盖',row.coverage,'%',100);
    metric(stat,'全脑靶外超阈体积',row.spill,' cm³',.001);
    const link=add(card,'a');
    link.href=row.figure;
    const img=add(link,'img');
    img.src=row.figure;
    img.width=928;
    img.height=303;
    img.alt=row.subject+' '+data.targets[row.target]+' '+row.title+'的矢状、冠状和轴位彩色场图';
    img.loading='lazy';
    const footer=add(card,'p',undefined,'meta');
    const detail=add(footer,'a','原始结果表');
    detail.href=relative(row.table);
    if(row.token)add(footer,'span',row.token,'code');
  }
}
function show(preferredExperiment,preferredSubject,preferredTarget){
  exp.value=preferredExperiment;
  refresh(preferredSubject,preferredTarget);
  page=0;
  display();
}
const packed=Object.entries(data.experiments).filter(([key])=>data.entries.some(x=>x.experiment===key))
  .sort(([a],[b])=>a.localeCompare(b,'zh-CN'))
  .map(([key,label])=>[key,key==='pool'?label:key+' '+label]);
choices(exp,[['all','各实验汇总'],...packed],'04');
show('04','sub2','gp222');
exp.onchange=()=>{refresh(subject.value,target.value);page=0;display();};
subject.onchange=()=>{refresh(subject.value,target.value);page=0;display();};
target.onchange=sort.onchange=()=>{page=0;display();};
q('prev').onclick=()=>{page--;display();};
q('next').onclick=()=>{page++;display();};
for(const button of document.querySelectorAll('[data-target]')){
  button.onclick=()=>show('04',button.dataset.target==='stn295'?'sub1':'sub2',button.dataset.target);
}
for(const note of data.notes)add(q('notes'),'li',note);
q('snapshot').textContent='快照 '+data.snapshot+' · '+data.entries.length+' 项可浏览记录 · '+data.count+' 张独立图';
</script></body></html>
"""


def publish(catalog: Catalog) -> None:
    """核验图像和结果表后发布本地浏览页与清单。"""
    for entry in catalog.entries:
        if not (OUT / entry["figure"]).is_file() or not (ROOT / entry["table"]).is_file():
            raise FileNotFoundError(entry)
    data = dict(entries=catalog.entries, targets=TARGETS, experiments=EXPERIMENTS,
                notes=catalog.notes, count=len(catalog.panels), snapshot=datetime.now().astimezone().isoformat(timespec="minutes"))
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = OUT / "full_audit.html"
    temporary = html.with_suffix(".tmp")
    temporary.write_text(HTML.replace("@@DATA@@", encoded), encoding="utf-8")
    temporary.replace(html)
    manifest = OUT / "full_audit.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(catalog.entries[0]))
        writer.writeheader()
        writer.writerows(catalog.entries)
    print(f"全量索引保留 {len(catalog.entries)} 项结果，{len(catalog.panels)} 张独立图。", flush=True)


def main() -> None:
    """建立落盘快照，生成样图或完整图集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    catalog = Catalog()
    baseline, by_method = add_baselines(catalog)
    candidates = add_candidate_groups(catalog)
    add_routing_and_ratios(catalog, baseline)
    add_search_results(catalog, by_method)
    add_diagnostics(catalog, baseline, candidates)
    print(f"实验计数 {dict(Counter(row['experiment'] for row in catalog.entries))}，独立图 {len(catalog.panels)}", flush=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    chosen = set(catalog.panels)
    if args.sample:
        chosen = {Path(entry["figure"]).stem for entry in catalog.entries
                  if (entry["experiment"], entry["subject"], entry["target"]) in {("04", "sub2", "gp222"),
                                                                               ("04", "sub2", "dca228"),
                                                                               ("22", "sub1", "stn295")}}
    render(catalog, chosen)
    if not args.sample:
        publish(catalog)
        publish_targeted(catalog.entries, ROOT, OUT, TARGETS, EXPERIMENTS, ARMS)


if __name__ == "__main__":
    main()
