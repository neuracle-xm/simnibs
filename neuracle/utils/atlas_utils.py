"""
Atlas 标准化与 ROI 路径工具。

该模块负责：
1. 约定 atlas 原始目录、标准化目录和 manifest 目录
2. 生成/读取 atlas registry
3. 根据 atlas 名称和脑区名称定位标准化 ROI 文件
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NEURACLE_ROOT = REPO_ROOT / "neuracle"
ATLAS_ROOT = NEURACLE_ROOT / "atlas"
ATLAS_SOURCE_DIR = ATLAS_ROOT / "atlas"
ATLAS_STANDARDIZED_DIR = ATLAS_ROOT / "standardized"
ATLAS_MANIFESTS_DIR = ATLAS_ROOT / "manifests"
ATLAS_VALIDATION_DIR = ATLAS_ROOT / "validation"
ATLAS_SCRIPTS_DIR = ATLAS_ROOT / "scripts"
ATLAS_COLOR_TABLE_DIR = ATLAS_SOURCE_DIR / "颜色查找表"
ATLAS_REGISTRY_PATH = ATLAS_MANIFESTS_DIR / "atlas_registry.json"
SIMNIBS_MNI_TEMPLATE = (
    REPO_ROOT / "simnibs" / "resources" / "templates" / "MNI152_T1_1mm.nii.gz"
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "roi"


def _to_repo_relative(path: Path) -> str:
    # registry 内统一保存仓库相对路径，避免把本机绝对路径写死进去。
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _resolve_repo_path(path_str: str | Path) -> Path:
    # 运行时再把相对路径恢复成绝对路径，调用方不需要关心路径拼接。
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _build_area_entries(label_table: Path, roi_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(label_table)
    areas: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["index"])
        if index == 0:
            continue
        label_en = row["label_en"].strip()
        label_zh = row["label_zh"].strip()
        # 使用标签索引做前缀，避免不同脑区在 slugify 后发生文件名冲突。
        filename = f"{index:04d}_{_slugify(label_en)}.nii.gz"
        areas.append(
            {
                "index": index,
                "label_en": label_en,
                "label_zh": label_zh,
                "roi_filename": filename,
                "roi_path": _to_repo_relative(roi_dir / filename),
            }
        )
    return areas


def build_atlas_registry() -> dict[str, Any]:
    """根据现有 atlas 资源构建 registry。"""
    registry: dict[str, Any] = {
        "schema_version": 1,
        "simnibs_mni_template": _to_repo_relative(SIMNIBS_MNI_TEMPLATE),
        "atlas_root": _to_repo_relative(ATLAS_ROOT),
        "standardized_root": _to_repo_relative(ATLAS_STANDARDIZED_DIR),
        "atlases": {},
    }

    atlas_specs = [
        {
            "name": "BN",
            "raw_atlas": ATLAS_SOURCE_DIR / "BN" / "BN_Atlas_246_1mm.nii.gz",
            "raw_label_table": ATLAS_SOURCE_DIR / "BN" / "BN_Atlas_246_labels_zh.csv",
            "color_table": ATLAS_COLOR_TABLE_DIR / "BN_Atlas_246_labels_zh.csv",
            "space_template": ATLAS_SOURCE_DIR / "BN" / "MNI152_T1_1mm.nii.gz",
            "space_name": "bn_mni152_1mm",
            "is_discrete": True,
            "interpolation": "nearest",
        },
        {
            "name": "JulichBrainAtlas",
            "raw_atlas": (
                ATLAS_SOURCE_DIR
                / "JulichBrainAtlas"
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz"
            ),
            "raw_label_table": (
                ATLAS_SOURCE_DIR
                / "JulichBrainAtlas"
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv"
            ),
            "color_table": (
                ATLAS_COLOR_TABLE_DIR
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv"
            ),
            "space_template": (
                ATLAS_SOURCE_DIR
                / "JulichBrainAtlas"
                / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
            ),
            "space_name": "icbm152_09c",
            "is_discrete": True,
            "interpolation": "nearest",
        },
    ]

    for resolution in (64, 128, 256, 512, 1024):
        atlas_name = f"DiFuMo_{resolution}"
        difumo_dir = ATLAS_SOURCE_DIR / "DiFuMo" / atlas_name
        atlas_specs.append(
            {
                "name": atlas_name,
                "raw_atlas": difumo_dir / "2mm" / "maps.nii.gz",
                "raw_label_table": difumo_dir
                / f"labels_{resolution}_dictionary_zh.csv",
                "color_table": ATLAS_COLOR_TABLE_DIR
                / f"labels_{resolution}_dictionary_zh.csv",
                "space_template": ATLAS_SOURCE_DIR
                / "DiFuMo"
                / "tpl-MNI152NLin6Asym_res-02_T1w_difumo-grid.nii.gz",
                "space_name": "mni152nlin6asym_difumo_grid",
                "is_discrete": False,
                "interpolation": "linear",
            }
        )

    for spec in atlas_specs:
        # standardized 目录下统一组织 atlas、标签表和 ROI，便于在线侧按固定规则读取。
        standardized_dir = ATLAS_STANDARDIZED_DIR / spec["name"]
        roi_dir = standardized_dir / "rois"
        label_table_target = standardized_dir / "labels_zh.csv"
        atlas_target = standardized_dir / "atlas_simnibs_mni.nii.gz"
        areas = _build_area_entries(Path(spec["raw_label_table"]), roi_dir)

        registry["atlases"][spec["name"]] = {
            "name": spec["name"],
            "raw_atlas": _to_repo_relative(Path(spec["raw_atlas"])),
            "raw_label_table": _to_repo_relative(Path(spec["raw_label_table"])),
            "color_table": _to_repo_relative(Path(spec["color_table"])),
            "space_template": _to_repo_relative(Path(spec["space_template"])),
            "space_name": spec["space_name"],
            "is_discrete": spec["is_discrete"],
            "interpolation": spec["interpolation"],
            "standardized_dir": _to_repo_relative(standardized_dir),
            "standardized_atlas": _to_repo_relative(atlas_target),
            "standardized_label_table": _to_repo_relative(label_table_target),
            "roi_dir": _to_repo_relative(roi_dir),
            "areas": areas,
        }

    return registry


def write_atlas_registry(path: Path = ATLAS_REGISTRY_PATH) -> Path:
    """写出 atlas registry JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = build_atlas_registry()
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_atlas_registry(path: Path = ATLAS_REGISTRY_PATH) -> dict[str, Any]:
    if not path.exists():
        write_atlas_registry(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_area_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _resolve_area_entry(area: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(area)
    resolved["roi_path"] = str(_resolve_repo_path(area["roi_path"]))
    return resolved


def _resolve_atlas_spec(spec: dict[str, Any]) -> dict[str, Any]:
    # 对外返回的 spec 一律展开成绝对路径，避免各处重复 resolve。
    resolved = dict(spec)
    for key in (
        "raw_atlas",
        "raw_label_table",
        "color_table",
        "space_template",
        "standardized_dir",
        "standardized_atlas",
        "standardized_label_table",
        "roi_dir",
    ):
        resolved[key] = str(_resolve_repo_path(spec[key]))
    resolved["areas"] = [_resolve_area_entry(area) for area in spec["areas"]]
    return resolved


def get_atlas_spec(
    atlas_name: str, registry: dict[str, Any] | None = None
) -> dict[str, Any]:
    if registry is None:
        registry = load_atlas_registry()
    try:
        return _resolve_atlas_spec(registry["atlases"][atlas_name])
    except KeyError as exc:
        raise KeyError(f"未找到 atlas: {atlas_name}") from exc


def _find_matching_areas(
    atlas_name: str,
    area_name: str,
    registry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # 同时支持英文名、中文名、文件名 stem 以及直接传 index。
    spec = get_atlas_spec(atlas_name, registry)
    area_key = _normalize_area_key(area_name)
    matches: list[dict[str, Any]] = []
    for area in spec["areas"]:
        if area_key in {
            _normalize_area_key(area["label_en"]),
            _normalize_area_key(area["label_zh"]),
            _normalize_area_key(Path(area["roi_filename"]).stem),
            str(area["index"]),
        }:
            matches.append(area)
    return spec, matches


def _build_merged_roi_path(
    spec: dict[str, Any], area_name: str, matches: list[dict[str, Any]]
) -> Path:
    merged_dir = Path(spec["standardized_dir"]) / "merged_rois"
    merged_dir.mkdir(parents=True, exist_ok=True)
    match_token = "_".join(
        str(area["index"])
        for area in sorted(matches, key=lambda item: int(item["index"]))
    )
    return merged_dir / f"{_slugify(area_name)}__merged__{match_token}.nii.gz"


def _ensure_merged_roi(
    spec: dict[str, Any], area_name: str, matches: list[dict[str, Any]]
) -> Path:
    # 同名脑区只在首次命中时生成一次合并 mask，后续直接复用缓存结果。
    merged_path = _build_merged_roi_path(spec, area_name, matches)
    source_paths = [Path(area["roi_path"]) for area in matches]
    if merged_path.exists():
        merged_mtime = merged_path.stat().st_mtime
        if all(
            source_path.exists() and source_path.stat().st_mtime <= merged_mtime
            for source_path in source_paths
        ):
            return merged_path

    if len(source_paths) == 1:
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths[0], merged_path)
        return merged_path

    # 多个 component 时做体素级并集，输出仍保持二值 mask。
    first_img = nib.load(str(source_paths[0]))
    merged_mask = np.asarray(first_img.dataobj, dtype=np.uint8) > 0
    for source_path in source_paths[1:]:
        source_img = nib.load(str(source_path))
        merged_mask |= np.asarray(source_img.dataobj, dtype=np.uint8) > 0

    header = first_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(merged_mask.astype(np.uint8), first_img.affine, header),
        merged_path,
    )
    return merged_path


def get_standardized_roi_path(
    atlas_name: str,
    area_name: str,
    registry: dict[str, Any] | None = None,
) -> Path:
    """根据 atlas 名称和脑区名称定位标准化 ROI 文件；重名脑区会自动合并。"""
    spec, matches = _find_matching_areas(atlas_name, area_name, registry)
    if len(matches) == 1:
        return Path(matches[0]["roi_path"])
    if len(matches) > 1:
        return _ensure_merged_roi(spec, area_name, matches)
    raise KeyError(f"未在 atlas {atlas_name} 中找到脑区: {area_name}")


def iter_atlas_specs(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if registry is None:
        registry = load_atlas_registry()
    return [get_atlas_spec(name, registry) for name in registry["atlases"]]
