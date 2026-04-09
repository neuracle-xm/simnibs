from __future__ import annotations

import colorsys
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = REPO_ROOT / "atlas"
BN_MAPPING_PATH = ATLAS_DIR / "BN" / "BN_abbreviation_mapping.csv"
DIFUMO_MAPPING_PATH = ATLAS_DIR / "DiFuMo_translation_mapping.csv"
JULICH_MAPPING_PATH = (
    ATLAS_DIR / "JulichBrainAtlas" / "JulichBrainAtlas_translation_mapping.csv"
)


MULTI_REPLACEMENTS: list[tuple[str, str]] = [
    ("No network found", "未发现网络"),
    ("Planum temporale", "颞平面"),
    ("Transverse sinus", "横窦"),
    ("parieto-occipital", "顶枕"),
    ("antero-superior", "前上部"),
    ("postero-superior", "后上部"),
    ("postero-inferior", "后下部"),
    ("fronto-parietal", "额顶"),
    ("temporo-parietal", "颞顶"),
    ("parieto-temporal", "顶颞"),
    ("postcentral and precentral gyri", "中央后回和中央前回"),
    ("precentral and postcentral gyri", "中央前回和中央后回"),
    ("precentral gyrus", "中央前回"),
    ("postcentral gyrus", "中央后回"),
    ("angular gyrus", "角回"),
    ("fusiform gyrus", "梭状回"),
    ("cingulate cortex", "扣带皮层"),
    ("calcarine cortex", "距状皮层"),
    ("inferior frontal gyrus", "额下回"),
    ("middle frontal gyrus", "额中回"),
    ("superior frontal gyrus", "额上回"),
    ("superior temporal gyrus", "颞上回"),
    ("middle temporal gyrus", "颞中回"),
    ("inferior temporal gyrus", "颞下回"),
    ("parahippocampal gyrus", "海马旁回"),
    ("orbitofrontal cortex", "眶额皮层"),
    ("frontal pole", "额极"),
    ("temporal pole", "颞极"),
    ("occipital pole", "枕极"),
    ("lateral occipital", "外侧枕叶"),
    ("precuneus", "楔前叶"),
    ("cuneus", "楔叶"),
    ("insula", "脑岛"),
    ("insulae", "脑岛"),
    ("thalamus", "丘脑"),
    ("amygdala", "杏仁核"),
    ("hippocampus", "海马"),
    ("cerebellum", "小脑"),
    ("brainstem", "脑干"),
    ("parietal", "顶叶"),
    ("occipital", "枕叶"),
    ("frontal", "额叶"),
    ("temporal", "颞叶"),
    ("sulcus", "沟"),
    ("gyrus", "回"),
    ("cortex", "皮层"),
    ("posterior", "后部"),
    ("anterior", "前部"),
    ("superior", "上部"),
    ("inferior", "下部"),
    ("medial", "内侧"),
    ("lateral", "外侧"),
    ("dorsal", "背侧"),
    ("ventral", "腹侧"),
    ("mesial", "内侧"),
    ("central", "中央"),
    ("with", "伴"),
    ("parts of", ""),
    ("part of", ""),
    ("and", "和"),
    ("of", "的"),
    ("sinus", "窦"),
    ("nucleus", "核"),
    ("gyri", "脑回"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_rgba(index: int, total: int) -> tuple[int, int, int, int]:
    hue = ((index - 1) % max(total, 1)) / max(total, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return int(r * 255), int(g * 255), int(b * 255), 255


def apply_phrase_replacements(text: str) -> str:
    result = text
    for src, dst in sorted(
        MULTI_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True
    ):
        pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
        result = pattern.sub(dst, result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def cleanup_pure_chinese_label(text: str) -> str:
    result = text
    result = re.sub(r"（[^（）]*[A-Za-z][^（）]*）", "", result)
    result = re.sub(r"\([^()]*[A-Za-z][^()]*\)", "", result)
    result = re.sub(r"[A-Za-z][A-Za-z0-9./+\-]*", "", result)
    result = result.replace(",", "、")
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"\s*（\s*", "（", result)
    result = re.sub(r"\s*）\s*", "）", result)
    result = re.sub(r"\s*、\s*", "、", result)
    result = re.sub(r"\s+", "", result)
    return result


def maybe_add_english_note(original: str, translated: str) -> str:
    return cleanup_pure_chinese_label(translated)


def translate_free_text_label(label: str) -> str:
    translated = apply_phrase_replacements(label)
    return maybe_add_english_note(label, translated)


def translate_julich_label(label: str) -> str:
    translated = apply_phrase_replacements(label)
    translated = re.sub(r"\(([^()]+)\)", lambda m: f"（{cleanup_pure_chinese_label(m.group(1))}）", translated)
    translated = re.sub(r"\s+", " ", translated).strip()
    translated = translated.replace(",", "、")
    translated = re.sub(r"（\s*）", "", translated)
    return translated


def load_bn_mapping() -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if not BN_MAPPING_PATH.exists():
        return mapping
    with BN_MAPPING_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["abbr"]] = row
    return mapping


def load_difumo_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not DIFUMO_MAPPING_PATH.exists():
        return mapping
    with DIFUMO_MAPPING_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["label_en"]] = row["label_zh"]
    return mapping


def load_julich_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not JULICH_MAPPING_PATH.exists():
        return mapping
    with JULICH_MAPPING_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["label_base_en"]] = row["label_base_zh"]
    return mapping


def parse_bn_label(label: str, mapping: dict[str, dict[str, str]]) -> tuple[str, str]:
    if label == "Unknown":
        return label, "未知"

    hemisphere = ""
    core = label
    if label.endswith("_L"):
        hemisphere = "左侧"
        core = label[:-2]
    elif label.endswith("_R"):
        hemisphere = "右侧"
        core = label[:-2]

    zh_core = mapping.get(core, {}).get("full_name_zh", core)

    if hemisphere:
        return label, f"{zh_core}（{hemisphere}）"
    return label, zh_core


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate_bn_labels() -> Path:
    source = ATLAS_DIR / "BN" / "BN_Atlas_246_LUT.txt"
    output = ATLAS_DIR / "BN" / "BN_Atlas_246_labels_zh.csv"
    mapping = load_bn_mapping()

    rows: list[list[object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            index_str, label_en, r, g, b, a = line.split()
            _, label_zh = parse_bn_label(label_en, mapping)
            rows.append(
                [
                    int(index_str),
                    label_en,
                    label_zh,
                    int(r),
                    int(g),
                    int(b),
                    255 if int(index_str) != 0 else 0,
                ]
            )

    write_csv(output, ["index", "label_en", "label_zh", "r", "g", "b", "a"], rows)
    return output


def generate_difumo_labels() -> list[Path]:
    outputs: list[Path] = []
    difumo_mapping = load_difumo_mapping()
    for atlas_dir in sorted(ATLAS_DIR.glob("DiFuMo_*")):
        csv_candidates = list(atlas_dir.glob("labels_*_dictionary.csv"))
        if not csv_candidates:
            continue
        source = csv_candidates[0]
        output = atlas_dir / f"{source.stem}_zh.csv"

        rows: list[list[object]] = []
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            items = list(reader)

        total = len(items)
        for item in items:
            index = int(item["Component"])
            label_en = item["Difumo_names"].strip()
            label_zh = difumo_mapping.get(label_en, translate_free_text_label(label_en))
            r, g, b, a = generate_rgba(index, total)
            rows.append([index, label_en, label_zh, r, g, b, a])

        write_csv(
            output,
            ["index", "label_en", "label_zh", "r", "g", "b", "a"],
            rows,
        )
        outputs.append(output)
    return outputs


def parse_julich_structures(
    xml_path: Path,
    hemisphere_suffix: str,
    hemisphere_zh: str,
    offset: int,
    mapping: dict[str, str],
) -> list[list[object]]:
    root = ET.parse(xml_path).getroot()
    structures = root.find("Structures")
    if structures is None:
        raise ValueError(f"Missing Structures node in {xml_path}")

    rows: list[list[object]] = []
    for node in structures.findall("Structure"):
        grayvalue = int(node.attrib["grayvalue"])
        label_base = (node.text or "").strip()
        label_en = f"{label_base}_{hemisphere_suffix}"
        label_zh_base = mapping.get(label_base, translate_julich_label(label_base))
        label_zh = f"{label_zh_base}（{hemisphere_zh}）"
        rows.append([grayvalue + offset, label_en, label_zh])
    return rows


def generate_julich_labels() -> Path:
    atlas_dir = ATLAS_DIR / "JulichBrainAtlas"
    output = atlas_dir / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv"
    mapping = load_julich_mapping()

    left_rows = parse_julich_structures(
        atlas_dir / "JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.xml",
        "lh",
        "左半球",
        0,
        mapping,
    )
    right_rows = parse_julich_structures(
        atlas_dir / "JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.xml",
        "rh",
        "右半球",
        len(left_rows),
        mapping,
    )
    combined = sorted(left_rows + right_rows, key=lambda row: int(row[0]))
    total = len(combined)

    rows: list[list[object]] = []
    for index, label_en, label_zh in combined:
        r, g, b, a = generate_rgba(int(index), total)
        rows.append([index, label_en, label_zh, r, g, b, a])

    write_csv(
        output,
        ["index", "label_en", "label_zh", "r", "g", "b", "a"],
        rows,
    )
    return output


def main() -> None:
    generated: list[Path] = []
    generated.append(generate_bn_labels())
    generated.extend(generate_difumo_labels())
    generated.append(generate_julich_labels())

    print("Generated label tables:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
