from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JULICH_DIR = REPO_ROOT / "atlas" / "JulichBrainAtlas"
OUTPUT_CSV = JULICH_DIR / "JulichBrainAtlas_translation_mapping.csv"


PAREN_TRANSLATIONS = {
    "PostCG": "中央后回",
    "IPL": "顶下小叶",
    "LOC": "外侧枕皮层",
    "CoS": "侧副沟",
    "Insula": "脑岛",
    "SFG": "额上回",
    "HESCHL": "赫氏回",
    "STS": "颞上沟",
    "STG": "颞上回",
    "FusG": "梭状回",
    "MFG": "额中回",
    "SPL": "顶上小叶",
    "PreCG": "中央前回",
    "Frontal Operculum": "额盖",
    "Amygdala": "杏仁核",
    "Hippocampus": "海马",
    "pACC": "前扣带皮层前部",
    "sACC": "胼胝体下前扣带皮层",
    "ACC": "前扣带皮层",
    "SFS": "额上沟",
    "PhG": "海马旁回",
    "POS": "顶枕沟",
    "FPole": "额极",
    "IFG": "额下回",
    "IFS": "额下沟",
    "PreCS": "中央前沟",
    "Cuneus": "楔叶",
    "LingG": "舌回",
    "POperc": "顶叶盖",
    "GapMap": "间隙图",
    "OFC": "眶额皮层",
    "SMA": "辅助运动区",
    "SMG": "缘上回",
    "CalcS": "距状沟",
    "OTS": "枕颞沟",
    "IPS": "顶内沟",
    "Basal Forebrain": "基底前脑",
    "Metathalamus": "后丘脑",
    "Subthalamus": "底丘脑",
    "Ventral Pallidum": "腹侧苍白球",
    "PostCS": "中央后沟",
    "MFG": "额中回",
    "IFG": "额下回",
    "SMA, mesial SFG": "辅助运动区、额上回内侧部",
    "Hippocampal Region, Entorhinal Cortex": "海马区、内嗅皮层",
    "IFS, PreCS": "额下沟、中央前沟",
    "MFG, IFG": "额中回、额下回",
    "STG, SMG": "颞上回、缘上回",
    "V1, 17, CalcS": "V1区、17区、距状沟",
    "Cerebellum, Dorsal Dentate Nucleus": "小脑、背侧齿状核",
    "Cerebellum, Ventral Dentate Nucleus": "小脑、腹侧齿状核",
    "Cerebellum, Fastigial Nucleus": "小脑、顶核",
    "Cerebellum, Interposed Nucleus": "小脑、中间核",
    "Hippocampus, Subicular complex": "海马、下托复合体",
    "Hippocampus, Transsubiculum": "海马、前下托",
    "Thalamus, ventral anterior Nucleus (ventral)": "丘脑、腹前核腹侧部",
}


TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    ("Thalamus", "丘脑"),
    ("Midbrain", "中脑"),
    ("Cerebellum", "小脑"),
    ("Ventral Striatum", "腹侧纹状体"),
    ("Nucleus Ruber", "红核"),
    ("Substantia Nigra", "黑质"),
    ("Fundus of Putamen", "壳核底部"),
    ("Fundus of Caudate Nucleus", "尾状核底部"),
    ("Basal Forebrain", "基底前脑"),
    ("Bed Nucleus", "床核"),
    ("Tuberculum", "结节"),
    ("Terminal islands", "终纹岛"),
    ("Entorhinal Cortex", "内嗅皮层"),
    ("Hippocampal Region", "海马区"),
    ("PiriformCortexMesial", "内侧梨状皮层"),
    ("temporobasal", "颞底部"),
    ("temporal", "颞部"),
    ("parvocellular part", "小细胞部"),
    ("magnocellular part", "大细胞部"),
    ("pars compacta", "致密部"),
    ("pars reticulata", "网状部"),
    ("anterior pulvinar", "前丘脑枕"),
    ("inferior pulvinar", "下丘脑枕"),
    ("medial pulvinar", "内侧丘脑枕"),
    ("lateral pulvinar", "外侧丘脑枕"),
    ("posterior Nucleus", "后核"),
    ("lateral dorsal Nucleus", "外侧背核"),
    ("zona incerta", "不确定带"),
    ("anteromedial Nucleus", "前内侧核"),
    ("ventral lateral posterior Nucleus", "腹外侧后核"),
    ("limitans Nucleus", "界核"),
    ("reticular Nucleus", "网状核"),
    ("paraventricular Nucleus", "室旁核"),
    ("mediodorsal Nucleus", "内背核"),
    ("parafascicular Nucleus", "束旁核"),
    ("lateral posterior Nucleus", "外侧后核"),
    ("ventral medial Nucleus", "腹内侧核"),
    ("centromedian Nucleus", "中央中核"),
    ("medioventral Nucleus", "内侧腹核"),
    ("ventral anterior Nucleus, magnocellular part", "腹前核大细胞部"),
    ("ventral intermediate Nucleus", "腹中间核"),
    ("anterior intralaminar nuclei", "前板内核群"),
    ("suprageniculate Nucleus", "膝上核"),
    ("ventral posterior lateral Nucleus", "腹后外侧核"),
    ("Subicular complex", "下托复合体"),
    ("Transsubiculum", "前下托"),
    ("ventral lateral anterior Nucleus", "腹外侧前核"),
    ("ventral posterior medial Nucleus, parvocellular part", "腹后内侧核小细胞部"),
    ("ventral anterior Nucleus (ventral)", "腹前核腹侧部"),
    ("ventral posterior medial Nucleus", "腹后内侧核"),
    ("anteroventral Nucleus", "前腹核"),
    ("subparafascicular Nucleus", "束旁下核"),
    ("ventroposterior inferior Nucleus", "腹后下核"),
    ("medial Accumbens", "内侧伏隔核"),
    ("lateral Accumbens", "外侧伏隔核"),
    ("preSMA", "前辅助运动区"),
    ("mesial SFG", "额上回内侧部"),
    ("V1, 17", "V1区、17区"),
    ("V2, 18", "V2区、18区"),
    ("Dorsal Dentate Nucleus", "背侧齿状核"),
    ("Ventral Dentate Nucleus", "腹侧齿状核"),
    ("Fastigial Nucleus", "顶核"),
    ("Interposed Nucleus", "中间核"),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def translate_parenthetical(text: str) -> str:
    value = clean(text)
    if value in PAREN_TRANSLATIONS:
        return PAREN_TRANSLATIONS[value]

    translated = value
    for src, dst in sorted(TEXT_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True):
        translated = re.sub(re.escape(src), dst, translated, flags=re.IGNORECASE)
    translated = translated.replace(",", "、")
    translated = re.sub(r"\s+", "", translated)
    return translated


def translate_label_base(label: str) -> str:
    value = clean(label)

    if value.startswith("Area "):
        prefix = value[5:]
        if "(" in prefix and prefix.endswith(")"):
            area_code, paren = prefix.split("(", 1)
            area_code = clean(area_code)
            paren = paren[:-1]
            paren_zh = translate_parenthetical(paren)
            return f"{area_code}区（{paren_zh}）" if paren_zh else f"{area_code}区"
        return f"{prefix}区"

    if "(" in value and value.endswith(")"):
        code, paren = value.split("(", 1)
        code = clean(code)
        paren = paren[:-1]
        paren_zh = translate_parenthetical(paren)
        return f"{code}区（{paren_zh}）" if paren_zh else f"{code}区"

    translated = translate_parenthetical(value)
    return translated if translated else value


def main() -> None:
    xml_path = JULICH_DIR / "JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.xml"
    root = ET.parse(xml_path).getroot()
    structures = root.find("Structures")
    if structures is None:
        raise ValueError("Missing Structures node in Julich XML")

    unique: dict[str, str] = {}
    for node in structures.findall("Structure"):
        label_base = clean((node.text or "").strip())
        unique[label_base] = translate_label_base(label_base)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_base_en", "label_base_zh"])
        for label_base in sorted(unique):
            writer.writerow([label_base, unique[label_base]])

    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
