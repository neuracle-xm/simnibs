from __future__ import annotations

import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = REPO_ROOT / "atlas"
OUTPUT_CSV = ATLAS_DIR / "DiFuMo_translation_mapping.csv"


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
    ("precentral sulcus", "中央前沟"),
    ("postcentral sulcus", "中央后沟"),
    ("central sulcus", "中央沟"),
    ("superior parietal lobule", "顶上小叶"),
    ("callosomarginal sulcus", "胼胝体缘沟"),
    ("interhemispheric fissure", "大脑纵裂"),
    ("intraparietal sulcus", "顶内沟"),
    ("superior frontal sulcus", "额上沟"),
    ("cingulate gyrus", "扣带回"),
    ("superior occipital gyrus", "枕上回"),
    ("supramarginal gyrus", "缘上回"),
    ("skull", "颅骨"),
    ("angular gyrus", "角回"),
    ("angular sulcus", "角沟"),
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
    ("planum polare", "颞极平面"),
    ("planum porale", "颞前平面"),
    ("lateral occipital", "外侧枕叶"),
    ("precuneus", "楔前叶"),
    ("paracentral lobule", "旁中央小叶"),
    ("cuneus", "楔叶"),
    ("insula", "脑岛"),
    ("thalamus", "丘脑"),
    ("amygdala", "杏仁核"),
    ("hippocampus", "海马"),
    ("caudate nucleus", "尾状核"),
    ("caudate", "尾状核"),
    ("putamen", "壳核"),
    ("globus pallidus", "苍白球"),
    ("subthalamic nuclei", "底丘脑核"),
    ("mammillary bodies", "乳头体"),
    ("optic chiasm", "视交叉"),
    ("corpus callosum", "胼胝体"),
    ("forceps major", "大钳"),
    ("forceps minor", "小钳"),
    ("fornix body", "穹隆体"),
    ("uncinate fasciculus", "钩束"),
    ("cerebellar peduncles", "小脑脚"),
    ("cerebellar peduncle", "小脑脚"),
    ("cerebral peduncles", "大脑脚"),
    ("medulla oblongata", "延髓"),
    ("pons", "脑桥"),
    ("fourth ventricle", "第四脑室"),
    ("third ventricle", "第三脑室"),
    ("ventricles", "脑室"),
    ("subsplenial area", "胼胝体压部下区"),
    ("central opercula", "中央盖区"),
    ("pars opercularis", "盖部"),
    ("pars triangularis", "三角部"),
    ("cerebellum", "小脑"),
    ("brainstem", "脑干"),
    ("corona radiata", "放射冠"),
    ("internal capsule", "内囊"),
    ("commissure", "连合"),
    ("fissure", "裂"),
    ("operculum", "盖"),
    ("orbital gyrus", "眶回"),
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
    ("middle", "中部"),
    ("with", "伴"),
    ("parts of", ""),
    ("part of", ""),
    ("and", "和"),
    ("of", "的"),
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def translate_fragment(text: str) -> str:
    result = clean_text(text)
    for src, dst in sorted(MULTI_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True):
        result = re.sub(re.escape(src), dst, result, flags=re.IGNORECASE)
    result = re.sub(r"\bLH\b", "左侧", result)
    result = re.sub(r"\bRH\b", "右侧", result)
    result = re.sub(r"[A-Za-z][A-Za-z0-9./+\-]*", "", result)
    result = result.replace(",", "、")
    result = re.sub(r"\s+", "", result)
    return result


def translate_csf_label(label: str) -> str:
    text = clean_text(label)
    if text.startswith("Cerebrospinal fluid (between ") and text.endswith(")"):
        inner = text[len("Cerebrospinal fluid (between ") : -1]
        if " and " in inner:
            left, right = inner.rsplit(" and ", 1)
            return f"{translate_fragment(left)}与{translate_fragment(right)}之间的脑脊液"
    if text.startswith("Cerebrospinal fluid between ") and " and " in text:
        inner = text[len("Cerebrospinal fluid between ") :].rstrip(")")
        left, right = inner.rsplit(" and ", 1)
        return f"{translate_fragment(left)}与{translate_fragment(right)}之间的脑脊液"
    if text.startswith("Cerebrospinal fluid (superior of ") and text.endswith(")"):
        inner = text[len("Cerebrospinal fluid (superior of ") : -1]
        return f"{translate_fragment(inner)}上方的脑脊液"
    if text == "Cerebrospinal fluid":
        return "脑脊液"
    return ""


def translate_label(label: str) -> str:
    if "Cerebrospinal fluid" in label:
        translated_csf = translate_csf_label(label)
        if translated_csf:
            return translated_csf

    result = clean_text(label)
    return translate_fragment(result)


def collect_unique_labels() -> list[str]:
    labels: set[str] = set()
    for csv_path in sorted(ATLAS_DIR.glob("DiFuMo_*/labels_*_dictionary.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labels.add(clean_text(row["Difumo_names"]))
    return sorted(labels)


def main() -> None:
    labels = collect_unique_labels()
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_en", "label_zh"])
        for label in labels:
            writer.writerow([label, translate_label(label)])
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
