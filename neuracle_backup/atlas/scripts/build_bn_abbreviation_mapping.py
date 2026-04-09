from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_XLSX = REPO_ROOT / "data" / "BNA_subregions.xlsx"
OUTPUT_TSV = REPO_ROOT / "atlas" / "BN" / "BN_abbreviation_mapping.csv"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_abbreviation(abbreviation: str) -> str:
    value = clean_text(abbreviation)
    if value == "TE1.0 and TE1.2":
        return "TE1.0/TE1.2"
    return value


def translate_bn_full_name(full_name_en: str) -> str:
    special = {
        "medial area 8": "8区内侧部",
        "dorsolateral area 8": "8区背外侧部",
        "lateral area 9": "9区外侧部",
        "dorsolateral area 6": "6区背外侧部",
        "medial area 6": "6区内侧部",
        "medial area 9": "9区内侧部",
        "medial area 10": "10区内侧部",
        "dorsal area 9/46": "9/46区背侧部",
        "inferior frontal junction": "额下回交界区",
        "area 46": "46区",
        "ventral area 9/46": "9/46区腹侧部",
        "ventrolateral area 8": "8区腹外侧部",
        "ventrolateral area 6": "6区腹外侧部",
        "lateral area10": "10区外侧部",
        "dorsal area 44": "44区背侧部",
        "inferior frontal sulcus": "额下沟",
        "caudal area 45": "45区尾侧部",
        "rostral area 45": "45区嘴侧部",
        "opercular area 44": "44区盖部",
        "ventral area 44": "44区腹侧部",
        "medial area 14": "14区内侧部",
        "orbital area 12/47": "12/47区眶部",
        "lateral area 11": "11区外侧部",
        "medial area 11": "11区内侧部",
        "area 13": "13区",
        "lateral area 12/47": "12/47区外侧部",
        "area 4(head and face region)": "4区（头面部）",
        "caudal dorsolateral area 6": "6区尾侧背外侧部",
        "area 4(upper limb region)": "4区（上肢区）",
        "area 4(trunk region)": "4区（躯干区）",
        "area 4(tongue and larynx region)": "4区（舌和喉部）",
        "caudal ventrolateral area 6": "6区尾侧腹外侧部",
        "area1/2/3 (lower limb region)": "1/2/3区（下肢区）",
        "area 4, (lower limb region)": "4区（下肢区）",
        "medial area 38": "38区内侧部",
        "area 41/42": "41/42区",
        "TE1.0 and TE1.2": "TE1.0和TE1.2区",
        "caudal area 22": "22区尾侧部",
        "lateral area 38": "38区外侧部",
        "rostral area 22": "22区嘴侧部",
        "caudal area 21": "21区尾侧部",
        "rostral area 21": "21区嘴侧部",
        "dorsolateral area37": "37区背外侧部",
        "anterior superior temporal sulcus": "前部颞上沟",
        "intermediate ventral area 20": "20区中间腹侧部",
        "extreme lateroventral area37": "37区极外侧腹侧部",
        "rostral area 20": "20区嘴侧部",
        "intermediate lateral area 20": "20区中间外侧部",
        "ventrolateral area 37": "37区腹外侧部",
        "caudolateral of area 20": "20区尾侧外侧部",
        "caudoventral of area 20": "20区尾侧腹侧部",
        "rostroventral area 20": "20区嘴侧腹侧部",
        "medioventral area37": "37区内侧腹侧部",
        "lateroventral area37": "37区外侧腹侧部",
        "rostral area 35/36": "35/36区嘴侧部",
        "caudal area 35/36": "35/36区尾侧部",
        "area TL (lateral PPHC, posterior parahippocampal gyrus)": "TL区（外侧海马旁后部皮层）",
        "area 28/34 (EC, entorhinal cortex)": "28/34区（内嗅皮层）",
        "area TI(temporal agranular insular cortex)": "TI区（颞叶无颗粒脑岛皮层）",
        "area TH (medial PPHC)": "TH区（内侧海马旁后部皮层）",
        "rostroposterior superior temporal sulcus": "后部颞上沟嘴侧部",
        "caudoposterior superior temporal sulcus": "后部颞上沟尾侧部",
        "rostral area 7": "7区嘴侧部",
        "caudal area 7": "7区尾侧部",
        "lateral area 5": "5区外侧部",
        "postcentral area 7": "7区中央后部",
        "intraparietal area 7(hIP3)": "7区顶内沟部",
        "caudal area 39(PGp)": "39区尾侧部",
        "rostrodorsal area 39(Hip3)": "39区嘴侧背侧部",
        "rostrodorsal area 40(PFt)": "40区嘴侧背侧部",
        "caudal area 40(PFm)": "40区尾侧部",
        "rostroventral area 39(PGa)": "39区嘴侧腹侧部",
        "rostroventral area 40(PFop)": "40区嘴侧腹侧部",
        "medial area 7(PEp)": "7区内侧部",
        "medial area 5(PEm)": "5区内侧部",
        "dorsomedial parietooccipital sulcus(PEr)": "顶枕沟背内侧部",
        "area 31 (Lc1)": "31区",
        "area 1/2/3(upper limb, head and face region)": "1/2/3区（上肢、头面部）",
        "area 1/2/3(tongue and larynx region)": "1/2/3区（舌和喉部）",
        "area 2": "2区",
        "area1/2/3(trunk region)": "1/2/3区（躯干区）",
        "hypergranular insula": "高颗粒脑岛",
        "ventral agranular insula": "腹侧无颗粒脑岛",
        "dorsal agranular insula": "背侧无颗粒脑岛",
        "ventral dysgranular and granular insula": "腹侧乏颗粒及颗粒脑岛",
        "dorsal granular insula": "背侧颗粒脑岛",
        "dorsal dysgranular insula": "背侧乏颗粒脑岛",
        "dorsal area 23": "23区背侧部",
        "rostroventral area 24": "24区嘴侧腹侧部",
        "pregenual area 32": "32区前膝部",
        "ventral area 23": "23区腹侧部",
        "caudodorsal area 24": "24区尾侧背侧部",
        "caudal area 23": "23区尾侧部",
        "subgenual area 32": "32区胼胝体下部",
        "caudal lingual gyrus": "舌回尾侧部",
        "rostral cuneus gyrus": "楔叶回嘴侧部",
        "caudal cuneus gyrus": "楔叶回尾侧部",
        "rostral lingual gyrus": "舌回嘴侧部",
        "ventromedial parietooccipital sulcus": "顶枕沟腹内侧部",
        "middle occipital gyrus": "枕中回",
        "area V5/MT+": "V5/MT+区",
        "occipital polar cortex": "枕极皮层",
        "inferior occipital gyrus": "枕下回",
        "medial superior occipital gyrus": "内侧枕上回",
        "lateral superior occipital gyrus": "外侧枕上回",
        "medial amygdala": "内侧杏仁核",
        "lateral amygdala": "外侧杏仁核",
        "rostral hippocampus": "海马嘴侧部",
        "caudal hippocampus": "海马尾侧部",
        "ventral caudate": "腹侧尾状核",
        "globus pallidus": "苍白球",
        "nucleus accumbens": "伏隔核",
        "ventromedial putamen": "腹内侧壳核",
        "dorsal caudate": "背侧尾状核",
        "dorsolateral putamen": "背外侧壳核",
        "medial pre-frontal thalamus": "内侧前额叶丘脑",
        "pre-motor thalamus": "运动前丘脑",
        "sensory thalamus": "感觉丘脑",
        "rostral temporal thalamus": "嘴侧颞丘脑",
        "posterior parietal thalamus": "顶后丘脑",
        "occipital thalamus": "枕丘脑",
        "caudal temporal thalamus": "尾侧颞丘脑",
        "lateral pre-frontal thalamus": "外侧前额叶丘脑",
    }
    return special.get(clean_text(full_name_en), clean_text(full_name_en))


def parse_xlsx_rows(xlsx_path: Path) -> list[dict[str, str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(xlsx_path) as zf:
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join((t.text or "") for t in si.findall(".//a:t", ns))
            for si in shared_root.findall("a:si", ns)
        ]
        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    last_lobe = ""
    last_gyrus = ""
    for row in sheet_root.findall(".//a:sheetData/a:row", ns)[1:]:
        values: dict[str, str] = {}
        for cell in row.findall("a:c", ns):
            ref = cell.attrib["r"]
            col = "".join(ch for ch in ref if ch.isalpha())
            value_node = cell.find("a:v", ns)
            value = value_node.text if value_node is not None else ""
            if cell.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            values[col] = clean_text(value)

        if values.get("A"):
            last_lobe = values["A"]
        if values.get("B"):
            last_gyrus = values["B"]
        if not values.get("F"):
            continue

        full = values["F"]
        if "," in full:
            abbr, full_name_en = [clean_text(part) for part in full.split(",", 1)]
        else:
            abbr, full_name_en = clean_text(full), clean_text(full)

        rows.append(
            {
                "abbr": normalize_abbreviation(abbr),
                "full_name_en": full_name_en,
                "full_name_zh": translate_bn_full_name(full_name_en),
                "lobe_en": clean_text(last_lobe),
                "gyrus_en": clean_text(last_gyrus),
                "label_id_l": values.get("D", ""),
                "label_id_r": values.get("E", ""),
                "source": "Brainnetome Atlas BNA_subregions.xlsx",
            }
        )
    return rows


def main() -> None:
    rows = parse_xlsx_rows(SOURCE_XLSX)
    unique_rows = {row["abbr"]: row for row in rows}

    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "abbr",
                "full_name_en",
                "full_name_zh",
            ]
        )
        for abbr in sorted(unique_rows):
            row = unique_rows[abbr]
            writer.writerow(
                [
                    row["abbr"],
                    row["full_name_en"],
                    row["full_name_zh"],
                ]
            )

    print(f"Wrote {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
