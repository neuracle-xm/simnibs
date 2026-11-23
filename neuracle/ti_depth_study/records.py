"""研究运行产物的可追溯读写。"""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """对文件内容计算完整 SHA-256，用于研究产物登记。

    Parameters
    ----------
    path : Path
        文件的工作区或外部只读路径。

    Returns
    -------
    str
        64 位十六进制散列。
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp() -> str:
    """生成带工作站时区的运行记录时间。

    Returns
    -------
    str
        ISO 8601 时间戳。
    """
    return datetime.now().astimezone().isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 研究表，兼容 PowerShell CSV 的 BOM。

    Parameters
    ----------
    path : Path
        有表头的 CSV 文件。

    Returns
    -------
    list of dict
        原样保留 CSV 的字符串字段。
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    """先写临时文件，再替换研究表，以便中断后读取完整状态。

    Parameters
    ----------
    path : Path
        目标 CSV 路径，父目录须存在。
    rows : list of dict
        字段顺序按表头，不存在的值写 NA。
    columns : tuple of str, optional
        有空表或可变字段时显式提供表头。
    """
    fields = columns if columns is not None else tuple(rows[0])
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="NA")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path: Path, content: dict[str, Any]) -> None:
    """原子更新 UTF-8 研究配置和状态清单。

    Parameters
    ----------
    path : Path
        目标 JSON 路径，父目录须存在。
    content : dict
        由 JSON 基本类型组成的配置或状态。
    """
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def artifact(study: Path, case_id: str, role: str, path: Path) -> dict[str, str]:
    """登记产物路径和散列，研究内路径以研究根为基准。

    Parameters
    ----------
    study : Path
        研究根目录。
    case_id, role : str
        产物所属案例和用途。
    path : Path
        产物的现有文件路径。

    Returns
    -------
    dict
        可直接写入 artifacts.csv 的一行。
    """
    full_path = path.resolve()
    relative = full_path.relative_to(study) if full_path.is_relative_to(study) else full_path
    return dict(case_id=case_id, role=role, path=relative.as_posix(), sha256=digest(full_path))


def update_index(study: Path, record: dict[str, Any]) -> None:
    """以实验、运行号和阶段为键更新研究索引的一行。

    Parameters
    ----------
    study : Path
        研究根目录，其中的 results 目录已存在。
    record : dict
        当前阶段的进度、科学判定、案例统计及报告路径。
    """
    index = study / "results/index.csv"
    key = ("experiment", "run_id", "stage")
    rows = read_csv(index) if index.exists() else []
    rows = [row for row in rows if any(row[field] != record[field] for field in key)]
    write_csv(index, sorted([*rows, record], key=lambda row: tuple(row[field] for field in key)))
