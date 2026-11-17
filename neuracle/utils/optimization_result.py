"""两种逆向方法统一输出实际电极及 mA 电流。"""

import json
import math
from pathlib import Path


def write_electrode_mapping(
    output_dir: Path,
    method: str,
    labels: list[str],
    current_a: list[float],
    current_b: list[float],
) -> Path:
    """校验业务结果并原子补充映射，保留 free 原有诊断字段。

    Parameters
    ----------
    output_dir : Path
        任务结果目录。
    method : str
        本次实际执行的方法。
    labels : list[str]
        A+、A−、B+、B− 四个电极名。
    current_a, current_b : list[float]
        实际生成结果场的成对电流，单位 mA；based 允许零幅值。

    Returns
    -------
    Path
        完整 electrode_mapping.json 路径。
    """
    if method not in ("leadfield_free", "leadfield_based"):
        raise ValueError("未知优化方法")
    if len(labels) != 4 or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("结果必须包含 A+、A−、B+、B− 四个电极标签")
    for currents in (current_a, current_b):
        if len(currents) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in currents
        ):
            raise ValueError("结果电流必须是两个有限数字，单位 mA")
        if currents[0] < 0 or currents[1] > 0 or abs(sum(currents)) > 1e-6:
            raise ValueError("结果电流顺序必须为正、负且总和为零")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "electrode_mapping.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(
        optimization_method=method,
        mapped_labels=labels,
        current_A=current_a,
        current_B=current_b,
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
