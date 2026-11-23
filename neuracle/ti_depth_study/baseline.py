"""通过现有 CLI 运行可追溯的原生基线及共同正向任务。"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist

from neuracle.ti_depth_study.records import digest, write_json
from neuracle.utils.constants import STANDARD_COND
from simnibs.utils.csv_reader import read_csv_positions


PHYSICS = {"montage": "EEG10-10_NSN", "anisotropy_type": "scalar", "electrode_radius": 6.0,
           "conductivity_config": STANDARD_COND.copy()}
FIELD_MODEL = "common_single_layer_scalar"


def content_id(config: dict[str, Any]) -> str:
    """根据完整的输入配置生成稳定实体 ID。

    Parameters
    ----------
    config : dict
        字段顺序无关的输入配置，包含实验的物理或上游散列条件。

    Returns
    -------
    str
        配置 JSON 的完整 SHA-256。
    """
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def inverse_config(target: dict[str, str], method: str) -> dict[str, Any]:
    """生成共同先导协议的原生逆向输入。

    Parameters
    ----------
    target : dict
        02 已登记的 atlas_name 和 label。
    method : str
        原生方法标识 leadfield_free 或 leadfield_based。

    Returns
    -------
    dict
        原生 CLI 可直接读取的 params.json 内容。
    """
    config = PHYSICS | {"roi_type": "atlas", "roi_param": {"atlas_param": {"name": target["atlas_name"], "area": target["label"]}},
                        "target_threshold": 0.10, "optimization_method": method}
    if method == "leadfield_free":
        config |= {"current_A": [2.0, -2.0], "current_B": [2.0, -2.0]}
    return config


def forward_config(labels: list[str], currents: tuple[list[float], list[float]]) -> dict[str, Any]:
    """从具名布局和两路有符号电流生成共同正向参数。

    Parameters
    ----------
    labels : list of str
        依次为 A+、A−、B+、B−。
    currents : tuple of list
        每路两个有符号电流，单位 mA。

    Returns
    -------
    dict
        原生正向 CLI 的 params.json 内容。
    """
    if len(labels) != 4 or len(set(labels)) != 4 or any(len(channel) != 2 for channel in currents):
        raise ValueError("正向布局需要四个互异帽位和两路成对电流")
    if any(channel[0] <= 0 or channel[1] >= 0 or not np.isclose(sum(channel), 0) for channel in currents):
        raise ValueError("单频通道电流无效或为零")
    return PHYSICS | {
        "electrode_A": [dict(name=name, current_mA=current) for name, current in zip(labels[:2], currents[0])],
        "electrode_B": [dict(name=name, current_mA=current) for name, current in zip(labels[2:], currents[1])],
    }


def run_task(
    study: Path, module: str, subject: str, task: str, config: dict[str, Any], commands: Path,
    required_globs: tuple[str, ...],
) -> Path:
    """调用现有仿真入口，成功产物按配置和散列复用。

    Parameters
    ----------
    study : Path
        数据与本地输出的研究根目录。
    module, subject, task : str
        原生入口模块短名、头模 ID 及实验任务 ID。
    config : dict
        定义该物理任务的 JSON 输入。
    commands : Path
        本次运行的 PowerShell 实际调用记录。
    required_globs : tuple of str
        每条都要命中至少一个文件才可验收的输出模式。

    Returns
    -------
    Path
        通过原生退出码、参数及输出散列检查的任务目录。
    """
    simulations = study / "simulations"
    simulations.mkdir(exist_ok=True)
    base = simulations / f"{module}_{task}"
    attempt = 1
    directory = base
    while directory.exists():
        manifest = directory / "study_execution.json"
        params = directory / "params.json"
        if manifest.is_file() and params.is_file() and json.loads(params.read_text(encoding="utf-8")) == config:
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            hashes = prior.get("sha256", {})
            matches = [sorted(directory.glob(pattern)) for pattern in required_globs]
            if prior["exit_code"] == 0 and all(matches) and all(hashes.get(path.relative_to(directory).as_posix()) == digest(path) for paths in matches for path in paths):
                return directory
        attempt += 1
        directory = base.with_name(f"{base.name}_try{attempt:02}")
    directory.mkdir()
    write_json(directory / "params.json", config)
    actual_task = directory.name.removeprefix(module + "_")
    invocation = [sys.executable, "-X", "utf8", "-u", "-m", f"neuracle.{module}", str(study), actual_task, subject]
    with commands.open("a", encoding="utf-8") as stream:
        rendered = " ".join("'" + part.replace("'", "''") + "'" for part in invocation)
        stream.write(f"\n# {Path.cwd()}\n& {rendered}\n")
    print(f"{subject} {module} {actual_task}", flush=True)
    with (directory / "study_console.log").open("wb") as console:
        process = subprocess.run(invocation, stdout=console, stderr=subprocess.STDOUT, check=False)
    matches = [sorted(directory.glob(pattern)) for pattern in required_globs]
    code = process.returncode
    write_json(directory / "study_execution.json", {"exit_code": code, "argv": invocation,
               "sha256": {path.relative_to(directory).as_posix(): digest(path) for paths in matches for path in paths} if code == 0 else {}})
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(f"# exit {code}\n")
    if code != 0 or not all(matches):
        missing = [pattern for pattern, paths in zip(required_globs, matches) if not paths]
        raise RuntimeError(f"{module} {actual_task} 退出码 {code}，缺失 {missing}，详见 {directory / 'study_console.log'}")
    return directory


def candidate_layouts(cap: Path, origins: dict[tuple[str, ...], set[str]]) -> dict[tuple[str, ...], set[str]]:
    """扩展每个原生布局的单帽位相邻扰动，并按具名物理配置去重。

    Parameters
    ----------
    cap : Path
        当前头模的 NSN 帽位 CSV。
    origins : dict
        四个有序帽位到贡献该布局的路线和案例键集合。

    Returns
    -------
    dict
        合并种子和每电极至多一个最近合法替换。扰动保留最小圆片间距。
    """
    kinds, points, _, names, _, _ = read_csv_positions(str(cap))
    positions = {name: np.asarray(point) for kind, point, name in zip(kinds, points, names) if kind in {"Electrode", "ReferenceElectrode"} and name}
    layouts = {labels: methods.copy() for labels, methods in origins.items()}
    min_center_dist = 2 * PHYSICS["electrode_radius"] + 5.0
    for seed, methods in sorted(origins.items()):
        if len(seed) != 4 or len(set(seed)) != 4 or not all(name in positions for name in seed):
            raise ValueError(f"原生布局无法映射到四个互异的个体帽位: {seed}")
        for index, name in enumerate(seed):
            neighbors = sorted((candidate for candidate in positions if candidate not in seed),
                               key=lambda candidate: (np.linalg.norm(positions[candidate] - positions[name]), candidate))
            for neighbor in neighbors:
                layout = (*seed[:index], neighbor, *seed[index+1:])
                if np.min(pdist([positions[electrode] for electrode in layout])) >= min_center_dist:
                    layouts.setdefault(layout, set()).update(f"neighbor:{method}" for method in methods)
                    break
    return layouts
