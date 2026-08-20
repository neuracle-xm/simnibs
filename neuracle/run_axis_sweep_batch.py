"""批量执行可扩展的 MNI 三轴 TI 逆向扫参实验。

默认仅生成和校验 102 个任务的执行计划；只有显式传入 ``--execute`` 才会
顺序启动 SimNIBS。每个任务继续复用 ``run_inverse_debug.py``，因此绕过前端、
FastAPI、数据库和任务队列，但仍走本项目的 Neuracle 调用链。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from neuracle import run_inverse_debug as inverse_runner


class CampaignError(ValueError):
    """表示批量实验配置不合法。"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析轴向批量实验命令行参数。

    Parameters
    ----------
    argv : list[str] | None
        显式参数列表；为 None 时读取 ``sys.argv``。

    Returns
    -------
    argparse.Namespace
        已解析的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="规划或顺序执行右半脑 MNI 三轴 TI 逆向扫参",
    )
    parser.add_argument("--config", required=True, help="批量实验 JSON 配置")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真实启动 SimNIBS；不传时只生成、校验并打印计划",
    )
    parser.add_argument("--only-axis", choices=("O", "X", "Y", "Z"), help="只运行指定轴；O 表示共享原点")
    parser.add_argument("--only-repeat", type=int, help="只运行指定重复序号，例如 1")
    parser.add_argument("--only-parameter-set", help="只运行指定 parameter_sets.name")
    parser.add_argument("--limit", type=int, help="最多选择前 N 个任务，适合先做单点验证")
    parser.add_argument(
        "--rerun-succeeded",
        action="store_true",
        help="连已成功的任务也重新执行；默认断点续跑并跳过成功项",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="任一任务失败后立即停止")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    """读取批量实验 JSON object。

    Parameters
    ----------
    path : Path
        批量配置路径。

    Returns
    -------
    dict[str, Any]
        JSON 顶层对象。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"读取配置失败：{path}，{exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError("批量配置顶层必须是 JSON object")
    return value


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """递归合并基础参数和覆盖参数。

    Parameters
    ----------
    base : dict[str, Any]
        基础参数对象。
    overrides : dict[str, Any]
        参数组提供的局部覆盖。

    Returns
    -------
    dict[str, Any]
        不修改输入对象的合并结果。
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def safe_name(value: str, field: str) -> str:
    """校验可安全用于任务 ID 和目录名的字符串。

    Parameters
    ----------
    value : str
        待校验名称。
    field : str
        错误消息使用的字段名。

    Returns
    -------
    str
        已通过字符集校验的名称。
    """
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise CampaignError(f"{field} 只能包含字母、数字、点、下划线和短横线")
    return value


def coordinate_slug(axis: str, value: float) -> str:
    """把轴和坐标转换为稳定的任务名片段。

    Parameters
    ----------
    axis : str
        坐标轴名称；共享原点使用 ``O``。
    value : float
        轴向 MNI 坐标。

    Returns
    -------
    str
        不包含路径特殊字符的坐标片段。
    """
    if axis == "O":
        return "o_000"
    sign = "p" if value >= 0 else "m"
    magnitude = f"{abs(value):g}".replace(".", "d")
    return f"{axis.lower()}_{sign}{magnitude.zfill(3)}"


def require_dict(container: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需的 JSON object 字段。

    Parameters
    ----------
    container : dict[str, Any]
        字段所在对象。
    key : str
        字段名。

    Returns
    -------
    dict[str, Any]
        字段对象。
    """
    value = container.get(key)
    if not isinstance(value, dict):
        raise CampaignError(f"{key} 必须是 JSON object")
    return value


def require_list(container: dict[str, Any], key: str) -> list[Any]:
    """读取必需的非空 JSON 数组字段。

    Parameters
    ----------
    container : dict[str, Any]
        字段所在对象。
    key : str
        字段名。

    Returns
    -------
    list[Any]
        非空字段数组。
    """
    value = container.get(key)
    if not isinstance(value, list) or not value:
        raise CampaignError(f"{key} 必须是非空数组")
    return value


def positions_from_campaign(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """展开轴向配置中的全部 MNI 位点。

    Parameters
    ----------
    campaign : dict[str, Any]
        包含原点开关和三轴坐标的实验配置。

    Returns
    -------
    list[dict[str, Any]]
        按原点、X、Y、Z 顺序生成的位点对象。
    """
    positions: list[dict[str, Any]] = []
    if campaign.get("include_origin", True):
        positions.append({"axis": "O", "value": 0.0, "center": [0.0, 0.0, 0.0]})
    axes = require_dict(campaign, "axes")
    for axis in ("X", "Y", "Z"):
        values = axes.get(axis)
        if not isinstance(values, list):
            raise CampaignError(f"campaign.axes.{axis} 必须是数组")
        seen: set[float] = set()
        for raw_value in values:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise CampaignError(f"campaign.axes.{axis} 坐标必须为数字")
            value = float(raw_value)
            if value == 0:
                raise CampaignError(f"campaign.axes.{axis} 不应包含 0；请使用 include_origin")
            if value in seen:
                raise CampaignError(f"campaign.axes.{axis} 包含重复坐标：{value:g}")
            seen.add(value)
            center = [0.0, 0.0, 0.0]
            center["XYZ".index(axis)] = value
            positions.append({"axis": axis, "value": value, "center": center})
    return positions


def build_jobs(raw: dict[str, Any], config_path: Path) -> tuple[list[dict[str, Any]], Path]:
    """展开轴向实验的参数组、重复轮次和位点。

    Parameters
    ----------
    raw : dict[str, Any]
        原始批量配置。
    config_path : Path
        配置文件路径，用于解析相对路径。

    Returns
    -------
    tuple[list[dict[str, Any]], Path]
        完整任务列表和状态目录。
    """
    campaign = require_dict(raw, "campaign")
    runtime = require_dict(raw, "runtime")
    base_params = require_dict(raw, "base_params")
    campaign_name = safe_name(campaign.get("name"), "campaign.name")
    prefix = safe_name(runtime.get("run_name_prefix"), "runtime.run_name_prefix")
    repetitions = require_list(campaign, "repetitions")
    parameter_sets = require_list(campaign, "parameter_sets")
    positions = positions_from_campaign(campaign)
    state_value = campaign.get("state_dir", f"campaigns/{campaign_name}")
    if not isinstance(state_value, str) or not state_value:
        raise CampaignError("campaign.state_dir 必须是非空字符串")
    state_dir = inverse_runner.resolve_path(state_value, config_path.parent)

    runtime_template = {key: copy.deepcopy(value) for key, value in runtime.items() if key != "run_name_prefix"}
    jobs: list[dict[str, Any]] = []
    ids: set[str] = set()
    for parameter_set in parameter_sets:
        if not isinstance(parameter_set, dict):
            raise CampaignError("campaign.parameter_sets 的元素必须是 object")
        set_name = safe_name(parameter_set.get("name"), "parameter_sets.name")
        overrides = parameter_set.get("overrides", {})
        if not isinstance(overrides, dict):
            raise CampaignError(f"parameter_sets.{set_name}.overrides 必须是 object")
        set_params = deep_merge(base_params, overrides)
        for repetition in repetitions:
            if not isinstance(repetition, dict):
                raise CampaignError("campaign.repetitions 的元素必须是 object")
            index = repetition.get("index")
            seed = repetition.get("seed")
            if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                raise CampaignError("repetitions.index 必须是正整数")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise CampaignError("repetitions.seed 必须是整数或 null")
            for position in positions:
                slug = coordinate_slug(position["axis"], position["value"])
                job_id = f"{set_name}__r{index:02d}__{slug}"
                if job_id in ids:
                    raise CampaignError(f"生成了重复任务 ID：{job_id}")
                ids.add(job_id)
                params = copy.deepcopy(set_params)
                roi_param = params.setdefault("roi_param", {})
                mni_param = roi_param.setdefault("mni_param", {})
                mni_param["center"] = position["center"]
                params["roi_type"] = "mni_pos"
                runtime_job = copy.deepcopy(runtime_template)
                job_token = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8]
                runtime_job["run_name"] = f"{prefix}_{job_token}_{set_name}_r{index:02d}_{slug}"
                optimizer_options = runtime_job.get("optimizer_options") or {}
                if not isinstance(optimizer_options, dict):
                    raise CampaignError("runtime.optimizer_options 必须是 object 或 null")
                optimizer_options = copy.deepcopy(optimizer_options)
                optimizer_options["seed"] = seed
                runtime_job["optimizer_options"] = optimizer_options
                jobs.append({
                    "job_id": job_id,
                    "campaign": campaign_name,
                    "parameter_set": set_name,
                    "repeat": index,
                    "seed": seed,
                    "axis": position["axis"],
                    "coordinate": position["value"],
                    "center": position["center"],
                    "run_name": runtime_job["run_name"],
                    "runner_config": {
                        "campaign_job": {
                            "campaign": campaign_name,
                            "job_id": job_id,
                            "parameter_set": set_name,
                            "repeat": index,
                            "seed": seed,
                            "axis": position["axis"],
                            "coordinate": position["value"],
                        },
                        "runtime": runtime_job,
                        "params": params,
                    },
                })
    return jobs, state_dir


def select_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """按照命令行筛选条件选择任务。

    Parameters
    ----------
    jobs : list[dict[str, Any]]
        已展开的全部任务。
    args : argparse.Namespace
        命令行筛选参数。

    Returns
    -------
    list[dict[str, Any]]
        保持原顺序的选中任务。
    """
    selected = [
        job for job in jobs
        if (args.only_axis is None or job["axis"] == args.only_axis)
        and (args.only_repeat is None or job["repeat"] == args.only_repeat)
        and (args.only_parameter_set is None or job["parameter_set"] == args.only_parameter_set)
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise CampaignError("--limit 必须是正整数")
        selected = selected[: args.limit]
    if not selected:
        raise CampaignError("筛选后没有任务")
    return selected


def write_json_atomic(path: Path, value: Any) -> None:
    """通过同目录临时文件原子更新 JSON。

    Parameters
    ----------
    path : Path
        目标 JSON 路径。
    value : Any
        可被 JSON 序列化的值。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(temporary, path)


def load_status(path: Path) -> dict[str, Any]:
    """读取断点状态，文件缺失时返回空状态。

    Parameters
    ----------
    path : Path
        状态 JSON 路径。

    Returns
    -------
    dict[str, Any]
        包含任务状态映射的对象。
    """
    if not path.is_file():
        return {"jobs": {}}
    value = load_json(path)
    jobs = value.get("jobs")
    if not isinstance(jobs, dict):
        raise CampaignError(f"状态文件 jobs 字段无效：{path}")
    return value


def plan_payload(raw: dict[str, Any], config_path: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """构建可追溯的批量执行计划快照。

    Parameters
    ----------
    raw : dict[str, Any]
        原始批量配置。
    config_path : Path
        配置文件路径。
    jobs : list[dict[str, Any]]
        本次选择的任务。

    Returns
    -------
    dict[str, Any]
        包含配置摘要和任务列表的计划对象。
    """
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_config": str(config_path),
        "source_sha256": digest,
        "task_count": len(jobs),
        "simulation_runs": len(jobs),
        "estimated_note": "任务顺序执行；总耗时约等于所有单次优化耗时之和",
        "jobs": [{key: value for key, value in job.items() if key != "runner_config"} for job in jobs],
    }


def validate_and_write_job_configs(
    jobs: list[dict[str, Any]],
    state_dir: Path,
) -> None:
    """写入每个任务的单次配置并复用运行器校验。

    Parameters
    ----------
    jobs : list[dict[str, Any]]
        待处理任务。
    state_dir : Path
        批量实验状态目录。
    """
    config_dir = state_dir / "jobs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        path = config_dir / f"{job['job_id']}.json"
        write_json_atomic(path, job["runner_config"])
        inverse_runner.normalize_config(job["runner_config"], path)


def find_new_run_dir(output_root: Path, run_name: str, before: set[Path]) -> Path | None:
    """定位某任务在子进程执行期间新建的结果目录。

    Parameters
    ----------
    output_root : Path
        仿真结果根目录。
    run_name : str
        任务运行名前缀。
    before : set[Path]
        启动任务前已存在的匹配目录。

    Returns
    -------
    Path | None
        最新的新建目录；没有新目录时返回 None。
    """
    after = {path.resolve() for path in output_root.glob(f"{run_name}_*") if path.is_dir()}
    created = after - before
    return max(created, key=lambda path: path.stat().st_mtime) if created else None


def execute_jobs(
    jobs: list[dict[str, Any]],
    state_dir: Path,
    rerun_succeeded: bool,
    stop_on_error: bool,
) -> int:
    """顺序执行任务并持久化可恢复状态。

    Parameters
    ----------
    jobs : list[dict[str, Any]]
        本次选择的任务。
    state_dir : Path
        计划、任务配置和状态文件目录。
    rerun_succeeded : bool
        是否重新运行已成功任务。
    stop_on_error : bool
        是否在首个失败任务后停止。

    Returns
    -------
    int
        全部成功为 0，存在失败或中断为 1。
    """
    status_path = state_dir / "campaign_status.json"
    status = load_status(status_path)
    status.setdefault("campaign", jobs[0]["campaign"])
    status.setdefault("created_at", datetime.now().astimezone().isoformat())
    status_jobs = status.setdefault("jobs", {})
    failures = 0
    for ordinal, job in enumerate(jobs, 1):
        previous = status_jobs.get(job["job_id"], {})
        if previous.get("status") == "SUCCEEDED" and not rerun_succeeded:
            print(f"[{ordinal}/{len(jobs)}] 跳过已成功：{job['job_id']}")
            continue
        config_path = state_dir / "jobs" / f"{job['job_id']}.json"
        output_root = inverse_runner.resolve_path(
            job["runner_config"]["runtime"]["output_root"],
            config_path.parent,
        )
        before = {path.resolve() for path in output_root.glob(f"{job['run_name']}_*") if path.is_dir()} if output_root.is_dir() else set()
        status_jobs[job["job_id"]] = {
            "status": "RUNNING",
            "started_at": datetime.now().astimezone().isoformat(),
            "axis": job["axis"],
            "coordinate": job["coordinate"],
            "repeat": job["repeat"],
            "seed": job["seed"],
            "parameter_set": job["parameter_set"],
            "point": job.get("point"),
            "center": job.get("center"),
        }
        status["updated_at"] = datetime.now().astimezone().isoformat()
        write_json_atomic(status_path, status)
        print(f"\n[{ordinal}/{len(jobs)}] 启动：{job['job_id']}")
        command = [sys.executable, "-m", "neuracle.run_inverse_debug", "--config", str(config_path)]
        try:
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
            return_code = completed.returncode
        except KeyboardInterrupt:
            return_code = 130
        run_dir = find_new_run_dir(output_root, job["run_name"], before)
        succeeded = return_code == 0
        status_jobs[job["job_id"]].update({
            "status": "SUCCEEDED" if succeeded else "INTERRUPTED" if return_code == 130 else "FAILED",
            "completed_at": datetime.now().astimezone().isoformat(),
            "return_code": return_code,
            "run_dir": str(run_dir) if run_dir else None,
        })
        status["updated_at"] = datetime.now().astimezone().isoformat()
        write_json_atomic(status_path, status)
        if not succeeded:
            failures += 1
            print(f"任务失败：{job['job_id']}，退出码 {return_code}")
            if return_code == 130 or stop_on_error:
                break
    print(f"\n批量状态：{status_path}")
    print(f"本次选择 {len(jobs)} 个任务，失败/中断 {failures} 个。")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """运行轴向 TI 逆向批量计划或执行流程。

    Parameters
    ----------
    argv : list[str] | None
        显式参数列表；为 None 时读取 ``sys.argv``。

    Returns
    -------
    int
        成功为 0，配置错误为 2，任务失败为 1。
    """
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    try:
        raw = load_json(config_path)
        jobs, state_dir = build_jobs(raw, config_path)
        selected = select_jobs(jobs, args)
        validate_and_write_job_configs(selected, state_dir)
        write_json_atomic(state_dir / "campaign_plan.json", plan_payload(raw, config_path, selected))
        axes = {axis: sum(job["axis"] == axis for job in selected) for axis in ("O", "X", "Y", "Z")}
        repeats = sorted({job["repeat"] for job in selected})
        parameter_sets = sorted({job["parameter_set"] for job in selected})
        print("批量实验计划校验通过。")
        print(f"状态目录：{state_dir}")
        print(f"任务数：{len(selected)}；轴分布：{axes}")
        print(f"重复轮次：{repeats}；参数组：{parameter_sets}")
        if not args.execute:
            print("当前为计划模式，未调用 SimNIBS。确认后添加 --execute 才会真实运行。")
            return 0
        return execute_jobs(selected, state_dir, args.rerun_succeeded, args.stop_on_error)
    except (CampaignError, inverse_runner.ConfigurationError) as exc:
        print(f"批量实验配置错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
