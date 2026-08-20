"""批量执行任意 MNI 三维坐标的 TI 逆向实验。

默认仅生成、校验执行计划；只有显式传入 ``--execute`` 才会顺序启动
SimNIBS。底层复用轴向扫参脚本的状态管理与 ``run_inverse_debug.py`` 调用链。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from neuracle import run_axis_sweep_batch as batch
from neuracle import run_inverse_debug as inverse_runner

DEFAULT_CONFIG = SCRIPT_DIR / "coordinate_campaign.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析任意坐标批量实验命令行参数。

    Parameters
    ----------
    argv : list[str] | None
        显式参数列表；为 None 时读取 ``sys.argv``。

    Returns
    -------
    argparse.Namespace
        已解析的命令行参数。
    """
    parser = argparse.ArgumentParser(description="规划或顺序执行任意 MNI 三维坐标 TI 逆向实验")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="三维坐标批量实验 JSON 配置")
    parser.add_argument("--execute", action="store_true", help="真实启动 SimNIBS；不传时仅校验计划")
    parser.add_argument("--only-point", help="只运行 coordinates 中指定 name 的位点")
    parser.add_argument("--only-repeat", type=int, help="只运行指定重复序号")
    parser.add_argument("--only-parameter-set", help="只运行指定参数组")
    parser.add_argument("--limit", type=int, help="最多选择前 N 个任务")
    parser.add_argument("--rerun-succeeded", action="store_true", help="重新运行状态为成功的任务")
    parser.add_argument("--stop-on-error", action="store_true", help="任一任务失败后停止")
    return parser.parse_args(argv)


def number_slug(value: float) -> str:
    """把坐标分量转换为稳定的任务名片段。

    Parameters
    ----------
    value : float
        一个 MNI 坐标分量。

    Returns
    -------
    str
        带符号且不包含小数点的名称片段。
    """
    sign = "p" if value >= 0 else "m"
    magnitude = f"{abs(value):g}".replace(".", "d")
    return f"{sign}{magnitude.zfill(3)}"


def parse_points(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """校验并规范化任意三维 MNI 坐标。

    Parameters
    ----------
    campaign : dict[str, Any]
        包含 ``coordinates`` 的实验配置。

    Returns
    -------
    list[dict[str, Any]]
        唯一名称和有限三维坐标组成的位点列表。
    """
    raw_points = batch.require_list(campaign, "coordinates")
    points: list[dict[str, Any]] = []
    names: set[str] = set()
    centers: set[tuple[float, float, float]] = set()
    for ordinal, raw_point in enumerate(raw_points, 1):
        if isinstance(raw_point, list):
            raw_center = raw_point
            raw_name = None
        elif isinstance(raw_point, dict):
            raw_center = raw_point.get("center")
            raw_name = raw_point.get("name")
        else:
            raise batch.CampaignError(f"coordinates[{ordinal}] 必须是 [x,y,z] 或 object")
        if not isinstance(raw_center, list) or len(raw_center) != 3:
            raise batch.CampaignError(f"coordinates[{ordinal}].center 必须恰好包含 x、y、z 三个数字")
        center: list[float] = []
        for component in raw_center:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise batch.CampaignError(f"coordinates[{ordinal}].center 只能包含数字")
            value = float(component)
            if not math.isfinite(value):
                raise batch.CampaignError(f"coordinates[{ordinal}].center 不能包含无穷大或 NaN")
            center.append(value)
        center_key = tuple(center)
        if center_key in centers:
            raise batch.CampaignError(f"coordinates 包含重复坐标：{center}")
        centers.add(center_key)
        generated_name = "xyz_" + "_".join(number_slug(value) for value in center)
        name = batch.safe_name(raw_name, f"coordinates[{ordinal}].name") if raw_name is not None else generated_name
        if name in names:
            raise batch.CampaignError(f"coordinates 包含重复名称：{name}")
        names.add(name)
        points.append({"name": name, "center": center})
    return points


def build_jobs(raw: dict[str, Any], config_path: Path) -> tuple[list[dict[str, Any]], Path]:
    """展开任意坐标实验的参数组、重复轮次和位点。

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
    campaign = batch.require_dict(raw, "campaign")
    runtime = batch.require_dict(raw, "runtime")
    base_params = batch.require_dict(raw, "base_params")
    campaign_name = batch.safe_name(campaign.get("name"), "campaign.name")
    prefix = batch.safe_name(runtime.get("run_name_prefix"), "runtime.run_name_prefix")
    repetitions = batch.require_list(campaign, "repetitions")
    parameter_sets = batch.require_list(campaign, "parameter_sets")
    points = parse_points(campaign)

    state_value = campaign.get("state_dir", f"campaigns/{campaign_name}")
    if not isinstance(state_value, str) or not state_value:
        raise batch.CampaignError("campaign.state_dir 必须是非空字符串")
    state_dir = inverse_runner.resolve_path(state_value, config_path.parent)

    runtime_template = {key: copy.deepcopy(value) for key, value in runtime.items() if key != "run_name_prefix"}
    jobs: list[dict[str, Any]] = []
    ids: set[str] = set()
    for parameter_set in parameter_sets:
        if not isinstance(parameter_set, dict):
            raise batch.CampaignError("campaign.parameter_sets 的元素必须是 object")
        set_name = batch.safe_name(parameter_set.get("name"), "parameter_sets.name")
        overrides = parameter_set.get("overrides", {})
        if not isinstance(overrides, dict):
            raise batch.CampaignError(f"parameter_sets.{set_name}.overrides 必须是 object")
        set_params = batch.deep_merge(base_params, overrides)
        for repetition in repetitions:
            if not isinstance(repetition, dict):
                raise batch.CampaignError("campaign.repetitions 的元素必须是 object")
            index, seed = repetition.get("index"), repetition.get("seed")
            if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                raise batch.CampaignError("repetitions.index 必须是正整数")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise batch.CampaignError("repetitions.seed 必须是整数或 null")
            for point in points:
                job_id = f"{set_name}__r{index:02d}__{point['name']}"
                if job_id in ids:
                    raise batch.CampaignError(f"生成了重复任务 ID：{job_id}")
                ids.add(job_id)
                params = copy.deepcopy(set_params)
                params.setdefault("roi_param", {}).setdefault("mni_param", {})["center"] = point["center"]
                params["roi_type"] = "mni_pos"
                runtime_job = copy.deepcopy(runtime_template)
                token = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8]
                runtime_job["run_name"] = f"{prefix}_{token}_{set_name}_r{index:02d}_{point['name']}"
                optimizer_options = runtime_job.get("optimizer_options") or {}
                if not isinstance(optimizer_options, dict):
                    raise batch.CampaignError("runtime.optimizer_options 必须是 object 或 null")
                runtime_job["optimizer_options"] = {**copy.deepcopy(optimizer_options), "seed": seed}
                jobs.append({
                    "job_id": job_id,
                    "campaign": campaign_name,
                    "parameter_set": set_name,
                    "repeat": index,
                    "seed": seed,
                    "axis": "XYZ",
                    "coordinate": point["name"],
                    "point": point["name"],
                    "center": point["center"],
                    "run_name": runtime_job["run_name"],
                    "runner_config": {
                        "campaign_job": {
                            "campaign": campaign_name,
                            "job_id": job_id,
                            "parameter_set": set_name,
                            "repeat": index,
                            "seed": seed,
                            "point": point["name"],
                            "center": point["center"],
                        },
                        "runtime": runtime_job,
                        "params": params,
                    },
                })
    return jobs, state_dir


def select_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """按照位点、轮次和参数组筛选任务。

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
        if (args.only_point is None or job["point"] == args.only_point)
        and (args.only_repeat is None or job["repeat"] == args.only_repeat)
        and (args.only_parameter_set is None or job["parameter_set"] == args.only_parameter_set)
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise batch.CampaignError("--limit 必须是正整数")
        selected = selected[:args.limit]
    if not selected:
        raise batch.CampaignError("筛选后没有任务")
    return selected


def main(argv: list[str] | None = None) -> int:
    """运行任意坐标 TI 逆向批量计划或执行流程。

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
        raw = batch.load_json(config_path)
        jobs, state_dir = build_jobs(raw, config_path)
        selected = select_jobs(jobs, args)
        batch.validate_and_write_job_configs(selected, state_dir)
        batch.write_json_atomic(state_dir / "campaign_plan.json", batch.plan_payload(raw, config_path, selected))
        points = sorted({job["point"] for job in selected})
        repeats = sorted({job["repeat"] for job in selected})
        parameter_sets = sorted({job["parameter_set"] for job in selected})
        print("任意三维坐标批量实验计划校验通过。")
        print(f"状态目录：{state_dir}")
        print(f"任务数：{len(selected)}；位点：{points}")
        print(f"重复轮次：{repeats}；参数组：{parameter_sets}")
        if not args.execute:
            print("当前为计划模式，未调用 SimNIBS。确认后添加 --execute 才会真实运行。")
            return 0
        return batch.execute_jobs(selected, state_dir, args.rerun_succeeded, args.stop_on_error)
    except (batch.CampaignError, inverse_runner.ConfigurationError) as exc:
        print(f"批量实验配置错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
