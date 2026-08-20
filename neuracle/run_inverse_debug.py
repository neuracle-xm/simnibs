"""TI 逆向仿真独立调试运行器。

该脚本不依赖 FastAPI、数据库或 Electron。它读取手工维护的 JSON 配置，
校验产品参数和头模输入，并在非 dry-run 模式下调用本项目的 Neuracle API。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 支持从项目根目录直接执行当前脚本文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from neuracle.logger import setup_logging
from neuracle.parameters.schemas import AnisotropyType, AtlasParam, MNIParam, ROIParam
from neuracle.ti_inverse import run_ti_inverse

logger = logging.getLogger(__name__)
TISSUE_NAMES = (
    "WM",
    "GM",
    "CSF",
    "Bone",
    "Scalp",
    "Eyes",
    "CompactBone",
    "SpongyBone",
    "Blood",
    "Muscle",
)
ANISOTROPY_TYPES = {"scalar", "dir", "vn", "mc"}
ROI_TYPES = {"atlas", "mni_pos"}
CURRENT_LIMIT_MA = 4.0
DEFAULT_ELECTRODE_RADIUS_MM = 6.0
DEFAULT_NON_ROI_THRESHOLD_V_PER_M = 0.1
DEFAULT_MNI_NON_ROI_RADIUS_MM = 25.0


class ConfigurationError(ValueError):
    """表示调试配置或输入文件不满足运行条件。"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Parameters
    ----------
    argv : list[str] | None
        显式参数列表；为 None 时读取 ``sys.argv``。

    Returns
    -------
    argparse.Namespace
        已解析的命令行参数。
    """
    parser = argparse.ArgumentParser(description="独立准备并运行 TI 逆向仿真")
    parser.add_argument("--config", required=True, help="调试配置 JSON 路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验并打印规范化参数，不创建目录、不调用 SimNIBS",
    )
    return parser.parse_args(argv)


def load_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON object。

    Parameters
    ----------
    path : Path
        JSON 文件路径。

    Returns
    -------
    dict[str, Any]
        JSON 顶层对象。

    Raises
    ------
    ConfigurationError
        文件不存在、JSON 无效或顶层不是对象时抛出。
    """
    if not path.is_file():
        raise ConfigurationError(f"配置文件不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"读取配置文件失败：{path}，{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("配置文件顶层必须是 JSON object")
    return data


def require_object(container: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需的 object 字段。

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
        raise ConfigurationError(f"{key} 必须是 JSON object")
    return value


def require_string(container: dict[str, Any], key: str, prefix: str) -> str:
    """读取非空字符串字段。

    Parameters
    ----------
    container : dict[str, Any]
        字段所在对象。
    key : str
        字段名。
    prefix : str
        错误消息中的字段前缀。

    Returns
    -------
    str
        去除首尾空白后的字符串。
    """
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{prefix}.{key} 必须是非空字符串")
    return value.strip()


def require_number(container: dict[str, Any], key: str, prefix: str) -> float:
    """读取有限数值字段。

    Parameters
    ----------
    container : dict[str, Any]
        字段所在对象。
    key : str
        字段名。
    prefix : str
        错误消息中的字段前缀。

    Returns
    -------
    float
        有限浮点数。
    """
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{prefix}.{key} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{prefix}.{key} 必须是有限数字")
    return number


def resolve_path(value: str, base_dir: Path) -> Path:
    """将配置路径解析为绝对路径。

    Parameters
    ----------
    value : str
        绝对路径或相对配置文件的路径。
    base_dir : Path
        相对路径的基准目录。

    Returns
    -------
    Path
        规范化后的绝对路径。
    """
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def validate_current_pair(value: Any, field_name: str) -> list[float]:
    """校验一组逆向仿真正负电流。

    Parameters
    ----------
    value : Any
        待校验的电流配置，单位 mA。
    field_name : str
        字段名。

    Returns
    -------
    list[float]
        规范化后的正负电流。
    """
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError(f"params.{field_name} 必须包含两个电流值")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ConfigurationError(f"params.{field_name} 的元素必须是数字")
    pair = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in pair):
        raise ConfigurationError(f"params.{field_name} 的元素必须是有限数字")
    if pair[0] <= 0 or pair[1] >= 0:
        raise ConfigurationError(f"params.{field_name} 必须按正电流、负电流顺序填写")
    if abs(sum(pair)) > 1e-9:
        raise ConfigurationError(f"params.{field_name} 电流总和必须为 0")
    if pair[0] > CURRENT_LIMIT_MA or pair[1] < -CURRENT_LIMIT_MA:
        raise ConfigurationError(
            f"params.{field_name} 单电极电流必须位于 [-{CURRENT_LIMIT_MA:g}, {CURRENT_LIMIT_MA:g}] mA"
        )
    return pair


def validate_conductivity(value: Any) -> dict[str, float]:
    """校验完整的组织电导率配置。

    Parameters
    ----------
    value : Any
        待校验的电导率对象。

    Returns
    -------
    dict[str, float]
        按 Neuracle 组织顺序规范化的电导率。
    """
    if not isinstance(value, dict):
        raise ConfigurationError("params.conductivity_config 必须是 JSON object")
    missing = [name for name in TISSUE_NAMES if name not in value]
    unknown = [name for name in value if name not in TISSUE_NAMES]
    if missing:
        raise ConfigurationError(f"conductivity_config 缺少组织：{', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"conductivity_config 包含未知组织：{', '.join(unknown)}")
    normalized: dict[str, float] = {}
    for name in TISSUE_NAMES:
        raw_value = value[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ConfigurationError(f"conductivity_config.{name} 必须是数字")
        conductivity = float(raw_value)
        if not math.isfinite(conductivity) or conductivity <= 0:
            raise ConfigurationError(f"conductivity_config.{name} 必须是有限正数")
        normalized[name] = conductivity
    return normalized


def validate_roi(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """校验 Atlas 或 MNI ROI 分支。

    Parameters
    ----------
    params : dict[str, Any]
        参数对象。

    Returns
    -------
    tuple[str, dict[str, Any]]
        ROI 类型和规范化后的 ROI 参数。
    """
    roi_type = require_string(params, "roi_type", "params")
    if roi_type not in ROI_TYPES:
        raise ConfigurationError("params.roi_type 只支持 atlas 或 mni_pos")
    roi_param = require_object(params, "roi_param")
    atlas_param = roi_param.get("atlas_param")
    mni_param = roi_param.get("mni_param")
    if roi_type == "atlas":
        if not isinstance(atlas_param, dict) or mni_param is not None:
            raise ConfigurationError("atlas 模式必须只提供 roi_param.atlas_param")
        return roi_type, {
            "atlas_param": {
                "name": require_string(atlas_param, "name", "params.roi_param.atlas_param"),
                "area": require_string(atlas_param, "area", "params.roi_param.atlas_param"),
            },
            "mni_param": None,
        }
    if not isinstance(mni_param, dict) or atlas_param is not None:
        raise ConfigurationError("mni_pos 模式必须只提供 roi_param.mni_param")
    center = mni_param.get("center")
    if not isinstance(center, list) or len(center) != 3:
        raise ConfigurationError("params.roi_param.mni_param.center 必须是 [x, y, z]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in center):
        raise ConfigurationError("MNI center 的三个元素必须是数字")
    normalized_center = [float(item) for item in center]
    if not all(math.isfinite(item) for item in normalized_center):
        raise ConfigurationError("MNI center 的三个元素必须是有限数字")
    radius = require_number(mni_param, "radius", "params.roi_param.mni_param")
    if radius <= 0:
        raise ConfigurationError("MNI ROI 半径必须大于 0")
    return roi_type, {
        "atlas_param": None,
        "mni_param": {"center": normalized_center, "radius": radius},
    }


def find_head_model_file(head_model_dir: Path, names: tuple[str, ...]) -> Path | None:
    """从候选文件名中返回第一个存在的头模文件。

    Parameters
    ----------
    head_model_dir : Path
        头模目录。
    names : tuple[str, ...]
        候选文件名。

    Returns
    -------
    Path | None
        第一个存在的文件；全部缺失时返回 None。
    """
    for name in names:
        candidate = head_model_dir / name
        if candidate.is_file():
            return candidate
    return None


def normalize_config(
    raw_config: dict[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """校验并规范化完整调试配置。

    Parameters
    ----------
    raw_config : dict[str, Any]
        原始配置。
    config_path : Path
        配置文件路径。
    Returns
    -------
    tuple[dict[str, Any], list[str]]
        规范化配置和非阻塞警告列表。
    """
    runtime = require_object(raw_config, "runtime")
    params = require_object(raw_config, "params")
    base_dir = config_path.parent
    neuracle_root = PROJECT_ROOT.resolve()
    head_model_dir = resolve_path(
        require_string(runtime, "head_model_dir", "runtime"), base_dir
    )
    if not head_model_dir.is_dir():
        raise ConfigurationError(f"头模目录不存在：{head_model_dir}")
    head_model_id = require_string(runtime, "head_model_id", "runtime")
    mesh_path = head_model_dir / f"{head_model_id}.msh"
    if not mesh_path.is_file():
        raise ConfigurationError(f"头模 mesh 不存在：{mesh_path}")
    t1_path = find_head_model_file(head_model_dir, ("T1.nii.gz", "T1.nii"))
    if t1_path is None:
        raise ConfigurationError(f"头模缺少 T1.nii.gz/T1.nii：{head_model_dir}")
    output_root = resolve_path(
        require_string(runtime, "output_root", "runtime"), base_dir
    )
    run_name = require_string(runtime, "run_name", "runtime")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ConfigurationError("runtime.run_name 只能包含字母、数字、点、下划线和短横线")
    n_workers_raw = runtime.get("n_workers", 8)
    if isinstance(n_workers_raw, bool) or not isinstance(n_workers_raw, int) or n_workers_raw <= 0:
        raise ConfigurationError("runtime.n_workers 必须是正整数")
    optimizer_options = runtime.get("optimizer_options")
    if optimizer_options is not None:
        if not isinstance(optimizer_options, dict):
            raise ConfigurationError("runtime.optimizer_options 必须是对象或 null")
        allowed_optimizer_options = {"maxiter", "popsize", "seed"}
        unknown_options = set(optimizer_options) - allowed_optimizer_options
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise ConfigurationError(f"不支持的 optimizer_options：{names}")
        for name, minimum in (("maxiter", 0), ("popsize", 1)):
            value = optimizer_options.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
            ):
                raise ConfigurationError(
                    f"runtime.optimizer_options.{name} 必须是大于等于 {minimum} 的整数"
                )
        seed = optimizer_options.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ConfigurationError("runtime.optimizer_options.seed 必须是整数或 null")
    montage = require_string(params, "montage", "params")
    montage_path = Path(montage)
    if montage_path.is_absolute():
        resolved_montage = montage_path
    else:
        resolved_montage = head_model_dir / "eeg_positions" / montage
        if resolved_montage.suffix.lower() != ".csv":
            resolved_montage = resolved_montage.with_suffix(".csv")
    if not resolved_montage.is_file():
        raise ConfigurationError(f"Montage 文件不存在：{resolved_montage}")
    current_a = validate_current_pair(params.get("current_A"), "current_A")
    current_b = validate_current_pair(params.get("current_B"), "current_B")
    roi_type, roi_param = validate_roi(params)
    target_threshold = require_number(params, "target_threshold", "params")
    if target_threshold <= 0:
        raise ConfigurationError("params.target_threshold 必须大于 0")
    anisotropy_type = require_string(params, "anisotropy_type", "params")
    if anisotropy_type not in ANISOTROPY_TYPES:
        raise ConfigurationError("params.anisotropy_type 只支持 scalar、dir、vn、mc")
    conductivity = validate_conductivity(params.get("conductivity_config"))
    electrode_radius = params.get("electrode_radius", DEFAULT_ELECTRODE_RADIUS_MM)
    if isinstance(electrode_radius, bool) or not isinstance(electrode_radius, (int, float)):
        raise ConfigurationError("params.electrode_radius 必须是数字")
    electrode_radius = float(electrode_radius)
    if not math.isfinite(electrode_radius) or electrode_radius <= 0:
        raise ConfigurationError("params.electrode_radius 必须是有限正数")
    dti_path = find_head_model_file(
        head_model_dir,
        ("DTI_coregT1_tensor.nii.gz", "DTI_coregT1_tensor.nii"),
    )
    warnings: list[str] = []
    if anisotropy_type != "scalar" and dti_path is None:
        raise ConfigurationError(f"{anisotropy_type} 各向异性需要 DTI 张量文件")
    if roi_type == "atlas":
        warnings.append("Atlas ROI 的名称和标准化 mask 将由 Neuracle 运行环境进一步校验")
    normalized = {
        "runtime": {
            "neuracle_root": str(neuracle_root),
            "head_model_dir": str(head_model_dir),
            "head_model_id": head_model_id,
            "output_root": str(output_root),
            "run_name": run_name,
            "n_workers": n_workers_raw,
            "optimizer_options": optimizer_options,
        },
        "params": {
            "montage": montage,
            "current_A": current_a,
            "current_B": current_b,
            "roi_type": roi_type,
            "roi_param": roi_param,
            "target_threshold": target_threshold,
            "conductivity_config": conductivity,
            "anisotropy_type": anisotropy_type,
            "electrode_radius": electrode_radius,
        },
        "derived": {
            "mesh_path": str(mesh_path),
            "t1_path": str(t1_path),
            "dti_path": str(dti_path) if dti_path else None,
            "montage_path": str(resolved_montage),
            "goal": "focality",
            "e_postproc": "max_TI",
            "optimizer": "differential_evolution",
            "polish": False,
            "focality_threshold": [
                DEFAULT_NON_ROI_THRESHOLD_V_PER_M,
                target_threshold,
            ],
            "non_roi": (
                {
                    "method": "volume_sphere_difference",
                    "center": roi_param["mni_param"]["center"],
                    "radius": DEFAULT_MNI_NON_ROI_RADIUS_MM,
                    "space": "mni",
                }
                if roi_type == "mni_pos"
                else {
                    "method": "volume_mask_difference",
                    "atlas": roi_param["atlas_param"]["name"],
                    "area": roi_param["atlas_param"]["area"],
                    "space": "mni",
                }
            ),
            "current_A_ampere": [item / 1000 for item in current_a],
            "current_B_ampere": [item / 1000 for item in current_b],
        },
    }
    return normalized, warnings


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。

    Parameters
    ----------
    path : Path
        文件路径。

    Returns
    -------
    str
        十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    """以排他模式写入格式化 JSON。

    Parameters
    ----------
    path : Path
        输出路径。
    data : dict[str, Any]
        待写入对象。
    """
    with path.open("x", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def create_run_dir(output_root: Path, run_name: str) -> Path:
    """创建不会覆盖历史结果的时间戳运行目录。

    Parameters
    ----------
    output_root : Path
        调试结果根目录。
    run_name : str
        运行名称。

    Returns
    -------
    Path
        新建运行目录。
    """
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir()
    return run_dir


def build_manifest(
    config_path: Path,
    normalized: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """构建可复现实验清单。

    Parameters
    ----------
    config_path : Path
        原始配置路径。
    normalized : dict[str, Any]
        规范化配置。
    run_dir : Path
        本次运行目录。

    Returns
    -------
    dict[str, Any]
        运行清单。
    """
    neuracle_root = Path(normalized["runtime"]["neuracle_root"])
    inverse_entry = neuracle_root / "neuracle" / "ti_inverse.py"
    optimization_source = (
        neuracle_root
        / "neuracle"
        / "ti_optimization"
        / "electrodes_roi.py"
    )
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "orchestrator_python": sys.executable,
        "source_fingerprints": {
            "ti_inverse.py": sha256_file(inverse_entry),
            "electrodes_roi.py": sha256_file(optimization_source),
        },
    }


def build_result_inventory(run_dir: Path, head_model_id: str, return_code: int) -> dict[str, Any]:
    """检查逆向仿真的关键输出文件。

    Parameters
    ----------
    run_dir : Path
        本次运行目录。
    head_model_id : str
        头模 ID。
    return_code : int
        Neuracle 子进程退出码。

    Returns
    -------
    dict[str, Any]
        关键产物存在性和结果文件总数。
    """
    expected = {
        "electrode_mapping": run_dir / "electrode_mapping.json",
        "max_ti_nifti": run_dir / f"{head_model_id}_optimization_max_TI.nii.gz",
        "mapped_mesh": (
            run_dir
            / "mapped_electrodes_simulation"
            / f"{head_model_id}_tes_mapped_opt_head_mesh.msh"
        ),
    }
    files = [path for path in run_dir.rglob("*") if path.is_file()]
    return {
        "completed_at": datetime.now().astimezone().isoformat(),
        "return_code": return_code,
        "success": return_code == 0,
        "file_count": len(files),
        "expected_outputs": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in expected.items()
        },
    }


def build_roi_param(params: dict[str, Any]) -> ROIParam:
    """把规范化后的 ROI 配置转换为项目数据结构。

    Parameters
    ----------
    params : dict[str, Any]
        已通过校验的参数对象。

    Returns
    -------
    ROIParam
        可传给 :func:`neuracle.ti_inverse.run_ti_inverse` 的 ROI 参数。
    """
    roi_param = params["roi_param"]
    if params["roi_type"] == "atlas":
        atlas_param = roi_param["atlas_param"]
        return ROIParam(atlas_param=AtlasParam(name=atlas_param["name"], area=atlas_param["area"]))
    mni_param = roi_param["mni_param"]
    return ROIParam(mni_param=MNIParam(center=mni_param["center"], radius=mni_param["radius"]))


def execute_inverse(normalized: dict[str, Any], run_dir: Path) -> int:
    """使用本项目 API 执行一次 TI 逆向优化。

    Parameters
    ----------
    normalized : dict[str, Any]
        已规范化的运行配置。
    run_dir : Path
        本次优化的输出目录。

    Returns
    -------
    int
        成功为 0，运行失败为 1，用户中断为 130。
    """
    runtime = normalized["runtime"]
    params = normalized["params"]
    optimizer_options = runtime["optimizer_options"]
    if optimizer_options is not None:
        optimizer_options = {key: value for key, value in optimizer_options.items() if value is not None}
    try:
        run_ti_inverse(
            head_model_id=runtime["head_model_id"],
            head_model_dir=runtime["head_model_dir"],
            output_dir=str(run_dir),
            montage=params["montage"],
            current_A=params["current_A"],
            current_B=params["current_B"],
            roi_type=params["roi_type"],
            roi_param=build_roi_param(params),
            target_threshold=params["target_threshold"],
            conductivity_config=params["conductivity_config"],
            anisotropy=AnisotropyType(params["anisotropy_type"]),
            n_workers=runtime["n_workers"],
            electrode_radius=params["electrode_radius"],
            optimizer_options=optimizer_options,
        )
    except KeyboardInterrupt:
        logger.warning("用户中断 TI 逆向优化")
        return 130
    except Exception:
        logger.exception("TI 逆向优化执行失败")
        return 1
    return 0


def run_debug(normalized: dict[str, Any], config_path: Path) -> int:
    """准备运行目录并启动 Neuracle 调试适配器。

    Parameters
    ----------
    normalized : dict[str, Any]
        规范化配置。
    config_path : Path
        原始配置路径。

    Returns
    -------
    int
        适配器退出码。
    """
    runtime = normalized["runtime"]
    run_dir = create_run_dir(Path(runtime["output_root"]), runtime["run_name"])
    execution = {
        **normalized,
        "runtime": {**runtime, "output_dir": str(run_dir)},
    }
    execution_path = run_dir / "execution.json"
    manifest_path = run_dir / "manifest.json"
    write_json_exclusive(execution_path, execution)
    # 保留与产品 simulations 目录一致的参数快照，便于批量实验报告直接读取。
    write_json_exclusive(run_dir / "params.json", normalized["params"])
    write_json_exclusive(manifest_path, build_manifest(config_path, normalized, run_dir))
    setup_logging(str(run_dir / "logs"))
    print(f"运行目录：{run_dir}")
    return_code = execute_inverse(normalized, run_dir)
    result = build_result_inventory(run_dir, runtime["head_model_id"], return_code)
    write_json_exclusive(run_dir / "result_inventory.json", result)
    print(f"Neuracle 退出码：{return_code}")
    print(f"产物清单：{run_dir / 'result_inventory.json'}")
    return return_code


def main(argv: list[str] | None = None) -> int:
    """运行独立逆向调试流程。

    Parameters
    ----------
    argv : list[str] | None
        显式参数列表；为 None 时读取 ``sys.argv``。

    Returns
    -------
    int
        成功为 0，配置错误为 2，Neuracle 失败时透传其退出码。
    """
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    try:
        raw_config = load_json_object(config_path)
        normalized, warnings = normalize_config(
            raw_config,
            config_path,
        )
        print("配置校验通过。")
        for warning in warnings:
            print(f"警告：{warning}")
        if args.dry_run:
            print(json.dumps(normalized, ensure_ascii=False, indent=2))
            return 0
        return run_debug(normalized, config_path)
    except ConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("用户中断运行。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
