"""
TI 逆向优化入口

提供 TI 逆向优化流程的入口函数 run_ti_inverse()。

逆向仿真是"已知目标电场分布，反求电极电流配置"的优化过程。
需要迭代优化以找到最优电流配置。

用法:
    from neuracle.ti_inverse import run_ti_inverse
    from neuracle.parameters.schemas import ROIParam, MNIParam

    run_ti_inverse(
        head_model_id="ernie",
        head_model_dir="/path/to/data_root/head_models/m2m_ernie",
        output_dir="/path/to/data_root/simulations/ti_inverse_demo",
        montage="EEG10-10_Neuroelectrics",
        current_A=[1.0, -1.0],
        current_B=[1.0, -1.0],
        roi_type="mni_pos",
        roi_param=ROIParam(mni_param=MNIParam(center=[-50, -30, 40], radius=10)),
        target_threshold=0.2,
        conductivity_config={"WM": 0.126, "GM": 0.275, ...},
        anisotropy=AnisotropyType.SCALAR,
    )
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

# 支持直接执行当前脚本文件时使用 `from neuracle...` 绝对导入
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuracle.atlas.standardized import get_standardized_roi_path
from neuracle.logger import setup_logging
from neuracle.parameters.schemas import (
    AnisotropyType,
    AtlasParam,
    MNIParam,
    ROIParam,
)
from neuracle.parameters.validator import ValidationError, validate_inverse_params
from neuracle.ti_optimization import (
    get_electrode_mapping,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils import (
    NON_ROI_THRESHOLD,
    cond_dict_to_list,
    find_montage_file,
)
from neuracle.utils.constants import (
    EXIT_INVALID_ARGS,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_VALUE_ERROR,
    N_WORKERS,
)
from neuracle.utils.error_message import clear_error_message, write_error_message
from neuracle.utils.find_nifty import find_optional_nifti_file
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger("neuracle.ti_inverse")


def run_ti_inverse(
    head_model_id: str,
    head_model_dir: str,
    output_dir: str,
    montage: str,
    current_A: list[float],
    current_B: list[float],
    roi_type: Literal["atlas", "mni_pos"],
    roi_param: ROIParam,
    target_threshold: float,
    conductivity_config: dict[str, float],
    anisotropy: AnisotropyType,
    n_workers: int = 8,
) -> None:
    """
    运行 TI 逆向优化，生成失败则抛出异常。

    流程步骤：
    步骤 0：初始化与检查，解析 ROI 参数
    步骤 1：优化器初始化
    步骤 2：目标设置
    步骤 3：电极与 ROI 配置
    步骤 4：执行优化
    步骤 5：电极映射获取
    步骤 6：NIfTI 导出

    Parameters
    ----------
    head_model_id : str
        头模 ID
    head_model_dir : str
        头模目录路径
    output_dir : str
        仿真输出目录路径
    montage : str
        电极导联名称
    current_A : list[float]
        电极组 A 初始电流值（mA）
    current_B : list[float]
        电极组 B 初始电流值（mA）
    roi_type : Literal["atlas", "mni_pos"]
        ROI 类型
    roi_param : ROIParam
        ROI 参数配置
    target_threshold : float
        目标电场强度阈值
    conductivity_config : dict[str, float]
        组织电导率配置
    anisotropy : AnisotropyType
        各向异性类型
    n_workers : int
        并行工作进程数
    """
    roi_info = (
        f"atlas={roi_param.atlas_param}"
        if roi_param.atlas_param
        else f"mni={roi_param.mni_param}"
        if roi_param.mni_param
        else "None"
    )
    logger.info(
        "开始 TI 逆向优化: head_model_id=%s, head_model_dir=%s, output_dir=%s, montage=%s, anisotropy=%s, "
        "current_A=%s, current_B=%s, roi_type=%s, roi=%s, "
        "target_threshold=%s, conductivity=%s",
        head_model_id,
        head_model_dir,
        output_dir,
        montage,
        anisotropy,
        current_A,
        current_B,
        roi_type,
        roi_info,
        target_threshold,
        conductivity_config,
    )

    # ===== 步骤 0：初始化与检查 =====
    subject_dir = Path(head_model_dir)
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"subject 目录不存在: {subject_dir}。请先执行头模生成任务。"
        )
    output_dir_path = Path(output_dir)
    t1_file_path = find_optional_nifti_file(subject_dir, ("T1.nii.gz", "T1.nii"))
    if t1_file_path is None:
        raise FileNotFoundError(f"未找到 T1 文件: {subject_dir}")
    dti_file_path = find_optional_nifti_file(
        subject_dir,
        ("DTI_coregT1_tensor.nii.gz", "DTI_coregT1_tensor.nii"),
    )
    net_electrode_file = find_montage_file(str(subject_dir), montage)
    mesh_path = subject_dir / f"{head_model_id}.msh"
    if not mesh_path.exists():
        raise FileNotFoundError(f"头模 mesh 文件不存在: {mesh_path}")

    roi_center = None
    roi_radius = None
    roi_center_space = "subject"
    roi_mask_path = None
    roi_mask_space = None
    focality_threshold = target_threshold

    if roi_type == "atlas" and roi_param.atlas_param:
        roi_mask_path = str(
            get_standardized_roi_path(
                roi_param.atlas_param.name,
                roi_param.atlas_param.area,
            )
        )
        if not os.path.exists(roi_mask_path):
            raise FileNotFoundError(
                f"标准化 ROI 不存在: {roi_mask_path}。请先运行 atlas 标准化和 ROI 生成脚本。"
            )
        roi_mask_space = "mni"
    elif roi_type == "mni_pos" and roi_param.mni_param:
        roi_center = roi_param.mni_param.center
        roi_radius = roi_param.mni_param.radius
        roi_center_space = "mni"

    # ===== 步骤 1：优化器初始化 =====
    opt = init_optimization(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir_path),
        anisotropy_type=anisotropy,
        cond=cond_dict_to_list(conductivity_config),
        fname_tensor=dti_file_path,
    )
    logger.info("优化器初始化完成")

    # ===== 步骤 2：目标设置 =====
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    logger.info("优化目标设置完成")

    # ===== 步骤 3：电极与 ROI 配置 =====
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        mesh_file_path=mesh_path,
        electrode_current1=[c / 1000 for c in current_A],
        electrode_current2=[c / 1000 for c in current_B],
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
    )
    logger.info("电极与 ROI 配置完成")

    # ===== 步骤 4：执行优化 =====
    run_optimization(opt=opt, n_workers=n_workers)
    logger.info("优化执行完成")

    # ===== 步骤 5：电极映射获取 =====
    electrode_A, electrode_B = get_electrode_mapping(output_dir=str(output_dir_path))
    logger.info("电极映射获取完成: A=%s, B=%s", electrode_A, electrode_B)

    # ===== 步骤 6：NIfTI 导出 =====
    msh_name = f"{head_model_id}_tes_mapped_opt_head_mesh.msh"
    msh_path = str(output_dir_path / "mapped_electrodes_simulation" / msh_name)
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=str(output_dir_path),
        reference=t1_file_path,
        field_name="max_TI",
        prefix=f"{head_model_id}_optimization",
    )
    logger.info("NIfTI 导出完成: %s", ti_nifti_path)

    logger.info("TI 逆向优化完成")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建 TI 逆向优化命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        命令行解析器
    """
    parser = argparse.ArgumentParser(
        prog="python neuracle/ti_inverse.py",
        description="运行 TI 逆向优化流程",
    )
    parser.add_argument("data_root", help="数据根目录")
    parser.add_argument("task_id", help="仿真任务 ID")
    parser.add_argument("head_model_id", help="头模 ID，不带 m2m_ 前缀")
    return parser


def resolve_ti_inverse_dirs(
    data_root: str,
    task_id: str,
    head_model_id: str,
) -> tuple[str, str]:
    """
    根据目录约定解析 TI 逆向优化目录。

    Parameters
    ----------
    data_root : str
        数据根目录
    task_id : str
        仿真任务 ID
    head_model_id : str
        头模 ID

    Returns
    -------
    tuple[str, str]
        仿真目录、头模目录
    """
    root_dir = Path(data_root)
    simulation_dir = root_dir / "simulations" / f"ti_inverse_{task_id}"
    head_model_dir = root_dir / "head_models" / f"m2m_{head_model_id}"
    if not simulation_dir.is_dir():
        raise FileNotFoundError(f"仿真目录不存在: {simulation_dir}")
    return str(simulation_dir), str(head_model_dir)


def load_ti_inverse_params(
    simulation_dir: str,
    head_model_dir: str,
) -> tuple[dict, str]:
    """
    根据已解析目录读取 TI 逆向优化输入参数。

    Parameters
    ----------
    simulation_dir : str
        仿真目录
    head_model_dir : str
        头模目录

    Returns
    -------
    tuple[dict, str]
        参数字典、仿真目录
    """
    simulation_dir_path = Path(simulation_dir)
    head_model_dir_path = Path(head_model_dir)
    params_path = simulation_dir_path / "params.json"
    if not head_model_dir_path.is_dir():
        raise FileNotFoundError(f"头模目录不存在: {head_model_dir_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"参数文件不存在: {params_path}")
    with params_path.open("r", encoding="utf-8") as file_obj:
        params_data = json.load(file_obj)
    if not isinstance(params_data, dict):
        raise ValueError("params.json 顶层必须是 JSON 对象")
    params_dict = {
        "head_model_dir": str(head_model_dir_path),
        "montage": params_data.get("montage"),
        "current_A": params_data.get("current_A"),
        "current_B": params_data.get("current_B"),
        "roi_type": params_data.get("roi_type"),
        "roi_param": params_data.get("roi_param"),
        "target_threshold": params_data.get("target_threshold"),
        "conductivity_config": params_data.get("conductivity_config"),
        "anisotropy": params_data.get("anisotropy_type"),
    }
    return params_dict, str(simulation_dir_path)


def build_inverse_params(
    params_dict: dict,
) -> tuple[
    str,
    list[float],
    list[float],
    Literal["atlas", "mni_pos"],
    ROIParam,
    float,
    dict[str, float],
    AnisotropyType,
]:
    """
    将已验证的参数字典转换为运行时参数。

    Parameters
    ----------
    params_dict : dict
        已通过校验的参数字典

    Returns
    -------
    tuple[str, list[float], list[float], Literal["atlas", "mni_pos"], ROIParam, float, dict[str, float], AnisotropyType]
        montage、电流组 A、电流组 B、ROI 类型、ROI 参数、目标阈值、电导率、各向异性
    """
    roi_param_data = params_dict["roi_param"]
    roi_param = ROIParam()
    if roi_param_data.get("mni_param"):
        roi_param.mni_param = MNIParam(
            center=roi_param_data["mni_param"]["center"],
            radius=roi_param_data["mni_param"]["radius"],
        )
    if roi_param_data.get("atlas_param"):
        roi_param.atlas_param = AtlasParam(
            name=roi_param_data["atlas_param"]["name"],
            area=roi_param_data["atlas_param"]["area"],
        )
    return (
        params_dict["montage"],
        params_dict["current_A"],
        params_dict["current_B"],
        params_dict["roi_type"],
        roi_param,
        params_dict["target_threshold"],
        params_dict["conductivity_config"],
        AnisotropyType(params_dict["anisotropy"]),
    )


def main(argv: list[str] | None = None) -> int:
    """
    TI 逆向优化命令行入口。

    Parameters
    ----------
    argv : list[str] | None
        命令行参数列表，默认读取 sys.argv

    Returns
    -------
    int
        进程退出码
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    task_dir = Path(args.data_root) / "simulations" / f"ti_inverse_{args.task_id}"
    log_dir = task_dir / "logs" if task_dir.is_dir() else None
    try:
        simulation_dir, head_model_dir = resolve_ti_inverse_dirs(
            args.data_root,
            args.task_id,
            args.head_model_id,
        )
        log_dir = Path(simulation_dir) / "logs"
        setup_logging(str(log_dir))
        params_dict, simulation_dir = load_ti_inverse_params(
            simulation_dir,
            head_model_dir,
        )
        validate_inverse_params(params_dict)
        (
            montage,
            current_a,
            current_b,
            roi_type,
            roi_param,
            target_threshold,
            conductivity_config,
            anisotropy,
        ) = build_inverse_params(params_dict)
        run_ti_inverse(
            head_model_id=args.head_model_id,
            head_model_dir=params_dict["head_model_dir"],
            output_dir=simulation_dir,
            montage=montage,
            current_A=current_a,
            current_B=current_b,
            roi_type=roi_type,
            roi_param=roi_param,
            target_threshold=target_threshold,
            conductivity_config=conductivity_config,
            anisotropy=anisotropy,
            n_workers=N_WORKERS,
        )
    except ValidationError as exc:
        logger.exception("TI 逆向优化参数校验失败: %s", exc)
        if log_dir is not None:
            write_error_message(log_dir, EXIT_INVALID_ARGS, str(exc))
        return EXIT_INVALID_ARGS
    except ValueError as exc:
        logger.exception("TI 逆向优化数值处理失败: %s", exc)
        if log_dir is not None:
            write_error_message(log_dir, EXIT_VALUE_ERROR, str(exc))
        return EXIT_VALUE_ERROR
    except Exception as exc:
        logger.exception("TI 逆向优化执行失败: %s", exc)
        if log_dir is not None:
            write_error_message(log_dir, EXIT_RUNTIME_ERROR, str(exc))
        return EXIT_RUNTIME_ERROR
    if log_dir is not None:
        clear_error_message(log_dir)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
