"""
TI 正向仿真入口

提供 TI 正向仿真流程的入口函数 run_ti_forward()。

正向仿真用于计算给定电极配置下的电场分布。
"已知电极电流，求解电场"，不涉及优化。

用法:
    from neuracle.ti_forward import run_ti_forward
    from neuracle.parameters.schemas import ElectrodeWithCurrent

    run_ti_forward(
        head_model_dir="/path/to/data_root/head_models/m2m_ernie",
        output_dir="/path/to/data_root/simulations/ti_forward_demo",
        montage="EEG10-10_Neuroelectrics",
        electrode_A=[ElectrodeWithCurrent(name="F5", current_mA=1.0),
                      ElectrodeWithCurrent(name="P5", current_mA=-1.0)],
        electrode_B=[ElectrodeWithCurrent(name="F6", current_mA=1.0),
                      ElectrodeWithCurrent(name="P6", current_mA=-1.0)],
        conductivity_config={"WM": 0.126, "GM": 0.275, ...},
        anisotropy=AnisotropyType.SCALAR,
    )
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 支持直接执行当前脚本文件时使用 `from neuracle...` 绝对导入
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuracle.logger import setup_logging
from neuracle.parameters.schemas import AnisotropyType, ElectrodeWithCurrent
from neuracle.parameters.validator import ValidationError, validate_forward_params
from neuracle.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)
from neuracle.utils import (
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
from neuracle.utils.find_nifty import find_optional_nifti_file
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger("neuracle.ti_forward")


def run_ti_forward(
    head_model_id: str,
    head_model_dir: str,
    output_dir: str,
    montage: str,
    electrode_A: list[ElectrodeWithCurrent],
    electrode_B: list[ElectrodeWithCurrent],
    conductivity_config: dict[str, float],
    anisotropy: AnisotropyType,
    n_workers: int = 8,
) -> None:
    """
    运行 TI 正向仿真，生成失败则抛出异常。

    流程步骤：
    步骤 0：初始化与检查
    步骤 1：会话初始化
    步骤 2：电极对 A 配置
    步骤 3：电极对 B 配置
    步骤 4：TDCS 仿真执行
    步骤 5：TI 计算
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
    electrode_A : list[ElectrodeWithCurrent]
        电极组 A，每个元素包含 name 和 current_mA
    electrode_B : list[ElectrodeWithCurrent]
        电极组 B，每个元素包含 name 和 current_mA
    conductivity_config : dict[str, float]
        组织电导率配置
    anisotropy : AnisotropyType
        各向异性类型
    n_workers : int
        并行工作进程数
    """
    logger.info(
        "开始 TI 正向仿真: head_model_id=%s, head_model_dir=%s, output_dir=%s, montage=%s, anisotropy=%s, "
        "electrode_A=%s, electrode_B=%s, conductivity=%s",
        head_model_id,
        head_model_dir,
        output_dir,
        montage,
        anisotropy,
        electrode_A,
        electrode_B,
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

    eeg_cap = find_montage_file(str(subject_dir), montage)
    mesh_path = subject_dir / f"{head_model_id}.msh"
    if not mesh_path.exists():
        raise FileNotFoundError(f"头模 mesh 文件不存在: {mesh_path}")

    electrode_A_names = [e.name for e in electrode_A]
    electrode_A_currents = [e.current_mA / 1000 for e in electrode_A]
    electrode_B_names = [e.name for e in electrode_B]
    electrode_B_currents = [e.current_mA / 1000 for e in electrode_B]

    # ===== 步骤 1：会话初始化 =====
    S = setup_session(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir_path),
        anisotropy_type=anisotropy,
        cond=cond_dict_to_list(conductivity_config),
        fname_tensor=dti_file_path,
        eeg_cap=eeg_cap,
    )
    logger.info("会话初始化完成")

    # ===== 步骤 2：电极对 A 配置 =====
    setup_electrode_pair1(
        session=S,
        electrode_pair1=electrode_A_names,
        current1=electrode_A_currents,
    )
    logger.info("电极对 A 配置完成")

    # ===== 步骤 3：电极对 B 配置 =====
    setup_electrode_pair2(
        session=S,
        electrode_pair2=electrode_B_names,
        current2=electrode_B_currents,
    )
    logger.info("电极对 B 配置完成")

    # ===== 步骤 4：TDCS 仿真执行 =====
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=str(subject_dir),
        output_dir=str(output_dir_path),
        n_workers=n_workers,
    )
    logger.info("TDCS 仿真完成")

    # ===== 步骤 5：TI 计算 =====
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=str(output_dir_path),
    )
    logger.info("TI 计算完成")

    # ===== 步骤 6：NIfTI 导出 =====
    ti_nifti_path = export_ti_to_nifti(
        msh_path=ti_mesh_path,
        output_dir=str(output_dir_path),
        reference=t1_file_path,
        field_name="max_TI",
        prefix=f"{head_model_id}_simulation",
    )
    logger.info("NIfTI 导出完成: %s", ti_nifti_path)

    logger.info("TI 正向仿真完成")


def resolve_ti_forward_dirs(
    data_root: str,
    task_id: str,
    head_model_id: str,
) -> tuple[str, str]:
    """
    根据目录约定解析 TI 正向仿真目录。

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
    simulation_dir = root_dir / "simulations" / f"ti_forward_{task_id}"
    head_model_dir = root_dir / "head_models" / f"m2m_{head_model_id}"
    if not simulation_dir.is_dir():
        raise FileNotFoundError(f"仿真目录不存在: {simulation_dir}")
    return str(simulation_dir), str(head_model_dir)


def load_ti_forward_params(
    simulation_dir: str,
    head_model_dir: str,
) -> tuple[dict, str]:
    """
    根据已解析目录读取 TI 正向仿真输入参数。

    原理
    ----
    头模文件和仿真参数分离存储：
    - 头模相关文件位于 head_models
    - 当前任务参数、日志和结果位于 simulations
    该函数在日志初始化后执行头模目录检查和 JSON 读取，让失败原因可以写入任务日志。

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

    Raises
    ------
    FileNotFoundError
        当头模目录或 params.json 不存在时抛出
    ValueError
        当 params.json 顶层结构不是对象时抛出
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
        "electrode_A": params_data.get("electrode_A"),
        "electrode_B": params_data.get("electrode_B"),
        "conductivity_config": params_data.get("conductivity_config"),
        "anisotropy": params_data.get("anisotropy_type"),
    }
    return params_dict, str(simulation_dir_path)


def build_forward_params(
    params_dict: dict,
) -> tuple[
    str,
    list[ElectrodeWithCurrent],
    list[ElectrodeWithCurrent],
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
    tuple[str, list[ElectrodeWithCurrent], list[ElectrodeWithCurrent], dict[str, float], AnisotropyType]
        montage、电极组 A、电极组 B、电导率、各向异性
    """
    electrode_a = [
        ElectrodeWithCurrent(name=item["name"], current_mA=item["current_mA"])
        for item in params_dict["electrode_A"]
    ]
    electrode_b = [
        ElectrodeWithCurrent(name=item["name"], current_mA=item["current_mA"])
        for item in params_dict["electrode_B"]
    ]
    anisotropy = AnisotropyType(params_dict["anisotropy"])
    return (
        params_dict["montage"],
        electrode_a,
        electrode_b,
        params_dict["conductivity_config"],
        anisotropy,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建 TI 正向仿真命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        命令行解析器
    """
    parser = argparse.ArgumentParser(
        prog="python neuracle/ti_forward.py",
        description="运行 TI 正向仿真流程",
    )
    parser.add_argument("data_root", help="数据根目录")
    parser.add_argument("task_id", help="仿真任务 ID")
    parser.add_argument("head_model_id", help="头模 ID，不带 m2m_ 前缀")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    TI 正向仿真命令行入口。

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
    try:
        simulation_dir, head_model_dir = resolve_ti_forward_dirs(
            args.data_root,
            args.task_id,
            args.head_model_id,
        )
        setup_logging(str(Path(simulation_dir) / "logs"))
        params_dict, simulation_dir = load_ti_forward_params(
            simulation_dir,
            head_model_dir,
        )
        validate_forward_params(params_dict)
        (
            montage,
            electrode_a,
            electrode_b,
            conductivity_config,
            anisotropy,
        ) = build_forward_params(params_dict)
        run_ti_forward(
            head_model_id=args.head_model_id,
            head_model_dir=params_dict["head_model_dir"],
            output_dir=simulation_dir,
            montage=montage,
            electrode_A=electrode_a,
            electrode_B=electrode_b,
            conductivity_config=conductivity_config,
            anisotropy=anisotropy,
            n_workers=N_WORKERS,
        )
    except ValidationError as exc:
        logger.exception("TI 正向仿真参数校验失败: %s", exc)
        return EXIT_INVALID_ARGS
    except ValueError as exc:
        logger.exception("TI 正向仿真数值处理失败: %s", exc)
        return EXIT_VALUE_ERROR
    except Exception:
        logger.exception("TI 正向仿真执行失败")
        return EXIT_RUNTIME_ERROR
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
