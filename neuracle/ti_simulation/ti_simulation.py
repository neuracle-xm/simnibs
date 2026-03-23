"""
TI Simulation - Temporal Interference 正向仿真

原理：
    1. 设置两个电极对，配置电流
    2. 运行两次 TDCS 仿真（每个电极对一次）
    3. 从仿真结果中提取电场数据
    4. 使用 TI_utils 计算 TI 最大调制振幅
    5. 生成可视化输出

用法：
    python -m neuracle.ti_simulation <subid> [--electrode-pair1 <pair1>]
            [--electrode-pair2 <pair2>] [--current1 <current1>]
            [--current2 <current2>] [--electrode-shape <shape>]
            [--electrode-dim <width height>] [--electrode-thickness <thickness>]
            [--n-workers <n_workers>]
"""

import argparse
import logging
import os
from copy import deepcopy

import numpy as np

from simnibs import ElementTags, mesh_io, run_simnibs, sim_struct
from simnibs.utils import TI_utils as TI

logger = logging.getLogger(__name__)


def run_ti_forward_simulation(
    subject_dir: str,
    output_dir: str,
    electrode_pair1: list[str],
    electrode_pair2: list[str],
    current1: float = 0.001,
    current2: float = 0.001,
    electrode_shape: str = "ellipse",
    electrode_dimensions: list[float] | None = None,
    electrode_thickness: float = 2.0,
    n_workers: int = 1,
) -> tuple[str, str]:
    """
    执行 TI 正向仿真 - 运行两组电极的 TDCS 仿真

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}）
    output_dir : str
        仿真输出目录
    electrode_pair1 : list
        第一个电极对 [elec1_name, elec2_name]，如 ['F5', 'P5']
    electrode_pair2 : list
        第二个电极对 [elec1_name, elec2_name]，如 ['F6', 'P6']
    current1 : float, optional
        第一组电极对电流强度，单位 A (default: 0.001，即 1mA)
    current2 : float, optional
        第二组电极对电流强度，单位 A (default: 0.001，即 1mA)
    electrode_shape : str, optional
        电极形状，可选 'ellipse', 'rect', 'custom' (default: 'ellipse')
    electrode_dimensions : list, optional
        电极尺寸 [width, height]，单位 mm (default: [40, 40])
    electrode_thickness : float, optional
        电极厚度，单位 mm (default: 2.0)
    n_workers : int, optional
        并行工作进程数 (default: 1)
        Note: 即使设置了多个worker，实际也还是没用

    Returns
    -------
    tuple
        (mesh1_path, mesh2_path) - 两组电极仿真结果网格路径
    """
    if electrode_dimensions is None:
        electrode_dimensions = [40, 40]

    logger.info("开始 TI 正向仿真")
    logger.info("Subject 目录: %s", subject_dir)
    logger.info("输出目录: %s", output_dir)
    logger.info("电极对1: %s, 电流: %s A", electrode_pair1, current1)
    logger.info("电极对2: %s, 电流: %s A", electrode_pair2, current2)

    # 设置会话
    S = sim_struct.SESSION()
    S.subpath = subject_dir
    S.pathfem = output_dir
    S.open_in_gmsh = False  # 禁止自动打开 Gmsh

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 配置第一个电极对
    logger.info("配置第一个电极对: %s", electrode_pair1)
    tdcs1 = _setup_electrode_pair(
        session=S,
        electrode_pair=electrode_pair1,
        currents=[current1, -current1],
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
    )

    # 配置第二个电极对
    logger.info("配置第二个电极对: %s", electrode_pair2)
    tdcs2 = _setup_electrode_pair(
        session=S,
        electrode_pair=electrode_pair2,
        currents=[current2, -current2],
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
    )

    # 运行仿真
    logger.info("开始运行 TDCS 仿真，n_workers=%s", n_workers)
    run_simnibs(S, cpus=n_workers)
    logger.info("TDCS 仿真完成")

    # 返回两组仿真结果路径
    # 从 subject_dir 提取 subject ID（支持 m2m_ernie 或 ernie 格式）
    subid = os.path.basename(subject_dir).replace("m2m_", "", 1)
    mesh1_path = os.path.join(output_dir, f"{subid}_TDCS_1_scalar.msh")
    mesh2_path = os.path.join(output_dir, f"{subid}_TDCS_2_scalar.msh")

    logger.info("TDCS 仿真结果路径: %s, %s", mesh1_path, mesh2_path)
    return mesh1_path, mesh2_path


def calculate_ti_envelope(
    mesh1_path: str,
    mesh2_path: str,
    output_dir: str,
) -> tuple[str, np.ndarray]:
    """
    计算 TI 包络

    Parameters
    ----------
    mesh1_path : str
        第一组电极仿真结果网格路径
    mesh2_path : str
        第二组电极仿真结果网格路径
    output_dir : str
        输出目录

    Returns
    -------
    tuple
        (ti_mesh_path, ti_max) - TI 结果网格路径和最大调制振幅数组
    """
    logger.info("开始计算 TI 场...")
    logger.info("读取网格文件: %s, %s", mesh1_path, mesh2_path)

    m1 = mesh_io.read_msh(mesh1_path)
    m2 = mesh_io.read_msh(mesh2_path)

    # 裁剪网格，去除电极元素
    tags_keep = np.hstack(
        (
            np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
            np.arange(
                ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1
            ),
        )
    )
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    # 计算 TI 最大调制振幅
    ef1 = m1.field["E"]
    ef2 = m2.field["E"]
    ti_max = TI.get_maxTI(ef1.value, ef2.value)

    # 生成可视化输出
    logger.info("生成 TI 可视化输出...")
    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(ef1.norm(), "magnE - pair 1")
    mout.add_element_field(ef2.norm(), "magnE - pair 2")
    mout.add_element_field(ti_max, "TImax")

    ti_mesh_path = os.path.join(output_dir, "TI.msh")
    mesh_io.write_msh(mout, ti_mesh_path)

    # 生成视图
    v = mout.view(
        visible_tags=[1002, 1006],
        visible_fields="TImax",
    )
    v.write_opt(ti_mesh_path)

    logger.info("TI 计算完成，输出文件: %s", ti_mesh_path)
    return ti_mesh_path, ti_max


def run_ti_simulation(
    subject_dir: str,
    output_dir: str,
    electrode_pair1: list[str],
    electrode_pair2: list[str],
    current1: float = 0.001,
    current2: float = 0.001,
    electrode_shape: str = "ellipse",
    electrode_dimensions: list[float] | None = None,
    electrode_thickness: float = 2.0,
    n_workers: int = 1,
) -> tuple[str, np.ndarray]:
    """
    执行 TI 正向仿真

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}）
    output_dir : str
        仿真输出目录
    electrode_pair1 : list
        第一个电极对 [elec1_name, elec2_name]，如 ['F5', 'P5']
    electrode_pair2 : list
        第二个电极对 [elec1_name, elec2_name]，如 ['F6', 'P6']
    current1 : float, optional
        第一组电极对电流强度，单位 A (default: 0.001，即 1mA)
    current2 : float, optional
        第二组电极对电流强度，单位 A (default: 0.001，即 1mA)
    electrode_shape : str, optional
        电极形状，可选 'ellipse', 'rect', 'custom' (default: 'ellipse')
    electrode_dimensions : list, optional
        电极尺寸 [width, height]，单位 mm (default: [40, 40])
    electrode_thickness : float, optional
        电极厚度，单位 mm (default: 2.0)
    n_workers : int, optional
        并行工作进程数 (default: 1)
        Note: 即使设置了多个worker，实际也还是没用

    Returns
    -------
    tuple
        (ti_mesh_path, ti_max) - TI 结果网格路径和最大调制振幅数组
    """
    # 运行正向仿真
    mesh1_path, mesh2_path = run_ti_forward_simulation(
        subject_dir=subject_dir,
        output_dir=output_dir,
        electrode_pair1=electrode_pair1,
        electrode_pair2=electrode_pair2,
        current1=current1,
        current2=current2,
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
        n_workers=n_workers,
    )

    # 计算 TI 包络
    ti_mesh_path, ti_max = calculate_ti_envelope(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=output_dir,
    )

    return ti_mesh_path, ti_max


def _setup_electrode_pair(
    session: sim_struct.SESSION,
    electrode_pair: list[str],
    currents: list[float],
    electrode_shape: str,
    electrode_dimensions: list[float],
    electrode_thickness: float,
) -> sim_struct.TDCSLIST:
    """
    配置单个电极对

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象
    electrode_pair : list
        电极对 [elec1_name, elec2_name]
    currents : list
        电流配置 [current1, current2]
    electrode_shape : str
        电极形状
    electrode_dimensions : list
        电极尺寸 [width, height]
    electrode_thickness : float
        电极厚度

    Returns
    -------
    TDCSLIST
        配置好的 TDCS 列表对象
    """
    tdcs = session.add_tdcslist()
    tdcs.currents = currents

    electrode = tdcs.add_electrode()
    electrode.channelnr = 1
    electrode.centre = electrode_pair[0]
    electrode.shape = electrode_shape
    electrode.dimensions = electrode_dimensions
    electrode.thickness = electrode_thickness

    electrode = tdcs.add_electrode()
    electrode.channelnr = 2
    electrode.centre = electrode_pair[1]
    electrode.shape = electrode_shape
    electrode.dimensions = electrode_dimensions
    electrode.thickness = electrode_thickness

    return tdcs


def parse_arguments(argv: list | None = None) -> argparse.Namespace:
    """
    解析命令行参数

    Parameters
    ----------
    argv : list or None
        命令行参数列表

    Returns
    -------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        prog="python -m neuracle.ti_simulation",
        description="Run TI (Temporal Interference) simulation for tDCS",
    )
    parser.add_argument("subid", help="Subject ID (e.g., ernie)")
    parser.add_argument(
        "--electrode-pair1",
        type=str,
        nargs=2,
        default=["F5", "P5"],
        help="First electrode pair, e.g., F5 P5",
    )
    parser.add_argument(
        "--electrode-pair2",
        type=str,
        nargs=2,
        default=["F6", "P6"],
        help="Second electrode pair, e.g., F6 P6",
    )
    parser.add_argument(
        "--current1",
        type=float,
        default=0.001,
        help="First electrode pair current amplitude in A (default: 0.001)",
    )
    parser.add_argument(
        "--current2",
        type=float,
        default=0.001,
        help="Second electrode pair current amplitude in A (default: 0.001)",
    )
    parser.add_argument(
        "--electrode-shape",
        type=str,
        default="ellipse",
        choices=["ellipse", "rect", "custom"],
        help="Electrode shape (default: ellipse)",
    )
    parser.add_argument(
        "--electrode-dim",
        type=float,
        nargs=2,
        default=[40, 40],
        help="Electrode dimensions [width height] in mm (default: 40 40)",
    )
    parser.add_argument(
        "--electrode-thickness",
        type=float,
        default=2.0,
        help="Electrode thickness in mm (default: 2.0)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """主函数"""
    args = parse_arguments()
    subject_dir = os.path.join(os.getcwd(), "m2m_" + args.subid)
    output_dir = os.path.join(os.getcwd(), "TI_" + args.subid)

    run_ti_simulation(
        subject_dir=subject_dir,
        output_dir=output_dir,
        electrode_pair1=args.electrode_pair1,
        electrode_pair2=args.electrode_pair2,
        current1=args.current1,
        current2=args.current2,
        electrode_shape=args.electrode_shape,
        electrode_dimensions=args.electrode_dim,
        electrode_thickness=args.electrode_thickness,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
