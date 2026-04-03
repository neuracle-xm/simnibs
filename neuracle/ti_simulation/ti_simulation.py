"""
TI Simulation - Temporal Interference 正向仿真

原理：
    1. 设置两个电极对，配置电流
    2. 运行两次 TDCS 仿真（每个电极对一次）
    3. 从仿真结果中提取电场数据
    4. 使用 TI_utils 计算 TI 最大调制振幅
    5. 生成可视化输出

用法：
    from neuracle.ti_simulation import (
        setup_session,
        setup_electrode_pair1,
        setup_electrode_pair2,
        run_tdcs_simulation,
        calculate_ti,
    )

    # 1. 配置会话
    S = setup_session(subject_dir, output_dir)

    # 2. 配置电极对
    setup_electrode_pair1(S, ["F5", "P5"], current1=0.001)
    setup_electrode_pair2(S, ["F6", "P6"], current2=0.001)

    # 3. 运行仿真
    mesh1, mesh2 = run_tdcs_simulation(S, subject_dir, output_dir, n_workers=24)

    # 4. 计算 TI
    ti_mesh, ti_max = calculate_ti(mesh1, mesh2, output_dir)
"""

import logging
import os
from copy import deepcopy

import numpy as np

from simnibs import ElementTags, mesh_io, run_simnibs, sim_struct
from simnibs.utils import TI_utils as TI

logger = logging.getLogger(__name__)


def setup_session(
    subject_dir: str,
    output_dir: str,
    msh_file_path: str | None = None,
    anisotropy_type: str = "scalar",
    cond: list | None = None,
    fname_tensor: str | None = None,
    eeg_cap: str | None = None,
) -> sim_struct.SESSION:
    """
    配置会话参数

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}）
    msh_file_path : str, optional
        头模网格文件路径；如果提供，则优先使用该 mesh 文件
    output_dir : str
        仿真输出目录
    anisotropy_type : str, optional
        电导率各向异性类型 (default: "scalar")
    cond : list, optional
        电导率列表
    fname_tensor : str, optional
        DTI 张量文件路径
    eeg_cap : str, optional
        EEG 电极帽 CSV 文件路径（不含 .csv 扩展名）

    Returns
    -------
    sim_struct.SESSION
        配置好的会话对象
    """
    S = sim_struct.SESSION()
    S.subpath = subject_dir
    S.fnamehead = msh_file_path
    S.pathfem = output_dir
    S.open_in_gmsh = False
    S.anisotropy_type = anisotropy_type
    logger.info("电导率各向异性类型: %s", anisotropy_type)

    if cond is not None:
        S.cond = cond
        logger.info("使用自定义电导率值: %s", cond)
    else:
        logger.info("使用标准电导率值 (standard_cond)")

    if fname_tensor is not None:
        S.fname_tensor = fname_tensor
        logger.info("DTI 张量文件: %s", fname_tensor)

    S.eeg_cap = eeg_cap
    logger.info("EEG 电极帽: %s", eeg_cap)
    if msh_file_path is not None:
        logger.info("头模网格文件: %s", msh_file_path)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("会话配置完成")

    return S


def setup_electrode_pair1(
    session: sim_struct.SESSION,
    electrode_pair1: list[str],
    current1: list[float],
    electrode_shape: str = "ellipse",
    electrode_dimensions: list[float] | None = None,
    electrode_thickness: float = 2.0,
) -> sim_struct.TDCSLIST:
    """
    配置第一个电极对

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象
    electrode_pair1 : list
        第一个电极对 [elec1_name, elec2_name]
    current1 : list[float]
        第一组电极对电流列表 [anode_current, cathode_current]，单位 A
    electrode_shape : str
        电极形状
    electrode_dimensions : list
        电极尺寸 [width, height]
    electrode_thickness : float
        电极厚度

    Returns
    -------
    sim_struct.TDCSLIST
        配置好的 TDCS 列表对象
    """
    if electrode_dimensions is None:
        electrode_dimensions = [40, 40]

    logger.info("配置第一个电极对: %s", electrode_pair1)
    tdcs1 = _setup_electrode_pair(
        session=session,
        electrode_pair=electrode_pair1,
        currents=current1,
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
    )
    return tdcs1


def setup_electrode_pair2(
    session: sim_struct.SESSION,
    electrode_pair2: list[str],
    current2: list[float],
    electrode_shape: str = "ellipse",
    electrode_dimensions: list[float] | None = None,
    electrode_thickness: float = 2.0,
) -> sim_struct.TDCSLIST:
    """
    配置第二个电极对

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象
    electrode_pair2 : list
        第二个电极对 [elec1_name, elec2_name]
    current2 : list[float]
        第二组电极对电流列表 [anode_current, cathode_current]，单位 A
    electrode_shape : str
        电极形状
    electrode_dimensions : list
        电极尺寸 [width, height]
    electrode_thickness : float
        电极厚度

    Returns
    -------
    sim_struct.TDCSLIST
        配置好的 TDCS 列表对象
    """
    if electrode_dimensions is None:
        electrode_dimensions = [40, 40]

    logger.info("配置第二个电极对: %s", electrode_pair2)
    tdcs2 = _setup_electrode_pair(
        session=session,
        electrode_pair=electrode_pair2,
        currents=current2,
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
    )
    return tdcs2


def run_tdcs_simulation(
    session: sim_struct.SESSION,
    subject_dir: str,
    output_dir: str,
    n_workers: int = 1,
) -> tuple[str, str]:
    """
    运行 TDCS 仿真

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象
    subject_dir : str
        Subject 目录（用于提取 subid）
    output_dir : str
        仿真输出目录
    n_workers : int
        并行工作进程数

    Returns
    -------
    tuple
        (mesh1_path, mesh2_path) - 两组电极仿真结果网格路径
    """
    logger.info("开始运行 TDCS 仿真，n_workers=%s", n_workers)
    run_simnibs(session, cpus=n_workers)
    logger.info("TDCS 仿真完成")

    base_name = os.path.splitext(os.path.basename(session.fnamehead))[0]
    mesh1_path = os.path.join(output_dir, f"{base_name}_TDCS_1_scalar.msh")
    mesh2_path = os.path.join(output_dir, f"{base_name}_TDCS_2_scalar.msh")

    logger.info("TDCS 仿真结果路径: %s, %s", mesh1_path, mesh2_path)
    return mesh1_path, mesh2_path


def calculate_ti(
    mesh1_path: str,
    mesh2_path: str,
    output_dir: str,
) -> str:
    """
    计算 TI 场

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
    str
        TI 结果网格路径
    """
    logger.info("开始计算 TI 场...")
    logger.info("读取网格文件: %s, %s", mesh1_path, mesh2_path)

    m1 = mesh_io.read_msh(mesh1_path)
    m2 = mesh_io.read_msh(mesh2_path)

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

    ef1 = m1.field["E"]
    ef2 = m2.field["E"]
    ti_max = TI.get_maxTI(ef1.value, ef2.value)

    logger.info("生成 TI 可视化输出...")
    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(ti_max, "max_TI")

    ti_mesh_path = os.path.join(output_dir, "TI.msh")
    mesh_io.write_msh(mout, ti_mesh_path)

    v = mout.view(
        visible_tags=[1002, 1006],
        visible_fields="max_TI",
    )
    v.write_opt(ti_mesh_path)

    logger.info("TI 计算完成，输出文件: %s", ti_mesh_path)
    return ti_mesh_path


def _setup_electrode_pair(
    session: sim_struct.SESSION,
    electrode_pair: list[str],
    currents: list[float],
    electrode_shape: str,
    electrode_dimensions: list[float],
    electrode_thickness: float,
) -> sim_struct.TDCSLIST:
    """
    配置单个电极对（内部函数）

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象
    electrode_pair : list
        电极列表，支持多电极配置 [elec1_name, elec2_name, ...]
    currents : list
        电流配置，长度应与 electrode_pair 一致
        同一通道的多个电极电流会合并
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

    # 添加所有电极，channelnr 从 1 开始连续分配
    for i, elec_name in enumerate(electrode_pair):
        electrode = tdcs.add_electrode()
        electrode.channelnr = i + 1
        electrode.centre = elec_name
        electrode.shape = electrode_shape
        electrode.dimensions = electrode_dimensions
        electrode.thickness = electrode_thickness

    tdcs.currents = currents

    return tdcs
