"""
Session 模块 - SimNIBS 会话配置

提供 SimNIBS SESSION 对象的配置功能，用于设置仿真所需的头部模型路径、输出目录、
网格、输出目录及张量文件等参数。

原理：
    SimNIBS 仿真需要一个 SESSION 对象来管理所有仿真参数，包括：
    - 头部模型路径 (fnamehead)
    - 输出目录 (pathfem)
    - DTI 张量文件 (fname_tensor)
    - EEG 电极帽配置 (eeg_cap)

用法：
    from neuracle.ti_simulation import setup_session

    S = setup_session(
        subject_dir="path/to/m2m_subid",
        output_dir="path/to/output",
        msh_file_path="path/to/model.msh"
    )
"""

import logging
import os

from simnibs import sim_struct

logger = logging.getLogger(__name__)


def setup_session(
    subject_dir: str,
    output_dir: str,
    msh_file_path: str | None = None,
    fname_tensor: str | None = None,
    eeg_cap: str | None = None,
) -> sim_struct.SESSION:
    """
    配置 SimNIBS 会话参数

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}），包含头部模型数据
    output_dir : str
        仿真输出目录，存放仿真结果文件
    msh_file_path : str, optional
        头模网格文件路径；如果提供，则优先使用该 mesh 文件
    fname_tensor : str, optional
        DTI 张量文件路径，用于各向异性电导率计算
    eeg_cap : str, optional
        EEG 电极帽 CSV 文件路径（不含 .csv 扩展名）

    Returns
    -------
    sim_struct.SESSION
        配置好的 SimNIBS 会话对象

    Notes
    -----
    SESSION 对象是 SimNIBS 仿真的核心配置对象，需要配置以下主要属性：
    - subpath: Subject 数据目录
    - fnamehead: 头模网格文件路径
    - pathfem: 仿真输出目录
    - open_in_gmsh: 是否在 Gmsh 中打开结果（设置为 False 避免阻塞）
    - fname_tensor: DTI 张量文件路径
    - eeg_cap: EEG 电极帽配置
    """
    S = sim_struct.SESSION()
    S.subpath = subject_dir
    S.fnamehead = msh_file_path
    S.pathfem = output_dir
    S.open_in_gmsh = False
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
