"""
CHARM 步骤8: 导出 Niivue 用 central 重建表面。

固定复用 CHARM 第 6 步生成的左右半球 central 重建表面，
合并后写出为受试者目录下固定文件 `surface.gii.gz`。
"""

import logging
import os

from neuracle.utils.surface_to_niivue import export_surface_to_niivue_gifti
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def export_niivue_surfaces(subject_dir: str) -> str:
    """
    从 CHARM 重建表面导出 Niivue 可用的 central GIfTI 文件。

    原理：
        1. 定位 `m2m_{subid}/surfaces/lh.central.gii` 与 `rh.central.gii`
        2. 读取左右半球 central 重建表面
        3. 合并左右半球
        4. 在 `{subject_dir}/surface.gii.gz` 中写出 `.gii.gz`

    Parameters
    ----------
    subject_dir : str
        受试者目录路径，例如 `m2m_ernie`

    Returns
    -------
    str
        生成的 GIfTI 文件路径

    Raises
    ------
    FileNotFoundError
        当受试者目录或重建表面文件不存在时
    """
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError("受试者目录不存在: %s" % subject_dir)
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    if not os.path.isdir(sub_files.surface_folder):
        raise FileNotFoundError("重建表面目录不存在: %s" % sub_files.surface_folder)
    output_path = os.path.join(subject_dir, "surface.gii.gz")
    logger.info("开始导出 Niivue central 重建表面: %s", subject_dir)
    logger.info("输入表面目录: %s", sub_files.surface_folder)
    logger.info("输出文件: %s", output_path)
    generated_path = export_surface_to_niivue_gifti(subject_dir=subject_dir)
    logger.info("Niivue central 重建表面导出完成")
    return generated_path
