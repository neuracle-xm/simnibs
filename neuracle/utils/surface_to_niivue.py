"""
CHARM 重建表面导出为 Niivue 可用的 GIfTI。

固定复用 CHARM 第 6 步生成的左右半球 central 表面，
合并后写出为单一 `surface.gii` 文件，供 Niivue 加载。
"""

import logging
import os

from simnibs.mesh_tools import mesh_io
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def read_central_surfaces(subject_dir: str) -> tuple[Msh, Msh]:
    """
    读取左右半球 central 皮层表面。

    原理：
        1. 通过 `SubjectFiles` 定位 `surfaces/lh.central.gii` 与 `rh.central.gii`
        2. 使用 SimNIBS 的 `read_gifti_surface()` 读取 GIfTI 表面
        3. 返回左右半球各自的 `Msh` 对象

    Parameters
    ----------
    subject_dir : str
        CHARM 受试者目录，例如 `m2m_ernie`

    Returns
    -------
    tuple[simnibs.mesh_tools.mesh_io.Msh, simnibs.mesh_tools.mesh_io.Msh]
        左右半球表面 `(left_surface, right_surface)`

    Raises
    ------
    FileNotFoundError
        当受试者目录或任一半球表面文件不存在时
    """
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError("受试者目录不存在: %s" % subject_dir)
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    left_surface_path = sub_files.get_surface("lh", "central")
    right_surface_path = sub_files.get_surface("rh", "central")
    if not left_surface_path.exists():
        raise FileNotFoundError("左半球表面文件不存在: %s" % left_surface_path)
    if not right_surface_path.exists():
        raise FileNotFoundError("右半球表面文件不存在: %s" % right_surface_path)
    logger.info("正在读取左半球 central 表面: %s", left_surface_path)
    left_surface = mesh_io.read_gifti_surface(str(left_surface_path))
    logger.info("正在读取右半球 central 表面: %s", right_surface_path)
    right_surface = mesh_io.read_gifti_surface(str(right_surface_path))
    return left_surface, right_surface


def join_hemisphere_surfaces(left_surface: Msh, right_surface: Msh) -> Msh:
    """
    合并左右半球表面。

    Parameters
    ----------
    left_surface : simnibs.mesh_tools.mesh_io.Msh
        左半球表面
    right_surface : simnibs.mesh_tools.mesh_io.Msh
        右半球表面

    Returns
    -------
    simnibs.mesh_tools.mesh_io.Msh
        合并后的表面网格
    """
    logger.info(
        "正在合并左右半球 central 表面: 左顶点=%d, 右顶点=%d",
        left_surface.nodes.nr,
        right_surface.nodes.nr,
    )
    return left_surface.join_mesh(right_surface)


def write_niivue_surface_gifti(surface_mesh: Msh, output_path: str) -> str:
    """
    将合并后的表面写出为 Niivue 可加载的 GIfTI 文件。

    Parameters
    ----------
    surface_mesh : simnibs.mesh_tools.mesh_io.Msh
        合并后的表面网格
    output_path : str
        输出文件路径

    Returns
    -------
    str
        输出文件路径
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    logger.info("正在写出 GIfTI 表面: %s", output_path)
    mesh_io.write_gifti_surface(
        surface_mesh,
        output_path,
        element_tag=file_finder.ElementTags.GM_TH_SURFACE,
    )
    return output_path


def export_surface_to_niivue_gifti(subject_dir: str) -> str:
    """
    将 central 皮层重建表面导出为 Niivue 可用的 GIfTI 文件。

    原理：
        1. 从 `surfaces/` 目录读取左右半球的 central 重建表面
        2. 合并左右半球
        3. 在受试者目录内部写出固定文件 `surface.gii`

    Parameters
    ----------
    subject_dir : str
        CHARM 受试者目录，例如 `m2m_ernie`

    Returns
    -------
    str
        输出 GIfTI 文件路径

    Raises
    ------
    FileNotFoundError
        当目录或表面文件不存在时
    """
    output_path = os.path.join(subject_dir, "surface.gii")
    left_surface, right_surface = read_central_surfaces(subject_dir=subject_dir)
    merged_surface = join_hemisphere_surfaces(left_surface, right_surface)
    return write_niivue_surface_gifti(
        surface_mesh=merged_surface, output_path=output_path
    )
