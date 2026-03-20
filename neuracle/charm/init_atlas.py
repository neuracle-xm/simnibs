"""
CHARM 步骤4: Atlas 初始仿射配准与颈部校正

将 MNI Atlas 仿射配准到输入图像空间，并进行颈部校正。

原理：
    1. 设置 samseg 相关的 atlas 路径和参数
    2. 根据 init_type 设置决定初始化方式（atlas 或 mni）
    3. 执行仿射配准，输出模板到受试者空间的变换
    4. 可选进行颈部校正（neck optimization）

输入：
    - T1fs.nii.gz (或 T1fs_denoised.nii.gz)
    - T2_reg.nii.gz (可选)

输出：
    - segmentation/template_coregistered.nii.gz

用法：
    python -m neuracle.charm.init_atlas <subid> [--use-transform <transform_file>]
"""

import argparse
import logging
import os

import numpy as np

from neuracle.charm.nifti_utils import MAX_THREADS, _read_settings, _setup_atlas
from simnibs.segmentation import charm_utils, simnibs_samseg
from simnibs.utils import file_finder, settings_reader

logger = logging.getLogger(__name__)


def init_atlas(
    subject_dir: str,
    use_transform: str | None = None,
    init_transform: str | None = None,
    noneck: bool = False,
) -> None:
    """
    执行 Atlas 初始仿射配准

    Parameters
    ----------
    subject_dir : str
        Subject directory (m2m_{subid})
    use_transform : str or None, optional
        用户提供的世界到世界变换矩阵路径 (default: None)
    init_transform : str or None, optional
        初始化用的变换矩阵路径 (default: None)
    noneck : bool, optional
        是否跳过颈部校正 (default: False)

    Returns
    -------
    None
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    settings = _read_settings(sub_files.settings)
    samseg_settings = settings["samseg"]
    num_threads = settings["general"]["threads"]
    if isinstance(num_threads, int) and num_threads > 0:
        simnibs_samseg.setGlobalDefaultNumberOfThreads(num_threads)
        logger.info("使用 %d 个线程，而非全部可用线程。", num_threads)
    else:
        simnibs_samseg.setGlobalDefaultNumberOfThreads(MAX_THREADS)
    show_figs = False
    show_movies = False
    visualizer = simnibs_samseg.initVisualizer(show_figs, show_movies)
    (
        template_name,
        atlas_settings,
        atlas_path,
        atlas_level1,
        atlas_level2,
        atlas_affine_name,
        gmm_parameters,
    ) = _setup_atlas(samseg_settings, sub_files.T2_reg, None)
    os.makedirs(sub_files.segmentation_folder, exist_ok=True)
    if os.path.exists(sub_files.T1_denoised):
        input_t1 = sub_files.T1_denoised
        logger.info("使用去噪后的 T1 图像: %s", input_t1)
    else:
        input_t1 = sub_files.reference_volume
        logger.info("使用参考 T1 图像: %s", input_t1)
    if use_transform is not None:
        logger.info("使用用户提供的世界到世界变换矩阵: %s", use_transform)
        trans_mat = _read_transform(use_transform)
    else:
        if samseg_settings["init_type"] == "atlas":
            logger.info("使用 atlas 方法进行仿射初始化")
            trans_mat = None
        elif samseg_settings["init_type"] == "mni":
            logger.info("使用 MNI 模板进行仿射初始化")
            mni_template = file_finder.Templates().mni_volume
            mni_settings = settings["initmni"]
            trans_mat = charm_utils._init_atlas_affine(
                input_t1, mni_template, mni_settings
            )
        else:
            logger.info("仿射初始化类型未知 (%s)，默认为 'atlas'", samseg_settings["init_type"])
            trans_mat = None
    if init_transform is not None:
        logger.info("使用初始变换矩阵: %s", init_transform)
        init_transform = _read_transform(init_transform)
    if noneck:
        logger.info("跳过颈部校正")
    logger.info("开始仿射配准和颈部校正。")
    charm_utils._register_atlas_to_input_affine(
        input_t1,
        template_name,
        atlas_affine_name,
        atlas_level1,
        atlas_level2,
        sub_files.segmentation_folder,
        sub_files.template_coregistered,
        settings["initatlas"],
        atlas_settings["neck_optimization"]["neck_tissues"],
        visualizer,
        noneck,
        world_to_world_transform_matrix=trans_mat,
        init_transform=init_transform,
    )
    logger.info("Atlas 配准完成")


def _read_transform(transform_file: str) -> np.ndarray:
    """
    读取变换矩阵

    Parameters
    ----------
    transform_file : str
        变换矩阵文件路径

    Returns
    -------
    np.ndarray
        4x4 变换矩阵
    """
    transform = np.loadtxt(transform_file)
    assert transform.shape == (4, 4), (
        f"`transform` should have shape (4, 4), got {transform.shape}"
    )
    RAS2LPS = np.diag([-1, -1, 1, 1])
    return RAS2LPS @ transform @ RAS2LPS


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    解析命令行参数

    Parameters
    ----------
    argv : list[str] or None
        命令行参数列表

    Returns
    -------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        prog="python -m neuracle.charm.init_atlas",
        description="Register atlas to input images for CHARM pipeline",
    )
    parser.add_argument("subid", help="Subject ID (e.g., sub001)")
    parser.add_argument(
        "--use-transform",
        type=str,
        help="Use provided world-to-world transformation matrix",
    )
    parser.add_argument(
        "--init-transform",
        type=str,
        help="Initialization transformation matrix",
    )
    parser.add_argument(
        "--noneck",
        action="store_true",
        help="Skip neck correction",
    )
    return parser.parse_args(argv)


def main() -> None:
    """主函数"""
    args = parse_arguments()
    subject_dir = os.path.join(os.getcwd(), "m2m_" + args.subid)
    init_atlas(
        subject_dir=subject_dir,
        use_transform=args.use_transform,
        init_transform=args.init_transform,
        noneck=args.noneck,
    )


if __name__ == "__main__":
    main()
