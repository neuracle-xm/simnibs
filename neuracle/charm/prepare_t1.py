"""
CHARM 步骤1: T1 图像准备

将原始 T1 加权 MRI 图像转换为 SimNIBS 所需格式，执行 qform/sform 检查。

原理：
    1. 加载原始 T1 NIfTI 文件
    2. 检查并修正 qform/sform 编码
    3. 去除单例维度（singleton dimensions）
    4. 转换为 float32 并保存到 subject_dir

用法：
    python -m neuracle.charm.prepare_t1 <subid> <T1_file> [--forcesform]
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from neuracle.charm.nifti_utils import _check_q_and_s_form
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def prepare_t1(
    subject_dir: str,
    t1: str,
    force_qform: bool = False,
    force_sform: bool = False,
) -> None:
    """
    准备 T1 图像

    Parameters
    ----------
    subject_dir : str
        Subject directory (m2m_{subid})
    t1 : str
        Path to input T1 NIfTI file
    force_qform : bool, optional
        强制使用 qform 编码 (default: False)
    force_sform : bool, optional
        强制使用 sform 编码 (default: False)

    Returns
    -------
    None
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    logger.info("正在准备 T1 图像: %s", t1)
    if force_qform and force_sform:
        logger.error("不能同时强制使用 qform 和 sform")
        raise ValueError("Cannot force both qform and sform, please use only one")
    t1_tmp = nib.load(t1)
    t1_tmp = _check_q_and_s_form(t1_tmp, force_qform, force_sform)
    if (np.array(t1_tmp.shape) == 1).any():
        logger.info("检测到单例维度，正在去除")
        data_tmp = np.squeeze(t1_tmp.get_fdata())
        t1_tmp = nib.Nifti1Image(data_tmp, t1_tmp.affine)
    t1_tmp.set_data_dtype(np.float32)
    nib.save(t1_tmp, sub_files.reference_volume)
    logger.info("T1 图像已保存至: %s", sub_files.reference_volume)


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
        prog="python -m neuracle.charm.prepare_t1",
        description="Prepare T1 image for CHARM pipeline",
    )
    parser.add_argument("subid", help="Subject ID (e.g., sub001)")
    parser.add_argument("t1", help="Path to T1 NIfTI file")
    parser.add_argument(
        "--forceqform",
        action="store_true",
        help="Force use of qform matrix",
    )
    parser.add_argument(
        "--forcesform",
        action="store_true",
        help="Force use of sform matrix",
    )
    return parser.parse_args(argv)


def main() -> None:
    """主函数"""
    args = parse_arguments()
    subject_dir = os.path.join(os.getcwd(), "m2m_" + args.subid)
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)
    prepare_t1(
        subject_dir=subject_dir,
        t1=args.t1,
        force_qform=args.forceqform,
        force_sform=args.forcesform,
    )


if __name__ == "__main__":
    main()
