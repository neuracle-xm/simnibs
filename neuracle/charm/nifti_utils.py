"""
NIfTI 图像处理工具函数
"""

import logging
import os

import nibabel as nib
import numpy as np

from simnibs.utils import file_finder, settings_reader

logger = logging.getLogger(__name__)

# 最大线程数常量
MAX_THREADS = 8


def _check_q_and_s_form(
    scan: nib.Nifti1Image,
    force_qform: bool = False,
    force_sform: bool = False,
) -> nib.Nifti1Image:
    """
    检查并修正 qform/sform 编码

    Parameters
    ----------
    scan : nib.Nifti1Image
        输入 NIfTI 图像
    force_qform : bool
        强制使用 qform
    force_sform : bool
        强制使用 sform

    Returns
    -------
    nib.Nifti1Image
        修正后的图像
    """
    if not scan.get_qform(coded=True)[1] > 0 and force_sform is False:
        raise ValueError(
            "The qform_code is 0. Please check the header of the input scan. "
            "You can use the sform instead by using --forcesform option."
        )
    if not np.allclose(scan.get_qform(), scan.get_sform(), rtol=1e-5, atol=1e-6):
        if not (force_qform or force_sform):
            raise ValueError(
                "The qform and sform matrices do not match. "
                "Please use --forceqform (preferred) or --forcesform option"
            )
        if force_qform:
            (qmat, qcode) = scan.get_qform(coded=True)
            scan.set_sform(qmat, code=qcode)
        elif force_sform:
            if not scan.get_sform(coded=True)[1] > 0:
                raise ValueError(
                    "The sform_code is 0, but you are forcing it. "
                    "Please fix the sform_code or use the qform instead."
                )
            (mat_tmp, code_tmp) = scan.get_sform(coded=True)
            scan.set_qform(mat_tmp, code=code_tmp)
            (mat_tmp, code_tmp) = scan.get_qform(coded=True)
            scan.set_sform(mat_tmp, code=code_tmp)
    return scan


def _read_settings(fn_settings: str) -> dict:
    """
    读取 CHARM 设置文件

    Parameters
    ----------
    fn_settings : str
        设置文件路径

    Returns
    -------
    dict
        设置字典
    """
    return settings_reader.read_ini(fn_settings)


def _setup_atlas(
    samseg_settings: dict,
    t2_reg: str,
    use_settings: str | None,
) -> tuple:
    """
    设置 Atlas 路径和参数

    Parameters
    ----------
    samseg_settings : dict
        Samseg 设置
    t2_reg : str
        T2 配准后文件路径
    use_settings : str or None
        自定义设置文件路径

    Returns
    -------
    tuple
        (template_name, atlas_settings, atlas_path, atlas_level1, atlas_level2,
         atlas_affine_name, gmm_parameters)
    """
    atlas_name = samseg_settings["atlas_name"]
    logger.info("使用 %s 作为 charm atlas。", atlas_name)
    atlas_path = os.path.join(file_finder.templates.charm_atlas_path, atlas_name)
    atlas_settings = settings_reader.read_ini(
        os.path.join(atlas_path, atlas_name + ".ini")
    )
    atlas_settings_names = atlas_settings["names"]
    template_name = os.path.join(atlas_path, atlas_settings_names["template_name"])
    atlas_affine_name = os.path.join(atlas_path, atlas_settings_names["affine_atlas"])
    atlas_level1 = os.path.join(atlas_path, atlas_settings_names["atlas_level1"])
    atlas_level2 = os.path.join(atlas_path, atlas_settings_names["atlas_level2"])
    custom_gmm_parameters = samseg_settings["gmm_parameter_file"]
    if not use_settings or not custom_gmm_parameters:
        if os.path.exists(t2_reg):
            gmm_parameters = os.path.join(
                atlas_path, atlas_settings_names["gaussian_parameters_t2"]
            )
        else:
            gmm_parameters = os.path.join(
                atlas_path, atlas_settings_names["gaussian_parameters_t1"]
            )
    else:
        settings_dir = os.path.dirname(use_settings)
        gmm_parameters = os.path.join(settings_dir, custom_gmm_parameters)
        if not os.path.exists(gmm_parameters):
            raise FileNotFoundError(
                f"Could not find gmm parameter file: {gmm_parameters}"
            )
    return (
        template_name,
        atlas_settings,
        atlas_path,
        atlas_level1,
        atlas_level2,
        atlas_affine_name,
        gmm_parameters,
    )
