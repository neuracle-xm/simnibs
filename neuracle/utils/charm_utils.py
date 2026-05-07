"""
CHARM 流程辅助工具函数

本模块提供 CHARM 分割流程中所需的辅助函数，包括：
    - qform/sform 编码检查与修正
    - CHARM 配置文件的读取
    - Atlas 路径和参数的设置

用法：
    from neuracle.utils.charm_utils import check_q_and_s_form, read_settings
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from simnibs import SIMNIBSDIR
from simnibs.utils import file_finder, settings_reader

logger = logging.getLogger(__name__)

ALLOWED_OVERRIDE_FIELDS = {
    "segment": {"downsampling_targets"},
    "mesh": {"elem_sizes", "skin_facet_size", "facet_distances"},
}


def check_q_and_s_form(
    scan: nib.Nifti1Image,
    force_qform: bool = False,
    force_sform: bool = False,
) -> nib.Nifti1Image:
    """
    检查并修正 qform/sform 编码。

    Parameters
    ----------
    scan : nib.Nifti1Image
        输入 nifti 图像
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
        logger.error("qform_code 为 0，请检查输入图像的头部信息: %s", scan)
        raise ValueError(
            "qform_code 为 0，请检查输入图像的头部信息。"
            "可以使用 --forcesform 选项强制使用 sform。"
        )
    if not np.allclose(scan.get_qform(), scan.get_sform(), rtol=1e-5, atol=1e-6):
        if not (force_qform or force_sform):
            logger.error("qform 和 sform 矩阵不匹配")
            raise ValueError(
                "qform 和 sform 矩阵不匹配。"
                "请使用 --forceqform（推荐）或 --forcesform 选项。"
            )
        if force_qform:
            (qmat, qcode) = scan.get_qform(coded=True)
            scan.set_sform(qmat, code=qcode)
        elif force_sform:
            if not scan.get_sform(coded=True)[1] > 0:
                logger.error("sform_code 为 0 但被强制使用")
                raise ValueError(
                    "sform_code 为 0 但被强制使用。请修复 sform_code 或使用 qform。"
                )
            (mat_tmp, code_tmp) = scan.get_sform(coded=True)
            scan.set_qform(mat_tmp, code=code_tmp)
            (mat_tmp, code_tmp) = scan.get_qform(coded=True)
            scan.set_sform(mat_tmp, code=code_tmp)
    return scan


def read_settings() -> dict:
    """
    读取 CHARM 设置文件。

    Returns
    -------
    dict
        设置字典
    """
    src_settings = os.path.join(SIMNIBSDIR, "charm.ini")
    settings = settings_reader.read_ini(src_settings)
    override_path = get_charm_override_path()
    if not override_path.exists():
        logger.info("未找到 CHARM 覆盖配置，使用默认配置: %s", src_settings)
        return settings
    override_settings = read_charm_override(override_path)
    validate_charm_override(override_settings)
    logger.info(
        "CHARM 覆盖配置内容: %s",
        json.dumps(override_settings, ensure_ascii=False, sort_keys=True),
    )
    apply_charm_override(settings, override_settings)
    logger.info("已加载 CHARM 覆盖配置: %s", override_path)
    return settings


def get_charm_override_path() -> Path:
    """
    获取 CHARM 覆盖配置文件路径。

    Returns
    -------
    Path
        覆盖配置文件路径
    """
    return Path(__file__).resolve().parents[1] / "charm_override.json"


def read_charm_override(config_file: Path) -> dict[str, Any]:
    """
    读取 CHARM 覆盖配置文件。

    Parameters
    ----------
    config_file : Path
        覆盖配置文件路径

    Returns
    -------
    dict[str, Any]
        覆盖配置字典
    """
    logger.info("读取 CHARM 覆盖配置: %s", config_file)
    with config_file.open("r", encoding="utf-8") as file_obj:
        try:
            config = json.load(file_obj)
        except json.JSONDecodeError as exc:
            logger.error("CHARM 覆盖配置不是合法 JSON: %s", config_file)
            raise ValueError(f"CHARM 覆盖配置不是合法 JSON: {config_file}") from exc
    if not isinstance(config, dict):
        logger.error("CHARM 覆盖配置顶层必须是对象: %s", config_file)
        raise ValueError(f"CHARM 覆盖配置顶层必须是对象: {config_file}")
    return config


def validate_charm_override(config: dict[str, Any]) -> None:
    """
    校验 CHARM 覆盖配置。

    Parameters
    ----------
    config : dict[str, Any]
        覆盖配置字典

    Returns
    -------
    None
    """
    for section, values in config.items():
        if section not in ALLOWED_OVERRIDE_FIELDS:
            logger.error("CHARM 覆盖配置包含不允许的 section: %s", section)
            raise ValueError(f"CHARM 覆盖配置包含不允许的 section: {section}")
        if not isinstance(values, dict):
            logger.error("CHARM 覆盖配置的 section 必须是对象: %s", section)
            raise ValueError(f"CHARM 覆盖配置的 section 必须是对象: {section}")
        allowed_keys = ALLOWED_OVERRIDE_FIELDS[section]
        for key, value in values.items():
            if key not in allowed_keys:
                logger.error("CHARM 覆盖配置包含不允许的字段: %s.%s", section, key)
                raise ValueError(f"CHARM 覆盖配置包含不允许的字段: {section}.{key}")
            validate_charm_override_value(section, key, value)


def validate_charm_override_value(section: str, key: str, value: Any) -> None:
    """
    校验单个 CHARM 覆盖字段。

    Parameters
    ----------
    section : str
        配置 section 名称
    key : str
        配置字段名称
    value : Any
        配置字段值

    Returns
    -------
    None
    """
    if section == "segment" and key == "downsampling_targets":
        if not isinstance(value, list):
            logger.error("%s.%s 必须是数组", section, key)
            raise ValueError(f"{section}.{key} 必须是数组")
        return
    if section == "mesh" and key == "elem_sizes":
        if not isinstance(value, dict):
            logger.error("%s.%s 必须是对象", section, key)
            raise ValueError(f"{section}.{key} 必须是对象")
        return
    if section == "mesh" and key == "skin_facet_size":
        if isinstance(value, bool):
            if value is False:
                return
            logger.error("%s.%s 只能是数字或 false", section, key)
            raise ValueError(f"{section}.{key} 只能是数字或 false")
        if not isinstance(value, int) and not isinstance(value, float):
            logger.error("%s.%s 必须是数字或 false", section, key)
            raise ValueError(f"{section}.{key} 必须是数字或 false")
        return
    if section == "mesh" and key == "facet_distances":
        if not isinstance(value, dict):
            logger.error("%s.%s 必须是对象", section, key)
            raise ValueError(f"{section}.{key} 必须是对象")
        return
    logger.error("未识别的 CHARM 覆盖字段: %s.%s", section, key)
    raise ValueError(f"未识别的 CHARM 覆盖字段: {section}.{key}")


def apply_charm_override(
    base_settings: dict[str, Any],
    override_settings: dict[str, Any],
) -> None:
    """
    应用 CHARM 覆盖配置。

    Parameters
    ----------
    base_settings : dict[str, Any]
        默认配置
    override_settings : dict[str, Any]
        覆盖配置

    Returns
    -------
    None
    """
    for section, values in override_settings.items():
        for key, value in values.items():
            logger.info("覆盖 CHARM 配置: %s.%s", section, key)
            base_settings[section][key] = value


def setup_atlas(
    samseg_settings: dict,
    t2_reg: str,
    use_settings: str | None,
) -> tuple:
    """
    设置 Atlas 路径和参数。

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
            logger.error("找不到 GMM 参数文件: %s", gmm_parameters)
            raise FileNotFoundError(f"找不到 GMM 参数文件: {gmm_parameters}")
    return (
        template_name,
        atlas_settings,
        atlas_path,
        atlas_level1,
        atlas_level2,
        atlas_affine_name,
        gmm_parameters,
    )
