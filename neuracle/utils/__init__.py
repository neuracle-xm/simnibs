"""
neuracle.utils 模块

提供通用工具函数和数据结构。
"""

import os

from neuracle.utils.atlas_utils import (
    ATLAS_COLOR_TABLE_DIR,
    ATLAS_MANIFESTS_DIR,
    ATLAS_REGISTRY_PATH,
    ATLAS_ROOT,
    ATLAS_SCRIPTS_DIR,
    ATLAS_SOURCE_DIR,
    ATLAS_STANDARDIZED_DIR,
    ATLAS_VALIDATION_DIR,
    SIMNIBS_MNI_TEMPLATE,
    get_atlas_spec,
    get_standardized_roi_path,
    iter_atlas_specs,
    load_atlas_registry,
    write_atlas_registry,
)
from neuracle.utils.cond_utils import TISSUE_ORDER, cond_dict_to_list
from neuracle.utils.env import get_aliyun_config, get_rabbitmq_config, load_env
from neuracle.utils.params_utils import (
    dict_to_ack_test_params,
    dict_to_forward_params,
    dict_to_inverse_params,
    dict_to_model_params,
)

N_WORKERS = 8
NON_ROI_THRESHOLD = 0.1

# 标准电导率值 (S/m)
STANDARD_COND = {
    "White Matter": 0.126,
    "Gray Matter": 0.275,
    "CSF": 1.654,
    "Bone": 0.01,
    "Scalp": 0.465,
    "Eye balls": 0.5,
    "Compact Bone": 0.008,
    "Spongy Bone": 0.025,
    "Blood": 0.6,
    "Muscle": 0.16,
}

# 电极帽 CSV 文件名常量
EEG10_10_CUTINI_2011 = "EEG10-10_Cutini_2011"
EEG10_10_NEUROELECTRICS = "EEG10-10_Neuroelectrics"
EEG10_10_UI_JURAK_2007 = "EEG10-10_UI_Jurak_2007"
EEG10_20_OKAMOTO_2004 = "EEG10-20_Okamoto_2004"
EEG10_20_EXTENDED_SPM12 = "EEG10-20_extended_SPM12"


def find_montage_file(dir_path: str, montage: str) -> str:
    """
    在头模目录中查找指定名称的 montage CSV 文件

    Parameters
    ----------
    dir_path : str
        头模目录路径
    montage : str
        montage 文件名或路径

    Returns
    -------
    str
        完整的 CSV 文件路径

    Raises
    ------
    FileNotFoundError
        当 montage 文件不存在时
    """
    candidate_paths = []
    if os.path.isabs(montage):
        candidate_paths.append(montage)
    else:
        candidate_paths.append(os.path.join(dir_path, "eeg_positions", montage))
        candidate_paths.append(
            os.path.join(dir_path, "eeg_positions", f"{montage}.csv")
        )
    for montage_path in candidate_paths:
        if os.path.exists(montage_path):
            return montage_path

    raise FileNotFoundError(f"montage 文件不存在: {montage}")


__all__ = [
    "N_WORKERS",
    "NON_ROI_THRESHOLD",
    "STANDARD_COND",
    "TISSUE_ORDER",
    "ATLAS_ROOT",
    "ATLAS_SOURCE_DIR",
    "ATLAS_STANDARDIZED_DIR",
    "ATLAS_MANIFESTS_DIR",
    "ATLAS_VALIDATION_DIR",
    "ATLAS_SCRIPTS_DIR",
    "ATLAS_COLOR_TABLE_DIR",
    "ATLAS_REGISTRY_PATH",
    "SIMNIBS_MNI_TEMPLATE",
    "cond_dict_to_list",
    "dict_to_ack_test_params",
    "dict_to_forward_params",
    "dict_to_inverse_params",
    "dict_to_model_params",
    "get_rabbitmq_config",
    "get_aliyun_config",
    "load_atlas_registry",
    "write_atlas_registry",
    "get_atlas_spec",
    "get_standardized_roi_path",
    "iter_atlas_specs",
    "load_env",
    "find_montage_file",
    "EEG10_10_CUTINI_2011",
    "EEG10_10_NEUROELECTRICS",
    "EEG10_10_UI_JURAK_2007",
    "EEG10_20_OKAMOTO_2004",
    "EEG10_20_EXTENDED_SPM12",
]
