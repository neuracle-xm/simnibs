"""
参数转换模块

将字典转换为对应的 dataclass 参数类型，用于参数解析和校验。

该模块是参数处理的中间层：
1. validator.py 验证字典格式正确（字段存在、类型正确、值范围合法）
2. converter.py 将验证通过的字典转换为强类型的 dataclass
3. 业务代码直接使用 dataclass，不用处理原始字典

这样做的好处：
- validator 不关心业务逻辑，只关心数据格式
- converter 转换后的 dataclass 有类型提示，方便 IDE 和静态检查
- 业务代码直接使用 dataclass，不用处理原始字典
"""

import logging
from pathlib import Path

from neuracle.parameters.schemas import (
    AtlasParam,
    ElectrodeWithCurrent,
    ForwardParams,
    InverseParams,
    MNIParam,
    ModelParams,
    ROIParam,
)
from neuracle.utils.constants import ELECTRODE_RADIUS

logger = logging.getLogger(__name__)


def _get_subject_dir(head_model_dir: str) -> Path:
    """获取 subject 目录的完整路径。"""
    path_obj = Path(head_model_dir)
    if ".." in path_obj.parts:
        raise ValueError(f"非法 head_model_dir: {head_model_dir}")
    return path_obj


def _get_t1_and_dti_path(data: dict) -> tuple[str, str | None]:
    """获取 T1 和 DTI 文件路径。"""
    subject_dir = _get_subject_dir(data["head_model_dir"])
    if subject_dir.name == "m2m_ernie":
        t1_file_path = str(subject_dir / "T1.nii.gz")
        dti_file_path = str(subject_dir / "DTI_coregT1_tensor.nii.gz")
    else:
        t1_file_path = data["T1_file_path"]
        dti_file_path = data.get("DTI_file_path")
    return t1_file_path, dti_file_path


def dict_to_model_params(data: dict) -> ModelParams:
    """
    将字典转换为 ModelParams。

    Parameters
    ----------
    data : dict
        包含 T1_file_path、head_model_dir、T2_file_path 的字典

    Returns
    -------
    ModelParams
        模型参数据类实例
    """
    return ModelParams(
        head_model_dir=data["head_model_dir"],
        T1_file_path=data["T1_file_path"],
        T2_file_path=data.get("T2_file_path"),
    )


def dict_to_forward_params(data: dict) -> ForwardParams:
    """
    将字典转换为 ForwardParams。

    Parameters
    ----------
    data : dict
        包含 head_model_dir、T1_file_path、montage、electrode_A、electrode_B、
        conductivity_config、anisotropy、电极半径、DTI_file_path 的字典

    Returns
    -------
    ForwardParams
        正向仿真参数据类实例
    """
    electrode_a = [
        ElectrodeWithCurrent(name=e["name"], current_mA=e["current_mA"])
        for e in data["electrode_A"]
    ]
    electrode_b = [
        ElectrodeWithCurrent(name=e["name"], current_mA=e["current_mA"])
        for e in data["electrode_B"]
    ]

    t1_file_path, dti_file_path = _get_t1_and_dti_path(data)

    return ForwardParams(
        head_model_dir=data["head_model_dir"],
        T1_file_path=t1_file_path,
        montage=data["montage"],
        electrode_A=electrode_a,
        electrode_B=electrode_b,
        conductivity_config=data["conductivity_config"],
        anisotropy=data["anisotropy"],
        electrode_radius=data.get("electrode_radius", ELECTRODE_RADIUS),
        DTI_file_path=dti_file_path,
    )


def dict_to_inverse_params(data: dict) -> InverseParams:
    """
    将字典转换为 InverseParams。

    Parameters
    ----------
    data : dict
        包含 head_model_dir、T1_file_path、montage、current_A、current_B、
        roi_type、roi_param、target_threshold、conductivity_config、
        anisotropy、电极半径、DTI_file_path 的字典

    Returns
    -------
    InverseParams
        逆向计算参数据类实例
    """
    roi_param_data = data["roi_param"]
    roi_param = ROIParam()
    if roi_param_data.get("mni_param"):
        roi_param.mni_param = MNIParam(
            center=roi_param_data["mni_param"]["center"],
            radius=roi_param_data["mni_param"]["radius"],
        )
    if roi_param_data.get("atlas_param"):
        roi_param.atlas_param = AtlasParam(
            name=roi_param_data["atlas_param"]["name"],
            area=roi_param_data["atlas_param"]["area"],
        )

    t1_file_path, dti_file_path = _get_t1_and_dti_path(data)

    return InverseParams(
        head_model_dir=data["head_model_dir"],
        T1_file_path=t1_file_path,
        montage=data["montage"],
        current_A=data["current_A"],
        current_B=data["current_B"],
        roi_type=data["roi_type"],
        roi_param=roi_param,
        target_threshold=data["target_threshold"],
        conductivity_config=data["conductivity_config"],
        anisotropy=data["anisotropy"],
        electrode_radius=data.get("electrode_radius", ELECTRODE_RADIUS),
        DTI_file_path=dti_file_path,
    )
