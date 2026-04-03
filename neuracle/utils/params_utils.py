"""
参数转换工具模块

将字典转换为对应的 dataclass 参数类型，用于 RabbitMQ 消息解析和参数校验。
"""

import logging

from neuracle.rabbitmq.schemas import (
    AckTestParams,
    AtlasParam,
    ElectrodeWithCurrent,
    ForwardParams,
    InverseParams,
    MNIParam,
    ModelParams,
    ROIParam,
)

logger = logging.getLogger(__name__)


def dict_to_model_params(data: dict, task_id: str) -> ModelParams:
    """
    将字典转换为 ModelParams。

    Parameters
    ----------
    data : dict
        包含 T1_file_path、dir_path、T2_file_path、DTI_file_path 的字典
    task_id : str
        任务 ID，从消息顶层获取

    Returns
    -------
    ModelParams
        模型参数据类实例
    """
    return ModelParams(
        id=task_id,
        T1_file_path=data["T1_file_path"],
        dir_path=data["dir_path"],
        T2_file_path=data.get("T2_file_path"),
        DTI_file_path=data.get("DTI_file_path"),
    )


def dict_to_forward_params(data: dict, task_id: str) -> ForwardParams:
    """
    将字典转换为 ForwardParams。

    Parameters
    ----------
    data : dict
        包含 dir_path、T1_file_path、montage、electrode_A、electrode_B、
        conductivity_config、anisotropy、DTI_file_path 的字典
    task_id : str
        任务 ID，从消息顶层获取

    Returns
    -------
    ForwardParams
        正向仿真参数据类实例
    """
    electrode_A = [
        ElectrodeWithCurrent(name=e["name"], current_mA=e["current_mA"])
        for e in data["electrode_A"]
    ]
    electrode_B = [
        ElectrodeWithCurrent(name=e["name"], current_mA=e["current_mA"])
        for e in data["electrode_B"]
    ]
    return ForwardParams(
        id=task_id,
        dir_path=data["dir_path"],
        T1_file_path=data["T1_file_path"],
        montage=data["montage"],
        electrode_A=electrode_A,
        electrode_B=electrode_B,
        conductivity_config=data["conductivity_config"],
        anisotropy=data["anisotropy"],
        DTI_file_path=data.get("DTI_file_path"),
    )


def dict_to_inverse_params(data: dict, task_id: str) -> InverseParams:
    """
    将字典转换为 InverseParams。

    Parameters
    ----------
    data : dict
        包含 dir_path、T1_file_path、montage、current_A、current_B、
        roi_type、roi_param、target_threshold、conductivity_config、
        anisotropy、DTI_file_path 的字典
    task_id : str
        任务 ID，从消息顶层获取

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

    return InverseParams(
        id=task_id,
        dir_path=data["dir_path"],
        T1_file_path=data["T1_file_path"],
        montage=data["montage"],
        current_A=data["current_A"],
        current_B=data["current_B"],
        roi_type=data["roi_type"],
        roi_param=roi_param,
        target_threshold=data["target_threshold"],
        conductivity_config=data["conductivity_config"],
        anisotropy=data["anisotropy"],
        DTI_file_path=data.get("DTI_file_path"),
    )


def dict_to_ack_test_params(data: dict, task_id: str) -> AckTestParams:
    """
    将字典转换为 AckTestParams。

    Parameters
    ----------
    data : dict
        包含 sleep_seconds 的字典
    task_id : str
        任务 ID，从消息顶层获取

    Returns
    -------
    AckTestParams
        心跳测试参数据类实例
    """
    return AckTestParams(
        id=task_id,
        sleep_seconds=float(data.get("sleep_seconds", 30.0)),
    )
