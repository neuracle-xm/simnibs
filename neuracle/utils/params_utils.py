"""
参数转换工具模块

将字典转换为对应的 dataclass 参数类型。
"""

from neuracle.rabbitmq.schemas import (
    AckTestParams,
    AtlasParam,
    ForwardParams,
    InverseParams,
    MNIParam,
    ModelParams,
    ROIParam,
)


def dict_to_model_params(data: dict, task_id: str) -> ModelParams:
    """将字典转换为 ModelParams（不包含 id，id 在消息顶层）"""
    return ModelParams(
        id=task_id,
        T1_file_path=data["T1_file_path"],
        dir_path=data["dir_path"],
        T2_file_path=data.get("T2_file_path"),
        DTI_file_path=data.get("DTI_file_path"),
    )


def dict_to_forward_params(data: dict, task_id: str) -> ForwardParams:
    """将字典转换为 ForwardParams（不包含 id，id 在消息顶层）"""
    return ForwardParams(
        id=task_id,
        dir_path=data["dir_path"],
        montage=data["montage"],
        electrode_A=data["electrode_A"],
        electrode_B=data["electrode_B"],
        current_A=data["current_A"],
        current_B=data["current_B"],
        cond=data["cond"],
        anisotropy=data["anisotropy"],
        DTI_file_path=data.get("DTI_file_path"),
    )


def dict_to_inverse_params(data: dict, task_id: str) -> InverseParams:
    """将字典转换为 InverseParams（不包含 id，id 在消息顶层）"""
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
        montage=data["montage"],
        current_A=data["current_A"],
        current_B=data["current_B"],
        roi_type=data["roi_type"],
        roi_param=roi_param,
        target_threshold=data["target_threshold"],
        cond=data["cond"],
        anisotropy=data["anisotropy"],
        electrode_pair1_center=data.get("electrode_pair1_center"),
        electrode_pair2_center=data.get("electrode_pair2_center"),
        electrode_radius=data.get("electrode_radius"),
        electrode_current1=data.get("electrode_current1"),
        electrode_current2=data.get("electrode_current2"),
        DTI_file_path=data.get("DTI_file_path"),
    )


def dict_to_ack_test_params(data: dict, task_id: str) -> AckTestParams:
    """将字典转换为 AckTestParams（不包含 id，id 在消息顶层）"""
    return AckTestParams(
        id=task_id,
        sleep_seconds=float(data.get("sleep_seconds", 30.0)),
    )
