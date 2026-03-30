"""
RabbitMQ 消息构建辅助函数

提供便捷的消息构建接口，用于构造符合规范的 JSON 消息。
"""

from typing import Literal


def build_model_message(
    id: str,
    dir_path: str,
    T1_file_path: str,
    T2_file_path: str | None = None,
    DTI_file_path: str | None = None,
) -> dict:
    """
    构建头模生成消息

    Args:
        id: 任务唯一标识符
        dir_path: 头模生成输出目录路径
        T1_file_path: T1 加权 MRI 图像路径
        T2_file_path: T2 加权 MRI 图像路径（可选）
        DTI_file_path: DTI 扩散张量图像路径（可选）

    Returns:
        dict: 符合规范的 JSON 消息字典
    """
    message = {
        "id": id,
        "type": "model",
        "params": {
            "dir_path": dir_path,
            "T1_file_path": T1_file_path,
        },
    }
    if T2_file_path:
        message["params"]["T2_file_path"] = T2_file_path
    if DTI_file_path:
        message["params"]["DTI_file_path"] = DTI_file_path
    return message


def build_forward_message(
    id: str,
    dir_path: str,
    msh_file_path: str,
    montage: str,
    electrode_A: list[str],
    electrode_B: list[str],
    current_A: list[float],
    current_B: list[float],
    cond: dict[str, float],
    anisotropy: bool,
    DTI_file_path: str | None = None,
) -> dict:
    """
    构建正向仿真消息

    Args:
        id: 任务唯一标识符
        dir_path: 头模所在文件夹路径
        msh_file_path: 头模网格文件路径
        montage: 电极导联配置文件路径
        electrode_A: 电极组 A 名称列表
        electrode_B: 电极组 B 名称列表
        current_A: 电极组 A 电流值列表
        current_B: 电极组 B 电流值列表
        cond: 组织电导率配置
        anisotropy: 是否启用各向异性电导率
        DTI_file_path: DTI 张量文件路径（可选）

    Returns:
        dict: 符合规范的 JSON 消息字典
    """
    message = {
        "id": id,
        "type": "forward",
        "params": {
            "dir_path": dir_path,
            "msh_file_path": msh_file_path,
            "montage": montage,
            "electrode_A": electrode_A,
            "electrode_B": electrode_B,
            "current_A": current_A,
            "current_B": current_B,
            "cond": cond,
            "anisotropy": anisotropy,
        },
    }
    if DTI_file_path:
        message["params"]["DTI_file_path"] = DTI_file_path
    return message


def build_inverse_message(
    id: str,
    dir_path: str,
    msh_file_path: str,
    montage: str,
    current_A: list[float],
    current_B: list[float],
    roi_type: Literal["atlas", "mni_pos"],
    roi_param: dict,
    target_threshold: float,
    cond: dict[str, float],
    anisotropy: bool,
    DTI_file_path: str | None = None,
) -> dict:
    """
    构建逆向仿真消息

    Args:
        id: 任务唯一标识符
        dir_path: 头模所在文件夹路径
        msh_file_path: 头模网格文件路径
        montage: 电极导联配置文件路径
        current_A: 电极组 A 电流值列表
        current_B: 电极组 B 电流值列表
        roi_type: ROI 类型（"atlas" 或 "mni_pos"）
        roi_param: ROI 参数配置
        target_threshold: 目标电场强度阈值
        cond: 组织电导率配置
        anisotropy: 是否启用各向异性电导率
        DTI_file_path: DTI 张量文件路径（可选）

    Returns:
        dict: 符合规范的 JSON 消息字典
    """
    message = {
        "id": id,
        "type": "inverse",
        "params": {
            "dir_path": dir_path,
            "msh_file_path": msh_file_path,
            "montage": montage,
            "current_A": current_A,
            "current_B": current_B,
            "roi_type": roi_type,
            "roi_param": roi_param,
            "target_threshold": target_threshold,
            "cond": cond,
            "anisotropy": anisotropy,
        },
    }
    if DTI_file_path:
        message["params"]["DTI_file_path"] = DTI_file_path
    return message


def build_progress_message(
    id: str,
    msg_type: Literal["model", "forward", "inverse", "ack_test"],
    progress_rate: int,
    message: str | None = None,
    result: dict | None = None,
) -> dict:
    """
    构建返回消息

    Args:
        id: 对应任务的 ID
        msg_type: 任务类型
        progress_rate: 进度百分比 (0-100)
        message: 错误消息或状态描述
        result: 完成时返回的结果

    Returns:
        dict: 符合规范的 JSON 消息字典
    """
    progress_msg = {
        "id": id,
        "type": msg_type,
        "progress_rate": progress_rate,
        "message": message,
        "result": result,
    }
    return progress_msg


def build_ack_test_message(id: str, sleep_seconds: float = 30.0) -> dict:
    """
    构建 ack 时机验证消息

    Args:
        id: 任务唯一标识符
        sleep_seconds: 模拟长任务持续时间

    Returns:
        dict: 符合规范的 JSON 消息字典
    """
    return {
        "id": id,
        "type": "ack_test",
        "params": {
            "sleep_seconds": sleep_seconds,
        },
    }
