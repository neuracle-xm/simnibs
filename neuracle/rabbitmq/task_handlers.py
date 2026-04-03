"""
任务处理工具函数

提供 RabbitMQ 任务处理相关的辅助函数，包括：
- 目录路径规范化
- ACK 调度
- RabbitMQ 配置掩码
"""

import logging
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 调试模式：True 时跳过 output_dir 删除
DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 内置头模路径
BUILT_IN_DIR_PATH = "m2m_ernie"
BUILT_IN_DTI_FILE_PATH = "DTI_coregT1_tensor.nii.gz"


def normalize_dir_path(dir_path: str) -> str:
    """规范化相对目录路径，禁止绝对路径和父级跳转。

    Parameters
    ----------
    dir_path : str
        原始目录路径

    Returns
    -------
    str
        规范化后的目录路径

    Raises
    ------
    ValueError
        当路径为绝对路径或包含父级跳转 (..) 时抛出
    """
    normalized = dir_path.replace("\\", "/").strip().strip("/")
    path_obj = Path(normalized)
    if path_obj.is_absolute() or ".." in path_obj.parts:
        logger.error("非法路径访问: %s", dir_path)
        raise ValueError(f"非法 dir_path: {dir_path}")
    return normalized


def get_subject_dir(dir_path: str) -> Path:
    """获取 subject 目录的完整路径。

    Parameters
    ----------
    dir_path : str
        subject 目录名（如 m2m_ernie）

    Returns
    -------
    Path
        完整的 subject 目录路径
    """
    return DATA_ROOT / Path(normalize_dir_path(dir_path))


def get_task_output_dir(dir_path: str, suffix: str, task_id: str | None = None) -> Path:
    """获取任务输出目录的完整路径。

    Parameters
    ----------
    dir_path : str
        subject 目录名
    suffix : str
        任务类型后缀（如 TI_simulation, TI_optimization）
    task_id : str, optional
        任务 ID

    Returns
    -------
    Path
        任务输出目录路径
    """
    normalized = normalize_dir_path(dir_path)
    if task_id:
        return DATA_ROOT / f"{normalized}_{suffix}_{task_id}"
    return DATA_ROOT / f"{normalized}_{suffix}"


def get_model_mesh_path(dir_path: str) -> Path:
    """获取头模网格文件路径。

    Parameters
    ----------
    dir_path : str
        subject 目录名

    Returns
    -------
    Path
        model.msh 文件路径
    """
    return get_subject_dir(dir_path) / "model.msh"


def resolve_local_dti_path(dir_path: str, dti_file_path: str | None) -> str | None:
    """解析本地 DTI 文件路径。

    Parameters
    ----------
    dir_path : str
        subject 目录名
    dti_file_path : str, optional
        DTI 文件路径

    Returns
    -------
    str or None
        本地 DTI 文件完整路径，仅对内置头模有效
    """
    if dir_path == BUILT_IN_DIR_PATH:
        dti_file_path = BUILT_IN_DTI_FILE_PATH
        logger.info("使用内置头模，DTI使用内置路径: %s", dti_file_path)
        return str(get_subject_dir(dir_path) / dti_file_path)
    return None


def ensure_data_root() -> None:
    """确保数据根目录存在，不存在则创建。"""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def reset_task_output_dir(output_dir: str) -> None:
    """删除任务输出目录，避免重跑时被残留结果污染。

    Parameters
    ----------
    output_dir : str
        任务输出目录路径
    """
    if os.path.isdir(output_dir):
        logger.info("清理旧任务输出目录: %s", output_dir)
        shutil.rmtree(output_dir)


def _ack_in_consumer_thread(
    channel: Any,
    delivery_tag: int,
    task_id: str,
    msg_type: str,
    reason: str,
) -> None:
    """在 RabbitMQ 消费线程中实际执行 ack。

    Parameters
    ----------
    channel : Any
        RabbitMQ 通道
    delivery_tag : int
        消息 delivery tag
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    reason : str
        ack 原因
    """
    try:
        if channel is None or not getattr(channel, "is_open", False):
            logger.warning(
                "ack_sent 跳过，channel 不可用: task_id=%s type=%s delivery_tag=%s reason=%s",
                task_id,
                msg_type,
                delivery_tag,
                reason,
            )
            return
        channel.basic_ack(delivery_tag=delivery_tag)
        logger.info(
            "ack_sent: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )
    except Exception:
        logger.exception(
            "发送 ack 失败: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )


def schedule_ack(
    channel: Any,
    delivery_tag: int,
    task_id: str,
    msg_type: str,
    reason: str,
) -> None:
    """将 ack 请求从工作线程投递回消费线程执行。

    Parameters
    ----------
    channel : Any
        RabbitMQ 通道
    delivery_tag : int
        消息 delivery tag
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    reason : str
        ack 原因
    """
    connection = getattr(channel, "connection", None)
    if connection is None or not getattr(connection, "is_open", False):
        logger.warning(
            "ack_scheduled 跳过，connection 不可用: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )
        return
    logger.info(
        "ack_scheduled: task_id=%s type=%s delivery_tag=%s reason=%s",
        task_id,
        msg_type,
        delivery_tag,
        reason,
    )
    connection.add_callback_threadsafe(
        partial(
            _ack_in_consumer_thread,
            channel,
            delivery_tag,
            task_id,
            msg_type,
            reason,
        )
    )


def mask_rabbitmq_config(config: dict[str, Any]) -> dict[str, Any]:
    """隐藏 RabbitMQ 配置中的敏感信息，避免密码写入日志。

    Parameters
    ----------
    config : dict[str, Any]
        原始配置字典

    Returns
    -------
    dict[str, Any]
        密码被掩码处理后的配置字典
    """
    masked = dict(config)
    if masked.get("password"):
        masked["password"] = "***"
    return masked
