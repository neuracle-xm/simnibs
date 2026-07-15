"""子进程结构化错误消息文件工具。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from neuracle.utils.constants import ERROR_MESSAGE_FILENAME

logger = logging.getLogger(__name__)


def clear_error_message(log_dir: Path) -> None:
    """清理任务成功前遗留的结构化错误文件。

    Parameters
    ----------
    log_dir : pathlib.Path
        当前任务的日志目录。

    Returns
    -------
    None
    """
    error_message_path = log_dir / ERROR_MESSAGE_FILENAME
    try:
        error_message_path.unlink(missing_ok=True)
    except OSError:
        logger.exception("清理结构化错误文件失败: %s", error_message_path)


def write_error_message(log_dir: Path, return_code: int, message: str) -> None:
    """写入子进程退出码和错误消息。

    Parameters
    ----------
    log_dir : pathlib.Path
        当前任务的日志目录。
    return_code : int
        子进程返回给主进程的整数退出码。
    message : str
        当前异常的文本消息。

    Returns
    -------
    None
    """
    error_message_path = log_dir / ERROR_MESSAGE_FILENAME
    payload = {"return_code": return_code, "message": message}
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        error_message_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("结构化错误文件写入完成: %s", error_message_path)
    except OSError:
        logger.exception("写入结构化错误文件失败: %s", error_message_path)
