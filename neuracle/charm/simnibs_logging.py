"""
CHARM 日志辅助模块。

为 Neuracle 拆分后的 CHARM 流程补充 SimNIBS 文件日志。
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from neuracle.logger.formatters import get_formatter
from neuracle.logger.native_output import native_output_file_logging

_CHARM_SIMNIBS_HANDLER_ATTR = "_neuracle_charm_simnibs_handler"
_CHARM_LOGGER_NAMES = ("simnibs", "neuracle")


def attach_simnibs_charm_file_logger(
    subject_dir: str, filename: str = "simnibs_charm.log"
) -> logging.FileHandler | None:
    """
    为 CHARM 流程挂载 SimNIBS 文件日志处理器。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})。
    filename : str, optional
        SimNIBS 和 Neuracle 日志文件名，默认写入 `simnibs_charm.log`。

    Returns
    -------
    logging.FileHandler | None
        新增的文件处理器；如果当前已经挂载 CHARM 处理器，则返回 None。
    """
    primary_logger = logging.getLogger(_CHARM_LOGGER_NAMES[0])
    for handler in primary_logger.handlers:
        if getattr(handler, _CHARM_SIMNIBS_HANDLER_ATTR, False):
            return None
    log_path = _get_charm_log_path(subject_dir, filename)
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    # 先清空旧文件，再用追加模式写入，避免 Python 日志和底层输出互相覆盖。
    with open(log_path, "w", encoding="utf-8"):
        pass
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(get_formatter())
    setattr(handler, _CHARM_SIMNIBS_HANDLER_ATTR, True)
    for logger_name in _CHARM_LOGGER_NAMES:
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(logging.DEBUG)
        target_logger.addHandler(handler)
    return handler


@contextmanager
def simnibs_charm_file_logging(
    subject_dir: str,
    filename: str = "simnibs_charm.log",
) -> Generator[None, None, None]:
    """
    在代码块执行期间记录 CHARM 相关 SimNIBS 文件日志。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})。
    filename : str, optional
        SimNIBS 日志文件名，默认写入 `simnibs_charm.log`。

    Yields
    ------
    None
        不产生上下文对象，只负责进入和退出时管理日志处理器。
    """
    handler = attach_simnibs_charm_file_logger(subject_dir, filename=filename)
    active_handler = handler or _find_simnibs_charm_file_logger()
    try:
        with native_output_file_logging(
            active_handler,
            logger_name=_CHARM_LOGGER_NAMES,
        ):
            yield
    finally:
        detach_simnibs_charm_file_logger(handler)


def detach_simnibs_charm_file_logger(
    handler: logging.FileHandler | None,
) -> None:
    """
    移除 CHARM 流程临时挂载的 SimNIBS 文件日志处理器。

    Parameters
    ----------
    handler : logging.FileHandler | None
        需要移除并关闭的文件处理器；None 表示当前调用没有新增处理器。

    Returns
    -------
    None
        无返回值。
    """
    if handler is None:
        return
    for logger_name in _CHARM_LOGGER_NAMES:
        target_logger = logging.getLogger(logger_name)
        if handler in target_logger.handlers:
            target_logger.removeHandler(handler)
    handler.close()


def _find_simnibs_charm_file_logger() -> logging.FileHandler | None:
    """
    查找当前已经挂载的 CHARM SimNIBS 文件日志处理器。

    Returns
    -------
    logging.FileHandler | None
        已挂载的 CHARM 文件处理器；不存在时返回 None。
    """
    primary_logger = logging.getLogger(_CHARM_LOGGER_NAMES[0])
    for handler in primary_logger.handlers:
        if getattr(handler, _CHARM_SIMNIBS_HANDLER_ATTR, False):
            return handler
    return None


def _get_charm_log_path(subject_dir: str, filename: str) -> str:
    """
    生成 CHARM SimNIBS 日志文件路径。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})。
    filename : str
        SimNIBS 日志文件名。

    Returns
    -------
    str
        位于受试者 `logs` 子目录下的日志文件路径。
    """
    return os.path.join(subject_dir, "logs", filename)
