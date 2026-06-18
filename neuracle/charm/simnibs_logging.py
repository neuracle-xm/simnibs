"""
CHARM 日志辅助模块。

为 Neuracle 拆分后的 CHARM 流程补充 SimNIBS 文件日志。
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from neuracle.logger.formatters import get_formatter

_CHARM_SIMNIBS_HANDLER_ATTR = "_neuracle_charm_simnibs_handler"


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
        SimNIBS 日志文件名，默认写入 `simnibs_charm.log`。

    Returns
    -------
    logging.FileHandler | None
        新增的文件处理器；如果当前已经挂载 CHARM SimNIBS 处理器，则返回 None。
    """
    simnibs_logger = logging.getLogger("simnibs")
    for handler in simnibs_logger.handlers:
        if getattr(handler, _CHARM_SIMNIBS_HANDLER_ATTR, False):
            return None
    log_dir = os.path.join(subject_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(get_formatter())
    setattr(handler, _CHARM_SIMNIBS_HANDLER_ATTR, True)
    simnibs_logger.setLevel(logging.DEBUG)
    simnibs_logger.addHandler(handler)
    return handler


@contextmanager
def simnibs_charm_file_logging(
    subject_dir: str, filename: str = "simnibs_charm.log"
) -> Iterator[None]:
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
    try:
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
    simnibs_logger = logging.getLogger("simnibs")
    simnibs_logger.removeHandler(handler)
    handler.close()
