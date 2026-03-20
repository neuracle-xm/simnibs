"""
Neuracle 模块日志配置

提供日志记录器配置函数，支持多级别文件日志输出和自动回滚。

使用方式：
    import logging
    from neuracle.logger import setup_logging
    setup_logging()  # 使用默认路径

    # 或指定日志目录
    setup_logging(log_dir='/path/to/logs')

    # 之后在任何地方使用标准 logging
    logger = logging.getLogger('neuracle')
    logger.info("message")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Final, Optional

# 日志文件配置
_MAX_BYTES: Final[int] = 1 * 1024 * 1024  # 1MB
_BACKUP_COUNT: Final[int] = 20

# 日志格式
_LOG_FORMAT: Final[str] = '%(asctime)s - %(processName)s - %(threadName)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
_DATE_FORMAT: Final[str] = '%Y-%m-%d %H:%M:%S'


def _create_level_handler(level: int, filename: str, log_dir: str) -> RotatingFileHandler:
    """
    创建指定级别的文件处理器

    原理：
        使用 RotatingFileHandler 实现日志文件自动回滚。当日志文件达到
        指定大小时，会自动关闭当前文件，并将其重命名为带数字后缀的备份文件，
        然后创建一个新的日志文件继续写入。备份文件数量超过限制时，最旧的
        备份文件会被删除。

    Args:
        level: 日志级别（logging.DEBUG, logging.INFO 等）
        filename: 日志文件名
        log_dir: 日志目录路径

    Returns:
        RotatingFileHandler: 配置好的文件处理器
    """
    log_path = os.path.join(log_dir, filename)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding='utf-8'
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    return handler


def _create_console_handler() -> logging.StreamHandler:
    """
    创建控制台处理器

    原理：
        使用 StreamHandler 输出到标准错误流（stderr），支持彩色输出（在某些终端中）。

    Returns:
        logging.StreamHandler: 配置好的控制台处理器
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    return console_handler


def setup_logging(log_dir: Optional[str] = None) -> None:
    """
    配置 neuracle 日志系统

    原理：
        配置 'neuracle' logger 的处理器。由于 logging 模块的单例机制，
        所有以 'neuracle' 开头的子 logger 都会继承此配置，不会创建重复的处理器。

        支持同时输出到：
        - 控制台（INFO 级别及以上）
        - 文件（debug.log, info.log, warning.log, error.log）

    Args:
        log_dir: 日志目录路径，默认为 neuracle/log/

    用法：
        >>> import logging
        >>> from neuracle.logger import setup_logging
        >>> setup_logging()  # 使用默认路径
        >>>
        >>> # 或指定日志目录
        >>> setup_logging(log_dir='/path/to/logs')
        >>>
        >>> # 在任何模块中使用
        >>> logger = logging.getLogger('neuracle.module')
        >>> logger.info("message")
    """
    # 默认日志目录
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('neuracle')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # 添加控制台处理器（INFO 级别）
    logger.addHandler(_create_console_handler())
    # 添加各级别文件处理器
    logger.addHandler(_create_level_handler(logging.DEBUG, 'debug.log', log_dir))
    logger.addHandler(_create_level_handler(logging.INFO, 'info.log', log_dir))
    logger.addHandler(_create_level_handler(logging.WARNING, 'warning.log', log_dir))
    logger.addHandler(_create_level_handler(logging.ERROR, 'error.log', log_dir))
