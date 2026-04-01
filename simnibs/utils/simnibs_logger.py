import io
import logging
import re
import sys
import warnings
import numpy as np


class CatProgressFilter(logging.Filter):
    """过滤 CAT12 进度条噪声"""

    # 匹配 CAT12 进度条模式
    SPINNER_PATTERN = re.compile(r"[-\\|/][-\\|/]")
    SELECTING_PATTERN = re.compile(r"Selecting intersections.*?\d{1,3}\s*%")
    LOADING_PERCENT_PATTERN = re.compile(r"Loading\s+\.\.\d+%")
    PERCENTAGE_PATTERN = re.compile(r"\d{1,3}\s*%")

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤掉 CAT12 进度条噪声"""
        msg = record.getMessage()
        # 过滤旋转进度条（如 /-\|/-\）
        if self.SPINNER_PATTERN.search(msg):
            return False
        # 过滤 "Selecting intersections ... XX%" 模式
        if self.SELECTING_PATTERN.search(msg):
            return False
        # 过滤 "Loading ..X%Loading ..Y%..." 模式（meshfix 进度条）
        if self.LOADING_PERCENT_PATTERN.search(msg):
            return False
        # 过滤单独的数字百分比（通常是进度更新）
        if self.PERCENTAGE_PATTERN.fullmatch(msg.strip()):
            return False
        return True


# 在模块加载时立即重置 stderr 编码，避免 Windows GBK 问题
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
elif hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


class Utf8StreamHandler(logging.StreamHandler):
    """支持 UTF-8 编码的 StreamHandler，避免 Windows GBK 编码问题"""

    def __init__(self) -> None:
        # 确保在调用 super().__init__() 之前 stderr 是 UTF-8 编码
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        # 调用父类构造函数，父类会设置 self.stream = sys.stderr
        super().__init__()
        # 确保 self.stream 也是 UTF-8 编码
        if hasattr(self.stream, "reconfigure"):
            try:
                self.stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    def stop(self) -> None:
        pass


global logger
logger = logging.getLogger('simnibs')
logger.propagate = False  # 防止消息传播到根 logger，避免被其他 logger 的 handlers 捕获
sh = Utf8StreamHandler()
formatter = logging.Formatter('[ %(name)s ] %(levelname)s: %(message)s')
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
sh.addFilter(CatProgressFilter())
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)
logging.addLevelName(25, 'SUMMARY')
logging.addLevelName(26, 'SUMMARY')

def log_warnings(message, category, filename, lineno, file=None, line=None):
    logger.warning(warnings.formatwarning(message, category, filename, lineno))


# This is causing errors in pytest
#warnings.showwarning = log_warnings


def register_excepthook(logger):
    def log_excep(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.debug(
            "Traceback",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, None)
        )
    sys.excepthook = log_excep


def unregister_excepthook():
    sys.excepthook = sys.__excepthook__


def format_time(running_time):
    """Format time in seconds as hours:minutes:seconds.
    
    PARAMETERS
    ----------
    running_time : float
        Time in seconds.
    
    RETURNS
    ----------
    running_time : str
        The time formatted as hours:minutes:seconds.
    """
    hrs = np.uint16(np.floor(running_time/(60.**2)))
    mts = np.uint16(np.floor(running_time/60.-hrs*60))
    sec = np.uint16(np.round(running_time-hrs*60.**2-mts*60.))

    return "{:02d}:{:02d}:{:02d}".format(hrs,mts,sec)