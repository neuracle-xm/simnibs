"""
原生标准输出捕获模块。

提供文件描述符级别的 stdout/stderr 文件重定向，用于捕获 C/C++ 扩展绕过
Python logging 直接写到标准输出或标准错误的内容。
"""

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NamedTuple

_STDOUT_FD = 1
_STDERR_FD = 2
# fd 重定向是进程级全局状态，禁止嵌套进入，避免互相覆盖保存的原始句柄。
_native_output_redirect_active = False


class _StreamHandlerRedirectState(NamedTuple):
    """
    终端日志处理器临时重定向状态。

    Attributes
    ----------
    handler : logging.StreamHandler
        被临时切换输出流的日志处理器。
    original_stream : Any
        处理器原始输出流。
    redirected_stream : Any
        指向原终端 fd 副本的临时输出流。
    """

    handler: logging.StreamHandler
    original_stream: Any
    redirected_stream: Any


class _FdRedirectState(NamedTuple):
    """
    标准输出直接重定向状态。

    Attributes
    ----------
    stream_fd : int
        被重定向的标准输出文件描述符。
    saved_fd : int
        原始终端文件描述符副本。
    """

    stream_fd: int
    saved_fd: int


class _PythonStreamRedirectState(NamedTuple):
    """
    Python 标准流临时重定向状态。

    Attributes
    ----------
    stdout : Any
        原始 sys.stdout。
    stderr : Any
        原始 sys.stderr。
    redirected_stdout : Any
        指向日志文件的临时 stdout。
    redirected_stderr : Any
        指向日志文件的临时 stderr。
    """

    stdout: Any
    stderr: Any
    redirected_stdout: Any
    redirected_stderr: Any


def _flush_standard_streams() -> None:
    """
    刷新 Python 标准输出对象。

    Returns
    -------
    None
        无返回值。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except OSError:
            pass


def _start_fd_redirect(stream_fd: int, target_fd: int) -> _FdRedirectState | None:
    """
    将标准输出文件描述符直接重定向到目标文件描述符。

    Parameters
    ----------
    stream_fd : int
        需要重定向的文件描述符，例如 stdout=1 或 stderr=2。
    target_fd : int
        目标文件描述符。

    Returns
    -------
    _FdRedirectState | None
        重定向状态；当前环境不支持 fd 重定向时返回 None。
    """
    try:
        saved_fd = os.dup(stream_fd)
    except OSError:
        return None
    try:
        # 直接模式不经过 pipe，避免部分 ITK/C++ 输出在 Windows pipe 下阻塞。
        os.dup2(target_fd, stream_fd)
    except OSError:
        try:
            os.close(saved_fd)
        except OSError:
            pass
        return None
    return _FdRedirectState(stream_fd=stream_fd, saved_fd=saved_fd)


def _stop_fd_redirect(state: _FdRedirectState) -> None:
    """
    恢复被直接重定向的标准输出文件描述符。

    Parameters
    ----------
    state : _FdRedirectState
        `_start_fd_redirect()` 返回的重定向状态。

    Returns
    -------
    None
        无返回值。
    """
    try:
        os.dup2(state.saved_fd, state.stream_fd)
    except OSError:
        pass
    try:
        os.close(state.saved_fd)
    except OSError:
        pass


def _open_text_stream_from_fd(fd: int, stream: Any) -> Any:
    """
    基于指定 fd 副本创建文本输出流。

    Parameters
    ----------
    fd : int
        目标文件描述符。
    stream : Any
        原始 Python 文本流，用于复用编码和错误处理策略。

    Returns
    -------
    Any
        指向目标 fd 副本的新文本输出流。
    """
    encoding = getattr(stream, "encoding", None) or "utf-8"
    errors = getattr(stream, "errors", None) or "replace"
    return os.fdopen(
        os.dup(fd),
        "w",
        buffering=1,
        encoding=encoding,
        errors=errors,
    )


def _redirect_python_standard_streams(
    target_fd: int,
) -> _PythonStreamRedirectState | None:
    """
    将 Python 层 sys.stdout/sys.stderr 临时指向日志文件。

    fd 级别重定向可以捕获 C/C++ 输出，但 Windows 交互式终端中的
    sys.stdout/sys.stderr 可能仍持有旧控制台句柄，普通 print 会触发
    WinError 6。这里同步替换 Python 标准流，避免 print 写到失效句柄。

    Parameters
    ----------
    target_fd : int
        日志文件描述符。

    Returns
    -------
    _PythonStreamRedirectState | None
        重定向状态；创建文本流失败时返回 None。
    """
    redirected_stdout = None
    redirected_stderr = None
    try:
        redirected_stdout = _open_text_stream_from_fd(target_fd, sys.stdout)
        redirected_stderr = _open_text_stream_from_fd(target_fd, sys.stderr)
    except OSError:
        # 任一标准流创建失败都回收已创建的流，避免留下孤立文件句柄。
        for stream in (redirected_stdout, redirected_stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass
        return None
    state = _PythonStreamRedirectState(
        stdout=sys.stdout,
        stderr=sys.stderr,
        redirected_stdout=redirected_stdout,
        redirected_stderr=redirected_stderr,
    )
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr
    return state


def _restore_python_standard_streams(
    state: _PythonStreamRedirectState | None,
) -> None:
    """
    恢复 Python 层 sys.stdout/sys.stderr。

    Parameters
    ----------
    state : _PythonStreamRedirectState | None
        `_redirect_python_standard_streams()` 返回的状态。

    Returns
    -------
    None
        无返回值。
    """
    if state is None:
        return
    # 先恢复 sys 对象，再关闭临时流，避免后续 print/logging 写入已关闭对象。
    sys.stdout = state.stdout
    sys.stderr = state.stderr
    for stream in (state.redirected_stdout, state.redirected_stderr):
        try:
            stream.close()
        except OSError:
            pass


def _redirect_logger_stream_handlers_from_fd_map(
    logger_name: str | tuple[str, ...] | list[str],
    saved_fd_map: dict[int, int],
) -> list[_StreamHandlerRedirectState]:
    """
    根据标准流 fd 映射重定向指定 logger 的终端处理器。

    Parameters
    ----------
    logger_name : str | tuple[str, ...] | list[str]
        需要处理终端 handler 的 logger 名称。
    saved_fd_map : dict[int, int]
        标准流到原终端 fd 的映射。

    Returns
    -------
    list[_StreamHandlerRedirectState]
        需要在退出时恢复的处理器状态。
    """
    redirected_states: list[_StreamHandlerRedirectState] = []
    for name in _normalize_logger_names(logger_name):
        redirected_states.extend(
            _redirect_single_logger_stream_handlers_from_fd_map(name, saved_fd_map)
        )
    return redirected_states


def _redirect_single_logger_stream_handlers_from_fd_map(
    logger_name: str,
    saved_fd_map: dict[int, int],
) -> list[_StreamHandlerRedirectState]:
    """
    根据标准流 fd 映射重定向单个 logger 的终端处理器。

    Parameters
    ----------
    logger_name : str
        需要处理终端 handler 的 logger 名称。
    saved_fd_map : dict[int, int]
        标准流到原终端 fd 的映射。

    Returns
    -------
    list[_StreamHandlerRedirectState]
        需要在退出时恢复的处理器状态。
    """
    redirected_states: list[_StreamHandlerRedirectState] = []
    target_logger = logging.getLogger(logger_name)
    for handler in target_logger.handlers:
        # 文件 handler 本来就写文件，只有控制台 StreamHandler 需要避开被替换的 stderr。
        if isinstance(handler, logging.FileHandler):
            continue
        if not isinstance(handler, logging.StreamHandler):
            continue
        stream = handler.stream
        try:
            stream_fd = stream.fileno()
        except (AttributeError, OSError):
            continue
        saved_fd = saved_fd_map.get(stream_fd)
        if saved_fd is None:
            continue
        # 控制台 handler 写原终端副本，Python 日志不会被 fd 重定向重复写入目标文件。
        redirected_stream = _open_text_stream_from_fd(saved_fd, stream)
        handler.setStream(redirected_stream)
        redirected_states.append(
            _StreamHandlerRedirectState(
                handler=handler,
                original_stream=stream,
                redirected_stream=redirected_stream,
            )
        )
    return redirected_states


def _normalize_logger_names(
    logger_name: str | tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """
    将单个或多个 logger 名称统一为元组。

    Parameters
    ----------
    logger_name : str | tuple[str, ...] | list[str]
        logger 名称或名称列表。

    Returns
    -------
    tuple[str, ...]
        规范化后的 logger 名称。
    """
    if isinstance(logger_name, str):
        return (logger_name,)
    return tuple(logger_name)


def _restore_logger_stream_handlers(
    states: list[_StreamHandlerRedirectState],
) -> None:
    """
    恢复被临时重定向的终端日志处理器。

    Parameters
    ----------
    states : list[_StreamHandlerRedirectState]
        `_redirect_logger_stream_handlers_from_fd_map()` 返回的处理器状态。

    Returns
    -------
    None
        无返回值。
    """
    for state in reversed(states):
        state.handler.setStream(state.original_stream)
        try:
            state.redirected_stream.close()
        except OSError:
            pass


@contextmanager
def native_output_file_logging(
    handler: logging.FileHandler | None,
    logger_name: str | tuple[str, ...] | list[str] | None = None,
) -> Generator[None, None, None]:
    """
    捕获底层 C/C++ 标准输出并写入文件日志处理器。

    Parameters
    ----------
    handler : logging.FileHandler | None
        目标文件日志处理器，用于复用同一个文件描述符和锁。
    logger_name : str | tuple[str, ...] | list[str] | None, optional
        需要临时重定向终端 handler 的 logger 名称。传入后可避免这些 logger 的
        Python 控制台日志在直接重定向期间写入被替换的 stderr。

    Yields
    ------
    None
        不产生上下文对象，只负责进入和退出时管理标准输出重定向。
    """
    global _native_output_redirect_active
    if _native_output_redirect_active or handler is None or handler.stream is None:
        yield
        return
    # 进入 fd 级重定向前必须刷新，避免旧缓冲内容被写入新目标。
    _flush_standard_streams()
    redirect_states: list[_FdRedirectState] = []
    redirected_handlers: list[_StreamHandlerRedirectState] = []
    redirected_python_streams: _PythonStreamRedirectState | None = None
    _native_output_redirect_active = True
    try:
        try:
            # 直接模式用于长时间 CHARM 步骤：底层输出只落盘，避免 pipe 卡死。
            for stream_fd in (_STDOUT_FD, _STDERR_FD):
                state = _start_fd_redirect(stream_fd, handler.stream.fileno())
                if state is not None:
                    redirect_states.append(state)
            if logger_name is not None:
                redirected_handlers = _redirect_logger_stream_handlers_from_fd_map(
                    logger_name,
                    {state.stream_fd: state.saved_fd for state in redirect_states},
                )
            redirected_python_streams = _redirect_python_standard_streams(
                handler.stream.fileno()
            )
            yield
        finally:
            _flush_standard_streams()
            # 退出顺序很关键：先恢复 Python stream，再恢复 fd，最后恢复 logger handler。
            _restore_python_standard_streams(redirected_python_streams)
            handler.flush()
            for state in reversed(redirect_states):
                _stop_fd_redirect(state)
            _restore_logger_stream_handlers(redirected_handlers)
    finally:
        _native_output_redirect_active = False
