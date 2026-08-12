"""数值 warning 的局部异常转换工具。"""

import warnings
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

_DIVIDE_WARNING_MESSAGE = r"(divide by zero|invalid value) encountered in .*divide"
_P = ParamSpec("_P")
_R = TypeVar("_R")


def raise_divide_runtime_warnings(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """
    在函数执行期间将除法相关的 RuntimeWarning 提升为异常。

    原理
    ----
    NumPy 的除零和无效除法默认只发出 ``RuntimeWarning``，计算可能继续并
    产生 ``inf`` 或 ``NaN``。该装饰器使用局部 warning 上下文将目标 warning
    转换为异常，并在函数返回或抛出异常后恢复调用方原有的 warning 配置。

    Parameters
    ----------
    func : Callable[_P, _R]
        需要启用数值 warning 异常转换的函数。

    Returns
    -------
    Callable[_P, _R]
        保留原函数元数据的包装函数。
    """
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """
        在局部 warning 上下文中调用原函数。

        Parameters
        ----------
        *args : _P.args
            原函数的位置参数。
        **kwargs : _P.kwargs
            原函数的关键字参数。

        Returns
        -------
        _R
            原函数返回值。
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=_DIVIDE_WARNING_MESSAGE,
                category=RuntimeWarning,
            )
            return func(*args, **kwargs)
    return wrapper
