"""
命令行参数解析工具

为 neuracle 的终端入口脚本提供公共的参数解析与组装能力。
"""

import argparse
from typing import Any

from neuracle.parameters.schemas import AnisotropyType
from neuracle.utils import STANDARD_COND


def parse_electrode_specs(specs: list[str]) -> list[dict[str, float]]:
    """
    解析电极参数列表。

    Parameters
    ----------
    specs : list[str]
        终端传入的电极字符串列表，格式为 ``NAME:CURRENT``

    Returns
    -------
    list[dict[str, float]]
        电极参数字典列表
    """
    electrodes = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"电极参数格式错误: {spec}，应为 NAME:CURRENT")
        name, current_str = spec.split(":", 1)
        name = name.strip()
        current_str = current_str.strip()
        if not name:
            raise ValueError(f"电极名称不能为空: {spec}")
        try:
            current_mA = float(current_str)
        except ValueError as exc:
            raise ValueError(f"电极电流不是数字: {spec}") from exc
        electrodes.append({"name": name, "current_mA": current_mA})
    return electrodes


def parse_conductivity_specs(specs: list[str] | None) -> dict[str, float]:
    """
    解析电导率参数。

    Parameters
    ----------
    specs : list[str] | None
        终端传入的电导率字符串列表，格式为 ``TISSUE=VALUE``

    Returns
    -------
    dict[str, float]
        组织电导率配置
    """
    conductivity_config = dict(STANDARD_COND)
    if not specs:
        return conductivity_config
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"电导率参数格式错误: {spec}，应为 TISSUE=VALUE")
        tissue, value_str = spec.split("=", 1)
        tissue = tissue.strip()
        value_str = value_str.strip()
        if tissue not in STANDARD_COND:
            valid_tissues = ", ".join(STANDARD_COND.keys())
            raise ValueError(f"未知组织名称: {tissue}，可选值: {valid_tissues}")
        try:
            conductivity_config[tissue] = float(value_str)
        except ValueError as exc:
            raise ValueError(f"电导率不是数字: {spec}") from exc
    return conductivity_config


def parse_anisotropy(value: str) -> AnisotropyType:
    """
    解析各向异性类型。

    Parameters
    ----------
    value : str
        终端传入的各向异性类型字符串

    Returns
    -------
    AnisotropyType
        各向异性枚举值
    """
    try:
        return AnisotropyType(value)
    except ValueError as exc:
        valid_values = ", ".join(item.value for item in AnisotropyType)
        raise ValueError(f"anisotropy 非法: {value}，可选值: {valid_values}") from exc


def build_mni_roi_param(center: list[float], radius: float) -> dict[str, Any]:
    """
    构造 MNI ROI 参数字典。

    Parameters
    ----------
    center : list[float]
        MNI 中心坐标
    radius : float
        ROI 半径

    Returns
    -------
    dict[str, Any]
        逆向仿真使用的 ROI 参数字典
    """
    return {"mni_param": {"center": center, "radius": radius}}


def build_atlas_roi_param(name: str, area: str) -> dict[str, Any]:
    """
    构造 atlas ROI 参数字典。

    Parameters
    ----------
    name : str
        atlas 名称
    area : str
        atlas 区域名称

    Returns
    -------
    dict[str, Any]
        逆向仿真使用的 ROI 参数字典
    """
    return {"atlas_param": {"name": name, "area": area}}


def add_conductivity_argument(parser: argparse.ArgumentParser) -> None:
    """
    为命令行解析器添加电导率参数。

    Parameters
    ----------
    parser : argparse.ArgumentParser
        目标解析器
    """
    parser.add_argument(
        "--conductivity",
        action="append",
        default=None,
        metavar="TISSUE=VALUE",
        help="组织电导率覆盖项，可重复传入；未传时使用内置默认值",
    )
