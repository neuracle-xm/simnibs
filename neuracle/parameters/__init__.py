"""
参数相关模块

提供参数数据结构定义、转换、验证和进度管理功能。
"""

from neuracle.parameters.converter import (
    dict_to_forward_params,
    dict_to_inverse_params,
    dict_to_model_params,
)
from neuracle.parameters.progress import (
    ModelProgress,
    load_progress,
    save_progress,
)
from neuracle.parameters.schemas import (
    AnisotropyType,
    AtlasParam,
    ElectrodeWithCurrent,
    ForwardParams,
    InverseParams,
    MNIParam,
    ModelParams,
    ROIParam,
)
from neuracle.parameters.validator import (
    ValidationError,
    validate_forward_params,
    validate_inverse_params,
    validate_model_params,
)

__all__ = [
    "AnisotropyType",
    "AtlasParam",
    "ElectrodeWithCurrent",
    "ForwardParams",
    "InverseParams",
    "MNIParam",
    "ModelParams",
    "ROIParam",
    "ModelProgress",
    "ValidationError",
    "dict_to_model_params",
    "dict_to_forward_params",
    "dict_to_inverse_params",
    "validate_model_params",
    "validate_forward_params",
    "validate_inverse_params",
    "load_progress",
    "save_progress",
]
