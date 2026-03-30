"""
RabbitMQ 模块

提供 SimNIBS 与 RabbitMQ 集成的功能。
"""

from .listener import RabbitMQListener
from .message_builder import (
    build_ack_test_message,
    build_forward_message,
    build_inverse_message,
    build_model_message,
    build_progress_message,
)
from .schemas import (
    AckTestParams,
    AtlasParam,
    ForwardParams,
    ForwardResult,
    InverseParams,
    InverseResult,
    MNIParam,
    ModelParams,
    ModelResult,
    ReturnMessage,
    ROIParam,
)
from .sender import RabbitMQSender
from .sender_thread import SenderThread
from .validator import (
    ValidationError,
    validate_ack_test_params,
    validate_forward_params,
    validate_inverse_params,
    validate_message,
    validate_model_params,
)

__all__ = [
    "RabbitMQListener",
    "RabbitMQSender",
    "SenderThread",
    # schemas
    "AtlasParam",
    "AckTestParams",
    "MNIParam",
    "ROIParam",
    "ModelParams",
    "ForwardParams",
    "InverseParams",
    "ModelResult",
    "ForwardResult",
    "InverseResult",
    "ReturnMessage",
    # validator
    "ValidationError",
    "validate_model_params",
    "validate_forward_params",
    "validate_inverse_params",
    "validate_ack_test_params",
    "validate_message",
    # message_builder
    "build_model_message",
    "build_forward_message",
    "build_inverse_message",
    "build_ack_test_message",
    "build_progress_message",
]
