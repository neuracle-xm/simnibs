"""
RabbitMQ 消息数据结构定义

定义所有消息类型的参数和结果数据类。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AtlasParam:
    """Atlas ROI 参数"""

    name: str
    area: str


@dataclass
class MNIParam:
    """MNI 位置 ROI 参数"""

    center: list[float]
    radius: float


@dataclass
class ROIParam:
    """ROI 参数"""

    atlas_param: AtlasParam | None = None
    mni_param: MNIParam | None = None


@dataclass
class ModelParams:
    """头模生成参数"""

    id: str
    T1_file_path: str
    dir_path: str
    T2_file_path: str | None = None
    DTI_file_path: str | None = None


@dataclass
class ForwardParams:
    """正向仿真参数"""

    id: str
    dir_path: str
    msh_file_path: str
    montage: str
    electrode_A: list[str]
    electrode_B: list[str]
    current_A: list[float]
    current_B: list[float]
    cond: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None


@dataclass
class InverseParams:
    """逆向仿真参数"""

    id: str
    dir_path: str
    msh_file_path: str
    montage: str
    current_A: list[float]
    current_B: list[float]
    roi_type: Literal["atlas", "mni_pos"]
    roi_param: ROIParam
    target_threshold: float
    cond: dict[str, float]
    anisotropy: bool
    electrode_pair1_center: list[list[float]] | None = None
    electrode_pair2_center: list[list[float]] | None = None
    electrode_radius: list[float] | None = None
    electrode_current1: list[float] | None = None
    electrode_current2: list[float] | None = None
    DTI_file_path: str | None = None


@dataclass
class ModelResult:
    """头模生成结果"""

    msh_file_path: str | None = None


@dataclass
class ForwardResult:
    """正向仿真结果"""

    TI_file: str | None = None


@dataclass
class InverseResult:
    """逆向仿真结果"""

    TI_file: str | None = None
    electrode_A: list[str] | None = None
    electrode_B: list[str] | None = None


@dataclass
class ReturnMessage:
    """返回消息"""

    id: str
    type: Literal["model", "forward", "inverse"]
    progress_rate: int
    message: str | None = None
    result: ModelResult | ForwardResult | InverseResult | None = None
