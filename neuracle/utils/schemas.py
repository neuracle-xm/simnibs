"""
RabbitMQ 消息数据结构定义

定义所有消息类型的参数和结果数据类。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AtlasParam:
    """Atlas ROI 参数

    Attributes
    ----------
    name : str
        Atlas 名称，如 "BN_Atlas_246_1mm"
    area : str
        脑区名称，如 "Precentral Gyrus"
    """

    name: str
    area: str


@dataclass
class MNIParam:
    """MNI 位置 ROI 参数

    Attributes
    ----------
    center : list[float]
        MNI 坐标中心点 [x, y, z]
    radius : float
        球形 ROI 半径（mm）
    """

    center: list[float]
    radius: float


@dataclass
class ROIParam:
    """ROI 参数，支持 Atlas 或 MNI 两种定义方式

    Attributes
    ----------
    atlas_param : AtlasParam | None
        Atlas 定义的 ROI
    mni_param : MNIParam | None
        MNI 球形 ROI
    """

    atlas_param: AtlasParam | None = None
    mni_param: MNIParam | None = None


@dataclass
class ModelParams:
    """头模生成参数

    Attributes
    ----------
    id : str
        任务 ID
    T1_file_path : str
        T1 加权 MRI 文件路径
    dir_path : str
        subject 目录路径
    T2_file_path : str | None
        T2 加权 MRI 文件路径（可选）
    DTI_file_path : str | None
        DTI tensor 文件路径（可选）
    """

    id: str
    T1_file_path: str
    dir_path: str
    T2_file_path: str | None = None
    DTI_file_path: str | None = None


@dataclass
class ElectrodeWithCurrent:
    """带电流强度的电极定义

    Attributes
    ----------
    name : str
        电极名称
    current_mA : float
        电流强度（mA）
    """

    name: str
    current_mA: float


@dataclass
class ForwardParams:
    """正向仿真参数

    Attributes
    ----------
    id : str
        任务 ID
    dir_path : str
        subject 目录路径
    T1_file_path : str
        T1 加权 MRI 文件路径（用于插值参考）
    montage : str
        电极 montage 文件名
    electrode_A : list[ElectrodeWithCurrent]
        电极组 A
    electrode_B : list[ElectrodeWithCurrent]
        电极组 B
    conductivity_config : dict[str, float]
        电导率配置
    anisotropy : bool
        是否启用各向异性电导率
    DTI_file_path : str | None
        DTI tensor 文件路径（可选）
    """

    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    electrode_A: list[ElectrodeWithCurrent]
    electrode_B: list[ElectrodeWithCurrent]
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None


@dataclass
class InverseParams:
    """逆向优化参数

    Attributes
    ----------
    id : str
        任务 ID
    dir_path : str
        subject 目录路径
    T1_file_path : str
        T1 加权 MRI 文件路径（用于插值参考）
    montage : str
        电极 montage 文件名
    current_A : list[float]
        电极组 A 电流强度列表（mA）
    current_B : list[float]
        电极组 B 电流强度列表（mA）
    roi_type : Literal["atlas", "mni_pos"]
        ROI 类型：atlas 或 mni_pos
    roi_param : ROIParam
        ROI 参数
    target_threshold : float
        目标阈值
    conductivity_config : dict[str, float]
        电导率配置
    anisotropy : bool
        是否启用各向异性电导率
    DTI_file_path : str | None
        DTI tensor 文件路径（可选）
    """

    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    current_A: list[float]
    current_B: list[float]
    roi_type: Literal["atlas", "mni_pos"]
    roi_param: ROIParam
    target_threshold: float
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None


@dataclass
class AckTestParams:
    """ACK 时机验证参数

    Attributes
    ----------
    id : str
        任务 ID
    sleep_seconds : float
        睡眠等待时间（秒），默认 30.0
    """

    id: str
    sleep_seconds: float = 30.0


@dataclass
class ModelResult:
    """头模生成结果

    Attributes
    ----------
    msh_file_path : str | None
        生成的 mesh 文件路径
    """

    msh_file_path: str | None = None


@dataclass
class ForwardResult:
    """正向仿真结果

    Attributes
    ----------
    TI_file : str | None
        TI 场 nifti 文件路径
    """

    TI_file: str | None = None


@dataclass
class InverseResult:
    """逆向仿真结果

    Attributes
    ----------
    TI_file : str | None
        TI 场 nifti 文件路径
    electrode_A : list[str] | None
        电极组 A 对应的电极名称列表
    electrode_B : list[str] | None
        电极组 B 对应的电极名称列表
    """

    TI_file: str | None = None
    electrode_A: list[str] | None = None
    electrode_B: list[str] | None = None


@dataclass
class ReturnMessage:
    """返回消息

    Attributes
    ----------
    id : str
        任务 ID
    type : Literal["model", "forward", "inverse", "ack_test"]
        消息类型
    progress_rate : int
        进度百分比（0-100）
    message : str | None
        消息内容
    result : ModelResult | ForwardResult | InverseResult | None
        任务结果
    """

    id: str
    type: Literal["model", "forward", "inverse", "ack_test"]
    progress_rate: int
    message: str | None = None
    result: ModelResult | ForwardResult | InverseResult | None = None
