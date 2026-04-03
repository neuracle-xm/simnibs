"""
任务进度管理

提供进度枚举和进度文件的加载保存功能。
"""

from enum import IntEnum
from pathlib import Path


class ModelProgress(IntEnum):
    """Model 任务进度枚举"""

    START = 0
    PREPARE_T1_DONE = 10
    PREPARE_T2_DONE = 20
    DENOISE_DONE = 35
    INIT_ATLAS_DONE = 50
    SEGMENTATION_DONE = 70
    SURFACES_DONE = 85
    COMPLETED = 100


class ForwardProgress(IntEnum):
    """Forward 任务进度枚举"""

    START = 0
    SESSION_SETUP = 10
    ELECTRODE_PAIR1_DONE = 20
    ELECTRODE_PAIR2_DONE = 35
    TDCS_SIMULATION_DONE = 70
    TI_CALCULATED = 85
    NIFTI_EXPORTED = 95
    COMPLETED = 100


class InverseProgress(IntEnum):
    """Inverse 任务进度枚举"""

    START = 0
    OPTIMIZATION_INIT = 10
    GOAL_SETUP = 20
    ELECTRODES_ROI_SETUP = 35
    OPTIMIZATION_DONE = 85
    ELECTRODE_MAPPING_DONE = 90
    NIFTI_EXPORTED = 95
    COMPLETED = 100


def load_progress(progress_file: Path) -> int:
    """加载进度，返回当前百分比，未找到则返回 0"""
    if progress_file.exists():
        return int(progress_file.read_text().strip())
    return 0


def save_progress(progress_file: Path, progress: int) -> None:
    """保存进度"""
    progress_file.write_text(str(progress))
