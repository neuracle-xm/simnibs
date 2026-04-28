"""
任务进度管理

提供头模生成任务的进度枚举和进度文件的加载保存功能。

进度持久化
---------
进度信息保存到 subject_dir/.progress.txt 文件中。
文件内容为单个整数（0-100），表示当前进度百分比。
当消息被重发时，可从此文件恢复执行进度。

使用场景
-------
在 Model 任务中（支持断点续传）：
- 任务开始时检查 progress_file 是否存在
- 每个步骤完成后保存当前进度
- 若消息被重发，从进度文件恢复
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


def load_progress(progress_file: Path) -> int:
    """
    加载进度

    Parameters
    ----------
    progress_file : Path
        进度文件路径

    Returns
    -------
    int
        当前百分比，未找到则返回 0
    """
    if progress_file.exists():
        return int(progress_file.read_text().strip())
    return 0


def save_progress(progress_file: Path, progress: int) -> None:
    """
    保存进度

    Parameters
    ----------
    progress_file : Path
        进度文件路径
    progress : int
        进度百分比
    """
    progress_file.write_text(str(progress))
