"""
Storage 存储模块

提供文件存储功能，包括：
- 本地路径管理
- 任务输出目录管理

依赖
----
- config: 配置模块，提供环境变量读取

示例
----
>>> from neuracle.storage import get_subject_dir
>>> subject_dir = get_subject_dir("m2m_ernie")
"""

from neuracle.storage.paths import (
    DATA_ROOT,
    ensure_data_root,
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    normalize_dir_path,
    reset_task_output_dir,
    resolve_local_dti_path,
)
from neuracle.utils.constants import (
    BUILT_IN_DIR_PATH,
    BUILT_IN_DTI_FILE_PATH,
    PROJECT_ROOT,
)

__all__ = [
    "DATA_ROOT",
    "PROJECT_ROOT",
    "BUILT_IN_DIR_PATH",
    "BUILT_IN_DTI_FILE_PATH",
    "normalize_dir_path",
    "get_subject_dir",
    "get_task_output_dir",
    "get_model_mesh_path",
    "resolve_local_dti_path",
    "ensure_data_root",
    "reset_task_output_dir",
]
