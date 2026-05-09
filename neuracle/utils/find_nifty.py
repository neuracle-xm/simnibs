from pathlib import Path


def find_optional_nifti_file(
    subject_dir: Path,
    candidate_names: tuple[str, ...],
) -> str | None:
    """
    在指定目录中按顺序查找 NIfTI 文件。

    Parameters
    ----------
    subject_dir : Path
        任务目录路径
    candidate_names : tuple[str, ...]
        候选文件名，按优先级排序

    Returns
    -------
    str | None
        找到则返回完整路径字符串，否则返回 None
    """
    for candidate_name in candidate_names:
        candidate_path = subject_dir / candidate_name
        if candidate_path.exists():
            return str(candidate_path)
    return None
