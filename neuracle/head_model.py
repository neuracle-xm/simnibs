"""
头模生成入口

提供 CHARM 头模生成流程的入口函数 generate_head_model()。

用法:
    from neuracle.head_model import generate_head_model

    generate_head_model(
        head_model_dir="/path/to/data_root/head_models/m2m_{head_model_id}",
    )
"""

import argparse
import logging
import sys
from pathlib import Path

# 支持直接执行当前脚本文件时使用 `from neuracle...` 绝对导入
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuracle.charm import (
    create_mesh,
    create_surfaces,
    denoise_inputs,
    export_niivue_surfaces,
    init_atlas,
    prepare_t1,
    prepare_t2,
    run_segmentation,
)
from neuracle.charm.simnibs_logging import simnibs_charm_file_logging
from neuracle.logger import setup_logging
from neuracle.parameters.converter import dict_to_model_params
from neuracle.parameters.progress import (
    ModelProgress,
    load_progress,
    save_progress,
)
from neuracle.parameters.validator import ValidationError, validate_model_params
from neuracle.utils.constants import (
    EXIT_INVALID_ARGS,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_VALUE_ERROR,
)
from neuracle.utils.error_message import clear_error_message, write_error_message
from neuracle.utils.find_nifty import find_optional_nifti_file

logger = logging.getLogger("neuracle.head_model")


def generate_head_model(
    head_model_dir: str,
) -> None:
    """
    运行 CHARM 头模生成流程，并记录 SimNIBS 文件日志。

    Parameters
    ----------
    head_model_dir : str
        头模目录路径。

    Returns
    -------
    None
        无返回值。
    """
    with simnibs_charm_file_logging(head_model_dir):
        _generate_head_model_impl(head_model_dir=head_model_dir)


def _generate_head_model_impl(
    head_model_dir: str,
) -> None:
    """
    运行 CHARM 头模生成流程，生成失败则抛出异常。

    流程步骤：
    步骤 0：初始化，创建 subject 目录
    步骤 1：T1 图像预处理
    步骤 2：T2 图像预处理
    步骤 3：去噪处理
    步骤 4：Atlas 初始化
    步骤 5：组织分割
    步骤 6：表面重建
    步骤 7：Mesh 生成
    步骤 8：导出 Niivue central 表面

    每个步骤完成后保存进度到 .progress.txt，支持断点续传。

    Parameters
    ----------
    head_model_dir : str
        头模目录路径
    """
    logger.info(
        "开始头模生成: 目录=%s",
        head_model_dir,
    )

    # ===== 步骤 0：初始化 =====
    subject_dir = Path(head_model_dir)

    progress_file = subject_dir / ".progress.txt"
    current_progress = load_progress(progress_file)
    logger.info("从进度 %d 开始执行", current_progress)

    # ===== 步骤 1：准备输入文件 =====
    t1_file_path = find_optional_nifti_file(subject_dir, ("T1.nii.gz", "T1.nii"))
    t2_file_path = find_optional_nifti_file(subject_dir, ("T2.nii.gz", "T2.nii"))
    t1_local_path = Path(t1_file_path) if t1_file_path else None
    t2_local_path = Path(t2_file_path) if t2_file_path else None

    if current_progress < ModelProgress.PREPARE_T1_DONE:
        if not t1_local_path or not t1_local_path.exists():
            raise FileNotFoundError(f"T1 文件不存在: {t1_local_path}")
        if t2_local_path and not t2_local_path.exists():
            raise FileNotFoundError(f"T2 文件不存在: {t2_local_path}")

    # ===== 步骤 2：T1 图像预处理 =====
    if current_progress < ModelProgress.PREPARE_T1_DONE:
        if t1_local_path and t1_local_path.exists():
            try:
                prepare_t1(str(subject_dir), str(t1_local_path))
            except Exception:
                try:
                    prepare_t1(str(subject_dir), str(t1_local_path), force_qform=True)
                except Exception:
                    prepare_t1(str(subject_dir), str(t1_local_path), force_sform=True)
        save_progress(progress_file, ModelProgress.PREPARE_T1_DONE)
        logger.info("T1 预处理完成")

    # ===== 步骤 3：T2 图像预处理 =====
    if current_progress < ModelProgress.PREPARE_T2_DONE:
        if t2_local_path and t2_local_path.exists():
            try:
                prepare_t2(
                    str(subject_dir),
                    str(t2_local_path),
                    register_t2=True,
                )
            except Exception:
                try:
                    prepare_t2(
                        str(subject_dir),
                        str(t2_local_path),
                        register_t2=True,
                        force_qform=True,
                    )
                except Exception:
                    prepare_t2(
                        str(subject_dir),
                        str(t2_local_path),
                        register_t2=True,
                        force_sform=True,
                    )
        save_progress(progress_file, ModelProgress.PREPARE_T2_DONE)
        logger.info("T2 预处理完成")

    # ===== 步骤 4：去噪处理 =====
    if current_progress < ModelProgress.DENOISE_DONE:
        # 确保 CHARM 管线需要的子目录存在
        (subject_dir / "segmentation").mkdir(parents=True, exist_ok=True)
        denoise_inputs(str(subject_dir))
        save_progress(progress_file, ModelProgress.DENOISE_DONE)
        logger.info("去噪处理完成")

    # ===== 步骤 5：Atlas 初始化 =====
    if current_progress < ModelProgress.INIT_ATLAS_DONE:
        init_atlas(str(subject_dir))
        save_progress(progress_file, ModelProgress.INIT_ATLAS_DONE)
        logger.info("Atlas 初始化完成")

    # ===== 步骤 6：组织分割 =====
    if current_progress < ModelProgress.SEGMENTATION_DONE:
        run_segmentation(str(subject_dir))
        save_progress(progress_file, ModelProgress.SEGMENTATION_DONE)
        logger.info("组织分割完成")

    # ===== 步骤 7：表面重建 =====
    if current_progress < ModelProgress.SURFACES_DONE:
        create_surfaces(str(subject_dir))
        save_progress(progress_file, ModelProgress.SURFACES_DONE)
        logger.info("表面重建完成")

    # ===== 步骤 7：Mesh 生成 =====
    if current_progress < ModelProgress.MESH_DONE:
        create_mesh(str(subject_dir))
        save_progress(progress_file, ModelProgress.MESH_DONE)
        logger.info("Mesh 生成完成")

    # ===== 步骤 8：导出 Niivue central 表面 =====
    if current_progress < ModelProgress.EXPORT_SURFACE_DONE:
        export_niivue_surfaces(str(subject_dir))
        save_progress(progress_file, ModelProgress.EXPORT_SURFACE_DONE)
        progress_file.unlink()
        logger.info("头模生成完成: %s", subject_dir)


def resolve_head_model_inputs(data_root: str, task_id: str) -> dict[str, str | None]:
    """
    根据 data_root 和 task_id 解析头模输入参数。

    原理
    ----
    命令行只传入任务根目录和任务标识，程序内部拼出任务目录，
    并在该目录下自动查找 T1/T2 输入文件，减少外部调用方拼参负担。

    Parameters
    ----------
    data_root : str
        数据根目录
    task_id : str
        任务 ID

    Returns
    -------
    dict[str, str | None]
        头模生成所需参数字典

    Raises
    ------
    FileNotFoundError
        当任务目录不存在或缺少 T1 文件时抛出
    """
    subject_dir = Path(data_root) / "head_models" / f"m2m_{task_id}"
    if not subject_dir.exists() or not subject_dir.is_dir():
        raise FileNotFoundError(f"任务目录不存在: {subject_dir}")
    t1_file_path = find_optional_nifti_file(subject_dir, ("T1.nii.gz", "T1.nii"))
    if t1_file_path is None:
        raise FileNotFoundError(f"未找到 T1 文件: {subject_dir}")
    t2_file_path = find_optional_nifti_file(subject_dir, ("T2.nii.gz", "T2.nii"))
    return {
        "head_model_dir": str(subject_dir),
        "T1_file_path": t1_file_path,
        "T2_file_path": t2_file_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建头模生成命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        命令行解析器
    """
    parser = argparse.ArgumentParser(
        prog="python neuracle/head_model.py",
        description="运行 CHARM 头模生成流程",
    )
    parser.add_argument("data_root", help="数据根目录")
    parser.add_argument("task_id", help="任务目录名")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    头模生成命令行入口。

    Parameters
    ----------
    argv : list[str] | None
        命令行参数列表，默认读取 sys.argv

    Returns
    -------
    int
        进程退出码
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    task_dir = Path(args.data_root) / "head_models" / f"m2m_{args.task_id}"
    log_dir = task_dir / "logs" if task_dir.is_dir() else None
    try:
        params_dict = resolve_head_model_inputs(args.data_root, args.task_id)
        log_dir = Path(params_dict["head_model_dir"]) / "logs"
        setup_logging(str(log_dir))
        validate_model_params(params_dict)
        params = dict_to_model_params(params_dict)
        generate_head_model(
            head_model_dir=params.head_model_dir,
        )
    except ValidationError as exc:
        logger.exception("头模生成参数校验失败: %s", exc)
        if log_dir is not None:
            write_error_message(log_dir, EXIT_INVALID_ARGS, str(exc))
        return EXIT_INVALID_ARGS
    except ValueError as exc:
        logger.exception("头模生成数值处理失败: %s", exc)
        if log_dir is not None:
            write_error_message(log_dir, EXIT_VALUE_ERROR, str(exc))
        return EXIT_VALUE_ERROR
    except Exception as exc:
        logger.exception("头模生成执行失败: %s", exc)
        if log_dir is not None:
            write_error_message(log_dir, EXIT_RUNTIME_ERROR, str(exc))
        return EXIT_RUNTIME_ERROR
    if log_dir is not None:
        clear_error_message(log_dir)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
