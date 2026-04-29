"""
头模生成入口

提供 CHARM 头模生成流程的入口函数 generate_head_model()。

用法:
    from neuracle.head_model import generate_head_model

    generate_head_model(
        dir_path="m2m_subject001",
        T1_file_path="/path/to/T1.nii.gz",
        T2_file_path="/path/to/T2.nii.gz",   # 可选
        DTI_file_path="/path/to/DTI.nii.gz",  # 可选
    )
"""

import logging
from pathlib import Path

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
from neuracle.parameters.progress import (
    ModelProgress,
    load_progress,
    save_progress,
)
from neuracle.storage.paths import (
    ensure_data_root,
    get_subject_dir,
)

logger = logging.getLogger(__name__)


def generate_head_model(
    dir_path: str,
    T1_file_path: str,
    T2_file_path: str | None = None,
    DTI_file_path: str | None = None,
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
    dir_path : str
        subject 输出目录名（如 m2m_ernie）
    T1_file_path : str
        T1 加权 MRI 图像路径
    T2_file_path : str, optional
        T2 加权 MRI 图像路径
    DTI_file_path : str, optional
        DTI 扩散张量图像路径
    """
    logger.info(
        "开始头模生成: dir_path=%s, T1_file=%s, T2_file=%s, DTI_file=%s",
        dir_path,
        T1_file_path,
        T2_file_path,
        DTI_file_path,
    )

    # ===== 步骤 0：初始化 =====
    subject_dir = get_subject_dir(dir_path)
    ensure_data_root()
    subject_dir.mkdir(parents=True, exist_ok=True)

    progress_file = subject_dir / ".progress.txt"
    current_progress = load_progress(progress_file)
    logger.info("从进度 %d 开始执行", current_progress)

    # ===== 步骤 1：准备输入文件 =====
    t1_local_path = Path(T1_file_path) if T1_file_path else None
    t2_local_path = Path(T2_file_path) if T2_file_path else None
    dti_local_path = Path(DTI_file_path) if DTI_file_path else None

    if current_progress < ModelProgress.PREPARE_T1_DONE:
        if not t1_local_path or not t1_local_path.exists():
            raise FileNotFoundError(f"T1 文件不存在: {t1_local_path}")
        if t2_local_path and not t2_local_path.exists():
            raise FileNotFoundError(f"T2 文件不存在: {t2_local_path}")
        if dti_local_path and not dti_local_path.exists():
            raise FileNotFoundError(f"DTI 文件不存在: {dti_local_path}")

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
                prepare_t2(str(subject_dir), str(t2_local_path))
            except Exception:
                try:
                    prepare_t2(str(subject_dir), str(t2_local_path), force_qform=True)
                except Exception:
                    prepare_t2(str(subject_dir), str(t2_local_path), force_sform=True)
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
