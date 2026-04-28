"""
TI 逆向优化入口

提供 TI 逆向优化流程的入口函数 run_ti_inverse()。

逆向仿真是"已知目标电场分布，反求电极电流配置"的优化过程。
需要迭代优化以找到最优电流配置。

用法:
    from neuracle.ti_inverse import run_ti_inverse
    from neuracle.parameters.schemas import ROIParam, MNIParam

    run_ti_inverse(
        dir_path="m2m_ernie",
        montage="EEG10-10_Neuroelectrics",
        current_A=[1.0, -1.0],
        current_B=[1.0, -1.0],
        roi_type="mni_pos",
        roi_param=ROIParam(mni_param=MNIParam(center=[-50, -30, 40], radius=10)),
        target_threshold=0.2,
        conductivity_config={"WM": 0.126, "GM": 0.275, ...},
        anisotropy=AnisotropyType.SCALAR,
    )
"""

import logging
import os
import re
import shutil
from typing import Literal

from neuracle.atlas.standardized import get_standardized_roi_path
from neuracle.parameters.schemas import AnisotropyType, ROIParam
from neuracle.storage.paths import (
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    reset_task_output_dir,
)
from neuracle.ti_optimization import (
    get_electrode_mapping,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils import (
    NON_ROI_THRESHOLD,
    cond_dict_to_list,
    find_montage_file,
)
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger(__name__)


def run_ti_inverse(
    dir_path: str,
    montage: str,
    current_A: list[float],
    current_B: list[float],
    roi_type: Literal["atlas", "mni_pos"],
    roi_param: ROIParam,
    target_threshold: float,
    conductivity_config: dict[str, float],
    anisotropy: AnisotropyType,
    DTI_file_path: str | None = None,
    n_workers: int = 8,
    debug: bool = False,
) -> None:
    """
    运行 TI 逆向优化，生成失败则抛出异常。

    流程步骤：
    步骤 0：初始化与检查，解析 ROI 参数
    步骤 1：优化器初始化
    步骤 2：目标设置
    步骤 3：电极与 ROI 配置
    步骤 4：执行优化
    步骤 5：电极映射获取
    步骤 6：NIfTI 导出
    步骤 7：清理

    Parameters
    ----------
    dir_path : str
        头模目录名
    montage : str
        电极导联名称
    current_A : list[float]
        电极组 A 初始电流值（mA）
    current_B : list[float]
        电极组 B 初始电流值（mA）
    roi_type : Literal["atlas", "mni_pos"]
        ROI 类型
    roi_param : ROIParam
        ROI 参数配置
    target_threshold : float
        目标电场强度阈值
    conductivity_config : dict[str, float]
        组织电导率配置
    anisotropy : AnisotropyType
        各向异性类型
    DTI_file_path : str, optional
        DTI 张量文件路径
    n_workers : int
        并行工作进程数
    debug : bool
        调试模式，为 True 时不清理临时输出目录
    """
    roi_info = (
        f"atlas={roi_param.atlas_param}"
        if roi_param.atlas_param
        else f"mni={roi_param.mni_param}"
        if roi_param.mni_param
        else "None"
    )
    logger.info(
        "开始 TI 逆向优化: dir_path=%s, montage=%s, anisotropy=%s, "
        "current_A=%s, current_B=%s, roi_type=%s, roi=%s, "
        "target_threshold=%s, conductivity=%s",
        dir_path,
        montage,
        anisotropy,
        current_A,
        current_B,
        roi_type,
        roi_info,
        target_threshold,
        conductivity_config,
    )

    # ===== 步骤 0：初始化与检查 =====
    subject_dir = get_subject_dir(dir_path)
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"subject 目录不存在: {subject_dir}。请先执行头模生成任务。"
        )

    task_id = "inverse"
    output_dir = get_task_output_dir(dir_path, "TI_optimization", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not debug:
        reset_task_output_dir(str(output_dir))

    net_electrode_file = find_montage_file(str(subject_dir), montage)
    mesh_path = get_model_mesh_path(dir_path)

    roi_center = None
    roi_radius = None
    roi_center_space = "subject"
    roi_mask_path = None
    roi_mask_space = None
    focality_threshold = target_threshold

    if roi_type == "atlas" and roi_param.atlas_param:
        roi_mask_path = str(
            get_standardized_roi_path(
                roi_param.atlas_param.name,
                roi_param.atlas_param.area,
            )
        )
        if not os.path.exists(roi_mask_path):
            raise FileNotFoundError(
                f"标准化 ROI 不存在: {roi_mask_path}。请先运行 atlas 标准化和 ROI 生成脚本。"
            )
        roi_mask_space = "mni"
    elif roi_type == "mni_pos" and roi_param.mni_param:
        roi_center = roi_param.mni_param.center
        roi_radius = roi_param.mni_param.radius
        roi_center_space = "mni"

    # ===== 步骤 1：优化器初始化 =====
    opt = init_optimization(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy,
        cond=cond_dict_to_list(conductivity_config),
        fname_tensor=DTI_file_path,
    )
    logger.info("优化器初始化完成")

    # ===== 步骤 2：目标设置 =====
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    logger.info("优化目标设置完成")

    # ===== 步骤 3：电极与 ROI 配置 =====
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        mesh_file_path=mesh_path,
        electrode_current1=[c / 1000 for c in current_A],
        electrode_current2=[c / 1000 for c in current_B],
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
    )
    logger.info("电极与 ROI 配置完成")

    # ===== 步骤 4：执行优化 =====
    run_optimization(opt=opt, n_workers=n_workers)
    logger.info("优化执行完成")

    # ===== 步骤 5：电极映射获取 =====
    electrode_A, electrode_B = get_electrode_mapping(output_dir=str(output_dir))
    logger.info("电极映射获取完成: A=%s, B=%s", electrode_A, electrode_B)

    # ===== 步骤 6：NIfTI 导出 =====
    subid = re.search("m2m_(.+)", dir_path).group(1)
    msh_name = f"{subid}_tes_mapped_opt_head_mesh.msh"
    msh_path = str(output_dir / "mapped_electrodes_simulation" / msh_name)
    # 从 subject 目录查找 T1 文件作为参考空间
    t1_candidates = list(subject_dir.glob("T1*.nii.gz"))
    reference_file = str(t1_candidates[0]) if t1_candidates else ""
    # 导出到 subject 目录，避免被清理步骤删除
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=str(subject_dir),
        reference=reference_file,
        field_name="max_TI",
        prefix=f"{subid}_optimization",
    )
    logger.info("NIfTI 导出完成: %s", ti_nifti_path)

    # ===== 步骤 7：清理 =====
    if not debug:
        shutil.rmtree(output_dir, ignore_errors=True)

    logger.info("TI 逆向优化完成")
