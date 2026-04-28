"""
TI 正向仿真入口

提供 TI 正向仿真流程的入口函数 run_ti_forward()。

正向仿真用于计算给定电极配置下的电场分布。
"已知电极电流，求解电场"，不涉及优化。

用法:
    from neuracle.ti_forward import run_ti_forward
    from neuracle.parameters.schemas import ElectrodeWithCurrent

    run_ti_forward(
        dir_path="m2m_ernie",
        montage="EEG10-10_Neuroelectrics",
        electrode_A=[ElectrodeWithCurrent(name="F5", current_mA=1.0),
                      ElectrodeWithCurrent(name="P5", current_mA=-1.0)],
        electrode_B=[ElectrodeWithCurrent(name="F6", current_mA=1.0),
                      ElectrodeWithCurrent(name="P6", current_mA=-1.0)],
        conductivity_config={"WM": 0.126, "GM": 0.275, ...},
        anisotropy=AnisotropyType.SCALAR,
    )
"""

import logging
import re
import shutil

from neuracle.parameters.schemas import AnisotropyType, ElectrodeWithCurrent
from neuracle.storage.paths import (
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    reset_task_output_dir,
)
from neuracle.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)
from neuracle.utils import (
    cond_dict_to_list,
    find_montage_file,
)
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger(__name__)


def run_ti_forward(
    dir_path: str,
    montage: str,
    electrode_A: list[ElectrodeWithCurrent],
    electrode_B: list[ElectrodeWithCurrent],
    conductivity_config: dict[str, float],
    anisotropy: AnisotropyType,
    DTI_file_path: str | None = None,
    n_workers: int = 8,
    debug: bool = False,
) -> None:
    """
    运行 TI 正向仿真，生成失败则抛出异常。

    流程步骤：
    步骤 0：初始化与检查
    步骤 1：会话初始化
    步骤 2：电极对 A 配置
    步骤 3：电极对 B 配置
    步骤 4：TDCS 仿真执行
    步骤 5：TI 计算
    步骤 6：NIfTI 导出
    步骤 7：清理

    Parameters
    ----------
    dir_path : str
        头模目录名
    montage : str
        电极导联名称
    electrode_A : list[ElectrodeWithCurrent]
        电极组 A，每个元素包含 name 和 current_mA
    electrode_B : list[ElectrodeWithCurrent]
        电极组 B，每个元素包含 name 和 current_mA
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
    logger.info(
        "开始 TI 正向仿真: dir_path=%s, montage=%s, anisotropy=%s, "
        "electrode_A=%s, electrode_B=%s, conductivity=%s",
        dir_path,
        montage,
        anisotropy,
        electrode_A,
        electrode_B,
        conductivity_config,
    )

    # ===== 步骤 0：初始化与检查 =====
    subject_dir = get_subject_dir(dir_path)
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"subject 目录不存在: {subject_dir}。请先执行头模生成任务。"
        )

    task_id = "forward"
    output_dir = get_task_output_dir(dir_path, "TI_simulation", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not debug:
        reset_task_output_dir(str(output_dir))

    eeg_cap = find_montage_file(str(subject_dir), montage)
    mesh_path = get_model_mesh_path(dir_path)

    electrode_A_names = [e.name for e in electrode_A]
    electrode_A_currents = [e.current_mA / 1000 for e in electrode_A]
    electrode_B_names = [e.name for e in electrode_B]
    electrode_B_currents = [e.current_mA / 1000 for e in electrode_B]

    # ===== 步骤 1：会话初始化 =====
    S = setup_session(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy,
        cond=cond_dict_to_list(conductivity_config),
        fname_tensor=DTI_file_path,
        eeg_cap=eeg_cap,
    )
    logger.info("会话初始化完成")

    # ===== 步骤 2：电极对 A 配置 =====
    setup_electrode_pair1(
        session=S,
        electrode_pair1=electrode_A_names,
        current1=electrode_A_currents,
    )
    logger.info("电极对 A 配置完成")

    # ===== 步骤 3：电极对 B 配置 =====
    setup_electrode_pair2(
        session=S,
        electrode_pair2=electrode_B_names,
        current2=electrode_B_currents,
    )
    logger.info("电极对 B 配置完成")

    # ===== 步骤 4：TDCS 仿真执行 =====
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        n_workers=n_workers,
    )
    logger.info("TDCS 仿真完成")

    # ===== 步骤 5：TI 计算 =====
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=str(output_dir),
    )
    logger.info("TI 计算完成")

    # ===== 步骤 6：NIfTI 导出 =====
    subid = re.search("m2m_(.+)", dir_path).group(1)
    # 从 subject 目录查找 T1 文件作为参考空间
    t1_candidates = list(subject_dir.glob("T1*.nii.gz"))
    reference_file = str(t1_candidates[0]) if t1_candidates else ""
    # 导出到 subject 目录，避免被清理步骤删除
    ti_nifti_path = export_ti_to_nifti(
        msh_path=ti_mesh_path,
        output_dir=str(subject_dir),
        reference=reference_file,
        field_name="max_TI",
        prefix=f"{subid}_simulation",
    )
    logger.info("NIfTI 导出完成: %s", ti_nifti_path)

    # ===== 步骤 7：清理 =====
    if not debug:
        shutil.rmtree(output_dir, ignore_errors=True)

    logger.info("TI 正向仿真完成")
