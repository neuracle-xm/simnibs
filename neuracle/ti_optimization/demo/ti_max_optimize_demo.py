"""
TI Max Optimization Demo - 最小化 ROI 内最大电场

演示如何使用步骤函数执行 TI Max 优化。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- 电极对2: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- ROI 中心: [-41.0, -13.0, 66.0] (subject space)
- ROI 半径: 20mm
"""

import os

from neuracle.logger import setup_logging
from neuracle.ti_optimization import (
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils.ti_export import export_ti_to_nifti

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ti_optimization 目录
TI_OPT_DIR = os.path.dirname(SCRIPT_DIR)
# neuracle 目录
NEURACLE_DIR = os.path.dirname(TI_OPT_DIR)
# 项目根目录 (simnibs)
PROJECT_ROOT = os.path.dirname(NEURACLE_DIR)
# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    """主函数"""
    # 启用日志
    setup_logging(os.path.join(NEURACLE_DIR, "log"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")
    output_dir = os.path.join(DATA_DIR, "TI_max_optimize_ernie")
    mesh_file_path = os.path.join(subject_dir, "model.msh")

    print("=" * 60)
    print("TI Max Optimization: 最小化 ROI 内最大电场")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 1. 初始化优化结构
    print("\n[1/5] 初始化优化结构...")
    opt = init_optimization(
        subject_dir=subject_dir,
        output_dir=output_dir,
        msh_file_path=mesh_file_path,
    )

    # 2. 配置目标函数
    print("[2/5] 配置目标函数 (max)...")
    setup_goal(
        opt=opt,
        goal="max",
    )

    # 3. 配置电极对和 ROI
    print("[3/5] 配置电极对和 ROI...")
    setup_electrodes_and_roi(
        opt=opt,
        goal="max",
        mesh_file_path=mesh_file_path,
        electrode_pair1_center=[[0, 0]],
        electrode_pair2_center=[[0, 0]],
        electrode_radius=[10],
        electrode_current1=[0.002, -0.002],
        electrode_current2=[0.002, -0.002],
        roi_center=[-41.0, -13.0, 66.0],
        roi_radius=20.0,
    )

    # 4. 运行优化
    print("[4/5] 运行优化算法...")
    output_folder = run_optimization(
        opt=opt,
        n_workers=24,
    )

    # 5. 导出 NIfTI 格式
    print("[5/5] 导出 NIfTI 格式...")
    msh_path = os.path.join(
        output_folder,
        "mapped_electrodes_simulation",
        "model_tes_mapped_opt_head_mesh.msh",
    )
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=output_dir,
        reference=os.path.join(subject_dir, "T1.nii.gz"),
        field_name="max_TI",
    )

    print("=" * 60)
    print("TI Max 优化完成!")
    print(f"输出目录: {output_folder}")
    print(f"TI NIfTI 文件: {ti_nifti_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
