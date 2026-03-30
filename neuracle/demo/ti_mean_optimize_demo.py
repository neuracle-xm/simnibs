"""
TI Mean Optimization Demo - 最大化 ROI 内平均电场

演示如何使用步骤函数执行 TI Mean 优化。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- 电极对2: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- ROI 中心: [-41.0, -13.0, 66.0] (subject space)
- ROI 半径: 20mm
"""

import os

from neuracle.logger import setup_logging
from neuracle.ti_optimize import (
    export_mz3,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# neuracle 目录
NEURACLE_DIR = os.path.dirname(SCRIPT_DIR)
# 项目根目录 (simnibs)
PROJECT_ROOT = os.path.dirname(NEURACLE_DIR)
# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# EEG 电极文件路径
EEG_ELECTRODE_FILE = os.path.join(
    PROJECT_ROOT,
    "simnibs",
    "resources",
    "ElectrodeCaps_MNI",
    "EEG10-10_UI_Jurak_2007.csv",
)


def main():
    """主函数"""
    # 启用日志
    setup_logging(os.path.join(PROJECT_ROOT, "neuracle", "log"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")
    output_dir = os.path.join(DATA_DIR, "TI_mean_optimize_ernie")

    print("=" * 60)
    print("TI Mean Optimization: 最大化 ROI 内平均电场")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 1. 初始化优化结构
    print("\n[1/5] 初始化优化结构...")
    opt = init_optimization(
        subject_dir=subject_dir,
        output_dir=output_dir,
    )

    # 2. 配置目标函数
    print("[2/5] 配置目标函数 (mean)...")
    setup_goal(
        opt=opt,
        goal="mean",
    )

    # 3. 配置电极对和 ROI
    print("[3/5] 配置电极对和 ROI...")
    setup_electrodes_and_roi(
        opt=opt,
        goal="mean",
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

    # 5. 导出 MZ3 格式
    print("[5/5] 导出 MZ3 格式...")
    mz3_path = export_mz3(
        output_dir=output_dir,
        surface_type="central",
    )

    print("=" * 60)
    print("TI Mean 优化完成!")
    print(f"输出目录: {output_folder}")
    print(f"MZ3 文件: {mz3_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
