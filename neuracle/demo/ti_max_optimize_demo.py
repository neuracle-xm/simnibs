"""
TI Max Optimization Demo - 最小化 ROI 内最大电场

演示如何使用 run_ti_optimization 函数执行 TI Max 优化。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- 电极对2: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- ROI 中心: [-41.0, -13.0, 66.0] (subject space)
- ROI 半径: 20mm
"""

import os

from neuracle.logger import setup_logging
from neuracle.ti_optimize import run_ti_optimization

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
    output_dir = os.path.join(DATA_DIR, "TI_max_optimize_ernie")

    print("=" * 60)
    print("TI Max Optimization: 最小化 ROI 内最大电场")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 执行 TI Max 优化
    output_folder = run_ti_optimization(
        subject_dir=subject_dir,
        output_dir=output_dir,
        goal="max",
        roi_center=[-41.0, -13.0, 66.0],
        roi_radius=20.0,
        electrode_radius=[10],
        electrode_current1=[0.002, -0.002],
        electrode_current2=[0.002, -0.002],
        map_to_net_electrodes=True,
        run_mapped_electrodes_simulation=True,
        net_electrode_file=EEG_ELECTRODE_FILE,
        n_workers=24,
    )

    print("=" * 60)
    print("TI Max 优化完成!")
    print(f"输出目录: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()
