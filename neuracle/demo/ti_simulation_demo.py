"""
TI Simulation Demo - Temporal Interference 正向仿真示例

演示如何使用 run_ti_simulation 函数执行 TI 正向仿真。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: F5-P5, 电流 1mA
- 电极对2: F6-P6, 电流 1mA
- 电极形状: 椭圆 40x40 mm
- 电极厚度: 2 mm
- n_workers: 24
"""

import os

from neuracle.logger import setup_logging
from neuracle.ti_simulation import run_ti_simulation

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# neuracle 目录
NEURACLE_DIR = os.path.dirname(SCRIPT_DIR)
# 项目根目录 (simnibs)
PROJECT_ROOT = os.path.dirname(NEURACLE_DIR)
# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    """主函数"""
    # 启用日志
    setup_logging(os.path.join(PROJECT_ROOT, "neuracle", "log"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")
    output_dir = os.path.join(DATA_DIR, "TI_ernie")

    print("=" * 60)
    print("TI Simulation: Temporal Interference 正向仿真")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 执行 TI 仿真
    # 配置两个电极对：
    # - 电极对1: F5-P5 (左半球)
    # - 电极对2: F6-P6 (右半球)
    # 输出:
    # - TI_ernie/ernie_TDCS_1_scalar.msh - 第一对电极仿真结果
    # - TI_ernie/ernie_TDCS_2_scalar.msh - 第二对电极仿真结果
    # - TI_ernie/TI.msh - TI 场可视化结果
    ti_mesh_path, ti_max = run_ti_simulation(
        subject_dir=subject_dir,
        output_dir=output_dir,
        electrode_pair1=["F5", "P5"],
        electrode_pair2=["F6", "P6"],
        current1=0.001,  # 1mA
        current2=0.001,  # 1mA
        electrode_shape="ellipse",
        electrode_dimensions=[40, 40],
        electrode_thickness=2.0,
        n_workers=24,
    )

    print("=" * 60)
    print("TI 仿真完成!")
    print(f"TI 网格文件: {ti_mesh_path}")
    print(f"TI 最大调制振幅范围: [{ti_max.min():.4f}, {ti_max.max():.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
