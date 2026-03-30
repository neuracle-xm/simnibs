"""
TI Simulation Demo - Temporal Interference 正向仿真示例

演示如何使用步骤函数执行 TI 正向仿真。

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
from neuracle.ti_simulation import (
    calculate_ti,
    export_mz3,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)

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

    # 1. 配置会话
    print("\n[1/6] 配置会话参数...")
    S = setup_session(
        subject_dir=subject_dir,
        output_dir=output_dir,
    )

    # 2. 配置第一个电极对
    print("[2/6] 配置第一个电极对 (F5-P5)...")
    setup_electrode_pair1(
        session=S,
        electrode_pair1=["F5", "P5"],
        current1=0.001,  # 1mA
    )

    # 3. 配置第二个电极对
    print("[3/6] 配置第二个电极对 (F6-P6)...")
    setup_electrode_pair2(
        session=S,
        electrode_pair2=["F6", "P6"],
        current2=0.001,  # 1mA
    )

    # 4. 运行 TDCS 仿真
    print("[4/6] 运行 TDCS 仿真...")
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=subject_dir,
        output_dir=output_dir,
        n_workers=24,
    )

    # 5. 计算 TI 场
    print("[5/6] 计算 TI 场...")
    ti_mesh_path, ti_max = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=output_dir,
    )

    # 6. 导出 MZ3 格式
    print("[6/6] 导出 MZ3 格式...")
    mz3_path = export_mz3(
        ti_mesh_path=ti_mesh_path,
        output_dir=output_dir,
        surface_type="central",
    )

    print("=" * 60)
    print("TI 仿真完成!")
    print(f"TI 网格文件: {ti_mesh_path}")
    print(f"MZ3 文件: {mz3_path}")
    print(f"TI 最大调制振幅范围: [{ti_max.min():.4f}, {ti_max.max():.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
