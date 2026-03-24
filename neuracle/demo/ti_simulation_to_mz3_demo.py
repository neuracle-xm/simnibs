"""
TI simulation to MZ3 demo.

将 TI 仿真结果导出为 `.mz3` 以便在 surf-ice/BrainVisa 中查看。
MZ3 只支持顶点标量，因此这里默认写入电场强度 `magnE`，
而不是向量场 `E`。
"""

import os

from neuracle.logger import setup_logging
from neuracle.mesh_tools import msh_to_mz3

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
    ti_mesh_path = os.path.join(DATA_DIR, "TI_ernie", "TI.msh")
    output_dir = os.path.join(DATA_DIR, "TI_ernie_mz3")

    # 检查 TI 仿真结果是否存在
    if not os.path.exists(ti_mesh_path):
        print("错误: TI 仿真结果不存在!")
        print("请先运行: python neuracle/demo/ti_simulation_demo.py")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TI Simulation to MZ3: TI 仿真结果可视化")
    print("=" * 60)
    print(f"Input MSH: {ti_mesh_path}")
    print(f"Output directory: {output_dir}")
    print("Scalar field: magnE")
    print("Surface mapping: white=1001, central=1002, pial=1003")

    # 三种皮层表面类型
    surface_types = ["central", "pial", "white"]

    # 依次转换三种表面
    for surface_type in surface_types:
        print(f"\n转换 {surface_type} 表面 (带电场强度标量)...")
        output_path = msh_to_mz3(
            msh_path=ti_mesh_path,
            output_dir=output_dir,
            surface_type=surface_type,
            field_name="TImax",
        )
        print(f"  -> {output_path}")

    print("=" * 60)
    print("转换完成!")
    print(f"输出目录: {output_dir}")
    print("生成文件:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    print("=" * 60)
    print("提示: 生成的 .mz3 文件可在 surf-ice 或 BrainVisa 中打开，")
    print("      magnE 会作为顶点标量显示，可用于彩色渲染。")


if __name__ == "__main__":
    main()
