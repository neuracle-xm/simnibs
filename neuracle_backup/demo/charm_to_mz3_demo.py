"""
CHARM to MZ3 demo.

将 CHARM 生成的带表面标签的 `.msh` 导出为 `.mz3`。
这里的 surface_type 映射为 SimNIBS 现成表面标签:
    - white   -> WM_TH_SURFACE (1001)
    - central -> GM_TH_SURFACE (1002)
    - pial    -> CSF_TH_SURFACE (1003)
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
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")
    msh_path = os.path.join(subject_dir, "ernie.msh")
    output_dir = os.path.join(DATA_DIR, "m2m_ernie_mz3")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CHARM to MZ3: 将头模型转换为 MZ3 表面网格格式")
    print("=" * 60)
    print(f"Input MSH: {msh_path}")
    print(f"Output directory: {output_dir}")
    print("Surface mapping: white=1001, central=1002, pial=1003")

    # 三种皮层表面类型
    surface_types = ["central", "pial", "white"]

    # 依次转换三种表面
    for surface_type in surface_types:
        print(f"\n转换 {surface_type} 表面...")
        output_path = msh_to_mz3(
            msh_path=msh_path,
            output_dir=output_dir,
            surface_type=surface_type,
            field_name=None,
        )
        print(f"  -> {output_path}")

    print("=" * 60)
    print("转换完成!")
    print(f"输出目录: {output_dir}")
    print("生成文件:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
