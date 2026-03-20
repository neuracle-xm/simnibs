"""
CHARM 步骤4: Atlas 初始仿射配准与颈部校正示例

演示如何使用 init_atlas 函数将 MNI Atlas 仿射配准到输入图像空间。

数据来源: data/m2m_ernie/
- T1fs.nii.gz 或 T1fs_denoised.nii.gz
- T2_reg.nii.gz (可选)
"""

import os

from neuracle.charm.init_atlas import init_atlas
from neuracle.logger import setup_logging

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

    print("=" * 60)
    print("CHARM 步骤4: Atlas 初始仿射配准与颈部校正")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行 Atlas 配准
    # 这将:
    # 1. 设置 samseg 相关的 atlas 路径和参数
    # 2. 根据 init_type 设置决定初始化方式（atlas 或 mni）
    # 3. 执行仿射配准
    # 4. 可选进行颈部校正
    # 输出:
    # - segmentation/template_coregistered.nii.gz
    init_atlas(
        subject_dir=subject_dir,
        use_transform=None,  # 可选：使用预先计算的变换矩阵
        init_transform=None,  # 可选：初始变换矩阵
        noneck=False,  # 设置为 True 跳过颈部校正
    )

    print("=" * 60)
    print("Atlas 仿射配准完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
