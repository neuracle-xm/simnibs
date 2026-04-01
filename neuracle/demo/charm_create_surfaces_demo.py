"""
CHARM 步骤6: 皮层表面重建示例

演示如何使用 create_surfaces 函数从分割结果重建皮层表面。

数据来源: data/m2m_ernie/
- segmentation/tissue_labeling_upsampled.nii.gz
- segmentation/norm_image.nii.gz
"""

import os

from neuracle.charm.create_surfaces import create_surfaces
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
    setup_logging(os.path.join(PROJECT_ROOT, "neuracle", "log", "create_surfaces"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")

    print("=" * 60)
    print("CHARM 步骤6: 皮层表面重建")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行表面重建
    # 这将:
    # 1. 如果提供 FreeSurfer 目录，从 FreeSurfer 表面加载
    # 2. 否则使用 CAT12 方法进行表面重建
    # 3. 可选地根据表面更新分割结果
    # 输出:
    # - surfaces/lh.white
    # - surfaces/rh.white
    # - surfaces/lh.pial
    # - surfaces/rh.pial
    # - surfaces/lh.central
    # - surfaces/rh.central
    create_surfaces(
        subject_dir=subject_dir,
        fs_dir=None,  # 可选：FreeSurfer subjects 目录
    )

    print("=" * 60)
    print("皮层表面重建完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
