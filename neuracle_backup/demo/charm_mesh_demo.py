"""
CHARM 步骤7: 四面体网格生成示例

演示如何使用 create_mesh_step 函数从组织标签图像生成 tetrahedral 头模型网格。

数据来源: data/m2m_ernie/
- segmentation/tissue_labeling_upsampled.nii.gz
"""

import os

from neuracle.charm.mesh import create_mesh_step
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
    setup_logging(os.path.join(PROJECT_ROOT, "neuracle", "log", "charm_mesh"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")

    print("=" * 60)
    print("CHARM 步骤7: 四面体网格生成")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行网格生成
    # 这将:
    # 1. 加载上采样的组织标签图像
    # 2. 裁剪图像至感兴趣区域
    # 3. 使用 CGAL 进行四面体网格生成
    # 4. 变换 EEG 电极位置到受试者空间
    # 5. 输出最终的 .msh 文件
    # 输出:
    # - {subid}.msh (头模型网格)
    # - eeg_positions/*.csv, *.geo (EEG 电极位置)
    # - mni_transf/final_labels.nii.gz, final_labels_MNI.nii.gz
    create_mesh_step(
        subject_dir=subject_dir,
        debug=False,  # 设置为 True 会保留中间结果
    )

    print("=" * 60)
    print("四面体网格生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
