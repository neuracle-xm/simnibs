"""
CHARM 步骤5: 体积与表面分割示例

演示如何使用 run_segmentation 函数执行分割、偏置场校正，并生成上采样的组织标记图像。

数据来源: data/m2m_ernie/
- segmentation/template_coregistered.nii.gz
- T1fs.nii.gz / T2_reg.nii.gz (偏置校正输出)
"""

import os

from neuracle.charm.segment import run_segmentation
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
    setup_logging(os.path.join(PROJECT_ROOT, "neuracle", "log", "charm_segment"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")

    print("=" * 60)
    print("CHARM 步骤5: 体积与表面分割")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行分割
    # 这将:
    # 1. 使用 samseg 进行参数估计和分割
    # 2. 输出偏置场校正后的图像
    # 3. 进行后处理（形态学操作）
    # 4. 生成上采样（1mm）的组织标签图像
    # 输出:
    # - segmentation/tissue_labeling_upsampled.nii.gz
    # - segmentation/tissue_labeling_upsampled_LUT.txt
    run_segmentation(
        subject_dir=subject_dir,
        debug=False,  # 设置为 True 会保留中间结果
    )

    print("=" * 60)
    print("分割完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
