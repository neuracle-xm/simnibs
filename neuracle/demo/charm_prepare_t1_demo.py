"""
CHARM 步骤1: T1 图像准备示例

演示如何使用 prepare_t1 函数将原始 T1 加权 MRI 图像转换为 SimNIBS 所需格式。

数据来源: data/m2m_ernie/T1.nii.gz
"""

import os

from neuracle.charm.prepare_t1 import prepare_t1
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
    t1_input = os.path.join(DATA_DIR, "m2m_ernie", "T1.nii.gz")

    print("=" * 60)
    print("CHARM 步骤1: T1 图像准备")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Input T1: {t1_input}")

    # 执行 T1 图像准备
    # 这将:
    # 1. 检查并修正 qform/sform 编码
    # 2. 去除单例维度
    # 3. 转换为 float32
    # 4. 保存到 subject_dir (使用 file_finder 的 reference_volume)
    prepare_t1(
        subject_dir=subject_dir,
        t1=t1_input,
        force_qform=False,
        force_sform=False,
    )

    print("=" * 60)
    print("T1 图像准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
