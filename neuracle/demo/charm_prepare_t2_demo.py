"""
CHARM 步骤2: T2 图像配准与准备示例

演示如何使用 prepare_t2 函数处理 T2 加权图像，包括可选的 T2-to-T1 刚性配准。

数据来源: data/m2m_ernie/T2_reg.nii.gz
"""

import os

from neuracle.charm.prepare_t2 import prepare_t2
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
    t2_input = os.path.join(DATA_DIR, "m2m_ernie", "T2_reg.nii.gz")

    print("=" * 60)
    print("CHARM 步骤2: T2 图像配准与准备")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Input T2: {t2_input}")

    # 执行 T2 图像准备（不进行配准）
    # 如果 register_t2=True，会执行 T2-to-T1 刚性配准
    # 这将:
    # 1. 检查并修正 qform/sform 编码
    # 2. 去除单例维度
    # 3. 转换为 float32
    # 4. 保存到 subject_dir/T2_reg.nii.gz
    prepare_t2(
        subject_dir=subject_dir,
        t2=t2_input,
        register_t2=True,
        force_qform=False,
        force_sform=True,
    )

    print("=" * 60)
    print("T2 图像准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
