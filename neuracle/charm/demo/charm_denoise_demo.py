"""
CHARM 步骤3: 图像降噪示例

演示如何使用 denoise_inputs 函数对输入图像进行 SANLM 降噪。

数据来源: data/m2m_ernie/
- T1fs.nii.gz 或 reference_volume
- T2_reg.nii.gz (如果存在)
"""

import os

from neuracle.charm.denoise import denoise_inputs
from neuracle.logger import setup_logging

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# neuracle 目录
NEURACLE_DIR = os.path.dirname(SCRIPT_DIR)
# 项目根目录 (simnibs)
PROJECT_ROOT = os.path.dirname(os.path.dirname(NEURACLE_DIR))
# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    """主函数"""
    # 启用日志
    setup_logging(os.path.join(PROJECT_ROOT, "neuracle", "log"))

    # 设置路径
    subject_dir = os.path.join(DATA_DIR, "m2m_ernie")

    print("=" * 60)
    print("CHARM 步骤3: 图像降噪")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行降噪
    # 这将:
    # 1. 读取 settings.ini 中的降噪设置
    # 2. 如果 T1_denoised 不存在，对 T1 进行 SANLM 降噪
    # 3. 如果 T2_reg 存在且 T2_reg_denoised 不存在，对 T2 进行降噪
    # 输出:
    # - T1fs_denoised.nii.gz
    # - T2_reg_denoised.nii.gz (如果存在 T2)
    denoise_inputs(subject_dir=subject_dir)

    print("=" * 60)
    print("图像降噪完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
