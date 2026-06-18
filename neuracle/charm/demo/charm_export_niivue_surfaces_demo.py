"""
CHARM 步骤8: 导出 Niivue central 重建表面示例。

演示如何从 CHARM 第 6 步生成的左右半球 central 重建表面中，
合并并导出受试者目录下固定文件 `surface.gii.gz`。
"""

from neuracle.charm.export_niivue_surfaces import export_niivue_surfaces
from neuracle.charm.simnibs_logging import simnibs_charm_file_logging
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main() -> None:
    """
    示例主函数。

    Returns
    -------
    None
    """
    setup_logging(str(PROJECT_ROOT / "log" / "charm_export_niivue_surfaces"))
    subject_dir = DATA_ROOT / "m2m_ernie"
    print("=" * 60)
    print("CHARM 步骤8: 导出 Niivue central 重建表面")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    with simnibs_charm_file_logging(
        str(subject_dir), filename="simnibs_charm_export_niivue_surfaces.log"
    ):
        output_path = export_niivue_surfaces(subject_dir=str(subject_dir))
    print("生成文件:")
    print(f"  - {output_path}")
    print("=" * 60)
    print("Niivue central 重建表面导出完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
