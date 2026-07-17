"""
TI Max Optimization Demo - 最大化 ROI 内电场的 99.9 百分位

演示如何使用步骤函数执行 TI Max 优化。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: ElectrodeArrayPair, 半径 6mm, 电流 2mA
- 电极对2: ElectrodeArrayPair, 半径 6mm, 电流 2mA
- ROI 中心: [-41.0, -13.0, 66.0] (subject space)
- ROI 半径: 20mm
- 自定义电导率: WM 0.14 S/m, GM 0.30 S/m，其余使用标准值
- 电导率各向异性类型: vn
"""

from pathlib import Path

from neuracle.logger import setup_logging
from neuracle.ti_optimization import (
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils import cond_dict_to_list
from neuracle.utils.constants import (
    DATA_ROOT,
    ELECTRODE_RADIUS,
    PROJECT_ROOT,
    STANDARD_COND,
)
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs.utils import file_finder


def main() -> None:
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "ti_max_optimize"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"
    output_dir = DATA_ROOT / "TI_max_optimize_ernie"
    sub_files = file_finder.SubjectFiles(subpath=str(subject_dir))
    subid = sub_files.subid
    conductivity_config = {**STANDARD_COND, "WM": 0.14, "GM": 0.30}
    cond = cond_dict_to_list(conductivity_config)
    anisotropy_type = "vn"

    print("=" * 60)
    print("TI Max Optimization: 最大化 ROI 内电场的 99.9 百分位")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 1. 初始化优化结构
    print("\n[1/5] 初始化优化结构...")
    opt = init_optimization(
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        msh_file_path=sub_files.fnamehead,
        anisotropy_type=anisotropy_type,
        cond=cond,
    )

    # 2. 配置目标函数
    print("[2/5] 配置目标函数 (max)...")
    setup_goal(
        opt=opt,
        goal="max",
    )

    # 3. 配置电极对和 ROI
    print("[3/5] 配置电极对和 ROI...")
    setup_electrodes_and_roi(
        opt=opt,
        goal="max",
        mesh_file_path=sub_files.fnamehead,
        electrode_pair1_center=[[0, 0]],
        electrode_pair2_center=[[0, 0]],
        electrode_radius=[ELECTRODE_RADIUS],
        electrode_current1=[0.002, -0.002],
        electrode_current2=[0.002, -0.002],
        roi_center=[-41.0, -13.0, 66.0],
        roi_radius=20.0,
    )

    # 4. 运行优化
    print("[4/5] 运行优化算法...")
    output_folder = run_optimization(
        opt=opt,
        n_workers=24,
    )

    # 5. 导出 NIfTI 格式
    print("[5/5] 导出 NIfTI 格式...")
    msh_path = (
        Path(output_folder)
        / "mapped_electrodes_simulation"
        / f"{subid}_tes_mapped_opt_head_mesh.msh"
    )
    ti_nifti_path = export_ti_to_nifti(
        msh_path=str(msh_path),
        output_dir=str(output_dir),
        reference=str(subject_dir / "T1.nii.gz"),
        field_name="max_TI",
        prefix=f"{subid}_optimization",
    )

    print("=" * 60)
    print("TI Max 优化完成!")
    print(f"输出目录: {output_folder}")
    print(f"TI NIfTI 文件: {ti_nifti_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
