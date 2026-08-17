"""
TI Caudate Focality Optimization Demo - 左侧尾状核 focality 优化

演示如何使用步骤函数执行 TI Focality 优化。

数据来源: data/m2m_bed84e96-8da9-413f-81ed-b64e7aaa4cbd/

默认配置：
- 电极对1: ElectrodeArrayPair, 半径 6mm, 电流 2mA
- 电极对2: ElectrodeArrayPair, 半径 6mm, 电流 2mA
- ROI: Brainnetome 左侧尾状核 (vCa_L + dCa_L)
- Non-ROI: Brainnetome 左侧 11 区内侧部 (A11m_L)
- Focality 阈值: [0.1, 0.2] V/m
- 电导率: SimNIBS 标准值
- 电导率各向异性类型: scalar
"""

from pathlib import Path
from uuid import uuid4

from neuracle.logger import setup_logging
from neuracle.ti_optimization import (
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils.constants import (
    DATA_ROOT,
    ELECTRODE_RADIUS,
    NEURACLE_DIR,
    PROJECT_ROOT,
)
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs.utils import file_finder


def main() -> None:
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "ti_caudate_focality_optimize_demo"))

    # 设置路径
    subject_id = "bed84e96-8da9-413f-81ed-b64e7aaa4cbd"
    subject_dir = DATA_ROOT / f"m2m_{subject_id}"
    run_id = uuid4().hex[:8]
    output_dir = DATA_ROOT / f"TI_caudate_focality_optimize_{subject_id}_{run_id}"
    roi_dir = (
        NEURACLE_DIR / "atlas" / "standardized" / "BN_Atlas_246_1mm" / "rois"
    )
    roi_mask_paths = [
        roi_dir / "0219_vCa_L.nii.gz",
        roi_dir / "0227_dCa_L.nii.gz",
    ]
    # Modak et al. (2024) 报道左侧尾状核 TI 在中部眶额皮层出现 off-target 激活。
    non_roi_mask_path = roi_dir / "0047_A11m_L.nii.gz"
    sub_files = file_finder.SubjectFiles(subpath=str(subject_dir))
    subid = sub_files.subid
    anisotropy_type = "scalar"

    print("=" * 60)
    print("TI Caudate Focality Optimization: 左侧尾状核 focality 优化")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Electrode radius: {ELECTRODE_RADIUS} mm")

    # 1. 初始化优化结构
    print("\n[1/5] 初始化优化结构...")
    opt = init_optimization(
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        msh_file_path=sub_files.fnamehead,
        anisotropy_type=anisotropy_type,
    )

    # 2. 配置目标函数
    print("[2/5] 配置目标函数 (focality)...")
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[0.1, 0.2],
    )

    # 3. 配置电极对和 ROI
    print("[3/5] 配置电极对和 ROI...")
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        mesh_file_path=sub_files.fnamehead,
        electrode_pair1_center=[[0, 0]],
        electrode_pair2_center=[[0, 0]],
        electrode_radius=[ELECTRODE_RADIUS],
        electrode_current1=[0.002, -0.002],
        electrode_current2=[0.002, -0.002],
        roi_mask_path=[str(path) for path in roi_mask_paths],
        roi_mask_space="mni",
        non_roi_mask_path=str(non_roi_mask_path),
        non_roi_mask_space="mni",
    )

    # 4. 运行优化
    print("[4/5] 运行优化算法...")
    output_folder = run_optimization(
        opt=opt,
        n_workers=8,
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
        prefix=f"{subid}_caudate_optimization",
    )

    print("=" * 60)
    print("TI 左侧尾状核 Focality 优化完成!")
    print(f"输出目录: {output_folder}")
    print(f"TI NIfTI 文件: {ti_nifti_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
