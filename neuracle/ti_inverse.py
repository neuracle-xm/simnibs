"""
TI 逆向优化入口

提供 TI 逆向优化流程的入口函数 run_ti_inverse()。

逆向仿真是"已知目标电场分布，反求电极电流配置"的优化过程。
需要迭代优化以找到最优电流配置。

用法:
    from neuracle.ti_inverse import run_ti_inverse
    from neuracle.parameters.schemas import ROIParam, MNIParam

    run_ti_inverse(
        dir_path="m2m_ernie",
        montage="EEG10-10_Neuroelectrics",
        current_A=[1.0, -1.0],
        current_B=[1.0, -1.0],
        roi_type="mni_pos",
        roi_param=ROIParam(mni_param=MNIParam(center=[-50, -30, 40], radius=10)),
        target_threshold=0.2,
        conductivity_config={"WM": 0.126, "GM": 0.275, ...},
        anisotropy=AnisotropyType.SCALAR,
    )
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Literal

# 支持直接执行当前脚本文件时使用 `from neuracle...` 绝对导入
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuracle.atlas.standardized import get_standardized_roi_path
from neuracle.logger import setup_logging
from neuracle.parameters.converter import dict_to_inverse_params
from neuracle.parameters.schemas import AnisotropyType, ROIParam
from neuracle.parameters.validator import ValidationError, validate_inverse_params
from neuracle.storage.paths import (
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    reset_task_output_dir,
)
from neuracle.ti_optimization import (
    get_electrode_mapping,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils import (
    NON_ROI_THRESHOLD,
    cond_dict_to_list,
    find_montage_file,
)
from neuracle.utils.cli_utils import (
    add_conductivity_argument,
    build_atlas_roi_param,
    build_mni_roi_param,
    parse_anisotropy,
    parse_conductivity_specs,
)
from neuracle.utils.constants import (
    EXIT_INVALID_ARGS,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
)
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger(__name__)


def run_ti_inverse(
    dir_path: str,
    montage: str,
    current_A: list[float],
    current_B: list[float],
    roi_type: Literal["atlas", "mni_pos"],
    roi_param: ROIParam,
    target_threshold: float,
    conductivity_config: dict[str, float],
    anisotropy: AnisotropyType,
    DTI_file_path: str | None = None,
    n_workers: int = 8,
    debug: bool = False,
) -> None:
    """
    运行 TI 逆向优化，生成失败则抛出异常。

    流程步骤：
    步骤 0：初始化与检查，解析 ROI 参数
    步骤 1：优化器初始化
    步骤 2：目标设置
    步骤 3：电极与 ROI 配置
    步骤 4：执行优化
    步骤 5：电极映射获取
    步骤 6：NIfTI 导出
    步骤 7：清理

    Parameters
    ----------
    dir_path : str
        头模目录名
    montage : str
        电极导联名称
    current_A : list[float]
        电极组 A 初始电流值（mA）
    current_B : list[float]
        电极组 B 初始电流值（mA）
    roi_type : Literal["atlas", "mni_pos"]
        ROI 类型
    roi_param : ROIParam
        ROI 参数配置
    target_threshold : float
        目标电场强度阈值
    conductivity_config : dict[str, float]
        组织电导率配置
    anisotropy : AnisotropyType
        各向异性类型
    DTI_file_path : str, optional
        DTI 张量文件路径
    n_workers : int
        并行工作进程数
    debug : bool
        调试模式，为 True 时不清理临时输出目录
    """
    roi_info = (
        f"atlas={roi_param.atlas_param}"
        if roi_param.atlas_param
        else f"mni={roi_param.mni_param}"
        if roi_param.mni_param
        else "None"
    )
    logger.info(
        "开始 TI 逆向优化: dir_path=%s, montage=%s, anisotropy=%s, "
        "current_A=%s, current_B=%s, roi_type=%s, roi=%s, "
        "target_threshold=%s, conductivity=%s",
        dir_path,
        montage,
        anisotropy,
        current_A,
        current_B,
        roi_type,
        roi_info,
        target_threshold,
        conductivity_config,
    )

    # ===== 步骤 0：初始化与检查 =====
    subject_dir = get_subject_dir(dir_path)
    if not subject_dir.is_dir():
        raise FileNotFoundError(
            f"subject 目录不存在: {subject_dir}。请先执行头模生成任务。"
        )

    task_id = "inverse"
    output_dir = get_task_output_dir(dir_path, "TI_optimization", task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not debug:
        reset_task_output_dir(str(output_dir))

    net_electrode_file = find_montage_file(str(subject_dir), montage)
    mesh_path = get_model_mesh_path(dir_path)

    roi_center = None
    roi_radius = None
    roi_center_space = "subject"
    roi_mask_path = None
    roi_mask_space = None
    focality_threshold = target_threshold

    if roi_type == "atlas" and roi_param.atlas_param:
        roi_mask_path = str(
            get_standardized_roi_path(
                roi_param.atlas_param.name,
                roi_param.atlas_param.area,
            )
        )
        if not os.path.exists(roi_mask_path):
            raise FileNotFoundError(
                f"标准化 ROI 不存在: {roi_mask_path}。请先运行 atlas 标准化和 ROI 生成脚本。"
            )
        roi_mask_space = "mni"
    elif roi_type == "mni_pos" and roi_param.mni_param:
        roi_center = roi_param.mni_param.center
        roi_radius = roi_param.mni_param.radius
        roi_center_space = "mni"

    # ===== 步骤 1：优化器初始化 =====
    opt = init_optimization(
        subject_dir=str(subject_dir),
        msh_file_path=str(mesh_path),
        output_dir=str(output_dir),
        anisotropy_type=anisotropy,
        cond=cond_dict_to_list(conductivity_config),
        fname_tensor=DTI_file_path,
    )
    logger.info("优化器初始化完成")

    # ===== 步骤 2：目标设置 =====
    setup_goal(
        opt=opt,
        goal="focality",
        focality_threshold=[focality_threshold, NON_ROI_THRESHOLD],
        net_electrode_file=net_electrode_file,
    )
    logger.info("优化目标设置完成")

    # ===== 步骤 3：电极与 ROI 配置 =====
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality",
        mesh_file_path=mesh_path,
        electrode_current1=[c / 1000 for c in current_A],
        electrode_current2=[c / 1000 for c in current_B],
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_center_space=roi_center_space,
        roi_mask_path=roi_mask_path,
        roi_mask_space=roi_mask_space,
    )
    logger.info("电极与 ROI 配置完成")

    # ===== 步骤 4：执行优化 =====
    run_optimization(opt=opt, n_workers=n_workers)
    logger.info("优化执行完成")

    # ===== 步骤 5：电极映射获取 =====
    electrode_A, electrode_B = get_electrode_mapping(output_dir=str(output_dir))
    logger.info("电极映射获取完成: A=%s, B=%s", electrode_A, electrode_B)

    # ===== 步骤 6：NIfTI 导出 =====
    subid = re.search("m2m_(.+)", dir_path).group(1)
    msh_name = f"{subid}_tes_mapped_opt_head_mesh.msh"
    msh_path = str(output_dir / "mapped_electrodes_simulation" / msh_name)
    # 从 subject 目录查找 T1 文件作为参考空间
    t1_candidates = list(subject_dir.glob("T1*.nii.gz"))
    reference_file = str(t1_candidates[0]) if t1_candidates else ""
    # 导出到 subject 目录，避免被清理步骤删除
    ti_nifti_path = export_ti_to_nifti(
        msh_path=msh_path,
        output_dir=str(subject_dir),
        reference=reference_file,
        field_name="max_TI",
        prefix=f"{subid}_optimization",
    )
    logger.info("NIfTI 导出完成: %s", ti_nifti_path)

    # ===== 步骤 7：清理 =====
    if not debug:
        shutil.rmtree(output_dir, ignore_errors=True)

    logger.info("TI 逆向优化完成")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建 TI 逆向优化命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        命令行解析器
    """
    parser = argparse.ArgumentParser(
        prog="python neuracle/ti_inverse.py",
        description="运行 TI 逆向优化流程",
    )
    parser.add_argument("dir_path", help="头模目录名，例如 m2m_ernie")
    parser.add_argument("montage", help="电极导联名称或 CSV 路径")
    parser.add_argument(
        "--current-a",
        nargs="+",
        type=float,
        required=True,
        metavar="CURRENT",
        help="电极组 A 初始电流列表，单位 mA",
    )
    parser.add_argument(
        "--current-b",
        nargs="+",
        type=float,
        required=True,
        metavar="CURRENT",
        help="电极组 B 初始电流列表，单位 mA",
    )
    parser.add_argument(
        "--roi-type",
        required=True,
        choices=["atlas", "mni_pos"],
        help="ROI 类型",
    )
    parser.add_argument("--atlas-name", help="atlas 模式下的图谱名称")
    parser.add_argument("--atlas-area", help="atlas 模式下的区域名称")
    parser.add_argument(
        "--mni-center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="mni_pos 模式下的 MNI 中心坐标",
    )
    parser.add_argument("--mni-radius", type=float, help="mni_pos 模式下的 ROI 半径")
    parser.add_argument(
        "--target-threshold", type=float, required=True, help="目标电场强度阈值"
    )
    add_conductivity_argument(parser)
    parser.add_argument(
        "--anisotropy",
        default=AnisotropyType.SCALAR.value,
        choices=[item.value for item in AnisotropyType],
        help="各向异性类型，默认 scalar",
    )
    parser.add_argument(
        "--dti-file-path", dest="DTI_file_path", help="DTI 张量文件路径"
    )
    parser.add_argument("--n-workers", type=int, default=8, help="并行工作进程数")
    parser.add_argument("--debug", action="store_true", help="调试模式，不清理临时目录")
    return parser


def build_roi_param_dict(args: argparse.Namespace) -> dict[str, object]:
    """
    根据命令行参数构造 ROI 参数字典。

    Parameters
    ----------
    args : argparse.Namespace
        命令行解析结果

    Returns
    -------
    dict[str, object]
        ROI 参数字典
    """
    if args.roi_type == "atlas":
        if not args.atlas_name or not args.atlas_area:
            raise ValueError(
                "roi_type=atlas 时必须同时提供 --atlas-name 和 --atlas-area"
            )
        return build_atlas_roi_param(args.atlas_name, args.atlas_area)
    if not args.mni_center or args.mni_radius is None:
        raise ValueError("roi_type=mni_pos 时必须提供 --mni-center 和 --mni-radius")
    return build_mni_roi_param(args.mni_center, args.mni_radius)


def main(argv: list[str] | None = None) -> int:
    """
    TI 逆向优化命令行入口。

    Parameters
    ----------
    argv : list[str] | None
        命令行参数列表，默认读取 sys.argv

    Returns
    -------
    int
        进程退出码
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        params_dict = {
            "dir_path": args.dir_path,
            "T1_file_path": "",
            "montage": args.montage,
            "current_A": args.current_a,
            "current_B": args.current_b,
            "roi_type": args.roi_type,
            "roi_param": build_roi_param_dict(args),
            "target_threshold": args.target_threshold,
            "conductivity_config": parse_conductivity_specs(args.conductivity),
            "anisotropy": parse_anisotropy(args.anisotropy),
            "DTI_file_path": args.DTI_file_path,
        }
        setup_logging()
        validate_inverse_params(params_dict)
        params = dict_to_inverse_params(params_dict)
        run_ti_inverse(
            dir_path=params.dir_path,
            montage=params.montage,
            current_A=params.current_A,
            current_B=params.current_B,
            roi_type=params.roi_type,
            roi_param=params.roi_param,
            target_threshold=params.target_threshold,
            conductivity_config=params.conductivity_config,
            anisotropy=params.anisotropy,
            DTI_file_path=params.DTI_file_path,
            n_workers=args.n_workers,
            debug=args.debug,
        )
    except (ValidationError, ValueError) as exc:
        logger.error("TI 逆向优化参数校验失败: %s", exc)
        return EXIT_INVALID_ARGS
    except Exception:
        logger.exception("TI 逆向优化执行失败")
        return EXIT_RUNTIME_ERROR
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
