"""
TI Optimization - Temporal Interference 逆向仿真优化

原理：
    1. 基于 SimNIBS 的 TesFlexOptimization 框架
    2. 支持四种目标函数：mean, max, focality, focality_inv
    3. 使用 ElectrodeArrayPair 配置两组电极对
    4. 使用球形 ROI 定义优化区域

用法：
    from neuracle.ti_optimize import (
        init_optimization,
        setup_goal,
        setup_electrodes_and_roi,
        run_optimization,
        export_mz3,
    )

    # 1. 初始化
    opt = init_optimization(subject_dir, output_dir)

    # 2. 配置目标函数
    setup_goal(opt, goal="focality", focality_threshold=[0.1, 0.2])

    # 3. 配置电极和 ROI
    setup_electrodes_and_roi(opt, subject_dir, goal, roi_center, roi_radius, ...)

    # 4. 运行优化
    run_optimization(opt, n_workers=24)

    # 5. 导出结果
    mz3_path = export_mz3(output_dir)
"""

import json
import logging
import os

from neuracle.mesh_tools import msh_to_mz3
from simnibs import opt_struct

logger = logging.getLogger(__name__)


def init_optimization(
    subject_dir: str,
    output_dir: str,
    msh_file_path: str | None = None,
    anisotropy_type: str = "scalar",
    cond: list | None = None,
    fname_tensor: str | None = None,
) -> opt_struct.TesFlexOptimization:
    """
    初始化优化结构

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}）
    msh_file_path : str, optional
        头模网格文件路径；如果提供，则优先使用该 mesh 文件
    output_dir : str
        优化输出目录
    anisotropy_type : str
        各向异性类型 (default: "scalar")
    cond : list, optional
        电导率列表
    fname_tensor : str, optional
        DTI 张量文件路径

    Returns
    -------
    opt_struct.TesFlexOptimization
        初始化好的优化对象
    """
    opt = opt_struct.TesFlexOptimization()
    opt.subpath = subject_dir
    opt.fn_mesh = msh_file_path
    opt.output_folder = output_dir
    opt.open_in_gmsh = False
    opt.anisotropy_type = anisotropy_type
    logger.info("电导率各向异性类型: %s", anisotropy_type)

    if cond is not None:
        opt.cond = cond
        logger.info("使用自定义电导率值")
    else:
        logger.info("使用标准电导率值 (standard_cond)")

    if fname_tensor is not None:
        opt.fname_tensor = fname_tensor
        logger.info("DTI 张量文件: %s", fname_tensor)
    if msh_file_path is not None:
        logger.info("头模网格文件: %s", msh_file_path)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("优化结构初始化完成")

    return opt


def setup_goal(
    opt: opt_struct.TesFlexOptimization,
    goal: str,
    e_postproc: str = "max_TI",
    focality_threshold: list[float] | None = None,
    min_electrode_distance: float = 5.0,
    constrain_electrode_locations: bool = False,
    overlap_factor: float = 1.0,
    map_to_net_electrodes: bool = True,
    run_mapped_electrodes_simulation: bool = True,
    net_electrode_file: str | None = None,
    optimizer: str = "differential_evolution",
    polish: bool = False,
    run_final_simulation: bool = True,
    disable_SPR_for_volume_roi: bool = True,
    weights: list[float] | None = None,
    track_focality: bool = False,
    detailed_results: bool = False,
) -> None:
    """
    配置目标函数和优化选项

    Parameters
    ----------
    opt : opt_struct.TesFlexOptimization
        优化对象
    goal : str
        目标函数类型："mean", "max", "focality", "focality_inv"
    e_postproc : str
        E-field 后处理方式 (default: "max_TI")
    focality_threshold : list[float], optional
        focality 阈值
    min_electrode_distance : float
        电极最小距离（mm）
    constrain_electrode_locations : bool
        是否限制电极阵列位置
    overlap_factor : float
        允许 lambda 区域的重叠因子
    map_to_net_electrodes : bool
        是否将优化电极映射到 EEG 网格
    run_mapped_electrodes_simulation : bool
        是否用映射位置运行仿真
    net_electrode_file : str, optional
        EEG 电极位置 CSV 文件路径
    optimizer : str
        优化算法
    polish : bool
        是否使用 L-BFGS-B 细化
    run_final_simulation : bool
        是否运行最终仿真
    disable_SPR_for_volume_roi : bool
        是否对体素 ROI 禁用 SPR 插值
    weights : list[float], optional
        ROI 特定权重
    track_focality : bool
        是否追踪 focality 变化
    detailed_results : bool
        是否写入详细结果
    """
    opt.goal = goal
    opt.e_postproc = e_postproc

    if goal in ["focality", "focality_inv"]:
        if focality_threshold is None:
            focality_threshold = [0.1, 0.2]
        opt.threshold = focality_threshold

    opt.min_electrode_distance = min_electrode_distance
    opt.constrain_electrode_locations = constrain_electrode_locations
    opt.overlap_factor = overlap_factor
    opt.map_to_net_electrodes = map_to_net_electrodes
    opt.run_mapped_electrodes_simulation = run_mapped_electrodes_simulation
    opt.net_electrode_file = net_electrode_file
    opt.optimizer = optimizer
    opt.polish = polish
    opt.run_final_electrode_simulation = run_final_simulation
    opt.disable_SPR_for_volume_roi = disable_SPR_for_volume_roi
    if weights is not None:
        opt.weights = weights
    opt.track_focality = track_focality
    opt.detailed_results = detailed_results

    logger.info("目标函数配置完成: %s", goal)


def setup_electrodes_and_roi(
    opt: opt_struct.TesFlexOptimization,
    goal: str,
    electrode_pair1_center: list[list[float]] | None = None,
    electrode_pair2_center: list[list[float]] | None = None,
    electrode_radius: list[float] | None = None,
    electrode_current1: list[float] | None = None,
    electrode_current2: list[float] | None = None,
    roi_center: list[float] | None = None,
    roi_radius: float | None = None,
    non_roi_center: list[float] | None = None,
    non_roi_radius: float | None = None,
) -> None:
    """
    配置电极对和 ROI

    Parameters
    ----------
    opt : opt_struct.TesFlexOptimization
        优化对象
    goal : str
        目标函数类型
    electrode_pair1_center : list[list[float]], optional
        第一组电极阵列中心位置
    electrode_pair2_center : list[list[float]], optional
        第二组电极阵列中心位置
    electrode_radius : list[float], optional
        电极半径
    electrode_current1 : list[float], optional
        第一组电极电流
    electrode_current2 : list[float], optional
        第二组电极电流
    roi_center : list[float], optional
        ROI 球形区域中心
    roi_radius : float, optional
        ROI 球形区域半径
    non_roi_center : list[float], optional
        Non-ROI 球形区域中心
    non_roi_radius : float, optional
        Non-ROI 球形区域半径
    """
    # 默认值
    if electrode_pair1_center is None:
        electrode_pair1_center = [[0, 0]]
    if electrode_pair2_center is None:
        electrode_pair2_center = [[0, 0]]
    if electrode_radius is None:
        electrode_radius = [10]
    if electrode_current1 is None:
        electrode_current1 = [0.002, -0.002]
    if electrode_current2 is None:
        electrode_current2 = [0.002, -0.002]
    if roi_center is None:
        roi_center = [-41.0, -13.0, 66.0]
    if roi_radius is None:
        roi_radius = 20.0

    # 配置第一组电极对
    logger.info("配置第一组电极对")
    electrode_layout1 = opt.add_electrode_layout("ElectrodeArrayPair")
    electrode_layout1.center = electrode_pair1_center
    electrode_layout1.radius = electrode_radius
    electrode_layout1.current = electrode_current1

    # 配置第二组电极对
    logger.info("配置第二组电极对")
    electrode_layout2 = opt.add_electrode_layout("ElectrodeArrayPair")
    electrode_layout2.center = electrode_pair2_center
    electrode_layout2.radius = electrode_radius
    electrode_layout2.current = electrode_current2

    # 配置 ROI
    logger.info("配置 ROI: 中心=%s, 半径=%s", roi_center, roi_radius)
    roi = opt.add_roi()
    roi.method = "surface"
    roi.surface_type = "central"
    roi.roi_sphere_center_space = "subject"
    roi.roi_sphere_center = roi_center
    roi.roi_sphere_radius = roi_radius

    # 配置 Non-ROI（仅 focality 需要）- 使用 roi_sphere_operator="difference" 自动计算
    if goal in ["focality", "focality_inv"]:
        if non_roi_center is None:
            non_roi_center = roi_center
        if non_roi_radius is None:
            non_roi_radius = 25.0
        logger.info("配置 Non-ROI: 中心=%s, 半径=%s", non_roi_center, non_roi_radius)
        non_roi = opt.add_roi()
        non_roi.method = "surface"
        non_roi.surface_type = "central"
        non_roi.roi_sphere_center_space = "subject"
        non_roi.roi_sphere_center = non_roi_center
        non_roi.roi_sphere_radius = non_roi_radius
        non_roi.roi_sphere_operator = ["difference"]

    logger.info("电极对和 ROI 配置完成")


def run_optimization(
    opt: opt_struct.TesFlexOptimization,
    n_workers: int | None = None,
) -> str:
    """
    运行优化算法

    Parameters
    ----------
    opt : opt_struct.TesFlexOptimization
        优化对象
    n_workers : int, optional
        并行计算的 CPU 核心数

    Returns
    -------
    str
        优化结果输出目录
    """
    logger.info("开始优化...")
    opt.run(cpus=n_workers)
    logger.info("优化完成，结果保存在: %s", opt.output_folder)
    return opt.output_folder


def export_mz3(
    output_dir: str,
    surface_type: str = "central",
) -> str:
    """
    导出优化结果到 MZ3 格式

    Parameters
    ----------
    output_dir : str
        优化输出目录
    surface_type : str
        表面类型 (default: "central")

    Returns
    -------
    str
        MZ3 文件路径
    """

    # 查找优化结果 mesh 文件
    msh_path = None
    for f in os.listdir(output_dir):
        if f.endswith(".msh"):
            msh_path = os.path.join(output_dir, f)
            break

    if msh_path is None:
        raise FileNotFoundError(f"在 {output_dir} 中未找到 .msh 文件")

    logger.info("导出优化结果到 MZ3 格式...")
    mz3_path = msh_to_mz3(
        msh_path=msh_path,
        output_dir=output_dir,
        surface_type=surface_type,
        field_name="max_TI",
    )
    logger.info("MZ3 导出完成: %s", mz3_path)
    return mz3_path


def get_electrode_mapping(output_dir: str) -> tuple[list[str], list[str]]:
    """
    获取优化电极映射到的 EEG 电极标签

    Parameters
    ----------
    output_dir : str
        优化输出目录

    Returns
    -------
    tuple[list[str], list[str]]
        (electrode_A, electrode_B) - 两组电极的映射标签列表

    Raises
    ------
    FileNotFoundError
        当 electrode_mapping.json 不存在时
    """
    mapping_path = os.path.join(output_dir, "electrode_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"电极映射文件不存在: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    mapped_labels = mapping.get("mapped_labels", [])
    n = len(mapped_labels) // 2
    electrode_A = mapped_labels[:n]
    electrode_B = mapped_labels[n:]

    logger.info(
        "电极映射结果: electrode_A=%s, electrode_B=%s", electrode_A, electrode_B
    )
    return electrode_A, electrode_B
