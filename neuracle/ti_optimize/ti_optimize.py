"""
TI Optimization - Temporal Interference 逆向仿真优化

原理：
    1. 基于 SimNIBS 的 TesFlexOptimization 框架
    2. 支持四种目标函数：mean, max, focality, focality_inv
    3. 使用 ElectrodeArrayPair 配置两组电极对
    4. 使用球形 ROI 定义优化区域

用法：
    from neuracle.ti_optimize import run_ti_optimization
    run_ti_optimization(
        subject_dir="path/to/m2m_subid",
        output_dir="path/to/output",
        goal="focality",
        ...
    )
"""

import logging
import os

import numpy as np

from simnibs import mesh_io, opt_struct

logger = logging.getLogger(__name__)


def run_ti_optimization(
    subject_dir: str,
    output_dir: str,
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
    auto_non_roi: bool = False,
    focality_threshold: list[float] | None = None,
    e_postproc: str = "max_TI",
    min_electrode_distance: float = 5.0,
    constrain_electrode_locations: bool = False,
    overlap_factor: float = 1.0,
    map_to_net_electrodes: bool = False,
    run_mapped_electrodes_simulation: bool = False,
    net_electrode_file: str | None = None,
    optimizer: str = "differential_evolution",
    polish: bool = False,
    run_final_simulation: bool = True,
    anisotropy_type: str = "scalar",
    cond: list | None = None,
    fname_tensor: str | None = None,
    disable_SPR_for_volume_roi: bool = True,
    weights: list[float] | None = None,
    track_focality: bool = False,
    detailed_results: bool = False,
    n_workers: int | None = None,
) -> str:
    """
    执行 TI 逆向仿真优化

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}）
    output_dir : str
        优化输出目录
    goal : str
        目标函数类型："mean", "max", "focality", "focality_inv"
    electrode_pair1_center : list[list[float]] | None
        第一组电极阵列中心位置 [[x1, y1], [x2, y2]]，归一化坐标
    electrode_pair2_center : list[list[float]] | None
        第二组电极阵列中心位置 [[x1, y1], [x2, y2]]，归一化坐标
    electrode_radius : list[float] | None
        电极半径（mm）
    electrode_current1 : list[float] | None
        第一组电极电流 [current1, -current1]，单位 A
    electrode_current2 : list[float] | None
        第二组电极电流 [current1, -current1]，单位 A
    roi_center : list[float] | None
        ROI 球形区域中心 [x, y, z]（subject space, mm）
    roi_radius : float | None
        ROI 球形区域半径（mm）
    non_roi_center : list[float] | None
        Non-ROI 球形区域中心（仅 goal="focality" 时需要）
    non_roi_radius : float | None
        Non-ROI 球形区域半径（仅 goal="focality" 时需要）
    auto_non_roi : bool
        是否自动计算非ROI区域 (default: False)
        当为 True 时，自动计算一个包含整个头部的球形区域作为 non-ROI
    focality_threshold : list[float] | None
        focality 阈值 [non_roi_threshold, roi_threshold]（仅 goal="focality" 时需要），单位 V/m
    e_postproc : str
        E-field 后处理方式 (default: "max_TI")
    min_electrode_distance : float
        电极最小距离（mm）(default: 5.0)
    constrain_electrode_locations : bool
        是否限制电极阵列位置在额、顶、枕区 (default: False)
    overlap_factor : float
        允许 lambda 区域的重叠因子 (default: 1.0)
    map_to_net_electrodes : bool
        是否将优化电极映射到 EEG 网格最近位置 (default: False)
    run_mapped_electrodes_simulation : bool
        是否用映射位置运行仿真 (default: False, 需 map_to_net_electrodes=True)
    net_electrode_file : str | None
        EEG 电极位置 CSV 文件路径 (default: None)
    optimizer : str
        优化算法 (default: "differential_evolution")
    polish : bool
        是否使用 L-BFGS-B 细化 (default: False)
    run_final_simulation : bool
        是否运行最终仿真 (default: True)
    anisotropy_type : str
        各向异性类型 (default: "scalar")
        可选值：
        - "scalar": 标量电导率（各向同性）
        - "vector": 向量电导率（仅白质各向异性）
        - "tensor": 张量电导率（基于 DWI 张量）
    cond : list, optional
        电导率列表，默认为标准电导率 (standard_cond)
        每个元素为 (tissue_index, conductivity) 元组
    fname_tensor : str, optional
        DTI 张量文件路径 (.nii.gz)，仅当 anisotropy_type="tensor" 时需要
    disable_SPR_for_volume_roi : bool
        是否对体素 ROI 禁用 SPR 插值 (default: True)
    weights : list[float] | None
        ROI 特定权重 (default: None)
    track_focality : bool
        是否追踪 focality 变化 (default: False)
    detailed_results : bool
        是否写入详细结果 (default: False)
    n_workers : int | None
        并行计算的 CPU 核心数 (default: None, 使用所有可用核心)，传递给 opt.run(cpus=)

    Returns
    -------
    str
        优化结果输出目录
    """
    logger.info("开始 TI 逆向仿真优化")
    logger.info("Subject 目录: %s", subject_dir)
    logger.info("输出目录: %s", output_dir)
    logger.info("目标函数: %s", goal)

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

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化优化结构
    opt = opt_struct.TesFlexOptimization()
    opt.subpath = subject_dir
    opt.output_folder = output_dir
    opt.open_in_gmsh = False  # 禁止自动打开 Gmsh

    # 配置目标函数
    opt.goal = goal
    opt.e_postproc = e_postproc

    # 配置 focality 阈值
    if goal in ["focality", "focality_inv"]:
        if focality_threshold is None:
            focality_threshold = [0.1, 0.2]
        opt.threshold = focality_threshold

    # 配置其他优化选项
    opt.min_electrode_distance = min_electrode_distance
    opt.constrain_electrode_locations = constrain_electrode_locations
    opt.overlap_factor = overlap_factor
    opt.map_to_net_electrodes = map_to_net_electrodes
    opt.run_mapped_electrodes_simulation = run_mapped_electrodes_simulation
    if net_electrode_file is not None:
        opt.net_electrode_file = net_electrode_file
    opt.optimizer = optimizer
    opt.polish = polish
    opt.run_final_electrode_simulation = run_final_simulation
    opt.anisotropy_type = anisotropy_type
    logger.info("电导率各向异性类型: %s", anisotropy_type)

    # 设置电导率值
    if cond is not None:
        opt.cond = cond
        logger.info("使用自定义电导率值")
    else:
        logger.info("使用标准电导率值 (standard_cond)")

    # 设置 DTI 张量文件（用于 tensor 各向异性）
    if fname_tensor is not None:
        opt.fname_tensor = fname_tensor
        logger.info("DTI 张量文件: %s", fname_tensor)

    opt.disable_SPR_for_volume_roi = disable_SPR_for_volume_roi
    if weights is not None:
        opt.weights = weights
    opt.track_focality = track_focality
    opt.detailed_results = detailed_results

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

    # 配置 Non-ROI（仅 focality 需要）
    if goal in ["focality", "focality_inv"]:
        if auto_non_roi:
            # 自动计算非ROI区域：读取网格计算头部外接球
            logger.info("自动计算非ROI区域")
            msh_path = os.path.join(subject_dir, os.listdir(subject_dir)[0])
            for f in os.listdir(subject_dir):
                if f.endswith(".msh"):
                    msh_path = os.path.join(subject_dir, f)
                    break
            mesh = mesh_io.read_msh(msh_path)
            # 计算头部节点中心的最大距离作为外接球
            head_nodes = mesh.nodes[mesh.nodes.tag == 1006]
            center = head_nodes.center_of_mass()
            distances = np.linalg.norm(head_nodes.value - center, axis=1)
            non_roi_center_auto = center.tolist()
            non_roi_radius_auto = float(distances.max()) + 10.0
            logger.info("自动计算 Non-ROI: 中心=%s, 半径=%.1f", non_roi_center_auto, non_roi_radius_auto)
            non_roi = opt.add_roi()
            non_roi.method = "surface"
            non_roi.surface_type = "central"
            non_roi.roi_sphere_center_space = "subject"
            non_roi.roi_sphere_center = non_roi_center_auto
            non_roi.roi_sphere_radius = non_roi_radius_auto
            non_roi.roi_sphere_operator = ["difference"]
        else:
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

    # 运行优化
    logger.info("开始优化...")
    opt.run(cpus=n_workers)
    logger.info("优化完成，结果保存在: %s", output_dir)

    return output_dir
