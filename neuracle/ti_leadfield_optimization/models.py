"""
Leadfield-based TI 遗传算法优化的数据结构。

该模块只定义 demo 及内部算法需要的配置和结果对象，不改动
``neuracle.parameters`` 中现有业务参数。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from simnibs.mesh_tools.mesh_io import Msh


@dataclass(frozen=True)
class CurrentPair:
    """两路 TI 载波的电流幅值，单位为 mA。"""

    current_a_ma: float
    current_b_ma: float


@dataclass(frozen=True)
class ElectrodeChromosome:
    """由 A+、A-、B+、B- 四个 montage 索引组成的染色体。"""

    electrode_indices: tuple[int, int, int, int]


@dataclass(frozen=True)
class FitnessMetrics:
    """单个六基因染色体的 SimNIBS focality 及诊断指标。"""

    objective: float
    score: float
    roc_distance: float
    roi_sensitivity: float
    non_roi_false_positive_rate: float
    roi_rest_ratio: float
    roi_mean_v_per_m: float
    rest_mean_v_per_m: float
    roi_max_v_per_m: float
    currents: CurrentPair


@dataclass(frozen=True)
class LeadfieldGAResult:
    """遗传算法返回的最优电极、电流和收敛信息。"""

    chromosome: ElectrodeChromosome
    electrode_names: tuple[str, str, str, str]
    metrics: FitnessMetrics
    convergence: tuple[float, ...]
    random_seed: int


@dataclass(frozen=True)
class RegionMasks:
    """与 leadfield element 顺序一致的 ROI、non-ROI/Rest mask 和体积权重。"""

    roi_mask: npt.NDArray[np.bool_]
    rest_mask: npt.NDArray[np.bool_]
    element_volumes: npt.NDArray[np.float64]
    roi_volume: float
    rest_volume: float


@dataclass(frozen=True)
class LeadfieldData:
    """已全量加载到内存的 volumetric TDCS leadfield 及元数据。"""

    values: npt.NDArray[np.float64]
    mesh: Msh
    electrode_names: tuple[str, ...]
    electrode_rows: dict[str, int | None]
    path: Path
    attributes: dict[str, Any]


@dataclass(frozen=True)
class GASettings:
    """工程化遗传算法、独立电流搜索及 focality 参数。"""

    max_num_iteration: int = 30
    population_size: int = 80
    mutation_probability: float = 0.2
    elit_ratio: float = 0.05
    crossover_probability: float = 0.5
    parents_portion: float = 0.3
    crossover_type: str = "uniform"
    max_iteration_without_improv: int | None = 10
    function_timeout_seconds: float = 120.0
    current_min_ma: float = 0.0
    current_max_ma: float = 4.0
    current_step_ma: float = 0.05
    non_roi_threshold_v_per_m: float = 0.1
    roi_threshold_v_per_m: float = 0.2
    random_seed: int = 20220815
    duplicate_electrode_penalty: float = 10000.0


@dataclass(frozen=True)
class LeadfieldGADemoConfig:
    """基于现有头模和独立 atlas ROI/non-ROI 运行验证 demo 的配置。"""

    head_model_id: str
    head_model_dir: Path
    mesh_path: Path
    t1_path: Path
    montage_path: Path
    roi_mask_paths: tuple[Path, ...]
    non_roi_mask_paths: tuple[Path, ...]
    leadfield_dir: Path
    result_dir: Path
    conductivity_config: dict[str, float]
    anisotropy_type: str = "scalar"
    dti_path: Path | None = None
    electrode_radius_mm: float = 6.0
    electrode_thickness_mm: float = 2.0
    n_workers: int = 8
    ga: GASettings = field(default_factory=GASettings)
    run_direct_fem_validation: bool = True
