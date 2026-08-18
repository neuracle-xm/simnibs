"""
Leadfield-based TI 遗传算法优化模块。

模块不接入现有 ``ti_inverse`` 入口，只供同目录下的 demo 进行论文
方法复现和效果验证。
"""

from neuracle.ti_leadfield_optimization.fitness import (
    LeadfieldFitnessEvaluator,
    calculate_region_metrics,
    generate_current_pairs,
)
from neuracle.ti_leadfield_optimization.leadfield import (
    ensure_leadfield,
    load_leadfield,
)
from neuracle.ti_leadfield_optimization.models import (
    CurrentPair,
    ElectrodeChromosome,
    FitnessMetrics,
    GASettings,
    LeadfieldData,
    LeadfieldGADemoConfig,
    LeadfieldGAResult,
    RegionMasks,
)
from neuracle.ti_leadfield_optimization.optimizer import run_genetic_optimization
from neuracle.ti_leadfield_optimization.roi import build_atlas_region_masks

__all__ = [
    "CurrentPair",
    "ElectrodeChromosome",
    "FitnessMetrics",
    "GASettings",
    "LeadfieldData",
    "LeadfieldFitnessEvaluator",
    "LeadfieldGADemoConfig",
    "LeadfieldGAResult",
    "RegionMasks",
    "build_atlas_region_masks",
    "calculate_region_metrics",
    "ensure_leadfield",
    "generate_current_pairs",
    "load_leadfield",
    "run_genetic_optimization",
]
