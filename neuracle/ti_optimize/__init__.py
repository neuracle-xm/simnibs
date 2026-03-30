"""
TI Optimization 模块

提供 Temporal Interference 逆向仿真优化功能。
"""

from neuracle.ti_optimize.ti_optimize import (
    export_mz3,
    get_electrode_mapping,
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)

__all__ = [
    "init_optimization",
    "setup_goal",
    "setup_electrodes_and_roi",
    "run_optimization",
    "get_electrode_mapping",
    "export_mz3",
]
