"""
TI Simulation Module

Temporal Interference (TI) 正向仿真模块，提供从电极对设置到 TI 场计算的全流程功能。
"""

from neuracle.ti_simulation.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)

__all__ = [
    "setup_session",
    "setup_electrode_pair1",
    "setup_electrode_pair2",
    "run_tdcs_simulation",
    "calculate_ti",
]
