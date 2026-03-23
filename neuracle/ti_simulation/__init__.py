"""
TI Simulation Module

Temporal Interference (TI) 正向仿真模块，提供从电极对设置到 TI 场计算的全流程功能。
"""

from neuracle.ti_simulation.ti_simulation import (
    run_ti_simulation,
    run_ti_forward_simulation,
    calculate_ti_envelope,
)

__all__ = [
    "run_ti_simulation",
    "run_ti_forward_simulation",
    "calculate_ti_envelope",
]
