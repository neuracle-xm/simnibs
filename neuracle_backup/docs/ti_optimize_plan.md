# TI 优化方案

## 概述

TI (Temporal Interference) 优化模块基于 SimNIBS 的 TesFlexOptimization 框架，提供四种优化目标函数，用于优化经颅电刺激中的电极配置。

## 优化目标

| 目标函数 | 说明 |
|---------|------|
| `mean` | 最大化 ROI 内平均电场 |
| `max` | 最小化 ROI 内最大电场 |
| `focality` | 最大化 ROI 内 focality（聚焦度） |
| `focality_inv` | 反向 focality 优化 |

## 原理

1. ** ElectrodeArrayPair 配置**：使用双电极阵列对，每组电极对阵列中心、半径和电流可独立配置
2. **球形 ROI**：使用球形区域定义优化目标区域
3. **Focality 优化**：额外定义 Non-ROI 区域，计算 ROI 与 Non-ROI 的电场比率
4. **EEG 电极映射**：支持将优化后的电极位置映射到标准 EEG 网格

## 使用方法

```python
from neuracle.ti_optimize import (
    init_optimization,
    setup_goal,
    setup_electrodes_and_roi,
    run_optimization,
)

# 1. 初始化优化结构
opt = init_optimization(subject_dir, output_dir)

# 2. 配置目标函数
setup_goal(opt, goal="focality", focality_threshold=[0.1, 0.2])

# 3. 配置电极和 ROI
setup_electrodes_and_roi(
    opt,
    goal="focality",
    roi_center=[-41.0, -13.0, 66.0],
    roi_radius=20.0,
    electrode_pair1_center=[[0, 0]],
    electrode_pair2_center=[[0, 0]],
    electrode_radius=[10],
)

# 4. 运行优化
output_dir = run_optimization(opt, n_workers=24)
```

## 导出 API

```python
from neuracle.ti_optimize import (
    init_optimization,       # 初始化优化结构
    setup_goal,             # 配置目标函数
    setup_electrodes_and_roi, # 配置电极和 ROI
    run_optimization,        # 运行优化
    get_electrode_mapping,   # 获取电极映射结果
)
```

## 参数说明

### 必需参数

- `subject_dir`: Subject 目录（m2m_{subid}）
- `output_dir`: 优化输出目录
- `goal`: 目标函数类型

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `roi_center` | ROI 球形区域中心 [x, y, z] | [-41.0, -13.0, 66.0] |
| `roi_radius` | ROI 球形区域半径（mm） | 20.0 |
| `electrode_radius` | 电极半径（mm） | [10] |
| `electrode_current1` | 第一组电极电流 [A, -A] | [0.002, -0.002] |
| `electrode_current2` | 第二组电极电流 [A, -A] | [0.002, -0.002] |
| `n_workers` | 并行计算的 CPU 核心数 | None |
| `map_to_net_electrodes` | 是否映射到 EEG 网格 | False |
| `net_electrode_file` | EEG 电极位置 CSV 文件 | None |

## 文件结构

```
neuracle/ti_optimize/
├── __init__.py          # 模块初始化
└── ti_optimize.py       # TI 优化函数

neuracle/demo/
├── ti_mean_optimize_demo.py          # Mean 优化示例
├── ti_max_optimize_demo.py           # Max 优化示例
├── ti_focality_optimize_demo.py     # Focality 优化示例
└── ti_focality_inv_optimize_demo.py # Focality Inverse 优化示例
```
