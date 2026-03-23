# TI 正向仿真方案

## 1. 概述

### 1.1 目标
在 `neuracle/ti_simulation/` 下创建 TI（Temporal Interference，时间干涉）正向仿真模块，提供从电极对设置到 TI 场计算的全流程功能。

### 1.2 参考文档
- SimNIBS 官方文档：https://simnibs.github.io/simnibs/build/html/documentation/documentation.html
- TI.py 示例：`simnibs/examples/simulations/TI.py`
- TI_utils.py 工具函数：`simnibs/utils/TI_utils.py`

---

## 2. 文件结构

```
neuracle/
├── ti_simulation/
│   ├── __init__.py              # 包初始化，导出公共 API
│   └── ti_simulation.py          # 主函数：run_ti_simulation, run_ti_forward_simulation, calculate_ti_envelope
└── demo/
    └── ti_simulation_demo.py     # 示例代码
```

---

## 3. 函数设计

### 3.1 `run_ti_simulation()`

**功能**：执行 TI 正向仿真（完整流程：仿真 + TI 计算）。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `subject_dir` | str | Subject 目录（m2m_{subid}） |
| `output_dir` | str | 仿真输出目录 |
| `electrode_pair1` | list[str] | 第一个电极对 [elec1_name, elec2_name]，如 ['F5', 'P5'] |
| `electrode_pair2` | list[str] | 第二个电极对 [elec1_name, elec2_name]，如 ['F6', 'P6'] |
| `current1` | float | 第一组电极对电流强度，单位 A（default: 0.001，即 1mA） |
| `current2` | float | 第二组电极对电流强度，单位 A（default: 0.001，即 1mA） |
| `electrode_shape` | str | 电极形状，可选 'ellipse', 'rect', 'custom'（default: 'ellipse'） |
| `electrode_dimensions` | list[float] \| None | 电极尺寸 [width, height]，单位 mm（default: [40, 40]） |
| `electrode_thickness` | float | 电极厚度，单位 mm（default: 2.0） |
| `n_workers` | int | 并行工作进程数（default: 1） |

**返回值**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `ti_mesh_path` | str | TI 结果网格文件路径 |
| `ti_max` | np.ndarray | TI 最大调制振幅数组 |

**输出文件**：
- `{output_dir}/{subid}_TDCS_1_scalar.msh` - 第一对电极仿真结果
- `{output_dir}/{subid}_TDCS_2_scalar.msh` - 第二对电极仿真结果
- `{output_dir}/TI.msh` - TI 场可视化结果

### 3.2 `run_ti_forward_simulation()`

**功能**：执行两组电极的 TDCS 正向仿真。

**参数**：与 `run_ti_simulation()` 相同，但不需要 `n_workers`（直接传给 `run_simnibs`）。

**返回值**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `mesh1_path` | str | 第一组电极仿真结果网格路径 |
| `mesh2_path` | str | 第二组电极仿真结果网格路径 |

### 3.3 `calculate_ti_envelope()`

**功能**：从两个仿真结果计算 TI 包络。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `mesh1_path` | str | 第一组电极仿真结果网格路径 |
| `mesh2_path` | str | 第二组电极仿真结果网格路径 |
| `output_dir` | str | 输出目录 |

**返回值**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `ti_mesh_path` | str | TI 结果网格文件路径 |
| `ti_max` | np.ndarray | TI 最大调制振幅数组 |

### 3.4 `_setup_electrode_pair()`

**功能**：配置单个电极对。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `session` | sim_struct.SESSION | SimNIBS 会话对象 |
| `electrode_pair` | list[str] | 电极对 [elec1_name, elec2_name] |
| `currents` | list[float] | 电流配置 [current1, current2] |
| `electrode_shape` | str | 电极形状 |
| `electrode_dimensions` | list[float] | 电极尺寸 [width, height] |
| `electrode_thickness` | float | 电极厚度 |

**返回值**：`sim_struct.TDCSLIST` - 配置好的 TDCS 列表对象

---

## 4. 实现步骤

### 步骤 1：创建文件夹和基础文件
```
neuracle/ti_simulation/
├── __init__.py
├── ti_simulation.py
```

### 步骤 2：实现 `run_ti_forward_simulation()` 主函数
- 导入必要的模块（sim_struct, run_simnibs, mesh_io, ElementTags, TI_utils）
- 设置 SESSION 参数（`open_in_gmsh = False` 禁止自动打开 Gmsh）
- 配置两个电极对
- 调用 `run_simnibs(S, cpus=n_workers)` 执行仿真
- 返回两组仿真结果路径

### 步骤 3：实现 `calculate_ti_envelope()` 函数
- 读取网格文件
- 裁剪网格（去除电极元素）
- 使用 TI_utils 计算 TI 最大调制振幅
- 生成可视化输出

### 步骤 4：实现 `run_ti_simulation()` 组合函数
- 调用 `run_ti_forward_simulation()` 执行仿真
- 调用 `calculate_ti_envelope()` 计算 TI 包络
- 返回最终结果

### 步骤 5：实现辅助函数
- `_setup_electrode_pair()` - 电极对配置

### 步骤 6：添加日志记录
- 使用 `logger = logging.getLogger(__name__)`
- 在关键步骤记录日志（仿真开始、电极配置、TI 计算完成等）
- 错误时记录 error 日志

### 步骤 7：创建示例 demo
```
neuracle/demo/ti_simulation_demo.py
```

---

## 5. 关键代码片段

### 5.1 会话配置
```python
S = sim_struct.SESSION()
S.subpath = subject_dir
S.pathfem = output_dir
S.open_in_gmsh = False  # 禁止自动打开 Gmsh
```

### 5.2 电极对配置
```python
tdcs = session.add_tdcslist()
tdcs.currents = currents  # [current1, -current1]，电流方向相反

electrode = tdcs.add_electrode()
electrode.channelnr = 1
electrode.centre = electrode_pair[0]  # 如 'F5'
electrode.shape = electrode_shape
electrode.dimensions = electrode_dimensions
electrode.thickness = electrode_thickness
```

### 5.3 TI 计算
```python
from simnibs.utils import TI_utils as TI
ef1 = m1.field['E']
ef2 = m2.field['E']
ti_max = TI.get_maxTI(ef1.value, ef2.value)
```

### 5.4 网格裁剪（去除电极元素）
```python
tags_keep = np.hstack((
    np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
    np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1)
))
m1 = m1.crop_mesh(tags=tags_keep)
m2 = m2.crop_mesh(tags=tags_keep)
```

---

## 6. Demo 示例设计

### 6.1 `ti_simulation_demo.py`

**功能**：演示如何使用 `run_ti_simulation()` 函数执行 TI 正向仿真。

**数据来源**：`data/m2m_ernie/`

**配置**：
- 电极对1：F5-P5，电流 1mA
- 电极对2：F6-P6，电流 1mA
- 电极形状：椭圆 40x40 mm
- 电极厚度：2 mm
- n_workers: 24

**输出**：
- `data/TI_ernie/ernie_TDCS_1_scalar.msh`
- `data/TI_ernie/ernie_TDCS_2_scalar.msh`
- `data/TI_ernie/TI.msh`（TI 场可视化）

---

## 7. 注意事项

1. **路径处理**：使用 `os.path.join` 处理跨平台路径
2. **日志规范**：遵循 CLAUDE.md 规范，使用模块级 logger
3. **文档要求**：NumPy 风格 docstring，标注类型和返回值
4. **导入规范**：所有 import 放在模块头部
5. **线程设置**：通过 `n_workers` 参数控制并行工作进程数
6. **Subject ID 提取**：使用 `os.path.basename(subject_dir).replace("m2m_", "", 1)` 处理 m2m_ 前缀
7. **错误处理**：使用 try-except 捕获关键错误，记录日志后重新抛出

---

## 8. 验证计划

1. 运行 `ti_simulation_demo.py`，确认：
   - 仿真正常执行
   - 输出文件生成（ernie_TDCS_1_scalar.msh, ernie_TDCS_2_scalar.msh, TI.msh）
   - 日志正确记录
   - 不自动打开 Gmsh
2. 检查输出网格文件是否包含正确的场数据（E 字段）
3. 验证 TI.msh 可在 Gmsh 中正确打开
4. 验证 TI 最大调制振幅范围是否合理（[0, ~25]）
