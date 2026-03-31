# ForwardParams / InverseParams 参数结构重构方案

本文档描述 `ForwardParams` 和 `InverseParams` 数据结构的重构方案，目标是简化参数结构，使其更直观。

## 1. 概述

### 变更背景

原有参数结构存在以下问题：
- `electrode_A/B` 与 `current_A/B` 分离，需要保证长度一致
- `cond` 电导率配置与其他参数分离
- `InverseParams` 中电极优化相关参数过多

### 变更原则

1. **电极与电流绑定**：`electrode_A/B` 直接包含电流信息，不再分离
2. **电导率统一配置**：新增 `conductivity_config` 统一管理电导率
3. **精简逆向参数**：移除人类用户不需要理解的电极优化中间参数
4. **统一命名**：使用人类友好的命名 `ForwardParamsHuman` / `InverseParamsHuman`

## 2. ForwardParams 重构

### 2.1 当前结构

```python
@dataclass
class ForwardParams:
    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    electrode_A: list[str]
    electrode_B: list[str]
    current_A: list[float]
    current_B: list[float]
    cond: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None
```

### 2.2 新结构

```python
@dataclass
class ElectrodeWithCurrent:
    name: str
    current_mA: float


@dataclass
class ForwardParams:
    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    electrode_A: list[ElectrodeWithCurrent]
    electrode_B: list[ElectrodeWithCurrent]
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None
```

### 2.3 字段变更说明

| 字段 | 变更类型 | 说明 |
|---|---|---|
| `electrode_A/B` | **重构** | 从 `list[str]` 改为对象数组，每个对象包含 `name` 和 `current_mA` |
| `current_A/B` | **移除** | 电流信息已合并到 `electrode_A/B` 中 |
| `cond` | **重命名** | 改名为 `conductivity_config`，结构不变 |
| 其他字段 | **保留** | `id`、`dir_path`、`T1_file_path`、`montage`、`anisotropy`、`DTI_file_path` 均保留 |

### 2.4 示例

```json
{
  "id": "forward_001",
  "type": "forward",
  "params": {
    "dir_path": "C:/data/m2m_ernie",
    "T1_file_path": "C:/data/m2m_ernie/T1.nii.gz",
    "montage": "EEG10-10_UI_Jurak_2007",
    "electrode_A": [
      { "name": "F5", "current_mA": 2.0 },
      { "name": "P5", "current_mA": -2.0 }
    ],
    "electrode_B": [
      { "name": "F6", "current_mA": 1.0 },
      { "name": "P6", "current_mA": -1.0 }
    ],
    "conductivity_config": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654
    },
    "anisotropy": false
  }
}
```

## 3. InverseParams 重构

### 3.1 当前结构

```python
@dataclass
class InverseParams:
    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    current_A: list[float]
    current_B: list[float]
    roi_type: Literal["atlas", "mni_pos"]
    roi_param: ROIParam
    target_threshold: float
    cond: dict[str, float]
    anisotropy: bool
    electrode_pair1_center: list[list[float]] | None = None
    electrode_pair2_center: list[list[float]] | None = None
    electrode_radius: list[float] | None = None
    electrode_current1: list[float] | None = None
    electrode_current2: list[float] | None = None
    DTI_file_path: str | None = None
```

### 3.2 新结构

```python
@dataclass
class AtlasParam:
    name: str
    area: str


@dataclass
class MNIParam:
    center: list[float]
    radius: float


@dataclass
class ROIParam:
    atlas_param: AtlasParam | None = None
    mni_param: MNIParam | None = None


@dataclass
class InverseParams:
    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    current_A: list[float]
    current_B: list[float]
    roi_type: Literal["atlas", "mni_pos"]
    roi_param: ROIParam
    target_threshold: float
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None
```

### 3.3 字段变更说明

| 字段 | 变更类型 | 说明 |
|---|---|---|
| `cond` | **重命名** | 改名为 `conductivity_config`，结构不变 |
| `roi_param` | **重构** | 内联化 `atlas_param` 和 `mni_param` 到 `ROIParam` 中，不再使用单独的 `AtlasParam` / `MNIParam` 类 |
| `electrode_pair1_center` | **移除** | 移除电极优化相关参数 |
| `electrode_pair2_center` | **移除** | 移除电极优化相关参数 |
| `electrode_radius` | **移除** | 移除电极优化相关参数 |
| `electrode_current1` | **移除** | 移除电极优化相关参数 |
| `electrode_current2` | **移除** | 移除电极优化相关参数 |
| 其他字段 | **保留** | `id`、`dir_path`、`T1_file_path`、`montage`、`current_A`、`current_B`、`roi_type`、`target_threshold`、`anisotropy`、`DTI_file_path` 均保留 |

### 3.4 示例

```json
{
  "id": "inverse_001",
  "type": "inverse",
  "params": {
    "dir_path": "C:/data/m2m_ernie",
    "T1_file_path": "C:/data/m2m_ernie/T1.nii.gz",
    "montage": "EEG10-10_Cutini_2011",
    "current_A": [0.002, -0.002],
    "current_B": [0.001, -0.001],
    "roi_type": "mni_pos",
    "roi_param": {
      "atlas_param": null,
      "mni_param": {
        "center": [-38.5, -22.5, 58.3],
        "radius": 15.0
      }
    },
    "target_threshold": 0.5,
    "conductivity_config": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654
    },
    "anisotropy": false
  }
}
```

## 5. 影响范围

### 5.1 需要修改的文件

#### ForwardParams 变更

1. `neuracle/rabbitmq/schemas.py` - 新增 `ElectrodeWithCurrent` 类，修改 `ForwardParams` 的 `electrode_A/B` 和 `cond` 字段
2. `neuracle/rabbitmq/validator.py` - 修改 `validate_forward_params` 验证逻辑，验证 `electrode_A/B` 为对象数组，验证 `conductivity_config`
3. `neuracle/utils/params_utils.py` - 修改 `dict_to_forward_params` 转换逻辑
4. `neuracle/main.py` - 修改 `handle_forward_task` 中调用 `setup_electrode_pair1/2` 的地方，从 `electrode_A/B` 对象中提取 `name` 和 `current_mA`

#### InverseParams 变更

1. `neuracle/rabbitmq/schemas.py` - 修改 `InverseParams` 中 `cond` -> `conductivity_config`，移除电极优化参数
2. `neuracle/rabbitmq/validator.py` - 修改 `validate_inverse_params` 验证逻辑，验证 `conductivity_config`，移除电极优化参数验证
3. `neuracle/utils/params_utils.py` - 修改 `dict_to_inverse_params` 转换逻辑
4. `neuracle/main.py` - 修改 `handle_inverse_task` 中调用 `setup_electrodes_and_roi` 的地方，移除电极优化参数

#### 其他文件

1. `neuracle/rabbitmq/__init__.py` - 新增导出 `ElectrodeWithCurrent`
2. `neuracle/demo/main_sender_demo.py` - 修改 `create_forward_message` 和 `create_inverse_message` 函数，使用新的 `electrode_A/B` 结构和 `conductivity_config`
3. `neuracle/demo/backend_side_demo.py` - 修改所有 forward/inverse 示例消息，使用新的 `electrode_A/B` 结构和 `conductivity_config`
4. `neuracle/demo/simnibs_side_demo.py` - 修改 forward 示例消息，使用新的 `electrode_A/B` 结构

#### 文档更新

1. `neuracle/docs/rabbitmq_data_structure.md` - 更新 ForwardParams 和 InverseParams 的 JSON 示例，移除 `msh_file_path` 字段说明，更新字段类型描述
2. `neuracle/docs/rabbitmq_service_design.md` - 更新第 6.2 节 forward 执行顺序中关于 `electrode_A/B` 的描述，更新第 10 节差异描述
3. `neuracle/docs/ti_nifti_export_plan.md` - 更新 ForwardParams 和 InverseParams 的结构定义