# TI NIfTI 导出修改方案

## 修改目标

1. 正向和逆向仿真结果只导出 `TI_max` 到 NIfTI 格式（使用灰质+白质插值）
2. 不再导出 `magnE - pair 1` 和 `magnE - pair 2`
3. 不再生成 MZ3 文件
4. `ForwardParams` 和 `InverseParams` 添加 `T1_file_path` 参数，用于指定插值参考文件

## 修改文件清单

### 1. `neuracle/rabbitmq/schemas.py`

**ForwardParams 修改**：
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

**InverseParams 修改**：
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
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None
```

### 2. `neuracle/ti_simulation/ti_simulation.py`

**前置导入**：
```python
from simnibs import ElementTags
```

**calculate_ti 函数修改**：

原始版本：
```python
def calculate_ti(
    mesh1_path: str,
    mesh2_path: str,
    output_dir: str,
) -> str:
```

修改后（本次方案）：
```python
def calculate_ti(
    mesh1_path: str,
    mesh2_path: str,
    output_dir: str,
) -> str:
    # 只返回 TI mesh 路径
```

**内部修改**：
1. `keep_tissues=[ElementTags.WM, ElementTags.GM]`（白质+灰质）
2. 移除 `magnE - pair 1` 和 `magnE - pair 2` 字段，只保留 `TImax`
3. 返回值保持 `ti_mesh_path`（供后续 export_ti_to_nifti 使用）

### 3. `neuracle/ti_optimize/ti_optimize.py`

export_mz3 函数已移动到 `utils/ti_export_utils.py`，原位置可选择移除或保留。

### 4. `neuracle/utils/` 新建文件

创建新文件 `neuracle/utils/ti_export_utils.py`：

```python
"""
TI 导出工具函数
"""
import logging
import os

from simnibs import ElementTags
from simnibs.utils.transformations import interpolate_to_volume
from simnibs import mesh_io

logger = logging.getLogger(__name__)


def export_ti_to_nifti(
    msh_path: str,
    output_dir: str,
    reference: str,
    field_name: str = "TImax",
) -> str:
    """
    将 TI 场从 mesh 导出到 NIfTI 格式

    Parameters
    ----------
    msh_path : str
        源 mesh 文件路径
    output_dir : str
        输出目录
    reference : str
        参考 NIfTI 文件路径（用于确定输出空间）
    field_name : str
        要导出的字段名 (default: "TImax")

    Returns
    -------
    str
        NIfTI 文件路径
    """
    logger.info("将 TI 结果导出到 NIfTI 格式...")
    mesh = mesh_io.read_msh(msh_path)

    nifti_path = os.path.join(output_dir, f"{field_name}.nii.gz")
    interpolate_to_volume(
        mesh, reference, nifti_path,
        method="linear",
        keep_tissues=[ElementTags.WM, ElementTags.GM]
    )
    logger.info("NIfTI 导出完成: %s", nifti_path)
    return nifti_path


def export_ti_to_mz3(
    ti_mesh_path: str,
    output_dir: str,
    surface_type: str = "central",
) -> str:
    """
    导出 TI 结果到 MZ3 格式

    Parameters
    ----------
    ti_mesh_path : str
        TI 结果网格路径
    output_dir : str
        输出目录
    surface_type : str
        表面类型 (default: "central")

    Returns
    -------
    str
        MZ3 文件路径
    """
    from neuracle.mesh_tools import msh_to_mz3

    logger.info("导出 TI 结果到 MZ3 格式...")
    mz3_path = msh_to_mz3(
        msh_path=ti_mesh_path,
        output_dir=output_dir,
        surface_type=surface_type,
        field_name="TImax",
    )
    logger.info("MZ3 导出完成: %s", mz3_path)
    return mz3_path
```

### 5. `neuracle/main.py`

**handle_forward_task 函数修改**：

修改前：
```python
# 85% - calculate_ti
ti_mesh_path = calculate_ti(
    mesh1_path=mesh1_path,
    mesh2_path=mesh2_path,
    output_dir=str(output_dir),
)
send_progress(message_queue, task_id, "forward", 85)

# 95% - export_mz3
mz3_path = sim_export_mz3(
    ti_mesh_path=ti_mesh_path,
    output_dir=str(output_dir),
    surface_type="central",
)
send_progress(message_queue, task_id, "forward", 95)

ti_file_key = upload_task_result(
    bucket,
    Path(mz3_path),
    f"{normalize_dir_path(params.dir_path)}_TI_simulation_{task_id}/TI.mz3",
)
```

修改后：
```python
# 85% - calculate_ti
ti_mesh_path = calculate_ti(
    mesh1_path=mesh1_path,
    mesh2_path=mesh2_path,
    output_dir=str(output_dir),
)
send_progress(message_queue, task_id, "forward", 85)

# 95% - export_ti_to_nifti
ti_nifti_path = export_ti_to_nifti(
    msh_path=ti_mesh_path,
    output_dir=str(output_dir),
    reference=str(subject_dir / params.T1_file_path),
    field_name="TImax",
)
send_progress(message_queue, task_id, "forward", 95)

ti_file_key = upload_task_result(
    bucket,
    Path(ti_nifti_path),
    f"{normalize_dir_path(params.dir_path)}_TI_simulation_{task_id}/TI.nii.gz",
)
```

**handle_inverse_task 函数修改**：

修改前：
```python
# 95% - export_mz3
modelid = Path(mesh_path).stem
msh_name = f"{modelid}_tes_mapped_opt_surface_mesh.msh"
mz3_path = optimize_export_mz3(
    output_dir=str(output_dir / "mapped_electrodes_simulation"),
    msh_name=msh_name,
    surface_type="central",
)
send_progress(message_queue, task_id, "inverse", 95)

ti_file_key = upload_task_result(
    bucket,
    Path(mz3_path),
    f"{normalize_dir_path(params.dir_path)}_TI_optimization_{task_id}/TI.mz3",
)
```

修改后：
```python
# 95% - export_ti_to_nifti
modelid = Path(mesh_path).stem
msh_name = f"{modelid}_tes_mapped_opt_surface_mesh.msh"
msh_path = str(output_dir / "mapped_electrodes_simulation" / msh_name)
ti_nifti_path = export_ti_to_nifti(
    msh_path=msh_path,
    output_dir=str(output_dir),
    reference=str(subject_dir / params.T1_file_path),
    field_name="max_TI",
)
send_progress(message_queue, task_id, "inverse", 95)

ti_file_key = upload_task_result(
    bucket,
    Path(ti_nifti_path),
    f"{normalize_dir_path(params.dir_path)}_TI_optimization_{task_id}/TI.nii.gz",
)
```

**导入调整**：
- 移除 `sim_export_mz3` 和 `optimize_export_mz3` 的导入（不再使用）
- 添加 `from neuracle.utils.ti_export_utils import export_ti_to_nifti` 的导入

### 6. Demo 文件修改

以下 demo 文件需要相应更新：

- `neuracle/demo/ti_simulation_demo.py`
- `neuracle/demo/ti_max_optimize_demo.py`
- `neuracle/demo/ti_focality_optimize_demo.py`
- `neuracle/demo/ti_focality_inv_optimize_demo.py`
- `neuracle/demo/ti_mean_optimize_demo.py`

**修改要点**：
1. `calculate_ti` 返回值保持 `str`（mesh路径）
2. 正向 demo：添加 `export_ti_to_nifti` 调用，生成 NIfTI 文件
3. 移除 `export_mz3` 相关导入和调用
4. 优化 demo：移除 `export_mz3`，改为调用 `export_ti_to_nifti`（从 `neuracle.utils.ti_export_utils` 导入）

### 7. `neuracle/ti_simulation/__init__.py`

保持导出不变，`export_mz3` 保留

## 实施步骤

### 步骤 1：修改 `schemas.py`
- 在 `ForwardParams` 中添加 `T1_file_path: str` 字段（必需）
- 在 `InverseParams` 中添加 `T1_file_path: str` 字段（必需）

### 步骤 2：新建 `utils/ti_export_utils.py`
- 实现 `export_ti_to_nifti(msh_path, output_dir, reference, field_name)` 函数
- 实现 `export_ti_to_mz3(ti_mesh_path, output_dir, surface_type)` 函数
- 添加 `logger = logging.getLogger(__name__)`
- 添加 `from simnibs import ElementTags` 和 `from simnibs.utils.transformations import interpolate_to_volume`

### 步骤 3：修改 `ti_simulation.py`
- `calculate_ti` 函数移除 `reference` 参数
- `calculate_ti` 函数只保留 `TImax` 字段，移除 `magnE - pair 1` 和 `magnE - pair 2`
- 返回值保持 `str`（mesh 路径）
- 添加 `from simnibs import ElementTags` 导入

### 步骤 4：修改 `main.py`
- 添加 `from neuracle.utils.ti_export_utils import export_ti_to_nifti` 导入
- `handle_forward_task`：
  - 85% calculate_ti 获取 ti_mesh_path
  - 95% 调用 export_ti_to_nifti 生成 NIfTI
  - 上传结果改为 `.nii.gz`
- `handle_inverse_task`：
  - 95% 调用 export_ti_to_nifti 替代 export_mz3
  - 上传结果改为 `.nii.gz`

### 步骤 5：修改各 demo 文件
- `ti_simulation_demo.py`：
  - 移除 `export_mz3` 导入
  - 添加 `export_ti_to_nifti` 导入
  - calculate_ti 后添加 export_ti_to_nifti 调用
- `ti_max_optimize_demo.py`、`ti_focality_optimize_demo.py`、`ti_focality_inv_optimize_demo.py`、`ti_mean_optimize_demo.py`：
  - 移除 `export_mz3` 导入
  - 添加 `export_ti_to_nifti` 导入
  - 替换 export_mz3 调用为 export_ti_to_nifti

### 步骤 6：测试运行
- 运行 `ti_simulation_demo.py` 验证正向仿真流程
- 检查输出目录确认生成 `TI.nii.gz` 文件
