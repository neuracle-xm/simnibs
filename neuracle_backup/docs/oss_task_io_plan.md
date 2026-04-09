# 三类任务接入 OSS 的改造方案

## 0. 当前本地任务实际依赖的文件清单

### 0.1 TI 正向仿真

| 类别 | 文件/目录 | 来源 | 是否必需 | 备注 |
| --- | --- | --- | --- | --- |
| 网格文件 | `dir_path/model.msh` | 根据 `dir_path` 固定拼接 | 是 | 正向仿真核心输入 |
| 电极帽文件 | `dir_path/eeg_positions/<montage>.csv` | `find_montage_file(dir_path, montage)` 本地查找 | 是 | `montage` 仅传名称，例如 `EEG10-10_Cutini_2011` |
| DTI 文件 | `DTI_file_path` 指向的文件 | 参数显式传入 | 是 | 当前文档按最保守口径统一视为必需 |

### 0.2 TI 逆向仿真

| 类别 | 文件/目录 | 来源 | 是否必需 | 备注 |
| --- | --- | --- | --- | --- |
| 网格文件 | `dir_path/model.msh` | 根据 `dir_path` 固定拼接 | 是 | 逆向仿真核心输入 |
| 电极帽文件 | `dir_path/eeg_positions/<montage>.csv` | `find_montage_file(dir_path, montage)` 本地查找 | 是 | `montage` 仅传名称，例如 `EEG10-10_Cutini_2011` |
| DTI 文件 | `DTI_file_path` 指向的文件 | 参数显式传入 | 是 | 当前文档按最保守口径统一视为必需 |
| subject 目录网格 | `dir_path/<subid>.msh` | `SubjectFiles(subpath=dir_path)` | 是 | `dir_path` 需保持 `m2m_*` 目录语义 |
| central surface | `dir_path/surfaces/lh.central.gii` | `SubjectFiles(subpath=dir_path)` | 是 | 当前 ROI 使用 `surface + central` |
| central surface | `dir_path/surfaces/rh.central.gii` | `SubjectFiles(subpath=dir_path)` | 是 | 当前 ROI 使用 `surface + central` |
| MNI 变换文件 | `dir_path/toMNI/MNI2Conform_nonl.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| MNI 变换文件 | `dir_path/toMNI/Conform2MNI_nonl.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| 参考体积 | `dir_path/label_prep/T1_upsampled.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| 参考体积回退文件 | `dir_path/T1.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| atlas ROI 文件 | 仓库内标准化 ROI 文件 | `get_standardized_roi_path(...)` | 是 | 当前文档按最保守口径统一视为必需 |
| 约束定位文件 | `dir_path/eeg_positions/Fiducials.csv` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |

### 0.3 当前可排除文件/目录

| 文件/目录 | 是否可排除 | 备注 |
| --- | --- | --- |
| `dir_path/segmentation/` | 是 | 当前 TI 正向/逆向链路未直接读取 |
| `dir_path/final_tissues.nii.gz` | 是 | 当前 TI 正向/逆向链路未直接读取 |
| `final_tissues_LUT.txt` | 是 | 当前 TI 正向/逆向不作为任务输入 |
| `*.lg.png` | 是 | 当前 TI 正向/逆向不作为计算输入 |

## 1. 总体原则

| 项目 | 处理方式 |
| --- | --- |
| `dir_path` | 作为 `data/` 下的相对目录使用；为空时默认使用 `m2m_ernie` |
| `montage` | 只传名称，从 `dir_path/eeg_positions/` 中查找对应 `.csv` |
| 正向网格文件 | 固定使用 `dir_path/model.msh` |
| 逆向网格文件 | 固定使用 `dir_path/model.msh` |
| 本地缓存命中 | `data/` 下已有 `dir_path` 时直接复用 |
| 本地缓存未命中 | 从 OSS 下载所有以 `dir_path` 开头的文件到 `data/dir_path/` |
| DTI 文件 | 使用 `dir_path/DTI_file_path`；未传则默认使用 `DTI_coregT1_tensor.nii.gz` |
| 任务完成后输出 | 上传指定结果到 OSS |
| 任务完成后清理 | 删除正向/逆向任务输出目录 |

## 2. 本地目录规则

| 任务 | 本地输入目录 | 本地输出目录 |
| --- | --- | --- |
| 头模生成 | `data/dir_path/` | `data/dir_path/` |
| TI 正向仿真 | `data/dir_path/` | `data/{dir_path}_TI_simulation_{task_id}/` |
| TI 逆向仿真 | `data/dir_path/` | `data/{dir_path}_TI_optimization_{task_id}/` |

## 3. 头模生成方案

### 3.1 输入

| 参数 | 来源 |
| --- | --- |
| `T1_file_path` | OSS |
| `T2_file_path` | OSS |
| `DTI_file_path` | OSS |
| `dir_path` | 请求参数 |

### 3.2 执行流程

| 步骤 | 处理 |
| --- | --- |
| 1 | 根据 `dir_path` 在 `data/` 下创建本地目录 |
| 2 | 从 OSS 下载 `T1/T2/DTI` 到 `data/dir_path/` |
| 3 | 在 `data/dir_path/` 下执行头模生成任务 |

### 3.3 上传范围

| 类型 | OSS 路径 |
| --- | --- |
| 目录 | `{dir_path}/eeg_positions/` |
| 目录 | `{dir_path}/label_prep/` |
| 目录 | `{dir_path}/surfaces/` |
| 目录 | `{dir_path}/toMNI/` |
| 文件 | `{dir_path}/model.msh` |
| 文件 | `{dir_path}/model.msh.opt` |

## 4. TI 正向仿真方案

### 4.1 输入

| 参数 | 处理 |
| --- | --- |
| `dir_path` | 先检查 `data/dir_path/` 是否存在 |
| `montage` | 只传名称，从 `data/dir_path/eeg_positions/` 中查找对应 `.csv` |
| `DTI_file_path` | 使用 `data/dir_path/DTI_file_path`；未传则不找 |

### 4.2 数据准备

| 场景 | 处理 |
| --- | --- |
| `data/dir_path/` 已存在 | 直接使用 |
| `data/dir_path/` 不存在 | 创建目录后，从 OSS 下载所有以 `dir_path` 开头的文件到 `data/dir_path/` |

### 4.3 执行参数

| 项目 | 固定取值 |
| --- | --- |
| 网格文件 | `data/dir_path/model.msh` |
| `montage` 文件 | `data/dir_path/eeg_positions/` 下匹配文件 |
| `DTI` 文件 | `data/dir_path/DTI_file_path`；未传则默认使用 `DTI_coregT1_tensor.nii.gz` |
| `output_dir` | `data/{dir_path}_TI_simulation_{task_id}/` |

### 4.4 上传与返回

| 项目 | 处理 |
| --- | --- |
| 上传文件 | `TI.mz3` |
| 上传 OSS 路径 | `{dir_path}_TI_simulation_{task_id}/TI.mz3` |
| 返回字段 | `TI_file` 返回 `{dir_path}_TI_simulation_{task_id}/TI.mz3` |
| 任务结束清理 | 删除 `data/{dir_path}_TI_simulation_{task_id}/` |

## 5. TI 逆向仿真方案

### 5.1 输入

| 参数 | 处理 |
| --- | --- |
| `dir_path` | 先检查 `data/dir_path/` 是否存在 |
| `montage` | 只传名称，从 `data/dir_path/eeg_positions/` 中查找对应 `.csv` |
| `DTI_file_path` | 使用 `data/dir_path/DTI_file_path`；未传则不找 |

### 5.2 数据准备

| 场景 | 处理 |
| --- | --- |
| `data/dir_path/` 已存在 | 直接使用 |
| `data/dir_path/` 不存在 | 创建目录后，从 OSS 下载所有以 `dir_path` 开头的文件到 `data/dir_path/` |

### 5.3 执行参数

| 项目 | 固定取值 |
| --- | --- |
| 网格文件 | `data/dir_path/model.msh` |
| `montage` 文件 | `data/dir_path/eeg_positions/` 下匹配文件 |
| `DTI` 文件 | `data/dir_path/DTI_file_path`；未传则默认使用 `DTI_coregT1_tensor.nii.gz` |
| `output_dir` | `data/{dir_path}_TI_optimization_{task_id}/` |

### 5.4 MZ3 导出规则

| 项目 | 处理 |
| --- | --- |
| MZ3 来源 | 使用 `data/{dir_path}_TI_optimization_{task_id}/mapped_electrodes_simulation/` 下的 `.msh` 文件 |
| MZ3 本地输出目录 | `data/{dir_path}_TI_optimization_{task_id}/mapped_electrodes_simulation/` |
| MZ3 上传 OSS 路径 | `{dir_path}_TI_optimization_{task_id}/TI.mz3` |

### 5.5 上传与返回

| 项目 | 处理 |
| --- | --- |
| 上传文件 | `mapped_electrodes_simulation/` 下生成的 `.mz3` |
| 上传 OSS 路径 | `{dir_path}_TI_optimization_{task_id}/TI.mz3` |
| 返回字段 | `TI_file` 返回 `{dir_path}_TI_optimization_{task_id}/TI.mz3` |
| 任务结束清理 | 删除 `data/{dir_path}_TI_optimization_{task_id}/` |

## 6. 文档对应的代码改造点

| 文件 | 需要调整的内容 |
| --- | --- |
| `neuracle/main.py` | 三类任务接入 `data/` 目录规则、OSS 下载与上传流程 |
| `neuracle/ti_simulation/ti_simulation.py` | 保持正向仿真生成 `TI.msh` 和 `TI.mz3` |
| `neuracle/ti_optimize/ti_optimize.py` | 逆向 `.mz3` 导出改为读取 `mapped_electrodes_simulation/` 下的 `.msh` |
| `neuracle/oss_tool/__init__.py` | 补充按前缀下载、上传本地文件、上传目录的能力 |

## 7. 上传白名单

| 任务 | 上传内容 |
| --- | --- |
| 头模生成 | 上传到 `{dir_path}/eeg_positions/`、`{dir_path}/label_prep/`、`{dir_path}/surfaces/`、`{dir_path}/toMNI/`、`{dir_path}/model.msh`、`{dir_path}/model.msh.opt` |
| TI 正向仿真 | 上传到 `{dir_path}_TI_simulation_{task_id}/TI.mz3` |
| TI 逆向仿真 | 上传到 `{dir_path}_TI_optimization_{task_id}/TI.mz3` |

## 8. 删除规则

| 任务 | 删除内容 |
| --- | --- |
| 头模生成 | 不删 `data/dir_path/` |
| TI 正向仿真 | 删除 `data/{dir_path}_TI_simulation_{task_id}/`（DEBUG=True 时跳过删除） |
| TI 逆向仿真 | 删除 `data/{dir_path}_TI_optimization_{task_id}/`（DEBUG=True 时跳过删除） |

### 8.1 DEBUG 模式

通过 `neuracle/main.py` 顶部的 `DEBUG` 常量控制：
- `DEBUG=True` 时，跳过正向/逆向仿真 output_dir 的删除
- 默认为 `False`，任务完成后正常删除 output_dir

## 9. 具体修改步骤

### 9.1 第一步：补充 OSS 工具能力

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/oss_tool/__init__.py` | 增加按单文件下载能力 |
| `neuracle/oss_tool/__init__.py` | 增加按前缀批量下载能力，支持下载所有以 `dir_path` 开头的文件 |
| `neuracle/oss_tool/__init__.py` | 增加上传本地文件到指定 OSS key 的能力 |
| `neuracle/oss_tool/__init__.py` | 增加上传本地目录到指定 OSS 前缀的能力 |

### 9.2 第二步：封装本地缓存目录规则

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/main.py` | 增加统一的 `data/` 本地目录解析逻辑 |
| `neuracle/main.py` | 头模生成固定使用 `data/dir_path/` |
| `neuracle/main.py` | 正向仿真固定使用 `data/dir_path/` 和 `data/{dir_path}_TI_simulation/` |
| `neuracle/main.py` | 逆向仿真固定使用 `data/dir_path/` 和 `data/{dir_path}_TI_optimization/` |

### 9.3 第三步：改造头模生成任务

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/main.py` | 接收 `T1/T2/DTI` 的 OSS 路径和 `dir_path` |
| `neuracle/main.py` | 根据 `dir_path` 创建 `data/dir_path/` |
| `neuracle/main.py` | 下载 `T1/T2/DTI` 到 `data/dir_path/` |
| `neuracle/main.py` | 任务完成后上传 `{dir_path}/eeg_positions/`、`{dir_path}/label_prep/`、`{dir_path}/surfaces/`、`{dir_path}/toMNI/`、`{dir_path}/model.msh`、`{dir_path}/model.msh.opt` |

### 9.4 第四步：改造 TI 正向仿真任务

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/main.py` | 移除显式 `msh_file_path` 输入，统一改为 `data/dir_path/model.msh` |
| `neuracle/main.py` | 检查 `data/dir_path/` 是否存在，不存在时下载所有以 `dir_path` 开头的 OSS 文件 |
| `neuracle/main.py` | montage 只按名称在 `data/dir_path/eeg_positions/` 中查找 |
| `neuracle/main.py` | `DTI_file_path` 未传时默认使用 `DTI_coregT1_tensor.nii.gz` |
| `neuracle/main.py` | `dir_path` 为空时默认使用 `m2m_ernie` |
| `neuracle/main.py` | 任务完成后上传 `TI.mz3` 到 `{dir_path}_TI_simulation_{task_id}/TI.mz3` |
| `neuracle/main.py` | 返回 `TI_file={dir_path}_TI_simulation_{task_id}/TI.mz3` |
| `neuracle/main.py` | 上传完成后删除 `data/{dir_path}_TI_simulation_{task_id}/` |

### 9.5 第五步：改造 TI 逆向仿真任务

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/main.py` | 移除显式 `msh_file_path` 输入，统一改为 `data/dir_path/model.msh` |
| `neuracle/main.py` | 检查 `data/dir_path/` 是否存在，不存在时下载所有以 `dir_path` 开头的 OSS 文件 |
| `neuracle/main.py` | montage 只按名称在 `data/dir_path/eeg_positions/` 中查找 |
| `neuracle/main.py` | `DTI_file_path` 未传时默认使用 `DTI_coregT1_tensor.nii.gz` |
| `neuracle/main.py` | `dir_path` 为空时默认使用 `m2m_ernie` |
| `neuracle/ti_optimize/ti_optimize.py` | `.mz3` 导出源改为 `data/{dir_path}_TI_optimization_{task_id}/mapped_electrodes_simulation/` 下的 `.msh` |
| `neuracle/main.py` | 上传生成的 `.mz3` 到 `{dir_path}_TI_optimization_{task_id}/TI.mz3` |
| `neuracle/main.py` | 返回 `TI_file={dir_path}_TI_optimization_{task_id}/TI.mz3` |
| `neuracle/main.py` | 上传完成后删除 `data/{dir_path}_TI_optimization_{task_id}/` |

### 9.6 第六步：修改参数结构和校验

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/rabbitmq/schemas.py` | 删除正向和逆向的 `msh_file_path` 字段 |
| `neuracle/utils/params_utils.py` | 删除正向和逆向对 `msh_file_path` 的解析 |
| `neuracle/rabbitmq/validator.py` | 删除正向和逆向对 `msh_file_path` 的校验 |
| `neuracle/rabbitmq/schemas.py` | 保留 `dir_path`、`montage`、`DTI_file_path`、头模生成的 OSS 文件参数 |

### 9.7 第七步：补充日志与进度

| 文件 | 修改内容 |
| --- | --- |
| `neuracle/main.py` | 下载前记录本地缓存命中/未命中 |
| `neuracle/main.py` | 上传前记录目标 OSS key |
| `neuracle/main.py` | 返回结果中记录最终上传的 OSS 路径 |
| `neuracle/main.py` | 删除输出目录前记录待删除目录 |
