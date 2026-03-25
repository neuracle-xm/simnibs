# RabbitMQ 消息数据结构规范

本文档定义了后端（Backend）与 SimNIBS 之间的 RabbitMQ 消息传递数据结构和格式。

---

## 目录

1. [队列设计](#1-队列设计)
2. [消息推送格式](#2-消息推送格式)
3. [进度返回格式](#3-进度返回格式)
4. [完整消息示例](#4-完整消息示例)
5. [数据结构实现](#5-数据结构实现)
6. [测试说明](#6-测试说明)

---

## 1. 队列设计

### 1.1 队列命名

| 队列名称 | 方向 | 说明 |
|----------|------|------|
| `backend_to_simnibs` | Backend → SimNIBS | 任务推送（头模生成、正向仿真、逆向仿真） |
| `simnibs_to_backend` | SimNIBS → Backend | 进度回报与结果返回 |

### 1.2 连接配置

- **交换机**: 使用默认交换机 `''`（空字符串）
- **队列绑定**: 直接绑定到队列，无需交换机中转

---

## 2. 消息推送格式

所有消息均为 JSON 格式，包含以下通用字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 任务唯一标识符 |
| `type` | string | 是 | 任务类型：`model` / `forward` / `inverse` |
| `params` | object | 是 | 任务参数配置 |

---

### 2.1 头模生成消息 (Model Generation)

```json
{
  "id": "model_id",
  "type": "model",
  "params": {
    "T1_file_path": "m2m_modelId_T1_nii",
    "T2_file_path": "m2m_modelId_T2_nii",
    "DTI_file_path": "m2m_modelId_DTI_nii",
    "dir_path": "m2m_modelId"
  }
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 模型唯一标识符 |
| `params.T1_file_path` | string | 是 | T1 加权 MRI 图像路径 |
| `params.T2_file_path` | string | 否 | T2 加权 MRI 图像路径，可为空 |
| `params.DTI_file_path` | string | 否 | DTI 扩散张量图像路径，可为空 |
| `params.dir_path` | string | 是 | 头模生成输出目录路径 |

---

### 2.2 正向仿真消息 (Forward Simulation)

```json
{
  "id": "task_id",
  "type": "forward",
  "params": {
    "dir_path": "头模所在文件夹",
    "msh_file_path": "头模文件",
    "montage": "使用的电极导联文件",
    "electrode_A": ["F5", "P5"],
    "electrode_B": ["F5", "P5"],
    "current_A": [0.002, -0.002],
    "current_B": [0.001, -0.001],
    "cond": {},
    "anisotropy": false,
    "DTI_file_path": "张量文件路径"
  }
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 任务唯一标识符 |
| `params.dir_path` | string | 是 | 头模所在文件夹路径 |
| `params.msh_file_path` | string | 是 | 头模网格文件路径 (.msh) |
| `params.montage` | string | 是 | 电极导联配置文件路径 |
| `params.electrode_A` | string[] | 是 | 电极组 A 名称列表 |
| `params.electrode_B` | string[] | 是 | 电极组 B 名称列表 |
| `params.current_A` | float[] | 是 | 电极组 A 电流值列表 (安培)，总和必须为 0 |
| `params.current_B` | float[] | 是 | 电极组 B 电流值列表 (安培)，总和必须为 0 |
| `params.cond` | object | 是 | 组织电导率配置 JSON |
| `params.anisotropy` | boolean | 是 | 是否启用各向异性电导率 |
| `params.DTI_file_path` | string | 否 | DTI 张量文件路径，可为空 |

---

### 2.3 逆向仿真消息 (Inverse Simulation)

```json
{
  "id": "task_id",
  "type": "inverse",
  "params": {
    "dir_path": "头模所在文件夹",
    "msh_file_path": "头模文件",
    "montage": "使用的电极导联文件",
    "current_A": [0.002, -0.002],
    "current_B": [0.001, -0.001],
    "roi_type": "atlas",
    "roi_param": {
      "atlas_param": {
        "name": "脑图谱名称",
        "area": "选择的脑区"
      },
      "mni_param": null
    },
    "target_threshold": 0.5,
    "cond": {},
    "anisotropy": false,
    "DTI_file_path": "张量文件路径"
  }
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 任务唯一标识符 |
| `params.dir_path` | string | 是 | 头模所在文件夹路径 |
| `params.msh_file_path` | string | 是 | 头模网格文件路径 (.msh) |
| `params.montage` | string | 是 | 电极导联配置文件路径 |
| `params.current_A` | float[] | 是 | 电极组 A 电流值列表 (安培)，总和必须为 0 |
| `params.current_B` | float[] | 是 | 电极组 B 电流值列表 (安培)，总和必须为 0 |
| `params.roi_type` | string | 是 | ROI 类型：`atlas` 或 `mni_pos` |
| `params.roi_param` | object | 是 | ROI 参数配置 |
| `params.roi_param.atlas_param` | object | 否 | 图谱模式参数，当 `roi_type` 为 `atlas` 时必填 |
| `params.roi_param.atlas_param.name` | string | 否 | 脑图谱名称 |
| `params.roi_param.atlas_param.area` | string | 否 | 选择的脑区名称 |
| `params.roi_param.mni_param` | object | 否 | MNI 坐标模式参数，当 `roi_type` 为 `mni_pos` 时必填 |
| `params.roi_param.mni_param.center` | float[3] | 否 | MNI 坐标中心点 [x, y, z] |
| `params.roi_param.mni_param.radius` | float | 否 | 靶区半径范围 |
| `params.target_threshold` | float | 是 | 目标电场强度阈值，必须 >= 0 |
| `params.cond` | object | 是 | 组织电导率配置 JSON |
| `params.anisotropy` | boolean | 是 | 是否启用各向异性电导率 |
| `params.DTI_file_path` | string | 否 | DTI 张量文件路径，可为空 |

---

## 3. 进度返回格式

消息通过 `simnibs_to_backend` 队列发送。

```json
{
  "id": "model_id 或 task_id",
  "type": "model | forward | inverse",
  "progress_rate": 75,
  "message": null,
  "result": null
}
```

### 3.1 进度消息字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 对应任务的 ID |
| `type` | string | 是 | 任务类型：`model` / `forward` / `inverse` |
| `progress_rate` | integer | 是 | 进度百分比 (0-100) |
| `message` | string | 否 | 错误消息或状态描述，正常为 null |
| `result` | object | 否 | 完成时返回的结果，过程中为 null |

### 3.2 完成消息 (Result)

#### 头模生成完成

```json
{
  "id": "model_id",
  "type": "model",
  "progress_rate": 100,
  "message": null,
  "result": {
    "msh_file_path": "头模最终生成的结果文件路径"
  }
}
```

#### 正向仿真完成

```json
{
  "id": "task_id",
  "type": "forward",
  "progress_rate": 100,
  "message": null,
  "result": {
    "T1_mni": "配准到 MNI 空间的头模 .nii.gz",
    "TI_file": "正向仿真最终结果 .mz3"
  }
}
```

#### 逆向仿真完成

```json
{
  "id": "task_id",
  "type": "inverse",
  "progress_rate": 100,
  "message": null,
  "result": {
    "T1_mni": "配准到 MNI 空间的头模 .nii.gz",
    "TI_file": "逆向仿真最终结果 .mz3",
    "electrode_A": ["F5", "P5"],
    "electrode_B": ["F5", "P5"]
  }
}
```

---

## 4. 完整消息示例

### 4.1 头模生成

**请求消息** (通过 `backend_to_simnibs` 发送):
```json
{
  "id": "subj_001",
  "type": "model",
  "params": {
    "T1_file_path": "/data/m2m_subj_001/T1fs.nii",
    "T2_file_path": "/data/m2m_subj_001/T2fs.nii",
    "DTI_file_path": "/data/m2m_subj_001/DTI.nii",
    "dir_path": "/data/m2m_subj_001"
  }
}
```

**进度回报** (通过 `simnibs_to_backend` 发送):
```json
{
  "id": "subj_001",
  "type": "model",
  "progress_rate": 50,
  "message": null,
  "result": null
}
```

**完成消息** (通过 `simnibs_to_backend` 发送):
```json
{
  "id": "subj_001",
  "type": "model",
  "progress_rate": 100,
  "message": null,
  "result": {
    "msh_file_path": "/data/m2m_subj_001/subj_001.msh"
  }
}
```

### 4.2 正向仿真

**请求消息** (通过 `backend_to_simnibs` 发送):
```json
{
  "id": "fwd_001",
  "type": "forward",
  "params": {
    "dir_path": "/data/m2m_subj_001",
    "msh_file_path": "/data/m2m_subj_001/subj_001.msh",
    "montage": "/data/montages/standard_64chan.csv",
    "electrode_A": ["F5", "P5"],
    "electrode_B": ["F5", "P5"],
    "current_A": [0.002, -0.002],
    "current_B": [0.001, -0.001],
    "cond": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654,
      "Bone": 0.01,
      "Scalp": 0.465,
      "Eye balls": 0.5,
      "Compact Bone": 0.008,
      "Spongy Bone": 0.025,
      "Blood": 0.6,
      "Muscle": 0.16
    },
    "anisotropy": true,
    "DTI_file_path": "/data/m2m_subj_001/DTI.nii"
  }
}
```

**完成消息** (通过 `simnibs_to_backend` 发送):
```json
{
  "id": "fwd_001",
  "type": "forward",
  "progress_rate": 100,
  "message": null,
  "result": {
    "T1_mni": "/data/results/subj_001_T1_mni.nii.gz",
    "TI_file": "/data/results/subj_001_forward_result.mz3"
  }
}
```

### 4.3 逆向仿真 (Atlas ROI)

**请求消息** (通过 `backend_to_simnibs` 发送):
```json
{
  "id": "inv_001",
  "type": "inverse",
  "params": {
    "dir_path": "/data/m2m_subj_001",
    "msh_file_path": "/data/m2m_subj_001/subj_001.msh",
    "montage": "/data/montages/standard_64chan.csv",
    "current_A": [0.002, -0.002],
    "current_B": [0.001, -0.001],
    "roi_type": "atlas",
    "roi_param": {
      "atlas_param": {
        "name": "AAL3",
        "area": "Precentral_L"
      },
      "mni_param": null
    },
    "target_threshold": 0.5,
    "cond": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654,
      "Bone": 0.01,
      "Scalp": 0.465,
      "Eye balls": 0.5,
      "Compact Bone": 0.008,
      "Spongy Bone": 0.025,
      "Blood": 0.6,
      "Muscle": 0.16
    },
    "anisotropy": true,
    "DTI_file_path": "/data/m2m_subj_001/DTI.nii"
  }
}
```

### 4.4 逆向仿真 (MNI Position ROI)

**请求消息** (通过 `backend_to_simnibs` 发送):
```json
{
  "id": "inv_002",
  "type": "inverse",
  "params": {
    "dir_path": "/data/m2m_subj_001",
    "msh_file_path": "/data/m2m_subj_001/subj_001.msh",
    "montage": "/data/montages/standard_64chan.csv",
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
    "cond": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654,
      "Bone": 0.01,
      "Scalp": 0.465,
      "Eye balls": 0.5,
      "Compact Bone": 0.008,
      "Spongy Bone": 0.025,
      "Blood": 0.6,
      "Muscle": 0.16
    },
    "anisotropy": false,
    "DTI_file_path": null
  }
}
```

**完成消息** (通过 `simnibs_to_backend` 发送):
```json
{
  "id": "inv_001",
  "type": "inverse",
  "progress_rate": 100,
  "message": null,
  "result": {
    "T1_mni": "/data/results/subj_001_T1_mni.nii.gz",
    "TI_file": "/data/results/subj_001_inverse_result.mz3",
    "electrode_A": ["F3", "FC5"],
    "electrode_B": ["P3", "PO7"]
  }
}
```

### 4.5 错误消息示例

```json
{
  "id": "subj_001",
  "type": "model",
  "progress_rate": 25,
  "message": "T1 file not found or invalid format",
  "result": null
}
```

---

## 5. 数据结构实现

### 5.1 模块位置

数据结构和验证器位于 `neuracle/rabbitmq/` 目录下，测试示例位于 `neuracle/demo/` 目录下：

| 文件 | 说明 |
|------|------|
| `schemas.py` | 消息数据类定义（使用 dataclass） |
| `validator.py` | 参数合法性验证函数 |
| `message_builder.py` | 消息构建辅助函数 |
| `simnibs_side_demo.py` | SimNIBS 端测试示例（接收消息并返回结果） |
| `backend_side_demo.py` | Backend 端测试示例（发送消息并接收结果） |

详细实现请参考源代码文件。

---

## 6. 测试说明

### 6.1 架构说明

测试演示 Backend 和 SimNIBS 两个独立进程的通信：

```
┌─────────────┐     backend_to_simnibs      ┌─────────────┐
│   Backend   │ ──────────────────────────► │   SimNIBS   │
│   (发送)    │                             │   (接收)    │
└─────────────┘                             └─────────────┘
       ▲                                           │
       │     simnibs_to_backend                    │
       └──────────────────────────────────────────┘
                     (返回结果)
```

### 6.2 运行步骤

1. **先启动 SimNIBS 端**（持续监听，不自动停止）
2. **再启动 Backend 端**（发送消息并持续监听结果）

### 6.3 运行说明

**SimNIBS 端命令：**
```bash
python -m neuracle.demo.simnibs_side_demo
```

**Backend 端命令：**
```bash
python -m neuracle.demo.backend_side_demo normal # 发送正常测试消息
python -m neuracle.demo.backend_side_demo error # 发送错误测试消息
```

### 6.2 测试场景覆盖矩阵

| 消息类型 | 场景 | 覆盖情况 |
|----------|------|----------|
| model | 必填参数 | T1_file_path, dir_path |
| model | 可选参数 | + T2_file_path, DTI_file_path |
| forward | anisotropy=False | 无 DTI |
| forward | anisotropy=True | + DTI_file_path |
| forward | anisotropy=False with DTI | + DTI_file_path |
| forward | 多电极配置 | 4+ 电极 |
| inverse | Atlas ROI | roi_type=atlas |
| inverse | MNI Position ROI | roi_type=mni_pos |
| inverse | 边界值 | target_threshold=0 |
| 错误消息 | 参数缺失 | 缺少必填字段 |
| 错误消息 | 空值 | id="" |
| 错误消息 | 长度不匹配 | electrode_A 和 current_A |
| 错误消息 | 电流总和不等于0 | current_A=[0.001] |
| 错误消息 | 非法枚举值 | roi_type="invalid" |
| 错误消息 | 负数 | target_threshold=-0.5 |
| 错误消息 | 未知类型 | type="unknown" |
| 进度消息 | 进行中 | progress_rate=0-99 |
| 进度消息 | 完成 | progress_rate=100 |
| 进度消息 | 错误 | message!=null |

---

## 附录

### A. 文件格式说明

| 扩展名 | 格式 | 说明 |
|--------|------|------|
| `.msh` | MSH | SimNIBS 网格文件格式 |
| `.nii.gz` | NIfTI | 3D 医学影像格式 (压缩) |
| `.mz3` | MZ3 | 电场仿真结果格式 |
| `.csv` | CSV | 电极导联配置文件 |

### B. 组织名称与 Tissue Tag 映射

用于将组织名称映射为 SimNIBS 内部使用的 tissue tag：

| 组织名称 | Tissue Tag |
|----------|------------|
| White Matter | 1 |
| Gray Matter | 2 |
| CSF | 3 |
| Bone | 4 |
| Scalp | 5 |
| Eye balls | 6 |
| Compact Bone | 7 |
| Spongy Bone | 8 |
| Blood | 9 |
| Muscle | 10 |
