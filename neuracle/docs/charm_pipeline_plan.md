# CHARM Pipeline 步骤拆分方案

## 一、背景

`simnibs/segmentation/charm_main.py` 中的 `run()` 函数包含完整的 CHARM 分割流程。本方案将其拆分为独立的步骤脚本，支持单独运行某个步骤。

## 二、步骤划分

| 步骤 | 脚本文件             | 功能                         | 对应 charm_main.py     |
| ---- | -------------------- | ---------------------------- | ---------------------- |
| 1    | `prepare_t1.py`      | T1 图像准备与格式转换        | `_prepare_t1()`        |
| 2    | `prepare_t2.py`      | T2 图像配准与准备            | `_prepare_t2()`        |
| 3    | `denoise.py`         | 输入图像降噪                 | `_denoise_inputs()`    |
| 4    | `init_atlas.py`      | atlas 初始仿射配准与颈部校正 | `initatlas` 阶段       |
| 5    | `segment.py`         | 体积与表面分割               | `segment` 阶段         |
| 6    | `create_surfaces.py` | 皮层表面重建                 | `create_surfaces` 阶段 |
| 7    | `mesh.py`            | 四面体网格生成               | `mesh_image` 阶段      |

## 三、目录结构

```
neuracle/charm/
├── __init__.py
├── prepare_t1.py      # 步骤1: T1准备
├── prepare_t2.py      # 步骤2: T2准备
├── denoise.py         # 步骤3: 降噪
├── init_atlas.py      # 步骤4: atlas初始化
├── segment.py         # 步骤5: 分割
├── create_surfaces.py # 步骤6: 表面重建
├── mesh.py            # 步骤7: 网格生成
└── utils.py           # 公共工具函数
```

## 四、每个步骤的输入输出

### 步骤1: prepare_t1.py
- **输入**: 原始 T1 NIfTI 文件
- **输出**: `m2m_{subid}/T1fs.nii.gz`
- **功能**: 格式转换、维度检查、qform/sform 校验

### 步骤2: prepare_t2.py
- **输入**: 原始 T2 NIfTI 文件 + T1fs.nii.gz
- **输出**: `m2m_{subid}/T2_reg.nii.gz`
- **功能**: T2-to-T1 刚性配准（可选）

### 步骤3: denoise.py
- **输入**: T1fs.nii.gz, T2_reg.nii.gz（可选）
- **输出**: `m2m_{subid}/T1fs_denoised.nii.gz`, `T2_reg_denoised.nii.gz`（可选）
- **功能**: SANLM 滤波降噪

### 步骤4: init_atlas.py
- **输入**: T1fs.nii.gz（+ T2_reg.nii.gz）
- **输出**: `m2m_{subid}/segmentation/template_coregistered.nii.gz`
- **功能**: MNI atlas 仿射配准 + 颈部校正

### 步骤5: segment.py
- **输入**: template_coregistered.nii.gz + 组织概率图
- **输出**: `m2m_{subid}/segmentation/tissue_labeling_upsampled.nii.gz`
- **功能**: 分割、偏置场校正、形态学操作

### 步骤6: create_surfaces.py
- **输入**: tissue_labeling_upsampled.nii.gz + norm_image.nii.gz
- **输出**: `m2m_{subid}/surfaces/{lh,rh}.*` 表面文件
- **功能**: 皮层表面重建（white、pial、central）

### 步骤7: mesh.py
- **输入**: tissue_labeling_upsampled.nii.gz
- **输出**: `m2m_{subid}/{subid}.msh`
- **功能**: 四面体网格生成 + EEG 位置变换

## 五、配置文件

各步骤依赖 `charm.ini` 中的设置项：

| 步骤            | 依赖的配置节                                   |
| --------------- | ---------------------------------------------- |
| prepare_t1      | 无                                             |
| prepare_t2      | 无                                             |
| denoise         | `preprocess.denoise`                           |
| init_atlas      | `samseg.init_type`, `initmni.*`, `initatlas.*` |
| segment         | `segment.*`, `samseg.*`                        |
| create_surfaces | `surfaces.*`, `samseg.conductivity_mapping`    |
| mesh            | `mesh.*`                                       |

## 六、使用方式

```bash
# 运行完整流程
python -m neuracle.charm.prepare_t1 <subid> <T1_file>
python -m neuracle.charm.prepare_t2 <subid> <T2_file>
python -m neuracle.charm.denoise <subid>
python -m neuracle.charm.init_atlas <subid>
python -m neuracle.charm.segment <subid>
python -m neuracle.charm.create_surfaces <subid>
python -m neuracle.charm.mesh <subid>

# 或通过 argparse（类似原有 charm CLI）
python -m neuracle.charm.run --prepare-t1 <subid> <T1_file>
python -m neuracle.charm.run --prepare-t2 <subid> <T2_file>
# ... etc
```
