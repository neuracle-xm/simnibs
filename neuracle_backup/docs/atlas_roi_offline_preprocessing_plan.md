# Atlas ROI 与 MNI 坐标 ROI 接入方案

## 1. 当前实现状态

本文档描述的是**当前代码已经落地的 atlas ROI / MNI ROI 接入方式**，并补充离线预处理的使用要求。

当前实现对应的关键代码位置：

- 在线接入：[neuracle/main.py](/C:/Users/50609/Documents/simnibs/neuracle/main.py)
- ROI 组装：[neuracle/ti_optimize/ti_optimize.py](/C:/Users/50609/Documents/simnibs/neuracle/ti_optimize/ti_optimize.py)
- atlas registry 与 ROI 查找：[neuracle/utils/atlas_utils.py](/C:/Users/50609/Documents/simnibs/neuracle/utils/atlas_utils.py)
- 离线标准化脚本：[neuracle/atlas/scripts/standardize_atlases.py](/C:/Users/50609/Documents/simnibs/neuracle/atlas/scripts/standardize_atlases.py)
- ROI 生成脚本：[neuracle/atlas/scripts/generate_standardized_rois.py](/C:/Users/50609/Documents/simnibs/neuracle/atlas/scripts/generate_standardized_rois.py)
- registry 生成脚本：[neuracle/atlas/scripts/build_standardized_registry.py](/C:/Users/50609/Documents/simnibs/neuracle/atlas/scripts/build_standardized_registry.py)
- validation 脚本：[neuracle/atlas/scripts/validate_standardized_atlases.py](/C:/Users/50609/Documents/simnibs/neuracle/atlas/scripts/validate_standardized_atlases.py)

当前已经支持：

1. `atlas_param.name + area` 在线定位标准化 ROI。
2. BN、JulichBrainAtlas、DiFuMo_64/128/256/512/1024。
3. atlas ROI 以真实 MNI mask 方式传给 SimNIBS。
4. MNI 坐标 ROI 以 `roi_sphere_center_space = "mni"` 方式传给 SimNIBS。
5. DiFuMo 原始 4D `maps.nii.gz` 离线标准化后再 `argmax` 生成算法使用的 3D 离散 atlas。
6. 同名脑区对应多个 component 时，自动合并成一个缓存 mask。

## 2. 非仓库产物说明

`neuracle/atlas/standardized/`、`neuracle/atlas/validation/` 和 `neuracle/atlas/manifests/atlas_registry.json` 属于**离线生成产物**。

这些文件默认不依赖 GitHub 仓库中的现成内容，使用前需要**先在本地离线运行一遍预处理脚本**。也就是说：

- 新机器首次运行 atlas ROI 相关功能前，必须先生成一次标准化 atlas、ROI mask 和 registry。
- 如果只拉代码、不做离线预处理，在线 atlas ROI 会因为找不到标准化 ROI 文件而失败。

## 3. 首次使用前必须执行的离线步骤

项目要求先激活 `conda` 环境：

```powershell
& conda shell.powershell hook | Out-String | Invoke-Expression
conda activate simnibs
```

然后在仓库根目录按顺序执行：

```powershell
python -m neuracle.atlas.scripts.build_standardized_registry
python -m neuracle.atlas.scripts.standardize_atlases
python -m neuracle.atlas.scripts.generate_standardized_rois
python -m neuracle.atlas.scripts.validate_standardized_atlases
```

这四步分别负责：

1. 生成 `atlas_registry.json`
2. 将 atlas 统一到 SimNIBS MNI 模板空间
3. 生成每个脑区的单独 ROI mask
4. 生成基础 validation 报告和 overlay 图

如果 atlas 原始资源有更新，或者脚本逻辑有更新，需要重新执行上述流程。需要覆盖旧产物时可使用：

```powershell
python -m neuracle.atlas.scripts.standardize_atlases --force
python -m neuracle.atlas.scripts.generate_standardized_rois --force
```

## 4. 目录约定

当前目录结构按下面的分层工作：

```text
neuracle/atlas/
├── atlas/                 # 原始 atlas 数据
├── manifests/             # atlas_registry.json
├── scripts/               # 离线预处理脚本
├── standardized/          # 离线生成的标准化 atlas 与 ROI
└── validation/            # 离线生成的校验报告与 overlay 图
```

其中：

- `atlas/` 是原始输入
- `standardized/` 是算法在线阶段真正读取的 atlas 产物
- `manifests/atlas_registry.json` 是在线查找入口
- `validation/` 是离线校验输出

## 5. registry 设计

当前 registry 由 `build_standardized_registry.py` 生成，格式与实现保持一致：

- registry 内部保存的是**仓库相对路径**
- 运行时由 `atlas_utils.py` 自动解析为绝对路径
- 每个脑区对应一个 `index`、`label_en`、`label_zh` 和 `roi_filename`

这样做的原因：

1. 避免把本机绝对路径写死到 JSON。
2. 方便跨机器、跨目录迁移。
3. 在线代码仍然可以直接拿到绝对路径，不需要自己拼接。

## 6. 各 atlas 的当前实现

### 6.1 BN

输入资源：

- `neuracle/atlas/atlas/BN/BN_Atlas_246_1mm.nii.gz`
- `neuracle/atlas/atlas/BN/BN_Atlas_246_labels_zh.csv`

当前实现：

- 按离散 atlas 处理
- 使用最近邻重采样
- 当前校验结果表明，标准化后的 BN atlas 与 SimNIBS 模板 `shape` 与 `affine` 一致

说明：

- BN 很接近 SimNIBS 默认模板
- 工程上仍统一走标准化目录和 ROI 生成目录，避免在线阶段为 BN 单独分支

### 6.2 JulichBrainAtlas

输入资源：

- `neuracle/atlas/atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz`
- `neuracle/atlas/atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv`

当前实现：

- 按离散 atlas 处理
- 使用最近邻重采样
- 生成标准化 atlas 和逐脑区 ROI

### 6.3 DiFuMo

输入资源：

- `neuracle/atlas/atlas/DiFuMo/DiFuMo_*/2mm/maps.nii.gz`
- `neuracle/atlas/atlas/DiFuMo/DiFuMo_*/labels_*_dictionary_zh.csv`

当前实现：

- 以原始 4D `maps.nii.gz` 作为算法权威输入
- 先重采样到 SimNIBS MNI 空间
- 再在目标空间执行 `argmax`
- 生成算法使用的 3D 离散 atlas
- 再按标签表切分成单脑区 ROI

额外说明：

- `standardize_atlases.py` 中已将 DiFuMo 处理改成**流式 argmax**
- 不再把全部 4D 重采样结果 `np.stack` 到内存中
- 这是为了解决 `DiFuMo_1024` 会把内存打满的问题

## 7. 重名脑区的处理

当前实现和原先方案有一个重要差异：

- 原方案里默认 `atlas_param.area` 应当唯一命中一个脑区
- 当前代码已经改为：**如果同名脑区匹配到多个 component，则自动合并成一个缓存 mask**

实现位置在 [atlas_utils.py](/C:/Users/50609/Documents/simnibs/neuracle/utils/atlas_utils.py) 的 `get_standardized_roi_path()`。

这主要是为了适配 DiFuMo 标签表中真实存在的重复命名情况，例如：

- 不同 `Component` 可能共享相同的 `Difumo_names`

当前行为：

1. 按 `label_en` / `label_zh` / `roi_filename stem` / `index` 查找
2. 如果只匹配一个 ROI，直接返回该 ROI
3. 如果匹配多个 ROI，生成 `merged_rois/*.nii.gz` 缓存文件
4. 在线阶段直接使用这个合并后的 mask

因此，当前在线 atlas ROI 已不再要求 `area` 必须唯一。

## 8. 在线阶段的最终接入方式

### 8.1 atlas ROI

在线请求示例：

```json
{
  "roi_type": "atlas",
  "roi_param": {
    "atlas_param": {
      "name": "DiFuMo_1024",
      "area": "Hippocampus anterior LH"
    }
  }
}
```

当前实现流程：

1. `main.py` 调用 `get_standardized_roi_path(...)`
2. 从 registry 中解析 atlas 和 ROI 路径
3. 如果同名脑区有多个 component，则先自动生成合并 mask
4. 将 ROI 以 `mask_space = "mni"` 的方式交给 `TesFlexOptimization`

最终传给 SimNIBS 的 ROI 配置是：

- `roi.method = "surface"`
- `roi.surface_type = "central"`
- `roi.mask_path = <standardized_roi_mask>`
- `roi.mask_space = "mni"`
- `roi.mask_value = 1`

### 8.2 MNI 坐标 ROI

在线请求示例：

```json
{
  "roi_type": "mni_pos",
  "roi_param": {
    "mni_param": {
      "center": [-38.5, -22.5, 58.3],
      "radius": 15.0
    }
  }
}
```

当前实现流程：

1. `main.py` 读取 `center` 与 `radius`
2. 不在业务层预先转换到 subject
3. 直接将球形 ROI 以 `roi_sphere_center_space = "mni"` 的形式传给 SimNIBS

最终传给 SimNIBS 的 ROI 配置是：

- `roi.method = "surface"`
- `roi.surface_type = "central"`
- `roi.roi_sphere_center = center`
- `roi.roi_sphere_radius = radius`
- `roi.roi_sphere_center_space = "mni"`

## 9. 当前 validation 的实际含义

当前 `validate_standardized_atlases.py` 已实现的是**轻量空间一致性校验**，不是完整 QA。

它当前会输出：

- `validation_report.json`
- 每个 atlas 一张中间 `z` 切面的 overlay PNG

`validation_report.json` 当前检查：

1. 标准化 atlas 文件是否存在
2. `shape` 是否与 SimNIBS 模板一致
3. `affine` 是否与 SimNIBS 模板一致
4. `roi_dir` 是否存在

所以当前 validation 能说明：

- atlas 标准化产物已经生成
- atlas 已经落到 SimNIBS 模板空间对应网格
- ROI 输出目录存在

但当前 validation 还**没有**覆盖：

- 逐脑区 ROI 数量和标签表的严格一致性
- 每个标签值是否都被正确保留
- 多切面的 overlay 检查
- 每个 ROI 映射到 subject 后的完整人工验收

## 10. demo 与当前代码的一致性

当前已经存在并可运行的 demo：

- `neuracle/demo/atlas_bn_roi_demo.py`
- `neuracle/demo/atlas_julich_roi_demo.py`
- `neuracle/demo/atlas_difumo_roi_demo.py`
- `neuracle/demo/mni_roi_demo.py`

说明：

1. BN demo 选用 `A8m_L`
2. Julich demo 选用 `Area 3b (PostCG)_lh`
3. DiFuMo demo 已覆盖 `64/128/256/512/1024`
4. 只有 `DiFuMo_1024` 明确提供了左侧海马 `Hippocampus anterior LH`
5. `DiFuMo_128/256/512` 只能选择最接近海马的 component，标签本身不分左右
6. `DiFuMo_64` 没有海马相关标签，demo 中使用替代脑区

demo 使用的 subject 数据为：

- `data/m2m_ernie/T1.nii.gz`

demo 输出目录为：

- `data/roi_demo_outputs/`

## 11. 与原方案相比的已知差异

以下内容与最初的规划相比已经发生调整，并以当前代码为准：

1. 本文档不再是“仅描述方案，不修改代码”，当前代码已经实现了核心链路。
2. atlas ROI 在线阶段不是在业务层显式调用 `mni_mask_to_sub(...)`，而是将 `mask_space = "mni"` 的 ROI 直接交给 SimNIBS。
3. `atlas_registry.json` 现在使用仓库相对路径，而不是绝对路径。
4. 重名脑区不再报歧义，而是自动合并为一个缓存 mask。
5. validation 当前只实现了轻量空间校验，尚未覆盖完整标签一致性 QA。
6. DiFuMo 标准化实现已经特别处理了内存峰值问题。

## 12. 结论

当前 atlas ROI / MNI ROI 方案已经可以这样总结：

1. atlas 必须先离线标准化到 SimNIBS MNI 空间。
2. atlas ROI 在线阶段只读取离线生成的标准化 ROI。
3. atlas ROI 通过 `mask_space = "mni"` 交给 SimNIBS。
4. MNI 球形 ROI 通过 `roi_sphere_center_space = "mni"` 交给 SimNIBS。
5. DiFuMo 重名 component 会在在线查找阶段自动合并。
6. 首次部署或换机器后，必须先本地离线运行一遍 atlas 预处理脚本，否则 atlas ROI 功能不可用。
