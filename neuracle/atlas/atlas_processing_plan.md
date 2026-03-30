# 脑图谱整理实施方案

## 目标

对 `atlas/` 目录下的脑图谱数据做统一整理，完成以下 4 项工作：

1. 将记录脑区 `label` 的文件中的脑区名称翻译为中文。
2. 对缺少 `RGBA` 颜色表的图谱补充用于可视化的常用颜色表。
3. 将 `JulichBrainAtlas` 中的左右半球数据合并为一份，包括 `label` 和 `nii`。
4. 将所有 `label` 文件统一整理为 `.csv` 格式。

## 当前目录现状

当前仓库的目录角色如下：

- `atlas/`：存放各类 atlas 原始数据。
- `scripts/`：存放当前已经写好的下载、翻译映射和标签表生成脚本。
- `.venv/`：当前项目虚拟环境。

### 1. BN

- `atlas/BN/BN_Atlas_246_1mm.nii.gz`
- `atlas/BN/BN_Atlas_246_LUT.txt`

说明：
- `BN_Atlas_246_LUT.txt` 已经是带编号、标签名和 `RGBA` 的文本表。
- 该图谱主要需要做中文翻译和格式统一检查。

### 2. DiFuMo

- `atlas/DiFuMo/DiFuMo_64/labels_64_dictionary.csv`
- `atlas/DiFuMo/DiFuMo_128/labels_128_dictionary.csv`
- `atlas/DiFuMo/DiFuMo_256/labels_256_dictionary.csv`
- `atlas/DiFuMo/DiFuMo_512/labels_512_dictionary.csv`
- `atlas/DiFuMo/DiFuMo_1024/labels_1024_dictionary.csv`
- 对应 `atlas/DiFuMo/DiFuMo_*/2mm/maps.nii.gz`
- 对应 `atlas/DiFuMo/DiFuMo_*/3mm/maps.nii.gz`

说明：
- `labels_*_dictionary.csv` 里记录组件名称和网络信息。
- 当前已存在统一翻译映射表 `atlas/DiFuMo/DiFuMo_translation_mapping.csv`，供 `64/128/256/512/1024` 共用。
- 当前已生成各维度中文标签文件：
  - `atlas/DiFuMo/DiFuMo_64/labels_64_dictionary_zh.csv`
  - `atlas/DiFuMo/DiFuMo_128/labels_128_dictionary_zh.csv`
  - `atlas/DiFuMo/DiFuMo_256/labels_256_dictionary_zh.csv`
  - `atlas/DiFuMo/DiFuMo_512/labels_512_dictionary_zh.csv`
  - `atlas/DiFuMo/DiFuMo_1024/labels_1024_dictionary_zh.csv`
- `DiFuMo` 的 `maps.nii.gz` 是 4D 成分权重图/软分配图。
- 每个体素不是单个离散标签值，而是一个长度为 `K` 的成分权重向量，其中 `K` 为 64、128、256、512 或 1024。
- 当前已在所有 2mm 版本下派生 3D winner-take-all 离散 atlas：
  - `atlas/DiFuMo/DiFuMo_64/2mm/DiFuMo64.nii.gz`
  - `atlas/DiFuMo/DiFuMo_128/2mm/DiFuMo128.nii.gz`
  - `atlas/DiFuMo/DiFuMo_256/2mm/DiFuMo256.nii.gz`
  - `atlas/DiFuMo/DiFuMo_512/2mm/DiFuMo512.nii.gz`
  - `atlas/DiFuMo/DiFuMo_1024/2mm/DiFuMo1024.nii.gz`
- 当前还额外保存了与 `DiFuMo` 配套的模板文件：
  - `atlas/DiFuMo/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz`
  - `atlas/DiFuMo/tpl-MNI152NLin6Asym_res-02_T1w_difumo-grid.nii.gz`
- `atlas/DiFuMo/DiFuMo_1024/2mm/resampled_maps.nii.gz` 为中间文件，在本方案中不作为标准输入，后续处理统一以各目录下原始 `maps.nii.gz` 为准。

### 3. JulichBrainAtlas

- `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.nii.gz`
- `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.xml`
- `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.nii.gz`
- `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.xml`

说明：
- 这里按“将 `lh` + `rh` 合并成一份双侧数据”执行。
- `xml` 可作为标签来源，最终统一导出为 `.csv`。
- 当前已建立独立翻译映射表 `atlas/JulichBrainAtlas/JulichBrainAtlas_translation_mapping.csv`，并已生成：
  - `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv`
  - `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz`
- 当前目录中还存在一个额外模板文件 `atlas/JulichBrainAtlas/mni_icbm152_t1_tal_nlin_asym_09c.nii`。

## 执行原则

### 标签统一格式

统一输出为 UTF-8 编码的 `.csv`。

当前项目按图谱类型采用以下字段：

```txt
BN:
index,label_en,label_zh,r,g,b,a

DiFuMo:
index,label_en,label_zh,r,g,b,a

JulichBrainAtlas:
index,label_en,label_zh,r,g,b,a
```

说明：
- `index`：标签值或组件编号。
- `label_en`：英文原名。
- `label_zh`：中文译名。
- `r g b a`：RGBA 颜色。

### 中文翻译原则

- 优先保留专业解剖学术语的一致性。
- 缩写不直接删除，必要时保留在中文后面。
- 左右半球统一写法：
  - `_L` -> `左侧`
  - `_R` -> `右侧`
  - `lh` -> `左半球`
  - `rh` -> `右半球`
- 最终标签文件优先输出纯中文，不保留英文括注。
- 对缩写或结构名复杂的 atlas，先建立独立翻译映射表，再据此生成最终标签文件。

### RGBA 颜色表原则

缺少颜色表时，采用常用离散色板生成颜色。

这里需要明确区分两个概念：

1. 色板名称
2. 最终写入标签文件的 `RGBA` 数值

色板名称只是生成颜色的方法来源，例如 `tab20`、`Set3`、`glasbey`。  
实际在可视化软件中真正起作用的，不是色板名称，而是每个标签最终对应的 `R G B A` 数值。

也就是说：

- 软件即使不知道颜色来自哪个色板，只要读到了具体的 `RGBA` 值，也可以正常染色。
- 因此最终交付物中，核心是“每个标签都有稳定、明确的 `RGBA` 值”。
- 色板名称仅用于批量生成颜色，属于实现策略，不是可视化的必需字段。

示例：

```txt
1   Frontal_Area   额叶区   255 0 0 255
2   Temporal_Area  颞叶区   0 255 0 255
3   Occipital_Area 枕叶区   0 0 255 255
```

对于上面的标签表：

- 可视化软件只需要读取最后 4 列 `RGBA`
- 并不需要知道这些值最初是由哪一种色板生成的

生成颜色时可参考以下常用离散色板：

1. `tab20`
2. `Set3`
3. `glasbey` 类高区分度调色板

约束：
- 背景值 `0` 统一保留为 `0 0 0 0`。
- 非背景标签统一采用 `alpha=255`，用于可视化。
- 相邻编号尽量避免颜色过于接近。

说明：
- 颜色表用途明确为可视化，不再优先兼容 `BN` 的原始 alpha 习惯。
- `BN` 若保留原始文件，也将同步额外输出一份统一可视化格式的标签表。
- 对最终输出文件而言，重点是保留稳定的 `RGBA` 数值，而不是保留色板名称。
- 如无额外需要，标签文件中可以不单独增加“palette”字段。

## 分图谱处理方案

### 一、BN

处理内容：

1. 读取 `BN_Atlas_246_LUT.txt`。
2. 翻译全部脑区名称为中文。
3. 将颜色字段整理为统一的可视化 `RGBA` 格式。
4. 输出统一格式的 `.csv`。

建议输出：

- `atlas/BN/BN_abbreviation_mapping.csv`
- `atlas/BN/BN_Atlas_246_labels_zh.csv`

备注：
- 原文件已经是 `.txt`，但最终整理结果统一导出为 `.csv`。
- `BN_abbreviation_mapping.csv` 单独保存缩写、英文全称和中文全称的对应关系。

### 二、DiFuMo

处理内容：

1. 读取各分辨率下的 `labels_*_dictionary.csv`。
2. 翻译 `Difumo_names` 为中文。
3. 为每个组件编号生成用于可视化的 `RGBA`。
4. 将 `csv` 转为统一 `.csv` 标签文件。
5. 基于原始 4D `maps.nii.gz` 生成一份 3D winner-take-all 离散 atlas：
   - 对每个体素，取第 4 维上概率最大的那个成分；
   - 将该成分编号写入对应体素；
   - 若该体素所有成分值都为 `0`，则写为背景 `0`。
6. 为这份 3D 离散 atlas 配套标签说明文件与颜色表。

建议输出：

- `atlas/DiFuMo/DiFuMo_translation_mapping.csv`
- `atlas/DiFuMo/DiFuMo_64/labels_64_dictionary_zh.csv`
- `atlas/DiFuMo/DiFuMo_128/labels_128_dictionary_zh.csv`
- `atlas/DiFuMo/DiFuMo_256/labels_256_dictionary_zh.csv`
- `atlas/DiFuMo/DiFuMo_512/labels_512_dictionary_zh.csv`
- `atlas/DiFuMo/DiFuMo_1024/labels_1024_dictionary_zh.csv`
- `atlas/DiFuMo/DiFuMo_64/2mm/DiFuMo64.nii.gz`
- `atlas/DiFuMo/DiFuMo_128/2mm/DiFuMo128.nii.gz`
- `atlas/DiFuMo/DiFuMo_256/2mm/DiFuMo256.nii.gz`
- `atlas/DiFuMo/DiFuMo_512/2mm/DiFuMo512.nii.gz`
- `atlas/DiFuMo/DiFuMo_1024/2mm/DiFuMo1024.nii.gz`
- `atlas/DiFuMo/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz`
- `atlas/DiFuMo/tpl-MNI152NLin6Asym_res-02_T1w_difumo-grid.nii.gz`

备注：
- 原始 4D `maps.nii.gz` 保留不改。
- 当前 2mm 版本已经使用派生出的 `DiFuMo*.nii.gz` 进行离散 atlas 可视化。
- 原始 4D 图更适合保留用于后续个体空间映射和连续权重分析。
- `DiFuMo_translation_mapping.csv` 只维护一份，供 `64/128/256/512/1024` 各版本共用。
- 当前尚未为 3mm 版本生成对应离散 atlas，如后续需要可按同一规则补齐。

### 三、JulichBrainAtlas

处理内容分两部分。

#### 1. 标签文件合并

1. 解析 `lh.xml` 与 `rh.xml`。
2. 提取每个结构的：
   - `grayvalue`
   - `id`
   - 英文标签
   - 半球信息
3. 翻译为中文。
4. 合并成一份统一 `.csv`。

其中，`Julich` XML 中几个字段的角色定义如下：

- `grayvalue`
  - 是与 `nii.gz` 体素值直接对应的标签值
  - 是合并、重编号、配色和标签映射时的核心字段
- `id`
  - 是 `Julich` 体系内部的结构标识符
  - 不直接作为 `nii.gz` 的体素值使用
  - 当前最终标签文件中不输出，但可在独立翻译映射表中保留或后续追溯使用
- `num`
  - 主要是 XML 中的记录顺序号
  - 不作为标签值使用
  - 在当前任务中不保留到最终输出文件中，避免与真实标签编号混淆

合并规则为：重新编号，左半球与右半球使用不同的新编号。

原因：

- 单个合并后的 `nii` 需要唯一标签值。
- 如果左右半球都复用同一 `grayvalue`，合并后将无法只靠数值区分半球。

建议编号规则：

- 左半球：沿用原编号或原 `grayvalue`
- 右半球：在左半球最大值基础上整体偏移

示例：

```txt
1   Area 3b (PostCG)_lh   中央后回3b区（左半球）   ...   julich_id=127
208 Area 3b (PostCG)_rh   中央后回3b区（右半球）   ...   julich_id=127
```

`Julich` 最终标签文件建议字段如下：

```txt
index,label_en,label_zh,r,g,b,a
```

说明：

- `index`：合并并重编号后的唯一标签值，用于与合并后的 `nii.gz` 对应
- 最终标签文件不再输出 `source` 与 `julich_id`
- `JulichBrainAtlas_translation_mapping.csv` 单独保存原始结构名与中文翻译映射

建议输出：

- `atlas/JulichBrainAtlas/JulichBrainAtlas_translation_mapping.csv`
- `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv`

#### 2. NIfTI 合并

1. 读取 `lh.nii.gz` 与 `rh.nii.gz`。
2. 检查两者空间信息是否一致：
   - shape
   - affine
   - voxel size
3. 对右半球图像的标签值按新编号规则整体偏移。
4. 将左右半球体素合并到一个新 `nii.gz`。
5. 输出合并后的双侧图谱。

建议输出：

- `atlas/JulichBrainAtlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz`

备注：
- 如果左右图存在重叠体素，需要定义冲突处理规则，建议优先保留非零值，若同一体素左右都非零则记录日志并人工复核。

## 实施顺序

建议按以下顺序执行：

1. 先做标签表整理：
   - BN
   - DiFuMo
   - Julich XML
2. 再统一翻译与可视化颜色表字段。
3. 已完成 `BN`、`DiFuMo`、`JulichBrainAtlas` 的主要标签文件整理与中文化。
4. 已完成 `JulichBrainAtlas` 的双侧 `nii.gz` 合并。
5. 已完成 `DiFuMo` 全部 2mm 版本的离散 atlas 生成，以及配套 `MNI152NLin6Asym` 模板下载与同网格模板生成。
6. 对全部输出做格式校验。

## 结果验证方案

验证分为两层：

1. 文件级验证
2. 结果级验证

原则：

- 不能只以“脚本运行完成且无报错”作为正确性的判断依据。
- 必须同时检查空间一致性、数据结构、标签合法性和可视化落点是否合理。

### 一、文件级验证

文件级验证用于确认输出在格式和空间元数据上是成立的。

#### 1. 输出文件完整性

需要检查：

- 预期输出文件是否全部生成。
- 文件命名是否符合当前 atlas 目录下的约定。
- 输出文件是否落在预期 atlas 子目录中。

#### 2. 空间信息一致性

凡是同一 atlas 体系下互相配套使用的结果，都应保持空间信息一致。

至少检查以下内容：

- shape
- affine
- voxel spacing
- orientation
- voxel grid

约束：

- 同一 atlas 的图像文件、派生离散 atlas 和配套模板之间应保持预期网格关系。
- 若 shape 一致但 affine 或 orientation 不一致，仍应判定为异常。

#### 3. 维度检查

不同类型 atlas 的输出维度应符合预期：

- `DiFuMo` 原始连续图应为 4D。
- `DiFuMo` 的离散 atlas 应为 3D。
- `BN` 与 `Julich` 应为 3D。

#### 4. 数据类型和值域检查

需要区分连续图和离散图：

- 连续图：
  - `DiFuMo` 原始 4D 图允许浮点数。
- 离散图：
  - `BN`
  - `Julich`
- `DiFuMo` 派生离散 atlas

对离散图必须检查：

- 体素值应为整数标签或可安全转换为整数标签。
- 不应出现由错误插值产生的大量非整数值。
- 标签值范围应合理，不应出现明显超出标签表编号范围的值。

#### 5. 标签文件对应性检查

输出图像必须能和标签表一一对应。

至少检查：

- 图像中的非零标签值是否都能在对应标签 `.csv` 的 `index` 列中找到。
- 标签表中是否存在大量未被图像使用的编号。

说明：

- “标签表中存在未出现编号”不一定是错误，因为某些区域可能在当前空间或当前样例中未出现。
- 但“图像里出现标签表没有的编号”通常应视为明确错误。

### 二、结果级验证

结果级验证用于确认“虽然文件格式正确，但内容是否真的对齐到正确空间”。

#### 1. 可视化叠加检查

将同一 atlas 体系下的输出图与对应模板做 overlay，检查以下内容：

- 脑区是否总体落在脑实质范围内，而不是明显偏到脑外。
- Atlas 是否和对应模板轮廓大体吻合。
- 是否出现整体平移、缩放异常或明显错位。

这是识别“形变场方向用反了”“affine 没对齐”“目标网格写错了”最直接的方法。

#### 2. 左右方向检查

需要确认：

- 左右半球没有镜像翻转。
- `Julich` 合并后的结果仍保持正确半球分布。

如果出现：

- 左侧结构落到右脑
- 右侧结构落到左脑

通常提示方向解释、坐标系或形变场方向有问题。

#### 3. 插值策略检查

不同 atlas 的结果外观应符合各自插值策略。

`DiFuMo` 原始 4D 连续图：

- 应表现为连续、平滑的强度变化。
- 不应出现明显最近邻插值导致的块状边界。

`BN`、`Julich`、`DiFuMo` 派生离散 atlas：

- 应表现为离散分区。
- 标签边界可以是分段式的，但不应出现大量小数标签。

#### 4. 背景检查

需要检查背景值是否仍然合理保留：

- 背景应主要保持为 `0`。
- 不应出现整个体积几乎都被非零标签填满的异常结果。
- `DiFuMo` 派生离散 atlas 中原本全零位置在目标空间仍应尽量保持为背景 `0`。

#### 5. 标签覆盖与分布检查

需要对输出图中的标签分布做基本统计：

- 统计唯一标签值数量。
- 统计各标签体素数。
- 检查是否存在异常孤立标签或极少量随机标签噪声。

对于 demo：

- 不要求每个标签都必须出现。
- 但若只出现极少数标签，或出现大量不合理零散标签，通常说明处理流程有问题。

### 三、按 atlas 类型的专项验证

#### 1. DiFuMo 连续图

检查项：

- 输出应为 4D。
- 第 4 维长度应与组件数一致，例如 64、128、256、512、1024。
- 连续值结果应可视化为平滑权重图。
- 若后续从该结果生成离散 atlas，两者应来自同一空间网格。

#### 2. DiFuMo 离散 atlas

检查项：

- 输出应为 3D。
- 体素值应为 `0..K` 范围内的整数，其中 `0` 为背景。
- 非零标签值应能在对应 `labels_*_dictionary_zh.csv` 中找到。
- 不应出现超出组件编号范围的值。

#### 3. BN

检查项：

- 输出应为 3D 离散标签图。
- 标签值应保持整数。
- 非零标签值应能映射到 `atlas/BN/BN_Atlas_246_labels_zh.csv`。
- 若与原始 MNI atlas 相比出现异常碎裂或大量缺失，需复查最近邻重采样过程。

#### 4. Julich

检查项：

- 输入必须是已合并好的双侧 atlas。
- 输出应为 3D 离散标签图。
- 标签编号应与双侧合并后的标签表一致。
- 左右半球结构在 `T1` 上的分布应符合实际解剖位置。

### 四、建议自动化校验项

建议后续增加一个统一校验脚本，例如：

- `scripts/validate_outputs.py`

脚本可自动执行以下检查：

1. 检查预期输出文件是否存在。
2. 读取各输出图像，比较 shape、affine 和 spacing。
3. 检查图像维度是否符合 atlas 类型预期。
4. 对离散 atlas 检查是否近似整数标签。
5. 统计唯一标签值，并与对应标签表的 `index` 集合比对。
6. 统计非零体素比例，识别明显异常结果。

### 五、建议人工复核方式

自动校验通过后，仍建议至少做一轮人工复核。

建议方式：

1. 随机选择轴状、冠状、矢状若干层面查看 overlay。
2. 对 `BN` 或 `Julich` 随机抽取几个解剖上容易识别的脑区，确认位置大体合理。
3. 对 `DiFuMo` 检查连续图、离散图与配套模板是否位于同一预期网格关系下。

### 六、通过标准

一个 atlas 整理结果可暂时视为“通过”，至少需要同时满足：

1. 输出文件齐全。
2. 配套图像之间的空间元数据一致或符合预期对应关系。
3. 连续图/离散图的维度与数据类型正确。
4. 离散标签图未出现明显错误插值造成的小数标签。
5. 标签编号与标签表可对应。
6. 可视化叠加后，atlas 位置总体合理，没有明显翻转、偏移或跑出脑外。

## 技术实现建议

建议使用 Python 脚本统一处理，原因：

- 便于解析 `csv/xml/txt/nii.gz`
- 便于批量翻译映射和颜色生成
- 便于输出统一格式

建议脚本拆分：

1. `translate_labels.py`
   - 负责读取标签文件
   - 维护英文到中文的翻译映射
2. `generate_rgba.py`
   - 负责为缺失颜色表的数据生成用于可视化的 RGBA
3. `build_difumo_wta.py`
   - 负责将 4D `maps.nii.gz` 转成 3D winner-take-all 离散 atlas
4. `merge_julich.py`
   - 负责 XML 合并和 NIfTI 合并
5. `validate_outputs.py`
   - 负责检查字段完整性、编号唯一性、颜色合法性

## 校验项

完成后至少检查以下内容：

1. 全部标签文件都已转成 `.csv`。
2. 所有 `.csv` 都是 UTF-8 编码。
3. 每条标签都有：
   - 编号
   - 英文名
   - 中文名
   - RGBA
4. `Julich` 合并后标签编号唯一。
5. `Julich` 合并后的 `nii.gz` 与标签表一一对应。
6. `BN` 原有颜色未被破坏。
7. `DiFuMo` 每个分辨率版本的组件数与标签数一致。
8. `DiFuMo` 派生离散 atlas 中标签值范围与标签表一致。
9. `DiFuMo` 中全零体素被正确保留为背景 `0`。

## 风险与注意事项

1. `Julich` 的左右半球 XML 看起来标签文本相同，不能只拼接文本，必须配套重编号策略。
2. `DiFuMo` 原始数据是成分图谱，派生为 winner-take-all 后会损失“一个体素属于多个成分”的信息。
3. 部分脑区英文名称可能存在多种中文译法，需要统一词表。
4. 如果后续软件依赖特定 LUT 格式，可能还需要再导出一份兼容格式。

## 预期产物

本轮整理完成后，目录中应新增：

- 每个图谱对应的中文 `.csv` 标签文件
- `BN`、`DiFuMo`、`JulichBrainAtlas` 对应的独立翻译映射表 `.csv`
- `DiFuMo` 各分辨率、各空间版本对应的离散 atlas 文件
- `JulichBrainAtlas` 的双侧合并 `.nii.gz`
- 可选：翻译词表与处理日志
