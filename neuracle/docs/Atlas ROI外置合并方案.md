# Atlas ROI 外置合并方案

## 1. 背景与目标

**现有业务运行时**：`neuracle/atlas/standardized/<atlas>/rois/` 将 atlas 切分为逐脑区二值 mask。TI 逆向优化在 `neuracle/ti_inverse.py` 中根据用户选择的 `atlas name + area` 调用 `get_standardized_roi_path()`，随后把得到的单脑区 mask 作为 MNI 空间 ROI 传给 SimNIBS。

**现有业务运行时**：部分单脑区很小。它们经过 MNI 到个体空间的非线性变换、离散重采样，再与个体 mesh 中的 WM/GM 元素相交后，可能只剩很少甚至没有可用于优化的元素，使逆向优化对个体差异和离散误差非常敏感。

本方案的目标是：

1. 本次只做合并 ROI 试验，不修改前端、参数结构或 TI 逆向业务调用入口。
2. 通过 `neuracle/ti_optimization/demo/` 下的独立 demo 调用合并逻辑并验证结果。
3. 对每个已注册 atlas 固定选择实测体积最小的 5 个 ROI，验证其自动合并结果；DiFuMo64 作为不合并的 `keep` 对照。
4. 保留原始 atlas、标准化 atlas、逐脑区 ROI 及其 registry，不覆盖、不改名、不重新编号。
5. 全部匹配和取舍使用确定性自动规则，不保留人工审核分支；输出实际指标以便评估规则效果。

本方案只设计合并机制，不在本轮修改 atlas 数据或现有代码。

### 1.1 阶段术语

全文统一使用以下阶段标签：

| 标签 | 是否依赖个体头模 | 本方案中的含义 |
|---|---|---|
| **现有业务运行时** | 是 | 当前 `ti_inverse.py` 的真实调用路径，仅用于说明现状；本次试验不修改该路径 |
| **离线/Atlas 阶段** | 否 | 只读取标准化 atlas 和单区 ROI，执行体积统计、父区匹配、自动分组、小 ROI 选样、mask 并集及 MNI 空间校验 |
| **Demo 运行时/个体阶段** | 是 | demo 读取 `subject_dir`，将原 ROI 和合并 ROI 映射到个体头模，统计 WM/GM mesh 元素，并可选择运行完整优化 |

“离线/Atlas 阶段”是处理边界，不代表新增独立命令入口。初版由 demo 在进入个体头模处理前调用这些内部函数，产物仍只写入 demo 输出目录。

## 2. 当前实现与问题证据

### 2.1【现有业务运行时】当前调用路径

```text
params.json 中的 atlas name + area
    -> get_standardized_roi_path()
    -> 单脑区标准化 MNI mask
    -> setup_electrodes_and_roi()
    -> 与个体 WM/GM mesh 相交
    -> TI focality 逆向优化
```

`get_standardized_roi_path()` 当前已有一种“合并”，但它只处理 **同名脑区匹配到多个 component** 的情况，主要用于 DiFuMo 重名标签。该行为不是解剖父区合并，且被 atlas demo 等其他调用方共用，因此不应直接扩展成逆向优化专用的自动父区合并。

### 2.2【离线/Atlas 阶段】当前标准化产物的体积基线

以下数据由当前本地 `atlas_registry.json` 和标准化离散 atlas 统计得到，体素体积均为 1 mm³：

| Atlas | 注册 ROI 数 | 最小体积 | 中位体积 | 90% 分位体积 |
|---|---:|---:|---:|---:|
| Brainnetome 246 | 246 | 661 mm³ | 4297 mm³ | 7867 mm³ |
| Julich bilateral | 414 | 5 mm³ | 1137 mm³ | 4635 mm³ |
| DiFuMo64 | 64 | 8917 mm³ | 17912 mm³ | 25490 mm³ |
| DiFuMo128 | 128 | 5874 mm³ | 10644 mm³ | 15520 mm³ |
| DiFuMo256 | 256 | 3631 mm³ | 6396 mm³ | 10270 mm³ |
| DiFuMo512 | 512 | 1653 mm³ | 3294 mm³ | 4887 mm³ |
| DiFuMo1024 | 1024 | 863 mm³ | 1712 mm³ | 2264 mm³ |

这说明问题在 Julich 和高分辨率 DiFuMo 上尤其明显；同时 DiFuMo64 已经相对较大，不能不分图谱地继续粗暴扩张。

### 2.3【跨阶段说明】合并能够解决和不能保证解决的问题

离线/Atlas 阶段只负责生成更大的合并 MNI mask；它本身不能证明个体上的有效元素已经增加。是否真正提高了有效采样数，只能在 Demo 运行时/个体阶段将 mask 映射到头模并与 WM/GM mesh 相交后判断。

但合并不保证逆向优化一定成功，原因包括：

- **可选 Demo 完整优化阶段**：电极候选位置和电流约束可能仍无法覆盖目标；
- **可选 Demo 完整优化阶段**：focality 的 ROI 下限和 Non-ROI 上限可能互相冲突；
- **可选 Demo 完整优化阶段**：目标过大时，要求更大范围同时达到阈值可能反而更难；
- **Demo 运行时/个体阶段**：无论标签名称是什么，合并 mask 都可能与当前允许的 WM/GM mesh 元素没有交集或交集过少；单纯扩大不能替代实际匹配校验。

因此，方案追求的是“大小适中且语义稳定的父区”，而不是越大越好。

## 3.【离线/Atlas 阶段】合并原则及理由

本节全部针对标准化 MNI atlas 的自动合并，不读取个体头模，也不判断个体 WM/GM mesh 匹配结果。

### 3.1 优先使用官方层级，其次使用 mask 实际重叠

如果 atlas 提供可追溯的官方区域树或脑区编号到父区的显式映射，则直接使用该层级；没有官方层级时，才通过标准化 mask 之间的实际体素重叠建立试验性父子关系。空间上最近但没有层级或实际重叠依据的脑区不自动合并，因为最近邻不一定属于同一解剖或功能系统。

名称通常不用于猜测候选。Julich 是明确例外：本地标签名只经过固定格式归一化后，与官方 Julich-Brain v3.1 区域树的叶节点做**精确匹配**，用于确认“本地 ROI 是官方树中的哪个节点”；真正的父子关系仍来自官方区域树，不从名称中的括号、缩写或关键词自行推断。

### 3.2 默认保持左右半球分离

有官方左右编号映射时按该映射保持同侧；Julich 的 `_lh`、`_rh` 只在名称精确匹配官方树时分别归一化为官方节点的 ` - left hemisphere`、` - right hemisphere`，随后按官方节点侧别保持同侧；通过 mask 匹配时，父区必须与源 ROI 存在实际体素重叠，不能仅凭 `LH`、`RH` 等名称后缀预设候选。TI 逆向优化通常具有明确的侧化目标；自动把左右半球合并会显著扩大目标、降低局灶性，并可能得到与用户意图相反的电极配置。

不带官方侧别映射的 ROI 不通过名称或坐标符号猜测侧别，只保留实际重叠结果。

### 3.3 只做成员 mask 的体素并集

合并结果为：

```text
merged_mask = mask_1 OR mask_2 OR ... OR mask_n
```

不做 dilation、closing、凸包填充或跨空隙插值。这样每个输出体素都能追溯到现有 atlas 的某个原始脑区，不会凭形态学操作创造没有 atlas 依据的新组织。

### 3.4 解剖语义优先，体积只作为护栏

组成员首先由官方区域树、官方显式编号映射或 mask 实际重叠确定；体积用于发现过小、无增长或过大的异常组，不反过来驱动任意邻区加入。

初版生成报告应给出：原 ROI 体积、合并后体积、放大倍数、连通分量数和成员列表。具体体积上下限在代表性失败数据上验证后再定，不把未经验证的固定 mm³ 阈值写成医学标准。

## 4.【离线/Atlas 阶段】各 atlas 的合并策略

### 4.1 Brainnetome：按官方 Gyrus、同侧合并

`BNA_subregions.xlsx` 已提供 `Lobe`、`Gyrus`、左右标签编号和细分区描述。当前数据包含 24 个 Gyrus 组，每个组含 2～8 对左右细分区，没有单成员 Gyrus 组。因此可直接使用官方 Gyrus 作为父区，不需要从缩写猜测。

合并键为：

```text
(Gyrus, hemisphere)
```

示例：

| 用户所选细分区 | 合并组 | 成员 | 合并理由 |
|---|---|---|---|
| `rHipp_L`（215）或 `cHipp_L`（217） | `Hipp_L` | 215 + 217 | 两者分别是左侧海马嘴侧部和尾侧部，合并后才表达完整的同侧海马分区 |
| `mAmyg_L`（211）或 `lAmyg_L`（213） | `Amyg_L` | 211 + 213 | 两者是左侧杏仁核内侧、外侧细分区，父级解剖结构一致 |
| `A8m_L` 等左侧 SFG 子区 | `SFG_L` | 1、3、5、7、9、11、13 | BNA 原始表将这些细分区共同归入左侧 Superior Frontal Gyrus |

选择 Gyrus 而不是整个 Lobe，是因为整个脑叶通常过大，会明显牺牲目标特异性；Gyrus 是 BNA 当前层级中兼顾体积与解剖可解释性的最小可靠父级。

### 4.2 Julich：按官方 v3.1 区域树自动合并

Julich-Brain 是基于细胞构筑概率图建立的皮层区和皮层下核团图谱，而不是 BNA 的宏观脑回分区。[Amunts 等人的图谱论文](https://pubmed.ncbi.nlm.nih.gov/32732281/)和 [Julich-Brain v3.1 官方发布说明](https://julich-brain-atlas.de/news/2024-06-new-release-julich-brain-atlas-adds-52-new-maps)均说明了其细胞构筑分区性质及皮层、皮层下结构范围。因此 Julich 不参考 BNA Gyrus，也不通过跨图谱重叠把细胞构筑区强行归入 BNA 脑回。

官方 siibra 接口把每个 parcellation 表示为一棵区域树，每个 `Region` 都可访问 `parent`；官方示例还能直接看到杏仁核组及其子区等层级。[区域树说明](https://siibra-python.readthedocs.io/en/latest/examples/01_atlases_and_parcellations/002_explore_region_hierarchy.html)、[Region 父级元数据说明](https://siibra-python.readthedocs.io/en/latest/examples/01_atlases_and_parcellations/004_brain_region_metadata.html)和 [Julich v3.1 区域查找示例](https://siibra-python.readthedocs.io/en/latest/examples/01_atlases_and_parcellations/003_find_regions.html)共同构成本方案的父子关系依据。

为避免 demo 运行时依赖网络或新增 siibra 包，实施时把官方 Julich-Brain v3.1 区域树固化为只读本地快照，并记录来源 URL、版本和内容哈希。自动流程为：

1. 将本地标签结尾的 `_lh`、`_rh` 分别固定归一化为官方命名中的 ` - left hemisphere`、` - right hemisphere`，同时只统一空白和大小写；除此之外不改写语义词。
2. 归一化后的完整名称必须与官方 v3.1 区域树的叶节点精确匹配；不使用模糊、子串、关键词或相似度匹配。
3. 从命中的叶节点沿官方 `parent` 链向上查找；跳过只汇总左右侧同名区的双侧节点，选择其同侧子树中至少覆盖 2 个当前本地 ROI 的最近祖先。
4. 将该祖先下能够精确对应到当前本地图谱、且与源 ROI 同侧的全部叶节点作为成员，合并键为官方祖先节点 ID 和侧别。
5. 名称无法精确命中官方叶节点，或沿父链找不到满足成员数要求的同侧祖先时，自动记为 `unmatched`，不进入人工审核，也不使用 BNA 或模糊名称兜底。

实际试验结果示例：

| 用户所选细分区 | 合并组 | 成员 | 合并理由 |
|---|---|---|---|
| `Area PirTB (PiriformCortexMesial, temporobasal)_lh`（17）或 `Area PirT (PiriformCortexMesial, temporal)_lh`（130） | Julich 官方 `piriform cortex` 左侧组 | 17 + 130 | 两个叶节点精确命中官方树后，最近的同侧有效祖先均为 `piriform cortex` |
| `Pv (Thalamus, paraventricular Nucleus)_lh`（61） | Julich 官方 `medial group` 左侧组 | 61（Pv）+ 74（MD）+ 138（MV） | 三个左侧丘脑核叶节点位于官方树同一个 `medial group` 祖先下 |
| `VPM (Thalamus, ventral posterior medial Nucleus)_rh`（385） | Julich 官方 `ventral group` 右侧组 | 238（VA）+ 252（VLP）+ 332（VM）+ 344（VAmc）+ 358（VIM）+ 375（VPL）+ 383（VLA）+ 385（VPM）+ 400（VPMpc）+ 401（VPi） | 这些右侧丘脑核均由固化的 v3.1 区域树遍历到同一个 `ventral group` 祖先，不使用名称模糊匹配 |

表中成员来自 `roi_merge_report.json` 的实际自动遍历结果。本地名称只负责精确定位官方叶节点，分组关系仍完全取自固化的 Julich-Brain v3.1 区域树。

这样合并的理由是：它保留 Julich 自身的细胞构筑语义，允许极小叶节点扩展到官方定义的最近可用父级，同时避免把跨 atlas 空间重叠误当成官方解剖父子关系。

### 4.3 DiFuMo：利用多分辨率空间重叠寻找较粗功能父区

DiFuMo64/128/256/512/1024 是不同分辨率的功能成分集合，并不是保证严格嵌套的解剖树。因此不能把名称前缀相同的 component 直接视为父子，也不能只靠 `anterior/posterior/LH/RH` 等词删除后缀来合并。

demo 的离线/Atlas 阶段采用以下确定性自动映射过程。

#### 4.3.1 固定相邻分辨率

每个源 atlas 只使用预先指定的一个较粗参考 atlas，不递归切换层级：

```text
DiFuMo128  → DiFuMo64
DiFuMo256  → DiFuMo128
DiFuMo512  → DiFuMo256
DiFuMo1024 → DiFuMo512
```

DiFuMo64 没有更粗的同系列 atlas，且当前最小 ROI 已有 8917 mm³，因此本次直接作为 `keep` 对照基线，不做父区匹配，也不生成合并成员。

#### 4.3.2 计算实际体素重叠候选

对源 atlas 的每一个细 component `S_i`，分别与参考父 atlas 的所有粗 component `P_j` 计算实际重叠。只有重叠体素数大于 0 的父 component 才进入候选列表，并记录：

```text
overlap_count  = voxel_count(S_i ∩ P_j)
coverage       = overlap_count / voxel_count(S_i)
Dice           = 2 × overlap_count / (voxel_count(S_i) + voxel_count(P_j))
overlap_volume = overlap_count × voxel_volume
```

标签名称、名称前缀和 `anterior/posterior/LH/RH` 等后缀均不参与候选生成或指标计算。一个名称看似不一致的父 component，只要实际空间重叠排名最高，仍可成为自动选择结果。

#### 4.3.3 为每个源 component 选择唯一父 component

候选列表按照以下固定优先级排序：

1. coverage 从高到低；
2. Dice 从高到低；
3. 重叠体积从高到低；
4. 父 component index 从小到大。

排序后的第一名作为该源 component 的唯一父 component，同时在报告中保存完整 `candidate_rankings` 和第二名 coverage。没有任何正体素重叠时自动记为 `unmatched`，不使用名称或人工选择兜底。

#### 4.3.4 反向分组并决定 merged 或 keep

程序先对源 atlas 的全部 component 完成父 component 选择，再按照所选父 component 反向分组；这一步发生在 demo 选择每个 atlas 的 5 个小 ROI 样本之前。例如：

```text
父 component 14
├─ 源 component 33
├─ 源 component 63
└─ 源 component 121
```

同一父 component 下至少有 2 个源 component 时，该组成员全部写入 `member_indices`，每个成员的决策均为 `merged`；只有 1 个源 component 时不能扩大 ROI，决策为 `keep`。

#### 4.3.5 生成同一源 atlas 的成员并集

父 component 只作为分组参照，不直接作为最终 ROI。实际输出始终是同一源 atlas 下所有成员二值 mask 的逻辑并集：

```text
merged_mask = source_mask_1 OR source_mask_2 OR ... OR source_mask_n
```

合并过程不做 dilation、closing、凸包填充或跨空隙插值，并在报告中记录源 ROI、参考父 component、实际成员、合并体积和体积放大倍数。

实际试验结果示例：

| 用户所选细分区 | 合并组 | 成员 | 合并理由 |
|---|---|---|---|
| DiFuMo128 `Precuneus posterior`（121） | DiFuMo64 `Cingulate gyrus mid-posterior`（14）参照组 | 33（Posterior cingulate cortex）+ 63（Posterior cingulate cortex superior）+ 121（Precuneus posterior） | 源 ROI 与父 component 的实测 coverage 为 0.425、Dice 为 0.287；同一父 component 接纳这 3 个 DiFuMo128 成员 |
| DiFuMo512 `Precuneus RH`（413） | DiFuMo256 `Precuneus superior`（256）参照组 | 131（Precuneus mid-superior LH）+ 253（Precuneus middle）+ 413（Precuneus RH） | 源 ROI 与父 component 的实测 coverage 为 0.719、Dice 为 0.431，并按固定重叠排序选为第一父候选 |
| DiFuMo1024 `Superior parietal lobule posterior LH`（998） | DiFuMo512 `Superior occipital sulcus superior LH`（75）参照组 | 686（Intraparietal sulcus posterior LH）+ 998（Superior parietal lobule posterior LH） | 源 ROI 与父 component 的实测 coverage 为 0.384、Dice 为 0.231；成员来自相同父 component 的实际空间接纳结果，而不是标签相似性 |

采用多分辨率重叠的理由是：高分辨率 DiFuMo 的小 component 可以在同一 atlas 家族的更粗功能表示中找到更稳健的目标，而实际体素重叠直接反映 component 的空间关系。各分辨率由独立分解得到，因此 demo 必须完整输出匹配指标，不能把自动结果当成已经确认的严格父子关系。

## 5.【离线/Atlas 阶段】试验模块与自动合并算法

### 5.1【结构说明】代码和产物范围

本次只计划增加内部试验 helper 和 demo：

```text
neuracle/
├─ atlas/
│  ├─ merge_metadata/
│  │  └─ julich_brain_v3_1_hierarchy.json  # 官方区域树离线快照，含来源、版本和哈希
│  └─ roi_merge_experiment.py
└─ ti_optimization/
   └─ demo/
      ├─ merged_atlas_roi_demo.py
      └─ merged_atlas_roi_focality_compare_demo.py

data/
├─ roi_merge_demo_outputs/<run_id>/
│  ├─ merged_masks/                    # 离线/Atlas 阶段生成
│  ├─ overlays/                        # 离线/Atlas 阶段的 MNI overlay
│  ├─ roi_merge_report.json            # 两个阶段完成后汇总
│  └─ roi_merge_report.csv             # 两个阶段完成后汇总
└─ roi_merge_focality_comparisons/
   └─ small_roi_25_focality_3ma/
      ├─ queue_state.json              # 可恢复队列状态
      ├─ focality_jobs.csv             # 逐任务状态
      ├─ focality_batch_report.json    # 逐 ROI 比较结果
      └─ jobs/                         # 每次优化的独立输出目录
```

约束如下：

- `roi_merge_experiment.py` 只提供给 demo 导入的内部函数，不从 `neuracle.atlas.__init__` 导出；
- 不修改 `neuracle/ti_inverse.py`、`neuracle/ti_forward.py`、参数 schema、validator 或现有 atlas registry；
- 不新增 `python -m neuracle.atlas...` 形式的生成或验证入口；
- 合并 mask 和报告只写入 demo 输出目录，不写入原始 atlas 或 `standardized/*/rois/`。

### 5.2【离线/Atlas 阶段】各 atlas 的自动父区来源

当前 registry 中的七个 atlas 都进入试验：

| 源 atlas | 自动父区来源 | 匹配方式 |
|---|---|---|
| `BN_Atlas_246_1mm` | BNA 官方 Gyrus | 使用官方左右脑区编号到 Gyrus 的显式映射 |
| `JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152` | Julich-Brain v3.1 官方区域树离线快照 | 归一化名称精确匹配官方叶节点，再遍历官方父链 |
| `DiFuMo64` | 无 | `keep` 对照基线，不做合并 |
| `DiFuMo128` | DiFuMo64 | 实际体素重叠 |
| `DiFuMo256` | DiFuMo128 | 实际体素重叠 |
| `DiFuMo512` | DiFuMo256 | 实际体素重叠 |
| `DiFuMo1024` | DiFuMo512 | 实际体素重叠 |

Julich 的快照只增加外部合并所需元数据，不修改现有 atlas、标准化 ROI 或 registry。demo 只读取本地快照，不能在运行时联网查询官方服务；如果快照版本或哈希不符则明确失败。DiFuMo64 的 `keep` 样本用于对照“小 ROI 合并”与“原 ROI 直接进入个体匹配”的差异。

### 5.3【离线/Atlas 阶段】确定性自动决策

自动决策按 atlas 分成四条确定性分支：

1. BNA 使用官方编号到同侧 Gyrus 的显式映射。
2. Julich 使用归一化完整名称精确匹配官方 v3.1 叶节点，再按第 4.2 节遍历官方父链；无法精确匹配时记为 `unmatched`。
3. DiFuMo128～1024 只保留与指定粗一级 atlas 实际重叠体素数大于 0 的父 component，再按 coverage 降序、Dice 降序、重叠体积降序、父区 index 升序排序；没有实际重叠时记为 `unmatched`。
4. DiFuMo64 不执行父区匹配，所有 ROI 固定记为 `keep` 基线。
5. BNA、Julich 和 DiFuMo128～1024 的有效组内成员数至少为 2，且并集体积大于当前源 ROI 时记为 `merged`；只有一个成员或并集没有增长时记为 `keep`。

算法不存在人工审核状态。coverage 很低或第一、第二名非常接近时仍按固定排序规则选择，但必须在报告中输出完整排名和差距，便于判断这种全自动策略是否值得继续使用。

每个结果记录至少包含：

```json
{
  "atlas_name": "DiFuMo1024",
  "source_index": 1,
  "source_volume_mm3": 0.0,
  "parent_source": "spatial_overlap",
  "parent_atlas_name": "DiFuMo512",
  "parent_index": 1,
  "hierarchy_parent_id": null,
  "coverage": 0.0,
  "dice": 0.0,
  "overlap_volume_mm3": 0.0,
  "second_best_coverage": 0.0,
  "member_indices": [],
  "merged_volume_mm3": 0.0,
  "volume_ratio": 0.0,
  "decision": "merged"
}
```

Julich 的名称字段用于精确定位官方树节点，但不参与父级相似度计算或排序；其他 atlas 的名称只附带输出，不进入计算和排序。DiFuMo64 的父级和重叠指标字段为 `null`，`decision` 固定为 `keep`。

## 6.【跨阶段】合并 ROI Demo 设计

### 6.1【结构说明】Demo 文件与执行边界

新增：

```text
neuracle/ti_optimization/demo/merged_atlas_roi_demo.py
```

demo 直接调用内部自动合并 helper，不经过 `ti_inverse.py`，也不形成新的业务调用入口。默认使用现有 demo 头模 `data/m2m_ernie`，输出到独立、带 `run_id` 的目录，避免覆盖已有结果。

一次 demo 按固定顺序经过两个阶段：

```text
离线/Atlas 阶段
    -> 自动父区匹配
    -> 自动分组和小 ROI 选样
    -> 合并 mask 生成及 MNI 空间校验

Demo 运行时/个体阶段
    -> 读取 subject_dir
    -> 原 ROI 与合并 ROI 映射到个体头模
    -> 统计 WM/GM mesh 元素并生成 match_status
    -> 可选：执行完整 TI 优化
```

建议参数：

```text
--subject-dir
--output-dir
--samples-per-atlas       默认 5
--offline-only            默认关闭；开启后只执行阶段四的 MNI 离线验证
```

默认模式先完成离线/Atlas 阶段，再完成个体 WM/GM mesh 匹配，但不执行完整 TI 优化。该合并 demo 不提供 `--run-optimization`；完整优化只存在于第 6.5 节的独立成对比较 Demo 中，避免离线验证时误启动优化。

### 6.2【离线/Atlas 阶段】小 ROI 自动选样

每个 atlas 都直接按当前标准化单区 mask 的实际体积升序、source index 升序排序，固定选择前 5 个。选样先于父区决策，不因名称、是否容易合并或最终 `decision` 替换样本；这样能如实暴露算法对最小 ROI 的覆盖能力，而不是只挑合并成功的区域。DiFuMo64 的 5 个样本预期全部为 `keep` 对照，其他 atlas 若出现 `unmatched` 或意外 `keep`，报告必须保留该结果并继续处理剩余样本，不能换用更大的 ROI 补足。

以下数值于 2026-08-17 直接从当前 `neuracle/atlas/standardized/` 文件计算。各文件体素均为 1 mm³，因此非零体素数与体积数值相同；demo 必须重新计算 `source_volume_mm3`，若输入文件变化导致数值与本表不同，应记录基线漂移但继续使用本次实测值，不能静默沿用文档值。

| Atlas | Index | ROI | 当前实测体积 |
|---|---:|---|---:|
| Brainnetome 246 | 213 | `lAmyg_L` | 661 mm³ |
| Brainnetome 246 | 117 | `TI_L` | 785 mm³ |
| Brainnetome 246 | 214 | `lAmyg_R` | 1018 mm³ |
| Brainnetome 246 | 235 | `Stha_L` | 1054 mm³ |
| Brainnetome 246 | 116 | `A28/34_R` | 1077 mm³ |
| Julich bilateral | 17 | `Area PirTB (PiriformCortexMesial, temporobasal)_lh` | 5 mm³ |
| Julich bilateral | 385 | `VPM (Thalamus, ventral posterior medial Nucleus)_rh` | 7 mm³ |
| Julich bilateral | 178 | `VPM (Thalamus, ventral posterior medial Nucleus)_lh` | 9 mm³ |
| Julich bilateral | 130 | `Area PirT (PiriformCortexMesial, temporal)_lh` | 18 mm³ |
| Julich bilateral | 61 | `Pv (Thalamus, paraventricular Nucleus)_lh` | 20 mm³ |
| DiFuMo64 | 4 | `Cingulate cortex posterior` | 8917 mm³ |
| DiFuMo64 | 60 | `Cuneus` | 9661 mm³ |
| DiFuMo64 | 18 | `Precuneus superior` | 10320 mm³ |
| DiFuMo64 | 3 | `Calcarine cortex posterior` | 12035 mm³ |
| DiFuMo64 | 54 | `Precuneus anterior` | 12111 mm³ |
| DiFuMo128 | 52 | `Posterior cingulate cortex inferior` | 5874 mm³ |
| DiFuMo128 | 72 | `Calcarine cortex posterior` | 6278 mm³ |
| DiFuMo128 | 121 | `Precuneus posterior` | 6691 mm³ |
| DiFuMo128 | 16 | `Precuneus superior` | 7047 mm³ |
| DiFuMo128 | 33 | `Posterior cingulate cortex` | 7097 mm³ |
| DiFuMo256 | 78 | `Lingual gyrus mid-posterior` | 3631 mm³ |
| DiFuMo256 | 195 | `Lingual gyrus medial` | 3700 mm³ |
| DiFuMo256 | 218 | `Precuneus inferior` | 3730 mm³ |
| DiFuMo256 | 15 | `Posterior cingulate cortex posterior` | 3765 mm³ |
| DiFuMo256 | 151 | `Parieto-occipital sulcus anterior` | 4059 mm³ |
| DiFuMo512 | 216 | `Precuneus postero-inferior RH` | 1653 mm³ |
| DiFuMo512 | 331 | `Parieto-occipital sulcus middle` | 1814 mm³ |
| DiFuMo512 | 47 | `Paracingulate sulcus posterior RH` | 1846 mm³ |
| DiFuMo512 | 432 | `Subparietal sulcus inferior LH` | 1854 mm³ |
| DiFuMo512 | 413 | `Precuneus RH` | 1900 mm³ |
| DiFuMo1024 | 288 | `Descending occipital gyrus superior` | 863 mm³ |
| DiFuMo1024 | 621 | `Lingual gyrus posterior` | 913 mm³ |
| DiFuMo1024 | 885 | `Middle temporal gyrus posterior inferior LH` | 1011 mm³ |
| DiFuMo1024 | 998 | `Superior parietal lobule posterior LH` | 1013 mm³ |
| DiFuMo1024 | 20 | `Parieto-occipital sulcus middle` | 1014 mm³ |

覆盖范围固定为：

```text
BN_Atlas_246_1mm
JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152
DiFuMo64
DiFuMo128
DiFuMo256
DiFuMo512
DiFuMo1024
```

默认每个 atlas 5 个样本，共验证 35 个 ROI，其中 30 个用于自动合并验证，DiFuMo64 的 5 个用于 `keep` 对照验证。

### 6.3【离线/Atlas 阶段】合并 mask 验证

对每个样本自动完成：

1. 保存原 ROI 的 index、名称和实测体积；保存自动决策、父区依据、成员列表及适用的匹配指标。
2. 对 `merged` 样本验证合并 mask 严格等于所有成员 mask 的逻辑并集，并记录合并体积。
3. 对 DiFuMo64 的 `keep` 样本验证后续直接使用的 mask 与源 ROI 完全一致，不要求生成伪合并 NIfTI，合并体积字段记为源体积。
4. 验证实际输出为 `uint8` 二值 NIfTI，shape、affine 与源 ROI 一致，并生成对应的 MNI 空间多切面 overlay。
5. 样本数不足、体积基线漂移、非 DiFuMo64 样本未得到 `merged`、并集不一致、空间信息不一致或结果 mask 为空时，为样本追加对应的 `acceptance_issues`；能够得到有效结果 mask 的样本仍进入个体阶段，其余样本记为跳过，demo 继续处理后续样本。

本阶段不读取 `subject_dir`，也不产生 `empty`、`no_gain` 或 `expanded`。

### 6.4【Demo 运行时/个体阶段】WM/GM 实际匹配

仅在 6.3 节全部通过后，对每个样本执行：

1. 读取 `subject_dir`；默认值为 `data/m2m_ernie`。
2. 将原 ROI 和合并 ROI 分别从 MNI 空间映射到该个体头模。
3. 分别统计两个 ROI 与当前允许的 WM/GM mesh 元素的实际交集数量。
4. 按实际结果自动给出：
   - `empty`：合并 ROI 的有效元素数为 0；
   - `no_gain`：合并 ROI 的有效元素数小于或等于原 ROI；
   - `expanded`：合并 ROI 的有效元素数大于原 ROI。
5. 把个体头模路径、原 ROI 元素数、合并 ROI 元素数、增长比例和 `match_status` 追加到 JSON、CSV。

同一 atlas ROI 在不同个体头模上的 `match_status` 可以不同；这些字段只属于本次 demo 运行结果，不写入 atlas registry 或离线合并关系。

### 6.5【Demo 运行时/个体阶段】合并前后 focality 成对优化

新增独立的 `merged_atlas_roi_focality_compare_demo.py`。本试验 Demo 不接收命令行参数，直接固定读取 `data/roi_merge_demo_outputs/20260817_110807_8c9b0ef7/roi_merge_report.json`，使用 `data/m2m_ernie`，并从第 6.2 节的 35 个离线样本中只保留当前实际判定为 `merged` 的 25 个 ROI。10 个 `keep` 样本不写入比较 Demo 的固定清单，也不进入优化队列。报告内容与硬编码参数共同计算 SHA-256 指纹；已有队列与当前输入不一致时直接停止，避免把不同实验条件续写进同一份结果。

这 10 个 `keep` 包含两类：

1. 5 个 DiFuMo64 固定 `keep` 对照：DiFuMo64 没有更粗一级的同系列 atlas，因此不执行父区匹配。
2. 5 个自动匹配后形成的单成员 `keep`：DiFuMo128 index 52、72、16，DiFuMo512 index 47，以及 DiFuMo1024 index 621。它们虽然找到了实际重叠的父 component，但该父 component 最终只接纳当前 ROI，`member_indices` 只有自身，输出体积与源 ROI 相同，无法形成至少两个成员的有效并集，因此按确定性规则保留原 ROI，并在离线报告中记录 `expected_merged_got_keep`。

每个 `decision=merged` 样本排入两个任务：

```text
original 条件：roi_mask_path = source_roi_path
merged 条件：roi_mask_path = result_mask_path
```

每个固定 ROI 都必须在当前报告中仍为 `decision=merged` 且包含有效的合并 mask，否则 Demo 在构造队列前直接报错。25 个 ROI 各排入一个 original 和一个 merged，因此队列固定包含 50 个优化任务；`keep` 和 `unmatched` 均不进入该比较实验。

队列最多同时提交 4 个独立逆向优化进程，并始终使用固定的 `goal="focality"`。每个优化内部的 `n_workers` 固定为 4；同一时刻最多存在 4 个优化任务，因此最多使用约 16 个内部 worker。任务启动和结束后都原子更新 `queue_state.json`、`focality_jobs.csv` 和 `focality_batch_report.json`。再次运行同一 Demo 时，已完成或已失败的任务不会重复提交，上次中断时仍为 `running` 的任务恢复成 `pending`，并写入新的 `attempt_<n>` 目录，保留旧 attempt 产物。

original 和 merged 都固定使用 Non-ROI 阈值 0.1 V/m、ROI 阈值 0.2 V/m、两对电极、每对 `+0.003 A/-0.003 A`（3 mA）、WM 0.14 S/m、GM 0.30 S/m、`vn` 各向异性、优化器 seed 42 和相同电极半径。对同一个 ROI，唯一实验变量是 ROI mask。

报告至少包含 `optimization_success`、`optimization_goal_value`、FEM/目标函数评估次数、优化电极位置、运行时间、结果 mesh、导出 NIfTI 和异常信息。目标函数值越小表示当前 focality 目标越优；不得用两个不同目标函数的 demo 代替这组配对实验。

`expanded` 只说明合并后获得了更多有效 mesh 元素，不等于逆向优化一定成功。本批次已由用户执行完整队列，实际得到 49 个完成任务和 1 个失败任务，因此 25 个 ROI 中有 24 个具备完整 original/merged 配对结果。

### 6.6【结果分析阶段】使用合并前 ROI 定义重算 merged focality

新增只读结果分析脚本 `merged_atlas_roi_focality_recalculate_demo.py`，直接读取第 6.5 节已经生成的 `queue_state.json` 和每个成功任务根目录下的 `*_tes_flex_opt_head_mesh.msh`，不重新执行优化、FEM 或电极映射。脚本固定输出 `focality_recalculated_comparison.csv` 和 `focality_recalculated_comparison.json`，并覆盖队列中的全部 25 个 ROI；缺少 original 或 merged 成功结果的样本保留在结果中并记录错误，不用其他样本替代。

每个成功配对保留以下四个 focality 值：

1. `original_focality`：original 优化任务记录的最终 `optimization_goal_value`；
2. `merged_focality`：merged 优化任务记录的最终 `optimization_goal_value`；
3. `merged_on_original_roi_focality`：读取 merged 根目录 mesh 中唯一的 `max_TI` 元素场，但使用 original 根目录 mesh 中的 `ROI` 和 `non-ROI` 元素掩膜，按照与优化器相同的阈值 `[0.1, 0.2] V/m` 和 focality ROC 公式重新计算目标值。
4. `merged_on_original_roi_merged_non_roi_focality`：同样读取 merged 根目录 mesh 中的 `max_TI` 元素场，ROI 使用 original 根目录 mesh 的 `ROI` 元素掩膜，non-ROI 改用 merged 根目录 mesh 的 `non-ROI` 元素掩膜，再按相同阈值和公式重算目标值。

脚本在 JSON 报告中分别把后三种 focality 与 `original_focality` 比较，统一统计可比较数量、当前方法更优、合并前更优、相同和无法比较数量；逐 ROI 表只保留四个数值，不重复放置单行比较结论。

重算只允许使用未映射电极的根目录优化 mesh，因为前两个 `optimization_goal_value` 对应连续电极位置优化结果；不得混入 `mapped_electrodes_simulation` 下的映射后 mesh。脚本同时校验 original/merged mesh 的元素数量、元素编号、元素类型和组织标签顺序一致，并要求 `ROI`、`non-ROI`、`max_TI` 字段各只有一个；任一条件不满足时，该样本记录为失败而不是猜测字段。

在 `simnibs_env` 中执行：

```powershell
python -m neuracle.ti_optimization.demo.merged_atlas_roi_focality_recalculate_demo
```

本次运行共处理 25 个 ROI，成功重算 24 个，失败 1 个。focality 目标值越小越好，三种计算方式相对合并前 focality 的比较统计见逐 ROI 表下方。Julich index 385 的 original 优化任务已失败，错误为 `ValueError: zero-size array to reduction operation maximum which has no identity`，因此该行只能保留 merged focality，不能重算或比较。完整机器可读结果保存在：

```text
data/roi_merge_focality_comparisons/small_roi_25_focality_3ma/focality_recalculated_comparison.csv
data/roi_merge_focality_comparisons/small_roi_25_focality_3ma/focality_recalculated_comparison.json
```

| Atlas | Index | ROI | 合并前 focality | 合并后 focality | merged 场按合并前 ROI/non-ROI 重算 | merged 场按合并前 ROI + 合并后 non-ROI 重算 |
|---|---:|---|---:|---:|---:|---:|
| BN_Atlas_246_1mm | 213 | `lAmyg_L` | -86.6661 | -100.9182 | -102.6197 | -102.6353 |
| BN_Atlas_246_1mm | 117 | `TI_L` | -110.6547 | -78.4863 | -83.3521 | -83.5172 |
| BN_Atlas_246_1mm | 214 | `lAmyg_R` | -84.7640 | -102.7599 | -102.3627 | -102.3767 |
| BN_Atlas_246_1mm | 235 | `Stha_L` | -82.6506 | -64.2788 | -79.5874 | -79.6915 |
| BN_Atlas_246_1mm | 116 | `A28/34_R` | -85.0838 | -86.5412 | -61.8305 | -61.9551 |
| JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152 | 17 | `Area PirTB (PiriformCortexMesial, temporobasal)_lh` | -119.9334 | -110.9605 | -106.2412 | -106.2413 |
| JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152 | 385 | `VPM (Thalamus, ventral posterior medial Nucleus)_rh` | — | -65.9534 | — | — |
| JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152 | 178 | `VPM (Thalamus, ventral posterior medial Nucleus)_lh` | -107.2179 | -78.2443 | -74.2230 | -74.2389 |
| JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152 | 130 | `Area PirT (PiriformCortexMesial, temporal)_lh` | -117.4264 | -110.9605 | -110.4396 | -110.4398 |
| JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152 | 61 | `Pv (Thalamus, paraventricular Nucleus)_lh` | -78.0626 | -60.0075 | -61.0473 | -61.0486 |
| DiFuMo128 | 121 | `Precuneus posterior` | -108.9840 | -86.0161 | -93.7483 | -94.4545 |
| DiFuMo128 | 33 | `Posterior cingulate cortex` | -89.9054 | -86.0161 | -87.9151 | -88.4376 |
| DiFuMo256 | 78 | `Lingual gyrus mid-posterior` | -115.2172 | -106.1220 | -107.9825 | -108.1723 |
| DiFuMo256 | 195 | `Lingual gyrus medial` | -99.4358 | -96.7483 | -97.0320 | -97.4371 |
| DiFuMo256 | 218 | `Precuneus inferior` | -101.4920 | -90.5508 | -94.6860 | -94.7956 |
| DiFuMo256 | 15 | `Posterior cingulate cortex posterior` | -95.5895 | -90.5508 | -83.1057 | -83.2055 |
| DiFuMo256 | 151 | `Parieto-occipital sulcus anterior` | -90.6158 | -95.3202 | -91.9154 | -92.1383 |
| DiFuMo512 | 216 | `Precuneus postero-inferior RH` | -102.4626 | -90.4394 | -94.2166 | -94.3628 |
| DiFuMo512 | 331 | `Parieto-occipital sulcus middle` | -95.3969 | -87.3391 | -88.1823 | -88.2316 |
| DiFuMo512 | 432 | `Subparietal sulcus inferior LH` | -91.3057 | -96.6642 | -86.0982 | -86.1623 |
| DiFuMo512 | 413 | `Precuneus RH` | -104.8939 | -97.9845 | -96.9433 | -97.1673 |
| DiFuMo1024 | 288 | `Descending occipital gyrus superior` | -122.0620 | -119.7849 | -104.0444 | -104.0629 |
| DiFuMo1024 | 885 | `Middle temporal gyrus posterior inferior LH` | -99.9790 | -90.2087 | -90.9370 | -91.0126 |
| DiFuMo1024 | 998 | `Superior parietal lobule posterior LH` | -125.1980 | -125.9712 | -112.5430 | -112.5671 |
| DiFuMo1024 | 20 | `Parieto-occipital sulcus middle` | -102.6427 | -98.0317 | -95.2290 | -95.2691 |

三种 focality 计算方式相对合并前结果的统计如下。百分比以 24 个可比较 ROI 为分母：

| 比较方式 | 可比较 | 当前方式更优 | 合并前更优 | 相同 | 无法比较 |
|---|---:|---:|---:|---:|---:|
| 合并后 focality vs. 合并前 focality | 24 | 6（25.0%） | 18（75.0%） | 0 | 1 |
| merged 场按合并前 ROI/non-ROI 重算 vs. 合并前 focality | 24 | 3（12.5%） | 21（87.5%） | 0 | 1 |
| merged 场按合并前 ROI + 合并后 non-ROI 重算 vs. 合并前 focality | 24 | 3（12.5%） | 21（87.5%） | 0 | 1 |

## 7.【分阶段】试验验收标准

### 7.1【离线/Atlas 阶段】验收标准

本节标准用于判定和记录试验结果，不作为自动修正规则。任一标准未通过时，demo 必须在 JSON、CSV 汇总中记录具体原因并继续处理其他样本；不得为了让验收通过而修改分组算法、替换样本或扩大成员范围。只要报告成功写出，验收项未通过本身不导致非零退出；只有输入缺失、报告无法写出等使试验无法执行的运行错误才使用非零退出码。

1. 不设计任何需要人工选择的状态；所有结果只能由固定自动规则生成。
2. 七个 atlas 均按实测体积固定选出最小的 5 个 ROI，共 35 个；不足时明确记录，不人工补样。
3. 相同 atlas 输入重复执行离线/Atlas 阶段时，source、parent、成员列表和输出 mask 哈希一致。
4. BNA 候选来自官方编号映射，Julich 候选来自名称精确匹配后的官方 v3.1 区域树，DiFuMo128～1024 候选来自实际体素重叠；均不使用名称相似度或人工兜底。
5. 30 个待合并样本的 mask 严格等于成员并集且体积大于原 ROI；DiFuMo64 的 5 个样本固定为 `keep` 且结果 mask 等于源 ROI；所有 NIfTI 空间信息与源 ROI 一致。
6. 本阶段生成逐样本合并 NIfTI、MNI overlay 和父区匹配指标，不包含个体 WM/GM 元素数。

### 7.2【Demo 运行时/个体阶段】验收标准

1. 原 ROI 与合并 ROI 都输出当前 `subject_dir` 下的 WM/GM mesh 元素数。
2. `empty`、`no_gain` 或 `expanded` 只由两个实际元素数自动产生。
3. JSON、CSV 必须记录头模路径和个体匹配结果，且不得把这些结果写回 atlas registry。
4. `merged_atlas_roi_demo.py` 本身不提供优化参数；只有显式运行零参数的独立 focality 比较 Demo 才能启动批量优化。
5. 当前实际判定为 `merged` 的 25 个 ROI 进入队列，固定生成 50 个任务；`keep` 和 `unmatched` 不进入队列，且同一时刻最多运行 4 个逆向优化。
6. original 和 merged 的优化成功与否、目标值和异常必须分别记录，不能改写离线合并决策；验收不通过只记录，不触发自动改码或替换 ROI。

### 7.3【范围边界】验收标准

1. `ti_inverse.py`、`ti_forward.py`、参数 schema、validator、公共 atlas API 和现有 atlas 文件均无改动。
2. 不新增测试文件；该 demo 和其报告就是本次试验的验证载体。

## 8.【分阶段】实施顺序

### 阶段一【离线/Atlas 阶段】：实现内部自动合并 helper

固化带来源、版本和哈希的 Julich-Brain v3.1 官方区域树快照；实现体积统计、官方层级遍历、实际重叠指标、固定排序、自动分组和二值并集，只提供内部函数，不增加公共 API 或命令入口。

### 阶段二【离线/Atlas 阶段】：实现 demo 的自动选样与 mask 验证

实现七个 atlas 的自动小 ROI 选样、合并 mask、MNI overlay 和离线报告字段。所有函数遵循 `neuracle/CLAUDE.md` 的类型标注、NumPy docstring、中文注释和模块级 logger 约定。

### 阶段三【Demo 运行时/个体阶段】：实现 WM/GM 实际匹配

读取 `subject_dir`，映射原 ROI 与合并 ROI，统计个体 WM/GM mesh 元素数并生成 `empty`、`no_gain` 或 `expanded`。该结果只追加到本次 demo 报告。

### 阶段四【离线验证】：执行静态与 MNI mask 验证

在 `simnibs_env` 中执行语法检查、demo `--help`、自动选样和 mask 并集验证，并运行 `git diff --check`。不创建测试代码。

### 阶段五【Demo 运行时/个体阶段】：实现成对 focality 优化试验

实现零参数的独立合并前后批量比较 Demo，固定 3 mA 电流、25 个实际 `merged` 小 ROI 和最多 4 个并发逆向优化，并提供可恢复队列和增量报告。初次实现时只完成静态检查，随后由用户运行 Demo 消费完整队列；最终 49 个任务完成、1 个任务失败，未通过验收的结果只记录，不据此自动修改代码。
