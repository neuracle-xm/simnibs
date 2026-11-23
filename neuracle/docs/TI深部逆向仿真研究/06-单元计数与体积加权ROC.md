# 06　相同的场，单元大小会改变 ROC 排名吗？

现行 ROC 给每个采样点一票。若某一小片组织被划成许多小四面体，它对评分的贡献会超过同体积的粗网格区域。这个对照只替换权重，保留两条路线原有的目标与靶外区域。

## 实验对照

对区域 X，分别计算单元超阈比例和体积比例 pᵥ(X,t)=Σvᵢ·1[Fᵢ≥t]/Σvᵢ。将原 ROC 距离中的 TPR、FPR 同时换成相应区域的 pᵥ；两臂的阈值、区域边界和电流完全相同。

## 两条路线的操作

在 sub1/sub2 的 STN、rHipp、A4ul 上，从共同候选库逐布局算两种分数，列首选是否翻转，并按单元体积四分位拆出对 TPR、FPR 的贡献。based 从裁剪网格取单元体积，free 绑定实际 ROI 采样点对应的四面体；一旦采样点到元素并非一一对应，先在共同正向网格完成重排。

## 结果与判据

保留逐布局两种 ROC、靶体积、有效单元数和首选的 D_hot。若大幅排名变化集中在细网格区，且新首选经复算更接近靶区、三个皮层对照均通过，才进入同预算闭环。分数变化而实际场无改善时，只报告离散化影响。

本页比较规则和参数为拟议实验设置。候选库、独立正向复算、三例分工和完整皮层验收均沿用[共同实验协议](02-共同实验协议.md)。实现和数据依据见[ROC 单元计数定义](../../../simnibs/optimization/tes_flex_optimization/measures.py) · [本地单元体积取法](../../ti_leadfield_optimization/regions.py)。

## 交给 Codex 的执行单

**输入与启动条件。** 读取 03 的 `shared_inputs.csv`，锁同一 `data_id`、脑域、T、Rest 和各候选的 F/v。先导 sub1/stn295；开发加入 sub2、rhipp216、a4ul58。 本篇需要专用研究分析。按 02 的入口约定实现或复用后，先调 `python -m neuracle.demo.ti_depth.exp06 --study-root "$studyRoot" --run-id run-001 --stage pilot`，再以同一运行号和 `--stage development` 完成下列开发步骤。两条命令是下一轮实现后的调用方式。

1. 为每个候选按原阈值生成 T 和 Rest 各自的布尔超阈数组。一次用元素数作分母，一次用体积和作分母，保存四个分子、分母及两种 ROC 距离。
2. 选一个场，按单元体积四分位分组校验所有组的超阈体积和等于全域超阈体积。对两列分数独立排序，照同一平手规则取首选。
3. 若翻转的布局经独立指标判断有开发收益，再在 based 候选和 free 回调中仅替换权重，用相同预算重搜并复算。

### 结果存放与判读

研究根 `$studyRoot` 默认为 `C:\Users\50609\Documents\simnibs\neuracle\TI-study-v1`。本篇首次运行写入 `results/06/run-001/`，先打开 `results/06/run-001/report.md`。同次执行的清单、命令、案例状态和文件索引按[02 的磁盘交付约定](02-共同实验协议.md#磁盘交付约定)保存。

- **专属表**　`results/06/run-001/roc_weights.csv`。每候选一行，含 `candidate_id,count_tpr,vol_tpr,count_fpr,vol_fpr,rank_count,rank_volume,roi_mm3,rest_mm3`，另存分子分母。
- **图或中间证据**　`figures/roc_rank.png` 横轴计数排名、纵轴体积排名，标记两种首选。
- **阅读顺序**　在专属表找 `rank_count=1` 与 `rank_volume=1` 的行，再看 `paired.csv` 的位置、覆盖和皮层读数；同一首选意味着此次仅分数改变。
- **本篇交付条件**　给出每个靶的翻转数、原解和新解、独立终点；说明网格权重是否解释收益。
