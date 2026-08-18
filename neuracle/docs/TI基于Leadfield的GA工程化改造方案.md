# TI 基于 Leadfield 的 GA 工程化改造方案

## 1. 改造目标

本次改造不再复现论文参数，仅保留固定 montage leadfield 上搜索两路 TI 电极和电流
的工程路径。现有 `neuracle.ti_inverse` 和 `neuracle.ti_optimization` 不做改动。

具体目标如下：

1. 缩短 GA 适应度计算时间。
2. 目标函数改为 SimNIBS TES 逆向优化的 `focality`。
3. 两路电流分别独立优化，范围均为 `0.05–4.00 mA`，步长为 `0.05 mA`，不设置
   两路总和约束。
4. 固定 montage 改为 `EEG10-10_NSN.csv`。

## 2. 目标函数

直接调用 SimNIBS `tes_flex_optimization.measures.ROC`，保持逆向优化中
`goal="focality"` 的定义：

- 非 ROI 阈值：`0.1 V/m`；
- ROI 阈值：`0.2 V/m`；
- sensitivity：ROI 中 `max_TI >= 0.2 V/m` 的 element 比例；
- false-positive rate：非 ROI 中 `max_TI >= 0.1 V/m` 的 element 比例；
- ROC distance：`sqrt(false_positive_rate^2 + (sensitivity - 1)^2)`；
- GA 最小化目标：`-100 * (sqrt(2) - ROC distance)`。

ROI/Rest mean、峰值和 ratio 继续作为结果诊断指标，但不再参与 GA 目标函数。

## 3. GA 加速方式

旧实现每个四电极染色体内部枚举电流组合。独立 `0.05–4.00 mA` 会产生
`80 * 80 = 6400` 组电流，不能继续使用内部穷举。

新染色体使用六个整数基因：

```text
(A+, A-, B+, B-, current_A_tick, current_B_tick)
```

其中电流值等于 `tick * 0.05 mA`。这样每次 GA 适应度调用只重建一次 `max_TI`，
由 GA 同时搜索电极和两路电流。

进一步优化：

- 适应度循环只计算 SimNIBS focality，不重复计算体积加权诊断统计；
- 对电极对正负方向交换、A/B 两路整体交换形成的等价解使用规范化缓存键；
- 默认种群由 100 降为 64、最大代数由 25 降为 20，并在连续 6 代无改进时提前停止；
- 最优解确定后再计算一次完整 ROI/Rest 诊断指标并导出场数据。

## 4. Montage 与缓存隔离

Demo 使用：

```text
data/m2m_ernie/eeg_positions/EEG10-10_NSN.csv
```

NSN montage 会改变 leadfield 的电极集合，因此使用新的
`data/ti_leadfield_ga_ernie_nsn_10_10` 输出根目录，避免误用旧 montage 的 HDF5、
直接 FEM 和 JSON 缓存。

## 5. 验证边界

本次修改后执行以下低成本验证，不运行完整 leadfield/FEM demo：

1. Python 语法编译与 import 检查；
2. NSN montage 路径和电极数量检查；
3. 电流 tick 边界、步长和无总和约束检查；
4. 使用小型合成 leadfield 对比模块 objective 与 SimNIBS `ROC` 的数值一致性；
5. 验证等价染色体缓存键和六基因 GA 解码。

## 6. 实际运行结果与验证

2026 年 8 月 18 日已在 `simnibs_env` 中完整运行修改后的 demo。ROI 为
Brainnetome atlas 中 `rHipp_R` 与 `cHipp_R` 的并集，即完整右侧海马；NSN montage
包含 61 个电极，生成的 leadfield shape 为 `(60, 1357778, 3)`，数据类型为
`float64`。

### 6.1 GA 最优解

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | F8 | C6 | 1.80 mA |
| B | F7 | PO4 | 3.40 mA |

两路电流分别满足 `0.05–4.00 mA`、步长 `0.05 mA` 的独立约束，没有两路总和
约束。GA 使用 64 个个体、最多 20 代及连续 6 代无改进提前停止；本次实际执行了
809 个不重复适应度计算，等价解缓存命中 75 次。

### 6.2 Leadfield 与直接 FEM 对比

直接 FEM 使用与 leadfield 优化相同的电极、电流、导电率、ROI、阈值及
`max_TI` 计算方式重新仿真。两条路径的结果如下：

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 58.429177 | 59.359489 | 1.567% |
| ROC distance | 0.829922 | 0.820619 | 1.121% |
| ROI sensitivity | 56.816% | 59.069% | 3.815% |
| 非 ROI false-positive rate | 70.872% | 71.125% | 0.356% |
| ROI mean | 0.202154 V/m | 0.203650 V/m | 0.735% |
| 非 ROI mean | 0.193906 V/m | 0.194680 V/m | 0.397% |
| ROI/Rest ratio | 1.042533 | 1.046079 | 0.339% |
| ROI max | 0.481246 V/m | 0.484483 V/m | 0.668% |

验证规则只检查 Leadfield 与直接 FEM 的 focality score 相对误差是否不超过 5%。
本次误差为 **1.567%**，因此 `comparison.json` 中：

```text
leadfield_fem_focality_score_relative_error_within_5_percent = true
passed = true
```

这里的 `passed=true` 表示两种计算路径的一致性验证通过，并不表示当前结果已经具有
很好的空间聚焦性。`score` 是便于阅读的正值 `-objective`；GA 实际最小化
`objective = -100 * (sqrt(2) - ROC distance)`，所以 score 越大越好，理论最大值为
141.421。当前 Leadfield/直接 FEM 的 ROI sensitivity 分别只有 56.8%/59.1%，同时
非 ROI false-positive rate 均约为 71%，ROI/Rest mean ratio 也仅约为 1.04。因此，
本次结果证明了 Leadfield 与直接 FEM 数值基本一致，但从这些 focality 指标看，右侧
海马的实际空间聚焦程度仍然不强。

本次首次执行在写 `comparison.json` 时还暴露了 NumPy `bool_` 不能直接 JSON 序列化
的问题。该问题发生在全部数值计算完成之后，已修复为统一转换 NumPy scalar/array，
并在打开目标文件前先完成序列化；随后利用已有 Leadfield、GA 和直接 FEM 结果恢复了
上述比较文件，没有重新执行昂贵计算。

主要结果文件如下：

- `data/ti_leadfield_ga_ernie_nsn_10_10/results/optimization_result.json`：最优解与
  Leadfield 指标；
- `data/ti_leadfield_ga_ernie_nsn_10_10/results/ti_leadfield_ga_result.msh`：Leadfield
  优化场及 ROI mask；
- `data/ti_leadfield_ga_ernie_nsn_10_10/results/final_validation/optimized/ernie_TI.msh`：
  直接 FEM 验证场；
- `data/ti_leadfield_ga_ernie_nsn_10_10/results/final_validation/comparison.json`：两条
  路径的指标、相对误差及验证结论。

## 7. 运行耗时

本次运行开始时，demo 的绝对耗时日志尚未正确写入文件，原因是 `python -m` 下
demo logger 名称为 `__main__`。因此，以下耗时由同一次运行的日志边界计算。主流程从
`2026-08-18 11:23:09` 到 `11:49:31`，总耗时约 **26 分 22 秒**。结果恢复阶段另耗时
约 73 秒，不计入主流程耗时，因为它只是修复 JSON 写入后的恢复操作。

| 阶段 | 起止时间 | 耗时 |
| --- | --- | ---: |
| 首次生成 NSN leadfield | 11:23:10–11:36:32 | 13 分 22 秒 |
| 加载完整 float64 leadfield | 11:36:32–11:36:42 | 10 秒 |
| 构建右海马 ROI 与非 ROI | 11:36:42–11:38:38 | 1 分 56 秒 |
| 六基因 GA 优化 | 11:38:38–11:43:04 | 4 分 26 秒 |
| 优化结果 mesh/NIfTI 导出 | 11:43:04–11:44:31 | 1 分 27 秒 |
| 直接 FEM、TI 计算及 ROI 统计 | 11:44:31–11:49:31 | 5 分 |

与上一版运行相比，GA 阶段由 1 小时 28 分 39 秒降至 4 分 26 秒，约为原来的
`1/20`；整个主流程由 1 小时 57 分 35 秒降至约 26 分 22 秒，约为原来的 `1/4.5`。
整体耗时不能视为严格的同条件性能基准，因为本次同时更换了 montage、目标函数、GA
搜索空间，并且不再执行 baseline 直接 FEM；但 GA 内部取消每个染色体的电流组合
穷举后，适应度搜索已不再是总耗时瓶颈，首次 leadfield 生成成为耗时最长的阶段。

demo 日志配置现已修正；后续新运行会直接记录每个阶段的 `stage_elapsed`、累计
`total_elapsed` 以及最终 `total_elapsed_seconds`，无需再通过日志时间戳推算。

## 8. 扩大六基因 GA 搜索

完整枚举两路 `0.05–4.00 mA` 电流会使每个电极候选固定增加 6400 次 focality
计算，预计耗时过长，因此继续使用六基因联合搜索，仅适度扩大 GA 搜索预算：

- 染色体保持 `(A+, A-, B+, B-, current_A_tick, current_B_tick)`；
- population size 从 64 增加到 80；
- 最大代数从 20 增加到 30；
- 连续无改进提前停止代数从 6 增加到 10；
- 两路电流仍分别为 `0.05–4.00 mA`、步长 `0.05 mA`，不设置总和约束；
- NSN montage、完整右海马 ROI、SimNIBS focality 目标和直接 FEM 验证保持不变；
- 继续复用现有 NSN leadfield，新结果写入
  `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended`，不覆盖首次六基因实验；
- 使用独立日志目录记录阶段耗时、总耗时和实际不重复适应度计算次数。

理论最大 GA 候选预算由 `64 * 20` 增加为 `80 * 30`，约为原来的 1.875 倍；
实际耗时还会受到初始种群、重复候选缓存和提前停止影响，以运行日志为准。

### 8.1 第一轮扩大搜索实际结果

2026 年 8 月 18 日已完成第一轮扩大搜索。运行复用了 NSN 10-10 leadfield，完整
流程耗时 **10 分 25.033 秒**（625.033 秒），各阶段耗时如下：

| 阶段 | 耗时 |
| --- | ---: |
| 复用并校验 leadfield | 0.091 秒 |
| 加载完整 float64 leadfield | 6.608 秒 |
| 构建完整右海马 ROI 与非 ROI | 1 分 09.074 秒 |
| 六基因 GA 优化 | 5 分 09.806 秒 |
| 重建并导出最优 TI 电场 | 47.736 秒 |
| 直接 FEM 对照与 ROI 统计 | 3 分 11.658 秒 |
| 写入结果并验收 | 0.001 秒 |

GA 共执行 1533 个不重复适应度计算，等价解缓存命中 143 次。最优 objective 在
第 25 代出现，之后到第 30 代没有继续改善；连续无改进只有 5 代，尚未达到提前停止
阈值 10 代，因此本轮由最大代数结束。最优解为：

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | FCz | O2 | 2.00 mA |
| B | T8 | FT8 | 3.60 mA |

Leadfield 与直接 FEM 的验证结果如下：

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 72.911916 | 70.818361 | 2.871% |
| ROC distance | 0.685094 | 0.706030 | 2.965% |
| ROI sensitivity | 66.443% | 84.898% | 21.737% |
| 非 ROI false-positive rate | 59.728% | 68.969% | 13.398% |
| ROI mean | 0.215502 V/m | 0.245723 V/m | 12.299% |
| 非 ROI mean | 0.139335 V/m | 0.173456 V/m | 19.672% |
| ROI/Rest ratio | 1.546649 | 1.416625 | 8.407% |
| ROI max | 0.561982 V/m | 0.688759 V/m | 18.407% |

focality score 相对误差为 **2.871%**，小于 5%，因此一致性验收
`passed=true`。但直接 FEM 的非 ROI false-positive rate 为 68.969%，ROI/Rest
ratio 仅为 1.417，所以第一轮扩大搜索同样没有获得很强的右海马空间聚焦。

第一轮结果保存在
`data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended/results`，其中直接 FEM 对照文件为
`final_validation/comparison.json`。

### 8.2 第二轮扩大搜索

第一轮扩大搜索在第 25 代获得新的最优值，运行到第 30 代时仍未满足连续 10 代
无改进的停止条件，说明当前代数上限可能截断了后续搜索。因此第二轮继续保留六基因
联合优化，只扩大搜索预算：

- population size 从 80 增加到 120；
- 最大代数从 30 增加到 50；
- 连续无改进提前停止代数从 10 增加到 15；
- 理论候选预算从 `80 * 30 = 2400` 增加到 `120 * 50 = 6000`，约为第一轮的
  2.5 倍；
- 继续复用相同 NSN leadfield，结果写入
  `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended_v2`；
- 使用 `ti_leadfield_ga_nsn_10_10_ga_extended_v2_demo` 独立日志目录，不覆盖
  第一轮扩大搜索的结果和耗时记录。

### 8.3 第二轮扩大搜索实际结果

2026 年 8 月 18 日已完成第二轮扩大搜索。运行复用了 NSN 10-10 leadfield，未重新
执行 leadfield FEM；完整流程的绝对计时为 **28 分 44.765 秒**（1724.765 秒）。

| 阶段 | 耗时 |
| --- | ---: |
| 复用并校验 leadfield | 0.092 秒 |
| 加载完整 float64 leadfield | 8.832 秒 |
| 构建完整右海马 ROI 与非 ROI | 1 分 50.886 秒 |
| 六基因 GA 优化 | 20 分 08.397 秒 |
| 重建并导出最优 TI 电场 | 1 分 29.640 秒 |
| 直接 FEM 对照与 ROI 统计 | 5 分 06.832 秒 |
| 写入结果并验收 | 0.002 秒 |

GA 完成了配置的 50 代搜索，共执行 3740 个不重复适应度计算，等价解缓存命中
341 次。最优 objective 首次出现在第 38 代，之后到第 50 代没有继续改善；由于连续
无改进尚未达到 15 代，因此本次由最大代数结束，而不是提前停止。最优解为：

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | F8 | AF7 | 1.85 mA |
| B | T8 | CP6 | 2.80 mA |

两路电流均满足各自 `0.05–4.00 mA`、步长 `0.05 mA` 的约束，不存在两路电流总和
约束。

### 8.4 第二轮 Leadfield 与直接 FEM 验证

本轮继续使用非 ROI `0.1 V/m`、ROI `0.2 V/m` 的 SimNIBS focality 阈值。最优解
经直接 FEM 重新计算后的对比如下：

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 73.235728 | 70.890148 | 3.203% |
| ROC distance | 0.681856 | 0.705312 | 3.326% |
| ROI sensitivity | 66.331% | 72.309% | 8.267% |
| 非 ROI false-positive rate | 59.293% | 64.868% | 8.594% |
| ROI mean | 0.211156 V/m | 0.224917 V/m | 6.118% |
| 非 ROI mean | 0.148481 V/m | 0.158156 V/m | 6.117% |
| ROI/Rest ratio | 1.422108 | 1.422119 | 0.00081% |
| ROI max | 0.537457 V/m | 0.572787 V/m | 6.168% |

验收程序只判断 focality score 的 Leadfield/直接 FEM 相对误差是否不超过 5%。本轮
误差为 **3.203%**，因此 `comparison.json` 中
`leadfield_fem_focality_score_relative_error_within_5_percent=true` 且
`passed=true`。这仍然只说明两条计算路径在验收指标上基本一致，并不代表空间聚焦
效果已经合格。

从实际 focality 指标看，Leadfield 中 ROI 达到 `0.2 V/m` 的比例为 66.331%，但
非 ROI 达到 `0.1 V/m` 的比例仍为 59.293%；直接 FEM 中两者分别为 72.309% 和
64.868%。也就是说，ROI 命中率有所保证，但大约六成以上的非 ROI 也超过了更低的
非 ROI 阈值。结合直接 FEM 的 ROI/Rest mean ratio 仅为 1.422，可判断本轮仍未形成
很强的右海马空间聚焦。

与第一轮扩大搜索（population 80、30 代）相比，第二轮 Leadfield score 从
72.911916 提高到 73.235728，提升 **0.444%**；直接 FEM score 从 70.818361 提高到
70.890148，仅提升 **0.101%**。相应地，GA 耗时从 5 分 09.806 秒增加到
20 分 08.397 秒，约为 3.90 倍；总耗时从 10 分 25.033 秒增加到 28 分 44.765 秒，
约为 2.76 倍。因此，扩大搜索确实找到了略优解，但耗时增加明显，focality 改善较小，
当前瓶颈更可能是 montage、ROI/非 ROI 阈值及目标函数所定义的可达解空间，而不只是
GA 搜索轮数不足。

本轮结果与验证文件位于：

- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended_v2/results/optimization_result.json`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended_v2/results/convergence.csv`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended_v2/results/ti_leadfield_ga_result.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended_v2/results/final_validation/optimized/ernie_TI.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_extended_v2/results/final_validation/comparison.json`。

## 9. 第一轮 GA 条件下的 2–4 mA 电流搜索

为检查较低电流候选是否限制了搜索，本轮恢复第一轮扩大搜索的 GA 条件，同时收紧
两路独立电流的下界：

- population size：80；
- 最大代数：30；
- 连续 10 代无改进时提前停止；
- 两路电流分别独立搜索 `2.00–4.00 mA`，步长 `0.05 mA`，不设置总和约束；
- montage、完整右海马 ROI、非 ROI `0.1 V/m`、ROI `0.2 V/m` 和 SimNIBS
  focality 目标保持不变；
- 复用现有 NSN 10-10 leadfield；
- 结果写入 `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4`；
- 日志写入 `log/ti_leadfield_ga_nsn_10_10_ga_current_2_4_demo`，不覆盖前三次实验；
- GA 结束后继续运行直接 FEM，并以 focality score 相对误差不超过 5% 作为
  Leadfield/直接 FEM 一致性验收条件。

### 9.1 实际耗时与最优解

2026 年 8 月 18 日已完成本轮完整 demo。运行复用了 NSN 10-10 leadfield，总耗时
为 **9 分 15.236 秒**（555.236 秒）。

| 阶段 | 耗时 |
| --- | ---: |
| 复用并校验 leadfield | 0.085 秒 |
| 加载完整 float64 leadfield | 6.577 秒 |
| 构建完整右海马 ROI 与非 ROI | 1 分 10.687 秒 |
| 六基因 GA 优化 | 4 分 19.168 秒 |
| 重建并导出最优 TI 电场 | 43.751 秒 |
| 直接 FEM 对照与 ROI 统计 | 2 分 54.909 秒 |
| 写入结果并验收 | 0.001 秒 |

GA 共执行 1442 个不重复适应度计算，等价解缓存命中 246 次。第 20 代获得
71.502320 的 score，随后连续 9 代没有改善，但在第 30 代又提高到最终最优值，因此
本轮由最大代数结束，没有触发连续 10 代无改进的提前停止。最优解为：

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | F6 | AFz | 4.00 mA |
| B | T8 | CP6 | 2.60 mA |

其中 A 路电流达到本轮搜索上限 4.00 mA，两路电流均满足独立 `2.00–4.00 mA`、
步长 `0.05 mA` 和无总和约束的配置。

### 9.2 Leadfield 与直接 FEM 验证

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 71.535647 | 71.508571 | 0.03785% |
| ROC distance | 0.698857 | 0.699128 | 0.03873% |
| ROI sensitivity | 60.764% | 65.475% | 7.196% |
| 非 ROI false-positive rate | 57.832% | 60.793% | 4.871% |
| ROI mean | 0.203414 V/m | 0.211605 V/m | 3.871% |
| 非 ROI mean | 0.149849 V/m | 0.155890 V/m | 3.875% |
| ROI/Rest ratio | 1.357459 | 1.357402 | 0.00424% |
| ROI max | 0.499067 V/m | 0.531874 V/m | 6.168% |

focality score 相对误差为 **0.03785%**，小于 5%，因此
`leadfield_fem_focality_score_relative_error_within_5_percent=true` 且
`passed=true`。本轮 Leadfield 与直接 FEM 的 focality score 一致性明显好于前两轮，
但这仍然不等价于聚焦效果合格。

直接 FEM 中 ROI 达到 `0.2 V/m` 的比例为 65.475%，非 ROI 达到 `0.1 V/m` 的比例
仍为 60.793%，ROI/Rest mean ratio 只有 1.357。因此，本轮结果仍表现为 ROI 和较大
范围的非 ROI 同时超过各自阈值，没有形成很强的右海马空间聚焦。

与相同 population、代数和提前停止条件的第一轮扩大搜索相比：

- 总耗时由 10 分 25.033 秒降至 9 分 15.236 秒，减少 69.797 秒（11.167%）；
- GA 耗时由 5 分 09.806 秒降至 4 分 19.168 秒，减少 50.638 秒（16.345%）；
- Leadfield score 从 72.911916 降至 71.535647，下降 1.888%；
- 直接 FEM score 从 70.818361 升至 71.508571，提高 0.975%。

原第一轮最优电流 2.00/3.60 mA 本身仍在新的 `2.00–4.00 mA` 搜索范围内，但本轮
GA 没有再次找到该电极与电流组合。由于 GA 是有限代随机搜索，改变电流基因边界也会
改变初始种群和后续进化轨迹。因此，本轮差异不能简单解释为“提高电流下界改善或降低
了聚焦”，只能说明在当前随机种子和 80×30 搜索预算下获得了上述候选。第 30 代仍有
新改善也表明搜索可能尚未完全收敛。

本轮主要结果位于：

- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4/results/optimization_result.json`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4/results/convergence.csv`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4/results/ti_leadfield_ga_result.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4/results/final_validation/optimized/ernie_TI.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4/results/final_validation/comparison.json`。

## 10. ROI 阈值 0.1 V/m 实验

本轮只把 ROI 阈值从 `0.2 V/m` 降为 `0.1 V/m`，用于观察目标阈值变化对 GA 搜索
和最终 focality 的影响。其余条件与第 9 节完全相同：

- population size：80；最大代数：30；连续 10 代无改进时提前停止；
- 两路电流分别独立搜索 `2.00–4.00 mA`，步长 `0.05 mA`，无总和约束；
- montage、完整右海马 ROI、非 ROI 阈值 `0.1 V/m`、SimNIBS focality 目标、
  random seed 和直接 FEM 验收规则保持不变；
- 本轮 ROI 和非 ROI 的 ROC 判定阈值均为 `0.1 V/m`；
- 复用现有 NSN 10-10 leadfield；
- 结果写入 `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4_roi_0_1`；
- 日志写入 `log/ti_leadfield_ga_nsn_10_10_ga_current_2_4_roi_0_1_demo`，不覆盖前述
  实验结果。

### 10.1 实际耗时与最优解

2026 年 8 月 18 日已完成本轮完整 demo。运行复用了 NSN 10-10 leadfield，总耗时
为 **10 分 47.036 秒**（647.036 秒）。

| 阶段 | 耗时 |
| --- | ---: |
| 复用并校验 leadfield | 0.094 秒 |
| 加载完整 float64 leadfield | 6.929 秒 |
| 构建完整右海马 ROI 与非 ROI | 1 分 15.281 秒 |
| 六基因 GA 优化 | 5 分 23.828 秒 |
| 重建并导出最优 TI 电场 | 47.401 秒 |
| 直接 FEM 对照与 ROI 统计 | 3 分 13.446 秒 |
| 写入结果并验收 | 0.001 秒 |

GA 共执行 1526 个不重复适应度计算，等价解缓存命中 150 次。第 28 代获得最终最优
值，随后运行到第 30 代结束，没有触发提前停止。最优解为：

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | CP6 | CP4 | 3.05 mA |
| B | F8 | Fp2 | 2.10 mA |

两路电流均满足独立 `2.00–4.00 mA`、步长 `0.05 mA` 和无总和约束的配置。

与 ROI 阈值为 `0.2 V/m` 的上一轮相比，总耗时增加 91.800 秒（16.534%），GA
耗时增加 64.660 秒（24.949%）。本轮不重复适应度计算从 1442 次增加到 1526 次，
同时运行时系统负载也会影响单次测量，因此耗时变化不能解释为阈值计算本身变慢。

### 10.2 Leadfield 与直接 FEM 验证

本轮 ROI sensitivity 与非 ROI false-positive rate 都按 `max_TI >= 0.1 V/m`
统计。结果如下：

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 108.312665 | 106.808272 | 1.389% |
| ROC distance | 0.331087 | 0.346131 | 4.346% |
| ROI sensitivity | 82.197% | 88.939% | 7.580% |
| 非 ROI false-positive rate | 27.915% | 32.798% | 14.888% |
| ROI mean | 0.120092 V/m | 0.130695 V/m | 8.113% |
| 非 ROI mean | 0.091907 V/m | 0.099895 V/m | 7.996% |
| ROI/Rest ratio | 1.306676 | 1.308329 | 0.126% |
| ROI max | 0.383645 V/m | 0.423344 V/m | 9.377% |

focality score 相对误差为 **1.389%**，小于 5%，因此
`leadfield_fem_focality_score_relative_error_within_5_percent=true` 且
`passed=true`。这表示本轮 Leadfield 与直接 FEM 的 focality score 一致性验收通过。

直接 FEM 中，右海马 ROI 有 88.939% 的 element 达到 `0.1 V/m`，非 ROI 中达到
同一阈值的比例为 32.798%。与上一轮相比，非 ROI 使用的阈值同为 `0.1 V/m`，其直接
FEM false-positive rate 从 60.793% 降低 27.995 个百分点，相对降低 46.050%。这说明
新目标找到的候选在 `0.1 V/m` 这一判定点上具有更好的 ROI/非 ROI 分类表现。

但是，上一轮 ROI sensitivity 使用的是 `0.2 V/m`，本轮使用 `0.1 V/m`，所以两轮
score、ROC distance 和 ROI sensitivity 的绝对值不能直接比较。本轮直接 FEM 的
ROI/Rest mean ratio 仅为 1.308，且仍有约三分之一的非 ROI 超过 `0.1 V/m`，因此
可以说阈值化 focality 明显改善，但仍不能据此认定已经形成高度集中的右海马空间聚焦。

本轮主要结果位于：

- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4_roi_0_1/results/optimization_result.json`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4_roi_0_1/results/convergence.csv`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4_roi_0_1/results/ti_leadfield_ga_result.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4_roi_0_1/results/final_validation/optimized/ernie_TI.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_2_4_roi_0_1/results/final_validation/comparison.json`。

## 11. MNI 坐标球形 ROI 与 0–4 mA 实验

本轮不再使用右海马 Brainnetome atlas mask 直接定义 ROI，改用 SimNIBS 原生 MNI
球形 ROI：

- MNI 球心：`[26, -21, -15] mm`；
- 半径：`15 mm`；
- 运行时仅把球心从 MNI 空间变换到 ernie subject 空间，再按 mesh element center
  到球心的距离选取 WM/GM 四面体，不读取海马 atlas mask；
- 15 mm 半径大于此前完整右海马 mask 的约 12.8 mm 等效球半径，避免使用过小 ROI，
  但球形区域也会覆盖海马周围的部分组织，因此应解释为“右海马区域附近的几何 ROI”，
  不能视为严格的海马解剖边界；
- 两路电流分别独立搜索 `0.00–4.00 mA`，步长 `0.05 mA`，不设置总和约束；
- ROI 阈值恢复为 `0.2 V/m`，非 ROI 阈值保持 `0.1 V/m`；
- GA 保持 population 80、最大 30 代、连续 10 代无改进提前停止和相同 random seed；
- montage、leadfield、导电率、直接 FEM 和 5% focality score 一致性验收规则不变；
- 结果写入 `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_mni_sphere_15mm`；
- 日志写入 `log/ti_leadfield_ga_nsn_10_10_ga_current_0_4_mni_sphere_15mm_demo`，不覆盖
  之前实验。

### 11.1 ROI 覆盖、实际耗时与最优解

15 mm MNI 球形 ROI 在 ernie WM/GM mesh 中命中 10,531 个四面体，体积为
13,468.998 mm³；Rest 包含 1,347,247 个四面体，体积为 1,296,224.156 mm³。球形
ROI 体积约为此前完整右海马 atlas ROI 的两倍，确认半径没有取得过小。

2026 年 8 月 18 日已完成本轮完整 demo。运行复用了 NSN 10-10 leadfield，总耗时
为 **7 分 38.173 秒**（458.173 秒）。

| 阶段 | 耗时 |
| --- | ---: |
| 复用并校验 leadfield | 0.091 秒 |
| 加载完整 float64 leadfield | 6.761 秒 |
| 构建 MNI 球形 ROI 与非 ROI | 2.105 秒 |
| 六基因 GA 优化 | 4 分 43.171 秒 |
| 重建并导出最优 TI 电场 | 47.115 秒 |
| 直接 FEM 对照与球形 ROI 统计 | 1 分 58.873 秒 |
| 写入结果并验收 | 0.001 秒 |

GA 共执行 1503 个不重复适应度计算，等价解缓存命中 190 次。第 21 代获得
69.554065 的 score，随后连续 8 代没有改善，但在第 30 代提高到最终最优值，因此
本轮由最大代数结束，没有触发连续 10 代无改进的提前停止。最优解为：

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | AF3 | F8 | 2.50 mA |
| B | C6 | FT8 | 3.00 mA |

两路最优电流均为非零值，满足各自独立 `0.00–4.00 mA`、步长 `0.05 mA` 和无总和
约束的配置。

与第 9 节同为 ROI `0.2 V/m`、population 80 和 30 代的 atlas ROI 实验相比，本轮
总耗时减少 97.063 秒（17.481%）。其中首次 ROI 构建由 70.687 秒降至 2.105 秒，
直接 FEM 阶段中的第二次 ROI 统计也随之缩短；GA 耗时则从 259.168 秒增加到
283.171 秒。由于本轮同时改变了 ROI 定义和电流下界，整体耗时只能用于工程测量，
不能作为单一因素的严格性能基准。

### 11.2 Leadfield 与直接 FEM 验证

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 69.678904 | 68.481922 | 1.718% |
| ROC distance | 0.717425 | 0.729394 | 1.641% |
| ROI sensitivity | 72.861% | 78.824% | 7.565% |
| 非 ROI false-positive rate | 66.411% | 69.798% | 4.852% |
| ROI mean | 0.231190 V/m | 0.241357 V/m | 4.212% |
| 非 ROI mean | 0.176526 V/m | 0.184445 V/m | 4.293% |
| ROI/Rest ratio | 1.309667 | 1.308556 | 0.0848% |
| ROI max | 0.400631 V/m | 0.418495 V/m | 4.269% |

focality score 相对误差为 **1.718%**，小于 5%，因此
`leadfield_fem_focality_score_relative_error_within_5_percent=true` 且
`passed=true`。`comparison.json` 还显式记录了
`definition=mni_sphere`、MNI 球心和 15 mm 半径，便于确认两条路径使用的是同一几何
ROI。

一致性验收通过仍不等于空间聚焦良好。直接 FEM 中，球形 ROI 达到 `0.2 V/m` 的
element 比例为 78.824%，但非 ROI 达到 `0.1 V/m` 的比例仍为 69.798%，ROI/Rest
mean ratio 只有 1.309。因此，本轮对较大的右海马附近球形区域产生了较高场强，但
全脑 WM/GM 的超阈值范围仍然很大，聚焦程度依然不强。

直接 FEM 运行还出现两项质量提示：

- C6–FT8 通道的 SimNIBS current calibration error 估计为 18.25%，超过 10%；
- 电极网格质量统计出现一次 `divide by zero encountered in divide` RuntimeWarning。

这些提示没有中断计算，demo 正常退出且最终 score 一致性验收通过；但它们会降低对
直接 FEM 绝对数值的置信度。本轮结果适合用于当前算法实验和趋势比较，在将该 montage
视为实际刺激方案前，应进一步检查 C6/FT8 电极放置和网格质量。

本轮主要结果位于：

- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_mni_sphere_15mm/results/optimization_result.json`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_mni_sphere_15mm/results/convergence.csv`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_mni_sphere_15mm/results/ti_leadfield_ga_result.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_mni_sphere_15mm/results/final_validation/optimized/ernie_TI.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_mni_sphere_15mm/results/final_validation/comparison.json`。

## 12. 完整右海马 ROI 与完整右杏仁核 non-ROI 实验

本轮恢复使用 Brainnetome 解剖图谱定义完整右侧海马 ROI，并把 non-ROI 从“其余全部
WM/GM”改成一个明确的相邻深部脑区——完整右侧杏仁核：

- ROI：`0216_rHipp_R.nii.gz` 与 `0218_cHipp_R.nii.gz` 的并集；
- non-ROI：`0212_mAmyg_R.nii.gz` 与 `0214_lAmyg_R.nii.gz` 的并集；
- 两个区域分别映射到 ernie subject mesh，并各自与 WM/GM 四面体求交；non-ROI 不再
  包含右杏仁核以外的其余脑组织；
- 如果映射后的 ROI 与 non-ROI 重叠、任一区域为空或 element 体积非法，程序在 GA
  前直接报错；
- 两路电流分别独立搜索 `0.00–4.00 mA`，步长 `0.05 mA`，无总和约束；
- ROI 阈值为 `0.2 V/m`，non-ROI 阈值为 `0.1 V/m`；
- GA 保持 population 80、最大 30 代、连续 10 代无改进提前停止和相同 random seed；
- montage、leadfield、导电率、直接 FEM 和 5% focality score 一致性验收规则不变；
- 使用新的日志与结果目录，避免覆盖前述实验。

选择右杏仁核的依据是它位于右海马前方且同为深部结构，能够较严格地检验相邻深部
脑区选择性。人体海马 TI 研究的主要深浅选择性对照是上覆皮层，同时还检查了位于
目标海马前方的杏仁核是否出现非目标效应；因此本轮把右杏仁核作为“一个指定 atlas
non-ROI”是对其空间特异性思路的工程化近似。需要注意，本轮 focality 只衡量右海马
相对于右杏仁核的区分能力，不再代表其余全脑区域的场强受控或全脑聚焦。

文献依据：Grossman 等人的早期 TI 研究以“刺激海马而不募集上覆皮层”为深部选择性
证据；Violante 等人的人体海马 TI 研究同样以海马和上覆皮层为主要对照，并额外检查
了位于目标海马前方的杏仁核是否出现非目标效应。本轮使用 Brainnetome 的完整右杏仁核
作为单一 atlas non-ROI，属于结合上述空间特异性思路与当前图谱能力作出的工程选择，
并非复刻论文中的原始 ROI。

- Grossman et al., *Noninvasive Deep Brain Stimulation via Temporally Interfering Electric Fields*：
  <https://doi.org/10.1016/j.cell.2017.05.024>；
- Violante et al., *Non-invasive temporal interference electrical stimulation of the human hippocampus*：
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10620081/>；
- Fan et al., *The Human Brainnetome Atlas*：
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4961028/>。

### 12.1 实际覆盖、耗时与最优解

2026 年 8 月 18 日已完成本轮完整 demo。映射后的完整右海马 ROI 包含 5,370 个
WM/GM 四面体，体积为 6,646.374 mm³；完整右杏仁核 non-ROI 包含 835 个四面体，
体积为 1,895.335 mm³。两者没有重叠。运行复用了现有 NSN 10-10 leadfield，总耗时
为 **11 分 59.356 秒**（719.356 秒）。

| 阶段 | 耗时 |
| --- | ---: |
| 复用并校验 leadfield | 0.102 秒 |
| 加载完整 float64 leadfield | 6.839 秒 |
| 映射右海马 ROI 与右杏仁核 non-ROI | 2 分 22.198 秒 |
| 六基因 GA 优化 | 4 分 02.502 秒 |
| 重建并导出最优 TI 电场 | 55.892 秒 |
| 直接 FEM 对照与 atlas 区域统计 | 4 分 31.759 秒 |
| 写入结果并验收 | 0.002 秒 |

GA 共执行 1,293 个不重复适应度计算，等价解缓存命中 113 次。第 13 代获得最终最优
值，随后因连续无改进提前结束，`convergence.csv` 最后记录到第 25 代。最优解为：

| 通道 | 正极 | 负极 | 电流 |
| --- | --- | --- | ---: |
| A | AF8 | Oz | 3.40 mA |
| B | O2 | POz | 3.10 mA |

两路电流均满足各自独立 `0.00–4.00 mA`、步长 `0.05 mA` 和无总和约束的配置。

### 12.2 Leadfield 与直接 FEM 验证

本轮 ROI sensitivity 按右海马 `max_TI >= 0.2 V/m` 统计，non-ROI false-positive
rate 按右杏仁核 `max_TI >= 0.1 V/m` 统计。结果如下：

| 指标 | Leadfield | 直接 FEM | 相对误差 |
| --- | ---: | ---: | ---: |
| focality score | 54.371528 | 54.281356 | 0.166% |
| ROC distance | 0.870498 | 0.871400 | 0.103% |
| ROI sensitivity | 20.819% | 22.216% | 6.287% |
| 右杏仁核 false-positive rate | 36.168% | 39.281% | 7.927% |
| ROI mean | 0.127825 V/m | 0.131162 V/m | 2.544% |
| 右杏仁核 mean | 0.093186 V/m | 0.095400 V/m | 2.320% |
| ROI/non-ROI mean ratio | 1.371718 | 1.374865 | 0.229% |
| ROI max | 0.380202 V/m | 0.393893 V/m | 3.476% |

focality score 相对误差为 **0.166%**，小于 5%，因此
`leadfield_fem_focality_score_relative_error_within_5_percent=true` 且
`passed=true`。这说明 Leadfield 与直接 FEM 在本轮指定的“右海马/右杏仁核”目标函数
上高度一致。

但是，该一致性验收通过不等于本轮达到了良好聚焦。直接 FEM 中只有 22.216% 的右海马
element 达到 `0.2 V/m`，却有 39.281% 的右杏仁核 element 达到 `0.1 V/m`；右海马
与右杏仁核的平均场强比也只有 1.375。因此本轮结果对相邻深部结构的阈值区分仍然较差，
不能认定已经形成良好的右海马选择性刺激。

同时，由于 non-ROI 只包含右杏仁核，本轮 score 即使更高也不能证明其他皮层或全脑区域
场强较低；它和此前使用全脑 Rest 的 score、false-positive rate 不是同一个统计域，
不能直接横向比较。JSON 与代码中的 `rest_mask`、`rest_mean_v_per_m`、
`roi_rest_ratio` 是为了兼容既有内部接口保留的字段名，本轮分别表示右杏仁核 mask、
右杏仁核均值和 ROI/non-ROI 均值比。结果 mesh 另行写入了语义明确的
`non_ROI_mask` 字段。

直接 FEM 的两路 current calibration error 分别为 3.0% 和 9.3%，均未超过 10%；
电极网格质量统计仍出现一次 `divide by zero encountered in divide` RuntimeWarning，
atlas 映射也出现 `invalid value encountered in dot` RuntimeWarning。这些警告没有中断
计算，且 Leadfield 与直接 FEM 映射得到的 ROI/non-ROI element 数和体积完全一致。

本轮主要结果位于：

- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_hippocampus_vs_amygdala/results/optimization_result.json`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_hippocampus_vs_amygdala/results/convergence.csv`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_hippocampus_vs_amygdala/results/ti_leadfield_ga_result.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_hippocampus_vs_amygdala/results/final_validation/optimized/ernie_TI.msh`；
- `data/ti_leadfield_ga_ernie_nsn_10_10_ga_current_0_4_hippocampus_vs_amygdala/results/final_validation/comparison.json`。

日志位于
`log/ti_leadfield_ga_nsn_10_10_ga_current_0_4_hippocampus_vs_amygdala_demo`，与此前
实验相互独立。
