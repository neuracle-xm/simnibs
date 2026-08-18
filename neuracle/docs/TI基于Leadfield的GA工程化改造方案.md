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
