# TI 逆向坐标批量脚本适配方案

## 背景

新增的 `run_inverse_debug.py`、`run_axis_sweep_batch.py`、`run_coordinate_batch.py`
和 `coordinate_campaign.json` 来自其他开发环境，当前包含外部仓库路径、其他用户目录、
缺失的 `invoke_neuracle.py` 依赖以及仅在脚本同目录执行时成立的导入方式，无法直接在本项目运行。

## 目标

1. 单次调试入口直接复用本项目 `neuracle.ti_inverse.run_ti_inverse()`，不再依赖外部仓库或缺失脚本。
2. 批量入口统一通过 `python -m neuracle...` 调用，保证从项目根目录运行时导入稳定。
3. 配置使用 `%LOCALAPPDATA%` 表达 NSN-F 数据目录，移除具体 Windows 用户名和外部虚拟环境路径。
4. 保留默认计划模式；只有显式传入 `--execute` 才启动耗时仿真。
5. 将批量配置中的 `seed`、`maxiter` 和 `popsize` 传给本项目 TesFlex 优化器，保证配置与实际行为一致。

## 修改设计

### 单次调试入口

- 仓库根目录由脚本位置自动确定，允许 `--neuracle-root` 显式覆盖，但不再要求 JSON 配置该字段。
- 使用当前 Python 进程导入 Neuracle API；运行命令必须由 `simnibs_env` 执行。
- 在独立时间戳目录写入 `execution.json`、`params.json`、`manifest.json` 和结果清单。
- 运行前继续校验头模、T1、Montage、ROI、电流、电导率和各向异性输入。

### 批量入口

- 模块之间使用 `neuracle.*` 绝对导入，批量子进程使用 `python -m neuracle.run_inverse_debug`。
- 坐标批量入口继续默认读取 `coordinate_campaign.json`。
- 轴向入口未收到配套的 `axis_sweep_campaign.json`，因此将 `--config` 设为必填，避免引用不存在的默认文件。
- 计划文件和断点状态仍写入配置指定的 `state_dir`。

### 优化器参数

- 为 `run_ti_inverse()` 增加可选 `optimizer_options` 参数，默认值为 `None`，不改变现有调用行为。
- 在优化器初始化后设置 `opt.optimizer_options`；当包含 `seed` 时同步设置 `opt.seed`。
- 新增脚本只允许传入 `maxiter`、`popsize` 和 `seed`。

## 验证

按照项目约定不新增测试代码，也不启动耗时 TI 优化。验证内容为：

1. 在 `simnibs_env` 中编译相关 Python 文件。
2. 执行三个入口的 `--help`，验证模块导入和 CLI。
3. 执行坐标批量入口的计划模式，验证本机 ernie 头模、Montage、参数规范化和三任务计划生成。
4. 检查 JSON 格式、Git diff 和空白错误。
