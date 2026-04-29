# CHARM 重建表面直连 Niivue 方案

## 1. 目标

新增一条基于 CHARM 重建表面的可视化导出链路：

1. 直接使用 CHARM 第 6 步生成的皮层重建表面
2. 作为 CHARM 第 8 步导出 Niivue 可直接加载的 `.gii`
3. 当前实现固定使用 `central` 表面
4. 输出文件固定写到受试者目录下，文件名固定为 `surface.gii`

---

## 2. 背景

CHARM 在第 6 步已经生成了左右半球皮层重建表面：

1. `surfaces/lh.white.gii`
2. `surfaces/rh.white.gii`
3. `surfaces/lh.pial.gii`
4. `surfaces/rh.pial.gii`
5. `surfaces/lh.central.gii`
6. `surfaces/rh.central.gii`

这些表面本身就是面向皮层几何表达的重建结果，适合作为前端皮层显示的输入。

Niivue 官方支持直接加载 `GIfTI (.gii)` mesh，因此不需要额外改造为其他格式，就可以直接复用这套重建表面。

---

## 3. 三种重建表面的含义

虽然当前实现只使用 `central`，但文档里需要把三种表面的语义写清楚，避免后续误用。

### 3.1 `white`

`white` 表示白质侧皮层表面。

几何上，它位于灰质与白质的交界附近，偏向皮层内侧。更具体地说：

1. 它贴近白质外边界
2. 它描述的是皮层内侧的褶皱形态
3. 在脑沟区域会更靠近沟底内侧

适用场景：

1. 关注白质侧皮层边界
2. 关注皮层内侧几何形态
3. 做与白质表面相关的表面分析

局限：

1. 对纯可视化而言，它通常偏内缩
2. 单独显示时，整体观感不如中间层自然

### 3.2 `pial`

`pial` 表示皮层外侧表面。

几何上，它位于灰质与脑脊液接触的一侧，偏向皮层外侧。更具体地说：

1. 它贴近皮层最外层
2. 它描述的是脑回冠部与外层脑沟轮廓
3. 它更接近人们直观看到的脑表外轮廓

适用场景：

1. 关注皮层外轮廓
2. 关注靠近脑脊液侧的表面形态
3. 需要强调脑表外层边界时

局限：

1. 它更偏外层，容易受外轮廓形态主导
2. 在某些视角下，沟内信息不如中间层平衡

### 3.3 `central`

`central` 表示位于 `white` 与 `pial` 之间的中间皮层表面。

几何上，它不贴近皮层内边界，也不贴近皮层外边界，而是处于两者之间的中间位置。更具体地说：

1. 它比 `white` 更不内缩
2. 它比 `pial` 更不外鼓
3. 它通常更适合作为“整层皮层”的代表表面

适用场景：

1. 前端默认皮层显示
2. 关注整体沟回形态
3. 希望在可读性和几何稳定性之间取得平衡

相对优势：

1. 对脑沟脑回显示更均衡
2. 单层显示时观感通常最稳定
3. 作为默认表面更符合当前 Niivue 使用场景

---

## 4. 为什么当前固定使用 `central`

本次实现固定使用 `central`，不导出 `white` 和 `pial`，原因如下：

1. 当前目标是改善 Niivue 中的皮层表面显示效果
2. `central` 最适合作为单一默认显示表面
3. 它比 `white` 与 `pial` 更适合承载“整体皮层形态”的视觉表达
4. 当前需求是收敛到单一稳定产物，而不是同时暴露多个切换选项

因此本次功能边界明确为：

1. 读取 `lh.central.gii`
2. 读取 `rh.central.gii`
3. 合并左右半球
4. 输出单一文件 `surface.gii`

---

## 5. 输入输出定义

### 5.1 输入

输入目录为 CHARM 受试者目录：

- `m2m_{subid}/`

本次固定输入文件为：

- `m2m_{subid}/surfaces/lh.central.gii`
- `m2m_{subid}/surfaces/rh.central.gii`

### 5.2 输出

本次固定输出到受试者目录本身，不新建额外子目录。

固定输出文件为：

- `m2m_{subid}/surface.gii`

这里要明确：

1. 输出格式固定为 `GIfTI (.gii)`
2. 输出文件名固定为 `surface.gii`
3. 输出对象是左右半球合并后的单一 `central` 表面

---

## 6. 表面处理策略

### 6.1 读取

固定使用：

1. `simnibs.mesh_tools.mesh_io.read_gifti_surface()` 读取 `lh.central.gii`
2. `simnibs.mesh_tools.mesh_io.read_gifti_surface()` 读取 `rh.central.gii`

### 6.2 合并

使用现有 mesh join 能力合并左右半球。

要求：

1. 右半球索引正确偏移
2. 不改变原始顶点坐标
3. 不引入平滑或重建后处理

### 6.3 写出

使用：

1. `simnibs.mesh_tools.mesh_io.write_gifti_surface()`

固定写出为：

- `{subject_dir}/surface.gii`

---

## 7. 接口建议

### 7.1 上层接口

建议位置：

- `neuracle/charm/export_niivue_surfaces.py`

建议接口：

- `export_niivue_surfaces(subject_dir)`

职责：

1. 校验 `surfaces/` 目录存在
2. 固定读取左右半球 `central`
3. 合并左右半球
4. 固定写出 `surface.gii`

### 7.2 底层接口

建议位置：

- `neuracle/utils/surface_to_niivue.py`

建议拆分为几个最小函数：

1. `read_central_surfaces(subject_dir)`
2. `join_hemisphere_surfaces(left_surface, right_surface)`
3. `write_niivue_surface_gifti(surface_mesh, output_path)`

---

## 8. 错误处理要求

需要显式处理：

1. `subject_dir` 不存在
2. `surfaces/` 目录不存在
3. `lh.central.gii` 不存在
4. `rh.central.gii` 不存在
5. GIfTI 读取失败
6. 左右半球合并失败
7. `surface.gii` 写出失败

---

## 9. 验收标准

1. 能从 `m2m_{subid}/surfaces/` 成功读取 `lh.central.gii` 与 `rh.central.gii`
2. 能合并左右半球
3. 能在 `m2m_{subid}/surface.gii` 生成单一输出文件
4. 输出文件可被 Niivue 直接加载
5. 默认不依赖额外平滑步骤

---

## 10. 本次结论

本次方案明确收敛为单一能力：

1. 基于 CHARM 重建表面
2. 固定使用 `central`
3. 作为 CHARM 第 8 步导出
4. 输出到受试者目录下的固定文件 `surface.gii`

这样可以最大限度减少前端接入复杂度，同时保留适合皮层显示的表面来源。
