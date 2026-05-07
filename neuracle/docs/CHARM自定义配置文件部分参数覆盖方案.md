# CHARM 自定义配置文件部分参数覆盖方案

## 1. 目标

在 `neuracle` 侧固定读取一份 `json` 配置文件，不走 `--usesettings`，也不要求外部调用时传入配置路径。

本次只覆盖 4 个参数：

1. `segment.downsampling_targets`
2. `mesh.elem_sizes`
3. `mesh.skin_facet_size`
4. `mesh.facet_distances`

其余 CHARM 参数继续使用 `simnibs/charm.ini` 默认值。

## 2. 总体思路

配置读取逻辑分两层：

1. 先读取 `simnibs/charm.ini` 作为默认基线配置
2. 再读取 `neuracle` 根目录下的自定义 `json` 覆盖文件
3. 如果覆盖文件不存在，则完全使用默认配置
4. 如果覆盖文件存在，则只覆盖允许的 4 个字段
5. 未出现在 `json` 中的字段，继续使用 `simnibs/charm.ini` 中的值

这样可以保证：

1. 默认行为与当前流程一致
2. 自定义配置只影响少量明确字段
3. 调用方式不需要变化

## 3. 配置文件位置

覆盖文件固定放在：

`neuracle/charm_override.json`

这样做的原因是：

1. 路径固定，最直接
2. 便于人工编辑和排查
3. 不需要调用方关心额外的目录结构

## 4. 可覆盖范围

本次只允许覆盖以下 4 个字段：

1. `segment.downsampling_targets`
2. `mesh.elem_sizes`
3. `mesh.skin_facet_size`
4. `mesh.facet_distances`

除了这 4 个字段以外：

1. 不允许覆盖 `segment` 中的其他字段
2. 不允许覆盖 `mesh` 中的其他字段
3. 不允许覆盖其他 section

如果 `json` 中出现超出范围的 section 或 key，应直接报错。

## 5. 逻辑流程

运行时的逻辑流程如下：

1. 读取默认配置
2. 检查 `neuracle/charm_override.json` 是否存在
3. 如果不存在，直接使用默认配置继续执行
4. 如果存在，读取该文件
5. 校验该文件中是否只包含允许的 section 和字段
6. 将允许字段覆盖到默认配置中
7. 使用覆盖后的最终配置执行 `segment` 和 `mesh` 流程

整个流程对外部调用方透明，不增加新的输入参数。

## 6. 部分字段覆盖规则

允许只写其中一部分字段。

这意味着：

1. `json` 中写了的字段，会覆盖默认值
2. `json` 中没写的字段，继续使用 `simnibs/charm.ini` 中的值

例如：

1. 如果只写了 `mesh.skin_facet_size`
2. 那么 `segment.downsampling_targets`
3. `mesh.elem_sizes`
4. `mesh.facet_distances`

仍然继续使用默认配置中的原值。

## 7. 字段内部覆盖规则

这 4 个字段都按“整体替换”处理。

具体含义是：

1. `segment.downsampling_targets` 一旦出现在 `json` 中，就用 `json` 中的整个值替换默认值
2. `mesh.skin_facet_size` 一旦出现在 `json` 中，就用 `json` 中的整个值替换默认值
3. `mesh.elem_sizes` 一旦出现在 `json` 中，就用 `json` 中的整个对象替换默认值
4. `mesh.facet_distances` 一旦出现在 `json` 中，就用 `json` 中的整个对象替换默认值

特别说明：

1. `mesh.elem_sizes` 不做对象内部的逐项合并
2. `mesh.facet_distances` 不做对象内部的逐项合并

这样做是为了避免默认配置中的旧子项残留，导致最终行为不清晰。

## 8. 校验原则

覆盖文件需要做基础校验，至少包括：

1. 顶层只允许 `segment` 和 `mesh`
2. `segment` 下只允许 `downsampling_targets`
3. `mesh` 下只允许 `elem_sizes`、`skin_facet_size`、`facet_distances`
4. `downsampling_targets` 必须是数组
5. `elem_sizes` 必须是对象
6. `skin_facet_size` 必须是数字或 `false`
7. `facet_distances` 必须是对象

如果校验失败，应立即报错，而不是静默忽略。

## 9. 影响范围

本次改动只影响 `neuracle` 自己的配置读取层，不改变 SimNIBS 原始配置体系。

边界如下：

1. 不改 `simnibs/charm.ini`
2. 不改 SimNIBS CLI 的 `--usesettings`
3. 不增加新的外部调用参数
4. 不改变现有 CHARM 步骤的调用方式
5. 只影响 `segment` 和 `mesh` 对这 4 个字段的取值

## 10. 实施顺序

建议按以下顺序落地：

1. 先固定覆盖文件路径为 `neuracle/charm_override.json`
2. 再接入默认配置加覆盖配置的两层读取逻辑
3. 增加允许字段范围校验
4. 增加字段类型校验
5. 最后验证 `segment` 和 `mesh` 两步能正确读到覆盖后的参数

## 11. 验证 Demo

建议补一个最小验证 demo，用来确认覆盖逻辑符合预期。

验证目标：

1. 没有 `neuracle/charm_override.json` 时，流程完全使用 `simnibs/charm.ini`
2. 只写部分字段时，只有这些字段被覆盖
3. 未写字段继续沿用默认值
4. 出现不允许的字段时，流程会直接报错

建议至少覆盖以下三组场景：

### 11.1 默认回退场景

验证方式：

1. 暂时移走或删除 `neuracle/charm_override.json`
2. 运行 `segment` 或 `mesh` 对应步骤
3. 观察日志或运行结果中的参数取值

预期结果：

1. `segment.downsampling_targets` 使用 `simnibs/charm.ini` 的默认值
2. `mesh.elem_sizes` 使用 `simnibs/charm.ini` 的默认值
3. `mesh.skin_facet_size` 使用 `simnibs/charm.ini` 的默认值
4. `mesh.facet_distances` 使用 `simnibs/charm.ini` 的默认值

### 11.2 部分覆盖场景

验证方式：

1. 在 `neuracle/charm_override.json` 中只保留一个字段，例如只保留 `mesh.skin_facet_size`
2. 运行 `mesh` 步骤
3. 检查生效参数

预期结果：

1. `mesh.skin_facet_size` 使用 `json` 中的值
2. `segment.downsampling_targets` 继续使用默认值
3. `mesh.elem_sizes` 继续使用默认值
4. `mesh.facet_distances` 继续使用默认值

### 11.3 非法字段场景

验证方式：

1. 在 `neuracle/charm_override.json` 中加入一个不在白名单中的字段
2. 运行 `segment` 或 `mesh` 步骤

预期结果：

1. 流程在配置读取阶段直接报错
2. 不会静默忽略非法字段
3. 报错信息中应明确指出是哪一个字段不被允许

### 11.4 整体替换场景

验证方式：

1. 在 `neuracle/charm_override.json` 中写入 `mesh.elem_sizes`
2. 只保留你希望使用的那些组织项
3. 运行 `mesh` 步骤

预期结果：

1. `mesh.elem_sizes` 按 `json` 中的整个对象生效
2. 不会自动把 `simnibs/charm.ini` 中未写出的子项补回来
3. `mesh.facet_distances` 也应遵循同样规则

这个 demo 的价值是：

1. 能验证默认回退是否正常
2. 能验证部分覆盖是否正常
3. 能验证白名单校验是否正常
4. 能验证 `elem_sizes` 和 `facet_distances` 的整体替换语义是否正常

## 12. 结论

这次需求的核心不是替换 CHARM 的整套配置，而是在 `neuracle` 内部增加一层轻量覆盖机制。

最合适的方案是：

1. 保留 `simnibs/charm.ini` 作为默认基线
2. 在 `neuracle` 根目录下固定读取 `charm_override.json`
3. 允许只写部分字段
4. 未写字段继续使用默认值
5. 对出现的字段按整体替换处理
6. 只开放以下 4 个字段：
   - `segment.downsampling_targets`
   - `mesh.elem_sizes`
   - `mesh.skin_facet_size`
   - `mesh.facet_distances`

这样改动集中、行为清晰、兼容性也最好。
