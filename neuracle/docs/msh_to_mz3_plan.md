# MSH 转 MZ3 修订方案

## 1. 目标

在 `neuracle/mesh_tools/` 下提供一个稳定的 `.msh -> .mz3.gz` 转换器，用于把 SimNIBS 网格中的三角表面与可选的标量场导出为 surf-ice/BrainVisa 可读格式。

本次修订重点是纠正两个错误前提：

1. 不再手工从四面体重建界面，而是直接复用 SimNIBS 已经写入 `.msh` 的表面三角形 tag。
2. 不再把体网格上的场值直接写到 MZ3，而是先插值到表面顶点。

---

## 2. MZ3 约束

MZ3 这里仅写入：

- 三角面 `(Nf, 3)`，`uint32`
- 顶点坐标 `(Nv, 3)`，`float32`
- 可选顶点标量 `(Nv,)` 或 `(Nv, Ns)`，`float32`

注意：

- 面索引必须是 0-based。
- 标量长度必须与顶点数 `Nv` 一致。
- 向量场如 `E`、`J` 不能直接按 scalar 写入，应改用 `magnE`、`magnJ` 等标量字段。

---

## 3. 表面提取策略

### 3.1 使用 SimNIBS 既有表面 tag

SimNIBS 已定义表面标签：

- `WM_TH_SURFACE = 1001`
- `GM_TH_SURFACE = 1002`
- `CSF_TH_SURFACE = 1003`

对应实现应使用：

```python
surface_mesh = mesh.crop_mesh(tags=surface_tag, elm_type=2)
vertices = surface_mesh.nodes.node_coord
faces = surface_mesh.elm.node_number_list[:, :3] - 1
```

### 3.2 `surface_type` 语义

为兼容现有调用，保留以下映射：

- `white` -> `WM_TH_SURFACE (1001)`
- `central` -> `GM_TH_SURFACE (1002)`
- `pial` -> `CSF_TH_SURFACE (1003)`

说明：

- 这里的 `central` 指的是 `GM_TH_SURFACE`，也就是 SimNIBS 网格里现成的灰质表面标签。
- 它不是严格意义上的 mid-cortical central surface。
- 如果未来需要真实 cortical central layers，应单独基于更细的 central layer tags 实现，不能继续复用当前 API 语义。

---

## 4. 场数据提取策略

### 4.1 正确做法

从 `.msh` 读取场后，必须插值到表面顶点：

```python
field = mesh.field[field_name]
surface_field = field.interpolate_to_surface(surface_mesh)
cdata = surface_field.value
```

### 4.2 标量字段要求

只允许标量字段写入 MZ3。

推荐字段：

- `magnE`
- `magnJ`
- `TImax`
- `v`

为了兼容常见误用，可做别名解析：

- `E` -> `magnE`
- `J` -> `magnJ`

如果字段不存在，或字段是向量/张量，应显式报错。

---

## 5. 函数设计

### 5.1 `read_msh_surface()`

功能：

- 从带表面标签的 `.msh` 中裁剪指定三角表面

返回：

- `vertices: np.ndarray`
- `faces: np.ndarray`
- `surface_mesh: mesh_io.Msh`

### 5.2 `write_mz3()`

功能：

- 将顶点、三角面和可选顶点标量写入 `.mz3.gz`

校验：

- `vertices.shape == (Nv, 3)`
- `faces.shape == (Nf, 3)`
- `cdata.shape[0] == Nv`

### 5.3 `msh_to_mz3()`

功能：

1. 读取 `.msh`
2. 提取指定表面
3. 可选地将标量字段插值到表面
4. 写出 `.mz3.gz`

---

## 6. Demo 约定

### 6.1 `charm_to_mz3_demo.py`

输出三种表面：

- `white`
- `central`
- `pial`

仅导出几何，不写场数据。

### 6.2 `ti_simulation_to_mz3_demo.py`

输出三种表面，并写入电场强度标量：

- 默认使用 `magnE`
- 不再直接传 `E`

---

## 7. 验证点

1. `crop_mesh(tags=..., elm_type=2)` 能正确提取三角面。
2. 导出的 `faces` 为 0-based。
3. `cdata.shape[0] == len(vertices)`。
4. `magnE`、`TImax` 等标量字段可以正常显示。
5. 对 `E`、`J` 这种向量字段，若未做别名解析则应报错；若做了解析，应明确写入的是 `magnE` / `magnJ`。
