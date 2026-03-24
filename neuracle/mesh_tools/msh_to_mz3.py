"""
MSH to MZ3 converter.

将 SimNIBS 的带表面标签的 `.msh` 网格导出为 surf-ice/BrainVisa 可读的 `.mz3`。
几何提取基于 SimNIBS 已有的三角表面 tag，场数据则插值到表面顶点。
"""

import logging
import os
import struct
from typing import Optional

import numpy as np

from simnibs import mesh_io
from simnibs.utils.mesh_element_properties import ElementTags

logger = logging.getLogger(__name__)

SURFACE_TAGS = {
    "white": int(ElementTags.WM_TH_SURFACE),
    "central": int(ElementTags.GM_TH_SURFACE),
    "pial": int(ElementTags.CSF_TH_SURFACE),
}

SCALAR_FIELD_ALIASES = {
    "e": "magnE",
    "j": "magnJ",
}


def _normalize_scalar_data(cdata: np.ndarray, n_vertices: int) -> np.ndarray:
    """Normalize scalar data to float32 with shape (Nv, Ns)."""
    cdata = np.asarray(cdata)

    if cdata.ndim == 1:
        cdata = cdata[:, np.newaxis]

    if cdata.shape[0] != n_vertices:
        raise ValueError(
            f"标量数据长度与顶点数不一致: {cdata.shape[0]} != {n_vertices}"
        )

    return np.ascontiguousarray(cdata.astype(np.float32))


def _mz3_bytes(array: np.ndarray, dtype: str) -> bytes:
    """Serialize array row-by-row.

    Notes
    -----
    NumPy 默认是 row-major，直接写 contiguous bytes 即可得到
    `(N, 3)` 数组按行连续的二进制布局。
    """
    arr = np.asarray(array, dtype=dtype)
    return np.ascontiguousarray(arr).tobytes()


def write_mz3(
    vertices: np.ndarray,
    faces: np.ndarray,
    mz3_path: str,
    cdata: Optional[np.ndarray] = None,
) -> None:
    """
    将三角表面网格写入 MZ3 文件 (非压缩)。

    Parameters
    ----------
    vertices : np.ndarray
        顶点坐标, shape `(Nv, 3)`, `float32`
    faces : np.ndarray
        面索引, shape `(Nf, 3)`, `uint32`, 0-indexed
    mz3_path : str
        输出路径
    cdata : np.ndarray | None, optional
        顶点标量数据, shape `(Nv,)` 或 `(Nv, Ns)`
    """
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.uint32))

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices 形状错误: 期望 (Nv, 3), 实际为 {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces 形状错误: 期望 (Nf, 3), 实际为 {faces.shape}")

    if cdata is not None:
        cdata = _normalize_scalar_data(cdata, len(vertices))

    attr = 0
    if len(faces) > 0:
        attr |= 1
    if len(vertices) > 0:
        attr |= 2
    if cdata is not None:
        attr |= 8

    output_dir = os.path.dirname(mz3_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(mz3_path, "wb") as f:
        f.write(struct.pack("<HH", 23117, attr))
        f.write(struct.pack("<I", len(faces)))
        f.write(struct.pack("<I", len(vertices)))
        f.write(struct.pack("<I", 0))

        if len(faces) > 0:
            f.write(_mz3_bytes(faces, "<u4"))
        if len(vertices) > 0:
            f.write(_mz3_bytes(vertices, "<f4"))
        if cdata is not None:
            f.write(_mz3_bytes(cdata, "<f4"))

    logger.info("MZ3 文件已写入: %s", mz3_path)


def read_msh_surface(
    msh_path: str,
    surface_type: str,
    mesh: Optional[mesh_io.Msh] = None,
) -> tuple[np.ndarray, np.ndarray, mesh_io.Msh]:
    """
    从带表面标签的 MSH 中提取三角表面。

    Notes
    -----
    - `white` 使用 `WM_TH_SURFACE (1001)`
    - `central` 使用 `GM_TH_SURFACE (1002)`
    - `pial` 使用 `CSF_TH_SURFACE (1003)`
    """
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"MSH 文件不存在: {msh_path}")

    if surface_type not in SURFACE_TAGS:
        raise ValueError(
            f"不支持的 surface_type: {surface_type}; 可选值: {', '.join(SURFACE_TAGS)}"
        )

    if mesh is None:
        mesh = mesh_io.read_msh(msh_path)
    surface_tag = SURFACE_TAGS[surface_type]
    # SimNIBS 的 crop_mesh 在同时提供多个筛选条件时使用 OR，而不是 AND。
    # 这里 surface_tag 本身就是表面三角面标签，因此只按 tag 裁剪。
    surface_mesh = mesh.crop_mesh(tags=surface_tag)

    if surface_mesh.elm.nr == 0:
        raise ValueError(
            f"未找到 surface_type={surface_type} 对应的三角表面 tag={surface_tag}"
        )

    if not np.all(surface_mesh.elm.elm_type == 2):
        unique_types = np.unique(surface_mesh.elm.elm_type)
        raise ValueError(
            f"表面 tag={surface_tag} 裁剪结果包含非三角元素: {unique_types.tolist()}"
        )

    vertices = surface_mesh.nodes.node_coord.astype(np.float32, copy=False)
    faces = (surface_mesh.elm.node_number_list[:, :3] - 1).astype(np.uint32, copy=False)

    logger.info(
        "提取 %s 表面(tag=%d): %d 顶点, %d 面",
        surface_type,
        surface_tag,
        len(vertices),
        len(faces),
    )

    return vertices, faces, surface_mesh


def _resolve_scalar_field_name(mesh: mesh_io.Msh, field_name: str) -> str:
    if field_name in mesh.field:
        return field_name

    alias = SCALAR_FIELD_ALIASES.get(field_name.lower())
    if alias and alias in mesh.field:
        logger.info("将场名 '%s' 解析为标量场 '%s'", field_name, alias)
        return alias

    available_fields = ", ".join(sorted(mesh.field.keys()))
    raise ValueError(
        f"场数据 '{field_name}' 不存在。可用字段: {available_fields or '无'}"
    )


def _extract_surface_scalar_data(
    mesh: mesh_io.Msh,
    surface_mesh: mesh_io.Msh,
    field_name: str,
) -> np.ndarray:
    resolved_name = _resolve_scalar_field_name(mesh, field_name)
    field = mesh.field[resolved_name]

    if field.nr_comp not in (1,):
        raise ValueError(
            f"字段 '{resolved_name}' 不是标量场，nr_comp={field.nr_comp}。"
            "请改用标量字段，例如 magnE、magnJ、TImax 或 v。"
        )

    interp = field.interpolate_to_surface(surface_mesh)
    cdata = _normalize_scalar_data(interp.value, surface_mesh.nodes.nr)
    logger.info(
        "场数据 '%s' 已插值到表面顶点: %s",
        resolved_name,
        cdata.shape,
    )
    return cdata


def msh_to_mz3(
    msh_path: str,
    output_dir: str,
    surface_type: str = "central",
    field_name: Optional[str] = None,
) -> str:
    """
    将带表面标签的 MSH 网格转换为 MZ3 文件。

    Parameters
    ----------
    msh_path : str
        输入 MSH 文件路径
    output_dir : str
        输出目录
    surface_type : str
        表面类型,可选: `"white"`, `"central"`, `"pial"`
    field_name : str | None
        要写入的标量场名称,例如 "magnE - pair 1"、"TImax" 等。

    Returns
    -------
    str
        生成的 MZ3 文件路径
    """
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"MSH 文件不存在: {msh_path}")

    if surface_type not in SURFACE_TAGS:
        raise ValueError(
            f"不支持的 surface_type: {surface_type}; 可选值: {', '.join(SURFACE_TAGS)}"
        )

    logger.info("开始转换: %s -> %s", msh_path, output_dir)
    logger.info("表面类型: %s", surface_type)

    mesh = mesh_io.read_msh(msh_path)
    vertices, faces, surface_mesh = read_msh_surface(msh_path, surface_type, mesh=mesh)

    cdata = None
    if field_name is not None:
        cdata = _extract_surface_scalar_data(mesh, surface_mesh, field_name)

    subject_id = os.path.basename(msh_path).replace(".msh", "")
    output_path = os.path.join(output_dir, f"{subject_id}_{surface_type}.mz3")
    write_mz3(vertices, faces, output_path, cdata)

    logger.info("转换完成: %s", output_path)
    return output_path
