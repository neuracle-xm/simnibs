"""
MSH to MZ3 converter

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
    """
    规范化标量数据为 float32 类型，形状 (Nv, Ns)。

    Parameters
    ----------
    cdata : np.ndarray
        输入标量数据
    n_vertices : int
        顶点数量

    Returns
    -------
    np.ndarray
        规范化后的 float32 数组，形状 (Nv, Ns)

    Raises
    ------
    ValueError
        当标量数据长度与顶点数不一致时
    """
    cdata = np.asarray(cdata)

    if cdata.ndim == 1:
        cdata = cdata[:, np.newaxis]

    if cdata.shape[0] != n_vertices:
        logger.error("标量数据长度与顶点数不一致: %s != %s", cdata.shape[0], n_vertices)
        raise ValueError(
            f"标量数据长度与顶点数不一致: {cdata.shape[0]} != {n_vertices}"
        )

    return np.ascontiguousarray(cdata.astype(np.float32))


def _mz3_bytes(array: np.ndarray, dtype: str) -> bytes:
    """
    将数组序列化为字节串。

    Parameters
    ----------
    array : np.ndarray
        输入数组
    dtype : str
        数据类型字符串（如 "<u4", "<f4"）

    Returns
    -------
    bytes
        数组的连续字节表示

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
    将三角表面网格写入 MZ3 文件（非压缩）。

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

    Raises
    ------
    ValueError
        当 vertices 或 faces 形状不正确时
    """
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.uint32))

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        logger.error("vertices 形状错误: 期望 (Nv, 3), 实际为 %s", vertices.shape)
        raise ValueError(f"vertices 形状错误: 期望 (Nv, 3), 实际为 {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        logger.error("faces 形状错误: 期望 (Nf, 3), 实际为 %s", faces.shape)
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

    Parameters
    ----------
    msh_path : str
        MSH 文件路径
    surface_type : str
        表面类型，可选 "white"、"central"、"pial"
    mesh : mesh_io.Msh | None, optional
        已加载的 mesh 对象，如为 None 则从文件读取

    Returns
    -------
    tuple[np.ndarray, np.ndarray, mesh_io.Msh]
        (顶点坐标数组, 面索引数组, 表面 mesh 对象)

    Raises
    ------
    FileNotFoundError
        当 MSH 文件不存在时
    ValueError
        当 surface_type 不支持，或找不到对应的三角表面，
        或裁剪结果包含非三角元素时

    Notes
    -----
    - `white` 使用 `WM_TH_SURFACE (1001)`
    - `central` 使用 `GM_TH_SURFACE (1002)`
    - `pial` 使用 `CSF_TH_SURFACE (1003)`

    某些优化结果文件本身已经是纯三角表面网格，不再包含
    central/white/pial 的体表面 tag。此时直接使用整张表面。
    """
    if not os.path.exists(msh_path):
        logger.error("MSH 文件不存在: %s", msh_path)
        raise FileNotFoundError(f"MSH 文件不存在: {msh_path}")

    if surface_type not in SURFACE_TAGS:
        logger.error(
            "不支持的 surface_type: %s; 可选值: %s",
            surface_type,
            ", ".join(SURFACE_TAGS),
        )
        raise ValueError(
            f"不支持的 surface_type: {surface_type}; 可选值: {', '.join(SURFACE_TAGS)}"
        )

    if mesh is None:
        mesh = mesh_io.read_msh(msh_path)
    surface_tag = SURFACE_TAGS[surface_type]
    # SimNIBS 的 crop_mesh 在同时提供多个筛选条件时使用 OR，而不是 AND。
    # 这里 surface_tag 本身就是表面三角面标签，因此只按 tag 裁剪。
    try:
        surface_mesh = mesh.crop_mesh(tags=surface_tag)
    except ValueError as exc:
        # 某些优化结果文件本身已经是纯三角表面网格，不再包含
        # central/white/pial 的体表面 tag。此时直接使用整张表面。
        if mesh.elm.nr > 0 and np.all(mesh.elm.elm_type == 2):
            logger.info(
                "未找到表面 tag=%d，输入已是纯三角表面网格，直接使用整张网格: %s",
                surface_tag,
                msh_path,
            )
            surface_mesh = mesh
        else:
            raise exc

    if surface_mesh.elm.nr == 0:
        logger.error(
            "未找到 surface_type=%s 对应的三角表面 tag=%d",
            surface_type,
            surface_tag,
        )
        raise ValueError(
            f"未找到 surface_type={surface_type} 对应的三角表面 tag={surface_tag}"
        )

    if not np.all(surface_mesh.elm.elm_type == 2):
        unique_types = np.unique(surface_mesh.elm.elm_type)
        logger.error(
            "表面 tag=%d 裁剪结果包含非三角元素: %s",
            surface_tag,
            unique_types.tolist(),
        )
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
    """
    解析标量场名称，支持别名映射。

    Parameters
    ----------
    mesh : mesh_io.Msh
        mesh 对象
    field_name : str
        字段名称或别名

    Returns
    -------
    str
        解析后的实际字段名称

    Raises
    ------
    ValueError
        当字段不存在时

    Notes
    -----
    支持的别名：'e' -> 'magnE'，'j' -> 'magnJ'
    """
    if field_name in mesh.field:
        return field_name

    alias = SCALAR_FIELD_ALIASES.get(field_name.lower())
    if alias and alias in mesh.field:
        logger.info("将场名 '%s' 解析为标量场 '%s'", field_name, alias)
        return alias

    available_fields = ", ".join(sorted(mesh.field.keys()))
    logger.error("场数据 '%s' 不存在。可用字段: %s", field_name, available_fields or "无")
    raise ValueError(
        f"场数据 '{field_name}' 不存在。可用字段: {available_fields or '无'}"
    )


def _extract_surface_scalar_data(
    mesh: mesh_io.Msh,
    surface_mesh: mesh_io.Msh,
    field_name: str,
) -> np.ndarray:
    """
    从 mesh 中提取表面标量数据。

    Parameters
    ----------
    mesh : mesh_io.Msh
        原始 mesh 对象
    surface_mesh : mesh_io.Msh
        表面 mesh 对象
    field_name : str
        字段名称

    Returns
    -------
    np.ndarray
        插值到表面的标量数据，形状 (Nv,) 或 (Nv, Ns)

    Raises
    ------
    ValueError
        当字段不是标量场时

    Notes
    -----
    如果输入网格不包含体元素，字段将按表面网格直接转换为顶点标量。
    """
    resolved_name = _resolve_scalar_field_name(mesh, field_name)
    field = mesh.field[resolved_name]

    if field.nr_comp not in (1,):
        logger.error(
            "字段 '%s' 不是标量场，nr_comp=%s。请改用标量字段，例如 magnE、magnJ、TImax 或 v。",
            resolved_name,
            field.nr_comp,
        )
        raise ValueError(
            f"字段 '{resolved_name}' 不是标量场，nr_comp={field.nr_comp}。"
            "请改用标量字段，例如 magnE、magnJ、TImax 或 v。"
        )

    if len(mesh.elm.tetrahedra) == 0:
        logger.info(
            "输入网格不包含体元素，字段 '%s' 将按表面网格直接转换为顶点标量",
            resolved_name,
        )
        return _extract_scalar_data_from_surface_field(surface_mesh, resolved_name)

    interp = field.interpolate_to_surface(surface_mesh)
    cdata = _normalize_scalar_data(interp.value, surface_mesh.nodes.nr)
    logger.info(
        "场数据 '%s' 已插值到表面顶点: %s",
        resolved_name,
        cdata.shape,
    )
    return cdata


def _extract_scalar_data_from_surface_field(
    surface_mesh: mesh_io.Msh,
    field_name: str,
) -> np.ndarray:
    """
    从表面网格字段中提取标量数据。

    Parameters
    ----------
    surface_mesh : mesh_io.Msh
        表面 mesh 对象
    field_name : str
        字段名称

    Returns
    -------
    np.ndarray
        顶点标量数据

    Raises
    ------
    ValueError
        当表面网格中缺少指定字段时
    TypeError
        当字段类型不支持时

    Notes
    -----
    如果字段是 NodeData 类型，直接返回值；
    如果是 ElementData 类型，则由表面三角标量平均到顶点。
    """
    if field_name not in surface_mesh.field:
        available_fields = ", ".join(sorted(surface_mesh.field.keys()))
        logger.error(
            "表面网格中缺少字段 '%s'。可用字段: %s",
            field_name,
            available_fields or "无",
        )
        raise ValueError(
            f"表面网格中缺少字段 '{field_name}'。可用字段: {available_fields or '无'}"
        )

    field = surface_mesh.field[field_name]

    if isinstance(field, mesh_io.NodeData):
        cdata = _normalize_scalar_data(field.value, surface_mesh.nodes.nr)
        logger.info(
            "字段 '%s' 已直接作为表面顶点标量导出: %s",
            field_name,
            cdata.shape,
        )
        return cdata

    if isinstance(field, mesh_io.ElementData):
        values = np.asarray(field.value)
        if values.ndim > 1:
            values = np.squeeze(values, axis=-1)
        node_values = surface_mesh.elm2node_matrix().dot(values)
        cdata = _normalize_scalar_data(node_values, surface_mesh.nodes.nr)
        logger.info(
            "字段 '%s' 已由表面三角标量平均到顶点: %s",
            field_name,
            cdata.shape,
        )
        return cdata

    logger.error("不支持的字段类型: %s", type(field).__name__)
    raise TypeError(f"不支持的字段类型: {type(field)!r}")


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
    surface_type : str, optional
        表面类型，可选 "white"、"central"、"pial"，默认为 "central"
    field_name : str | None, optional
        要写入的标量场名称，例如 "magnE - pair 1"、"TImax" 等

    Returns
    -------
    str
        生成的 MZ3 文件路径

    Raises
    ------
    FileNotFoundError
        当 MSH 文件不存在时
    ValueError
        当 surface_type 不支持时

    Notes
    -----
    MZ3 格式可供 surf-ice/BrainVisa 等软件读取。
    输出文件名格式为 {subject_id}_{surface_type}.mz3
    """
    if not os.path.exists(msh_path):
        logger.error("MSH 文件不存在: %s", msh_path)
        raise FileNotFoundError(f"MSH 文件不存在: {msh_path}")

    if surface_type not in SURFACE_TAGS:
        logger.error(
            "不支持的 surface_type: %s; 可选值: %s",
            surface_type,
            ", ".join(SURFACE_TAGS),
        )
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
