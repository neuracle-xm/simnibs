"""
Volumetric TDCS leadfield 的生成、缓存校验和全量加载。

该模块使用 SimNIBS ``TDCSLEADFIELD`` 为固定 montage 预计算单位电流电场，
并将 HDF5 dataset 一次性读入 ``float64`` 内存数组供 GA 重复访问。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import simnibs
from neuracle.ti_leadfield_optimization.models import (
    LeadfieldData,
    LeadfieldGADemoConfig,
)
from neuracle.utils import cond_dict_to_list
from simnibs import ElementTags, sim_struct
from simnibs.mesh_tools.mesh_io import Msh

logger = logging.getLogger(__name__)

LEADFIELD_DATASET = "/mesh_leadfield/leadfields/tdcs_leadfield"
LEADFIELD_MESH_PATH = "/mesh_leadfield/"
MANIFEST_FILENAME = "leadfield_manifest.json"


def _sha256_file(path: Path) -> str:
    """计算文件 SHA-256，用于确保 leadfield 缓存与输入完全一致。

    Parameters
    ----------
    path : pathlib.Path
        需要计算摘要的文件。

    Returns
    -------
    str
        小写十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    """生成参与 FEM 配置的输入文件签名。

    Parameters
    ----------
    path : pathlib.Path
        输入文件路径。

    Returns
    -------
    dict[str, Any]
        包含绝对路径、大小、修改时间和 SHA-256 的字典。
    """
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _build_configuration(config: LeadfieldGADemoConfig) -> dict[str, Any]:
    """构建决定 leadfield 数值的完整配置摘要。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        Demo 中的头模、montage、电极和电导率配置。

    Returns
    -------
    dict[str, Any]
        可序列化且可与 manifest 直接比较的配置。
    """
    dti_signature = None
    if config.dti_path is not None:
        dti_signature = _file_signature(config.dti_path)
    return {
        "mesh": _file_signature(config.mesh_path),
        "montage": _file_signature(config.montage_path),
        "dti": dti_signature,
        "conductivity": config.conductivity_config,
        "anisotropy_type": config.anisotropy_type,
        "electrode": {
            "shape": "ellipse",
            "radius_mm": config.electrode_radius_mm,
            "thickness_mm": config.electrode_thickness_mm,
        },
        "field": "E",
        "interpolation": None,
        "tissues": [int(ElementTags.WM), int(ElementTags.GM)],
        "simnibs_version": simnibs.__version__,
    }


def _json_value(value: Any) -> Any:
    """将 HDF5 属性转换为 JSON 可序列化的 Python 值。

    Parameters
    ----------
    value : Any
        HDF5 属性值。

    Returns
    -------
    Any
        字符串、标量或递归转换后的列表。
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _read_hdf5_metadata(path: Path) -> dict[str, Any]:
    """读取 leadfield dataset 形状、类型和 SimNIBS 属性。

    Parameters
    ----------
    path : pathlib.Path
        Leadfield HDF5 文件。

    Returns
    -------
    dict[str, Any]
        Dataset 元数据。
    """
    with h5py.File(path, "r") as h5:
        dataset = h5[LEADFIELD_DATASET]
        attributes = {key: _json_value(value) for key, value in dataset.attrs.items()}
        return {
            "path": str(path.resolve()),
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "attributes": attributes,
        }


def _write_manifest(path: Path, configuration: dict[str, Any], hdf5_path: Path) -> None:
    """写入 leadfield 缓存 manifest。

    Parameters
    ----------
    path : pathlib.Path
        Manifest 文件路径。
    configuration : dict[str, Any]
        生成 leadfield 时的配置摘要。
    hdf5_path : pathlib.Path
        生成的 leadfield HDF5 文件。

    Returns
    -------
    None
        函数只写入 JSON 文件。
    """
    payload = {
        "configuration": configuration,
        "leadfield": _read_hdf5_metadata(hdf5_path),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _read_manifest(path: Path) -> dict[str, Any]:
    """读取已有 leadfield manifest 供缓存命中校验。

    Parameters
    ----------
    path : pathlib.Path
        Manifest JSON 文件。

    Returns
    -------
    dict[str, Any]
        Manifest 内容。
    """
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _configure_leadfield(
    config: LeadfieldGADemoConfig,
) -> sim_struct.TDCSLEADFIELD:
    """创建与 TI 正向仿真电极几何一致的 volumetric leadfield 配置。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        Demo 配置。

    Returns
    -------
    simnibs.simulation.sim_struct.TDCSLEADFIELD
        可直接运行的 SimNIBS leadfield 对象。
    """
    leadfield = sim_struct.TDCSLEADFIELD()
    leadfield.subpath = str(config.head_model_dir)
    leadfield.fnamehead = str(config.mesh_path)
    leadfield.pathfem = str(config.leadfield_dir)
    leadfield.eeg_cap = str(config.montage_path)
    leadfield.field = "E"
    leadfield.interpolation = None
    leadfield.tissues = [ElementTags.WM, ElementTags.GM]
    leadfield.anisotropy_type = config.anisotropy_type
    if config.dti_path is not None:
        leadfield.fname_tensor = str(config.dti_path)
    conductivity_values = cond_dict_to_list(config.conductivity_config)
    if len(conductivity_values) > len(leadfield.cond):
        raise ValueError("自定义电导率数量超过 SimNIBS 支持的组织数量")
    for index, conductivity in enumerate(conductivity_values):
        leadfield.cond[index].value = conductivity
    leadfield.electrode.shape = "ellipse"
    leadfield.electrode.dimensions = [
        2 * config.electrode_radius_mm,
        2 * config.electrode_radius_mm,
    ]
    leadfield.electrode.thickness = [config.electrode_thickness_mm]
    return leadfield


def _find_generated_leadfield(directory: Path) -> Path:
    """在专用缓存目录中定位唯一的 leadfield HDF5。

    Parameters
    ----------
    directory : pathlib.Path
        Leadfield 输出目录。

    Returns
    -------
    pathlib.Path
        唯一的 HDF5 文件。

    Raises
    ------
    RuntimeError
        未找到或找到多个 HDF5 时抛出。
    """
    candidates = sorted(directory.glob("*leadfield*.hdf5"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"leadfield HDF5 数量异常: 期望 1，实际 {len(candidates)}，目录={directory}"
        )
    return candidates[0]


def ensure_leadfield(config: LeadfieldGADemoConfig) -> Path:
    """校验并复用缓存，或首次生成 volumetric TDCS leadfield。

    Parameters
    ----------
    config : LeadfieldGADemoConfig
        头模、montage、组织和电极配置。

    Returns
    -------
    pathlib.Path
        通过 manifest 校验的 leadfield HDF5 路径。

    Raises
    ------
    RuntimeError
        缓存不完整或配置已变更时抛出，避免覆盖旧结果。
    """
    configuration = _build_configuration(config)
    manifest_path = config.leadfield_dir / MANIFEST_FILENAME
    hdf5_candidates = sorted(config.leadfield_dir.glob("*leadfield*.hdf5"))
    if manifest_path.exists() and len(hdf5_candidates) == 1:
        manifest = _read_manifest(manifest_path)
        if manifest.get("configuration") != configuration:
            raise RuntimeError(
                f"leadfield 缓存配置已变更，请更换缓存目录: {config.leadfield_dir}"
            )
        logger.info("复用已校验的 leadfield: %s", hdf5_candidates[0])
        return hdf5_candidates[0]
    if manifest_path.exists() or hdf5_candidates:
        raise RuntimeError(
            f"leadfield 缓存不完整，请更换缓存目录: {config.leadfield_dir}"
        )
    config.leadfield_dir.mkdir(parents=True, exist_ok=True)
    logger.info("首次生成 volumetric TDCS leadfield: %s", config.leadfield_dir)
    leadfield = _configure_leadfield(config)
    leadfield.run(cpus=config.n_workers)
    hdf5_path = _find_generated_leadfield(config.leadfield_dir)
    _write_manifest(manifest_path, configuration, hdf5_path)
    logger.info("leadfield 生成完成: %s", hdf5_path)
    return hdf5_path


def _decode_name(value: Any) -> str:
    """将 HDF5 中的电极名称统一转为 Python 字符串。

    Parameters
    ----------
    value : Any
        HDF5 属性中的 bytes 或字符串。

    Returns
    -------
    str
        UTF-8 电极名称。
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_leadfield(path: str | Path) -> LeadfieldData:
    """一次性加载完整 ``float64`` leadfield 并校验 element-data 语义。

    Parameters
    ----------
    path : str or pathlib.Path
        SimNIBS TDCS leadfield HDF5 文件。

    Returns
    -------
    LeadfieldData
        常驻内存的 leadfield、内嵌 mesh 和电极到 row 的映射。

    Raises
    ------
    ValueError
        Dataset 不是 volumetric E element data，或形状与 mesh 不匹配时抛出。
    """
    hdf5_path = Path(path)
    logger.info("一次性加载完整 float64 leadfield: %s", hdf5_path)
    with h5py.File(hdf5_path, "r") as h5:
        dataset = h5[LEADFIELD_DATASET]
        attributes = {key: _json_value(value) for key, value in dataset.attrs.items()}
        values = np.asarray(dataset[...], dtype=np.float64)
        raw_names = dataset.attrs.get("electrode_names")
        raw_reference = dataset.attrs.get("reference_electrode")
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"leadfield 形状应为 (N, M, 3)，实际 {values.shape}")
    if attributes.get("field") != "E":
        raise ValueError(f"leadfield field 应为 E，实际 {attributes.get('field')}")
    if attributes.get("d_type") != "element_data":
        raise ValueError(
            f"leadfield d_type 应为 element_data，实际 {attributes.get('d_type')}"
        )
    if attributes.get("units") != "V/m":
        raise ValueError(f"leadfield 单位应为 V/m，实际 {attributes.get('units')}")
    if raw_names is None or raw_reference is None:
        raise ValueError("leadfield 缺少 electrode_names 或 reference_electrode 属性")
    electrode_names = tuple(_decode_name(name) for name in raw_names)
    reference_name = _decode_name(raw_reference)
    if len(set(electrode_names)) != len(electrode_names):
        raise ValueError("leadfield 电极名称必须唯一")
    if reference_name not in electrode_names:
        raise ValueError(f"参考电极不在 electrode_names 中: {reference_name}")
    non_reference_names = [name for name in electrode_names if name != reference_name]
    if values.shape[0] != len(non_reference_names):
        raise ValueError(
            f"leadfield row 数与非参考电极数不一致: {values.shape[0]} != {len(non_reference_names)}"
        )
    electrode_rows = {name: index for index, name in enumerate(non_reference_names)}
    electrode_rows[reference_name] = None
    mesh = Msh.read_hdf5(str(hdf5_path), LEADFIELD_MESH_PATH)
    if values.shape[1] != mesh.elm.nr:
        raise ValueError(
            f"leadfield element 数与 mesh 不一致: {values.shape[1]} != {mesh.elm.nr}"
        )
    if len(electrode_names) < 4:
        raise ValueError("固定 montage 至少需要四个名称唯一的电极")
    logger.info(
        "leadfield 加载完成: shape=%s, dtype=%s, electrodes=%s",
        values.shape,
        values.dtype,
        len(electrode_names),
    )
    return LeadfieldData(
        values=values,
        mesh=mesh,
        electrode_names=electrode_names,
        electrode_rows=electrode_rows,
        path=hdf5_path,
        attributes=attributes,
    )
