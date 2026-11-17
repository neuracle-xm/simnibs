"""
Volumetric TDCS leadfield 的生成、缓存校验和按规模加载。

该模块使用 SimNIBS ``TDCSLEADFIELD`` 为固定 montage 预计算单位电流电场，
并按内存预算将 HDF5 dataset 转换为 ``float64`` 内存或磁盘映射数组供 GA 访问。
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numba
import numpy as np
from filelock import FileLock

import simnibs
from neuracle.ti_leadfield_optimization.models import (
    LeadfieldConfig,
    LeadfieldData,
)
from neuracle.utils import cond_dict_to_list
from simnibs import ElementTags, sim_struct
from simnibs.mesh_tools.mesh_io import Msh
from simnibs.simulation.pardiso import get_libmkl

logger = logging.getLogger(__name__)

LEADFIELD_DATASET = "/mesh_leadfield/leadfields/tdcs_leadfield"
LEADFIELD_MESH_PATH = "/mesh_leadfield/"
MANIFEST_FILENAME = "leadfield_manifest.json"
CACHE_SCHEMA_VERSION = 1
GENERATOR_VERSION = "neuracle-volumetric-1"
LEADFIELD_MEMORY_BUDGET_BYTES = 512 * 1024**2
LEADFIELD_CHUNK_ELEMENTS = 65536


class _CacheLeadfield(sim_struct.TDCSLEADFIELD):
    """仅缩短共享缓存输出名，保留 SimNIBS 的物理建模和序列化类型。"""

    @property
    def type(self) -> str:
        """保持 MAT 文件可由标准 SimNIBS 加载。

        Returns
        -------
        str
            标准 TDCSLEADFIELD 类型标识。
        """
        return "TDCSLEADFIELD"

    def _lf_name(self) -> str:
        """使用固定 HDF5 名，身份由父目录和 manifest 表达。

        Returns
        -------
        str
            缓存内的 leadfield 文件名。
        """
        return "leadfield.hdf5"

    def _el_name(self) -> str:
        """使用固定电极 mesh 名，避免重复头模和 montage 名称。

        Returns
        -------
        str
            缓存内的电极 mesh 文件名。
        """
        return "electrodes.msh"

    def _mesh_roi_name(self) -> str:
        """使用固定 ROI mesh 名，保留原来的输出内容。

        Returns
        -------
        str
            缓存内的 ROI mesh 文件名。
        """
        return "roi.msh"


def _check_output_paths(leadfield: sim_struct.TDCSLEADFIELD) -> None:
    """在 Windows FEM 开始前检查输出路径，避免晚于计算才写入失败。

    Parameters
    ----------
    leadfield : simnibs.simulation.sim_struct.TDCSLEADFIELD
        已确定构建目录、短输出名和日志时间戳的配置。

    Returns
    -------
    None
        路径可容纳在传统 Windows MAX_PATH 内时正常返回。

    Raises
    ------
    ValueError
        数据根目录过深，缩短输出名后仍达到 260 个 UTF-16 单元。
    """
    if os.name != "nt":
        return
    directory = Path(leadfield.pathfem).resolve()
    filenames = [
        leadfield._lf_name(),
        leadfield._el_name(),
        leadfield._mesh_roi_name(),
        "simnibs_simulation_" + leadfield.time_str + ".mat",
        "simnibs_simulation_" + leadfield.time_str + ".log",
        MANIFEST_FILENAME + ".tmp",
    ]
    for filename in filenames:
        path = directory / filename
        length = len(str(path).encode("utf-16-le")) // 2
        if length >= 260:
            raise ValueError(
                f"leadfield 输出路径过长（{length} 个 UTF-16 单元）：{path}；"
                "请将应用 DATA 数据目录迁移到更短的路径后重试"
            )


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
        仅包含文件内容 SHA-256 的字典，不受搬迁和 mtime 影响。
    """
    return {"sha256": _sha256_file(path)}


def _build_configuration(config: LeadfieldConfig) -> dict[str, Any]:
    """从实际 SimNIBS 对象解析 FEM 身份，不包含路径、ROI 或 GA 设置。

    Parameters
    ----------
    config : LeadfieldConfig
        已解析的头模和内部物理配置。

    Returns
    -------
    dict[str, Any]
        规范化有效配置；数值统一为浮点数，文件采用流式内容摘要。
    """
    leadfield = _configure_leadfield(config)
    leadfield._add_electrodes_from_cap()
    names = [electrode.name for electrode in leadfield.electrode]
    if (
        len(names) < 4
        or len(set(names)) != len(names)
        or any(not name for name in names)
    ):
        raise ValueError("montage 必须包含至少四个唯一且非空的电极名")
    if any(
        not np.all(np.isfinite(electrode.centre)) for electrode in leadfield.electrode
    ):
        raise ValueError("montage 电极坐标必须是有限数字")
    positions = np.asarray(
        [electrode.centre for electrode in leadfield.electrode], dtype=float
    )
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError("montage 存在坐标完全重合的电极，无法生成独立电流 leadfield")
    anisotropy = None
    if config.anisotropy_type != "scalar":
        if config.dti_path is None or not config.dti_path.is_file():
            raise ValueError("各向异性 FEM 必须提供实际 tensor 文件")
        anisotropy = {
            "tensor": _file_signature(config.dti_path),
            "maxratio": float(leadfield.aniso_maxratio),
            "maxcond": float(leadfield.aniso_maxcond),
            "tissues": [int(tag) for tag in leadfield.anisotropic_tissues],
        }
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "simnibs_version": simnibs.__version__,
        "mesh": _file_signature(config.mesh_path),
        "montage": _file_signature(config.montage_path),
        "electrode_names": names,
        "reference_electrode": names[0],
        "conductivity": {
            key: float(value) for key, value in config.conductivity_config.items()
        },
        "anisotropy_type": config.anisotropy_type,
        "anisotropy": anisotropy,
        "electrode": {
            "shape": leadfield.electrode[0].shape,
            "definition": leadfield.electrode[0].definition,
            "dimensions_mm": [
                float(value) for value in leadfield.electrode[0].dimensions
            ],
            "thickness_mm": [
                float(value) for value in leadfield.electrode[0].thickness
            ],
            "holes": [],
            "plug": [],
            "dimensions_sponge": None,
            "material": "saline",
            "conductivity_s_per_m": float(
                leadfield.cond[int(ElementTags.SALINE_START) - 1].value
            ),
        },
        "boundary": {
            "ground": "reference_surface_nodes_0V_Dirichlet",
            "input_type": "tag",
            "current_a": 1.0,
            "weigh_by_area": True,
        },
        "field": leadfield.field,
        "tissues": [int(tag) for tag in leadfield.tissues],
        "interpolation": leadfield.interpolation,
        "normalization": "1A",
        "solver_options": leadfield.solver_options,
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
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("manifest 顶层必须是对象")
    return payload


def _configure_leadfield(
    config: LeadfieldConfig,
) -> sim_struct.TDCSLEADFIELD:
    """创建与 TI 正向仿真电极几何一致的 volumetric leadfield 配置。

    Parameters
    ----------
    config : LeadfieldConfig
        有效 FEM 配置。

    Returns
    -------
    simnibs.simulation.sim_struct.TDCSLEADFIELD
        可直接运行的 SimNIBS leadfield 对象。
    """
    if config.anisotropy_type not in ("scalar", "dir", "vn", "mc"):
        raise ValueError("未知各向异性模式")
    if (
        not isinstance(config.n_workers, int)
        or isinstance(config.n_workers, bool)
        or config.n_workers < 1
    ):
        raise ValueError("FEM 计算线程数必须是正整数")
    physical_values = [
        *config.conductivity_config.values(),
        config.electrode_radius_mm,
        config.electrode_thickness_mm,
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for value in physical_values
    ):
        raise ValueError("电导率及电极尺寸必须是有限正数")
    leadfield = _CacheLeadfield()
    leadfield.subpath = str(config.head_model_dir)
    leadfield.fnamehead = str(config.mesh_path)
    leadfield.pathfem = str(config.leadfield_dir)
    leadfield.eeg_cap = str(config.montage_path)
    leadfield.field = "E"
    leadfield.solver_options = "pardiso"
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
    _check_output_paths(leadfield)
    return leadfield


def _run_leadfield(leadfield: sim_struct.TDCSLEADFIELD, n_threads: int) -> None:
    """以单进程复用 PARDISO 分解，并临时限制原生计算线程数。

    Parameters
    ----------
    leadfield : simnibs.simulation.sim_struct.TDCSLEADFIELD
        已配置 PARDISO 的 leadfield，保留原电极与边界条件。
    n_threads : int
        MKL 线程预算；Numba 使用该预算与初始化容量中的较小值。

    Returns
    -------
    None
        在配置目录写入 leadfield，正常或异常结束均恢复线程设置。

    Notes
    -----
    cpus=1 选择 SimNIBS 的串行电极循环，首次求解的分解结果供后续
    右端项复用。MKL 使用当前调用线程的运行时设置，避免环境变量在
    库初始化后不生效，不创建 Python FEM 子进程。
    """
    if leadfield.solver_options != "pardiso":
        raise ValueError("单进程多线程 leadfield 必须使用 PARDISO")
    if isinstance(n_threads, bool) or not isinstance(n_threads, int) or n_threads < 1:
        raise ValueError("FEM 计算线程数必须是正整数")
    if n_threads > np.iinfo(np.int32).max:
        raise ValueError("FEM 计算线程数超过 MKL 接口范围")
    library = get_libmkl()
    set_mkl_threads = library.MKL_Set_Num_Threads_Local
    set_mkl_threads.argtypes = [ctypes.c_int]
    set_mkl_threads.restype = ctypes.c_int
    previous_numba_threads = numba.get_num_threads()
    numba_threads = min(n_threads, numba.config.NUMBA_NUM_THREADS)
    previous_mkl_threads = set_mkl_threads(n_threads)
    try:
        numba.set_num_threads(numba_threads)
        logger.info(
            "leadfield FEM 使用单进程 PARDISO，MKL 线程预算=%s，Numba 线程数=%s",
            n_threads,
            numba_threads,
        )
        leadfield.run(cpus=1)
    finally:
        set_mkl_threads(previous_mkl_threads)
        numba.set_num_threads(previous_numba_threads)


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


def configuration_key(configuration: dict[str, Any]) -> str:
    """规范化配置 JSON 后计算 SHA-256，拒绝 NaN/Infinity。

    Parameters
    ----------
    configuration : dict[str, Any]
        有效 FEM 配置。

    Returns
    -------
    str
        不受字典键顺序影响的缓存键。
    """
    payload = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_leadfield(path: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    """校验完整 HDF5 数值、归一化、电极和内嵌 mesh 的元素顺序。

    Parameters
    ----------
    path : Path
        待发布或复用的 HDF5。
    configuration : dict[str, Any]
        期望的实际 FEM 配置。

    Returns
    -------
    dict[str, Any]
        文件摘要、dataset 元数据及内嵌 mesh 顺序摘要。
    """
    mesh = Msh.read_hdf5(str(path), LEADFIELD_MESH_PATH)
    mesh_digest = hashlib.sha256()
    for array in (
        mesh.nodes.node_coord,
        mesh.elm.node_number_list,
        mesh.elm.elm_type,
        mesh.elm.tag1,
        mesh.elm.tag2,
    ):
        mesh_digest.update(np.ascontiguousarray(array).tobytes())
    if mesh.elm.nr == 0 or not np.all(mesh.elm.elm_type == 4):
        raise ValueError("leadfield 必须包含非空 tetrahedron mesh")
    if not np.all(np.isin(mesh.elm.tag1, configuration["tissues"])):
        raise ValueError("leadfield 内嵌 mesh 组织不匹配")
    if not np.all(np.isfinite(mesh.nodes.node_coord)):
        raise ValueError("leadfield mesh 坐标包含非有限值")
    connectivity = mesh.elm.node_number_list
    if np.any(connectivity < 1) or np.any(connectivity > mesh.nodes.nr):
        raise ValueError("leadfield mesh 节点索引越界")
    with h5py.File(path, "r") as h5:
        dataset = h5[LEADFIELD_DATASET]
        attributes = {key: _json_value(value) for key, value in dataset.attrs.items()}
        expected = {
            "field": "E",
            "d_type": "element_data",
            "units": "V/m",
            "current": "1A",
            "interpolation": "None",
            "tissues": configuration["tissues"],
            "electrode_names": configuration["electrode_names"],
            "reference_electrode": configuration["reference_electrode"],
        }
        for key, value in expected.items():
            if attributes.get(key) != value:
                raise ValueError(f"leadfield 元数据 {key} 不匹配")
        if dataset.shape != (len(configuration["electrode_names"]) - 1, mesh.elm.nr, 3):
            raise ValueError("leadfield dataset 与电极/mesh 形状不匹配")
        if dataset.dtype.kind != "f" or dataset.dtype.itemsize not in (4, 8):
            raise ValueError("leadfield dataset 必须是 float32/float64")
        for row in range(dataset.shape[0]):
            for start in range(0, dataset.shape[1], 65536):
                if not np.all(np.isfinite(dataset[row, start : start + 65536, :])):
                    raise ValueError("leadfield 数值包含 NaN/Infinity")
        metadata = {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "attributes": attributes,
        }
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "mesh_order_sha256": mesh_digest.hexdigest(),
        "dataset": metadata,
    }


def _validate_cache(directory: Path, configuration: dict[str, Any]) -> Path:
    """核对完成 manifest 和 HDF5，任何不一致均禁止复用。

    Parameters
    ----------
    directory : Path
        已发布的指纹目录。
    configuration : dict[str, Any]
        本次 FEM 配置。

    Returns
    -------
    Path
        通过完整性检查的文件。
    """
    manifest = _read_manifest(directory / MANIFEST_FILENAME)
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("state") != "complete"
        or manifest.get("cache_key") != configuration_key(configuration)
        or manifest.get("configuration") != configuration
    ):
        raise ValueError("leadfield manifest 配置或完成状态不匹配")
    path = _find_generated_leadfield(directory)
    if manifest.get("leadfield") != validate_leadfield(path, configuration):
        raise ValueError("leadfield 内容摘要或元素元数据不匹配")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """同目录写入并替换 JSON，仅发布完整内容。

    Parameters
    ----------
    path : Path
        目标路径。
    payload : dict[str, Any]
        完整数据。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _isolate_cache(directory: Path, cache_key: str, state: str) -> None:
    """用短目录隔离残缺或无效缓存，并保留完整 key 与原目录名。

    Parameters
    ----------
    directory : pathlib.Path
        已获得对应 key 锁后确认需隔离的目录。
    cache_key : str
        完整缓存身份，不通过截断 key 缩短路径。
    state : str
        failed 或 invalid，分别表示残缺构建和无效完整缓存。

    Returns
    -------
    None
        在同一父目录重命名并写入诊断记录，不删除数据。
    """
    if state not in ("failed", "invalid"):
        raise ValueError("未知 leadfield 隔离状态")
    isolated = directory.parent / ("." + state + "-" + uuid.uuid4().hex)
    directory.rename(isolated)
    _write_json_atomic(
        isolated / "isolation.json",
        {"cache_key": cache_key, "state": state, "original_directory": directory.name},
    )
    logger.warning("隔离 leadfield 缓存，cache_key=%s，目录=%s", cache_key, isolated)


def ensure_leadfield(config: LeadfieldConfig, task_dir: Path | None = None) -> Path:
    """按内容寻址缓存，持有操作系统跨进程锁，验证完成后原子发布。

    Parameters
    ----------
    config : LeadfieldConfig
        leadfield_dir 为该头模的共享缓存根目录。
    task_dir : Path, optional
        写入本次缓存命中状态和耗时的任务目录。

    Returns
    -------
    Path
        完整发布且只读复用的 HDF5。

    Notes
    -----
    FileLock 使用 OS 锁，进程崩溃自动释放；锁文件保留但不表示仍被占用。
    持锁后隔离同键残缺目录，不删除其他键或正在构建的内容。
    """
    started = time.perf_counter()
    configuration = _build_configuration(config)
    cache_key = configuration_key(configuration)
    root = config.leadfield_dir
    directory = root / cache_key
    building = root / (".building-" + cache_key)
    leadfield = _configure_leadfield(replace(config, leadfield_dir=building))
    root.mkdir(parents=True, exist_ok=True)
    generated_seconds = 0.0
    cache_hit = False
    logger.info("等待 leadfield 缓存锁: %s", cache_key)
    with FileLock(str(root / (cache_key + ".lock"))):
        abandoned_directories = list(root.glob(".building-" + cache_key + "-*"))
        if building.exists():
            abandoned_directories.append(building)
        for abandoned in abandoned_directories:
            _isolate_cache(abandoned, cache_key, "failed")
        if directory.exists():
            try:
                hdf5_path = _validate_cache(directory, configuration)
                cache_hit = True
                logger.info("复用完整 leadfield 缓存: %s", cache_key)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                logger.warning("leadfield 缓存无效，隔离后重建: %s", exc)
                _isolate_cache(directory, cache_key, "invalid")
        if not cache_hit:
            building.mkdir()
            generation_start = time.perf_counter()
            try:
                logger.info("生成 leadfield 缓存: %s", cache_key)
                _run_leadfield(leadfield, n_threads=config.n_workers)
                generated = _find_generated_leadfield(building)
                metadata = validate_leadfield(generated, configuration)
                if _build_configuration(config) != configuration:
                    raise RuntimeError("FEM 生成期间输入文件发生变化，拒绝发布缓存")
                manifest = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "state": "complete",
                    "cache_key": cache_key,
                    "configuration": configuration,
                    "leadfield": metadata,
                }
                _write_json_atomic(building / MANIFEST_FILENAME, manifest)
                building.rename(directory)
                hdf5_path = directory / generated.name
                generated_seconds = time.perf_counter() - generation_start
            except BaseException:
                logger.exception("leadfield 生成失败，未发布完成缓存: %s", building)
                raise
    if task_dir is not None:
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            task_dir / "leadfield_reference.json",
            {
                "cache_key": cache_key,
                "manifest_version": CACHE_SCHEMA_VERSION,
                "manifest_path": str(directory / MANIFEST_FILENAME),
                "cache_hit": cache_hit,
                "generation_seconds": generated_seconds,
                "cache_seconds": time.perf_counter() - started,
            },
        )
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


def _read_leadfield_values(
    dataset: h5py.Dataset, memory_budget_bytes: int, temporary_dir: Path
) -> np.ndarray:
    """按预算分块转换电场，超出预算时返回只读磁盘映射。

    Parameters
    ----------
    dataset : h5py.Dataset
        形状为非空 (电极行数, 元素数, 3) 的 float32/float64 数据。
    memory_budget_bytes : int
        常驻数组预算字节数，0 强制使用磁盘。
    temporary_dir : Path
        磁盘映射临时文件目录。

    Returns
    -------
    numpy.ndarray
        float64 数组；磁盘映射及所有视图释放后自动删除其临时文件。

    Notes
    -----
    HDF5 直接转换到目标缓冲区，逐块校验，避免原类型和布尔全量副本。
    磁盘路径先顺序写入再只读映射，避免构建过程中积累脏映射页。
    """
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes < 0
    ):
        raise ValueError("leadfield 内存预算必须是非负整数")
    if (
        dataset.ndim != 3
        or dataset.shape[2] != 3
        or any(size == 0 for size in dataset.shape)
    ):
        raise ValueError("leadfield 形状必须为非空 (N, M, 3)")
    if dataset.dtype.kind != "f" or dataset.dtype.itemsize not in (4, 8):
        raise ValueError("leadfield dataset 必须是 float32/float64")
    required_bytes = dataset.size * np.dtype(np.float64).itemsize
    values = None
    if required_bytes <= memory_budget_bytes:
        try:
            values = np.empty(dataset.shape, dtype=np.float64)
        except MemoryError:
            logger.warning("leadfield 内存分配失败，改用磁盘映射")
    stream = None
    try:
        if values is None:
            temporary_dir.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(temporary_dir).free < required_bytes + 64 * 1024**2:
                raise OSError(
                    "leadfield 磁盘映射空间不足，需容纳完整 float64 数组及 64 MiB 余量"
                )
            stream = tempfile.TemporaryFile(
                prefix="leadfield-", suffix=".bin", dir=temporary_dir
            )
        logger.info(
            "加载 leadfield: 模式=%s, 数组字节数=%s, 内存预算=%s",
            "内存" if values is not None else "磁盘映射",
            required_bytes,
            memory_budget_bytes,
        )
        buffer = (
            np.empty(
                (1, min(dataset.shape[1], LEADFIELD_CHUNK_ELEMENTS), 3),
                dtype=np.float64,
            )
            if stream is not None else None
        )
        for row in range(dataset.shape[0]):
            for start in range(0, dataset.shape[1], LEADFIELD_CHUNK_ELEMENTS):
                stop = min(start + LEADFIELD_CHUNK_ELEMENTS, dataset.shape[1])
                block = (
                    buffer[:, :stop - start, :]
                    if stream is not None
                    else values[row:row + 1, start:stop, :]
                )
                dataset.read_direct(
                    block, source_sel=np.s_[row:row + 1, start:stop, :]
                )
                if not np.all(np.isfinite(block)):
                    raise ValueError("leadfield 包含 NaN/Infinity")
                if stream is not None:
                    stream.write(memoryview(block).cast("B"))
        if stream is not None:
            stream.flush()
            values = np.memmap(stream, dtype=np.float64, mode="r", shape=dataset.shape)
            weakref.finalize(values._mmap, stream.close)
            stream = None
        return values
    finally:
        if stream is not None:
            stream.close()


def load_leadfield(
    path: str | Path,
    *,
    memory_budget_bytes: int = LEADFIELD_MEMORY_BUDGET_BYTES,
    temporary_dir: str | Path | None = None,
) -> LeadfieldData:
    """按规模加载 ``float64`` leadfield 并校验 element-data 语义。

    Parameters
    ----------
    path : str or pathlib.Path
        SimNIBS TDCS leadfield HDF5 文件。
    memory_budget_bytes : int, optional
        内存数组预算，默认 512 MiB；超出时使用磁盘映射，0 强制磁盘映射。
    temporary_dir : str or pathlib.Path, optional
        临时文件目录，默认 neuracle/private_gitignore/leadfield。

    Returns
    -------
    LeadfieldData
        内存或只读磁盘映射 leadfield、内嵌 mesh 和电极到 row 的映射。

    Raises
    ------
    ValueError
        Dataset 不是 volumetric E element data，或形状与 mesh 不匹配时抛出。
    """
    hdf5_path = Path(path)
    scratch_dir = (
        Path(temporary_dir)
        if temporary_dir is not None
        else Path(__file__).resolve().parents[1] / "private_gitignore" / "leadfield"
    )
    logger.info("按规模加载 float64 leadfield: %s", hdf5_path)
    with h5py.File(hdf5_path, "r") as h5:
        dataset = h5[LEADFIELD_DATASET]
        attributes = {key: _json_value(value) for key, value in dataset.attrs.items()}
        values = _read_leadfield_values(dataset, memory_budget_bytes, scratch_dir)
        raw_names = dataset.attrs.get("electrode_names")
        raw_reference = dataset.attrs.get("reference_electrode")
    if attributes.get("current") != "1A":
        raise ValueError("leadfield 必须按 1 A 归一化")
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
