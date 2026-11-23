"""构建共同评估四面体，并按几何对齐两路载波。"""

from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, Geometry, IntArray, ti_envelope
from simnibs.mesh_tools.mesh_io import Msh, read_msh
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.mesh_element_properties import ElementTags
from simnibs.utils.region_of_interest import mask_image_to_index_mask, mni_mask_to_sub
from simnibs.utils.transformations import subject2mni_coords


def brain_rows(mesh: Msh) -> IntArray:
    """选取网格中 WM/GM 四面体的零基行号。

    Parameters
    ----------
    mesh : Msh
        参考头模或载波网格。

    Returns
    -------
    ndarray
        按网格原有顺序排列的四面体行。
    """
    return np.flatnonzero((mesh.elm.elm_type == 4) & np.isin(mesh.elm.tag1, [ElementTags.WM, ElementTags.GM]))


def prepare_geometry(
    head: Path, cortex_atlas: Path, target_paths: dict[str, Path], output: Path,
) -> list[dict[str, object]]:
    """对独立头模应用个体配准，保存固定网格、区域及质控信息。

    Parameters
    ----------
    head : Path
        包含 CHARM 完整网格和变换的 m2m 目录。
    cortex_atlas : Path
        标准化 Brainnetome 整数标签图。
    target_paths : dict
        靶键到标准化二值 ROI 路径的映射。
    output : Path
        要生成的 geometry.npz；同目录 roi/ 保存个体空间掩膜。

    Returns
    -------
    list of dict
        每靶的体积、元素数、个体重心和回到 MNI 后的侧别读数。
    """
    subject = SubjectFiles(subpath=str(head))
    mesh = read_msh(subject.fnamehead, skip_data=True)
    rows = brain_rows(mesh)
    if not len(rows):
        raise ValueError("参考网格没有灰白质四面体")
    tags = mesh.elm.tag1[rows]
    volume = mesh.elements_volumes_and_areas().value[rows]
    centroid = mesh.elements_baricenters().value[rows]
    atlas = nib.load(cortex_atlas)
    labels = np.asanyarray(atlas.dataobj)
    cortical_template = nib.Nifti1Image(((labels >= 1) & (labels <= 210)).astype(np.uint8), atlas.affine)
    cortical_image = mni_mask_to_sub(cortical_template, subject)
    cortical_mask = np.isin(rows, mask_image_to_index_mask("elm_center", cortical_image, mesh, 1)) & (tags == ElementTags.GM)
    if not np.any(cortical_mask):
        raise ValueError("个体脑网格上的 Brainnetome 皮层代理区为空")
    output.parent.mkdir(parents=True, exist_ok=True)
    roi_dir = output.parent / "roi"
    roi_dir.mkdir(exist_ok=True)
    nib.save(cortical_image, roi_dir / "cortex.nii.gz")
    masks = {}
    checks: list[dict[str, object]] = []
    for key, path in target_paths.items():
        target_image = mni_mask_to_sub(nib.load(path), subject)
        roi_path = roi_dir / f"{key}.nii.gz"
        nib.save(target_image, roi_path)
        mask = np.isin(rows, mask_image_to_index_mask("elm_center", target_image, mesh, 1))
        masks[f"target_{key}"] = mask
        center = np.average(centroid[mask], axis=0, weights=volume[mask]) if mask.any() else None
        mni_x = float(subject2mni_coords(center.reshape(1, 3), str(head))[0, 0]) if center is not None else np.nan
        checks.append(dict(target=key, roi_path=str(roi_path), elements=int(mask.sum()), roi_mm3=float(volume[mask].sum()),
                           centroid_mm=center.tolist() if center is not None else "NA", mni_centroid_x_mm=mni_x,
                           status="ok" if mni_x > 0 else "empty_target" if center is None else "side_mismatch"))
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, element_id=mesh.elm.elm_number[rows], node_mm=mesh.nodes.node_coord,
                        connectivity=mesh.elm.node_number_list[rows] - 1, volume_mm3=volume, centroid_mm=centroid,
                        tissue_tag=tags, cortex_mask=cortical_mask, **masks)
    temporary.replace(output)
    return checks


def load_geometry(path: Path) -> tuple[Geometry, IntArray, dict[str, BoolArray]]:
    """加载共同几何、组织标签及分靶掩膜。

    Parameters
    ----------
    path : Path
        prepare_geometry 生成的 geometry.npz。

    Returns
    -------
    geometry, tissues, targets : tuple
        全部数组共用固定参考元素行序，targets 以靶键索引。
    """
    with np.load(path) as data:
        geo = Geometry(data["element_id"], data["node_mm"], data["connectivity"], data["volume_mm3"], data["centroid_mm"], data["cortex_mask"])
        targets = {key.removeprefix("target_"): data[key] for key in data.files if key.startswith("target_")}
        return geo, data["tissue_tag"], targets


def _key_rows(connectivity: IntArray, tissues: IntArray) -> NDArray:
    """用四个排序节点编号和组织标签组成可检索的元素键。

    Parameters
    ----------
    connectivity, tissues : ndarray
        同序的四节点下标和组织编号。

    Returns
    -------
    ndarray
        结构化元素键，按节点及组织字典序比较。
    """
    columns = np.column_stack((np.sort(connectivity, axis=1), tissues)).astype(np.int64)
    return columns.view(np.dtype([(f"c{i}", np.int64) for i in range(columns.shape[1])])).ravel()


def align_element_values(mesh: Msh, field: FloatArray, geometry: Geometry, tissues: IntArray) -> tuple[FloatArray, BoolArray]:
    """将已有逐元素标量或向量映射到参考四面体。

    Parameters
    ----------
    mesh : Msh
        定义读数顺序的求解网格。
    field : ndarray
        原网格逐元素值，第一维对应其元素行。
    geometry : Geometry
        共同参考元素 ID、连通关系和重心。
    tissues : ndarray
        与参考元素逐行对应的组织标签。

    Returns
    -------
    aligned, exact : tuple of ndarray
        同序场及是否按节点与组织精确匹配。其余位置尝试在同组织
        的载波四面体内以参考重心取分片常量，不可匹配位置保留 NaN。
    """
    if field.shape[0] != mesh.elm.nr:
        raise ValueError("场读数与求解网格行数不符")
    rows = brain_rows(mesh)
    conn = mesh.elm.node_number_list[rows] - 1
    reference = geometry.connectivity
    nodes = mesh.nodes.node_coord
    if (conn.shape == reference.shape and np.array_equal(mesh.elm.tag1[rows], tissues)
            and np.array_equal(np.sort(conn, axis=1), np.sort(reference, axis=1))
            and np.max(np.abs(nodes[conn] - geometry.node_mm[conn]), initial=0.0) <= 1e-5):
        return np.asarray(field[rows], dtype=np.float64), np.ones(len(rows), dtype=bool)
    node_ids, sources = np.unique(conn, return_inverse=True)
    ref_ids, refs = np.unique(reference, return_inverse=True)
    dist, nearest = cKDTree(geometry.node_mm[ref_ids]).query(nodes[node_ids], distance_upper_bound=1e-5)
    matching_nodes = np.where(np.isfinite(dist), nearest, -1)
    source_keys = _key_rows(matching_nodes[sources].reshape(-1, 4), mesh.elm.tag1[rows])
    ref_keys = _key_rows(refs.reshape(-1, 4), tissues)
    order = np.argsort(source_keys)
    sorted_keys = source_keys[order]
    pos = np.searchsorted(sorted_keys, ref_keys)
    exact = pos < len(order)
    exact[exact] &= sorted_keys[pos[exact]] == ref_keys[exact]
    aligned = np.full((len(reference), *field.shape[1:]), np.nan)
    aligned[exact] = field[rows[order[pos[exact]]]]
    missing = np.flatnonzero(~exact)
    if missing.size:
        elements = mesh.find_tetrahedron_with_points(geometry.centroid_mm[missing], compute_baricentric=False)
        inside = elements > 0
        same_tissue = np.zeros(len(elements), dtype=bool)
        same_tissue[inside] = mesh.elm.tag1[elements[inside] - 1] == tissues[missing[inside]]
        aligned[missing[same_tissue]] = field[elements[same_tissue] - 1]
    return aligned, exact


def align_carrier(mesh: Msh, geometry: Geometry, tissues: IntArray) -> tuple[FloatArray, BoolArray]:
    """将载波网格的三维元素 E 向量映射到参考四面体。

    Parameters
    ----------
    mesh : Msh
        包含元素 E 向量的载波网格。
    geometry, tissues : Geometry and ndarray
        共同参考网格及其对应的组织标签。

    Returns
    -------
    aligned, exact : tuple of ndarray
        对齐的 E 向量及精确匹配标志。
    """
    field = mesh.field["E"].value
    if field.shape != (mesh.elm.nr, 3):
        raise ValueError("单频网格缺少逐元素三维向量 E")
    return align_element_values(mesh, field, geometry, tissues)


def prepare_field(mesh_a: Path, mesh_b: Path, geometry_path: Path, output: Path) -> dict[str, float]:
    """对齐两路正向载波，重算包络并保存有效掩膜。

    Parameters
    ----------
    mesh_a, mesh_b : Path
        共同 FEM 正向任务中对应 A/B 的载波网格。
    geometry_path : Path
        参考 geometry.npz。
    output : Path
        可供后续实验复用的场实体 NPZ。

    Returns
    -------
    dict
        精确匹配、同组织取样及无法匹配的体积，单位 mm³。
    """
    geometry, tissues, _ = load_geometry(geometry_path)
    e_a, exact_a = align_carrier(read_msh(mesh_a), geometry, tissues)
    e_b, exact_b = align_carrier(read_msh(mesh_b), geometry, tissues)
    envelope = ti_envelope(e_a, e_b)
    valid = np.isfinite(envelope)
    exact = exact_a & exact_b & valid
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, E_A=e_a, E_B=e_b, max_TI=envelope, valid_mask=valid, exact_mask=exact)
    temporary.replace(output)
    volume = geometry.volume_mm3
    return dict(exact_mm3=float(volume[exact].sum()), resampled_mm3=float(volume[valid & ~exact].sum()), missing_mm3=float(volume[~valid].sum()))
