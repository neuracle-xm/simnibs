"""把共同正向场固定，复核个体 ROI 的变换、求解网格与位移敏感性。"""

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import map_coordinates

from neuracle.atlas.loader import get_atlas_spec
from neuracle.atlas.registry import SIMNIBS_MNI_TEMPLATE
from neuracle.demo.ti_depth.exp02 import PILOT_TARGETS
from neuracle.ti_depth_study.diagnostics import FieldCase, ResultRows, run_diagnostic
from neuracle.ti_depth_study.fields import align_element_values, brain_rows, load_geometry
from neuracle.ti_depth_study.figures import plot_roi_slices
from neuracle.ti_depth_study.metrics import BoolArray, FloatArray, measure_field
from neuracle.ti_depth_study.records import artifact, digest, read_csv, write_csv
from neuracle.ti_leadfield_optimization.leadfield import LEADFIELD_MESH_PATH
from simnibs.mesh_tools.mesh_io import Msh, read_msh
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.region_of_interest import mask_image_to_index_mask
from simnibs.utils.transformations import subject2mni_coords, warp_volume


SHIFTS = ((0., 0., 0.), *(tuple(float(direction * mm) if axis == index else 0.0 for axis in range(3))
                           for mm in (1, 2) for index in range(3) for direction in (-1, 1)))


def sample_shifted(mask: nib.Nifti1Image, centroids: FloatArray, offset: tuple[float, float, float]) -> BoolArray:
    """在相同参考重心按最近邻取平移后的离散 ROI。

    Parameters
    ----------
    mask : Nifti1Image
        已在个体空间的二值标签图。
    centroids : ndarray
        共同网格中心的个体 RAS 坐标。
    offset : tuple of float
        掩膜沿物理世界坐标平移的 mm 向量。

    Returns
    -------
    ndarray of bool
        当前中心在平移后前景内的标志，出界填背景。
    """
    voxel = nib.affines.apply_affine(np.linalg.inv(mask.affine), centroids - np.asarray(offset))
    labels = np.asanyarray(mask.dataobj)
    return map_coordinates(labels, voxel.T, order=0, mode="constant", cval=0, prefilter=False) == 1


def weighted_dice(first: BoolArray, second: BoolArray, volume: FloatArray) -> float:
    """计算同一组织参考网格上两份掩膜的体积 Dice。

    Parameters
    ----------
    first, second, volume : ndarray
        同序的两个布尔掩膜及参考单元体积。

    Returns
    -------
    float
        共有体积的二倍除以两个体积之和。两个区域均空时为 NaN。
    """
    total = volume[first].sum() + volume[second].sum()
    return float(2 * volume[first & second].sum() / total) if total else np.nan


def solver_geometry(case: FieldCase, subject: SubjectFiles, image: nib.Nifti1Image) -> dict[str, Any]:
    """在每条路线实际求解网格中核对目标的物理取样与共同域对应。

    Parameters
    ----------
    case : FieldCase
        固定场及其原生路由和任务目录。
    subject : SubjectFiles
        对应的独立个体头模。
    image : Nifti1Image
        求解器实际参考体素网格上的个体 ROI。

    Returns
    -------
    dict
        求解网格 ROI 体积、元素数、体积分布、质心与共同域 Dice。
    """
    row = case.native
    if row["method"] == "leadfield_based":
        task = case.study / "simulations" / row["inverse_task"]
        reference = json.loads((task / "leadfield_reference.json").read_text(encoding="utf-8"))
        manifest_path = Path(reference["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        solve_path = manifest_path.parent / manifest["leadfield"]["filename"]
        mesh = Msh.read_hdf5(str(solve_path), LEADFIELD_MESH_PATH)
        solve_hash = manifest["leadfield"]["sha256"]
    else:
        solve_path = Path(subject.fnamehead)
        mesh = read_msh(str(solve_path), skip_data=True)
        solve_hash = digest(solve_path)
    brain = brain_rows(mesh)
    marked = mask_image_to_index_mask("elm_center", image, mesh, 1)
    roi = np.intersect1d(brain, marked)
    if not roi.size:
        raise ValueError("实际求解网格交集没有目标四面体")
    volume = mesh.elements_volumes_and_areas().value[roi]
    barycenters = mesh.elements_baricenters().value[roi]
    center = np.average(barycenters, axis=0, weights=volume)
    _, tags, _ = load_geometry(case.geometry_path)
    aligned, exact = align_element_values(mesh, np.isin(np.arange(mesh.elm.nr), roi).astype(float), case.geometry, tags)
    finite = np.isfinite(aligned)
    missing_volume = float(case.geometry.volume_mm3[~finite].sum())
    dice = weighted_dice(aligned == 1, case.target, case.geometry.volume_mm3) if missing_volume == 0 else "NA"
    return dict(solver_path=str(solve_path), solver_sha256=solve_hash, solver_roi_n=int(roi.size), solver_roi_mm3=float(volume.sum()),
                solver_roi_median_tet_mm3=float(np.median(volume)), solver_roi_center_mm=center.tolist(),
                solver_roi_center_MNI_mm=subject2mni_coords(center.reshape(1, 3), subject.subpath)[0].tolist(),
                solver_to_reference_dice=dice, missing_mm3=missing_volume, exact_mm3=float(case.geometry.volume_mm3[exact].sum()))


def evaluate(case: FieldCase, out: Path, artifacts: list[dict[str, str]]) -> ResultRows:
    """登记模板与个体掩膜链，并在冻结原生场上复评 13 个位置。

    Parameters
    ----------
    case, out, artifacts : FieldCase, Path and list of dict
        已验收的共同正向病例、运行位置和散列清单。

    Returns
    -------
    rows, pairs : tuple of lists
        固定场的位移读数和相对原位的测量差值。
    """
    row = case.native
    subject = SubjectFiles(subpath=str(case.study / "head_models" / f"m2m_{row['subject']}"))
    registered = next(item for item in read_csv(case.study / "results/02/run-001/targets.csv") if item["target"] == row["target"])
    spec = get_atlas_spec(registered["atlas_name"])
    source_mask = Path(registered["mask_path"])
    raw_atlas = Path(spec["raw_atlas"])
    template = Path(spec["space_template"])
    image = nib.load(case.geometry_path.parent / "roi" / f"{row['target']}.nii.gz")
    folder = out / "masks" / row["subject"]
    folder.mkdir(parents=True, exist_ok=True)
    native_path = folder / f"{row['target']}_T1.nii.gz"
    if not native_path.is_file():
        warp_volume(str(source_mask), subject.subpath, str(native_path), transformation_direction="mni2subject",
                    transformation_type="nonl", reference=subject.reference_volume, order=0, binary=True)
    native = nib.squeeze_image(nib.load(native_path))
    tissues = nib.squeeze_image(nib.load(subject.final_labels))
    if len(native.shape) != 3 or len(tissues.shape) != 3:
        raise ValueError("原生 ROI 和组织标签需要三维体素数据")
    mask = np.asanyarray(native.dataobj) == 1
    if not mask.any():
        raise ValueError("原生 T1 网格上的 ROI 为空")
    if tissues.shape != native.shape or not np.array_equal(tissues.affine, native.affine):
        tissues = resample_from_to(tissues, native, order=0)
    brain = np.isin(np.asanyarray(tissues.dataobj), [1, 2])
    part = mask & brain
    if not part.any():
        raise ValueError("原生 T1 ROI 与 WM/GM 的交集为空")
    voxel_mm3 = float(abs(np.linalg.det(native.affine[:3, :3])))
    native_center = nib.affines.apply_affine(native.affine, np.argwhere(part).mean(axis=0))
    group = dict(case_id=row["case_id"], subject=row["subject"], target=row["target"], method=row["method"])
    chain = group | solver_geometry(case, subject, image) | dict(label=registered["label"], foreground=1, raw_atlas=str(raw_atlas),
                     raw_atlas_sha256=digest(raw_atlas), raw_template=str(template), raw_template_sha256=digest(template),
                     standard_template_sha256=digest(SIMNIBS_MNI_TEMPLATE), roi_sha256=digest(source_mask),
                     reference_T1_sha256=digest(Path(subject.reference_volume)), native_roi_sha256=digest(native_path),
                     template_correspondence="pending_anatomical_review", native_axes="".join(nib.aff2axcodes(native.affine)),
                     native_roi_mm3=float(mask.sum() * voxel_mm3), native_brain_intersection_mm3=float(part.sum() * voxel_mm3),
                     native_excluded_mm3=float((mask & ~brain).sum() * voxel_mm3), native_center_mm=native_center.tolist(),
                     mni_to_subject_sha256=digest(Path(subject.mni2conf_nonl)), subject_to_mni_sha256=digest(Path(subject.conf2mni_nonl)))
    chain_path = out / "mask_chain.csv"
    previous = read_csv(chain_path) if chain_path.is_file() else []
    write_csv(chain_path, [*(item for item in previous if item["case_id"] != row["case_id"]), chain])
    geo = case.geometry
    vol = geo.volume_mm3
    masks = {f"shift{index:02}": sample_shifted(image, geo.centroid_mm, vector) for index, vector in enumerate(SHIFTS)}
    mask_path = folder / f"{row['target']}_tet_masks.npz"
    if not mask_path.is_file():
        np.savez_compressed(mask_path, offsets_mm=np.asarray(SHIFTS), element_id=geo.element_id, **masks)
    figure = out / "figures/roi_overlay.png"
    if not figure.is_file():
        plot_roi_slices(Path(subject.reference_volume), native_path, figure, {})
    artifacts.extend(artifact(case.study, row["case_id"], role, path) for role, path in (
        ("native_ROI", native_path), ("mesh_masks", mask_path), ("ROI_overlay", figure), ("use_atlas_mask", source_mask)))
    results = []
    for (key, target), vector in zip(masks.items(), SHIFTS):
        dx, dy, dz = vector
        center = np.average(geo.centroid_mm[target], axis=0, weights=vol[target]) if target.any() else None
        error = float(np.linalg.norm(center - native_center)) if center is not None else "NA"
        measured = measure_field(geo, target, case.e_a, case.e_b, case.valid)
        results.append(measured | dict(field_id=case.field_path.stem, mesh_source="common_reference_WM_GM", mask_id=key,
                         dx_mm=dx, dy_mm=dy, dz_mm=dz, volume_mm3=float(vol[target].sum()), n_elements=int(target.sum()),
                         centroid_error_mm=error, Dice=weighted_dice(case.target, target, vol), Dice_reference="03_nominal_T",
                         mean_target_vpm=float(np.average(case.field[target], weights=vol[target])) if target.any() else "NA"))
    nominal = results[0]
    pairs = [dict(case_id=f"{row['case_id']}:{item['mask_id']}", reference_case_id=f"{row['case_id']}:shift00", arm=item["mask_id"],
                  reference_arm="shift00", field_source="common_forward", delta_D_hot_mm=item["D_hot_mm"]-nominal["D_hot_mm"],
                  delta_Cov010_pp=100*(item["Cov010"]-nominal["Cov010"])) for item in results[1:] if item["status"] == nominal["status"] == "ok"]
    return results, pairs


def main() -> None:
    """从仓库根运行 21 的固定场及个体 ROI 位移对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--baseline-id", default="run-001")
    parser.add_argument("--stage", choices=("pilot", "development"), default="pilot")
    args = parser.parse_args()
    study = args.study_root.resolve()
    targets = ("stn295",) if args.stage == "pilot" else PILOT_TARGETS
    columns = ("field_id", "mesh_source", "mask_id", "dx_mm", "dy_mm", "dz_mm", "volume_mm3", "n_elements", "centroid_error_mm", "Dice", "Cov010", "D_hot_mm")
    run_diagnostic(study, "21", args.run_id, args.baseline_id, args.stage, targets, evaluate, "roi_shift.csv", columns,
                   {"offsets_mm": SHIFTS, "interpolation": "nearest", "nominal_source": "03_solver_grid_ROI"})
    out = study / "results/21" / args.run_id
    chain = out / "mask_chain.csv"
    if chain.is_file():
        path = out / "artifacts.csv"
        write_csv(path, [*read_csv(path), artifact(study, "run", "mask_chain", chain)])


if __name__ == "__main__":
    main()
