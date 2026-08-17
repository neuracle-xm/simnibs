"""
验证 TI focality 优化结果，并生成 ROI / Non-ROI 可视化文件。

脚本同时使用最终 head mesh 做阈值统计，并使用导出的 NIfTI 生成区域标签
和正交切面图。默认优先验证映射到 EEG net 前的优化电极仿真结果。
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from nibabel.processing import resample_from_to
from numpy.typing import NDArray

from simnibs import mesh_io

matplotlib.use("Agg")
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns
    -------
    argparse.Namespace
        包含结果目录、阈值和可选参考影像的参数。
    """
    parser = argparse.ArgumentParser(
        description="验证 TI focality 结果，并生成 ROI / Non-ROI 可视化文件。"
    )
    parser.add_argument("result_dir", type=Path, help="TI focality 优化输出目录")
    parser.add_argument(
        "--mesh",
        type=Path,
        help="待验证的 head mesh；默认优先使用映射电极前的 mesh",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="可视化背景 T1 NIfTI；不提供时使用黑色背景",
    )
    parser.add_argument(
        "--non-roi-threshold",
        type=float,
        default=0.1,
        help="Non-ROI 上限，单位 V/m，默认 0.1",
    )
    parser.add_argument(
        "--roi-threshold",
        type=float,
        default=0.2,
        help="ROI 下限，单位 V/m，默认 0.2",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="NIfTI mask 二值化阈值，默认 0.5",
    )
    return parser.parse_args()


def find_result_mesh(result_dir: Path, mesh_path: Path | None) -> Path:
    """
    查找需要验证的最终仿真 head mesh。

    Parameters
    ----------
    result_dir : pathlib.Path
        TI focality 优化输出目录。
    mesh_path : pathlib.Path or None
        用户显式指定的 mesh；为 None 时自动查找。

    Returns
    -------
    pathlib.Path
        唯一的最终仿真 head mesh 路径。

    Raises
    ------
    FileNotFoundError
        指定路径不存在或目录中没有候选 mesh。
    RuntimeError
        同一优先级下找到多个候选 mesh。
    """
    if mesh_path is not None:
        resolved_mesh_path = mesh_path.resolve()
        if not resolved_mesh_path.is_file():
            raise FileNotFoundError(f"mesh 文件不存在: {resolved_mesh_path}")
        return resolved_mesh_path
    original_candidates = sorted(result_dir.glob("*_tes_flex_opt_head_mesh.msh"))
    mapped_candidates = sorted(
        result_dir.glob(
            "mapped_electrodes_simulation/*_tes_mapped_opt_head_mesh.msh"
        )
    )
    candidates = original_candidates or mapped_candidates
    if not candidates:
        raise FileNotFoundError(f"结果目录中没有找到最终 head mesh: {result_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"找到多个候选 head mesh，请通过 --mesh 指定: {candidates}"
        )
    return candidates[0].resolve()


def find_nifti_triplet(result_dir: Path) -> tuple[Path, Path, Path]:
    """
    查找具有相同前缀的 max_TI、ROI 和 non-ROI NIfTI。

    Parameters
    ----------
    result_dir : pathlib.Path
        TI focality 优化输出目录。

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        max_TI、ROI、non-ROI NIfTI 路径。

    Raises
    ------
    FileNotFoundError
        未找到完整的三件套。
    RuntimeError
        找到多组三件套，无法自动确定。
    """
    triplets: list[tuple[Path, Path, Path]] = []
    roi_suffix = "_ROI.nii.gz"
    for roi_path in sorted(result_dir.glob(f"*{roi_suffix}")):
        prefix = roi_path.name[: -len(roi_suffix)]
        field_path = result_dir / f"{prefix}_max_TI.nii.gz"
        non_roi_path = result_dir / f"{prefix}_non-ROI.nii.gz"
        if field_path.is_file() and non_roi_path.is_file():
            triplets.append((field_path, roi_path, non_roi_path))
    if not triplets:
        raise FileNotFoundError(
            "结果目录中没有找到同前缀的 max_TI、ROI 和 non-ROI NIfTI"
        )
    if len(triplets) > 1:
        raise RuntimeError(f"找到多组 NIfTI 结果，无法自动确定: {triplets}")
    return triplets[0]


def summarize_values(
    values: NDArray[np.float64], threshold: float, comparison: str
) -> dict[str, object]:
    """
    统计区域内 TI 场值及阈值满足比例。

    Parameters
    ----------
    values : numpy.ndarray
        区域内各四面体单元的 max_TI，单位 V/m。
    threshold : float
        需要验证的阈值，单位 V/m。
    comparison : {"greater_equal", "less"}
        ROI 使用 greater_equal，Non-ROI 使用 less。

    Returns
    -------
    dict[str, object]
        数量、分位数、满足比例及全单元判定。

    Raises
    ------
    ValueError
        区域为空、含非有限值或比较方式无效。
    """
    if values.size == 0:
        raise ValueError("待验证区域不包含四面体单元")
    if not np.all(np.isfinite(values)):
        raise ValueError("待验证区域的 max_TI 包含 NaN 或 Inf")
    if comparison == "greater_equal":
        satisfied = values >= threshold
        requirement = f">= {threshold:g} V/m"
    elif comparison == "less":
        satisfied = values < threshold
        requirement = f"< {threshold:g} V/m"
    else:
        raise ValueError(f"不支持的比较方式: {comparison}")
    return {
        "requirement": requirement,
        "element_count": int(values.size),
        "satisfied_element_count": int(np.count_nonzero(satisfied)),
        "satisfied_fraction": float(np.mean(satisfied)),
        "all_satisfied": bool(np.all(satisfied)),
        "min_v_per_m": float(np.min(values)),
        "mean_v_per_m": float(np.mean(values)),
        "median_v_per_m": float(np.median(values)),
        "p95_v_per_m": float(np.percentile(values, 95)),
        "max_v_per_m": float(np.max(values)),
    }


def validate_mesh(
    mesh_path: Path, roi_threshold: float, non_roi_threshold: float
) -> dict[str, object]:
    """
    在最终 head mesh 的四面体单元上验证 focality 阈值。

    Parameters
    ----------
    mesh_path : pathlib.Path
        含 max_TI、ROI 和 non-ROI 字段的最终 head mesh。
    roi_threshold : float
        ROI 下限，单位 V/m。
    non_roi_threshold : float
        Non-ROI 上限，单位 V/m。

    Returns
    -------
    dict[str, object]
        ROI、Non-ROI 和总体判定结果。

    Notes
    -----
    SimNIBS focality 的 ROC 目标按单元数量计算比例，因此这里不做体积加权。
    严格通过要求 ROI 全部单元达到下限，且 Non-ROI 全部单元低于上限。
    """
    mesh = mesh_io.read_msh(str(mesh_path))
    required_fields = {"max_TI", "ROI", "non-ROI"}
    missing_fields = required_fields - mesh.field.keys()
    if missing_fields:
        raise ValueError(f"head mesh 缺少字段: {sorted(missing_fields)}")
    tetrahedra = mesh.elm.elm_type == 4
    field_values = np.asarray(mesh.field["max_TI"].value, dtype=np.float64)
    roi_mask = tetrahedra & (mesh.field["ROI"].value > 0.5)
    non_roi_mask = tetrahedra & (mesh.field["non-ROI"].value > 0.5)
    overlap_count = int(np.count_nonzero(roi_mask & non_roi_mask))
    if overlap_count:
        raise ValueError(f"ROI 与 Non-ROI 重叠了 {overlap_count} 个四面体单元")
    roi_summary = summarize_values(
        field_values[roi_mask], roi_threshold, "greater_equal"
    )
    non_roi_summary = summarize_values(
        field_values[non_roi_mask], non_roi_threshold, "less"
    )
    false_positive_fraction = 1.0 - float(non_roi_summary["satisfied_fraction"])
    sensitivity = float(roi_summary["satisfied_fraction"])
    roc_distance = float(np.linalg.norm([false_positive_fraction, sensitivity - 1.0]))
    all_satisfied = bool(
        roi_summary["all_satisfied"] and non_roi_summary["all_satisfied"]
    )
    return {
        "mesh_path": str(mesh_path),
        "evaluation_domain": "tetrahedral elements",
        "roi": roi_summary,
        "non_roi": non_roi_summary,
        "focality_roc_distance": roc_distance,
        "all_requirements_satisfied": all_satisfied,
    }


def load_aligned_nifti(
    field_path: Path, roi_path: Path, non_roi_path: Path, mask_threshold: float
) -> tuple[nib.Nifti1Image, NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """
    读取并检查同空间的结果 NIfTI。

    Parameters
    ----------
    field_path : pathlib.Path
        max_TI NIfTI 路径。
    roi_path : pathlib.Path
        ROI NIfTI 路径。
    non_roi_path : pathlib.Path
        Non-ROI NIfTI 路径。
    mask_threshold : float
        ROI 和 Non-ROI mask 二值化阈值。

    Returns
    -------
    tuple
        max_TI image、场数组、ROI mask 和 Non-ROI mask。

    Raises
    ------
    ValueError
        NIfTI 维度、shape、affine 或 mask 内容无效。
    """
    images = [nib.load(str(path)) for path in (field_path, roi_path, non_roi_path)]
    if any(image.ndim != 3 for image in images):
        raise ValueError("max_TI、ROI 和 non-ROI NIfTI 必须都是三维影像")
    if any(image.shape != images[0].shape for image in images[1:]):
        raise ValueError("max_TI、ROI 和 non-ROI NIfTI 的 shape 不一致")
    if any(not np.allclose(image.affine, images[0].affine) for image in images[1:]):
        raise ValueError("max_TI、ROI 和 non-ROI NIfTI 的 affine 不一致")
    field_data = np.asarray(images[0].get_fdata(), dtype=np.float64)
    roi_mask = np.asarray(images[1].dataobj) > mask_threshold
    non_roi_mask = np.asarray(images[2].dataobj) > mask_threshold
    if not np.any(roi_mask) or not np.any(non_roi_mask):
        raise ValueError("NIfTI 二值化后的 ROI 或 Non-ROI 为空")
    if np.any(roi_mask & non_roi_mask):
        raise ValueError("NIfTI 中的 ROI 与 Non-ROI 存在重叠")
    return images[0], field_data, roi_mask, non_roi_mask


def save_region_labels(
    field_image: nib.Nifti1Image,
    roi_mask: NDArray[np.bool_],
    non_roi_mask: NDArray[np.bool_],
    output_path: Path,
) -> None:
    """
    保存 ROI / Non-ROI 合并标签 NIfTI。

    Parameters
    ----------
    field_image : nibabel.Nifti1Image
        提供空间信息的 max_TI 影像。
    roi_mask : numpy.ndarray
        ROI 二值 mask。
    non_roi_mask : numpy.ndarray
        Non-ROI 二值 mask。
    output_path : pathlib.Path
        输出标签 NIfTI 路径。标签 1 为 ROI，标签 2 为 Non-ROI。
    """
    labels = np.zeros(field_image.shape, dtype=np.uint8)
    labels[roi_mask] = 1
    labels[non_roi_mask] = 2
    header = field_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(labels, field_image.affine, header), str(output_path))


def extract_slice(
    data: NDArray[np.generic], axis: int, index: int
) -> NDArray[np.generic]:
    """
    提取适合 matplotlib 显示的正交切面。

    Parameters
    ----------
    data : numpy.ndarray
        三维影像数组。
    axis : int
        切面法向轴，0、1、2 分别对应 sagittal、coronal、axial。
    index : int
        该轴上的体素索引。

    Returns
    -------
    numpy.ndarray
        转置后的二维切面。
    """
    return np.take(data, index, axis=axis).T


def region_center(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """
    计算区域体素坐标的中位数中心。

    Parameters
    ----------
    mask : numpy.ndarray
        三维二值区域 mask。

    Returns
    -------
    numpy.ndarray
        三个体素轴上的整数中心坐标。
    """
    return np.rint(np.median(np.argwhere(mask), axis=0)).astype(np.int64)


def draw_contour(
    axis: plt.Axes,
    mask_slice: NDArray[np.bool_],
    color: str,
) -> None:
    """
    在切面上绘制非空区域轮廓。

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        目标绘图坐标轴。
    mask_slice : numpy.ndarray
        二维二值 mask。
    color : str
        轮廓颜色。
    """
    if np.any(mask_slice):
        axis.contour(mask_slice.astype(float), levels=[0.5], colors=[color], linewidths=1.2)


def save_overlay_png(
    field_image: nib.Nifti1Image,
    field_data: NDArray[np.float64],
    roi_mask: NDArray[np.bool_],
    non_roi_mask: NDArray[np.bool_],
    reference_path: Path | None,
    roi_threshold: float,
    output_path: Path,
) -> None:
    """
    生成以 ROI 和 Non-ROI 为中心的正交切面图。

    Parameters
    ----------
    field_image : nibabel.Nifti1Image
        max_TI NIfTI，用于目标空间。
    field_data : numpy.ndarray
        max_TI 三维数组，单位 V/m。
    roi_mask : numpy.ndarray
        ROI 二值 mask。
    non_roi_mask : numpy.ndarray
        Non-ROI 二值 mask。
    reference_path : pathlib.Path or None
        T1 背景影像；为 None 时使用黑色背景。
    roi_threshold : float
        ROI 阈值，用于保证颜色范围至少覆盖该值。
    output_path : pathlib.Path
        输出 PNG 路径。
    """
    plot_field_image = nib.as_closest_canonical(
        nib.Nifti1Image(field_data, field_image.affine)
    )
    plot_roi_image = nib.as_closest_canonical(
        nib.Nifti1Image(roi_mask.astype(np.uint8), field_image.affine)
    )
    plot_non_roi_image = nib.as_closest_canonical(
        nib.Nifti1Image(non_roi_mask.astype(np.uint8), field_image.affine)
    )
    field_data = np.asarray(plot_field_image.get_fdata(), dtype=np.float64)
    roi_mask = np.asarray(plot_roi_image.dataobj) > 0.5
    non_roi_mask = np.asarray(plot_non_roi_image.dataobj) > 0.5
    if reference_path is None:
        background = np.zeros_like(field_data)
    else:
        if not reference_path.is_file():
            raise FileNotFoundError(f"参考影像不存在: {reference_path}")
        reference_image = nib.load(str(reference_path))
        if reference_image.ndim != 3:
            raise ValueError("参考 T1 必须是三维影像")
        resampled_reference = resample_from_to(
            reference_image, plot_field_image, order=1
        )
        background = np.asarray(resampled_reference.get_fdata(), dtype=np.float64)
    positive_background = background[np.isfinite(background) & (background > 0)]
    background_max = float(np.percentile(positive_background, 99.5)) if positive_background.size else 1.0
    field_max = max(float(np.percentile(field_data[np.isfinite(field_data)], 99.5)), roi_threshold)
    centers = [("ROI center", region_center(roi_mask)), ("Non-ROI center", region_center(non_roi_mask))]
    plane_names = ["Sagittal", "Coronal", "Axial"]
    figure, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    image_artist = None
    for row, (region_name, center) in enumerate(centers):
        for axis_index, plane_name in enumerate(plane_names):
            plot_axis = axes[row, axis_index]
            index = int(center[axis_index])
            background_slice = extract_slice(background, axis_index, index)
            field_slice = extract_slice(field_data, axis_index, index)
            roi_slice = extract_slice(roi_mask, axis_index, index)
            non_roi_slice = extract_slice(non_roi_mask, axis_index, index)
            plot_axis.imshow(background_slice, cmap="gray", origin="lower", vmin=0, vmax=background_max)
            masked_field = np.ma.masked_less_equal(field_slice, 0)
            image_artist = plot_axis.imshow(masked_field, cmap="inferno", origin="lower", vmin=0, vmax=field_max, alpha=0.62)
            draw_contour(plot_axis, roi_slice, "lime")
            draw_contour(plot_axis, non_roi_slice, "cyan")
            plot_axis.set_title(f"{region_name} - {plane_name} ({index})")
            plot_axis.axis("off")
    legend_handles = [
        Line2D([0], [0], color="lime", lw=2, label="ROI"),
        Line2D([0], [0], color="cyan", lw=2, label="Non-ROI"),
    ]
    figure.legend(handles=legend_handles, loc="lower center", ncol=2)
    if image_artist is not None:
        figure.colorbar(image_artist, ax=axes, shrink=0.72, label="max_TI (V/m)")
    figure.suptitle("TI focality result with ROI / Non-ROI contours")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_summary(report: dict[str, object]) -> None:
    """
    将验证结果打印到控制台。

    Parameters
    ----------
    report : dict[str, object]
        validate_mesh 返回的验证报告。
    """
    roi = report["roi"]
    non_roi = report["non_roi"]
    assert isinstance(roi, dict)
    assert isinstance(non_roi, dict)
    print("=" * 72)
    print("TI focality 结果验证")
    print("=" * 72)
    print(f"Mesh: {report['mesh_path']}")
    print(
        "ROI      : "
        f"{roi['satisfied_element_count']}/{roi['element_count']} "
        f"({float(roi['satisfied_fraction']):.2%}) 满足 {roi['requirement']}；"
        f"全区域通过={roi['all_satisfied']}"
    )
    print(
        "Non-ROI  : "
        f"{non_roi['satisfied_element_count']}/{non_roi['element_count']} "
        f"({float(non_roi['satisfied_fraction']):.2%}) 满足 {non_roi['requirement']}；"
        f"全区域通过={non_roi['all_satisfied']}"
    )
    print(f"ROC 距离 : {float(report['focality_roc_distance']):.6f}（越接近 0 越好）")
    print(f"严格总判定: {report['all_requirements_satisfied']}")


def main() -> int:
    """
    执行阈值验证并写出 JSON、区域标签 NIfTI 和切面 PNG。

    Returns
    -------
    int
        ROI 与 Non-ROI 全部满足要求时返回 0，否则返回 1。
    """
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"结果目录不存在: {result_dir}")
    mesh_path = find_result_mesh(result_dir, args.mesh)
    field_path, roi_path, non_roi_path = find_nifti_triplet(result_dir)
    report = validate_mesh(mesh_path, args.roi_threshold, args.non_roi_threshold)
    field_image, field_data, roi_mask, non_roi_mask = load_aligned_nifti(
        field_path, roi_path, non_roi_path, args.mask_threshold
    )
    report.update(
        {
            "max_ti_nifti": str(field_path.resolve()),
            "roi_nifti": str(roi_path.resolve()),
            "non_roi_nifti": str(non_roi_path.resolve()),
            "region_labels": {"0": "background", "1": "ROI", "2": "non-ROI"},
        }
    )
    report_path = result_dir / "ti_focality_validation.json"
    labels_path = result_dir / "ti_focality_regions.nii.gz"
    overlay_path = result_dir / "ti_focality_overlay.png"
    save_region_labels(field_image, roi_mask, non_roi_mask, labels_path)
    save_overlay_png(
        field_image,
        field_data,
        roi_mask,
        non_roi_mask,
        args.reference.resolve() if args.reference else None,
        args.roi_threshold,
        overlay_path,
    )
    report["region_labels_nifti"] = str(labels_path.resolve())
    report["overlay_png"] = str(overlay_path.resolve())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_summary(report)
    print(f"JSON 报告 : {report_path}")
    print(f"区域标签  : {labels_path}（1=ROI，2=Non-ROI）")
    print(f"切面图    : {overlay_path}")
    return 0 if report["all_requirements_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
