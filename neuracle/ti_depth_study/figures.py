"""在个体 T1 上绘制研究靶区和共享切面的正向场。"""

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_roi_slices(t1_path: Path, roi_path: Path, output: Path, field_paths: dict[str, Path], hotspots: dict[str, list[float]] | None = None) -> None:
    """用相同个体空间三切面与色标绘制 ROI 和可选场。

    Parameters
    ----------
    t1_path, roi_path : Path
        个体解剖 T1 与配准后的二值 ROI。
    output : Path
        PNG 输出。没有场时输出几何核验图。
    field_paths : dict
        标签到共同正向 NIfTI 的映射。全部使用同一 V/m 色标。
    hotspots : dict, optional
        每份场的个体物理空间主热点坐标，标在相同切面上的正交投影。
    """
    roi_image = nib.as_closest_canonical(nib.load(roi_path))
    t1_image = resample_from_to(nib.load(t1_path), roi_image, order=1)
    roi = np.asanyarray(roi_image.dataobj) > 0
    if not roi.any():
        raise ValueError(f"绘图目标掩膜为空: {roi_path}")
    t1 = np.asanyarray(t1_image.dataobj)
    center = np.round(np.argwhere(roi).mean(axis=0)).astype(int)
    t1_max = float(np.percentile(t1[t1 > 0], 99))
    data = {label: np.asanyarray(resample_from_to(nib.load(path), roi_image, order=1).dataobj) for label, path in field_paths.items()}
    vmax = max((float(np.nanmax(value)) for value in data.values()), default=0.0)
    panels = data or {roi_path.name.removesuffix(".nii.gz"): None}
    fig = Figure(figsize=(10, 3.0 * len(panels)), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(panels), 3, squeeze=False)
    legend = None
    for row, (label, field) in enumerate(panels.items()):
        for dim, (ax, name) in enumerate(zip(axes[row], ("Sagittal", "Coronal", "Axial"))):
            slice_t1 = np.take(t1, center[dim], axis=dim).T
            slice_roi = np.take(roi, center[dim], axis=dim).T
            ax.imshow(slice_t1, cmap="gray", origin="lower", vmin=0.0, vmax=t1_max)
            if field is not None:
                current = np.take(field, center[dim], axis=dim).T
                legend = ax.imshow(np.ma.masked_where(~np.isfinite(current) | (current <= 0), current),
                                   cmap="inferno", origin="lower", alpha=0.62, vmin=0.0, vmax=vmax)
            if slice_roi.any():
                ax.contour(slice_roi, levels=[0.5], colors=["cyan"], linewidths=0.8)
            if hotspots and label in hotspots:
                voxel = nib.affines.apply_affine(np.linalg.inv(roi_image.affine), hotspots[label])
                point = np.delete(voxel, dim)
                ax.scatter(point[0], point[1], marker="x", s=18, color="#2de5d1", label="Hotspot projection")
                if dim == 0:
                    ax.legend(loc="lower right", fontsize=6)
            ax.set_title(f"{label} | {name}")
            ax.set_axis_off()
    if legend is not None:
        fig.colorbar(legend, ax=axes, label="max_TI (V/m)", shrink=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145)
