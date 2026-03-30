"""
CHARM Pipeline 步骤脚本包

将 SimNIBS charm 分割流程拆分为独立的步骤脚本。
"""

from .create_surfaces import create_surfaces
from .denoise import denoise_inputs
from .init_atlas import init_atlas
from .mesh import create_mesh_step as create_mesh
from .prepare_t1 import prepare_t1
from .prepare_t2 import prepare_t2
from .segment import run_segmentation

__all__ = [
    "prepare_t1",
    "prepare_t2",
    "denoise_inputs",
    "init_atlas",
    "run_segmentation",
    "create_surfaces",
    "create_mesh",
]
