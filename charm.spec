# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for charm.py - SimNIBS head mesh creation tool

Usage:
    pyinstaller charm.spec

This will create a folder distribution (not --onefile) containing the
charm executable with all dependencies.
"""

import os
import sys
from pathlib import Path

# Get the SimNIBS directory - use current directory when spec is executed
SIMNIBSDIR = os.path.abspath(os.getcwd())

block_cipher = None

# Analysis phase
a = Analysis(
    ['simnibs/cli/charm.py'],
    pathex=[SIMNIBSDIR],
    binaries=[],
    datas=[
        # Include the charm.ini configuration file
        (os.path.join(SIMNIBSDIR, 'simnibs', 'charm.ini'), 'simnibs'),

        # Include external binaries for Windows
        (os.path.join(SIMNIBSDIR, 'simnibs', 'external', 'bin', 'win', '*.exe'), 'simnibs/external/bin/win'),
        (os.path.join(SIMNIBSDIR, 'simnibs', 'external', 'bin', 'win', '*.dll'), 'simnibs/external/bin/win'),

        # Include atlas data for segmentation
        (os.path.join(SIMNIBSDIR, 'simnibs', 'segmentation', 'atlases', '*'), 'simnibs/segmentation/atlases'),

        # Include electrode cap templates (ElectrodeCaps_MNI)
        (os.path.join(SIMNIBSDIR, 'simnibs', 'resources', 'ElectrodeCaps_MNI', '*'), 'simnibs/resources/ElectrodeCaps_MNI'),

        # Include MNI templates
        (os.path.join(SIMNIBSDIR, 'simnibs', 'resources', 'templates', '*'), 'simnibs/resources/templates'),

        # Include LUT files
        (os.path.join(SIMNIBSDIR, 'simnibs', 'resources', '*LUT.txt'), 'simnibs/resources'),

        # Include coil models (if needed)
        (os.path.join(SIMNIBSDIR, 'simnibs', 'resources', 'coil_models', '*'), 'simnibs/resources/coil_models'),

        # Include samseg module and data
        (os.path.join(SIMNIBSDIR, 'simnibs', 'segmentation', 'simnibs_samseg', '*'), 'simnibs/segmentation/simnibs_samseg'),

        # Include segmentation Python scripts (needed for multiprocessing)
        (os.path.join(SIMNIBSDIR, 'simnibs', 'segmentation', '*.py'), 'simnibs/segmentation'),
    ],
    hiddenimports=[
        # SimNIBS modules
        'simnibs',
        'simnibs.__version__',
        'simnibs.cli',
        'simnibs.cli.utils',
        'simnibs.cli.utils.args_charm',
        'simnibs.cli.utils.args_general',
        'simnibs.cli.utils.helpers',
        'simnibs.segmentation',
        'simnibs.segmentation.charm_main',
        'simnibs.segmentation.charm_utils',
        'simnibs.segmentation.simnibs_samseg',
        'simnibs.segmentation.simnibs_samseg.simnibs_segmentation_utils',
        'simnibs.segmentation.brain_surface',
        'simnibs.mesh_tools',
        'simnibs.mesh_tools.mesh_io',
        'simnibs.mesh_tools.meshing',
        'simnibs.mesh_tools.cgal',
        'simnibs.utils',
        'simnibs.utils.file_finder',
        'simnibs.utils.transformations',
        'simnibs.utils.simnibs_logger',
        'simnibs.utils.settings_reader',
        'simnibs.utils.cond_utils',
        'simnibs.utils.html_writer',
        'simnibs.utils.spawn_process',

        # Cython extensions (these are compiled .pyd files on Windows)
        'simnibs.mesh_tools.cython_msh',
        'simnibs.segmentation._marching_cubes_lewiner_cy',
        'simnibs.segmentation._cat_c_utils',
        'simnibs.segmentation._thickness',
        'simnibs.mesh_tools.cgal.create_mesh_surf',
        'simnibs.mesh_tools.cgal.create_mesh_vol',
        'simnibs.mesh_tools.cgal.cgal_misc',
        'simnibs.mesh_tools.cgal.polygon_mesh_processing',

        # Third-party libraries
        'numpy',
        'scipy',
        'nibabel',
        'nibabel.arraywriters',
        'nibabel.brikhead',
        'nibabel.builtin_namespaces',
        'nibabel.cifti2',
        'nibabel.dataobj_files',
        'nibabel.easing',
        'nibabel.filebasedimages',
        'nibabel.fileholders',
        'nibabel.fileslice',
        'nibabel.floating',
        'nibabel.imageclasses',
        'nibabel.info',
        'nibabel.keywordonly',
        'nibabel.loadsave',
        'nibabel.minc',
        'nibabel.nifti',
        'nibabel.nifti2',
        'nibabel.onetime',
        'nibabel.openers',
        'nibabel.parrec',
        'nibabel.processing',
        'nibabel.proxy_api',
        'nibabel.py3k',
        'nibabel.pydicom_compat',
        'nibabel.spatialimages',
        'nibabel.tempfiles',
        'nibabel.trackvis',
        'nibabel.tripquery',
        'nibabel.volumeutils',
        'nibabel.nifti.fileslice',
        'nibabel.nifti.volumeutils',
        'nibabel.arrayproxy',
        'nibabel.deprecator',
        'nibabel.chiral',
        'nibabel.affines',
        'nibabel.filename_parser',
        'nibabel.trk',
        'nibabel.gifti',
        'nibabel.minc1',
        'nibabel.brikhead.afni',
        'nibabel.parrec',
        'h5py',
        'jsonschema',
        'matplotlib',
        'PIL',
        'requests',

        # Note: petsc4py and mpi4py are excluded as they are not needed for charm
        # and cause issues in frozen environment

        # numba (for JIT compilation)
        'numba',
        'numba.core.typing',
        'numba.core.extending',

        # Standard library modules that might be missed
        'logging',
        'argparse',
        'tempfile',
        'glob',
        'shutil',
        'time',
        're',
        'subprocess',
    ],
    hookspath=['./'],  # Use current directory for hooks
    hooksconfig={},
    runtime_hooks=['pyi_rth_simnibs.py'],
    excludes=[
        # Exclude simulation-related packages that cause issues
        'petsc4py',
        'mpi4py',

        # Exclude GUI-related packages to reduce size
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'matplotlib.backends.backend_qt5agg',
        'jupyterlab',
        'pytest',
        'mock',

        # Exclude other unnecessary packages
        'IPython',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter phase - remove unnecessary files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Executable phase
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='charm',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Collect binaries and data files
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='charm',
)
