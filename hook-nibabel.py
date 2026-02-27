"""
PyInstaller hook for nibabel to ensure all modules are included.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all nibabel submodules
hiddenimports = collect_submodules('nibabel')

# Collect any data files
datas = collect_data_files('nibabel')
