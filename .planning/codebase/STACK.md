# Technology Stack

**Analysis Date:** 2026-03-19

## Languages

**Primary:**
- Python 3.11+ - Core runtime and application logic

**Secondary:**
- C++ - CGAL mesh generation extensions, computational kernels
- Cython - Python to C++ bindings for performance-critical operations
- C - CAT12 segmentation interface, computational utilities

## Runtime

**Environment:**
- Python 3.11.10 (Windows)
- Conda virtual environment required
- Cross-platform support (Windows, Linux, macOS)

**Package Manager:**
- pip 24.2+ - Primary Python package manager
- conda - For managing scientific libraries and C++ dependencies

## Frameworks

**Core:**
- setuptools - Build system and package management
- setuptools-scm - Version management from git tags
- NumPy 1.26.4 - Fundamental scientific computing

**Scientific:**
- SciPy 1.14.1 - Scientific algorithms and mathematical functions
- PETSc 3.21.5 - Parallel finite element equation solver
- FMM3D 1.0.0 - Fast Multipole Method for electromagnetic calculations
- PyGPC 0.4.1 - Generalized Polynomial Chaos for uncertainty quantification

**Medical Imaging:**
- nibabel 5.2.1 - Neuroimaging file I/O
- samseg 0.4a0 - MRI brain segmentation
- CAT12 - Structural MRI segmentation (via external dependency)

**GUI Development:**
- PyQt5 - Desktop application framework
- PyOpenGL 3.1.6 - 3D visualization
- matplotlib-base 3.9.2 - Plotting and visualization
- Pillow 10.4.0 - Image processing

## Key Dependencies

**Critical:**
- petsc4py 3.21.5 - PETSc Python interface for FEM solving
- h5py 3.11.0 - HDF5 file format support
- numba 0.60.0 - Just-in-time compilation for numerical code
- fmm3dpy 1.0.0 - Fast Multipole Method implementation
- jsonschema 4.23.0 - JSON validation

**Infrastructure:**
- TBB (Intel Threading Building Blocks) - Parallel processing
- Intel MKL 2024.2.1 - Mathematical kernel library
- mumps 0.0.2 - Multi-frontal Massively Parallel Solver
- python-mumps 0.0.2 - Python interface to MUMPS

## Configuration

**Environment:**
- Requires conda virtual environment
- Platform-specific build requirements:
  - Windows: MSVC >= 14.0, conda boost
  - Linux: GCC >= 6.3, system boost
  - macOS: Apple Clang == 10.0.1, Homebrew boost
- CGAL 5.5.1 (header-only, downloaded during build)
- Eigen3 3.x - Linear algebra library for CGAL

**Build:**
- pyproject.toml - Modern Python packaging configuration
- setup.py - C extension compilation with Cython
- environment_win.yml, environment_linux.yml, environment_macOS.yml - Platform-specific conda environments

## Platform Requirements

**Development:**
- Conda environment with specific dependencies
- CGAL >= 5.5, Boost >= 1.57
- Build tools appropriate for platform (MSVC, GCC, Clang)
- Git for version management

**Production:**
- Windows installer (.exe) generated
- Python 3.11+ runtime
- Scientific computing libraries (NumPy, SciPy)
- FEM solver (PETSc) - compiled specifically for target platform

## Third-Party Integrations

**Mesh Generation:**
- CGAL (GPLv3+) - Computational Geometry Algorithms Library
- Mmg (LGPL) - Mesh adaptation
- meshfix (GPLv3+) - Mesh repair utilities

**Medical Imaging Tools:**
- FSL (FreeSurfer License) - MRI processing (via dwi2cond)
- JupyterLab 4.3.0 - Interactive analysis environment

---

*Stack analysis: 2026-03-19*