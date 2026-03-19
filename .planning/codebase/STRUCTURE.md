# Codebase Structure

**Analysis Date:** 2026-03-19

## Directory Layout

```
simnibs/
├── __init__.py           # Package initialization and main exports
├── segmentation/         # MRI segmentation and preprocessing
├── mesh_tools/          # Mesh generation and I/O operations
├── simulation/          # FEM simulation computation
├── optimization/        # Stimulation parameter optimization
├── utils/              # Shared utilities and cross-cutting concerns
├── cli/                # Command-line interface tools
├── external/           # External binaries and wheels
├── resources/          # Templates, atlases, and reference data
├── _internal_resources/# Internal assets (HTML, icons, testing files)
├── matlab_tools/       # MATLAB integration utilities
├── eeg/                # EEG-specific functionality
└── GUI/                # GUI application components
```

## Directory Purposes

**`simnibs/segmentation/`**:
- Purpose: MRI preprocessing, brain segmentation, and surface reconstruction
- Contains: CHARM pipeline, CAT12 integration, samseg, atlases
- Key files:
  - `charm_main.py` - Main segmentation pipeline
  - `brain_surface.py` - Surface reconstruction
  - `atlases/` - Reference atlases for registration
  - `simnibs_samseg/` - Samseg segmentation implementation

**`simnibs/mesh_tools/`**:
- Purpose: Mesh generation, processing, and I/O operations
- Contains: CGAL extensions, mesh structures, format converters
- Key files:
  - `mesh_io.py` - Core mesh data structures (Msh, Nodes, Elements)
  - `meshing.py` - Mesh generation operations
  - `cgal/` - CGAL C++ extensions (Cython)

**`simnibs/simulation/`**:
- Purpose: FEM simulation engine and computational methods
- Contains: Session management, solvers, field calculations
- Key files:
  - `sim_struct.py` - SESSION data structure and management
  - `fem.py` - FEM solver implementation
  - `electrode_placement.py` - Electrode positioning
  - `tms_coil/` - TMS coil models and calculations

**`simnibs/optimization/`**:
- Purpose: Stimulation parameter optimization algorithms
- Contains: TDCS/TMS optimization, uncertainty quantification
- Key files:
  - `tdcs_optimization.py` - TDCS optimization
  - `tms_optimization.py` - TMS optimization
  - `pygpc/` - Polynomial chaos expansion for uncertainty

**`simnibs/utils/`**:
- Purpose: Shared utilities and cross-cutting concerns
- Contains: File management, transforms, navigation
- Key files:
  - `file_finder.py` - SubjectFiles class for m2m structure
  - `transformations.py` - Coordinate transformations
  - `cond_utils.py` - Conductivity management

**`simnibs/cli/`**:
- Purpose: Command-line interface tools
- Contains: Individual CLI tools implementing subcommands
- Key files: `charm.py`, `run_simnibs.py`, `eeg_positions.py`

## Key File Locations

**Entry Points:**
- `C:\Users\50609\Documents\simnibs\simnibs\simulation\run_simnibs.py` - Main simulation runner
- `C:\Users\50609\Documents\simnibs\simnibs\segmentation\charm_main.py` - CHARM pipeline
- `C:\Users\50609\Documents\simnibs\simnibs\__init__.py` - Package initialization

**Configuration:**
- `C:\Users\50609\Documents\simnibs\pyproject.toml` - Project configuration and scripts
- `C:\Users\50609\Documents\simnibs\simnibs\utils\simnibs_logger.py` - Logging configuration

**Core Logic:**
- `C:\Users\50609\Documents\simnibs\simnibs\simulation\sim_struct.py` - SESSION data structures
- `C:\Users\50609\Documents\simnibs\simnibs\mesh_tools\mesh_io.py` - Mesh data structures
- `C:\Users\50609\Documents\simnibs\simnibs\fem.py` - FEM solver implementation

**Testing:**
- Distributed across modules in `tests/` subdirectories

## Naming Conventions

**Files:**
- snake_case for Python files (e.g., `charm_main.py`)
- PascalCase for classes (e.g., `SubjectFiles`, `SESSION`)
- camelCase for MATLAB integration (e.g., `gifti/`)

**Functions:**
- snake_case for regular functions
- camelCase for compatibility with MATLAB interface

**Variables:**
- snake_case for general variables
- PascalCase for class instances
- ALL_CAPS for constants

**Classes:**
- PascalCase for public classes (e.g., `TDCSoptimize`)
- snake_case for method names
- _private_prefix for internal methods

## Where to Add New Code

**New Feature:**
- Primary code: Based on domain (segmentation/, simulation/, etc.)
- Tests: Add alongside source in `tests/` subdirectory
- CLI tool: Add to `cli/` directory and register in `pyproject.toml`

**New Component/Module:**
- Implementation: Add appropriate module directory
- Dependencies: Add to `pyproject.toml` requirements
- Documentation: Add docstrings following NumPy format

**Utilities:**
- Shared helpers: Add to `utils/` module
- Cross-cutting concerns: Place in appropriate utility module
- Type annotations: Always include in new code

## Special Directories

**`external/`**:
- Purpose: Platform-specific binaries and wheels
- Generated: No - manually maintained
- Committed: Yes - required for distribution

**`resources/`**:
- Purpose: Templates, atlases, coil models
- Generated: No - reference data
- Committed: Yes - essential for functionality

**`_internal_resources/`**:
- Purpose: HTML templates, icons, testing files
- Generated: No - internal assets
- Committed: Yes - required for GUI and testing

**`private_gitignore/`**:
- Purpose: Temporary scripts and development tools
- Generated: Yes - for temporary development files
- Committed: No - excluded by .gitignore

---

*Structure analysis: 2026-03-19*