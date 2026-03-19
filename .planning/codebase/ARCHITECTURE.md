# Architecture

**Analysis Date:** 2026-03-19

## Pattern Overview

**Overall:** Layered modular architecture with domain separation

**Key Characteristics:**
- Modular separation of concerns: segmentation, meshing, simulation, optimization
- Scientific computing pattern with heavy use of numerical libraries
- Event-driven workflow pattern for the CHARM pipeline
- Data-driven architecture with SESSION as central data structure

## Layers

**Segmentation Layer (`simnibs/segmentation/`)**:
- Purpose: MRI preprocessing and brain segmentation
- Location: `C:\Users\50609\Documents\simnibs\simnibs\segmentation`
- Contains: CHARM pipeline, surface reconstruction, atlases
- Depends on: CAT12, samseg, nibabel
- Used by: Mesh generation, subject setup

**Mesh Tools Layer (`simnibs/mesh_tools/`)**:
- Purpose: Mesh generation, I/O, and processing
- Location: `C:\Users\50609\Documents\simnibs\simnibs\mesh_tools`
- Contains: Msh data structure, CGAL extensions, meshing algorithms
- Depends on: CGAL, Boost, Eigen3, numpy
- Used by: Simulation layer

**Simulation Layer (`simnibs/simulation/`)**:
- Purpose: FEM simulation computation and field calculation
- Location: `C:\Users\50609\Documents\simnibs\simnibs\simulation`
- Contains: SESSION structure, FEM solvers, TMS coils
- Depends on: PETSc, MUMPS, scipy, numpy
- Used by: Optimization layer

**Optimization Layer (`simnibs/optimization/`)**:
- Purpose: Stimulation parameter optimization
- Location: `C:\Users\50609\Documents\simnibs\simnibs\optimization`
- Contains: TDCS/TMS optimization algorithms, GPC uncertainty quantification
- Depends on: scipy.optimize, pygpc, scipy.spatial
- Used by: CLI tools, applications

**Utilities Layer (`simnibs/utils/`)**:
- Purpose: Shared functionality and cross-cutting concerns
- Location: `C:\Users\50609\Documents\simnibs\simnibs\utils`
- Contains: File management, coordinate transforms, subject navigation
- Depends on: Standard libraries, nibabel, scipy
- Used by: All layers

## Data Flow

**Processing Pipeline:**

1. **Input Stage**
   - MRI images (T1, optionally T2)
   - Subject setup files
   - Coil model definitions

2. **Segmentation Stage**
   - CHARM pipeline runs (registration, segmentation, surface creation)
   - Outputs: segmented tissues, brain surfaces, subject folder structure

3. **Meshing Stage**
   - Tetrahedral mesh generation using CGAL
   - Outputs: .msh files with conductivity properties

4. **Simulation Stage**
   - SESSION collects simulation configurations
   - FEM assembly and solving with PETSc/MUMPS
   - Outputs: field distributions, electrode potentials

5. **Postprocessing Stage**
   - Field mapping to surfaces/atlases
   - Statistical analysis with GPC
   - Visualization output

**State Management:**
- SESSION object maintains simulation state
- SubjectFiles class manages file relationships
- Templates class handles reference data

## Key Abstractions

**SESSION** (`simulation/sim_struct.py`):
- Purpose: Central container for simulation configuration
- Examples: `C:\Users\50609\Documents\simnibs\simnibs\simulation\sim_struct.py`
- Pattern: Data container with validation and preparation methods

**SubjectFiles** (`utils/file_finder.py`):
- Purpose: Manages subject file structure (m2m folders)
- Examples: `C:\Users\50609\Documents\simnibs\simnibs\utils\file_finder.py`
- Pattern: File locator with path resolution

**Msh** (`mesh_tools/mesh_io.py`):
- Purpose: Mesh data structure and I/O operations
- Examples: `C:\Users\50609\Documents\simnibs\simnibs\mesh_tools\mesh_io.py`
- Pattern: Rich data structure with validation methods

## Entry Points

**CLI Tools** (`pyproject.toml`):
- `simnibs` - Main simulation runner (`simulation/run_simnibs.py`)
- `charm` - CHARM segmentation pipeline (`segmentation/charm_main.py`)
- `optimize_tes` - TES optimization (`optimization/tdcs_optimization.py`)
- `optimize_tms` - TMS optimization (`optimization/tms_optimization.py`)
- `eeg_positions` - EEG electrode positioning utilities

**Primary Entry Point**:
- Location: `C:\Users\50609\Documents\simnibs\simnibs\simulation\run_simnibs.py`
- Triggers: Simulation execution
- Responsibilities: SESSION management, workflow coordination

## Error Handling

**Strategy:** Comprehensive logging with configurable levels

**Patterns:**
- SimnibsLogger class for consistent logging
- Exception handling with informative messages
- File validation and path resolution checks
- FEM solver error detection and reporting

## Cross-Cutting Concerns

**Logging:** Centralized via `simnibs_logger.py` with configurable output
**Validation:** Type hints and parameter validation in core structures
**Performance:** Parallel processing, memory management, and efficient solvers
**Visualization:** HTML report generation with embedded matplotlib/brainsprite

---

*Architecture analysis: 2026-03-19*