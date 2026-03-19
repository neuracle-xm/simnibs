# External Integrations

**Analysis Date:** 2026-03-19

## APIs & External Services

**Model Repositories:**
- GitHub - Primary code repository
  - URL: https://github.com/simnibs/simnibs
  - Download: CLI tools, documentation
- GitHub Releases - Automated publishing platform
  - Action: GitHubRelease@0 for automated uploads
- GitHub - SimNIBS Coils repository
  - Service: Download additional coil models
  - Client: requests library
  - Endpoint: https://github.com/simnibs/simnibs-coils
  - Auth: None required

**Scientific Data:**
- Neuroimaging file formats
  - NIfTI (.nii, .nii.gz) - MRI data
  - MSH (.msh) - Mesh files
  - VTK - Visualization toolkit format
  - GZIP - Compressed file support

## Data Storage

**Databases:**
- File-based only (No external databases)
- HDF5 (.h5) - Hierarchical data format for simulation results
  - Client: h5py library
  - Usage: Storage of field solutions and simulation data

**File Storage:**
- Local filesystem - Primary storage for subject data
- Network file systems - Supported for data sharing
- GitHub repository - Distribution of coil models

**Caching:**
- Filesystem caching for intermediate results
- No external caching services

## Authentication & Identity

**Auth Provider:**
- GitHub OAuth - CI/CD authentication
  - Implementation: Azure Pipelines connection
  - Purpose: Automated publishing to GitHub releases
- No user authentication in application

## Monitoring & Observability

**Error Tracking:**
- None detected - No external error tracking service
- Built-in Python exception handling

**Logs:**
- File-based logging
- Console logging for CLI tools
- No centralized log aggregation

## CI/CD & Deployment

**Hosting:**
- GitHub repository - Code storage
- GitHub Releases - Distribution platform
- Windows executable installer via conda-pack

**CI Pipeline:**
- Azure Pipelines - Continuous integration
  - Configuration: .azure-pipelines/*.yml
  - Triggers: master branch commits
  - Jobs: Build, Test, Release (Windows platform)
  - Deployment: Automated GitHub release drafts

## Environment Configuration

**Required env vars:**
- CONDA_PREFIX - Conda environment path (required for build)
- None for runtime operation

**Secrets location:**
- GitHub connection credentials - Azure Pipelines secure variables
- No API keys or service accounts in codebase

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- HTTP requests to GitHub - Coil model downloads
  - Client: requests.get() for GitHub zip downloads
  - Endpoints:
    - https://github.com/simnibs/simnibs-coils/archive/master.zip
    - https://api.github.com (for releases)
  - Error handling: Temporary files with cleanup

## External File Formats

**Medical Imaging:**
- NIfTI-1/2 - Neuroimaging Toolkit format (via nibabel)
- Analyze - Old neuroimaging format (via nibabel)
- DICOM - Not directly supported (converts via external tools)

**Mesh Formats:**
- Gmsh - Mesh visualization (via gmsh.info)
- VTK - Visualization Toolkit format
- FreeMesh - Finite element mesh format

**Scientific Formats:**
- HDF5 - Hierarchical Data Format (via h5py)
- JSON - Configuration and metadata (via jsonschema)
- XML - Not prominently used

## Integration Notes

**Third-Party Libraries:**
- CAT12 - External segmentation tool (GPLv2+)
- ADMlib - Auxiliary dipole method (GPLv2, academic only)
- PyVista - 3D visualization (MIT license)

**External Dependencies:**
- FSL (FreeSurfer License) - MRI processing suite
- dwi2cond - White matter conductivity estimation
- Charm-gems - CHARM segmentation utilities

---

*Integration audit: 2026-03-19*