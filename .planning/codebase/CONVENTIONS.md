# Coding Conventions

**Analysis Date:** 2026-03-19

## Naming Patterns

**Files:**
- All lowercase with underscores: `mesh_io.py`, `file_finder.py`, `sim_struct.py`
- Test files prefixed with `test_`: `test_mesh_io.py`, `test_fem.py`
- Module names: All lowercase, descriptive

**Classes:**
- PascalCase: `Nodes`, `Elements`, `Msh`, `SESSION`, `TDCS`, `TMSLEADFIELD`
- Exception classes: PascalCase ending with `Error`: `InvalidMeshError`

**Functions:**
- All lowercase with underscores: `read_msh`, `write_msh`, `get_fields_at_coordinates`
- Private methods prefixed with single underscore: `_GetitemTester`, `_increment_version_dev_branch`
- Dunder methods surrounded by double underscores: `__init__`, `__getitem__`

**Variables:**
- All lowercase with underscores: `node_coord`, `elm_type`, `subpath`
- Constants: UPPER_SNAKE_CASE: `CONCURRENT_MESH_3`, `LINKED_WITH_TBB`
- Instance variables: same as general variables
- Class attributes: same as general variables

**Parameters:**
- All lowercase with underscores: `matlab_struct`, `node_coord`, `elm_type`
- Optional parameters: explicitly documented in docstrings

## Code Style

**Formatting:**
- No explicit formatter configuration found
- Line length appears to follow PEP 8 (79-99 characters)
- Consistent indentation with 4 spaces
- Trailing commas in function calls and data structures when spanning multiple lines

**Linting:**
- No specific linter configuration files found (no .flake8, .pylintrc)
- Type hints used throughout: `Union`, `np.ndarray`, `Optional`
- Imports organized with standard library, third-party, local imports

**Docstrings:**
- Format: Multi-line strings with proper indentation
- Style: NumPy format for functions and classes
- Includes: Description, Parameters, Returns, Examples where applicable
- Language: Primary language is English with technical terms in English

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import time`, `import copy`
2. Third-party imports: `import numpy as np`, `import scipy.io`, `import h5py`
3. Local imports: `from ..mesh_tools import mesh_io`, `from ..utils import file_finder`

**Path Aliases:**
- Relative imports used extensively: `from .. import __version__`
- Absolute imports for external modules: `import scipy.spatial`

**Import Patterns:**
- Aliased imports: `import numpy as np`
- Explicit imports: `from typing import Union`
- Conditional imports with try/except blocks for optional dependencies

## Error Handling

**Patterns:**
```python
# Custom exceptions
class InvalidMeshError(ValueError):
    pass

# Value validation with clear messages
if not is_conda:
    raise Exception("Cannot run setup without conda")

# Logging errors
logger.error(f"Failed to process {filename}")
logger.debug("Traceback", exc_info=(exc_type, exc_value, exc_traceback))

# Warning handling
warnings.warn(f"Deprecated function: {func_name}")
```

**Exception Types:**
- `ValueError` for invalid data/parameters
- `OSError` for file system issues
- `RuntimeError` for execution failures
- Custom exceptions for domain-specific errors

**Logging Strategy:**
- Use custom logger: `logger = logging.getLogger('simnibs')`
- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL, SUMMARY
- Contextual information included in messages
- Exception info logged with traceback for debug level

## Comments

**When to Comment:**
- Complex algorithms requiring explanation
- Non-obvious behavior
- Important implementation details
- Performance considerations
- Platform-specific code

**Docstring Pattern:**
```python
def function_name(param1, param2=None):
    """Function description.

    Parameters
    ----------
    param1 : type
        Description of parameter
    param2 : type, optional
        Description of optional parameter

    Returns
    -------
    type
        Description of return value
    """
```

## Function Design

**Size:**
- Functions generally 10-50 lines
- Complex operations broken into smaller helper functions
- Functions have single responsibility

**Parameters:**
- 1-7 parameters typical
- Optional parameters with default values
- Type hints for all parameters
- Clear parameter naming

**Return Values:**
- Single return value pattern
- Type hints for return values
- None for void functions
- Custom objects for complex data structures

## Module Design

**Exports:**
- `__all__` list defined for public API
- Import lists at top of modules
- Conditional imports for optional dependencies

**Barrel Files:**
- Not commonly used
- Modules directly import from submodules

**Module Organization:**
- Clear separation of concerns
- Related functionality grouped together
- Platform-specific code isolated

---

*Convention analysis: 2026-03-19*