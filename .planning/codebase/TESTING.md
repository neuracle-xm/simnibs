# Testing Patterns

**Analysis Date:** 2026-03-19

## Test Framework

**Runner:**
- Framework: pytest
- Config: `conftest.py` (custom pytest configuration)
- Test discovery: Auto-discovery in `tests/` directories

**Assertion Library:**
- Standard `assert` statements
- NumPy testing utilities: `np.all()`, `np.array_equal()`
- No additional assertion libraries detected

**Run Commands:**
```bash
pytest                           # Run all tests
pytest -m slow                  # Run slow tests only
pytest --runslow                # Run slow tests with flag
pytest --tb=short               # Short traceback format
```

## Test File Organization

**Location:**
- Co-located with source code in `tests/` subdirectories
- Pattern: `simnibs/module/tests/`

**Naming:**
- Files prefixed with `test_`: `test_mesh_io.py`, `test_fem.py`
- Test class names: `TestClassName`
- Test method names: `test_feature_name()`

**Structure:**
```
simnibs/
├── module/
│   ├── __init__.py
│   ├── module.py
│   └── tests/
│       ├── __init__.py
│       └── test_module.py
```

## Test Structure

**Suite Organization:**
```python
import pytest
import numpy as np
from simnibs.module import module_name

class TestFeatureName:
    def test_basic_functionality(self):
        # Arrange
        input_data = ...
        # Act
        result = module_name.function(input_data)
        # Assert
        assert result.expected_value == result.actual_value

    def test_edge_case(self):
        # Test boundary conditions
        pass

    @pytest.mark.slow
    def test_performance_critical(self):
        # Test slow operations
        pass
```

**Patterns:**
- Arrange-Act-Assert pattern commonly used
- Setup done in test methods or fixtures
- Clear separation between test cases
- Individual test methods for each scenario

## Mocking

**Framework:**
- Standard `unittest.mock` not extensively used
- Minimal mocking detected
- Real objects used when possible

**Patterns:**
```python
# Limited mocking observed
# Tests tend to use real objects with proper fixtures
```

**What to Mock:**
- External file I/O (minimal)
- Network requests (minimal)
- Time-dependent functions (minimal)

**What NOT to Mock:**
- Core data structures (numpy arrays, scipy objects)
- Internal simulation logic
- File I/O that's critical to functionality

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture(scope='module')
def sphere3_msh():
    fn = os.path.join(SIMNIBSDIR, '_internal_resources', 'testing_files', 'sphere3.msh')
    return mesh_io.read_msh(fn)

@pytest.fixture
def atlas_itk_msh():
    fn = os.path.join(SIMNIBSDIR, '_internal_resources', 'testing_files', 'cube_atlas', 'atlas.txt.gz')
    return itk_mesh_io.itk_to_msh(fn)
```

**Location:**
- Test-specific fixtures in individual test files
- Shared fixtures in `conftest.py`
- Test data in `_internal_resources/testing_files/`

**Factory Pattern:**
- Custom object creation in test methods
- Complex objects created with helper functions
- Minimal use of dedicated factory classes

## Coverage

**Requirements:**
- Not explicitly enforced
- Test files exist for all major modules
- Integration tests present

**View Coverage:**
```bash
pytest --cov=simnibs.module           # Coverage report
pytest --cov=simnibs --cov-report=html # HTML coverage report
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes
- Location: `tests/test_*.py` files
- Pattern: Test isolated functionality with mocked dependencies
- Examples: `test_mesh_io.py`, `test_transformations.py`

**Integration Tests:**
- Scope: Multiple components working together
- Pattern: Real data flow between modules
- Examples: Tests requiring mesh files and simulation runs
- Marked with `@pytest.mark.slow`

**E2E Tests:**
- Framework: Not detected
- Focus: Not implemented
- Alternative: Integration tests cover full workflows

## Common Patterns

**Async Testing:**
- Limited async functionality in codebase
- Standard synchronous test patterns used

**Error Testing:**
```python
def test_error_conditions(self):
    with pytest.raises(ValueError):
        function_that_should_fail()
    with pytest.raises(OSError):
        function_with_io_error()
```

**Parametrized Tests:**
- Limited use detected
- Standard test methods with multiple scenarios

**Fixture Scopes:**
- `function`: Default, per-test
- `class`: Shared within test class
- `module`: Shared across all tests in module
- `session`: Shared across all test runs

**Test Data Management:**
- External test files stored in `_internal_resources/testing_files/`
- Temporary files created with `tempfile` module
- Cleanup with `shutil.rmtree()` in teardown

**Test Organization:**
- Tests grouped by functionality
- Each test class focuses on one feature
- Related tests in same class
- Clear naming conventions

---

*Testing analysis: 2026-03-19*