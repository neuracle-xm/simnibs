"""
PyInstaller runtime hook for SimNIBS to handle petsc4py import issues.

This hook modifies the import behavior to prevent issues with petsc4py
when used in a frozen environment.
"""

import sys

# Check if we're in a frozen environment
if getattr(sys, 'frozen', False):
    import types

    # First, mock petsc4py and mpi4py BEFORE any simnibs imports
    petsc_module = types.ModuleType('petsc4py')
    petsc_lib = types.ModuleType('petsc4py.lib')
    sys.modules['petsc4py'] = petsc_module
    sys.modules['petsc4py.lib'] = petsc_lib
    sys.modules['mpi4py'] = types.ModuleType('mpi4py')

    # Create a minimal dummy fem module that provides what sim_struct needs
    # The sim_struct module imports from fem, so we need to provide a mock
    dummy_fem_module = types.ModuleType('simnibs.simulation.fem')

    # Add any required attributes to the fem module
    # sim_struct imports: from . import fem
    # We don't need to add much as charm doesn't directly use fem
    sys.modules['simnibs.simulation.fem'] = dummy_fem_module

    # For the optimization module
    dummy_opt_struct = types.ModuleType('simnibs.optimization.opt_struct')
    sys.modules['simnibs.optimization.opt_struct'] = dummy_opt_struct

