#!/usr/bin/env python
"""Script to run PyInstaller with proper environment setup."""
import os
import sys
import subprocess

# Ensure we're in the right directory
os.chdir(r'C:\Users\mdrs\Documents\simnibs')

# Find the simnibs conda environment python
conda_env_path = r'C:\Users\mdrs\miniforge3\envs\simnibs'
simnibs_python = os.path.join(conda_env_path, 'python.exe')

if not os.path.exists(simnibs_python):
    print(f"ERROR: Python not found at {simnibs_python}")
    sys.exit(1)

print(f"Using Python: {simnibs_python}")

# Run pyinstaller with the correct Python
result = subprocess.run(
    [simnibs_python, '-m', 'PyInstaller', '-y', 'charm.spec'],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Return code: {result.returncode}")

sys.exit(result.returncode)
