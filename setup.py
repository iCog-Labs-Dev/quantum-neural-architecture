from setuptools import setup, find_packages
import os

# Collect all packages under src/ plus the top-level utility package
src_packages = find_packages(where="src")
package_dirs = {"": "src"}

# Include utility/ from project root
if os.path.isdir("utility"):
    src_packages.append("utility")
    package_dirs["utility"] = "utility"

setup(
    name="qnn",
    version="0.1",
    packages=src_packages,
    package_dir=package_dirs,
)