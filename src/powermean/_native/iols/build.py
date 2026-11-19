"""Explicit local D19 CMake build; wheel builds use scikit-build-core.

Run with a supported Python and existing pybind11, NumPy, Eigen and LAPACK.
No dependency is installed and no package runtime builds or loads an artifact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def configure_command(
    build_dir: Path,
    *,
    eigen_include: Path,
    pybind11_dir: Path,
    openblas64_library: Path | None = None,
) -> tuple[str, ...]:
    """Configure the project for this interpreter with explicit existing headers."""
    root = Path(__file__).resolve().parents[4]
    command: tuple[str, ...] = (
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build_dir),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DEIGEN3_INCLUDE_DIR={eigen_include}",
        f"-Dpybind11_DIR={pybind11_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    if openblas64_library is not None:
        command += (f"-DPOWERMEAN_IOLS_OPENBLAS64_LIBRARY={openblas64_library}",)
    return command


def build_native_iols(
    build_dir: Path,
    *,
    eigen_include: Path,
    pybind11_dir: Path,
    install_prefix: Path,
    openblas64_library: Path | None = None,
) -> Path:
    """Build and stage the extension in an explicit directory outside the package."""
    command = configure_command(
        build_dir,
        eigen_include=eigen_include,
        pybind11_dir=pybind11_dir,
        openblas64_library=openblas64_library,
    )
    subprocess.run(command, check=True)
    subprocess.run(
        ("cmake", "--build", str(build_dir), "--config", "Release"), check=True
    )
    subprocess.run(
        (
            "cmake",
            "--install",
            str(build_dir),
            "--config",
            "Release",
            "--prefix",
            str(install_prefix),
        ),
        check=True,
    )
    return install_prefix / "powermean" / "_native" / "iols"


def main() -> None:
    """Read explicit build inputs for an offline local build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--eigen-include", type=Path, required=True)
    parser.add_argument("--pybind11-dir", type=Path, required=True)
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument("--openblas64-library", type=Path)
    args = parser.parse_args()
    build_native_iols(
        args.build_dir,
        eigen_include=args.eigen_include,
        pybind11_dir=args.pybind11_dir,
        install_prefix=args.install_prefix,
        openblas64_library=args.openblas64_library,
    )


if __name__ == "__main__":
    main()
