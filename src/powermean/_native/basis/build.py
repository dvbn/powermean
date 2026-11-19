"""Explicit local D19 CMake build of the echelon extension; wheels use scikit-build.

Run with a supported Python and an existing pybind11.  No dependency is installed
and no package runtime builds or loads an artifact.  The IOLS kernel is left out of
this build (``POWERMEAN_BUILD_IOLS=OFF``); ``build.py`` of ``_native/iols`` builds it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def configure_command(build_dir: Path, *, pybind11_dir: Path) -> tuple[str, ...]:
    """Configure the project for this interpreter with an explicit pybind11."""
    root = Path(__file__).resolve().parents[4]
    return (
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build_dir),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11_dir}",
        "-DPOWERMEAN_BUILD_IOLS=OFF",
        "-DPOWERMEAN_BUILD_BASIS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    )


def build_native_basis(
    build_dir: Path, *, pybind11_dir: Path, install_prefix: Path
) -> Path:
    """Build and stage the extension in an explicit directory outside the package."""
    subprocess.run(configure_command(build_dir, pybind11_dir=pybind11_dir), check=True)
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
    return install_prefix / "powermean" / "_native" / "basis"


def main() -> None:
    """Read explicit build inputs for an offline local build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--pybind11-dir", type=Path, required=True)
    parser.add_argument("--install-prefix", type=Path, required=True)
    args = parser.parse_args()
    build_native_basis(
        args.build_dir,
        pybind11_dir=args.pybind11_dir,
        install_prefix=args.install_prefix,
    )


if __name__ == "__main__":
    main()
