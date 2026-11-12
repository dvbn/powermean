"""Restricted native package: the exact FE incidence row reduction.

``_echelon_cpp`` (built from ``echelon.cpp`` by the top-level CMake project when
``POWERMEAN_BUILD_BASIS`` is on) is an optional accelerator of
:func:`powermean.fe.certificate_basis._operational_echelon`.  Only that module
imports it, only after explicit ``basis_backend="cpp"`` selection. The Python
loop stays the specification and handles representation/work/allocation declines.
A missing or broken explicitly requested extension is an error. Defaults do not
probe this module.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
