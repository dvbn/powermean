"""Typed views of the SciPy objects that the carried Newton files use.

SciPy publishes no type information: its modules have no ``py.typed`` marker, and no
stub package is among the dependencies.  A static ``import scipy.linalg`` therefore
fails the strict type check, and the carried files may not silence it.  This module
imports the two modules of SciPy that they use with :func:`importlib.import_module`,
which runs the same import, and gives the module objects the typed interfaces below.
Nothing is wrapped or copied: ``linalg.solve`` is ``scipy.linalg.solve``,
``sparse.issparse`` is ``scipy.sparse.issparse``, and so on, so every operation of the
carried files is SciPy's own.

:mod:`powermean._vendor.vendor_newton` imports this module under the name ``scipy``,
so that its calls read as in its source, ``scipy.linalg.solve(..., assume_a="sym")``;
:mod:`powermean._vendor.vendor_linear_loss` imports ``sparse`` from it, as its source
imports ``sparse`` from SciPy.

The interfaces are those of the calls that the carried files make, and no wider: the
one keyword of ``norm`` is declared by :class:`NormKeywords`, and ``issparse`` is a
``TypeIs`` of :class:`powermean._vendor._typing.SparseMatrix`.  Where the carried files
test a design with ``issparse``, the type checker therefore takes its sparse member in
the branch where the test is true and its dense member in the other.  ``TypeIs`` is in
``typing`` from Python 3.13; the type checker reads it from its own stubs of
``typing_extensions``, under ``TYPE_CHECKING``, and nothing is imported for it when the
package runs.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import numpy as np

from powermean._vendor._typing import FloatArray, SparseMatrix

if TYPE_CHECKING:
    from typing_extensions import TypeIs


class Solve(Protocol):
    """``scipy.linalg.solve`` as the Newton step calls it."""

    def __call__(
        self, a: FloatArray, b: FloatArray, /, *, check_finite: bool, assume_a: str
    ) -> FloatArray:
        """The solution ``x`` of ``a x = b``."""
        ...


class NormKeywords(TypedDict):
    """The one keyword with which the line search calls ``scipy.linalg.norm``."""

    ord: int


class Norm(Protocol):
    """``scipy.linalg.norm`` as the line search calls it, with the keyword ``ord``.

    ``ord`` is also a builtin, so the keyword is declared by :class:`NormKeywords`
    rather than as a parameter: a call without ``ord``, or with another keyword, is a
    type error.
    """

    def __call__(self, a: FloatArray, /, **options: Unpack[NormKeywords]) -> np.float64:
        """The norm of ``a`` of the order ``ord``."""
        ...


class DiaMatrix(Protocol):
    """``scipy.sparse.dia_matrix`` built from one diagonal and its offset."""

    def __call__(
        self, arg1: tuple[FloatArray, int], /, *, shape: tuple[int, int]
    ) -> SparseMatrix:
        """The sparse matrix with the given diagonal."""
        ...


class Linalg(Protocol):
    """The objects of ``scipy.linalg`` that ``vendor_newton`` uses."""

    LinAlgWarning: type[RuntimeWarning]
    norm: Norm
    solve: Solve


class Sparse(Protocol):
    """The objects of ``scipy.sparse`` that ``vendor_linear_loss`` uses."""

    dia_matrix: DiaMatrix

    def issparse(self, x: object, /) -> TypeIs[SparseMatrix]:
        """Whether ``x`` is a sparse matrix or a sparse array of SciPy."""
        ...


linalg = cast("Linalg", importlib.import_module("scipy.linalg"))
sparse = cast("Sparse", importlib.import_module("scipy.sparse"))
