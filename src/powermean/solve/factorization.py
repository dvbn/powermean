"""Factorizations of the certificate: an equilibrated LU and a Cholesky factor.

The certificate of
:mod:`powermean.solve.certificate` factorizes the observed bread ``C = -dS/dgamma'`` of
:mod:`powermean.systems.joint` in two ways, one for each of its uses.

Equilibrated LU.  With the scale ``s``, ``s_k`` the square root of the
largest of the absolute entries of row ``k`` and of column ``k``, the diagonal
congruence

    Cs = C / (s s'),   that is   C = diag(s) Cs diag(s),

keeps the sign structure of ``C`` and leaves the Newton step unchanged.  ``Cs`` is
factorized by LU with partial pivoting (``scipy.linalg.lu_factor``), and LAPACK's
``dgecon`` estimates its reciprocal condition number in the 1-norm from that factor
and ``||Cs||_1``: the conditioning of the factorized system in scaled coordinates.
:meth:`Factor.solve` returns ``C^-1 b = Cs^-1 (b / s) / s``.  A matrix with an entry
that is not finite raises ``FloatingPointError``.  An exact zero row or column, a
factor that SciPy reports exactly singular (its ``LinAlgWarning``), an exact zero
pivot, an estimate that fails, or ``rcond <= rcond_min`` raise
:class:`SingularJacobianError`, which keeps the estimate when one was formed (NaN
otherwise) and, for a factor that exists but fails the threshold, that factor, for
diagnosis only.  Nothing is ridged, floored, symmetrized or replaced by a generalized
inverse.  ``SingularJacobianError`` derives from ``Exception``, and
``Factor`` is a frozen record.

Cholesky factor, package code.  The curvature gate and the scaled score of the
certificate use ``C_sym = (C + C')/2``; the observed bread of the system is symmetric up
to the rounding of its products.  Each entry of ``C_sym`` is ``(C_ij + C_ji)/2``
rounded once to the nearest double, so that the symmetrization neither overflows at
the largest doubles nor loses a subnormal entry: the sum of the two entries is halved
where it is finite, and their halves are added where it overflows, which happens only
when both are at least ``2^970`` in absolute value.  :func:`symmetric_cholesky` returns
the lower factor ``L``, ``L L' = C_sym``, of ``scipy.linalg.cholesky``, which raises
``numpy.linalg.LinAlgError`` where LAPACK's ``potrf`` meets a leading minor that is not
positive: ``C_sym`` is then not positive definite in floating point.  Positive
definiteness is decided by that factorization and by no threshold on eigenvalues:
attempting the Cholesky factorization is the usual numerical test, whereas an
eigenvalue test (below ``k eps`` times the largest, or not positive) would add a
tolerance that the options do not have.  How well conditioned the matrix
is, is the separate test of ``rcond`` above (``rcond_min``).
:func:`forward_solve` returns ``L^-1 v`` by forward substitution
(``scipy.linalg.solve_triangular``), whose Euclidean norm is ``||C_sym^(-1/2) v||``.

SciPy publishes no type information, so this module reaches the functions above with
:func:`importlib.import_module` and gives the modules the typed interfaces of the calls
made here, as ``powermean._vendor._scipy`` does for the carried files, which this
module may not import.  Nothing
is wrapped or copied: every operation is SciPy's.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from powermean.model.variance import FloatArray

PivotArray = NDArray[np.int32]


class LuFactor(Protocol):
    """``scipy.linalg.lu_factor`` as called here."""

    def __call__(
        self, a: FloatArray, /, *, check_finite: bool
    ) -> tuple[FloatArray, PivotArray]:
        """The factors ``L`` and ``U`` in one matrix, and the pivots."""
        ...


class LuSolve(Protocol):
    """``scipy.linalg.lu_solve`` as called here."""

    def __call__(
        self, lu_and_piv: tuple[FloatArray, PivotArray], b: FloatArray, /
    ) -> FloatArray:
        """The solution ``x`` of ``a x = b`` from the factor of ``a``."""
        ...


class Cholesky(Protocol):
    """``scipy.linalg.cholesky`` with the lower factor."""

    def __call__(
        self, a: FloatArray, /, *, lower: bool, check_finite: bool
    ) -> FloatArray:
        """The Cholesky factor of ``a``."""
        ...


class SolveTriangular(Protocol):
    """``scipy.linalg.solve_triangular`` on a lower triangular matrix."""

    def __call__(
        self, a: FloatArray, b: FloatArray, /, *, lower: bool, check_finite: bool
    ) -> FloatArray:
        """The solution ``x`` of ``a x = b`` by substitution."""
        ...


class Linalg(Protocol):
    """The objects of ``scipy.linalg`` that this module uses."""

    LinAlgWarning: type[RuntimeWarning]
    lu_factor: LuFactor
    lu_solve: LuSolve
    cholesky: Cholesky
    solve_triangular: SolveTriangular


class Dgecon(Protocol):
    """``scipy.linalg.lapack.dgecon`` as called here."""

    def __call__(
        self, a: FloatArray, anorm: float, /, *, norm: str
    ) -> tuple[float, int]:
        """The reciprocal condition number estimated from an LU factor, and ``info``."""
        ...


class Lapack(Protocol):
    """The object of ``scipy.linalg.lapack`` that this module uses."""

    dgecon: Dgecon


linalg = cast("Linalg", importlib.import_module("scipy.linalg"))
lapack = cast("Lapack", importlib.import_module("scipy.linalg.lapack"))


# ------------------------------------------------------------- equilibrated LU


class SingularJacobianError(Exception):
    """Exact zero pivot or ``rcond <= rcond_min``.

    ``rcond`` is the estimate when one was computed (NaN otherwise), so that a failed
    factorization keeps it; ``factor`` is the LU of a matrix that is ill-conditioned
    but not exactly singular, for diagnosis only.
    """

    def __init__(
        self, message: str, rcond: float = np.nan, factor: Factor | None = None
    ) -> None:
        super().__init__(message)
        self.rcond = float(rcond)
        self.factor = factor


@dataclass(frozen=True, eq=False)
class Factor:
    """LU of the equilibrated matrix: ``C = Ds Cs Ds`` and ``Cs = P L U``.

    ``lu`` is the pair that ``scipy.linalg.lu_factor`` returns for ``Cs``, ``ds`` the
    scale ``s`` and ``rcond`` LAPACK's estimate of the reciprocal condition number of
    ``Cs`` in the 1-norm.
    """

    lu: tuple[FloatArray, PivotArray]
    ds: FloatArray
    rcond: float

    def solve(self, rhs: FloatArray) -> FloatArray:
        """``C^-1 rhs``: ``C = diag(s) Cs diag(s)``, so ``x = Cs^-1 (rhs / s) / s``."""
        scale = self.ds[:, None] if rhs.ndim == 2 else self.ds
        solution: FloatArray = linalg.lu_solve(self.lu, rhs / scale) / scale
        return solution


def factorize(matrix: FloatArray, rcond_min: float) -> Factor:
    """Equilibrated dense LU with a LAPACK 1-norm reciprocal condition estimate.

    Equilibration ``Cs = C / (s s')``, ``s_k = sqrt(max(rowmax_k, colmax_k))`` of
    ``|C|``, is a diagonal congruence, so it keeps the sign structure and does not
    alter the Newton step.

    Parameters
    ----------
    matrix : array
        The square matrix ``C``.
    rcond_min : float
        The threshold that the estimate must exceed, strictly.

    Returns
    -------
    Factor
        The factor of ``Cs``, its scale and the estimate.

    Raises
    ------
    FloatingPointError
        An entry of ``matrix`` is not finite.
    SingularJacobianError
        An exact zero row or column, an exactly singular factor or zero pivot, an
        estimate that failed, or an estimate at most ``rcond_min``.
    """
    if not np.isfinite(matrix).all():
        raise FloatingPointError("non-finite Jacobian")
    absolute = np.abs(matrix)
    scale = np.sqrt(np.maximum(absolute.max(axis=0), absolute.max(axis=1)))
    if np.any(scale == 0):
        raise SingularJacobianError("zero row/column in Jacobian")
    scaled = matrix / scale[:, None] / scale[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("error", linalg.LinAlgWarning)
        try:
            lu = linalg.lu_factor(scaled, check_finite=False)
        except linalg.LinAlgWarning as exc:  # exactly singular factor
            raise SingularJacobianError(str(exc)) from exc
    if np.any(np.diag(lu[0]) == 0):
        raise SingularJacobianError("zero pivot", 0.0)
    anorm = np.abs(scaled).sum(axis=0).max()
    rcond, info = lapack.dgecon(lu[0], anorm, norm="1")
    if info != 0 or not np.isfinite(rcond):
        raise SingularJacobianError("condition estimate failed")
    if rcond <= rcond_min:
        raise SingularJacobianError(
            f"rcond {rcond:.3e} <= rcond_min", rcond, Factor(lu, scale, float(rcond))
        )
    return Factor(lu, scale, float(rcond))


# ----------------------------------------------------------------- Cholesky factor


def symmetric_cholesky(matrix: FloatArray) -> FloatArray:
    """The lower Cholesky factor ``L`` of ``C_sym = (C + C')/2``: ``L L' = C_sym``.

    Each entry of ``C_sym`` is ``(C_ij + C_ji)/2`` rounded once to the nearest double.
    Where the sum ``C_ij + C_ji`` is finite it is halved: below ``2^-1021`` in absolute
    value the sum is exact and only the halving rounds, and above it only the sum
    rounds.  Where the sum overflows, both entries are at least ``2^970`` in absolute
    value, so that their halves are exact and only the sum of the halves rounds.
    Halving each entry first would round the subnormal ``nextafter(0, 1)`` to zero, and
    the sum alone overflows a diagonal of ``1.7e308``.  No entry is floored, cut off or
    added to.

    Parameters
    ----------
    matrix : array
        The square matrix ``C``, finite.

    Returns
    -------
    array
        ``L``, lower triangular, zero above its diagonal.

    Raises
    ------
    numpy.linalg.LinAlgError
        ``C_sym`` is not positive definite in floating point: LAPACK's ``potrf`` met a
        leading minor that is not positive.
    """
    with np.errstate(over="ignore"):
        total = matrix + matrix.T
    halves = 0.5 * matrix + 0.5 * matrix.T
    symmetric = np.where(np.isinf(total), halves, 0.5 * total)
    lower: FloatArray = linalg.cholesky(symmetric, lower=True, check_finite=False)
    return lower


def forward_solve(lower: FloatArray, rhs: FloatArray) -> FloatArray:
    """``L^-1 rhs`` by forward substitution, ``L`` lower triangular.

    ``L`` is the factor of :func:`symmetric_cholesky`, so that the Euclidean norm of
    the result is ``||C_sym^(-1/2) rhs||``: ``||L^-1 v||^2 = v' C_sym^-1 v``.
    """
    solution: FloatArray = linalg.solve_triangular(
        lower, rhs, lower=True, check_finite=False
    )
    return solution
