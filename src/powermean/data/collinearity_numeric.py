"""Checked arithmetic and the LAPACK boundary of the collinearity analysis.

Internal module: the numerical operations that
:mod:`powermean.data.collinearity_geometry` and :mod:`powermean.data.collinearity` run,
the record of every LAPACK attempt and the exact integer domains of the records.
Nothing here absorbs, ranks a subset, decides an omission, fits or forms an indicator
matrix.

Coordinates.  Columns are scaled by the powers of two of
:func:`powermean.data.design.column_scale` only if multiplying back restores every
entry bit for bit (:func:`reversible`), else ``FloatingPointError``: stricter than
the global helper, whose domain is unchanged.

Numerical rank.  The number of singular values (NumPy's SVD, ``compute_uv=False``,
``hermitian=False``) strictly greater than a cutoff; equality is numerical zero.  A
matrix's own cutoff is ``eps max(rows, columns) sigma_max``, the rule of
``solve/linear.py`` and of NumPy's ``matrix_rank``, which detects numerical, not exact,
dependence.

The LAPACK boundary.  Every SVD, pivoted economic QR (LAPACK ``dgeqp3``) and
``dgelsd`` solve is one attempt of the ledger, recorded in order as a
:class:`Factorization`.  Its outputs are checked before they are indexed or
converted: real floating arrays of the shapes that the call declares, every entry
finite; singular values nonnegative and nonincreasing; QR pivots an integer
permutation; a solve's rank an exact integer within its matrix's sizes.  The values
kept (singular values, ``|diag R|``, a solution) are the nearest doubles of the
outputs and must be usable (:func:`_doubles`): no double is infinite (an overflow),
and every tiny output, nonzero and below the normal doubles before rounding, converts
exactly (else it underflowed inexactly, to zero or to another subnormal).  Exact
zeros and subnormals stay, as does the ordinary rounding of a normal value of a wider
type; nothing is floored, clipped or flushed.  An output that fails is recorded
``unusable``, a ``LinAlgError`` of the call itself ``failed``, and both then raise
``numpy.linalg.LinAlgError`` under any warning policy, since the cast reports no
condition of its own and these checks decide.  Nothing else is caught.

Integers.  The indices, counts, ranks and sizes that the records retain are exact
integers, Python or NumPy but not booleans, checked before any conversion
(:func:`exact_integer`, :func:`integers`): nothing fractional is narrowed and nothing
of more than one dimension is flattened.

Norms.  ``numpy.hypot.reduce`` of the magnitudes, at least the largest magnitude, so
a nonzero vector never has a zero norm; overflow and an invalid value raise.
Underflow does not: this platform's ``hypot`` flags the exact subnormal
``hypot(3 2^-1074, 4 2^-1074)``, the norms only feed the screen, whose other side is
at least ``2^-537``, and zero columns are told by their entries.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, TypeGuard, cast

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.design import column_scale
from powermean.fe.absorb_numeric import checked_product, group_sums
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

# The machine epsilon of a double, 2^-52, of every rank cutoff eps max(M, N) sigma.
EPSILON = float(np.finfo(np.float64).eps)
# The smallest normal double, 2^-1022, as a double: below it a nonzero value is tiny.
SMALLEST_NORMAL = np.float64(np.finfo(np.float64).smallest_normal)
KINDS = ("gram_norm", "gram_qr", "direct_svd", "subset_svd", "span_svd", "gelsd")
OUTCOMES = ("ok", "failed", "unusable")
Solved = tuple[FloatArray, FloatArray, int, FloatArray]


class Linalg(Protocol):
    """``scipy.linalg.qr`` and ``scipy.linalg.lstsq`` as this module calls them."""

    def qr(
        self, a: FloatArray, /, *, mode: str, pivoting: bool, check_finite: bool
    ) -> tuple[FloatArray, FloatArray, IndexArray]:
        """``Q``, ``R`` and the pivots ``P`` of ``a[:, P] = Q R``."""
        ...

    def lstsq(
        self,
        a: FloatArray,
        b: FloatArray,
        /,
        *,
        cond: float,
        check_finite: bool,
        lapack_driver: str,
    ) -> Solved:
        """The solution, the residual sums of squares, the rank, singular values."""
        ...


linalg = cast("Linalg", importlib.import_module("scipy.linalg"))


# ---------------------------------------------------------------------- domains


def is_integer(value: object) -> TypeGuard[int | np.integer[Any]]:
    """Whether ``value`` is an exact integer: a Python or NumPy integer, not a bool."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def exact_integer(value: object, what: str) -> int:
    """A nonnegative exact integer as an ``int``, checked before any conversion."""
    if not (is_integer(value) and value >= 0):
        raise ValueError(f"{what} must be a nonnegative integer")
    return int(value)


def integers(values: object, what: str) -> tuple[int, ...]:
    """Exact nonnegative integers of a tuple, list or one-dimensional integer array.

    Each entry is checked before it is converted: a fractional entry is refused, not
    narrowed, and an array of more than one dimension is refused, not flattened.
    """
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
            raise ValueError(f"{what} must be a vector of integers")
        values = values.tolist()
    if not isinstance(values, tuple | list):
        raise ValueError(f"{what} must be a tuple, list or vector of integers")
    return tuple(exact_integer(value, what) for value in values)


def increasing(values: object, what: str) -> tuple[int, ...]:
    """Strictly increasing exact nonnegative integers (:func:`integers`)."""
    found = integers(values, what)
    if list(found) != sorted(set(found)):
        raise ValueError(f"{what} must be strictly increasing")
    return found


def real(value: object, what: str, *, positive: bool = False) -> float:
    """A finite nonnegative real number, positive if asked, as a ``float``."""
    if isinstance(value, bool) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"{what} must be a finite real number")
    found, sign = float(value), "positive" if positive else "nonnegative"
    if not (np.isfinite(found) and found >= 0) or (positive and found == 0):
        raise ValueError(f"{what} must be finite and {sign}")
    return found


def owned(value: object) -> FloatArray:
    """An owned read-only finite vector of doubles, real, checked before conversion."""
    found = np.asarray(value)
    if found.ndim != 1 or (found.size and found.dtype.kind not in "fiu"):
        raise ValueError("recorded values must be a finite real vector")
    stored = immutable_array(found.astype(np.float64))
    if not np.isfinite(stored).all():
        raise ValueError("recorded values must be a finite real vector")
    return stored


def count_above(values: FloatArray, cutoff: float) -> int:
    """How many ``values`` are strictly greater than ``cutoff``; equality is not."""
    return int(np.count_nonzero(np.asarray(values) > cutoff))


# ------------------------------------------------------------------- the ledger


@dataclass(frozen=True, eq=False)
class Factorization:
    """One attempted decomposition or solve: kind, purpose, shape, outcome, result.

    ``values`` are singular values (``|diag R|`` for ``gram_qr``), ``min(shape)`` of
    them, present with ``cutoff`` and ``rank`` exactly when the outcome is ``ok``;
    ``rank`` counts them above ``cutoff``, except for ``gelsd``, whose rank is
    LAPACK's report at ``cutoff = cond s_1`` and need not equal that count.  A
    ``failed`` or ``unusable`` attempt has no values, cutoff or rank.
    """

    kind: str
    purpose: str
    shape: tuple[int, int]
    outcome: str
    values: FloatArray
    cutoff: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        """Check every field before storing it owned; ``ValueError`` otherwise."""
        if self.kind not in KINDS or self.outcome not in OUTCOMES:
            raise ValueError("unknown factorization kind or outcome")
        if not isinstance(self.purpose, str):
            raise ValueError("the purpose of an attempt is a string")
        shape = integers(self.shape, "shape")
        if len(shape) != 2:
            raise ValueError("shape must be two nonnegative sizes")
        values = owned(self.values)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)
        if self.outcome != "ok":
            if values.size or self.cutoff is not None or self.rank is not None:
                raise ValueError("a failed or unusable attempt has no result")
            return
        self._result(min(shape), values)

    def _result(self, size: int, values: FloatArray) -> None:
        """An ``ok`` attempt: ``size`` nonnegative values, its cutoff and rank."""
        if not size or values.size != size or np.any(values < 0):
            raise ValueError("an ok attempt has min(shape) nonnegative values")
        cutoff = real(self.cutoff, "the cutoff")
        rank = exact_integer(self.rank, "the rank")
        if rank > size or (
            self.kind != "gelsd" and rank != count_above(values, cutoff)
        ):
            raise ValueError("rank must count the values above the cutoff")
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "rank", rank)


Ledger = list[Factorization]


class Rank(NamedTuple):
    """The singular values of a matrix, the cutoff used and the rank at it."""

    values: FloatArray
    cutoff: float
    rank: int


def _attempt[T](ledger: Ledger, entry: Factorization, call: Callable[[], T]) -> T:
    """Run ``call``; a failed decomposition is recorded as ``entry``, then raised."""
    try:
        return call()
    except np.linalg.LinAlgError:
        ledger.append(entry)
        raise


def _failed(
    kind: str, purpose: str, shape: tuple[int, int], outcome: str
) -> Factorization:
    """The record of an attempt that produced no usable output."""
    return Factorization(kind, purpose, shape, outcome, np.zeros(0))


def _unusable(ledger: Ledger, entry: Factorization) -> np.linalg.LinAlgError:
    """Record an unusable LAPACK output; the error to raise."""
    ledger.append(entry)
    return np.linalg.LinAlgError(
        f"{entry.purpose}: the {entry.kind} output is unusable"
    )


# ------------------------------------------------------------------- arithmetic


def reversible(matrix: FloatArray, what: str) -> tuple[FloatArray, FloatArray]:
    """``matrix / d`` and the power-of-two column scales ``d``, exactly reversible."""
    scales = column_scale(matrix)
    # Quotients are below 2 in magnitude; an underflow is caught by the reversal.
    with np.errstate(under="ignore"):
        scaled: FloatArray = matrix / scales
        restored = scaled * scales
    if not np.array_equal(restored, matrix):
        raise FloatingPointError(f"{what}: the column scaling is not reversible")
    return immutable_array(scaled), scales


def norm(vector: FloatArray, what: str) -> float:
    """The 2-norm by ``hypot.reduce`` of the magnitudes (module notes)."""
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            return float(np.hypot.reduce(np.abs(vector)))
    except FloatingPointError as error:
        raise FloatingPointError(f"{what}: the norm is not representable") from error


def screened(within: float, original: float, screen: float) -> bool:
    """The candidate screen ``||R_j|| <= sqrt(s) ||U_j||``; equality passes."""
    with np.errstate(all="raise"):
        bound = float(np.sqrt(screen)) * original
    return within <= bound


def gram(z: FloatArray) -> FloatArray:
    """``Z'Z`` by range-checked products and row-order sums, column by column."""
    n, k = z.shape
    found = np.zeros((k, k))
    for j in range(k):
        products = checked_product(z[:, j : j + 1], z[:, j:], "Gram product")
        found[j, j:] = found[j:, j] = group_sums(products, np.zeros(n, np.intp), 1)[0]
    return found


# ----------------------------------------------------------------------- LAPACK


def _doubles(found: np.ndarray[Any, np.dtype[Any]]) -> FloatArray | None:
    """Finite real floating values as their nearest doubles if usable, else None."""
    if not np.isfinite(found).all():
        return None
    # Not reported here, so that no warning policy preempts the checks below.
    with np.errstate(over="ignore", under="ignore"):
        values: FloatArray = found.astype(np.float64)
    tiny = np.abs(found) < SMALLEST_NORMAL
    exact = np.array_equal(values[tiny], found[tiny])
    return values if np.isfinite(values).all() and exact else None


def _singular(output: object, count: int) -> FloatArray | None:
    """``count`` real singular values, finite, nonnegative, nonincreasing, or None."""
    found = np.asarray(output)
    if found.dtype.kind != "f" or found.shape != (count,):
        return None
    values = _doubles(found)
    if values is None:
        return None
    ordered = bool(np.all(values[1:] <= values[:-1]) and np.all(values >= 0))
    return values if count and ordered else None


def _pivoted(
    r: object, pivots: object, shape: tuple[int, int]
) -> tuple[FloatArray, IndexArray] | None:
    """``|diag R|`` and the pivots of an economic QR of a ``shape`` matrix, or None."""
    factor, order = np.asarray(r), np.asarray(pivots)
    if factor.dtype.kind != "f" or factor.shape != (min(shape), shape[1]):
        return None
    if order.dtype.kind not in "iu" or order.shape != (shape[1],):
        return None
    permutation = np.array_equal(np.sort(order), np.arange(shape[1]))
    if not (permutation and np.isfinite(factor).all()):
        return None
    diagonal = _doubles(np.abs(np.diag(factor)))
    return None if diagonal is None else (diagonal, frozen_index(order.astype(np.intp)))


def _solution(
    solution: object, rank: object, values: object, shape: tuple[int, int]
) -> tuple[FloatArray, int, FloatArray] | None:
    """A vector solve's solution, rank and singular values, checked, or None."""
    found, singular = np.asarray(solution), _singular(values, min(shape))
    if found.dtype.kind != "f" or found.shape != (shape[1],) or singular is None:
        return None
    if not (is_integer(rank) and 0 <= rank <= min(shape)):
        return None
    coefficients = _doubles(found)
    return None if coefficients is None else (coefficients, int(rank), singular)


def numerical_rank(
    matrix: FloatArray,
    ledger: Ledger,
    *,
    kind: str,
    purpose: str,
    cutoff: float | None = None,
) -> Rank:
    """Checked singular values and the rank at ``cutoff`` (the matrix's own rule)."""
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    output = _attempt(
        ledger,
        _failed(kind, purpose, shape, "failed"),
        lambda: np.linalg.svd(matrix, compute_uv=False, hermitian=False),
    )
    values = _singular(output, min(shape))
    if values is None:
        raise _unusable(ledger, _failed(kind, purpose, shape, "unusable"))
    used = EPSILON * max(shape) * float(values[0]) if cutoff is None else cutoff
    rank = count_above(values, used)
    ledger.append(Factorization(kind, purpose, shape, "ok", values, used, rank))
    return Rank(immutable_array(values), used, rank)


def pivoted_diagonal(
    matrix: FloatArray, ledger: Ledger, *, cutoff: float
) -> tuple[FloatArray, IndexArray, int]:
    """``|diag R|``, the pivots of the pivoted economic QR, and the candidate rank."""
    shape, purpose = (int(matrix.shape[0]), int(matrix.shape[1])), "Gram matrix"
    _, r, pivots = _attempt(
        ledger,
        _failed("gram_qr", purpose, shape, "failed"),
        lambda: linalg.qr(matrix, mode="economic", pivoting=True, check_finite=False),
    )
    found = _pivoted(r, pivots, shape)
    if found is None:
        raise _unusable(ledger, _failed("gram_qr", purpose, shape, "unusable"))
    diagonal, order = found
    rank = count_above(diagonal, cutoff)
    ledger.append(
        Factorization("gram_qr", purpose, shape, "ok", diagonal, cutoff, rank)
    )
    return immutable_array(diagonal), order, rank


def least_squares(
    a: FloatArray, b: FloatArray, ledger: Ledger, *, cond: float, purpose: str
) -> tuple[FloatArray, int]:
    """``dgelsd`` through ``scipy.linalg.lstsq`` at ``cond``: the solution and rank.

    ``b`` is a vector.  The residual sums of squares are discarded unread; SciPy may
    form them with an overflow that it does not report, hence the local error state.
    """
    shape = (int(a.shape[0]), int(a.shape[1]))

    def call() -> Solved:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return linalg.lstsq(
                a, b, cond=cond, check_finite=False, lapack_driver="gelsd"
            )

    solution, _, rank, values = _attempt(
        ledger, _failed("gelsd", purpose, shape, "failed"), call
    )
    found = _solution(solution, rank, values, shape)
    if found is None:
        raise _unusable(ledger, _failed("gelsd", purpose, shape, "unusable"))
    coefficients, reported, singular = found
    used = cond * float(singular[0])
    ledger.append(
        Factorization("gelsd", purpose, shape, "ok", singular, used, reported)
    )
    return immutable_array(coefficients), reported
