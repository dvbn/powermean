"""The linear path without effects: weighted least squares of the log-response.

Log OLS is ``omega = 0``.  A solver returns a candidate, never a claim of convergence.

The problem.  On a log-response ``l``, a dense design ``X``, an offset ``o`` and
nonnegative weights ``a``, :func:`solve` computes over every row of positive weight

    beta = argmin_b  1/2 sum_i a_i (l_i - o_i - x_i' b)^2,   e = l - o - X beta,

whose normal equations are ``X'(a e) = 0``, sums over the rows.  The caller forms
``l = log Y``: ``l`` is any finite real, negative ones included, and is not
exponentiated.  The offset has coefficient one and is not multiplied by an order, no
coefficient is divided by one, no intercept is added and nothing is centered: every
column given is kept.  The three kinds of weights share this solve; they differ in
inference.  Rows of zero weight are validated and keep their position, as
in the ``SamplingUnits`` of :mod:`powermean.systems.joint`, but enter no arithmetic.

The factorization.  ``beta`` solves ``sqrt(a) X b = sqrt(a) (l - o)`` in the
least-squares sense from the singular value decomposition of ``sqrt(a) X`` by
``scipy.linalg.lstsq`` with the LAPACK driver ``gelsd`` (SciPy's default), never from
the normal equations or an inverse.  LAPACK counts as zero the singular values at most
``cutoff`` times the largest, with ``cutoff = eps max(M, N)`` for the weighted design of
``M`` rows and ``N`` columns: the default of NumPy's ``lstsq`` since NumPy 2.0, on the
same driver, and the threshold of NumPy's ``matrix_rank``, whose documentation gives it
as MATLAB's and as the bound that recognizes a column that is a linear combination of
others in floating point, whose computed singular value is roundoff rather than zero.
SciPy's default, ``eps``, can count that roundoff as a direction.  The cutoff
decides whether every coefficient is identified, and nothing else: it is neither the
certificate's ``tol`` nor its ``rcond_min``, and it is not an option.

The rank.  At full column rank the coefficients and residuals are formed.  Below it,
fewer rows of positive weight than columns included, none is: the library's
minimum-norm solution is discarded, and no ridge, dropped column, pseudoinverse or
other solver replaces it.  The rank and singular values are kept as the
library reported them.  Its residual sums of squares are never read; the residuals are
recomputed from the unweighted rows.  Near-collinearity is not collinearity:
singular values above the cutoff are solved, however small their ratio,
which they report.

The candidate.  :class:`LinearCandidate` is owned and read-only, and never a
certification claim.  The certificate evaluates the joint system and the
status of a solve reads a Newton candidate; neither receives it.  No standard error is
formed, so the coefficient gate is untouched.

Failures.  A value that is not representable in float64 (``l - o``, the weighted design
or response, the solution or the residuals) ends the solve with ``numerical_breakdown``
and its message; a decomposition that does not converge (``LinAlgError``), with
``factorization_failure``.  No other exception is caught.  NumPy's report of overflow is
suppressed where those values are formed, each being checked instead, and in the
residual sums of squares that SciPy forms unread.  Invalid inputs raise
``ValueError``.  Inputs are not modified, and no output shares memory with them.  SciPy
has no type information: ``scipy.linalg`` is reached by :func:`importlib.import_module`
with the typed interface of the one call made, as in
:mod:`powermean.solve.factorization`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, cast, get_args

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array, _finite_result
from powermean.systems.joint import SamplingUnits, _positions

LinearReason = Literal[
    "full_column_rank",
    "rank_deficient",
    "factorization_failure",
    "numerical_breakdown",
]
REASONS: tuple[LinearReason, ...] = get_args(LinearReason)
# The driver of scipy.linalg.lstsq: LAPACK's dgelsd for doubles, SciPy's default.
DRIVER = "gelsd"
# The machine epsilon of a double, 2^-52, of the rank cutoff eps max(M, N).
EPSILON = float(np.finfo(np.float64).eps)


class Lstsq(Protocol):
    """``scipy.linalg.lstsq`` as this module calls it, with the driver ``gelsd``."""

    def __call__(
        self,
        a: FloatArray,
        b: FloatArray,
        /,
        *,
        cond: float,
        check_finite: bool,
        lapack_driver: str,
    ) -> tuple[FloatArray, FloatArray, int, FloatArray]:
        """The solution, the residual sums of squares, the rank, the singular values."""
        ...


class Linalg(Protocol):
    """The object of ``scipy.linalg`` that this module uses."""

    lstsq: Lstsq


linalg = cast("Linalg", importlib.import_module("scipy.linalg"))


# ---------------------------------------------------------------------- the records


@dataclass(frozen=True, eq=False)
class LeastSquaresFactorization:
    """The singular value decomposition of the weighted design, as the library made it.

    ``driver`` is ``"gelsd"``; ``cutoff`` is ``eps max(M, N)``, relative to the largest
    singular value, and neither ``tol`` nor ``rcond_min``; ``rank`` is the library's
    count at the cutoff; ``singular_values`` are those of ``sqrt(a) X`` on the rows of
    positive weight, ``min(M, N)`` of them in decreasing order, an owned read-only copy.
    """

    driver: str
    cutoff: float
    rank: int
    singular_values: FloatArray

    def __post_init__(self) -> None:
        """Store the singular values as an owned read-only copy."""
        object.__setattr__(self, "singular_values", _immutable(self.singular_values))


@dataclass(frozen=True, eq=False)
class LinearCandidate:
    """The candidate of one least-squares solve and its diagnostics; not a solution.

    ``reason`` is one of :data:`REASONS`, none a status of the certificate:
    ``"full_column_rank"``, with coefficients; ``"rank_deficient"``, the rank at the
    cutoff below ``dimension``, the number of columns, all kept;
    ``"factorization_failure"``, a decomposition that did not converge; or
    ``"numerical_breakdown"``, a value that is not representable.  ``message`` is the
    text of the error that ended the solve, or ``""``.  ``units`` holds the positions of
    the rows of positive weight (``identifiers``), to which ``weights`` (the raw ``a``)
    and ``residuals`` are aligned, of the rows of zero weight (``inactive``), and the
    number of input rows (``nobs``).  ``factorization`` is ``None`` when none was
    formed.  ``coefficients`` (``beta``) and ``residuals`` (``e = l - o - X beta``,
    unweighted) exist at full column rank only.  The arrays are owned read-only copies;
    no status, standard error or verdict is held.
    """

    reason: LinearReason
    message: str
    dimension: int
    units: SamplingUnits
    weights: FloatArray
    factorization: LeastSquaresFactorization | None = None
    coefficients: FloatArray | None = None
    residuals: FloatArray | None = None

    def __post_init__(self) -> None:
        """Store the arrays as owned read-only copies."""
        object.__setattr__(self, "weights", _immutable(self.weights))
        for name in ("coefficients", "residuals"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _immutable(value))

    @property
    def underdetermined(self) -> bool:
        """Whether fewer rows of positive weight than columns entered the solve."""
        return int(self.units.identifiers.size) < self.dimension


# ----------------------------------------------------------------------- the inputs


def _response_and_design(
    response: ArrayLike, design: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """A nonempty vector ``l`` and a matrix with one row per value and a column."""
    values = _finite_array(response, "response")
    matrix = _finite_array(design, "design")
    if values.ndim != 1 or not values.size:
        raise ValueError("response must be a nonempty vector")
    if matrix.ndim != 2 or matrix.shape[0] != values.size or not matrix.shape[1]:
        raise ValueError(
            "design must be a matrix with one row per response and a column"
        )
    return values, matrix


def _offset(offset: ArrayLike | None, rows: int) -> FloatArray:
    """The offset ``o``: zero if omitted, a scalar for every row, else aligned."""
    if offset is None:
        return np.zeros(rows)
    shift = _finite_array(offset, "offset")
    if shift.ndim == 0:
        shift = np.full(rows, float(shift))
    if shift.shape != (rows,):
        raise ValueError("offset must be a scalar or a vector aligned with response")
    return shift


def _weights(weights: ArrayLike | None, rows: int) -> FloatArray:
    """The weights ``a``: nonnegative, one per row, not all zero; one if omitted."""
    if weights is None:
        return np.ones(rows)
    values = _finite_array(weights, "weights")
    if values.shape != (rows,):
        raise ValueError("weights must be a vector aligned with response")
    if np.any(values < 0):
        raise ValueError("weights must be nonnegative")
    if not np.any(values > 0):
        raise ValueError("weights must have a positive value")
    return values


class Problem(NamedTuple):
    """The rows of positive weight, copied from the inputs, and their positions."""

    response: FloatArray
    design: FloatArray
    offset: FloatArray
    weights: FloatArray
    units: SamplingUnits


def _problem(
    response: ArrayLike,
    design: ArrayLike,
    offset: ArrayLike | None,
    weights: ArrayLike | None,
) -> Problem:
    """The validated inputs on the rows of positive weight; no arithmetic is done."""
    values, matrix = _response_and_design(response, design)
    rows = int(values.size)
    shift = _offset(offset, rows)
    mass = _weights(weights, rows)
    active = mass > 0
    units = SamplingUnits(_positions(active), _positions(~active), rows)
    return Problem(values[active], matrix[active], shift[active], mass[active], units)


# ------------------------------------------------------------------ the arithmetic


class Weighted(NamedTuple):
    """``l - o`` and the weighted arrays of the factorization, all finite."""

    target: FloatArray
    design: FloatArray
    response: FloatArray


def _weighted(problem: Problem) -> Weighted:
    """``l - o``, ``sqrt(a) X`` and ``sqrt(a) (l - o)`` on the rows of positive weight.

    Each is formed once and must be finite: a value that is not representable raises
    ``FloatingPointError``, the first in that order.
    """
    root = np.sqrt(problem.weights)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        target: FloatArray = problem.response - problem.offset
        design: FloatArray = root[:, None] * problem.design
        response: FloatArray = root * target
    return Weighted(
        target=_finite_result(target, "response net of the offset"),
        design=_finite_result(design, "weighted design"),
        response=_finite_result(response, "weighted response"),
    )


def _factorize(weighted: Weighted) -> tuple[FloatArray, LeastSquaresFactorization]:
    """The library's least-squares solution and the factorization that it reports.

    The residual sums of squares are discarded unread.  Raises
    ``numpy.linalg.LinAlgError`` where the decomposition does not converge.
    """
    rows, columns = weighted.design.shape
    cutoff = max(rows, columns) * EPSILON
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        solution, _, rank, singular_values = linalg.lstsq(
            weighted.design,
            weighted.response,
            cond=cutoff,
            check_finite=False,
            lapack_driver=DRIVER,
        )
    factorization = LeastSquaresFactorization(
        driver=DRIVER, cutoff=cutoff, rank=int(rank), singular_values=singular_values
    )
    return solution, factorization


def _fit(
    problem: Problem, weighted: Weighted, solution: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The coefficients and the residuals ``e = (l - o) - X beta``, both finite.

    The residuals are recomputed from the unweighted rows, not taken from the library;
    a value that is not representable raises ``FloatingPointError``.
    """
    coefficients = _finite_result(solution, "least-squares solution")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        residuals: FloatArray = weighted.target - problem.design @ coefficients
    return coefficients, _finite_result(residuals, "residual vector")


def _record(
    problem: Problem,
    reason: LinearReason,
    message: str,
    *,
    factorization: LeastSquaresFactorization | None = None,
    fit: tuple[FloatArray, FloatArray] | None = None,
) -> LinearCandidate:
    """The candidate of a solve that ended with ``reason``."""
    return LinearCandidate(
        reason=reason,
        message=message,
        dimension=int(problem.design.shape[1]),
        units=problem.units,
        weights=problem.weights,
        factorization=factorization,
        coefficients=None if fit is None else fit[0],
        residuals=None if fit is None else fit[1],
    )


# ------------------------------------------------------------------------- solve


def solve(
    response: ArrayLike,
    design: ArrayLike,
    *,
    offset: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> LinearCandidate:
    """Weighted least squares of ``l - o`` on ``X``: a candidate and its diagnostics.

    Parameters
    ----------
    response : array_like
        The log-response ``l``, ``log Y`` as the caller formed it: one finite real
        value per row, negative ones included.
    design : array_like
        The dense design ``X``, finite and real, one row per value of ``response`` and
        at least one column; every column is kept and none is added.
    offset : float or array_like, optional
        The offset ``o``, with coefficient one: zero if omitted, the same value for
        every row if a scalar, else one finite value per row.
    weights : array_like, optional
        The observation weights ``a``, one finite nonnegative value per row, not all
        zero; one for every row if omitted.  Rows of zero weight enter no arithmetic.

    Returns
    -------
    LinearCandidate
        How the solve ended, its factorization and, at full column rank, the
        coefficients and the residuals; never a certification claim.

    Raises
    ------
    ValueError
        An input is not real, not finite or not of its shape, a weight is negative,
        or no weight is positive.
    """
    problem = _problem(response, design, offset, weights)
    try:
        weighted = _weighted(problem)
    except FloatingPointError as error:
        return _record(problem, "numerical_breakdown", str(error))
    try:
        solution, factorization = _factorize(weighted)
    except np.linalg.LinAlgError as error:
        return _record(problem, "factorization_failure", str(error))
    if factorization.rank < problem.design.shape[1]:
        return _record(problem, "rank_deficient", "", factorization=factorization)
    try:
        fit = _fit(problem, weighted, solution)
    except FloatingPointError as error:
        return _record(
            problem, "numerical_breakdown", str(error), factorization=factorization
        )
    return _record(
        problem, "full_column_rank", "", factorization=factorization, fit=fit
    )
