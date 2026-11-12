"""Joint system without effects: moment, Jacobians, criterion and row contributions.

The joint system without effects (``b_i = x_i``), on the transformed problem: an
outcome ``z >= 0``, a dense design ``X`` with rows ``x_i'``, coefficients ``gamma``, a
transformed offset ``o`` and fixed nonnegative weights ``a``.  The index is
``q = X gamma + o``; the order layer has already multiplied the offset by ``omega``,
and no order or exposure enters here.  The row terms ``r``, ``c``, ``w`` and ``l`` are
those of :mod:`powermean.model.rows`, and each weight multiplies the terms of its row
once:

    moment                S = X'(a r)          = sum_i a_i r_i x_i
    observed bread        C = X' diag(a c) X   = -dS/dgamma'
    expected information  H = X' diag(a w) X   the conditional expectation of C
    criterion             L = sum_i a_i l_i    dL/dgamma = -S, its Hessian is C
    contributions         X * (a r)[:, None]   row i is a_i r_i x_i'; they sum to S

Signs: the literal Jacobian is ``-C`` (``jacobian``), the expected Jacobian ``-H``
(``expected_jacobian``) and the gradient of the criterion ``-S``.  The curvatures
coincide for every outcome only at ``tau = 1`` and ``kappa = 0``; neither stands in
for the other, and ``C`` is neither floored nor regularized where it is singular or
indefinite (zero rows at ``tau = 2`` have no curvature).  The aggregates are sums,
never divided by the number of rows or by the total weight: a factor common to ``S``
and ``C`` would leave the roots and the Newton step ``C^-1 S`` unchanged, but would
multiply by its square root the scaled score ``||C^(-1/2) S||`` that the certificate
measures where ``C`` is positive definite.

Weights.  ``a`` enters alike for the three kinds of the design; the kinds differ in
the meat and the counts, which are not formed here.  An evaluation keeps the raw ``r``
and ``a`` so that frequency weights can enter a meat as replicated rows, with
``a_i r_i^2`` and not the square of a contribution.  Rows of zero weight are
validated and keep their position (:class:`SamplingUnits`), but they enter no index,
row or aggregate, so an extreme value there cannot overflow.  No row of positive
weight is left out.

Two paths.  ``criterion`` evaluates the criterion of the rows alone, as a line search
needs; the other methods evaluate the rows, never the criterion.  Either can be
representable where the other is not (:mod:`powermean.model.rows`).  A requested
aggregate that is not representable in float64 raises ``FloatingPointError``: the
weighted row term is formed first, then its products with the design and their sum,
and nothing is rescaled or clipped.

The weight is inside the row term in every product: ``X' (a r)``,
``X' (X * (a c)[:, None])`` and ``X * (a r)[:, None]``.  ``evaluate`` returns a
:class:`JointEvaluation`, not the rows alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.model.rows import (
    RowEvaluation,
    _immutable,
    evaluate_criterion,
    evaluate_rows,
)
from powermean.model.variance import (
    FloatArray,
    NB2Variance,
    PowerVariance,
    _finite_array,
    _finite_result,
)

IndexArray = NDArray[np.intp]


# ----------------------------------------------------------------------- the inputs


def _outcome_and_design(
    z: ArrayLike, design: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """A nonnegative vector ``z`` and a matrix with one row per value."""
    outcome = _finite_array(z, "z")
    matrix = _finite_array(design, "design")
    if outcome.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != outcome.size:
        raise ValueError("z must be a vector aligned with the rows of design")
    if not outcome.size or not matrix.shape[1]:
        raise ValueError("z and design must be nonempty, and design must have a column")
    if np.any(outcome < 0):
        raise ValueError("z must be nonnegative")
    return outcome, matrix


def _offset(offset: ArrayLike | None, rows: int) -> FloatArray:
    """The transformed offset ``o``: zero if omitted, a scalar for every row."""
    if offset is None:
        return np.zeros(rows)
    shift = _finite_array(offset, "offset")
    if shift.ndim == 0:
        shift = np.full(rows, float(shift))
    if shift.shape != (rows,):
        raise ValueError("offset must be a scalar or a vector aligned with z")
    return shift


def _weights(weights: ArrayLike | None, rows: int) -> FloatArray:
    """The weights ``a``: nonnegative, one per row, not all zero; one if omitted."""
    if weights is None:
        return np.ones(rows)
    values = _finite_array(weights, "weights")
    if values.shape != (rows,):
        raise ValueError("weights must be a vector aligned with z")
    if np.any(values < 0):
        raise ValueError("weights must be nonnegative")
    if not np.any(values > 0):
        raise ValueError("weights must have a positive value")
    return values


def _positions(mask: NDArray[np.bool_]) -> IndexArray:
    """The positions at which ``mask`` holds, read-only, in bytes of their own."""
    found = np.flatnonzero(mask).tobytes()
    positions: IndexArray = np.frombuffer(found, dtype=np.intp)
    return positions


# --------------------------------------------------------------------- the formulas


def _negative(matrix: FloatArray) -> FloatArray:
    """``-matrix``, read-only: ``-C`` or ``-H``, where ``C`` may be indefinite."""
    return _immutable(-matrix)


def _weighted_moment(
    design: FloatArray, weights: FloatArray, residual: FloatArray
) -> FloatArray:
    """``S = X'(a r)``; the weight multiplies ``r`` first."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value: FloatArray = design.T @ (weights * residual)
    return _immutable(_finite_result(value, "moment"))


def _weighted_product(
    design: FloatArray, weights: FloatArray, row_weight: FloatArray, name: str
) -> FloatArray:
    """``X' (X * (a v)[:, None])``: ``C`` or ``H``."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value: FloatArray = design.T @ (design * (weights * row_weight)[:, None])
    return _immutable(_finite_result(value, name))


def _weighted_contributions(
    design: FloatArray, weights: FloatArray, residual: FloatArray
) -> FloatArray:
    """The rows ``a_i r_i x_i'``: ``X * (a r)[:, None]``."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value: FloatArray = design * (weights * residual)[:, None]
    return _immutable(_finite_result(value, "contribution matrix"))


def _weighted_criterion(weights: FloatArray, terms: FloatArray) -> float:
    """``L = sum_i a_i l_i``, the criterion of the joint system, with weights."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total = float(np.sum(weights * terms))
    if not math.isfinite(total):
        raise FloatingPointError("criterion is not representable in float64")
    return total


# ---------------------------------------------------------------------- the records


@dataclass(frozen=True, eq=False)
class SamplingUnits:
    """The sampling units of the joint system without effects: its active rows.

    ``identifiers`` are the positions among the input rows of the rows of positive
    weight, in increasing order: the ``i``-th contribution, and the ``i``-th entry of
    every row array of an evaluation, belong to the input row ``identifiers[i]``.
    ``inactive`` are the positions of the rows of zero weight, which enter no
    arithmetic; ``nobs`` counts every input row.  Read-only arrays.  The alignment of
    one system, not the sample accounting of the interface.
    """

    identifiers: IndexArray
    inactive: IndexArray
    nobs: int


@dataclass(frozen=True, eq=False)
class JointEvaluation:
    """One evaluation of the joint system: rows, weights, sums and contributions.

    Row arrays are on the active rows, in the order of ``units.identifiers``, under the
    names ``log_mean`` (``q``), ``residual`` (``r``), ``curvature`` (``c``) and
    ``expected_curvature`` (``w``), unweighted, beside the raw ``weights``.
    ``moment``, ``bread`` and ``information`` are ``S``, ``C`` and ``H``; ``jacobian``
    is ``-C`` and ``expected_jacobian`` is ``-H``.  No criterion
    (:meth:`JointSystem.criterion`).  Read-only arrays.
    """

    gamma: FloatArray
    log_mean: FloatArray
    residual: FloatArray
    curvature: FloatArray
    expected_curvature: FloatArray
    weights: FloatArray
    moment: FloatArray
    bread: FloatArray
    information: FloatArray
    contributions: FloatArray
    units: SamplingUnits

    @property
    def jacobian(self) -> FloatArray:
        """``-C``, the literal Jacobian ``dS/dgamma'`` of the moment."""
        return _negative(self.bread)

    @property
    def expected_jacobian(self) -> FloatArray:
        """``-H``, the expected Jacobian of the moment."""
        return _negative(self.information)


@runtime_checkable
class EstimatingSystem(Protocol):
    """An estimating system on dense arrays without effects.

    ``dimension`` and ``units``; ``evaluate`` returns the moment, its observed and
    expected Jacobians and the unit contributions, and ``criterion`` the criterion
    alone, whose gradient is minus the moment.  Separate calls: a line search needs the
    criterion only, and the derivatives do not need it.
    """

    @property
    def dimension(self) -> int:
        """The number of coefficients."""
        ...

    @property
    def units(self) -> SamplingUnits:
        """The sampling units, to which the contributions are aligned."""
        ...

    def evaluate(self, theta: ArrayLike, /) -> JointEvaluation:
        """Moment, Jacobians and contributions at ``theta``, without the criterion."""
        ...

    def criterion(self, theta: ArrayLike, /) -> float:
        """The criterion at ``theta``, alone."""
        ...


# ----------------------------------------------------------------------- the system


@dataclass(frozen=True, eq=False, init=False)
class JointSystem:
    """The joint system without effects, on dense arrays.

    Parameters
    ----------
    z : array
        Transformed outcome, nonnegative and finite, one value per row.
    design : array
        Dense design ``X``, finite, one row per value of ``z`` and at least one column.
        No column is removed for its rank.
    variance : PowerVariance or NB2Variance
        The working variance, of exact type, as :mod:`powermean.model.rows` requires.
    offset : float or array, optional
        Transformed offset ``o``: zero if omitted, the same value for every row if a
        scalar, else one value per row.
    weights : array, optional
        Fixed nonnegative observation weights ``a``, one per row and not all zero; one
        for every row if omitted.

    Notes
    -----
    The inputs are stored validated and resolved, as owned read-only arrays; the
    ``active_`` fields hold the rows of positive weight, on which every evaluation
    runs.  Invalid inputs raise ``ValueError``, another variance type ``TypeError``.
    """

    z: FloatArray = field(repr=False)
    design: FloatArray = field(repr=False)
    variance: PowerVariance | NB2Variance
    offset: FloatArray = field(repr=False)
    weights: FloatArray = field(repr=False)
    units: SamplingUnits = field(repr=False)
    active_z: FloatArray = field(repr=False)
    active_design: FloatArray = field(repr=False)
    active_offset: FloatArray = field(repr=False)
    active_weights: FloatArray = field(repr=False)

    def __init__(
        self,
        z: ArrayLike,
        design: ArrayLike,
        *,
        variance: PowerVariance | NB2Variance,
        offset: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> None:
        if type(variance) not in (PowerVariance, NB2Variance):
            raise TypeError("variance must be a supported built-in working variance")
        outcome, matrix = _outcome_and_design(z, design)
        values = _weights(weights, outcome.size)
        given = {
            "z": outcome,
            "design": matrix,
            "offset": _offset(offset, outcome.size),
            "weights": values,
        }
        active = values > 0
        for name, value in given.items():
            owned = _immutable(value)
            kept = owned if active.all() else _immutable(owned[active])
            object.__setattr__(self, name, owned)
            object.__setattr__(self, f"active_{name}", kept)
        units = SamplingUnits(_positions(active), _positions(~active), int(active.size))
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "units", units)

    @property
    def dimension(self) -> int:
        """The number of coefficients: the columns of the design."""
        return int(self.design.shape[1])

    def _coefficients(self, gamma: ArrayLike) -> FloatArray:
        """``gamma`` as a finite vector, one value per column."""
        coefficients = _finite_array(gamma, "gamma")
        if coefficients.shape != (self.dimension,):
            raise ValueError("gamma must match the number of design columns")
        return coefficients

    def _index(self, coefficients: FloatArray) -> FloatArray:
        """``q = X gamma + o`` on the active rows."""
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            q: FloatArray = self.active_design @ coefficients + self.active_offset
        return _finite_result(q, "index")

    def _rows(self, coefficients: FloatArray) -> RowEvaluation:
        """The rows of :func:`powermean.model.rows.evaluate_rows` at the index."""
        q = self._index(coefficients)
        return evaluate_rows(self.active_z, q, variance=self.variance)

    def log_mean(self, gamma: ArrayLike) -> FloatArray:
        """The index ``q = X gamma + o`` of the active rows."""
        return _immutable(self._index(self._coefficients(gamma)))

    def residual(self, gamma: ArrayLike) -> FloatArray:
        """The score residual ``r`` of the active rows, unweighted."""
        return self._rows(self._coefficients(gamma)).score

    def _moment(self, rows: RowEvaluation) -> FloatArray:
        """``S`` from the score residuals of the rows."""
        return _weighted_moment(self.active_design, self.active_weights, rows.score)

    def _bread(self, rows: RowEvaluation) -> FloatArray:
        """``C`` from the curvature weights of the rows."""
        curvature = rows.observed_curvature
        return _weighted_product(
            self.active_design, self.active_weights, curvature, "observed bread"
        )

    def _information(self, rows: RowEvaluation) -> FloatArray:
        """``H`` from the information weights of the rows."""
        weight = rows.fisher_curvature
        return _weighted_product(
            self.active_design, self.active_weights, weight, "expected information"
        )

    def _contributions(self, rows: RowEvaluation) -> FloatArray:
        """The contributions from the score residuals of the rows."""
        return _weighted_contributions(
            self.active_design, self.active_weights, rows.score
        )

    def contributions(self, gamma: ArrayLike) -> FloatArray:
        """The rows ``a_i r_i x_i'`` of the active rows."""
        return self._contributions(self._rows(self._coefficients(gamma)))

    def moment(self, gamma: ArrayLike) -> FloatArray:
        """``S = X'(a r)``, a sum."""
        return self._moment(self._rows(self._coefficients(gamma)))

    def jacobian(self, gamma: ArrayLike) -> FloatArray:
        """``-C``, the literal Jacobian ``dS/dgamma'``."""
        return _negative(self._bread(self._rows(self._coefficients(gamma))))

    def expected_jacobian(self, gamma: ArrayLike) -> FloatArray:
        """``-H``, the expected Jacobian."""
        return _negative(self._information(self._rows(self._coefficients(gamma))))

    def evaluate(self, gamma: ArrayLike) -> JointEvaluation:
        """The rows, the moment, both Jacobians and the contributions at ``gamma``.

        One evaluation of the rows serves every aggregate; the criterion is not
        evaluated.  Raises ``FloatingPointError`` where the index, a row or an
        aggregate is not representable in float64.
        """
        coefficients = self._coefficients(gamma)
        rows = self._rows(coefficients)
        return JointEvaluation(
            gamma=_immutable(coefficients),
            log_mean=rows.log_mean,
            residual=rows.score,
            curvature=rows.observed_curvature,
            expected_curvature=rows.fisher_curvature,
            weights=self.active_weights,
            moment=self._moment(rows),
            bread=self._bread(rows),
            information=self._information(rows),
            contributions=self._contributions(rows),
            units=self.units,
        )

    def criterion(self, gamma: ArrayLike) -> float:
        """``L = sum_i a_i l_i`` at ``gamma``, from the criterion of the rows alone.

        No score, curvature or information weight is evaluated.  Raises
        ``FloatingPointError`` where the index, the criterion of a row or their
        weighted sum is not representable in float64.
        """
        q = self._index(self._coefficients(gamma))
        terms = evaluate_criterion(self.active_z, q, variance=self.variance)
        return _weighted_criterion(self.active_weights, terms)
