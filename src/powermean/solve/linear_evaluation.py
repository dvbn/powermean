"""The fresh evaluation of the linear path at a proposed point: the original equation.

A solver proposes, the certificate decides, and the certificate is computed after the
solver returns, from fresh evaluations of the original system.
The least-squares solve of :mod:`powermean.solve.linear` returns
a candidate with its own residuals, rank and singular values, none of which certifies
it.  This module evaluates the original linear equation afresh at any proposed
coefficient vector ``b``, from the retained inputs of the solve and nothing of the
candidate:

    e = (l - o) - X b                the residual of the log response net of the
                                     raw offset, on the rows of positive weight
    S = X'(a e)                      the normal-equation score, a sum
    C = X' diag(a) X                 the curvature of the least-squares criterion,
                                     a sum, which does not depend on b
    contributions   a_i e_i x_i'     the row scores, which sum to S

on exactly the rows of positive weight, the ``SamplingUnits`` of the solve; a row of
zero weight enters no arithmetic.  The log response ``l`` is any finite real vector
and is never exponentiated, the raw offset enters with coefficient one and no order
multiplier, a working variance has no role, and the three kinds of weights share the
sums: they differ in the counts of the inference, which the caller states.  The sums
are those of the products of :mod:`powermean.systems.joint` with a unit curvature
weight (``_weighted_moment``, ``_weighted_product``, ``_weighted_contributions``), so
that the linear equation is evaluated by the accepted arithmetic and not by a
``JointEvaluation``, which has a log mean, a working variance and an expected
information that the linear path has not.  A quantity that is not representable in
float64 raises ``FloatingPointError``, as the joint system does; nothing is rescaled
or clipped.

:class:`LinearProblem` holds the retained inputs of one solve, validated as
:func:`powermean.solve.linear.solve` validates them and stored as owned read-only
arrays, with the positions of the rows of positive weight.  :class:`LinearEvaluation`
holds one evaluation: the point, the residuals, the raw weights, the sums, the
contributions and the units, under the names that the consumers of the joint
evaluation read (``gamma`` for the point in the coordinates of the solver, ``moment``,
``bread``, ``contributions``, ``weights``, ``units``), with ``information`` the
expected information of the least-squares equation, which is its curvature ``C``
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array, _finite_result
from powermean.solve.linear import _offset, _response_and_design, _weights
from powermean.systems.joint import (
    SamplingUnits,
    _positions,
    _weighted_contributions,
    _weighted_moment,
    _weighted_product,
)


@dataclass(frozen=True, eq=False)
class LinearEvaluation:
    """One fresh evaluation of the linear equation at a point (module docstring).

    Attributes
    ----------
    gamma : array
        The point ``b``, in the coordinates of the solver.
    residual : array
        ``e = (l - o) - X b`` on the rows of positive weight.
    weights : array
        The raw weights ``a`` of those rows.
    moment : array
        ``S = X'(a e)``.
    bread : array
        ``C = X' diag(a) X``.
    contributions : array
        The rows ``a_i e_i x_i'``.
    units : SamplingUnits
        The positions of the rows of positive weight among the input rows of the
        solve, of the rows of zero weight, and the number of input rows.

    Notes
    -----
    Read-only arrays.  ``information`` is ``C``: the expected information of the
    least-squares equation coincides with its curvature.
    """

    gamma: FloatArray
    residual: FloatArray
    weights: FloatArray
    moment: FloatArray
    bread: FloatArray
    contributions: FloatArray
    units: SamplingUnits

    @property
    def information(self) -> FloatArray:
        """``C`` again: the expected information of a least-squares equation."""
        return self.bread

    @property
    def jacobian(self) -> FloatArray:
        """``-C``, the literal Jacobian ``dS/db'`` of the normal-equation score."""
        return _immutable(-self.bread)


@dataclass(frozen=True, eq=False, init=False)
class LinearProblem:
    """The retained inputs of one least-squares solve, for its fresh evaluation.

    Parameters
    ----------
    response : array_like
        The log response ``l``, one finite real value per input row.
    design : array_like
        The design ``X`` that the solve received (scaled by the preparation), one
        row per input row and at least one column.
    offset : float or array_like, optional
        The raw offset ``o``, with coefficient one: zero if omitted, the same value
        for every row if a scalar, else one finite value per row.
    weights : array_like, optional
        The weights ``a``, one finite nonnegative value per row, not all zero; one
        for every row if omitted.

    Notes
    -----
    Validated as :func:`powermean.solve.linear.solve` validates its inputs, with its
    errors; stored as owned read-only arrays, with the ``active_`` arrays on the rows
    of positive weight and the units of the solve.
    """

    response: FloatArray = field(repr=False)
    design: FloatArray = field(repr=False)
    offset: FloatArray = field(repr=False)
    weights: FloatArray = field(repr=False)
    units: SamplingUnits = field(repr=False)
    active_response: FloatArray = field(repr=False)
    active_design: FloatArray = field(repr=False)
    active_offset: FloatArray = field(repr=False)
    active_weights: FloatArray = field(repr=False)

    def __init__(
        self,
        response: ArrayLike,
        design: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> None:
        values, matrix = _response_and_design(response, design)
        rows = int(values.size)
        given = {
            "response": values,
            "design": matrix,
            "offset": _offset(offset, rows),
            "weights": _weights(weights, rows),
        }
        active = given["weights"] > 0
        for name, value in given.items():
            owned = _immutable(value)
            kept = owned if active.all() else _immutable(owned[active])
            object.__setattr__(self, name, owned)
            object.__setattr__(self, f"active_{name}", kept)
        units = SamplingUnits(_positions(active), _positions(~active), rows)
        object.__setattr__(self, "units", units)

    @property
    def dimension(self) -> int:
        """The number of coefficients: the columns of the design."""
        return int(self.design.shape[1])

    def _coefficients(self, b: ArrayLike) -> FloatArray:
        """``b`` as a finite vector, one value per column."""
        coefficients = _finite_array(b, "b")
        if coefficients.shape != (self.dimension,):
            raise ValueError("b must match the number of design columns")
        return coefficients

    def residual(self, b: ArrayLike) -> FloatArray:
        """``e = (l - o) - X b`` on the rows of positive weight, read-only.

        Raises ``FloatingPointError`` where it is not representable.
        """
        coefficients = self._coefficients(b)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            target: FloatArray = self.active_response - self.active_offset
            residual: FloatArray = target - self.active_design @ coefficients
        return _immutable(_finite_result(residual, "linear residual"))

    def evaluate(self, b: ArrayLike) -> LinearEvaluation:
        """The residuals, the sums and the contributions at ``b`` (module docstring).

        Parameters
        ----------
        b : array_like
            The proposed point, one finite real value per column of the design.

        Returns
        -------
        LinearEvaluation
            The fresh evaluation.

        Raises
        ------
        ValueError
            ``b`` is not finite and real, or not of the length of the design.
        FloatingPointError
            A residual, a sum or a contribution is not representable in float64.
        """
        coefficients = self._coefficients(b)
        residual = self.residual(coefficients)
        design, weights = self.active_design, self.active_weights
        unit = np.ones(weights.shape)
        return LinearEvaluation(
            gamma=_immutable(coefficients),
            residual=residual,
            weights=weights,
            moment=_weighted_moment(design, weights, residual),
            bread=_weighted_product(design, weights, unit, "observed bread"),
            contributions=_weighted_contributions(design, weights, residual),
            units=self.units,
        )
