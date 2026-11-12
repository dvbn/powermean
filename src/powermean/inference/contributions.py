"""The influence record of a candidate: its map, contributions, weights and count.

A candidate ``gamma_hat`` of a sum moment ``S(gamma) = sum_i s_i(gamma)`` has the
first-order representation

    gamma_hat - gamma_0 = A sum_i s_i + o(1),    A = C^-1,    C = -dS/dgamma'

where ``C`` is the observed bread of the equation actually solved, at the candidate:
never the expected information, nor the curvature of an artificial regression.
:class:`Influence` keeps what every covariance of the candidate needs, so that a
covariance can be formed again, with other cluster codes, without refitting: the
candidate, the literal Jacobian ``-C``, the map ``A``, which
:func:`powermean.inference.influence.moment_influence_map` forms from ``-C``, the
contributions ``s_i`` of the sampling units, their raw weights, the units themselves
(the positions of the rows among the input rows), the kind of the weights and the
statistical count ``N``.  ``A s_i`` sums to the first-order change of ``gamma``; the
normalization that the primitives of :mod:`powermean.inference.vcov` expect is formed
by :mod:`powermean.inference.convention`, not here.

Two sources.

- :func:`joint_influence` reads a :class:`~powermean.systems.joint.JointEvaluation`,
  the record that the certificate forms afresh at its candidate: ``-C`` is its
  ``jacobian`` (``C`` its ``bread``), and ``s_i = a_i r_i x_i`` its contributions, on
  its rows of positive weight.  It is the derivative of the sum, not of the average:
  ``-C``, not ``-C/n``.
- :func:`linear_influence` reads a :class:`~powermean.solve.linear.LinearCandidate` of
  the linear path, at full column rank, with the design ``X`` that its solve received
  (every input row, in their order), which the candidate does not hold.  The normal
  equations ``X'(a e) = 0`` give ``C = X' diag(a) X`` and ``s_i = a_i e_i x_i``, from
  the candidate's coefficients and residuals, formed by the products of the joint
  system with a unit curvature.  It is not a ``JointEvaluation``, and it is no fresh
  reassessment of the candidate, which would also need the original response and
  offset: a covariance is never a certificate.

Counts.  ``N`` is the number of observations of the ``"stata"`` factors: for
frequency weights the total of the counts of the rows of positive
weight, summed in integer arithmetic; otherwise the number of those rows.  It is never
``units.nobs``, which counts rows of zero weight, and for frequency weights never the
number of stored rows.  Frequency weights are exact counts, as the integer frequency
weights of Stata: each must be an integer-valued double of at most ``2^53``, and so
must their total; a fractional or larger value raises ``ValueError``, and nothing is
rounded.  The caller states the kind of the weights, which is never inferred from
their values.

Rows of zero weight enter no arithmetic, alignment or count: the evaluation and the
candidate hold only the rows of positive weight, and the design is read on those rows
alone, after a check of every row as the solve checked them.  The arrays of a record
are owned and read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.inference.influence import moment_influence_map
from powermean.inference.numeric import finite_array
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray
from powermean.solve.linear import LinearCandidate
from powermean.systems.joint import (
    JointEvaluation,
    SamplingUnits,
    _weighted_contributions,
    _weighted_product,
)

# The largest count below which every integer is a double: a larger count may be the
# rounding of another, so it is refused, as the dense reference refuses it.
MAXIMUM_COUNT = 2**53


@dataclass(frozen=True, eq=False)
class Influence:
    """What every covariance of one candidate needs, without refitting.

    Attributes
    ----------
    gamma : array
        The candidate, one value per coefficient.
    jacobian : array
        ``-C``, the literal Jacobian of the sum moment at the candidate, observed.
    influence_map : array
        ``A = C^-1``, :func:`~powermean.inference.influence.moment_influence_map` of
        ``-C``.
    contributions : array
        The rows ``s_i``, one per row of positive weight, in the order of
        ``units.identifiers``.
    weights : array
        The raw weights of those rows.
    units : SamplingUnits
        The positions of those rows among the input rows, of the rows of zero weight,
        and the number of input rows.
    frequency : bool
        Whether the weights are frequency weights, as the caller stated.
    count : int
        ``N``: the total of the frequency counts, else the number of rows of positive
        weight.

    Notes
    -----
    The arrays must be aligned with the coefficients and the units; otherwise
    ``ValueError``.  They are stored as given, owned and read-only by their builders.
    """

    gamma: FloatArray
    jacobian: FloatArray
    influence_map: FloatArray
    contributions: FloatArray
    weights: FloatArray
    units: SamplingUnits
    frequency: bool
    count: int

    def __post_init__(self) -> None:
        """Check that the arrays are aligned with the coefficients and the units."""
        size, rows = self.gamma.size, self.units.identifiers.size
        aligned = (
            self.gamma.shape == (size,)
            and self.jacobian.shape == (size, size)
            and self.influence_map.shape == (size, size)
            and self.contributions.shape == (rows, size)
            and self.weights.shape == (rows,)
        )
        if not aligned:
            raise ValueError("the arrays of an influence record are not aligned")


def statistical_count(weights: FloatArray, *, frequency: bool) -> int:
    """``N``: the exact total of frequency counts, else the number of rows.

    Parameters
    ----------
    weights : array
        The raw weights of the rows of positive weight.
    frequency : bool
        Whether they are frequency weights.

    Returns
    -------
    int
        The total of the counts, summed as integers, or the number of rows.

    Raises
    ------
    ValueError
        A frequency weight that is not a positive integer of at most ``2^53``, or a
        total above ``2^53``.
    """
    if not frequency:
        return int(weights.size)
    total = 0
    for value in weights.tolist():
        if not (value.is_integer() and 0 < value <= MAXIMUM_COUNT):
            raise ValueError(
                "frequency weights must be integer counts of at most 2**53"
            )
        total += int(value)
    if total > MAXIMUM_COUNT:
        raise ValueError("the total frequency count must not exceed 2**53")
    return total


def _check_frequency(frequency: object) -> None:
    """``frequency`` is a ``bool``; the kind of the weights is never inferred."""
    if not isinstance(frequency, bool):
        raise TypeError("frequency must be a bool")


def joint_influence(evaluation: JointEvaluation, *, frequency: bool) -> Influence:
    """The influence record of an evaluation of the joint system without effects.

    Parameters
    ----------
    evaluation : JointEvaluation
        The evaluation at the candidate, as the certificate formed it afresh.
    frequency : bool
        Whether the weights of the system are frequency weights.

    Returns
    -------
    Influence
        ``-C`` from the observed bread, its map, the contributions ``a_i r_i x_i``,
        the raw weights, the units and ``N``.

    Raises
    ------
    TypeError
        ``evaluation`` is not a ``JointEvaluation`` or ``frequency`` not a ``bool``.
    ValueError
        Invalid frequency counts, or a map that cannot be formed or represented.
    numpy.linalg.LinAlgError
        ``C`` is not of full rank.
    """
    if type(evaluation) is not JointEvaluation:
        raise TypeError("evaluation must be a JointEvaluation")
    _check_frequency(frequency)
    count = statistical_count(evaluation.weights, frequency=frequency)
    jacobian = evaluation.jacobian
    return Influence(
        gamma=evaluation.gamma,
        jacobian=jacobian,
        influence_map=moment_influence_map(jacobian),
        contributions=evaluation.contributions,
        weights=evaluation.weights,
        units=evaluation.units,
        frequency=frequency,
        count=count,
    )


def _active_design(design: ArrayLike, candidate: LinearCandidate) -> FloatArray:
    """The rows of positive weight of the design of the solve, checked whole first."""
    matrix = finite_array(design, "design")
    if matrix.shape != (candidate.units.nobs, candidate.dimension):
        raise ValueError(
            "design must have one row per input row of the solve "
            "and one column per coefficient"
        )
    active: FloatArray = matrix[candidate.units.identifiers]
    return active


def linear_influence(
    candidate: LinearCandidate, design: ArrayLike, *, frequency: bool
) -> Influence:
    """The influence record of a candidate of the linear path, with its design.

    Parameters
    ----------
    candidate : LinearCandidate
        The candidate of :func:`powermean.solve.linear.solve`, at full column rank.
    design : array_like
        The design ``X`` of that solve: one finite real row per input row, in their
        order, and one column per coefficient.
    frequency : bool
        Whether the weights of the solve are frequency weights.

    Returns
    -------
    Influence
        ``-C`` with ``C = X' diag(a) X``, its map, the contributions ``a_i e_i x_i``
        from the candidate's residuals, the raw weights, the units and ``N``.

    Raises
    ------
    TypeError
        ``candidate`` is not a ``LinearCandidate`` or ``frequency`` not a ``bool``.
    ValueError
        A candidate without coefficients, a design of another shape or that is not
        finite and real, invalid frequency counts, or a map that cannot be formed.
    FloatingPointError
        ``C`` or a contribution is not representable in float64.
    """
    if type(candidate) is not LinearCandidate:
        raise TypeError("candidate must be a LinearCandidate")
    _check_frequency(frequency)
    coefficients, residuals = candidate.coefficients, candidate.residuals
    if coefficients is None or residuals is None:
        raise ValueError(
            f"a candidate that ended with {candidate.reason!r} has no covariance"
        )
    count = statistical_count(candidate.weights, frequency=frequency)
    active = _active_design(design, candidate)
    weights = candidate.weights
    # The least-squares curvature of every row is one: the products of the joint
    # system with the row weight one give C = X' diag(a) X.
    bread = _weighted_product(active, weights, np.ones(weights.shape), "observed bread")
    jacobian = _immutable(-bread)
    return Influence(
        gamma=coefficients,
        jacobian=jacobian,
        influence_map=moment_influence_map(jacobian),
        contributions=_weighted_contributions(active, weights, residuals),
        weights=weights,
        units=candidate.units,
        frequency=frequency,
        count=count,
    )
