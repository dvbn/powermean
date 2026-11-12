"""The reported scale of the certificate under a named HC covariance, at the fit.

The coefficient gate measures the coefficient correction in units of its standard
error.  The initial coefficient gate reads the standard errors of the covariance
requested at the fit: for a named HC method, that
method's covariance at the certificate's candidate.  The providers below supply them
through the callbacks of the solve stage, as :mod:`powermean.inference.provider` and
:mod:`powermean.inference.linear_provider` supply the robust and one-way ones; the
solve stage imports neither.

:class:`HCScale` serves the joint system without effects, on every equation that the
fit certifies.  The convex initializer and every continuation stage of a selected
root outside ``[1, 2]`` hold the rows, the design, the offset and the weights of the
target and differ from it in the working variance alone
(:func:`powermean.solve.joint_root.same_system`), so the provider holds those rows and
reads the rest from the evaluation it receives: its expected curvature gives the hats,
its observed Jacobian, or the expected one, the bread.  Each call binds the
evaluation to the held rows (its raw weights, and its contributions formed again from
the held design and its residuals, bit for bit) and forms the hats, the influence rows
and the adjusted meat afresh by the internal composition of
:mod:`powermean.inference.hc`.  The method's factor and the reference are those of the
request's preflight (:func:`~powermean.inference.hc.hc_preflight`), which the fit forms
once on its rows before it solves, so that a refusal costs no solve; the preflight
depends on the design, the weights, the method and the convention alone, which every
equation of the fit shares.

:class:`LinearHCScale` serves the linear path: the
:func:`~powermean.inference.hc.linear_hc` of the solve captured before it ran
(:class:`~powermean.inference.hc.LinearSolve`), at the fresh evaluation's point, which
must be the captured candidate's (its coefficients, weights and units, bit for bit).

The standard errors are those of the direct map of the order scale
(:func:`~powermean.inference.ssc.direct_errors`): zero where a variance is zero,
NaN where it is negative, nothing floored.  An error of the covariance (a computed hat
at one on a row that is not structural, a value out of range) is raised, not caught,
and nothing of one call is kept for another.  The handoff forms the fit-time
covariance again by the entry points of :mod:`powermean.inference.hc` and requires the
certificate's standard errors to be its own, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from powermean.data.inference_requests import HC_METHODS, HCMethod
from powermean.inference.bread import BREADS, Bread
from powermean.inference.hc import (
    HCCovariance,
    LinearSolve,
    _joint_composition,
    linear_hc,
)
from powermean.inference.ssc import AppliedConvention, ResolvedConvention, direct_errors
from powermean.model.order_scale import OrderScale
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array
from powermean.solve.linear_evaluation import LinearEvaluation
from powermean.solve.reported import ReportedScale
from powermean.systems.joint import JointEvaluation, _weighted_contributions

NOT_ROWS = "the evaluation's scores are not those of the fit's rows"
NOT_POINT = "the evaluation is not that of the captured solve's point"


def _same(found: NDArray[Any] | None, given: NDArray[Any] | None) -> bool:
    """Whether two arrays have one shape and the same bytes; never for an absent one."""
    if found is None or given is None:
        return False
    return found.shape == given.shape and found.tobytes() == given.tobytes()


@dataclass(frozen=True, eq=False)
class HCScale:
    """The reported scale at the candidate of a joint evaluation, by a named HC method.

    Parameters
    ----------
    applied : AppliedConvention
        The convention of the request applied to the fit's counts by its preflight:
        the method, its own factor and the reference of the convention in force.
    design : array
        The active design of the system that the fit solves, one row per row of
        positive weight.
    weights : array
        The raw weights of those rows.
    frequency : bool
        Whether the weights are frequency counts.
    order : OrderScale
        The order scale of the candidate's coordinates, the direct map.
    bread : {"observed", "expected"}, default "observed"
        The bread of the influence rows; the hats are those of the expected
        information either way.

    Notes
    -----
    An instance is a :class:`~powermean.solve.reported.ReportedScaleProvider`.  The
    arrays are stored as owned read-only copies; another type of argument raises
    ``TypeError``, misaligned arrays or another bread ``ValueError``.
    """

    applied: AppliedConvention
    design: FloatArray
    weights: FloatArray
    frequency: bool
    order: OrderScale
    bread: Bread = "observed"

    def __post_init__(self) -> None:
        """Check the arguments and own the rows.

        The values are read as ``object``: this is the run-time check of what the
        fields declare.
        """
        applied: object = self.applied
        if type(applied) is not AppliedConvention:
            raise TypeError("applied must be an AppliedConvention")
        if self.applied.method is None or self.applied.kind != "robust":
            raise ValueError("the applied convention must name an HC method")
        frequency: object = self.frequency
        if not isinstance(frequency, bool):
            raise TypeError("frequency must be a bool")
        order: object = self.order
        if type(order) is not OrderScale:
            raise TypeError("order must be an OrderScale")
        if self.bread not in BREADS:
            raise ValueError("bread must be 'observed' or 'expected'")
        matrix = _finite_array(self.design, "design")
        a = _finite_array(self.weights, "weights")
        if matrix.ndim != 2 or a.shape != (matrix.shape[0],):
            raise ValueError("design and weights must be aligned")
        if self.order.dimension != matrix.shape[1]:
            raise ValueError("order must have one scale per column of the design")
        object.__setattr__(self, "design", _immutable(matrix))
        object.__setattr__(self, "weights", _immutable(a))

    def covariance(self, evaluation: JointEvaluation) -> HCCovariance:
        """The named HC covariance of the evaluation's equation at its point.

        Raises
        ------
        TypeError
            ``evaluation`` is not a ``JointEvaluation``.
        ValueError
            The evaluation's weights or contributions are not those of the held rows.
        """
        if type(evaluation) is not JointEvaluation:
            raise TypeError("evaluation must be a JointEvaluation")
        scores = _weighted_contributions(self.design, self.weights, evaluation.residual)
        if not (
            _same(evaluation.weights, self.weights)
            and _same(evaluation.contributions, scores)
        ):
            raise ValueError(NOT_ROWS)
        return _joint_composition(
            self.design,
            self.weights,
            evaluation,
            self.applied,
            frequency=self.frequency,
            bread=self.bread,
        )

    def __call__(self, evaluation: JointEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, from that evaluation alone."""
        found = self.covariance(evaluation)
        return ReportedScale(
            gamma=evaluation.gamma,
            standard_errors=direct_errors(found.covariance, self.order),
            order=self.order,
        )


@dataclass(frozen=True, eq=False)
class LinearHCScale:
    """The reported scale at the point of a linear evaluation, by a named HC method.

    Parameters
    ----------
    solved : LinearSolve
        The equation captured before its one solve, with the candidate of that solve.
    method : HC method
        The exact method string.
    convention : ResolvedConvention
        The convention in force for the request at the fit.
    order : OrderScale
        The accepted order scale of the point's coordinates, the direct map.

    Notes
    -----
    An instance is a :class:`~powermean.solve.linear_certificate.LinearScaleProvider`;
    another type of argument raises ``TypeError``, another method ``ValueError``.
    """

    solved: LinearSolve
    method: HCMethod
    convention: ResolvedConvention
    order: OrderScale

    def __post_init__(self) -> None:
        """Check the arguments by exact type.

        The values are read as ``object``: this is the run-time check of what the
        fields declare.
        """
        solved: object = self.solved
        if type(solved) is not LinearSolve:
            raise TypeError("solved must be a LinearSolve")
        if self.method not in HC_METHODS:
            raise ValueError(f"unknown HC method {self.method!r}")
        convention: object = self.convention
        if type(convention) is not ResolvedConvention:
            raise TypeError("convention must be a ResolvedConvention")
        order: object = self.order
        if type(order) is not OrderScale:
            raise TypeError("order must be an OrderScale")

    def covariance(self, evaluation: LinearEvaluation) -> HCCovariance:
        """The named HC covariance of the captured solve, at the evaluation's point.

        Raises
        ------
        TypeError
            ``evaluation`` is not a ``LinearEvaluation``.
        ValueError
            The evaluation's point, weights or units are not the captured candidate's.
        """
        if type(evaluation) is not LinearEvaluation:
            raise TypeError("evaluation must be a LinearEvaluation")
        candidate = self.solved.candidate
        if not (
            _same(evaluation.gamma, candidate.coefficients)
            and _same(evaluation.weights, candidate.weights)
            and _same(evaluation.units.identifiers, candidate.units.identifiers)
        ):
            raise ValueError(NOT_POINT)
        return linear_hc(self.solved, self.method, self.convention)

    def __call__(self, evaluation: LinearEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, from the captured solve at that point."""
        found = self.covariance(evaluation)
        return ReportedScale(
            gamma=evaluation.gamma,
            standard_errors=direct_errors(found.covariance, self.order),
            order=self.order,
        )


__all__ = ["HCScale", "LinearHCScale"]
