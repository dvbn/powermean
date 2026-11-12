"""The fresh certificate of a point of the linear path, by the accepted gates.

A public log OLS fit is certified
before it is reported: :func:`certify_linear` evaluates the original linear equation
afresh at the proposed point (:mod:`powermean.solve.linear_evaluation`), from the
retained log response, raw offset, scaled design and typed weights, and reads nothing
of the least-squares candidate: not its residuals, not its rank, not its singular
values.  Full column rank is the condition of the solve, not a certificate, and the
cutoff of the singular values is not the conditioning gate.

From the fresh sums ``S`` and ``C`` the quantities are those of
:mod:`powermean.solve.quantities`, unchanged: the curvature gate and the scaled score
``||C_sym^(-1/2) S||`` (:func:`~powermean.solve.quantities.curvature`), the Newton
correction ``p = C^-1 S`` with LAPACK's estimate of the reciprocal condition number of
the equilibrated ``C`` (:func:`~powermean.solve.quantities.newton_step`), the index
correction as the root mean square of ``X p`` over the observations of the rows of
positive weight, each counting its frequency weight when the caller states
``frequency`` (:func:`~powermean.solve.quantities.root_mean_square`), and the
coefficient gate on the reported corrections against the standard errors at
the point (:func:`~powermean.solve.reported.coefficient_check`), which a provider of
the inference stage supplies through the callback :class:`LinearScaleProvider`,
called once with the fresh evaluation when a correction is formed, and whose record
must name the very point.  The gates, their order, the strict and secondary levels,
the equality bounds and the threshold of the conditioning are those of
:func:`powermean.solve.certificate.failed_gates`, with the same
:class:`~powermean.solve.assessment.CertificateControls`: nothing is added, nothing
is relaxed, and a point whose fresh score is not at the tolerance fails whatever the
solve reported.  Where the equilibrated ``C`` fails its test of conditioning no
correction is formed, the index and the coefficient corrections are missing and the
provider is not called, even at exactly zero residuals: nothing is replaced by a
mathematical zero.  Exactly zero standard errors with an exactly zero correction
certify an exact root only when every other gate passes, and the inference is then
degenerate.

The records mirror those of the joint certificate under the same names, so that the
result layer reads either: :class:`LinearCertificate` is not a
:class:`~powermean.solve.certificate.Certificate` and holds a
:class:`~powermean.solve.linear_evaluation.LinearEvaluation`, never a joint one.
This module imports nothing of the inference stage.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray
from powermean.solve.assessment import CertificateControls
from powermean.solve.certificate import (
    NOT_FINITE,
    GateName,
    Level,
    failed_gates,
)
from powermean.solve.linear_evaluation import LinearEvaluation, LinearProblem
from powermean.solve.quantities import curvature, newton_step, root_mean_square
from powermean.solve.reported import (
    CoefficientCheck,
    ReportedScale,
    coefficient_check,
)


class LinearScaleProvider(Protocol):
    """The reported scale of the coefficients at the point of a linear evaluation."""

    def __call__(self, evaluation: LinearEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, formed from that evaluation only."""
        ...


class LinearMeasures(NamedTuple):
    """What the gates read of a point of the linear path (``GateMeasures``)."""

    evaluation: LinearEvaluation | None
    positive_definite: bool
    conditioned: bool
    rcond: float
    correction: FloatArray | None
    scaled_score: float
    index_correction: float
    reported: ReportedScale | None
    coefficients: CoefficientCheck | None

    @property
    def evaluated(self) -> bool:
        """Whether a representable evaluation exists (the gate ``evaluation``)."""
        return self.evaluation is not None


UNEVALUATED = LinearMeasures(
    evaluation=None,
    positive_definite=False,
    conditioned=False,
    rcond=math.nan,
    correction=None,
    scaled_score=math.nan,
    index_correction=math.nan,
    reported=None,
    coefficients=None,
)


@dataclass(frozen=True, eq=False)
class LinearCertificate:
    """The fresh certificate of one point of the linear path.

    Attributes
    ----------
    gamma : array
        The point, in the coordinates of the solver; an owned read-only copy, NaN
        included.
    controls : CertificateControls
        ``tol``, ``tol_secondary``, ``coef_tol`` and ``rcond_min``.
    frequency : bool
        Whether the weights of the solve were frequency weights.
    evaluation : LinearEvaluation or None
        The fresh evaluation at the point, or ``None`` when the point is not finite
        or its evaluation not representable.
    positive_definite : bool
        Whether ``C_sym`` has a Cholesky factor.
    conditioned : bool
        Whether the equilibrated ``C`` factorized with ``rcond > rcond_min``.
    rcond : float
        The estimate of the reciprocal condition number, NaN when none was formed.
    correction : array or None
        The Newton correction ``p = C^-1 S``, or ``None`` when it was not formed.
    scaled_score, index_correction : float
        The two quantities of the tolerances, NaN when undefined.
    reported : ReportedScale or None
        What the provider returned at the point, if it was called.
    coefficients : CoefficientCheck or None
        The coefficient gate, when a correction and a provider were present.
    errors : mapping
        The text of each error that a step raised, by its gate.
    strict, secondary : tuple of str
        The gates that do not pass at ``tol`` and at ``tol_secondary``.
    """

    gamma: FloatArray
    controls: CertificateControls
    frequency: bool
    evaluation: LinearEvaluation | None
    positive_definite: bool
    conditioned: bool
    rcond: float
    correction: FloatArray | None
    scaled_score: float
    index_correction: float
    reported: ReportedScale | None
    coefficients: CoefficientCheck | None
    errors: Mapping[GateName, str]
    strict: tuple[GateName, ...]
    secondary: tuple[GateName, ...]

    @property
    def level(self) -> Level | None:
        """``"strict"``, else ``"secondary"`` if that level passes, else ``None``."""
        if not self.strict:
            return "strict"
        if not self.secondary:
            return "secondary"
        return None

    @property
    def accepted(self) -> bool:
        """Whether the point passes every gate at one of the two levels."""
        return self.level is not None

    @property
    def degenerate(self) -> bool:
        """Whether a reported standard error is zero: degenerate inference."""
        return self.coefficients is not None and self.coefficients.degenerate


def _certificate(
    point: FloatArray,
    controls: CertificateControls,
    frequency: bool,
    measures: LinearMeasures,
    errors: Mapping[GateName, str],
) -> LinearCertificate:
    """The record of the measures of a point, with its gates at both levels."""
    return LinearCertificate(
        gamma=point,
        controls=controls,
        frequency=frequency,
        evaluation=measures.evaluation,
        positive_definite=measures.positive_definite,
        conditioned=measures.conditioned,
        rcond=measures.rcond,
        correction=measures.correction,
        scaled_score=measures.scaled_score,
        index_correction=measures.index_correction,
        reported=measures.reported,
        coefficients=measures.coefficients,
        errors=MappingProxyType(dict(errors)),
        strict=failed_gates(measures, controls.tol),
        secondary=failed_gates(measures, controls.tol_secondary),
    )


def _scale_of(value: object, evaluation: LinearEvaluation) -> ReportedScale:
    """The provider's record, checked: a ``ReportedScale`` of the same point.

    Typed as ``object`` because it is the run-time check of what a provider returns.
    """
    if not isinstance(value, ReportedScale):
        raise TypeError("a reported-scale provider must return a ReportedScale")
    if value.gamma.tobytes() != evaluation.gamma.tobytes():
        raise ValueError("the reported scale was formed at another point")
    return value


def linear_index_correction(
    problem: LinearProblem, correction: FloatArray, *, frequency: bool
) -> float:
    """The root mean square of ``X p`` over the observations of the active rows.

    ``X`` is the design of the rows of positive weight; each row counts its
    frequency weight of observations when ``frequency`` holds, and one otherwise
    (:func:`powermean.solve.quantities.root_mean_square`, unchanged).
    """
    with np.errstate(over="ignore", invalid="ignore"):
        change = problem.active_design @ correction
    multiplicity = problem.active_weights if frequency else np.ones(change.shape)
    return root_mean_square(change, multiplicity)


def _measures(
    problem: LinearProblem,
    evaluation: LinearEvaluation,
    controls: CertificateControls,
    reported: LinearScaleProvider | None,
    frequency: bool,
) -> tuple[LinearMeasures, dict[GateName, str]]:
    """Every quantity of a representable evaluation, and the errors met."""
    moment, bread = evaluation.moment, evaluation.bread
    curved = curvature(moment, bread)
    step = newton_step(moment, bread, controls.rcond_min)
    errors: dict[GateName, str] = {}
    if not curved.positive_definite:
        errors["curvature"] = curved.error
    index = math.nan
    scale: ReportedScale | None = None
    check: CoefficientCheck | None = None
    if step.correction is None:
        errors["conditioning"] = step.error
    else:
        index = linear_index_correction(problem, step.correction, frequency=frequency)
        if reported is not None:
            scale = _scale_of(reported(evaluation), evaluation)
            check = coefficient_check(
                scale.corrections(step.correction),
                scale.standard_errors,
                controls.coef_tol,
            )
    measures = LinearMeasures(
        evaluation=evaluation,
        positive_definite=curved.positive_definite,
        conditioned=step.correction is not None,
        rcond=step.rcond,
        correction=step.correction,
        scaled_score=curved.scaled_score,
        index_correction=index,
        reported=scale,
        coefficients=check,
    )
    return measures, errors


def _check(problem: object, controls: object, frequency: object) -> None:
    """The types of the arguments of :func:`certify_linear`, each refused.

    Typed as ``object`` because it is the run-time check of what a caller passes.
    """
    if type(problem) is not LinearProblem:
        raise TypeError("problem must be a LinearProblem")
    if type(controls) is not CertificateControls:
        raise TypeError("controls must be CertificateControls")
    if not isinstance(frequency, bool):
        raise TypeError("frequency must be a bool")


def _point(problem: LinearProblem, b: ArrayLike) -> FloatArray:
    """``b`` as an owned read-only vector of the problem's length, NaN allowed."""
    if np.iscomplexobj(b):
        raise ValueError("b must be real-valued")
    values = np.asarray(b, dtype=np.float64)
    if values.shape != (problem.dimension,):
        raise ValueError("b must match the number of design columns")
    return _immutable(values)


def certify_linear(
    problem: LinearProblem,
    b: ArrayLike,
    controls: CertificateControls,
    *,
    reported: LinearScaleProvider | None,
    frequency: bool,
) -> LinearCertificate:
    """The fresh certificate of the point ``b`` of the linear ``problem``.

    Parameters
    ----------
    problem : LinearProblem
        The original linear equation, evaluated afresh at the point.
    b : array_like
        The proposed point, one real value per column of the design.
    controls : CertificateControls
        ``tol``, ``tol_secondary``, ``coef_tol`` and ``rcond_min``.
    reported : LinearScaleProvider or None
        The reported scale and the standard errors at the point, called once with
        the fresh evaluation when a correction is formed; ``None`` leaves the
        coefficient gate failed.
    frequency : bool
        Whether the weights of the solve are frequency weights, the number of
        observations of each row.

    Returns
    -------
    LinearCertificate
        The evaluation, the quantities, the errors and the gates at both levels.

    Raises
    ------
    TypeError
        An argument is not of its type, or the provider returns another record.
    ValueError
        ``b`` is not real or not of the length of the design, or the provider's
        record names another point.
    """
    _check(problem, controls, frequency)
    point = _point(problem, b)
    if not np.isfinite(point).all():
        return _certificate(
            point, controls, frequency, UNEVALUATED, {"evaluation": NOT_FINITE}
        )
    try:
        evaluation = problem.evaluate(point)
    except FloatingPointError as error:
        return _certificate(
            point, controls, frequency, UNEVALUATED, {"evaluation": str(error)}
        )
    measures, errors = _measures(problem, evaluation, controls, reported, frequency)
    return _certificate(point, controls, frequency, measures, errors)
