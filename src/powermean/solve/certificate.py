"""The certificate of a candidate of the joint system without effects.

A solver proposes, the certificate decides: :class:`Certificate` is the independent
verification of a candidate against the original equations.

Fresh evaluation.  :func:`certify` evaluates the original joint system at the
candidate ``gamma``, once and afresh (``JointSystem.evaluate``), and reads nothing of a
solver: no native flag or reason, no cached gradient, hessian, loss or raw prediction.
It evaluates no criterion, which the derivatives do not need and which can be
unrepresentable where they are representable (:mod:`powermean.model.rows`).  From the
sums ``S`` and ``C`` of that evaluation it forms the quantities of
:mod:`powermean.solve.quantities`: the curvature, the scaled score
``||C_sym^(-1/2) S||``, the Newton correction ``p = C^-1 S`` with LAPACK's estimate
``rcond`` of the reciprocal condition number of the equilibrated ``C``, and the index
correction, the root mean square of ``X p`` over the observations of the rows of
positive weight, each counting its frequency weight when the caller states
``frequency``.  The coefficient correction is the reported correction against the
standard errors at the candidate (:mod:`powermean.solve.reported`): ``M p`` for a
dense map, or the direct map of the accepted order scale entry by entry, by the form
that the provider bound to the candidate.  No argument is modified.  The gates are
read through :class:`GateMeasures`, which the fresh linear certificate
(``powermean.solve.linear_certificate``) answers with its own evaluation, so that the
accepted gate arithmetic is shared and no joint evaluation stands in for a linear one.

Gates, in this order, of which a quantity that is NaN passes none:

    evaluation      the candidate is finite and its evaluation representable
    curvature       C_sym is positive definite: its Cholesky factorization exists
    conditioning    the equilibrated C factorizes with rcond > rcond_min, strictly
    score           scaled score <= tolerance
    index           index correction <= tolerance
    coefficient     every reported coefficient correction passes the gate at coef_tol

The strict level takes ``tolerance = tol`` and requires every gate.  The secondary
level is tried only where the strict one fails: it takes ``tolerance =
tol_secondary`` for the score and the index and changes nothing else, so that it too
requires the evaluation, the curvature, the conditioning and the coefficients.  No
ordering of the two tolerances is assumed.  The certificate is accepted when it
passes at one of them; the statuses of a solve are composed from it and from the
solver's native reason in :mod:`powermean.solve.status`.  The effect equations and the
inner accuracy of the certificate concern absorbed effects: a system without
effects has neither, and no gate stands in for them.

Selected roots.  The gate ``curvature`` is the requirement of a system with a convex
criterion, decided by the working variance of the
equation itself (:func:`outside_convex`), never by a caller.  A power variance with
``tau`` outside ``[1, 2]`` is a selected local root, certified
as a regular root and not as a minimum: there the gate is waived where the scaled
score has a metric, and the score is ``||C_sym^(-1/2) S||`` where ``C_sym`` is
positive definite, else ``||L_M^-1 S||`` with the lower factor ``L_M`` of the solve's
fixed positive-definite merit metric, as the cancelled certificate forms it.
Without that factor, or with a
convex variance, nothing changes: the gate and the NaN score stand.  Conditioning,
index, coefficients and every other gate are the same at both kinds of root; the
certificate records whether the gate applied and which metric the score used.

Failures.  A gate that cannot be evaluated does not pass.  Without a representable
evaluation no quantity is formed.  Where ``C_sym`` is not positive definite the scaled
score is undefined (NaN), while the correction can still be formed.  Where the
factorization fails (an exact zero row or column, an exactly singular factor, or
``rcond <= rcond_min``), no correction is formed: the index
and coefficient corrections are then missing and the provider of the reported scale
is not called.  Nothing is repaired: no ridge, floor, pseudoinverse or expected
curvature takes the place of ``C``.  The certificate keeps the candidate, the
evaluation, every quantity formed, the text of each error and the gates that failed
at each level.  Without a provider of the reported scale the coefficient gate fails:
missing standard errors are never a success.  A provider must return the record of
the very candidate that it received; a record of another candidate is refused with
``ValueError``.  An exception of the provider, as of the system, is not caught,
except the ``FloatingPointError`` of an evaluation that is not representable.

Assessment of the retry.  :class:`CertificateAssessor` is the
:class:`~powermean.solve.assessment.Assessor` of the solver: it receives the system,
the coefficients and the controls of the certificate, and no native detail, and
returns :func:`assessment` of the certificate: ``accepted`` when it passes at either
level, ``score`` the scaled score, and ``error`` the larger of the scaled score and the
index correction, NaN when either is NaN.  The rule of the retry, with its factor
0.99, and the budgets of the solve are the solver's, unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NamedTuple, Protocol, get_args

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, PowerVariance, _finite_array
from powermean.solve.assessment import Assessment, CertificateControls
from powermean.solve.factorization import forward_solve
from powermean.solve.quantities import (
    curvature,
    euclidean_norm,
    index_correction,
    newton_step,
)
from powermean.solve.reported import (
    CoefficientCheck,
    ReportedScale,
    ReportedScaleProvider,
    coefficient_check,
)
from powermean.systems.joint import JointEvaluation, JointSystem

GateName = Literal[
    "evaluation", "curvature", "conditioning", "score", "index", "coefficient"
]
GATES: tuple[GateName, ...] = get_args(GateName)
Level = Literal["strict", "secondary"]
# The metric of the scaled score: the symmetrized observed bread where it is positive
# definite, the fixed merit metric of a selected root's solve, or none (NaN score).
ScoreMetric = Literal["current_symmetric_hessian", "fixed_merit", "none"]
NOT_FINITE = "the candidate is not finite"


def outside_convex(variance: object) -> bool:
    """Whether ``variance`` is a power variance with ``tau`` outside ``[1, 2]``.

    The joint criterion is convex for NB2 and for the power family on ``[1, 2]``;
    outside it the equation's root is a selected local root, and the certificate's
    curvature gate does not apply (module docstring).
    """
    return isinstance(variance, PowerVariance) and not 1.0 <= variance.tau <= 2.0


class GateMeasures(Protocol):
    """What :func:`failed_gates` reads of the measures of a candidate.

    The measures of the joint system (:class:`Measures`) and those of the linear
    path (``powermean.solve.linear_certificate``) both answer it: whether a
    representable evaluation exists, the two factorization gates, the two scaled
    quantities and the coefficient gate.
    """

    @property
    def evaluated(self) -> bool:
        """Whether the candidate is finite and its evaluation representable."""
        ...

    @property
    def positive_definite(self) -> bool:
        """Whether ``C_sym`` has a Cholesky factor."""
        ...

    @property
    def conditioned(self) -> bool:
        """Whether the equilibrated ``C`` factorized with ``rcond > rcond_min``."""
        ...

    @property
    def scaled_score(self) -> float:
        """``||C_sym^(-1/2) S||``, NaN where undefined."""
        ...

    @property
    def index_correction(self) -> float:
        """The root mean square of the index correction, NaN where undefined."""
        ...

    @property
    def coefficients(self) -> CoefficientCheck | None:
        """The coefficient gate, with a correction and a provider."""
        ...


class Measures(NamedTuple):
    """What the gates read: the evaluation and the quantities formed from it.

    ``conditioned`` says that the factorization passed its test, and ``correction``
    is formed only then; ``reported`` and ``coefficients`` only with a correction and
    a provider.  A quantity that was not formed is NaN or ``None``.
    """

    evaluation: JointEvaluation | None
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


UNEVALUATED = Measures(
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


def failed_gates(measures: GateMeasures, tolerance: float) -> tuple[GateName, ...]:
    """The gates that ``measures`` do not pass, in their order.

    ``tolerance`` bounds the scaled score and the index correction, which pass when
    they are at most ``tolerance``: a NaN does not.  Every other gate is the same at
    either level.
    """
    coefficients = measures.coefficients
    passed: dict[GateName, bool] = {
        "evaluation": measures.evaluated,
        "curvature": measures.positive_definite,
        "conditioning": measures.conditioned,
        "score": measures.scaled_score <= tolerance,
        "index": measures.index_correction <= tolerance,
        "coefficient": coefficients is not None and coefficients.accepted,
    }
    return tuple(gate for gate in GATES if not passed[gate])


@dataclass(frozen=True, eq=False)
class Certificate:
    """The certificate of one candidate: its evaluation, quantities and gates.

    Attributes
    ----------
    gamma : array
        The candidate, an owned read-only copy, NaN included.
    controls : CertificateControls
        ``tol``, ``tol_secondary``, ``coef_tol`` and ``rcond_min``.
    frequency : bool
        Whether the weights of the system were frequency weights.
    evaluation : JointEvaluation or None
        The fresh evaluation at the candidate, or ``None`` when the candidate is not
        finite or its evaluation not representable.
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
        What the provider returned at the candidate, if it was called.
    coefficients : CoefficientCheck or None
        The coefficient gate, when a correction and a provider were present.
    errors : mapping
        The text of each error that a step raised, by its gate: ``evaluation``,
        ``curvature`` (the Cholesky factorization) or ``conditioning`` (the LU).
    strict, secondary : tuple of str
        The gates that do not pass at ``tol`` and at ``tol_secondary``.
    require_positive_definite : bool
        Whether the gate ``curvature`` applies: true for a convex working variance,
        false for a selected root outside ``[1, 2]`` (module docstring).
    score_metric : {"current_symmetric_hessian", "fixed_merit", "none"}
        The metric of the scaled score: ``C_sym``, the fixed merit metric of a
        selected root's solve, or none (the score is NaN).
    """

    gamma: FloatArray
    controls: CertificateControls
    frequency: bool
    evaluation: JointEvaluation | None
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
    require_positive_definite: bool = True
    score_metric: ScoreMetric = "none"

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
        """Whether the candidate passes every gate at one of the two levels."""
        return self.level is not None

    @property
    def degenerate(self) -> bool:
        """Whether a reported standard error is zero: degenerate inference."""
        return self.coefficients is not None and self.coefficients.degenerate


def _route_gates(
    gates: tuple[GateName, ...], require: bool, metric: ScoreMetric
) -> tuple[GateName, ...]:
    """The failed gates of the route: curvature waived for a selected root's metric.

    A selected root whose scaled score has a metric does not need ``C_sym`` positive
    definite; without a metric, or for a convex variance, the gates stand as failed.
    """
    if not require and metric != "none":
        return tuple(gate for gate in gates if gate != "curvature")
    return gates


def _certificate(
    candidate: FloatArray,
    controls: CertificateControls,
    frequency: bool,
    measures: Measures,
    errors: Mapping[GateName, str],
    *,
    require: bool,
    metric: ScoreMetric,
) -> Certificate:
    """The record of the measures of a candidate, with its gates at both levels."""
    return Certificate(
        gamma=candidate,
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
        strict=_route_gates(failed_gates(measures, controls.tol), require, metric),
        secondary=_route_gates(
            failed_gates(measures, controls.tol_secondary), require, metric
        ),
        require_positive_definite=require,
        score_metric=metric,
    )


def _scale_of(value: object, evaluation: JointEvaluation) -> ReportedScale:
    """The provider's record, checked: a ``ReportedScale`` of the same candidate.

    Typed as ``object`` because it is the run-time check of what a provider returns.
    """
    if not isinstance(value, ReportedScale):
        raise TypeError("a reported-scale provider must return a ReportedScale")
    if value.gamma.tobytes() != evaluation.gamma.tobytes():
        raise ValueError("the reported scale was formed at another candidate")
    return value


def _coefficients(
    scale: ReportedScale, correction: FloatArray, coef_tol: float
) -> CoefficientCheck:
    """The coefficient gate on the reported corrections, by the map the provider holds.

    The dense ``M p``, or the direct map of the accepted order scale entry by entry
    (:meth:`ReportedScale.corrections`), so that a retry assessment and the final
    certificate report the correction of a mixed-scale candidate without an
    intermediate that overflows or underflows.
    """
    corrections = scale.corrections(correction)
    return coefficient_check(corrections, scale.standard_errors, coef_tol)


def _merit_score(moment: FloatArray, merit: FloatArray) -> float:
    """``||L_M^-1 S||`` by the fixed merit factor, as the cancelled certificate does."""
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = forward_solve(merit, moment)
    return euclidean_norm(whitened)


def _measures(
    system: JointSystem,
    evaluation: JointEvaluation,
    controls: CertificateControls,
    reported: ReportedScaleProvider | None,
    frequency: bool,
    *,
    merit: FloatArray | None,
) -> tuple[Measures, dict[GateName, str], ScoreMetric]:
    """Every quantity of a representable evaluation, the errors met, the metric.

    The curvature gate's requirement is the variance's (:func:`outside_convex`); a
    selected root whose ``C_sym`` is not positive definite reads its scaled score in
    the fixed merit metric, where the solve formed one, and records no curvature error.
    """
    moment, bread = evaluation.moment, evaluation.bread
    curved = curvature(moment, bread)
    step = newton_step(moment, bread, controls.rcond_min)
    errors: dict[GateName, str] = {}
    scaled = curved.scaled_score
    metric: ScoreMetric = (
        "current_symmetric_hessian" if curved.positive_definite else "none"
    )
    if not curved.positive_definite:
        if merit is not None and outside_convex(system.variance):
            scaled, metric = _merit_score(moment, merit), "fixed_merit"
        else:
            errors["curvature"] = curved.error
    index = math.nan
    scale: ReportedScale | None = None
    check: CoefficientCheck | None = None
    if step.correction is None:
        errors["conditioning"] = step.error
    else:
        index = index_correction(system, step.correction, frequency=frequency)
        if reported is not None:
            scale = _scale_of(reported(evaluation), evaluation)
            check = _coefficients(scale, step.correction, controls.coef_tol)
    measures = Measures(
        evaluation=evaluation,
        positive_definite=curved.positive_definite,
        conditioned=step.correction is not None,
        rcond=step.rcond,
        correction=step.correction,
        scaled_score=scaled,
        index_correction=index,
        reported=scale,
        coefficients=check,
    )
    return measures, errors, metric


# ----------------------------------------------------------------- the certificate


def _check(system: object, controls: object, frequency: object) -> None:
    """The types of the arguments of :func:`certify`, each refused with ``TypeError``.

    Typed as ``object`` because it is the run-time check of what a caller passes.
    """
    if type(system) is not JointSystem:
        raise TypeError("system must be a JointSystem")
    if type(controls) is not CertificateControls:
        raise TypeError("controls must be CertificateControls")
    if not isinstance(frequency, bool):
        raise TypeError("frequency must be a bool")


def _candidate(system: JointSystem, gamma: ArrayLike) -> FloatArray:
    """``gamma`` as an owned read-only vector of the system's length, NaN allowed."""
    if np.iscomplexobj(gamma):
        raise ValueError("gamma must be real-valued")
    values = np.asarray(gamma, dtype=np.float64)
    if values.shape != (system.dimension,):
        raise ValueError("gamma must match the number of design columns")
    return _immutable(values)


def _merit(system: JointSystem, merit_cholesky: ArrayLike | None) -> FloatArray | None:
    """The fixed merit factor: finite, lower triangular, with positive diagonal.

    Refused (``ValueError``) for a convex variance, whose gate never reads it, and
    for a matrix of another size.
    """
    if merit_cholesky is None:
        return None
    if not outside_convex(system.variance):
        raise ValueError("a merit metric scales the score of a selected root only")
    factor = _finite_array(merit_cholesky, "merit_cholesky")
    if factor.shape != (system.dimension, system.dimension):
        raise ValueError("merit_cholesky must be square, one row per coefficient")
    if not np.array_equal(factor, np.tril(factor)) or not np.all(np.diag(factor) > 0):
        raise ValueError(
            "merit_cholesky must be lower triangular with positive diagonal"
        )
    return _immutable(factor)


def certify(
    system: JointSystem,
    gamma: ArrayLike,
    controls: CertificateControls,
    *,
    reported: ReportedScaleProvider | None,
    frequency: bool,
    merit_cholesky: ArrayLike | None = None,
) -> Certificate:
    """The certificate of the candidate ``gamma`` of ``system``.

    Parameters
    ----------
    system : JointSystem
        The original system, evaluated afresh at the candidate.
    gamma : array_like
        The candidate, one real value per column of the design.
    controls : CertificateControls
        ``tol``, ``tol_secondary``, ``coef_tol`` and ``rcond_min``.
    reported : ReportedScaleProvider or None
        The reported scale and the standard errors at the candidate, called once
        with the fresh evaluation when a correction is formed; ``None`` leaves the
        coefficient gate failed.
    frequency : bool
        Whether the weights of the system are frequency weights, the number of
        observations of each row; ``False`` for analytic or probability weights and
        for a system without weights.
    merit_cholesky : array_like, optional
        The lower factor ``L_M`` of the fixed merit metric of a selected root's
        solve (module docstring), for a power variance outside ``[1, 2]`` only.

    Returns
    -------
    Certificate
        The evaluation, the quantities, the errors and the gates at both levels.

    Raises
    ------
    TypeError
        An argument is not of its type, or the provider returns another record.
    ValueError
        ``gamma`` is not real or not of the length of the design, the provider's
        record names another candidate, or a merit factor that is not finite, not of
        the system's size, not lower triangular with positive diagonal, or given for
        a convex variance.
    """
    _check(system, controls, frequency)
    candidate = _candidate(system, gamma)
    merit = _merit(system, merit_cholesky)
    require = not outside_convex(system.variance)
    if not np.isfinite(candidate).all():
        return _certificate(
            candidate,
            controls,
            frequency,
            UNEVALUATED,
            {"evaluation": NOT_FINITE},
            require=require,
            metric="none",
        )
    try:
        evaluation = system.evaluate(candidate)
    except FloatingPointError as error:
        return _certificate(
            candidate,
            controls,
            frequency,
            UNEVALUATED,
            {"evaluation": str(error)},
            require=require,
            metric="none",
        )
    measures, errors, metric = _measures(
        system, evaluation, controls, reported, frequency, merit=merit
    )
    return _certificate(
        candidate, controls, frequency, measures, errors, require=require, metric=metric
    )


# ------------------------------------------------------------ the retry of the solver


def assessment(certificate: Certificate) -> Assessment:
    """What the retry of the solver reads of a certificate.

    ``accepted`` when the certificate passes at either level, ``score`` the scaled
    score, ``error`` the larger of the scaled score and the index correction, NaN
    when either is NaN (Python's ``max`` would keep the first of a NaN pair).
    """
    score, index = certificate.scaled_score, certificate.index_correction
    error = math.nan if math.isnan(score) or math.isnan(index) else max(score, index)
    return Assessment(accepted=certificate.accepted, score=score, error=error)


@dataclass(frozen=True)
class CertificateAssessor:
    """The assessor of the retry of the convex Newton solver, by the certificate.

    Parameters
    ----------
    reported : ReportedScaleProvider or None
        The reported scale and the standard errors at a candidate.
    frequency : bool
        Whether the weights of the system are frequency weights.

    Notes
    -----
    An instance is an :class:`~powermean.solve.assessment.Assessor`: called with the
    system, a point and the controls of the certificate, it certifies the point afresh
    and returns its :func:`assessment`.  It keeps no state between calls.
    """

    reported: ReportedScaleProvider | None
    frequency: bool

    def __call__(
        self,
        system: JointSystem,
        gamma: FloatArray,
        controls: CertificateControls,
        /,
    ) -> Assessment:
        """The assessment of ``gamma`` on ``system`` with the controls given."""
        certificate = certify(
            system,
            gamma,
            controls,
            reported=self.reported,
            frequency=self.frequency,
        )
        return assessment(certificate)
