"""The joint system outside ``tau in [1, 2]``: nearest convex start, then root Newton.

Root Newton: the actual Jacobian by equilibrated LU, the merit ``S' M^-1 S / 2`` with
a fixed positive-definite ``M``, Armijo on differences, native stops, then the
certificate.  A joint fit outside ``[1, 2]`` starts from the certified solution at the
nearer of ``tau = 1`` and ``tau = 2``, with the continuation as fallback, and its result
is a selected local root, certified as a regular root, not as unique.  The nearest
convex start is PPML below 1 and Gamma above 2, on the same sample and formulation;
the cancelled and mixed Gamma start is unchanged, and caller starts are kept.  The
convex Newton solver, its start, options and certificate are unchanged and
never see a variance outside ``[1, 2]``.

The steps of :func:`solve_joint_root`, in order:

1. The initializer.  A caller's start is accepted as given and recorded; it is not
   certified, and no continuation leaves from it.  Otherwise the nearest convex member
   of the family is fitted on the same system (the same rows, scaled design, offset
   and weights; no row dropped, added or reselected) by the unchanged convex Newton
   solver from its unchanged automatic start, and concluded by the unchanged convex
   certificate with the fit's own reported-scale provider.  An automatic start without
   a finite value ends the fit there, as on the convex route.
2. The merit metric.  From a caller's start or a certified initializer, the lower
   Cholesky factor of the fixed metric at that start: ``"gamma"`` (the default), the
   observed bread of the Gamma equation of the same system, ``X' diag(a z/m) X``; or
   ``"gram"``, the Gram matrix ``X'X`` of the active scaled design, as the cancelled
   solver uses the structural Gram.  A factorization that fails refuses the requested
   metric; no other metric is substituted.
3. The root solve.  The shared kernel of :mod:`powermean.solve.root_kernel`: the
   fresh equations of the joint system under each stage variance (the row kernel of
   :mod:`powermean.model.rows` and the weighted sums of :mod:`powermean.systems.joint`),
   the direction ``C delta = S`` with the actual observed bread ``C = -dS/dgamma'``
   by the equilibrated LU of :mod:`powermean.solve.factorization` (never a Cholesky
   factor, ridge or expected curvature), and every stage point certified afresh by
   :func:`powermean.solve.certificate.certify` on that stage's equation with the fixed
   merit factor.  The continuation leaves from the initializer's own ``tau``.
4. The conclusion.  The point of the target stage, certified afresh on the target
   equation, with the status that
   :func:`powermean.solve.status.compose_status` gives that certificate and the
   native reason of the retained attempt. Where no root stage opened (the automatic
   initializer was not certified, or the metric could not be factorized) the
   initializer's point is certified on the target equation for diagnosis without the
   reported scale, since no selected root exists whose standard errors could enter
   the coefficient gate.  The status is then the initializer's own; for a metric
   that cannot be formed at the start it is ``nonfinite`` where the target equation
   is not representable there and ``singular_system`` otherwise, with the native
   reason of the metric's failure (:func:`joint_root_status`).

Budgets.  The convex initializer's iterations and evaluations are charged to
``max_iter`` and ``max_evaluations`` before the root solve receives what remains, as in
the cancelled solver.  Every operation of the root solve is counted in the
candidate's work record.  The selected root is not claimed unique, global or a local
minimum of the quasi-likelihood: the certificate establishes a regular root.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.admission_records import WorkLog
from powermean.model.rows import RowEvaluation, _immutable, evaluate_rows
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    PowerVariance,
    _finite_result,
)
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.cancelled_certificate import whitened_score
from powermean.solve.cancelled_records import MeritMetric, RootOptions
from powermean.solve.certificate import (
    Certificate,
    CertificateAssessor,
    assessment,
    certify,
    outside_convex,
)
from powermean.solve.factorization import factorize, symmetric_cholesky
from powermean.solve.joint_root_records import (
    InitializerKind,
    JointRootCandidate,
    JointRootInitializer,
    JointRootStage,
    StageSummary,
)
from powermean.solve.newton_convex import solve as newton_solve
from powermean.solve.newton_trace import NativeReason, Timing
from powermean.solve.no_point import StartFailure
from powermean.solve.options import SolverOptions
from powermean.solve.reported import ReportedScaleProvider
from powermean.solve.root_kernel import (
    RootBudget,
    RootKernel,
    RootRun,
    StageOutcome,
    Whitened,
    run_continuation,
    run_root_stage,
)
from powermean.solve.starts import Start, automatic_start
from powermean.solve.status import Conclusion, Status, compose_status, conclude
from powermean.systems.joint import JointSystem, _weighted_moment, _weighted_product

# The nearest convex members of the power family.
PPML = PowerVariance(1.0)
GAMMA = PowerVariance(2.0)
NOT_ROOT = "the joint root solve needs a power variance with tau outside [1, 2]"


# ---------------------------------------------------------------- the system


def nearest_convex(variance: object) -> tuple[InitializerKind, PowerVariance]:
    """The nearest convex member of the family: PPML below 1, Gamma above 2."""
    if not isinstance(variance, PowerVariance) or not outside_convex(variance):
        raise ValueError(NOT_ROOT)
    if variance.tau < 1.0:
        return "ppml_tau1", PPML
    return "gamma_tau2", GAMMA


def same_system(system: JointSystem, variance: PowerVariance) -> JointSystem:
    """The joint system of the same rows, design, offset and weights under ``variance``.

    The system itself where ``variance`` is its own; otherwise a system built from its
    stored arrays, so that its active rows and units are the same.
    """
    if variance == system.variance:
        return system
    return JointSystem(
        system.z,
        system.design,
        variance=variance,
        offset=system.offset,
        weights=system.weights,
    )


@dataclass(frozen=True, eq=False)
class JointPoint:
    """One fresh state of the joint equations: the point, its rows and ``S``."""

    theta: FloatArray
    score: FloatArray
    rows: RowEvaluation


def equations_of(
    system: JointSystem,
) -> Callable[[FloatArray, CancelledFamily], JointPoint]:
    """Fresh rows and ``S = X'(a r)`` of the system's rows under a stage variance.

    The arithmetic of :meth:`JointSystem.evaluate`: the same coefficients check,
    index, row kernel and weighted sum, for the stage variance given.
    """

    def equations(theta: FloatArray, variance: CancelledFamily) -> JointPoint:
        coefficients = system._coefficients(theta)
        q = system._index(coefficients)
        rows = evaluate_rows(system.active_z, q, variance=variance)
        score = _weighted_moment(
            system.active_design, system.active_weights, rows.score
        )
        return JointPoint(_immutable(coefficients), score, rows)

    return equations


def _bread(system: JointSystem, rows: RowEvaluation) -> FloatArray:
    """The observed bread ``C = X' diag(a c) X`` of the rows, as the system forms it."""
    return _weighted_product(
        system.active_design,
        system.active_weights,
        rows.observed_curvature,
        "observed bread",
    )


def direction_of(
    system: JointSystem, rcond_min: float, work: WorkLog
) -> Callable[[JointPoint], FloatArray]:
    """The root-Newton step ``C delta = S`` by the equilibrated LU of the actual ``C``.

    ``SingularJacobianError`` (an exact zero row or column, a singular factor or
    ``rcond <= rcond_min``) and ``FloatingPointError`` reach the attempt, which stops
    with ``factorization_failure`` or ``numerical_breakdown``.
    """

    def direction(state: JointPoint) -> FloatArray:
        matrix = work.run("root_bread", lambda: _bread(system, state.rows))
        factor = work.run("root_lu", lambda: factorize(matrix, rcond_min))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = work.run("root_direction_solve", lambda: factor.solve(state.score))
        return _finite_result(delta, "root direction")

    return direction


def _whitener(merit: FloatArray, work: WorkLog) -> Callable[[FloatArray], Whitened]:
    """``u = L_M^-1 S``, its norm and ``phi`` by the fixed factor (cancelled helper)."""

    def whiten(score: FloatArray) -> Whitened:
        return whitened_score(score, merit, work)

    return whiten


def _gram(system: JointSystem) -> FloatArray:
    """The Gram matrix ``X'X`` of the active scaled design, unweighted."""
    design = system.active_design
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value: FloatArray = design.T @ design
    return _finite_result(value, "design Gram")


def merit_factor(
    system: JointSystem, point: FloatArray, metric: MeritMetric, work: WorkLog
) -> FloatArray:
    """The lower Cholesky factor ``L_M`` of the fixed merit metric at ``point``.

    ``"gamma"``: the observed bread of the Gamma equation of the same rows at the
    start, ``X' diag(a z/m) X``; ``"gram"``: :func:`_gram`.  ``LinAlgError`` where
    the matrix is not positive definite, ``FloatingPointError`` or ``ValueError``
    where it cannot be formed: the requested metric is refused, never replaced.
    """
    if metric == "gamma":
        state = work.run(
            "merit_gamma_equations", lambda: equations_of(system)(point, GAMMA)
        )
        matrix = _bread(system, state.rows)
    else:
        matrix = work.run("merit_gram", lambda: _gram(system))
    return work.run("merit_cholesky", lambda: symmetric_cholesky(matrix))


# ------------------------------------------------------------ the certificates


def summary_of(certificate: Certificate) -> StageSummary:
    """The gates and quantities of a certificate, without its evaluation."""
    return StageSummary(
        certificate.level,
        certificate.strict,
        certificate.secondary,
        certificate.scaled_score,
        certificate.index_correction,
        certificate.rcond,
        certificate.positive_definite,
        certificate.conditioned,
        certificate.score_metric,
        MappingProxyType(dict(certificate.errors)),
    )


@dataclass(frozen=True, eq=False)
class _StageCertifier:
    """The fresh certificate of a stage point on that stage's equation (step 3)."""

    system: JointSystem
    controls: CertificateControls
    reported: ReportedScaleProvider | None
    frequency: bool
    merit: FloatArray
    work: WorkLog

    def certify(self, variance: CancelledFamily, point: FloatArray) -> Certificate:
        """Certify ``point`` on the equation of ``variance`` with the merit factor."""
        if not isinstance(variance, PowerVariance):
            raise TypeError(NOT_ROOT)
        stage = same_system(self.system, variance)
        return self.work.run(
            "root_certificate",
            lambda: certify(
                stage,
                point,
                self.controls,
                reported=self.reported,
                frequency=self.frequency,
                merit_cholesky=self.merit,
            ),
        )

    def accepted(self, certified: Certificate) -> bool:
        """Whether the certificate passes at either level."""
        return certified.accepted

    def assessment(self, certified: Certificate) -> Assessment:
        """The retry reading of the certificate (:func:`certificate.assessment`)."""
        return assessment(certified)

    def status(self, certified: Certificate, native: NativeReason) -> Status:
        """The accepted status composition of the certificate and the reason."""
        return compose_status(certified, native)

    def point(self, certified: Certificate) -> FloatArray:
        """The point that the certificate assessed."""
        return certified.gamma


# ------------------------------------------------------------------- the solve


@dataclass(frozen=True, eq=False)
class _Fit:
    """One solve's fixed inputs: the target system, settings, provider and work."""

    system: JointSystem
    options: SolverOptions
    root_options: RootOptions
    reported: ReportedScaleProvider | None
    frequency: bool
    work: WorkLog
    clock: tuple[float, float]

    def timing(self) -> Timing:
        """The time since the solve began."""
        wall, cpu = self.clock
        return Timing(time.perf_counter() - wall, time.process_time() - cpu)


def _check(
    system: JointSystem,
    options: SolverOptions,
    root_options: RootOptions,
    start: Start | None,
) -> None:
    """The preconditions of :func:`solve_joint_root`, refused with built-in errors."""
    if type(system) is not JointSystem:
        raise TypeError("system must be a JointSystem")
    if type(options) is not SolverOptions:
        raise TypeError("options must be a SolverOptions")
    if type(root_options) is not RootOptions:
        raise TypeError("root_options must be a RootOptions")
    if start is not None and type(start) is not Start:
        raise TypeError("start must be a Start or None")
    if not outside_convex(system.variance):
        raise ValueError(NOT_ROOT)
    if start is not None and start.coefficients.shape != (system.dimension,):
        raise ValueError("start must have one value per column of the design")


def _initializer(
    fit: _Fit, start: Start | None, intercept: int | None
) -> JointRootInitializer | StartFailure:
    """The caller's start, or the nearest convex fit of the same system (step 1)."""
    if start is not None:
        point = FloatStorage.store(start.coefficients)
        return JointRootInitializer(
            "user_start", None, False, None, None, point, None, None
        )
    kind, variance = nearest_convex(fit.system.variance)
    convex = same_system(fit.system, variance)
    try:
        opening = automatic_start(convex, intercept=intercept)
    except FloatingPointError as error:
        assert intercept is not None  # the zero start of no intercept is finite
        with np.errstate(over="ignore"):
            mean = float(np.mean(convex.active_z))
        return StartFailure(message=str(error), mean=mean, intercept=intercept)
    assess = CertificateAssessor(reported=fit.reported, frequency=fit.frequency)
    candidate = newton_solve(convex, opening, fit.options, assess=assess)
    concluded = conclude(
        convex, candidate, reported=fit.reported, frequency=fit.frequency
    )
    certificate = concluded.certificate
    return JointRootInitializer(
        kind,
        variance,
        certificate.accepted,
        concluded.status,
        concluded.reason,
        FloatStorage.store(certificate.gamma),
        candidate,
        certificate,
    )


def _merit(
    fit: _Fit, initializer: JointRootInitializer
) -> tuple[FloatArray | None, NativeReason]:
    """The fixed merit factor at the start, or ``None`` with the reason of its absence.

    An uncertified automatic initializer runs no root solve: its own native reason.
    A metric that is not positive definite is a ``factorization_failure``; one whose
    matrix cannot be formed in range a ``numerical_breakdown``.
    """
    if initializer.kind != "user_start" and not initializer.certified:
        if initializer.reason is None:
            raise ValueError("an automatic initializer keeps its native reason")
        return None, initializer.reason
    try:
        factor = merit_factor(
            fit.system, initializer.point, fit.root_options.merit_metric, fit.work
        )
    except np.linalg.LinAlgError:
        return None, "factorization_failure"
    except (FloatingPointError, ValueError):
        return None, "numerical_breakdown"
    return factor, "native_stop"


def _opening(initializer: JointRootInitializer, start: Start | None) -> Start:
    """The start of the fit: the caller's, or the convex fit's automatic start."""
    if start is not None:
        return start
    if initializer.newton is None:
        raise ValueError("an automatic initializer keeps its convex candidate")
    return initializer.newton.start


def joint_root_status(
    candidate: JointRootCandidate, certificate: Certificate
) -> Status:
    """The status of a joint root conclusion (module docstring, step 4).

    With a root stage (including a zero-step exact root), the accepted composition of
    the target certificate and the
    native reason.  Without, the uncertified initializer's own status; else, where the
    merit metric could not be formed at the start, ``nonfinite`` if the target
    equation is not representable there and ``singular_system`` otherwise, as the
    cancelled solver reports a merit metric that does not factorize.
    """
    if candidate.stages:
        return compose_status(certificate, candidate.reason)
    initializer = candidate.initializer
    if initializer.kind != "user_start" and not initializer.certified:
        if initializer.status is None:
            raise ValueError("an automatic initializer keeps its status")
        return initializer.status
    if "evaluation" in certificate.strict:
        return "nonfinite"
    return "singular_system"


def _without_root(
    fit: _Fit,
    initializer: JointRootInitializer,
    start: Start | None,
    reason: NativeReason,
) -> Conclusion:
    """The conclusion where no root stage opened (module docstring, step 4)."""
    controls = certificate_controls(fit.options)
    certificate = fit.work.run(
        "root_certificate",
        lambda: certify(
            fit.system,
            initializer.point,
            controls,
            reported=None,
            frequency=fit.frequency,
        ),
    )
    candidate = JointRootCandidate(
        start=_opening(initializer, start),
        options=fit.options,
        root_options=fit.root_options,
        initializer=initializer,
        merit_metric=fit.root_options.merit_metric,
        _merit_cholesky=None,
        stages=(),
        target_stage=None,
        continuation_used=False,
        _point=FloatStorage.store(certificate.gamma),
        reason=reason,
        work=fit.work.snapshot(),
        timing=fit.timing(),
    )
    return Conclusion(
        candidate, certificate, reason, joint_root_status(candidate, certificate)
    )


def _stage(outcome: StageOutcome[Certificate]) -> JointRootStage:
    """The record of one stage outcome, with the summary of its retained point."""
    variance = outcome.variance
    if not isinstance(variance, PowerVariance):
        raise TypeError(NOT_ROOT)
    return JointRootStage(
        variance,
        outcome.attempts,
        outcome.retained,
        summary_of(outcome.certified),
        outcome.status,
        outcome.iterations,
        outcome.evaluations,
    )


def _root(
    fit: _Fit, initializer: JointRootInitializer, merit: FloatArray
) -> tuple[tuple[StageOutcome[Certificate], ...], int, bool, Certificate]:
    """The stages, the target stage's index, continuation and the target certificate."""
    target = fit.system.variance
    if not isinstance(target, PowerVariance):
        raise TypeError(NOT_ROOT)
    options, work = fit.options, fit.work
    certifier = _StageCertifier(
        fit.system,
        certificate_controls(options),
        fit.reported,
        fit.frequency,
        merit,
        work,
    )
    kernel = RootKernel(
        equations_of(fit.system),
        direction_of(fit.system, options.rcond_min, work),
        _whitener(merit, work),
        options,
        fit.root_options,
        RootBudget(max(0, options.max_evaluations - initializer.evaluations)),
        work,
    )
    iterations = RootBudget(max(0, options.max_iter - initializer.iterations))
    run = RootRun(kernel, iterations, certifier, initializer.certified)
    direct = run_root_stage(run, target, initializer.point)
    _, origin = nearest_convex(target)
    stages, final, used = run_continuation(
        run, target, initializer.point, direct, origin=origin.tau
    )
    certificate = final.certified
    if used and stages[-1].variance != target:
        certificate = certifier.certify(target, certificate.gamma)
    index = next(position for position, stage in enumerate(stages) if stage is final)
    return stages, index, used, certificate


def solve_joint_root(
    system: JointSystem,
    options: SolverOptions,
    root_options: RootOptions,
    *,
    start: Start | None,
    intercept: int | None,
    reported: ReportedScaleProvider | None,
    frequency: bool,
) -> Conclusion | StartFailure:
    """Solve the joint equation outside ``[1, 2]`` and conclude (module docstring).

    Parameters
    ----------
    system : JointSystem
        The joint system of the fit, with a power variance outside ``[1, 2]``.
    options : SolverOptions
        The resolved options: the convex initializer's, the root solve's native
        tolerance, budgets, contraction, retry and ``rcond_min``, and the
        certificate's controls.
    root_options : RootOptions
        ``root_armijo``, ``root_max_backtracks``, ``root_max_iter``,
        ``merit_metric``, ``continuation`` and ``continuation_step``.
    start : Start or None
        The caller's start in the solver's coordinates, or ``None`` for the nearest
        convex initializer.
    intercept : int or None
        The intercept column of the design, for the convex initializer's automatic
        start.
    reported : ReportedScaleProvider or None
        The reported scale and standard errors of the fit's requested inference.
    frequency : bool
        Whether the weights are frequency weights.

    Returns
    -------
    Conclusion or StartFailure
        The conclusion of the root candidate, or the failure of the convex
        initializer's automatic start, where no candidate exists.

    Raises
    ------
    TypeError, ValueError
        A precondition does not hold.
    SpecificationError
        ``newton_provider="sklearn"`` without a verified installation, from the
        convex initializer.
    """
    _check(system, options, root_options, start)
    clock = (time.perf_counter(), time.process_time())
    fit = _Fit(system, options, root_options, reported, frequency, WorkLog(), clock)
    initializer = _initializer(fit, start, intercept)
    if isinstance(initializer, StartFailure):
        return initializer
    merit, absence = _merit(fit, initializer)
    if merit is None:
        return _without_root(fit, initializer, start, absence)
    stages, index, used, certificate = _root(fit, initializer, merit)
    final = stages[index]
    reason = final.attempts[final.retained].native
    candidate = JointRootCandidate(
        start=_opening(initializer, start),
        options=options,
        root_options=root_options,
        initializer=initializer,
        merit_metric=root_options.merit_metric,
        _merit_cholesky=FloatStorage.store(merit),
        stages=tuple(_stage(stage) for stage in stages),
        target_stage=index,
        continuation_used=used,
        _point=FloatStorage.store(certificate.gamma),
        reason=reason,
        work=fit.work.snapshot(),
        timing=fit.timing(),
    )
    return Conclusion(
        candidate, certificate, reason, joint_root_status(candidate, certificate)
    )
