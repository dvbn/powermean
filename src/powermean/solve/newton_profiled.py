"""The internal profiled Newton solver of the joint system with absorbed effects.

The explicitly selected internal organization ``absorbed='profiled'`` for the power
family with ``tau`` in ``[1, 2]`` and NB2.  No public route, default organization,
certificate, covariance or other organization.

:func:`solve` takes the accepted immutable final sample, the working
variance of its request, the resolved options, the explicit absorbed options, a required
trigger with its admission of an outcome and an optional independent full-point
assessor, and returns a candidate with its trace (:class:`ProfiledCandidate`), never a
certified fit.

The internal NumericFit entry can select the retained
nonlinear backend. Its execution tag and history route say so explicitly: trials
and the first cold opening solve one conditional FE block, with the remaining FE
coordinates updated by Newton, not profiled to completion. Warm retry openings
and fresh final work remain all-FE. Public/direct and ineligible calls retain the
original organization below.

Slopes.  The damped Newton iteration of :mod:`powermean.solve.newton_convex` with the
provider of ``newton_provider`` (both drive the same profiled linear loss), from zero
slopes (the intercept is absorbed) or the caller's start, with one budget of pointwise
evaluations and one of accepted steps for the whole solve, and the whole-point state of
:mod:`powermean.solve.profiled_loss`.  After a native stop the fresh final work runs;
its failure is the operative reason ``final_failure``.  The retry is the existing rule
(:func:`powermean.solve.newton_retry.run_attempts`), decided on an operative native
stop with the caller's assessor of whole points: ``retry=1`` without one is refused
before any work.  The second attempt starts from the first's whole point with what
remains of the budgets.

Effects only.  Without an active slope there is no provider, start, Newton step or
retry: the effects are profiled from the cold state, ZC is evaluated, and the fresh
final profile runs; the operative reason is ``profile_only``.

Indications and the trigger.  An alarm stops the attempt with the operative reason
``alarm``: not a native stop, not retry eligible, the native flag kept in the alarm.
The caller's trigger is then called once, and its trace read, by
:mod:`powermean.solve.profiled_trigger` (whose request, views and record this module
still provides): the trace is bound to this sample, its origin and this request, its
first fresh diagnostic is admitted again on the solver's own problem, and the
coordinator's outcome is kept only when the caller's admission, the coordinator's own,
admits it.  The solver never resumes, re-solves, certifies, adopts a new
sample or deletes a row: its stopped point and its sample are unchanged.

Evaluations.  The profiled linear loss charges each evaluation to the solve's one budget
before its profile, rows and absorption
(:class:`~powermean.solve.profiled_linear_loss.Admissions`); a retry receives what
remains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.fe.profile import ProfileState
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve import support, support_gamma
from powermean.solve.assessment import Assessment, CertificateControls
from powermean.solve.assessment import certificate_controls as controls_of
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_convex import (
    IndicationError,
    InnerLimitError,
    NewtonProblem,
    NewtonProvider,
    ProfiledIteration,
    resolve_provider,
    run_index_attempt,
    run_profiled_attempt,
)
from powermean.solve.newton_retry import improved, verdict
from powermean.solve.newton_trace import NativeReason, Timing
from powermean.solve.options import Provider, SolverOptions
from powermean.solve.point_loss import PointLoss
from powermean.solve.profiled_indications import Alarm
from powermean.solve.profiled_linear_loss import Admissions, ProfiledPointLoss
from powermean.solve.profiled_loss import (
    FinalWork,
    ProfiledEvaluator,
    SolveWork,
    WorkLog,
    final_work,
)
from powermean.solve.profiled_trigger import (
    RECESSIONS,
    Admission,
    EventView,
    FullPoint,
    OutcomeView,
    RequestView,
    TraceView,
    Trigger,
    TriggerRecord,
    TriggerRequest,
    TriggerStatus,
    triggered,
)
from powermean.solve.reduced_nonlinear_loss import ReducedEvaluator
from powermean.solve.starts import Start
from powermean.systems.profiled import PointProfile, ProfiledProblem

# The names of this module, and those of the trigger boundary defined in
# solve/profiled_trigger.py, whose import routes from here are kept.
__all__ = [
    "RECESSIONS",
    "EventView",
    "Family",
    "FullAssessor",
    "FullPoint",
    "OutcomeView",
    "ProfiledAttempt",
    "ProfiledCandidate",
    "ProfiledReason",
    "RequestView",
    "TraceView",
    "Trigger",
    "TriggerRecord",
    "TriggerRequest",
    "TriggerStatus",
    "solve",
    "stopped_point",
]

if TYPE_CHECKING:
    from powermean.solve.reduced_nonlinear import ConditionalPoint

Family = PowerVariance | NB2Variance
ProfiledReason = Literal[
    NativeReason, "inner_limit", "alarm", "final_failure", "profile_only"
]
# ---------------------------------------------------------------------- the points


class FullAssessor(Protocol):
    """An independent assessment of a whole point (the full certificate's role)."""

    def __call__(
        self, point: FullPoint, controls: CertificateControls, /
    ) -> Assessment:
        """The assessment of ``point`` under the controls of the certificate."""
        ...


@dataclass(frozen=True, eq=False)
class ProfiledAttempt:
    """One attempt: its iteration, operative reason, points, final work and alarm.

    ``iteration`` is the provider's trace (``None`` without slopes); ``reason`` the
    operative stop and ``message`` its text; ``opening`` the attempt's actual start
    (a conditional point for the retained backend's first cold opening);
    ``point`` the accepted point at the stop, or the whole point the attempt started
    from when it profiled nothing (``None`` for a cold start); ``loss`` its averaged
    criterion when evaluated; ``final`` the fresh final work; ``alarm`` the indication
    that stopped it; ``assessment`` the retry rule's assessment of it, if made.
    """

    iteration: ProfiledIteration | None
    reason: ProfiledReason
    message: str
    opening: PointProfile | ConditionalPoint | None
    point: PointProfile | ConditionalPoint | None
    loss: float | None
    final: FinalWork | None
    alarm: Alarm | None
    assessment: Assessment | None = None

    @property
    def iterations(self) -> int:
        """The accepted Newton steps of the attempt."""
        return 0 if self.iteration is None else self.iteration.iterations


def stopped_point(attempt: ProfiledAttempt) -> PointProfile | ConditionalPoint:
    """The point an attempt stopped at: its complete fresh final point, else ``point``.

    ``ValueError`` when the attempt holds no profiled point.
    """
    final = attempt.final
    if final is not None and final.point is not None and final.complete:
        return final.point
    if attempt.point is None:
        raise ValueError("the attempt holds no profiled point")
    return attempt.point


# ------------------------------------------------------------------ the candidate


@dataclass(frozen=True, eq=False)
class ProfiledCandidate:
    """The candidate of one profiled solve, with its trace; not a certified solution.

    ``attempts`` holds one attempt, or two when the retry ran, and ``retained`` the
    index of the attempt whose point is the candidate; ``start`` and ``provider`` are
    ``None`` without slopes.  ``work`` holds every profile and absorption call;
    ``trigger`` the one call of the trigger after an alarm.  The assessment fields are
    those of :class:`~powermean.solve.newton_trace.NewtonCandidate`.
    """

    sample: FESample
    variance: Family
    start: Start | None
    options: SolverOptions
    absorbed: AbsorbedOptions
    provider: Provider | None
    provider_version: str | None
    attempts: tuple[ProfiledAttempt, ...]
    retained: int
    initial_assessment: Assessment | None
    no_measurable_progress: bool
    assessments: int
    assessment_timing: Timing
    work: SolveWork
    trigger: TriggerRecord | None
    timing: Timing
    execution: Literal["profiled", "reduced_nonlinear"] = "profiled"

    @property
    def reason(self) -> ProfiledReason:
        """The operative reason of the retained attempt."""
        return self.attempts[self.retained].reason

    @property
    def iterations(self) -> int:
        """The accepted Newton steps of every attempt, charged to ``max_iter``."""
        return sum(attempt.iterations for attempt in self.attempts)

    def point(self) -> FullPoint:
        """The whole point of the retained attempt (``ValueError`` without one)."""
        retained = self.attempts[self.retained]
        return FullPoint(self.sample, self.variance, stopped_point(retained))


# ---------------------------------------------------------------------- the solve


@dataclass(frozen=True, eq=False)
class _Context:
    """The fixed parts of one solve, shared by its attempts."""

    sample: FESample
    variance: Family
    problem: ProfiledProblem
    options: SolverOptions
    absorbed: AbsorbedOptions
    log: WorkLog
    budget: Admissions


class _Assessments:
    """The caller's assessments of whole points: each one made, counted and timed."""

    def __init__(self, assess: FullAssessor, context: _Context) -> None:
        self.assess = assess
        self.context = context
        self.controls = controls_of(context.options)
        self.count = 0
        self.wall = 0.0
        self.cpu = 0.0

    def __call__(self, point: PointProfile | ConditionalPoint) -> Assessment:
        """The assessment of ``point``, which must be an :class:`Assessment`."""
        full = FullPoint(self.context.sample, self.context.variance, point)
        wall, cpu = time.perf_counter(), time.process_time()
        result = self.assess(full, self.controls)
        self.wall += time.perf_counter() - wall
        self.cpu += time.process_time() - cpu
        self.count += 1
        if not isinstance(result, Assessment):
            raise TypeError("an assessor must return an Assessment")
        return result


def _check(
    sample: FESample,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    trigger: object,
    admit: object,
    *,
    assess: FullAssessor | None,
) -> None:
    """The preconditions of :func:`solve`, each refused with a built-in error."""
    if type(sample) is not FESample:
        raise TypeError("sample must be an FESample")
    if type(options) is not SolverOptions or type(absorbed) is not AbsorbedOptions:
        raise TypeError("options must be a SolverOptions and an AbsorbedOptions")
    if not callable(trigger):
        raise TypeError("trigger must be the caller's callable")
    if not callable(admit):
        raise TypeError("admit must be the caller's admission of an outcome")
    if absorbed.absorbed != "profiled":
        raise ValueError("this is the profiled organization, not 'index'")
    if sample.omega == 0:
        raise ValueError("omega = 0 is the linear path, not a Newton solve")
    if options.retry == 1 and assess is None:
        raise ValueError("retry=1 needs an independent assessment of the whole point")


def _problem(sample: FESample, variance: Family) -> ProfiledProblem:
    """The profiled problem of the sample's final rows and active scaled columns."""
    grouping, design = sample.grouping, sample.design
    if grouping is None:
        raise ValueError("the profiled solver needs fixed-effect blocks")
    matrix = np.zeros((sample.count, 0)) if design is None else design.matrix
    return ProfiledProblem(
        z=sample.problem.response,
        design=matrix,
        offset=sample.problem.offset,
        weights=sample.data.weight_values,
        grouping=grouping,
        variance=variance,
        columns=() if design is None else design.column_names,
    )


def _evaluator(
    context: _Context, attempt: int, warm: ProfileState | None
) -> ProfiledEvaluator:
    """The whole-point state of one attempt."""
    return ProfiledEvaluator(
        context.problem,
        context.absorbed,
        context.options.curvature,
        context.log,
        attempt=attempt,
        warm=warm,
        positions=context.sample.positions,
    )


def _finished(
    evaluator: ProfiledEvaluator,
    iteration: ProfiledIteration,
    origin: PointProfile | ConditionalPoint | None,
) -> ProfiledAttempt:
    """The attempt of an iteration: its operative reason and its final work.

    The fresh final work runs after a native stop only.
    """
    accepted, opening = evaluator.accepted, evaluator.opening
    point = opening if opening is not None else origin
    loss = None
    if accepted is not None:
        point, loss = accepted.point, accepted.loss
    found = (opening, point, loss)
    if iteration.native is None:
        stop: ProfiledReason = "alarm"
        if iteration.stop == "inner_limit":
            stop = "inner_limit"
        alarm = evaluator.alarm
        return ProfiledAttempt(iteration, stop, iteration.message, *found, None, alarm)
    if iteration.native != "native_stop" or accepted is None:
        return ProfiledAttempt(
            iteration, iteration.native, iteration.message, *found, None, None
        )
    final = final_work(evaluator, accepted.point, accepted.loss)
    reason: ProfiledReason = "native_stop"
    if final.alarm is not None:
        reason = "alarm"
    elif not final.complete:
        reason = "final_failure"
    return ProfiledAttempt(iteration, reason, final.reason, *found, final, final.alarm)


class _Newton(NamedTuple):
    """The provider and the fixed inputs of the iteration of one solve."""

    provider: NewtonProvider
    point: PointLoss
    problem: NewtonProblem


def _attempt(
    context: _Context,
    newton: _Newton,
    theta: FloatArray,
    *,
    origin: PointProfile | ConditionalPoint | None,
    max_iter: int,
) -> ProfiledAttempt:
    """One attempt from ``theta`` and the effects of ``origin`` (cold without one)."""
    index = 0 if origin is None else 1
    warm = None if origin is None else origin.state
    space = context.problem.__dict__.get("_reduced_nonlinear")
    if space is not None:
        reduced = ReducedEvaluator(
            context, newton.point, space, attempt=index, warm=warm
        )
        iteration = run_index_attempt(
            newton.provider, newton.point, newton.problem, theta,
            context.options, evaluation=reduced, max_iter=max_iter,
            budget=context.budget,
        )
        return _finished(reduced, iteration, origin)
    evaluator = _evaluator(context, index, warm)
    iteration = run_profiled_attempt(
        newton.provider,
        newton.point,
        newton.problem,
        theta,
        context.options,
        evaluation=evaluator,
        max_iter=max_iter,
        budget=context.budget,
    )
    return _finished(evaluator, iteration, origin)


class _Attempts(NamedTuple):
    """The attempts of a solve, the one retained, and what the retry rule read."""

    attempts: tuple[ProfiledAttempt, ...]
    retained: int
    initial: Assessment | None
    no_measurable_progress: bool


def _opened(attempt: ProfiledAttempt) -> PointProfile | ConditionalPoint:
    """The actual opening point of an attempt that evaluated it."""
    if attempt.opening is None:
        raise ValueError("the attempt did not profile its start")
    return attempt.opening


def _attempts(
    context: _Context, newton: _Newton, start: Start, assess: _Assessments | None
) -> _Attempts:
    """The first attempt and, if the existing rule allows it, the second.

    The statements of :func:`powermean.solve.newton_retry.run_attempts`, read on the
    operative reason and on whole points.
    """
    options, budget = context.options, context.budget
    first = _attempt(
        context, newton, start.coefficients, origin=None, max_iter=options.max_iter
    )
    if assess is None or options.retry == 0 or first.reason != "native_stop":
        return _Attempts((first,), 0, None, False)
    origin = stopped_point(first)
    assessed = assess(origin)
    first = replace(first, assessment=assessed)
    budgets_remain = first.iterations < options.max_iter and budget.used < budget.limit
    if assessed.accepted or not assessed.finite_error or not budgets_remain:
        return _Attempts((first,), 0, None, False)
    initial = assess(_opened(first))
    if not improved(assessed, initial):
        return _Attempts((first,), 0, initial, False)
    left = options.max_iter - first.iterations
    second = _attempt(context, newton, origin.theta, origin=origin, max_iter=left)
    reassessed = assess(stopped_point(second))
    second = replace(second, assessment=reassessed)
    no_progress, keep_first = verdict(assessed, reassessed)
    return _Attempts((first, second), 0 if keep_first else 1, initial, no_progress)


def _profile_only(context: _Context) -> ProfiledAttempt:
    """The branch without slopes: profile, ZC, fresh final work; no Newton step."""
    evaluator = _evaluator(context, 0, None)
    reason: ProfiledReason
    try:
        evaluator.start(np.zeros(0))
        opening = evaluator.inspect()
    except (InnerLimitError, IndicationError, FloatingPointError) as error:
        reason = "numerical_breakdown"
        if isinstance(error, InnerLimitError):
            reason = "inner_limit"
        elif isinstance(error, IndicationError):
            reason = "alarm"
        point = evaluator.opening
        found = (point, point, None, None, evaluator.alarm)
        return ProfiledAttempt(None, reason, str(error), *found)
    final = final_work(evaluator, opening, None)
    reason = "profile_only"
    if final.alarm is not None:
        reason = "alarm"
    elif not final.complete:
        reason = "final_failure"
    points = (opening, opening, None)
    return ProfiledAttempt(None, reason, final.reason, *points, final, final.alarm)


def _request(
    context: _Context, attempt: ProfiledAttempt, alarm: Alarm
) -> TriggerRequest:
    """The trigger's request: the stopped point, the alarm, the TIGHTER controls."""
    variance = context.variance
    gpml = type(variance) is PowerVariance and variance.tau == 2.0
    point = FullPoint(context.sample, variance, stopped_point(attempt))
    return TriggerRequest(
        context.sample,
        variance,
        None if gpml else support.TIGHTER,
        support_gamma.TIGHTER if gpml else None,
        alarm,
        point,
    )


def solve(
    sample: FESample,
    variance: Family,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    *,
    trigger: Trigger,
    admit: Admission,
    assess: FullAssessor | None = None,
    start: Start | None = None,
) -> ProfiledCandidate:
    """Solve the profiled joint problem of ``sample``: a candidate and its trace.

    Parameters
    ----------
    sample : FESample
        The accepted immutable final sample, with fixed-effect blocks.
    variance : PowerVariance or NB2Variance
        The working variance of the sample's request: ``tau`` in ``[1, 2]``, or NB2.
    options : SolverOptions
        The resolved options: the provider, the native tolerance, the budgets, the
        line search, the retry and the curvature, and the certificate's controls for
        the assessor only.
    absorbed : AbsorbedOptions
        The explicit absorbed options; ``absorbed`` must be ``'profiled'``.
    trigger : Trigger
        The caller's trigger, bound to the actual completed outcome of the sample
        stage; called at most once, after an alarm.
    admit : Admission
        The caller's admission of the trigger call's outcome, the coordinator's own
        (``powermean.interface.sample.admitted_outcome``); only an outcome it admits is
        kept, apart from the stopped solve.
    assess : FullAssessor, optional
        The independent assessment of a whole point; required with ``retry=1``.
    start : Start, optional
        The start of the slopes in the coordinates of the sample's scaled design; zero
        slopes when omitted.  Refused without slopes.

    Returns
    -------
    ProfiledCandidate
        The candidate and its trace; no claim of convergence or existence.

    Raises
    ------
    SpecificationError
        ``newton_provider="sklearn"`` without a verified installed scikit-learn.
    TypeError, ValueError
        A precondition does not hold; nothing is run.
    """
    _check(sample, options, absorbed, trigger, admit, assess=assess)
    call = _ProfiledCall(sample, variance, options, absorbed, trigger, admit, assess, start)
    return _run_profiled(call, _problem(sample, variance))


@dataclass(frozen=True)
class _ProfiledCall:
    """The unchanged solver inputs, separate from their owned numeric representation."""

    sample: FESample
    variance: Family
    options: SolverOptions
    absorbed: AbsorbedOptions
    trigger: Trigger
    admit: Admission
    assess: FullAssessor | None
    start: Start | None


def _solve_numeric(call: _ProfiledCall, problem: ProfiledProblem) -> ProfiledCandidate:
    """Owned entry; the chosen nonlinear backend is explicitly recorded."""
    _check(call.sample, call.options, call.absorbed, call.trigger, call.admit,
           assess=call.assess)
    return _run_profiled(call, problem)


def _run_profiled(call: _ProfiledCall, problem: ProfiledProblem) -> ProfiledCandidate:
    """Shared starts, provider, budgets, retry and final-work lifecycle."""
    sample, variance, options, absorbed = (call.sample, call.variance,
                                          call.options, call.absorbed)
    trigger, admit, assess, start = call.trigger, call.admit, call.assess, call.start
    if start is not None and (
        type(start) is not Start or start.coefficients.shape != (problem.n_slopes,)
    ):
        raise ValueError("start must be a Start with one value per slope column")
    provider = (
        None if not problem.n_slopes else resolve_provider(options.newton_provider)
    )
    wall, cpu = time.perf_counter(), time.process_time()
    budget = Admissions(options.max_evaluations)
    found = (sample, variance, problem, options, absorbed, WorkLog(), budget)
    context = _Context(*found)
    assessments = None if assess is None else _Assessments(assess, context)
    if provider is None:
        outcome = _Attempts((_profile_only(context),), 0, None, False)
    else:
        if start is None:
            start = Start(np.zeros(problem.n_slopes), "automatic")
        zero_offset = np.zeros(problem.n_rows)
        zero_offset.flags.writeable = False
        point = ProfiledPointLoss(
            problem,
            offset=zero_offset,
            variance=variance,
            curvature=options.curvature,
            budget=budget,
        )
        design = NewtonProblem(problem.design, problem.z, problem.weights)
        newton = _Newton(provider, point, design)
        outcome = _attempts(context, newton, start, assessments)
    last = outcome.attempts[-1]
    record = None
    if last.alarm is not None:
        record = triggered(trigger, admit, _request(context, last, last.alarm))
    timing = Timing(0.0, 0.0)
    if assessments is not None:
        timing = Timing(assessments.wall, assessments.cpu)
    return ProfiledCandidate(
        sample=sample,
        variance=variance,
        start=start,
        options=options,
        absorbed=absorbed,
        provider=None if provider is None else provider.name,
        provider_version=None if provider is None else provider.version,
        attempts=outcome.attempts,
        retained=outcome.retained,
        initial_assessment=outcome.initial,
        no_measurable_progress=outcome.no_measurable_progress,
        assessments=0 if assessments is None else assessments.count,
        assessment_timing=timing,
        work=context.log.frozen(),
        trigger=record,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
        execution=("profiled" if problem.__dict__.get("_reduced_nonlinear") is None
                   else "reduced_nonlinear"),
    )
