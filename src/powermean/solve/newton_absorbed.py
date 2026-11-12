"""The internal fitted-index Newton solver of the joint system with absorbed effects.

The explicitly selected internal organization ``absorbed='index'`` (Newton in the
fitted index, as ``ppmlhdfe`` organizes it) for the power family with ``tau`` in
``[1, 2]`` and NB2.
No public route, default organization, certificate, covariance, linear path,
IOLS, cancellation, nonconvex continuation or instrument.

:func:`solve` takes the accepted immutable final sample, the working
variance of its request, the resolved options, the explicit absorbed options, a required
trigger with its admission of an outcome, an optional independent whole-point assessor
and an optional full start, and returns a candidate with its trace
(:class:`IndexCandidate`), never a certified fit.

The iteration.  The damped Newton iteration of :mod:`powermean.solve.newton_convex`
with the provider of ``newton_provider``, through one shared loop and its
representation seams (:func:`~powermean.solve.newton_convex.run_index_attempt`), with
the whole-point state of :mod:`powermean.solve.absorbed_loss`: the accepted point is
``(gamma, q)``, each direction is the full joint Newton direction in the fitted index
(the slopes' Schur step by the actual provider in K-by-K storage, the forced full-RHS
nuisance solve, :mod:`powermean.solve.absorbed_direction`), each trial moves ``gamma``
and ``q`` along it unprofiled, and the native test reads the full score.  Without a
slope column the same loop runs with the provider's zero-slope storage, its inner solve
and convergence method skipped explicitly; the provider's identity is recorded.  One
budget of pointwise evaluations and one of accepted steps serve the whole solve.

The start.  Automatic: zero slopes and ``log(mean z)`` in every level of the first
block, then one sweep at ``profile_tol``, an initializer whose budget status is no
failure (a range refusal stops): the family's accepted profiler, or, for the first
attempt with one block, no slope and exact PPML or the exact NB2 ``kappa = 0`` alias,
the local evaluation of that same sweep
(:mod:`powermean.solve.absorbed_start`), recorded as its own work.  The caller's
:class:`~powermean.solve.absorbed_direction.IndexStart` gives the slopes and a
coefficient witness bound to the grouping, whose index is reconstructed without a
profile.  A retry starts at the retained ``(gamma, q)`` without a sweep, its schedule
reset to ``absorb_tol_start`` and its forcing history to ``1e-3``, the
budget and the logged work shared.

The final work, after a native stop only, at the same ``q`` and in this order:
recovery and normalization of the effects under the raw weights at exactly
``absorb_tol_final``, with the explicit reconstruction checks; the fresh rows and the
accepted fresh equations of every original block; the slope absorption when there are
slopes, then the full-RHS solve, at exactly ``absorb_tol_final`` and with the most
recently issued forcing factor (``1e-3`` when none was issued), no factor issued for
it, its exception guarded by the final rows' raw-weight total and ``gradient_tol``
(success needs ``af`` and either the relative bound or a native residual
at most ``gradient_tol``, never ``af`` alone); the indications.  It stops at the first
failed part, the operative reason ``final_failure``, every completed part kept; it is
outside the iteration's evaluation budget and never takes another Newton step.  A
nonzero final equation is evidence for the caller's assessor, not a second native rule.

Indications and the trigger are those of the profiled route
(:mod:`powermean.solve.profiled_trigger`): an alarm stops the attempt (``alarm``, not a
native stop, not retry eligible); the caller's trigger is called once with the stopped
whole point and the family's TIGHTER controls, its trace bound, its first diagnostic
admitted again and its outcome kept only when the caller's admission admits it.  The
solver never resumes, certifies, adopts a new sample or deletes a row.

The opening of the retry.  The retry rule assesses the first attempt's initialized start
through the same counted and timed call as every assessment.  When the assessor is the
exact private pair :class:`_OpeningDispatch`, recognized by its exact type at that one
statement, that call goes to the pair's ``opening`` callback; every other call of the
pair goes to its ``regular`` callback, unchanged.  Any other assessor receives every
call, the opening's included, as before: no attribute of it is read and no argument is
added.  The attempts, budgets, rule and selection are unchanged.

The equation-aware stop.  When the assessor is the
exact private adapter :class:`_EquationStop`, recognized by its exact type in
:func:`solve`, every attempt of the solve tests its accepted steps with
:class:`_ProbedEvaluator`: the two native formulas first, unchanged and in their order;
only when both pass, one probe of the unchanged ``fresh_equations(_prepared(problem,
base), q)`` at the accepted trial, ``q`` its full index and ``base =
base_index(problem, theta)`` of its slopes (the transformed offset included), on the
problem, weights, family and grouping of the final work.  The native stop needs both
formulas and ``maximum <= tol`` (:func:`_passes`, equality passing), ``tol`` the solver
option read once at the solve's start.  A finite failed probe continues the same
attempt through the unchanged loop: it is no assessment, final work, reading or retry,
and the schedule and forcing history go on.  A refused probe (``FloatingPointError``)
waits for the step hook: the accepted step is counted and its indications run first,
an alarm keeping its precedence and the combined flag; then the refusal ends the
attempt as ``numerical_breakdown`` with its reason.  The probes are not pointwise
evaluations: they consume no slot of the one budget and create no allowance, at most
one per accepted step, none at a start or a held trial.  Each probe is logged in the
fit's :class:`ProbeLog` (:class:`EquationProbe`: its attempt and accepted step, the
two formulas' values, the SHA-256 of its ``theta``, base and ``q`` bytes, its bound,
maximum or refusal, outcome, the step's combined native flag and its own seconds,
inside the iteration's).  :func:`solve` starts the log before any work and finishes it
with its candidate; a started log is refused by any other solve.  After a combined
native stop, the final work, the retry rule, the budgets and the selection are those
above.  Any other assessor and every other call of this solver keep the original
iteration.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, Protocol

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.fe.absorb_numeric import PreparedWeights
from powermean.fe.profile import _bound
from powermean.fe.profile_arithmetic import EquationCheck, fresh_equations
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve import support, support_gamma
from powermean.solve.absorbed_direction import (
    CallSite,
    Direction,
    IndexContext,
    IndexLog,
    IndexPoint,
    IndexStart,
    IndexWork,
    recover,
)
from powermean.solve.absorbed_loss import FinalRows, IndexEvaluator, final_rows
from powermean.solve.absorbed_rhs import RHSResult
from powermean.solve.absorbed_start import LocalInitializer
from powermean.solve.assessment import Assessment, CertificateControls
from powermean.solve.assessment import certificate_controls as controls_of
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_convex import (
    NewtonProblem,
    NewtonProvider,
    ProfiledIteration,
    Solver,
    resolve_provider,
    run_index_attempt,
)
from powermean.solve.newton_retry import improved, verdict
from powermean.solve.newton_trace import NativeReason, Timing
from powermean.solve.options import Provider, SolverOptions
from powermean.solve.profiled_indications import Alarm
from powermean.solve.profiled_linear_loss import Admissions, ProfiledPointLoss
from powermean.solve.profiled_trigger import (
    Admission,
    Trigger,
    TriggerRecord,
    TriggerRequest,
    WholePoint,
    triggered,
)
from powermean.systems.profiled import (
    Absorbed,
    PointProfile,
    ProfiledProblem,
    _prepared,
    base_index,
)

Family = PowerVariance | NB2Variance
IndexReason = Literal[NativeReason, "inner_limit", "alarm", "final_failure"]
# The whole point of the fitted-index organization.
IndexWholePoint = WholePoint[IndexPoint]

_ACCEPTED_CONTEXT_ABSORB = IndexContext.absorb
_ACCEPTED_CONTEXT_NUISANCE = IndexContext.nuisance


# ---------------------------------------------------------------- final work


@dataclass(frozen=True, eq=False)
class IndexFinal:
    """The fresh final work at the same point after a native stop.

    ``point`` is the accepted point (the same ``q``) with its recovery; ``rows`` the
    fresh rows at ``q``; ``equations`` the accepted fresh equations of every original
    block at ``q``; ``absorbed`` the final slope absorption (``None`` without slopes or
    when not reached), kept when completed whatever follows; ``rhs`` the final full-RHS
    solve; ``alarm`` an indication at the final point; ``reason`` empty when every part
    completed, else what failed first.
    """

    point: IndexPoint
    rows: FinalRows | None
    equations: EquationCheck | None
    absorbed: Absorbed | None
    rhs: RHSResult | None
    alarm: Alarm | None
    reason: str

    @property
    def complete(self) -> bool:
        """Whether the recovery, rows, equations, absorption and RHS all completed."""
        return not self.reason


def final_work(evaluator: IndexEvaluator, point: IndexPoint, loss: float) -> IndexFinal:
    """The fresh final work at ``point`` in the frozen order (module docstring).

    ``loss`` is the accepted criterion, against which GC compares the final one at the
    same ``q``.  The full-RHS solve reads the evaluator's most recently issued forcing
    factor and issues none.  A failed part ends the work with its reason; a cached
    result is never reused, and ``q`` is never replaced.
    """
    context = evaluator.context
    problem, controls = context.problem, context.controls
    final = controls.absorb_tol_final
    recovery = recover(
        problem,
        point,
        tolerance=final,
        max_sweeps=controls.absorb_max_sweeps,
        acceleration=controls.absorb_acceleration,
    )
    found = IndexPoint(point.theta, point.base, point.index, recovery)
    if not recovery.complete:
        return IndexFinal(found, None, None, None, None, None, recovery.reason)
    rows: FinalRows | None = None
    equations: EquationCheck | None = None
    absorbed: Absorbed | None = None
    prepared: PreparedWeights | None = None
    plain_context = (
        type(context) is IndexContext
        and IndexContext.absorb is _ACCEPTED_CONTEXT_ABSORB
        and IndexContext.nuisance is _ACCEPTED_CONTEXT_NUISANCE
        and "absorb" not in context.__dict__
        and "nuisance" not in context.__dict__
    )
    try:
        rows = final_rows(problem, point.index, context.curvature)
        equations = fresh_equations(_prepared(problem, point.base), point.index)
        if problem.n_slopes:
            site: CallSite = (evaluator.attempt, evaluator.steps, "final", final, None)
            if plain_context:
                absorbed, prepared = context._absorb(
                    site, rows.curvature, index=point.index
                )
            else:
                absorbed = context.absorb(site, rows.curvature)
            result = absorbed.result
            if not result.converged or absorbed.failure:
                failure = absorbed.failure or result.reason
                return IndexFinal(found, rows, equations, absorbed, None, None, failure)
        site = (evaluator.attempt, evaluator.steps, "final", final, None)
        system = (rows.curvature, rows.right)
        if plain_context and prepared is not None:
            rhs = context._nuisance(
                site, system, evaluator.forcing, rows.total, prepared=prepared
            )
        else:
            rhs = context.nuisance(site, system, evaluator.forcing, rows.total)
        if not rhs.converged:
            reason = f"full right-hand side: {rhs.reason}"
            return IndexFinal(found, rows, equations, absorbed, rhs, None, reason)
        alarm = evaluator.final_alarm(point.index, (loss, rows.loss))
        return IndexFinal(found, rows, equations, absorbed, rhs, alarm, "")
    except FloatingPointError as error:
        return IndexFinal(found, rows, equations, absorbed, None, None, str(error))


# ------------------------------------------------------------------ attempts


@dataclass(frozen=True, eq=False)
class IndexAttempt:
    """One attempt: its iteration, operative reason, points, final work and alarm.

    ``iteration`` is the provider's trace; ``reason`` the operative stop and
    ``message`` its text; ``opening`` the attempt's initialized start; ``point`` the
    accepted point at the stop (``None`` when no point was evaluated); ``loss`` and
    ``gradient`` its criterion and full score when evaluated; ``direction`` the last
    completed full direction; ``initializer`` the automatic start's one-sweep profile,
    or its local start in the local domain; ``final`` the fresh final work;
    ``alarm`` the indication that stopped it; ``assessment`` the retry rule's
    assessment of it, if made.
    """

    iteration: ProfiledIteration
    reason: IndexReason
    message: str
    opening: IndexPoint | None
    point: IndexPoint | None
    loss: float | None
    gradient: FloatArray | None
    direction: Direction | None
    initializer: PointProfile | LocalInitializer | None
    final: IndexFinal | None
    alarm: Alarm | None
    assessment: Assessment | None = None

    @property
    def iterations(self) -> int:
        """The accepted Newton steps of the attempt."""
        return self.iteration.iterations


def stopped_point(attempt: IndexAttempt) -> IndexPoint:
    """The point an attempt stopped at: its final point (recovered) or ``point``.

    ``ValueError`` when the attempt holds no point.
    """
    if attempt.final is not None:
        return attempt.final.point
    if attempt.point is None:
        raise ValueError("the attempt holds no index point")
    return attempt.point


@dataclass(frozen=True, eq=False)
class IndexCandidate:
    """The candidate of one fitted-index solve, with its trace; not a certified fit.

    ``attempts`` holds one attempt, or two when the retry ran, and ``retained`` the
    index of the attempt whose point is the candidate; ``start`` the caller's full
    start (``None`` for the automatic one); ``provider`` and ``provider_version`` the
    actual provider, which ran even without slopes, and ``factorized`` whether it
    factorized a slope system (its convergence method never runs on this route).
    ``work`` holds every initializer, absorption, full-RHS and direction call;
    ``trigger`` the one call of the trigger after an alarm.  The assessment fields are
    those of :class:`~powermean.solve.newton_trace.NewtonCandidate`.
    """

    sample: FESample
    variance: Family
    start: IndexStart | None
    options: SolverOptions
    absorbed: AbsorbedOptions
    provider: Provider
    provider_version: str | None
    factorized: bool
    attempts: tuple[IndexAttempt, ...]
    retained: int
    initial_assessment: Assessment | None
    no_measurable_progress: bool
    assessments: int
    assessment_timing: Timing
    work: IndexWork
    trigger: TriggerRecord | None
    timing: Timing

    @property
    def reason(self) -> IndexReason:
        """The operative reason of the retained attempt."""
        return self.attempts[self.retained].reason

    @property
    def iterations(self) -> int:
        """The accepted Newton steps of every attempt, charged to ``max_iter``."""
        return sum(attempt.iterations for attempt in self.attempts)

    def point(self) -> IndexWholePoint:
        """The whole point of the retained attempt (``ValueError`` without one)."""
        retained = self.attempts[self.retained]
        return WholePoint(self.sample, self.variance, stopped_point(retained))


# ---------------------------------------------------------------------- solve


class IndexAssessor(Protocol):
    """An independent assessment of a whole index point (the certificate's role)."""

    def __call__(
        self, point: IndexWholePoint, controls: CertificateControls, /
    ) -> Assessment:
        """The assessment of ``point`` under the controls of the certificate."""
        ...


@dataclass(frozen=True, eq=False)
class _OpeningDispatch:
    """A private pair of assessors: ``opening`` for the retry rule's start only.

    Its call is ``regular``'s, unchanged.  Only :meth:`_Assessments.opening`, at the
    retry rule's assessment of the first attempt's initialized start, calls
    ``opening`` (module docstring).
    """

    regular: IndexAssessor
    opening: IndexAssessor

    def __call__(
        self, point: IndexWholePoint, controls: CertificateControls, /
    ) -> Assessment:
        """``regular``'s assessment of ``point``."""
        return self.regular(point, controls)


# ------------------------------------------------- the private equation-aware stop

ProbeOutcome = Literal["pass", "continue", "refused"]
REUSED = "a probe log serves one solve"
UNFINISHED = "the probe log was not finished by the solve of this candidate"


def array_digest(values: FloatArray) -> str:
    """The SHA-256 of the binary64 bytes of ``values``: a content binding, no copy."""
    found = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(found.tobytes()).hexdigest()


def _passes(maximum: float, tol: float) -> bool:
    """The equations' part of the combined native stop: ``maximum <= tol``."""
    return maximum <= tol


@dataclass(frozen=True)
class EquationProbe:
    """One fresh-equations probe of the private equation-aware stop (module docstring).

    ``attempt`` and ``step`` locate the accepted step it tested (its accepted steps in
    the attempt, that one counted); ``score`` and ``decrement`` are the two native
    formulas' values that passed there, ``max |G|`` and half the unhalved decrement at
    the direction's origin, against ``gradient_tol``; ``theta``, ``base`` and ``q`` the
    SHA-256 of the bytes of the accepted slopes, of their ``base_index`` (``None`` when
    it was refused) and of the accepted index; ``maximum`` the largest absolute fresh
    equation, ``None`` when refused, against ``tol``; ``outcome`` is ``pass``,
    ``continue`` or ``refused`` (``reason`` its refusal); ``stopped`` the step's
    combined native flag as the step hook received it (``None`` before the hook);
    ``wall`` and ``cpu`` the probe's own seconds, inside the iteration's and never
    added to them.
    """

    attempt: int
    step: int
    score: float
    decrement: float
    gradient_tol: float
    theta: str
    base: str | None
    q: str
    tol: float
    maximum: float | None
    outcome: ProbeOutcome
    reason: str
    stopped: bool | None
    wall: float
    cpu: float


@dataclass(frozen=True)
class StopProbes:
    """The frozen probes of one solve under the private equation-aware stop.

    ``tol`` bounds the fresh equations and ``gradient_tol`` the two native formulas,
    each read once at the solve's start; ``probes`` holds every probe, in order.  An
    empty tuple is a stop that ran and whose native formulas never passed.
    """

    tol: float
    gradient_tol: float
    probes: tuple[EquationProbe, ...]


class ProbeLog:
    """The probes of one solve as they are made (module docstring); one log per fit.

    :func:`solve` starts it with the two bounds before any work and finishes it with
    its candidate; each probe is logged when it is made and keeps its step's combined
    native flag once the step hook has received it.  A started log is refused by any
    other solve, and its receipt is that of the solve which finished it only.
    """

    def __init__(self) -> None:
        self.controls: tuple[float, float] | None = None
        self.probes: list[EquationProbe] = []
        self.candidate: IndexCandidate | None = None

    @property
    def active(self) -> bool:
        """Whether a solve started the log: its stop ran, whatever it probed."""
        return self.controls is not None

    def start(self, tol: float, gradient_tol: float) -> None:
        """A solve starts with its two bounds; ``ValueError`` for a started log."""
        if self.controls is not None:
            raise ValueError(REUSED)
        self.controls = (tol, gradient_tol)

    def made(self, probe: EquationProbe) -> None:
        """A probe, logged as it was made."""
        self.probes.append(probe)

    def stepped(self, attempt: int, step: int, stopped: bool) -> None:
        """The combined native flag of an accepted step, kept on that step's probe."""
        if not self.probes:
            return
        last = self.probes[-1]
        if (last.attempt, last.step, last.stopped) == (attempt, step, None):
            self.probes[-1] = replace(last, stopped=stopped)

    def finish(self, candidate: IndexCandidate) -> None:
        """The candidate of the solve that started the log; ``ValueError`` otherwise."""
        if self.controls is None or self.candidate is not None:
            raise ValueError(REUSED)
        self.candidate = candidate

    def receipt(self, candidate: object) -> StopProbes:
        """The frozen probes of the solve that returned ``candidate``.

        ``ValueError`` unless that very solve started and finished the log.
        """
        controls, finished = self.controls, self.candidate
        if controls is None or finished is None or finished is not candidate:
            raise ValueError(UNFINISHED)
        return StopProbes(controls[0], controls[1], tuple(self.probes))


@dataclass(frozen=True, eq=False)
class _EquationStop:
    """The private equation-aware stop: an assessor and the fit's own probe log.

    Its call is ``assess``'s, unchanged, and the retry rule's opening is ``assess``'s
    (:meth:`_Assessments.opening`).  :func:`solve` recognizes it by its exact type only
    and starts ``log`` (module docstring).
    """

    assess: IndexAssessor
    log: ProbeLog

    def __call__(
        self, point: IndexWholePoint, controls: CertificateControls, /
    ) -> Assessment:
        """``assess``'s assessment of ``point``."""
        return self.assess(point, controls)


class _Stop(NamedTuple):
    """The private equation-aware stop of one solve: its log, the equations' bound."""

    log: ProbeLog
    tol: float


def _started(assess: IndexAssessor | None, options: SolverOptions) -> _Stop | None:
    """The stop of a solve whose assessor is the exact private adapter, else ``None``.

    The adapter's log is started with ``tol`` and ``gradient_tol``, each read once
    here (``ValueError`` for a log that a solve started); no other assessor is read.
    """
    if type(assess) is _EquationStop:
        tol = options.tol
        assess.log.start(tol, options.gradient_tol)
        return _Stop(assess.log, tol)
    return None


class _ProbedEvaluator(IndexEvaluator):
    """The attempt state of the private equation-aware stop (module docstring).

    The native test of :class:`IndexEvaluator` and, only when both of its formulas pass
    at an accepted step, one probe of that accepted point; the step hook of
    :class:`IndexEvaluator` before the refusal of the step's probe.

    Parameters
    ----------
    context : IndexContext
        The fixed parts of the solve.
    stop : _Stop
        The solve's probe log and the bound of the equations.
    attempt : int
        The index of the attempt in the solve.
    origin : IndexStart, IndexPoint or None
        The origin of :class:`IndexEvaluator`.
    """

    def __init__(
        self,
        context: IndexContext,
        stop: _Stop,
        *,
        attempt: int,
        origin: IndexStart | IndexPoint | None,
    ) -> None:
        super().__init__(context, attempt=attempt, origin=origin)
        self.stop = stop
        self.refusal: FloatingPointError | None = None

    def converged(self, solver: Solver) -> bool:
        """The two native formulas, then, at their pass, the probe's ``maximum <= tol``.

        A refused probe is no stop: its refusal waits for the step hook.
        """
        if not super().converged(solver):
            return False
        probe = self._probe(solver.tol)
        self.stop.log.made(probe)
        return probe.outcome == "pass"

    def _probe(self, gradient_tol: float) -> EquationProbe:
        """The fresh equations at the accepted trial, bound and timed (docstring).

        Only ``FloatingPointError``, the refusal of the base or of the equations, is
        caught, and held for the step hook; nothing is replaced by a passing value.
        """
        accepted, problem, tol = self._accepted(), self.context.problem, self.stop.tol
        wall, cpu = time.perf_counter(), time.process_time()
        base: FloatArray | None = None
        maximum: float | None = None
        reason = ""
        try:
            base = base_index(problem, accepted.theta)
            maximum = fresh_equations(_prepared(problem, base), accepted.index).maximum
        except FloatingPointError as error:
            self.refusal, reason = error, str(error)
        outcome: ProbeOutcome = "refused"
        if maximum is not None:
            outcome = "pass" if _passes(maximum, tol) else "continue"
        binding = None if base is None else array_digest(base)
        return EquationProbe(
            attempt=self.attempt,
            step=self.steps + 1,
            score=float(np.max(np.abs(self._view().gradient))),
            decrement=0.5 * self._direction().decrement,
            gradient_tol=gradient_tol,
            theta=array_digest(accepted.theta),
            base=binding,
            q=array_digest(accepted.index),
            tol=tol,
            maximum=maximum,
            outcome=outcome,
            reason=reason,
            stopped=None,
            wall=time.perf_counter() - wall,
            cpu=time.process_time() - cpu,
        )

    def stepped(self, converged: bool, gradient: FloatArray) -> None:
        """The step's combined flag on its probe, the unchanged hook, then a refusal.

        The accepted step is counted and its indications run first, an alarm keeping
        its precedence; a refused probe of the step then ends the attempt as a
        numerical breakdown with the probe's reason.
        """
        self.stop.log.stepped(self.attempt, self.steps + 1, bool(converged))
        refusal, self.refusal = self.refusal, None
        super().stepped(converged, gradient)
        if refusal is not None:
            raise refusal


@dataclass(frozen=True, eq=False)
class _Context:
    """The fixed parts of one solve, shared by its attempts.

    ``stop`` is the private equation-aware stop of the solve, ``None`` without it.
    """

    sample: FESample
    variance: Family
    options: SolverOptions
    index: IndexContext
    stop: _Stop | None = None


class _Assessments:
    """The caller's assessments of whole points: each one made, counted and timed."""

    def __init__(self, assess: IndexAssessor, context: _Context) -> None:
        self.assess = assess
        self.context = context
        self.controls = controls_of(context.options)
        self.count = 0
        self.wall = 0.0
        self.cpu = 0.0

    def __call__(self, point: IndexPoint) -> Assessment:
        """The assessment of ``point``, which must be an :class:`Assessment`."""
        return self._made(self.assess, point)

    def opening(self, point: IndexPoint) -> Assessment:
        """The assessment of the retry rule's start (module docstring).

        The ``opening`` callback of the exact private pair, also when the private
        equation-aware stop holds it; for any other assessor, :meth:`__call__` itself.
        """
        assess = self.assess
        if type(assess) is _EquationStop:
            assess = assess.assess
        if type(assess) is _OpeningDispatch:
            return self._made(assess.opening, point)
        return self(point)

    def _made(self, assess: IndexAssessor, point: IndexPoint) -> Assessment:
        """One assessment of ``point`` by ``assess``: made, counted and timed."""
        whole = WholePoint(self.context.sample, self.context.variance, point)
        wall, cpu = time.perf_counter(), time.process_time()
        result = assess(whole, self.controls)
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
    callables: tuple[object, object],
    *,
    assess: IndexAssessor | None,
) -> None:
    """The preconditions of :func:`solve`, each refused with a built-in error."""
    if type(sample) is not FESample:
        raise TypeError("sample must be an FESample")
    if type(options) is not SolverOptions or type(absorbed) is not AbsorbedOptions:
        raise TypeError("options must be a SolverOptions and an AbsorbedOptions")
    trigger, admit = callables
    if not callable(trigger):
        raise TypeError("trigger must be the caller's callable")
    if not callable(admit):
        raise TypeError("admit must be the caller's admission of an outcome")
    if absorbed.absorbed != "index":
        raise ValueError("this is the fitted-index organization, not 'profiled'")
    if sample.omega == 0:
        raise ValueError("omega = 0 is the linear path, not a Newton solve")
    if options.retry == 1 and assess is None:
        raise ValueError("retry=1 needs an independent assessment of the whole point")


def _problem(sample: FESample, variance: Family) -> ProfiledProblem:
    """The accepted problem of the sample's final rows and active scaled columns."""
    grouping, design = sample.grouping, sample.design
    if grouping is None:
        raise ValueError("the fitted-index solver needs fixed-effect blocks")
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


def _bound_start(start: object, problem: ProfiledProblem) -> None:
    """Refuse a start that is not a full start of this problem (``ValueError``).

    The effects' binding is the accepted profile's: positions, names, codes and labels.
    """
    if type(start) is not IndexStart:
        raise ValueError("start must be an IndexStart")
    if start.slopes.shape != (problem.n_slopes,):
        raise ValueError("start must hold one value per slope column")
    _bound(start.effects, _prepared(problem, problem.offset))


class _Newton(NamedTuple):
    """The provider and the fixed arrays of the iteration of one solve."""

    provider: NewtonProvider
    problem: NewtonProblem


def _point_of(evaluator: IndexEvaluator) -> IndexPoint | None:
    """The accepted whole point of an attempt, its base derived; ``None`` without one.

    ``FloatingPointError`` when that base is not representable.
    """
    accepted, opening = evaluator.accepted, evaluator.opening
    if accepted is None or opening is None:
        return None
    if evaluator.previous is None:
        return opening
    base = base_index(evaluator.context.problem, accepted.theta)
    return IndexPoint(accepted.theta, base, accepted.index)


def _record(
    evaluator: IndexEvaluator,
    iteration: ProfiledIteration,
    stop: tuple[IndexReason, str],
    point: IndexPoint | None,
    *,
    final: IndexFinal | None = None,
    alarm: Alarm | None = None,
) -> IndexAttempt:
    """The record of an attempt that stopped for ``stop``: its reason and message."""
    accepted = evaluator.accepted
    return IndexAttempt(
        iteration,
        stop[0],
        stop[1],
        evaluator.opening,
        point,
        None if accepted is None else accepted.loss,
        None if accepted is None else accepted.gradient,
        evaluator.direction_found,
        evaluator.initializer,
        final,
        alarm,
    )


def _finished(evaluator: IndexEvaluator, iteration: ProfiledIteration) -> IndexAttempt:
    """The attempt of an iteration: its operative reason and its final work.

    The fresh final work runs after a native stop only.  An accepted point whose base
    is not representable is a numerical breakdown of the attempt.
    """
    try:
        point = _point_of(evaluator)
    except FloatingPointError as error:
        return _record(evaluator, iteration, ("numerical_breakdown", str(error)), None)
    if iteration.native is None:
        reason: IndexReason = "alarm"
        if iteration.stop == "inner_limit":
            reason = "inner_limit"
        stop = (reason, iteration.message)
        return _record(evaluator, iteration, stop, point, alarm=evaluator.alarm)
    accepted = evaluator.accepted
    if iteration.native != "native_stop" or point is None or accepted is None:
        return _record(
            evaluator, iteration, (iteration.native, iteration.message), point
        )
    final = final_work(evaluator, point, accepted.loss)
    reason = "native_stop"
    if final.alarm is not None:
        reason = "alarm"
    elif not final.complete:
        reason = "final_failure"
    stop = (reason, final.reason)
    return _record(evaluator, iteration, stop, point, final=final, alarm=final.alarm)


def _attempt(
    context: _Context,
    newton: _Newton,
    origin: IndexStart | IndexPoint | None,
    *,
    attempt: int,
    max_iter: int,
) -> IndexAttempt:
    """One attempt from ``origin``: the caller's start, a retained point, or none.

    Under the solve's private equation-aware stop its state is a
    :class:`_ProbedEvaluator`, a retry's included; otherwise an :class:`IndexEvaluator`.
    """
    problem = context.index.problem
    if isinstance(origin, IndexPoint):
        coefficients = origin.theta
    elif isinstance(origin, IndexStart):
        coefficients = origin.slopes
    else:
        coefficients = np.zeros(problem.n_slopes)
    stop = context.stop
    evaluator: IndexEvaluator
    if stop is None:
        evaluator = IndexEvaluator(context.index, attempt=attempt, origin=origin)
    else:
        evaluator = _ProbedEvaluator(
            context.index, stop, attempt=attempt, origin=origin
        )
    iteration = run_index_attempt(
        newton.provider,
        context.index.point,
        newton.problem,
        coefficients,
        context.options,
        evaluation=evaluator,
        max_iter=max_iter,
        budget=context.index.admissions,
    )
    return _finished(evaluator, iteration)


class _Attempts(NamedTuple):
    """The attempts of a solve, the one retained, and what the retry rule read."""

    attempts: tuple[IndexAttempt, ...]
    retained: int
    initial: Assessment | None
    no_measurable_progress: bool


def _opened(attempt: IndexAttempt) -> IndexPoint:
    """The initialized start of an attempt that evaluated it."""
    if attempt.opening is None:
        raise ValueError("the attempt did not initialize its start")
    return attempt.opening


def _attempts(
    context: _Context,
    newton: _Newton,
    start: IndexStart | None,
    assess: _Assessments | None,
) -> _Attempts:
    """The first attempt and, if the existing rule allows it, the second.

    The statements of :func:`powermean.solve.newton_retry.run_attempts`, read on the
    operative reason and on whole index points; one budget for both attempts.
    """
    options, budget = context.options, context.index.admissions
    first = _attempt(context, newton, start, attempt=0, max_iter=options.max_iter)
    if assess is None or options.retry == 0 or first.reason != "native_stop":
        return _Attempts((first,), 0, None, False)
    origin = stopped_point(first)
    assessed = assess(origin)
    first = replace(first, assessment=assessed)
    budgets_remain = first.iterations < options.max_iter and budget.used < budget.limit
    if assessed.accepted or not assessed.finite_error or not budgets_remain:
        return _Attempts((first,), 0, None, False)
    initial = assess.opening(_opened(first))
    if not improved(assessed, initial):
        return _Attempts((first,), 0, initial, False)
    left = options.max_iter - first.iterations
    second = _attempt(context, newton, origin, attempt=1, max_iter=left)
    reassessed = assess(stopped_point(second))
    second = replace(second, assessment=reassessed)
    no_progress, keep_first = verdict(assessed, reassessed)
    return _Attempts((first, second), 0 if keep_first else 1, initial, no_progress)


def _request(context: _Context, attempt: IndexAttempt, alarm: Alarm) -> TriggerRequest:
    """The trigger's request: the stopped whole point, the alarm, TIGHTER controls."""
    variance = context.variance
    gpml = type(variance) is PowerVariance and variance.tau == 2.0
    point = WholePoint(context.sample, variance, stopped_point(attempt))
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
    assess: IndexAssessor | None = None,
    start: IndexStart | None = None,
) -> IndexCandidate:
    """Solve the joint problem of ``sample`` in the fitted index: a candidate, a trace.

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
        The explicit absorbed options; ``absorbed`` must be ``'index'``.
    trigger : Trigger
        The caller's trigger, bound to the actual completed outcome of the sample
        stage; called at most once, after an alarm.
    admit : Admission
        The caller's admission of the trigger call's outcome, the coordinator's own;
        only an outcome it admits is kept, apart from the stopped solve.
    assess : IndexAssessor, optional
        The independent assessment of a whole index point; required with ``retry=1``.
        The exact private adapter :class:`_EquationStop` also selects the private
        equation-aware stop for every attempt (module docstring).
    start : IndexStart, optional
        The caller's full start (slopes and a coefficient witness bound to the
        sample's grouping); the automatic start when omitted.

    Returns
    -------
    IndexCandidate
        The candidate and its trace; no claim of convergence or existence.

    Raises
    ------
    SpecificationError
        ``newton_provider="sklearn"`` without a verified installed scikit-learn.
    TypeError, ValueError
        A precondition does not hold, or the adapter's probe log served another
        solve; nothing is run.
    """
    _check(sample, options, absorbed, (trigger, admit), assess=assess)
    stop = _started(assess, options)
    call = _IndexCall(
        sample, variance, options, absorbed, trigger, admit, assess, start
    )
    return _run_index(call, _problem(sample, variance), stop)


@dataclass(frozen=True)
class _IndexCall:
    """Unchanged index arguments, independent of the private owned representation."""

    sample: FESample
    variance: Family
    options: SolverOptions
    absorbed: AbsorbedOptions
    trigger: Trigger
    admit: Admission
    assess: IndexAssessor | None
    start: IndexStart | None


def _solve_numeric(call: _IndexCall, problem: ProfiledProblem) -> IndexCandidate:
    """Private owned entry to the SAME index loop, start and retry organization."""
    _check(call.sample, call.options, call.absorbed, (call.trigger, call.admit),
           assess=call.assess)
    stop = _started(call.assess, call.options)
    return _run_index(call, problem, stop)


def _run_index(
    call: _IndexCall, problem: ProfiledProblem, stop: _Stop | None
) -> IndexCandidate:
    """The single existing index execution body, with its original ordering."""
    sample, variance, options, absorbed = (
        call.sample, call.variance, call.options, call.absorbed
    )
    trigger, admit, assess, start = call.trigger, call.admit, call.assess, call.start
    if start is not None:
        _bound_start(start, problem)
    provider = resolve_provider(options.newton_provider)
    wall, cpu = time.perf_counter(), time.process_time()
    budget = Admissions(options.max_evaluations)
    zero_offset = np.zeros(problem.n_rows)
    zero_offset.flags.writeable = False
    point = ProfiledPointLoss(
        problem,
        offset=zero_offset,
        variance=variance,
        curvature=options.curvature,
        budget=budget,
    )
    index = IndexContext(
        problem,
        absorbed,
        options.curvature,
        sample.positions,
        IndexLog(),
        point,
        budget,
        options.gradient_tol,
    )
    context = _Context(sample, variance, options, index, stop)
    assessments = None if assess is None else _Assessments(assess, context)
    arrays = NewtonProblem(problem.design, problem.z, problem.weights)
    outcome = _attempts(context, _Newton(provider, arrays), start, assessments)
    last = outcome.attempts[-1]
    record = None
    if last.alarm is not None:
        record = triggered(trigger, admit, _request(context, last, last.alarm))
    timing = Timing(0.0, 0.0)
    if assessments is not None:
        timing = Timing(assessments.wall, assessments.cpu)
    candidate = IndexCandidate(
        sample=sample,
        variance=variance,
        start=start,
        options=options,
        absorbed=absorbed,
        provider=provider.name,
        provider_version=provider.version,
        factorized=bool(problem.n_slopes),
        attempts=outcome.attempts,
        retained=outcome.retained,
        initial_assessment=outcome.initial,
        no_measurable_progress=outcome.no_measurable_progress,
        assessments=0 if assessments is None else assessments.count,
        assessment_timing=timing,
        work=index.log.frozen(),
        trigger=record,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
    )
    if stop is not None:
        stop.log.finish(candidate)
    return candidate
