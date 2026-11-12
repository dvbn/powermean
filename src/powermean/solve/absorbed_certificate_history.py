"""Admitted snapshots of absorbed candidates: reasons, selected points, partial history.

A solver proposes, the certificate decides.  Internal module.  No certificate verdict
is formed here, and no optimizer is replayed.

Admission (:func:`profiled_history`, :func:`index_history`, :func:`linear_history`)
checks the coherence of the evidence, not that recorded work established success: the
exact candidate type, sample, family, options and absorbed organization; one or two
attempts and a valid retained index; every logged call under the candidate's controls,
naming one of its attempts and a step within that attempt's accepted steps, with
counters that are nonnegative integers within the call's budget and absent exactly
when it was refused; no final call for an attempt without final work; each final
record admitted by :mod:`powermean.solve.absorbed_certificate_final` (its owned parts,
the accepted sequence, its partials bound to its OWN final point, its calls bound to
what they returned); and the operative reason by the accepted precedence: a final alarm
gives ``alarm``, else a failed final gives ``final_failure``, else ``native_stop``
(``profile_only`` on the profiled ``K = 0`` branch).  Other reasons stay historical
facts.  The WLS candidate is re-admitted by its accepted constructor
(``dataclasses.replace``), which rebinds every record.

The snapshot keeps the sample and family, retained index, ordered attempts, operative
reason and message, native reason (``None`` without an iteration), the retained final
record with its present and unreached parts, alarm, trigger, historical work and
timing; unavailable parts stay ``None``, never zero or success.  An attempt whose final
record had to be re-formed is held with that owned record (and its alarm), and the
candidate with those attempts; otherwise both are the caller's objects.  The selected
point is the accepted ``stopped_point`` rule: a profiled final point only when that
final is complete, an index final point whenever a final exists, else the attempt's
point; the WLS point attribute if it was formed.  ``ValueError`` (``TypeError`` for a
wrong type) refuses an incoherent history.

Root candidates (:func:`root_history`, route ``"root"``;
:mod:`powermean.solve.absorbed_root`).  The coherence of an
:class:`~powermean.solve.absorbed_root_records.AbsorbedRootCandidate`: the exact
sample, family (a power variance outside ``[1, 2]``), options, root options and the
requested organization's options; the initializer: a caller's start (uncertified, no
convex candidate) or the nearest convex member (PPML below 1, Gamma above 2) solved
by the requested organization on the SAME sample object, its certified flag, status
and reason those of its own audit; stages on variances outside ``[1, 2]``, the
direct one on the target, each with one or two attempts and a valid retained one, the
native reason that of the target stage's retained attempt, a merit wherever a stage
ran; a selected point and its certificate together or neither, none without a stage;
an operative reason in :data:`ROOT_REASONS`.  The snapshot's attempts are the
stages, ``retained`` the target stage; no final work, alarm or trigger belongs to the
root stages (the initializer's are in its own audit).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from powermean.data.sample_fe_state import FESample, same_bits
from powermean.fe.absorb import AbsorptionResult
from powermean.fe.profile import Work
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.solve.absorbed_certificate_final import (
    REFUSED,
    index_final,
    logged_final,
    profiled_final,
    require,
)
from powermean.solve.absorbed_certificate_point import problem_of
from powermean.solve.absorbed_direction import IndexPoint, IndexWork
from powermean.solve.absorbed_rhs import RHSCall
from powermean.solve.absorbed_root_records import (
    AbsorbedRootCandidate,
    AbsorbedRootInitializer,
    AbsorbedRootStage,
    AbsorbedRootWork,
    RootPoint,
)
from powermean.solve.cancelled_records import (
    RootAttemptRecord,
    RootOptions,
    iter_continuation_variances,
)
from powermean.solve.certificate import outside_convex
from powermean.solve.linear_absorbed_point import LinearPoint
from powermean.solve.linear_absorbed_state import AbsorbedLinearCandidate, LinearWork
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexAttempt, IndexCandidate, IndexFinal
from powermean.solve.newton_profiled import ProfiledAttempt, ProfiledCandidate
from powermean.solve.newton_trace import Timing
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_indications import Alarm
from powermean.solve.profiled_loss import (
    AbsorptionCall,
    FinalWork,
    ProfileCall,
    SolveWork,
)
from powermean.solve.profiled_trigger import TriggerRecord
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.systems.profiled import PointProfile, ProfiledProblem

Route = Literal["profiled", "reduced", "index", "linear", "root"]
Family = PowerVariance | NB2Variance
FinalRecord = FinalWork | IndexFinal | AbsorptionResult
NATIVE = (
    "native_stop",
    "iteration_limit",
    "evaluation_limit",
    "line_search_failure",
    "factorization_failure",
    "non_descent_direction",
    "numerical_breakdown",
)
FINAL_REASONS = ("native_stop", "final_failure", "profile_only")
# An uncertified convex initializer's operative reason is the root candidate's.
ROOT_REASONS = (*NATIVE, "inner_limit", "alarm", "final_failure", "profile_only")


@dataclass(frozen=True, eq=False)
class CandidateHistory:
    """The admitted history of one candidate (module docstring); no verdict.

    ``partials`` and ``unreached`` name the parts of the retained final record that
    are present and absent, in the order of the final work; ``point`` is the selected
    point, ``None`` when the candidate holds none.
    """

    route: Route
    candidate: (
        ProfiledCandidate
        | IndexCandidate
        | AbsorbedLinearCandidate
        | AbsorbedRootCandidate
    )
    sample: FESample
    variance: Family | None
    retained: int | None
    attempts: (
        tuple[ProfiledAttempt, ...]
        | tuple[IndexAttempt, ...]
        | tuple[AbsorbedRootStage, ...]
    )
    reason: str
    message: str
    native: str | None
    final: FinalRecord | None
    partials: tuple[str, ...]
    unreached: tuple[str, ...]
    alarm: Alarm | None
    trigger: TriggerRecord | None
    work: SolveWork | IndexWork | LinearWork | AbsorbedRootWork
    timing: Timing | None
    point: PointProfile | ConditionalPoint | IndexPoint | LinearPoint | RootPoint | None


def _expected(final: FinalWork | IndexFinal, k: int, profiled: bool) -> str:
    """The accepted operative reason of an attempt with ``final``."""
    if final.alarm is not None:
        return "alarm"
    if not final.complete:
        return "final_failure"
    return "profile_only" if profiled and not k else "native_stop"


def _parts(
    names: tuple[str, ...], values: tuple[object, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The names of the present and of the absent parts, in order."""
    present = tuple(n for n, v in zip(names, values, strict=True) if v is not None)
    return present, tuple(n for n in names if n not in present)


# ---------------------------------------------------------------- the calls


def _count(value: object) -> bool:
    """Whether ``value`` is a nonnegative integer, never a boolean."""
    return type(value) is int and value >= 0


def _counted(call: AbsorptionCall | RHSCall) -> bool:
    """A logged absorption or RHS call's counters: absent iff refused, else in range."""
    counts = (call.sweeps, call.visits, call.operators, call.diagnostic_passes)
    if call.status == REFUSED:
        return all(c is None for c in counts) and call.maximum is None
    sweeps = call.sweeps
    in_budget = sweeps is not None and sweeps <= call.max_sweeps
    return all(_count(c) for c in counts) and in_budget


def _profile_counted(call: ProfileCall) -> bool:
    """A logged profile call's work: absent iff refused, else exact and in order."""
    work = call.work
    if call.status == REFUSED:
        return work is None and call.scalar is None and call.maximum is None
    if type(work) is not Work:
        return False
    pairs = (
        (work.sweeps_attempted, work.sweeps_completed),
        (work.blocks_attempted, work.blocks_completed),
        (work.checks_attempted, work.checks_completed),
    )
    return all(_count(a) and _count(c) and c <= a for a, c in pairs)


def _controls(candidate: ProfiledCandidate | IndexCandidate) -> None:
    """Every logged call ran under the candidate's controls, within its domains."""
    options, work, attempts = candidate.absorbed, candidate.work, candidate.attempts
    budget, flag = options.absorb_max_sweeps, options.absorb_acceleration
    rhs: tuple[RHSCall, ...] = work.rhs if isinstance(work, IndexWork) else ()
    calls: list[AbsorptionCall | RHSCall] = [*work.absorptions, *rhs]
    require(all(c.max_sweeps == budget and c.acceleration is flag for c in calls))
    require(all(p.max_sweeps in (options.profile_max_sweeps, 1) for p in work.profiles))
    for call in (*work.profiles, *calls):
        attempt, step = call.attempt, call.iteration
        require(type(attempt) is int and 0 <= attempt < len(attempts))
        require(type(step) is int and 0 <= step <= attempts[attempt].iterations)
    require(all(_profile_counted(c) for c in work.profiles))
    grouping = candidate.sample.grouping
    for call in work.profiles:
        if call.scope == "conditional":
            require(type(candidate) is ProfiledCandidate
                    and candidate.execution == "reduced_nonlinear")
            require(type(call.block) is int and grouping is not None
                    and 0 <= call.block < len(grouping.blocks))
        else:
            require(call.scope == "all_fe" and call.block is None)
    require(all(_counted(c) for c in calls))


def _common(
    candidate: ProfiledCandidate | IndexCandidate, organization: str
) -> ProfiledProblem:
    """The types, controls and retained index of a nonlinear candidate; its problem."""
    sample, absorbed = candidate.sample, candidate.absorbed
    if type(candidate) is ProfiledCandidate:
        require(candidate.execution in ("profiled", "reduced_nonlinear"))
    require(type(sample) is FESample and type(candidate.options) is SolverOptions)
    require(type(absorbed) is AbsorbedOptions and absorbed.absorbed == organization)
    attempts, retained = candidate.attempts, candidate.retained
    require(isinstance(attempts, tuple) and 1 <= len(attempts) <= 2)
    require(type(retained) is int and 0 <= retained < len(attempts))
    return problem_of(sample, candidate.variance)


def _unfinished(candidate: ProfiledCandidate | IndexCandidate, index: int) -> None:
    """An attempt without final work: no final reason and no logged final call."""
    attempt = candidate.attempts[index]
    require(attempt.reason not in FINAL_REASONS)
    require(not any(logged_final(candidate.work, index)))


# ------------------------------------------------------------------- profiled


def _profiled_attempt(
    problem: ProfiledProblem, candidate: ProfiledCandidate, index: int
) -> FinalWork | None:
    """One profiled attempt: its reason and alarm; its owned final record."""
    attempt = candidate.attempts[index]
    require(type(attempt) is ProfiledAttempt)
    allowed = (*NATIVE, "inner_limit", "alarm", "final_failure", "profile_only")
    require(attempt.reason in allowed)
    if attempt.final is None:
        _unfinished(candidate, index)
        return None
    expected = _expected(
        attempt.final, problem.n_slopes,
        profiled=candidate.execution == "profiled",
    )
    require(attempt.reason == expected and attempt.alarm is attempt.final.alarm)
    return profiled_final(problem, candidate, index, attempt)


def profiled_history(candidate: object) -> CandidateHistory:
    """The admitted history of a :class:`ProfiledCandidate` (module docstring)."""
    if type(candidate) is not ProfiledCandidate:
        raise TypeError("candidate must be a ProfiledCandidate")
    problem = _common(candidate, "profiled")
    _controls(candidate)
    count = range(len(candidate.attempts))
    finals = tuple(_profiled_attempt(problem, candidate, i) for i in count)
    attempts = tuple(
        a if f is None or f is a.final else replace(a, final=f, alarm=f.alarm)
        for a, f in zip(candidate.attempts, finals, strict=True)
    )
    if attempts != candidate.attempts:
        candidate = replace(candidate, attempts=attempts)
    attempt = attempts[candidate.retained]
    final = attempt.final
    point = final.point if final is not None and final.complete else attempt.point
    names = ("point", "equations", "absorbed", "alarm")
    present: tuple[str, ...] = ()
    unreached: tuple[str, ...] = ()
    if final is not None:
        values = (final.point, final.equations, final.absorbed, final.alarm)
        present, unreached = _parts(names, values)
    native = None if attempt.iteration is None else attempt.iteration.native
    return CandidateHistory(
        "profiled" if candidate.execution == "profiled" else "reduced",
        candidate,
        candidate.sample,
        candidate.variance,
        candidate.retained,
        attempts,
        attempt.reason,
        attempt.message,
        native,
        final,
        present,
        unreached,
        attempt.alarm,
        candidate.trigger,
        candidate.work,
        candidate.timing,
        point,
    )


# ---------------------------------------------------------------------- index


def _index_attempt(
    problem: ProfiledProblem, candidate: IndexCandidate, index: int
) -> IndexFinal | None:
    """One index attempt: its reason and alarm; its owned final record."""
    attempt = candidate.attempts[index]
    require(type(attempt) is IndexAttempt)
    require(attempt.reason in (*NATIVE, "inner_limit", "alarm", "final_failure"))
    if attempt.final is None:
        _unfinished(candidate, index)
        return None
    expected = _expected(attempt.final, problem.n_slopes, profiled=False)
    require(attempt.reason == expected and attempt.alarm is attempt.final.alarm)
    return index_final(problem, candidate, index, attempt)


def index_history(candidate: object) -> CandidateHistory:
    """The admitted history of an :class:`IndexCandidate` (module docstring)."""
    if type(candidate) is not IndexCandidate:
        raise TypeError("candidate must be an IndexCandidate")
    problem = _common(candidate, "index")
    _controls(candidate)
    count = range(len(candidate.attempts))
    finals = tuple(_index_attempt(problem, candidate, i) for i in count)
    attempts = tuple(
        a if f is None or f is a.final else replace(a, final=f, alarm=f.alarm)
        for a, f in zip(candidate.attempts, finals, strict=True)
    )
    if attempts != candidate.attempts:
        candidate = replace(candidate, attempts=attempts)
    attempt = attempts[candidate.retained]
    final = attempt.final
    point = final.point if final is not None else attempt.point
    names = ("recovery", "rows", "equations", "absorbed", "rhs", "alarm")
    present: tuple[str, ...] = ()
    unreached: tuple[str, ...] = ()
    if final is not None:
        values = (
            final.point.recovered,
            final.rows,
            final.equations,
            final.absorbed,
            final.rhs,
            final.alarm,
        )
        present, unreached = _parts(names, values)
    return CandidateHistory(
        "index",
        candidate,
        candidate.sample,
        candidate.variance,
        candidate.retained,
        attempts,
        attempt.reason,
        attempt.message,
        attempt.iteration.native,
        final,
        present,
        unreached,
        attempt.alarm,
        candidate.trigger,
        candidate.work,
        candidate.timing,
        point,
    )


# ----------------------------------------------------------------------- root


def _convex_status(initializer: AbsorbedRootInitializer) -> tuple[bool, str, str]:
    """The certified flag, status and reason of a convex initializer's own audit."""
    audit = initializer.audit
    require(audit is not None)
    assert audit is not None
    certificate = audit.certificate
    certified = certificate is not None and certificate.accepted
    reason = audit.history.reason
    status = reason
    if certified:
        assert certificate is not None
        status = "verified" if certificate.level == "strict" else "verified_secondary"
    return certified, status, reason


def _root_initializer(candidate: AbsorbedRootCandidate) -> None:
    """The start of a root candidate: a caller's, or the same-sample convex fit."""
    initializer = candidate.initializer
    require(type(initializer) is AbsorbedRootInitializer)
    start = initializer.point
    require(start is None or type(start) is RootPoint)
    require(start is not None or not candidate.stages)
    gap = initializer.gap
    if initializer.kind == "user_start":
        require(candidate.start is not None)
        require(initializer.variance is None and not initializer.certified)
        require(initializer.candidate is None and initializer.audit is None)
        require(gap is None and initializer.status is None)
        require(initializer.reason is None)
        return
    below = candidate.variance.tau < 1.0
    kind, tau = ("ppml_tau1", 1.0) if below else ("gamma_tau2", 2.0)
    require(candidate.start is None and initializer.kind == kind)
    require(initializer.variance == PowerVariance(tau))
    convex = initializer.candidate
    require(type(convex) is IndexCandidate or type(convex) is ProfiledCandidate)
    assert convex is not None
    require(convex.sample is candidate.sample)
    require(convex.variance == initializer.variance)
    require(convex.absorbed == candidate.absorbed)
    require(convex.options == candidate.options)
    audit = initializer.audit
    require(audit is not None)
    assert audit is not None
    require(audit.history.sample is candidate.sample)
    require(audit.history.candidate is convex)
    require(audit.history.variance == initializer.variance)
    certified, status, reason = _convex_status(initializer)
    require(initializer.certified == certified)
    require(initializer.status == status and initializer.reason == reason)
    require(gap is None or (type(gap) is float and gap >= 0.0))


def _root_stages(candidate: AbsorbedRootCandidate) -> None:
    """The stages, the target stage and its native reason."""
    stages, target = candidate.stages, candidate.target_stage
    require(isinstance(stages, tuple))
    require(candidate.continuation_used == (len(stages) > 1))
    if not stages:
        require(target is None and candidate.native is None)
        require(candidate.point is None)
        return
    require(candidate.merit is not None)
    require(type(target) is int and 0 <= target < len(stages))
    assert target is not None
    for stage in stages:
        require(type(stage) is AbsorbedRootStage)
        require(type(stage.variance) is PowerVariance)
        require(outside_convex(stage.variance))
        attempts, retained = stage.attempts, stage.retained
        require(isinstance(attempts, tuple) and 1 <= len(attempts) <= 2)
        require(type(retained) is int and 0 <= retained < len(attempts))
    require(stages[0].variance == candidate.variance)
    start = candidate.initializer.point
    require(start is not None)
    assert start is not None
    opening = start.gamma
    for position, stage in enumerate(stages):
        _root_attempts(candidate, stage)
        require(same_bits(stage.attempts[0].opening, opening))
        chosen = stage.attempts[stage.retained]
        opening = start.gamma if position == 0 else chosen.point
    _root_continuation(candidate)
    final = stages[target]
    require(candidate.native == final.attempts[final.retained].native)


def _root_attempts(candidate: AbsorbedRootCandidate, stage: AbsorbedRootStage) -> None:
    """Exact typed attempts, openings, dimensions, counts and summary verdict."""
    start = candidate.initializer.point
    assert start is not None
    for attempt in stage.attempts:
        require(type(attempt) is RootAttemptRecord and attempt.native in NATIVE)
        counts = (attempt.iterations, attempt.evaluations, attempt.backtracks)
        require(all(_count(count) for count in counts))
        require(attempt.iterations <= candidate.root_options.root_max_iter)
        require(attempt.opening.shape == start.gamma.shape)
        require(attempt.point.shape == start.gamma.shape)
        require(np.isfinite(attempt.opening).all() and np.isfinite(attempt.point).all())
    if len(stage.attempts) == 2:
        first, second = stage.attempts
        require(candidate.options.retry == 1 and first.native == "native_stop")
        require(same_bits(second.opening, first.point))
    require(stage.iterations == sum(a.iterations for a in stage.attempts))
    require(stage.evaluations == sum(a.evaluations for a in stage.attempts))
    summary = stage.summary
    level = (
        "strict"
        if not summary.strict
        else "secondary"
        if not summary.secondary
        else None
    )
    require(summary.level == level)
    if level is not None:
        require(
            stage.status == ("verified" if level == "strict" else "verified_secondary")
        )


def _root_continuation(candidate: AbsorbedRootCandidate) -> None:
    """Only the canonical continuation prefix, from each last certified point."""
    stages = candidate.stages
    reached = len(stages) - 1 if stages[-1].variance == candidate.variance else 0
    require(candidate.target_stage == reached)
    if len(stages) == 1:
        return
    require(candidate.initializer.certified)
    require(candidate.root_options.continuation == "fallback")
    require(stages[0].summary.level is None)
    origin = 1.0 if candidate.variance.tau < 1 else 2.0
    schedule = tuple(
        iter_continuation_variances(
            candidate.variance,
            candidate.root_options.continuation_step,
            len(stages) - 1,
            origin=origin,
        )
    )
    require(tuple(stage.variance for stage in stages[1:]) == schedule)
    require(all(stage.summary.level is not None for stage in stages[1:-1]))
    direct = stages[0]
    require(
        candidate.initializer.iterations + direct.iterations
        < candidate.options.max_iter
    )
    require(
        candidate.initializer.evaluations + direct.evaluations
        < candidate.options.max_evaluations
    )


def _root_work(candidate: AbsorbedRootCandidate) -> None:
    """Bind aggregate root counts to actual stages, directions and merit actions."""
    work = candidate.work
    require(type(work) is AbsorbedRootWork)
    counts = work.counts.counts
    require(len({c.name for c in counts}) == len(counts))
    require(
        all(
            _count(c.begun) and _count(c.completed) and c.completed <= c.begun
            for c in counts
        )
    )
    begun = {c.name: c.begun for c in counts}
    require(begun.get("root_equations", 0) == candidate.root_evaluations)
    require(begun.get("root_hessian", 0) == len(work.directions))
    require(all(_count(count) for count in work.merit_actions.values()))
    completed = {c.name: c.completed for c in counts}
    require(
        completed.get("merit_solve", 0)
        <= work.merit_actions.get("triangular", 0)
        <= begun.get("merit_solve", 0)
    )
    require(all(_count(count) for d in work.directions for count in d.actions.values()))
    # The initializer can itself stop on its budget; no invented root work follows.
    require(
        candidate.root_iterations
        <= max(0, candidate.options.max_iter - candidate.initializer.iterations)
    )
    require(
        candidate.root_evaluations
        <= max(0, candidate.options.max_evaluations - candidate.initializer.evaluations)
    )


def root_history(candidate: object) -> CandidateHistory:
    """The admitted history of an :class:`AbsorbedRootCandidate` (module docstring)."""
    if type(candidate) is not AbsorbedRootCandidate:
        raise TypeError("candidate must be an AbsorbedRootCandidate")
    sample, variance = candidate.sample, candidate.variance
    require(type(sample) is FESample and sample.omega != 0)
    require(sample.grouping is not None)
    require(type(candidate.options) is SolverOptions)
    require(type(candidate.root_options) is RootOptions)
    absorbed = candidate.absorbed
    require(type(absorbed) is AbsorbedOptions)
    require(absorbed.absorbed in ("index", "profiled"))
    require(type(variance) is PowerVariance and outside_convex(variance))
    _root_initializer(candidate)
    _root_stages(candidate)
    point, certificate = candidate.point, candidate.certificate
    require((point is None) == (certificate is None))
    require(point is None or type(point) is RootPoint)
    require(candidate.reason in ROOT_REASONS)
    _root_work(candidate)
    return CandidateHistory(
        "root",
        candidate,
        sample,
        variance,
        candidate.target_stage,
        candidate.stages,
        candidate.reason,
        candidate.message,
        candidate.native,
        None,
        (),
        (),
        None,
        None,
        candidate.work,
        candidate.timing,
        point,
    )


# ------------------------------------------------------------------------ WLS


def linear_history(candidate: object) -> CandidateHistory:
    """The admitted history of an :class:`AbsorbedLinearCandidate`, re-admitted."""
    if type(candidate) is not AbsorbedLinearCandidate:
        raise TypeError("candidate must be an AbsorbedLinearCandidate")
    admitted = replace(candidate)
    names = ("prepared", "joint", "seed", "final", "point", "normalized", "equations")
    present, unreached = _parts(names, tuple(getattr(admitted, n) for n in names))
    return CandidateHistory(
        "linear",
        admitted,
        admitted.sample,
        None,
        None,
        (),
        admitted.status,
        admitted.reason,
        None,
        admitted.final,
        present,
        unreached,
        None,
        None,
        admitted.work,
        None,
        admitted.point,
    )
