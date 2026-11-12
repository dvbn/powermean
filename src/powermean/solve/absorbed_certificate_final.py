"""The retained final work of an absorbed candidate: owned parts, sequence and calls.

Internal module: the result-local admission of one attempt's final record, used by
:mod:`powermean.solve.absorbed_certificate_history`.  No certificate verdict is formed
and no optimizer is replayed.

Owned parts (:mod:`powermean.solve.absorbed_certificate_final_parts`).  The final record
is re-formed first so that it keeps no array that another holder can write: each array
of the records that do not own their arrays at construction (the final profile's base,
the point equations, the final rows, the absorbed weights and Schur matrix, the recovery
effects and normalization, the alarms) that is not read-only over bytes of its own is
replaced by such a copy, of the same dtype, shape and bytes.  A record whose arrays are
all closed, as every record of the accepted producers is, is kept with its identity.
The checks read the re-formed record, which is the one retained.

Sequence (:func:`walk`).  The parts are those of the accepted ``final_work``, in order,
each reached only after the previous one succeeded.  Profiled: the fresh profile, the
rows at its OWN index (recomputed bit for bit), the cold absorption with the solve's
curvature when there are slopes.  Index: the recovery, the rows and the equations at
the attempt's ``q`` (recomputed), the absorption when there are slopes, the full
right-hand-side solve.  The first part that is absent or failed ends the work with its
reason: a failed part's own, or, for a part refused by range, a nonempty reason that
its logged call (if it logs one) holds as a refused call; nothing later is present or
logged, and no alarm.  After every part succeeded, a reason means a refused alarm stage.

Calls.  A final call exists only where the sequence reached it.  A returned record's
call holds its status, reason, counters, largest equation and controls; a final full
right-hand side's call also its forcing (in ``[0, 0.9999]``), bounds, ``sigma``, native
flag and guard, which holds ``af``, the rows' weight total and ``gradient_tol``.
``ValueError`` (:data:`INCOHERENT`) otherwise.
"""

from __future__ import annotations

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.fe.profile import _bound
from powermean.fe.profile_arithmetic import EquationCheck, fresh_equations
from powermean.solve.absorbed_certificate_final_parts import (
    INCOHERENT,
    owned_index,
    owned_profiled,
)
from powermean.solve.absorbed_direction import IndexPoint, IndexWork, Recovery
from powermean.solve.absorbed_loss import FinalRows, final_rows
from powermean.solve.absorbed_rhs import FORCING_CAP, RHSCall, RHSGuard, RHSResult
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexAttempt, IndexCandidate, IndexFinal
from powermean.solve.newton_profiled import ProfiledAttempt, ProfiledCandidate
from powermean.solve.profiled_indications import Alarm
from powermean.solve.profiled_loss import (
    AbsorptionCall,
    FinalWork,
    ProfileCall,
    SolveWork,
)
from powermean.systems.profiled import (
    Absorbed,
    PointEquations,
    PointProfile,
    ProfiledProblem,
    _prepared,
    point_equations,
)

REFUSED = "refused"
Call = ProfileCall | AbsorptionCall | RHSCall
# a part (None when absent), whether it succeeded, the reason its failure gives,
# whether it logs a call, and that call (None when none is logged)
Stage = tuple[object, bool, str, bool, Call | None]
Logged = tuple[tuple[ProfileCall, ...], tuple[AbsorptionCall, ...], tuple[RHSCall, ...]]


def require(condition: object) -> None:
    """``ValueError`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(INCOHERENT)


def identical(found: object, given: object) -> bool:
    """Whether two arrays (or tuples of arrays) are identical bit for bit."""
    if isinstance(found, tuple) and isinstance(given, tuple):
        return len(found) == len(given) and all(
            identical(a, b) for a, b in zip(found, given, strict=True)
        )
    return same_bits(
        np.asarray(found, dtype=np.float64), np.asarray(given, dtype=np.float64)
    )


# ----------------------------------------------------------------- sequence


def walk(reason: str, stages: tuple[Stage, ...], alarm: Alarm | None) -> None:
    """The accepted order of a final work's parts and its reason (module docstring)."""
    for position, (part, succeeded, failure, logs, call) in enumerate(stages):
        if part is not None and succeeded:
            continue
        later = stages[position + 1 :]
        require(alarm is None)
        require(all(p is None and c is None for p, _, _, _, c in later))
        if part is not None:
            require(reason == failure)
            return
        refused = call is not None and call.status == REFUSED and call.reason == reason
        require(reason and (refused if logs else call is None))
        return
    require(not reason or alarm is None)


def _absorption_stage(absorbed: Absorbed | None, call: AbsorptionCall | None) -> Stage:
    """The slope absorption: converged without a Schur failure, else its reason."""
    if absorbed is None:
        return (None, False, "", True, call)
    result = absorbed.result
    succeeded = result.converged and not absorbed.failure
    return (absorbed, succeeded, absorbed.failure or result.reason, True, call)


def logged_final(work: SolveWork | IndexWork, index: int) -> Logged:
    """The logged final calls of attempt ``index``: profiles, absorptions and RHS."""
    rhs: tuple[RHSCall, ...] = work.rhs if isinstance(work, IndexWork) else ()
    return (
        tuple(c for c in work.profiles if c.attempt == index and c.role == "final"),
        tuple(c for c in work.absorptions if c.attempt == index and c.role == "final"),
        tuple(c for c in rhs if c.attempt == index and c.role == "final"),
    )


# -------------------------------------------------------------------- calls


def _bound_profile(
    call: ProfileCall, point: PointProfile, options: AbsorbedOptions
) -> None:
    """The final profile call: its controls, and the status, work and maximum."""
    result = point.result
    require(call.scope == "all_fe" and call.block is None)
    limits = (options.profile_tol, options.profile_max_sweeps)
    require((call.tolerance, call.max_sweeps) == limits)
    require((result.tolerance, result.max_sweeps) == limits)
    found = (call.status, call.reason, call.work, call.scalar, call.maximum)
    given = (result.status, result.reason, result.work, point.scalar, result.maximum)
    require(found == given)


def _bound_absorption(
    call: AbsorptionCall | None, absorbed: Absorbed, options: AbsorbedOptions
) -> None:
    """The final absorption call: its controls, and the status and work returned."""
    if call is None:
        raise ValueError(INCOHERENT)
    result = absorbed.result
    final, flag = options.absorb_tol_final, options.absorb_acceleration
    require((call.tolerance, call.score) == (final, None))
    require((result.tolerance, result.acceleration) == (final, flag))
    found = (call.status, call.reason, call.sweeps, call.visits, call.operators)
    given = (result.status, result.reason, result.sweeps, result.visits)
    require(found == (*given, result.operators) and call.failure == absorbed.failure)
    passes = (call.diagnostic_passes, call.maximum)
    require(passes == (result.diagnostic_passes, result.maximum))
    maximum = result.maximum
    passed = maximum is not None and maximum <= result.tolerance and not result.reason
    require(passed or not result.converged)


def _bound_rhs(
    call: RHSCall | None, rhs: RHSResult, rows: FinalRows, candidate: IndexCandidate
) -> None:
    """The final RHS call: controls, status, work, bounds, forcing, ``sigma``, guard."""
    if call is None:
        raise ValueError(INCOHERENT)
    final, work = candidate.absorbed.absorb_tol_final, rhs.work
    require((call.tolerance, call.score, rhs.tolerance) == (final, None, final))
    require((rhs.max_sweeps, rhs.acceleration) == (call.max_sweeps, call.acceleration))
    require(0.0 <= rhs.forcing <= FORCING_CAP)
    found: tuple[object, ...] = (call.status, call.reason, call.sweeps, call.visits)
    given: tuple[object, ...] = (rhs.status, rhs.reason, work.sweeps, work.visits)
    require(found == given)
    found = (call.operators, call.diagnostic_passes, call.maximum, call.initial)
    given = (work.operators, work.diagnostic_passes, rhs.maximum, rhs.initial)
    require(found == given)
    found = (call.absolute, call.relative, call.final_accuracy, call.native)
    given = (rhs.absolute, rhs.relative, rhs.final_accuracy, rhs.native)
    require(found == given and call.guard == rhs.guard)
    require((call.forcing, call.sigma) == (rhs.forcing, rhs.sigma))
    gradient_tol = candidate.options.gradient_tol
    require(rhs.guard == RHSGuard(final, rows.total, gradient_tol))
    require(not rhs.converged or not rhs.reason)


# ---------------------------------------------------------------- profiled


def _profiled_types(final: FinalWork) -> None:
    """Each present part of a profiled final record has its accepted type."""
    require(final.point is None or type(final.point) is PointProfile)
    require(final.equations is None or type(final.equations) is PointEquations)
    require(final.absorbed is None or type(final.absorbed) is Absorbed)
    require(final.alarm is None or type(final.alarm) is Alarm)
    require(type(final.reason) is str)


def _profiled_point(
    problem: ProfiledProblem,
    candidate: ProfiledCandidate,
    attempt: ProfiledAttempt,
    point: PointProfile,
    final: FinalWork,
) -> None:
    """The final profile at the attempt's slopes from its state; its own equations."""
    equations, absorbed, selected = final.equations, final.absorbed, attempt.point
    if selected is None:
        raise ValueError(INCOHERENT)
    result = point.result
    require(identical(point.theta, selected.theta))
    require(identical(result.start.coefficients, selected.state.coefficients))
    _bound(point.state, _prepared(problem, point.base))
    if equations is not None:
        fresh = point_equations(problem, point.index)
        require(
            identical(
                (fresh.gradient, fresh.curvature, fresh.information),
                (equations.gradient, equations.curvature, equations.information),
            )
        )
        require(fresh.loss == equations.loss and point.converged)
        weights = (
            equations.curvature
            if candidate.options.curvature == "observed"
            else equations.information
        )
        require(absorbed is None or identical(absorbed.weights, weights))


def profiled_final(
    problem: ProfiledProblem,
    candidate: ProfiledCandidate,
    index: int,
    attempt: ProfiledAttempt,
) -> FinalWork:
    """An attempt's profiled final record, owned and bound (module docstring).

    The record is bound to its OWN final point, whose ``q`` may differ from the
    attempt's; ``ValueError`` (:data:`INCOHERENT`) when it is not coherent.
    """
    if type(attempt.final) is not FinalWork or attempt.point is None:
        raise ValueError(INCOHERENT)
    _profiled_types(attempt.final)
    final = owned_profiled(attempt.final)
    profiles, absorptions, rhs = logged_final(candidate.work, index)
    require(len(profiles) == 1 and len(absorptions) <= 1 and not rhs)
    absorption = absorptions[0] if absorptions else None
    point, absorbed = final.point, final.absorbed
    succeeded = point is not None and point.converged
    failure = "" if point is None else point.result.reason
    stages: list[Stage] = [(point, succeeded, failure, True, profiles[0])]
    stages.append((final.equations, True, "", False, None))
    if problem.n_slopes:
        stages.append(_absorption_stage(absorbed, absorption))
    else:
        require(absorbed is None and absorption is None)
    walk(final.reason, tuple(stages), final.alarm)
    if point is not None:
        _bound_profile(profiles[0], point, candidate.absorbed)
        _profiled_point(problem, candidate, attempt, point, final)
    if absorbed is not None:
        _bound_absorption(absorption, absorbed, candidate.absorbed)
    return final


# -------------------------------------------------------------------- index


def _index_types(final: IndexFinal) -> None:
    """Each present part of an index final record has its accepted type."""
    point = final.point
    require(type(point) is IndexPoint and type(point.recovered) is Recovery)
    require(final.rows is None or type(final.rows) is FinalRows)
    require(final.equations is None or type(final.equations) is EquationCheck)
    require(final.absorbed is None or type(final.absorbed) is Absorbed)
    require(final.rhs is None or type(final.rhs) is RHSResult)
    require(final.alarm is None or type(final.alarm) is Alarm)
    require(type(final.reason) is str)


def _index_point(
    problem: ProfiledProblem,
    candidate: IndexCandidate,
    attempt: IndexAttempt,
    final: IndexFinal,
) -> None:
    """The final point is the attempt's; its rows, equations and weights fresh."""
    point, mine = final.point, attempt.point
    if mine is None:
        raise ValueError(INCOHERENT)
    require(
        identical(
            (point.theta, point.base, point.index), (mine.theta, mine.base, mine.index)
        )
    )
    rows, equations, absorbed = final.rows, final.equations, final.absorbed
    if rows is not None:
        fresh = final_rows(problem, point.index, candidate.options.curvature)
        require(
            identical(
                (fresh.gradient, fresh.curvature, fresh.right),
                (rows.gradient, rows.curvature, rows.right),
            )
        )
        require((fresh.loss, fresh.total) == (rows.loss, rows.total))
        require(absorbed is None or identical(absorbed.weights, rows.curvature))
    if equations is not None:
        check = fresh_equations(_prepared(problem, point.base), point.index)
        require(
            identical(check.errors, equations.errors)
            and check.maximum == equations.maximum
        )


def index_final(
    problem: ProfiledProblem,
    candidate: IndexCandidate,
    index: int,
    attempt: IndexAttempt,
) -> IndexFinal:
    """An attempt's index final record, owned and bound (module docstring).

    The record keeps the attempt's ``theta``, base and ``q``; ``ValueError``
    (:data:`INCOHERENT`) when it is not coherent.
    """
    if type(attempt.final) is not IndexFinal or attempt.point is None:
        raise ValueError(INCOHERENT)
    _index_types(attempt.final)
    final = owned_index(attempt.final)
    profiles, absorptions, rhs_calls = logged_final(candidate.work, index)
    require(not profiles and len(absorptions) <= 1 and len(rhs_calls) <= 1)
    absorption = absorptions[0] if absorptions else None
    rhs_call = rhs_calls[0] if rhs_calls else None
    recovery, rows = final.point.recovered, final.rows
    absorbed, rhs = final.absorbed, final.rhs
    complete = recovery is not None and recovery.complete
    failure = "" if recovery is None else recovery.reason
    stages: list[Stage] = [(recovery, complete, failure, False, None)]
    stages += [(rows, True, "", False, None), (final.equations, True, "", False, None)]
    if problem.n_slopes:
        stages.append(_absorption_stage(absorbed, absorption))
    else:
        require(absorbed is None and absorption is None)
    solved = rhs is not None and rhs.converged
    why = "" if rhs is None else f"full right-hand side: {rhs.reason}"
    stages.append((rhs, solved, why, True, rhs_call))
    walk(final.reason, tuple(stages), final.alarm)
    _index_point(problem, candidate, attempt, final)
    if absorbed is not None:
        _bound_absorption(absorption, absorbed, candidate.absorbed)
    if rhs is not None and rows is not None:  # the walk placed the rows first
        _bound_rhs(rhs_call, rhs, rows, candidate)
    return final
