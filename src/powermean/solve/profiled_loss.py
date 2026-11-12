"""The whole-point state of a profiled Newton attempt: trials, schedule, alarms, work.

During iteration the solver stops when fitted means at zero rows collapse or the
criterion decreases without bound, and nothing is dropped on the basis of an iterate.
Internal module.

:class:`ProfiledEvaluator` is the state of one attempt of the profiled Newton solver,
the :class:`~powermean.solve.newton_convex.ProfiledEvaluation` that the profiled linear
loss of :mod:`powermean.solve.profiled_linear_loss` and the iteration hooks of
:mod:`powermean.solve.newton_convex` call: the start, the trials, the accepted point,
the absorption schedule and the indications.
:func:`final_work` is the fresh final work after a native stop.  The points are those
of :mod:`powermean.systems.profiled` (the accepted profiles and absorption, unchanged).

Points and transactions.  Every trial profiles its own slopes from
the warm state, the effects of the most recent successfully evaluated point: the
accepted one, or a completed trial of the current line search.  A trial
whose profile refuses by range, or whose criterion or
gradient is not representable, is refused by the existing search and leaves the warm
state as it was.  A completed trial is held, never accepted by itself: when the search
accepts a trial, that trial's whole bundle (slopes, base, effects, index, profile, loss
and gradient) becomes the accepted bundle, and a refused or interrupted search restores
the accepted bundle's effects as the warm state.  The derivatives are evaluated at the
accepted point only.  A profile or an absorption that spends its sweeps raises
:class:`~powermean.solve.newton_convex.InnerLimitError`: the attempt stops,
and it is not a refused trial with a new allowance.

Schedule: :func:`next_tolerance`.  The first direction of an
attempt absorbs at ``a0 = absorb_tol_start``; after an accepted step, with ``s`` the
largest absolute averaged gradient of that accepted point, the
next direction uses ``max(b, a / 10)`` when ``s < a`` and ``a > b``, with
``b = min(a0, af)``, and ``a`` otherwise; the tolerance is constant within an iteration
and its trials; the final absorption uses exactly ``af``.  Every profile uses
``profile_tol`` and ``profile_max_sweeps``, which nothing relaxes.  The first
direction of an attempt and the final absorption start cold; a later direction starts
from the warm slope state only with strictly positive current curvature on every
retained row: the slope coefficients of the last accepted origin's
successful direction (:func:`~powermean.systems.profiled._slope_state`, bound to the
grouping and the ordered slope columns), which the absorber rebuilds and checks afresh
with the current weights at the current tolerance, so no success survives a change of
tolerance.  A direction whose absorption and Schur matrix succeed leaves its state
pending for its origin; only an accepted search promotes it, and a failed direction or
a refused or interrupted search promotes nothing.  Any zero uses the original cold
calculation and creates no slope state; accepting it clears the old seed.  The slope
state is apart from the nonlinear warm state of the profiles and is never taken from
a held or trial profile.

Indications: ZC and GC of
:mod:`powermean.solve.profiled_indications` (whose names this module still provides),
evaluated at the completed initial profile, at each newly accepted point after the
native convergence test (before the iteration could honor it), and at the fresh final
point; never at a held or refused trial.  Indications at one point coalesce into one
:class:`~powermean.solve.profiled_indications.Alarm`; the attempt stops with
:class:`~powermean.solve.newton_convex.IndicationError`.  Neither the presence nor the
absence of an alarm is an existence certificate.

Work.  Every profile and absorption call is logged (:class:`WorkLog`) with its attempt,
the accepted steps before it, its role, controls, status and the work it returned,
rejected, refused and final calls included; a call refused before the accepted routine
counted anything has no work record, not zero work.  A completed absorption whose Schur
matrix then proves not representable keeps its status and work, with that later range
failure recorded apart; the direction still refuses, and the final work
keeps the completed absorption.  Nothing here is charged to the pointwise-evaluation
budget of the iteration: the profiled linear loss admits each evaluation before the work
of this module starts (:mod:`powermean.solve.profiled_linear_loss`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from powermean.fe.absorb import AbsorptionState
from powermean.fe.profile import BUDGET, ProfileState, Work
from powermean.fe.profile_nb2 import ScalarWork
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_convex import IndicationError, InnerLimitError
from powermean.solve.options import Curvature
from powermean.solve.profiled_indications import (
    INDEX_MARGIN,
    LOG_MEAN_TOLERANCE,
    Alarm,
    ChordAlarm,
    Where,
    ZeroRowAlarm,
    chord_alarm,
    zero_row_alarm,
)
from powermean.systems.profiled import (
    Absorbed,
    PointEquations,
    PointProfile,
    ProfiledProblem,
    _absorb_slopes,
    _positive_slope_curvature,
    _slope_state,
    absorb_slopes,
    point_equations,
    profile_point,
)

if TYPE_CHECKING:
    from powermean.solve.reduced_nonlinear import ConditionalPoint

_REDUCED_PROFILE = profile_point

_REDUCED_ABSORB = absorb_slopes
_REDUCED_PRIVATE_ABSORB = _absorb_slopes

# The names of this module, and those of the indications defined in
# solve/profiled_indications.py, whose import routes from here are kept.
__all__ = [
    "INDEX_MARGIN",
    "LOG_MEAN_TOLERANCE",
    "REFUSED",
    "TIGHTENING",
    "AbsorptionCall",
    "AbsorptionRole",
    "Alarm",
    "Bundle",
    "ChordAlarm",
    "FinalWork",
    "ProfileCall",
    "ProfileRole",
    "ProfiledEvaluator",
    "SolveWork",
    "Where",
    "WorkLog",
    "ZeroRowAlarm",
    "chord_alarm",
    "final_work",
    "next_tolerance",
    "zero_row_alarm",
]

# ppmlhdfe 2.3.4, ppmlhdfe.mata, line 823: the partial-out tolerance tightens tenfold.
TIGHTENING = 10.0

ProfileRole = Literal["initial", "trial", "final"]
AbsorptionRole = Literal["direction", "final"]
REFUSED = "refused"


# ------------------------------------------------------------------ the schedule


def next_tolerance(
    current: float, score: float, *, start: float, final: float
) -> float:
    """The absorption tolerance of the next direction.

    ``max(min(start, final), current / 10)`` when ``score < current`` and ``current``
    exceeds ``min(start, final)``; otherwise ``current``.  Equality does not tighten; a
    zero score tightens; the rounded division by ten may underflow, and the positive
    endpoint then wins (an endpoint, not a floor of a model quantity).
    ``FloatingPointError`` for a score that is not finite: an evaluation failure, not
    progress.
    """
    if not math.isfinite(score):
        raise FloatingPointError(
            "the accepted gradient is not representable in float64"
        )
    target = min(start, final)
    if score < current and current > target:
        return max(target, current / TIGHTENING)
    return current


# ----------------------------------------------------------------------- the work


@dataclass(frozen=True)
class ProfileCall:
    """One profile call: where, its controls, status, reason and the work returned.

    ``work`` and ``scalar`` are ``None`` when the call was refused before the accepted
    routine counted anything (``status`` ``refused``): unavailable, not zero.
    ``scope='conditional'`` names one original block, not a completed all-FE
    profile. The retained-nonlinear backend uses it for its first cold opening
    and trials; warm retry openings and final profiles remain all-FE.
    """

    attempt: int
    iteration: int
    role: ProfileRole
    tolerance: float
    max_sweeps: int
    status: str
    reason: str
    work: Work | None
    scalar: ScalarWork | None
    maximum: float | None
    scope: Literal["all_fe", "conditional"] = "all_fe"
    block: int | None = None


@dataclass(frozen=True)
class AbsorptionCall:
    """One absorption call: where, its controls, the score that set them, its work.

    ``score`` is the accepted gradient maximum that set ``tolerance`` (``None`` for the
    first direction of an attempt and for the final call); the four counters are those
    of :class:`~powermean.fe.absorb.AbsorptionResult`, ``None`` when the call was
    refused before absorbing (``status`` ``refused``).  ``failure`` names a range
    failure that followed a returned absorption (its Schur matrix not representable);
    the status, reason and counters are then the completed absorption's own, and the
    call is no success.  It is empty otherwise.
    """

    attempt: int
    iteration: int
    role: AbsorptionRole
    tolerance: float
    score: float | None
    max_sweeps: int
    acceleration: bool
    status: str
    reason: str
    sweeps: int | None
    visits: int | None
    operators: int | None
    diagnostic_passes: int | None
    maximum: float | None
    failure: str = ""


@dataclass(frozen=True, eq=False)
class SolveWork:
    """Every profile and absorption call of one solve, in order."""

    profiles: tuple[ProfileCall, ...]
    absorptions: tuple[AbsorptionCall, ...]


class WorkLog:
    """The calls of one solve as they happen, frozen into :class:`SolveWork`."""

    def __init__(self) -> None:
        self.profiles: list[ProfileCall] = []
        self.absorptions: list[AbsorptionCall] = []

    def frozen(self) -> SolveWork:
        """The calls so far, as an immutable record."""
        return SolveWork(tuple(self.profiles), tuple(self.absorptions))


# --------------------------------------------------------------------- the points


@dataclass(frozen=True, eq=False)
class Bundle:
    """One whole accepted or held point: its profile, its loss and its gradient.

    ``gradient`` is ``None`` at the start of an attempt, whose first gradient the
    derivatives evaluate.
    """

    point: PointProfile | ConditionalPoint
    loss: float
    gradient: FloatArray | None

    def __post_init__(self) -> None:
        """Own the gradient."""
        if self.gradient is not None:
            object.__setattr__(self, "gradient", immutable_array(self.gradient))

    @property
    def theta(self) -> FloatArray:
        """The slopes."""
        return self.point.theta

    @property
    def index(self) -> FloatArray:
        """The profiled index of every row."""
        return self.point.index

    @property
    def state(self) -> ProfileState:
        """The block coefficients."""
        return self.point.state


class ProfiledEvaluator:
    """The whole-point state of one profiled Newton attempt (module docstring).

    Parameters
    ----------
    problem : ProfiledProblem
        The profiled problem of the sample.
    controls : AbsorbedOptions
        The profile and absorption controls, read as they are.
    curvature : {'observed', 'expected'}
        The curvature of the direction (``a c`` or ``a w`` in the absorption).
    log : WorkLog
        The calls of the solve, to which this attempt appends.
    attempt : int
        The index of the attempt in the solve.
    warm : ProfileState or None
        The effects that the first profile starts from; cold if ``None``.
    positions : array of int
        The original positions of the rows, for the records of an indication.
    """

    def __init__(
        self,
        problem: ProfiledProblem,
        controls: AbsorbedOptions,
        curvature: Curvature,
        log: WorkLog,
        *,
        attempt: int,
        warm: ProfileState | None,
        positions: IndexArray,
    ) -> None:
        self.problem = problem
        self.controls = controls
        self.curvature = curvature
        self.log = log
        self.attempt = attempt
        self.warm = warm
        self.positions = positions
        self.gpml = (
            type(problem.variance) is PowerVariance and problem.variance.tau == 2
        )
        self.tolerance = controls.absorb_tol_start
        self.score: float | None = None
        self.steps = 0
        self.opening: PointProfile | ConditionalPoint | None = None
        self.pending: PointProfile | None = None
        self.accepted: Bundle | None = None
        self.previous: Bundle | None = None
        self.held: list[Bundle] = []
        self.alarm: Alarm | None = None
        self.slope_seed: AbsorptionState | None = None
        self.slope_pending: tuple[Bundle | None, AbsorptionState] | None = None
        self.slope_cold_origin: Bundle | None = None

    # ------------------------------------------------------------- profiles

    def _profiled(
        self, theta: FloatArray, role: ProfileRole, state: ProfileState | None
    ) -> PointProfile:
        """One logged profile at ``theta`` from ``state``, whatever its status.

        A call refused before the routine counted anything is logged as refused and
        its ``FloatingPointError`` raised again.
        """
        controls = self.controls
        found = (self.attempt, self.steps, role)
        limits = (controls.profile_tol, controls.profile_max_sweeps)
        try:
            point = profile_point(
                self.problem,
                theta,
                tolerance=limits[0],
                max_sweeps=limits[1],
                state=state,
            )
        except FloatingPointError as error:
            refused = ProfileCall(
                *found, *limits, REFUSED, str(error), None, None, None
            )
            self.log.profiles.append(refused)
            raise
        result = point.result
        call = ProfileCall(
            *found,
            *limits,
            result.status,
            result.reason,
            result.work,
            point.scalar,
            result.maximum,
        )
        self.log.profiles.append(call)
        return point

    def _completed(self, theta: FloatArray, role: ProfileRole) -> PointProfile:
        """A converged profile from the warm state, or the attempt's failure.

        ``InnerLimitError`` when the sweeps are spent; ``FloatingPointError`` when it
        refuses by range (a trial is then refused by the search).
        """
        point = self._profiled(theta, role, self.warm)
        if point.result.status == BUDGET:
            raise InnerLimitError(f"profile: {point.result.reason}")
        if not point.converged:
            raise FloatingPointError(f"profile: {point.result.reason}")
        return point

    def start(self, coef: FloatArray) -> FloatArray:
        """The profiled index of the start of the attempt (its first profile)."""
        point = self._completed(coef, "initial")
        self.opening = point
        self.warm = point.state
        return point.index

    def trial(self, coef: FloatArray) -> FloatArray:
        """The profiled index of a trial of the line search, profiled afresh."""
        self.pending = None
        point = self._completed(coef, "trial")
        self.pending = point
        return point.index

    def evaluated(self, coef: FloatArray, loss: float, gradient: FloatArray) -> None:
        """The trial ``coef`` is evaluated: it is held and warms the next profile."""
        point = self.pending
        if point is None or not np.array_equal(point.theta, coef):
            raise ValueError("only the trial just profiled can be evaluated")
        self.held.append(Bundle(point, loss, gradient))
        self.warm = point.state
        self.pending = None

    def current(self, coef: FloatArray) -> FloatArray:
        """The profiled index of the accepted point, which ``coef`` must be."""
        accepted = self.accepted
        if accepted is None or not np.array_equal(accepted.theta, coef):
            raise ValueError("the derivatives are evaluated at the accepted point only")
        return accepted.index

    # ----------------------------------------------------------- directions

    def _absorbed(
        self, weights: FloatArray, role: AbsorptionRole, tolerance: float,
        *, index: FloatArray | None = None,
    ) -> Absorbed:
        """One logged absorption of the slopes with ``weights`` at ``tolerance``.

        A call refused before absorbing is logged as refused, without work, and its
        ``FloatingPointError`` raised again.  A returned absorption is logged with its
        own status and work, and with the range failure of its Schur matrix when that
        matrix is not representable; the caller refuses then.  A direction starts from
        the attempt's warm slope state (:attr:`slope_seed`) through the private
        :func:`~powermean.systems.profiled._absorb_slopes`, which checks eligibility
        after the original current-weight validation and drops an ineligible seed
        before absorption.  Without a seed it makes the cold call as before;
        the final absorption is cold and first clears the slope
        state of the attempt.
        """
        controls = self.controls
        score = self.score if role == "direction" else None
        found = (self.attempt, self.steps, role, tolerance, score)
        limits = (controls.absorb_max_sweeps, controls.absorb_acceleration)
        seed = self.slope_seed if role == "direction" else None
        if role == "final":
            self.slope_seed, self.slope_pending = None, None
            self.slope_cold_origin = None
        try:
            reduced = self.problem.__dict__.get("_reduced_execution")
            direct = None
            if (reduced is not None and index is not None
                    and absorb_slopes is _REDUCED_ABSORB
                    and _absorb_slopes is _REDUCED_PRIVATE_ABSORB):
                direct = reduced.project(
                    weights, index, tolerance=tolerance, acceleration=limits[1],
                    final_observed=role == "final" and self.curvature == "observed",
                )
            if direct is not None:
                absorbed, _ = direct
            elif seed is None:
                absorbed = absorb_slopes(
                    self.problem,
                    weights,
                    tolerance=tolerance,
                    max_sweeps=limits[0],
                    acceleration=limits[1],
                )
            else:
                absorbed, _ = _absorb_slopes(
                    self.problem,
                    weights,
                    tolerance=tolerance,
                    max_sweeps=limits[0],
                    acceleration=limits[1],
                    state=seed,
                )
        except FloatingPointError as error:
            call = AbsorptionCall(
                *found, *limits, REFUSED, str(error), None, None, None, None, None
            )
            self.log.absorptions.append(call)
            raise
        result = absorbed.result
        counts = (result.sweeps, result.visits, result.operators)
        call = AbsorptionCall(
            *found,
            *limits,
            result.status,
            result.reason,
            *counts,
            result.diagnostic_passes,
            result.maximum,
            absorbed.failure,
        )
        self.log.absorptions.append(call)
        return absorbed

    def direction(self, weights: FloatArray) -> FloatArray:
        """The slopes absorbed with the curvature ``weights`` at the current tolerance.

        From this origin's warm slope state only with strictly positive fresh current
        curvature on every row; any zero uses the original cold calculation, retaining
        every row.  The attempt's first direction is cold;
        a direction whose absorption and Schur matrix succeed leaves its own state
        pending for :meth:`end`, tagged with the accepted origin, and any direction
        first discards the pending state of an earlier one.
        ``InnerLimitError`` when the sweeps are spent; ``FloatingPointError`` for a
        range failure (a Schur matrix that is not representable included, after its
        completed absorption is logged), a breakdown or a level without positive
        curvature.
        """
        self.slope_pending = None
        self.slope_cold_origin = None
        bound = (
            {} if self.problem.__dict__.get("_reduced_execution") is None
            else {"index": self._accepted().index}
        )
        absorbed = self._absorbed(weights, "direction", self.tolerance, **bound)
        result = absorbed.result
        if result.status == BUDGET:
            raise InnerLimitError(f"absorption: {result.reason}")
        if not result.converged:
            raise FloatingPointError(f"absorption: {result.reason}")
        if absorbed.failure:
            raise FloatingPointError(absorbed.failure)
        if _positive_slope_curvature(absorbed.weights):
            self.slope_pending = (self.accepted, _slope_state(result))
        else:
            self.slope_cold_origin = self.accepted
        return result.residual

    # --------------------------------------------------------- transactions

    def begin(self) -> None:
        """A line search starts: no trial is held."""
        self.held = []
        self.pending = None

    def end(self, coef: FloatArray | None) -> None:
        """The search accepted the trial ``coef`` (its bundle commits), or none.

        Without an accepted trial the accepted bundle's effects warm the next profile
        again: the rollback of a refused or interrupted search, which also discards the
        pending slope state and keeps the slope seed.  When the trial's bundle commits,
        the pending slope state of the search's origin becomes the slope seed of the
        next origin.  A successful ineligible direction records only its origin;
        accepting its trial clears the old seed, without copying a state.
        """
        held, self.held, self.pending = self.held, [], None
        slopes, self.slope_pending = self.slope_pending, None
        cold_origin, self.slope_cold_origin = self.slope_cold_origin, None
        accepted = self._accepted()
        if coef is None:
            self.warm = accepted.state
            return
        found = [bundle for bundle in held if np.array_equal(bundle.theta, coef)]
        if not found:
            raise ValueError("the accepted trial was not evaluated in this search")
        self.previous, self.accepted = accepted, found[-1]
        self.warm = found[-1].state
        if slopes is not None and slopes[0] is accepted:
            self.slope_seed = slopes[1]
        elif cold_origin is accepted:
            self.slope_seed = None

    def _accepted(self) -> Bundle:
        """The accepted bundle; ``ValueError`` before the start is evaluated."""
        if self.accepted is None:
            raise ValueError("the start of the attempt is not evaluated")
        return self.accepted

    # ---------------------------------------------------------- the hooks

    def _opened(self) -> PointProfile | ConditionalPoint:
        """The actual opening point; ``ValueError`` before it is constructed."""
        if self.opening is None:
            raise ValueError("the start of the attempt is not profiled")
        return self.opening

    def started(self, loss: float) -> None:
        """The start is evaluated with ``loss``: it is accepted; ZC is evaluated."""
        opening = self._opened()
        self.accepted = Bundle(opening, loss, None)
        _stop(self._alarm("initial", opening, chord=None, converged=False))

    def inspect(self) -> PointProfile | ConditionalPoint:
        """ZC at the profiled start of a solve without slopes; the start if no alarm."""
        opening = self._opened()
        _stop(self._alarm("initial", opening, chord=None, converged=False))
        return opening

    def stepped(self, converged: bool, gradient: FloatArray) -> None:
        """A step was accepted and tested natively: indications, then the schedule.

        ``gradient`` is the engine's averaged gradient at the accepted point.
        """
        self.steps += 1
        accepted = self._accepted()
        chord = None
        if self.gpml and self.previous is not None:
            chord = chord_alarm(
                self.problem.z,
                self.problem.weights,
                (self.previous.loss, self.previous.index),
                (accepted.loss, accepted.index),
            )
        _stop(self._alarm("accepted", accepted.point, chord=chord, converged=converged))
        score = float(np.max(np.abs(gradient)))
        self.tolerance = next_tolerance(
            self.tolerance,
            score,
            start=self.controls.absorb_tol_start,
            final=self.controls.absorb_tol_final,
        )
        self.score = score

    def _alarm(
        self,
        where: Where,
        point: PointProfile | ConditionalPoint,
        *,
        chord: ChordAlarm | None,
        converged: bool,
    ) -> Alarm | None:
        """ZC at ``point`` and ``chord`` coalesced into one kept alarm, or ``None``."""
        zero = zero_row_alarm(self.problem.z, point.index, self.positions)
        if zero is None and chord is None:
            return None
        self.alarm = Alarm(where, self.attempt, self.steps, converged, zero, chord)
        return self.alarm


def _stop(alarm: Alarm | None) -> None:
    """``IndicationError`` naming the coalesced indications, if there is an alarm."""
    if alarm is None:
        return
    kinds = [
        k for k, v in (("ZC", alarm.zero_rows), ("GC", alarm.chord)) if v is not None
    ]
    raise IndicationError(
        f"existence indication {'+'.join(kinds)} at the {alarm.where} point"
    )


# ---------------------------------------------------------------- final work


@dataclass(frozen=True, eq=False)
class FinalWork:
    """The fresh final work of a native stop, or of a profile without slopes.

    ``point`` is the fresh final profile; ``equations`` the rows at its index;
    ``absorbed`` the fresh absorption at ``absorb_tol_final`` (``None`` without a slope
    column, or when not reached), kept with its work when it completed and its Schur
    matrix then proved not representable; ``alarm`` an indication at the final point;
    ``reason`` is empty when every part completed, else what failed.
    """

    point: PointProfile | None
    equations: PointEquations | None
    absorbed: Absorbed | None
    alarm: Alarm | None
    reason: str

    @property
    def complete(self) -> bool:
        """Whether the fresh profile, rows and absorption all completed."""
        return not self.reason


def final_work(
    evaluator: ProfiledEvaluator, start: PointProfile | ConditionalPoint,
    loss: float | None,
) -> FinalWork:
    """Fresh final profile, rows and absorption from the point ``start``.

    A new profile at ``profile_tol`` from ``start``'s effects at its slopes; the rows at
    its index; a cold absorption at exactly ``absorb_tol_final`` with the curvature of
    the solve, when there are slopes; then ZC at the final index and, for GPML with the
    accepted ``loss``, GC against ``start``.  A failed part ends the work with its
    reason (operative ``final_failure``), the completed absorption kept when only its
    Schur matrix failed; a cached success is never reused.
    """
    point: PointProfile | None = None
    equations: PointEquations | None = None
    absorbed: Absorbed | None = None
    problem, controls = evaluator.problem, evaluator.controls
    try:
        point = evaluator._profiled(start.theta, "final", start.state)
        if not point.converged:
            return FinalWork(point, None, None, None, point.result.reason)
        equations = point_equations(problem, point.index)
        if problem.n_slopes:
            observed = evaluator.curvature == "observed"
            weights = equations.curvature if observed else equations.information
            bound = (
                {} if problem.__dict__.get("_reduced_execution") is None
                else {"index": point.index}
            )
            absorbed = evaluator._absorbed(
                weights, "final", controls.absorb_tol_final, **bound
            )
            if not absorbed.result.converged:
                reason = absorbed.result.reason
                return FinalWork(point, equations, absorbed, None, reason)
            if absorbed.failure:
                return FinalWork(point, equations, absorbed, None, absorbed.failure)
        chord = None
        if evaluator.gpml and loss is not None:
            chord = chord_alarm(
                problem.z,
                problem.weights,
                (loss, start.index),
                (equations.loss, point.index),
            )
        alarm = evaluator._alarm("final", point, chord=chord, converged=True)
        return FinalWork(point, equations, absorbed, alarm, "")
    except FloatingPointError as error:
        return FinalWork(point, equations, absorbed, None, str(error))
