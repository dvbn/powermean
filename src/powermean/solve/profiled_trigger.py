"""The trigger boundary of the profiled Newton solver: request, trace, admission.

When an indication stops the solver during iteration, the existence analysis is rerun
freshly with tighter tolerances; nothing is dropped on the basis of an iterate.
Internal module, separate from :mod:`powermean.solve.newton_profiled`.  Nothing of the
interface layer is imported here: the caller's dependencies arrive as the typed
protocols below.

The request.  After an alarm the solver calls the caller's :class:`Trigger` once with a
:class:`TriggerRequest`: the stopped whole point (:class:`FullPoint`), the alarm and the
exact lower binding (the sample, the variance and the existing TIGHTER controls of the
family, of the support analysis for ``tau < 2`` and NB2 and of the recession analysis
for ``tau = 2``).  The caller owns the
trigger, bound to the actual completed outcome of the sample stage; it runs the normal
coordinator once and returns its frozen record through the structural views below
(:class:`TraceView`), a propagated failure included.

The reading (:func:`triggered`).  The trace must start from this very sample, its
origin's sample must be this sample, and its request must be this
trigger's; its first fresh diagnostic must be the family's, and the solver admits that
diagnostic's admitted result again on its own problem, formed from the sample (every
original unscaled candidate column, every block with its level labels, the typed
weights, the transformed outcome and offset, the original positions), which replays
every witness.  The coordinator's outcome is admitted by the caller's
:class:`Admission`, the coordinator's own admission of the call's origin, derived
request and recorded walk: a status, a structural stand-in or a foreign
or refused outcome is not an admitted outcome and is never kept as one.  The
classification (:data:`TriggerStatus`) reads that evidence: support without separated
rows is a false alarm; witnessed separation is ``separation_recomposed`` when the
admitted outcome is complete, else ``separation_unresolved``; a GPML recession is
``existence_failure``, never a deletion; anything else is ``inconclusive``; a trace
that carries a propagated failure is ``failed``, its admitted evidence kept; a trace
that fails a binding or an admission check is ``refused``.  The solver never resumes,
re-solves, certifies, adopts a new sample or deletes a row: its stopped point and its
sample are unchanged.

The point view.  A whole point
is a :class:`WholePoint` of a :class:`PointView`, the common honest data of either
absorbed organization: the slopes, the base and the full index of every row.  The
profiled organization's whole point keeps its name and meaning, :data:`FullPoint`, a
whole point of a :class:`~powermean.systems.profiled.PointProfile`; the fitted-index
organization passes its own point, with its optional recovered effects, and no profile
record is formed for it.  The request, the reading of a trace, its admission and its
classification read the sample, the variance and the controls only, unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, Protocol, cast

import numpy as np

from powermean.data.collinearity import CollinearityProblem, DiagnosticControls
from powermean.data.sample_fe_source import same_blocks
from powermean.data.sample_fe_state import FESample
from powermean.data.weights import ObservationWeights
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve.newton_trace import Timing
from powermean.solve.profiled_indications import Alarm
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.solve.support import (
    SupportControls,
    SupportResult,
    require_current,
    select_method,
)
from powermean.solve.support_gamma import GammaProblem, GammaResult
from powermean.solve.support_lp import LPControls
from powermean.solve.support_problem import NONE, STALE, SupportProblem
from powermean.systems.profiled import PointProfile

Family = PowerVariance | NB2Variance
TriggerStatus = Literal[
    "false_alarm",
    "separation_recomposed",
    "separation_unresolved",
    "existence_failure",
    "inconclusive",
    "failed",
    "refused",
]
RECESSIONS = ("negative_sum_recession", "zero_sum_recession")


class PointView(Protocol):
    """The common data of a whole point of either absorbed organization."""

    @property
    def theta(self) -> FloatArray:
        """The slopes, in the coordinates of the sample's scaled design."""
        ...

    @property
    def base(self) -> FloatArray:
        """``X theta + o``, the transformed offset entered once."""
        ...

    @property
    def index(self) -> FloatArray:
        """The full index of every row."""
        ...


@dataclass(frozen=True, eq=False)
class WholePoint[P: PointView]:
    """One whole point of a solve: its sample, variance and point.

    ``point`` holds at least the slopes, the base and the full index of every row
    (:class:`PointView`): a profiled point, with its block coefficients and profiled
    index, or a fitted-index point, with its optional recovered effects.  The trigger
    receives the stopped one; the caller's assessor of the retry reads any one.
    """

    sample: FESample
    variance: Family
    point: P


# Requested profiled-route whole points; conditional trials have their own type
# and do not claim completion of an all-FE profile.
FullPoint = WholePoint[PointProfile | ConditionalPoint]


@dataclass(frozen=True, eq=False)
class TriggerRequest:
    """What the trigger receives: the stopped point and its exact lower binding.

    ``support`` and ``recession`` are the TIGHTER controls of the family, ``None`` for
    the other family, as the coordinator's request takes them.  ``point`` is the
    stopped whole point of either organization.
    """

    sample: FESample
    variance: Family
    support: SupportControls | None
    recession: LPControls | None
    alarm: Alarm
    point: WholePoint[PointView]


class EventView(Protocol):
    """One recorded event of the coordinator, as the solver reads it."""

    @property
    def kind(self) -> str:
        """``entry``, ``closure``, ``raw``, ``admitted`` or ``final``."""
        ...

    @property
    def stage(self) -> str:
        """The stage of the walk the event belongs to."""
        ...

    @property
    def problem(self) -> SupportProblem | GammaProblem | CollinearityProblem | None:
        """The problem of a ``raw`` or ``admitted`` event."""
        ...

    @property
    def controls(self) -> SupportControls | LPControls | DiagnosticControls | None:
        """The controls of a ``raw`` or ``admitted`` event."""
        ...

    @property
    def result(self) -> object:
        """The analysis returned (``raw``) or kept by the admission (``admitted``).

        Typed ``object``: a ``raw`` result is the caller's own record of
        what returned, which the solver never reads; an ``admitted`` one is read only
        as the family's analysis, admitted again.
        """
        ...


class OutcomeView(Protocol):
    """An outcome of the coordinator, as the solver reads it."""

    @property
    def status(self) -> str:
        """The status of the composition."""
        ...

    @property
    def sample(self) -> FESample | None:
        """The completed sample, when the status is complete."""
        ...


class OriginView(Protocol):
    """The completed outcome a trigger call started from, as the solver reads it."""

    @property
    def sample(self) -> FESample | None:
        """The origin's completed sample."""
        ...


class RequestView(Protocol):
    """The request the coordinator derived for the call, as the solver reads it."""

    @property
    def omega(self) -> float:
        """The mean order."""
        ...

    @property
    def variance(self) -> Family | None:
        """The working variance."""
        ...

    @property
    def support(self) -> SupportControls | None:
        """The support-analysis controls of the call."""
        ...

    @property
    def recession(self) -> LPControls | None:
        """The recession-analysis controls of the call."""
        ...


class TraceView(Protocol):
    """The frozen record of one trigger call of the coordinator."""

    @property
    def origin(self) -> OriginView:
        """The completed outcome the call started from, bound when it entered."""
        ...

    @property
    def start(self) -> FESample:
        """The origin's sample, from which the call started."""
        ...

    @property
    def request(self) -> RequestView:
        """The request derived for the call."""
        ...

    @property
    def events(self) -> Sequence[EventView]:
        """Every recorded event, in order."""
        ...

    @property
    def outcome(self) -> OutcomeView | None:
        """The outcome of a completed call; ``None`` after a failure."""
        ...

    @property
    def unfinished(self) -> str | None:
        """The stage entered and not completed when a failure propagated."""
        ...

    @property
    def failure(self) -> Exception | None:
        """The exception that propagated out of the call, as it was raised."""
        ...


class Trigger(Protocol):
    """The caller's trigger: one fresh composition from the stopped point's sample."""

    def __call__(self, request: TriggerRequest, /) -> TraceView:
        """The frozen trace of the call, a propagated failure included."""
        ...


class Admission(Protocol):
    """The caller's admission of a trace's outcome: the coordinator's own (upper).

    It admits the call's origin, its derived request and the outcome's recorded walk
    as the coordinator does, without analyzing anything, and returns the trace's
    outcome; ``ValueError`` when that outcome is not admitted.
    """

    def __call__(self, trace: TraceView, /) -> OutcomeView:
        """The trace's own outcome, admitted again."""
        ...


@dataclass(frozen=True, eq=False)
class TriggerRecord:
    """The one trigger call of a solve and the solver's reading of its trace.

    ``status`` is the classification and ``reason`` its reason; ``diagnostic`` the
    first fresh diagnostic as the solver admitted it again on its own problem;
    ``removed`` the original positions it witnessed as separated; ``outcome`` the
    coordinator's outcome as the caller's admission admitted it, kept apart and never
    adopted (``None`` when the call holds none or it was not admitted); ``admission``
    why a held outcome was not admitted (empty otherwise); ``timing`` that of the call.
    """

    request: TriggerRequest
    trace: TraceView
    status: TriggerStatus
    reason: str
    diagnostic: SupportResult | GammaResult | None
    removed: IndexArray
    outcome: OutcomeView | None
    admission: str
    timing: Timing


def _first(trace: TraceView, kind: str, stages: tuple[str, ...]) -> EventView | None:
    """The first event of ``kind`` in one of ``stages``, or ``None``."""
    for event in trace.events:
        if event.kind == kind and event.stage in stages:
            return event
    return None


def _support_problem(request: TriggerRequest) -> SupportProblem:
    """The support problem of the solver's sample: its original equation, all blocks."""
    sample, data = request.sample, request.sample.data
    return SupportProblem(
        sample.problem.response,
        data.design,
        data.column_names,
        sample.grouping,
        sample.positions,
        data.weight_values,
        sample.problem.offset,
        request.variance,
        sample.omega,
    )


def _gamma_problem(request: TriggerRequest) -> GammaProblem:
    """The Gamma recession problem of the solver's sample, with its typed weights."""
    sample, data = request.sample, request.sample.data
    weights = data.weights
    if weights is None:
        weights = ObservationWeights(np.ones(data.nobs))
    # Only a GPML solve requests recession controls; GammaProblem refuses other
    # families.
    variance = cast("PowerVariance", request.variance)
    return GammaProblem(
        sample.problem.response,
        data.design,
        data.column_names,
        sample.grouping,
        sample.positions,
        weights,
        sample.problem.offset,
        variance,
        sample.omega,
    )


def _again(request: TriggerRequest, event: EventView) -> SupportResult | GammaResult:
    """The admitted first diagnostic, admitted again on the solver's own problem.

    ``ValueError`` (the accepted refusals) for a result of another sample, blocks,
    level labels, controls or method, or whose evidence fails its replay.
    """
    result = event.result
    if isinstance(result, SupportResult) and request.support is not None:
        problem = _support_problem(request)
        require_current(result, problem)
        if not same_blocks(result.problem.grouping, problem.grouping):
            raise ValueError(STALE)
        method = select_method(problem, request.support)
        if result.controls != request.support or result.method != method:
            raise ValueError("the diagnostic ran under other controls or method")
        return replace(result, problem=problem)
    if isinstance(result, GammaResult) and request.recession is not None:
        found = _gamma_problem(request)
        result.problem.require_same(found)
        if not same_blocks(result.problem.grouping, found.grouping):
            raise ValueError(STALE)
        if result.controls != request.recession:
            raise ValueError("the diagnostic ran under other controls")
        checked = replace(result, problem=found, controls=request.recession)
        witness = checked.witness
        if witness is not None:
            witness.require(found, controls=request.recession, stage=witness.stage)
        return checked
    raise ValueError("the first diagnostic is not the family's analysis")


def _bound(request: TriggerRequest, trace: TraceView) -> EventView | None:
    """The admitted event of the first diagnostic of a trace bound to this request.

    ``None`` when the first diagnostic was not admitted or none ran; ``ValueError``
    for a trace of another sample, origin or request, or whose first diagnostic is not
    the family's.
    """
    if trace.start is not request.sample:
        raise ValueError("the trace started from another sample than the solver's")
    if trace.origin.sample is not request.sample:
        raise ValueError("the trace's origin is not the outcome of the solver's sample")
    asked = trace.request
    found = (asked.omega, asked.variance, asked.support, asked.recession)
    if found != (
        request.sample.omega,
        request.variance,
        request.support,
        request.recession,
    ):
        raise ValueError("the trace's request is not this trigger's")
    stage = "support" if request.support is not None else "recession"
    raw = _first(trace, "raw", ("support", "recession"))
    if raw is None:
        return None
    if raw.stage != stage:
        raise ValueError("the first diagnostic is not the family's analysis")
    admitted = _first(trace, "admitted", (stage,))
    if admitted is not None and admitted.problem is not raw.problem:
        raise ValueError("the admitted diagnostic is not the first one")
    return admitted


def _admitted(trace: TraceView, admit: Admission) -> tuple[OutcomeView | None, str]:
    """The trace's outcome as the caller's admission admitted it, or why it was not.

    ``None`` with no reason when the trace holds no outcome.  The admission's
    ``ValueError``, an admitted object other than the trace's own outcome, or a
    complete status without its sample (any other status with one) is no admitted
    outcome: ``None`` with the reason.
    """
    outcome = trace.outcome
    if outcome is None:
        return None, ""
    try:
        found = admit(trace)
    except ValueError as error:
        return None, str(error)
    if found is not outcome:
        return None, "the admission returned another outcome than the trace's"
    complete = found.status == "complete"
    sample = found.sample
    if complete != (sample is not None) or (complete and type(sample) is not FESample):
        return None, "a complete outcome holds its sample, and only it"
    return found, ""


def classified(
    diagnostic: SupportResult | GammaResult,
    failure: Exception | None,
    admitted: OutcomeView | None,
) -> tuple[TriggerStatus, str, IndexArray]:
    """The classification of an admitted first diagnostic: status, reason, removal.

    ``failure`` is the trace's propagated failure and ``admitted`` the call's outcome
    as the caller's admission admitted it (``None`` when none was): witnessed
    separation is ``separation_recomposed`` only with a complete admitted outcome.
    """
    removed = diagnostic.positions if isinstance(diagnostic, SupportResult) else NONE
    if failure is not None:
        return "failed", f"{type(failure).__name__}: {failure}", removed
    status: TriggerStatus = "inconclusive"
    if isinstance(diagnostic, SupportResult):
        if removed.size:
            complete = admitted is not None and admitted.status == "complete"
            status = "separation_recomposed" if complete else "separation_unresolved"
        elif diagnostic.status == "complete":
            status = "false_alarm"
    elif diagnostic.status in RECESSIONS:
        status = "existence_failure"
    elif diagnostic.status == "no_recession":
        status = "false_alarm"
    return status, diagnostic.reason, removed


class _Reading(NamedTuple):
    """What the solver reads from a trace."""

    status: TriggerStatus
    reason: str
    diagnostic: SupportResult | GammaResult | None
    removed: IndexArray
    outcome: OutcomeView | None
    admission: str


def _read(request: TriggerRequest, trace: TraceView, admit: Admission) -> _Reading:
    """Bind the trace, admit its first diagnostic again and its outcome; classify."""
    try:
        event = _bound(request, trace)
        diagnostic = None if event is None else _again(request, event)
    except ValueError as error:
        return _Reading("refused", str(error), None, NONE, None, "")
    if diagnostic is not None:
        outcome, admission = _admitted(trace, admit)
        status, reason, removed = classified(diagnostic, trace.failure, outcome)
        return _Reading(status, reason, diagnostic, removed, outcome, admission)
    failure = trace.failure
    if failure is None:
        reason = "the trace holds neither an admitted diagnostic nor a failure"
        return _Reading("refused", reason, None, NONE, None, "")
    text = f"{type(failure).__name__}: {failure} (unfinished: {trace.unfinished})"
    return _Reading("failed", text, None, NONE, None, "")


def triggered(
    trigger: Trigger, admit: Admission, request: TriggerRequest
) -> TriggerRecord:
    """The one call of the trigger, and the solver's reading of its trace."""
    wall, cpu = time.perf_counter(), time.process_time()
    trace = trigger(request)
    timing = Timing(time.perf_counter() - wall, time.process_time() - cpu)
    reading = _read(request, trace, admit)
    return TriggerRecord(request, trace, *reading, timing)
