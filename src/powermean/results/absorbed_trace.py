"""Detached snapshots of recorded trace events, traces, and trigger records.

This module defines the explicit finite detached DTOs for recorded trace events, traces,
and triggers without importing ``powermean.interface`` or using structural
``_OutcomeView`` protocols.  Coordinator outcome and invocation records live in
:mod:`powermean.results.absorbed_outcome_records`.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.data.collinearity import (
    CollinearityProblem,
    DiagnosticControls,
)
from powermean.data.sample_fe import Closure
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_attempts import RetainedAlarm, _owned_alarm
from powermean.results.absorbed_outcome_records import (
    RetainedClosureSnapshot,
    RetainedCollinearityInvocation,
    RetainedOutcomeSnapshot,
    RetainedRecessionInvocation,
    RetainedRound,
    RetainedSampleRequest,
    RetainedSupportInvocation,
    _owned_closure,
    build_sample_request_snapshot,
)
from powermean.results.absorbed_partitions import (
    PartitionIdentityMap,
    RetainedGroupingSnapshot,
)
from powermean.results.absorbed_points import (
    PointIdentityMap,
    RetainedPointSnapshot,
    snapshot_point_record,
)
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    snapshot_fe_sample,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_collinearity import (
    RetainedCollinearityAnalysis,
    RetainedCollinearityProblem,
    snapshot_collinearity_analysis,
)
from powermean.results.retained_diagnostic_problems import (
    RetainedGammaProblem,
    RetainedSupportProblem,
)
from powermean.results.retained_gamma import (
    RetainedGammaResult,
    snapshot_gamma_result,
)
from powermean.results.retained_inputs import (
    PointKey,
    ProblemIdentityMap,
    ProblemKey,
    SampleKey,
)
from powermean.results.retained_support import (
    RawSupportFields,
    RetainedRawSupport,
    RetainedSupportResult,
    snapshot_raw_support,
    snapshot_support_result,
)
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.linear_absorbed_point import LinearPoint
from powermean.solve.newton_trace import Timing
from powermean.solve.profiled_trigger import TriggerRecord, TriggerStatus
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.solve.support import SupportControls, SupportResult
from powermean.solve.support_gamma import GammaProblem, GammaResult
from powermean.solve.support_lp import LPControls
from powermean.solve.support_problem import SupportProblem
from powermean.systems.profiled import PointProfile

Family = PowerVariance | NB2Variance
DiagnosticProblem = SupportProblem | GammaProblem | CollinearityProblem
DiagnosticControl = SupportControls | LPControls | DiagnosticControls
DetachedDiagnosticProblem = (
    RetainedSupportProblem | RetainedGammaProblem | RetainedCollinearityProblem
)
DetachedDiagnosticResult = (
    RetainedSupportResult
    | RetainedRawSupport
    | RetainedGammaResult
    | RetainedCollinearityAnalysis
)


def snapshot_diagnostic_problem(
    problem: DiagnosticProblem | None,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> DetachedDiagnosticProblem | None:
    """Detach a concrete diagnostic problem record without re-encoding ``Grouping``."""
    if problem is None:
        return None
    dg = None if partition_map is None else partition_map.register_grouping
    if type(problem) is SupportProblem:
        rp = None if problem_map is None else problem_map.register_support
        return RetainedSupportProblem.from_problem(
            problem, register_problem=rp, detach_group=dg
        )
    if type(problem) is GammaProblem:
        rg = None if problem_map is None else problem_map.register_gamma
        return RetainedGammaProblem.from_problem(
            problem, register_problem=rg, detach_group=dg
        )
    if type(problem) is CollinearityProblem:
        rc = None if problem_map is None else problem_map.register_collinearity
        return RetainedCollinearityProblem.from_problem(
            problem, register_problem=rc, detach_group=dg
        )
    raise TypeError(f"unexpected diagnostic problem type {type(problem)!r}")


@dataclass(frozen=True)
class RetainedEventSnapshot:
    """Detached snapshot of a ``RecordedEvent`` in a trigger trace."""

    kind: str
    stage: str
    problem: DetachedDiagnosticProblem | None = None
    controls: DiagnosticControl | None = None
    result: DetachedDiagnosticResult | None = None
    _closure: Closure | None = None
    _removed: IndexArray | None = None
    mode: str | None = None
    unowned: str = ""
    problem_key: ProblemKey | None = None

    def __post_init__(self) -> None:
        """Store detached event fields preserving raw vs admitted distinctions."""
        cls = None if self._closure is None else _owned_closure(self._closure)
        pkey = None if self.problem is None else self.problem.key
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "problem_key", pkey)
        object.__setattr__(self, "_closure", cls)
        object.__setattr__(self, "_removed", _optional(self._removed))
        object.__setattr__(self, "mode", None if self.mode is None else str(self.mode))
        object.__setattr__(self, "unowned", str(self.unowned))

    @property
    def closure(self) -> Closure | None:
        """Return a fresh owned ``Closure`` or ``None``."""
        return None if self._closure is None else _owned_closure(self._closure)

    @property
    def removed(self) -> IndexArray | None:
        """Return a fresh owned copy of ``removed``, or ``None``."""
        return _optional(self._removed)


@dataclass(frozen=True)
class RetainedTraceSnapshot:
    """Detached snapshot of ``RecordedTrace`` without live exceptions or tracebacks."""

    origin: RetainedOutcomeSnapshot
    start: RetainedSampleSnapshot
    request: RetainedSampleRequest
    events: tuple[RetainedEventSnapshot, ...]
    outcome: RetainedOutcomeSnapshot | None
    unfinished: str | None
    failure_type: str | None
    failure_message: str | None


def _detach_trigger_diagnostic(
    diag: SupportResult | GammaResult | None,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedSupportResult | RetainedGammaResult | None:
    """Detach ``TriggerRecord.diagnostic`` into its finite retained representation."""
    if diag is None:
        return None
    if type(diag) is SupportResult:
        return snapshot_support_result(
            diag, problem_map=problem_map, partition_map=partition_map
        )
    if type(diag) is GammaResult:
        return snapshot_gamma_result(
            diag, problem_map=problem_map, partition_map=partition_map
        )
    raise TypeError(f"unexpected trigger diagnostic type {type(diag)!r}")


@dataclass(frozen=True)
class RetainedTriggerSnapshot:
    """Detached snapshot of :class:`~powermean.solve.profiled_trigger.TriggerRecord`."""

    status: TriggerStatus
    reason: str
    diagnostic: RetainedSupportResult | RetainedGammaResult | None
    _removed: IndexArray
    outcome: RetainedOutcomeSnapshot | None
    admission: str
    timing: Timing
    variance: Family
    support: SupportControls | None
    recession: LPControls | None
    alarm: RetainedAlarm
    sample: RetainedSampleSnapshot
    trace: RetainedTraceSnapshot
    _point_theta: FloatArray
    _point_base: FloatArray
    _point_index: FloatArray
    point_key: PointKey | None = None
    point: RetainedPointSnapshot | None = None

    @property
    def removed(self) -> IndexArray:
        """Return a fresh owned copy of ``removed``."""
        return owned_array(self._removed)

    @property
    def point_theta(self) -> FloatArray:
        """Return a fresh owned copy of ``point_theta``."""
        return owned_array(self._point_theta)

    @property
    def point_base(self) -> FloatArray:
        """Return a fresh owned copy of ``point_base``."""
        return owned_array(self._point_base)

    @property
    def point_index(self) -> FloatArray:
        """Return a fresh owned copy of ``point_index``."""
        return owned_array(self._point_index)

    @property
    def recomposed(self) -> RetainedOutcomeSnapshot | None:
        """Return the detached recomposed coordinator outcome, if any."""
        return self.outcome

    @property
    def origin_sample_key(self) -> SampleKey:
        """Return the ``SampleKey`` of the sample from which the trigger started."""
        return self.sample.key

    @property
    def recomposed_sample_key(self) -> SampleKey | None:
        """Return the ``SampleKey`` of the recomposed outcome's sample, if any."""
        if self.outcome is None or self.outcome.sample is None:
            return None
        return self.outcome.sample.key


def snapshot_trigger(
    record: TriggerRecord,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    trace_snapshot: RetainedTraceSnapshot,
    outcome_snapshot: RetainedOutcomeSnapshot | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
    point_map: PointIdentityMap | None = None,
) -> RetainedTriggerSnapshot:
    """Snapshot a ``TriggerRecord`` into a :class:`RetainedTriggerSnapshot`."""
    if type(record) is not TriggerRecord:
        raise TypeError("record must be a concrete TriggerRecord")
    req = record.request
    diag = _detach_trigger_diagnostic(
        record.diagnostic, problem_map=problem_map, partition_map=partition_map
    )
    alarm_snap = _owned_alarm(req.alarm)
    assert alarm_snap is not None
    pt = req.point.point
    pt_snap: RetainedPointSnapshot | None = None
    pt_key: PointKey | None = None
    if isinstance(pt, PointProfile | ConditionalPoint | IndexPoint | LinearPoint):
        if point_map is not None:
            pt_snap = point_map.get(
                pt, sample_snapshot=sample_snapshot, variance=req.variance
            )
        else:
            gmap = None if partition_map is None else partition_map.grouping_map
            pt_snap = snapshot_point_record(
                pt,
                sample_snapshot=sample_snapshot,
                variance=req.variance,
                grouping_map=gmap,
            )
        pt_key = pt_snap.key
    return RetainedTriggerSnapshot(
        record.status,
        str(record.reason),
        diag,
        owned_array(record.removed),
        outcome_snapshot,
        str(record.admission),
        record.timing,
        req.variance,
        req.support,
        req.recession,
        alarm_snap,
        sample_snapshot,
        trace_snapshot,
        owned_array(pt.theta),
        owned_array(pt.base),
        owned_array(pt.index),
        pt_key,
        pt_snap,
    )


__all__ = [
    "DetachedDiagnosticProblem",
    "DetachedDiagnosticResult",
    "DiagnosticControl",
    "DiagnosticProblem",
    "Family",
    "RawSupportFields",
    "RetainedClosureSnapshot",
    "RetainedCollinearityAnalysis",
    "RetainedCollinearityInvocation",
    "RetainedCollinearityProblem",
    "RetainedEventSnapshot",
    "RetainedGammaProblem",
    "RetainedGammaResult",
    "RetainedGroupingSnapshot",
    "RetainedOutcomeSnapshot",
    "RetainedRawSupport",
    "RetainedRecessionInvocation",
    "RetainedRound",
    "RetainedSampleRequest",
    "RetainedSupportInvocation",
    "RetainedSupportProblem",
    "RetainedSupportResult",
    "RetainedTraceSnapshot",
    "RetainedTriggerSnapshot",
    "_detach_trigger_diagnostic",
    "_owned_closure",
    "build_sample_request_snapshot",
    "snapshot_collinearity_analysis",
    "snapshot_diagnostic_problem",
    "snapshot_fe_sample",
    "snapshot_gamma_result",
    "snapshot_raw_support",
    "snapshot_support_result",
    "snapshot_trigger",
]
