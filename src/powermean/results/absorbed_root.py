"""The retained record of an absorbed joint root solve outside ``tau in [1, 2]``.

The solver, its options, the start that produced the reported solution, the path and the
work are recorded.  The selected-root route of :mod:`powermean.solve.absorbed_root`
keeps, beside the snapshots that every absorbed route keeps (its history, with its
stages as attempts, and the certificate of its selected point), what this module
detaches:

- the operation of the root stages and the organization requested at the fit, which
  its convex initializer ran: recorded apart, neither labelled as the other;
- the resolved root-Newton options;
- the initializer: its kind, convex variance, certified flag, status and reason, the
  start's index gap, the budgets charged, the start in the full coordinates, and the
  convex fit's OWN history and certificate snapshots (the inner checks and work that
  actually ran, at their unchanged bounds), detached by the snapshot
  routines with the same identity maps; the convex certificate's slope map and rows
  are kept as discarded, so that no inference is ever formed at the initializer;
- the fixed merit metric (its matrix, scale, factor record and pivots) or the reason
  of its absence;
- the target stage, the continuation flag, the stage variances, every direction
  record, the merit solves and the counted work.

Nothing here solves, certifies or recomputes; the arrays are owned copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from powermean.fe.admission_records import AdmissionWork
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.absorbed_groupings import GroupingIdentityMap
from powermean.results.absorbed_history import (
    PointIdentityMap,
    RetainedHistorySnapshot,
    RetainedPointSnapshot,
    _candidate_inner_and_start,
    snapshot_candidate_history,
    snapshot_point_record,
)
from powermean.results.absorbed_owned import (
    RetainedCertificateSnapshot,
    snapshot_certificate,
)
from powermean.results.absorbed_root_records import (
    RetainedMeritSnapshot,
    frozen_counts,
    snapshot_directions,
    snapshot_merit,
)
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    SampleIdentityMap,
)
from powermean.results.absorbed_trace import RetainedTriggerSnapshot
from powermean.results.owned import _optional, owned_array
from powermean.solve.absorbed_root_records import (
    AbsorbedRootCandidate,
    RootDirectionRecord,
)
from powermean.solve.cancelled_records import RootOptions

__all__ = [
    "RetainedRootInitializer",
    "RetainedRootSnapshot",
    "snapshot_root",
]


@dataclass(frozen=True)
class RetainedRootInitializer:
    """Owned snapshot of the start of a root solve (module docstring)."""

    kind: str
    variance: PowerVariance | None
    certified: bool
    status: str | None
    reason: str | None
    gap: float | None
    iterations: int
    evaluations: int
    start: RetainedPointSnapshot | None
    history: RetainedHistorySnapshot | None
    certificate: RetainedCertificateSnapshot | None


@dataclass(frozen=True)
class RetainedRootSnapshot:
    """Owned record of one absorbed joint root solve (module docstring)."""

    operation: str
    organization: str
    root_options: RootOptions
    initializer: RetainedRootInitializer
    merit: RetainedMeritSnapshot | None
    merit_reason: str
    target_stage: int | None
    continuation_used: bool
    stage_variances: tuple[float, ...]
    directions: tuple[RootDirectionRecord, ...]
    merit_actions: Mapping[str, int]
    work: AdmissionWork
    iterations: int
    evaluations: int
    root_iterations: int
    root_evaluations: int
    _caller_start: FloatArray | None = None

    def __post_init__(self) -> None:
        """Own the caller's slopes and freeze the counts."""
        object.__setattr__(self, "_caller_start", _optional(self._caller_start))
        object.__setattr__(self, "merit_actions", frozen_counts(self.merit_actions))

    @property
    def caller_start(self) -> FloatArray | None:
        """Return a fresh owned copy of the caller's slopes, or ``None``."""
        return _optional(self._caller_start)


def _initializer(
    candidate: AbsorbedRootCandidate,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    trigger_snapshot: RetainedTriggerSnapshot | None,
    point_map: PointIdentityMap | None,
    sample_map: SampleIdentityMap | None,
    grouping_map: GroupingIdentityMap | None,
) -> RetainedRootInitializer:
    """Detach the initializer: its start point, its convex history and certificate."""
    initializer = candidate.initializer
    grp_snap = sample_snapshot._final_grouping
    inner, _ = _candidate_inner_and_start(
        candidate, "root", grouping_snapshot=grp_snap, grouping_map=grouping_map
    )
    start: RetainedPointSnapshot | None = None
    if initializer.point is not None:
        if point_map is not None:
            start = point_map.get(
                initializer.point,
                sample_snapshot=sample_snapshot,
                variance=candidate.variance,
                inner=inner,
            )
        else:
            start = snapshot_point_record(
                initializer.point,
                sample_snapshot=sample_snapshot,
                variance=candidate.variance,
                inner=inner,
                sample_map=sample_map,
                grouping_map=grouping_map,
            )
    history: RetainedHistorySnapshot | None = None
    certificate: RetainedCertificateSnapshot | None = None
    audit = initializer.audit
    if audit is not None:
        history = snapshot_candidate_history(
            audit.history,
            sample_snapshot=sample_snapshot,
            trigger_snapshot=trigger_snapshot,
            point_map=point_map,
            sample_map=sample_map,
            grouping_map=grouping_map,
        )
        if audit.certificate is not None:
            certificate = snapshot_certificate(
                audit.certificate,
                sample_snapshot=sample_snapshot,
                selected_point=history.selected_point,
                keep_slope_map=False,
                point_map=point_map,
                sample_map=sample_map,
                grouping_map=grouping_map,
            )
    return RetainedRootInitializer(
        str(initializer.kind),
        initializer.variance,
        bool(initializer.certified),
        initializer.status,
        initializer.reason,
        initializer.gap,
        int(initializer.iterations),
        int(initializer.evaluations),
        start,
        history,
        certificate,
    )


def snapshot_root(
    candidate: AbsorbedRootCandidate,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    trigger_snapshot: RetainedTriggerSnapshot | None = None,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedRootSnapshot:
    """Detach the record of a root candidate (module docstring).

    ``trigger_snapshot`` is the admitted snapshot of the convex initializer's own
    trigger record, where its history holds one (the accepted history snapshot
    refuses a held trigger without it).
    """
    if type(candidate) is not AbsorbedRootCandidate:
        raise TypeError("candidate must be an AbsorbedRootCandidate")
    initializer = _initializer(
        candidate,
        sample_snapshot=sample_snapshot,
        trigger_snapshot=trigger_snapshot,
        point_map=point_map,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    given = candidate.start
    return RetainedRootSnapshot(
        operation=candidate.operation,
        organization=candidate.organization,
        root_options=candidate.root_options,
        initializer=initializer,
        merit=snapshot_merit(candidate.merit),
        merit_reason=str(candidate.merit_reason),
        target_stage=candidate.target_stage,
        continuation_used=bool(candidate.continuation_used),
        stage_variances=tuple(float(s.variance.tau) for s in candidate.stages),
        directions=snapshot_directions(candidate.work.directions),
        merit_actions=candidate.work.merit_actions,
        work=candidate.work.counts,
        iterations=int(candidate.iterations),
        evaluations=int(candidate.evaluations),
        root_iterations=int(candidate.root_iterations),
        root_evaluations=int(candidate.root_evaluations),
        _caller_start=None if given is None else owned_array(given.view()),
    )
