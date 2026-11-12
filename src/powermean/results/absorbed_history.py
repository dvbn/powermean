"""Detached snapshots of absorbed candidate histories across all four routes.

This module defines :class:`RetainedHistorySnapshot`, :func:`_point_mapper`,
:func:`_candidate_inner_and_start`, :func:`snapshot_candidate_history`, and
:func:`snapshot_history`, and re-exports linear decomposition and state symbols from
:mod:`powermean.results.absorbed_linear_history`.  The selected-root route (``"root"``)
keeps its stages as its attempts (:mod:`powermean.results.absorbed_root_records`), its
target stage as ``retained``, no final work, and the resolved solver and requested
organization's options; its initializer's own history is snapshotted by
:mod:`powermean.results.absorbed_root`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from powermean.results.absorbed_attempts import (
    RetainedAlarm,
    RetainedIndexAttempt,
    RetainedIndexFinal,
    RetainedIndexStart,
    RetainedLocalInitializer,
    RetainedProfiledAttempt,
    RetainedProfiledFinal,
    _owned_alarm,
    snapshot_index_attempt,
    snapshot_profiled_attempt,
)
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
)
from powermean.results.absorbed_linear_history import (
    RetainedDecomposition,
    RetainedLinearState,
    _snapshot_linear_state,
)
from powermean.results.absorbed_points import (
    AbsorbedRoute,
    Family,
    PointIdentityMap,
    PointMapper,
    RawPointRecord,
    RetainedFreshEquations,
    RetainedLinearNormalization,
    RetainedPointSnapshot,
    _compute_beta,
    _snapshot_linear_point,
    snapshot_fresh_equations,
    snapshot_linear_normalization,
    snapshot_point_record,
)
from powermean.results.absorbed_root_records import (
    RetainedRootStage,
    snapshot_root_stage,
)
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    SampleIdentityMap,
)
from powermean.results.absorbed_states import (
    RetainedAbsorptionResult,
    RetainedAbsorptionState,
    RetainedPreparedWeights,
    RetainedRecovery,
    _owned_absorption_result,
    _owned_recovery,
)
from powermean.results.absorbed_trace import RetainedTriggerSnapshot
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import PointKey, SampleKey
from powermean.solve.absorbed_certificate_history import CandidateHistory, Route
from powermean.solve.absorbed_certificate_point import InnerControls
from powermean.solve.absorbed_direction import IndexWork
from powermean.solve.absorbed_root_records import (
    AbsorbedRootCandidate,
    AbsorbedRootStage,
    AbsorbedRootWork,
)
from powermean.solve.assessment import Assessment
from powermean.solve.linear_absorbed_point import LinearControls, LinearPoint
from powermean.solve.linear_absorbed_state import (
    AbsorbedLinearCandidate,
    LinearWork,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexAttempt, IndexCandidate
from powermean.solve.newton_profiled import ProfiledAttempt, ProfiledCandidate
from powermean.solve.newton_trace import Timing
from powermean.solve.options import Provider, SolverOptions
from powermean.solve.profiled_loss import SolveWork
from powermean.solve.starts import Start

__all__ = [
    "AbsorbedRoute",
    "PointIdentityMap",
    "RetainedAbsorptionResult",
    "RetainedAbsorptionState",
    "RetainedDecomposition",
    "RetainedFreshEquations",
    "RetainedHistorySnapshot",
    "RetainedIndexAttempt",
    "RetainedLinearNormalization",
    "RetainedLinearState",
    "RetainedLocalInitializer",
    "RetainedPointSnapshot",
    "RetainedPreparedWeights",
    "RetainedProfiledAttempt",
    "RetainedRecovery",
    "_candidate_inner_and_start",
    "_compute_beta",
    "_owned_absorption_result",
    "_owned_recovery",
    "_point_mapper",
    "_snapshot_linear_state",
    "snapshot_candidate_history",
    "snapshot_fresh_equations",
    "snapshot_history",
    "snapshot_linear_normalization",
    "snapshot_point_record",
]


@dataclass(frozen=True)
class RetainedHistorySnapshot:
    """Detached snapshot of an admitted ``CandidateHistory`` across all four routes."""

    route: Route
    sample_key: SampleKey
    sample: RetainedSampleSnapshot
    variance: Family | None
    retained: int | None
    attempts: (
        tuple[RetainedProfiledAttempt, ...]
        | tuple[RetainedIndexAttempt, ...]
        | tuple[RetainedRootStage, ...]
    )
    reason: str
    message: str
    native: str | None
    final: RetainedProfiledFinal | RetainedIndexFinal | RetainedAbsorptionResult | None
    partials: tuple[str, ...]
    unreached: tuple[str, ...]
    alarm: RetainedAlarm | None
    trigger: RetainedTriggerSnapshot | None
    work: SolveWork | IndexWork | LinearWork | AbsorbedRootWork
    timing: Timing | None
    point: RetainedPointSnapshot | None
    linear_state: RetainedLinearState | None
    _start: Start | RetainedIndexStart | None
    provider: Provider | None
    provider_version: str | None
    initial_assessment: Assessment | None
    no_measurable_progress: bool | None
    assessments: int | None
    assessment_timing: Timing | None
    options: SolverOptions | None
    absorbed_options: AbsorbedOptions | None
    factorized: bool | None
    linear_controls: LinearControls | None

    @property
    def start(self) -> Start | RetainedIndexStart | None:
        """Return a fresh owned copy of the candidate start record, or ``None``."""
        if isinstance(self._start, Start):
            return Start(owned_array(self._start.coefficients), self._start.source)
        return self._start

    @property
    def selected_point(self) -> RetainedPointSnapshot | None:
        """Return the selected point snapshot, or ``None``."""
        return self.point

    @property
    def selected(self) -> RetainedPointSnapshot | None:
        """Compatibility alias for ``selected_point``."""
        return self.point

    @property
    def selected_point_key(self) -> PointKey | None:
        """Return the ``PointKey`` of the selected point, or ``None``."""
        return None if self.point is None else self.point.key

    @property
    def linear(self) -> RetainedLinearState | None:
        """Compatibility alias for ``linear_state``."""
        return self.linear_state


def _point_mapper(
    sample_snapshot: RetainedSampleSnapshot,
    variance: Family | None,
    inner: InnerControls | None,
    selected_raw: RawPointRecord | None,
    selected_snapshot: RetainedPointSnapshot | None,
    *,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> PointMapper:
    """Return a typed ``PointMapper`` backed by a :class:`PointIdentityMap`."""
    if point_map is not None:
        point_map.bind_context(
            sample_snapshot=sample_snapshot,
            variance=variance,
            inner=inner,
            selected_raw=selected_raw,
            selected_snapshot=selected_snapshot,
        )
        return point_map.get
    pmap = PointIdentityMap(
        sample_snapshot=sample_snapshot,
        variance=variance,
        inner=inner,
        selected_raw=selected_raw,
        selected_snapshot=selected_snapshot,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    return pmap.get


def _candidate_inner_and_start(
    cand: (
        ProfiledCandidate
        | IndexCandidate
        | AbsorbedLinearCandidate
        | AbsorbedRootCandidate
    ),
    route: Route,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> tuple[InnerControls, Start | RetainedIndexStart | None]:
    """Extract ``InnerControls`` and owned start record from ``cand``.

    A root candidate's controls are the requested organization's, as its points'
    admission keeps them; its start the caller's slopes, as given.
    """
    if isinstance(cand, AbsorbedRootCandidate):
        a, given = cand.absorbed, cand.start
        ptol = a.profile_tol if a.absorbed == "profiled" else None
        c_a = (a.absorb_tol_final, a.absorb_max_sweeps, a.absorb_acceleration)
        root_st = None if given is None else Start(owned_array(given.view()), "user")
        return InnerControls(*c_a, ptol), root_st
    if isinstance(cand, ProfiledCandidate):
        a, ps = cand.absorbed, cand.start
        c_a = (a.absorb_tol_final, a.absorb_max_sweeps, a.absorb_acceleration)
        inn = InnerControls(*c_a, a.profile_tol)
        st = None if ps is None else Start(owned_array(ps.coefficients), ps.source)
        return inn, st
    if isinstance(cand, IndexCandidate):
        a, is_ = cand.absorbed, cand.start
        ptol = None if route != "profiled" else a.profile_tol
        c_a = (a.absorb_tol_final, a.absorb_max_sweeps, a.absorb_acceleration)
        inn = InnerControls(*c_a, ptol)
        idx_st = (
            None
            if is_ is None
            else RetainedIndexStart.from_start(
                is_,
                grouping_snapshot=grouping_snapshot,
                grouping_map=grouping_map,
            )
        )
        return inn, idx_st
    c = cand.controls
    c_a = (c.absorb_tol_final, c.absorb_max_sweeps, c.absorb_acceleration)
    return InnerControls(*c_a, None), None


def _snapshot_route_attempts(
    history: CandidateHistory,
    sample_snapshot: RetainedSampleSnapshot,
    selected_point: RetainedPointSnapshot | None,
    *,
    get_pt: PointMapper,
    grp_snap: RetainedGroupingSnapshot | None,
    sample_map: SampleIdentityMap | None,
    grouping_map: GroupingIdentityMap | None,
) -> tuple[
    tuple[RetainedProfiledAttempt, ...]
    | tuple[RetainedIndexAttempt, ...]
    | tuple[RetainedRootStage, ...],
    RetainedProfiledFinal | RetainedIndexFinal | RetainedAbsorptionResult | None,
    RetainedLinearState | None,
]:
    """Detach attempts, final work, and optional linear state for ``history``."""
    cand = history.candidate
    if history.route == "root":
        r_in = cast("tuple[AbsorbedRootStage, ...]", history.attempts)
        return tuple(snapshot_root_stage(s) for s in r_in), None, None
    if history.route in ("profiled", "reduced"):
        p_in = cast("tuple[ProfiledAttempt, ...]", history.attempts)
        pa = tuple(
            snapshot_profiled_attempt(
                a, get_pt, grouping_snapshot=grp_snap, grouping_map=grouping_map
            )
            for a in p_in
        )
        fin_p = None if history.retained is None else pa[history.retained].final
        return pa, fin_p, None
    if history.route == "index":
        i_in = cast("tuple[IndexAttempt, ...]", history.attempts)
        ia = tuple(
            snapshot_index_attempt(
                a, get_pt, grouping_snapshot=grp_snap, grouping_map=grouping_map
            )
            for a in i_in
        )
        fin_i = None if history.retained is None else ia[history.retained].final
        return ia, fin_i, None
    assert isinstance(cand, AbsorbedLinearCandidate)
    lin_state = _snapshot_linear_state(
        cand,
        selected_point,
        sample_key=sample_snapshot.key,
        grouping_snapshot=grp_snap,
        get_point=get_pt,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    return (), lin_state.final, lin_state


def snapshot_candidate_history(
    history: CandidateHistory,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    selected_point: RetainedPointSnapshot | None = None,
    trigger_snapshot: RetainedTriggerSnapshot | None = None,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedHistorySnapshot:
    """Snapshot an admitted ``CandidateHistory`` with shared selected point."""
    if history.trigger is not None and trigger_snapshot is None:
        raise ValueError(
            "history.trigger requires a lowered trigger_snapshot from admission"
        )
    cand, var = history.candidate, history.variance
    grp_snap = sample_snapshot._final_grouping
    inner, start_rec = _candidate_inner_and_start(
        cand, history.route, grouping_snapshot=grp_snap, grouping_map=grouping_map
    )
    if selected_point is None and history.point is not None:
        if isinstance(cand, AbsorbedLinearCandidate):
            assert isinstance(history.point, LinearPoint)
            idx = 0 if point_map is None else point_map.allocate_key(history.point)
            selected_point = _snapshot_linear_point(
                history.point,
                sample_snapshot,
                inn=inner,
                pkey=idx,
                linear_normalization=cand.normalized,
                equations=cand.equations,
                sample_map=sample_map,
            )
        elif point_map is not None:
            selected_point = point_map.get(
                history.point,
                sample_snapshot=sample_snapshot,
                variance=var,
                inner=inner,
            )
        else:
            selected_point = snapshot_point_record(
                history.point,
                sample_snapshot=sample_snapshot,
                variance=var,
                inner=inner,
                sample_map=sample_map,
                grouping_map=grouping_map,
            )
    get_pt = _point_mapper(
        sample_snapshot,
        var,
        inner,
        history.point,
        selected_point,
        point_map=point_map,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    atts, fin, lin_state = _snapshot_route_attempts(
        history,
        sample_snapshot,
        selected_point,
        get_pt=get_pt,
        grp_snap=grp_snap,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    nl = cand if isinstance(cand, ProfiledCandidate | IndexCandidate) else None
    fac = cand.factorized if isinstance(cand, IndexCandidate) else None
    l_ctrl = cand.controls if isinstance(cand, AbsorbedLinearCandidate) else None
    nat = None if history.native is None else str(history.native)
    prts = tuple(str(p) for p in history.partials)
    unr = tuple(str(u) for u in history.unreached)
    alm = _owned_alarm(history.alarm)
    rsn, msg = str(history.reason), str(history.message)
    nl_t = (
        (None, None, None, None, None, None, None, None, fac, l_ctrl)
        if nl is None
        else (
            nl.provider,
            nl.provider_version,
            nl.initial_assessment,
            nl.no_measurable_progress,
            nl.assessments,
            nl.assessment_timing,
            nl.options,
            nl.absorbed,
            fac,
            l_ctrl,
        )
    )
    r1 = (sample_snapshot, var, history.retained, atts, rsn, msg, nat, fin, prts, unr)
    r2 = (alm, trigger_snapshot, history.work, history.timing)
    if isinstance(cand, AbsorbedRootCandidate):
        # The solver and requested organization's options; no provider record.
        r3_root = (selected_point, lin_state, start_rec, None, None, None, None)
        nothing = (None, None, cand.options, cand.absorbed, None, None)
        root_rest = (*r1, *r2, *r3_root, *nothing)
        return RetainedHistorySnapshot(history.route, sample_snapshot.key, *root_rest)
    r3 = (selected_point, lin_state, start_rec, *nl_t)
    rest = (*r1, *r2, *r3)
    return RetainedHistorySnapshot(history.route, sample_snapshot.key, *rest)


snapshot_history = snapshot_candidate_history
