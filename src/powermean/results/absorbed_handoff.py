"""Lower absorbed FE envelope composition, initial scale cache, and covariance service.

This module defines :class:`CaptureMaps` for per-handoff identity wiring and anchor
clearing, composes lower-layer audit admission, detached
sample/point/history/certificate snapshots, the pre-discard initial covariance and
direct-SE cache, and guarded post-fit covariance recomputation via
:func:`powermean.inference.absorbed.fe_covariance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean.fe.certificate_basis import _operational_data, embed, lift
from powermean.inference.absorbed import fe_covariance
from powermean.inference.provider import direct_standard_errors
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, _finite_result
from powermean.results.absorbed_binding import (
    AdmittedAbsorbedGraph,
    admit_absorbed_graph,
)
from powermean.results.absorbed_groupings import GroupingIdentityMap
from powermean.results.absorbed_geometry_records import _OperationalBasisSnapshot
from powermean.results.absorbed_attempts import (
    _owned_absorbed_schur,
    _owned_alarm,
    _owned_iteration,
)
from powermean.results.absorbed_history import (
    PointIdentityMap,
    RetainedHistorySnapshot,
    _candidate_inner_and_start,
    _point_mapper,
    snapshot_candidate_history,
    snapshot_point_record,
)
from powermean.results.absorbed_points import PointMapper
from powermean.results.absorbed_owned import (
    RetainedCertificateSnapshot,
    RetainedGeometrySnapshot,
    _GeometryData,
    _OperationalCertificate,
    _capture_certificate,
)
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_retained import (
    RetainedAbsorbedEnvelope,
    RetainedInitialScaleCache,
    _OperationalHistory,
    _ProfiledAttemptData,
    _ProfiledFinalData,
    snapshot_covariance_request,
    snapshot_initial_scale,
)
from powermean.results.absorbed_root import RetainedRootSnapshot, snapshot_root
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    SampleIdentityMap,
)
from powermean.results.absorbed_trace import (
    RetainedOutcomeSnapshot,
    RetainedTriggerSnapshot,
)
from powermean.results.owned import owned_array
from powermean.results.retained_inputs import (
    InputIdentityMap,
    ProblemIdentityMap,
    RetainedInputSnapshot,
    owned_order_scale,
)
from powermean.solve.absorbed_certificate import CandidateAudit
from powermean.solve.absorbed_certificate_history import CandidateHistory
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleResult,
)
from powermean.solve.absorbed_root_records import AbsorbedRootCandidate
from powermean.solve.newton_profiled import ProfiledAttempt, ProfiledCandidate
from powermean.solve.profiled_loss import SolveWork
from powermean.solve.reported import ReportedScale
from powermean.solve.starts import Start
from powermean.systems.joint import SamplingUnits


@dataclass(frozen=True)
class CaptureMaps:
    """Per-handoff identity maps wired across input, grouping, sample, and point."""

    input_map: InputIdentityMap = field(default_factory=InputIdentityMap)
    grouping_map: GroupingIdentityMap = field(default_factory=GroupingIdentityMap)
    partition_map: PartitionIdentityMap = field(init=False)
    sample_map: SampleIdentityMap = field(init=False)
    point_map: PointIdentityMap = field(init=False)
    problem_map: ProblemIdentityMap = field(default_factory=ProblemIdentityMap)

    def __post_init__(self) -> None:
        """Wire shared input, grouping, partition, and sample maps across capture."""
        part_map = PartitionIdentityMap(grouping_map=self.grouping_map)
        samp_map = SampleIdentityMap(
            input_map=self.input_map,
            partition_map=part_map,
        )
        object.__setattr__(self, "partition_map", part_map)
        object.__setattr__(self, "sample_map", samp_map)
        object.__setattr__(
            self,
            "point_map",
            PointIdentityMap(
                sample_map=samp_map,
                grouping_map=self.grouping_map,
            ),
        )

    def clear_anchors(self) -> None:
        """Release all temporary original-object reference anchors after capture."""
        self.input_map.clear_anchors()
        self.grouping_map.clear_anchors()
        self.partition_map.clear_anchors()
        self.sample_map.clear_anchors()
        self.point_map.clear_anchors()
        self.problem_map.clear_anchors()


def _sampling_units(weights: FloatArray, count: int) -> SamplingUnits:
    """Build :class:`SamplingUnits` from retained raw weights without ``FESample``."""
    active_mask = weights > 0
    active: IndexArray = np.frombuffer(
        np.flatnonzero(active_mask).tobytes(), dtype=np.intp
    )
    inactive: IndexArray = np.frombuffer(
        np.flatnonzero(~active_mask).tobytes(), dtype=np.intp
    )
    return SamplingUnits(active, inactive, count)


def _compute_influence_from_snapshot(
    cert: RetainedCertificateSnapshot | _OperationalCertificate, slope_map: FloatArray
) -> FloatArray:
    """Compute ``u_i = b_i (B_i W)`` from retained geometry before discard."""
    geom = cert.geometry
    assert geom is not None
    if type(cert) is _OperationalCertificate and type(geom) is _GeometryData:
        k = geom.design.shape[1]
        basis = (
            cert.basis.numerical_basis()
            if isinstance(cert.basis, _OperationalBasisSnapshot) else cert.basis.basis
        )
        assert cert.rows is not None
        nuisance = lift(basis, embed(basis, slope_map[k:]))
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            rows: FloatArray = geom.design @ slope_map[:k] + nuisance
            found: FloatArray = cert.rows.b[:, None] * rows
        return owned_array(_finite_result(found, "influence row"))
    geom = cast("RetainedGeometrySnapshot", geom)
    k = geom.n_slopes
    basis = geom.basis.basis
    nuisance = lift(basis, embed(basis, slope_map[k:]))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        rows = geom.design @ slope_map[:k] + nuisance
        found = geom.rows.b[:, None] * rows
    return owned_array(_finite_result(found, "influence row"))


def check_cluster_request_rows(
    request: CovarianceRequest, sample: RetainedSampleSnapshot
) -> None:
    """Validate that a cluster ``CovarianceRequest`` is aligned to ``sample``."""
    if request.kind == "cluster":
        if (
            request.positions is None
            or not request.clusters
            or request.positions.shape != (sample.count,)
            or not np.array_equal(request.positions, sample.positions)
        ):
            raise ValueError("cluster request is not bound to the sample's rows")
        for _, codes in request.clusters:
            if codes.shape != (sample.count,):
                raise ValueError("cluster request is not bound to the sample's rows")


_check_cluster_request_rows = check_cluster_request_rows


def matches_initial_covariance_request(
    initial: RetainedInitialScaleCache | None,
    sample: RetainedSampleSnapshot,
    request: CovarianceRequest | None,
) -> bool:
    """Return True if ``request`` matches the cached initial covariance request."""
    if initial is None:
        return False
    if request is None:
        return True
    if type(request) is not CovarianceRequest:
        raise TypeError("request must be a CovarianceRequest or None")
    check_cluster_request_rows(request, sample)
    return request.identity() == initial.request.identity()


def _covariance_request_label(request: CovarianceRequest | None) -> str:
    """Human-readable label for a refused lower-level covariance request."""
    if request is None or request.kind == "robust":
        return "vcov('robust')"
    if request.kind == "hc":
        return f"vcov({request.method!r})"
    if len(request.clusters) == 1:
        return f"vcov(cluster={request.clusters[0][0]!r})"
    dims = ", ".join(repr(n) for n, _ in request.clusters)
    return f"vcov(cluster={dims})"


def _require_recompute_inputs(
    envelope: RetainedAbsorbedEnvelope,
    request: CovarianceRequest | None,
    request_label: str | None,
) -> tuple[CovarianceRequest, FloatArray, FloatArray]:
    """Validate non-initial covariance recomputation inputs."""
    if request is None:
        if envelope.empty:
            raise ValueError("the model has no slope parameters (K = 0)")
        raise ValueError("initial scale evidence was not formed for this candidate")
    label = (
        request_label
        if request_label is not None
        else _covariance_request_label(request)
    )
    envelope.require_influence(label)
    if envelope.empty:
        raise ValueError("the model has no slope parameters (K = 0)")
    obs = envelope.observed_influence
    if obs is None or envelope.order is None or envelope.selected_point is None:
        raise ValueError("observed influence rows were not formed for this candidate")
    check_cluster_request_rows(request, envelope.sample)
    return request, obs, envelope.sample.weight_values


def recompute_absorbed_scale(
    envelope: RetainedAbsorbedEnvelope,
    request: CovarianceRequest | None = None,
    *,
    request_label: str | None = None,
) -> RetainedInitialScaleCache:
    """Serve cached initial scale or recompute from retained observed influence rows."""
    if type(envelope) is not RetainedAbsorbedEnvelope:
        raise TypeError("envelope must be a RetainedAbsorbedEnvelope")
    if request is not None and type(request) is not CovarianceRequest:
        raise TypeError("request must be a CovarianceRequest or None")
    if matches_initial_covariance_request(
        envelope.initial_scale, envelope.sample, request
    ):
        assert envelope.initial_scale is not None
        return envelope.initial_scale
    req, obs, weights = _require_recompute_inputs(envelope, request, request_label)
    assert envelope.order is not None
    assert envelope.selected_point is not None
    units = _sampling_units(weights, envelope.sample.count)
    active = units.identifiers
    cov = fe_covariance(
        obs[active],
        weights[active],
        frequency=envelope.sample.frequency,
        request=req,
        units=units,
    )
    ses = direct_standard_errors(cov, envelope.order)
    rep = ReportedScale(
        gamma=envelope.selected_point.theta,
        standard_errors=ses,
        order=envelope.order,
    )
    rep_cov: FloatArray | None = None
    cov_err: str | None = None
    try:
        rep_cov = envelope.order.covariance(cov.covariance)
    except FloatingPointError as exc:
        cov_err = str(exc)
    head = (
        snapshot_covariance_request(req),
        envelope.order,
        bool((ses == 0.0).any()),
    )
    return RetainedInitialScaleCache(*head, cov, rep, rep_cov, cov_err, obs)


def compose_absorbed_envelope(
    admitted: AdmittedAbsorbedGraph,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    outcome_snapshot: RetainedOutcomeSnapshot | None = None,
    trigger_snapshot: RetainedTriggerSnapshot | None = None,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
    keep_influence: bool = True,
) -> RetainedAbsorbedEnvelope:
    """Compose a :class:`RetainedAbsorbedEnvelope` from admitted lower components.

    ``trigger_snapshot`` is the admitted snapshot of the fit's trigger record: for a
    selected-root fit, whose root stages call no trigger, that of its convex
    initializer, kept with the initializer's own history in the fit's record
    (:func:`~powermean.results.absorbed_root.snapshot_root`), which is detached
    after the history and the certificate.
    """
    candidate = admitted.audit.history.candidate
    root = isinstance(candidate, AbsorbedRootCandidate)
    cert = admitted.audit.certificate
    capture_history = (
        _capture_profiled_history
        if type(candidate) is ProfiledCandidate and cert is not None
        and _operational_data(cert.basis) is not None
        else snapshot_candidate_history
    )
    history_snap = capture_history(
        admitted.audit.history,
        sample_snapshot=sample_snapshot,
        trigger_snapshot=None if root else trigger_snapshot,
        point_map=point_map,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    cert_snap = (
        None
        if cert is None
        else _capture_certificate(
            cert,
            sample_snapshot=sample_snapshot,
            selected_point=history_snap.selected_point,
            sample_map=sample_map,
            grouping_map=grouping_map,
            keep_slope_map=keep_influence,
        )
    )
    raw_scale: FEScaleResult | None = (
        None
        if cert is None
        else cast("FEScaleResult | None", getattr(cert, "_scale", None))
    )
    initial_scale: RetainedInitialScaleCache | None = None
    obs_inf: FloatArray | None = None
    w_map: FloatArray | None = None
    if raw_scale is not None:
        initial_scale = snapshot_initial_scale(raw_scale, keep_influence=keep_influence)
        if keep_influence:
            obs_inf = raw_scale.evidence.influence
            w_map = raw_scale.input.slope_map
    elif (
        keep_influence
        and cert_snap is not None
        and cert_snap.geometry is not None
        and cert_snap.measures is not None
        and cert_snap.measures.slope_map is not None
        and cert_snap.point.n_slopes > 0
    ):
        w_map = cert_snap.measures.slope_map
        obs_inf = _compute_influence_from_snapshot(cert_snap, w_map)
    root_snap: RetainedRootSnapshot | None = None
    if isinstance(candidate, AbsorbedRootCandidate):
        root_snap = snapshot_root(
            candidate,
            sample_snapshot=sample_snapshot,
            trigger_snapshot=trigger_snapshot,
            point_map=point_map,
            sample_map=sample_map,
            grouping_map=grouping_map,
        )
    return RetainedAbsorbedEnvelope(
        history_snap.route,
        keep_influence,
        admitted.receipt,
        sample_snapshot,
        outcome_snapshot,
        cast("RetainedHistorySnapshot", history_snap),
        history_snap.selected_point,
        cast("RetainedCertificateSnapshot | None", cert_snap),
        owned_order_scale(admitted.sample.order_scale),
        initial_scale,
        obs_inf,
        w_map,
        root_snap,
    )


def _profiled_attempt_data(
    attempt: ProfiledAttempt, get_point: PointMapper,
    sample: RetainedSampleSnapshot, grouping_map: GroupingIdentityMap | None,
) -> _ProfiledAttemptData:
    """Capture explicit profiled attempt values without its presentation graph."""
    final = attempt.final
    detached = None
    if final is not None:
        equations = final.equations
        values = (None if equations is None else (
            float(equations.loss), owned_array(equations.gradient),
            owned_array(equations.curvature), owned_array(equations.information),
        ))
        absorbed = _owned_absorbed_schur(
            final.absorbed, grouping_snapshot=sample._final_grouping,
            grouping_map=grouping_map,
        )
        point = None if final.point is None else get_point(final.point)
        detached = _ProfiledFinalData(
            point, values, absorbed, _owned_alarm(final.alarm), str(final.reason),
        )
    opening = None if attempt.opening is None else get_point(attempt.opening)
    point = None if attempt.point is None else get_point(attempt.point)
    loss = None if attempt.loss is None else float(attempt.loss)
    return _ProfiledAttemptData(
        _owned_iteration(attempt.iteration), attempt.reason, str(attempt.message),
        opening, point, loss, detached, _owned_alarm(attempt.alarm), attempt.assessment,
    )


def _capture_profiled_history(
    history: CandidateHistory,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    trigger_snapshot: RetainedTriggerSnapshot | None = None,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> _OperationalHistory:
    """Detach the finite native history with unchanged point-key capture order."""
    if history.trigger is not None and trigger_snapshot is None:
        raise ValueError(
            "history.trigger requires a lowered trigger_snapshot from admission"
        )
    candidate = history.candidate
    assert type(candidate) is ProfiledCandidate
    inner, start = _candidate_inner_and_start(
        candidate, history.route, grouping_snapshot=sample_snapshot._final_grouping,
        grouping_map=grouping_map,
    )
    assert start is None or isinstance(start, Start)
    point = None
    if history.point is not None:
        point = (
            snapshot_point_record(
                history.point, sample_snapshot=sample_snapshot,
                variance=history.variance, inner=inner, sample_map=sample_map,
                grouping_map=grouping_map,
            ) if point_map is None else point_map.get(
                history.point, sample_snapshot=sample_snapshot,
                variance=history.variance, inner=inner,
            )
        )
    get_point = _point_mapper(
        sample_snapshot, history.variance, inner, history.point, point,
        point_map=point_map, sample_map=sample_map, grouping_map=grouping_map,
    )
    attempts = tuple(
        _profiled_attempt_data(a, get_point, sample_snapshot, grouping_map)
        for a in cast("tuple[ProfiledAttempt, ...]", history.attempts)
    )
    return _OperationalHistory(
        history.route, sample_snapshot, history.variance, history.retained, attempts,
        str(history.reason), str(history.message),
        None if history.native is None else str(history.native),
        tuple(str(p) for p in history.partials), tuple(str(u) for u in history.unreached),
        _owned_alarm(history.alarm), trigger_snapshot,
        cast("SolveWork", history.work), history.timing,
        point, start, candidate.provider, candidate.provider_version,
        candidate.initial_assessment, candidate.no_measurable_progress,
        candidate.assessments, candidate.assessment_timing,
        candidate.options, candidate.absorbed,
    )


def retain_absorbed_envelope(
    audit: CandidateAudit,
    *,
    prepared: PreparedFrame | None = None,
    input_snapshot: RetainedInputSnapshot | None = None,
    outcome_snapshot: RetainedOutcomeSnapshot | None = None,
    trigger_snapshot: RetainedTriggerSnapshot | None = None,
    request: CovarianceRequest | None = None,
    keep_influence: bool = True,
) -> RetainedAbsorbedEnvelope:
    """Admit a lower-layer ``CandidateAudit`` and detach its envelope."""
    admitted = admit_absorbed_graph(
        audit,
        prepared=prepared,
        request=request,
        keep_influence=keep_influence,
        same_preparation=prepared is not None or input_snapshot is not None,
        same_outcome_sample=outcome_snapshot is not None,
    )
    maps = CaptureMaps()
    in_snap = (
        maps.input_map.register_snapshot(
            input_snapshot, base=admitted.sample.base, prepared=prepared
        )
        if input_snapshot is not None
        else (None if prepared is None else maps.input_map.register(prepared))
    )
    sample_snap = maps.sample_map.register(
        admitted.sample, input_snapshot=in_snap, partition_map=maps.partition_map
    )
    try:
        return compose_absorbed_envelope(
            admitted,
            sample_snapshot=sample_snap,
            outcome_snapshot=outcome_snapshot,
            trigger_snapshot=trigger_snapshot,
            point_map=maps.point_map,
            sample_map=maps.sample_map,
            grouping_map=maps.grouping_map,
            keep_influence=keep_influence,
        )
    finally:
        maps.clear_anchors()


__all__ = [
    "CaptureMaps",
    "_check_cluster_request_rows",
    "_compute_influence_from_snapshot",
    "_sampling_units",
    "check_cluster_request_rows",
    "compose_absorbed_envelope",
    "matches_initial_covariance_request",
    "recompute_absorbed_scale",
    "retain_absorbed_envelope",
]
