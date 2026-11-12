"""Interface-layer post-fit covariance request normalization and absorbed result facade.

This module coordinates absorbed fit retention and re-exports post-fit covariance
normalization and result facade classes from
:mod:`powermean.interface.retained_covariance` and
:mod:`powermean.interface.retained_result`.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean._prepare.compose import PreparedFrame
from powermean.data.specs import Estimator
from powermean.data.weights import ObservationWeights
from powermean.interface.retained_admission import (
    admit_interface_graph,
    check_dependence_matches_sample,
)
from powermean.interface.retained_binding import SampleTarget
from powermean.interface.retained_context import PerFitPreparationContext
from powermean.interface.retained_covariance import (
    PostFitCovarianceSpec,
    _can_match_initial_cache,
    _encode_sample_cluster,
    _request_label,
    normalize_postfit_covariance_request,
    recompute_absorbed_scale_for_envelope,
)
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_preparation import RetainedPreparation
from powermean.interface.retained_result import RetainedAbsorbedResult
from powermean.interface.retention import bind_sample_metadata
from powermean.interface.sample import SampleOutcome
from powermean.results.absorbed_handoff import compose_absorbed_envelope
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.absorbed_scale_cache import RetainedInitialScaleCache
from powermean.solve.absorbed_certificate import CandidateAudit
from powermean.solve.absorbed_certificate_scale import CovarianceRequest

__all__ = [
    "AbsorbedHandoff",
    "PostFitCovarianceSpec",
    "RetainedAbsorbedResult",
    "_can_match_initial_cache",
    "_encode_sample_cluster",
    "_request_label",
    "bind_absorbed",
    "build_absorbed_handoff",
    "check_dependence_matches_sample",
    "normalize_postfit_covariance_request",
    "recompute_absorbed_scale",
    "retain_absorbed",
    "retain_absorbed_fit",
    "retain_prepared_dependence",
]


@dataclass(frozen=True)
class AbsorbedHandoff:
    """Typed original handoff package for an absorbed FE fit."""

    audit: CandidateAudit
    preparation: (
        RetainedPreparation | PerFitPreparationContext | PreparedFrame | None
    ) = None
    outcome: SampleOutcome | None = None
    dependence: BoundDependence | None = None
    estimator: Estimator | None = None
    request: CovarianceRequest | None = None
    keep_influence: bool = True


def _unwrap_envelope_and_dependence(
    target: RetainedAbsorbedResult | RetainedAbsorbedEnvelope,
) -> tuple[RetainedAbsorbedEnvelope, BoundDependence | None]:
    """Extract ``(envelope, dependence)`` from a result facade or lower envelope."""
    if type(target) is RetainedAbsorbedResult:
        return target.envelope, target.dependence
    if type(target) is RetainedAbsorbedEnvelope:
        return target, None
    raise TypeError(
        "target must be a RetainedAbsorbedResult or RetainedAbsorbedEnvelope"
    )


def recompute_absorbed_scale(
    target: RetainedAbsorbedResult | RetainedAbsorbedEnvelope,
    request: PostFitCovarianceSpec = None,
) -> RetainedInitialScaleCache:
    """Serve cached initial scale or recompute post-fit covariance."""
    envelope, dependence = _unwrap_envelope_and_dependence(target)
    return recompute_absorbed_scale_for_envelope(envelope, dependence, request)


def _unwrap_preparation(
    preparation: object,
) -> PerFitPreparationContext | PreparedFrame | None:
    """Validate and unwrap ``RetainedPreparation`` to ``PerFitPreparationContext``."""
    if preparation is None:
        return None
    if type(preparation) is RetainedPreparation:
        return preparation._context
    if type(preparation) is PerFitPreparationContext:
        return preparation
    if type(preparation) is PreparedFrame:
        return preparation
    raise TypeError("preparation must be a RetainedPreparation or PreparedFrame")


def retain_absorbed(handoff: AbsorbedHandoff) -> RetainedAbsorbedResult:
    """Admit ``handoff``, detach all retained snapshots, and return the facade."""
    if type(handoff) is not AbsorbedHandoff:
        raise TypeError("handoff must be an AbsorbedHandoff")
    prep_arg = _unwrap_preparation(handoff.preparation)
    admitted = admit_interface_graph(
        handoff.audit,
        preparation=prep_arg,
        outcome=handoff.outcome,
        dependence=handoff.dependence,
        estimator=handoff.estimator,
        request=handoff.request,
        keep_influence=handoff.keep_influence,
    )
    try:
        envelope = compose_absorbed_envelope(
            admitted.admitted_graph,
            sample_snapshot=admitted.sample_snapshot,
            outcome_snapshot=admitted.outcome_snapshot,
            trigger_snapshot=admitted.trigger_snapshot,
            point_map=admitted.point_map,
            sample_map=admitted.sample_map,
            grouping_map=admitted.grouping_map,
            keep_influence=handoff.keep_influence,
        )
    finally:
        if admitted.point_map is not None:
            admitted.point_map.clear_anchors()
        if admitted.sample_map is not None:
            admitted.sample_map.clear_anchors()
        if admitted.grouping_map is not None:
            admitted.grouping_map.clear_anchors()
    return RetainedAbsorbedResult(envelope, dependence=admitted.dependence)


def bind_absorbed(
    audit: CandidateAudit,
    *,
    preparation: (
        RetainedPreparation | PerFitPreparationContext | PreparedFrame | None
    ) = None,
    outcome: SampleOutcome | None = None,
    dependence: BoundDependence | None = None,
    estimator: Estimator | None = None,
    request: CovarianceRequest | None = None,
    keep_influence: bool = True,
) -> RetainedAbsorbedResult:
    """Convenience entry point wrapping :func:`retain_absorbed`."""
    return retain_absorbed(
        AbsorbedHandoff(
            audit=audit,
            preparation=preparation,
            outcome=outcome,
            dependence=dependence,
            estimator=estimator,
            request=request,
            keep_influence=keep_influence,
        )
    )


def retain_prepared_dependence(
    preparation: RetainedPreparation,
    sample: SampleTarget,
    *,
    weights: ObservationWeights | None = None,
) -> BoundDependence:
    """Bind retained dependence metadata from ``preparation`` to ``sample``."""
    if type(preparation) is not RetainedPreparation:
        raise TypeError("preparation must be a RetainedPreparation")
    return bind_sample_metadata(preparation, sample, weights=weights)


def build_absorbed_handoff(
    audit: CandidateAudit,
    *,
    preparation: RetainedPreparation | None = None,
    outcome: SampleOutcome | None = None,
    dependence: BoundDependence | None = None,
    estimator: Estimator | None = None,
    request: CovarianceRequest | None = None,
    keep_influence: bool = True,
) -> AbsorbedHandoff:
    """Package an :class:`AbsorbedHandoff`, binding ``dependence`` if needed."""
    resolved_dep = dependence
    if resolved_dep is None and preparation is not None:
        if type(preparation) is not RetainedPreparation:
            raise TypeError("preparation must be a RetainedPreparation")
        if type(audit) is not CandidateAudit:
            raise TypeError("audit must be a CandidateAudit")
        sample = audit.history.sample
        resolved_dep = bind_sample_metadata(
            preparation, sample, weights=sample.data.weights
        )
    return AbsorbedHandoff(
        audit=audit,
        preparation=preparation,
        outcome=outcome,
        dependence=resolved_dep,
        estimator=estimator,
        request=request,
        keep_influence=keep_influence,
    )


def retain_absorbed_fit(
    audit: CandidateAudit,
    *,
    preparation: RetainedPreparation | None = None,
    outcome: SampleOutcome | None = None,
    dependence: BoundDependence | None = None,
    estimator: Estimator | None = None,
    request: CovarianceRequest | None = None,
    keep_influence: bool = True,
) -> RetainedAbsorbedResult:
    """Build an :class:`AbsorbedHandoff` and retain its result facade."""
    return retain_absorbed(
        build_absorbed_handoff(
            audit,
            preparation=preparation,
            outcome=outcome,
            dependence=dependence,
            estimator=estimator,
            request=request,
            keep_influence=keep_influence,
        )
    )
