"""Post-fit covariance request normalization and initial-cache matching helpers.

This module normalizes post-fit covariance specifications
(:data:`PostFitCovarianceSpec`)
into :class:`~powermean.solve.absorbed_certificate_scale.CovarianceRequest` records and
evaluates initial-cache or recomputed scale results on a
:class:`~powermean.results.absorbed_retained.RetainedAbsorbedEnvelope`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from powermean import messages
from powermean.data.inference_requests import ClusterRequest
from powermean.errors import SpecificationError
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_time import _is_missing_metadata
from powermean.results.absorbed_handoff import (
    check_cluster_request_rows,
)
from powermean.results.absorbed_handoff import (
    recompute_absorbed_scale as _lower_recompute_absorbed_scale,
)
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.absorbed_scale_cache import RetainedInitialScaleCache
from powermean.solve.absorbed_certificate_scale import CovarianceRequest

PostFitCovarianceSpec = (
    CovarianceRequest | ClusterRequest | Literal["robust"] | Mapping[str, str] | None
)

__all__ = [
    "PostFitCovarianceSpec",
    "_can_match_initial_cache",
    "_encode_sample_cluster",
    "_request_label",
    "normalize_postfit_covariance_request",
    "recompute_absorbed_scale_for_envelope",
]


def _encode_sample_cluster_codes(
    envelope: RetainedAbsorbedEnvelope,
    dependence: BoundDependence | None,
    name: str,
) -> np.ndarray[Any, np.dtype[np.integer[Any]]]:
    """Encode first-appearance cluster codes on ``envelope.sample``."""
    if dependence is not None:
        return dependence.cluster_codes(name).codes
    if name not in envelope.sample.metadata:
        available = (
            messages.format_keys(envelope.sample.metadata)
            if envelope.sample.metadata
            else "none"
        )
        raise SpecificationError(
            "specification.cluster_not_retained",
            {"name": name, "retained": available},
        )
    raw_vals = envelope.sample.metadata[name]
    missing_count = sum(1 for val in raw_vals.tolist() if _is_missing_metadata(val))
    if missing_count > 0:
        raise SpecificationError(
            "specification.cluster_missing_values",
            {"name": name, "count": missing_count},
        )
    labels: dict[Any, int] = {}
    codes = np.empty(envelope.sample.count, dtype=np.intp)
    for pos, val in enumerate(raw_vals.tolist()):
        codes[pos] = labels.setdefault(val, len(labels))
    return codes


def _encode_sample_cluster(
    envelope: RetainedAbsorbedEnvelope,
    dependence: BoundDependence | None,
    name: str,
) -> CovarianceRequest:
    """Encode first-appearance cluster codes for ``name`` on ``envelope.sample``."""
    codes = _encode_sample_cluster_codes(envelope, dependence, name)
    return CovarianceRequest("cluster", name, codes, envelope.sample.positions)


def normalize_postfit_covariance_request(
    envelope: RetainedAbsorbedEnvelope,
    dependence: BoundDependence | None = None,
    request: PostFitCovarianceSpec = None,
) -> CovarianceRequest | None:
    """Normalize a post-fit covariance specification to :class:`CovarianceRequest`."""
    if request is None:
        return None
    if request == "robust":
        return CovarianceRequest("robust")
    if type(request) is CovarianceRequest:
        check_cluster_request_rows(request, envelope.sample)
        return request
    if type(request) is ClusterRequest:
        if (
            len(request.dimensions) == 1
            and request.multiway == CovarianceRequest("robust").multiway
            and request.psd == CovarianceRequest("robust").psd
        ):
            return _encode_sample_cluster(envelope, dependence, request.dimensions[0])
        clusters = tuple(
            (dim, _encode_sample_cluster_codes(envelope, dependence, dim))
            for dim in request.dimensions
        )
        return CovarianceRequest(
            kind="cluster",
            name=clusters[0][0] if len(clusters) == 1 else None,
            codes=clusters[0][1] if len(clusters) == 1 else None,
            positions=envelope.sample.positions,
            clusters=clusters,
            multiway=request.multiway,
            psd=request.psd,
        )
    if isinstance(request, Mapping):
        if set(request) != {"cluster"} or not isinstance(request.get("cluster"), str):
            raise ValueError(
                "vcov mapping must be {'cluster': <column_name>} with a string column"
            )
        return _encode_sample_cluster(envelope, dependence, request["cluster"])
    raise TypeError(f"unsupported post-fit covariance request {request!r}")


def _request_label(request: PostFitCovarianceSpec) -> str:
    """Human-readable label for a refused post-fit covariance request."""
    if request is None or request == "robust":
        return "vcov('robust')"
    if type(request) is CovarianceRequest:
        if request.kind == "robust":
            return "vcov('robust')"
        if request.kind == "hc":
            return f"vcov({request.method!r})"
        if len(request.clusters) == 1:
            return f"vcov(cluster={request.clusters[0][0]!r})"
        dims = ", ".join(repr(n) for n, _ in request.clusters)
        return f"vcov(cluster={dims})"
    if type(request) is ClusterRequest:
        dims = ", ".join(repr(d) for d in request.dimensions)
        return f"vcov(cluster={dims})"
    if isinstance(request, Mapping) and "cluster" in request:
        return f"vcov(cluster={request['cluster']!r})"
    return "vcov(...)"


def _can_match_initial_cache(
    envelope: RetainedAbsorbedEnvelope, request: PostFitCovarianceSpec
) -> bool:
    """Return True if ``request`` could match ``envelope.initial_scale``."""
    init = envelope.initial_scale
    if init is None:
        return False
    if request is None:
        return True
    init_req = init.request.request
    if request == "robust":
        return init_req.is_default_stata and init.request.kind == "robust"
    if type(request) is CovarianceRequest:
        return (
            init_req.kind == request.kind
            and init_req.bread == request.bread
            and init_req.method == request.method
            and tuple(n for n, _ in init_req.clusters)
            == tuple(n for n, _ in request.clusters)
            and init_req.multiway == request.multiway
            and init_req.psd == request.psd
            and init_req.ssc == request.ssc
        )
    if type(request) is ClusterRequest:
        return (
            init_req.kind == "cluster"
            and init_req.bread == "observed"
            and tuple(n for n, _ in init_req.clusters) == request.dimensions
            and init_req.multiway == request.multiway
            and init_req.psd == request.psd
            and init_req.ssc == CovarianceRequest("robust").ssc
        )
    if isinstance(request, Mapping) and set(request) == {"cluster"}:
        return (
            init_req.is_default_stata
            and init.request.kind == "cluster"
            and init.request.name == request["cluster"]
        )
    return False


def recompute_absorbed_scale_for_envelope(
    envelope: RetainedAbsorbedEnvelope,
    dependence: BoundDependence | None = None,
    request: PostFitCovarianceSpec = None,
) -> RetainedInitialScaleCache:
    """Serve cached initial scale or recompute post-fit covariance for ``envelope``."""
    if request is None:
        return _lower_recompute_absorbed_scale(envelope, None)
    label = _request_label(request)
    if not _can_match_initial_cache(envelope, request):
        envelope.require_influence(label)
    cov_req = normalize_postfit_covariance_request(envelope, dependence, request)
    return _lower_recompute_absorbed_scale(envelope, cov_req, request_label=label)
