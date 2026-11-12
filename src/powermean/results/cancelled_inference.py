"""Typed cancelled retained inference adapter; no refit or structural replay."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from powermean.errors import SpecificationError
from powermean.fe.admission_records import WorkLog
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.groups import encode_block
from powermean.inference.cancelled import cancelled_covariance, reserve_inference
from powermean.inference.convention import ClusterCodes
from powermean.inference.ssc import (
    STATA,
    AppliedConvention,
    ConventionCounts,
    ResolvedConvention,
    apply_convention,
    resolve_convention,
)
from powermean.inference.wz_covariance import convention as wz_convention
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.covariance import _missing

if TYPE_CHECKING:
    from powermean.results.cancelled_retained import RetainedCancelledEnvelope
    from powermean.results.retained_inference import InferenceRequest, RetainedInference
from powermean.systems.identified import IdentifiedState


def unavailable(method: str) -> SpecificationError:
    """Truthful first-release refusal, leaving the scientific obligation OPEN."""
    return SpecificationError(
        "specification.unsupported_inference",
        {
            "method": method,
            "case": "cancelled group equations in this private increment",
        },
    )


def bind_cluster(
    envelope: RetainedCancelledEnvelope,
    name: str,
    labels: ArrayLike,
    positions: ArrayLike,
) -> ClusterCodes:
    """Bind new labels to all frozen positional rows; never drop a missing row."""
    if not np.array_equal(positions, envelope.geometry.structure.positions):
        raise ValueError("post-fit cluster positions do not match the fitted rows")
    values = np.asarray(labels, dtype=object)
    if values.shape != (envelope.n_rows,):
        raise ValueError("post-fit cluster labels must align with every fitted row")
    missing = sum(_missing(item) for item in values.tolist())
    if missing:
        raise SpecificationError(
            "specification.cluster_missing_values", {"name": name, "count": missing}
        )
    return ClusterCodes(name, encode_block(values, name=name).codes)


def request_convention(
    geometry: NumericalGeometry,
    request: InferenceRequest,
    *,
    inherited: ResolvedConvention | None = None,
) -> AppliedConvention:
    """Validate the bounded observed/Stata/robust-or-one-way domain before solving.

    N is the cancelled sampling-unit count. Rank and K_reference here describe all
    identified estimated coordinates, including retained effects; cancelled scales
    are not estimated. Only Stata values are admitted, so no regression/HC1 dof
    assumption, nesting adjustment or native-standardization frame is introduced.
    """
    if request.bread != "observed" or request.multiway_path:
        raise unavailable(request.label)
    resolved = resolve_convention(
        request.ssc, request.inherited if request.inherited is not None else inherited
    )
    if request.adjustment != "wz" and resolved.values != STATA:
        raise unavailable(request.label)
    n = geometry.structure.n_rows
    for dimension in request.clusters:
        if dimension.codes.shape != (n,) or np.any(dimension.codes < 0):
            raise ValueError(
                "cluster codes must align with every original positional row"
            )
    groups = tuple(int(np.unique(c.codes).size) for c in request.clusters)
    counts = ConventionCounts(
        geometry.structure.n_groups,
        geometry.n_slopes_identified,
        geometry.n_identified,
        geometry.n_identified,
        geometry.n_identified,
        0,
        groups,
    )
    if request.adjustment == "wz":
        return wz_convention(request.ssc, counts, inherited=inherited)
    return apply_convention(resolved, counts, kind=request.kind)


def same_point_inference(
    state: IdentifiedState,
    rhs: FloatArray,
    order: OrderScale,
    request: InferenceRequest,
    applied: AppliedConvention,
    *,
    work: WorkLog,
    record: type[RetainedInference],
) -> RetainedInference:
    """Adapt actual cancelled sum-score inference to the common retained record."""
    cluster = None if not request.clusters else request.clusters[0]
    covariance, errors, reported, failure = cancelled_covariance(
        state, rhs, order, applied, cluster, work=work
    )
    return record(
        copy.deepcopy(request),
        applied,
        None,
        "observed",
        covariance,
        covariance.covariance,
        errors,
        reported,
        failure,
        state.theta,
        False,
        state.design.structure.n_groups,
        state.design.structure.n_rows,
        state.design.structure.n_rows,
        work=work.snapshot(),
    )


def _same_request(left: InferenceRequest, right: InferenceRequest) -> bool:
    return (
        left.kind == right.kind
        and left.adjustment == right.adjustment
        and left.bread == right.bread
        and len(left.clusters) == len(right.clusters)
        and all(
            a.name == b.name and np.array_equal(a.codes, b.codes)
            for a, b in zip(left.clusters, right.clusters, strict=True)
        )
    )


def retained_cancelled_inference(
    envelope: RetainedCancelledEnvelope,
    request: InferenceRequest,
    record: type[RetainedInference],
) -> RetainedInference:
    """Use the initial cache or the retained same-point state and W, without refit."""
    inherited = (
        None if envelope._initial is None else envelope._initial.applied.convention
    )
    applied = request_convention(envelope.geometry, request, inherited=inherited)
    if not envelope.accepted:
        raise ValueError("inference is withheld for an uncertified cancelled fit")
    initial = envelope.initial_scale
    if initial is not None and _same_request(envelope.request, request):
        return replace(
            initial,
            request=copy.deepcopy(request),
            applied=applied,
            from_initial_cache=True,
            covariance=None,
            work=WorkLog().snapshot(),
        )
    envelope.require_influence(request.label)
    state, rhs = envelope._state, envelope._rhs
    if state is None or rhs is None:
        raise ValueError("retained same-point cancelled inputs are unavailable")
    reserve_inference(envelope.geometry, envelope._inference_bounds)
    return same_point_inference(
        state,
        rhs.view(),
        envelope.order,
        request,
        applied,
        work=WorkLog(),
        record=record,
    )
