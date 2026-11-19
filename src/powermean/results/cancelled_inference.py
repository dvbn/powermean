"""Typed cancelled retained inference adapter; no refit or structural replay."""

from __future__ import annotations

import copy
import importlib
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.request_values import given
from powermean.errors import SpecificationError
from powermean.fe.admission_records import WorkLog
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.groups import encode_block
from powermean.formula.validity import raw_invalid_count
from powermean.inference.cancelled import cancelled_covariance, reserve_inference
from powermean.inference.convention import ClusterCodes
from powermean.inference.ssc import (
    AppliedConvention,
    ConventionCounts,
    ResolvedConvention,
    apply_convention,
    resolve_convention,
)
from powermean.inference.wz_covariance import convention as wz_convention
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray

if TYPE_CHECKING:
    from powermean.results.cancelled_retained import RetainedCancelledEnvelope
    from powermean.results.retained_inference import InferenceRequest, RetainedInference
    from powermean.systems.admitted import CapturedFit

    RowInputs = tuple[CapturedFit | None, list[warnings.WarningMessage] | None]
from powermean.systems.identified import IdentifiedState

# The observation-level engine of a multiway request, loaded on request only.
ROW_ENGINE = "powermean.results.cancelled_rows"


def unavailable(method: str) -> SpecificationError:
    """Truthful first-release refusal, leaving the scientific obligation OPEN."""
    return SpecificationError(
        "specification.unsupported_inference",
        {
            "method": method,
            "case": "a fit with a cancelled fixed effect",
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
    missing = raw_invalid_count(labels if isinstance(labels, np.ndarray) else values)
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
    """Validate the observed-bread robust, one-way or multiway domain before solving.

    N is the cancelled sampling-unit count. Rank and K_reference here describe all
    identified estimated coordinates, including retained effects; cancelled scales
    are not estimated. Every convention takes these counts and no nesting
    adjustment; a factor or reference with undefined degrees of freedom is refused
    by the convention's owner. A multiway request is counted here and answered from
    the original-row terms by ``ROW_ENGINE``. A requested dimension with fewer than
    two clusters is refused by name, one-way or multiway, at the fit and after it,
    with the joint route's refusal.
    """
    if request.bread != "observed":
        raise unavailable(request.label)
    resolved = resolve_convention(
        request.ssc, request.inherited if request.inherited is not None else inherited
    )
    n = geometry.structure.n_rows
    for dimension in request.clusters:
        if dimension.codes.shape != (n,) or np.any(dimension.codes < 0):
            raise ValueError(
                "cluster codes must align with every original positional row"
            )
    groups = tuple(int(np.unique(c.codes).size) for c in request.clusters)
    for dimension, count in zip(request.clusters, groups, strict=True):
        if count < 2:
            raise SpecificationError(
                "specification.invalid_value",
                {
                    "argument": "cluster",
                    "value": dimension.name,
                    "valid": "at least two clusters on the estimation sample",
                },
            )
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
    rows: RowInputs = (None, None),
) -> RetainedInference:
    """Adapt actual cancelled sum-score inference to the common retained record.

    A multiway request is composed from the same original-row terms by the common
    convention engine (``ROW_ENGINE``).  ``rows`` holds what it reads beside the
    point: the capture, for its frame columns, and, from a fit only, the list that
    receives the notes of this candidate in place of their issue, so that the fit
    issues those of its reported point once.
    """
    if request.multiway_path:
        engine = importlib.import_module(ROW_ENGINE)
        capture, held = rows
        return cast(
            "RetainedInference",
            engine.multiway_inference(
                state,
                rhs,
                order,
                request,
                applied,
                work=work,
                capture=capture,
                held=held,
            ),
        )
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


def _same_scale(
    initial: RetainedInference, request: InferenceRequest, applied: AppliedConvention
) -> bool:
    """Whether the cached scale is this request's: form, convention, D4 treatment."""
    cached = initial.request
    if cached.multiway_path != request.multiway_path or cached.form != request.form:
        return False
    if initial.applied.convention.values != applied.convention.values:
        return False
    if not request.signed:
        return True
    psd = request.psd if given(request.psd) else applied.psd
    return initial.psd == psd and (
        psd != "cgm" or initial.applied.sequence == applied.sequence
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
    if (
        initial is not None
        and _same_request(envelope.request, request)
        and _same_scale(initial, request, applied)
    ):
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
        rows=(envelope._capture, None),
    )
