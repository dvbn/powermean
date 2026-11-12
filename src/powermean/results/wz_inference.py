"""Normal retained WZ recomputation, using original owned sources.

Covariance changes read the uncorrected retained state and never invoke optimization.
The existing fit-time initial cache remains usable after influence discard; a different
request truthfully requires retained state.
"""

from __future__ import annotations

import numpy as np

from powermean.inference.absorbed_details import typed_details
from powermean.inference.absorbed_rank import absorbed_count
from powermean.inference.ssc import convention_counts
from powermean.inference.wz_binding import (
    WZSource,
    capture_rows,
    layout,
    reserve_validation,
    unavailable,
)
from powermean.inference.wz_covariance import (
    WZControls,
    bind_request,
    convention,
    ppml,
    reported,
    retained_covariance_record,
)
from powermean.inference.wz_operator import evaluate
from powermean.inference.wz_resources import joint_bounds
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.cancelled_inference import _same_request, request_convention
from powermean.results.cancelled_retained import RetainedCancelledEnvelope
from powermean.results.retained_inference import InferenceRequest, RetainedInference
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.options import SolverOptions


def joint_wz(
    envelope: RetainedAbsorbedEnvelope,
    request: InferenceRequest,
    terms: tuple[tuple[str, ...], ...],
    absorbed: AbsorbedOptions,
    options: SolverOptions,
    *,
    max_dense_bytes: int | None = None,
) -> RetainedInference:
    """Read the retained uncorrected joint point; never recover effects or refit.

    The normal result owns the original formula terms; the envelope owns the
    current rows, grouping, numerical point and variance. No verified marker or
    prior WZ success supplies source admission. Initial requests are served by
    the normal result's exact cache path before entering this recomputation.
    Convention counts and the operator share the freshly admitted actual P.
    """
    if request.adjustment != "wz" or len(request.clusters) != 1:
        raise unavailable("one pair cluster and adjustment='wz' required")
    if request.bread != "observed":
        raise unavailable("observed PPML bread required")
    sample, point, certificate = (
        envelope.sample,
        envelope.selected_point,
        envelope.certificate,
    )
    if point is None or certificate is None or certificate.geometry is None:
        raise unavailable("the current retained joint point and rank are unavailable")
    envelope.require_influence(request.label)
    bounds = joint_bounds(max_dense_bytes)
    reserve_validation((sample.count, sample.n_slopes), bounds)
    ppml(sample.omega, point.variance, weights=sample.data.weights is not None)
    group, design, order = sample.grouping, sample.design, envelope.order
    if group is None or design is None or order is None:
        raise unavailable("current original grouping, slopes and report map required")
    if (
        sample.positive_only
        or sample.conditioned
        or any(v is not None for v in sample.reasons)
    ):
        raise unavailable(
            "unconditioned original three-way design without omitted slopes required"
        )
    source = WZSource(terms, group, sample.data.row_labels)
    cluster = request.clusters[0]
    panel = layout(source, cluster.name, design.matrix.shape, absorbed, bounds)
    bind_request(source, cluster)
    initial = envelope.initial_scale
    details = None if initial is None else initial.details
    inherited = None if details is None else typed_details(details).applied.convention
    counts_absorbed = absorbed_count(
        tuple((block.name, block.codes) for block in group.blocks), [cluster.codes]
    )
    counts = convention_counts(
        sample.count,
        sample.n_slopes,
        certificate.geometry.dimension,
        counts_absorbed,
        [panel.pairs],
    )
    applied = convention(request.ssc, counts, inherited=inherited)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        mu = np.exp(point.q)
    rows = capture_rows(
        panel,
        design.matrix,
        sample.data.y,
        mu,
        sample.active_names,
        route="joint_retained_current",
    )
    result = evaluate(
        rows,
        options,
        bounds,
        ssc=applied.convention.preset or "custom",
        plan=panel.plan,
    )
    errors, matrix, failure = reported(result, order)
    return RetainedInference(
        request,
        applied,
        counts_absorbed,
        "observed",
        retained_covariance_record(result, applied),
        result.adjusted,
        errors,
        matrix,
        failure,
        point.theta,
        False,
        sample.count,
        sample.count,
        sample.count,
        wz=result,
    )


def cancelled_wz(
    envelope: RetainedCancelledEnvelope,
    request: InferenceRequest,
    source: WZSource,
    absorbed: AbsorbedOptions,
) -> RetainedInference:
    """Answer the settled WZ request from the retained current uncorrected point."""
    ppml(envelope.order.omega, envelope.variance)
    if request.adjustment != "wz" or len(request.clusters) != 1:
        raise unavailable("one pair cluster and adjustment='wz' required")
    if request.clusters[0].name != envelope.cancelled_name:
        raise unavailable("only actual pair cancellation is supported")
    initial = envelope.initial_scale
    inherited = None if initial is None else initial.applied.convention
    applied = request_convention(envelope.geometry, request, inherited=inherited)
    if (
        initial is not None
        and _same_request(envelope.request, request)
        and request.ssc == envelope.request.ssc
    ):
        return initial
    envelope.require_influence(request.label)
    capture, state = envelope._capture, envelope._state
    if capture is None or state is None:
        raise ValueError("the current uncorrected cancelled state was not retained")
    controls = WZControls(
        source, absorbed, envelope.options, envelope._inference_bounds
    )
    result = controls.cancelled(
        state, capture, request.clusters[0], ssc=applied.convention.preset or "custom"
    )
    errors, matrix, failure = reported(result, envelope.order)
    return RetainedInference(
        request,
        applied,
        None,
        "observed",
        retained_covariance_record(result, applied),
        result.adjusted,
        errors,
        matrix,
        failure,
        state.theta,
        False,
        envelope.count,
        envelope.n_rows,
        envelope.n_rows,
        wz=result,
    )
