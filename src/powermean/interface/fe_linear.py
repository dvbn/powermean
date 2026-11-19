"""Fixed-effect log-OLS (``omega = 0``) by plain absorption, without certificate.

Owner's ruling 52: least squares of ``log y`` on the regressors with the effects
absorbed is computed as ``reghdfe``, ``fixest::feols`` and ``pyfixest.feols``
compute it: the sample step (positive outcome, singletons, regressors collinear with
the effects), the absorbed solve of :mod:`powermean.solve.linear_absorbed` with its
own convergence, rank and normalization decisions, and the count of the absorbed
parameters.  No exact basis is constructed and no exact certificate is run.  The fit
is accepted when the linear solve ends ``"complete"`` and, with slopes, its
covariance is formed; any other end of the solve is the fit's named failure.

The covariance.  Slope influence rows of the within regression,
``u_i = (Xt'A Xt)^-1 xt_i a_i e_i`` (:func:`powermean.inference.absorbed_linear.
within_influence`); robust and clustered requests through the retained service
(:func:`powermean.results.retained_inference.retained_inference`) with the exact
count of ruling 5; the seven named HC methods with the leverage of the dummy-
variable regression (rulings 55 and 63; :func:`powermean.inference.absorbed_linear.
linear_composition`).  The record of the request is the one a joint fit retains, so
that post-fit requests, summaries and provenance read it unchanged.
"""

from __future__ import annotations

import numpy as np

from powermean.data.inference_conventions import ConventionValues
from powermean.data.sample_fe_state import FESample
from powermean.data.specs import Estimator
from powermean.errors import SpecificationError
from powermean.inference.absorbed_block_hc import BlockPreflight
from powermean.inference.absorbed_linear import (
    Coordinates,
    exact_coordinates,
    linear_composition,
    linear_preflight,
    remember_coordinates,
    within_influence,
)
from powermean.inference.absorbed_preflight import (
    AbsorbedFrameRecord,
    _active_request,
    _psd_request,
)
from powermean.inference.convention import ClusterCodes, Covariance
from powermean.inference.ssc import direct_errors, resolve_convention
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_postfit import retain_absorbed_fit
from powermean.interface.retained_preparation import RetainedPreparation
from powermean.interface.retained_result import RetainedAbsorbedResult
from powermean.interface.sample import SampleOutcome
from powermean.model.variance import FloatArray
from powermean.results.absorbed_retained import (
    RetainedAbsorbedEnvelope,
    RetainedInitialScaleCache,
    snapshot_covariance_request,
)
from powermean.results.retained_hc import RetainedHC, initial_hc_from_envelope
from powermean.results.retained_inference import (
    InferenceRequest,
    RetainedInference,
    retained_inference,
)
from powermean.solve import linear_absorbed
from powermean.solve.absorbed_certificate import CandidateAudit
from powermean.solve.absorbed_certificate_history import linear_history
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleDetails,
)
from powermean.solve.assessment import certificate_controls
from powermean.solve.linear_absorbed_state import AbsorbedLinearCandidate
from powermean.solve.m1_options import AbsorbedOptions
from powermean.solve.options import SolverOptions
from powermean.solve.reported import ReportedScale, saturated
from powermean.systems.joint import SamplingUnits


def linear_request(
    sample: FESample, request: CovarianceRequest
) -> tuple[BlockPreflight, Coordinates | None] | None:
    """The refusals of the covariance request that need no point, before the solve.

    Cluster alignment and at least two clusters, the conflicts of ``psd`` and of the
    convention, and for a named HC request its counts
    (:func:`~powermean.inference.absorbed_linear.linear_preflight`:
    ``specification.no_residual_dof`` of HC1, ``specification.conflicting``),
    returned for the composition.  The counts of a robust or clustered request are
    formed with its covariance, after the solve.
    """
    grouping = sample.grouping
    assert grouping is not None
    _, _, weights, frequency, active_clusters = _active_request(sample, request)
    _psd_request(request)
    inherited = (
        resolve_convention(request.convention) if request.ssc_inherited else None
    )
    resolved = resolve_convention(request.ssc, inherited)
    if request.kind == "cluster" and any(
        int(np.unique(codes).size) < 2 for _, codes in active_clusters
    ):
        raise ValueError("the cluster factor G/(G - 1) needs at least two clusters")
    if request.kind != "hc":
        return None
    assert request.method is not None
    try:
        return linear_preflight(
            int(sample.active.size),
            tuple((block.name, block.codes) for block in grouping.blocks),
            weights,
            request.method,
            resolved,
            frequency=frequency,
        )
    except SpecificationError as error:
        # A saturated fit forms no covariance: the refusal of a factor that needs
        # residual degrees of freedom is its missing covariance, as on the joint
        # route (:func:`powermean.inference.absorbed_preflight._unless_saturated`).
        if error.identifier == "specification.no_residual_dof" and linear_saturated(
            sample
        ):
            return None
        raise


def linear_saturated(sample: FESample) -> bool:
    """Whether the fit is saturated: as many rows as parameters (SATFIX, ruling 74).

    The rows of positive weight against the slopes plus the exact rank of the
    indicators of every block on them (:func:`~powermean.solve.reported.saturated`).
    The rank is formed only where it can decide: with more rows than slopes and
    levels together, the fit is not saturated whatever the rank.
    """
    grouping = sample.grouping
    assert grouping is not None
    n_slopes = int(sample.active.size)
    rows = np.flatnonzero(sample.data.weight_values > 0)
    if rows.size > n_slopes + sum(block.n_levels for block in grouping.blocks):
        return False
    if not rows.size:
        return False
    coordinates = exact_coordinates(
        [block.codes for block in grouping.blocks], rows, int(sample.count)
    )
    return saturated(int(rows.size), n_slopes + coordinates.rank)


def _inherited(request: CovarianceRequest) -> ConventionValues | None:
    """The explicit convention values a request inherits, as a retained request does."""
    if type(request.convention) is ConventionValues and request.ssc_inherited:
        return request.convention
    return None


def _robust_scale(
    envelope: RetainedAbsorbedEnvelope,
    request: CovarianceRequest,
    units: SamplingUnits,
    theta: FloatArray,
    influence: FloatArray | None,
) -> tuple[RetainedInitialScaleCache, RetainedInference]:
    """The cache of a robust or clustered request, formed by the retained service.

    Returned with the service's own record, the fit's initial inference (its notes
    are issued once, by that one computation).
    """
    order = envelope.order
    assert order is not None
    initial = retained_inference(
        envelope,
        InferenceRequest(
            "robust" if request.kind == "robust" else "cluster",
            tuple(ClusterCodes(name, codes) for name, codes in request.clusters),
            request.ssc,
            request.bread,
            _inherited(request),
            request.multiway,
            request.psd,
            request.adjustment,
        ),
    )
    applied = initial.applied
    if initial.covariance is not None:
        scaled = initial.covariance
        record = Covariance(
            covariance=scaled.covariance,
            kind=request.kind,
            factor=float(scaled.factor),
            count=initial.count,
            clusters=applied.counts.clusters[0] if applied.counts.clusters else None,
            estimate=scaled.estimate,
            reference=applied.reference.kind,
        )
    else:
        multiway = initial.multiway
        assert multiway is not None
        record = Covariance(
            covariance=multiway.covariance,
            kind=request.kind,
            factor=1.0 if applied.factor is None else float(applied.factor.value),
            count=initial.count,
            clusters=multiway.g_min,
            estimate=multiway.estimate,
            reference=applied.reference.kind,
        )
    frame = (
        None
        if initial.frame is None
        else AbsorbedFrameRecord(
            sequence=initial.frame.sequence,
            convention=initial.frame.convention,
            scales=initial.frame.scales,
            deviations=initial.frame.deviations,
            floored=initial.frame.floored,
            count=initial.frame.count,
            fingerprint=initial.frame.fingerprint,
            constant=initial.frame.constant,
            constant_total=initial.frame.constant_total,
            constant_diagnostics=initial.frame.constant_diagnostics,
        )
    )
    details = FEScaleDetails(
        bread=request.bread,
        requested_ssc=request.ssc,
        resolved_preset=applied.convention.preset or "custom",
        ssc_source=applied.convention.source,
        factor_owner="convention",
        units=units,
        count=initial.count,
        rank=applied.counts.rank,
        applied=applied,
        absorbed=initial.absorbed,
        multiway=initial.multiway,
        psd=initial.psd,
        adjustment=initial.adjustment,
        frame=frame,
        hc=None,
        reported_covariance=initial.reported_covariance,
        covariance_error=initial.covariance_error,
    )
    cache = RetainedInitialScaleCache(
        request=snapshot_covariance_request(request),
        _order=order,
        degenerate=bool((initial.standard_errors == 0.0).any()),
        _covariance=record,
        _reported=ReportedScale(
            gamma=theta, standard_errors=initial.standard_errors, order=order
        ),
        _reported_covariance=initial.reported_covariance,
        covariance_error=initial.covariance_error,
        _influence=influence,
        _details=details,
    )
    return cache, initial


def _hc_scale(
    envelope: RetainedAbsorbedEnvelope,
    sample: FESample,
    request: CovarianceRequest,
    units: SamplingUnits,
    theta: FloatArray,
    rows: FloatArray,
    influence: FloatArray | None,
    prepared: tuple[BlockPreflight, Coordinates | None],
) -> RetainedInitialScaleCache:
    """The cache of a named HC request, the record of a joint fit's HC scale."""
    order, design = envelope.order, sample.design
    assert order is not None
    assert design is not None
    preflight, coordinates = prepared
    weights = sample.data.weight_values
    grouping = sample.grouping
    assert grouping is not None
    if coordinates is not None:
        remember_coordinates(
            envelope._coordinates_memo(),
            [block.codes for block in grouping.blocks],
            preflight.rows,
            int(weights.size),
            coordinates,
        )
    found = linear_composition(
        design.matrix, weights, rows, preflight, coordinates, bread=request.bread
    )
    # An undefined request (a computed hat above 1 - sqrt(eps)) holds a missing
    # covariance, as on a joint fit: the standard errors of the record are those
    # of the unadjusted estimate, the reported covariance of the request is missing.
    errors = direct_errors(found.gate_covariance, order)
    reported: FloatArray | None = None
    error: str | None = None
    if found.undefined:
        reported = found.covariance
    else:
        try:
            reported = order.covariance(found.covariance)
        except FloatingPointError as failure:
            error = str(failure)
    details = FEScaleDetails(
        bread=request.bread,
        requested_ssc=request.ssc,
        resolved_preset=preflight.applied.convention.preset or "custom",
        ssc_source=preflight.applied.convention.source,
        factor_owner="method",
        units=units,
        count=preflight.count,
        rank=preflight.rank,
        applied=preflight.applied,
        absorbed=preflight.absorbed,
        multiway=None,
        psd=None,
        adjustment=None,
        frame=None,
        hc=found,
        reported_covariance=reported,
        covariance_error=error,
    )
    return RetainedInitialScaleCache(
        request=snapshot_covariance_request(request),
        _order=order,
        degenerate=bool((errors == 0.0).any()),
        _covariance=Covariance(
            covariance=found.covariance,
            kind="hc",
            factor=float(found.factor),
            count=preflight.count,
            clusters=None,
            estimate=found.estimate,
            reference=found.reference.kind,
        ),
        _reported=ReportedScale(gamma=theta, standard_errors=errors, order=order),
        _reported_covariance=reported,
        covariance_error=error,
        _influence=influence,
        _details=details,
    )


def _initial(
    retained: RetainedAbsorbedResult,
    sample: FESample,
    request: CovarianceRequest,
    candidate: AbsorbedLinearCandidate,
    prepared: tuple[BlockPreflight, Coordinates | None] | None,
    *,
    keep_influence: bool,
) -> RetainedInference | RetainedHC | None:
    """Form the fit's covariance on the retained envelope; its initial record.

    The influence rows and the slope map are bound to the envelope first, since
    the retained service reads them there; a fit with ``keep_influence=False``
    then discards them, the record of the request staying.
    """
    joint, absorbed = candidate.joint, candidate.prepared
    equations, point = candidate.equations, candidate.point
    assert joint is not None
    assert absorbed is not None
    assert equations is not None
    assert point is not None
    rows, slope_map = within_influence(
        joint.residual[:, 1:], absorbed.weights, equations.weighted
    )
    envelope = retained.envelope
    object.__setattr__(envelope, "_observed_influence", rows)
    object.__setattr__(envelope, "_slope_map", slope_map)
    object.__setattr__(envelope, "keep_influence", True)
    units = _active_request(sample, request)[0]
    kept = rows if keep_influence else None
    initial: RetainedInference | None = None
    if prepared is not None:
        cache = _hc_scale(
            envelope, sample, request, units, point.theta, rows, kept, prepared
        )
    else:
        cache, initial = _robust_scale(envelope, request, units, point.theta, kept)
    object.__setattr__(envelope, "initial_scale", cache)
    if not keep_influence:
        object.__setattr__(envelope, "_observed_influence", None)
        object.__setattr__(envelope, "_slope_map", None)
        object.__setattr__(envelope, "keep_influence", False)
    return initial_hc_from_envelope(envelope) if initial is None else initial


def linear_absorbed_fit(
    sample: FESample,
    outcome: SampleOutcome,
    inner: AbsorbedOptions,
    options: SolverOptions,
    prepared: tuple[BlockPreflight, Coordinates | None] | None,
    *,
    preparation: RetainedPreparation,
    dependence: BoundDependence,
    estimator: Estimator,
    request: CovarianceRequest,
    keep_influence: bool,
) -> tuple[RetainedAbsorbedResult, RetainedInference | RetainedHC | None, str, bool]:
    """Solve, retain and report a fixed-effect log-OLS fit (module docstring).

    ``prepared`` is :func:`linear_request` of the same sample and request, made
    before any solve.  Returns the retained result, the initial record of the
    covariance request
    (``None`` without slopes, when the solve did not complete or when the fit is
    saturated), the status: ``"verified"`` for an accepted fit, else the reason the
    solve ended, and whether the completed fit is saturated
    (:func:`linear_saturated`): its covariance is then missing, with the note
    ``covariance.saturated``, formed by its result as on the joint route.
    """
    candidate = linear_absorbed.solve(
        sample,
        absorb_tol_final=inner.absorb_tol_final,
        absorb_max_sweeps=inner.absorb_max_sweeps,
        absorb_acceleration=inner.absorb_acceleration,
    )
    audit = CandidateAudit(
        history=linear_history(candidate),
        controls=certificate_controls(options),
        certificate=None,
    )
    retained = retain_absorbed_fit(
        audit,
        preparation=preparation,
        outcome=outcome,
        dependence=dependence,
        estimator=estimator,
        request=request,
        keep_influence=keep_influence,
    )
    initial: RetainedInference | RetainedHC | None = None
    complete = (
        candidate.status == "complete"
        and candidate.point is not None
        and bool(sample.active.size)
    )
    is_saturated = complete and linear_saturated(sample)
    if is_saturated:
        object.__setattr__(retained.envelope, "_linear_saturated", True)
    elif complete:
        initial = _initial(
            retained, sample, request, candidate, prepared,
            keep_influence=keep_influence,
        )  # fmt: skip
    status = "verified" if retained.accepted else audit.history.reason
    return retained, initial, status, is_saturated


__all__ = ["linear_absorbed_fit", "linear_request", "linear_saturated"]
