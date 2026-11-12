"""Current robust/cluster/multiway absorbed inference with the eigenvalue adjustment."""

from __future__ import annotations

import warnings

import numpy as np

from powermean.data.inference_requests import Multiway
from powermean.data.request_values import OMITTED
from powermean.fe.absorb_numeric import checked_product
from powermean.inference.absorbed_adjusted import PsdInput, adjusted_scale
from powermean.inference.absorbed_hc_scale import hc_scale
from powermean.inference.absorbed_preflight import (
    AbsorbedFrameRecord,
    AbsorbedPreflight,
    _preflight_matches,
    preflight_absorbed_scale,
)
from powermean.inference.absorbed_rows import influence_rows
from powermean.inference.absorbed_warnings import _raw_multiway, _reissue_port_warnings
from powermean.inference.bread import same_point_influence
from powermean.inference.convention import Covariance
from powermean.inference.multiway import MultiwayCovariance, multiway_covariance
from powermean.inference.psd import PsdAdjustment, reported_errors
from powermean.inference.ssc import direct_errors, primitive_estimate, scaled_covariance
from powermean.model.rows import evaluate_rows
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_geometry import hessian as assemble_hessian
from powermean.solve.absorbed_certificate_scale import (
    FEScaleDetails,
    FEScaleEvidence,
    FEScaleInput,
    FEScaleResult,
)
from powermean.solve.reported import ReportedScale


def _bread_rows(
    scale_input: FEScaleInput, influence: FloatArray
) -> tuple[FloatArray, FloatArray | None, SparseMatrix | None]:
    """Requested current-point bread influence, preserving the existing order."""
    geometry = scale_input.geometry
    point, sample, rows = (geometry.point, geometry.point.sample, geometry.rows)
    basis, design, request = (geometry.basis, geometry.design, scale_input.request)
    exp_curv: FloatArray | None = None
    exp_hess: SparseMatrix | None = None
    if request.bread == "observed":
        bread_rows = influence
    else:
        if point.variance is None:
            exp_curv = np.array(rows.weights, dtype=np.float64)
        else:
            eval_res = evaluate_rows(
                sample.problem.response, point.q, variance=point.variance
            )
            exp_curv = checked_product(
                rows.weights, eval_res.fisher_curvature, "expected curvature"
            )
            if not np.isfinite(exp_curv).all() or (exp_curv < 0).any():
                raise FloatingPointError(
                    "the expected curvature is negative or not representable"
                )
        exp_hess = assemble_hessian(design, exp_curv, basis)
        bread_rows = same_point_influence(
            design, basis, rows.b, exp_hess, bread="expected"
        ).influence
    return (bread_rows, exp_curv, exp_hess)


def generalized_scale(
    scale_input: FEScaleInput, preflight: AbsorbedPreflight | None
) -> FEScaleResult:
    """Evaluate the full current-point inference composition at ``scale_input``."""
    request = scale_input.request
    geometry = scale_input.geometry
    point = geometry.point
    rows = geometry.rows
    order = scale_input.order
    influence = influence_rows(scale_input)
    pre = (
        preflight
        if preflight is not None and _preflight_matches(preflight, scale_input)
        else preflight_absorbed_scale(point.sample, request)
    )
    assert pre is not None
    active = pre.units.identifiers
    applied = pre.applied
    preset_label = (
        applied.convention.preset if applied.convention.preset is not None else "custom"
    )
    if request.kind == "hc":
        return hc_scale(scale_input, pre, influence)
    bread_rows, exp_curv, exp_hess = _bread_rows(scale_input, influence)
    active_rows = bread_rows[active]
    active_weights = rows.weights[active]
    multiway: MultiwayCovariance | None = None
    adjustment: PsdAdjustment | None = None
    frame_rec: AbsorbedFrameRecord | None = None
    held_warnings: list[warnings.WarningMessage] = []
    if request.uses_multiway:
        form: Multiway = "cgm" if request.multiway is OMITTED else request.multiway
        estimate, held_warnings = _raw_multiway(active_rows, pre.active_clusters, form)
        multiway = multiway_covariance(
            estimate, applied, form=form, dimensions=[n for n, _ in pre.active_clusters]
        )
        theta_cov = multiway.covariance
        factor_float = (
            float(applied.factor.value) if applied.factor is not None else 1.0
        )
        clusters_count: int | None = multiway.g_min
    else:
        estimate = primitive_estimate(
            active_rows,
            active_weights,
            frequency=rows.frequency,
            cluster=pre.active_clusters[0] if pre.active_clusters else None,
        )
        scaled = scaled_covariance(estimate, applied)
        theta_cov = scaled.covariance
        factor_float = float(scaled.factor)
        clusters_count = applied.counts.clusters[0] if applied.counts.clusters else None
    errors = direct_errors(theta_cov, order)
    rep_cov = None
    cov_err = None
    try:
        rep_cov = order.covariance(theta_cov)
    except FloatingPointError as exc:
        cov_err = str(exc)
    if pre.psd == "cgm":
        info = PsdInput(
            scale_input, pre, active_rows, exp_curv, exp_hess, form, cov_err
        )
        adjustment, frame_rec = adjusted_scale(info, rep_cov, held_warnings)
        rep_cov = adjustment.adjusted
        errors = reported_errors(rep_cov)
    _reissue_port_warnings(held_warnings, adjustment)
    cov_rec = Covariance(
        covariance=theta_cov,
        kind=request.kind,
        factor=factor_float,
        count=pre.count,
        clusters=clusters_count,
        estimate=estimate,
        reference=applied.reference.kind,
    )
    details = FEScaleDetails(
        bread=request.bread,
        requested_ssc=request.ssc,
        resolved_preset=preset_label,
        ssc_source=pre.resolved.source,
        factor_owner="convention",
        units=pre.units,
        count=pre.count,
        rank=pre.rank,
        applied=applied,
        absorbed=pre.absorbed,
        multiway=multiway,
        psd=pre.psd,
        adjustment=adjustment,
        frame=frame_rec,
        hc=None,
        reported_covariance=rep_cov,
        covariance_error=cov_err,
    )
    reported = ReportedScale(gamma=point.theta, standard_errors=errors, order=order)
    evidence = FEScaleEvidence(
        scale_input.binding,
        scale_input.request,
        influence,
        cov_rec,
        reported,
        details=details,
    )
    return FEScaleResult(scale_input, evidence)
