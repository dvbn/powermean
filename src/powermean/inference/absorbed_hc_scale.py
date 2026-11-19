"""Named HC scale composition at the actual absorbed certificate point."""

from __future__ import annotations

import numpy as np

from powermean.inference.absorbed_block_hc import BlockPreflight, block_composition
from powermean.inference.absorbed_preflight import AbsorbedPreflight
from powermean.inference.convention import (
    Covariance,
)
from powermean.inference.ssc import (
    direct_errors,
)
from powermean.model.rows import evaluate_rows
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_scale import (
    FEScaleDetails,
    FEScaleEvidence,
    FEScaleInput,
    FEScaleResult,
)
from powermean.solve.reported import ReportedScale


def hc_scale(
    scale_input: FEScaleInput, pre: AbsorbedPreflight, influence: FloatArray
) -> FEScaleResult:
    """Compose the unchanged current full-nuisance HC kernel and direct report map."""
    geometry = scale_input.geometry
    point, sample, rows = geometry.point, geometry.point.sample, geometry.rows
    basis, design, order = geometry.basis, geometry.design, scale_input.order
    request, applied = scale_input.request, pre.applied
    preset_label = applied.convention.preset or "custom"
    assert request.method is not None
    if point.variance is None:
        fisher = np.ones(sample.count, dtype=np.float64)
    else:
        fisher = np.array(
            evaluate_rows(
                sample.problem.response, point.q, variance=point.variance
            ).fisher_curvature,
            dtype=np.float64,
        )
    # Fixed-effect log-OLS forms its named HC in
    # :mod:`powermean.inference.absorbed_linear` and holds no scale input.
    held = pre.hc_preflight
    assert type(held) is BlockPreflight
    hc_cov = block_composition(
        design,
        basis,
        rows,
        fisher,
        held,
        bread=request.bread,
        observed_influence=influence,
    )
    # An undefined request (a hat above the near-one warning criterion) holds a missing
    # covariance: the fit's coefficient gate then reads the unadjusted estimate of
    # the same rows, and the reported covariance of the request is missing.
    errors = direct_errors(hc_cov.gate_covariance, order)
    rep_cov: FloatArray | None = None
    cov_err: str | None = None
    if hc_cov.undefined:
        rep_cov = hc_cov.covariance
    else:
        try:
            rep_cov = order.covariance(hc_cov.covariance)
        except FloatingPointError as exc:
            cov_err = str(exc)
    cov_rec = Covariance(
        covariance=hc_cov.covariance,
        kind="hc",
        factor=float(hc_cov.factor),
        count=pre.count,
        clusters=None,
        estimate=hc_cov.estimate,
        reference=hc_cov.reference.kind,
    )
    details = FEScaleDetails(
        bread=request.bread,
        requested_ssc=request.ssc,
        resolved_preset=preset_label,
        ssc_source=pre.resolved.source,
        factor_owner="method",
        units=pre.units,
        count=pre.count,
        rank=pre.rank,
        applied=applied,
        absorbed=pre.absorbed,
        multiway=None,
        psd=None,
        adjustment=None,
        frame=None,
        hc=hc_cov,
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
