"""Lower original-audit admission and value-binding checks before retained detachment.

A selected-root candidate (:mod:`powermean.solve.absorbed_root`) is re-admitted by its
reader, its selected root point by its admission, and its convex initializer's own
audit, where one ran, by the same checks as any convex audit, under the same requested
covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean.data.fingerprint import sample_fingerprint
from powermean.data.sample_fe_state import FESample, same_bits
from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_product,
    checked_quotient,
    group_sums,
)
from powermean.fe.admission_records import AdmissionWork
from powermean.fe.certificate_basis import (
    CertificateBasis,
    _same_structure,
    retention_basis,
)
from powermean.fe.connectivity import IndicatorRank
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_history import AbsorbedRoute
from powermean.results.absorbed_retained import RetainedAdmissionReceipt
from powermean.results.absorbed_root_binding import bind_root
from powermean.results.absorbed_sample_binding import (
    UNBOUND,
    _check_prepared_matches_sample,
    _check_sample_integrity,
    _check_sample_slopes,
    _same_array_values,
    check_prepared_matches_sample,
)
from powermean.solve.absorbed_certificate import (
    CandidateAudit,
    FEPointCertificate,
    failed_gates,
)
from powermean.solve.absorbed_certificate_history import (
    CandidateHistory,
    index_history,
    linear_history,
    profiled_history,
    root_history,
)
from powermean.solve.absorbed_certificate_point import (
    AdmittedPoint,
    EffectEquations,
    WorkStep,
    admit_point,
    effect_equations,
    fresh_rows,
)
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleInput,
    FEScaleResult,
    admitted_scale,
    bound_request,
)
from powermean.solve.absorbed_root_records import AbsorbedRootCandidate
from powermean.solve.assessment import CertificateControls, certificate_controls
from powermean.solve.linear_absorbed_numeric import raw_slope_sums
from powermean.solve.linear_absorbed_state import AbsorbedLinearCandidate
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.reported import coefficient_check

if TYPE_CHECKING:
    from powermean.results.absorbed_sample import (
        RetainedSampleSnapshot,
        SampleIdentityMap,
    )


@dataclass(frozen=True)
class AdmittedAbsorbedGraph:
    """Verified live audit and sample references and their finite admission receipt."""

    audit: CandidateAudit
    sample: FESample
    receipt: RetainedAdmissionReceipt


def _check_history(audit: CandidateAudit) -> CandidateHistory:
    """Re-admit candidate history with its accepted route reader and compare."""
    history = audit.history
    if type(history) is not CandidateHistory:
        raise TypeError("audit.history must be a CandidateHistory")
    candidate = history.candidate
    if type(candidate) is ProfiledCandidate:
        expected = profiled_history(candidate)
        if audit.controls != certificate_controls(candidate.options):
            raise ValueError("audit.controls does not match candidate.options")
    elif type(candidate) is IndexCandidate:
        expected = index_history(candidate)
        if audit.controls != certificate_controls(candidate.options):
            raise ValueError("audit.controls does not match candidate.options")
    elif type(candidate) is AbsorbedRootCandidate:
        expected = root_history(candidate)
        if audit.controls != certificate_controls(candidate.options):
            raise ValueError("audit.controls does not match candidate.options")
    elif type(candidate) is AbsorbedLinearCandidate:
        expected = linear_history(candidate)
    else:
        raise TypeError("unsupported candidate type in CandidateHistory")
    id_attrs = ("sample", "final", "alarm", "trigger", "timing", "point")
    val_attrs = ("route", "variance", "retained", "reason", "message", "native", "work")
    is_lin = type(candidate) is AbsorbedLinearCandidate
    same_cand = is_lin or history.candidate is expected.candidate
    att_ok = len(history.attempts) == len(expected.attempts) and all(
        a is b for a, b in zip(history.attempts, expected.attempts, strict=True)
    )
    seq_ok = (tuple(history.partials), tuple(history.unreached)) == (
        tuple(expected.partials),
        tuple(expected.unreached),
    )
    if (
        not (same_cand and att_ok and seq_ok and candidate.sample is history.sample)
        or any(getattr(history, a) is not getattr(expected, a) for a in id_attrs)
        or any(getattr(history, a) != getattr(expected, a) for a in val_attrs)
    ):
        raise ValueError("audit.history does not match the admitted candidate reading")
    return history


def _admit_selected_point(history: CandidateHistory) -> AdmittedPoint:
    """Admit ``history.point`` with ``admit_point`` without solving or profiling."""
    pt_rec, cand = history.point, history.candidate
    if (
        type(cand) is ProfiledCandidate
        or type(cand) is IndexCandidate
        or type(cand) is AbsorbedRootCandidate
    ):
        assert history.variance is not None
        whole: Any = WholePoint(history.sample, history.variance, cast("Any", pt_rec))
        return admit_point(whole, cand.absorbed)
    if type(cand) is AbsorbedLinearCandidate:
        return admit_point(pt_rec, cand.controls)
    raise TypeError("unsupported candidate type")


def _linear_effect_equations(admitted: AdmittedPoint) -> EffectEquations:
    """Compute fresh WLS fixed-effect equations without FreshEquations replay."""
    sample = admitted.sample
    assert sample.grouping is not None
    blocks, a = sample.grouping.blocks, sample.data.weight_values
    e = checked_difference(sample.problem.response, admitted.q, "fresh residual")
    weighted = checked_product(a, e, "weighted residual")
    raw_slope_sums(sample, weighted)
    sums = tuple(group_sums(weighted, b.codes, b.n_levels) for b in blocks)
    totals = tuple(group_sums(a, b.codes, b.n_levels) for b in blocks)
    means = tuple(
        checked_quotient(s, d, "level mean") for s, d in zip(sums, totals, strict=True)
    )
    return EffectEquations(means, max(float(np.max(np.abs(m))) for m in means))


def _check_fresh_evidence(cert: FEPointCertificate, admitted: AdmittedPoint) -> None:
    """Compare ``cert.rows`` and ``cert.effects`` against ``admitted`` equations."""
    try:
        er = fresh_rows(admitted)
    except FloatingPointError:
        cr_any = (cert.rows, cert.effects, cert.geometry) != (None, None, None)
        if cr_any or "evaluation" not in cert.errors:
            raise ValueError(UNBOUND) from None
        return
    cr = cert.rows
    if cr is None or not (
        same_bits(cr.b, er.b)
        and same_bits(cr.curvature, er.curvature)
        and same_bits(cr.weights, er.weights)
        and cr.frequency == er.frequency
    ):
        raise ValueError(UNBOUND)
    try:
        fn = _linear_effect_equations if admitted.kind == "linear" else effect_equations
        exp_eff = fn(admitted)
    except FloatingPointError:
        if cert.effects is not None or "effects" not in cert.errors:
            raise ValueError(UNBOUND) from None
        return
    if cert.effects is None:
        raise ValueError(UNBOUND)
    c_max = np.asarray([cert.effects.maximum], dtype=np.float64)
    e_max = np.asarray([exp_eff.maximum], dtype=np.float64)
    means_ok = len(cert.effects.means) == len(exp_eff.means) and all(
        same_bits(a, b) for a, b in zip(cert.effects.means, exp_eff.means, strict=True)
    )
    if not same_bits(c_max, e_max) or not means_ok:
        raise ValueError(UNBOUND)


def _validate_scale_inputs(
    raw_scale: FEScaleResult,
    cert: FEPointCertificate,
    sample: FESample,
    expected_request: CovarianceRequest | None,
) -> tuple[bytes, tuple[object, ...]]:
    """Validate ``raw_scale.input`` and ``raw_scale.evidence`` bindings."""
    geom, meas, order = cert.geometry, cert.measures, sample.order_scale
    assert geom is not None
    assert meas is not None
    assert meas.slope_map is not None
    assert order is not None
    scale_input = raw_scale.input
    if type(scale_input) is not FEScaleInput:
        raise TypeError("scale.input must be an FEScaleInput")
    admitted_scale(raw_scale, scale_input)
    if scale_input.geometry is not geom:
        raise ValueError("scale_input.geometry is not certificate.geometry")
    if scale_input.order.omega != order.omega or not same_bits(
        scale_input.order.scale, order.scale
    ):
        raise ValueError("scale_input.order does not match sample.order_scale")
    if not same_bits(scale_input.slope_map, meas.slope_map):
        raise ValueError("scale_input.slope_map does not match measures.slope_map")
    bound_req = bound_request(scale_input.request, geom)
    req_id = bound_req.identity()
    if expected_request is not None and req_id != expected_request.identity():
        raise ValueError("scale_input.request does not match expected request")
    w_b = meas.slope_map.tobytes()
    exp_items = (
        *geom.binding(),
        ("W", (meas.slope_map.shape, w_b)),
        ("order", (order.omega, order.scale.tobytes())),
        ("request", req_id),
    )
    if (
        scale_input.binding.items != exp_items
        or raw_scale.evidence.binding.items != exp_items
    ):
        raise ValueError("scale binding does not match current geometry and sample")
    return w_b, req_id


def _check_scale(
    cert: FEPointCertificate,
    sample: FESample,
    expected_request: CovarianceRequest | None,
) -> tuple[bool, bytes | None, tuple[object, ...] | None]:
    """Validate ``cert._scale`` and ``cert.coefficients`` against original inputs."""
    raw_scale: FEScaleResult | None = getattr(cert, "_scale", None)
    k = cert.point.n_slopes
    geom, meas, order = cert.geometry, cert.measures, sample.order_scale
    if (
        k == 0
        or geom is None
        or meas is None
        or meas.correction is None
        or meas.slope_map is None
        or order is None
    ):
        if raw_scale is not None or cert.coefficients is not None:
            raise ValueError("unexpected scale or coefficient check on point")
        w_m = None if meas is None else meas.slope_map
        return False, (None if w_m is None else w_m.tobytes()), None
    if raw_scale is None or cert.coefficients is None:
        raise ValueError("missing scale or coefficient check on conditioned point")
    if type(raw_scale) is not FEScaleResult:
        raise TypeError("certificate._scale must be an FEScaleResult")
    w_bytes, req_id = _validate_scale_inputs(raw_scale, cert, sample, expected_request)
    rep = raw_scale.evidence.reported
    cc = cert.coefficients
    corr = rep.corrections(meas.correction[:k])
    ec = coefficient_check(corr, rep.standard_errors, cert.controls.coef_tol)
    arrs_ok = (
        _same_array_values(cc.corrections, ec.corrections)
        and _same_array_values(cc.standard_errors, ec.standard_errors)
        and np.array_equal(cc.passed, ec.passed)
        and _same_array_values(cc.ratios, ec.ratios)
    )
    if cc.coef_tol != ec.coef_tol or not arrs_ok:
        raise ValueError("certificate.coefficients does not match admitted scale")
    return True, w_bytes, req_id


def _same_basis_indices(given: object, expected: tuple[IndexArray, ...]) -> bool:
    """Compare canonical index vectors without invoking an untrusted array callback."""
    return (
        type(given) is tuple
        and len(given) == len(expected)
        and all(
            type(x) is np.ndarray
            and x.dtype == y.dtype
            and x.shape == y.shape
            and np.array_equal(x, y)
            for x, y in zip(given, expected, strict=True)
        )
    )


def _same_basis_rank(given: object, expected: IndicatorRank) -> bool:
    """Compare the primitive rank fields, not caller-defined rank equality."""
    if type(given) is not IndicatorRank:
        return False
    rank = given
    return _same_structure(
        (
            rank.levels,
            rank.n_components,
            rank.pairwise_components,
            rank.lower,
            rank.upper,
        ),
        (
            expected.levels,
            expected.n_components,
            expected.pairwise_components,
            expected.lower,
            expected.upper,
        ),
    )


def _check_certificate_point_and_basis(
    cert: FEPointCertificate, history: CandidateHistory, admitted: AdmittedPoint
) -> None:
    """Validate ``cert.point`` and ``cert.basis`` against ``history`` and admitted.

    ``cert.basis`` is compared with a fresh-owned canonical expectation.  An
    authenticated fit construction is usable only after a complete current grouping
    comparison here; otherwise ``owned_basis`` performs its original validation.
    Every supplied field is still compared, never trusted as its own expectation.
    """
    pt = cert.point
    meta_ok = (pt.kind, pt.variance, pt.inner) == (
        admitted.kind,
        admitted.variance,
        admitted.inner,
    )
    vecs_ok = (
        same_bits(pt.theta, admitted.theta)
        and same_bits(pt.base, admitted.base)
        and same_bits(pt.q, admitted.q)
    )
    if (
        type(pt) is not AdmittedPoint
        or not (meta_ok and vecs_ok)
        or pt.sample is not history.sample
        or pt.record is not history.point
    ):
        raise ValueError(UNBOUND)
    assert history.sample.grouping is not None
    eb = retention_basis(history.sample, cert.basis)
    b = cert.basis
    if (
        type(b) is not CertificateBasis
        or b.grouping is not history.sample.grouping
        or not _same_structure(
            (b.route, b.rows, b.echelon, b.witnesses),
            (eb.route, eb.rows, eb.echelon, eb.witnesses),
        )
        or not _same_basis_rank(b.rank, eb.rank)
        or not (
            _same_basis_indices((b.selected,), (eb.selected,))
            and _same_basis_indices(b.references, eb.references)
            and _same_basis_indices(b.columns, eb.columns)
            and _same_basis_indices(b.positions, eb.positions)
        )
    ):
        raise ValueError(UNBOUND)


def _check_certificate(
    audit: CandidateAudit,
    history: CandidateHistory,
    expected_request: CovarianceRequest | None,
) -> tuple[bool, bytes | None, tuple[object, ...] | None]:
    """Validate ``audit.certificate`` against ``history.point`` and sample."""
    cert = audit.certificate
    if history.point is None:
        if cert is not None:
            raise ValueError("audit.certificate must be None when history has no point")
        return False, None, None
    if type(cert) is not FEPointCertificate:
        raise ValueError("a selected point requires its FEPointCertificate")
    if cert.controls != audit.controls:
        raise ValueError("certificate.controls does not match audit.controls")
    admitted = _admit_selected_point(history)
    _check_certificate_point_and_basis(cert, history, admitted)
    _check_fresh_evidence(cert, admitted)
    if cert.geometry is not None:
        geom = cert.geometry
        d_mat = history.sample.design
        z_mat = np.zeros((history.sample.count, 0), dtype=np.float64)
        exp_design: FloatArray = z_mat if d_mat is None else d_mat.matrix
        links_ok = (geom.point, geom.rows, geom.basis) == (
            cert.point,
            cert.rows,
            cert.basis,
        )
        if not links_ok or not same_bits(geom.design, exp_design):
            raise ValueError(UNBOUND)
    tol, tol_s = cert.controls.tol, cert.controls.tol_secondary
    if cert.strict != failed_gates(cert, tol) or cert.secondary != failed_gates(
        cert, tol_s
    ):
        raise ValueError("certificate gate verdicts do not match certificate evidence")
    return _check_scale(cert, history.sample, expected_request)


def _check_initializer(
    history: CandidateHistory, expected_request: CovarianceRequest | None
) -> None:
    """A selected-root candidate's convex initializer audit, checked as any audit.

    Its history is re-admitted by its accepted reader and its certificate bound to
    its point, sample and requested covariance; nothing without such an audit.
    """
    candidate = history.candidate
    if type(candidate) is not AbsorbedRootCandidate:
        return
    convex = candidate.initializer.audit
    if convex is None:
        return
    if type(convex) is not CandidateAudit:
        raise TypeError("the initializer's audit must be a CandidateAudit")
    if convex.history.sample is not history.sample:
        raise ValueError("the initializer did not solve the candidate's sample")
    checked = _check_history(convex)
    _check_certificate(convex, checked, expected_request)


def admit_absorbed_graph(
    audit: object,
    *,
    prepared: PreparedFrame | None = None,
    request: CovarianceRequest | None = None,
    keep_influence: bool = True,
    same_preparation: bool | None = None,
    same_outcome_sample: bool = False,
) -> AdmittedAbsorbedGraph:
    """Validate the lower audit/sample/certificate/scale graph before detachment."""
    if type(audit) is not CandidateAudit:
        raise TypeError("audit must be a CandidateAudit")
    if type(audit.controls) is not CertificateControls:
        raise TypeError("audit.controls must be CertificateControls")
    if not isinstance(keep_influence, bool):
        raise TypeError("keep_influence must be a bool")
    if request is not None and type(request) is not CovarianceRequest:
        raise TypeError("request must be a CovarianceRequest or None")
    history = _check_history(audit)
    sample = _check_sample_integrity(history.sample)
    if prepared is not None:
        check_prepared_matches_sample(prepared, sample)
    has_scale, w_bytes, req_id = _check_certificate(audit, history, request)
    _check_initializer(history, request)
    root_work: AdmissionWork | None = None
    root_steps: tuple[WorkStep, ...] = ()
    if type(history.candidate) is AbsorbedRootCandidate:
        root_work, root_steps = bind_root(history.candidate, audit.certificate)
    return _capture_admission(
        audit, history, (sample, None), has_scale, w_bytes, req_id, keep_influence,
        prepared, same_preparation, same_outcome_sample, root_work, root_steps,
    )


def _consume_fit_audit(
    audit: CandidateAudit, *, request: CovarianceRequest | None, keep_influence: bool,
    sample_map: SampleIdentityMap | None = None,
) -> AdmittedAbsorbedGraph:
    """Consume a just-produced internal audit, not a public imported graph.

    The sample coordinator and certificate/scale providers have made the genuine
    validity and warning/error decisions. Do not replay their history readers,
    sample/basis comparisons or fresh equations to verify our own implementation.
    Public admit_absorbed_graph keeps its original checks.
    """
    history, cert = audit.history, audit.certificate
    sample = history.sample
    scale = None if cert is None else cert._scale
    if scale is not None:
        binding = dict(scale.input.binding.items)
        w_bytes = binding["W"][1]
        request_id = binding["request"]
    else:
        slope_map = None if cert is None or cert.measures is None else cert.measures.slope_map
        w_bytes = None if slope_map is None else slope_map.tobytes()
        request_id = None
    root_work: AdmissionWork | None = None
    root_steps: tuple[WorkStep, ...] = ()
    if type(history.candidate) is AbsorbedRootCandidate:
        root_work, root_steps = bind_root(history.candidate, cert)
    snapshot = None if sample_map is None else sample_map._owned_snapshot(sample)
    return _capture_admission(
        audit, history, (sample, snapshot), scale is not None, w_bytes, request_id,
        keep_influence, None, None, False, root_work, root_steps,
    )


def _capture_admission(
    audit: CandidateAudit,
    history: CandidateHistory,
    sample_parts: tuple[FESample, RetainedSampleSnapshot | None],
    has_scale: bool,
    w_bytes: bytes | None,
    req_id: tuple[object, ...] | None,
    keep_influence: bool,
    prepared: PreparedFrame | None,
    same_preparation: bool | None,
    same_outcome_sample: bool,
    root_work: AdmissionWork | None,
    root_steps: tuple[WorkStep, ...],
) -> AdmittedAbsorbedGraph:
    """Build the same detached finite receipt from already admitted values."""
    sample, snapshot = sample_parts
    cert = audit.certificate
    pt = None if cert is None else cert.point
    var_name = None if history.variance is None else type(history.variance).__name__
    route: AbsorbedRoute = history.route
    n_kept = None if cert is None else cert.basis.n_kept
    fp = sample_fingerprint(sample.data) if snapshot is None else snapshot.fingerprint
    sp = prepared is not None if same_preparation is None else bool(same_preparation)
    vb = (
        (None, None, None)
        if pt is None
        else (pt.theta.tobytes(), pt.base.tobytes(), pt.q.tobytes())
    )
    return AdmittedAbsorbedGraph(
        audit,
        sample,
        RetainedAdmissionReceipt(
            route=route,
            omega=sample.omega,
            variance_name=var_name,
            nobs=sample.ledger.nobs,
            count=sample.count,
            n_slopes=int(sample.active.size),
            n_effects=n_kept,
            sample_digest=fp,
            has_selected_point=history.point is not None,
            has_certificate=cert is not None,
            has_initial_scale=has_scale,
            keep_influence=keep_influence,
            same_preparation=sp,
            same_outcome_sample=bool(same_outcome_sample),
            same_candidate_sample=True,
            same_selected_point=True,
            same_scale_input=has_scale,
            theta_bytes=vb[0],
            base_bytes=vb[1],
            q_bytes=vb[2],
            slope_map_bytes=w_bytes,
            request_identity=req_id,
            root_binding_work=root_work,
            root_binding_steps=root_steps,
        ),
    )


__all__ = [
    "UNBOUND",
    "AdmittedAbsorbedGraph",
    "_admit_selected_point",
    "_check_certificate",
    "_check_fresh_evidence",
    "_check_history",
    "_check_prepared_matches_sample",
    "_check_sample_integrity",
    "_check_sample_slopes",
    "_check_scale",
    "_same_array_values",
    "admit_absorbed_graph",
    "check_prepared_matches_sample",
]
