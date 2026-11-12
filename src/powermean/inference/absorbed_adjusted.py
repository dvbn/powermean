"""Current-point eigenvalue adjustment: native sequence with the private constant."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from powermean.data.inference_requests import Multiway
from powermean.fe.absorb_numeric import checked_product
from powermean.inference.absorbed_constant import ConstantRow, constant_influence
from powermean.inference.absorbed_preflight import (
    AbsorbedFrameRecord,
    AbsorbedPreflight,
)
from powermean.inference.absorbed_warnings import _raw_multiway, _reissue_port_warnings
from powermean.inference.multiway import multiway_covariance
from powermean.inference.psd import PsdAdjustment, adjust_psd
from powermean.model.order_scale import OrderScale
from powermean.model.rows import _immutable, evaluate_rows
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_scale import FEScaleInput
from powermean.solve.compact_matrix import CompactMatrix


@dataclass(frozen=True)
class PsdInput:
    """Current inputs, not an acceptance flag."""

    scale_input: FEScaleInput
    pre: AbsorbedPreflight
    active_rows: FloatArray
    exp_curv: FloatArray | None
    exp_hess: SparseMatrix | None
    form: Multiway
    cov_err: str | None


def _legacy_constant(
    info: PsdInput, held_warnings: list[warnings.WarningMessage]
) -> tuple[FloatArray | None, ConstantRow | None, tuple[str, ...]]:
    """The existing same-point private reporting-constant covariance row."""
    scale_input, pre = (info.scale_input, info.pre)
    geometry = scale_input.geometry
    point, sample, rows = (geometry.point, geometry.point.sample, geometry.rows)
    basis, design, order = (geometry.basis, geometry.design, scale_input.order)
    request, applied = (scale_input.request, pre.applied)
    active, active_rows = (pre.units.identifiers, info.active_rows)
    exp_curv, exp_hess, form = (info.exp_curv, info.exp_hess, info.form)
    if exp_curv is not None:
        staged_w = exp_curv
    elif point.variance is None:
        staged_w = np.array(rows.weights, dtype=np.float64)
    else:
        eval_res = evaluate_rows(
            sample.problem.response, point.q, variance=point.variance
        )
        staged_w = checked_product(
            rows.weights, eval_res.fisher_curvature, "expected curvature"
        )
    h_const = geometry.hessian if request.bread == "observed" else exp_hess
    assert h_const is not None
    if isinstance(h_const, CompactMatrix):
        h_const = h_const.materialize()
    const_row = constant_influence(design, basis, rows.b, h_const, staged_w)
    augmented = np.column_stack([active_rows, const_row.influence[active]])
    aug_est, _ = _raw_multiway(augmented, pre.active_clusters, form)
    aug_scaled = multiway_covariance(
        aug_est, applied, form=form, dimensions=[n for n, _ in pre.active_clusters]
    )
    full_order = OrderScale(order.omega, np.append(order.scale, 1.0))
    try:
        aug_rep = full_order.covariance(aug_scaled.covariance)
    except FloatingPointError:
        _reissue_port_warnings(held_warnings, None)
        raise
    size = active_rows.shape[1]
    constant_vec = _immutable(np.append(aug_rep[size, :size], aug_rep[size, size]))
    const_diags = tuple(aug_est.diagnostics)
    return (constant_vec, const_row, const_diags)


def adjusted_scale(
    info: PsdInput,
    rep_cov: FloatArray | None,
    held_warnings: list[warnings.WarningMessage],
) -> tuple[PsdAdjustment, AbsorbedFrameRecord]:
    """Existing native frame/sequence with actual point evidence."""
    pre = info.pre
    applied = pre.applied
    cov_err = info.cov_err
    if rep_cov is None:
        _reissue_port_warnings(held_warnings, None)
        raise FloatingPointError(
            (cov_err or "the reported covariance is not representable")
            + "; the eigenvalue adjustment acts on the "
            "reported covariance and nothing replaces it"
        )
    sequence = applied.sequence
    scales = pre.frame_scales
    assert scales is not None
    constant_vec: FloatArray | None = None
    const_row: ConstantRow | None = None
    const_diags: tuple[str, ...] = ()
    if sequence == "legacy":
        constant_vec, const_row, const_diags = _legacy_constant(info, held_warnings)
    try:
        adjustment = adjust_psd(
            rep_cov,
            intercept=None,
            scales=scales.scales,
            sequence=sequence,
            constant=constant_vec,
        )
    except FloatingPointError:
        _reissue_port_warnings(held_warnings, None)
        raise
    rep_cov = adjustment.adjusted
    frame_rec = AbsorbedFrameRecord(
        sequence=sequence,
        convention=applied.convention.label,
        scales=scales.scales,
        deviations=scales.deviations,
        floored=scales.floored,
        count=scales.count,
        fingerprint=pre.fingerprint or "",
        constant=const_row is not None,
        constant_total=None if const_row is None else const_row.total,
        constant_diagnostics=const_diags,
    )
    return (adjustment, frame_rec)
