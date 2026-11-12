"""The requested inference at a selected standard PPML point.

Internal: only the standard route of :mod:`powermean.interface.ppml_audit` calls it,
after the selected point's standard convergence, on the owned replica that its reading
read.

The work, by the existing primitives only, in order: the canonical observed
full-nuisance geometry on the fixed ``J`` (``assembled`` and ``fresh_geometry``: raw
``S`` and ``H``, the scale ``s``, the ordinary ``A`` and its 1-norm; the canonical
``Aplus`` construction is kept although no curvature factor is requested); the ordinary
sparse LU of ``A`` (``factorized(A, "ordinary")``, its index-width, range and
singularity refusals); finite ``L`` and ``U`` data, required since no conditioning audit
runs (a factor is no rcond or definiteness pass); ``W = F^T(E_s/s)/s`` by
:func:`~powermean.solve.absorbed_certificate_sparse.slope_map` with exactly ``K``
transpose right-hand sides, counted; a finite ``W``; the request bound to the sample's
rows; ``FEScaleInput(geometry, request, OrderScale, W)``; the built-in
:class:`~powermean.inference.absorbed.AbsorbedScale` of that request; the unchanged
``admitted_scale``.  The influence ``u_i = b_i (B_i W)``, the statistical count, the
frequency and cluster rules, the factors, the covariance and the direct standard errors
are the provider's own.  ``_ordinary`` is not called: it bundles the inverse-norm audit
and the correction solve.  Without slopes nothing is formed and no provider is called:
the empty restriction, ``not_applicable``.

The explicit residualized alternative
(:mod:`powermean.results.ppml_residualized_inference`) reuses
:func:`_geometry` unchanged and hands its projected ``W`` to :func:`_provided`,
which then
holds no ordinary full-system factor record and no full-system solves and names the
method that formed it; the provider's work is the same.

Status (:func:`standard_error_status`): a missing, infinite or negative standard error
refuses usable inference; an exact zero (signed zeros included) flags degenerate
inference, never floored; otherwise ``available``.  A refusal at any step is ``refused``
with its reason; it never starts another solver, mode, covariance or retry.

Validation (:func:`same_point_audit`), optional and separately timed: the unchanged
``certify_point`` on a fresh replica of an explicit owned capture of the selected point
(sample, family, absorbed options, request, controls, ``gradient_tol``, record), with
the built-in provider of the replica's request; no coefficient solve, profile or refit.
Its raw outcome, failures and a raised refusal included, is kept whatever the standard
status, labelled ``post_standard_same_point``; nothing falls back to it and the selected
point never changes.  With ``keep_influence=False`` its payload keeps no influence and
no ``W``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import cast

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.fe.certificate_basis import CertificateBasis, SupportCheck
from powermean.inference.absorbed import AbsorbedScale
from powermean.model.variance import FloatArray
from powermean.results.absorbed_assessment_capture import Capture, replicate
from powermean.results.absorbed_geometry_records import snapshot_measures
from powermean.results.absorbed_scale_cache import snapshot_initial_scale
from powermean.results.ppml_standard_assessment import standard_anchors
from powermean.results.ppml_standard_records import (
    InferenceMethod,
    InferenceStatus,
    RetainedSamePointAudit,
    StandardInference,
    StandardPointAssessment,
)
from powermean.solve.absorbed_certificate import certify_point
from powermean.solve.absorbed_certificate_factor import (
    Calls,
    FactorRecord,
    SuperLU,
    factor_record,
    factor_side,
    factor_step,
    factorized,
    failed_factor,
)
from powermean.solve.absorbed_certificate_geometry import (
    FEGeometry,
    assembled,
    assembly_step,
    fresh_geometry,
)
from powermean.solve.absorbed_certificate_point import (
    BOUND,
    AdmittedPoint,
    FreshRows,
    WorkStep,
)
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleInput,
    admitted_scale,
    bound_request,
)
from powermean.solve.absorbed_certificate_sparse import slope_map
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.profiled_trigger import WholePoint

ORIGIN = "post_standard_same_point"
NO_SLOPES = "the model has no slope parameters (K = 0): an empty restriction"
NOT_FINITE_FACTOR = "the ordinary factor holds a value that is not finite"
NOT_FINITE_MAP = "the slope map W holds a value that is not finite"
NOT_CONVERGED = "the selected point is not standard converged: inference not computed"


def standard_error_status(errors: FloatArray) -> tuple[InferenceStatus, str]:
    """The status of direct standard errors and its reason (module docstring)."""
    if not np.isfinite(errors).all():
        return "refused", "a standard error is missing or not finite"
    if (errors < 0).any():
        return "refused", "a standard error is negative"
    if (errors == 0).any():
        return "degenerate", "a standard error is exactly zero: degenerate inference"
    return "available", ""


def _geometry(
    point: AdmittedPoint,
    rows: FreshRows,
    basis: CertificateBasis,
    support: SupportCheck,
    steps: list[WorkStep],
) -> FEGeometry | str:
    """The observed geometry on the fixed ``J``, or the text of its range refusal."""
    try:
        assembly = assembled(point, rows, basis)
    except FloatingPointError as error:
        steps.append(WorkStep("assembly", "refused", str(error), None))
        return str(error)
    steps.append(assembly_step(assembly))
    try:
        geometry = fresh_geometry(point, rows, basis, support, assembly)
    except FloatingPointError as error:
        steps.append(WorkStep("equilibration", "refused", str(error), None))
        return str(error)
    steps.extend(geometry.steps())
    return geometry


def _factor(geometry: FEGeometry) -> tuple[FactorRecord, SuperLU | None]:
    """The ordinary factor of ``A``, finite; else its record and no factor.

    A scaling refusal of the geometry refuses before any call, as ``_ordinary`` does.
    """
    if geometry.ordinary is None or geometry.scale is None:
        return factor_record("ordinary", "refused", geometry.refusal), None
    record, factor = factorized(geometry.ordinary, "ordinary")
    if factor is None:
        return record, None
    lower, upper = factor_side(factor, "L"), factor_side(factor, "U")
    if not (np.isfinite(lower.data).all() and np.isfinite(upper.data).all()):
        return failed_factor(record, NOT_FINITE_FACTOR), None
    return record, factor


def _provided(
    geometry: FEGeometry,
    record: FactorRecord | None,
    slopes: FloatArray,
    *,
    request: CovarianceRequest,
    actions: Mapping[str, int] | None,
    steps: list[WorkStep],
    method: InferenceMethod = "full_joint",
) -> StandardInference:
    """The admitted provider result at the bound input, and its status.

    ``record`` and ``actions`` are the ordinary full-system factor record and its
    counted solves, ``None`` on the residualized route, whose ``method`` the record then
    names.
    """
    order = geometry.point.sample.order_scale
    if order is None:
        raise ValueError(BOUND)
    formed = (geometry, record, slopes)
    try:
        bound = bound_request(request, geometry)
    except ValueError as error:
        work = (*steps, WorkStep("request", "refused", str(error), None))
        return StandardInference(
            "refused",
            str(error),
            *formed,
            None,
            actions,
            work,
            requested=method,
            executed=method,
        )
    scale_input = FEScaleInput(geometry, bound, order, slopes)
    try:
        result = admitted_scale(AbsorbedScale(bound)(scale_input), scale_input)
    except (FloatingPointError, ValueError) as error:
        work = (*steps, WorkStep("provider", "refused", str(error), (("calls", 1),)))
        return StandardInference(
            "refused",
            str(error),
            *formed,
            None,
            actions,
            work,
            provider_calls=1,
            requested=method,
            executed=method,
        )
    status, reason = standard_error_status(result.evidence.reported.standard_errors)
    work = (*steps, WorkStep("provider", "completed", status, (("calls", 1),)))
    return StandardInference(
        status,
        reason,
        *formed,
        result,
        actions,
        work,
        provider_calls=1,
        requested=method,
        executed=method,
    )


def standard_inference(
    reading: StandardPointAssessment, request: CovarianceRequest
) -> StandardInference:
    """The requested inference at a converged reading's point (module docstring).

    ``request`` is the owned replica's covariance request of that reading.
    """
    point, rows, support = reading.point, reading.rows, reading.support
    slopes = point.n_slopes
    if not slopes:
        return StandardInference("not_applicable", NO_SLOPES)
    if rows is None or support is None or not reading.accepted:
        raise ValueError(NOT_CONVERGED)
    steps: list[WorkStep] = []
    geometry = _geometry(point, rows, reading.basis, support, steps)
    if isinstance(geometry, str):
        return StandardInference("refused", geometry, work=tuple(steps))
    record, factor = _factor(geometry)
    steps.append(factor_step(record))
    if factor is None or geometry.scale is None:
        return StandardInference(
            "refused", record.reason, geometry, record, work=tuple(steps)
        )
    calls = Calls()
    found = slope_map(factor, geometry.scale, slopes, calls)
    actions = calls.actions()
    steps.append(WorkStep("solves", "completed", "", tuple(actions.items())))
    if not np.isfinite(found).all():
        formed = (geometry, record, found, None, actions, tuple(steps))
        return StandardInference("refused", NOT_FINITE_MAP, *formed)
    return _provided(
        geometry, record, found, request=request, actions=actions, steps=steps
    )


# -------------------------------------------------------------- validation


def _unaudited(raised: str, wall: float) -> RetainedSamePointAudit:
    """A requested same-point audit that formed no certificate, and why."""
    nan = float("nan")
    return RetainedSamePointAudit(
        ORIGIN,
        False,
        raised,
        None,
        None,
        None,
        None,
        (),
        (),
        {},
        nan,
        nan,
        None,
        (),
        None,
        None,
        None,
        (),
        wall,
    )


def same_point_audit(
    captured: Capture | None,
    selected: AdmittedPoint | None,
    request: CovarianceRequest,
    *,
    keep_influence: bool,
) -> RetainedSamePointAudit:
    """The unchanged full audit of an explicit same-point capture (module docstring).

    ``captured`` holds the fit's sample, family, absorbed options, request, controls,
    ``gradient_tol`` and selected record; ``selected`` is the standard selected point
    and ``request`` the caller's.  Without them (no selected point, or inputs that
    changed) the audit is refused truthfully: nothing is reconstructed.
    """
    started = time.perf_counter()
    replica = None if captured is None else replicate(captured)
    if replica is None or selected is None:
        refusal = "no owned same-point capture of a selected point exists"
        return _unaudited(refusal, time.perf_counter() - started)
    mine = standard_anchors(replica)
    whole = WholePoint(mine.sample, mine.variance, mine.record)
    absorbed = cast("AbsorbedOptions", replica.roots[2])
    provider = AbsorbedScale(mine.request)
    try:
        found = certify_point(whole, mine.controls, absorbed=absorbed, scale=provider)
    except (FloatingPointError, ValueError) as error:
        return _unaudited(str(error), time.perf_counter() - started)
    point, scale, measures = found.point, found.scale, found.measures
    bound = (
        same_bits(point.theta, selected.theta)
        and same_bits(point.base, selected.base)
        and same_bits(point.q, selected.q)
        and mine.request.identity() == request.identity()
    )
    kept = None
    if measures is not None:
        kept = snapshot_measures(measures, keep_slope_map=keep_influence)
    cache = None
    if scale is not None:
        cache = snapshot_initial_scale(scale, keep_influence=keep_influence)
    return RetainedSamePointAudit(
        ORIGIN,
        bound,
        "",
        found.controls,
        found.level,
        found.accepted,
        found.degenerate,
        found.strict,
        found.secondary,
        dict(cast("Mapping[str, str]", found.errors)),
        found.scaled_score,
        found.index_correction,
        None if found.effects is None else found.effects.maximum,
        () if found.inner is None else found.inner.checks,
        kept,
        found.coefficients,
        cache,
        found.work,
        time.perf_counter() - started,
    )
