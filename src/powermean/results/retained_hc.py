"""Internal named HC computation from retained records, with fresh binding checks.

The contract, request/result records and capability-only queries are documented in
:mod:`powermean.results.retained_hc_records`. This module composes the kernels at the
selected point; no fit, sample selection or formula replay occurs. The reporting order
and column scale are bound before covariance computation. A selected root point of an
absorbed joint fit outside ``tau in [1, 2]`` (point kind ``"root"``) keeps its actual
signed observed curvature in the fresh rows (the expected-information hats of the full
nuisance geometry with the actual observed bread); every other check stands, and every
other point keeps the refusal of a negative curvature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from powermean import messages
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import (
    ConleyRequest,
    HacRequest,
)
from powermean.data.request_values import OMITTED
from powermean.errors import SpecificationError
from powermean.fe.absorb_numeric import checked_difference, checked_product
from powermean.inference.absorbed_details import typed_details
from powermean.inference.hc import (
    HCCovariance,
    _absorbed_composition,
    joint_hc,
    linear_hc,
)
from powermean.inference.ssc import (
    ResolvedConvention,
    apply_convention,
    direct_errors,
    owned_factor,
    resolve_convention,
)
from powermean.model.order_scale import OrderScale
from powermean.model.rows import evaluate_rows
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.cancelled_retained import RetainedCancelledEnvelope
from powermean.results.handoff import _same_basis, linear_problem_of, system_of
from powermean.results.point import frequency_of
from powermean.results.retained import RetainedFit
from powermean.results.retained_hc_records import (
    HCCapability,
    HCInferenceRequest,
    RetainedHC,
    Target,
    _envelope_capability,
    _fit_capability,
    hc_capability,
)
from powermean.results.retained_inference import (
    InferenceRequest,
    RetainedInference,
    retained_inference,
)
from powermean.solve.absorbed_certificate_geometry import hessian
from powermean.solve.absorbed_certificate_point import FreshRows
from powermean.solve.certificate import outside_convex
from powermean.systems.joint import JointEvaluation

# The later release that owns HAC and Conley.
NOT_BOUND = "the retained records are not bound to one point, equation and sample"


# ---------------------------------------------------------------------- request


# ------------------------------------------------------------------- capability


# ----------------------------------------------------------------------- answer


# ------------------------------------------------------------------- the checks


def _same(found: object, given: object) -> bool:
    """Whether two arrays have one shape, dtype and the same bytes."""
    if not isinstance(found, np.ndarray) or not isinstance(given, np.ndarray):
        return False
    return bool(
        found.shape == given.shape
        and found.dtype == given.dtype
        and found.tobytes() == given.tobytes()
    )


def _bound(condition: bool) -> None:
    """Refuse records that are not bound to one another (``ValueError``)."""
    if not condition:
        raise ValueError(NOT_BOUND)


def _same_order(found: OrderScale, given: OrderScale | None) -> bool:
    """Bind the reporting map by its order and column scale, without mapping values."""
    return bool(
        given is not None
        and found.omega == given.omega
        and _same(found.scale, given.scale)
    )


def _unsupported(request: HCInferenceRequest, case: str) -> SpecificationError:
    """The refusal of a method that a retained case does not support."""
    return SpecificationError(
        "specification.unsupported_inference", {"method": request.method, "case": case}
    )


def _resolved(
    request: HCInferenceRequest,
    inherited: str | ConventionValues | ResolvedConvention | None,
) -> ResolvedConvention:
    """The convention in force: explicit, else the private declaration, else the fit's.

    ``inherited`` is the recorded name or resolved convention of the retained initial
    request, or ``None`` where no covariance was formed at the fit.
    """
    found: ConventionValues | ResolvedConvention | None = request.inherited
    if found is None and inherited is not None:
        found = (
            inherited
            if isinstance(inherited, ResolvedConvention)
            else resolve_convention(inherited)
        )
    return resolve_convention(request.ssc, found)


def _reported(
    theta_covariance: FloatArray, order: OrderScale
) -> tuple[FloatArray, FloatArray | None, str | None]:
    """The direct errors and the reported covariance, each map applied once."""
    errors = direct_errors(theta_covariance, order)
    try:
        return errors, order.covariance(theta_covariance), None
    except FloatingPointError as exc:
        return errors, None, str(exc)


# ------------------------------------------------------------------- no-FE fits


def _linear_route(
    retained: RetainedFit, request: HCInferenceRequest, resolved: ResolvedConvention
) -> HCCovariance:
    """The captured solve, bound to the retained equation and point, then the kernel."""
    solved = retained.linear_solve
    if solved is None:
        raise _unsupported(
            request, "a linear fit whose solved equation was not captured with the fit"
        )
    captured, candidate = solved.request, solved.candidate
    problem = linear_problem_of(retained.sample)
    linear, theta = retained.linear, retained.theta
    _bound(
        captured.frequency == frequency_of(retained.sample)
        and all(
            _same(getattr(captured, name), getattr(problem, name))
            for name in ("response", "design", "offset", "weights")
        )
        and linear is not None
        and candidate.coefficients is not None
        and _same(candidate.coefficients, theta)
        and _same(candidate.coefficients, linear.coefficients)
        and _same(candidate.residuals, linear.residuals)
        and _same(candidate.units.identifiers, linear.units.identifiers)
        and _same(candidate.weights, linear.weights)
    )
    return linear_hc(solved, request.method, resolved)


def _joint_route(
    retained: RetainedFit, request: HCInferenceRequest, resolved: ResolvedConvention
) -> HCCovariance:
    """The retained equation at the certificate's evaluation, then the kernel."""
    certificate = retained.certificate
    evaluation = None if certificate is None else certificate.evaluation
    influence = retained.influence
    if certificate is None or evaluation is None or influence is None:
        raise _unsupported(request, "a fit without an evaluation of its point")
    _bound(
        _same(evaluation.gamma, retained.theta)
        and _same(certificate.gamma, retained.theta)
        and np.array_equal(influence.jacobian, -evaluation.bread)
    )
    system = system_of(retained.estimator, retained.sample)
    kernel = joint_hc
    method = request.method
    frequency = frequency_of(retained.sample)
    bread = request.bread
    if type(evaluation) is not JointEvaluation:
        raise TypeError("evaluation must be a JointEvaluation")
    return kernel(
        system,
        evaluation,
        method,
        resolved,
        frequency=frequency,
        bread=bread,
    )


def _fit_hc(retained: RetainedFit, request: HCInferenceRequest) -> RetainedHC:
    """Answer a named HC request from a certified no-FE fit (module docstring)."""
    if not retained.certified or retained.theta is None:
        raise ValueError("no selected point was certified for this candidate")
    if not retained.keep_influence or retained.influence is None:
        raise SpecificationError(
            "specification.influence_discarded", {"request": request.label}
        )
    sample, certificate = retained.sample, retained.certificate
    _bound(
        sample.omega == retained.estimator.mean.omega
        and certificate is not None
        and certificate.reported is not None
        and _same_basis(certificate.reported, sample.order_scale)
    )
    capability = _fit_capability(retained)
    default = retained.default
    resolved = _resolved(request, None if default is None else default.convention)
    if capability.route == "linear":
        found = _linear_route(retained, request, resolved)
    else:
        found = _joint_route(retained, request, resolved)
    sample = retained.sample
    errors, reported, error = _reported(found.covariance, sample.order_scale)
    return RetainedHC(
        request,
        capability,
        found,
        found.covariance,
        errors,
        reported,
        error,
        retained.theta,
        retained.names,
        found.applied.counts.count,
        int(found.leverage.active.size),
        sample.count,
    )


# ------------------------------------------------------------- absorbed envelope


def _fresh_rows(
    response: FloatArray,
    q: FloatArray,
    weights: FloatArray,
    variance: PowerVariance | NB2Variance | None,
    *,
    selected_root: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``b``, ``C`` and the unweighted expected curvature ``w`` at ``q``, afresh.

    The arithmetic of the accepted
    :func:`~powermean.solve.absorbed_certificate_point.fresh_rows` (checked products,
    the accepted row kernel), on the retained response, point and raw weights; ``w``
    is one for the WLS equation, else the row kernel's information weight.  A
    negative observed curvature is refused except at a ``selected_root`` point, as
    the accepted rows refuse it except at a root point.
    """
    if variance is None:
        e = checked_difference(response, q, "fresh residual")
        b = checked_product(weights, e, "weighted residual")
        c = np.array(weights, dtype=np.float64)
        w = np.ones(weights.shape)
    else:
        rows = evaluate_rows(response, q, variance=variance)
        b = checked_product(weights, rows.score, "weighted score")
        c = checked_product(weights, rows.observed_curvature, "weighted curvature")
        w = np.array(rows.fisher_curvature, dtype=np.float64)
    negative = bool((c < 0).any()) and not selected_root
    if not (np.isfinite(b).all() and np.isfinite(c).all()) or negative:
        raise FloatingPointError("the fresh curvature is negative or not representable")
    return b, c, w


def _same_sparse(found: object, given: object) -> bool:
    """Whether two owned CSC records hold the same shape, structure and values."""
    return bool(
        getattr(found, "shape", None) == getattr(given, "shape", None)
        and all(
            _same(getattr(found, name, None), getattr(given, name, None))
            for name in ("indptr", "indices", "data")
        )
    )


def _envelope_hc(
    envelope: RetainedAbsorbedEnvelope, request: HCInferenceRequest
) -> RetainedHC:
    """Answer a named HC request from a certified absorbed envelope."""
    point, certificate = envelope.selected_point, envelope.certificate
    if point is None or certificate is None or not certificate.accepted:
        raise ValueError("no selected point was certified for this candidate")
    if envelope.empty:
        raise ValueError("the model has no slope parameters (K = 0)")
    init = envelope.initial_scale
    if (
        init is not None
        and init.request.kind == "hc"
        and init.details is not None
        and init.details.hc is not None
        and init.request.method == request.method
        and init.request.bread == request.bread
    ):
        det = typed_details(init.details)
        resolved_init = _resolved(request, det.applied.convention)
        owned_factor(
            resolved_init,
            det.applied.counts,
            kind="robust",
            method=request.method,
        )
        if resolved_init.values == det.applied.convention.values:
            hc_rec = det.hc
            assert hc_rec is not None
            if resolved_init != det.applied.convention:
                applied_new = apply_convention(
                    resolved_init,
                    det.applied.counts,
                    kind="robust",
                    method=request.method,
                )
                hc_rec = replace(hc_rec, applied=applied_new)
            return RetainedHC(
                request,
                _envelope_capability(envelope),
                hc_rec,
                init.covariance.covariance,
                init.standard_errors,
                init.reported_covariance,
                init.covariance_error,
                point.theta,
                tuple(envelope.sample.active_names),
                det.count,
                int(det.units.identifiers.size),
                envelope.sample.count,
            )
    envelope.require_influence(request.label)
    capability = _envelope_capability(envelope)
    sample, order = envelope.sample, envelope.order
    rows, geometry, design = certificate.rows, certificate.geometry, sample.design
    if rows is not None and rows._influence_discarded:
        # the rows of a discarded envelope stay marked whatever its flag says
        raise SpecificationError(
            "specification.influence_discarded", {"request": request.label}
        )
    if rows is None or geometry is None or design is None or order is None:
        raise _unsupported(request, "an absorbed fit without its retained geometry")
    _bound(
        order.omega == sample.omega
        and _same(order.scale, design.scale)
        and _same_order(order, sample.order_scale)
        and _same_order(order, point.sample.order_scale)
        and _same_order(order, certificate.point.sample.order_scale)
        and point.sample.omega == sample.omega
        and certificate.point.sample.omega == sample.omega
    )
    matrix, weights, response = design.matrix, sample.weight_values, rows.response
    q = rows.q
    _bound(
        _same(certificate.point.theta, point.theta)
        and _same(certificate.point.q, point.q)
        and _same(q, point.q)
        and _same(geometry.design, matrix)
        and _same(rows.weights, weights)
        and _same(response, sample.problem.response)
        and rows.frequency == sample.frequency
    )
    selected_root = (
        point.kind == "root"
        and certificate.point.kind == "root"
        and outside_convex(rows.variance)
    )
    b, c, w = _fresh_rows(
        response, q, weights, rows.variance, selected_root=selected_root
    )
    basis = certificate.basis.basis
    _bound(
        _same(b, rows.b)
        and _same(c, rows.curvature)
        and _same_sparse(hessian(matrix, c, basis), geometry.hessian)
    )
    inherited: str | ConventionValues | ResolvedConvention | None = None
    if envelope.initial_scale is not None:
        if envelope.initial_scale.details is not None:
            inherited = typed_details(envelope.initial_scale.details).applied.convention
        else:
            inherited = envelope.initial_scale.request.convention
    found = _absorbed_composition(
        matrix,
        basis,
        FreshRows(b, c, weights, sample.frequency),
        w,
        request.method,
        convention=_resolved(request, inherited),
        bread=request.bread,
    )
    errors, reported, error = _reported(found.covariance, order)
    return RetainedHC(
        request,
        capability,
        found,
        found.covariance,
        errors,
        reported,
        error,
        point.theta,
        tuple(sample.active_names),
        found.applied.counts.count,
        int(found.leverage.active.size),
        sample.count,
    )


# --------------------------------------------------------------------- service


def retained_hc(target: Target, request: HCInferenceRequest) -> RetainedHC:
    """Answer a named HC request from the retained state of ``target``.

    Parameters
    ----------
    target : RetainedFit or RetainedAbsorbedEnvelope
        The accepted retained state of a no-FE fit or of an absorbed fit.
    request : HCInferenceRequest
        The method, the explicit convention, the bread.

    Returns
    -------
    RetainedHC
        The kernel's record, the covariance of ``theta``, the direct standard errors
        and the reported covariance, the counts.

    Raises
    ------
    TypeError
        Another type of target or request.
    ValueError
        No certified point; no slope; records that are not bound to one point,
        equation and sample; a design that is not of full column rank.
    SpecificationError
        ``specification.influence_discarded``, ``specification.unsupported_inference``
        before any computation; ``specification.no_residual_dof``,
        ``specification.conflicting`` and ``specification.unit_leverage`` from the
        kernels' preflight, before any evaluation or factorization.
    FloatingPointError
        A multiplier or intermediate that is not representable (no floor).
    """
    if type(request) is not HCInferenceRequest:
        raise TypeError("request must be an HCInferenceRequest")
    if type(target) is RetainedCancelledEnvelope:
        raise _unsupported(
            request, "cancelled group equations; HC extension remains open"
        )
    if type(target) is RetainedFit:
        return _fit_hc(target, request)
    if type(target) is RetainedAbsorbedEnvelope:
        return _envelope_hc(target, request)
    raise TypeError("target must be a RetainedFit or a RetainedAbsorbedEnvelope")


def deferred_refusal(request: HacRequest | ConleyRequest) -> SpecificationError:
    """The explicit later-release refusal of a reserved HAC or Conley request."""
    if type(request) is HacRequest:
        feature = "HAC covariance (deferred to a later release)"
    elif type(request) is ConleyRequest:
        feature = "Conley covariance (deferred to a later release)"
    else:
        raise TypeError("request must be a HacRequest or a ConleyRequest")
    return SpecificationError(
        "specification.unimplemented",
        {"feature": feature},
    )


def _unknown_key(key: object, valid: Sequence[str]) -> SpecificationError:
    """The refusal of a key that a reserved request mapping does not declare."""
    return SpecificationError(
        "specification.unknown_option",
        {"kind": "vcov key", "key": key, "valid": messages.format_keys(valid)},
    )


def _hac_mapping(vcov: Mapping[Any, Any]) -> SpecificationError:
    """The later-release refusal of a HAC mapping, its keys and values checked first."""
    allowed: tuple[str, ...] = ("hac", "unit", "lag", "step")
    for key in vcov:
        if key not in allowed:
            raise _unknown_key(key, allowed)
    return deferred_refusal(
        HacRequest(
            time=vcov["hac"],
            unit=vcov.get("unit", OMITTED),
            lag=vcov.get("lag", OMITTED),
            step=vcov.get("step", OMITTED),
        )
    )


def _conley_mapping(vcov: Mapping[Any, Any]) -> SpecificationError:
    """The later-release refusal of a Conley mapping, keys and values checked first."""
    allowed: tuple[str, ...] = ("conley", "cutoff", "units", "distance", "pool_times")
    for key in vcov:
        if key not in allowed:
            raise _unknown_key(key, allowed)
    coords = vcov["conley"]
    if (
        not isinstance(coords, Sequence)
        or isinstance(coords, (str, bytes))
        or len(coords) != 2
    ):
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "conley",
                "value": coords,
                "valid": "two coordinate column names [lat, lon]",
            },
        )
    return deferred_refusal(
        ConleyRequest(
            latitude=coords[0],
            longitude=coords[1],
            cutoff=vcov.get("cutoff", OMITTED),
            units=vcov.get("units", OMITTED),
            distance=vcov.get("distance", OMITTED),
            pool_times=vcov.get("pool_times", OMITTED),
        )
    )


def deferred_vcov(vcov: object, cluster: object = None) -> None:
    """Refuse a reserved HAC or Conley covariance request, before anything else.

    A typed :class:`~powermean.data.inference_requests.HacRequest` or
    :class:`~powermean.data.inference_requests.ConleyRequest` given as ``vcov`` or
    ``cluster``, or a ``vcov`` mapping with the key ``"hac"`` or ``"conley"``, whose
    keys and values are checked by its record first, is refused with
    ``specification.unimplemented`` naming the later release (:func:`deferred_refusal`):
    no covariance is substituted, nothing is solved and no retained state is read.  The
    one check of the fit's parsers (:mod:`powermean.interface.fe_options`) and of the
    requests on a result.  Any other value passes, for its own parser.
    """
    for value in (vcov, cluster):
        if isinstance(value, HacRequest | ConleyRequest):
            raise deferred_refusal(value)
    if isinstance(vcov, Mapping):
        if "hac" in vcov:
            raise _hac_mapping(vcov)
        if "conley" in vcov:
            raise _conley_mapping(vcov)


def retained_request(
    target: Target,
    request: HCInferenceRequest | InferenceRequest | HacRequest | ConleyRequest,
) -> RetainedHC | RetainedInference:
    """Route one post-fit request on retained state by its typed record.

    A named HC request goes to :func:`retained_hc`; a robust or cluster request to the
    unchanged :func:`~powermean.results.retained_inference.retained_inference`; a
    reserved HAC or Conley request is refused before the target is read.
    """
    if isinstance(request, HacRequest | ConleyRequest):
        raise deferred_refusal(request)
    if type(target) not in (
        RetainedFit,
        RetainedAbsorbedEnvelope,
        RetainedCancelledEnvelope,
    ):
        raise TypeError("target must be a RetainedFit or a RetainedAbsorbedEnvelope")
    if type(request) is HCInferenceRequest:
        return retained_hc(target, request)
    if type(request) is InferenceRequest:
        return retained_inference(target, request)
    raise TypeError(
        "request must be an HCInferenceRequest, an InferenceRequest, a HacRequest "
        "or a ConleyRequest"
    )


def initial_hc_from_envelope(envelope: RetainedAbsorbedEnvelope) -> RetainedHC:
    """Build ``RetainedHC`` directly from ``envelope.initial_scale``."""
    init = envelope.initial_scale
    point = envelope.selected_point
    if (
        init is None
        or point is None
        or init.request.kind != "hc"
        or init.details is None
        or init.details.hc is None
        or init.request.method is None
    ):
        raise ValueError("no initial HC scale was retained for this candidate")
    det = typed_details(init.details)
    assert det.hc is not None
    req = init.request.request
    assert req.method is not None
    inherited = (
        req.convention
        if type(req.convention) is ConventionValues and req.ssc_inherited
        else None
    )
    hc_req = HCInferenceRequest(
        req.method,
        req.ssc,
        req.bread,
        inherited,
    )
    return RetainedHC(
        hc_req,
        _envelope_capability(envelope),
        det.hc,
        init.covariance.covariance,
        init.standard_errors,
        init.reported_covariance,
        init.covariance_error,
        point.theta,
        tuple(envelope.sample.active_names),
        det.count,
        int(det.units.identifiers.size),
        envelope.sample.count,
    )


__all__ = [
    "HCCapability",
    "HCInferenceRequest",
    "RetainedHC",
    "deferred_refusal",
    "deferred_vcov",
    "hc_capability",
    "initial_hc_from_envelope",
    "retained_hc",
    "retained_request",
]
