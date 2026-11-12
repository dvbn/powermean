"""Compose distinct BBP sampling and published WZ coefficient corrections.

The existing normal cancelled domain has unit multipliers and observed one-way
inference. Cell integration here covers declared clusters made of complete
cancelled groups. Crossing/multiway cell pairing and joint-cell fitting remain
open mappings, not asserted sampling schemes. No correction work runs for False.
WZ uses the certified uncorrected PPML root and its own reserved analytical
operator on balanced directed complete N^2T or off-diagonal N(N-1)T support
(03.5.4 I1, 04.7).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.fe.admission_budget import admission_budget
from powermean.fe.geometry_checks import GeometryInputs
from powermean.inference.convention import ClusterCodes, IntegerArray
from powermean.inference.correction.budget import BiasBounds, enforce
from powermean.inference.correction.cell import CellBiasResult, bbp_cell_correction
from powermean.inference.correction.pairs import cluster_cell_pairs
from powermean.inference.correction.selector import BiasEvalOptions
from powermean.inference.correction.short_group import (
    ShortGroupBiasResult,
    bbp_short_group_correction,
)
from powermean.inference.wz_binding import (
    WZSource,
    capture_rows,
    layout,
    reserve_validation,
    support,
)
from powermean.inference.wz_coefficient import CoefficientBias, evaluate, workspace
from powermean.inference.wz_covariance import (
    WZControls,
    _cancelled_sources,
    bind_request,
    ppml,
)
from powermean.interface.fe_bias_options import BiasRequest
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray, PowerVariance, Variance
from powermean.results.cancelled_bias import BiasPoint, CancelledBias
from powermean.results.cancelled_retained import RetainedCancelledEnvelope
from powermean.results.retained_inference import InferenceRequest
from powermean.results.wz_bias import WZBias, WZBiasPoint
from powermean.solve.absorbed_certificate import FEPointCertificate
from powermean.solve.absorbed_certificate_point import admit_point
from powermean.solve.cancelled_certificate import CertifiedCancelledPoint
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.profiled_trigger import WholePoint
from powermean.systems.admitted import CapturedFit


def unavailable(case: str) -> SpecificationError:
    """Refuse an unestablished coefficient domain without changing BBP's domains."""
    return SpecificationError(
        "specification.unsupported_inference",
        {
            "method": "WZ coefficient",
            "case": (
                case + "; bias_correction=True is the general equation request "
                "on its existing domains"
            ),
        },
    )


def coefficient_ppml(omega: float, variance: object, *, weights: bool = False) -> None:
    """Reuse the exact PPML guard with coefficient-specific refusal metadata."""
    try:
        ppml(omega, variance, weights=weights)
    except SpecificationError as error:
        raise unavailable(
            "builtin PPML omega=1, tau=1 without statistical weights required"
        ) from error


def wz_pair_name(
    source: WZSource,
    shape: tuple[int, int],
    bounds: AttemptBounds,
    *,
    selected: str | None = None,
) -> str:
    """Validate a selected pair role, or identify a unique unnamed pair role.

    Each candidate uses layout's complete/off-diagonal support recognition,
    never a claimed label or flag. An explicit sampling/cancellation role is
    checked on its own. Without one, several matches on a complete N=T cube
    whose node and date labels coincide refuse rather than choose a role.
    """
    reserve_validation(shape, bounds)
    matches: list[str] = []
    for block in source.grouping.blocks:
        if selected is not None and block.name != selected:
            continue
        try:
            a, b, dates = source.components(block.name)
            support(a, b, dates)
        except SpecificationError:
            continue
        if len(a) == shape[0]:
            matches.append(block.name)
    if len(matches) != 1:
        raise unavailable(
            "WZ coefficient requires a unique balanced directed pair role on "
            "complete N^2T or off-diagonal N(N-1)T support"
        )
    return matches[0]


def unsupported_bias(case: str) -> SpecificationError:
    """An honest unavailable-domain refusal, not a zero correction or fallback."""
    return SpecificationError(
        "specification.unsupported_inference", {"method": "BBP", "case": case}
    )


@dataclass(frozen=True, slots=True)
class BiasSampling:
    """Per-fit row-aligned dependence data, never a structural validation token."""

    request: str
    method: Literal["short_group", "cell"]
    name: str
    codes: IndexStorage
    n_units: int


def bias_sampling(
    request: BiasRequest,
    cancelled: IntegerArray,
    name: str,
    inference: InferenceRequest | None,
    *,
    bounds: BiasBounds,
) -> BiasSampling | None:
    """Check current partitions before solver work; no pair enumeration occurs here."""
    if request is None:
        return None
    if request == "wz":
        raise unsupported_bias(
            "published WZ requires its distinct coefficient operator"
        )
    enforce(256 * cancelled.size, 128 * cancelled.size, bounds)
    if inference is None:
        codes, sampling_name = cancelled, name
    elif inference.kind == "cluster" and len(inference.clusters) == 1:
        cluster = inference.clusters[0]
        codes, sampling_name = cluster.codes, cluster.name
    else:
        raise unsupported_bias("declared one-way cancelled-group sampling required")
    if codes.shape != cancelled.shape:
        raise ValueError("BBP dependence must align with the current fit rows")
    forward: dict[int, int] = {}
    units: dict[int, int] = {}
    owned_codes = np.empty(cancelled.size, dtype=np.intp)
    for row, (group, unit) in enumerate(zip(cancelled, codes, strict=True)):
        g, u = int(group), int(unit)
        if g in forward and forward[g] != u:
            raise unsupported_bias("crossing cell pairing is not integrated")
        forward[g] = u
        if u not in units:
            units[u] = len(units)
        owned_codes[row] = units[u]
    same = len(forward) == len(units)
    if request != "cell" and not same:
        raise unsupported_bias(
            "True/short_group requires the actual cancelled-group sampling partition; "
            "an explicit cell request is required for complete-group dependence"
        )
    return BiasSampling(
        request,
        "cell" if request == "cell" else "short_group",
        sampling_name,
        IndexStorage.store(owned_codes),
        len(units),
    )


@dataclass(frozen=True, slots=True)
class BiasFitInputs:
    """Already aligned normal-fit inputs; each kernel freshly captures numeric bytes."""

    design: GeometryInputs
    z: FloatArray
    offset: FloatArray
    variance: Variance
    bounds: BiasBounds
    rcond_min: float


def apply_cancelled_bias(
    envelope: RetainedCancelledEnvelope,
    sampling: BiasSampling | None,
    inputs: BiasFitInputs,
) -> CancelledBias | None:
    """Evaluate once at the certified uncorrected point, without refitting or SSC V."""
    if sampling is None:
        return None
    if not envelope.accepted:
        return CancelledBias(
            sampling.request, sampling.method, sampling.name, sampling.n_units, None
        )
    geometry = envelope.geometry
    n, p, d, k = (
        geometry.structure.n_rows,
        geometry.n_columns_full,
        geometry.n_identified,
        geometry.n_slopes_identified,
    )
    # Reserve the still-live sampling codes/options, identified/full reporting
    # temporaries and four small owned outputs in addition to the kernel plan.
    # These are conservative managed arrays/visits, not measured OS memory/work.
    capture_bytes = 8 * (16 * n + 32 * p + 32 * d + 64 * k)
    capture_work = 128 * (n + p * d + p + k)
    enforce(capture_bytes + 1, capture_work + 1, inputs.bounds)
    kernel_bounds = BiasBounds(
        inputs.bounds.max_bytes - capture_bytes,
        inputs.bounds.max_work_units - capture_work,
    )
    theta = envelope.certificate.gamma
    options = BiasEvalOptions(
        order=envelope.order,
        centered_offset=inputs.offset,
        bounds=kernel_bounds,
        rcond_min=inputs.rcond_min,
    )
    result: ShortGroupBiasResult | CellBiasResult
    if sampling.method == "short_group":
        result = bbp_short_group_correction(
            envelope.geometry,
            inputs.z,
            theta,
            inputs=inputs.design,
            variance=inputs.variance,
            options=options,
        )
        pairs = 0
    else:
        cell_pairs = cluster_cell_pairs(sampling.codes.view(), bounds=kernel_bounds)
        result = bbp_cell_correction(
            envelope.geometry,
            inputs.z,
            theta,
            inputs=inputs.design,
            pairs=cell_pairs,
            variance=inputs.variance,
            options=options,
        )
        pairs = cell_pairs.n_pairs
    full = envelope.geometry.expand(theta - result.bias_theta)
    effects = full[envelope.geometry.n_slopes_full :]
    reported_effects = (
        OrderScale(envelope.order.omega, np.ones(effects.size)).coefficients(effects)
        if effects.size
        else effects
    )
    point = BiasPoint(
        FloatStorage.store(result.uncorrected),
        FloatStorage.store(result.bias),
        FloatStorage.store(result.corrected),
        FloatStorage.store(reported_effects),
    )
    return CancelledBias(
        sampling.request,
        sampling.method,
        sampling.name,
        sampling.n_units,
        point,
        (
            ("total_bytes", result.workspace_plan.total_bytes + capture_bytes),
            ("work_units", result.workspace_plan.work_units + capture_work),
            ("binding_reporting_bytes", capture_bytes),
            ("binding_reporting_work_units", capture_work),
            ("ordered_pairs", pairs),
            ("max_bytes", inputs.bounds.max_bytes),
            ("max_work_units", inputs.bounds.max_work_units),
        ),
    )


def _wz_triplet(
    result: CoefficientBias, theta: FloatArray, order: OrderScale, name: str
) -> WZBias:
    """Map slope bias once and subtract it from the original reported root."""
    uncorrected, bias = (
        order.coefficients(theta),
        order.coefficients(result.bias.view()),
    )
    with np.errstate(over="raise", invalid="raise"):
        corrected = uncorrected - bias
    if not np.isfinite(corrected).all():
        raise FloatingPointError("WZ corrected coefficient is not finite")
    return WZBias(
        name,
        WZBiasPoint(
            FloatStorage.store(uncorrected),
            FloatStorage.store(bias),
            FloatStorage.store(corrected),
        ),
        result,
    )


def apply_joint_wz_bias(
    certificate: FEPointCertificate | None,
    controls: WZControls,
    cluster: ClusterCodes,
) -> WZBias:
    """Use the same certified joint point before optional inference discard."""
    if certificate is None or not certificate.accepted:
        return WZBias(cluster.name, None)
    point = certificate.point
    sample, source = point.sample, controls.source
    coefficient_ppml(
        sample.omega, point.variance, weights=sample.data.weights is not None
    )
    design, group = sample.design, sample.grouping
    if design is None or group is None:
        raise unavailable(
            "WZ coefficients require current original slopes and grouping"
        )
    panel = layout(
        source,
        cluster.name,
        design.matrix.shape,
        controls.absorbed,
        controls.bounds,
        workspace=workspace,
    )
    order = sample.order_scale
    assert order is not None
    if (
        sample.positive_only
        or sample.conditioned
        or any(v is not None for v in sample.reasons)
        or sample.data.row_labels != source.row_labels
        or not same_bits(sample.positions, source.grouping.positions)
        or certificate.geometry is None
        or certificate.geometry.point is not point
    ):
        raise unavailable(
            "WZ coefficients require the original unconditioned joint root"
        )
    for actual, original in zip(group.blocks, source.grouping.blocks, strict=True):
        if (actual.name, actual.levels) != (
            original.name,
            original.levels,
        ) or not same_bits(actual.codes, original.codes):
            raise unavailable("current declared FE source labels or codes changed")
    bind_request(source, cluster)
    # Fresh point/base/index ownership without another solve or certification.
    assert isinstance(point.variance, PowerVariance)
    fresh = admit_point(
        WholePoint(sample, point.variance, point.record), controls.absorbed
    )
    if any(
        not same_bits(a, b)
        for a, b in (
            (fresh.theta, point.theta),
            (fresh.base, point.base),
            (fresh.q, point.q),
        )
    ):
        raise unavailable(
            "current joint point differs from its uncorrected certificate"
        )
    with np.errstate(over="raise", invalid="raise", under="raise"):
        mu = np.exp(fresh.q)
    rows = capture_rows(
        panel,
        design.matrix,
        sample.data.y,
        mu,
        sample.active_names,
        route="joint_coefficient_current",
    )
    result = evaluate(rows, controls.options, controls.bounds, plan=panel.plan)
    return _wz_triplet(result, fresh.theta, order, cluster.name)


def apply_cancelled_wz_bias(
    final: CertifiedCancelledPoint,
    capture: CapturedFit,
    order: OrderScale,
    controls: WZControls,
    cluster: ClusterCodes,
) -> WZBias:
    """Read the genuine cancelled point while it is live, without a companion fit."""
    certificate, state = final.certificate, final.state
    if not certificate.accepted or state is None:
        return WZBias(cluster.name, None)
    coefficient_ppml(order.omega, certificate.variance)
    geometry = capture.admission.geometry
    n, k = geometry.structure.n_rows, geometry.n_slopes_full
    p, d = geometry.n_columns_full, geometry.n_identified
    extra = (
        admission_budget(geometry.design).prospective_bytes,
        4096
        * (
            n * (p + 1) * (d + 1)
            + (p + 1) ** 3
            + (capture.admission.controls.max_sweeps + 1) * n * (k + 4)
        ),
    )
    panel = layout(
        controls.source,
        cluster.name,
        (n, k),
        controls.absorbed,
        controls.bounds,
        extra=extra,
        workspace=workspace,
    )
    bind_request(controls.source, cluster)
    if (
        state is None
        or state.design is not geometry
        or geometry.n_slopes_identified != k
    ):
        raise unavailable("all current original WZ slopes must remain identified")
    if capture.obs_weights is not None and not np.all(capture.obs_weights == 1):
        raise unavailable("only unit equation multipliers and no statistical weights")
    pair = next(b for b in controls.source.grouping.blocks if b.name == cluster.name)
    structure = geometry.structure
    if (
        not same_bits(pair.codes, structure.cancel_codes)
        or pair.levels != structure.cancel_levels
    ):
        raise unavailable("current cancellation must own the exact pair source")
    _cancelled_sources(controls.source, capture, controls.bounds.max_bytes)
    if not same_bits(certificate.gamma, state.theta):
        raise unavailable("current cancelled point differs from its certificate")
    fresh = capture.equations(certificate.gamma, variance=certificate.variance)
    if not same_bits(fresh.rows().q, state.rows().q):
        raise unavailable("current cancelled rows differ from the original root")
    with np.errstate(over="raise", invalid="raise", under="raise"):
        mu = fresh.rows().pi * fresh.full_state.group_t[structure.cancel_codes]
    rows = capture_rows(
        panel,
        capture.admission.raw_slopes,
        capture.z,
        mu,
        capture.admission.check.source_columns,
        route="pair_cancelled_coefficient_current",
    )
    full = geometry.expand(certificate.gamma)
    result = evaluate(rows, controls.options, controls.bounds, plan=panel.plan)
    return _wz_triplet(result, full[:k], order, cluster.name)
