"""Compose distinct BBP sampling and published WZ coefficient corrections.

The existing normal cancelled domain has unit multipliers and observed one-way
inference. Cell integration here covers declared clusters made of complete
cancelled groups. Crossing/multiway cell pairing and joint-cell fitting remain
open mappings, not asserted sampling schemes: a covariance request of that kind
does not declare the correction's sampling, which stays the cancelled groups
(ruling 82). No correction work runs for False.
WZ uses the certified uncorrected PPML root and its own reserved analytical
operator on present directed pair-date cells. Version 0.1 admits one country set
with each retained pair present at every observed date or absent altogether.
A pair-cancelled fit hands the same point's covariance
information to its coefficient (owner D1/D2); nothing is cached across fits.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import FloatStorage, IndexArray, IndexStorage
from powermean.fe.geometry_checks import GeometryInputs
from powermean.formula.frame import Design
from powermean.inference.convention import ClusterCodes, IntegerArray
from powermean.inference.correction.budget import BiasBounds, enforce
from powermean.inference.correction.cell import CellBiasResult, bbp_cell_correction
from powermean.inference.correction.pairs import cluster_cell_pairs
from powermean.inference.correction.selector import BiasEvalOptions
from powermean.inference.correction.short_group import (
    ShortGroupBiasResult,
    ShortGroupFitPoint,
    ShortGroupReport,
    _short_group_correction,
)
from powermean.inference.wz_binding import (
    WZCells,
    WZSource,
    capture_rows,
    layout,
    omitted_slopes,
    recognize,
    reserve_validation,
)
from powermean.inference.wz_coefficient import CoefficientBias, evaluate, workspace
from powermean.inference.wz_covariance import (
    WZControls,
    bind_request,
    ppml,
)
from powermean.inference.wz_operator import Information
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
from powermean.solve.cancelled_records import CancelledCertificate
from powermean.solve.profiled_trigger import WholePoint
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import _admitted_check


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


def wz_formula_intercept(design: Design) -> int | None:
    """Read the literal intercept's original position from the fitted spec once.

    Formulaic distinguishes its literal ``1`` factor from a lookup of a data
    column named Intercept. This reads structured fitted terms only, never rows
    or formula text; the scalar travels with the existing fit-owned WZ source.
    """
    for term, columns in design.spec.term_slices.items():
        if (
            len(term.factors) == 1
            and term.factors[0].eval_method.value == "literal"
            and term.factors[0].expr == "1"
        ):
            return int(columns.start)
    return None


def _self_pairs(cells: WZCells) -> bool:
    """Whether some row's origin label equals its destination label."""
    found = cells.found
    position = {label: k for k, label in enumerate(found.origins)}
    same = np.fromiter(
        (position.get(label, -1) for label in found.destinations),
        dtype=np.intp,
        count=len(found.destinations),
    )
    return bool(np.any(same[cells.destination] == cells.origin))


def wz_pair_name(
    source: WZSource,
    shape: tuple[int, int],
    bounds: AttemptBounds,
    *,
    selected: str | None = None,
) -> str:
    """Validate a selected pair role, or identify a unique unnamed pair role.

    Each candidate uses layout's present-cell support recognition, never a
    claimed label or flag. An explicit sampling/cancellation role is checked on
    its own. Without one, a single structural candidate is taken. A legacy
    complete/off-diagonal balanced role can also be recognized when it alone
    covers every one of its pair-date cells. Several time-complete candidates
    (including rectangular interaction triangles) require a declared pair role;
    equality or overlap of their labels cannot resolve that ambiguity.
    """
    reserve_validation(shape, bounds)
    matches: list[tuple[str, WZCells]] = []
    for block in source.grouping.blocks:
        if selected is not None and block.name != selected:
            continue
        try:
            cells = recognize(source, block.name)
        except SpecificationError:
            continue
        if cells.origin.size == shape[0]:
            matches.append((block.name, cells))
    if len(matches) == 1:
        return matches[0][0]
    balanced = [
        (name, cells)
        for name, cells in matches
        if shape[0] == cells.found.pairs * len(cells.found.dates)
    ]
    if len(balanced) == 1:
        name, cells = balanced[0]
        found = cells.found
        # Preserve the original complete/off-diagonal criteria only AFTER the
        # support has singled out a time-complete role, independently of labels.
        nn = len(found.origins)
        complete = found.pairs == nn * nn
        if (
            set(found.origins) == set(found.destinations)
            and (complete or found.pairs == nn * (nn - 1))
            and (complete or not _self_pairs(cells))
        ):
            return name
    raise unavailable(
        "WZ coefficient requires a unique directed pair role on present "
        "pair-date cells; name the pair cluster explicitly"
    )


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


def _units_by_rows(
    cancelled: IntegerArray, codes: IntegerArray
) -> tuple[IndexArray, int, int] | None:
    """Units numbered by first appearance, with the numbers of groups and units.

    One walk over the rows; ``None`` when a cancelled group meets two units.
    """
    forward: dict[int, int] = {}
    units: dict[int, int] = {}
    owned_codes = np.empty(cancelled.size, dtype=np.intp)
    for row, (group, unit) in enumerate(zip(cancelled, codes, strict=True)):
        g, u = int(group), int(unit)
        if g in forward and forward[g] != u:
            return None
        forward[g] = u
        if u not in units:
            units[u] = len(units)
        owned_codes[row] = units[u]
    return owned_codes, len(forward), len(units)


def _units(
    cancelled: IntegerArray, codes: IntegerArray
) -> tuple[IndexArray, int, int] | None:
    """The result of ``_units_by_rows``, by sorting where both are integer vectors.

    A group meets two units exactly when the unit of its first row is not the
    unit of each of its rows, and the first appearances of the sorted distinct
    units give the numbering of the walk. Other inputs take the walk.
    """
    if cancelled.ndim != 1 or cancelled.dtype.kind != "i" or codes.dtype.kind != "i":
        return _units_by_rows(cancelled, codes)
    _, index, group = np.unique(cancelled, return_index=True, return_inverse=True)
    if not np.array_equal(codes[index][group], codes):
        return None
    _, first, unit = np.unique(codes, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.intp)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size, dtype=np.intp)
    return rank[unit], index.size, first.size


def bias_sampling(
    request: BiasRequest,
    cancelled: IntegerArray,
    name: str,
    inference: InferenceRequest | None,
    *,
    bounds: BiasBounds,
) -> BiasSampling | None:
    """Check current partitions before solver work; no pair enumeration occurs here.

    The sampling of the correction is the one-way cluster request when it declares
    one: the cancelled groups, or under ``"cell"`` a partition of complete cancelled
    groups.  Any other covariance request (robust, named HC, several dimensions, a
    dimension that crosses the groups, or one coarser than them beside
    ``True``/``"short_group"``) declares none: the correction is that of the fit
    without ``vcov``, at the cancelled-group sampling, and the request is the
    covariance at the uncorrected root, as after the fit (owner's ruling 82).
    """
    if request is None:
        return None
    if request == "wz":
        raise unsupported_bias(
            "published WZ requires its distinct coefficient operator"
        )
    enforce(256 * cancelled.size, 128 * cancelled.size, bounds)
    found, sampling_name = None, name
    if (
        inference is not None
        and inference.kind == "cluster"
        and len(inference.clusters) == 1
    ):
        cluster = inference.clusters[0]
        if cluster.codes.shape != cancelled.shape:
            raise ValueError("BBP dependence must align with the current fit rows")
        found = _units(cancelled, cluster.codes)
        if found is not None and (request == "cell" or found[1] == found[2]):
            sampling_name = cluster.name
        else:
            found = None
    if found is None:
        found = _units(cancelled, cancelled)
    assert found is not None  # the cancelled groups partition the rows
    owned_codes, _, n_units = found
    return BiasSampling(
        request,
        "cell" if request == "cell" else "short_group",
        sampling_name,
        IndexStorage.store(owned_codes),
        n_units,
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


def _same_float_rows(values: FloatArray, captured: FloatArray) -> bool:
    """Compare binary64 rows bit for bit without copying either vector."""
    return (
        type(values) is np.ndarray
        and values.dtype == np.float64
        and np.array_equal(values.view(np.uint64), captured.view(np.uint64))
    )


def _matching_short_group(
    point: ShortGroupFitPoint | None,
    envelope: RetainedCancelledEnvelope,
    certificate: CancelledCertificate,
    inputs: BiasFitInputs,
) -> ShortGroupFitPoint | None:
    """Bind the selected artifacts to the same built-in unit-weight PPML inputs."""
    variance = inputs.variance
    if (
        point is None
        or type(variance) is not PowerVariance
        or variance.tau != 1.0
        or type(certificate.variance) is not PowerVariance
        or certificate.variance != variance
        or envelope._order.omega != 1.0
        or certificate.controls.rcond_min != inputs.rcond_min
    ):
        return None
    capture, state = point.capture, point.state
    if (
        state.design is not envelope.geometry
        or capture.obs_weights is not None
        or not same_bits(state.theta, certificate.gamma)
        or not _same_float_rows(inputs.z, capture.z)
        or not _same_float_rows(inputs.offset, capture.centered_offset)
        or state.geometry_check
        is not _admitted_check(envelope.geometry, inputs.design, capture.admission)
    ):
        return None
    return point


def apply_cancelled_bias(
    envelope: RetainedCancelledEnvelope,
    sampling: BiasSampling | None,
    inputs: BiasFitInputs,
    *,
    fit_point: ShortGroupFitPoint | None = None,
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
    # temporaries (including any selected-columns null-orthogonal projection) and
    # four small owned outputs in addition to the kernel plan.
    # These are conservative managed arrays/visits, not measured OS memory/work.
    norm_bytes, norm_work = geometry.normalization_workspace()
    capture_bytes = 8 * (16 * n + 32 * p + 32 * d + 64 * k) + norm_bytes
    capture_work = 128 * (n + p * d + p + k) + norm_work
    enforce(capture_bytes + 1, capture_work + 1, inputs.bounds)
    kernel_bounds = BiasBounds(
        inputs.bounds.max_bytes - capture_bytes,
        inputs.bounds.max_work_units - capture_work,
    )
    certificate = envelope.certificate
    theta = certificate.gamma
    # The fit's one geometry check stands for its own geometry and inputs; a
    # fit that kept no capture leaves the kernel to check afresh.
    capture = envelope._capture
    options = BiasEvalOptions(
        order=envelope.order,
        centered_offset=inputs.offset,
        bounds=kernel_bounds,
        rcond_min=inputs.rcond_min,
        admitted=None if capture is None else capture.admission,
    )
    result: ShortGroupBiasResult | ShortGroupReport | CellBiasResult
    if sampling.method == "short_group":
        result = _short_group_correction(
            envelope.geometry,
            inputs.z,
            theta,
            inputs=inputs.design,
            variance=inputs.variance,
            options=options,
            fit_point=_matching_short_group(fit_point, envelope, certificate, inputs),
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
    full = envelope.geometry.normalized_full(theta - result.bias_theta)
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
    *,
    shared: Information | None = None,
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
        controls.bounds,
        workspace=workspace,
    )
    order = sample.order_scale
    assert order is not None
    if (
        sample.positive_only
        or sample.conditioned
        or omitted_slopes(
            sample.data.column_names, sample.reasons, source.formula_intercept
        )
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
    coordinates = controls.joint_coordinates(panel)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        mu = np.exp(fresh.q)
    rows = capture_rows(
        panel,
        design.matrix,
        sample.data.y,
        mu,
        sample.active_names,
        route="joint_coefficient_current",
        coordinates=coordinates,
    )
    result = evaluate(
        rows, controls.options, controls.bounds, plan=coordinates.plan, shared=shared
    )
    return _wz_triplet(result, fresh.theta, order, cluster.name)


def apply_cancelled_wz_bias(
    final: CertifiedCancelledPoint,
    capture: CapturedFit,
    order: OrderScale,
    controls: WZControls,
    cluster: ClusterCodes,
    *,
    shared: Information | None = None,
) -> WZBias:
    """Read the genuine cancelled point while it is live, without a companion fit.

    ``shared`` is this fit's covariance information from its last certification;
    the coefficient reuses it only for freshly captured rows with the identical
    binding digest (the same accepted point), else forms its own.
    """
    certificate, state = final.certificate, final.state
    if not certificate.accepted or state is None:
        return WZBias(cluster.name, None)
    coefficient_ppml(order.omega, certificate.variance)
    geometry = capture.admission.geometry
    n, k = geometry.structure.n_rows, geometry.n_slopes_full
    panel = layout(
        controls.source, cluster.name, (n, k), controls.bounds, workspace=workspace
    )
    if state.design is not geometry or geometry.n_slopes_identified != k:
        raise unavailable("all current original WZ slopes must remain identified")
    if not same_bits(certificate.gamma, state.theta):
        raise unavailable("current cancelled point differs from its certificate")
    fresh = capture.equations(certificate.gamma, variance=certificate.variance)
    if not same_bits(fresh.rows().q, state.rows().q):
        raise unavailable("current cancelled rows differ from the original root")
    rows, coordinates = controls.cancelled_rows(
        fresh,
        capture,
        cluster,
        route="pair_cancelled_coefficient_current",
        panel=panel,
    )
    full = geometry.expand(certificate.gamma)
    result = evaluate(
        rows, controls.options, controls.bounds, plan=coordinates.plan, shared=shared
    )
    return _wz_triplet(result, full[:k], order, cluster.name)
