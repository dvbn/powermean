"""Value admission of current absorbed inference evidence.

The closed covariance kernels remain the numerical owners. Here their carried
records must agree with the actual request, fresh geometry and report map; a
method name or a covariance shape is not a substitute for those bindings.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from powermean.data.inference_conventions import ConventionValues
from powermean.data.request_values import Omitted
from powermean.data.sample_fe_state import same_bits
from powermean.model.order_scale import OrderScale
from powermean.solve.absorbed_certificate_geometry import FEGeometry
from powermean.solve.absorbed_scale_details import FEScaleDetails, record_fields
from powermean.solve.absorbed_scale_records import CovarianceRecord
from powermean.solve.absorbed_scale_request import CovarianceRequest
from powermean.solve.reported import ReportedScale

STALE = "the scale result is not bound to this input"
APPLIED = (
    "convention",
    "counts",
    "kind",
    "method",
    "factor",
    "components",
    "cluster_count",
    "reference",
    "psd",
)
COUNTS = (
    "count",
    "n_slopes",
    "rank",
    "k_reference",
    "k_factor",
    "nested_adjustment",
    "clusters",
)
RESOLVED = ("requested", "values", "source", "origin")
FACTOR = ("numerator", "denominator", "formula", "owner")


class ScaleContext(Protocol):
    """The fresh input facts read by admission, with no upward layer import."""

    @property
    def geometry(self) -> FEGeometry:
        """Fresh admitted geometry."""
        ...

    @property
    def request(self) -> CovarianceRequest:
        """Original requested covariance."""
        ...

    @property
    def order(self) -> OrderScale:
        """Actual report map."""
        ...


def _convention(value: object) -> ConventionValues:
    """The already settled preset values, or an exact custom-value record."""
    if type(value) is ConventionValues:
        return value
    presets = {
        "stata": ConventionValues("N/(N-1)", "G/(G-1)", "min", "normal", "cgm"),
        "none": ConventionValues("none", "none", "min", "normal", "none"),
        "regression": ConventionValues(
            "N/(N-K*)", "(N-1)/(N-K*)*G/(G-1)", "min", "t", "cgm"
        ),
    }
    if not isinstance(value, str) or value not in presets:
        raise ValueError(STALE)
    return presets[value]


def _ssc(
    details: FEScaleDetails, request: CovarianceRequest, applied: dict[str, object]
) -> None:
    """Requested presence, resolved values/source and owner must agree."""
    resolved = record_fields(applied["convention"], RESOLVED)
    omitted = isinstance(request.ssc, Omitted)
    value = request.convention if omitted else request.ssc
    source = (
        "inherited"
        if omitted and request.ssc_inherited
        else ("default" if omitted else "explicit")
    )
    expected = _convention(value)
    origin = "custom" if type(value) is ConventionValues else "preset"
    same_request = (
        isinstance(details.requested_ssc, Omitted)
        if omitted
        else details.requested_ssc == request.ssc
    )
    same_resolved_request = (
        isinstance(resolved["requested"], Omitted)
        if omitted
        else resolved["requested"] == request.ssc
    )
    if not (
        same_request
        and same_resolved_request
        and details.ssc_source == source
        and resolved["source"] == source
        and resolved["values"] == expected
        and resolved["origin"] == origin
    ):
        raise ValueError(STALE)
    if not isinstance(resolved["values"], ConventionValues):
        raise TypeError("the applied convention needs ConventionValues")
    label = next(
        (
            name
            for name in ("stata", "none", "regression")
            if _convention(name) == expected
        ),
        "custom",
    )
    if details.resolved_preset != label:
        raise ValueError(STALE)
    if applied["kind"] != ("robust" if request.kind == "hc" else request.kind):
        raise ValueError(STALE)
    if applied["method"] != request.method:
        raise ValueError(STALE)
    if details.factor_owner != (
        "method" if request.kind == "hc" or request.adjustment == "wz" else "convention"
    ):
        raise ValueError(STALE)


def _counts(
    details: FEScaleDetails,
    covariance: CovarianceRecord,
    context: ScaleContext,
    applied: dict[str, object],
) -> dict[str, object]:
    """Counts, units and exact full rank are those of this fresh sample."""
    geometry = context.geometry
    sample = geometry.point.sample
    active = geometry.rows.weights > 0
    expected_n = (
        sum(int(v) for v in geometry.rows.weights[active].tolist())
        if geometry.rows.frequency
        else int(np.count_nonzero(active))
    )
    counts = record_fields(applied["counts"], COUNTS)
    if not (
        details.units.nobs == sample.count
        and np.array_equal(details.units.identifiers, np.flatnonzero(active))
        and np.array_equal(details.units.inactive, np.flatnonzero(~active))
        and details.count == covariance.count == counts["count"] == expected_n
        and details.rank == counts["rank"] == geometry.dimension
        and counts["n_slopes"] == geometry.n_slopes
    ):
        raise ValueError(STALE)
    if not all(type(counts[n]) is int for n in COUNTS[:-1]):
        raise TypeError("inference counts must be exact integers")
    if details.absorbed is not None:
        absorbed = record_fields(
            details.absorbed, ("blocks", "pairwise", "absorbed", "nested_adjustment")
        )
        number, nested = absorbed["absorbed"], absorbed["nested_adjustment"]
        if type(number) is not int or type(nested) is not int:
            raise TypeError("absorbed convention counts must be exact integers")
        reference = geometry.n_slopes + number
        if (
            counts["k_reference"] != reference
            or counts["k_factor"] != reference + nested
            or counts["nested_adjustment"] != nested
        ):
            raise ValueError(STALE)
    return counts


def _hc(
    details: FEScaleDetails, covariance: CovarianceRecord, context: ScaleContext
) -> None:
    """Full nuisance rank, actual bread, method and covariance must agree."""
    names = (
        "method",
        "covariance",
        "estimate",
        "applied",
        "leverage",
        "multipliers",
        "bread",
        "layout",
        "influence",
    )
    hc = record_fields(details.hc, names)
    leverage = hc["leverage"]
    if not (
        hc["method"] == context.request.method
        and hc["bread"] == context.request.bread
        and hc["layout"] == "absorbed"
        and hc["applied"] == details.applied
        and same_bits(np.asarray(hc["covariance"]), covariance.covariance)
        and getattr(leverage, "rank", None) == context.geometry.dimension
        and getattr(leverage, "p", None) == context.geometry.dimension
        and getattr(leverage, "geometry", None) == "expected"
        and details.multiway is None
        and details.adjustment is None
    ):
        raise ValueError(STALE)


def _integer(value: object) -> int:
    """An exact scalar count, not a coerced self-description."""
    if type(value) is not int:
        raise TypeError("inference metadata counts must be integers")
    return value


def _indices(value: object) -> tuple[int, ...]:
    """An exact component-position tuple."""
    if type(value) is not tuple:
        raise TypeError("component positions must be a tuple")
    return tuple(_integer(item) for item in value)


def _cluster_ratio(applied: dict[str, object], groups: int) -> tuple[int, int]:
    """Check the applied convention's exact integer factor witness."""
    resolved = record_fields(applied["convention"], RESOLVED)
    values = resolved["values"]
    if type(values) is not ConventionValues:
        raise TypeError("the applied convention needs ConventionValues")
    if values.cluster == "none":
        return 1, 1
    if values.cluster == "G/(G-1)":
        return groups, groups - 1
    counts = record_fields(applied["counts"], COUNTS)
    n, k = _integer(counts["count"]), _integer(counts["k_factor"])
    return (n - 1) * groups, (n - k) * (groups - 1)


def _component(
    component: object,
    raw: object,
    request: CovarianceRequest,
    applied: dict[str, object],
) -> None:
    """Bind dimensions, signs, groups, unscaled component and factor ownership."""
    parts = record_fields(
        component,
        (
            "dimensions",
            "positions",
            "sign",
            "n_clusters",
            "factor",
            "covariance",
        ),
    )
    raw_parts = record_fields(
        raw,
        (
            "dimensions",
            "sign",
            "n_clusters",
            "factor",
            "covariance",
        ),
    )
    positions = _indices(parts["positions"])
    if not positions or any(i < 0 or i >= len(request.clusters) for i in positions):
        raise ValueError(STALE)
    if positions != tuple(sorted(set(positions))):
        raise ValueError(STALE)
    signed = request.multiway != "ddg"
    expected_sign = -1 if signed and len(positions) % 2 == 0 else 1
    if not (
        parts["dimensions"] == tuple(request.clusters[i][0] for i in positions)
        and raw_parts["dimensions"] == tuple(f"cluster_{i}" for i in positions)
        and parts["sign"] == raw_parts["sign"] == expected_sign
        and parts["n_clusters"] == raw_parts["n_clusters"]
        and raw_parts["factor"] == 1.0
        and same_bits(
            np.asarray(parts["covariance"]), np.asarray(raw_parts["covariance"])
        )
    ):
        raise ValueError(STALE)
    factor = record_fields(parts["factor"], FACTOR)
    if factor["owner"] != "convention":
        raise ValueError(STALE)
    groups = _integer(parts["n_clusters"])
    if applied["cluster_count"] == "min":
        common = parts["factor"] == applied["factor"]
    else:
        common = (factor["numerator"], factor["denominator"]) == _cluster_ratio(
            applied, groups
        )
    if not common:
        raise ValueError(STALE)


def _multiway(
    details: FEScaleDetails, covariance: CovarianceRecord, request: CovarianceRequest
) -> None:
    """Actual dimensions/form/components are required, not a labelled matrix."""
    names = ("covariance", "estimate", "applied", "form", "dimensions", "components")
    multiway = record_fields(details.multiway, names)
    form = "cgm" if isinstance(request.multiway, Omitted) else request.multiway
    if not (
        request.uses_multiway
        and multiway["form"] == form
        and multiway["dimensions"] == tuple(name for name, _ in request.clusters)
        and multiway["applied"] == details.applied
        and same_bits(np.asarray(multiway["covariance"]), covariance.covariance)
        and getattr(multiway["estimate"], "components", None) is not None
    ):
        raise ValueError(STALE)
    components = multiway["components"]
    raw = record_fields(
        multiway["estimate"],
        (
            "covariance",
            "method",
            "nobs",
            "df_resid",
            "correction",
            "factor",
            "components",
            "eigenvalues",
            "rank",
            "diagnostics",
        ),
    )["components"]
    if (
        type(components) is not tuple
        or type(raw) is not tuple
        or len(components) != len(raw)
    ):
        raise ValueError(STALE)
    applied = record_fields(details.applied, APPLIED)
    for component, raw_component in zip(components, raw, strict=True):
        _component(component, raw_component, request, applied)


def _reported(
    details: FEScaleDetails,
    covariance: CovarianceRecord,
    reported: ReportedScale,
    context: ScaleContext,
) -> None:
    """Range-safe direct errors and exact consistency of the reported payloads.

    The reported and the eigenvalue-adjusted payloads must agree exactly.
    """
    if details.adjustment is not None:
        request = context.request
        raw = context.order.covariance(covariance.covariance)
        adjusted: object = getattr(details.adjustment, "adjusted", None)
        sequence = "legacy" if context.request.convention == "stata" else "modern"
        scales = np.asarray(getattr(details.frame, "scales", None))
        full_scales = np.append(scales, 1.0) if sequence == "legacy" else scales
        if not (
            request.uses_multiway
            and details.psd == "cgm"
            and (isinstance(request.multiway, Omitted) or request.multiway == "cgm")
            and details.frame is not None
            and details.reported_covariance is not None
            and getattr(details.adjustment, "sequence", None) == sequence
            and getattr(details.frame, "sequence", None) == sequence
            and same_bits(np.asarray(getattr(details.adjustment, "raw", None)), raw)
            and same_bits(np.asarray(adjusted), details.reported_covariance)
            and getattr(details.frame, "count", None) == details.units.identifiers.size
            and getattr(details.frame, "constant", None) == (sequence == "legacy")
            and same_bits(
                full_scales,
                np.asarray(getattr(details.adjustment, "scales", None)),
            )
        ):
            raise ValueError(STALE)
        variances = np.diag(np.asarray(adjusted, dtype=np.float64))
    else:
        variances = np.diag(covariance.covariance)
    errors = np.full(variances.shape, np.nan)
    valid = np.isfinite(variances) & (variances >= 0)
    errors[valid] = np.sqrt(variances[valid])
    if details.adjustment is None:
        errors = context.order.standard_errors(errors)
    if not same_bits(reported.standard_errors, errors):
        raise ValueError(STALE)


def check_details_admission(
    details: FEScaleDetails,
    covariance: CovarianceRecord,
    reported: ReportedScale,
    context: ScaleContext,
) -> None:
    """Require original requested composition, not self-labelled covariance shape."""
    if details.bread != context.request.bread:
        raise ValueError("evidence bread does not match request bread")
    applied = record_fields(details.applied, APPLIED)
    _ssc(details, context.request, applied)
    _counts(details, covariance, context, applied)
    if context.request.kind == "hc":
        _hc(details, covariance, context)
    elif context.request.uses_multiway:
        _multiway(details, covariance, context.request)
    elif details.multiway is not None or details.hc is not None:
        raise ValueError(STALE)
    factor = applied["factor"]
    if factor is not None:
        parts = record_fields(factor, FACTOR)
        if parts["owner"] != details.factor_owner or covariance.factor != getattr(
            factor, "value", None
        ):
            raise ValueError(STALE)
    if context.request.adjustment == "wz":
        _wz(details, covariance, context, applied)
    elif details.wz is not None:
        raise ValueError(STALE)
    _reported(details, covariance, reported, context)


def _wz(
    details: FEScaleDetails,
    covariance: CovarianceRecord,
    context: ScaleContext,
    applied: dict[str, object],
) -> None:
    """Bind the published matrix, sole factor and request at the lower boundary."""
    wz = details.wz
    provenance = getattr(wz, "provenance", None)
    groups = covariance.clusters
    factor = record_fields(applied["factor"], FACTOR)
    if not (
        wz is not None
        and groups is not None
        and (factor["numerator"], factor["denominator"], factor["owner"])
        == (groups, groups - 1, "method")
        and getattr(provenance, "pairs", None) == groups
        and getattr(provenance, "observations", None)
        == context.geometry.point.sample.count
        and getattr(provenance, "factor_owner", None) == "published_WZ_only"
        and same_bits(np.asarray(getattr(wz, "adjusted", None)), covariance.covariance)
        and details.adjustment is None
        and details.multiway is None
        and details.hc is None
        and details.psd is None
    ):
        raise ValueError(STALE)
