"""Sample-bound absorbed inference structural preflight."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from powermean.data.fingerprint import sample_fingerprint
from powermean.data.inference_conventions import PsdTreatment
from powermean.data.request_values import OMITTED
from powermean.data.sample_fe_state import FESample
from powermean.errors import SpecificationError
from powermean.fe.certificate_basis import CertificateBasis, owned_basis
from powermean.inference.absorbed_code_tuples import CachedCodeTuples
from powermean.inference.absorbed_rank import AbsorbedCount, absorbed_count
from powermean.inference.contributions import statistical_count
from powermean.inference.convention import ClusterCodes, Covariance, _active_codes
from powermean.inference.hc import HCPreflight, hc_preflight
from powermean.inference.psd import FrameScales, Sequence, column_scales
from powermean.inference.ssc import (
    AppliedConvention,
    ResolvedConvention,
    apply_convention,
    convention_counts,
    resolve_convention,
)
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleDetails,
    FEScaleInput,
    IntegerArray,
)
from powermean.systems.joint import SamplingUnits


def _positions(mask: NDArray[np.bool_]) -> IndexArray:
    """The positions at which ``mask`` holds, read-only, in bytes of their own."""
    found: IndexArray = np.frombuffer(np.flatnonzero(mask).tobytes(), dtype=np.intp)
    return found


def sampling_units(sample: FESample) -> SamplingUnits:
    """The sample's final rows as sampling units: positive and zero weights apart."""
    active = sample.data.weight_values > 0
    return SamplingUnits(_positions(active), _positions(~active), sample.count)


@dataclass(frozen=True, eq=False)
class AbsorbedFrameRecord:
    """Owned frame record of one eigenvalue adjustment at an absorbed point."""

    sequence: Sequence
    convention: str
    scales: FloatArray
    deviations: FloatArray
    floored: NDArray[np.bool_]
    count: int
    fingerprint: str
    constant: bool
    constant_total: float | None
    constant_diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        """Own all arrays."""
        object.__setattr__(self, "scales", _immutable(self.scales))
        object.__setattr__(self, "deviations", _immutable(self.deviations))
        object.__setattr__(
            self,
            "floored",
            np.frombuffer(np.asarray(self.floored, dtype=np.bool_).tobytes(), np.bool_),
        )


@dataclass(frozen=True, eq=False)
class AbsorbedPreflight:
    """Immutable structural per-fit preflight bound by value to a sample and request."""

    request_identity: tuple[object, ...]
    nobs: int
    active_bytes: bytes
    omitted_bytes: bytes
    positions_bytes: bytes
    weights_bytes: bytes
    frequency: bool
    design_shape: tuple[int, int]
    design_bytes: bytes
    scale_bytes: bytes
    omega: float
    basis_kept: int
    block_codes_bytes: tuple[bytes, ...]
    basis_positions_bytes: tuple[bytes, ...]
    units: SamplingUnits
    count: int
    rank: int
    resolved: ResolvedConvention
    applied: AppliedConvention
    absorbed: AbsorbedCount | None
    active_clusters: tuple[tuple[str, IntegerArray], ...]
    hc_preflight: HCPreflight | None = None
    code_tuples: CachedCodeTuples | None = None
    psd: PsdTreatment | None = None
    frame_scales: FrameScales | None = None
    fingerprint: str | None = None


def _empty_preflight(
    sample: FESample,
    request: CovarianceRequest,
    units: SamplingUnits,
    *,
    resolved: ResolvedConvention,
    weights: FloatArray,
    frequency: bool,
    active_clusters: tuple[tuple[str, IntegerArray], ...],
    prepare_basis: Callable[[FESample], CertificateBasis] | None = None,
) -> CertificateBasis:
    """Validate the slope-empty covariance domain before any solve."""
    assert sample.grouping is not None
    active = units.identifiers
    basis0 = owned_basis(sample) if prepare_basis is None else prepare_basis(sample)
    if prepare_basis is not None and basis0.grouping is not sample.grouping:
        raise ValueError("the prepared basis belongs to another sample grouping")
    count0 = statistical_count(weights[active], frequency=frequency)
    if request.kind == "hc":
        assert request.method is not None
        counts0 = convention_counts(count0, 0, basis0.n_kept, None)
        apply_convention(resolved, counts0, kind="robust", method=request.method)
    else:
        groups0 = [int(np.unique(c).size) for _, c in active_clusters]
        if request.kind == "cluster" and any(g < 2 for g in groups0):
            raise ValueError("the cluster factor G/(G - 1) needs at least two clusters")
        blocks0 = tuple((b.name, b.codes[active]) for b in sample.grouping.blocks)
        absorbed0 = absorbed_count(blocks0, [c for _, c in active_clusters])
        counts0 = convention_counts(count0, 0, basis0.n_kept, absorbed0, groups0)
        apply_convention(resolved, counts0, kind=request.kind)
    return basis0


def _psd_request(request: CovarianceRequest) -> None:
    """Existing PSD option/form conflicts, before resolving conventions."""
    if request.psd is not OMITTED:
        signed = len(request.clusters) >= 2 and (
            request.multiway is OMITTED or request.multiway == "cgm"
        )
        if not signed:
            second = (
                "one cluster dimension"
                if len(request.clusters) < 2
                else "multiway='ddg'"
            )
            raise SpecificationError(
                "specification.conflicting",
                {"first": f"psd={request.psd!r}", "second": second},
            )


def _active_request(
    sample: FESample, request: CovarianceRequest
) -> tuple[
    SamplingUnits, IndexArray, FloatArray, bool, tuple[tuple[str, IntegerArray], ...]
]:
    """Bind active units and cluster classifications before any solve."""
    units = sampling_units(sample)
    active = units.identifiers
    weights = sample.data.weight_values
    frequency = sample.data.weights is not None and sample.data.weights.frequency
    if request.kind == "cluster" and (
        request.positions is None
        or not request.clusters
        or request.positions.shape != (sample.count,)
        or (not np.array_equal(request.positions, sample.positions))
    ):
        raise ValueError("the cluster codes are not aligned with the sample's rows")
    active_clusters = tuple(
        (
            (name, _active_codes(ClusterCodes(name, codes), units))
            for name, codes in request.clusters
        )
    )
    return (units, active, weights, frequency, active_clusters)


def preflight_absorbed_scale(
    sample: FESample, request: CovarianceRequest
) -> AbsorbedPreflight | None:
    """Validate structural covariance requirements and bind current sample facts."""
    return _preflight(sample, request, None)[0]


def _preflight_numeric(
    sample: FESample, request: CovarianceRequest,
    prepare_basis: Callable[[FESample], CertificateBasis],
    prepare_count: Callable[[CertificateBasis, list[IntegerArray]], AbsorbedCount | None]
    | None = None,
) -> tuple[AbsorbedPreflight | None, CertificateBasis]:
    """Prepare this operation's basis at the ORIGINAL ordered preflight site."""
    return _preflight(sample, request, prepare_basis, prepare_count)


def _preflight(
    sample: FESample, request: CovarianceRequest,
    prepare_basis: Callable[[FESample], CertificateBasis] | None,
    prepare_count: Callable[[CertificateBasis, list[IntegerArray]], AbsorbedCount | None]
    | None = None,
) -> tuple[AbsorbedPreflight | None, CertificateBasis]:
    """One preflight implementation; the public entry always constructs afresh."""
    if type(sample) is not FESample:
        raise TypeError("sample must be an FESample")
    if type(request) is not CovarianceRequest:
        raise TypeError("request must be a CovarianceRequest")
    if sample.grouping is None:
        raise ValueError("an absorbed sample needs fixed-effect blocks")
    units, active, weights, frequency, active_clusters = _active_request(
        sample, request
    )
    _psd_request(request)
    inherited = (
        resolve_convention(request.convention) if request.ssc_inherited else None
    )
    resolved = resolve_convention(request.ssc, inherited)
    if sample.design is None or sample.active.size == 0:
        basis = _empty_preflight(
            sample,
            request,
            units,
            resolved=resolved,
            weights=weights,
            frequency=frequency,
            active_clusters=active_clusters,
            prepare_basis=prepare_basis,
        )
        return None, basis
    design = sample.design
    k = int(sample.active.size)
    basis = owned_basis(sample) if prepare_basis is None else prepare_basis(sample)
    if prepare_basis is not None and basis.grouping is not sample.grouping:
        raise ValueError("the prepared basis belongs to another sample grouping")
    dimension = k + basis.n_kept
    count = statistical_count(weights[active], frequency=frequency)
    if request.kind == "hc":
        assert request.method is not None
        hc_pre = hc_preflight(
            design.matrix,
            weights,
            request.method,
            resolved,
            frequency=frequency,
            basis=basis,
        )
        code_tuples = CachedCodeTuples(basis)
        applied = hc_pre.applied
        absorbed: AbsorbedCount | None = hc_pre.absorbed
        psd: PsdTreatment | None = None
        frame_scales: FrameScales | None = None
        fp: str | None = None
    else:
        hc_pre = None
        code_tuples = None
        dimensions = [c for _, c in active_clusters]
        absorbed = (
            prepare_count(basis, dimensions)
            if (prepare_count is not None and active.size == sample.count
                and absorbed_count is _ABSORBED_COUNT) else None
        )
        if absorbed is None:
            blocks = tuple((b.name, b.codes[active]) for b in sample.grouping.blocks)
            absorbed = absorbed_count(blocks, dimensions)
        groups = [int(np.unique(c).size) for _, c in active_clusters]
        if request.kind == "cluster" and any(g < 2 for g in groups):
            raise ValueError("the cluster factor G/(G - 1) needs at least two clusters")
        counts = convention_counts(count, k, dimension, absorbed, groups)
        applied = apply_convention(resolved, counts, kind=request.kind)
        signed = len(request.clusters) >= 2 and (
            request.multiway is OMITTED or request.multiway == "cgm"
        )
        if signed:
            psd = (
                ("cgm" if request.psd == "cgm" else "none")
                if request.psd is not OMITTED
                else applied.psd
            )
        else:
            psd = None
        if psd == "cgm":
            frame_scales = column_scales(design.unscaled()[active])
            fp = sample_fingerprint(sample.data)
        else:
            frame_scales = None
            fp = None
    return AbsorbedPreflight(
        request_identity=request.identity(),
        nobs=sample.count,
        active_bytes=sample.active.tobytes(),
        omitted_bytes=units.inactive.tobytes(),
        positions_bytes=sample.positions.tobytes(),
        weights_bytes=weights.tobytes(),
        frequency=frequency,
        design_shape=design.matrix.shape,
        design_bytes=design.matrix.tobytes(),
        scale_bytes=design.scale.tobytes(),
        omega=sample.omega,
        basis_kept=basis.n_kept,
        block_codes_bytes=tuple(b.codes.tobytes() for b in basis.grouping.blocks),
        basis_positions_bytes=tuple(p.tobytes() for p in basis.positions),
        units=units,
        count=count,
        rank=dimension,
        resolved=resolved,
        applied=applied,
        absorbed=absorbed,
        active_clusters=active_clusters,
        hc_preflight=hc_pre,
        code_tuples=code_tuples,
        psd=psd,
        frame_scales=frame_scales,
        fingerprint=fp,
    ), basis


def default_details(
    pre: AbsorbedPreflight, scale_input: FEScaleInput, covariance: Covariance
) -> FEScaleDetails:
    """Reuse structural records without changing legacy covariance arithmetic."""
    rep_cov: FloatArray | None = None
    error: str | None = None
    try:
        rep_cov = scale_input.order.covariance(covariance.covariance)
    except FloatingPointError as failure:
        error = str(failure)
    return FEScaleDetails(
        bread="observed",
        requested_ssc=scale_input.request.ssc,
        resolved_preset=pre.resolved.preset or "custom",
        ssc_source=pre.resolved.source,
        factor_owner="convention",
        units=pre.units,
        count=pre.count,
        rank=pre.rank,
        applied=pre.applied,
        absorbed=pre.absorbed,
        reported_covariance=rep_cov,
        covariance_error=error,
    )


def _preflight_matches(pre: AbsorbedPreflight, scale_input: FEScaleInput) -> bool:
    """Verify that ``pre`` matches ``scale_input`` by exact value and byte bindings."""
    geometry = scale_input.geometry
    sample = geometry.point.sample
    basis = geometry.basis
    return (
        pre.request_identity == scale_input.request.identity()
        and pre.nobs == sample.count
        and (pre.active_bytes == sample.active.tobytes())
        and (pre.omitted_bytes == _positions(~(geometry.rows.weights > 0)).tobytes())
        and (pre.positions_bytes == sample.positions.tobytes())
        and (pre.weights_bytes == geometry.rows.weights.tobytes())
        and (pre.frequency == geometry.rows.frequency)
        and (pre.design_shape == geometry.design.shape)
        and (pre.design_bytes == geometry.design.tobytes())
        and (pre.scale_bytes == scale_input.order.scale.tobytes())
        and (pre.omega == scale_input.order.omega)
        and (pre.basis_kept == basis.n_kept)
        and (
            pre.block_codes_bytes
            == tuple(b.codes.tobytes() for b in basis.grouping.blocks)
        )
        and (pre.basis_positions_bytes == tuple(p.tobytes() for p in basis.positions))
    )


_ABSORBED_COUNT = absorbed_count
