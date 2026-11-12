"""Owned snapshots of covariance requests and pre-discard initial scale evaluations.

This module defines :class:`RetainedCovarianceRequest`,
:func:`snapshot_covariance_request`, :class:`RetainedInitialScaleCache`, and
:func:`snapshot_initial_scale`.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.data.inference_conventions import ConventionValues, PsdTreatment
from powermean.data.inference_requests import (
    HCMethod,
    Multiway,
    PairAdjustment,
)
from powermean.data.request_values import OMITTED, Omitted
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_input_values import owned_order_scale
from powermean.solve.absorbed_certificate_scale import (
    Bread,
    ComponentRecord,
    CovarianceRecord,
    CovarianceRequest,
    EstimateRecord,
    FEScaleDetails,
    FEScaleResult,
    IntegerArray,
    Kind,
    owned_component,
    owned_covariance,
    owned_details,
    owned_estimate,
    owned_reported,
)
from powermean.solve.absorbed_scale_details import owned_record
from powermean.solve.reported import ReportedScale

__all__ = [
    "RetainedCovarianceRequest",
    "RetainedInitialScaleCache",
    "snapshot_covariance_request",
    "snapshot_initial_scale",
]


@dataclass(frozen=True)
class RetainedCovarianceRequest:
    """Owned snapshot of ``CovarianceRequest`` preserving exact integer dtypes."""

    kind: Kind
    name: str | None = None
    _codes: IntegerArray | None = None
    _positions: IntegerArray | None = None
    convention: str | ConventionValues = "stata"
    _clusters: tuple[tuple[str, IntegerArray], ...] = ()
    method: HCMethod | None = None
    bread: Bread = "observed"
    multiway: Multiway | Omitted = OMITTED
    psd: PsdTreatment | Omitted = OMITTED
    ssc: str | ConventionValues | Omitted = OMITTED
    ssc_inherited: bool = False
    adjustment: PairAdjustment | Omitted = OMITTED

    def __init__(
        self,
        kind: Kind,
        name: str | None = None,
        codes: IntegerArray | None = None,
        positions: IntegerArray | None = None,
        convention: str | ConventionValues = "stata",
        *,
        request: CovarianceRequest | None = None,
    ) -> None:
        """Own ``codes``, ``positions``, and ``clusters`` with exact integer dtypes."""
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", None if name is None else str(name))
        owned_c = _optional(codes)
        owned_p = _optional(positions)
        object.__setattr__(self, "_codes", owned_c)
        object.__setattr__(self, "_positions", owned_p)
        object.__setattr__(self, "convention", owned_record(convention))
        clusters = () if request is None else request.clusters
        if clusters:
            owned_cl = tuple(
                (str(cname), _optional(ccodes))
                for cname, ccodes in clusters
                if ccodes is not None
            )
        elif kind == "cluster" and name is not None and owned_c is not None:
            owned_cl = ((str(name), owned_c),)
        else:
            owned_cl = ()
        object.__setattr__(self, "_clusters", owned_cl)
        object.__setattr__(self, "method", None if request is None else request.method)
        object.__setattr__(
            self, "bread", "observed" if request is None else request.bread
        )
        object.__setattr__(
            self, "multiway", OMITTED if request is None else request.multiway
        )
        object.__setattr__(self, "psd", OMITTED if request is None else request.psd)
        object.__setattr__(
            self, "adjustment", OMITTED if request is None else request.adjustment
        )
        object.__setattr__(
            self, "ssc", OMITTED if request is None else owned_record(request.ssc)
        )
        object.__setattr__(
            self, "ssc_inherited", False if request is None else request.ssc_inherited
        )

    @property
    def codes(self) -> IntegerArray | None:
        """Return a fresh owned copy of ``codes`` with its exact integer dtype."""
        return _optional(self._codes)

    @property
    def positions(self) -> IntegerArray | None:
        """Return a fresh owned copy of ``positions`` with its exact integer dtype."""
        return _optional(self._positions)

    @property
    def clusters(self) -> tuple[tuple[str, IntegerArray], ...]:
        """Return fresh owned copies of ``(name, codes)`` cluster dimensions."""
        out: list[tuple[str, IntegerArray]] = []
        for cname, ccodes in self._clusters:
            copied = _optional(ccodes)
            if copied is not None:
                out.append((cname, copied))
        return tuple(out)

    @property
    def request(self) -> CovarianceRequest:
        """Reconstruct a fresh :class:`CovarianceRequest` with owned arrays."""
        return CovarianceRequest(
            kind=self.kind,
            name=self.name,
            codes=self.codes,
            positions=self.positions,
            convention=self.convention,
            clusters=self.clusters,
            method=self.method,
            bread=self.bread,
            multiway=self.multiway,
            psd=self.psd,
            ssc=self.ssc,
            ssc_inherited=self.ssc_inherited,
            adjustment=self.adjustment,
        )

    def identity(self) -> tuple[object, ...]:
        """Return the exact request identity tuple."""
        return self.request.identity()


def snapshot_covariance_request(
    request: CovarianceRequest,
) -> RetainedCovarianceRequest:
    """Snapshot a :class:`CovarianceRequest` into an owned record."""
    return RetainedCovarianceRequest(
        kind=request.kind,
        name=request.name,
        codes=_optional(request.codes),
        positions=_optional(request.positions),
        convention=request.convention,
        request=request,
    )


@dataclass(frozen=True)
class RetainedInitialScaleCache:
    """Cached initial covariance, raw components, direct SE, and reporting map.

    Populated before semantic discard (``keep_influence=False``) and detached from the
    live :class:`FEScaleInput` so initial-request queries succeed after discard and
    direct standard errors survive even when reported-matrix squaring overflows.
    """

    request: RetainedCovarianceRequest
    _order: OrderScale
    degenerate: bool
    _covariance: CovarianceRecord
    _reported: ReportedScale
    _reported_covariance: FloatArray | None = None
    covariance_error: str | None = None
    _influence: FloatArray | None = None
    _details: FEScaleDetails | None = None

    def __post_init__(self) -> None:
        """Own all arrays and records on construction."""
        object.__setattr__(self, "_order", owned_order_scale(self._order))
        object.__setattr__(self, "degenerate", bool(self.degenerate))
        object.__setattr__(self, "_covariance", owned_covariance(self._covariance))
        object.__setattr__(self, "_reported", owned_reported(self._reported))
        object.__setattr__(
            self, "_reported_covariance", _optional(self._reported_covariance)
        )
        object.__setattr__(self, "_influence", _optional(self._influence))
        object.__setattr__(self, "_details", owned_details(self._details))

    @property
    def order(self) -> OrderScale:
        """Return a fresh owned ``OrderScale``."""
        return owned_order_scale(self._order)

    @property
    def covariance(self) -> CovarianceRecord:
        """Return a fresh owned copy of the solver-coordinate covariance record."""
        return owned_covariance(self._covariance)

    @property
    def estimate(self) -> EstimateRecord:
        """Return a fresh owned copy of the primitive covariance estimate."""
        return owned_estimate(self._covariance.estimate)

    @property
    def components(self) -> tuple[ComponentRecord, ...]:
        """Return fresh owned copies of the cluster variance components."""
        return tuple(owned_component(c) for c in self._covariance.estimate.components)

    @property
    def reported(self) -> ReportedScale:
        """Return a fresh owned copy of the reported scale record."""
        return owned_reported(self._reported)

    @property
    def details(self) -> FEScaleDetails | None:
        """Return detached current-point inference details, or ``None``."""
        return owned_details(self._details)

    @property
    def gamma(self) -> FloatArray:
        """Return a fresh owned copy of the solver-coordinate ``theta`` vector."""
        return owned_array(self._reported.gamma)

    @property
    def coefficients(self) -> FloatArray:
        """Return a fresh owned copy of the reported coefficients ``beta``."""
        return owned_array(self._order.coefficients(self._reported.gamma))

    @property
    def standard_errors(self) -> FloatArray:
        """Return a fresh owned copy of the direct reported standard errors."""
        return owned_array(self._reported.standard_errors)

    @property
    def reported_covariance(self) -> FloatArray | None:
        """Return a fresh owned copy of the reported covariance matrix, or ``None``."""
        return _optional(self._reported_covariance)

    def require_reported_covariance(self) -> FloatArray:
        """Return the reported covariance matrix, or raise ``FloatingPointError``."""
        if self._reported_covariance is None:
            raise FloatingPointError(
                self.covariance_error
                or "the reported covariance matrix is not representable"
            )
        return owned_array(self._reported_covariance)

    @property
    def influence(self) -> FloatArray | None:
        """Return a fresh owned copy of the cached observed influence, or ``None``."""
        return _optional(self._influence)

    def without_influence(self) -> RetainedInitialScaleCache:
        """Return a copy with ``_influence`` cleared for ``keep_influence=False``."""
        if self._influence is None:
            return self
        return RetainedInitialScaleCache(
            request=self.request,
            _order=self._order,
            degenerate=self.degenerate,
            _covariance=self._covariance,
            _reported=self._reported,
            _reported_covariance=self._reported_covariance,
            covariance_error=self.covariance_error,
            _influence=None,
            _details=self._details,
        )


def snapshot_initial_scale(
    scale: FEScaleResult,
    *,
    keep_influence: bool = True,
) -> RetainedInitialScaleCache:
    """Snapshot an admitted :class:`FEScaleResult` into an owned initial cache."""
    ev = scale.evidence
    rep = ev.reported
    order = scale.input.order
    if ev.details is not None:
        rep_cov = ev.details.reported_covariance
        cov_err = ev.details.covariance_error
    else:
        rep_cov = None
        cov_err = None
        try:
            rep_cov = order.covariance(ev.covariance.covariance)
        except FloatingPointError as exc:
            cov_err = str(exc)
    return RetainedInitialScaleCache(
        request=snapshot_covariance_request(ev.request),
        _order=order,
        degenerate=bool((rep.standard_errors == 0.0).any()),
        _covariance=ev.covariance,
        _reported=rep,
        _reported_covariance=rep_cov,
        covariance_error=cov_err,
        _influence=ev.influence if keep_influence else None,
        _details=ev.details,
    )
