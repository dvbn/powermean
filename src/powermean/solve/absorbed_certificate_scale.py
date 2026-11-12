"""The same-point scale contract of the absorbed certificate and its owned evidence.

Internal module: the contract implemented by the absorbed inference provider;
nothing here imports inference.

Input.  :class:`FEScaleInput` binds the fresh geometry by value, the requested
covariance (:class:`CovarianceRequest`), the direct report map and the slope map ``W``;
the input object and its request are identity tokens, never copied.

Evidence.  :class:`FEScaleEvidence` keeps its ORIGINAL binding and request as given and
owns its payload from construction, on direct construction and on
``dataclasses.replace`` alike: the influence rows, the covariance of the existing
convention, the reported scale, and optional :class:`FEScaleDetails` for the full joint
inference composition.  The covariance is three named records, each a
dataclass with exactly its declared fields: the convention covariance (``covariance``,
``kind``, ``factor``, ``count``, ``clusters``, ``estimate``, ``reference``), its
primitive estimate (``covariance``, ``method``, ``nobs``, ``df_resid``, ``correction``,
``factor``, ``components``, ``eigenvalues``, ``rank``, ``diagnostics``) and each
cluster component (``dimensions``, ``sign``, ``n_clusters``, ``factor``,
``covariance``); arrays of doubles and immutable scalar or tuple metadata
(``TypeError`` otherwise).  Each record is re-formed by ``dataclasses.replace`` with
copies of its arrays in bytes of their own (signed zeros, subnormals and NaN kept), its
class and every other field unchanged; the reported scale by its own constructor on a
fresh direct ``OrderScale``, never a dense map (or on a copy of a dense map it holds).
Nothing is recomputed, symmetrized or floored.

Admission (:func:`admitted_scale`): the result must hold that very input, evidence of
that input's binding (never rebound from the current input) and request; only then is
an owned snapshot formed and its data checked: the direct report map, the ``theta``
bytes, the shapes, the requested kind, and the current-sample/point inference details.
The certificate keeps the snapshot and hands out a fresh copy at each access
(:func:`owned_result`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np

from powermean.data.inference_conventions import (
    PSD_TREATMENTS,
)
from powermean.data.inference_requests import (
    MULTIWAY_FORMS,
)
from powermean.data.sample_fe_state import same_bits
from powermean.fe.reductions import immutable_array
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_geometry import FEGeometry
from powermean.solve.absorbed_scale_admission import (
    check_details_admission as _check_details_admission,
)
from powermean.solve.absorbed_scale_details import (
    FEScaleDetails,
    owned_details,
)
from powermean.solve.absorbed_scale_records import (
    ComponentRecord,
    CovarianceRecord,
    EstimateRecord,
    owned_component,
    owned_covariance,
    owned_estimate,
    owned_reported,
)
from powermean.solve.absorbed_scale_request import (
    CovarianceRequest,
    IntegerArray,
    Kind,
)
from powermean.solve.reported import ReportedScale

Bread = Literal["observed", "expected"]
FactorOwner = Literal["convention", "method", "ssc", "hc", "multiway"]
SSCSource = Literal["default", "explicit", "inherited"]
BREAD_VALUES: tuple[str, ...] = ("observed", "expected")
PRESET_NAMES: tuple[str, ...] = ("stata", "none", "regression")
MULTIWAY_VALUES: tuple[str, ...] = MULTIWAY_FORMS
PSD_VALUES: tuple[str, ...] = PSD_TREATMENTS

STALE = "the scale result is not bound to this input"
SCHEMA = "the covariance evidence is not the record of the existing convention"
COMPONENT = ("dimensions", "sign", "n_clusters", "factor", "covariance")
ESTIMATE = (
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
)
COVARIANCE = (
    "covariance",
    "kind",
    "factor",
    "count",
    "clusters",
    "estimate",
    "reference",
)


# ----------------------------------------------------------------- the input


def bound_request(request: object, geometry: FEGeometry) -> CovarianceRequest:
    """The provider's request, bound to the sample's final rows (``ValueError``)."""
    if type(request) is not CovarianceRequest:
        raise TypeError("a scale provider must hold a CovarianceRequest")
    sample = geometry.point.sample
    if request.kind == "cluster":
        positions = request.positions
        if (
            positions is None
            or not request.clusters
            or positions.shape != (sample.count,)
        ):
            raise ValueError("the cluster codes are not aligned with the sample's rows")
        if not np.array_equal(positions, sample.positions):
            raise ValueError("the cluster codes are bound to other rows")
        for _, codes in request.clusters:
            if codes.shape != (sample.count,):
                raise ValueError(
                    "the cluster codes are not aligned with the sample's rows"
                )
    return request


@dataclass(frozen=True)
class FEBinding:
    """The full binding of a scale input by value: identities and exact bytes."""

    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, eq=False)
class FEScaleInput:
    """What a provider receives: the geometry, the request, the report map and ``W``."""

    geometry: FEGeometry
    request: CovarianceRequest
    order: OrderScale
    slope_map: FloatArray
    binding: FEBinding = field(init=False)

    def __post_init__(self) -> None:
        """Check the parts, own ``W`` and bind the whole input by value."""
        geometry = self.geometry
        if type(geometry) is not FEGeometry or type(self.order) is not OrderScale:
            raise TypeError("a scale input binds an FEGeometry and an OrderScale")
        if type(self.request) is not CovarianceRequest:
            raise TypeError("a scale input binds a CovarianceRequest")
        k, slope_map = geometry.n_slopes, immutable_array(self.slope_map)
        if (
            not k
            or self.order.dimension != k
            or slope_map.shape != (geometry.dimension, k)
        ):
            raise ValueError("the slope map and the report map must match the slopes")
        items = (
            *geometry.binding(),
            ("W", (slope_map.shape, slope_map.tobytes())),
            ("order", (self.order.omega, self.order.scale.tobytes())),
            ("request", self.request.identity()),
        )
        object.__setattr__(self, "slope_map", slope_map)
        object.__setattr__(self, "binding", FEBinding(items))


# --------------------------------------------------------- the covariance records


# ---------------------------------------------------------------- the result


@dataclass(frozen=True, eq=False)
class FEScaleEvidence:
    """A provider's owned evidence and the ORIGINAL binding it was formed under.

    Construction (also by ``dataclasses.replace``) keeps the binding and the request
    and copies the influence rows, the covariance records, the reported scale, and
    optional ``details``.
    """

    binding: FEBinding
    request: CovarianceRequest
    influence: FloatArray
    covariance: CovarianceRecord
    reported: ReportedScale
    details: FEScaleDetails | None = None

    def __post_init__(self) -> None:
        """Check the parts and own the payload (module docstring)."""
        if type(self.binding) is not FEBinding:
            raise TypeError("evidence keeps the FEBinding it was formed under")
        if type(self.request) is not CovarianceRequest:
            raise TypeError("evidence keeps the CovarianceRequest it answers")
        influence = self.influence
        if not isinstance(influence, np.ndarray) or influence.dtype != np.float64:
            raise TypeError("the influence rows must be an array of doubles")
        object.__setattr__(self, "influence", immutable_array(influence))
        object.__setattr__(self, "covariance", owned_covariance(self.covariance))
        object.__setattr__(self, "reported", owned_reported(self.reported))
        object.__setattr__(self, "details", owned_details(self.details))


@dataclass(frozen=True, eq=False)
class FEScaleResult:
    """The exact input a provider received, and its evidence; a stale pair refuses."""

    input: FEScaleInput
    evidence: FEScaleEvidence

    def __post_init__(self) -> None:
        """``ValueError`` unless the evidence was formed under this input's binding."""
        if type(self.input) is not FEScaleInput:
            raise TypeError("a scale result holds an FEScaleInput")
        if type(self.evidence) is not FEScaleEvidence:
            raise TypeError("a scale result holds FEScaleEvidence")
        if self.evidence.binding != self.input.binding:
            raise ValueError(STALE)


class FEScaleProvider(Protocol):
    """The same-point scale provider: its request, and its result for an input."""

    @property
    def request(self) -> CovarianceRequest:
        """The requested covariance."""
        ...

    def __call__(self, scale_input: FEScaleInput, /) -> FEScaleResult:
        """The result at ``scale_input``, formed from it alone."""
        ...


def owned_result(result: FEScaleResult) -> FEScaleResult:
    """A copy of ``result``: the same input, its evidence re-formed and owned."""
    return FEScaleResult(result.input, replace(result.evidence))


def admitted_scale(result: object, scale_input: FEScaleInput) -> FEScaleResult:
    """The full admission of a provider's result at ``scale_input``: its snapshot.

    ``TypeError`` or ``ValueError`` (module docstring).
    """
    if (
        type(result) is not FEScaleResult
        or type(result.evidence) is not FEScaleEvidence
    ):
        raise TypeError("a scale provider must return an FEScaleResult")
    evidence, geometry = result.evidence, scale_input.geometry
    if result.input is not scale_input or result.input.binding != scale_input.binding:
        raise ValueError(STALE)
    if evidence.binding != scale_input.binding:
        raise ValueError(STALE)
    if evidence.request is not scale_input.request:
        raise ValueError("the evidence answers another covariance request")
    owned = owned_result(result)
    reported, mine, k = owned.evidence.reported, scale_input.order, geometry.n_slopes
    order = reported.order
    if (
        order is None
        or order.omega != mine.omega
        or not same_bits(order.scale, mine.scale)
    ):
        raise ValueError("the reported scale does not hold the direct report map")
    if not same_bits(reported.gamma, geometry.point.theta):
        raise ValueError(STALE)
    covariance, influence = owned.evidence.covariance, owned.evidence.influence
    shapes = (np.shape(covariance.covariance), np.shape(influence))
    if shapes != ((k, k), (geometry.point.sample.count, k)):
        raise ValueError(STALE)
    if covariance.kind != scale_input.request.kind:
        raise ValueError("the covariance is not the requested one")
    details = owned.evidence.details
    if details is None and not scale_input.request.is_default_stata:
        raise ValueError(STALE)
    if details is not None:
        _check_details_admission(details, covariance, reported, scale_input)
    return owned


__all__ = [
    "ComponentRecord",
    "CovarianceRecord",
    "CovarianceRequest",
    "EstimateRecord",
    "FEBinding",
    "FEScaleDetails",
    "FEScaleEvidence",
    "FEScaleInput",
    "FEScaleProvider",
    "FEScaleResult",
    "IntegerArray",
    "Kind",
    "admitted_scale",
    "bound_request",
    "owned_component",
    "owned_covariance",
    "owned_details",
    "owned_estimate",
    "owned_reported",
    "owned_result",
]
