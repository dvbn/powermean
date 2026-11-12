"""Schema-preserving owned covariance records at the absorbed certificate boundary."""

from __future__ import annotations

from dataclasses import Field, fields, is_dataclass, replace
from typing import Any, ClassVar, Protocol

import numpy as np

from powermean.fe.reductions import immutable_array
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.reported import ReportedScale

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


class ComponentRecord(Protocol):
    """One cluster component of the existing convention, a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def covariance(self) -> FloatArray:
        """Its unsigned covariance."""
        ...


class EstimateRecord(Protocol):
    """The primitive estimate of the existing convention, a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def covariance(self) -> FloatArray:
        """The covariance without the factor."""
        ...

    @property
    def components(self) -> tuple[ComponentRecord, ...]:
        """The cluster components."""
        ...

    @property
    def eigenvalues(self) -> FloatArray:
        """The eigenvalues of the covariance."""
        ...


class CovarianceRecord(Protocol):
    """The covariance of the existing convention, a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def covariance(self) -> FloatArray:
        """``V`` of ``theta``."""
        ...

    @property
    def kind(self) -> str:
        """``robust``, ``cluster``, or ``hc``."""
        ...

    @property
    def factor(self) -> float:
        """Scalar multiplier applied to the primitive estimate."""
        ...

    @property
    def count(self) -> int:
        """Statistical observation count ``N``."""
        ...

    @property
    def clusters(self) -> int | None:
        """Cluster count for one-way / minimum cluster dimension, or ``None``."""
        ...

    @property
    def estimate(self) -> EstimateRecord:
        """The primitive estimate."""
        ...

    @property
    def reference(self) -> str:
        """The reference distribution kind (``'normal'`` or ``'t'``)."""
        ...


def _value(value: object, kind: str) -> bool:
    """Whether a declared field holds a value of its kind (``?``: or ``None``)."""
    if kind.endswith("?") and value is None:
        return True
    kind = kind.rstrip("?")
    if kind == "doubles":
        return isinstance(value, np.ndarray) and value.dtype == np.float64
    if kind == "names":
        return isinstance(value, tuple) and all(isinstance(v, str) for v in value)
    if kind == "text":
        return isinstance(value, str)
    if kind == "real":
        return isinstance(value, float)
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _declared(record: object, names: tuple[str, ...], kinds: tuple[str, ...]) -> None:
    """``TypeError`` unless ``record`` is a dataclass of exactly the declared fields."""
    if isinstance(record, type) or not is_dataclass(record):
        raise TypeError(SCHEMA)
    if {f.name for f in fields(record)} != set(names):
        raise TypeError(SCHEMA)
    for name, kind in zip(names, kinds, strict=True):
        if kind and not _value(getattr(record, name), kind):
            raise TypeError(SCHEMA)


def owned_component(component: ComponentRecord) -> ComponentRecord:
    """A cluster component re-formed with a copy of its covariance (docstring)."""
    kinds = ("names", "integer", "integer", "real", "doubles")
    _declared(component, COMPONENT, kinds)
    return replace(component, covariance=immutable_array(component.covariance))


def owned_estimate(estimate: EstimateRecord) -> EstimateRecord:
    """A primitive estimate re-formed with copies of its arrays and components."""
    kinds = ("doubles", "text", "integer", "integer?", "text", "real?")
    _declared(estimate, ESTIMATE, (*kinds, "", "doubles", "integer", "names"))
    if not isinstance(estimate.components, tuple):
        raise TypeError(SCHEMA)
    components = tuple(owned_component(c) for c in estimate.components)
    return replace(
        estimate,
        covariance=immutable_array(estimate.covariance),
        components=components,
        eigenvalues=immutable_array(estimate.eigenvalues),
    )


def owned_covariance(covariance: CovarianceRecord) -> CovarianceRecord:
    """A convention covariance re-formed with copies of its arrays and estimate."""
    kinds = ("doubles", "text", "real", "integer", "integer?", "", "text")
    _declared(covariance, COVARIANCE, kinds)
    return replace(
        covariance,
        covariance=immutable_array(covariance.covariance),
        estimate=owned_estimate(covariance.estimate),
    )


def owned_reported(reported: ReportedScale) -> ReportedScale:
    """The reported scale re-formed by its constructor: owned ``gamma``, SE and map."""
    if type(reported) is not ReportedScale:
        raise TypeError("the evidence holds a ReportedScale")
    order = reported.order
    if order is not None:
        return ReportedScale(
            gamma=reported.gamma,
            standard_errors=reported.standard_errors,
            order=OrderScale(order.omega, order.scale),
        )
    return ReportedScale(
        gamma=reported.gamma,
        transform=reported.transform,
        standard_errors=reported.standard_errors,
    )
