"""Owned finite inference records at the solve/inference layer boundary.

Opaque mathematical records are checked by admission and narrowed to their actual
kernel classes by the inference adapter. No upper-layer import or unchecked Any
is needed here. This copier covers these finite diagnostics, never a fit graph.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from powermean.data.inference_conventions import ConventionValues, PsdTreatment
from powermean.data.request_values import OMITTED, Omitted
from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.model.variance import FloatArray
from powermean.systems.joint import SamplingUnits

Bread = Literal["observed", "expected"]
FactorOwner = Literal["convention", "method"]
SSCSource = Literal["default", "explicit", "inherited"]


@dataclass(frozen=True, eq=False)
class FEScaleDetails:
    """Current-point composition records, opaque until their schemas are checked."""

    bread: Bread
    requested_ssc: str | ConventionValues | Omitted
    resolved_preset: str
    ssc_source: SSCSource
    factor_owner: FactorOwner
    units: SamplingUnits
    count: int
    rank: int
    applied: object
    absorbed: object | None
    multiway: object | None = None
    psd: PsdTreatment | None = None
    adjustment: object | None = None
    frame: object | None = None
    hc: object | None = None
    reported_covariance: FloatArray | None = None
    covariance_error: str | None = None
    wz: object | None = None


def _array(value: NDArray[np.generic]) -> NDArray[np.generic]:
    """A fresh header, sharing storage only if its ultimate owner is bytes."""
    if value.dtype.hasobject:
        raise TypeError("inference details cannot contain object arrays")
    base: object = value
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if isinstance(base, bytes):
        return value.view()
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _copy(value: object, ancestors: frozenset[int]) -> object:
    """Detach every record occurrence; reject cycles and unsupported objects."""
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if type(value) is Omitted:
        return OMITTED
    # These exact named tuples contain bytes and integer shapes only.
    if type(value) in (FloatStorage, IndexStorage):
        return value
    if isinstance(value, np.generic):
        return value
    if isinstance(value, np.ndarray):
        return _array(value)
    if id(value) in ancestors:
        raise TypeError("inference details must be an acyclic finite record")
    nested = ancestors | {id(value)}
    if type(value) is tuple:
        return tuple(_copy(item, nested) for item in value)
    if type(value) is dict or isinstance(value, MappingProxyType):
        copied = {key: _copy(item, nested) for key, item in value.items()}
        return (
            MappingProxyType(copied) if isinstance(value, MappingProxyType) else copied
        )
    if is_dataclass(value) and not isinstance(value, type):
        changes = {f.name: _copy(getattr(value, f.name), nested) for f in fields(value)}
        return replace(value, **changes)
    raise TypeError("unsupported inference-details record")


def owned_record[RecordT](value: RecordT) -> RecordT:
    """Preserve each concrete record class, with detached nested wrappers/headers."""
    copied = _copy(value, frozenset())
    if type(copied) is not type(value):
        raise TypeError("ownership changed an inference record's class")
    return copied


def record_fields(value: object, names: tuple[str, ...]) -> dict[str, object]:
    """Read only an instance of exactly the declared dataclass field schema."""
    if isinstance(value, type) or not is_dataclass(value):
        raise TypeError("inference evidence must be a dataclass record")
    if {f.name for f in fields(value)} != set(names):
        raise TypeError("inference evidence has another record schema")
    return {name: getattr(value, name) for name in names}


def owned_details(details: FEScaleDetails | None) -> FEScaleDetails | None:
    """Own every nested record, not merely covariance/array payloads."""
    if details is None:
        return None
    if type(details) is not FEScaleDetails:
        raise TypeError("details must be an FEScaleDetails or None")
    if details.bread not in ("observed", "expected"):
        raise ValueError("details.bread must be 'observed' or 'expected'")
    if details.ssc_source not in ("default", "explicit", "inherited"):
        raise ValueError("invalid details.ssc_source")
    if details.factor_owner not in ("convention", "method"):
        raise ValueError("invalid details.factor_owner")
    if type(details.units) is not SamplingUnits:
        raise TypeError("details.units must be SamplingUnits")
    return owned_record(details)
