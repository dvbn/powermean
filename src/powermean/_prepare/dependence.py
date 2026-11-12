"""Fit-used and retained-only dependence provenance for frame preparation.

Provides :class:`UsedDependence`, :class:`ColumnProvenance`,
:class:`DependenceProvenance`, and :func:`prepare_dependence`.  Fit-used dependence
columns (``cluster``, ``clusters``, ``time``, ``unit``, ``coordinates``) join the step-2
``"clusters"`` missing group while preserving the seven-group accounting unchanged;
retained-only columns (``metadata``, ``retained``) record source dtype, role, and
missingness masks without excluding rows during sample formation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from powermean._prepare import inputs as _input_columns
from powermean._prepare.inputs import Columns, _column, _inputs, _values
from powermean.data import raw as _raw_columns
from powermean.data.raw import (
    AlignedColumns,
    AnyArray,
    Column,
    RawInputs,
    _OwnedMapping,
)
from powermean.formula import validity
from powermean.formula.frame import ModelFrame
from powermean.model.order import BoolArray, _flags

DependenceRole = Literal["cluster", "time", "unit", "space", "retained"]


@dataclass(frozen=True)
class UsedDependence:
    """Fit-used dependence columns and retained-only columns for preparation.

    Parameters
    ----------
    clusters : tuple of str, default ()
        One-way or multiway cluster column names used at the fit.
    time : str, optional
        Time column name used by a panel or time-series covariance at the fit.
    unit : str, optional
        Panel unit column name used by a panel covariance at the fit.
    coordinates : tuple of (str, str), optional
        Latitude and longitude column names ``(lat, lon)`` used by a spatial
        covariance at the fit.
    retained : tuple of str, default ()
        Columns declared via ``retain=`` to be kept on the sample for post-fit
        inference requests without excluding rows at the fit.
    """

    clusters: tuple[str, ...] = ()
    time: str | None = None
    unit: str | None = None
    coordinates: tuple[str, str] | None = None
    retained: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate column name collections and coordinate pair structure."""
        if not isinstance(self.clusters, tuple) or not all(
            isinstance(name, str) and name for name in self.clusters
        ):
            raise TypeError("clusters must be a tuple of non-empty strings")
        if self.time is not None and (not isinstance(self.time, str) or not self.time):
            raise TypeError("time must be a non-empty string or None")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise TypeError("unit must be a non-empty string or None")
        if self.coordinates is not None and (
            not isinstance(self.coordinates, tuple)
            or len(self.coordinates) != 2
            or not all(isinstance(c, str) and c for c in self.coordinates)
        ):
            raise TypeError("coordinates must be a 2-tuple of non-empty strings")
        if not isinstance(self.retained, tuple) or not all(
            isinstance(name, str) and name for name in self.retained
        ):
            raise TypeError("retained must be a tuple of non-empty strings")


@dataclass(frozen=True, init=False)
class ColumnProvenance:
    """Source dtype, role, missingness mask, and owned values of one metadata column.

    Parameters
    ----------
    name : str
        Column name in the source frame.
    roles : tuple of DependenceRole
        Roles declared for this column in canonical order.
    used_in_fit : bool
        Whether missing or non-finite values in this column exclude rows during
        step 2 of sample formation (group ``"clusters"``).
    source_dtype : str
        Exact string representation of the source pandas ``Series.dtype`` captured
        before :func:`_values` converts extension or categorical columns.
    pandas_kind : str
        Normalized kind tag of the source pandas ``Series.dtype``.
    missing_mask : numpy.ndarray
        Boolean mask of length ``nobs`` marking missing or non-finite entries.
    values : numpy.ndarray
        Raw column values extracted by :func:`_values`.
    """

    name: str
    roles: tuple[DependenceRole, ...]
    used_in_fit: bool
    source_dtype: str
    pandas_kind: str
    _missing_mask: BoolArray
    _column_record: Column

    def __init__(
        self,
        *,
        name: str,
        roles: tuple[DependenceRole, ...],
        used_in_fit: bool,
        source_dtype: str,
        pandas_kind: str,
        missing_mask: BoolArray,
        values: AnyArray,
    ) -> None:
        """Store owned copies of the missingness mask and column values."""
        if not isinstance(name, str) or not name:
            raise TypeError("name must be a non-empty string")
        if not roles:
            raise ValueError("roles cannot be empty")
        mask_arr = np.asarray(missing_mask, dtype=np.bool_)
        if mask_arr.ndim != 1:
            raise ValueError("missing_mask must be one-dimensional")
        mask = _flags(mask_arr)
        record = Column(values)
        if record.size != mask.size:
            raise ValueError("values and missing_mask must have the same length")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "roles", tuple(roles))
        object.__setattr__(self, "used_in_fit", bool(used_in_fit))
        object.__setattr__(self, "source_dtype", str(source_dtype))
        object.__setattr__(self, "pandas_kind", str(pandas_kind))
        object.__setattr__(self, "_missing_mask", mask)
        object.__setattr__(self, "_column_record", record)

    def __getstate__(self) -> Any:
        """Export declared rich state without carrying a private owner recipe.

        Generic/subclass copying retains object.__getstate__, including slots and
        existing reduction hooks. Only this exact private native view expands.
        """
        state = object.__getstate__(self)
        if type(self) is ColumnProvenance and "_retained_sources" in vars(self):
            state = dict(vars(self))
            for field in fields(self):
                state[field.name] = getattr(self, field.name)
            for name in ("_retained_sources",):
                state.pop(name, None)
            # The requested legacy fields now ARE the owned state. Future rich
            # copies do not reconstruct another full-row projection.
            vars(self).update(state)
            for name in tuple(vars(self)):
                if name not in state:
                    vars(self).pop(name)
        return state

    @classmethod
    def _from_retained(
        cls, *, name: str, source_dtype: str, pandas_kind: str,
        sources: tuple[Column, ...],
    ) -> ColumnProvenance:
        """Lossless original-row view from this invocation's native owned columns.

        No final FE codes or representatives stand in for original values. Only
        the fresh normal producer supplies these immutable native columns; generic
        constructors and callback-bearing inputs retain their eager copy order.
        """
        result = object.__new__(cls)
        for field, value in (
            ("name", name), ("roles", ("retained",)), ("used_in_fit", False),
            ("source_dtype", source_dtype), ("pandas_kind", pandas_kind),
            ("_retained_sources", sources),
        ):
            object.__setattr__(result, field, value)
        return result

    def __getattribute__(self, name: str) -> Any:
        """Resolve exact-native presentation fields; generic lookup stays unchanged."""
        if type(self) is ColumnProvenance and type(name) is str:
            state = object.__getattribute__(self, "__dict__")
            if "_retained_sources" in state and name not in state:
                if name == "_column_record":
                    positions = np.arange(self._retained_sources[0].size)
                    result = Column(self.take(positions))
                    object.__setattr__(self, name, result)
                    return result
                if name == "_missing_mask":
                    # The admitted native biuf rule, not a new call to a source callback
                    # after the fit. Non-native/changed-hook inputs were kept eager.
                    masks = tuple(
                        ~np.isfinite(c._stored) if c.dtype.kind == "f"
                        else np.zeros(c.size, dtype=np.bool_)
                        for c in self._retained_sources
                    )
                    result = _flags(np.logical_or.reduce(masks))
                    object.__setattr__(self, name, result)
                    return result
        return object.__getattribute__(self, name)

    @property
    def missing_mask(self) -> BoolArray:
        """A fresh read-only copy of the full-frame missingness mask."""
        return _flags(self._missing_mask)

    @property
    def values(self) -> AnyArray:
        """A fresh read-only copy of the full-frame raw column values."""
        if "_retained_sources" in vars(self):
            return self.take(np.arange(self._retained_sources[0].size))
        arr = self._column_record.values
        if arr.dtype.kind == "O":
            return arr
        stored: AnyArray = np.frombuffer(arr.tobytes(), dtype=arr.dtype)
        return stored.reshape(arr.shape)

    def take(self, positions: NDArray[np.intp]) -> AnyArray:
        """A fresh read-only copy of the column values at ``positions``."""
        if "_retained_sources" in vars(self):
            sources = self._retained_sources
            values = tuple(c._stored[positions] for c in sources)
            if len(values) == 1:
                return Column(values[0]).values
            # Match interaction_column: convert each original component to object
            # in its original row order BEFORE pairing, including scalar types.
            left, right = (v.astype(object) for v in values)
            invalid = np.zeros(left.size, dtype=np.bool_)
            for value in values:
                if value.dtype.kind == "f":
                    invalid |= ~np.isfinite(value)
            paired = np.fromiter(zip(left, right, strict=True), dtype=object,
                                 count=left.size)
            paired[invalid] = None
            paired.setflags(write=False)
            return paired
        arr = self._column_record.take(positions).values
        if arr.dtype.kind == "O":
            return arr
        stored: AnyArray = np.frombuffer(arr.tobytes(), dtype=arr.dtype)
        return stored.reshape(arr.shape)


@dataclass(frozen=True, init=False)
class DependenceProvenance:
    """Provenance of all fit-used and retained-only dependence columns on a frame.

    ``_columns`` is the record's :class:`~powermean.data.raw._OwnedMapping`: the exact
    ``dict`` that the constructor formed from the items given, carried as the
    record's current data and read through that one object.  :attr:`columns` hands
    out a fresh read-only view of a fresh dict on every access, as before.
    """

    used: UsedDependence
    _columns: Mapping[str, ColumnProvenance]
    _used_missing_union: BoolArray

    def __init__(
        self,
        *,
        used: UsedDependence,
        columns: Mapping[str, ColumnProvenance],
        used_missing_union: BoolArray,
    ) -> None:
        """Store an immutable mapping of column provenance and owned union mask."""
        union_arr = np.asarray(used_missing_union, dtype=np.bool_)
        if union_arr.ndim != 1:
            raise ValueError("used_missing_union must be one-dimensional")
        union = _flags(union_arr)
        object.__setattr__(self, "used", used)
        object.__setattr__(self, "_columns", _OwnedMapping(dict(columns)))
        object.__setattr__(self, "_used_missing_union", union)

    @property
    def columns(self) -> Mapping[str, ColumnProvenance]:
        """A fresh read-only mapping of column provenance records."""
        return MappingProxyType(dict(self._columns))

    @property
    def used_missing_union(self) -> BoolArray:
        """A fresh read-only copy of the missingness union of fit-used columns."""
        return _flags(self._used_missing_union)


def _pandas_kind(dtype: Any) -> str:
    """Classify a pandas Series dtype before conversion to a NumPy array."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "datetime_tz"
    if isinstance(dtype, pd.PeriodDtype):
        return "period"
    if not isinstance(dtype, np.dtype):
        return "extension"
    kind_map = {
        "M": "datetime",
        "i": "integer",
        "u": "unsigned",
        "f": "float",
        "b": "bool",
        "U": "string",
        "S": "string",
    }
    return kind_map.get(dtype.kind, "object")


def _role_table(
    columns: Columns, dependence: UsedDependence
) -> tuple[dict[str, list[DependenceRole]], UsedDependence]:
    """Build ordered per-column role lists and normalized UsedDependence."""
    roles_by_name: dict[str, list[DependenceRole]] = {}

    def add_role(name: str, role: DependenceRole) -> None:
        bucket = roles_by_name.setdefault(name, [])
        if role not in bucket:
            bucket.append(role)

    cluster_list: list[str] = []
    if columns.cluster is not None:
        add_role(columns.cluster, "cluster")
        cluster_list.append(columns.cluster)
    for name in dependence.clusters:
        add_role(name, "cluster")
        if name not in cluster_list:
            cluster_list.append(name)
    if dependence.time is not None:
        add_role(dependence.time, "time")
    if dependence.unit is not None:
        add_role(dependence.unit, "unit")
    if dependence.coordinates is not None:
        add_role(dependence.coordinates[0], "space")
        add_role(dependence.coordinates[1], "space")
    retained_list: list[str] = []
    for name in (*columns.metadata, *dependence.retained):
        add_role(name, "retained")
        if name not in retained_list:
            retained_list.append(name)
    normalized = UsedDependence(
        clusters=tuple(cluster_list),
        time=dependence.time,
        unit=dependence.unit,
        coordinates=dependence.coordinates,
        retained=tuple(retained_list),
    )
    return roles_by_name, normalized


def prepare_dependence(
    frame: ModelFrame,
    data: pd.DataFrame,
    columns: Columns,
    dependence: UsedDependence | None = None,
) -> tuple[RawInputs, dict[str, Any], DependenceProvenance]:
    """Read raw inputs, step-2 missing masks, and dependence column provenance.

    Fit-used dependence columns (``cluster``, ``clusters``, ``time``, ``unit``,
    ``coordinates``) union their invalid-value masks into ``missing["clusters"]``
    while keeping the existing seven-group coordinator unchanged.  Retained-only
    columns (``metadata``, ``retained``) record their source dtype, role, and
    missingness mask without excluding any row during sample formation.
    """
    return _prepare_dependence(frame, data, columns, dependence)


def _prepare_dependence(
    frame: ModelFrame, data: pd.DataFrame, columns: Columns,
    dependence: UsedDependence | None = None, *,
    _retained: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[RawInputs, dict[str, Any], DependenceProvenance]:
    """The single extraction body; owned composition alone supplies original terms."""
    inputs, missing = _inputs(frame, data, columns)
    resolved = UsedDependence() if dependence is None else dependence
    if type(resolved) is not UsedDependence:
        raise TypeError("dependence must be a UsedDependence or None")
    roles_by_name, normalized = _role_table(columns, resolved)
    nobs = len(data)
    union = np.zeros(nobs, dtype=np.bool_)
    has_used = False
    provenance_map: dict[str, ColumnProvenance] = {}
    extra_metadata: dict[str, Any] = {
        name: inputs.metadata[name] for name in inputs.metadata
    }
    retained_sources: dict[str, Column] = {}
    for name, role_list in roles_by_name.items():
        if _retained is not None and name in _retained and role_list == ["retained"]:
            terms = _retained[name]
            for term in terms:
                if term not in retained_sources:
                    prior = provenance_map.get(term)
                    retained_sources[term] = (
                        prior._column_record
                        if prior is not None and "_retained_sources" not in vars(prior)
                        else Column(_values(_column(data, term)))
                    )
            series = _column(data, terms[0])
            provenance_map[name] = ColumnProvenance._from_retained(
                name=name,
                source_dtype=str(series.dtype) if len(terms) == 1 else "object",
                pandas_kind=_pandas_kind(series.dtype) if len(terms) == 1 else "object",
                sources=tuple(retained_sources[term] for term in terms),
            )
            continue
        series = _column(data, name)
        source_dtype = str(series.dtype)
        pandas_kind = _pandas_kind(series.dtype)
        mask = validity.values_invalid(series)
        raw_vals = _values(series)
        used_in_fit = any(role != "retained" for role in role_list)
        if used_in_fit:
            union |= mask
            has_used = True
        if name not in extra_metadata:
            extra_metadata[name] = raw_vals
        provenance_map[name] = ColumnProvenance(
            name=name,
            roles=tuple(role_list),
            used_in_fit=used_in_fit,
            source_dtype=source_dtype,
            pandas_kind=pandas_kind,
            missing_mask=mask,
            values=raw_vals,
        )
    if has_used:
        missing["clusters"] = _flags(union)
    if len(extra_metadata) != len(inputs.metadata):
        if _retained is not None:
            # The first RawInputs already owns and checked y/design/weights.
            # Add only freshly checked metadata; no second numeric construction.
            object.__setattr__(inputs, "metadata",
                               AlignedColumns(extra_metadata, nobs))
        else:
            inputs = RawInputs(
                inputs.y,
                inputs.design,
                column_names=inputs.column_names,
                weights=inputs.weights,
                offset=inputs.offset,
                exposure=inputs.exposure,
                metadata=extra_metadata,
            )
    provenance = DependenceProvenance(
        used=normalized,
        columns=provenance_map,
        used_missing_union=union,
    )
    return inputs, missing, provenance

_RETAIN_GLOBALS = tuple((name, globals()[name]) for name in (
    "AlignedColumns", "Column", "ColumnProvenance", "RawInputs",
    "_inputs", "_column", "_values",
    "_pandas_kind", "_role_table", "_prepare_dependence",
))
_RETAIN_PROVENANCE = tuple((name, vars(ColumnProvenance).get(name)) for name in (
    "__init__", "__getattribute__", "__getattr__", "__getstate__",
    "take", "values", "missing_mask",
))
_RETAIN_FIELDS = tuple((name, vars(ColumnProvenance).get(name))
                       for name in ColumnProvenance.__dataclass_fields__)
_RETAIN_COLUMNS = tuple((kind, tuple((name, vars(kind).get(name)) for name in (
    "__init__", "__getattribute__", "values", "stored", "dtype", "size", "take",
    "nobs",
))) for kind in (Column, RawInputs, AlignedColumns, _input_columns.RawWeights))
_RETAIN_RAW_ALIASES = tuple((name, getattr(_raw_columns, name)) for name in (
    "Column", "RawInputs", "AlignedColumns", "RawWeights",
    "_elements", "_owned_objects", "_object_array",
    "_column_names", "_real_rows", "_immutable",
))
_RETAIN_INPUT_CONSTRUCTOR = _input_columns.RawInputs
_RETAIN_INPUT_WEIGHTS = _input_columns.RawWeights
_RETAIN_INVALID = validity.values_invalid


def _retained_original() -> bool:
    """Constant-size first-producer callback qualification, never per-row replay."""
    return (all(globals().get(name) is original for name, original in _RETAIN_GLOBALS)
            and all(vars(_raw_columns).get(name) is original
                    for name, original in _RETAIN_RAW_ALIASES)
            and _input_columns.RawInputs is _RETAIN_INPUT_CONSTRUCTOR
            and _input_columns.RawWeights is _RETAIN_INPUT_WEIGHTS
            and all(vars(ColumnProvenance).get(name) is original
                    for name, original in _RETAIN_PROVENANCE)
            and all(vars(ColumnProvenance).get(name) is original
                    for name, original in _RETAIN_FIELDS)
            and all(vars(kind).get(name) is original
                    for kind, methods in _RETAIN_COLUMNS for name, original in methods)
            and validity.values_invalid is _RETAIN_INVALID)
