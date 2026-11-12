"""Sample-aligned dependence metadata records and temporal/spatial view builders.

This module owns sample-aligned column metadata (:class:`RetainedColumnMetadata`) and
the positional dependence sidecar (:class:`BoundDependence`) with result-local input and
sample keys and freshly owned column arrays on ingress and every exposure.  Positional
binding from a concrete prepared frame and sample lives in
:mod:`powermean.interface.retained_binding`.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass, fields
from itertools import repeat
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from powermean import messages
from powermean._prepare.dependence import (
    ColumnProvenance,
    DependenceRole,
    UsedDependence,
)
from powermean._prepare.owned import _owned_index
from powermean.data.inference_requests import (
    ClusterRequest,
    ConleyRequest,
    HacRequest,
)
from powermean.data.raw import AnyArray, Column
from powermean.data.request_values import OMITTED, Omitted, given
from powermean.data.weights import WeightKind
from powermean.errors import SpecificationError
from powermean.inference.convention import ClusterCodes
from powermean.interface.sample_encoding import _integer_key_codes
from powermean.interface.retained_space import RetainedSpaceView
from powermean.interface.retained_time import (
    RetainedTimeView,
    _is_missing_metadata,
)
from powermean.model.order import IndexArray
from powermean.results.owned import owned_array
from powermean.results.retained_inputs import (
    InputKey,
    SampleKey,
    _copy_column,
)

__all__ = [
    "BoundDependence",
    "RetainedColumnMetadata",
]


def _ordinary_numeric_storage(column: Column) -> AnyArray | None:
    """Recognize only plain one-dimensional numeric storage, without callbacks."""
    if type(column) is Column:
        stored = column._stored
        if (
            type(stored) is np.ndarray
            and stored.ndim == 1
            and stored.dtype.kind in "biufc"
            and stored.dtype.metadata is None
        ):
            return stored
    return None


def _take_owned_column(column: Column, positions: IndexArray) -> Column:
    """Select an already admitted copy-free private column exactly once.

    Only normal NumericExecution composition calls this. Immutable scalar/tuple
    elements may share, but the outer tuple and every ndarray header detach.
    Public/foreign Column selection and deepcopy callbacks remain unchanged.
    """
    stored = column._stored
    if type(stored) is not tuple:
        return Column(stored[positions])
    # Select only final rows, even after a very restrictive sample filter. The
    # itemgetter operates on the exact immutable tuple, with no element callbacks
    # or N-original-row object-array materialization.
    picked = operator.itemgetter(*positions.tolist())(stored) if positions.size else ()
    selected = (picked,) if positions.size == 1 else picked
    result = object.__new__(Column)
    object.__setattr__(result, "_stored", selected)
    return result


def _owned_missing_count(column: Column) -> int:
    """The exact _is_missing_metadata rule without scalar walks for plain runs."""
    stored = column._stored
    if type(stored) is np.ndarray:
        kind = stored.dtype.kind
        if kind == "f" and stored.dtype.itemsize <= 8:
            return int(np.count_nonzero(~np.isfinite(stored)))
        if kind in "Mm":
            return int(np.count_nonzero(np.isnat(stored)))
        if kind != "f":
            # In particular complex NaN is NOT missing under the existing rule.
            return 0
        return sum(1 for value in stored.tolist() if _is_missing_metadata(value))
    if not stored:
        return 0
    kind = type(stored[0])
    if (kind in (int, bool, str, bytes, complex, tuple, float, type(None))
            and all(map(operator.is_, map(type, stored), repeat(kind)))):
        if kind is type(None):
            return len(stored)
        if kind is float:
            values = np.fromiter(stored, dtype=np.float64, count=len(stored))
            return int(np.count_nonzero(~np.isfinite(values)))
        return 0
    return sum(1 for value in stored if _is_missing_metadata(value))


@dataclass(frozen=True, init=False)
class RetainedColumnMetadata:
    """Sample-aligned metadata and full-frame provenance of one retained column."""

    name: str
    roles: tuple[DependenceRole, ...]
    used_in_fit: bool
    source_dtype: str
    pandas_kind: str
    full_missing_count: int
    sample_missing_count: int
    _column: Column

    def __init__(
        self,
        *,
        name: str,
        roles: tuple[DependenceRole, ...],
        used_in_fit: bool,
        source_dtype: str,
        pandas_kind: str,
        full_missing_count: int,
        values: AnyArray,
    ) -> None:
        """Store owned sample-aligned values and missingness counts."""
        col = Column(values, name=name)
        # Column already owns this numeric buffer. Other inputs keep the generic
        # copy path, including its object-copy callbacks and shared memo behavior.
        if type(values) is not np.ndarray or _ordinary_numeric_storage(col) is None:
            col = _copy_column(col)
        sample_missing = sum(
            1 for val in col.values.tolist() if _is_missing_metadata(val)
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "roles", tuple(roles))
        object.__setattr__(self, "used_in_fit", bool(used_in_fit))
        object.__setattr__(self, "source_dtype", str(source_dtype))
        object.__setattr__(self, "pandas_kind", str(pandas_kind))
        object.__setattr__(self, "full_missing_count", int(full_missing_count))
        object.__setattr__(self, "sample_missing_count", int(sample_missing))
        object.__setattr__(self, "_column", col)

    def __getstate__(self) -> Any:
        """Export declared rich state without carrying a private owner recipe.

        Generic/subclass copying retains object.__getstate__, including slots and
        existing reduction hooks. Only this exact private native view expands.
        """
        state = object.__getstate__(self)
        if (type(self) is RetainedColumnMetadata
                and "_retained_provenance" in vars(self)):
            state = dict(vars(self))
            for field in fields(self):
                state[field.name] = getattr(self, field.name)
            for name in ("_retained_provenance","_retained_positions"):
                state.pop(name, None)
            # The requested legacy fields now ARE the owned state. Future rich
            # copies do not reconstruct another full-row projection.
            vars(self).update(state)
            for name in tuple(vars(self)):
                if name not in state:
                    vars(self).pop(name)
        return state

    @classmethod
    def _from_owned_column(
        cls, *, name: str, roles: tuple[DependenceRole, ...], used_in_fit: bool,
        source_dtype: str, pandas_kind: str, full_missing_count: int, column: Column,
    ) -> RetainedColumnMetadata:
        """Carry one private selected column; no outward export just to recopy it."""
        result = object.__new__(cls)
        for field, value in (
            ("name", name), ("roles", roles), ("used_in_fit", used_in_fit),
            ("source_dtype", source_dtype), ("pandas_kind", pandas_kind),
            ("full_missing_count", full_missing_count),
            ("sample_missing_count", _owned_missing_count(column)), ("_column", column),
        ):
            object.__setattr__(result, field, value)
        return result

    @classmethod
    def _from_retained(
        cls, provenance: ColumnProvenance, positions: IndexArray,
    ) -> RetainedColumnMetadata:
        """A final-row projection of immutable original values, not FE level codes."""
        result = object.__new__(cls)
        for field, value in (
            ("name", provenance.name), ("roles", provenance.roles),
            ("used_in_fit", False), ("source_dtype", provenance.source_dtype),
            ("pandas_kind", provenance.pandas_kind),
            ("_retained_provenance", provenance),
            ("_retained_positions", positions.view()),
        ):
            object.__setattr__(result, field, value)
        return result

    def __getattribute__(self, name: str) -> Any:
        """Resolve exact-native presentation fields; generic lookup stays unchanged."""
        if type(self) is RetainedColumnMetadata and type(name) is str:
            state = object.__getattribute__(self, "__dict__")
            if "_retained_provenance" in state and name not in state:
                if name == "_column":
                    values = self._retained_provenance.take(self._retained_positions)
                    result = Column(values)
                    object.__setattr__(self, name, result)
                    return result
                if name == "full_missing_count":
                    missing = self._retained_provenance._missing_mask
                    result = int(np.count_nonzero(missing))
                    object.__setattr__(self, name, result)
                    return result
                if name == "sample_missing_count":
                    result = _owned_missing_count(self._column)
                    object.__setattr__(self, name, result)
                    return result
        return object.__getattribute__(self, name)

    @property
    def values(self) -> AnyArray:
        """Return a fresh read-only copy of the column values."""
        stored = _ordinary_numeric_storage(self._column)
        if stored is not None:
            # Never expose Column.values: its ndarray metadata is mutable even
            # when the underlying bytes are read-only.
            return owned_array(stored)
        return _copy_column(self._column).values


@dataclass(frozen=True)
class BoundDependence:
    """Positional dependence metadata sidecar bound to a final estimation sample."""

    nobs: int
    used_dependence: UsedDependence
    columns: Mapping[str, RetainedColumnMetadata]
    _positions: IndexArray
    _labels: pd.Index[Any]
    weight_kind: WeightKind | None = None
    frequency_counts: tuple[int, ...] | None = None
    _initial_time_view: RetainedTimeView | None = None
    _initial_space_view: RetainedSpaceView | None = None
    input_key: InputKey | None = None
    sample_key: SampleKey | None = None
    count: int = 0

    def __post_init__(self) -> None:
        """Store owned positional metadata, optional fit-time views, and keys."""
        pos = owned_array(np.asarray(self._positions, dtype=np.intp))
        if pos.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        owned_lbls = _owned_index(self._labels)
        if len(owned_lbls) != pos.size:
            raise ValueError("labels length must equal positions size")
        for name, meta in self.columns.items():
            if type(meta) is not RetainedColumnMetadata:
                raise TypeError("columns values must be RetainedColumnMetadata")
            if meta.values.shape != (pos.size,):
                raise ValueError(f"column {name!r} is not aligned with positions")
        freqs = (
            None
            if self.frequency_counts is None
            else tuple(int(c) for c in self.frequency_counts)
        )
        if freqs is not None and len(freqs) != pos.size:
            raise ValueError("frequency_counts length must equal positions size")
        object.__setattr__(self, "nobs", int(self.nobs))
        object.__setattr__(self, "count", int(pos.size))
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        object.__setattr__(self, "frequency_counts", freqs)
        object.__setattr__(
            self,
            "input_key",
            None if self.input_key is None else InputKey(int(self.input_key)),
        )
        object.__setattr__(
            self,
            "sample_key",
            None if self.sample_key is None else SampleKey(int(self.sample_key)),
        )
        object.__setattr__(self, "_positions", pos)
        object.__setattr__(self, "_labels", owned_lbls)

    @classmethod
    def _from_owned(
        cls, *, nobs: int, used_dependence: UsedDependence,
        columns: Mapping[str, RetainedColumnMetadata], positions: IndexArray,
        labels: pd.Index[Any], weight_kind: WeightKind | None,
        frequency_counts: tuple[int, ...] | None, input_key: InputKey | None,
        sample_key: SampleKey | None,
        initial_time: RetainedTimeView | None = None,
        initial_space: RetainedSpaceView | None = None,
    ) -> BoundDependence:
        """Compose private admitted parts, not public .values shape/copy replay."""
        result = object.__new__(cls)
        for field, value in (
            ("nobs", int(nobs)), ("count", int(positions.size)),
            ("used_dependence", used_dependence),
            ("columns", MappingProxyType(dict(columns))),
            ("_positions", positions.view()), ("_labels", _owned_index(labels)),
            ("weight_kind", weight_kind), ("frequency_counts", frequency_counts),
            ("input_key", None if input_key is None else InputKey(int(input_key))),
            ("sample_key", None if sample_key is None else SampleKey(int(sample_key))),
            ("_initial_time_view", initial_time), ("_initial_space_view", initial_space),
        ):
            object.__setattr__(result, field, value)
        return result

    @property
    def positions(self) -> IndexArray:
        """Return a fresh read-only copy of the final original row positions."""
        return owned_array(self._positions)

    @property
    def labels(self) -> pd.Index[Any]:
        """Return a fresh owned copy of the user Index at positions."""
        return _owned_index(self._labels)

    @property
    def retained_names(self) -> tuple[str, ...]:
        """Return the names of all retained metadata columns."""
        return tuple(self.columns)

    @property
    def initial_time_view(self) -> RetainedTimeView | None:
        """Return the RetainedTimeView bound at fit time, if any."""
        return self._initial_time_view

    @property
    def initial_space_view(self) -> RetainedSpaceView | None:
        """Return the RetainedSpaceView bound at fit time, if any."""
        return self._initial_space_view

    def with_keys(
        self,
        *,
        input_key: InputKey | None = None,
        sample_key: SampleKey | None = None,
    ) -> BoundDependence:
        """Return a copy of this sidecar carrying the specified result-local keys."""
        ik = self.input_key if input_key is None else input_key
        sk = self.sample_key if sample_key is None else sample_key
        if ik == self.input_key and sk == self.sample_key:
            return self
        return BoundDependence(
            nobs=self.nobs,
            used_dependence=self.used_dependence,
            columns=self.columns,
            _positions=self._positions,
            _labels=self._labels,
            weight_kind=self.weight_kind,
            frequency_counts=self.frequency_counts,
            _initial_time_view=self._initial_time_view,
            _initial_space_view=self._initial_space_view,
            input_key=ik,
            sample_key=sk,
        )

    def _require_column(
        self, name: str, *, for_cluster: bool = False
    ) -> RetainedColumnMetadata:
        meta = self.columns.get(name)
        if meta is not None:
            return meta
        available = (
            messages.format_keys(self.retained_names) if self.retained_names else "none"
        )
        identifier = (
            "specification.cluster_not_retained"
            if for_cluster
            else "specification.metadata_not_retained"
        )
        raise SpecificationError(identifier, {"name": name, "retained": available})

    def cluster_codes(self, name: str) -> ClusterCodes:
        """Form :class:`ClusterCodes` for ``name`` by first appearance on the sample."""
        meta = self._require_column(name, for_cluster=True)
        if meta.sample_missing_count > 0:
            raise SpecificationError(
                "specification.cluster_missing_values",
                {"name": name, "count": meta.sample_missing_count},
            )
        values = meta.values
        if (type(values) is np.ndarray and values.ndim == 1 and values.size
                and values.dtype.kind in "iu" and values.size == self.count):
            return ClusterCodes(name, _integer_key_codes((values,))[0])
        labels: dict[Any, int] = {}
        codes = np.empty(self.count, dtype=np.intp)
        for pos, val in enumerate(values.tolist()):
            codes[pos] = labels.setdefault(val, len(labels))
        return ClusterCodes(name, codes)

    def cluster_request_codes(
        self, request: ClusterRequest
    ) -> tuple[ClusterCodes, ...]:
        """Form :class:`ClusterCodes` for each dimension of ``request``."""
        return tuple(self.cluster_codes(dim) for dim in request.dimensions)

    def time_view(
        self,
        request: HacRequest | None = None,
        *,
        time: str | None = None,
        unit: str | None = None,
        step: int = 1,
        lag: int | Omitted = OMITTED,
        check_chronology: bool = False,
    ) -> RetainedTimeView:
        """Build a :class:`RetainedTimeView` from retained columns."""
        if request is not None:
            time_col = request.time
            unit_col = str(request.unit) if given(request.unit) else None
            step_val, lag_val = request.resolved_step, request.lag
        elif time is not None:
            time_col, unit_col, step_val, lag_val = time, unit, step, lag
        else:
            raise ValueError("time or request must be provided")
        t_meta = self._require_column(time_col)
        u_vals = None if unit_col is None else self._require_column(unit_col).values
        view = RetainedTimeView(
            time_column=time_col,
            raw_time=t_meta.values,
            unit_column=unit_col,
            raw_unit=u_vals,
            step=step_val,
            lag=lag_val,
            frequency_counts=self.frequency_counts,
        )
        if check_chronology:
            view.check_chronology(lag=lag_val)
        return view

    def space_view(
        self,
        request: ConleyRequest | None = None,
        *,
        latitude: str | None = None,
        longitude: str | None = None,
        pool_times: Literal[True] | Omitted | None = None,
        check_requirements: bool = False,
    ) -> RetainedSpaceView:
        """Build a :class:`RetainedSpaceView` from retained coordinate columns."""
        if request is not None:
            lat_col, lon_col = request.latitude, request.longitude
        elif latitude is not None and longitude is not None:
            lat_col, lon_col = latitude, longitude
        else:
            raise ValueError("latitude and longitude or request must be provided")
        view = RetainedSpaceView(
            latitude_column=lat_col,
            longitude_column=lon_col,
            raw_latitude=self._require_column(lat_col).values,
            raw_longitude=self._require_column(lon_col).values,
            frequency_counts=self.frequency_counts,
            pool_times=pool_times,
            request=request,
        )
        if check_requirements:
            view.check_conley_requirements(
                cutoff=None if request is None else request.cutoff,
                pool_times=pool_times,
            )
        return view

