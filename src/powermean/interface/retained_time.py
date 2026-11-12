"""Exact temporal ordering, inverse permutation, unit boundaries, and spacing evidence.

:class:`RetainedTimeView` captures the aligned time (and optional panel unit) metadata
of a retained estimation sample by original row positions:

- unit identifiers are encoded by first appearance of their exact values, never by
  sorting potentially heterogeneous unit labels directly;
- time values are validated as exact integers (never coerced to ``float64`` or
  truncated to fixed-width machine integers) so that timestamps beyond ``2**53`` and
  differences exceeding signed 64-bit subtraction limits stay exact;
- rows are ordered stably by ``(unit_code, time, position)`` (or ``(time, position)``
  for a single series), with the exact inverse permutation and unit boundary slices
  ``[unit_starts[g], unit_stops[g])`` recorded;
- within-unit time differences, ties, gaps relative to ``step``, and frequency
  multiplicity flags are recorded as evidence without imposing a blanket HAC
  capability verdict at retention time: an explicit ``lag=0`` request permits ties,
  gaps, and frequency replicas, whereas a positive or omitted lag requires strict
  chronology and unit-weight observations (:meth:`RetainedTimeView.check_chronology`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from powermean.data.raw import AnyArray, Column, _owned_objects
from powermean.data.request_values import (
    NONNEGATIVE_INTEGERS,
    OMITTED,
    POSITIVE_INTEGERS,
    Omitted,
    exact_integer,
    given,
)
from powermean.errors import SpecificationError
from powermean.model.order import IndexArray
from powermean.results.owned import owned_array
from powermean.results.retained_inputs import _copy_column

__all__ = [
    "RetainedTimeView",
    "_is_missing_metadata",
    "build_retained_time_view",
]


def _is_missing_metadata(value: object) -> bool:
    """Whether a metadata element is a missing marker or a non-finite float."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float | np.floating):
        return not math.isfinite(float(value))
    return isinstance(value, np.datetime64 | np.timedelta64) and bool(np.isnat(value))


def _exact_int_time(value: object, column: str) -> int:
    """Validate that ``value`` is an exact integer (not bool or float)."""
    if isinstance(value, bool | np.bool_):
        raise SpecificationError(
            "specification.invalid_value",
            {"argument": column, "value": value, "valid": "integers"},
        )
    if type(value) is int:
        return value
    if isinstance(value, np.integer):
        return int(value)
    raise SpecificationError(
        "specification.invalid_value",
        {"argument": column, "value": value, "valid": "integers"},
    )


def _encode_units(
    raw_unit: Column, column: str
) -> tuple[IndexArray, tuple[object, ...]]:
    """Encode unit labels by first appearance without sorting heterogeneous labels."""
    values = raw_unit.values.tolist()
    missing = sum(1 for val in values if _is_missing_metadata(val))
    if missing:
        raise SpecificationError(
            "specification.metadata_missing_values",
            {"name": column, "count": missing},
        )
    labels: dict[Any, int] = {}
    unique_labels: list[object] = []
    codes = np.empty(len(values), dtype=np.intp)
    for idx, val in enumerate(values):
        code = labels.get(val)
        if code is None:
            code = len(labels)
            labels[val] = code
            unique_labels.append(val)
        codes[idx] = code
    return owned_array(codes), _owned_objects(unique_labels)


def _extract_times(raw_time: Column, column: str) -> tuple[int, ...]:
    """Validate and extract exact Python integer times from ``raw_time``."""
    values = raw_time.values.tolist()
    missing = sum(1 for val in values if _is_missing_metadata(val))
    if missing:
        raise SpecificationError(
            "specification.metadata_missing_values",
            {"name": column, "count": missing},
        )
    return tuple(_exact_int_time(val, column) for val in values)


@dataclass(frozen=True, init=False)
class RetainedTimeView:
    """Aligned temporal and panel-unit ordering evidence on the retained sample."""

    time_column: str
    unit_column: str | None
    step: int
    lag: int | Omitted
    sorted_times: tuple[int, ...]
    within_unit_diffs: tuple[int | None, ...]
    tie_count: int
    gap_count: int
    has_ties: bool
    has_gaps: bool
    min_positive_step: int | None
    max_step: int | None
    has_frequency_replicas: bool
    max_frequency_count: int | None
    _raw_time: Column
    _raw_unit: Column | None
    _unit_codes: IndexArray | None
    _unit_labels: tuple[object, ...] | None
    _order: IndexArray
    _inverse: IndexArray
    _unit_starts: IndexArray
    _unit_stops: IndexArray

    def __init__(
        self,
        *,
        time_column: str,
        raw_time: ArrayLike | Sequence[object],
        unit_column: str | None = None,
        raw_unit: ArrayLike | Sequence[object] | None = None,
        step: int = 1,
        lag: int | Omitted = OMITTED,
        frequency_counts: Sequence[int] | None = None,
    ) -> None:
        """Build an immutable temporal ordering view from sample-aligned columns."""
        if not isinstance(time_column, str) or not time_column:
            raise TypeError("time_column must be a non-empty string")
        step = exact_integer("step", step, 1, POSITIVE_INTEGERS)
        if given(lag):
            lag = exact_integer("lag", lag, 0, NONNEGATIVE_INTEGERS)
        if (unit_column is None) != (raw_unit is None):
            raise ValueError("unit_column and raw_unit must be given together")
        time_col = _copy_column(Column(cast("Any", raw_time), name="raw_time"))
        times = _extract_times(time_col, time_column)
        nobs = len(times)

        unit_col: Column | None = None
        unit_codes: IndexArray | None = None
        unit_labels: tuple[object, ...] | None = None
        if raw_unit is not None and unit_column is not None:
            unit_col = _copy_column(Column(cast("Any", raw_unit), name="raw_unit"))
            if unit_col.size != nobs:
                raise ValueError("raw_unit and raw_time must have the same length")
            unit_codes, unit_labels = _encode_units(unit_col, unit_column)

        order, inverse, starts, stops = _compute_ordering(times, unit_codes)
        sorted_times = tuple(times[int(pos)] for pos in order.tolist())
        diffs, tie_count, gap_count, min_pos, max_step = _compute_diffs(
            sorted_times, starts, step
        )
        has_replicas, max_freq = _inspect_frequencies(frequency_counts, nobs)

        object.__setattr__(self, "time_column", time_column)
        object.__setattr__(self, "unit_column", unit_column)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "lag", lag)
        object.__setattr__(self, "sorted_times", sorted_times)
        object.__setattr__(self, "within_unit_diffs", diffs)
        object.__setattr__(self, "tie_count", tie_count)
        object.__setattr__(self, "gap_count", gap_count)
        object.__setattr__(self, "has_ties", tie_count > 0)
        object.__setattr__(self, "has_gaps", gap_count > 0)
        object.__setattr__(self, "min_positive_step", min_pos)
        object.__setattr__(self, "max_step", max_step)
        object.__setattr__(self, "has_frequency_replicas", has_replicas)
        object.__setattr__(self, "max_frequency_count", max_freq)
        object.__setattr__(self, "_raw_time", time_col)
        object.__setattr__(self, "_raw_unit", unit_col)
        object.__setattr__(self, "_unit_codes", unit_codes)
        object.__setattr__(self, "_unit_labels", unit_labels)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_inverse", inverse)
        object.__setattr__(self, "_unit_starts", starts)
        object.__setattr__(self, "_unit_stops", stops)

    @property
    def raw_time(self) -> AnyArray:
        """A fresh read-only copy of the raw time values."""
        return _copy_column(self._raw_time).values

    @property
    def raw_unit(self) -> AnyArray | None:
        """A fresh read-only copy of the raw unit values, or ``None``."""
        if self._raw_unit is None:
            return None
        return _copy_column(self._raw_unit).values

    @property
    def unit_codes(self) -> IndexArray | None:
        """A fresh read-only copy of the first-appearance unit codes, or ``None``."""
        if self._unit_codes is None:
            return None
        return owned_array(self._unit_codes)

    @property
    def unit_labels(self) -> tuple[object, ...] | None:
        """A fresh tuple of owned unit labels in first-appearance order, or ``None``."""
        if self._unit_labels is None:
            return None
        return _owned_objects(self._unit_labels)

    @property
    def order(self) -> IndexArray:
        """A fresh read-only copy of the stable unit-then-time sorting permutation."""
        return owned_array(self._order)

    @property
    def inverse(self) -> IndexArray:
        """A fresh read-only copy of the inverse permutation of :attr:`order`."""
        return owned_array(self._inverse)

    @property
    def unit_starts(self) -> IndexArray:
        """A fresh read-only copy of the inclusive start indices of each unit."""
        return owned_array(self._unit_starts)

    @property
    def unit_stops(self) -> IndexArray:
        """A fresh read-only copy of the exclusive stop indices of each unit."""
        return owned_array(self._unit_stops)

    def check_chronology(self, *, lag: int | Omitted | None = None) -> None:
        """Validate chronology for ``lag`` (or :attr:`lag` when ``None``).

        An explicit ``lag == 0`` permits ties, gaps, and frequency counts above one;
        a positive or omitted lag refuses frequency replicas, repeated periods within
        a unit, and interior gaps at :attr:`step`.
        """
        effective_lag = self.lag if lag is None else lag
        if given(effective_lag):
            effective_lag = exact_integer("lag", effective_lag, 0, NONNEGATIVE_INTEGERS)
            if effective_lag == 0:
                return
        if self.has_frequency_replicas:
            raise SpecificationError("specification.hac_replicas")
        request_name = (
            "automatic HAC lag selection"
            if isinstance(effective_lag, Omitted)
            else f"HAC with lag {effective_lag}"
        )
        if self.has_ties:
            raise SpecificationError(
                "specification.time_ties",
                {
                    "name": self.time_column,
                    "count": self.tie_count,
                    "request": request_name,
                },
            )
        if self.has_gaps:
            raise SpecificationError(
                "specification.time_gaps",
                {
                    "name": self.time_column,
                    "count": self.gap_count,
                    "step": self.step,
                    "request": request_name,
                },
            )


def _compute_ordering(
    times: tuple[int, ...], unit_codes: IndexArray | None
) -> tuple[IndexArray, IndexArray, IndexArray, IndexArray]:
    """Compute stable ``order``, ``inverse``, ``unit_starts``, and ``unit_stops``."""
    nobs = len(times)
    if nobs == 0:
        empty = owned_array(np.empty(0, dtype=np.intp))
        return empty, empty, empty, empty
    if unit_codes is None:
        ordered = sorted(range(nobs), key=lambda idx: (times[idx], idx))
        starts = np.array([0], dtype=np.intp)
        stops = np.array([nobs], dtype=np.intp)
    else:
        codes_list = [int(c) for c in unit_codes.tolist()]
        ordered = sorted(
            range(nobs), key=lambda idx: (codes_list[idx], times[idx], idx)
        )
        sorted_codes = [codes_list[idx] for idx in ordered]
        start_list: list[int] = [0]
        stop_list: list[int] = []
        for pos in range(1, nobs):
            if sorted_codes[pos] != sorted_codes[pos - 1]:
                stop_list.append(pos)
                start_list.append(pos)
        stop_list.append(nobs)
        starts = np.array(start_list, dtype=np.intp)
        stops = np.array(stop_list, dtype=np.intp)

    order_arr = np.array(ordered, dtype=np.intp)
    inverse_arr = np.empty(nobs, dtype=np.intp)
    inverse_arr[order_arr] = np.arange(nobs, dtype=np.intp)
    return (
        owned_array(order_arr),
        owned_array(inverse_arr),
        owned_array(starts),
        owned_array(stops),
    )


def _compute_diffs(
    sorted_times: tuple[int, ...], starts: IndexArray, step: int
) -> tuple[tuple[int | None, ...], int, int, int | None, int | None]:
    """Compute exact within-unit time differences and tie/gap counts."""
    start_set = {int(s) for s in starts.tolist()}
    diffs: list[int | None] = []
    repeated_periods: set[tuple[int, int]] = set()
    gap_count = 0
    min_pos: int | None = None
    max_step: int | None = None
    unit_idx = -1
    for pos, current in enumerate(sorted_times):
        if pos in start_set:
            unit_idx += 1
            diffs.append(None)
            continue
        delta = current - sorted_times[pos - 1]
        diffs.append(delta)
        if max_step is None or delta > max_step:
            max_step = delta
        if delta == 0:
            repeated_periods.add((unit_idx, current))
        else:
            if min_pos is None or delta < min_pos:
                min_pos = delta
            if delta != step:
                gap_count += 1
    return tuple(diffs), len(repeated_periods), gap_count, min_pos, max_step


def _inspect_frequencies(
    frequency_counts: Sequence[int] | None, nobs: int
) -> tuple[bool, int | None]:
    """Inspect optional exact frequency counts for replicas above one."""
    if frequency_counts is None:
        return False, None
    if len(frequency_counts) != nobs:
        raise ValueError("frequency_counts must have length equal to raw_time")
    if not frequency_counts:
        return False, None
    counts = [int(c) for c in frequency_counts]
    max_freq = max(counts)
    return max_freq > 1, max_freq


def build_retained_time_view(
    *,
    time_column: str,
    raw_time: ArrayLike | Sequence[object],
    unit_column: str | None = None,
    raw_unit: ArrayLike | Sequence[object] | None = None,
    step: int = 1,
    lag: int | Omitted = OMITTED,
    frequency_counts: Sequence[int] | None = None,
) -> RetainedTimeView:
    """Build a :class:`RetainedTimeView` from sample-aligned columns."""
    return RetainedTimeView(
        time_column=time_column,
        raw_time=raw_time,
        unit_column=unit_column,
        raw_unit=raw_unit,
        step=step,
        lag=lag,
        frequency_counts=frequency_counts,
    )
