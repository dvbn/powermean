"""Geographic coordinate validation, pole/antimeridian canonicalization, and membership.

:class:`RetainedSpaceView` captures the aligned latitude and longitude metadata of a
retained estimation sample by original row positions:

- raw coordinates are validated as finite real numbers in geographic degrees
  (``latitude in [-90, 90]``, ``longitude in [-180, 180]``) and retained as owned
  arrays;
- exact pole aliases (``abs(latitude) == 90`` -> ``canonical_longitude = 0.0``),
  antimeridian aliases (``longitude == 180.0`` -> ``-180.0``), and signed zero
  (``-0.0`` -> ``+0.0``) are canonicalized for location membership;
- canonical locations are numbered by first appearance, with both the row-to-location
  map and the inverse location-to-rows membership preserved;
- physical row counts per location and frequency-replica multiplicities are recorded
  distinctly alongside ``requires_pooling_confirmation`` and ``pool_times_confirmed``,
  while distance matrices, automatic cutoff heuristics, kernel weights, and score
  pooling are not formed here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from powermean.data.inference_requests import ConleyRequest, Distance, DistanceUnits
from powermean.data.raw import Column
from powermean.data.request_values import (
    CONFIRMATION,
    OMITTED,
    Omitted,
    given,
    invalid,
)
from powermean.data.request_values import (
    cutoff as check_cutoff,
)
from powermean.errors import SpecificationError
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.owned import owned_array

__all__ = [
    "RetainedSpaceView",
    "build_retained_space_view",
    "canonicalize_coordinates",
]


def _is_missing_coord(value: object) -> bool:
    """Whether a coordinate value is a missing marker or NaN."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float | np.floating):
        return math.isnan(float(value))
    return isinstance(value, np.datetime64 | np.timedelta64) and bool(np.isnat(value))


def _validate_degree(
    value: object, column: str, low: float, high: float, label: str
) -> float:
    """Validate that ``value`` is a finite real degree in ``[low, high]``."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise SpecificationError(
            "specification.coordinates_invalid",
            {
                "name": column,
                "detail": f"{label} {value!r} is not a real degree coordinate",
            },
        )
    try:
        coord = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise SpecificationError(
            "specification.coordinates_invalid",
            {
                "name": column,
                "detail": f"{label} {value!r} lies outside [{low:g}, {high:g}]",
            },
        ) from exc
    if not math.isfinite(coord) or coord < low or coord > high:
        raise SpecificationError(
            "specification.coordinates_invalid",
            {
                "name": column,
                "detail": f"{label} {value!r} lies outside [{low:g}, {high:g}]",
            },
        )
    return coord


def canonicalize_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    """Canonicalize a validated ``(latitude, longitude)`` degree pair.

    Normalizes ``-0.0`` to ``0.0``, maps all longitudes at ``latitude == 90.0`` or
    ``latitude == -90.0`` to ``0.0``, and maps ``longitude == 180.0`` to ``-180.0``.
    """
    lat = 0.0 if latitude == 0.0 else float(latitude)
    lon = 0.0 if longitude == 0.0 else float(longitude)
    if abs(lat) == 90.0:
        return lat, 0.0
    if lon == 180.0:
        return lat, -180.0
    return lat, lon


def _extract_coordinates(
    raw_lat: Column,
    lat_col: str,
    raw_lon: Column,
    lon_col: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate raw latitude/longitude columns and form canonical coordinate arrays."""
    lat_vals = raw_lat.values.tolist()
    lon_vals = raw_lon.values.tolist()
    if len(lat_vals) != len(lon_vals):
        raise ValueError("raw_latitude and raw_longitude must have the same length")
    lat_missing = sum(1 for val in lat_vals if _is_missing_coord(val))
    if lat_missing:
        raise SpecificationError(
            "specification.metadata_missing_values",
            {"name": lat_col, "count": lat_missing},
        )
    lon_missing = sum(1 for val in lon_vals if _is_missing_coord(val))
    if lon_missing:
        raise SpecificationError(
            "specification.metadata_missing_values",
            {"name": lon_col, "count": lon_missing},
        )
    nobs = len(lat_vals)
    raw_lat_arr = np.empty(nobs, dtype=np.float64)
    raw_lon_arr = np.empty(nobs, dtype=np.float64)
    can_lat_arr = np.empty(nobs, dtype=np.float64)
    can_lon_arr = np.empty(nobs, dtype=np.float64)
    for idx in range(nobs):
        lat = _validate_degree(lat_vals[idx], lat_col, -90.0, 90.0, "latitude")
        lon = _validate_degree(lon_vals[idx], lon_col, -180.0, 180.0, "longitude")
        can_lat, can_lon = canonicalize_coordinates(lat, lon)
        raw_lat_arr[idx] = lat
        raw_lon_arr[idx] = lon
        can_lat_arr[idx] = can_lat
        can_lon_arr[idx] = can_lon
    return (
        owned_array(raw_lat_arr),
        owned_array(raw_lon_arr),
        owned_array(can_lat_arr),
        owned_array(can_lon_arr),
    )


def _build_membership(
    can_lat: FloatArray,
    can_lon: FloatArray,
    multiplicities: tuple[int, ...],
) -> tuple[
    FloatArray,
    FloatArray,
    IndexArray,
    tuple[IndexArray, ...],
    IndexArray,
    tuple[int, ...],
]:
    """Encode unique canonical locations by first appearance and build membership."""
    nobs = int(can_lat.size)
    loc_map: dict[tuple[float, float], int] = {}
    unique_lats: list[float] = []
    unique_lons: list[float] = []
    buckets: list[list[int]] = []
    row_to_loc = np.empty(nobs, dtype=np.intp)
    for row in range(nobs):
        pair = (float(can_lat[row]), float(can_lon[row]))
        loc = loc_map.get(pair)
        if loc is None:
            loc = len(loc_map)
            loc_map[pair] = loc
            unique_lats.append(pair[0])
            unique_lons.append(pair[1])
            buckets.append([])
        row_to_loc[row] = loc
        buckets[loc].append(row)

    loc_to_rows = tuple(owned_array(np.array(rows, dtype=np.intp)) for rows in buckets)
    row_counts = owned_array(np.array([len(rows) for rows in buckets], dtype=np.intp))
    replica_counts = tuple(sum(multiplicities[row] for row in rows) for rows in buckets)
    return (
        owned_array(np.array(unique_lats, dtype=np.float64)),
        owned_array(np.array(unique_lons, dtype=np.float64)),
        owned_array(row_to_loc),
        loc_to_rows,
        row_counts,
        replica_counts,
    )


@dataclass(frozen=True, init=False)
class RetainedSpaceView:
    """Aligned geographic coordinates, canonical aliases, and location membership."""

    latitude_column: str
    longitude_column: str
    n_locations: int
    row_multiplicities: tuple[int, ...]
    location_replica_counts: tuple[int, ...]
    repeated_location_count: int
    requires_pooling_confirmation: bool
    pool_times_confirmed: bool
    cutoff: float | Omitted
    units: DistanceUnits
    distance: Distance
    _raw_latitude: FloatArray
    _raw_longitude: FloatArray
    _canonical_latitude: FloatArray
    _canonical_longitude: FloatArray
    _unique_canonical_latitude: FloatArray
    _unique_canonical_longitude: FloatArray
    _row_to_location: IndexArray
    _location_to_rows: tuple[IndexArray, ...]
    _location_row_counts: IndexArray

    def __init__(
        self,
        *,
        latitude_column: str,
        longitude_column: str,
        raw_latitude: ArrayLike | Sequence[object],
        raw_longitude: ArrayLike | Sequence[object],
        frequency_counts: Sequence[int] | None = None,
        pool_times: Literal[True] | Omitted | None = None,
        request: ConleyRequest | None = None,
    ) -> None:
        """Build an immutable spatial membership view from sample-aligned columns."""
        if not isinstance(latitude_column, str) or not latitude_column:
            raise TypeError("latitude_column must be a non-empty string")
        if not isinstance(longitude_column, str) or not longitude_column:
            raise TypeError("longitude_column must be a non-empty string")
        lat_col = Column(cast("Any", raw_latitude), name="raw_latitude")
        lon_col = Column(cast("Any", raw_longitude), name="raw_longitude")
        raw_lat, raw_lon, can_lat, can_lon = _extract_coordinates(
            lat_col, latitude_column, lon_col, longitude_column
        )
        nobs = int(raw_lat.size)
        if frequency_counts is None:
            multiplicities = (1,) * nobs
        else:
            if len(frequency_counts) != nobs:
                raise ValueError(
                    "frequency_counts must have length equal to raw_latitude"
                )
            multiplicities = tuple(int(c) for c in frequency_counts)
            if any(c < 1 for c in multiplicities):
                raise ValueError("frequency_counts must be positive integers")

        (
            uniq_lat,
            uniq_lon,
            row_to_loc,
            loc_to_rows,
            row_counts,
            replica_counts,
        ) = _build_membership(can_lat, can_lon, multiplicities)
        repeated_locs = sum(1 for count in replica_counts if count > 1)
        resolved_pool = (
            (request.pool_times if request is not None else OMITTED)
            if pool_times is None
            else pool_times
        )
        if given(resolved_pool) and resolved_pool is not True:
            raise invalid("pool_times", resolved_pool, CONFIRMATION)
        cutoff = OMITTED if request is None else request.cutoff
        units: DistanceUnits = "km" if request is None else request.resolved_units
        distance: Distance = (
            "triangular" if request is None else request.resolved_distance
        )

        object.__setattr__(self, "latitude_column", latitude_column)
        object.__setattr__(self, "longitude_column", longitude_column)
        object.__setattr__(self, "n_locations", len(loc_to_rows))
        object.__setattr__(self, "row_multiplicities", multiplicities)
        object.__setattr__(self, "location_replica_counts", replica_counts)
        object.__setattr__(self, "repeated_location_count", repeated_locs)
        object.__setattr__(self, "requires_pooling_confirmation", repeated_locs > 0)
        object.__setattr__(self, "pool_times_confirmed", resolved_pool is True)
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "_raw_latitude", raw_lat)
        object.__setattr__(self, "_raw_longitude", raw_lon)
        object.__setattr__(self, "_canonical_latitude", can_lat)
        object.__setattr__(self, "_canonical_longitude", can_lon)
        object.__setattr__(self, "_unique_canonical_latitude", uniq_lat)
        object.__setattr__(self, "_unique_canonical_longitude", uniq_lon)
        object.__setattr__(self, "_row_to_location", row_to_loc)
        object.__setattr__(self, "_location_to_rows", loc_to_rows)
        object.__setattr__(self, "_location_row_counts", row_counts)

    @property
    def raw_latitude(self) -> FloatArray:
        """A fresh read-only copy of the raw latitude values in degrees."""
        return owned_array(self._raw_latitude)

    @property
    def raw_longitude(self) -> FloatArray:
        """A fresh read-only copy of the raw longitude values in degrees."""
        return owned_array(self._raw_longitude)

    @property
    def canonical_latitude(self) -> FloatArray:
        """A fresh read-only copy of the canonicalized latitude values in degrees."""
        return owned_array(self._canonical_latitude)

    @property
    def canonical_longitude(self) -> FloatArray:
        """A fresh read-only copy of the canonicalized longitude values in degrees."""
        return owned_array(self._canonical_longitude)

    @property
    def unique_canonical_latitude(self) -> FloatArray:
        """A fresh read-only copy of unique canonical latitudes by first appearance."""
        return owned_array(self._unique_canonical_latitude)

    @property
    def unique_canonical_longitude(self) -> FloatArray:
        """A fresh read-only copy of unique canonical longitudes by first appearance."""
        return owned_array(self._unique_canonical_longitude)

    @property
    def row_to_location(self) -> IndexArray:
        """A fresh read-only copy of the row-to-location index map."""
        return owned_array(self._row_to_location)

    @property
    def location_to_rows(self) -> tuple[IndexArray, ...]:
        """A tuple of fresh read-only row-index arrays for each canonical location."""
        return tuple(owned_array(rows) for rows in self._location_to_rows)

    @property
    def location_row_counts(self) -> IndexArray:
        """A fresh read-only copy of the physical row counts per canonical location."""
        return owned_array(self._location_row_counts)

    def check_conley_requirements(
        self,
        *,
        cutoff: float | Omitted | None = None,
        pool_times: Literal[True] | Omitted | None = None,
    ) -> None:
        """Validate pooling confirmation and automatic-cutoff location count."""
        effective_pool = (
            (True if self.pool_times_confirmed else OMITTED)
            if pool_times is None
            else pool_times
        )
        if given(effective_pool) and effective_pool is not True:
            raise invalid("pool_times", effective_pool, CONFIRMATION)
        if self.requires_pooling_confirmation and effective_pool is not True:
            raise SpecificationError(
                "specification.conley_pooling",
                {"count": self.repeated_location_count},
            )
        effective_cutoff = self.cutoff if cutoff is None else cutoff
        if given(effective_cutoff):
            check_cutoff(effective_cutoff)
        elif self.n_locations < 4:
            raise SpecificationError(
                "specification.conley_locations",
                {"count": self.n_locations},
            )


def build_retained_space_view(
    *,
    latitude_column: str,
    longitude_column: str,
    raw_latitude: ArrayLike | Sequence[object],
    raw_longitude: ArrayLike | Sequence[object],
    frequency_counts: Sequence[int] | None = None,
    pool_times: Literal[True] | Omitted | None = None,
    request: ConleyRequest | None = None,
) -> RetainedSpaceView:
    """Build a :class:`RetainedSpaceView` from sample-aligned coordinate columns."""
    return RetainedSpaceView(
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        raw_latitude=raw_latitude,
        raw_longitude=raw_longitude,
        frequency_counts=frequency_counts,
        pool_times=pool_times,
        request=request,
    )
