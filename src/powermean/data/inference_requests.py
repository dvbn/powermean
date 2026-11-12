"""The canonical records of the expanded covariance requests.

Internal records.  Nothing here parses a ``vcov`` value, reads data, resolves a
convention or computes: the stages that implement the requests form these records from
what a caller passes.

==========================  ======================================================
:class:`HCRequest`          ``vcov="HC0"`` to ``"HC5"`` and ``"HC4m"``, exact strings
:class:`ClusterRequest`     cluster dimensions, the multiway form and the ``"psd"``
                            key, valid on a signed multiway request only
:class:`HacRequest`         ``{"hac": time, "unit": ..., "lag": ..., "step": ...}``
:class:`ConleyRequest`      ``{"conley": [lat, lon], "cutoff": ..., "units": ...,
                            "distance": ..., "pool_times": True}``
==========================  ======================================================

The omitted and explicit settings, the value rules and the serialization are those of
:mod:`powermean.data.request_values`: a setting that a request omits holds
:data:`~powermean.data.request_values.OMITTED`, ``None`` is never an omission, and
``from_dict`` reads a mapping only.  Units need an explicit cutoff.  No request record
has a field ``ssc``: a convention is a keyword of the call, never a key of a covariance
mapping, and its record,
:class:`~powermean.data.inference_conventions.ConventionValues`, is in
:mod:`powermean.data.inference_conventions` with the retained declarations.

Not decided here, and left to the stages that compute the requests: whether the data
hold an integer time on a regular grid, ties or gaps, coordinates in range, enough
distinct locations, the repeated locations or frequency counts that require
``pool_times``; the bound of an explicit lag, so that no bound is read from the wording
of ``specification.hac_lag_too_long``; leverage and residual degrees of freedom; the
convention in force, its source and the owner of a factor; retained metadata, their
alignment and missing values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from powermean.data.inference_conventions import PSD_TREATMENTS, PsdTreatment
from powermean.data.request_values import (
    AUTOMATIC_CUTOFF,
    CONFIRMATION,
    COORDINATES,
    DIMENSIONS,
    NONNEGATIVE_INTEGERS,
    OMITTED,
    POSITIVE_INTEGERS,
    Omitted,
    choice,
    column_name,
    column_names,
    cutoff,
    exact_integer,
    field_names,
    given,
    invalid,
    serialized,
    settings,
)
from powermean.errors import SpecificationError

HCMethod = Literal["HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5"]
Multiway = Literal["cgm", "ddg"]
PairAdjustment = Literal["none", "wz"]
DistanceUnits = Literal["km", "mi"]
Distance = Literal["triangular", "spherical"]

HC_METHODS: tuple[str, ...] = get_args(HCMethod)
MULTIWAY_FORMS: tuple[str, ...] = get_args(Multiway)
DISTANCE_UNITS: tuple[str, ...] = get_args(DistanceUnits)
DISTANCES: tuple[str, ...] = get_args(Distance)


@dataclass(frozen=True)
class HCRequest:
    """A named HC covariance: ``vcov="HC0"`` to ``"HC5"`` and ``"HC4m"``.

    The method owns its scalar factor and takes the reference distribution of the
    convention in force; neither is resolved here.

    Parameters
    ----------
    method : {"HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5"}
        The method, an exact string: ``"hc3"`` and ``"HC4M"`` are refused.

    Raises
    ------
    SpecificationError
        Another value (``specification.invalid_value``, argument ``vcov``).
    """

    method: HCMethod

    def __post_init__(self) -> None:
        """Check the method by exact string."""
        object.__setattr__(self, "method", choice("vcov", self.method, HC_METHODS))

    def to_dict(self) -> dict[str, object]:
        """The type and the method."""
        return serialized("HCRequest", {"method": self.method})

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> HCRequest:
        """The record of :meth:`to_dict` (:mod:`powermean.data.request_values`)."""
        return cls(**settings(record, "HCRequest", field_names(cls)))


@dataclass(frozen=True)
class ClusterRequest:
    """Cluster dimensions, their multiway form and the ``"psd"`` key.

    Parameters
    ----------
    dimensions : tuple of str
        One or more cluster variables, an interaction ``a^b`` among them, in their
        order; a repeated dimension is kept.
    multiway : {"cgm", "ddg"}, optional
        The multiway form; omitted, the signed form ``"cgm"`` with two or more
        dimensions.
    psd : {"cgm", "none"}, optional
        The treatment of an indefinite signed estimate, given for this
        request; valid on a signed multiway request only: two or more dimensions and
        the signed form.  Omitted, the convention in force decides.

    Raises
    ------
    SpecificationError
        A value outside its domain (``specification.invalid_value``); ``psd`` with one
        dimension or with ``multiway="ddg"`` (``specification.conflicting``).
    """

    dimensions: tuple[str, ...]
    multiway: Multiway | Omitted = OMITTED
    psd: PsdTreatment | Omitted = OMITTED
    adjustment: PairAdjustment | Omitted = OMITTED

    def __post_init__(self) -> None:
        """Check the dimensions, the form and where ``psd`` is given."""
        dimensions = column_names("cluster", self.dimensions, DIMENSIONS, 1)
        object.__setattr__(self, "dimensions", dimensions)
        if given(self.multiway):
            multiway = choice("multiway", self.multiway, MULTIWAY_FORMS)
            object.__setattr__(self, "multiway", multiway)
        if given(self.psd):
            psd = choice("psd", self.psd, PSD_TREATMENTS)
            object.__setattr__(self, "psd", psd)
            if not self.signed:
                second = (
                    "one cluster dimension" if len(dimensions) < 2 else "multiway='ddg'"
                )
                raise SpecificationError(
                    "specification.conflicting",
                    {"first": f"psd={psd!r}", "second": second},
                )

        if given(self.adjustment):
            adjustment = choice("adjustment", self.adjustment, ("none", "wz"))
            object.__setattr__(self, "adjustment", adjustment)
            if adjustment == "wz" and (
                len(dimensions) != 1 or given(self.multiway) or given(self.psd)
            ):
                raise SpecificationError(
                    "specification.conflicting",
                    {"first": "adjustment='wz'", "second": "multiway or psd"},
                )

    @property
    def signed(self) -> bool:
        """Whether the request is signed multiway: two or more dimensions, not DDG."""
        return len(self.dimensions) >= 2 and self.multiway != "ddg"

    def to_dict(self) -> dict[str, object]:
        """The type, the dimensions as a list, and the form and ``psd`` if given."""
        return serialized(
            "ClusterRequest",
            {
                "dimensions": self.dimensions,
                "multiway": self.multiway,
                "psd": self.psd,
                "adjustment": self.adjustment,
            },
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ClusterRequest:
        """The record of :meth:`to_dict` (:mod:`powermean.data.request_values`)."""
        return cls(**settings(record, "ClusterRequest", field_names(cls)))


@dataclass(frozen=True)
class HacRequest:
    """A HAC covariance: ``{"hac": time, "unit": ..., "lag": ..., "step": ...}``.

    No kernel, prewhitening or other setting exists: the kernel is Bartlett, without
    prewhitening.

    Parameters
    ----------
    time : str
        The time variable, the key ``"hac"``; that it holds integers on a regular grid
        is checked on the data when the request is computed.
    unit : str, optional
        The panel unit; omitted, one series.
    lag : int, optional
        A nonnegative integer, zero included; omitted, the automatic rule
        selects it.  ``None`` is refused.
    step : int, optional
        The step of the time grid, a positive integer; omitted, one period.

    Raises
    ------
    SpecificationError
        A value outside its domain (``specification.invalid_value``).
    """

    time: str
    unit: str | Omitted = OMITTED
    lag: int | Omitted = OMITTED
    step: int | Omitted = OMITTED

    def __post_init__(self) -> None:
        """Check each setting given; ``None`` is a value, not an omission."""
        object.__setattr__(self, "time", column_name("hac", self.time))
        if given(self.unit):
            object.__setattr__(self, "unit", column_name("unit", self.unit))
        if given(self.lag):
            lag = exact_integer("lag", self.lag, 0, NONNEGATIVE_INTEGERS)
            object.__setattr__(self, "lag", lag)
        if given(self.step):
            step = exact_integer("step", self.step, 1, POSITIVE_INTEGERS)
            object.__setattr__(self, "step", step)

    @property
    def automatic_lag(self) -> bool:
        """Whether the lag is omitted, so that the automatic rule selects it."""
        return isinstance(self.lag, Omitted)

    @property
    def resolved_step(self) -> int:
        """The step given, or one period where it is omitted."""
        return 1 if isinstance(self.step, Omitted) else self.step

    def to_dict(self) -> dict[str, object]:
        """The type, the time variable and the settings given."""
        return serialized(
            "HacRequest",
            {"time": self.time, "unit": self.unit, "lag": self.lag, "step": self.step},
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> HacRequest:
        """The record of :meth:`to_dict` (:mod:`powermean.data.request_values`)."""
        return cls(**settings(record, "HacRequest", field_names(cls)))


@dataclass(frozen=True)
class ConleyRequest:
    """A Conley covariance.

    ``{"conley": [lat, lon], "cutoff": ..., "units": ..., "distance": ...,
    "pool_times": True}``.

    Parameters
    ----------
    latitude, longitude : str
        The two coordinate variables, in degrees, the key ``"conley"`` in that order;
        their range is checked on the data when the request is computed.
    cutoff : float, optional
        A nonnegative finite number, zero included; omitted, the automatic rule
        selects it in kilometers.
    units : {"km", "mi"}, optional
        The units of an explicit cutoff; omitted, kilometers.  Refused without a
        cutoff.
    distance : {"triangular", "spherical"}, optional
        The distance; omitted, triangular.
    pool_times : True, optional
        The confirmation of pooling across times at one location, which
        is required where a location holds several rows or a frequency
        count above one; any value other than ``True`` is refused.

    Raises
    ------
    SpecificationError
        A value outside its domain (``specification.invalid_value``); units without a
        cutoff (``specification.conflicting``).
    """

    latitude: str
    longitude: str
    cutoff: float | Omitted = OMITTED
    units: DistanceUnits | Omitted = OMITTED
    distance: Distance | Omitted = OMITTED
    pool_times: Literal[True] | Omitted = OMITTED

    def __post_init__(self) -> None:
        """Check each setting given, then units against an omitted cutoff."""
        latitude = column_name("conley", self.latitude, COORDINATES)
        object.__setattr__(self, "latitude", latitude)
        longitude = column_name("conley", self.longitude, COORDINATES)
        object.__setattr__(self, "longitude", longitude)
        if given(self.cutoff):
            object.__setattr__(self, "cutoff", cutoff(self.cutoff))
        if given(self.units):
            units = choice("units", self.units, DISTANCE_UNITS)
            object.__setattr__(self, "units", units)
            if self.automatic_cutoff:
                raise SpecificationError(
                    "specification.conflicting",
                    {"first": f"units={units!r}", "second": AUTOMATIC_CUTOFF},
                )
        if given(self.distance):
            distance = choice("distance", self.distance, DISTANCES)
            object.__setattr__(self, "distance", distance)
        if given(self.pool_times) and self.pool_times is not True:
            raise invalid("pool_times", self.pool_times, CONFIRMATION)

    @property
    def automatic_cutoff(self) -> bool:
        """Whether the cutoff is omitted, so that the automatic rule selects it."""
        return isinstance(self.cutoff, Omitted)

    @property
    def resolved_units(self) -> DistanceUnits:
        """The units given, or kilometers where they are omitted."""
        return "km" if isinstance(self.units, Omitted) else self.units

    @property
    def resolved_distance(self) -> Distance:
        """The distance given, or the triangular distance where it is omitted."""
        return "triangular" if isinstance(self.distance, Omitted) else self.distance

    def to_dict(self) -> dict[str, object]:
        """The type, the two coordinate variables and the settings given."""
        return serialized(
            "ConleyRequest",
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "cutoff": self.cutoff,
                "units": self.units,
                "distance": self.distance,
                "pool_times": self.pool_times,
            },
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ConleyRequest:
        """The record of :meth:`to_dict` (:mod:`powermean.data.request_values`)."""
        return cls(**settings(record, "ConleyRequest", field_names(cls)))
