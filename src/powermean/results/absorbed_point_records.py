"""Detached point, linear normalization, and fresh equation snapshot records.

This leaf module defines :data:`Family`, :data:`ReferenceTuple`,
:func:`_owned_references`, :class:`RetainedLinearNormalization`,
:class:`RetainedFreshEquations`, and :class:`RetainedPointSnapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from powermean.fe.connectivity import IndicatorRank
from powermean.fe.profile_nb2 import ScalarWork
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_state_records import RetainedProfileResult
from powermean.results.absorbed_states import RetainedRecovery
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import (
    PartitionKey,
    PointKey,
    SampleKey,
    _tuple_arrays,
)
from powermean.solve.absorbed_certificate_point import InnerControls

Family = PowerVariance | NB2Variance
ReferenceTuple = tuple[IndexArray | FloatArray, ...]

__all__ = [
    "Family",
    "ReferenceTuple",
    "RetainedFreshEquations",
    "RetainedLinearNormalization",
    "RetainedPointSnapshot",
    "_owned_references",
]


def _owned_references(refs: ReferenceTuple) -> ReferenceTuple:
    """Return fresh owned copies of reference arrays."""
    return tuple(owned_array(r) for r in refs)


@dataclass(frozen=True)
class RetainedLinearNormalization:
    """Detached owned snapshot of ``PointNormalization`` without live ``FESample``."""

    point_key: PointKey
    sample_key: SampleKey
    _effects: tuple[FloatArray, ...]
    _references: ReferenceTuple
    _shifts: tuple[FloatArray, ...]
    _index: FloatArray
    gap: float
    rank: IndicatorRank

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "point_key", PointKey(int(self.point_key)))
        object.__setattr__(self, "sample_key", SampleKey(int(self.sample_key)))
        object.__setattr__(self, "_effects", _tuple_arrays(self._effects))
        object.__setattr__(self, "_references", _owned_references(self._references))
        object.__setattr__(self, "_shifts", _tuple_arrays(self._shifts))
        object.__setattr__(self, "_index", owned_array(self._index))
        object.__setattr__(self, "gap", float(self.gap))

    @property
    def effects(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the normalized block coefficient vectors."""
        return _tuple_arrays(self._effects)

    @property
    def references(self) -> ReferenceTuple:
        """Return fresh owned copies of the reference level vectors."""
        return _owned_references(self._references)

    @property
    def shifts(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the component shift vectors."""
        return _tuple_arrays(self._shifts)

    @property
    def index(self) -> FloatArray:
        """Return a fresh owned copy of the reconstructed normalized index."""
        return owned_array(self._index)

    def within(self, tolerance: float) -> bool:
        """Return True if ``gap <= tolerance``."""
        return bool(self.gap <= tolerance)


@dataclass(frozen=True)
class RetainedFreshEquations:
    """Detached owned snapshot of ``FreshEquations`` without live ``FESample``."""

    point_key: PointKey
    sample_key: SampleKey
    _residual: FloatArray
    _weighted: FloatArray
    _slope_sums: FloatArray
    _level_sums: tuple[FloatArray, ...]
    _level_totals: tuple[FloatArray, ...]
    _level_means: tuple[FloatArray, ...]
    largest_level_mean: float

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "point_key", PointKey(int(self.point_key)))
        object.__setattr__(self, "sample_key", SampleKey(int(self.sample_key)))
        object.__setattr__(self, "_residual", owned_array(self._residual))
        object.__setattr__(self, "_weighted", owned_array(self._weighted))
        object.__setattr__(self, "_slope_sums", owned_array(self._slope_sums))
        object.__setattr__(self, "_level_sums", _tuple_arrays(self._level_sums))
        object.__setattr__(self, "_level_totals", _tuple_arrays(self._level_totals))
        object.__setattr__(self, "_level_means", _tuple_arrays(self._level_means))
        object.__setattr__(self, "largest_level_mean", float(self.largest_level_mean))

    @property
    def residual(self) -> FloatArray:
        """Return a fresh owned copy of ``residual``."""
        return owned_array(self._residual)

    @property
    def weighted(self) -> FloatArray:
        """Return a fresh owned copy of ``weighted``."""
        return owned_array(self._weighted)

    @property
    def slope_sums(self) -> FloatArray:
        """Return a fresh owned copy of ``slope_sums``."""
        return owned_array(self._slope_sums)

    @property
    def level_sums(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of ``level_sums``."""
        return _tuple_arrays(self._level_sums)

    @property
    def level_totals(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of ``level_totals``."""
        return _tuple_arrays(self._level_totals)

    @property
    def level_means(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of ``level_means``."""
        return _tuple_arrays(self._level_means)


@dataclass(frozen=True)
class RetainedPointSnapshot:
    """Detached snapshot of an absorbed point with result-local keys.

    A ``"root"`` point (an exact point of the full joint system) keeps its kept
    incidence coordinates (``coordinates``), its effects and their accepted
    normalization; ``converged`` is ``None`` there: no inner approximation formed it.
    A ``"reduced"`` intermediate point keeps full raw effects and retained outer
    coordinates, not a fictitious all-FE profile result or convergence flag.
    """

    key: PointKey
    sample_key: SampleKey
    partition_key: PartitionKey | None
    kind: Literal["profiled", "reduced", "index", "linear", "root"]
    sample: RetainedSampleSnapshot
    variance: Family | None
    inner: InnerControls | None
    n_slopes: int
    converged: bool | None
    _theta: FloatArray
    _beta: FloatArray
    _base: FloatArray
    _q: FloatArray
    profile_result: RetainedProfileResult | None = None
    scalar_work: ScalarWork | None = None
    recovery: RetainedRecovery | None = None
    linear_normalization: RetainedLinearNormalization | None = None
    _raw_effects: tuple[FloatArray, ...] | None = None
    _normalized_effects: tuple[FloatArray, ...] | None = None
    _references: ReferenceTuple | None = None
    _shifts: tuple[FloatArray, ...] | None = None
    equations: RetainedFreshEquations | None = None
    _coordinates: FloatArray | None = None

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_theta", owned_array(self._theta))
        object.__setattr__(self, "_beta", owned_array(self._beta))
        object.__setattr__(self, "_base", owned_array(self._base))
        object.__setattr__(self, "_q", owned_array(self._q))
        if self._coordinates is not None:
            object.__setattr__(self, "_coordinates", owned_array(self._coordinates))
        if self._raw_effects is not None:
            object.__setattr__(self, "_raw_effects", _tuple_arrays(self._raw_effects))
        if self._normalized_effects is not None:
            object.__setattr__(
                self, "_normalized_effects", _tuple_arrays(self._normalized_effects)
            )
        if self._references is not None:
            object.__setattr__(self, "_references", _owned_references(self._references))
        if self._shifts is not None:
            object.__setattr__(self, "_shifts", _tuple_arrays(self._shifts))

    @property
    def route(self) -> Literal["profiled", "reduced", "index", "linear", "root"]:
        """Return the absorbed route kind."""
        return self.kind

    @property
    def coordinates(self) -> FloatArray | None:
        """Return detached retained-nonlinear or full root coordinates."""
        return None if self._coordinates is None else owned_array(self._coordinates)

    @property
    def theta(self) -> FloatArray:
        """Return a fresh owned copy of ``theta``."""
        return owned_array(self._theta)

    @property
    def beta(self) -> FloatArray:
        """Return a fresh owned copy of ``beta``."""
        return owned_array(self._beta)

    @property
    def base(self) -> FloatArray:
        """Return a fresh owned copy of ``base``."""
        return owned_array(self._base)

    @property
    def q(self) -> FloatArray:
        """Return a fresh owned copy of the authoritative fitted index ``q``."""
        return owned_array(self._q)

    @property
    def index(self) -> FloatArray:
        """Return a fresh owned copy of the authoritative fitted index ``q``."""
        return owned_array(self._q)

    @property
    def raw_effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of the raw block coefficients, or ``None``."""
        return None if self._raw_effects is None else _tuple_arrays(self._raw_effects)

    @property
    def effects(self) -> tuple[FloatArray, ...] | None:
        """Compatibility alias for ``raw_effects``."""
        return self.raw_effects

    @property
    def normalized_effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of the normalized block coefficients."""
        return (
            None
            if self._normalized_effects is None
            else _tuple_arrays(self._normalized_effects)
        )

    @property
    def references(self) -> ReferenceTuple | None:
        """Return fresh owned copies of the reference levels, or ``None``."""
        return None if self._references is None else _owned_references(self._references)

    @property
    def shifts(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of the component shifts, or ``None``."""
        return None if self._shifts is None else _tuple_arrays(self._shifts)

    @property
    def profile(self) -> RetainedProfileResult | None:
        """Compatibility alias for ``profile_result``."""
        return self.profile_result
