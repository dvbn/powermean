"""Detached inner state snapshots: recoveries, weights, index starts, and Schur records.

This module defines :func:`_owned_normalization`, :class:`RetainedRecovery`,
:func:`_owned_recovery`, :class:`RetainedPreparedWeights`, :class:`RetainedIndexStart`,
:class:`RetainedLocalInitializer` (all 14 fields including ``implementation``),
:func:`_snapshot_local_initializer`, :class:`RetainedAbsorbedSchur`, and
:func:`_owned_absorbed_schur`, and re-exports profile/absorption state records from
:mod:`powermean.results.absorbed_state_records`.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.fe.absorb_acceleration import CONVERGED
from powermean.fe.absorb_numeric import PreparedWeights
from powermean.fe.connectivity import IndicatorRank
from powermean.fe.groups import Grouping
from powermean.fe.profile import Work as ProfileWork
from powermean.fe.profile_arithmetic import EquationCheck
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
    detach_grouping,
)
from powermean.results.absorbed_state_records import (
    RetainedAbsorptionResult,
    RetainedAbsorptionState,
    RetainedProfileResult,
    RetainedProfileState,
    _owned_absorption_result,
    _owned_equation_check,
    _owned_fresh_check,
    _owned_profile_result,
    _resolve_state_grouping,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_input_values import _tuple_arrays
from powermean.solve.absorbed_direction import IndexStart, Normalization, Recovery
from powermean.solve.absorbed_start import IMPLEMENTATION, LocalInitializer
from powermean.systems.profiled import Absorbed

Family = PowerVariance | NB2Variance

__all__ = [
    "Family",
    "RetainedAbsorbedSchur",
    "RetainedAbsorptionResult",
    "RetainedAbsorptionState",
    "RetainedIndexStart",
    "RetainedLocalInitializer",
    "RetainedPreparedWeights",
    "RetainedProfileResult",
    "RetainedProfileState",
    "RetainedRecovery",
    "_owned_absorbed_schur",
    "_owned_absorption_result",
    "_owned_equation_check",
    "_owned_fresh_check",
    "_owned_normalization",
    "_owned_profile_result",
    "_owned_recovery",
    "_resolve_state_grouping",
    "_snapshot_local_initializer",
]


def _owned_normalization(n: Normalization | None) -> Normalization | None:
    """Return a fresh ``Normalization`` with owned arrays, or ``None``."""
    if n is None:
        return None
    return Normalization(
        _tuple_arrays(n.effects),
        _tuple_arrays(n.references),
        _tuple_arrays(n.shifts),
    )


@dataclass(frozen=True)
class RetainedRecovery:
    """Detached snapshot of ``Recovery`` on the fitted-index route."""

    tolerance: float
    _eta: FloatArray | None
    absorption: RetainedAbsorptionResult | None
    _effects: tuple[FloatArray, ...] | None
    gap: float | None
    _normalization: Normalization | None
    normalized_gap: float | None
    rank: IndicatorRank
    reason: str

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_eta", _optional(self._eta))
        if self._effects is not None:
            object.__setattr__(self, "_effects", _tuple_arrays(self._effects))
        object.__setattr__(
            self, "_normalization", _owned_normalization(self._normalization)
        )

    @classmethod
    def from_recovery(
        cls,
        rec: Recovery,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedRecovery:
        """Detach a ``Recovery`` into a :class:`RetainedRecovery`."""
        ab = _owned_absorption_result(
            rec.absorption,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(
            float(rec.tolerance),
            rec.eta,
            ab,
            rec.effects,
            None if rec.gap is None else float(rec.gap),
            rec.normalization,
            None if rec.normalized_gap is None else float(rec.normalized_gap),
            rec.rank,
            str(rec.reason),
        )

    @property
    def eta(self) -> FloatArray | None:
        """Return a fresh owned copy of ``eta = q - base``, or ``None``."""
        return _optional(self._eta)

    @property
    def effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of the recovered block effects, or ``None``."""
        return None if self._effects is None else _tuple_arrays(self._effects)

    @property
    def normalization(self) -> Normalization | None:
        """Return a fresh owned ``Normalization``, or ``None``."""
        return _owned_normalization(self._normalization)

    @property
    def complete(self) -> bool:
        """Return True if recovery and normalization completed within tolerance."""
        return not self.reason


def _owned_recovery(
    rec: Recovery | None,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedRecovery | None:
    """Detach an optional ``Recovery``."""
    if rec is None:
        return None
    return RetainedRecovery.from_recovery(
        rec, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
    )


@dataclass(frozen=True)
class RetainedPreparedWeights:
    """Detached snapshot of ``PreparedWeights`` without re-running group sums."""

    _grouping: RetainedGroupingSnapshot
    _weights: FloatArray
    _totals: tuple[FloatArray, ...]

    def __init__(
        self,
        grouping: Grouping | RetainedGroupingSnapshot,
        weights: FloatArray,
        totals: tuple[FloatArray, ...],
    ) -> None:
        """Detach ``grouping`` and store owned copies of ``weights`` and ``totals``."""
        object.__setattr__(self, "_grouping", detach_grouping(grouping))
        object.__setattr__(self, "_weights", owned_array(weights))
        object.__setattr__(self, "_totals", _tuple_arrays(totals))

    @classmethod
    def from_prepared(
        cls,
        pw: PreparedWeights,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedPreparedWeights:
        """Detach a ``PreparedWeights`` record."""
        grp = _resolve_state_grouping(
            pw.grouping,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(grp, pw.weights, pw.totals)

    @property
    def grouping(self) -> RetainedGroupingSnapshot:
        """Return a freshly materialized :class:`RetainedGroupingSnapshot`."""
        return self._grouping.rematerialize()

    @property
    def weights(self) -> FloatArray:
        """Return a fresh owned copy of the row weights."""
        return owned_array(self._weights)

    @property
    def totals(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the per-block level weight totals."""
        return _tuple_arrays(self._totals)

    @property
    def n_rows(self) -> int:
        """Return the number of rows."""
        return int(self._weights.size)

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks."""
        return len(self._totals)


@dataclass(frozen=True)
class RetainedIndexStart:
    """Detached snapshot of ``IndexStart`` with owned slope array on every access."""

    effects: RetainedProfileState
    _slopes: FloatArray

    @classmethod
    def from_start(
        cls,
        s: IndexStart,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedIndexStart:
        """Detach an ``IndexStart`` record."""
        eff = RetainedProfileState.from_state(
            s.effects,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(eff, owned_array(s.slopes))

    @property
    def slopes(self) -> FloatArray:
        """Return a fresh owned copy of ``slopes``."""
        return owned_array(self._slopes)


@dataclass(frozen=True)
class RetainedLocalInitializer:
    """Detached snapshot of all 14 fields of ``LocalInitializer``."""

    _theta: FloatArray
    _base: FloatArray
    variance: Family
    start: RetainedProfileState
    state: RetainedProfileState
    _index: FloatArray
    _initial: EquationCheck | None
    _check: EquationCheck | None
    status: str
    reason: str
    tolerance: float
    max_sweeps: int
    work: ProfileWork
    implementation: str = IMPLEMENTATION

    @property
    def theta(self) -> FloatArray:
        """Return a fresh owned copy of ``theta``."""
        return owned_array(self._theta)

    @property
    def base(self) -> FloatArray:
        """Return a fresh owned copy of ``base``."""
        return owned_array(self._base)

    @property
    def index(self) -> FloatArray:
        """Return a fresh owned copy of ``index``."""
        return owned_array(self._index)

    @property
    def initial(self) -> EquationCheck | None:
        """Return a fresh owned initial ``EquationCheck``, or ``None``."""
        return _owned_equation_check(self._initial)

    @property
    def check(self) -> EquationCheck | None:
        """Return a fresh owned final ``EquationCheck``, or ``None``."""
        return _owned_equation_check(self._check)

    @property
    def converged(self) -> bool:
        """Return True when ``status == 'converged'``."""
        return self.status == CONVERGED

    @property
    def maximum(self) -> float | None:
        """Return the largest absolute local equation at ``state``, or ``None``."""
        return None if self._check is None else self._check.maximum

    @property
    def sweeps(self) -> int:
        """Return the number of completed profile sweeps."""
        return int(self.work.sweeps_completed)

    @property
    def blocks(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the state block coefficient arrays."""
        return self.state.coefficients


def _snapshot_local_initializer(
    ir: LocalInitializer,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedLocalInitializer:
    """Detach all 14 ``LocalInitializer`` fields into a snapshot."""
    s0 = RetainedProfileState.from_state(
        ir.start,
        grouping_snapshot=grouping_snapshot,
        grouping_map=grouping_map,
    )
    s1 = RetainedProfileState.from_state(
        ir.state,
        grouping_snapshot=grouping_snapshot,
        grouping_map=grouping_map,
    )
    return RetainedLocalInitializer(
        owned_array(ir.theta),
        owned_array(ir.base),
        ir.variance,
        s0,
        s1,
        owned_array(ir.index),
        _owned_equation_check(ir.initial),
        _owned_equation_check(ir.check),
        str(ir.status),
        str(ir.reason),
        float(ir.tolerance),
        int(ir.max_sweeps),
        ir.work,
        str(ir.implementation),
    )


@dataclass(frozen=True)
class RetainedAbsorbedSchur:
    """Detached snapshot of ``Absorbed`` Schur record with owned arrays."""

    result: RetainedAbsorptionResult
    failure: str
    _weights: FloatArray
    _matrix: FloatArray | None = None

    @property
    def weights(self) -> FloatArray:
        """Return a fresh owned copy of ``weights``."""
        return owned_array(self._weights)

    @property
    def matrix(self) -> FloatArray | None:
        """Return a fresh owned copy of ``matrix``, or ``None``."""
        return _optional(self._matrix)


def _owned_absorbed_schur(
    ab: Absorbed | None,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedAbsorbedSchur | None:
    """Detach an optional ``Absorbed`` Schur record."""
    if ab is None:
        return None
    res = _owned_absorption_result(
        ab.result,
        grouping_snapshot=grouping_snapshot,
        grouping_map=grouping_map,
    )
    assert res is not None
    return RetainedAbsorbedSchur(
        res, str(ab.failure), owned_array(ab.weights), _optional(ab.matrix)
    )
