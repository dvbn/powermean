"""Detached profile and absorption state and result records.

This module defines :func:`_owned_equation_check`, :func:`_owned_fresh_check`,
:class:`RetainedProfileState`, :class:`RetainedProfileResult`,
:func:`_owned_profile_result`, :class:`RetainedAbsorptionState`,
:class:`RetainedAbsorptionResult`, and :func:`_owned_absorption_result`.  Each state
detaches its own ``state.grouping`` rather than blindly substituting a parent grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe.absorb import AbsorptionResult, AbsorptionState
from powermean.fe.absorb_acceleration import CONVERGED
from powermean.fe.absorb_numeric import FreshCheck
from powermean.fe.groups import Grouping
from powermean.fe.profile import ProfileResult, ProfileState
from powermean.fe.profile import Work as ProfileWork
from powermean.fe.profile_arithmetic import EquationCheck
from powermean.model.variance import FloatArray
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
    detach_grouping,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_input_values import (
    _same_column_values,
    _tuple_arrays,
)
from powermean.results.retained_labels import _stored_raw_elements

__all__ = [
    "RetainedAbsorptionResult",
    "RetainedAbsorptionState",
    "RetainedProfileResult",
    "RetainedProfileState",
    "_owned_absorption_result",
    "_owned_equation_check",
    "_owned_fresh_check",
    "_owned_profile_result",
    "_resolve_state_grouping",
]


def _owned_equation_check(c: EquationCheck | None) -> EquationCheck | None:
    """Return a fresh ``EquationCheck`` with owned error arrays, or ``None``."""
    if c is None:
        return None
    return EquationCheck(_tuple_arrays(c.errors), float(c.maximum))


def _owned_fresh_check(c: FreshCheck | None) -> FreshCheck | None:
    """Return a fresh ``FreshCheck`` with owned column errors, or ``None``."""
    if c is None:
        return None
    return FreshCheck(owned_array(c.column_errors), float(c.maximum))


def _grouping_matches_snapshot(
    grouping: Grouping, snapshot: RetainedGroupingSnapshot
) -> bool:
    """Return True when ``grouping`` matches the structure and labels of snapshot."""
    if (
        grouping.names != snapshot.names
        or grouping.n_rows != snapshot.n_rows
        or tuple(b.n_levels for b in grouping.blocks) != snapshot.n_levels
        or not np.array_equal(grouping.positions, snapshot._positions)
    ):
        return False
    for blk, sblk in zip(grouping.blocks, snapshot.blocks, strict=True):
        if not np.array_equal(blk.codes, sblk._codes) or not np.array_equal(
            blk.counts, sblk._counts
        ):
            return False
        raw_lvls = _stored_raw_elements(blk._levels)
        if len(raw_lvls) != len(sblk._levels_tuple):
            return False
        for a, b in zip(raw_lvls, sblk._levels_tuple, strict=True):
            if type(a) is not type(b) or a != b:
                return False
    if (snapshot._columns_tuple is not None) != hasattr(grouping, "source"):
        return False
    if snapshot._columns_tuple is not None and hasattr(grouping, "source"):
        src_cols = grouping.source.columns
        if tuple(src_cols.keys()) != tuple(snapshot._columns_tuple.keys()):
            return False
        for name, col in src_cols.items():
            snap_arr = snapshot.columns
            if snap_arr is None or not _same_column_values(col.values, snap_arr[name]):
                return False
    return True


def _resolve_state_grouping(
    grouping: Grouping,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedGroupingSnapshot:
    """Detach ``grouping`` via ``grouping_map``, or reuse a matching snapshot.

    The snapshot returned is an internally owned record (the map's, or the matching
    one given); the state record that receives it owns its own copy at its ingress and
    rematerializes it on every outward exposure.
    """
    if grouping_map is not None:
        return grouping_map.register(grouping)
    if grouping_snapshot is not None and _grouping_matches_snapshot(
        grouping, grouping_snapshot
    ):
        return (
            grouping_snapshot
            if grouping_snapshot._copy_free
            else grouping_snapshot.rematerialize()
        )
    return detach_grouping(grouping)


@dataclass(frozen=True)
class RetainedProfileState:
    """Detached snapshot of ``ProfileState`` without re-encoding ``Grouping``."""

    _grouping: RetainedGroupingSnapshot
    _coefficients: tuple[FloatArray, ...]

    def __init__(
        self,
        grouping: Grouping | RetainedGroupingSnapshot,
        coefficients: tuple[FloatArray, ...],
    ) -> None:
        """Detach ``grouping`` and store owned copies of ``coefficients``."""
        object.__setattr__(self, "_grouping", detach_grouping(grouping))
        object.__setattr__(self, "_coefficients", _tuple_arrays(coefficients))

    @classmethod
    def from_state(
        cls,
        state: ProfileState,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedProfileState:
        """Detach a ``ProfileState`` into a :class:`RetainedProfileState`."""
        grp = _resolve_state_grouping(
            state.grouping,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(grp, state.coefficients)

    @property
    def grouping(self) -> RetainedGroupingSnapshot:
        """Return a freshly materialized :class:`RetainedGroupingSnapshot`."""
        return self._grouping.rematerialize()

    @property
    def coefficients(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the block coefficient vectors."""
        return _tuple_arrays(self._coefficients)


@dataclass(frozen=True)
class RetainedProfileResult:
    """Detached snapshot of ``ProfileResult`` with owned arrays and states."""

    _index: FloatArray
    _state: RetainedProfileState
    _start: RetainedProfileState
    status: str
    reason: str
    tolerance: float
    max_sweeps: int
    _errors: tuple[FloatArray, ...] | None
    maximum: float | None
    work: ProfileWork

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_index", owned_array(self._index))
        if self._errors is not None:
            object.__setattr__(self, "_errors", _tuple_arrays(self._errors))

    @classmethod
    def from_result(
        cls,
        r: ProfileResult,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedProfileResult:
        """Detach a ``ProfileResult`` into a :class:`RetainedProfileResult`."""
        st = RetainedProfileState.from_state(
            r.state,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        st0 = RetainedProfileState.from_state(
            r.start,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(
            r.index,
            st,
            st0,
            str(r.status),
            str(r.reason),
            float(r.tolerance),
            int(r.max_sweeps),
            r.errors,
            None if r.maximum is None else float(r.maximum),
            r.work,
        )

    @property
    def index(self) -> FloatArray:
        """Return a fresh owned copy of the reconstructed index."""
        return owned_array(self._index)

    @property
    def state(self) -> RetainedProfileState:
        """Return the detached final :class:`RetainedProfileState`."""
        return self._state

    @property
    def start(self) -> RetainedProfileState:
        """Return the detached starting :class:`RetainedProfileState`."""
        return self._start

    @property
    def errors(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of the block equation errors, or ``None``."""
        return None if self._errors is None else _tuple_arrays(self._errors)

    @property
    def converged(self) -> bool:
        """Return True when ``status == 'converged'``."""
        return self.status == CONVERGED


def _owned_profile_result(
    r: ProfileResult | None,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedProfileResult | None:
    """Detach an optional ``ProfileResult``."""
    if r is None:
        return None
    return RetainedProfileResult.from_result(
        r, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
    )


@dataclass(frozen=True)
class RetainedAbsorptionState:
    """Detached snapshot of ``AbsorptionState`` without re-encoding ``Grouping``."""

    _grouping: RetainedGroupingSnapshot
    columns: tuple[str, ...]
    _coefficients: tuple[FloatArray, ...]

    def __init__(
        self,
        grouping: Grouping | RetainedGroupingSnapshot,
        columns: tuple[str, ...],
        coefficients: tuple[FloatArray, ...],
    ) -> None:
        """Detach ``grouping`` and store owned copies of ``coefficients``."""
        object.__setattr__(self, "_grouping", detach_grouping(grouping))
        object.__setattr__(self, "columns", tuple(str(c) for c in columns))
        object.__setattr__(self, "_coefficients", _tuple_arrays(coefficients))

    @classmethod
    def from_state(
        cls,
        state: AbsorptionState,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedAbsorptionState:
        """Detach an ``AbsorptionState`` into a :class:`RetainedAbsorptionState`."""
        grp = _resolve_state_grouping(
            state.grouping,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(grp, state.columns, state.coefficients)

    @property
    def grouping(self) -> RetainedGroupingSnapshot:
        """Return a freshly materialized :class:`RetainedGroupingSnapshot`."""
        return self._grouping.rematerialize()

    @property
    def coefficients(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the block coefficient matrices."""
        return _tuple_arrays(self._coefficients)

    @property
    def n_columns(self) -> int:
        """Return the number of absorbed columns."""
        return len(self.columns)


@dataclass(frozen=True)
class RetainedAbsorptionResult:
    """Detached snapshot of ``AbsorptionResult`` with owned arrays and state."""

    _residual: FloatArray
    _state: RetainedAbsorptionState
    status: str
    reason: str
    tolerance: float
    _column_errors: FloatArray | None
    maximum: float | None
    sweeps: int
    visits: int
    operators: int
    diagnostic_passes: int
    acceleration: bool

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_residual", owned_array(self._residual))
        object.__setattr__(self, "_column_errors", _optional(self._column_errors))

    @classmethod
    def from_result(
        cls,
        r: AbsorptionResult,
        *,
        grouping_snapshot: RetainedGroupingSnapshot | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> RetainedAbsorptionResult:
        """Detach an ``AbsorptionResult`` into a :class:`RetainedAbsorptionResult`."""
        st = RetainedAbsorptionState.from_state(
            r.state,
            grouping_snapshot=grouping_snapshot,
            grouping_map=grouping_map,
        )
        return cls(
            r.residual,
            st,
            str(r.status),
            str(r.reason),
            float(r.tolerance),
            r.column_errors,
            None if r.maximum is None else float(r.maximum),
            int(r.sweeps),
            int(r.visits),
            int(r.operators),
            int(r.diagnostic_passes),
            bool(r.acceleration),
        )

    @property
    def residual(self) -> FloatArray:
        """Return a fresh owned copy of the absorbed residual matrix."""
        return owned_array(self._residual)

    @property
    def state(self) -> RetainedAbsorptionState:
        """Return the detached :class:`RetainedAbsorptionState`."""
        return self._state

    @property
    def column_errors(self) -> FloatArray | None:
        """Return a fresh owned copy of ``column_errors``, or ``None``."""
        return _optional(self._column_errors)

    @property
    def converged(self) -> bool:
        """Return True when ``status == 'converged'``."""
        return self.status == CONVERGED


def _owned_absorption_result(
    r: AbsorptionResult | None,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedAbsorptionResult | None:
    """Detach an optional ``AbsorptionResult``."""
    if r is None:
        return None
    return RetainedAbsorptionResult.from_result(
        r, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
    )
