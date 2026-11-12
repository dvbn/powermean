"""Detached snapshots of primal and dual separation witnesses and column sums.

This module defines :class:`RetainedColumnSums`, :class:`RetainedPrimalWitness`,
:class:`RetainedDualCheck`, :func:`_detach_column_sums`, :func:`detach_primal_witness`,
and :func:`detach_dual_check`, and re-exports :class:`RetainedSupportProblem` from
:mod:`powermean.results.retained_diagnostic_problems`.  Each witness detaches its own
``w.problem`` / ``d.problem`` without parent substitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_diagnostic_problems import (
    Family,
    MarginalTuple,
    RetainedSupportProblem,
    _owned_marginals,
)
from powermean.results.retained_input_values import ProblemKey, _tuple_arrays
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.solve.support_witness import ColumnSums, DualWitness, PrimalWitness

__all__ = [
    "Family",
    "MarginalTuple",
    "RetainedColumnSums",
    "RetainedDualCheck",
    "RetainedPrimalWitness",
    "RetainedSupportProblem",
    "_detach_column_sums",
    "_owned_marginals",
    "detach_dual_check",
    "detach_primal_witness",
]


@dataclass(frozen=True)
class RetainedColumnSums:
    """Detached snapshot of ``ColumnSums`` with owned arrays on every exposure."""

    _numerators: FloatArray
    _denominators: FloatArray
    _ratios: FloatArray

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_numerators", owned_array(self._numerators))
        object.__setattr__(self, "_denominators", owned_array(self._denominators))
        object.__setattr__(self, "_ratios", owned_array(self._ratios))

    @classmethod
    def from_sums(cls, c: ColumnSums) -> RetainedColumnSums:
        """Detach a ``ColumnSums`` record."""
        return cls(c.numerators, c.denominators, c.ratios)

    @property
    def numerators(self) -> FloatArray:
        """Return a fresh owned copy of ``numerators``."""
        return owned_array(self._numerators)

    @property
    def denominators(self) -> FloatArray:
        """Return a fresh owned copy of ``denominators``."""
        return owned_array(self._denominators)

    @property
    def ratios(self) -> FloatArray:
        """Return a fresh owned copy of ``ratios``."""
        return owned_array(self._ratios)

    def passing(self, tolerance: float) -> bool:
        """Return whether every ratio is at most ``tolerance``."""
        return bool((self._ratios <= tolerance).all())


def _detach_column_sums(c: ColumnSums | None) -> RetainedColumnSums | None:
    """Detach an optional ``ColumnSums`` record."""
    return None if c is None else RetainedColumnSums.from_sums(c)


def _empty_index() -> IndexArray:
    """Return a fresh owned empty index array."""
    return owned_array(np.zeros(0, dtype=np.intp))


@dataclass(frozen=True)
class RetainedPrimalWitness:
    """Detached snapshot of ``PrimalWitness`` preserving optional failed/raw fields."""

    problem_key: ProblemKey
    problem: RetainedSupportProblem
    method: str
    tolerance: float
    threshold: float
    failure: tuple[str, ...]
    _proposed_slopes: FloatArray | None = None
    _proposed_effects: tuple[FloatArray, ...] | None = None
    _raw: FloatArray | None = None
    scale: float | None = None
    _slopes: FloatArray | None = None
    _effects: tuple[FloatArray, ...] | None = None
    _z: FloatArray | None = None
    positive_max: float | None = None
    zero_min: float | None = None
    _strict: IndexArray = field(default_factory=_empty_index)
    _unresolved: IndexArray = field(default_factory=_empty_index)

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_proposed_slopes", _optional(self._proposed_slopes))
        if self._proposed_effects is not None:
            object.__setattr__(
                self, "_proposed_effects", _tuple_arrays(self._proposed_effects)
            )
        object.__setattr__(self, "_raw", _optional(self._raw))
        object.__setattr__(self, "_slopes", _optional(self._slopes))
        if self._effects is not None:
            object.__setattr__(self, "_effects", _tuple_arrays(self._effects))
        object.__setattr__(self, "_z", _optional(self._z))
        object.__setattr__(self, "_strict", owned_array(self._strict))
        object.__setattr__(self, "_unresolved", owned_array(self._unresolved))

    @classmethod
    def from_witness(
        cls,
        w: PrimalWitness,
        *,
        problem_snapshot: RetainedSupportProblem | None = None,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedPrimalWitness:
        """Detach a ``PrimalWitness`` using its own ``w.problem`` identity."""
        prob = (
            RetainedSupportProblem.from_problem(
                w.problem,
                register_problem=(
                    None if problem_map is None else problem_map.register_support
                ),
                detach_group=(
                    None if partition_map is None else partition_map.register_grouping
                ),
            )
            if problem_map is not None or problem_snapshot is None
            else problem_snapshot
        )
        return cls(
            prob.key,
            prob,
            str(w.method),
            float(w.tolerance),
            float(w.threshold),
            tuple(str(f) for f in w.failure),
            w.proposed_slopes,
            w.proposed_effects,
            w.raw,
            None if w.scale is None else float(w.scale),
            w.slopes,
            w.effects,
            w.z,
            None if w.positive_max is None else float(w.positive_max),
            None if w.zero_min is None else float(w.zero_min),
            w.strict,
            w.unresolved,
        )

    @property
    def proposed_slopes(self) -> FloatArray | None:
        """Return a fresh owned copy of ``proposed_slopes``, or ``None``."""
        return _optional(self._proposed_slopes)

    @property
    def proposed_effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of ``proposed_effects``, or ``None``."""
        return (
            None
            if self._proposed_effects is None
            else _tuple_arrays(self._proposed_effects)
        )

    @property
    def raw(self) -> FloatArray | None:
        """Return a fresh owned copy of ``raw``, or ``None``."""
        return _optional(self._raw)

    @property
    def slopes(self) -> FloatArray | None:
        """Return a fresh owned copy of ``slopes``, or ``None``."""
        return _optional(self._slopes)

    @property
    def effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of ``effects``, or ``None``."""
        return None if self._effects is None else _tuple_arrays(self._effects)

    @property
    def z(self) -> FloatArray | None:
        """Return a fresh owned copy of ``z``, or ``None``."""
        return _optional(self._z)

    @property
    def strict(self) -> IndexArray:
        """Return a fresh owned copy of ``strict``."""
        return owned_array(self._strict)

    @property
    def unresolved(self) -> IndexArray:
        """Return a fresh owned copy of ``unresolved``."""
        return owned_array(self._unresolved)

    @property
    def valid(self) -> bool:
        """Return True when every primal predicate holds."""
        return not self.failure

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of the original positions of ``strict``."""
        return owned_array(self.problem.positions[self._strict])


@dataclass(frozen=True)
class RetainedDualCheck:
    """Detached snapshot of ``DualWitness`` preserving optional failed/raw fields."""

    problem_key: ProblemKey
    problem: RetainedSupportProblem
    tolerance: float
    threshold: float
    _raw: FloatArray
    _separated: IndexArray
    _constructed: FloatArray
    failure: tuple[str, ...]
    normalizer: float | None = None
    _normalized: FloatArray | None = None
    margin: float | None = None
    slopes: RetainedColumnSums | None = None
    levels: tuple[RetainedColumnSums, ...] = ()

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_raw", owned_array(self._raw))
        object.__setattr__(self, "_separated", owned_array(self._separated))
        object.__setattr__(self, "_constructed", owned_array(self._constructed))
        object.__setattr__(self, "_normalized", _optional(self._normalized))

    @classmethod
    def from_witness(
        cls,
        d: DualWitness,
        *,
        problem_snapshot: RetainedSupportProblem | None = None,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedDualCheck:
        """Detach a ``DualWitness`` using its own ``d.problem`` identity."""
        prob = (
            RetainedSupportProblem.from_problem(
                d.problem,
                register_problem=(
                    None if problem_map is None else problem_map.register_support
                ),
                detach_group=(
                    None if partition_map is None else partition_map.register_grouping
                ),
            )
            if problem_map is not None or problem_snapshot is None
            else problem_snapshot
        )
        sl = _detach_column_sums(d.slopes)
        lvls = tuple(RetainedColumnSums.from_sums(s) for s in d.levels)
        return cls(
            prob.key,
            prob,
            float(d.tolerance),
            float(d.threshold),
            d.raw,
            d.separated,
            d.constructed,
            tuple(str(f) for f in d.failure),
            None if d.normalizer is None else float(d.normalizer),
            d.normalized,
            None if d.margin is None else float(d.margin),
            sl,
            lvls,
        )

    @property
    def raw(self) -> FloatArray:
        """Return a fresh owned copy of ``raw``."""
        return owned_array(self._raw)

    @property
    def separated(self) -> IndexArray:
        """Return a fresh owned copy of ``separated``."""
        return owned_array(self._separated)

    @property
    def constructed(self) -> FloatArray:
        """Return a fresh owned copy of ``constructed``."""
        return owned_array(self._constructed)

    @property
    def normalized(self) -> FloatArray | None:
        """Return a fresh owned copy of ``normalized``, or ``None``."""
        return _optional(self._normalized)

    @property
    def passed(self) -> bool:
        """Return True when every dual check holds."""
        return not self.failure


def detach_primal_witness(
    w: PrimalWitness,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedPrimalWitness:
    """Detach ``w`` from its own ``w.problem`` identity."""
    return RetainedPrimalWitness.from_witness(
        w, problem_map=problem_map, partition_map=partition_map
    )


def detach_dual_check(
    d: DualWitness,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedDualCheck:
    """Detach ``d`` from its own ``d.problem`` identity."""
    return RetainedDualCheck.from_witness(
        d, problem_map=problem_map, partition_map=partition_map
    )
