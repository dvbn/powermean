"""Detached snapshots of support, recession, and collinearity diagnostic problems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from powermean.data.collinearity_records import CollinearityProblem
from powermean.data.weights import ObservationWeights
from powermean.fe.groups import Grouping
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_groupings import (
    RetainedGroupingSnapshot,
    detach_grouping,
)
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import PartitionKey, ProblemKey
from powermean.solve.support_gamma_problem import GammaProblem
from powermean.solve.support_problem import SupportProblem

Family = PowerVariance | NB2Variance
MarginalTuple = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
GroupingCallback = Callable[[Grouping], RetainedGroupingSnapshot]
SupportProblemCallback = Callable[
    [SupportProblem, Callable[[ProblemKey], "RetainedSupportProblem"]],
    "RetainedSupportProblem",
]
GammaProblemCallback = Callable[
    [GammaProblem, Callable[[ProblemKey], "RetainedGammaProblem"]],
    "RetainedGammaProblem",
]
CollinearityProblemCallback = Callable[
    [CollinearityProblem, Callable[[ProblemKey], "RetainedCollinearityProblem"]],
    "RetainedCollinearityProblem",
]

__all__ = [
    "CollinearityProblemCallback",
    "Family",
    "GammaProblemCallback",
    "GroupingCallback",
    "MarginalTuple",
    "RetainedCollinearityProblem",
    "RetainedGammaProblem",
    "RetainedSupportProblem",
    "SupportProblemCallback",
    "_owned_marginals",
    "_owned_observation_weights",
]


def _owned_marginals(m: MarginalTuple | None) -> MarginalTuple | None:
    """Return fresh owned copies of a 4-tuple of marginal arrays, or ``None``."""
    if m is None:
        return None
    return (
        owned_array(m[0]),
        owned_array(m[1]),
        owned_array(m[2]),
        owned_array(m[3]),
    )


def _owned_observation_weights(w: ObservationWeights) -> ObservationWeights:
    """Return a fresh ``ObservationWeights`` with owned values and preserved kind."""
    return ObservationWeights(owned_array(w.values), w.kind)


def _detach_problem_grouping(
    grouping: object,
    *,
    detach_group: GroupingCallback | None = None,
) -> tuple[RetainedGroupingSnapshot | None, PartitionKey | None]:
    """Detach an optional diagnostic problem ``Grouping`` and return its key."""
    if grouping is None:
        return None, None
    assert isinstance(grouping, Grouping)
    grp = detach_grouping(grouping) if detach_group is None else detach_group(grouping)
    return grp, grp.partition_key


@dataclass(frozen=True)
class RetainedSupportProblem:
    """Detached snapshot of ``SupportProblem`` with result-local keys."""

    key: ProblemKey
    _z: FloatArray
    _design: FloatArray
    columns: tuple[str, ...]
    _grouping: RetainedGroupingSnapshot | None
    partition_key: PartitionKey | None
    _positions: IndexArray
    _weights: FloatArray
    _offset: FloatArray
    variance: Family
    omega: float

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        for name in ("_z", "_design", "_positions", "_weights", "_offset"):
            object.__setattr__(self, name, owned_array(getattr(self, name)))

    @classmethod
    def from_problem(
        cls,
        problem: SupportProblem,
        *,
        register_problem: SupportProblemCallback | None = None,
        detach_group: GroupingCallback | None = None,
    ) -> RetainedSupportProblem:
        """Detach a ``SupportProblem`` by its own concrete object identity."""
        if type(problem) is not SupportProblem:
            raise TypeError("problem must be a concrete SupportProblem")

        def _build(key: ProblemKey) -> RetainedSupportProblem:
            grp, pkey = _detach_problem_grouping(
                problem.grouping,
                detach_group=detach_group,
            )
            cols = tuple(str(c) for c in problem.columns)
            head = (key, problem.z, problem.design, cols, grp, pkey)
            tail = (
                problem.positions,
                problem.weights,
                problem.offset,
                problem.variance,
                float(problem.omega),
            )
            return cls(*head, *tail)

        if register_problem is not None:
            return register_problem(problem, _build)
        return _build(ProblemKey(0))

    @property
    def z(self) -> FloatArray:
        """Return a fresh owned copy of ``z``."""
        return owned_array(self._z)

    @property
    def design(self) -> FloatArray:
        """Return a fresh owned copy of ``design``."""
        return owned_array(self._design)

    @property
    def grouping(self) -> RetainedGroupingSnapshot | None:
        """Return a fresh rematerialized grouping snapshot, or ``None``."""
        return None if self._grouping is None else self._grouping.rematerialize()

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of ``positions``."""
        return owned_array(self._positions)

    @property
    def weights(self) -> FloatArray:
        """Return a fresh owned copy of ``weights``."""
        return owned_array(self._weights)

    @property
    def offset(self) -> FloatArray:
        """Return a fresh owned copy of ``offset``."""
        return owned_array(self._offset)

    @property
    def n_rows(self) -> int:
        """Return the number of rows."""
        return int(self._z.size)

    @property
    def n_slopes(self) -> int:
        """Return the number of slope columns."""
        return int(self._design.shape[1])

    @property
    def n_blocks(self) -> int:
        """Return the number of fixed-effect blocks."""
        return 0 if self._grouping is None else len(self._grouping.blocks)

    @property
    def zero(self) -> IndexArray:
        """Return indices of rows with zero outcome."""
        return owned_array(np.flatnonzero(self._z == 0).astype(np.intp))

    @property
    def positive(self) -> IndexArray:
        """Return indices of rows with positive outcome."""
        return owned_array(np.flatnonzero(self._z > 0).astype(np.intp))


@dataclass(frozen=True)
class RetainedGammaProblem:
    """Detached snapshot of ``GammaProblem`` with freshly owned weights."""

    key: ProblemKey
    _z: FloatArray
    _design: FloatArray
    columns: tuple[str, ...]
    _grouping: RetainedGroupingSnapshot | None
    partition_key: PartitionKey | None
    _positions: IndexArray
    _weights: ObservationWeights
    _offset: FloatArray
    variance: PowerVariance
    omega: float

    def __post_init__(self) -> None:
        """Own all stored numeric arrays and typed observation weights."""
        for name in ("_z", "_design", "_positions", "_offset"):
            object.__setattr__(self, name, owned_array(getattr(self, name)))
        object.__setattr__(self, "_weights", _owned_observation_weights(self._weights))

    @classmethod
    def from_problem(
        cls,
        problem: GammaProblem,
        *,
        register_problem: GammaProblemCallback | None = None,
        detach_group: GroupingCallback | None = None,
    ) -> RetainedGammaProblem:
        """Detach a ``GammaProblem`` by its own concrete object identity."""
        if type(problem) is not GammaProblem:
            raise TypeError("problem must be a concrete GammaProblem")

        def _build(key: ProblemKey) -> RetainedGammaProblem:
            grp, pkey = _detach_problem_grouping(
                problem.grouping,
                detach_group=detach_group,
            )
            cols = tuple(str(c) for c in problem.columns)
            head = (key, problem.z, problem.design, cols, grp, pkey)
            tail = (
                problem.positions,
                problem.weights,
                problem.offset,
                problem.variance,
                float(problem.omega),
            )
            return cls(*head, *tail)

        if register_problem is not None:
            return register_problem(problem, _build)
        return _build(ProblemKey(0))

    @property
    def z(self) -> FloatArray:
        """Return a fresh owned copy of ``z``."""
        return owned_array(self._z)

    @property
    def design(self) -> FloatArray:
        """Return a fresh owned copy of ``design``."""
        return owned_array(self._design)

    @property
    def grouping(self) -> RetainedGroupingSnapshot | None:
        """Return a fresh rematerialized grouping snapshot, or ``None``."""
        return None if self._grouping is None else self._grouping.rematerialize()

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of ``positions``."""
        return owned_array(self._positions)

    @property
    def weights(self) -> ObservationWeights:
        """Return fresh owned ``ObservationWeights``."""
        return _owned_observation_weights(self._weights)

    @property
    def offset(self) -> FloatArray:
        """Return a fresh owned copy of ``offset``."""
        return owned_array(self._offset)

    @property
    def n_rows(self) -> int:
        """Return the number of rows."""
        return int(self._z.size)

    @property
    def n_slopes(self) -> int:
        """Return the number of slope columns."""
        return int(self._design.shape[1])

    @property
    def n_blocks(self) -> int:
        """Return the number of fixed-effect blocks."""
        return 0 if self._grouping is None else len(self._grouping.blocks)

    @property
    def n_parameters(self) -> int:
        """Return total slope plus fixed-effect level parameters."""
        levels = (
            0
            if self._grouping is None
            else sum(b.n_levels for b in self._grouping.blocks)
        )
        return self.n_slopes + levels

    @property
    def positive(self) -> IndexArray:
        """Return indices of rows with positive outcome."""
        return owned_array(np.flatnonzero(self._z > 0).astype(np.intp))


@dataclass(frozen=True)
class RetainedCollinearityProblem:
    """Detached snapshot of ``CollinearityProblem`` with result-local keys."""

    key: ProblemKey
    _design: FloatArray
    names: tuple[str, ...]
    _positions: IndexArray
    _grouping: RetainedGroupingSnapshot | None
    partition_key: PartitionKey | None

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_design", owned_array(self._design))
        object.__setattr__(self, "_positions", owned_array(self._positions))

    @classmethod
    def from_problem(
        cls,
        problem: CollinearityProblem,
        *,
        register_problem: CollinearityProblemCallback | None = None,
        detach_group: GroupingCallback | None = None,
    ) -> RetainedCollinearityProblem:
        """Detach a ``CollinearityProblem`` by its own concrete object identity."""
        if type(problem) is not CollinearityProblem:
            raise TypeError("problem must be a concrete CollinearityProblem")

        def _build(key: ProblemKey) -> RetainedCollinearityProblem:
            grp, pkey = _detach_problem_grouping(
                problem.grouping,
                detach_group=detach_group,
            )
            nms = tuple(str(n) for n in problem.names)
            return cls(key, problem.design, nms, problem.positions, grp, pkey)

        if register_problem is not None:
            return register_problem(problem, _build)
        return _build(ProblemKey(0))

    @property
    def design(self) -> FloatArray:
        """Return a fresh owned copy of ``design``."""
        return owned_array(self._design)

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of ``positions``."""
        return owned_array(self._positions)

    @property
    def grouping(self) -> RetainedGroupingSnapshot | None:
        """Return a fresh rematerialized grouping snapshot, or ``None``."""
        return None if self._grouping is None else self._grouping.rematerialize()

    @property
    def n_rows(self) -> int:
        """Return the number of rows."""
        return int(self._design.shape[0])

    @property
    def n_columns(self) -> int:
        """Return the number of design columns."""
        return int(self._design.shape[1])
