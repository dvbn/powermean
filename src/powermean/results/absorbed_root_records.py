"""Detached snapshots of the stages, directions and merit of an absorbed root solve.

The start and the continuation path are recorded.  The leaf records of the retained
selected-root route (:mod:`powermean.solve.absorbed_root`): every array is an owned copy
made at ingress and handed out afresh; the solver's immutable scalar records (a
direction record, a stage summary, the counted work) are kept with their mappings
frozen.  Nothing here reads a sample, a history or a certificate: those snapshots are
composed by :mod:`powermean.results.absorbed_root`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.absorbed_geometry_records import (
    RetainedSparseMatrix,
    _owned_factor_record,
    snapshot_sparse_matrix,
)
from powermean.results.owned import owned_array
from powermean.solve.absorbed_certificate_factor import FactorRecord
from powermean.solve.absorbed_root_records import (
    AbsorbedRootStage,
    MeritFactor,
    RootDirectionRecord,
    RootStageSummary,
)
from powermean.solve.cancelled_records import MeritMetric, RootAttemptRecord
from powermean.solve.newton_trace import NativeReason

__all__ = [
    "RetainedMeritSnapshot",
    "RetainedRootAttempt",
    "RetainedRootStage",
    "snapshot_directions",
    "snapshot_merit",
    "snapshot_root_stage",
]


@dataclass(frozen=True)
class RetainedRootAttempt:
    """Owned snapshot of one root attempt of the shared kernel."""

    _opening: FloatArray
    _point: FloatArray
    native: NativeReason
    iterations: int
    evaluations: int
    backtracks: int
    initial_merit: float
    final_merit: float
    message: str

    def __post_init__(self) -> None:
        """Own the points."""
        object.__setattr__(self, "_opening", owned_array(self._opening))
        object.__setattr__(self, "_point", owned_array(self._point))

    @property
    def opening(self) -> FloatArray:
        """Return a fresh owned copy of the opening point ``[theta, alpha]``."""
        return owned_array(self._opening)

    @property
    def point(self) -> FloatArray:
        """Return a fresh owned copy of the last accepted point ``[theta, alpha]``."""
        return owned_array(self._point)


def _owned_summary(summary: RootStageSummary) -> RootStageSummary:
    """A stage summary with its error mapping frozen anew."""
    return replace(summary, errors=MappingProxyType(dict(summary.errors)))


@dataclass(frozen=True)
class RetainedRootStage:
    """Owned snapshot of one variance stage: attempts, retained one, its gates."""

    variance: PowerVariance
    attempts: tuple[RetainedRootAttempt, ...]
    retained: int
    summary: RootStageSummary
    status: str
    iterations: int
    evaluations: int

    @property
    def native(self) -> NativeReason:
        """The native reason of the retained attempt."""
        return self.attempts[self.retained].native


def _snapshot_attempt(attempt: RootAttemptRecord) -> RetainedRootAttempt:
    """Detach one kernel attempt record."""
    return RetainedRootAttempt(
        owned_array(attempt.opening),
        owned_array(attempt.point),
        attempt.native,
        int(attempt.iterations),
        int(attempt.evaluations),
        int(attempt.backtracks),
        float(attempt.initial_merit),
        float(attempt.final_merit),
        str(attempt.message),
    )


def snapshot_root_stage(stage: AbsorbedRootStage) -> RetainedRootStage:
    """Detach one stage record of an absorbed root solve."""
    return RetainedRootStage(
        stage.variance,
        tuple(_snapshot_attempt(a) for a in stage.attempts),
        int(stage.retained),
        _owned_summary(stage.summary),
        str(stage.status),
        int(stage.iterations),
        int(stage.evaluations),
    )


def snapshot_directions(
    records: tuple[RootDirectionRecord, ...],
) -> tuple[RootDirectionRecord, ...]:
    """The direction records with their solve counts frozen anew."""
    return tuple(replace(r, actions=MappingProxyType(dict(r.actions))) for r in records)


@dataclass(frozen=True)
class RetainedMeritSnapshot:
    """Owned snapshot of a fixed merit metric: ``M``, its scale, factor record.

    The unit lower factor's values are not retained (as for the certificate's
    factors): the record keeps its permutations, nonzeros, pivots and provenance.
    """

    metric: MeritMetric
    matrix: RetainedSparseMatrix
    symmetric: RetainedSparseMatrix
    _scale: FloatArray
    _record: FactorRecord
    _diagonal: FloatArray
    _perm_r: IndexArray

    def __post_init__(self) -> None:
        """Own the arrays and the factor record."""
        object.__setattr__(self, "_scale", owned_array(self._scale))
        object.__setattr__(self, "_record", _owned_factor_record(self._record))
        object.__setattr__(self, "_diagonal", owned_array(self._diagonal))
        object.__setattr__(self, "_perm_r", owned_array(self._perm_r))

    @property
    def scale(self) -> FloatArray:
        """Return a fresh owned copy of the equilibration ``s`` of ``M``."""
        return owned_array(self._scale)

    @property
    def record(self) -> FactorRecord:
        """Return a fresh owned copy of the factor record (kind ``"merit"``)."""
        return _owned_factor_record(self._record)

    @property
    def diagonal(self) -> FloatArray:
        """Return a fresh owned copy of the positive pivots ``d``."""
        return owned_array(self._diagonal)

    @property
    def perm_r(self) -> IndexArray:
        """Return a fresh owned copy of the symmetric permutation."""
        return owned_array(self._perm_r)


def snapshot_merit(merit: MeritFactor | None) -> RetainedMeritSnapshot | None:
    """Detach a fixed merit metric, or ``None``."""
    if merit is None:
        return None
    matrix = snapshot_sparse_matrix(merit.matrix)
    symmetric = snapshot_sparse_matrix(merit.symmetric)
    assert matrix is not None
    assert symmetric is not None
    return RetainedMeritSnapshot(
        merit.metric,
        matrix,
        symmetric,
        owned_array(merit.scale),
        merit.record,
        owned_array(merit.diagonal),
        owned_array(merit.perm_r),
    )


def frozen_counts(counts: Mapping[str, int]) -> Mapping[str, int]:
    """A frozen copy of a count mapping."""
    return MappingProxyType({str(k): int(v) for k, v in counts.items()})
