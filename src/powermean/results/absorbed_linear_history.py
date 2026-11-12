"""Detached snapshots of linear WLS decompositions and candidate states.

This module defines :class:`RetainedDecomposition`, :class:`RetainedLinearState`, and
:func:`_snapshot_linear_state`.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.data.weights import WeightKind
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
)
from powermean.results.absorbed_point_records import (
    RetainedFreshEquations,
    RetainedLinearNormalization,
    RetainedPointSnapshot,
)
from powermean.results.absorbed_points import (
    PointMapper,
    snapshot_fresh_equations,
    snapshot_linear_normalization,
)
from powermean.results.absorbed_sample import SampleIdentityMap
from powermean.results.absorbed_state_records import (
    RetainedAbsorptionResult,
    RetainedAbsorptionState,
    _owned_absorption_result,
)
from powermean.results.absorbed_states import RetainedPreparedWeights
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_input_values import PointKey, SampleKey
from powermean.solve.linear_absorbed_numeric import (
    Decomposition,
)
from powermean.solve.linear_absorbed_numeric import (
    Kind as DecompositionKind,
)
from powermean.solve.linear_absorbed_numeric import (
    Outcome as DecompositionOutcome,
)
from powermean.solve.linear_absorbed_point import LinearControls
from powermean.solve.linear_absorbed_state import (
    AbsorbedLinearCandidate,
    LinearWork,
)
from powermean.solve.linear_absorbed_state import (
    Stage as LinearStage,
)
from powermean.solve.linear_absorbed_state import (
    Status as LinearStatus,
)

__all__ = [
    "RetainedDecomposition",
    "RetainedLinearState",
    "_snapshot_linear_state",
]


@dataclass(frozen=True)
class RetainedDecomposition:
    """Detached snapshot of all 7 stored fields of ``Decomposition`` without replay."""

    kind: DecompositionKind
    shape: tuple[int, int]
    outcome: DecompositionOutcome
    _values: FloatArray
    _pivots: IndexArray | None
    cutoff: float | None
    rank: int | None

    @classmethod
    def from_decomposition(cls, d: Decomposition) -> RetainedDecomposition:
        """Capture all 7 stored fields of ``d`` without invoking ``Decomposition``."""
        return cls(
            d.kind,
            (int(d.shape[0]), int(d.shape[1])),
            d.outcome,
            owned_array(d.values),
            _optional(d.pivots),
            None if d.cutoff is None else float(d.cutoff),
            None if d.rank is None else int(d.rank),
        )

    @property
    def values(self) -> FloatArray:
        """Return a fresh owned copy of ``values``."""
        return owned_array(self._values)

    @property
    def pivots(self) -> IndexArray | None:
        """Return a fresh owned copy of ``pivots``, or ``None``."""
        return _optional(self._pivots)


@dataclass(frozen=True)
class RetainedLinearState:
    """Detached snapshot of ``AbsorbedLinearCandidate`` stages (complete or partial)."""

    sample_key: SampleKey
    controls: LinearControls
    stage: LinearStage
    status: LinearStatus
    reason: str
    prepared: RetainedPreparedWeights | None
    joint: RetainedAbsorptionResult | None
    decompositions: tuple[RetainedDecomposition, ...]
    seed: RetainedAbsorptionState | None
    final: RetainedAbsorptionResult | None
    point_key: PointKey | None
    point: RetainedPointSnapshot | None
    normalized: RetainedLinearNormalization | None
    equations: RetainedFreshEquations | None
    _t: FloatArray
    _theta: FloatArray | None
    n_slopes: int
    weight_kind: WeightKind | None
    work: LinearWork

    @property
    def t(self) -> FloatArray:
        """Return a fresh owned copy of the signed response ``t``."""
        return owned_array(self._t)

    @property
    def theta(self) -> FloatArray | None:
        """Return a fresh owned copy of ``theta``, or ``None``."""
        return _optional(self._theta)

    @property
    def complete(self) -> bool:
        """Return whether every stage completed."""
        return self.status == "complete"

    @property
    def weights(self) -> RetainedPreparedWeights | None:
        """Compatibility alias for ``prepared``."""
        return self.prepared

    @property
    def normalization(self) -> RetainedLinearNormalization | None:
        """Compatibility alias for ``normalized``."""
        return self.normalized


def _snapshot_linear_state(
    cand: AbsorbedLinearCandidate,
    pt: RetainedPointSnapshot | None,
    *,
    sample_key: SampleKey = 0,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    get_point: PointMapper | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedLinearState:
    """Detach all stages of an admitted ``AbsorbedLinearCandidate``."""
    p, s = cand.prepared, cand.seed
    pw = (
        None
        if p is None
        else RetainedPreparedWeights.from_prepared(
            p, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
        )
    )
    dc = tuple(RetainedDecomposition.from_decomposition(d) for d in cand.decompositions)
    sd = (
        None
        if s is None
        else RetainedAbsorptionState.from_state(
            s, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
        )
    )
    jnt = _owned_absorption_result(
        cand.joint, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
    )
    fin = _owned_absorption_result(
        cand.final, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
    )
    skey = sample_key if pt is None else pt.sample_key
    pkey = PointKey(0) if pt is None else pt.key
    nrm = snapshot_linear_normalization(
        cand.normalized,
        point_key=pkey,
        sample_key=skey,
        get_point=get_point,
        sample_map=sample_map,
    )
    eqs = snapshot_fresh_equations(
        cand.equations,
        point_key=pkey,
        sample_key=skey,
        get_point=get_point,
        sample_map=sample_map,
    )
    return RetainedLinearState(
        skey,
        cand.controls,
        cand.stage,
        cand.status,
        str(cand.reason),
        pw,
        jnt,
        dc,
        sd,
        fin,
        None if pt is None else pt.key,
        pt,
        nrm,
        eqs,
        owned_array(cand.t),
        _optional(cand.theta),
        int(cand.n_slopes),
        cand.weight_kind,
        cand.work,
    )
