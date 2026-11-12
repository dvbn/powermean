"""Detached snapshots of iterative rectifier projections, summaries, and attempts.

This module defines :class:`RetainedProjectionRefinement`,
:class:`RetainedProjectionSolve`, :class:`RetainedRectifierSummary`,
:class:`RetainedRectifierFinal`, :class:`RetainedRectifierAttempt`,
:func:`_detach_projection`, and :func:`_detach_rectifier_outcome`.  Each witness and
absorption state inside a rectifier outcome detaches its own problem/grouping; the
refinement call's state, like the joint call's, through the projection's grouping
snapshot or map.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
)
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_state_records import (
    RetainedAbsorptionResult,
    _owned_absorption_result,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_diagnostic_problems import RetainedSupportProblem
from powermean.results.retained_input_values import _tuple_arrays
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.results.retained_support_witnesses import (
    RetainedDualCheck,
    RetainedPrimalWitness,
)
from powermean.solve.support_projection import (
    Projection,
    Refinement,
)
from powermean.solve.support_projection import (
    Status as ProjectionStatus,
)
from powermean.solve.support_rectifier import (
    Final as RectifierFinal,
)
from powermean.solve.support_rectifier import (
    RectifierControls,
    RectifierOutcome,
)

__all__ = [
    "RetainedProjectionRefinement",
    "RetainedProjectionSolve",
    "RetainedRectifierAttempt",
    "RetainedRectifierFinal",
    "RetainedRectifierSummary",
    "_detach_projection",
    "_detach_rectifier_outcome",
]


@dataclass(frozen=True)
class RetainedProjectionRefinement:
    """Detached snapshot of ``Refinement`` in ``support_projection``."""

    reason: str
    adopted: bool
    budget: int
    evaluations: int
    result: RetainedAbsorptionResult | None = None


@dataclass(frozen=True)
class RetainedProjectionSolve:
    """Detached snapshot of ``Projection``."""

    status: ProjectionStatus
    reason: str
    _scales: FloatArray
    absorption: RetainedAbsorptionResult | None = None
    _singular_values: FloatArray | None = None
    cutoff: float | None = None
    rank: int | None = None
    _pivots: IndexArray | None = None
    range_residual: float | None = None
    _slopes: FloatArray | None = None
    _effects: tuple[FloatArray, ...] | None = None
    _fitted: FloatArray | None = None
    _residual: FloatArray | None = None
    _level_means: tuple[FloatArray, ...] = ()
    _slope_ratios: FloatArray | None = None
    refinement: RetainedProjectionRefinement | None = None

    @property
    def scales(self) -> FloatArray:
        """Return a fresh owned copy of ``scales``."""
        return owned_array(self._scales)

    @property
    def singular_values(self) -> FloatArray | None:
        """Return a fresh owned copy of ``singular_values``, or ``None``."""
        return _optional(self._singular_values)

    @property
    def pivots(self) -> IndexArray | None:
        """Return a fresh owned copy of ``pivots``, or ``None``."""
        return _optional(self._pivots)

    @property
    def slopes(self) -> FloatArray | None:
        """Return a fresh owned copy of ``slopes``, or ``None``."""
        return _optional(self._slopes)

    @property
    def effects(self) -> tuple[FloatArray, ...] | None:
        """Return fresh owned copies of ``effects``, or ``None``."""
        return None if self._effects is None else _tuple_arrays(self._effects)

    @property
    def fitted(self) -> FloatArray | None:
        """Return a fresh owned copy of ``fitted``, or ``None``."""
        return _optional(self._fitted)

    @property
    def residual(self) -> FloatArray | None:
        """Return a fresh owned copy of ``residual``, or ``None``."""
        return _optional(self._residual)

    @property
    def level_means(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of ``level_means``."""
        return _tuple_arrays(self._level_means)

    @property
    def slope_ratios(self) -> FloatArray | None:
        """Return a fresh owned copy of ``slope_ratios``, or ``None``."""
        return _optional(self._slope_ratios)

    @property
    def solved(self) -> bool:
        """Return True when ``status == 'solved'``."""
        return self.status == "solved"

    @property
    def basic(self) -> IndexArray | None:
        """Return a fresh owned copy of the basic slope coordinates, or ``None``."""
        if self._pivots is None or self.rank is None:
            return None
        return owned_array(self._pivots[: self.rank])


@dataclass(frozen=True)
class RetainedRectifierSummary:
    """Detached snapshot of ``Attempt`` in ``support_rectifier``."""

    index: int
    status: str
    reason: str
    rank: int | None
    sweeps: int
    visits: int
    operators: int
    passes: int
    primal_valid: bool | None = None
    strict: int | None = None
    dual_passed: bool | None = None
    margin: float | None = None
    refinement: str = ""
    refinement_sweeps: int = 0


@dataclass(frozen=True)
class RetainedRectifierFinal:
    """Detached snapshot of ``Final`` in ``support_rectifier``."""

    _u: FloatArray
    projection: RetainedProjectionSolve
    _z: FloatArray | None = None
    _residual: FloatArray | None = None
    _weighted: FloatArray | None = None
    primal: RetainedPrimalWitness | None = None
    dual: RetainedDualCheck | None = None

    @property
    def u(self) -> FloatArray:
        """Return a fresh owned copy of ``u``."""
        return owned_array(self._u)

    @property
    def z(self) -> FloatArray | None:
        """Return a fresh owned copy of ``z``, or ``None``."""
        return _optional(self._z)

    @property
    def residual(self) -> FloatArray | None:
        """Return a fresh owned copy of ``residual``, or ``None``."""
        return _optional(self._residual)

    @property
    def weighted(self) -> FloatArray | None:
        """Return a fresh owned copy of ``weighted``, or ``None``."""
        return _optional(self._weighted)


@dataclass(frozen=True)
class RetainedRectifierAttempt:
    """Detached snapshot of ``RectifierOutcome``."""

    controls: RectifierControls
    penalty: int
    _weights: FloatArray | None
    charged: int
    attempts: tuple[RetainedRectifierSummary, ...]
    last: RetainedRectifierFinal | None
    witnesses: tuple[RetainedPrimalWitness, ...]
    dual: RetainedDualCheck | None
    completion: str
    reason: str

    @property
    def weights(self) -> FloatArray | None:
        """Return a fresh owned copy of the auxiliary weights, or ``None``."""
        return _optional(self._weights)


def _detach_refinement(
    r: Refinement | None,
    grp: RetainedGroupingSnapshot | None = None,
    *,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedProjectionRefinement | None:
    """Detach an optional ``Refinement``; its call's result as the joint call's."""
    if r is None:
        return None
    return RetainedProjectionRefinement(
        str(r.reason),
        bool(r.adopted),
        int(r.budget),
        int(r.evaluations),
        _owned_absorption_result(
            r.result,
            grouping_snapshot=grp,
            grouping_map=grouping_map,
        ),
    )


def _detach_projection(
    p: Projection,
    grp: RetainedGroupingSnapshot | None = None,
    *,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedProjectionSolve:
    """Detach a ``Projection`` into :class:`RetainedProjectionSolve`."""
    return RetainedProjectionSolve(
        p.status,
        str(p.reason),
        owned_array(p.scales),
        _owned_absorption_result(
            p.absorption,
            grouping_snapshot=grp,
            grouping_map=grouping_map,
        ),
        _optional(p.singular_values),
        None if p.cutoff is None else float(p.cutoff),
        None if p.rank is None else int(p.rank),
        _optional(p.pivots),
        None if p.range_residual is None else float(p.range_residual),
        _optional(p.slopes),
        None if p.effects is None else _tuple_arrays(p.effects),
        _optional(p.fitted),
        _optional(p.residual),
        _tuple_arrays(p.level_means),
        _optional(p.slope_ratios),
        _detach_refinement(p.refinement, grp, grouping_map=grouping_map),
    )


def _detach_rectifier_final(
    f: RectifierFinal | None,
    grp: RetainedGroupingSnapshot | None,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedRectifierFinal | None:
    """Detach an optional ``Final`` in ``support_rectifier``."""
    if f is None:
        return None
    gmap = None if partition_map is None else partition_map.grouping_map
    return RetainedRectifierFinal(
        owned_array(f.u),
        _detach_projection(f.projection, grp, grouping_map=gmap),
        _optional(f.z),
        _optional(f.residual),
        _optional(f.weighted),
        None
        if f.primal is None
        else RetainedPrimalWitness.from_witness(
            f.primal, problem_map=problem_map, partition_map=partition_map
        ),
        None
        if f.dual is None
        else RetainedDualCheck.from_witness(
            f.dual, problem_map=problem_map, partition_map=partition_map
        ),
    )


def _detach_rectifier_outcome(
    ir: RectifierOutcome,
    prob: RetainedSupportProblem | None = None,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedRectifierAttempt:
    """Detach a ``RectifierOutcome`` into :class:`RetainedRectifierAttempt`."""
    pw = RetainedPrimalWitness.from_witness
    dw = RetainedDualCheck.from_witness
    atts = tuple(
        RetainedRectifierSummary(
            int(a.index),
            str(a.status),
            str(a.reason),
            None if a.rank is None else int(a.rank),
            int(a.sweeps),
            int(a.visits),
            int(a.operators),
            int(a.passes),
            None if a.primal_valid is None else bool(a.primal_valid),
            None if a.strict is None else int(a.strict),
            None if a.dual_passed is None else bool(a.dual_passed),
            None if a.margin is None else float(a.margin),
            str(a.refinement),
            int(a.refinement_sweeps),
        )
        for a in ir.attempts
    )
    grp = None if prob is None else prob._grouping
    last = _detach_rectifier_final(
        ir.last,
        grp,
        problem_map=problem_map,
        partition_map=partition_map,
    )
    wits = tuple(
        pw(w, problem_map=problem_map, partition_map=partition_map)
        for w in ir.witnesses
    )
    du = (
        None
        if ir.dual is None
        else dw(ir.dual, problem_map=problem_map, partition_map=partition_map)
    )
    return RetainedRectifierAttempt(
        ir.controls,
        int(ir.penalty),
        _optional(ir.weights),
        int(ir.charged),
        atts,
        last,
        wits,
        du,
        str(ir.completion),
        str(ir.reason),
    )
