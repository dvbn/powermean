"""Detached snapshots of collinearity column diagnostics and analyses.

This module detaches ``ColumnDiagnostics`` and ``CollinearityAnalysis`` into explicit
finite typed records without retaining live ``Grouping`` or invoking diagnostic replay
constructors.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.data.collinearity_records import (
    CollinearityAnalysis,
    ColumnDiagnostics,
    DiagnosticControls,
    Targets,
)
from powermean.fe.absorb_numeric import FreshCheck
from powermean.model.order import IndexArray
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_state_records import (
    RetainedAbsorptionResult,
    _owned_absorption_result,
    _owned_fresh_check,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_collinearity_geometry import (
    RetainedFactorAttempt,
    RetainedRelationCheck,
    RetainedSpanTest,
    RetainedWithinColumns,
    _detach_factorization,
    _detach_relation,
    _detach_span_test,
    _detach_within,
)
from powermean.results.retained_diagnostic_problems import (
    RetainedCollinearityProblem,
)
from powermean.results.retained_input_values import ProblemKey
from powermean.results.retained_inputs import ProblemIdentityMap


@dataclass(frozen=True)
class RetainedColumnDiagnostics:
    """Detached snapshot of ``ColumnDiagnostics``."""

    index: int
    name: str
    kind: str
    scale: float | None = None
    original_norm: float | None = None
    within_norm: float | None = None
    screen: bool | None = None
    span: RetainedSpanTest | None = None
    relation: RetainedRelationCheck | None = None

    @classmethod
    def from_column(cls, c: ColumnDiagnostics) -> RetainedColumnDiagnostics:
        """Detach a ``ColumnDiagnostics`` record."""
        if type(c) is not ColumnDiagnostics:
            raise TypeError("c must be a concrete ColumnDiagnostics")
        return cls(
            index=int(c.index),
            name=str(c.name),
            kind=str(c.kind),
            scale=None if c.scale is None else float(c.scale),
            original_norm=(None if c.original_norm is None else float(c.original_norm)),
            within_norm=None if c.within_norm is None else float(c.within_norm),
            screen=None if c.screen is None else bool(c.screen),
            span=_detach_span_test(c.span),
            relation=_detach_relation(c.relation),
        )


def _detach_column_diag(c: ColumnDiagnostics) -> RetainedColumnDiagnostics:
    """Detach a ``ColumnDiagnostics`` into :class:`RetainedColumnDiagnostics`."""
    return RetainedColumnDiagnostics.from_column(c)


@dataclass(frozen=True)
class RetainedCollinearityAnalysis:
    """Detached snapshot of ``CollinearityAnalysis``."""

    problem_key: ProblemKey | None
    problem: RetainedCollinearityProblem | None
    controls: DiagnosticControls | None
    mode: str
    status: str
    cause: str
    message: str
    names: tuple[str, ...]
    _positions: IndexArray
    reasons: tuple[str | None, ...] | None
    _kept: IndexArray | None
    _omitted: IndexArray | None
    columns: tuple[RetainedColumnDiagnostics, ...]
    targets: Targets | None = None
    absorption: RetainedAbsorptionResult | None = None
    _fresh: FreshCheck | None = None
    within: RetainedWithinColumns | None = None
    factorizations: tuple[RetainedFactorAttempt, ...] = ()

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_positions", owned_array(self._positions))
        object.__setattr__(self, "_kept", _optional(self._kept))
        object.__setattr__(self, "_omitted", _optional(self._omitted))
        if self._fresh is not None:
            object.__setattr__(self, "_fresh", _owned_fresh_check(self._fresh))

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of ``positions``."""
        return owned_array(self._positions)

    @property
    def kept(self) -> IndexArray | None:
        """Return a fresh owned copy of ``kept``, or ``None``."""
        return _optional(self._kept)

    @property
    def omitted(self) -> IndexArray | None:
        """Return a fresh owned copy of ``omitted``, or ``None``."""
        return _optional(self._omitted)

    @property
    def fresh(self) -> FreshCheck | None:
        """Return a fresh owned ``FreshCheck``, or ``None``."""
        return None if self._fresh is None else _owned_fresh_check(self._fresh)


def snapshot_collinearity_analysis(
    analysis: CollinearityAnalysis,
    *,
    problem_snapshot: RetainedCollinearityProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedCollinearityAnalysis:
    """Detach ``CollinearityAnalysis`` into a ``RetainedCollinearityAnalysis``."""
    if type(analysis) is not CollinearityAnalysis:
        raise TypeError("analysis must be a concrete CollinearityAnalysis")
    if problem_map is None and problem_snapshot is None:
        problem_map = ProblemIdentityMap()
    if partition_map is None and problem_snapshot is None:
        partition_map = PartitionIdentityMap()
    if analysis.problem is None:
        prob: RetainedCollinearityProblem | None = None
    elif problem_map is not None or problem_snapshot is None:
        rp = None if problem_map is None else problem_map.register_collinearity
        dg = None if partition_map is None else partition_map.register_grouping
        prob = RetainedCollinearityProblem.from_problem(
            analysis.problem, register_problem=rp, detach_group=dg
        )
    else:
        prob = problem_snapshot
    pkey = None if prob is None else prob.key
    grp_snap = None if prob is None else prob._grouping
    gmap = None if partition_map is None else partition_map.grouping_map
    abs_res = (
        None
        if analysis.absorption is None
        else _owned_absorption_result(
            analysis.absorption,
            grouping_snapshot=grp_snap,
            grouping_map=gmap,
        )
    )
    fr = None if analysis.fresh is None else _owned_fresh_check(analysis.fresh)
    wth = _detach_within(analysis.within)
    cols = tuple(_detach_column_diag(c) for c in analysis.columns)
    facts = tuple(_detach_factorization(f) for f in analysis.factorizations)
    reasons = (
        None
        if analysis.reasons is None
        else tuple(None if r is None else str(r) for r in analysis.reasons)
    )
    return RetainedCollinearityAnalysis(
        problem_key=pkey,
        problem=prob,
        controls=analysis.controls,
        mode=str(analysis.mode),
        status=str(analysis.status),
        cause=str(analysis.cause),
        message=str(analysis.message),
        names=tuple(str(n) for n in analysis.names),
        _positions=owned_array(analysis.positions),
        reasons=reasons,
        _kept=_optional(analysis.kept),
        _omitted=_optional(analysis.omitted),
        columns=cols,
        targets=analysis.targets,
        absorption=abs_res,
        _fresh=fr,
        within=wth,
        factorizations=facts,
    )


__all__ = [
    "RetainedCollinearityAnalysis",
    "RetainedCollinearityProblem",
    "RetainedColumnDiagnostics",
    "RetainedFactorAttempt",
    "RetainedRelationCheck",
    "RetainedSpanTest",
    "RetainedWithinColumns",
    "_detach_column_diag",
    "_detach_factorization",
    "_detach_relation",
    "_detach_span_test",
    "_detach_within",
    "snapshot_collinearity_analysis",
]
