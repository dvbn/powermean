"""Detached snapshots of joint separation support runs and results.

This module detaches ``FEStep``, ``SupportResult``, and lowered ``RawSupportResult``
fields into explicit finite typed records without retaining live ``Grouping`` or
invoking ``SupportResult`` replay constructors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from powermean.model.order import IndexArray
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.owned import owned_array
from powermean.results.retained_diagnostic_problems import RetainedSupportProblem
from powermean.results.retained_inputs import (
    ProblemIdentityMap,
    ProblemKey,
    _tuple_arrays,
)
from powermean.results.retained_rectifier import (
    RetainedProjectionSolve,
    RetainedRectifierAttempt,
    RetainedRectifierFinal,
    RetainedRectifierSummary,
    _detach_projection,
    _detach_rectifier_outcome,
)
from powermean.results.retained_support_lp import (
    RetainedLPAttempt,
    RetainedLPDual,
    RetainedLPSolve,
    _detach_lp_outcome,
)
from powermean.results.retained_support_witnesses import (
    RetainedColumnSums,
    RetainedDualCheck,
    RetainedPrimalWitness,
)
from powermean.solve.support import (
    FEStep,
    Method,
    Status,
    SupportControls,
    SupportResult,
)
from powermean.solve.support_lp import LPOutcome
from powermean.solve.support_problem import SupportProblem
from powermean.solve.support_rectifier import RectifierOutcome
from powermean.solve.support_witness import DualWitness, PrimalWitness


@dataclass(frozen=True)
class RetainedFEStep:
    """Detached snapshot of ``FEStep``."""

    _levels: tuple[IndexArray, ...]
    _rows: IndexArray
    witness: RetainedPrimalWitness | None
    exact: bool

    @property
    def levels(self) -> tuple[IndexArray, ...]:
        """Return fresh owned copies of per-block separated level codes."""
        return _tuple_arrays(self._levels)

    @property
    def rows(self) -> IndexArray:
        """Return a fresh owned copy of separated row indices."""
        return owned_array(self._rows)


@dataclass(frozen=True)
class SupportWork:
    """Charged work summary for a detached support diagnostic."""

    lp_solves: int
    lp_iterations: int
    wls_attempts: int
    absorption_sweeps: int


@dataclass(frozen=True)
class RawSupportFields:
    """Lowered concrete fields of an un-admitted ``RawSupportResult``."""

    problem: SupportProblem
    controls: SupportControls
    method: Method | str
    fe: FEStep | None
    lp: LPOutcome | None
    ir: RectifierOutcome | None
    witnesses: Sequence[PrimalWitness]
    dual: DualWitness | None
    completion: str
    reason: str


def _union_strict(witnesses: tuple[RetainedPrimalWitness, ...]) -> IndexArray:
    """Return a fresh owned sorted union of witness ``strict`` row indices."""
    parts = [w._strict for w in witnesses if w._strict.size]
    if not parts:
        return owned_array(np.zeros(0, dtype=np.intp))
    return owned_array(np.unique(np.concatenate(parts)).astype(np.intp))


@dataclass(frozen=True)
class RetainedSupportResult:
    """Detached snapshot of an admitted ``SupportResult``."""

    problem_key: ProblemKey
    problem: RetainedSupportProblem
    controls: SupportControls
    method: Method
    fe: RetainedFEStep | None
    lp: RetainedLPAttempt | None
    ir: RetainedRectifierAttempt | None
    witnesses: tuple[RetainedPrimalWitness, ...]
    dual: RetainedDualCheck | None
    completion: str
    reason: str
    work: SupportWork

    @property
    def rectifier(self) -> tuple[RetainedRectifierSummary, ...]:
        """Return the rectifier attempt summaries."""
        return () if self.ir is None else self.ir.attempts

    @property
    def indicator(self) -> RetainedPrimalWitness | None:
        """Return the FE indicator step witness, if any."""
        return None if self.fe is None else self.fe.witness

    @property
    def separated(self) -> IndexArray:
        """Return a fresh owned copy of the separated current-row indices ``S``."""
        return _union_strict(self.witnesses)

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of the original positions of ``S``."""
        return owned_array(self.problem._positions[self.separated])

    @property
    def status(self) -> Status:
        """Return ``complete``, ``partial``, or ``inconclusive`` from the evidence."""
        if self.completion:
            return "complete"
        return "partial" if self.separated.size else "inconclusive"


@dataclass(frozen=True)
class RetainedRawSupport:
    """Detached snapshot of an un-admitted ``RawSupportResult``."""

    problem_key: ProblemKey
    problem: RetainedSupportProblem
    controls: SupportControls
    method: str
    fe: RetainedFEStep | None
    lp: RetainedLPAttempt | None
    ir: RetainedRectifierAttempt | None
    witnesses: tuple[RetainedPrimalWitness, ...]
    dual: RetainedDualCheck | None
    completion: str
    reason: str
    work: SupportWork

    @property
    def rectifier(self) -> tuple[RetainedRectifierSummary, ...]:
        """Return the rectifier attempt summaries."""
        return () if self.ir is None else self.ir.attempts

    @property
    def indicator(self) -> RetainedPrimalWitness | None:
        """Return the FE indicator step witness, if any."""
        return None if self.fe is None else self.fe.witness


def _detach_support_parts(
    raw: SupportResult | RawSupportFields,
    *,
    problem_snapshot: RetainedSupportProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> tuple[
    RetainedSupportProblem,
    RetainedFEStep | None,
    RetainedLPAttempt | None,
    RetainedRectifierAttempt | None,
    tuple[RetainedPrimalWitness, ...],
    RetainedDualCheck | None,
    SupportWork,
]:
    """Detach the shared nested sub-graphs of a support result or raw support result."""
    if problem_map is None and problem_snapshot is None:
        problem_map = ProblemIdentityMap()
    if partition_map is None and problem_snapshot is None:
        partition_map = PartitionIdentityMap()
    prob = (
        RetainedSupportProblem.from_problem(
            raw.problem,
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
    pw = RetainedPrimalWitness.from_witness
    fe_snap = (
        None
        if raw.fe is None
        else RetainedFEStep(
            _tuple_arrays(raw.fe.levels),
            owned_array(raw.fe.rows),
            None
            if raw.fe.witness is None
            else pw(
                raw.fe.witness,
                problem_snapshot=(
                    prob if raw.fe.witness.problem is raw.problem else None
                ),
                problem_map=problem_map,
                partition_map=partition_map,
            ),
            bool(raw.fe.exact),
        )
    )
    lp_snap = (
        None
        if raw.lp is None
        else _detach_lp_outcome(
            raw.lp,
            prob,
            problem_map=problem_map,
            partition_map=partition_map,
        )
    )
    ir_snap = (
        None
        if raw.ir is None
        else _detach_rectifier_outcome(
            raw.ir,
            prob,
            problem_map=problem_map,
            partition_map=partition_map,
        )
    )
    wits = tuple(
        pw(
            w,
            problem_snapshot=(prob if w.problem is raw.problem else None),
            problem_map=problem_map,
            partition_map=partition_map,
        )
        for w in raw.witnesses
    )
    du_snap = (
        None
        if raw.dual is None
        else RetainedDualCheck.from_witness(
            raw.dual,
            problem_snapshot=(prob if raw.dual.problem is raw.problem else None),
            problem_map=problem_map,
            partition_map=partition_map,
        )
    )
    work = SupportWork(
        0 if lp_snap is None else len(lp_snap.solves),
        0 if lp_snap is None else sum(s.iterations for s in lp_snap.solves),
        0 if ir_snap is None else ir_snap.charged,
        0 if ir_snap is None else sum(a.sweeps for a in ir_snap.attempts),
    )
    return prob, fe_snap, lp_snap, ir_snap, wits, du_snap, work


def snapshot_support_result(
    result: SupportResult,
    *,
    problem_snapshot: RetainedSupportProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedSupportResult:
    """Detach an admitted ``SupportResult`` into a :class:`RetainedSupportResult`."""
    if type(result) is not SupportResult:
        raise TypeError("result must be a concrete SupportResult")
    prob, fe, lp, ir, wits, du, work = _detach_support_parts(
        result,
        problem_snapshot=problem_snapshot,
        problem_map=problem_map,
        partition_map=partition_map,
    )
    return RetainedSupportResult(
        prob.key,
        prob,
        result.controls,
        cast("Method", str(result.method)),
        fe,
        lp,
        ir,
        wits,
        du,
        str(result.completion),
        str(result.reason),
        work,
    )


def snapshot_raw_support(
    raw: RawSupportFields,
    *,
    problem_snapshot: RetainedSupportProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedRawSupport:
    """Detach lowered ``RawSupportResult`` fields without ``SupportResult`` replay."""
    if type(raw.controls) is not SupportControls:
        raise TypeError("controls must be SupportControls")
    prob, fe_snap, lp_snap, ir_snap, wits, du_snap, work = _detach_support_parts(
        raw,
        problem_snapshot=problem_snapshot,
        problem_map=problem_map,
        partition_map=partition_map,
    )
    return RetainedRawSupport(
        prob.key,
        prob,
        raw.controls,
        str(raw.method),
        fe_snap,
        lp_snap,
        ir_snap,
        wits,
        du_snap,
        str(raw.completion),
        str(raw.reason),
        work,
    )


__all__ = [
    "RawSupportFields",
    "RetainedColumnSums",
    "RetainedDualCheck",
    "RetainedFEStep",
    "RetainedLPAttempt",
    "RetainedLPDual",
    "RetainedLPSolve",
    "RetainedPrimalWitness",
    "RetainedProjectionSolve",
    "RetainedRawSupport",
    "RetainedRectifierAttempt",
    "RetainedRectifierFinal",
    "RetainedRectifierSummary",
    "RetainedSupportProblem",
    "RetainedSupportResult",
    "SupportWork",
    "_detach_lp_outcome",
    "_detach_projection",
    "_detach_rectifier_outcome",
    "snapshot_raw_support",
    "snapshot_support_result",
]
