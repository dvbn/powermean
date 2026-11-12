"""Detached snapshots of GPML recession LP reports, solves, and results.

This module detaches ``LPReport``, ``Solve``, and ``GammaResult`` into explicit finite
typed records without retaining live ``Grouping``, live ``WarningMessage`` objects, or
invoking diagnostic replay constructors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.model.variance import FloatArray
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.owned import _optional
from powermean.results.retained_diagnostic_problems import (
    MarginalTuple,
    RetainedGammaProblem,
    _owned_marginals,
    _owned_observation_weights,
)
from powermean.results.retained_gamma_witnesses import (
    RetainedGammaDual,
    RetainedGammaIdentities,
    RetainedGammaMultipliers,
    RetainedGammaWitness,
    RetainedRawCheck,
    RetainedRawValues,
    _detach_gamma_dual,
    _detach_gamma_witness,
    _detach_raw_check,
    _detach_raw_values,
)
from powermean.results.retained_input_values import ProblemKey
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.solve.support_gamma import STATUS, GammaResult, Solve, Status
from powermean.solve.support_gamma_lp import LPReport, Stage
from powermean.solve.support_lp import LPControls


@dataclass(frozen=True)
class RetainedGammaLPReport:
    """Detached snapshot of ``LPReport`` without live ``WarningMessage`` objects."""

    stage: Stage
    controls: LPControls
    rows: int
    nonzeros: int
    status: int | None = None
    success: bool | None = None
    message: str | None = None
    iterations: int | None = None
    reported: float | None = None
    warnings: tuple[str, ...] = ()
    raised: str | None = None
    failure: str = "lp_failure"
    _point: FloatArray | None = None
    _marginals: MarginalTuple | None = None

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_point", _optional(self._point))
        object.__setattr__(self, "_marginals", _owned_marginals(self._marginals))

    @classmethod
    def from_report(cls, r: LPReport) -> RetainedGammaLPReport:
        """Detach an ``LPReport`` record."""
        return cls(
            r.stage,
            r.controls,
            int(r.rows),
            int(r.nonzeros),
            None if r.status is None else int(r.status),
            None if r.success is None else bool(r.success),
            None if r.message is None else str(r.message),
            None if r.iterations is None else int(r.iterations),
            None if r.reported is None else float(r.reported),
            tuple(str(w) for w in r.warnings),
            None if r.raised is None else str(r.raised),
            str(r.failure),
            r.point,
            r.marginals,
        )

    @property
    def point(self) -> FloatArray | None:
        """Return a fresh owned copy of ``point``, or ``None``."""
        return _optional(self._point)

    @property
    def marginals(self) -> MarginalTuple | None:
        """Return fresh owned copies of ``marginals``, or ``None``."""
        return _owned_marginals(self._marginals)

    @property
    def exception(self) -> str | None:
        """Compatibility alias for ``raised``."""
        return self.raised


def _detach_gamma_report(r: LPReport) -> RetainedGammaLPReport:
    """Detach an ``LPReport`` into :class:`RetainedGammaLPReport`."""
    return RetainedGammaLPReport.from_report(r)


@dataclass(frozen=True)
class RetainedGammaSolve:
    """Detached snapshot of one ``Solve`` in a GPML recession diagnostic."""

    report: RetainedGammaLPReport
    witness: RetainedGammaWitness | None = None
    dual: RetainedGammaDual | None = None

    @classmethod
    def from_solve(
        cls,
        s: Solve,
        *,
        problem_snapshot: RetainedGammaProblem | None = None,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedGammaSolve:
        """Detach a ``Solve`` record without parent problem substitution."""
        wit = _detach_gamma_witness(
            s.witness,
            problem_snapshot=problem_snapshot,
            problem_map=problem_map,
            partition_map=partition_map,
        )
        du = _detach_gamma_dual(
            s.dual,
            problem_snapshot=problem_snapshot,
            problem_map=problem_map,
            partition_map=partition_map,
        )
        return cls(_detach_gamma_report(s.report), wit, du)

    @property
    def continues(self) -> bool:
        """Return whether LP1's evidence is a passing dual without a strict witness."""
        return (
            self.witness is not None
            and not self.witness.valid
            and self.dual is not None
            and self.dual.passed
        )


def _detach_gamma_solve(
    s: Solve,
    *,
    problem_snapshot: RetainedGammaProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedGammaSolve:
    """Detach a ``Solve`` into :class:`RetainedGammaSolve`."""
    return RetainedGammaSolve.from_solve(
        s,
        problem_snapshot=problem_snapshot,
        problem_map=problem_map,
        partition_map=partition_map,
    )


@dataclass(frozen=True)
class RetainedGammaResult:
    """Detached snapshot of ``GammaResult`` without invoking replay constructors."""

    problem_key: ProblemKey
    problem: RetainedGammaProblem
    controls: LPControls
    solves: tuple[RetainedGammaSolve, ...]
    reason: str

    @property
    def status(self) -> Status:
        """Return the kind, ``no_recession``, or ``inconclusive`` from ``reason``."""
        return STATUS.get(self.reason, "inconclusive")

    @property
    def witness(self) -> RetainedGammaWitness | None:
        """Return the valid primal witness establishing recession, if any."""
        last = self.solves[-1].witness if self.solves else None
        return last if last is not None and last.valid else None

    @property
    def attempts(self) -> int:
        """Return the charged calls: one per program run."""
        return len(self.solves)

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        """Return ``(n_rows, positive_rows, n_parameters, constraint_nonzeros)``."""
        prob = self.problem
        pos = prob.positive
        per_row = np.count_nonzero(prob._design, axis=1)
        nonzeros = int(per_row[pos].sum()) + int(pos.size) * prob.n_blocks
        return (prob.n_rows, int(pos.size), prob.n_parameters, nonzeros)


def snapshot_gamma_result(
    result: GammaResult,
    *,
    problem_snapshot: RetainedGammaProblem | None = None,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedGammaResult:
    """Detach an admitted ``GammaResult`` into a :class:`RetainedGammaResult`."""
    if type(result) is not GammaResult:
        raise TypeError("result must be a concrete GammaResult")
    if problem_map is None and problem_snapshot is None:
        problem_map = ProblemIdentityMap()
    if partition_map is None and problem_snapshot is None:
        partition_map = PartitionIdentityMap()
    prob = (
        RetainedGammaProblem.from_problem(
            result.problem,
            register_problem=(
                None if problem_map is None else problem_map.register_gamma
            ),
            detach_group=(
                None if partition_map is None else partition_map.register_grouping
            ),
        )
        if problem_map is not None or problem_snapshot is None
        else problem_snapshot
    )
    solves = tuple(
        _detach_gamma_solve(
            s,
            problem_snapshot=prob,
            problem_map=problem_map,
            partition_map=partition_map,
        )
        for s in result.solves
    )
    return RetainedGammaResult(
        prob.key, prob, result.controls, solves, str(result.reason)
    )


__all__ = [
    "RetainedGammaDual",
    "RetainedGammaIdentities",
    "RetainedGammaLPReport",
    "RetainedGammaMultipliers",
    "RetainedGammaProblem",
    "RetainedGammaResult",
    "RetainedGammaSolve",
    "RetainedGammaWitness",
    "RetainedRawCheck",
    "RetainedRawValues",
    "_detach_gamma_dual",
    "_detach_gamma_report",
    "_detach_gamma_solve",
    "_detach_gamma_witness",
    "_detach_raw_check",
    "_detach_raw_values",
    "_owned_observation_weights",
    "snapshot_gamma_result",
]
