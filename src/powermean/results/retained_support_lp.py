"""Detached snapshots of joint separation LP solves, dual checks, and LP attempts.

This module defines :class:`RetainedLPSolve`, :class:`RetainedLPDual`,
:class:`RetainedLPAttempt`, and :func:`_detach_lp_outcome`.  Each witness inside an LP
attempt detaches its own ``problem`` without parent substitution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_diagnostic_problems import (
    MarginalTuple,
    RetainedSupportProblem,
    _owned_marginals,
)
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.results.retained_support_witnesses import (
    RetainedDualCheck,
    RetainedPrimalWitness,
)
from powermean.solve.support_lp import LPControls, LPOutcome

__all__ = [
    "RetainedLPAttempt",
    "RetainedLPDual",
    "RetainedLPSolve",
    "_detach_lp_outcome",
]


@dataclass(frozen=True)
class RetainedLPSolve:
    """Detached snapshot of ``LPSolve``."""

    _remaining: IndexArray
    status: int
    success: bool
    message: str
    iterations: int
    reported: float | None
    warnings: tuple[str, ...]
    failure: str
    _coefficients: FloatArray | None = None
    _marginals: MarginalTuple | None = None
    objective: float | None = None
    primal: RetainedPrimalWitness | None = None

    @property
    def remaining(self) -> IndexArray:
        """Return a fresh owned copy of ``remaining``."""
        return owned_array(self._remaining)

    @property
    def coefficients(self) -> FloatArray | None:
        """Return a fresh owned copy of ``coefficients``, or ``None``."""
        return _optional(self._coefficients)

    @property
    def marginals(self) -> MarginalTuple | None:
        """Return fresh owned copies of ``marginals``, or ``None``."""
        return _owned_marginals(self._marginals)


@dataclass(frozen=True)
class RetainedLPDual:
    """Detached snapshot of ``LPDual``."""

    _mapped: FloatArray
    _box: FloatArray
    _stationarity: FloatArray
    _box_ratios: FloatArray
    witness: RetainedDualCheck | None
    tolerance: float

    @property
    def mapped(self) -> FloatArray:
        """Return a fresh owned copy of ``mapped``."""
        return owned_array(self._mapped)

    @property
    def box(self) -> FloatArray:
        """Return a fresh owned copy of ``box``."""
        return owned_array(self._box)

    @property
    def stationarity(self) -> FloatArray:
        """Return a fresh owned copy of ``stationarity``."""
        return owned_array(self._stationarity)

    @property
    def box_ratios(self) -> FloatArray:
        """Return a fresh owned copy of ``box_ratios``."""
        return owned_array(self._box_ratios)

    @property
    def passed(self) -> bool:
        """Return whether the box contributions and the dual witness both pass."""
        box = bool((self._box_ratios <= self.tolerance).all())
        return box and self.witness is not None and self.witness.passed


@dataclass(frozen=True)
class RetainedLPAttempt:
    """Detached snapshot of ``LPOutcome``."""

    controls: LPControls
    method: str
    _options: dict[str, float | int | bool]
    _scales: FloatArray
    solves: tuple[RetainedLPSolve, ...]
    witnesses: tuple[RetainedPrimalWitness, ...]
    dual: RetainedLPDual | None
    completion: str
    reason: str

    @property
    def options(self) -> Mapping[str, float | int | bool]:
        """Return a fresh read-only mapping of the HiGHS options."""
        return MappingProxyType(dict(self._options))

    @property
    def scales(self) -> FloatArray:
        """Return a fresh owned copy of ``scales``."""
        return owned_array(self._scales)


def _detach_lp_outcome(
    lp: LPOutcome,
    prob: RetainedSupportProblem | None = None,
    *,
    problem_map: ProblemIdentityMap | None = None,
    partition_map: PartitionIdentityMap | None = None,
) -> RetainedLPAttempt:
    """Detach an ``LPOutcome`` into :class:`RetainedLPAttempt`."""
    _ = prob
    pw = RetainedPrimalWitness.from_witness
    solves = tuple(
        RetainedLPSolve(
            owned_array(s.remaining),
            int(s.status),
            bool(s.success),
            str(s.message),
            int(s.iterations),
            None if s.reported is None else float(s.reported),
            tuple(str(w) for w in s.warnings),
            str(s.failure),
            _optional(s.coefficients),
            _owned_marginals(s.marginals),
            None if s.objective is None else float(s.objective),
            None
            if s.primal is None
            else pw(
                s.primal,
                problem_map=problem_map,
                partition_map=partition_map,
            ),
        )
        for s in lp.solves
    )
    wits = tuple(
        pw(w, problem_map=problem_map, partition_map=partition_map)
        for w in lp.witnesses
    )
    d = lp.dual
    dual = (
        None
        if d is None
        else RetainedLPDual(
            owned_array(d.mapped),
            owned_array(d.box),
            owned_array(d.stationarity),
            owned_array(d.box_ratios),
            None
            if d.witness is None
            else RetainedDualCheck.from_witness(
                d.witness,
                problem_map=problem_map,
                partition_map=partition_map,
            ),
            float(d.tolerance),
        )
    )
    return RetainedLPAttempt(
        lp.controls,
        str(lp.method),
        dict(lp.options),
        owned_array(lp.scales),
        solves,
        wits,
        dual,
        str(lp.completion),
        str(lp.reason),
    )
