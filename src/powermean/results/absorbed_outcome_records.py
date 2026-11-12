"""Detached snapshots of sample coordinator requests, invocations, rounds, and outcomes.

This module defines :data:`RetainedClosureSnapshot`, :func:`_owned_closure`,
:class:`RetainedSampleRequest`, :func:`build_sample_request_snapshot`,
:class:`RetainedSupportInvocation`, :class:`RetainedRecessionInvocation`,
:class:`RetainedCollinearityInvocation`, :class:`RetainedRound`, and
:class:`RetainedOutcomeSnapshot`.  Each invocation detaches ``inv.problem`` and
``inv.result.problem`` / ``inv.analysis.problem`` from their own concrete originals.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from powermean._prepare.compose import PreparedFrame
from powermean.data.collinearity import (
    CollinearityAnalysis,
    CollinearityProblem,
    DiagnosticControls,
)
from powermean.data.sample_fe import Closure, ClosurePass, LevelSeparation, Visit
from powermean.data.sample_fe_state import RowLedger
from powermean.model.order import IndexArray
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.owned import owned_array
from powermean.results.retained_collinearity import (
    RetainedCollinearityAnalysis,
    snapshot_collinearity_analysis,
)
from powermean.results.retained_diagnostic_problems import (
    RetainedCollinearityProblem,
    RetainedGammaProblem,
    RetainedSupportProblem,
)
from powermean.results.retained_gamma import (
    RetainedGammaResult,
    snapshot_gamma_result,
)
from powermean.results.retained_input_values import (
    InputKey,
    ProblemKey,
    RetainedInputSnapshot,
    SampleKey,
    _owned_row_ledger,
)
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.results.retained_support import (
    RetainedSupportResult,
    snapshot_support_result,
)
from powermean.solve.support import SupportControls, SupportResult
from powermean.solve.support_gamma import GammaProblem, GammaResult
from powermean.solve.support_lp import LPControls
from powermean.solve.support_problem import SupportProblem

Family = PowerVariance | NB2Variance
RetainedClosureSnapshot = Closure

__all__ = [
    "Family",
    "RetainedClosureSnapshot",
    "RetainedCollinearityInvocation",
    "RetainedOutcomeSnapshot",
    "RetainedRecessionInvocation",
    "RetainedRound",
    "RetainedSampleRequest",
    "RetainedSupportInvocation",
    "_owned_closure",
    "build_sample_request_snapshot",
]


def _owned_closure(closure: Closure) -> Closure:
    """Return a fresh ``Closure`` with owned passes and row ledger."""
    if type(closure) is not Closure:
        raise TypeError("closure must be a concrete Closure")
    passes = tuple(
        ClosurePass(
            visits=tuple(Visit(v.block, owned_array(v.removed)) for v in p.visits),
            separation=(
                None
                if p.separation is None
                else LevelSeparation(
                    owned_array(p.separation.removed),
                    owned_array(p.separation.blocks),
                    tuple(owned_array(lvl) for lvl in p.separation.levels),
                )
            ),
        )
        for p in closure.passes
    )
    return Closure(passes=passes, ledger=_owned_row_ledger(closure.ledger))


@dataclass(frozen=True)
class RetainedSampleRequest:
    """Owned snapshot of ``SampleRequest``."""

    omega: float
    positive_only: bool
    variance: Family | None
    keep_singletons: bool
    support: SupportControls | None
    recession: LPControls | None
    collinearity: DiagnosticControls | None
    mode: str


def build_sample_request_snapshot(
    *,
    omega: float,
    positive_only: bool,
    variance: Family | None,
    keep_singletons: bool,
    support: SupportControls | None,
    recession: LPControls | None,
    collinearity: DiagnosticControls | None,
    mode: str,
) -> RetainedSampleRequest:
    """Build a :class:`RetainedSampleRequest` from lowered request fields."""
    return RetainedSampleRequest(
        float(omega),
        bool(positive_only),
        variance,
        bool(keep_singletons),
        support,
        recession,
        collinearity,
        str(mode),
    )


@dataclass(frozen=True, init=False)
class RetainedSupportInvocation:
    """Detached snapshot of an admitted ``SupportInvocation``."""

    problem_key: ProblemKey
    problem: RetainedSupportProblem
    controls: SupportControls
    result: RetainedSupportResult
    _removed: IndexArray

    def __init__(
        self,
        *,
        problem: SupportProblem,
        controls: SupportControls,
        result: SupportResult,
        removed: IndexArray,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> None:
        """Detach a support invocation's problem, controls, result, and removed rows."""
        pmap = ProblemIdentityMap() if problem_map is None else problem_map
        part_map = PartitionIdentityMap() if partition_map is None else partition_map
        prob = RetainedSupportProblem.from_problem(
            problem,
            register_problem=pmap.register_support,
            detach_group=part_map.register_grouping,
        )
        res = snapshot_support_result(
            result,
            problem_map=pmap,
            partition_map=part_map,
        )
        object.__setattr__(self, "problem_key", prob.key)
        object.__setattr__(self, "problem", prob)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "result", res)
        object.__setattr__(self, "_removed", owned_array(removed))

    @property
    def removed(self) -> IndexArray:
        """Return a fresh owned copy of ``removed``."""
        return owned_array(self._removed)


@dataclass(frozen=True, init=False)
class RetainedRecessionInvocation:
    """Detached snapshot of an admitted ``RecessionInvocation``."""

    problem_key: ProblemKey
    problem: RetainedGammaProblem
    controls: LPControls
    result: RetainedGammaResult

    def __init__(
        self,
        *,
        problem: GammaProblem,
        controls: LPControls,
        result: GammaResult,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> None:
        """Detach a recession invocation's problem, controls, and gamma result."""
        pmap = ProblemIdentityMap() if problem_map is None else problem_map
        part_map = PartitionIdentityMap() if partition_map is None else partition_map
        prob = RetainedGammaProblem.from_problem(
            problem,
            register_problem=pmap.register_gamma,
            detach_group=part_map.register_grouping,
        )
        res = snapshot_gamma_result(
            result,
            problem_map=pmap,
            partition_map=part_map,
        )
        object.__setattr__(self, "problem_key", prob.key)
        object.__setattr__(self, "problem", prob)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "result", res)


@dataclass(frozen=True, init=False)
class RetainedCollinearityInvocation:
    """Detached snapshot of an admitted ``CollinearityInvocation``."""

    problem_key: ProblemKey
    problem: RetainedCollinearityProblem
    controls: DiagnosticControls | None
    mode: str
    analysis: RetainedCollinearityAnalysis

    def __init__(
        self,
        *,
        problem: CollinearityProblem,
        controls: DiagnosticControls | None,
        mode: str,
        analysis: CollinearityAnalysis,
        problem_map: ProblemIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> None:
        """Detach a collinearity invocation's problem, controls, mode, and analysis."""
        pmap = ProblemIdentityMap() if problem_map is None else problem_map
        part_map = PartitionIdentityMap() if partition_map is None else partition_map
        prob = RetainedCollinearityProblem.from_problem(
            problem,
            register_problem=pmap.register_collinearity,
            detach_group=part_map.register_grouping,
        )
        ana = snapshot_collinearity_analysis(
            analysis,
            problem_map=pmap,
            partition_map=part_map,
        )
        object.__setattr__(self, "problem_key", prob.key)
        object.__setattr__(self, "problem", prob)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "mode", str(mode))
        object.__setattr__(self, "analysis", ana)


@dataclass(frozen=True, init=False)
class RetainedRound:
    """Detached snapshot of one coordinator ``Round``."""

    support: RetainedSupportInvocation | None
    _closure: Closure

    def __init__(
        self,
        *,
        closure: Closure,
        support: RetainedSupportInvocation | None = None,
    ) -> None:
        """Store owned copies of the closure and optional support invocation."""
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "_closure", _owned_closure(closure))

    @property
    def closure(self) -> Closure:
        """Return a fresh owned ``Closure``."""
        return _owned_closure(self._closure)


@dataclass(frozen=True)
class RetainedOutcomeSnapshot:
    """Detached snapshot of an admitted ``SampleOutcome`` with result-local keys."""

    input_snapshot: RetainedInputSnapshot
    request: RetainedSampleRequest
    status: str
    _ledger: RowLedger
    rounds: tuple[RetainedRound, ...]
    recession: RetainedRecessionInvocation | None = None
    collinearity: RetainedCollinearityInvocation | None = None
    sample: RetainedSampleSnapshot | None = None
    origin: RetainedOutcomeSnapshot | None = None
    input_key: InputKey = 0
    sample_key: SampleKey | None = None

    def __getstate__(self) -> Any:
        """Export declared rich state without carrying a private owner recipe.

        Generic/subclass copying retains object.__getstate__, including slots and
        existing reduction hooks. Only this exact private native view expands.
        """
        state = object.__getstate__(self)
        if type(self) is RetainedOutcomeSnapshot and "_numeric_details" in vars(self):
            state = dict(vars(self))
            for field in fields(self):
                state[field.name] = getattr(self, field.name)
            for name in ("_numeric_details",):
                state.pop(name, None)
            # The requested legacy fields now ARE the owned state. Future rich
            # copies do not reconstruct another full-row projection.
            vars(self).update(state)
            for name in tuple(vars(self)):
                if name not in state:
                    vars(self).pop(name)
        return state

    @classmethod
    def _from_numeric(
        cls, *, input_snapshot: RetainedInputSnapshot,
        request: RetainedSampleRequest,
        sample: RetainedSampleSnapshot, details: tuple[Any, Any, Any],
    ) -> RetainedOutcomeSnapshot:
        """Lossless private coordinator view, not a diagnostic export prerequisite."""
        result = object.__new__(cls)
        for name, value in (
            ("input_snapshot", input_snapshot), ("request", request),
            ("status", "complete"), ("_ledger", sample._ledger),
            ("sample", sample), ("input_key", input_snapshot.key),
            ("sample_key", sample.key), ("_numeric_details", details),
        ):
            object.__setattr__(result, name, value)
        return result

    def __getattribute__(self, name: str) -> Any:
        """Detailed diagnostic exports retain their full original typed records."""
        if (type(self) is RetainedOutcomeSnapshot and type(name) is str
                and name in ("rounds", "recession", "collinearity", "origin")):
            state = object.__getattribute__(self, "__dict__")
            details = state.get("_numeric_details")
            if details is not None:
                details[2](self, details[:2])
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        """Store detached coordinator outcome components with result-local keys."""
        if type(self._ledger) is not RowLedger:
            raise TypeError("ledger must be a concrete RowLedger")
        skey = None if self.sample is None else self.sample.key
        object.__setattr__(self, "input_key", self.input_snapshot.key)
        object.__setattr__(self, "sample_key", skey)
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "rounds", tuple(self.rounds))
        object.__setattr__(self, "_ledger", _owned_row_ledger(self._ledger))

    @property
    def prepared(self) -> PreparedFrame:
        """Return a fresh deeply owned ``PreparedFrame``."""
        s = self.input_snapshot
        if s.frame is None or s.columns is None:
            raise AttributeError("PreparedFrame is not available on this snapshot")
        return PreparedFrame(
            frame=s.frame,
            base=s.base,
            labels=s.labels,
            columns=s.columns,
            subset_missing=s.subset_missing,
            effects=s.effects,
            dependence=s.dependence,
        )

    @property
    def ledger(self) -> RowLedger:
        """Return a fresh deeply owned ``RowLedger``."""
        return _owned_row_ledger(self._ledger)
