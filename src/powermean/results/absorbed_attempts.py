"""Detached snapshots of absorbed Newton attempts, iterations, and final work."""

from __future__ import annotations

from dataclasses import dataclass

from powermean.fe.profile_arithmetic import EquationCheck
from powermean.model.variance import FloatArray
from powermean.results.absorbed_checks import (
    RetainedAlarm,
    RetainedChordAlarm,
    RetainedDirection,
    RetainedNativeCheck,
    RetainedRHSCheck,
    RetainedRHSResult,
    RetainedZeroRowAlarm,
    _owned_alarm,
    _owned_rhs_result,
)
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
)
from powermean.results.absorbed_points import Family, PointMapper, RetainedPointSnapshot
from powermean.results.absorbed_states import (
    RetainedAbsorbedSchur,
    RetainedAbsorptionResult,
    RetainedIndexStart,
    RetainedLocalInitializer,
    RetainedProfileState,
    _owned_absorbed_schur,
    _owned_absorption_result,
    _owned_equation_check,
    _snapshot_local_initializer,
)
from powermean.results.owned import _optional, owned_array
from powermean.solve.assessment import Assessment
from powermean.solve.newton_absorbed import IndexAttempt, IndexFinal, IndexReason
from powermean.solve.newton_convex import ProfiledIteration, ProfiledStop
from powermean.solve.newton_profiled import ProfiledAttempt, ProfiledReason
from powermean.solve.newton_trace import EvaluationCount, NativeReason, Timing
from powermean.systems.profiled import PointProfile

__all__ = [
    "Family",
    "RetainedAbsorbedSchur",
    "RetainedAbsorptionResult",
    "RetainedAlarm",
    "RetainedChordAlarm",
    "RetainedDirection",
    "RetainedFinalRows",
    "RetainedIndexAttempt",
    "RetainedIndexFinal",
    "RetainedIndexStart",
    "RetainedIterationState",
    "RetainedLocalInitializer",
    "RetainedNativeCheck",
    "RetainedPointEquations",
    "RetainedProfileState",
    "RetainedProfiledAttempt",
    "RetainedProfiledFinal",
    "RetainedProfiledIteration",
    "RetainedRHSCheck",
    "RetainedRHSResult",
    "RetainedZeroRowAlarm",
    "_owned_absorbed_schur",
    "_owned_absorption_result",
    "_owned_alarm",
    "_owned_equation_check",
    "_owned_iteration",
    "_owned_rhs_result",
    "_snapshot_local_initializer",
    "snapshot_index_attempt",
    "snapshot_profiled_attempt",
]


@dataclass(frozen=True)
class RetainedIterationState:
    """Detached snapshot of ``AcceptedState`` with owned arrays on every access."""

    _coefficients: FloatArray
    _raw_prediction: FloatArray | None
    loss: float | None
    _gradient: FloatArray | None

    @property
    def coefficients(self) -> FloatArray:
        """Return a fresh owned copy of ``coefficients``."""
        return owned_array(self._coefficients)

    @property
    def raw_prediction(self) -> FloatArray | None:
        """Return a fresh owned copy of ``raw_prediction``, or ``None``."""
        return _optional(self._raw_prediction)

    @property
    def gradient(self) -> FloatArray | None:
        """Return a fresh owned copy of ``gradient``, or ``None``."""
        return _optional(self._gradient)


@dataclass(frozen=True)
class RetainedProfiledIteration:
    """Detached snapshot of ``ProfiledIteration`` with owned arrays on every access."""

    _start: FloatArray
    state: RetainedIterationState
    native: NativeReason | None
    stop: ProfiledStop | None
    message: str
    iterations: int
    line_searches: int
    trials: int
    refused_trials: int
    evaluations: EvaluationCount
    warnings: tuple[str, ...]
    timing: Timing

    @property
    def start(self) -> FloatArray:
        """Return a fresh owned copy of ``start``."""
        return owned_array(self._start)


def _owned_iteration(it: ProfiledIteration | None) -> RetainedProfiledIteration | None:
    """Detach an optional ``ProfiledIteration``."""
    if it is None:
        return None
    st = it.state
    loss = None if st.loss is None else float(st.loss)
    c, rp = owned_array(st.coefficients), _optional(st.raw_prediction)
    state = RetainedIterationState(c, rp, loss, _optional(st.gradient))
    warns = tuple(str(w) for w in it.warnings)
    n_it, n_ls, n_tr = int(it.iterations), int(it.line_searches), int(it.trials)
    tail = (n_it, n_ls, n_tr, int(it.refused_trials), it.evaluations, warns, it.timing)
    return RetainedProfiledIteration(
        owned_array(it.start), state, it.native, it.stop, str(it.message), *tail
    )


@dataclass(frozen=True)
class RetainedPointEquations:
    """Detached snapshot of ``PointEquations`` with owned arrays on every access."""

    loss: float
    _gradient: FloatArray
    _curvature: FloatArray
    _information: FloatArray

    @property
    def gradient(self) -> FloatArray:
        """Return a fresh owned copy of ``gradient``."""
        return owned_array(self._gradient)

    @property
    def curvature(self) -> FloatArray:
        """Return a fresh owned copy of ``curvature``."""
        return owned_array(self._curvature)

    @property
    def information(self) -> FloatArray:
        """Return a fresh owned copy of ``information``."""
        return owned_array(self._information)


@dataclass(frozen=True)
class RetainedFinalRows:
    """Detached snapshot of ``FinalRows`` with owned arrays on every access."""

    loss: float
    _gradient: FloatArray
    _curvature: FloatArray
    _right: FloatArray
    total: float

    @property
    def gradient(self) -> FloatArray:
        """Return a fresh owned copy of ``gradient``."""
        return owned_array(self._gradient)

    @property
    def curvature(self) -> FloatArray:
        """Return a fresh owned copy of ``curvature``."""
        return owned_array(self._curvature)

    @property
    def right(self) -> FloatArray:
        """Return a fresh owned copy of ``right``."""
        return owned_array(self._right)


@dataclass(frozen=True)
class RetainedProfiledFinal:
    """Detached snapshot of ``FinalWork`` on the profiled route."""

    point: RetainedPointSnapshot | None
    equations: RetainedPointEquations | None
    absorbed: RetainedAbsorbedSchur | None
    alarm: RetainedAlarm | None
    reason: str

    @property
    def complete(self) -> bool:
        """Return True if the final work completed without failure."""
        return not self.reason


@dataclass(frozen=True)
class RetainedIndexFinal:
    """Detached snapshot of ``IndexFinal`` on the fitted-index route."""

    point: RetainedPointSnapshot | None
    rows: RetainedFinalRows | None
    _equations: EquationCheck | None
    absorbed: RetainedAbsorbedSchur | None
    rhs: RetainedRHSResult | None
    alarm: RetainedAlarm | None
    reason: str

    @property
    def equations(self) -> EquationCheck | None:
        """Return a fresh owned ``EquationCheck``, or ``None``."""
        return _owned_equation_check(self._equations)

    @property
    def complete(self) -> bool:
        """Return True if the final work completed without failure."""
        return not self.reason


@dataclass(frozen=True)
class RetainedProfiledAttempt:
    """Detached snapshot of ``ProfiledAttempt``."""

    iteration: RetainedProfiledIteration | None
    reason: ProfiledReason
    message: str
    opening: RetainedPointSnapshot | None
    point: RetainedPointSnapshot | None
    loss: float | None
    final: RetainedProfiledFinal | None
    alarm: RetainedAlarm | None
    assessment: Assessment | None

    @property
    def iterations(self) -> int:
        """Return the number of accepted Newton steps."""
        return 0 if self.iteration is None else int(self.iteration.iterations)


@dataclass(frozen=True)
class RetainedIndexAttempt:
    """Detached snapshot of ``IndexAttempt``."""

    iteration: RetainedProfiledIteration
    reason: IndexReason
    message: str
    opening: RetainedPointSnapshot | None
    point: RetainedPointSnapshot | None
    loss: float | None
    _gradient: FloatArray | None
    direction: RetainedDirection | None
    initializer: RetainedPointSnapshot | RetainedLocalInitializer | None
    final: RetainedIndexFinal | None
    alarm: RetainedAlarm | None
    assessment: Assessment | None

    @property
    def gradient(self) -> FloatArray | None:
        """Return a fresh owned copy of ``gradient``, or ``None``."""
        return _optional(self._gradient)

    @property
    def iterations(self) -> int:
        """Return the number of accepted Newton steps."""
        return int(self.iteration.iterations)


def snapshot_profiled_attempt(
    att: ProfiledAttempt,
    get_point: PointMapper,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedProfiledAttempt:
    """Detach a ``ProfiledAttempt`` and its optional ``FinalWork``."""
    f = att.final
    fin: RetainedProfiledFinal | None = None
    if f is not None:
        e = f.equations
        eq = (
            None
            if e is None
            else RetainedPointEquations(
                float(e.loss),
                owned_array(e.gradient),
                owned_array(e.curvature),
                owned_array(e.information),
            )
        )
        sch = _owned_absorbed_schur(
            f.absorbed, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
        )
        f_pt = None if f.point is None else get_point(f.point)
        fin = RetainedProfiledFinal(f_pt, eq, sch, _owned_alarm(f.alarm), str(f.reason))
    op = None if att.opening is None else get_point(att.opening)
    pt = None if att.point is None else get_point(att.point)
    loss = None if att.loss is None else float(att.loss)
    head = (_owned_iteration(att.iteration), att.reason, str(att.message), op, pt)
    return RetainedProfiledAttempt(
        *head, loss, fin, _owned_alarm(att.alarm), att.assessment
    )


def _snapshot_index_final(
    f: IndexFinal | None,
    get_point: PointMapper,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedIndexFinal | None:
    """Detach an optional ``IndexFinal`` record."""
    if f is None:
        return None
    r = f.rows
    rw = (
        None
        if r is None
        else RetainedFinalRows(
            float(r.loss),
            owned_array(r.gradient),
            owned_array(r.curvature),
            owned_array(r.right),
            float(r.total),
        )
    )
    sch = _owned_absorbed_schur(
        f.absorbed, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
    )
    f_pt = None if f.point is None else get_point(f.point)
    head = (f_pt, rw, _owned_equation_check(f.equations), sch)
    return RetainedIndexFinal(
        *head, _owned_rhs_result(f.rhs), _owned_alarm(f.alarm), str(f.reason)
    )


def snapshot_index_attempt(
    att: IndexAttempt,
    get_point: PointMapper,
    *,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedIndexAttempt:
    """Detach an ``IndexAttempt`` and its optional initializer and ``IndexFinal``."""
    fin = _snapshot_index_final(
        att.final,
        get_point,
        grouping_snapshot=grouping_snapshot,
        grouping_map=grouping_map,
    )
    ir = att.initializer
    if ir is None:
        init_s: RetainedPointSnapshot | RetainedLocalInitializer | None = None
    elif isinstance(ir, PointProfile):
        init_s = get_point(ir)
    else:
        init_s = _snapshot_local_initializer(
            ir, grouping_snapshot=grouping_snapshot, grouping_map=grouping_map
        )
    dr = att.direction
    d = (
        None
        if dr is None
        else RetainedDirection(
            owned_array(dr.d_gamma),
            owned_array(dr.d_q),
            float(dr.derivative),
            float(dr.decrement),
            float(dr.tolerance),
        )
    )
    it_snap = _owned_iteration(att.iteration)
    assert it_snap is not None
    op = None if att.opening is None else get_point(att.opening)
    pt = None if att.point is None else get_point(att.point)
    loss = None if att.loss is None else float(att.loss)
    head = (it_snap, att.reason, str(att.message), op, pt, loss)
    tail = (_optional(att.gradient), d, init_s, fin, _owned_alarm(att.alarm))
    return RetainedIndexAttempt(*head, *tail, att.assessment)
