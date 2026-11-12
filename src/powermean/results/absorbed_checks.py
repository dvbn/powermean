"""Detached alarm, RHS check, RHS result, and Newton direction snapshots.

This leaf module defines :class:`RetainedZeroRowAlarm`, :class:`RetainedChordAlarm`,
:class:`RetainedAlarm`, :func:`_owned_alarm`, :class:`RetainedRHSCheck`,
:class:`RetainedNativeCheck`, :class:`RetainedRHSResult`, :func:`_owned_rhs_result`, and
:class:`RetainedDirection`.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.fe.absorb_acceleration import CONVERGED
from powermean.fe.absorb_acceleration import Work as AbsorbWork
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import _tuple_arrays
from powermean.solve.absorbed_rhs import RHSGuard, RHSResult, within_forcing
from powermean.solve.profiled_indications import Alarm, Where

__all__ = [
    "RetainedAlarm",
    "RetainedChordAlarm",
    "RetainedDirection",
    "RetainedNativeCheck",
    "RetainedRHSCheck",
    "RetainedRHSResult",
    "RetainedZeroRowAlarm",
    "_owned_alarm",
    "_owned_rhs_result",
]


@dataclass(frozen=True)
class RetainedZeroRowAlarm:
    """Detached snapshot of ``ZeroRowAlarm`` with owned arrays on every exposure."""

    threshold: float
    lowest: float
    _rows: IndexArray
    _positions: IndexArray
    _indexes: FloatArray

    @property
    def rows(self) -> IndexArray:
        """Return a fresh owned copy of the triggering sample rows."""
        return owned_array(self._rows)

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of the triggering original positions."""
        return owned_array(self._positions)

    @property
    def indexes(self) -> FloatArray:
        """Return a fresh owned copy of the triggering indexes."""
        return owned_array(self._indexes)


@dataclass(frozen=True)
class RetainedChordAlarm:
    """Detached snapshot of ``ChordAlarm`` with owned arrays on every exposure."""

    kind: str
    losses: tuple[float, float]
    sigma: float
    lowest: float
    highest: float
    positive_lowest: float
    _before: FloatArray
    _after: FloatArray

    @property
    def before(self) -> FloatArray:
        """Return a fresh owned copy of the ``before`` index."""
        return owned_array(self._before)

    @property
    def after(self) -> FloatArray:
        """Return a fresh owned copy of the ``after`` index."""
        return owned_array(self._after)


@dataclass(frozen=True)
class RetainedAlarm:
    """Detached snapshot of ``Alarm`` with owned nested zero-row and chord alarms."""

    where: Where
    attempt: int
    iteration: int
    converged: bool
    zero_rows: RetainedZeroRowAlarm | None
    chord: RetainedChordAlarm | None

    @classmethod
    def from_alarm(cls, a: Alarm) -> RetainedAlarm:
        """Detach an ``Alarm`` and own its nested zero-row and chord arrays."""
        z, c = a.zero_rows, a.chord
        zr = (
            None
            if z is None
            else RetainedZeroRowAlarm(
                float(z.threshold),
                float(z.lowest),
                owned_array(z.rows),
                owned_array(z.positions),
                owned_array(z.indexes),
            )
        )
        ch = (
            None
            if c is None
            else RetainedChordAlarm(
                str(c.kind),
                (float(c.losses[0]), float(c.losses[1])),
                float(c.sigma),
                float(c.lowest),
                float(c.highest),
                float(c.positive_lowest),
                owned_array(c.before),
                owned_array(c.after),
            )
        )
        return cls(a.where, int(a.attempt), int(a.iteration), bool(a.converged), zr, ch)


def _owned_alarm(a: Alarm | None) -> RetainedAlarm | None:
    """Detach an optional ``Alarm``."""
    return None if a is None else RetainedAlarm.from_alarm(a)


@dataclass(frozen=True)
class RetainedRHSCheck:
    """Detached snapshot of ``RHSCheck`` with owned error vectors on every exposure."""

    _errors: tuple[FloatArray, ...]
    maximum: float

    @property
    def errors(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the per-block error vectors."""
        return _tuple_arrays(self._errors)


@dataclass(frozen=True)
class RetainedNativeCheck:
    """Detached snapshot of ``NativeCheck`` with owned sum vectors on every exposure."""

    _sums: tuple[FloatArray, ...]
    maximum: float

    @property
    def sums(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the per-block level sum vectors."""
        return _tuple_arrays(self._sums)


@dataclass(frozen=True)
class RetainedRHSResult:
    """Detached snapshot of ``RHSResult`` with owned arrays and checks."""

    _h: FloatArray
    _u: FloatArray
    _alpha: tuple[FloatArray, ...]
    check: RetainedRHSCheck | None
    native_check: RetainedNativeCheck | None
    status: str
    reason: str
    tolerance: float
    forcing: float
    initial: float | None
    guard: RHSGuard
    max_sweeps: int
    acceleration: bool
    work: AbsorbWork

    @classmethod
    def from_result(cls, r: RHSResult) -> RetainedRHSResult:
        """Detach an ``RHSResult`` and own all numeric vectors and checks."""
        chk = (
            None
            if r.check is None
            else RetainedRHSCheck(_tuple_arrays(r.check.errors), float(r.check.maximum))
        )
        nchk = (
            None
            if r.native_check is None
            else RetainedNativeCheck(
                _tuple_arrays(r.native_check.sums), float(r.native_check.maximum)
            )
        )
        grd = RHSGuard(
            r.guard.absorb_tol_final, r.guard.weight_total, r.guard.gradient_tol
        )
        return cls(
            owned_array(r.h),
            owned_array(r.u),
            _tuple_arrays(r.alpha),
            chk,
            nchk,
            str(r.status),
            str(r.reason),
            float(r.tolerance),
            float(r.forcing),
            None if r.initial is None else float(r.initial),
            grd,
            int(r.max_sweeps),
            bool(r.acceleration),
            r.work,
        )

    @property
    def h(self) -> FloatArray:
        """Return a fresh owned copy of ``h``."""
        return owned_array(self._h)

    @property
    def u(self) -> FloatArray:
        """Return a fresh owned copy of ``u``."""
        return owned_array(self._u)

    @property
    def alpha(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of ``alpha``."""
        return _tuple_arrays(self._alpha)

    @property
    def converged(self) -> bool:
        """Return True when ``status == 'converged'``."""
        return self.status == CONVERGED

    @property
    def maximum(self) -> float | None:
        """Return the largest absolute equation at the candidate, or ``None``."""
        return None if self.check is None else self.check.maximum

    @property
    def absolute(self) -> bool | None:
        """Return whether the candidate's maximum is at most ``tolerance``."""
        return None if self.check is None else self.check.maximum <= self.tolerance

    @property
    def relative(self) -> bool | None:
        """Return whether ``maximum <= forcing * initial``."""
        if self.check is None or self.initial is None:
            return None
        return within_forcing(self.check.maximum, self.forcing, self.initial)

    @property
    def final_accuracy(self) -> bool | None:
        """Return whether the candidate's maximum is at most ``af``."""
        return (
            None
            if self.check is None
            else self.check.maximum <= self.guard.absorb_tol_final
        )

    @property
    def sigma(self) -> float | None:
        """Return the candidate's native residual, if evaluated."""
        return None if self.native_check is None else self.native_check.maximum

    @property
    def native(self) -> bool | None:
        """Return whether ``sigma <= gtol``, or ``None`` if not evaluated."""
        return (
            None
            if self.native_check is None
            else self.native_check.maximum <= self.guard.gradient_tol
        )


def _owned_rhs_result(r: RHSResult | None) -> RetainedRHSResult | None:
    """Detach an optional ``RHSResult``."""
    return None if r is None else RetainedRHSResult.from_result(r)


@dataclass(frozen=True)
class RetainedDirection:
    """Detached snapshot of ``Direction`` with owned step arrays on every access."""

    _d_gamma: FloatArray
    _d_q: FloatArray
    derivative: float
    decrement: float
    tolerance: float

    @property
    def d_gamma(self) -> FloatArray:
        """Return a fresh owned copy of ``d_gamma``."""
        return owned_array(self._d_gamma)

    @property
    def d_q(self) -> FloatArray:
        """Return a fresh owned copy of ``d_q``."""
        return owned_array(self._d_q)
