"""The admitted inputs, the full point and the absorption evidence of the linear path.

Internal module; nothing here certifies a point or forms inference.  The
notation and the checked
arithmetic are those of :mod:`powermean.solve.linear_absorbed_numeric`.

Inputs.  The admitted sample, an ``FESample`` at ``omega = 0`` with fixed-effect
blocks, and :class:`LinearControls`, the three existing explicit controls checked in
the absorber's domains.

Records.  :class:`LinearPoint`, :class:`PointNormalization` and :class:`FreshEquations`
derive their values from their inputs, so no construction can supply them: the order
map applied once and the authoritative ``q``; the fitted-index route's normalization,
the only reuse of its solvers, with the gap of its index rebuilt in the same order; and
the original equations evaluated afresh at ``q`` (``e = l - q``, ``Xraw'(a e)``, every
level sum of ``a e``, the raw-weight level totals and the level means).

Absorption evidence.  :func:`_bound_absorption` admits one result of the absorber as
the outcome of one call (its values, columns and prepared weights) under the explicit
controls, with the convergence that the caller's ending states; it reads the result,
never a candidate.  The state is bound to the call; the result states the call's
tolerance (a float equal to the control) and acceleration (a bool equal to it); its
work counts are nonnegative integers, never booleans, with the sweeps within the
call's budget; its reason is a string, empty exactly on convergence; its residual is
the values less the lift of its state, bit for bit.  The fresh check of that residual
with the current weights is re-run for every result, converged or not, and decides
the diagnostics as the absorber states them (success is a completed check within the
tolerance): where the
check refuses by range, the result holds neither errors nor maximum and its status is
``range``; where it completes, the errors are its own bit for bit, the maximum is a
float equal to its maximum, and the check passes exactly when the result converged, so
that budget and breakdown keep a failing check and range a failing check or none.  No
iteration is replayed: the positive counts, and budget against breakdown, are the
absorber's report, checked in their domains only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.collinearity_numeric import is_integer, owned
from powermean.data.sample_fe_state import FESample, same_bits
from powermean.fe.absorb import (
    AbsorptionResult,
    AbsorptionState,
    _budget,
    _flag,
    _tolerance,
)
from powermean.fe.absorb_acceleration import RANGE
from powermean.fe.absorb_numeric import (
    FreshCheck,
    PreparedWeights,
    candidate,
    checked_difference,
    checked_product,
    checked_quotient,
    fresh_check,
    group_sums,
)
from powermean.fe.connectivity import IndicatorRank, indicator_rank
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_direction import normalize
from powermean.solve.linear_absorbed_numeric import (
    _own,
    base_index,
    raw_slope_sums,
    raw_weights,
    reconstruct,
)

BOUND = "the record is not bound to its sample, controls and earlier records"


# ------------------------------------------------------------------- the inputs


def linear_sample(sample: object) -> FESample:
    """The admitted sample of this path: an ``FESample`` at ``omega = 0`` with blocks.

    ``TypeError`` for another object; ``ValueError`` for another order or route, or for
    a sample without fixed effects.  The sample's own constructor binds its blocks to
    its final rows (their positions are the sample's).
    """
    if type(sample) is not FESample:
        raise TypeError("sample must be an FESample")
    if sample.omega != 0 or sample.problem.route != "linear":
        raise ValueError("the absorbed linear path solves omega = 0 only")
    blocks_of(sample)
    return sample


def blocks_of(sample: FESample) -> Grouping:
    """The final grouping of the sample; ``ValueError`` without fixed effects."""
    grouping = sample.grouping
    if grouping is None:
        raise ValueError("the absorbed linear path needs fixed-effect blocks")
    return grouping


@dataclass(frozen=True)
class LinearControls:
    """The three existing explicit controls of both declared absorptions.

    ``absorb_tol_final`` is the tolerance of the joint and of the final absorption and
    the bound of the normalization gap; ``absorb_max_sweeps`` the budget of each call,
    1 included; ``absorb_acceleration`` the accepted acceleration.  Each is checked in
    the absorber's domain with its refusal (``ValueError``); nothing is defaulted.
    """

    absorb_tol_final: float
    absorb_max_sweeps: int
    absorb_acceleration: bool

    def __post_init__(self) -> None:
        """Check each control; store the tolerance as a float, the budget as an int."""
        object.__setattr__(self, "absorb_tol_final", _tolerance(self.absorb_tol_final))
        object.__setattr__(self, "absorb_max_sweeps", _budget(self.absorb_max_sweeps))
        acceleration = _flag(self.absorb_acceleration)
        object.__setattr__(self, "absorb_acceleration", acceleration)


def _vector(value: object, size: int, what: str) -> FloatArray:
    """An owned read-only finite real vector of ``size`` values, or ``ValueError``."""
    found = owned(value)
    if found.size != size:
        raise ValueError(f"{what} must have {size} values")
    return found


# ------------------------------------------------------------------- the point


def _effects(effects: object, grouping: Grouping) -> tuple[FloatArray, ...]:
    """One owned finite vector per block, of its number of levels, or ``ValueError``."""
    if not isinstance(effects, tuple) or len(effects) != len(grouping.blocks):
        raise ValueError("effects must be a tuple of one vector per block")
    return tuple(
        _vector(alpha, block.n_levels, "effects")
        for block, alpha in zip(grouping.blocks, effects, strict=True)
    )


@dataclass(frozen=True, eq=False)
class LinearPoint:
    """A full point of the absorbed linear problem in original coordinates.

    ``theta``: the slopes in the sample's scaled coordinates (none without slopes);
    ``effects``: one vector per block, as the final absorption returned them.  Derived:
    ``beta``, the sample's order map of ``theta`` applied once (at ``omega = 0`` it only
    reverses the column scale; no map without slopes), ``base = o + Xraw beta`` and the
    authoritative ``index`` ``q``.  ``FloatingPointError`` where a value is not
    representable.
    """

    sample: FESample
    theta: FloatArray
    effects: tuple[FloatArray, ...]
    beta: FloatArray = field(init=False)
    base: FloatArray = field(init=False)
    index: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Own the inputs; form ``beta``, the base and ``q``."""
        sample = linear_sample(self.sample)
        grouping = blocks_of(sample)
        theta = _vector(self.theta, int(sample.active.size), "theta")
        effects = _effects(self.effects, grouping)
        order = sample.order_scale
        beta = immutable_array(np.zeros(0))
        if order is not None:
            beta = order.coefficients(theta)
        base = immutable_array(base_index(sample, beta))
        index = immutable_array(reconstruct(base, effects, grouping))
        _own(self, theta=theta, effects=effects, beta=beta, base=base, index=index)


@dataclass(frozen=True, eq=False)
class PointNormalization:
    """The accepted normalization of a point's effects, and its reconstruction gap.

    Derived from ``point``: the ``effects``, ``references`` and ``shifts`` of
    :func:`powermean.solve.absorbed_direction.normalize`; ``index``, those effects added
    to the point's base in the same order and arithmetic; ``gap = max |index - q|``;
    ``rank``, the indicator rank (bounded for three or more blocks, whose effects are a
    deterministic representative, not unique).  ``q`` stays authoritative.
    """

    point: LinearPoint
    effects: tuple[FloatArray, ...] = field(init=False)
    references: tuple[IndexArray, ...] = field(init=False)
    shifts: tuple[FloatArray, ...] = field(init=False)
    index: FloatArray = field(init=False)
    gap: float = field(init=False)
    rank: IndicatorRank = field(init=False)

    def __post_init__(self) -> None:
        """Normalize, rebuild the index the same way and measure the gap."""
        point = self.point
        if type(point) is not LinearPoint:
            raise ValueError("point must be a LinearPoint")
        grouping = blocks_of(point.sample)
        found = normalize(grouping, point.effects)
        effects = tuple(map(immutable_array, found.effects))
        index = immutable_array(reconstruct(point.base, effects, grouping))
        gap = checked_difference(index, point.index, "normalization gap")
        _own(
            self,
            effects=effects,
            references=tuple(map(frozen_index, found.references)),
            shifts=tuple(map(immutable_array, found.shifts)),
            index=index,
            gap=float(np.max(np.abs(gap))),
            rank=indicator_rank(grouping.blocks),
        )

    def within(self, tolerance: float) -> bool:
        """Whether the gap is at most ``tolerance``; equality passes."""
        return self.gap <= tolerance


@dataclass(frozen=True, eq=False)
class FreshEquations:
    """The original equations evaluated afresh at a point's ``q``, never rescaled.

    Derived from ``point``: ``residual`` ``e = l - q``, ``weighted`` ``a e`` (raw
    weights), ``slope_sums`` ``Xraw'(a e)`` (raw original active columns), every
    ``level_sums`` of ``a e``, the raw-weight ``level_totals`` and their quotients
    ``level_means``.  Evidence for the certificate, not a verdict.
    """

    point: LinearPoint
    residual: FloatArray = field(init=False)
    weighted: FloatArray = field(init=False)
    slope_sums: FloatArray = field(init=False)
    level_sums: tuple[FloatArray, ...] = field(init=False)
    level_totals: tuple[FloatArray, ...] = field(init=False)
    level_means: tuple[FloatArray, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Evaluate every equation at the point's ``q`` from the sample's raw arrays."""
        point = self.point
        if type(point) is not LinearPoint:
            raise ValueError("point must be a LinearPoint")
        sample = point.sample
        blocks = blocks_of(sample).blocks
        e = checked_difference(sample.problem.response, point.index, "fresh residual")
        a = raw_weights(sample)
        weighted = checked_product(a, e, "weighted residual")
        slope_sums = raw_slope_sums(sample, weighted)
        sums = tuple(group_sums(weighted, b.codes, b.n_levels) for b in blocks)
        totals = tuple(group_sums(a, b.codes, b.n_levels) for b in blocks)
        means = tuple(
            checked_quotient(s, d, "level mean")
            for s, d in zip(sums, totals, strict=True)
        )
        _own(
            self,
            residual=immutable_array(e),
            weighted=immutable_array(weighted),
            slope_sums=slope_sums,
            level_sums=tuple(map(immutable_array, sums)),
            level_totals=tuple(map(immutable_array, totals)),
            level_means=tuple(map(immutable_array, means)),
        )

    @property
    def largest_level_mean(self) -> float:
        """The largest absolute level mean over every block."""
        return max(float(np.max(np.abs(means))) for means in self.level_means)


# ------------------------------------------------------ the absorption evidence


def _bound_absorption(
    controls: LinearControls,
    result: object,
    call: tuple[FloatArray, tuple[str, ...], PreparedWeights],
    *,
    converged: bool,
) -> None:
    """A result of the absorber on these values and columns, under these controls.

    Its state is bound to the prepared grouping and the columns; it states the call's
    controls (:func:`_requested`) and reports its work and reason in their domains
    (:func:`_reported`); its residual is the values less the lift of its state, bit for
    bit; its diagnostics are those of the fresh check of that residual, re-run here for
    every result (:func:`_diagnosed`).  ``converged`` is the outcome that the caller's
    ending states for this call.  ``ValueError`` otherwise.
    """
    values, columns, prepared = call
    if (
        type(result) is not AbsorptionResult
        or type(result.state) is not AbsorptionState
    ):
        raise ValueError(BOUND)
    state = result.state
    same = state.grouping is prepared.grouping and state.columns == columns
    same = same and _requested(result, controls)
    same = same and _reported(result, controls.absorb_max_sweeps)
    if not same or result.converged != converged:
        raise ValueError(BOUND)
    try:
        residual = candidate(values, state.coefficients, prepared.grouping)
    except FloatingPointError as error:
        raise ValueError(BOUND) from error
    if not same_bits(result.residual, residual):
        raise ValueError(BOUND)
    check = _fresh(residual, prepared)
    if not _diagnosed(result, check, controls.absorb_tol_final):
        raise ValueError(BOUND)


def _requested(result: AbsorptionResult, controls: LinearControls) -> bool:
    """Whether the result states these controls: their float tolerance, bool flag."""
    tolerance, acceleration = result.tolerance, result.acceleration
    return (
        isinstance(tolerance, float)
        and tolerance == controls.absorb_tol_final
        and isinstance(acceleration, bool)
        and acceleration == controls.absorb_acceleration
    )


def _reported(result: AbsorptionResult, budget: int) -> bool:
    """Whether the work and the reason are in their domains, as the absorber reports.

    Every count is a nonnegative integer, never a bool, and the sweeps are at most the
    budget of the call; the reason is a string, empty exactly on convergence.  The
    positive counts themselves are not authenticated: no iteration is replayed.
    """
    counts = (result.sweeps, result.visits, result.operators, result.diagnostic_passes)
    reason = result.reason
    return (
        all(_count(count) for count in counts)
        and result.sweeps <= budget
        and isinstance(reason, str)
        and (reason == "") == result.converged
    )


def _count(value: object) -> bool:
    """Whether ``value`` is a nonnegative integer, never a bool."""
    return is_integer(value) and bool(value >= 0)


def _fresh(residual: FloatArray, prepared: PreparedWeights) -> FreshCheck | None:
    """The fresh check of ``residual`` with the current weights; ``None`` if refused."""
    try:
        return fresh_check(residual, prepared)
    except FloatingPointError:
        return None


def _diagnosed(
    result: AbsorptionResult, check: FreshCheck | None, tolerance: float
) -> bool:
    """Whether the result's diagnostics are ``check``'s, or absent where it refused.

    Absent errors and maximum, together, only with the status ``range`` and only where
    the fresh check refuses; else the errors are the check's bit for bit, the maximum a
    float equal to its maximum, and the check passes exactly when the result converged.
    """
    errors, maximum = result.column_errors, result.maximum
    if check is None:
        return errors is None and maximum is None and result.status == RANGE
    return (
        errors is not None
        and same_bits(errors, check.column_errors)
        and isinstance(maximum, float)
        and maximum == check.maximum
        and (check.maximum <= tolerance) == result.converged
    )
