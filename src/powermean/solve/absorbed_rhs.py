"""The forced full-RHS nuisance solve of the fitted-index Newton direction.

Newton in the fitted index, one weighted absorption per iteration, no dense indicator
matrix; the absorption controls are taken here as explicit arguments.  Internal
module: ``powermean`` exports nothing of it.

The equation.  With ``b = a r`` and ``C = diag(a c)`` (``a w`` only when the expected
curvature was explicitly requested) at the current index, the nuisance part of the full
joint Newton direction is the fitted ``h`` in the span of the original indicators ``D``:

    D' C h = D' b,      h = D alpha.

The right-hand side is complete: at a GPML zero row ``c = 0`` and ``r = -1``, so ``b``
is nonzero there; no ``r / c``, deleted zero row, floor, ridge, generalized inverse or
Fisher substitution enters.  The accepted ``absorb(values, C)`` solves
``D'C (values - D alpha) = 0`` and cannot represent this right-hand side;
:func:`solve_rhs` is the bounded RHS-aware solve built on its checked group sums,
products, lifts, prepared level totals and work record
(:mod:`powermean.fe.absorb_numeric`, :mod:`powermean.fe.absorb_acceleration`), which it
calls unchanged.

The candidate.  A candidate is a coefficient tuple ``alpha`` (one ``L_k``-by-1 array per
block, cold zeros on every call: no warm state, cache or API) and its ordered lift
``h = D alpha``; its authoritative residual is ``u = b - C h``, reconstructed from
``alpha`` with checked products and differences at every checked candidate, never a
separately drifting ``h`` or ``u``.  The fresh error of level ``g`` of block ``k`` is

    e_kg = group_sums(u)_kg / d_kg,      d_kg = sum_{i in kg} C_i,

in the existing absorption units, and NOT the C-weighted mean of ``u``.  With ``rho``
the maximum of ``|e|`` over every ORIGINAL block and level, and ``rho0`` that of the
call's cold initial check (``alpha = 0``, the existing first check, held fixed for the
call), success is

    rho <= tolerance  and  [rho <= forcing * rho0  or  (rho <= af and sigma <= gtol)],

each equality passing, decided in that order: the ceiling first, then the relative
bound, then ``rho <= af``, and only then ``sigma``, which is evaluated nowhere else.

The relative bound.  ``tolerance`` keeps
its absolute meaning and schedule.  ``forcing`` is a dimensionless factor in
``[0, 0.9999]``, and ``rho <= forcing * rho0`` is decided exactly by
:func:`within_forcing`, never through a rounded product: ``rho0 = 0`` passes at the
initial check, ``rho0 > 0`` never admits ``h = 0`` by this branch, ``forcing = 0``
admits an exactly zero residual only, and a positive subnormal ``rho0`` is kept.
:func:`next_forcing` is the guarded accepted-progress update of SciPy v1.16.2
``nonlin_solve`` (its four constants and its rule, not its code, norm, raw-norm
multiplier, Krylov solver or outer optimizer), which the caller
applies to the full native score ``s = max |G|`` of each new accepted origin.

The guarded exception.  The EXISTING final absorption accuracy ``af``,
the raw-weight total ``A`` that the native score arithmetic has already formed (never
``sum(C)``) and the EXISTING native gradient tolerance ``gtol`` arrive together in the
frozen :class:`RHSGuard`; no new control, floor on ``forcing * rho0`` or clamp exists.
``sigma`` (:func:`native_check`) is the native FE linear residual of the candidate:
``(-u) / A`` divided ROW BY ROW in the native float64 division of the score (an inexact
underflow is permitted there, as in the score itself), checked finite, then the
unchanged sequential level sums of every original block, then ``max |.|``; not the
stricter checked quotient, not a division after grouping, not the slope-plus-FE score.
Its block evaluations are counted before their arithmetic, the division inside the
first, and charged as diagnostic passes whether the solve then stops or continues; a
refusal ends the call with ``range`` at the same candidate, its check, ``u``, ``rho``
and work kept.  An unevaluated ``sigma`` is unavailable, never a pass.  At a final call
(``tolerance = af``) success is ``rho <= af`` and (relative or ``sigma <= gtol``), not
absolute-only.  A limitation: at ``rho0 = 0`` the relative branch passes
the cold candidate without ``sigma``, even where the native row-first residual is not
zero (``u = (1e16, 1, -1e16)``, ``A = 3``: the level sum of ``u`` is exactly zero, the
native residual ``-0.5``); no universal no-false-zero guarantee follows.

A target that the budget cannot reach ends with the existing ``budget``, ``breakdown``
or ``range`` status: nothing is floored, zeroed or solved again.  Success states the
requested finite residual bounds, not exact consistency, solvability or rank: an
inconsistent component that is small enough passes.

Forced visits.  One visit of block ``k``, counted before its first operation:
reconstruct ``h`` and ``u`` from ``alpha``, ``s = group_sums(u, k) / d_k``, and
``alpha_k <- alpha_k + s`` (checked).  One block takes the direct visit whatever the
acceleration; several blocks without acceleration take forward sweeps ``1, ..., G``,
every original block checked afresh after each.  A candidate commits only once a whole
sweep is reconstructed: a visit that refuses keeps the previous candidate and the work
attempted, and a fresh check that refuses keeps its new candidate.

Acceleration, several blocks.  ``T`` is the accepted symmetric walk
``Q_1, ..., Q_G, ..., Q_1`` and ``A = I - T``: applied to a lifted coefficient tuple
``p``, ``A p`` is the accumulated correction of the accepted
``walk(lift(p), prepared, symmetric_order)``.  The forced symmetric visits from
``alpha``, on a temporary copy that is never committed, give ``r``, the sum of their
corrections (not the rounded difference of two large states): if ``D'C h* = D'b``, each
forced visit acts on ``h* - h`` as the homogeneous projection does, so
``r = A (h* - h)``.  Then, with C-weighted products of lifted rows and the accepted
checked tuple arithmetic:

    p = r, rho = <r, r>_C;   v = A p, den = <p, v>_C, t = rho / den;
    alpha <- alpha + t p, r <- r - t v, the fresh full-RHS check (stop if it passes);
    rho_new = <r, r>_C, p <- r + (rho_new / rho) p, rho <- rho_new.

A ``rho`` or ``den`` that is not finite and positive, or a ``rho_new`` that is zero or
not finite while the check fails, is ``breakdown``: neither rank evidence nor
convergence, and nothing restarts.  On the positive-C quotient this is the accepted
symmetric conjugate gradient with an affine forcing; with additional C-null directions
it is only a quotient argument, and neither unique zero-row predictions nor solvability
follows.  A passing fresh check is not evidence that the equations are consistent.

Work and statuses (:class:`~powermean.fe.absorb_acceleration.Work`): the initial
check, and each native residual of the guarded exception, costs
diagnostic passes, one per block attempted, not sweeps; a forward traversal (G
visits) or the direct visit costs one sweep; every symmetric application, the forced
initialization included, reserves two sweeps and one operator before it starts and
attempts ``2G - 1`` visits, so that a budget of one cannot start an operator with
several blocks.  The statuses are ``converged``, ``budget``, ``range`` (a refused
operation of the checked arithmetic) and ``breakdown``; an unsuccessful one keeps the
last committed candidate, its check when that check completed, and the work attempted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb import _budget, _flag, _tolerance
from powermean.fe.absorb_acceleration import (
    BREAKDOWN,
    BUDGET,
    CONVERGED,
    RANGE,
    STATUSES,
    Work,
    _positive_finite,
    operator_cost,
)
from powermean.fe.absorb_numeric import (
    Attempted,
    Coefficients,
    PreparedWeights,
    added,
    checked_difference,
    checked_product,
    checked_quotient,
    checked_sum,
    forward_order,
    group_sums,
    lift,
    scaled,
    subtracted,
    symmetric_order,
    walk,
    weighted_inner,
    zero_coefficients,
)
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray

RIGHT = "b must be a real vector with one finite value per row"
RHO = (
    "zero or nonfinite recurrence residual norm while the full right-hand-side "
    "equations exceed a requested bound"
)
DEN = (
    "nonpositive or nonfinite curvature <p, A p>_C while the full right-hand-side "
    "equations exceed a requested bound"
)
FORCING = "forcing must be a finite real number in [0, 0.9999]"
OPERANDS = "the relative bound compares finite nonnegative numbers"
HISTORY = (
    "the forcing history needs finite nonnegative scores and a factor in [0, 0.9999]"
)
UNREPRESENTABLE = "the selected positive forcing factor is not representable in float64"
POSITIVE = "must be a finite positive real number"
NATIVE = "the native normalization (-u)/A is not representable in float64"

# SciPy v1.16.2 nonlin_solve's forcing constants: the first factor, the
# progress weight gamma, the cap eta_max and the safeguard threshold.
FORCING_INITIAL = 1e-3
FORCING_GAMMA = 0.9
FORCING_CAP = 0.9999
FORCING_THRESHOLD = 0.1


# --------------------------------------------------------------- the forcing


def within_forcing(maximum: float, forcing: float, initial: float) -> bool:
    """``maximum <= forcing * initial``, decided exactly.

    With ``maximum = p / q``, ``forcing = u / v`` and ``initial = r / t`` the exact
    ratios of the doubles (``float.as_integer_ratio``, positive denominators), the
    bound holds if and only if ``p v t <= u r q`` in Python's exact integers: no
    rounded product is formed, so a target below the smallest subnormal is neither
    rounded to zero nor up to it.  Equality passes.

    Raises
    ------
    ValueError
        An operand that is negative or not finite.
    """
    values = (float(maximum), float(forcing), float(initial))
    if not all(math.isfinite(x) and x >= 0.0 for x in values):
        raise ValueError(OPERANDS)
    (p, q), (u, v), (r, t) = (x.as_integer_ratio() for x in values)
    return p * v * t <= u * r * q


def next_forcing(previous: tuple[float, float], score: float) -> float:
    """The forcing factor of a new accepted origin (SciPy's rule).

    ``previous`` is the last issued pair ``(s_old, eta_old)`` and ``score`` the new
    origin's full native ``s = max |G|``.  In this order, each a binary64 operation:

        guard = 0.9 (eta_old eta_old)
        s_old = 0:      eta_old if s = 0, else 0.9999
        s > 2 s_old:    0.9999, the cap, without forming the ratio or its square
        eta_A = 0.9 (r r),  r = s / s_old
        guard >= 0.1:   min(0.9999, max(eta_A, guard))
        otherwise:      min(0.9999, eta_A)

    For ``s > 2 s_old`` the formula selects the cap anyway (``0.9 r^2 > 3.6``), so the
    shortcut only avoids an overflowing ratio or square.  A positive subnormal selected
    factor is kept; an ``eta_A`` that underflows to zero is discarded when the guard
    selects, and is an arithmetic failure when selected for a positive ``s``.

    Raises
    ------
    FloatingPointError
        The selected factor of a positive score rounds to zero.
    ValueError
        A score that is negative or not finite, or ``eta_old`` outside ``[0, 0.9999]``.
    """
    s_old, eta_old, s_new = float(previous[0]), float(previous[1]), float(score)
    scores = all(math.isfinite(x) and x >= 0.0 for x in (s_old, s_new))
    if not scores or not 0.0 <= eta_old <= FORCING_CAP:
        raise ValueError(HISTORY)
    guard = FORCING_GAMMA * (eta_old * eta_old)
    if s_old == 0.0:
        return eta_old if s_new == 0.0 else FORCING_CAP
    if s_new > 2.0 * s_old:
        return FORCING_CAP
    ratio = s_new / s_old
    eta = FORCING_GAMMA * (ratio * ratio)
    if not guard < FORCING_THRESHOLD:
        return min(FORCING_CAP, max(eta, guard))
    if eta == 0.0 < s_new:
        raise FloatingPointError(UNREPRESENTABLE)
    return min(FORCING_CAP, eta)


def _forcing(value: object) -> float:
    """A finite real number in ``[0, 0.9999]`` as a float, or ``ValueError``."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.number
    ):
        raise ValueError(FORCING)
    if np.iscomplexobj(value):
        raise ValueError(FORCING)
    found = float(value)
    if not 0.0 <= found <= FORCING_CAP:
        raise ValueError(FORCING)
    return found


# ------------------------------------------------------- the guarded exception


def _positive(value: object, name: str) -> float:
    """A finite positive real number as a float, or ``ValueError`` naming ``name``."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.number
    ):
        raise ValueError(f"{name} {POSITIVE}")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} {POSITIVE}")
    found = float(value)
    if not (math.isfinite(found) and found > 0.0):
        raise ValueError(f"{name} {POSITIVE}")
    return found


@dataclass(frozen=True)
class RHSGuard:
    """The existing accuracies of the guarded exception.

    Attributes
    ----------
    absorb_tol_final : float
        ``af``, the requested final absorption accuracy of the absorbed options.
    weight_total : float
        ``A``, the raw-weight total ``np.sum(a)`` that the native score arithmetic has
        already formed; never ``sum(C)``.
    gradient_tol : float
        ``gtol``, the native gradient tolerance of the solver options.

    Each is a finite positive real number, stored as a float; anything else is refused
    with ``ValueError`` here, before any diagnostic work.  No new control.
    """

    absorb_tol_final: float
    weight_total: float
    gradient_tol: float

    def __post_init__(self) -> None:
        """Check and store the three values."""
        for name in ("absorb_tol_final", "weight_total", "gradient_tol"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


@dataclass(frozen=True, eq=False)
class NativeCheck:
    """The native FE linear residual of one candidate.

    Attributes
    ----------
    sums : tuple of arrays of float
        One vector per block: the level sums of ``(-u) / A`` in level order,
        read-only and owned.
    maximum : float
        ``sigma``, the largest absolute level sum over every block.
    """

    sums: tuple[FloatArray, ...]
    maximum: float

    def __post_init__(self) -> None:
        """Own the sums, each finite; the maximum must be theirs."""
        owned = tuple(immutable_array(s) for s in self.sums)
        if not owned or any(s.ndim != 1 or not np.isfinite(s).all() for s in owned):
            raise ValueError("sums must be one finite vector per block")
        if self.maximum != max(float(np.max(np.abs(s))) for s in owned):
            raise ValueError("maximum must be the largest absolute level sum")
        object.__setattr__(self, "sums", owned)


def _normalized(u: FloatArray, total: float) -> FloatArray:
    """``(-u) / A`` row by row in the native float64 division, checked finite.

    The division of the native score (``v = (a * -r) / A``): an inexact underflow is
    permitted and a signed zero kept; a value that is not finite refuses with
    ``FloatingPointError``.  Not the stricter checked quotient.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        found: FloatArray = np.divide(np.negative(u), total)
    if not np.isfinite(found).all():
        raise FloatingPointError(NATIVE)
    return found


def native_check(
    u: FloatArray, prepared: PreparedWeights, total: float, attempted: Attempted
) -> NativeCheck:
    """``sigma``: the native FE linear residual of the rows ``u`` (one column).

    The rows divided by ``total`` first (:func:`_normalized`), then the unchanged
    sequential level sums of every original block in grouping order, then the largest
    absolute sum.  Each block is counted in ``attempted`` before its arithmetic, the
    division inside the first, so that a refusal is counted.  ``FloatingPointError``
    when the division or a level sum is not representable.
    """
    normalized: FloatArray | None = None
    sums: list[FloatArray] = []
    for block in prepared.grouping.blocks:
        attempted.begin()
        if normalized is None:
            normalized = _normalized(u, total)
        sums.append(group_sums(normalized, block.codes, block.n_levels)[:, 0])
    return NativeCheck(tuple(sums), max(float(np.max(np.abs(s))) for s in sums))


# ---------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class RHSCheck:
    """The fresh full-RHS equations of every original block at one candidate.

    Attributes
    ----------
    errors : tuple of arrays of float
        One vector per block, ``e_kg = group_sums(u)_kg / d_kg`` in level order,
        read-only and owned.
    maximum : float
        ``max |e_kg|`` over every block and level.
    """

    errors: tuple[FloatArray, ...]
    maximum: float

    def __post_init__(self) -> None:
        """Own the errors, each finite; the maximum must be theirs."""
        owned = tuple(immutable_array(e) for e in self.errors)
        if not owned or any(e.ndim != 1 or not np.isfinite(e).all() for e in owned):
            raise ValueError("errors must be one finite vector per block")
        if self.maximum != max(float(np.max(np.abs(e))) for e in owned):
            raise ValueError("maximum must be the largest absolute error")
        object.__setattr__(self, "errors", owned)


@dataclass(frozen=True, eq=False)
class RHSResult:
    """The outcome of one forced full-RHS solve: candidate, check, status and work.

    Attributes
    ----------
    h : array of float
        ``D alpha`` at the returned candidate, every row, read-only and owned.
    u : array of float
        ``b - C h`` there, every row, read-only and owned.
    alpha : tuple of arrays of float
        The candidate's coefficients, one vector per block, read-only and owned: span
        evidence of the direction, never a fitted effect.
    check : RHSCheck or None
        The fresh equations at the candidate; ``None`` when that check refused.
    native_check : NativeCheck or None
        The native residual of the guarded exception at the candidate, when it was
        evaluated there; ``None`` otherwise: unavailable, not a pass.
    status : str
        ``converged``, ``budget``, ``range`` or ``breakdown``.
    reason : str
        Empty on convergence; otherwise what stopped the solve.
    tolerance : float
        The requested maximum absolute equation, the ceiling; equality passes.
    forcing : float
        The requested relative factor ``eta``: the relative bound is ``forcing *
        initial``, compared exactly; equality passes.
    initial : float or None
        ``rho0``, the maximum of the cold initial check, fixed for the call; ``None``
        when that check refused.
    guard : RHSGuard
        The actual ``af``, ``A`` and ``gtol`` of the guarded exception.
    max_sweeps : int
        The budget of sweeps.
    acceleration : bool
        The realization requested (one block takes the direct visit either way).
    work : Work
        The reserved sweeps and the attempted visits, operators and passes.
    """

    h: FloatArray
    u: FloatArray
    alpha: tuple[FloatArray, ...]
    check: RHSCheck | None
    native_check: NativeCheck | None
    status: str
    reason: str
    tolerance: float
    forcing: float
    initial: float | None
    guard: RHSGuard
    max_sweeps: int
    acceleration: bool
    work: Work

    def __post_init__(self) -> None:
        """Check the status and the guard and own the arrays."""
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not isinstance(self.guard, RHSGuard):
            raise ValueError("guard must be an RHSGuard record")
        object.__setattr__(self, "h", immutable_array(self.h))
        object.__setattr__(self, "u", immutable_array(self.u))
        object.__setattr__(self, "alpha", tuple(immutable_array(a) for a in self.alpha))

    @property
    def converged(self) -> bool:
        """Whether the fresh check passed at the returned candidate."""
        return self.status == CONVERGED

    @property
    def maximum(self) -> float | None:
        """The largest absolute equation at the candidate, if its check completed."""
        return None if self.check is None else self.check.maximum

    @property
    def absolute(self) -> bool | None:
        """Whether the candidate's maximum is at most ``tolerance`` (the ceiling)."""
        return None if self.check is None else self.check.maximum <= self.tolerance

    @property
    def relative(self) -> bool | None:
        """Whether it is at most ``forcing * initial`` (:func:`within_forcing`)."""
        if self.check is None or self.initial is None:
            return None
        return within_forcing(self.check.maximum, self.forcing, self.initial)

    @property
    def final_accuracy(self) -> bool | None:
        """Whether the candidate's maximum is at most ``af``, if checked."""
        if self.check is None:
            return None
        return self.check.maximum <= self.guard.absorb_tol_final

    @property
    def sigma(self) -> float | None:
        """The candidate's native residual, if the guard evaluated it there."""
        return None if self.native_check is None else self.native_check.maximum

    @property
    def native(self) -> bool | None:
        """Whether ``sigma <= gtol``; ``None`` when ``sigma`` was not evaluated."""
        if self.native_check is None:
            return None
        return self.native_check.maximum <= self.guard.gradient_tol


# The place of a logged call in a solve: a direction's inner work or the final work.
Role = Literal["direction", "final"]


@dataclass(frozen=True)
class RHSCall:
    """One logged forced full-RHS call: where, its controls, its score, its work.

    The fields of :class:`~powermean.solve.profiled_loss.AbsorptionCall`, for
    :func:`solve_rhs`: the attempt, the accepted steps before it, the role, the
    tolerance and the accepted score that set it (``None`` for an attempt's first
    direction and the final call), the budget and realization, the status and reason,
    the work and the largest equation; the counters are ``None`` when the call was
    refused before solving (``status`` ``refused``).  Then the relative bound:
    the issued ``forcing``, the cold ``initial`` maximum, and whether the
    returned candidate met the ``absolute`` and the ``relative`` bound (``None`` where
    unavailable).  Then the guarded exception: the actual ``guard``
    (``af``, ``A``, ``gtol``, kept by a refused call too), whether the candidate met
    ``af`` (``final_accuracy``), and its ``sigma`` and ``native`` pass when the guard
    evaluated them there (``None`` otherwise: unavailable, not a pass).
    """

    attempt: int
    iteration: int
    role: Role
    tolerance: float
    score: float | None
    max_sweeps: int
    acceleration: bool
    status: str
    reason: str
    sweeps: int | None
    visits: int | None
    operators: int | None
    diagnostic_passes: int | None
    maximum: float | None
    forcing: float
    initial: float | None
    absolute: bool | None
    relative: bool | None
    guard: RHSGuard
    final_accuracy: bool | None
    sigma: float | None
    native: bool | None


@dataclass(frozen=True, eq=False)
class _Point:
    """A candidate: coefficients, their lift and residual, its check if completed.

    ``native`` is the native residual of the guarded exception at this candidate,
    once evaluated there.
    """

    alpha: Coefficients
    h: FloatArray
    u: FloatArray
    check: RHSCheck | None
    failure: str | None
    passes: int
    native: NativeCheck | None


@dataclass(frozen=True, eq=False)
class _Outcome:
    """What a realization returns: the last committed point, status, reason, work."""

    point: _Point
    status: str
    reason: str
    work: Work


@dataclass(frozen=True, eq=False)
class _Progress:
    """The recurrence between two updates: the point, ``r``, ``p``, ``rho``, work."""

    point: _Point
    r: Coefficients
    p: Coefficients
    rho: FloatArray
    work: Work


@dataclass(frozen=True)
class _Target:
    """The bounds of one call: the ceiling, ``forcing`` times ``initial``, the guard."""

    tolerance: float
    forcing: float
    initial: float
    guard: RHSGuard


# ------------------------------------------------------- candidates and visits


def _residual(b: FloatArray, prepared: PreparedWeights, h: FloatArray) -> FloatArray:
    """``u = b - C h``, each operation range-checked."""
    if (
        type(h) is np.ndarray
        and h.dtype == np.float64
        and type(b) is np.ndarray
        and b.dtype == np.float64
        and h.shape == b.shape
        and type(prepared) is PreparedWeights
        and not h.any()
        and not np.signbit(h).any()
        and np.isfinite(b).all()
    ):
        return b.copy()
    weighted = checked_product(prepared.weights[:, None], h, "curvature product")
    return checked_difference(b, weighted, "full right-hand side residual")


def _check(u: FloatArray, prepared: PreparedWeights, attempted: Attempted) -> RHSCheck:
    """``e_kg = group_sums(u)_kg / d_kg`` of every original block, each counted."""
    errors: list[FloatArray] = []
    for index, block in enumerate(prepared.grouping.blocks):
        attempted.begin()
        sums = group_sums(u, block.codes, block.n_levels)
        found = checked_quotient(sums, prepared.totals[index][:, None], "equation")
        errors.append(found[:, 0])
    return RHSCheck(tuple(errors), max(float(np.max(np.abs(e))) for e in errors))


def _evaluate(b: FloatArray, prepared: PreparedWeights, alpha: Coefficients) -> _Point:
    """``h`` and ``u`` reconstructed from ``alpha``, and the fresh check of them.

    ``FloatingPointError`` when ``h`` or ``u`` is not representable: no candidate
    exists then.  A refused check leaves a point without a check, its reason and the
    block evaluations attempted.
    """
    h = lift(alpha, prepared.grouping)
    u = _residual(b, prepared, h)
    attempted = Attempted()
    try:
        check = _check(u, prepared, attempted)
    except FloatingPointError as error:
        return _Point(alpha, h, u, None, str(error), attempted.count, None)
    return _Point(alpha, h, u, check, None, attempted.count, None)


def _visit(
    b: FloatArray,
    prepared: PreparedWeights,
    alpha: Coefficients,
    block: int,
    attempted: Attempted,
    *,
    u_current: FloatArray | None = None,
) -> tuple[Coefficients, FloatArray]:
    """One forced visit of ``block``: the updated tuple and the correction ``s``."""
    attempted.begin()
    u = (
        u_current
        if u_current is not None
        else _residual(b, prepared, lift(alpha, prepared.grouping))
    )
    encoded = prepared.grouping.blocks[block]
    sums = group_sums(u, encoded.codes, encoded.n_levels)
    correction = checked_quotient(
        sums, prepared.totals[block][:, None], "forced correction"
    )
    found = list(alpha)
    found[block] = checked_sum(alpha[block], correction, "forced coefficients")
    return tuple(found), correction


_ACCEPTED_VISIT = _visit
_ACCEPTED_RESIDUAL = _residual
_ACCEPTED_LIFT = lift


def _budget_outcome(point: _Point, work: Work, max_sweeps: int, cost: int) -> _Outcome:
    """The ``budget`` outcome at a checked point whose next unit does not fit."""
    reason = (
        f"the next unit of {cost} sweep(s) exceeds the budget of {max_sweeps} after "
        f"{work.sweeps} sweep(s)"
    )
    return _Outcome(point, BUDGET, reason, work)


# What a check decides: the outcome if the call ends there, the point, the work.
_Settled = tuple[_Outcome | None, _Point, Work]


def _settled(
    prepared: PreparedWeights, point: _Point, work: Work, target: _Target
) -> _Settled:
    """The outcome of a point whose check refused or met the bound, if any.

    Also the point and the work as they continue.  In the order of the predicate: the
    ceiling, the exact relative bound, ``rho <= af``, and only then the native residual
    (:func:`native_check`), whose attempted block evaluations are charged whatever
    follows; the point then carries it.  A refused native residual ends the call with
    ``range`` at this same point, its check and the work kept.
    """
    if point.check is None:
        return _Outcome(point, RANGE, point.failure or RANGE, work), point, work
    maximum, guard = point.check.maximum, target.guard
    if not maximum <= target.tolerance:
        return None, point, work
    if within_forcing(maximum, target.forcing, target.initial):
        return _Outcome(point, CONVERGED, "", work), point, work
    if not maximum <= guard.absorb_tol_final:
        return None, point, work
    attempted = Attempted()
    try:
        found = native_check(point.u, prepared, guard.weight_total, attempted)
    except FloatingPointError as error:
        work = work.checked(attempted.count)
        return _Outcome(point, RANGE, f"native residual: {error}", work), point, work
    work = work.checked(attempted.count)
    point = replace(point, native=found)
    if found.maximum <= guard.gradient_tol:
        return _Outcome(point, CONVERGED, "", work), point, work
    return None, point, work


# ----------------------------------------------------- forward and one block


def _sweeps(
    b: FloatArray,
    prepared: PreparedWeights,
    start: _Point,
    *,
    target: _Target,
    max_sweeps: int,
) -> _Outcome:
    """Forward sweeps of forced visits from the cold ``start``, all blocks checked.

    With one block this is the direct visit: one sweep, one visit, then the check.
    """
    order = forward_order(prepared.n_blocks)
    point = start
    work = Work().checked(point.passes)
    while True:
        settled, point, work = _settled(prepared, point, work, target)
        if settled is not None:
            return settled
        if work.sweeps + 1 > max_sweeps:
            return _budget_outcome(point, work, max_sweeps, 1)
        work = work.reserved(1)
        attempted = Attempted()
        try:
            alpha = point.alpha
            u_current: FloatArray | None = (
                point.u
                if (
                    _visit is _ACCEPTED_VISIT
                    and _residual is _ACCEPTED_RESIDUAL
                    and lift is _ACCEPTED_LIFT
                )
                else None
            )
            for block in order:
                if u_current is not None:
                    alpha, _ = _visit(
                        b, prepared, alpha, block, attempted, u_current=u_current
                    )
                    u_current = None
                else:
                    alpha, _ = _visit(b, prepared, alpha, block, attempted)
            moved = _evaluate(b, prepared, alpha)
        except FloatingPointError as error:
            return _Outcome(point, RANGE, str(error), work.visited(attempted.count))
        point, work = moved, work.visited(attempted.count).checked(moved.passes)


# ------------------------------------------------------- the forced recurrence


def _forced(
    b: FloatArray,
    prepared: PreparedWeights,
    alpha: Coefficients,
    attempted: Attempted,
    *,
    u_initial: FloatArray | None = None,
) -> Coefficients:
    """``r``: the sum of the corrections of the forced symmetric visits from ``alpha``.

    The visits run on a temporary tuple, which is discarded: nothing is committed.
    """
    temporary = alpha
    r = list(zero_coefficients(prepared.grouping, 1))
    u_current = (
        u_initial
        if (
            _visit is _ACCEPTED_VISIT
            and _residual is _ACCEPTED_RESIDUAL
            and lift is _ACCEPTED_LIFT
        )
        else None
    )
    for block in symmetric_order(prepared.n_blocks):
        if u_current is not None:
            temporary, correction = _visit(
                b, prepared, temporary, block, attempted, u_current=u_current
            )
            u_current = None
        else:
            temporary, correction = _visit(b, prepared, temporary, block, attempted)
        r[block] = checked_sum(r[block], correction, "forced residual")
    return tuple(r)


_ACCEPTED_FORCED = _forced


def _start(
    b: FloatArray,
    prepared: PreparedWeights,
    start: _Point,
    *,
    target: _Target,
    max_sweeps: int,
) -> _Outcome | _Progress:
    """The initial check and, if it fails and the budget admits, the forced ``r``."""
    sweeps, _ = operator_cost(prepared.n_blocks)
    point = start
    work = Work().checked(point.passes)
    settled, point, work = _settled(prepared, point, work, target)
    if settled is not None:
        return settled
    if work.sweeps + sweeps > max_sweeps:
        return _budget_outcome(point, work, max_sweeps, sweeps)
    work = work.reserved(sweeps, 1)
    attempted = Attempted()
    try:
        if _forced is _ACCEPTED_FORCED:
            r = _forced(
                b, prepared, point.alpha, attempted, u_initial=point.u
            )
        else:
            r = _forced(b, prepared, point.alpha, attempted)
        rows = lift(r, prepared.grouping)
        rho = weighted_inner(prepared, rows, rows)
    except FloatingPointError as error:
        return _Outcome(point, RANGE, str(error), work.visited(attempted.count))
    work = work.visited(attempted.count)
    if not _positive_finite(rho):
        return _Outcome(point, BREAKDOWN, RHO, work)
    return _Progress(point, r, r, rho, work)


def _update(
    b: FloatArray, prepared: PreparedWeights, progress: _Progress, attempted: Attempted
) -> tuple[_Point, Coefficients] | None:
    """``v = A p``, ``t``, the reconstructed candidate and ``r - t v``.

    ``None`` when ``den`` is not finite and positive: no candidate is formed then.
    """
    grouping = prepared.grouping
    p_rows = lift(progress.p, grouping)
    _, v, _ = walk(p_rows, prepared, symmetric_order(prepared.n_blocks), attempted)
    den = weighted_inner(prepared, p_rows, lift(v, grouping))
    if not _positive_finite(den):
        return None
    t = checked_quotient(progress.rho, den, "step")
    alpha = added(progress.point.alpha, scaled(progress.p, t))
    r = subtracted(progress.r, scaled(v, t))
    return _evaluate(b, prepared, alpha), r


def _next(
    prepared: PreparedWeights, progress: _Progress, r: Coefficients
) -> tuple[Coefficients, FloatArray] | None:
    """``p = r + (rho_new / rho) p`` and ``rho_new``; ``None`` if ``rho_new`` fails."""
    rows = lift(r, prepared.grouping)
    rho = weighted_inner(prepared, rows, rows)
    if not _positive_finite(rho):
        return None
    beta = checked_quotient(rho, progress.rho, "beta")
    return added(r, scaled(progress.p, beta)), rho


def _iterate(
    b: FloatArray,
    prepared: PreparedWeights,
    progress: _Progress,
    *,
    target: _Target,
    max_sweeps: int,
) -> _Outcome | _Progress:
    """One update: the reserved operator, the committed candidate, its check, ``p``."""
    sweeps, _ = operator_cost(prepared.n_blocks)
    point, work = progress.point, progress.work
    if work.sweeps + sweeps > max_sweeps:
        return _budget_outcome(point, work, max_sweeps, sweeps)
    work = work.reserved(sweeps, 1)
    attempted = Attempted()
    try:
        updated = _update(b, prepared, progress, attempted)
    except FloatingPointError as error:
        return _Outcome(point, RANGE, str(error), work.visited(attempted.count))
    work = work.visited(attempted.count)
    if updated is None:
        return _Outcome(point, BREAKDOWN, DEN, work)
    point, r = updated
    work = work.checked(point.passes)
    settled, point, work = _settled(prepared, point, work, target)
    if settled is not None:
        return settled
    try:
        found = _next(prepared, progress, r)
    except FloatingPointError as error:
        return _Outcome(point, RANGE, str(error), work)
    if found is None:
        return _Outcome(point, BREAKDOWN, RHO, work)
    p, rho = found
    return _Progress(point, r, p, rho, work)


def _conjugate_gradient(
    b: FloatArray,
    prepared: PreparedWeights,
    start: _Point,
    *,
    target: _Target,
    max_sweeps: int,
) -> _Outcome:
    """The forced symmetric conjugate gradient from the cold ``start`` (blocks > 1)."""
    progress = _start(b, prepared, start, target=target, max_sweeps=max_sweeps)
    while not isinstance(progress, _Outcome):
        progress = _iterate(b, prepared, progress, target=target, max_sweeps=max_sweeps)
    return progress


# ---------------------------------------------------------------------- entry


def _right(b: ArrayLike, n_rows: int) -> FloatArray:
    """An owned finite vector of ``n_rows`` doubles as one column, or ``ValueError``."""
    if np.iscomplexobj(b):
        raise ValueError(RIGHT)
    found: FloatArray = np.array(b, dtype=np.float64, copy=True)
    if found.shape != (n_rows,) or not np.isfinite(found).all():
        raise ValueError(RIGHT)
    return found[:, None]


def solve_rhs(
    b: ArrayLike,
    prepared: PreparedWeights,
    *,
    tolerance: float,
    forcing: float,
    guard: RHSGuard,
    max_sweeps: int,
    acceleration: bool,
) -> RHSResult:
    """Solve ``D' C h = D' b`` for ``h`` in the span of ``D``, from a cold start.

    Parameters
    ----------
    b : array_like
        The full right-hand side ``a r``, one finite value per row; zero-curvature rows
        keep their nonzero values.
    prepared : PreparedWeights
        The curvature weights ``C`` (``a c``, or ``a w`` with the expected curvature)
        with their level totals ``d``, every one positive
        (:func:`~powermean.fe.absorb_numeric.prepare_weights`).
    tolerance : float
        The ceiling: the requested maximum of ``|group_sums(u) / d|`` over every
        original block and level; positive and finite; equality passes.
    forcing : float
        The relative factor ``eta`` in ``[0, 0.9999]``: the relative bound is ``eta``
        times the maximum of the cold initial check, compared exactly; equality passes.
    guard : RHSGuard
        ``af``, ``A`` and ``gtol`` of the guarded exception: below the ceiling, a
        maximum at most ``af`` whose native residual is at most ``gtol`` also passes.
    max_sweeps : int
        The budget of sweeps, a positive integer, 1 included.
    acceleration : bool
        The forced symmetric conjugate gradient when ``True`` and there are several
        blocks; forward sweeps when ``False``; the direct visit with one block either
        way.

    Returns
    -------
    RHSResult
        The candidate's ``h``, ``u`` and coefficients, its check and native residual
        (when evaluated there), status, reason, the bounds and guard, and work; an
        unsuccessful status keeps the last committed candidate.

    Raises
    ------
    ValueError
        A malformed right-hand side, weights record, guard or control.
    """
    if not isinstance(prepared, PreparedWeights):
        raise ValueError("prepared must be a PreparedWeights record")
    if not isinstance(guard, RHSGuard):
        raise ValueError("guard must be an RHSGuard record")
    tol = _tolerance(tolerance)
    eta = _forcing(forcing)
    budget = _budget(max_sweeps)
    accelerated = _flag(acceleration)
    right = _right(b, prepared.n_rows)
    start = _evaluate(right, prepared, zero_coefficients(prepared.grouping, 1))
    initial = None if start.check is None else start.check.maximum
    if initial is None:
        failure = start.failure or RANGE
        outcome = _Outcome(start, RANGE, failure, Work().checked(start.passes))
    elif accelerated and prepared.n_blocks > 1:
        target = _Target(tol, eta, initial, guard)
        outcome = _conjugate_gradient(
            right, prepared, start, target=target, max_sweeps=budget
        )
    else:
        target = _Target(tol, eta, initial, guard)
        outcome = _sweeps(right, prepared, start, target=target, max_sweeps=budget)
    point = outcome.point
    return RHSResult(
        h=point.h[:, 0],
        u=point.u[:, 0],
        alpha=tuple(a[:, 0] for a in point.alpha),
        check=point.check,
        native_check=point.native,
        status=outcome.status,
        reason=outcome.reason,
        tolerance=tol,
        forcing=eta,
        initial=initial,
        guard=guard,
        max_sweeps=budget,
        acceleration=accelerated,
        work=outcome.work,
    )
