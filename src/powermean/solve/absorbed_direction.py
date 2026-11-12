"""The fitted-index Newton direction's assembly, and the same-point effect recovery.

Newton in the fitted index: one weighted absorption per iteration, effects recovered
at the end.  Reference levels are set by connected component; redundancies are exact
for one and two blocks, bounded pairwise for three or more.  Internal module:
``powermean`` exports nothing of it.

The direction.  ``X`` holds the retained scaled slopes, ``D`` the original indicators
(never formed), ``b = a r`` and ``C = a c`` (``a w`` with the expected curvature) at the
direction origin, ``A = sum a``.  The accepted absorption of ``X`` with the weights
``C`` gives ``Xt = X - P_C X``; the actual provider solves the Schur system ``(Xt' C Xt
/ A) d_gamma = Xt' b / A`` in K-by-K storage; the forced full-RHS solve of
:mod:`powermean.solve.absorbed_rhs` gives ``h`` with ``D' C h = D' b``.  The full joint
Newton direction in the fitted index is

    d_q = Xt d_gamma + h                      (formed in that order; h when K = 0)

and its two scalars at the origin are those of the full joint system:

    derivative  = v' d_q,            v = (-a r) / A
    decrement2  = d_q' ((C / A) d_q),

both formed from the unhalved direction, never from a slope-only view.  In exact
arithmetic ``d_q`` solves the normal equations of ``[X, D]`` (the Schur algebra); finite
inner tolerances give an approximate direction,
which the actual line search must accept, and no inner success flag is read as the
exact normal equations.  The full-RHS solve receives, besides the absolute tolerance of
the schedule, the forcing factor that its caller issued and the guard of
its exception: the existing ``absorb_tol_final``, the raw-weight total
``A`` of the caller's rows record and the existing ``gradient_tol``, passed explicitly
in one :class:`~powermean.solve.absorbed_rhs.RHSGuard`.  These bounds control the
nuisance residual only, so the approximate slope absorption still contributes ``D' C
Xt d_gamma`` to the joint residual.  Nothing here forms ``D`` or a FE-square matrix.
Each direction request checks and prepares ``C`` once, from its own rows: the record
that its slope absorption forms (the existing ``curvature_totals``, then
``prepare_weights``) is the one its full-RHS solve reads, and without slopes the
full-RHS solve forms it.  Nothing keeps that record; the final work prepares its own.
A direction's slope absorption may start from its evaluator's warm slope state
(:mod:`powermean.solve.absorbed_loss`); the full-RHS solve, the recovery and the final
work start cold.

The automatic start (:meth:`IndexContext.initialized`).  The plain unweighted
``log(mean z)`` in every level of the first block, the other blocks zero, then one
sweep: in the local domain (the first attempt, one block, no slope,
exact PPML or the exact NB2 ``kappa = 0`` alias) the local evaluation of that same sweep
(:mod:`powermean.solve.absorbed_start`), logged as its own record; everywhere else the
family's accepted profiler, logged as a profile call.  No profile record is fabricated
for the local start, nor a local record for the profiler.

Recovery.  The authoritative point is ``(gamma, q)``; its base is
``fl(X gamma + o)`` and its nuisance index ``eta = q - base`` (checked), never an
independently updated state.  :func:`recover` absorbs ``eta`` as one column under the
strictly positive RAW weights ``a``, cold, at the requested final tolerance with the
existing sweep budget and acceleration (a representation recovery: not a nonlinear
profile, and not a C-weighted recovery that would ignore zero-curvature rows), keeps the
returned absorption, and rebuilds ``q_rec`` by the accepted ordered base-plus-block
additions (:func:`~powermean.fe.profile_arithmetic.reconstruct_index`).  It requires
``max |q_rec - q| <= tolerance``, equality passing, with no relative multiplier or
roundoff allowance; this explicit check is stricter than the absorber's own group-mean
check, so a drift of ``eta`` orthogonal to ``D`` cannot pass as a recovery.  The
absorption takes this check and the normalized one as its requirement
(:func:`~powermean.fe.absorb.absorb`): with several blocks the weighted group means do
not bound the pointwise error of the fitted part, so a candidate within the tolerance
whose effects do not yet rebuild ``q`` continues the SAME cold recurrence, inside the
same budget and at the same tolerance, until both reconstructions hold.  An orthogonal
drift is never absorbed: the budget ends that absorption at a candidate whose own check
passed, and the explicit check refuses it as before.  ``q`` is never replaced.  The
record keeps ``eta`` as its own detached read-only copy, complete or partial:
the evidence cannot change after the recovery.
:func:`normalize` then sets, for each connected component and each block ``k >= 2`` in
block order, the component's smallest level code of block ``k`` as its reference: that
level's coefficient is subtracted from the component's block-``k`` levels and added to
its first-block levels (checked arithmetic), every reference and shift recorded, and the
same reconstruction check is repeated.  One block carries its levels unshifted.  No
intercept is added.  With three or more blocks a redundancy that connectivity does not
see may remain: the normalized coefficients are a deterministic representative, with
the indicator rank's bounds recorded, never a claim of unique effects.  Order-scale
conversion is outside this solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.accounting import frozen_index
from powermean.fe.absorb import AbsorptionResult, AbsorptionState, absorb
from powermean.fe.absorb_acceleration import Requirement
from powermean.fe.absorb_numeric import (
    Coefficients,
    PreparedWeights,
    checked_difference,
    checked_sum,
    prepare_weights,
)
from powermean.fe.connectivity import IndicatorRank, components, indicator_rank
from powermean.fe.groups import Grouping
from powermean.fe.profile import RANGE, ProfileState
from powermean.fe.profile_arithmetic import ProfileProblem, reconstruct_index
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_rhs import RHSCall, RHSGuard, RHSResult, Role, solve_rhs
from powermean.solve.absorbed_start import (
    IMPLEMENTATION,
    INITIAL_SWEEPS,
    LocalCall,
    LocalInitializer,
    applies,
    local_start,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.options import Curvature
from powermean.solve.point_loss import PointLoss
from powermean.solve.profiled_linear_loss import Admissions
from powermean.solve.profiled_loss import (
    REFUSED,
    AbsorptionCall,
    ProfileCall,
    ProfileRole,
)
from powermean.systems.profiled import (
    Absorbed,
    PointProfile,
    ProfiledProblem,
    _absorb_slopes,
    _prepare_curvature,
    _prepared,
    curvature_totals,
    profile_point,
)

# The identity of the one column that the recovery absorbs.
ETA = ("eta",)
DIRECTION = "the full direction is not representable in float64"

_ACCEPTED_CURVATURE_TOTALS = curvature_totals
_ACCEPTED_PREPARE_WEIGHTS = prepare_weights
_ACCEPTED_COMPONENTS = components
_ACCEPTED_RECONSTRUCT = reconstruct_index


# --------------------------------------------------------------------- direction


@dataclass(frozen=True, eq=False)
class Direction:
    """The full joint Newton direction at one origin, with its two scalars.

    Attributes
    ----------
    d_gamma : array of float
        The provider's Schur step of the slopes, ``K`` values (none when ``K = 0``).
    d_q : array of float
        ``Xt d_gamma + h`` on every row: the step of the fitted index.
    derivative : float
        ``v' d_q``, the derivative of the averaged criterion along ``d_q``.
    decrement : float
        ``d_q' ((C / A) d_q)``, the squared Newton decrement of the unhalved step.
    tolerance : float
        The inner tolerance of the slope absorption and the full-RHS solve.
    """

    d_gamma: FloatArray
    d_q: FloatArray
    derivative: float
    decrement: float
    tolerance: float

    def __post_init__(self) -> None:
        """Own the arrays."""
        object.__setattr__(self, "d_gamma", immutable_array(self.d_gamma))
        object.__setattr__(self, "d_q", immutable_array(self.d_q))


def complete_direction(
    absorbed: FloatArray | None,
    d_gamma: FloatArray,
    h: FloatArray,
    rows: tuple[FloatArray, FloatArray],
    *,
    tolerance: float,
) -> Direction:
    """``d_q = Xt d_gamma + h`` and its derivative and decrement at the origin.

    ``absorbed`` is ``Xt`` (``None`` without slopes, when ``d_q = h``) and ``rows`` the
    origin's ``(v, C / A)``.  ``FloatingPointError`` when ``d_q`` or a scalar is not
    representable; nothing is clipped.
    """
    v, scaled = rows
    with np.errstate(over="ignore", invalid="ignore"):
        if absorbed is None:
            d_q: FloatArray = np.array(h, dtype=np.float64, copy=True)
        else:
            d_q = absorbed @ d_gamma + h
        derivative = v @ d_q
        decrement = d_q @ (scaled * d_q)
    finite = np.isfinite(d_q).all() and np.isfinite([derivative, decrement]).all()
    if not finite:
        raise FloatingPointError(DIRECTION)
    return Direction(d_gamma, d_q, float(derivative), float(decrement), tolerance)


def nuisance_rhs(
    problem: ProfiledProblem,
    weights: ArrayLike,
    b: ArrayLike,
    *,
    tolerance: float,
    forcing: float,
    guard: RHSGuard,
    max_sweeps: int,
    acceleration: bool,
) -> RHSResult:
    """The forced full-RHS solve ``D' C h = D' b`` with the curvature ``weights``.

    The accepted checked curvature totals first (:func:`~powermean.systems.profiled
    .curvature_totals`: ``FloatingPointError`` for a level of a block without a positive
    total, before any work), then :func:`~powermean.solve.absorbed_rhs.solve_rhs` from a
    cold start with the absolute ``tolerance``, the issued ``forcing`` and the ``guard``
    of the exception; its status, bounds and work are returned as they are.
    """
    if (
        curvature_totals is _ACCEPTED_CURVATURE_TOTALS
        and prepare_weights is _ACCEPTED_PREPARE_WEIGHTS
    ):
        prepared = _prepare_curvature(problem, weights)
    else:
        found = curvature_totals(problem, weights)
        prepared = prepare_weights(problem.grouping, found)
    return solve_rhs(
        b,
        prepared,
        tolerance=tolerance,
        forcing=forcing,
        guard=guard,
        max_sweeps=max_sweeps,
        acceleration=acceleration,
    )


# ----------------------------------------------------------- normalization


_REDUCED_ABSORB = _absorb_slopes
_REDUCED_RHS = solve_rhs
_REDUCED_NUISANCE = nuisance_rhs


@dataclass(frozen=True, eq=False)
class Normalization:
    """Reference levels by connected component, and the normalized coefficients.

    Attributes
    ----------
    effects : tuple of arrays of float
        One vector per block, normalized; read-only and owned.
    references : tuple of arrays of int
        For each block after the first, the reference level code of each component
        (the smallest level code of that block in it).
    shifts : tuple of arrays of float
        For each block after the first, the coefficient of each component's reference,
        subtracted from the block and added to the first block's levels.
    """

    effects: tuple[FloatArray, ...]
    references: tuple[IndexArray, ...]
    shifts: tuple[FloatArray, ...]


@dataclass(frozen=True, eq=False)
class _NormalizationPlan:
    """Precomputed connected-component level memberships and reference codes."""

    first: IndexArray
    later_of: tuple[IndexArray, ...]
    references: tuple[IndexArray, ...]


def _plan_normalization(grouping: Grouping) -> _NormalizationPlan:
    """Compute the component memberships and reference level codes of ``grouping``."""
    parts = components(grouping.blocks)
    first = parts.of_levels[0]
    later_of: list[IndexArray] = []
    references: list[IndexArray] = []
    for k in range(1, len(grouping.blocks)):
        of = parts.of_levels[k]
        count = grouping.blocks[k].n_levels
        refs = np.full(parts.n_components, count, dtype=np.intp)
        np.minimum.at(refs, of, np.arange(count, dtype=np.intp))
        later_of.append(of)
        references.append(frozen_index(refs))
    return _NormalizationPlan(first, tuple(later_of), tuple(references))


def _normalize_with_plan(
    plan: _NormalizationPlan, effects: tuple[FloatArray, ...]
) -> Normalization:
    """Shift each component's later-block references onto its first-block levels."""
    found = [np.array(e, dtype=np.float64, copy=True) for e in effects]
    first = plan.first
    shifts: list[FloatArray] = []
    for k, (of, refs) in enumerate(
        zip(plan.later_of, plan.references, strict=True), start=1
    ):
        shift = found[k][refs]
        found[k] = checked_difference(found[k], shift[of], "normalized coefficients")
        found[0] = checked_sum(found[0], shift[first], "normalized coefficients")
        shifts.append(immutable_array(shift))
    return Normalization(
        tuple(immutable_array(e) for e in found), plan.references, tuple(shifts)
    )


def normalize(grouping: Grouping, effects: tuple[FloatArray, ...]) -> Normalization:
    """Shift each component's later-block references onto its first-block levels.

    Components from the accepted connectivity (``Components.of_levels``); every
    operation checked (``FloatingPointError`` otherwise).  Each row's sum of its levels'
    coefficients is preserved in exact arithmetic.
    """
    return _normalize_with_plan(_plan_normalization(grouping), effects)


_ACCEPTED_NORMALIZE = normalize


# ---------------------------------------------------------------------- recovery


@dataclass(frozen=True, eq=False)
class Recovery:
    """The same-point recovery of the effects of an index point, whatever its outcome.

    Attributes
    ----------
    tolerance : float
        The final absorption tolerance, also the bound of both reconstruction gaps.
    eta : array of float or None
        ``q - base``, a detached read-only copy owned by the record; ``None`` when it
        was not representable.
    absorption : AbsorptionResult or None
        The accepted cold absorption of ``eta`` under the raw weights, as returned.
    effects : tuple of arrays of float or None
        Its coefficients, one vector per block, when it converged.
    gap : float or None
        ``max |q_rec - q|`` of those coefficients.
    normalization : Normalization or None
        The component references and the normalized coefficients.
    normalized_gap : float or None
        ``max |q_rec - q|`` after the normalization.
    rank : IndicatorRank
        The indicator rank of the blocks, exact for one and two, else bounded.
    reason : str
        Empty when every part completed and both gaps are within the tolerance.
    """

    tolerance: float
    eta: FloatArray | None
    absorption: AbsorptionResult | None
    effects: tuple[FloatArray, ...] | None
    gap: float | None
    normalization: Normalization | None
    normalized_gap: float | None
    rank: IndicatorRank
    reason: str

    def __post_init__(self) -> None:
        """Own ``eta``: a detached read-only copy of the same values."""
        if self.eta is not None:
            object.__setattr__(self, "eta", immutable_array(self.eta))

    @property
    def complete(self) -> bool:
        """Whether the effects were recovered and normalized at the same point."""
        return not self.reason


@dataclass(frozen=True, eq=False)
class IndexPoint:
    """One whole point of the fitted-index route: slopes, base, full index, recovery.

    Attributes
    ----------
    theta : array of float
        The slopes ``gamma`` in the coordinates of the sample's scaled design.
    base : array of float
        ``fl(X gamma + o)``, the transformed offset entered once.
    index : array of float
        The authoritative fitted index ``q`` of every row.
    recovered : Recovery or None
        The same-point recovery of the effects, when the final work ran it.

    Notes
    -----
    The common point view of :mod:`powermean.solve.profiled_trigger`: no profile record
    is formed for an index point.  The arrays are owned read-only copies.
    """

    theta: FloatArray
    base: FloatArray
    index: FloatArray
    recovered: Recovery | None = None

    def __post_init__(self) -> None:
        """Own the arrays."""
        for name in ("theta", "base", "index"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


def reconstruct(
    problem: ProfiledProblem, base: FloatArray, effects: tuple[FloatArray, ...]
) -> FloatArray:
    """``((base + alpha_1[c_1]) + alpha_2[c_2]) + ...``, the accepted ordered sums."""
    return reconstruct_index(_prepared(problem, base), effects)


def _gap(rebuilt: FloatArray, index: FloatArray) -> float:
    """``max |rebuilt - index|``, the difference checked."""
    return float(np.max(np.abs(checked_difference(rebuilt, index, "recovery gap"))))


_VerifiedRecovery = tuple[tuple[FloatArray, ...], float, Normalization, float]


def _rebuilds(
    profile: ProfileProblem,
    index: FloatArray,
    tolerance: float,
    *,
    verified: list[_VerifiedRecovery | None] | None = None,
) -> Requirement:
    """The recovery's two reconstruction checks, as its absorption's requirement.

    Met by a candidate whose effects rebuild ``q`` within ``tolerance`` in the accepted
    order, before and after the normalization (the checks of :func:`recover`, same
    arithmetic).  A reconstruction that is not representable also ends the absorption:
    no later candidate is sought, and the explicit checks then refuse with the error,
    the absorption and the effects kept, as they always did.
    """
    plan: _NormalizationPlan | None = None

    def met(coefficients: Coefficients) -> bool:
        nonlocal plan
        if verified is not None:
            verified[0] = None
        effects = tuple(a[:, 0] for a in coefficients)
        try:
            gap = _gap(reconstruct_index(profile, effects), index)
            if gap > tolerance:
                return False
            if (
                normalize is _ACCEPTED_NORMALIZE
                and components is _ACCEPTED_COMPONENTS
            ):
                if plan is None:
                    plan = _plan_normalization(profile.grouping)
                normalization = _normalize_with_plan(plan, effects)
            else:
                normalization = normalize(profile.grouping, effects)
            after = _gap(reconstruct_index(profile, normalization.effects), index)
        except FloatingPointError:
            return True
        if after <= tolerance:
            if verified is not None:
                verified[0] = (effects, gap, normalization, after)
            return True
        return False

    return met


def _absorbed(
    profile: ProfileProblem,
    eta: FloatArray,
    index: FloatArray,
    *,
    tolerance: float,
    max_sweeps: int,
    acceleration: bool,
    verified: list[_VerifiedRecovery | None] | None = None,
) -> AbsorptionResult:
    """The cold absorption of ``eta`` under the raw weights at ``tolerance``.

    The absorption's requirement is :func:`_rebuilds`: its weighted group means do not
    bound the pointwise reconstruction gap with several blocks.
    """
    return absorb(
        eta[:, None],
        profile.prepared,
        ETA,
        tolerance=tolerance,
        max_sweeps=max_sweeps,
        acceleration=acceleration,
        requirement=_rebuilds(profile, index, tolerance, verified=verified),
    )


class _Partial:
    """The parts of a recovery as they complete, frozen into :class:`Recovery`."""

    def __init__(self, tolerance: float, rank: IndicatorRank) -> None:
        self.tolerance = tolerance
        self.rank = rank
        self.eta: FloatArray | None = None
        self.absorption: AbsorptionResult | None = None
        self.effects: tuple[FloatArray, ...] | None = None
        self.gap: float | None = None
        self.normalization: Normalization | None = None
        self.after: float | None = None

    def record(self, reason: str) -> Recovery:
        """The record of the parts so far, with the reason of the first failure."""
        return Recovery(
            self.tolerance,
            self.eta,
            self.absorption,
            self.effects,
            self.gap,
            self.normalization,
            self.after,
            self.rank,
            reason,
        )


def recover(
    problem: ProfiledProblem,
    point: IndexPoint,
    *,
    tolerance: float,
    max_sweeps: int,
    acceleration: bool,
) -> Recovery:
    """Recover and normalize the effects of ``point`` at the same ``q``.

    The one cold absorption of ``eta`` continues while its candidate, within
    ``tolerance`` in the absorber's own check, does not yet rebuild ``q`` (module
    docstring); both reconstruction checks are then made explicitly, unchanged.  Every
    part is kept as far as it completed; ``reason`` names the first that failed: a
    range refusal, an absorption that did not converge, or a reconstruction gap above
    ``tolerance`` before or after the normalization.  ``q`` is never replaced.
    """
    parts = _Partial(tolerance, indicator_rank(problem.grouping.blocks))
    verified: list[_VerifiedRecovery | None] = [None]
    try:
        parts.eta = checked_difference(point.index, point.base, "nuisance index")
        profile = _prepared(problem, point.base)
        if _absorbed is _ACCEPTED_ABSORBED:
            absorption = _absorbed(
                profile,
                parts.eta,
                point.index,
                tolerance=tolerance,
                max_sweeps=max_sweeps,
                acceleration=acceleration,
                verified=verified,
            )
        else:
            absorption = _absorbed(
                profile,
                parts.eta,
                point.index,
                tolerance=tolerance,
                max_sweeps=max_sweeps,
                acceleration=acceleration,
            )
        parts.absorption = absorption
        if not absorption.converged:
            return parts.record(f"recovery: {absorption.reason}")
        effects = tuple(a[:, 0] for a in absorption.state.coefficients)
        parts.effects = effects
        cached = verified[0]
        if (
            cached is not None
            and reconstruct_index is _ACCEPTED_RECONSTRUCT
            and normalize is _ACCEPTED_NORMALIZE
            and len(cached[0]) == len(effects)
            and all(
                np.array_equal(a, b) for a, b in zip(cached[0], effects, strict=True)
            )
        ):
            parts.gap = cached[1]
            parts.normalization = cached[2]
            parts.after = cached[3]
            return parts.record("")
        gap = _gap(reconstruct_index(profile, effects), point.index)
        parts.gap = gap
        if gap > tolerance:
            return parts.record(_beyond("recovered", gap, tolerance))
        normalization = normalize(problem.grouping, effects)
        parts.normalization = normalization
        rebuilt = reconstruct_index(profile, normalization.effects)
        after = _gap(rebuilt, point.index)
        parts.after = after
    except FloatingPointError as error:
        return parts.record(f"recovery: {error}")
    if after > tolerance:
        return parts.record(_beyond("normalized", after, tolerance))
    return parts.record("")


_ACCEPTED_ABSORBED = _absorbed


def _beyond(what: str, gap: float, tolerance: float) -> str:
    """The reason of a reconstruction gap above the tolerance."""
    return (
        f"recovery: the {what} effects rebuild the index within {gap:.3e}, above "
        f"the final tolerance {tolerance:.3e}"
    )


# ------------------------------------------------------------- the solve's work

# Where a logged inner call runs: the attempt, the accepted steps before it, the role,
# the tolerance and the accepted score that set it (``None`` when none did).
CallSite = tuple[int, int, Role, float, float | None]


@dataclass(frozen=True)
class DirectionCall:
    """One completed full direction: its tolerance and its scalars at the origin.

    ``score`` is ``max |G|`` at the origin; ``factorized`` whether the provider
    factorized a slope system for it.
    """

    attempt: int
    iteration: int
    tolerance: float
    derivative: float
    decrement: float
    score: float
    factorized: bool


@dataclass(frozen=True, eq=False)
class IndexWork:
    """Every initializer, absorption, RHS solve and direction of a solve.

    ``profiles`` holds the actual calls of the accepted profiler only, and
    ``initializers`` the local automatic starts only.
    """

    profiles: tuple[ProfileCall, ...]
    absorptions: tuple[AbsorptionCall, ...]
    rhs: tuple[RHSCall, ...]
    directions: tuple[DirectionCall, ...]
    initializers: tuple[LocalCall, ...]


class IndexLog:
    """The calls of one solve as they happen, frozen into :class:`IndexWork`."""

    def __init__(self) -> None:
        self.profiles: list[ProfileCall] = []
        self.absorptions: list[AbsorptionCall] = []
        self.rhs: list[RHSCall] = []
        self.directions: list[DirectionCall] = []
        self.initializers: list[LocalCall] = []

    def frozen(self) -> IndexWork:
        """The calls so far, as an immutable record."""
        return IndexWork(
            tuple(self.profiles),
            tuple(self.absorptions),
            tuple(self.rhs),
            tuple(self.directions),
            tuple(self.initializers),
        )


@dataclass(frozen=True, eq=False)
class IndexStart:
    """The caller's full start of the fitted-index route: slopes and effects.

    ``slopes`` holds one finite value per slope column, in the coordinates of the
    sample's scaled design (none without slopes); ``effects`` is the coefficient
    witness, a :class:`~powermean.fe.profile.ProfileState` bound to the sample's
    grouping.  The start's index is reconstructed from them by the accepted ordered
    additions, without a profile; no index is admitted as given.
    """

    slopes: FloatArray
    effects: ProfileState

    def __post_init__(self) -> None:
        """Check and own the slopes; the effects must be a profile state."""
        values: FloatArray = np.array(self.slopes, dtype=np.float64, copy=True)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("slopes must be a finite vector, possibly empty")
        if not isinstance(self.effects, ProfileState):
            raise ValueError("effects must be a ProfileState of the sample's grouping")
        object.__setattr__(self, "slopes", immutable_array(values))


_UNCOUNTED = (None, None, None, None, None)


@dataclass(frozen=True, eq=False)
class IndexContext:
    """The fixed parts of a solve that its attempts share, and its logged inner calls.

    ``problem`` the accepted problem of the sample; ``controls`` the absorbed options,
    read as they are; ``curvature`` that of the direction; ``positions`` the original
    rows, for an indication's records; ``log`` the calls of the solve; ``point`` its
    pointwise loss of zero offset, charging ``admissions``, the solve's one budget;
    ``gradient_tol`` the native gradient tolerance of the solver options, read by the
    guard of the full-RHS exception.  Every initializer, slope
    absorption and full-RHS call is logged with its own status and work, a call refused
    before the routine counted anything as ``refused`` without work (unavailable, not
    zero), and raised again; a full-RHS call also logs the forcing it received, its
    cold initial maximum, whether the returned candidate met each bound,
    and its guard with the exception's evidence.  A direction's slope
    absorption (:meth:`_absorb`) returns its fresh record of the curvature weights for
    that direction's full-RHS solve (:meth:`_nuisance`) only, and starts from the warm
    slope state that the direction's evaluator hands it, if any; :meth:`absorb`, the
    final work's, stays cold.
    """

    problem: ProfiledProblem
    controls: AbsorbedOptions
    curvature: Curvature
    positions: IndexArray
    log: IndexLog
    point: PointLoss
    admissions: Admissions
    gradient_tol: float

    def initialized(
        self, attempt: int, coef: FloatArray
    ) -> PointProfile | LocalInitializer:
        """The automatic start: ``log(mean z)`` in block 1, then one logged sweep.

        The plain unweighted mean of ``z`` over the final rows, in every level of the
        first block, the other blocks zero (a level that is not finite refuses before
        any call); then, in the local domain, the local evaluation of
        the one sweep (:meth:`_local`), and otherwise the family's accepted profile with
        one sweep at ``profile_tol``.  A budget status is an initializer, not a
        failure; a range refusal stops the attempt (``FloatingPointError``).
        """
        problem, controls = self.problem, self.controls
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            level = float(np.log(np.mean(problem.z)))
        if not math.isfinite(level):
            raise FloatingPointError("the automatic start has no finite level")
        levels = [block.n_levels for block in problem.grouping.blocks]
        coefficients = [np.full(levels[0], level)]
        coefficients += [np.zeros(count) for count in levels[1:]]
        state = ProfileState(problem.grouping, tuple(coefficients))
        if applies(problem, attempt):
            return self._local(attempt, coef, state)
        role: ProfileRole = "initial"
        found = (attempt, 0, role, controls.profile_tol, 1)
        try:
            point = profile_point(
                problem, coef, tolerance=controls.profile_tol, max_sweeps=1, state=state
            )
        except FloatingPointError as error:
            refused = ProfileCall(*found, REFUSED, str(error), None, None, None)
            self.log.profiles.append(refused)
            raise
        result = point.result
        call = ProfileCall(
            *found,
            result.status,
            result.reason,
            result.work,
            point.scalar,
            result.maximum,
        )
        self.log.profiles.append(call)
        if result.status == RANGE:
            raise FloatingPointError(f"initializer: {result.reason}")
        return point

    def _local(
        self, attempt: int, coef: FloatArray, state: ProfileState
    ) -> LocalInitializer:
        """The local automatic start, logged as its own call.

        A refusal before its first fresh check is logged ``refused`` without work and
        raised again; a ``range`` result is logged with its work, then stops the attempt
        (``FloatingPointError``); a converged or budget result is returned.
        """
        tolerance = self.controls.profile_tol
        variance = self.problem.variance
        role: Literal["initial"] = "initial"
        found = (attempt, 0, role, tolerance, INITIAL_SWEEPS, variance)
        try:
            result = local_start(self.problem, coef, state, tolerance=tolerance)
        except FloatingPointError as error:
            refused = LocalCall(
                *found, IMPLEMENTATION, REFUSED, str(error), None, None, None
            )
            self.log.initializers.append(refused)
            raise
        initial = None if result.initial is None else result.initial.maximum
        call = LocalCall(
            *found,
            result.implementation,
            result.status,
            result.reason,
            result.work,
            initial,
            result.maximum,
        )
        self.log.initializers.append(call)
        if result.status == RANGE:
            raise FloatingPointError(f"initializer: {result.reason}")
        return result

    def absorb(self, site: CallSite, weights: FloatArray) -> Absorbed:
        """One logged slope absorption with ``weights``, whatever its status."""
        absorbed, _ = self._absorb(site, weights)
        return absorbed

    def _absorb(
        self,
        site: CallSite,
        weights: FloatArray,
        *,
        state: AbsorptionState | None = None,
        index: FloatArray | None = None,
    ) -> tuple[Absorbed, PreparedWeights]:
        """:meth:`absorb`, with the fresh record of ``weights`` that it absorbed with.

        The record is prepared inside the logged refusal, by the existing chain of
        :func:`~powermean.systems.profiled.absorb_slopes` on this call's ``weights``,
        and returned for the full-RHS solve of the same direction (:meth:`_nuisance`);
        nothing keeps it.  ``state`` is the warm slope state of the direction's
        evaluator (:class:`~powermean.solve.absorbed_loss.IndexEvaluator`), the start
        of the absorption with that fresh record; cold when omitted, as :meth:`absorb`.
        """
        limits = (self.controls.absorb_max_sweeps, self.controls.absorb_acceleration)
        try:
            reduced = self.problem.__dict__.get("_reduced_execution")
            direct = None
            if (reduced is not None and index is not None
                    and _absorb_slopes is _REDUCED_ABSORB):
                direct = reduced.project(
                    weights, index, tolerance=site[3], acceleration=limits[1],
                    final_observed=site[2] == "final" and self.curvature == "observed",
                )
            if direct is not None:
                absorbed, prepared = direct
            else:
                absorbed, prepared = _absorb_slopes(
                    self.problem,
                    weights,
                    tolerance=site[3],
                    max_sweeps=limits[0],
                    acceleration=limits[1],
                    state=state,
                )
        except FloatingPointError as error:
            refused = AbsorptionCall(*site, *limits, REFUSED, str(error), *_UNCOUNTED)
            self.log.absorptions.append(refused)
            raise
        result = absorbed.result
        counts = (result.sweeps, result.visits, result.operators)
        self.log.absorptions.append(
            AbsorptionCall(
                *site,
                *limits,
                result.status,
                result.reason,
                *counts,
                result.diagnostic_passes,
                result.maximum,
                absorbed.failure,
            )
        )
        return absorbed, prepared

    def nuisance(
        self,
        site: CallSite,
        rows: tuple[FloatArray, FloatArray],
        forcing: float,
        total: float,
    ) -> RHSResult:
        """One logged full-RHS solve of ``rows = (C, b)`` with ``forcing``, as it ends.

        ``forcing`` is the factor its caller issued; the absolute
        tolerance is the site's; ``total`` is the raw-weight total ``A`` that the
        caller's rows record formed in the native score arithmetic, which the guard of
        the exception takes with the existing ``absorb_tol_final`` and ``gradient_tol``.
        """
        return self._nuisance(site, rows, forcing, total, prepared=None)

    def _nuisance(
        self,
        site: CallSite,
        rows: tuple[FloatArray, FloatArray],
        forcing: float,
        total: float,
        *,
        prepared: PreparedWeights | None,
    ) -> RHSResult:
        """:meth:`nuisance`, on the record of the same direction's slope absorption.

        Without ``prepared`` the curvature ``rows[0]`` is checked and prepared inside
        the logged refusal by :func:`nuisance_rhs`, as :meth:`nuisance` always does.
        ``prepared`` is the record that :meth:`_absorb` has just formed from the same
        ``C`` in this direction: the solve runs on it, inside the same refusal, and
        ``rows[0]`` is not read again.  The guard and the log are those of
        :meth:`nuisance`.
        """
        limits = (self.controls.absorb_max_sweeps, self.controls.absorb_acceleration)
        guard = RHSGuard(self.controls.absorb_tol_final, total, self.gradient_tol)
        try:
            reduced = self.problem.__dict__.get("_reduced_execution")
            direct = None
            if (reduced is not None and prepared is not None
                    and solve_rhs is _REDUCED_RHS
                    and nuisance_rhs is _REDUCED_NUISANCE):
                direct = reduced.nuisance(
                    rows[1], prepared, target=(site[3], forcing, guard), limits=limits
                )
            if direct is not None:
                result = direct
            elif prepared is None:
                result = nuisance_rhs(
                    self.problem,
                    rows[0],
                    rows[1],
                    tolerance=site[3],
                    forcing=forcing,
                    guard=guard,
                    max_sweeps=limits[0],
                    acceleration=limits[1],
                )
            else:
                result = solve_rhs(
                    rows[1],
                    prepared,
                    tolerance=site[3],
                    forcing=forcing,
                    guard=guard,
                    max_sweeps=limits[0],
                    acceleration=limits[1],
                )
        except FloatingPointError as error:
            refused = RHSCall(
                *site,
                *limits,
                REFUSED,
                str(error),
                *_UNCOUNTED,
                forcing=forcing,
                initial=None,
                absolute=None,
                relative=None,
                guard=guard,
                final_accuracy=None,
                sigma=None,
                native=None,
            )
            self.log.rhs.append(refused)
            raise
        work = result.work
        counts = (work.sweeps, work.visits, work.operators, work.diagnostic_passes)
        self.log.rhs.append(
            RHSCall(
                *site,
                *limits,
                result.status,
                result.reason,
                *counts,
                result.maximum,
                forcing=result.forcing,
                initial=result.initial,
                absolute=result.absolute,
                relative=result.relative,
                guard=result.guard,
                final_accuracy=result.final_accuracy,
                sigma=result.sigma,
                native=result.native,
            )
        )
        return result
