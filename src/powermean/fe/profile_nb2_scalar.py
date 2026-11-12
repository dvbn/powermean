"""Scalar NB2 level equations: analytic brackets, original-index evaluations, one step.

For NB2 each level solves a scalar monotone equation by a safeguarded Newton
iteration vectorized over levels, one safeguarded Newton step per block visit, at
``profile_tol`` 2e-11.  This module holds the scalar primitives of one block visit
only.  The state,
the ordered sweeps, the budget, the work and the statuses are
:mod:`powermean.fe.profile_nb2`; the shared checked problem, the ordered
reconstruction and the fresh equations are :mod:`powermean.fe.profile_arithmetic`,
reused and not copied.

Equation of one level ``g`` of block ``k``, with the other blocks held
fixed, ``v_i = b_i + sum_{j != k} alpha_j[c_j(i)]``, ``q_i = v_i + alpha``,
``m_i = exp(q_i)`` and ``kappa > 0``:

    F_g(alpha) = sum_{i in g} a_i (Z_i - m_i) / (1 + kappa m_i)
    C_g(alpha) = -F_g'(alpha) = sum_{i in g} a_i m_i (1 + kappa Z_i) / (1 + kappa m_i)^2

``F_g`` is the raw-weight sum of the unchanged NB2 score ``r_i`` of
:func:`~powermean.model.rows.evaluate_rows`, through the guarded
:func:`~powermean.fe.profile_arithmetic.active_scores`, the product and quotient
guards and the row-order level sums, so that ``e_g = F_g / d_g`` is the fresh-check
equation of that block, term for term (:func:`level_equations`).  ``C_g`` sums the
OBSERVED curvature ``c_i`` of the same kernel, never the expected ``m/(1 + kappa m)``
and never a floor: at a zero outcome it is ``m/(1 + kappa m)^2 > 0``, not Gamma's
structural zero (:func:`level_curvatures`).  With a positive level weight and finite
indices ``C_g > 0`` in exact arithmetic, and with positive weighted outcome ``F_g``
crosses zero exactly once; a represented ``C_g`` that is not positive (a genuine
final underflow of the kernel) selects the midpoint below and is never divided by.

Bracket.  With ``zbar_g = sum a Z / sum a > 0`` and the extrema of ``v``
over the rows of positive weight of the level, the root lies in

    [L_g, U_g] = [log zbar_g - max v, log zbar_g - min v],

since ``sum(a)(zbar - m_max)/(1 + kappa m_max) <= F <= sum(a)(zbar - m_min)/(1 + kappa
m_min)`` by the monotonicity of ``1/(1 + kappa m)`` and of ``m/(1 + kappa m)``.
``log zbar_g`` is formed from the weighted log sums (the maximum-shift
reduction, its vanishing normalized tails allowed), never from a sum ``a Z`` that can
overflow; the extrema use the active rows only; ``v`` is the ORIGINAL ordered
reconstruction with the block's coefficients replaced by zeros
(:func:`index_without`), and every trial score is evaluated at the ordered
reconstruction of trial coefficients, never at a separately drifting ``v + alpha``.
Equal ``v`` gives the closed-form level solution ``L = U`` for every ``kappa``.  The
algebra supplies the bracket, not a floating error bound: the endpoint signs are
checked freshly, ``F(L) >= 0`` and ``F(U) <= 0``, and a wrong rounded sign refuses
(``FloatingPointError``) without expansion, clipping, a changed sample or a new
tolerance; a degenerate bracket's sole point must meet the unchanged equation
tolerance or refuses for lack of interior progress (:func:`endpoint_check`).

Retention (the bracket is retained only while the scalar problem is unchanged).
Within one call the base, outcomes, weights and ``kappa`` are fixed by the problem, so
the scalar function of a level of block ``k`` is fixed by the ORDERED OTHER-BLOCK
INPUTS of its rows, ``alpha_j[c_j(i)]`` for every ``j != k`` in block order
(:func:`other_inputs`): bit-for-bit equal inputs give the same ordered reconstruction
of every row for every trial ``alpha``, hence the same rounded score.  Equal rounded
``v`` does not: with ``A = 2**53``, other coefficients ``(A, -A)`` and ``(A + 2, -A -
2)`` both give ``v = (0, 2)`` on base ``(0, 2)``, while a middle coefficient 1 gives
the full indexes ``(0, 4)`` and ``(2, 2)`` and scores of opposite signs.  A stored
bracket therefore carries these inputs and a level keeps it only while they are
bitwise unchanged on every one of its active rows (:func:`retained_levels`).

One safeguarded update per unfinished level.  A finished level
(``|e_g| <= profile_tol``) needs none.  An evaluated point inside the bracket
tightens it by its fresh score, positive ``F`` raising ``L`` and negative lowering
``U``, never widening (:func:`tightened`).  A coefficient inside or at an endpoint of
a nondegenerate bracket proposes ``alpha + F/C``, taken only if representable and
strictly inside the tightened bracket, else the checked finite interior midpoint
``L + (U - L)/2``; the quotient is formed on those levels only where ``C > 0``, so no
other level, masked out, outside or of curvature that is not positive, is ever
divided.  A coefficient outside moves to the nearer verified endpoint as the visit's
one update; no representable interior proposal or midpoint refuses as stagnation,
never as convergence by interval width (:func:`safeguarded_update`).  Nothing here
loops, retries, caches or chooses a control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe.absorb_numeric import checked_difference
from powermean.fe.groups import Block
from powermean.fe.profile_arithmetic import (
    _LEVEL_LOG_SUMS,
    BoolArray,
    Coefficients,
    ProfileProblem,
    _cached_rows,
    _detached_float,
    _immutable_backing,
    _representable,
    active_scores,
    level_log_sums,
    reconstruct_index,
    required_product,
    required_quotient,
)
from powermean.fe.reductions import group_sum, immutable_array
from powermean.model.rows import evaluate_rows
from powermean.model.variance import FloatArray

WRONG_SIGN = "a bracket endpoint has the wrong rounded sign"
NO_PROGRESS = (
    "a degenerate bracket's sole point fails the equation: no interior progress"
)
STAGNATION = "no representable interior proposal or midpoint: stagnation"


# ------------------------------------------------------------ coefficients, v


def replaced_block(
    coefficients: Coefficients, block: int, values: FloatArray
) -> Coefficients:
    """Fresh coefficient tuple with the vector of ``block`` replaced by ``values``."""
    found = list(coefficients)
    found[block] = np.array(values, dtype=np.float64, copy=True)
    return tuple(found)


def index_without(
    problem: ProfileProblem, coefficients: Coefficients, block: int
) -> FloatArray:
    """``v``: the ordered reconstruction with the coefficients of ``block`` zeroed.

    The same gathers and sums as the authoritative index, with an exact zero in the
    place of the block, so that ``v + alpha`` is the index of the level in real
    arithmetic; it serves the analytic endpoints only, never a score and never the
    identity of a stored bracket (:func:`other_inputs`).
    """
    zeros = np.zeros_like(coefficients[block])
    return reconstruct_index(problem, replaced_block(coefficients, block, zeros))


def other_inputs(
    problem: ProfileProblem, coefficients: Coefficients, block: int
) -> tuple[FloatArray, ...]:
    """The ordered other-block inputs of every row: ``alpha_j[c_j]`` for ``j != block``.

    One gathered vector per other block, in block order: the values that the ordered
    reconstruction adds to the base around the block's own coefficient.  They bind a
    stored bracket to its scalar problem (:func:`retained_levels`).
    """
    return tuple(
        immutable_array(alpha[encoded.codes])
        for j, (encoded, alpha) in enumerate(
            zip(problem.grouping.blocks, coefficients, strict=True)
        )
        if j != block
    )


# ---------------------------------------------------------------- the bracket


def level_log_means(problem: ProfileProblem, block: int) -> FloatArray:
    """``log zbar_g = LSE(log a + log Z) - LSE(log a)`` per level, each finite.

    Absent terms (a zero weight in both sums, a zero outcome in the first) are
    negative infinity; the accepted reduction
    :func:`~powermean.fe.profile_arithmetic.level_log_sums` refuses a level without
    a finite term, which the checked problem's support excludes.
    """
    encoded = problem.grouping.blocks[block]
    plain = (
        type(problem) is ProfileProblem
        and level_log_sums is _LEVEL_LOG_SUMS
        and type(encoded) is Block
        and _immutable_backing(encoded.codes)
    )
    if plain:
        cached: FloatArray | None = problem._nb2_means.get(block)
        if cached is not None:
            return _detached_float(cached)
        log_wz = problem._log_wz
    else:
        log_wz = problem.log_weights + problem.log_outcomes
    numerator = level_log_sums(log_wz, encoded.codes, encoded.n_levels)
    denominator = level_log_sums(problem.log_weights, encoded.codes, encoded.n_levels)
    found = checked_difference(numerator, denominator, "log level mean")
    if plain:
        owned = immutable_array(found)
        problem._nb2_means[block] = owned
        return _detached_float(owned)
    return found


def level_extrema(
    values: FloatArray, problem: ProfileProblem, block: int
) -> tuple[FloatArray, FloatArray]:
    """``min v`` and ``max v`` over the rows of positive weight of each level."""
    encoded = problem.grouping.blocks[block]
    plain_problem = type(problem) is ProfileProblem
    all_active = plain_problem and problem._all_active
    active = problem._active_mask if plain_problem else problem.active
    codes = encoded.codes if all_active else encoded.codes[active]
    active_values = values if all_active else values[active]
    low = np.full(encoded.n_levels, np.inf)
    high = np.full(encoded.n_levels, -np.inf)
    np.minimum.at(low, codes, active_values)
    np.maximum.at(high, codes, active_values)
    return low, high


def analytic_bracket(
    problem: ProfileProblem, without: FloatArray, block: int
) -> tuple[FloatArray, FloatArray]:
    """``[L_g, U_g] = [log zbar_g - max v, log zbar_g - min v]`` per level, checked.

    ``without`` is :func:`index_without` at the current coefficients.  Each endpoint
    is a range-checked difference (``FloatingPointError`` on overflow or an invalid
    operation); ``L_g <= U_g`` and ``L_g == U_g`` exactly where ``v`` is constant on
    the level's active rows.
    """
    low, high = level_extrema(without, problem, block)
    log_means = level_log_means(problem, block)
    lower = checked_difference(log_means, high, "lower endpoint")
    upper = checked_difference(log_means, low, "upper endpoint")
    return lower, upper


@dataclass(frozen=True, eq=False)
class Bracket:
    """The sign-checked bracket of every level of one block, with its identity.

    Attributes
    ----------
    lower, upper : array of float
        ``L_g`` and ``U_g`` per level, ``nan`` where no bracket is held (a level
        finished when the block's brackets were built), read-only and owned.
    context : tuple of arrays of float
        The ordered other-block inputs of every row (:func:`other_inputs`) under
        which the brackets hold: a level whose active rows keep them bit for bit is
        the same scalar problem and keeps its tightened bracket; any other change
        rebuilds it.  Read-only and owned.
    """

    lower: FloatArray
    upper: FloatArray
    context: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        """Own the arrays; the endpoints are ordered where present."""
        lower, upper = immutable_array(self.lower), immutable_array(self.upper)
        present = np.isfinite(lower)
        if (
            lower.shape != upper.shape
            or not np.array_equal(present, np.isfinite(upper))
            or (lower[present] > upper[present]).any()
        ):
            raise ValueError("lower and upper must be ordered where present")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self, "context", tuple(immutable_array(given) for given in self.context)
        )


def retained_levels(
    stored: Bracket | None,
    context: tuple[FloatArray, ...],
    problem: ProfileProblem,
    block: int,
) -> BoolArray:
    """The levels whose stored nondegenerate bracket is the unchanged scalar problem.

    A level is retained when a bracket is held for it, the ordered other-block inputs
    ``context`` (:func:`other_inputs` at the current coefficients) equal the stored
    ones bit for bit on every one of its rows of positive weight (the IEEE bit
    patterns, so that a changed sign of zero rebuilds too), and the bracket is not
    degenerate (a degenerate level is checked afresh at its sole point).  Equal
    rounded sums of other inputs do not retain: stale signs or bounds are never kept
    across changed inputs.
    """
    encoded = problem.grouping.blocks[block]
    if stored is None:
        return np.zeros(encoded.n_levels, dtype=bool)
    changed = np.zeros(problem.n_rows, dtype=bool)
    for before, now in zip(stored.context, context, strict=True):
        changed |= before.view(np.uint64) != now.view(np.uint64)
    plain_problem = type(problem) is ProfileProblem
    if not (plain_problem and problem._all_active):
        changed &= problem._active_mask if plain_problem else problem.active
    moved = np.bincount(encoded.codes[changed], minlength=encoded.n_levels) > 0
    present = np.isfinite(stored.lower)
    found: BoolArray = present & ~moved & (stored.lower < stored.upper)
    return found


# ------------------------------------------------------------- evaluations


def level_equations(
    problem: ProfileProblem, q: FloatArray, block: int
) -> tuple[FloatArray, FloatArray]:
    """``F_g = sum a_i r_i`` and ``e_g = F_g / d_g`` per level of ``block`` at ``q``.

    The arithmetic of :func:`~powermean.fe.profile_arithmetic.fresh_equations` for
    one block, through the same accepted guards (a lost score, weighted score or
    level equation refuses with ``FloatingPointError``) and the same row-order
    level sums, so that ``e_g`` is the fresh-check equation of the level, term for
    term, at the ORIGINAL ordered index ``q``.
    """
    plain_problem = type(problem) is ProfileProblem
    all_active = plain_problem and problem._all_active
    active = problem._active_mask if plain_problem else problem.active
    encoded = problem.grouping.blocks[block]
    weights = (
        problem.prepared.weights if all_active else problem.prepared.weights[active]
    )
    products = required_product(weights, active_scores(problem, q))
    codes = encoded.codes if all_active else encoded.codes[active]
    sums = group_sum(products, codes, encoded.n_levels)
    return sums, required_quotient(sums, problem.prepared.totals[block])


def level_curvatures(problem: ProfileProblem, q: FloatArray, block: int) -> FloatArray:
    """``C_g = sum a_i c_i`` with the OBSERVED curvature of the unchanged row kernel.

    ``c_i = m (1 + kappa Z) / (1 + kappa m)^2``, the exact negative derivative of the
    score, ``m / (1 + kappa m)^2`` at a zero outcome; the products refuse an overflow
    or an invalid operation and allow a final underflow, which the safeguard handles
    by never dividing by a curvature that is not positive.  No floor, no expected
    information, no Gamma zero.
    """
    encoded = problem.grouping.blocks[block]
    plain_problem = type(problem) is ProfileProblem
    if (
        plain_problem
        and problem._all_active
        and type(q) is np.ndarray
        and q.dtype == np.float64
        and q.shape == (problem.n_rows,)
    ):
        rows = _cached_rows(problem, q)
        weights = problem.prepared.weights
        codes = encoded.codes
    else:
        active = problem._active_mask if plain_problem else problem.active
        rows = evaluate_rows(problem.z[active], q[active], variance=problem.variance)
        weights = problem.prepared.weights[active]
        codes = encoded.codes[active]
    products = _representable(
        lambda: np.multiply(weights, rows.observed_curvature), "weighted curvature"
    )
    return group_sum(products, codes, encoded.n_levels)


def trial_equations(
    problem: ProfileProblem,
    coefficients: Coefficients,
    block: int,
    values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """``F_g`` and ``e_g`` at the ordered reconstruction of trial coefficients.

    ``values`` replaces the block's vector; the other blocks keep their current
    coefficients, so every row is evaluated at the index the trial state defines.
    """
    index = reconstruct_index(problem, replaced_block(coefficients, block, values))
    return level_equations(problem, index, block)


# ------------------------------------------------------- checks and the step


def endpoint_check(
    lower_sums: FloatArray,
    upper_sums: FloatArray,
    lower_errors: FloatArray,
    *,
    degenerate: BoolArray,
    tolerance: float,
    mask: BoolArray,
) -> None:
    """The fresh signs of a rebuilt bracket, or the sole point of a degenerate one.

    On the levels of ``mask``: ``F(L) >= 0`` and ``F(U) <= 0`` where the bracket is
    not degenerate, else ``FloatingPointError`` (``WRONG_SIGN``); where it is
    degenerate, ``|e(L)| <= tolerance``, else ``FloatingPointError``
    (``NO_PROGRESS``).  Nothing is expanded, clipped or relaxed.
    """
    wrong = mask & ~degenerate & ((lower_sums < 0) | (upper_sums > 0))
    if wrong.any():
        raise FloatingPointError(WRONG_SIGN)
    failing = mask & degenerate & (np.abs(lower_errors) > tolerance)
    if failing.any():
        raise FloatingPointError(NO_PROGRESS)


def tightened(
    lower: FloatArray,
    upper: FloatArray,
    point: FloatArray,
    sums: FloatArray,
    mask: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    """The bracket tightened by the fresh score of an evaluated inside point.

    On the levels of ``mask`` whose ``point`` lies in ``[L, U]``: a positive ``F``
    raises ``L`` to the point, a negative ``F`` lowers ``U`` to it, a zero ``F``
    moves nothing; an endpoint need not move its bound; nothing widens.  This is
    bookkeeping after an evaluation, not an update.
    """
    inside = mask & (lower <= point) & (point <= upper)
    new_lower: FloatArray = np.where(inside & (sums > 0), point, lower)
    new_upper: FloatArray = np.where(inside & (sums < 0), point, upper)
    return new_lower, new_upper


@dataclass(frozen=True, eq=False)
class Update:
    """The selected trial values of one block and the kind of each level's update.

    ``values`` is the block's trial vector (the current coefficient where no update
    applies); the four masks name the levels moved by an accepted Newton proposal,
    by the interior midpoint, by a move to the nearer endpoint of a nondegenerate
    bracket, and by the move to the sole point of a degenerate bracket.
    """

    values: FloatArray
    newton: BoolArray
    midpoint: BoolArray
    endpoint: BoolArray
    closed_form: BoolArray


def safeguarded_update(
    alpha: FloatArray,
    sums: FloatArray,
    curvatures: FloatArray,
    *,
    lower: FloatArray,
    upper: FloatArray,
    mask: BoolArray,
) -> Update:
    """One safeguarded Newton update per level of ``mask`` (the contract's rule).

    Outside the bracket: the nearer verified endpoint (the sole point where ``L ==
    U``).  Inside or at an endpoint: ``alpha + F/C`` when ``C > 0`` and the
    proposal is representable and strictly inside ``(L, U)``; else the midpoint
    ``L + (U - L)/2`` when it is representable and strictly inside; else
    ``FloatingPointError`` (``STAGNATION``).  The quotient is formed on the usable
    levels only (in ``mask``, inside or at an endpoint, ``C > 0``): no other level
    is divided, no curvature is floored and no bracket is expanded.  A quotient or
    sum that overflows makes its proposal unusable, as any proposal that is not
    finite.
    """
    below, above = mask & (alpha < lower), mask & (alpha > upper)
    outside = below | above
    degenerate = lower == upper
    inside = mask & ~outside
    usable = inside & (curvatures > 0)
    newton = np.array(alpha, dtype=np.float64, copy=True)
    with np.errstate(over="ignore"):
        newton[usable] = alpha[usable] + sums[usable] / curvatures[usable]
        midpoint = lower + (upper - lower) / 2
    accepted = usable & np.isfinite(newton) & (lower < newton) & (newton < upper)
    interior = (
        inside
        & ~accepted
        & np.isfinite(midpoint)
        & (lower < midpoint)
        & (midpoint < upper)
    )
    if (inside & ~accepted & ~interior).any():
        raise FloatingPointError(STAGNATION)
    values: FloatArray = np.where(
        below, lower, np.where(above, upper, np.where(accepted, newton, alpha))
    )
    values = np.where(interior, midpoint, values)
    return Update(
        values,
        accepted,
        interior,
        outside & ~degenerate,
        outside & degenerate,
    )
