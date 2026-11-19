"""The absorber-based iterative rectifier of joint separation, with fresh witnesses.

``design/02_computation.md``, Section 4 (the iterative-rectifier step that works through
the absorber and handles any number of effects); ``plan/evidence/M1.7_contract.txt``,
Sections A and C; decision D51.  Task M1.7, private.  Written from the published
algorithm as S9 describes it (``references/ppmlhdfe_separation_relu.mata``, lines 49 to
211; ``ppmlhdfe_functions.mata``, lines 86 to 177; ``nonexistence_primer.md``, the IR
section); no Stata code is copied.

The iteration.  ``u`` starts at ``u_P = 0``, ``u_O = 1``.  Each attempt is charged
before its call, a failed call included, and projects ``u`` on ``B = [X, F]`` by the
support WLS of :mod:`powermean.solve.support_projection` with the auxiliary weights 1
on ``O`` and ``K = ceil(|O| / d^2)`` on ``P``: the official primer's finite weighting
(``K = N0 / eps^2``), computed exactly from the binary64 ``d`` and refused unless that
integer is itself exactly a finite double (:func:`representable`; ``2^54`` is one,
``2^53 + 1`` and every integer from ``2^1024`` are not), in place of the executable
source's fixed ``M``, implied-epsilon floors and edited-to-zero diagnostics.  No
penalty is rounded, clamped or floored.  With an exact solve this bounds the
positive-row fitted norm at scale ``d``; it does not imply the stricter fresh ``e``
check, which remains mandatory.  The statistical weights are not these penalties and
never enter.  The projection is given the active ``e`` as its dual tolerance: it asks
its ``u`` column to meet the dual's level requirement and adopts a refined column only
where that requirement holds afresh, which binary64 does not guarantee on every sample
(``CONTRACT_V1.txt``, qualified by ``CONTRACT_V1_ADDENDUM2.txt``); an attempt's work
counts both absorber calls.

Stopping (contract C).  From the returned coefficients ``z`` is reconstructed afresh on
the current rows; the raw ``r = u - z`` and ``w r`` are kept.  The normalized direction
is checked by :func:`powermean.solve.support_witness.primal_witness`; a valid one adds
its strict set to ``S``, together with the FE step's witnesses.  When every zero row is
witnessed the set is complete (``covered``).  Otherwise the separate dual candidate
``w r`` with its entries on ``S`` set to zero is checked afresh by
:func:`powermean.solve.support_witness.dual_witness` on every slope column and FE level
(the raw residual's orthogonality is not inherited): passing, the set is numerically
complete (``dual``, a complete absence when ``S`` is empty).  When the candidate's
margin holds and only its column sums fail, the residual is refined before the verdict
(:func:`_refined_residual`: the classical iterative refinement of a least-squares
residual, at most :data:`REFINEMENTS` projections of ``r`` itself, each kept only if it
lowers the largest ratio and repeated only if it at least halves it, LAPACK's xGERFS
rule). A level of positive rows can have small residuals whose relative equation
still fails after refinement. When the check fails only on levels and refinement
stalls, remove their residual means by projecting on their indicators before
rechecking the dual. A common residual is removed exactly. The same check is
repeated afresh on the refined ``w r``: with
coefficients of order one on the zero rows, ``K (u - z)`` on the positive rows is
otherwise formed by cancellation and misses the sums by about ``K`` rounding units,
whatever the number of attempts.  For the same reason a projection that ends
``wls_uncertain`` is first refined on its own residual (:func:`_refined_projection`)
and is used only if a refinement step passes the projection's own rechecks.  A valid
primal alone stops
with the explicit partial set (``valid_primal``).  Otherwise the rectifier update
``u_P = 0``, ``u_O = max(z_O, 0)`` follows; an unchanged ``u`` stops (``no_change``), as
do the budget, a range failure or a projection failure (``absorption``,
``rank_uncertain``, ``wls_uncertain``), none of which is ever an absence.  The source's
optional heuristic acceleration (``relu_accelerate``) is off; the accepted absorber's
ordinary acceleration is an explicit, recorded projection control.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_product,
    checked_quotient,
    checked_sum,
    group_sums,
)
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray
from powermean.solve.support_problem import SupportProblem, require_same
from powermean.solve.support_projection import Projection, ProjectionControls, project
from powermean.solve.support_witness import (
    DualWitness,
    PrimalWitness,
    dual_witness,
    primal_witness,
    reconstruct,
    union,
)

if TYPE_CHECKING:
    from powermean.solve.support_overlap_lp import OverlapLP

# binary64: a significand of 53 bits and finite values below 2^1024.
# Refinement steps of the residual per attempt, and the rule that a step must at least
# halve the error to earn another: LAPACK's xGERFS (ITMAX = 5; dgerfs.f, "2 * BERR <=
# LSTRES"), the reference iterative refinement.
REFINEMENTS = 5
SIGNIFICAND_BITS = int(np.finfo(np.float64).nmant) + 1
EXPONENT_LIMIT = int(np.finfo(np.float64).maxexp)


@dataclass(frozen=True)
class RectifierControls:
    """The explicit controls of the rectifier (contract table).

    ``tolerance`` and ``threshold`` are ``e_IR`` and ``d_IR``, ``0 < e < d < 1``;
    ``max_attempts`` the WLS attempts of one invocation; ``projection`` the support
    WLS controls.
    """

    tolerance: float
    threshold: float
    max_attempts: int
    projection: ProjectionControls

    def __post_init__(self) -> None:
        """Check every control; refuse ``e >= d`` or ``d >= 1``."""
        for value in (self.tolerance, self.threshold):
            if isinstance(value, bool) or not isinstance(value, float):
                raise ValueError("the tolerance and threshold must be floats")
        if not (
            np.isfinite(self.threshold) and 0 < self.tolerance < self.threshold < 1
        ):
            raise ValueError("require 0 < tolerance < threshold < 1")
        budget = self.max_attempts
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("max_attempts must be a positive integer")
        if not isinstance(self.projection, ProjectionControls):
            raise ValueError("projection must be ProjectionControls")


def penalty(zero_rows: int, threshold: float) -> int:
    """``K = ceil(|O| / d^2)``, exactly, from the binary64 value of ``d``."""
    return math.ceil(Fraction(zero_rows) / Fraction(threshold) ** 2)


def representable(k: int) -> bool:
    """Whether the positive integer ``k`` is exactly a finite binary64 value.

    With ``k = m 2^t`` and ``m`` odd, ``k`` is a double exactly when ``m`` has at most
    53 bits and ``k < 2^1024``.  This decides the integer itself: a bound on the range
    of consecutive integers (``2^53``) would refuse exact penalties such as ``2^54``.
    """
    odd = k // (k & -k)
    return odd.bit_length() <= SIGNIFICAND_BITS and k.bit_length() <= EXPONENT_LIMIT


@dataclass(frozen=True)
class Attempt:
    """The summary of one charged WLS attempt.

    ``status`` and ``reason`` are the projection's; ``rank`` its slope rank;
    ``sweeps``, ``visits``, ``operators`` and ``passes`` the absorber's work, the
    joint call's and the refinement call's together (zero without blocks);
    ``primal_valid``, ``strict``, ``dual_passed`` and ``margin`` the fresh checks,
    ``None`` when not reached; ``refinement`` the projection's refinement reason (empty
    without one) and ``refinement_sweeps`` the sweeps of its call.
    """

    index: int
    status: str
    reason: str
    rank: int | None
    sweeps: int
    visits: int
    operators: int
    passes: int
    primal_valid: bool | None = None
    strict: int | None = None
    dual_passed: bool | None = None
    margin: float | None = None
    refinement: str = ""
    refinement_sweeps: int = 0


@dataclass(frozen=True, eq=False)
class Final:
    """The complete record of the last attempt: the raw quantities, unedited.

    ``u`` the working outcome, ``projection`` the support WLS, ``z`` the fresh
    reconstruction, ``residual`` ``r = u - z`` and ``weighted`` ``w r``; ``primal`` and
    ``dual`` the fresh checks.
    """

    u: FloatArray
    projection: Projection
    z: FloatArray | None = None
    residual: FloatArray | None = None
    weighted: FloatArray | None = None
    primal: PrimalWitness | None = None
    dual: DualWitness | None = None


@dataclass(frozen=True, eq=False)
class RectifierOutcome:
    """The rectifier of one invocation: work, witnesses and verdict.

    ``penalty`` is ``K`` and ``weights`` the auxiliary weights (``None`` when ``K`` is
    refused); ``charged`` the attempts charged; ``attempts`` their summaries and
    ``last`` the full last one; ``witnesses`` the rectifier's valid primal witnesses;
    ``dual`` the last dual check; ``completion`` is ``covered``, ``dual`` or empty and
    ``reason`` names the stop. The sample analyzer can attach ``overlap_lp`` work
    after the rectifier returns.
    """

    controls: RectifierControls
    penalty: int
    weights: FloatArray | None
    charged: int
    attempts: tuple[Attempt, ...]
    last: Final | None
    witnesses: tuple[PrimalWitness, ...]
    dual: DualWitness | None
    completion: str
    reason: str
    overlap_lp: OverlapLP | None = None


def _summary(index: int, final: Final, residuals: Sequence[Projection] = ()) -> Attempt:
    """The summary of an attempt from its projection and its fresh checks.

    ``residuals`` are the projections of the residual's refinements, when any ran:
    their absorber work is counted with the attempt's.
    """
    projection, primal, dual = final.projection, final.primal, final.dual
    refinement = projection.refinement
    refined = None if refinement is None else refinement.result
    counts = [0, 0, 0, 0]
    extra = []
    for residual in residuals:
        extra.append(residual.absorption)
        if residual.refinement is not None:
            extra.append(residual.refinement.result)
    for work in (projection.absorption, refined, *extra):
        if work is not None:
            found = (work.sweeps, work.visits, work.operators, work.diagnostic_passes)
            counts = [a + b for a, b in zip(counts, found, strict=True)]
    return Attempt(
        index,
        projection.status,
        projection.reason,
        projection.rank,
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        primal_valid=None if primal is None else primal.valid,
        strict=None if primal is None else int(primal.strict.size),
        dual_passed=None if dual is None else dual.passed,
        margin=None if dual is None else dual.margin,
        refinement="" if refinement is None else refinement.reason,
        refinement_sweeps=0 if refined is None else refined.sweeps,
    )


@dataclass(frozen=True, eq=False)
class _Step:
    """One attempt: its record, its stop (empty to continue) and the next ``u``."""

    final: Final
    reason: str
    completion: str = ""
    following: FloatArray | None = None
    refined: tuple[Projection, ...] = ()


def _step(
    problem: SupportProblem,
    controls: RectifierControls,
    u: FloatArray,
    weights: FloatArray,
    witnessed: tuple[PrimalWitness, ...],
    *,
    set_aside: bool,
) -> _Step:
    """The projection of ``u``, the fresh checks, the verdict and the rectified ``u``.

    ``witnessed`` are the valid witnesses before this attempt; ``set_aside`` is that of
    :func:`iterative_rectifier`, passed to every projection.
    """
    projection = project(
        problem,
        u,
        weights,
        controls.projection,
        dual_tolerance=controls.tolerance,
        set_aside=set_aside,
    )
    refined: list[Projection] = []
    if projection.status == "wls_uncertain":
        projection = _refined_projection(
            problem, controls, projection, weights, refined, set_aside=set_aside
        )
    final = Final(immutable_array(u), projection)
    slopes, effects = projection.slopes, projection.effects
    if not projection.solved or slopes is None or effects is None:
        return _Step(final, projection.status, refined=tuple(refined))
    try:
        z = reconstruct(problem, slopes, effects)
        residual = checked_difference(u, z, "rectifier residual")
        weighted = checked_product(weights, residual, "weighted residual")
    except FloatingPointError:
        return _Step(final, "range")
    primal = primal_witness(
        problem,
        slopes,
        effects,
        method="ir",
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    final = replace(
        final,
        z=immutable_array(z),
        residual=immutable_array(residual),
        weighted=immutable_array(weighted),
        primal=primal,
    )
    separated = union((*witnessed, primal))
    zero = problem.zero
    if not np.setdiff1d(zero, separated).size:
        return _Step(final, "covered", "covered")
    dual = dual_witness(
        problem,
        weighted,
        separated,
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    steps = 0
    while _sums_alone(dual) and not primal.valid and steps < REFINEMENTS:
        steps += 1
        # The margin holds and only the column sums fail: refine the residual.  With
        # a valid primal the attempt stops on its witness and a fresh invocation
        # follows the removal, so nothing is refined for it.
        projection, again = _refined_residual(
            problem, controls, residual, weights, set_aside=set_aside
        )
        refined.append(projection)
        if again is None:
            break
        candidate, again = _refinement_candidate(
            problem, controls, weights, dual, again
        )
        before, after = _worst(dual), _worst(candidate)
        if not after < before:
            break
        residual, weighted = again
        dual = candidate
        final = replace(
            final,
            residual=immutable_array(residual),
            weighted=immutable_array(weighted),
        )
        if not 2.0 * after <= before:
            break
    final = replace(final, dual=dual)
    if dual.passed:
        return _Step(final, "dual", "dual", refined=tuple(refined))
    if primal.valid:
        return _Step(final, "valid_primal", refined=tuple(refined))
    following = np.zeros(problem.n_rows)
    following[zero] = np.maximum(z[zero], 0.0)
    if np.array_equal(following, u):
        return _Step(final, "no_change", refined=tuple(refined))
    return _Step(final, "", following=following, refined=tuple(refined))


def _sums_alone(dual: DualWitness) -> bool:
    """Whether the dual's margin holds and only its column sums fail."""
    return bool(dual.failure) and set(dual.failure) <= {"slopes", "levels"}


def _worst(dual: DualWitness) -> float:
    """The largest column-sum ratio of a dual check, infinite when not evaluated."""
    if dual.margin is None:
        return float("inf")
    ratios = [] if dual.slopes is None else [np.max(dual.slopes.ratios, initial=0.0)]
    ratios += [np.max(level.ratios, initial=0.0) for level in dual.levels]
    return float(max(ratios, default=0.0))


def _refinement_candidate(
    problem: SupportProblem,
    controls: RectifierControls,
    weights: FloatArray,
    previous: DualWitness,
    refined: tuple[FloatArray, FloatArray],
) -> tuple[DualWitness, tuple[FloatArray, FloatArray]]:
    """Check a refined residual, correcting stalled positive-only levels directly."""
    candidate = dual_witness(
        problem,
        refined[1],
        previous.separated,
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    after = _worst(candidate)
    # Use the same progress requirement as the refinement loop. A tiny
    # improvement can leave the level equation unsatisfied and stop the loop.
    if candidate.failure != ("levels",) or 2.0 * after <= _worst(previous):
        return candidate, refined
    try:
        centered = _center_failed_levels(problem, controls, refined[0], candidate)
        if centered is None:
            return candidate, refined
        weighted = checked_product(weights, centered, "weighted residual")
    except FloatingPointError:
        return candidate, refined
    corrected = dual_witness(
        problem,
        weighted,
        previous.separated,
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    if corrected.passed or _worst(corrected) < after:
        return corrected, (centered, weighted)
    return candidate, refined


def _center_failed_levels(
    problem: SupportProblem,
    controls: RectifierControls,
    residual: FloatArray,
    dual: DualWitness,
) -> FloatArray | None:
    """Project on failed indicators of positive rows, whose weights are equal.

    Subtracting the residual mean adds the corresponding indicator to the fitted
    correction, leaving the zero rows alone. When every residual in the level
    is identical, take that exact value as the mean. Overlapping levels and the
    slope equations still need the full fresh check.
    """
    assert problem.grouping is not None
    following = residual
    for block, sums in zip(problem.grouping.blocks, dual.levels, strict=True):
        failing = sums.ratios > controls.tolerance
        failing[block.codes[problem.zero]] = False
        if not failing.any():
            continue
        low = np.full(block.n_levels, np.inf)
        high = np.full(block.n_levels, -np.inf)
        np.minimum.at(low, block.codes, following)
        np.maximum.at(high, block.codes, following)
        constant = failing & (low == high)
        correction = np.where(constant, low, 0.0)
        varying = failing & ~constant
        if varying.any():
            active = varying[block.codes]
            totals = group_sums(following[active], block.codes[active], block.n_levels)
            correction[varying] = checked_quotient(
                totals[varying],
                block.counts[varying].astype(np.float64),
                "level residual mean",
            )
        following = checked_difference(
            following, correction[block.codes], "centered level residual"
        )
    return None if following is residual else following


def _refined_projection(
    problem: SupportProblem,
    controls: RectifierControls,
    first: Projection,
    weights: FloatArray,
    work: list[Projection],
    *,
    set_aside: bool,
) -> Projection:
    """A projection whose equations miss the tolerance, refined on its residual.

    ``wls_uncertain`` states that the reconstructed residual misses a weighted level
    mean or a slope equation at the projection's absolute tolerance, which happens by
    rounding alone when the fit has coefficients of order one on rows of weight ``K``.
    The residual is projected with the same weights, at most :data:`REFINEMENTS`
    times; the coefficients and fitted values add up and the last residual is the
    residual of the sum.  The refined projection is ``solved`` exactly when a step is:
    that step's own recheck is the recheck of the summed solution's residual, by the
    same function at the same tolerance.  Otherwise ``first`` stands.  Every step is
    appended to ``work``.
    """
    slopes, effects = first.slopes, first.effects
    fitted, residual = first.fitted, first.residual
    if slopes is None or effects is None or fitted is None or residual is None:
        return first
    for _ in range(REFINEMENTS):
        step = project(
            problem,
            np.array(residual, dtype=np.float64),
            weights,
            controls.projection,
            dual_tolerance=controls.tolerance,
            set_aside=set_aside,
        )
        work.append(step)
        if step.status not in ("solved", "wls_uncertain") or step.residual is None:
            return first
        if step.slopes is None or step.effects is None or step.fitted is None:
            return first
        try:
            slopes = checked_sum(slopes, step.slopes, "refined slopes")
            effects = tuple(
                checked_sum(a, b, "refined effects")
                for a, b in zip(effects, step.effects, strict=True)
            )
            fitted = checked_sum(fitted, step.fitted, "refined fitted values")
        except FloatingPointError:
            return first
        residual = step.residual
        if step.solved:
            return replace(
                first,
                status="solved",
                reason="",
                slopes=immutable_array(slopes),
                effects=tuple(immutable_array(a) for a in effects),
                fitted=immutable_array(fitted),
                residual=immutable_array(residual),
                level_means=step.level_means,
                slope_ratios=step.slope_ratios,
            )
    return first


def _refined_residual(
    problem: SupportProblem,
    controls: RectifierControls,
    residual: FloatArray,
    weights: FloatArray,
    *,
    set_aside: bool,
) -> tuple[Projection, tuple[FloatArray, FloatArray] | None]:
    """One step of iterative refinement of the weighted least-squares residual.

    ``r = u - z`` loses its positive rows to cancellation when the fit has
    coefficients of order one there (``z_P`` is of order ``1 / K`` and is
    reconstructed from terms of order one), and ``w r`` then misses the dual's column
    sums by about ``K`` times the rounding unit.  Projecting ``r`` itself with the
    same weights and subtracting its fitted values, which are of the size of that
    error and so reconstructed without cancellation, is the classical refinement of a
    least-squares residual.  The refined ``w r`` is checked afresh by the same dual
    witness; a projection that does not solve leaves the first residual in place.
    """
    projection = project(
        problem,
        np.array(residual, dtype=np.float64),
        weights,
        controls.projection,
        dual_tolerance=controls.tolerance,
        set_aside=set_aside,
    )
    slopes, effects = projection.slopes, projection.effects
    if not projection.solved or slopes is None or effects is None:
        return projection, None
    try:
        fitted = reconstruct(problem, slopes, effects)
        better = checked_difference(residual, fitted, "refined rectifier residual")
        weighted = checked_product(weights, better, "refined weighted residual")
    except FloatingPointError:
        return projection, None
    return projection, (better, weighted)


def iterative_rectifier(
    problem: SupportProblem,
    controls: RectifierControls,
    initial: Sequence[PrimalWitness],
    *,
    set_aside: bool = False,
) -> RectifierOutcome:
    """The iterative rectifier of one fresh invocation on the current sample.

    Parameters
    ----------
    problem : SupportProblem
        The current sample, with at least one zero row.
    controls : RectifierControls
        The explicit tolerance, threshold, attempt budget and projection controls.
    initial : sequence of PrimalWitness
        Witnesses already established on this sample (the FE step's); each must be
        bound to ``problem`` and valid.
    set_aside : bool, default False
        Whether every projection sets aside the slopes that the effects and the
        earlier slopes absorb (:func:`powermean.solve.support_projection.project`):
        the second attempt of :func:`powermean.solve.support.analyze_support`.

    Returns
    -------
    RectifierOutcome
        The charged work, the witnesses and the stop: ``covered``, ``dual``,
        ``valid_primal``, ``no_change``, ``budget``, ``range``, ``absorption``,
        ``rank_uncertain`` or ``wls_uncertain``.
    """
    zero = problem.zero
    if not zero.size:
        raise ValueError("the rectifier needs a zero row")
    for witness in initial:
        require_same(witness.problem, problem)
        if not witness.valid:
            raise ValueError("initial witnesses must be valid")
    k = penalty(int(zero.size), controls.threshold)
    if not representable(k):
        return RectifierOutcome(controls, k, None, 0, (), None, (), None, "", "range")
    weights = np.ones(problem.n_rows)
    weights[problem.positive] = float(k)
    u = np.zeros(problem.n_rows)
    u[zero] = 1.0
    attempts: list[Attempt] = []
    witnesses: list[PrimalWitness] = []
    final: Final | None = None
    charged = 0
    stop = ("", "")
    while not stop[1]:
        witnessed = (*initial, *witnesses)
        if not np.setdiff1d(zero, union(witnessed)).size:
            stop = ("covered", "covered")
        elif charged >= controls.max_attempts:
            stop = ("", "budget")
        else:
            charged += 1
            step = _step(
                problem,
                controls,
                u,
                immutable_array(weights),
                witnessed,
                set_aside=set_aside,
            )
            final = step.final
            attempts.append(_summary(charged, final, step.refined))
            if final.primal is not None and final.primal.valid:
                witnesses.append(final.primal)
            if step.reason or step.following is None:
                stop = (step.completion, step.reason or "range")
            else:
                u = step.following
    return RectifierOutcome(
        controls,
        k,
        immutable_array(weights),
        charged,
        tuple(attempts),
        final,
        tuple(witnesses),
        None if final is None else final.dual,
        *stop,
    )
