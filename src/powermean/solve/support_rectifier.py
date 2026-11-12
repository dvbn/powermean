"""The absorber-based iterative rectifier of joint separation, with fresh witnesses.

This is the iterative-rectifier step, which works through the absorber and handles any
number of effects.  Internal module.  Written from the published algorithm as ppmlhdfe
2.3.4 describes it (``ppmlhdfe_separation_relu.mata``, lines 49 to 211;
``ppmlhdfe_functions.mata``, lines 86 to 177; ``nonexistence_primer.md``, the IR
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
where that requirement holds afresh, which binary64 does not guarantee on every sample;
an attempt's work counts both absorber calls.

Stopping.  From the returned coefficients ``z`` is reconstructed afresh on
the current rows; the raw ``r = u - z`` and ``w r`` are kept.  The normalized direction
is checked by :func:`powermean.solve.support_witness.primal_witness`; a valid one adds
its strict set to ``S``, together with the FE step's witnesses.  When every zero row is
witnessed the set is complete (``covered``).  Otherwise the separate dual candidate
``w r`` with its entries on ``S`` set to zero is checked afresh by
:func:`powermean.solve.support_witness.dual_witness` on every slope column and FE level
(the raw residual's orthogonality is not inherited): passing, the set is numerically
complete (``dual``, a complete absence when ``S`` is empty).  A valid primal alone stops
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

import numpy as np

from powermean.fe.absorb_numeric import checked_difference, checked_product
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

# binary64: a significand of 53 bits and finite values below 2^1024.
SIGNIFICAND_BITS = int(np.finfo(np.float64).nmant) + 1
EXPONENT_LIMIT = int(np.finfo(np.float64).maxexp)


@dataclass(frozen=True)
class RectifierControls:
    """The explicit controls of the rectifier.

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
    ``reason`` names the stop.
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


def _summary(index: int, final: Final) -> Attempt:
    """The summary of an attempt from its projection and its fresh checks."""
    projection, primal, dual = final.projection, final.primal, final.dual
    refinement = projection.refinement
    refined = None if refinement is None else refinement.result
    counts = [0, 0, 0, 0]
    for work in (projection.absorption, refined):
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


def _step(
    problem: SupportProblem,
    controls: RectifierControls,
    u: FloatArray,
    weights: FloatArray,
    witnessed: tuple[PrimalWitness, ...],
) -> _Step:
    """The projection of ``u``, the fresh checks, the verdict and the rectified ``u``.

    ``witnessed`` are the valid witnesses before this attempt.
    """
    projection = project(
        problem, u, weights, controls.projection, dual_tolerance=controls.tolerance
    )
    final = Final(immutable_array(u), projection)
    slopes, effects = projection.slopes, projection.effects
    if not projection.solved or slopes is None or effects is None:
        return _Step(final, projection.status)
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
    final = replace(final, dual=dual)
    if dual.passed:
        return _Step(final, "dual", "dual")
    if primal.valid:
        return _Step(final, "valid_primal")
    following = np.zeros(problem.n_rows)
    following[zero] = np.maximum(z[zero], 0.0)
    if np.array_equal(following, u):
        return _Step(final, "no_change")
    return _Step(final, "", following=following)


def iterative_rectifier(
    problem: SupportProblem,
    controls: RectifierControls,
    initial: Sequence[PrimalWitness],
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
            step = _step(problem, controls, u, immutable_array(weights), witnessed)
            final = step.final
            attempts.append(_summary(charged, final))
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
