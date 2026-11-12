"""Internal NB2 profiling of the fixed-effect blocks: factory, state, sweeps, work.

The inner NB2 profile only, a scalar equation per level.  No outer slope Newton, line
search or tightening schedule, no NB2 parameter estimation, no sample
deletion, separation, rank, existence decision, default, certificate or public path;
a completed profile is not a full-joint certificate and no convergence rate is
claimed.  The scalar primitives are :mod:`powermean.fe.profile_nb2_scalar`; the
shared checked problem, reconstruction and fresh equations are
:mod:`powermean.fe.profile_arithmetic`; the state, controls, counters and result
mechanics of the power profile :mod:`powermean.fe.profile` are reused as
they are.  The controls are ``profile_tol`` 2e-11 and ``profile_max_sweeps`` 20000 of
:class:`powermean.solve.absorbed.AbsorbedOptions`, unchanged, taken as explicit
arguments and never chosen here.

Family.  :func:`prepare_profile_nb2` is the NB2 factory of the shared record, by the
exact :class:`~powermean.model.variance.NB2Variance` type; the power factory and the
power loop refuse it, and :func:`profile_nb2` refuses a power problem.  At exactly
``kappa = 0`` the NB2 equation is the ``tau = 1`` power equation: the call
constructs a real checked ``PowerVariance(1)`` problem on the same
prepared weights, outcomes, base and logs, and runs the power profile with
the same state and controls, an identity of the same family equation and not an
optimizer fallback; its scalar work is zero.  A positive ``kappa``, however small or
large, is never replaced by Poisson or Gamma.

Loop.  The index is reconstructed from the
start and the fresh equations of every original block are evaluated there; an
initially passing check returns with zero sweeps.  Otherwise each sweep visits the
blocks in declared order, and a visit gives every UNFINISHED level of its block ONE
safeguarded update, vectorized over the levels, then the index is reconstructed and
the next block follows: the fresh current equations of the block, the bracket of
each unfinished level (retained while the ordered other-block inputs of its active
rows are bitwise unchanged, so that its rounded scalar function is the same, else
rebuilt from the analytic endpoints and their fresh signs; an equal rounded index
without the block does not retain), the tightening by the current point, the one
update of each level (Newton, midpoint, endpoint move or closed form), the fresh
evaluation of the trial, then the update applied, and the trial's tightening.  A
visit is not a completed scalar solve.  After EVERY complete sweep the fresh check of
all original blocks runs at the one returned index, the last allowed sweep included,
before exhaustion is declared; the sole accuracy predicate is ``max |e_g|`` at most
the tolerance, equality passing.  No hidden inner loop, scalar cap, retry or relaxed
tolerance exists.

Work.  :class:`~powermean.fe.profile.Work` keeps its block-level meanings
(``blocks_completed`` after a visit's update, reconstruction and trial evaluation are
finite).  :class:`ScalarWork` counts logical level evaluations, rebuilt brackets and
applied updates, apart, each when it completes: ``current`` (the levels evaluated at
the visit's current point), ``endpoint`` (the levels evaluated at a rebuilt lower or
upper endpoint, each evaluation counted once it completes, a degenerate sole point
evaluated as both), ``rebuilt`` (the levels whose rebuilt bracket passed its fresh
check), ``proposed`` (one per updated level, once its trial evaluation completed),
and the applied update kinds ``newton``, ``midpoint``, ``endpoint_moves`` and
``closed_form``, counted only with the trial's completion.  These are not row or
kernel operation counts: every current, endpoint or trial evaluation reconstructs
the full index and evaluates the row kernel on every active row, not only the
level's rows, and the curvature is a separate kernel evaluation at the current
index, not counted.  ``endpoint == 2 rebuilt`` and ``proposed`` equal to the sum of
the kinds are identities of completed visits only: on a refusal inside a visit the
completed evaluations stay counted (a lower endpoint evaluated before its upper
refused; both before a failed check), the refused bracket is not counted as rebuilt
and no update of the visit is applied or counted.  Nothing is retained between
calls.

Outcomes.  As the power profile: malformed controls, a wrong family or a state bound
to another grouping raise ``ValueError``; a starting index that is not representable
raises ``FloatingPointError``; the result carries ``converged``, ``budget`` or
``range``.  The ``range`` status also reports the scalar refusals
with their reason: a bracket endpoint whose fresh rounded sign is wrong, a degenerate
bracket whose sole point fails the equation (no interior progress) and no
representable interior proposal or midpoint (stagnation), none of which is a new
status, a widened bracket, a changed sample or a relaxed tolerance.  The last
complete finite state is returned and the caller's state is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb import _budget, _tolerance
from powermean.fe.absorb_numeric import prepare_weights
from powermean.fe.groups import Grouping
from powermean.fe.profile import (
    BUDGET,
    CONVERGED,
    RANGE,
    ProfileResult,
    ProfileState,
    Work,
    _bound,
    _Counters,
    _evaluate,
    _Point,
    _result,
    cold_state,
    profile,
)
from powermean.fe.profile_arithmetic import (
    BoolArray,
    Coefficients,
    ProfileProblem,
    _base,
    _build_profile_problem,
    _logs,
    _outcomes,
    reconstruct_index,
)
from powermean.fe.profile_nb2_scalar import (
    Bracket,
    analytic_bracket,
    endpoint_check,
    index_without,
    level_curvatures,
    level_equations,
    other_inputs,
    replaced_block,
    retained_levels,
    safeguarded_update,
    tightened,
    trial_equations,
)
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance

NB2_VARIANCE = "variance must be an NB2Variance"


# ---------------------------------------------------------------------- records


@dataclass(frozen=True)
class ScalarWork:
    """The completed level evaluations and applied updates of the scalar work."""

    current: int = 0
    endpoint: int = 0
    proposed: int = 0
    rebuilt: int = 0
    newton: int = 0
    midpoint: int = 0
    endpoint_moves: int = 0
    closed_form: int = 0


class _ScalarCounters:
    """The mutable scalar counters of one call; frozen into :class:`ScalarWork`."""

    __slots__ = ("values",)

    def __init__(self) -> None:
        """Start every counter at zero."""
        self.values = dict.fromkeys(ScalarWork.__dataclass_fields__, 0)

    def add(self, name: str, count: int) -> None:
        """Count ``count`` completed units of ``name``."""
        self.values[name] += int(count)

    def work(self) -> ScalarWork:
        """The frozen record."""
        return ScalarWork(**self.values)


@dataclass(frozen=True, eq=False)
class ProfileNB2Result:
    """The outcome of one NB2 profile: the profile result and the scalar work.

    Attributes
    ----------
    profile : ProfileResult
        The accepted result record: index, state, start, status, reason, controls,
        the signed equations of the last completed check and the block-level work.
    scalar : ScalarWork
        The scalar counters; all zero at ``kappa = 0``, where the power identity
        runs.
    """

    profile: ProfileResult
    scalar: ScalarWork

    def __post_init__(self) -> None:
        """Check the two records by type."""
        if not isinstance(self.profile, ProfileResult):
            raise ValueError("profile must be a ProfileResult record")
        if not isinstance(self.scalar, ScalarWork):
            raise ValueError("scalar must be a ScalarWork record")

    @property
    def status(self) -> str:
        """``converged``, ``budget`` or ``range``."""
        return self.profile.status

    @property
    def converged(self) -> bool:
        """Whether the fresh check passed at the returned index."""
        return self.profile.converged

    @property
    def state(self) -> ProfileState:
        """The last complete finite coefficient state."""
        return self.profile.state

    @property
    def index(self) -> FloatArray:
        """The reconstructed index of the returned state, every row."""
        return self.profile.index

    @property
    def work(self) -> Work:
        """The block-level counters."""
        return self.profile.work

    @property
    def reason(self) -> str:
        """Empty on convergence; otherwise what stopped the profile."""
        return self.profile.reason


# --------------------------------------------------------------------- factory


def prepare_profile_nb2(
    grouping: Grouping,
    z: ArrayLike,
    base: ArrayLike,
    weights: ArrayLike,
    *,
    variance: NB2Variance,
) -> ProfileProblem:
    """The checked NB2 problem of a grouping, outcomes, base index and weights.

    The shared validation of :class:`~powermean.fe.profile_arithmetic.ProfileProblem`
    with the exact :class:`~powermean.model.variance.NB2Variance`: ``ValueError``
    for another family, malformed inputs, a level without a finite positive total
    or without a row of positive weight and positive outcome (a refusal of this
    inner calculation, not a sample, existence or rank decision).
    """
    if type(variance) is not NB2Variance:
        raise ValueError(NB2_VARIANCE)
    prepared = prepare_weights(grouping, weights)
    outcomes = _outcomes(z, prepared.n_rows)
    checked_base = _base(base, prepared.n_rows)
    log_weights, log_outcomes = _logs(prepared.weights, outcomes)
    return _build_profile_problem(
        prepared,
        outcomes,
        checked_base,
        variance,
        log_weights,
        log_outcomes,
    )


def _power_identity(
    problem: ProfileProblem,
    tolerance: float,
    max_sweeps: int,
    state: ProfileState | None,
) -> ProfileNB2Result:
    """The ``kappa = 0`` identity: the checked ``tau = 1`` problem, the power loop."""
    if type(problem) is ProfileProblem:
        power = problem._with_base(problem.base, variance=PowerVariance(1.0))
    else:
        power = ProfileProblem(
            problem.prepared,
            problem.z,
            problem.base,
            PowerVariance(1.0),
            problem.log_weights,
            problem.log_outcomes,
        )
    return ProfileNB2Result(
        profile(power, tolerance=tolerance, max_sweeps=max_sweeps, state=state),
        ScalarWork(),
    )


# --------------------------------------------------------------------- a visit


def _brackets(
    problem: ProfileProblem,
    coefficients: Coefficients,
    block: int,
    *,
    stored: Bracket | None,
    unfinished: BoolArray,
    tolerance: float,
    scalar: _ScalarCounters,
) -> tuple[FloatArray, FloatArray, tuple[FloatArray, ...]]:
    """The bracket of every unfinished level: retained, or rebuilt and sign-checked.

    Returns ``L``, ``U`` (``nan`` on levels without a held bracket) and the ordered
    other-block inputs under which they hold.  A level keeps its stored bracket only
    while those inputs are bitwise unchanged on its active rows; a rebuilt level is
    evaluated at both analytic endpoints through the ordered reconstruction of the
    trial coefficients, each evaluation counted once it completes, and the bracket
    is counted as rebuilt once the contract's checks pass, before it is used.
    """
    alpha = coefficients[block]
    context = other_inputs(problem, coefficients, block)
    retained = retained_levels(stored, context, problem, block)
    lower = np.full(alpha.shape, np.nan)
    upper = np.full(alpha.shape, np.nan)
    if stored is not None:
        lower = np.where(retained, stored.lower, lower)
        upper = np.where(retained, stored.upper, upper)
    rebuild = unfinished & ~retained
    if rebuild.any():
        count = int(rebuild.sum())
        without = index_without(problem, coefficients, block)
        fresh_lower, fresh_upper = analytic_bracket(problem, without, block)
        degenerate = fresh_lower == fresh_upper
        low_sums, low_errors = trial_equations(
            problem, coefficients, block, np.where(rebuild, fresh_lower, alpha)
        )
        # Completed work is kept even if the upper evaluation or the check refuses.
        scalar.add("endpoint", count)
        high_sums, _ = trial_equations(
            problem, coefficients, block, np.where(rebuild, fresh_upper, alpha)
        )
        scalar.add("endpoint", count)
        endpoint_check(
            low_sums,
            high_sums,
            low_errors,
            degenerate=degenerate,
            tolerance=tolerance,
            mask=rebuild,
        )
        scalar.add("rebuilt", count)
        lower = np.where(rebuild, fresh_lower, lower)
        upper = np.where(rebuild, fresh_upper, upper)
    return lower, upper, context


def _visit(
    problem: ProfileProblem,
    coefficients: Coefficients,
    index: FloatArray,
    block: int,
    *,
    stored: Bracket | None,
    tolerance: float,
    scalar: _ScalarCounters,
) -> tuple[Coefficients, FloatArray, Bracket | None]:
    """One visit of ``block``: one safeguarded update of each unfinished level.

    The fresh current equations decide which levels need an update; the brackets are
    retained or rebuilt; the current point tightens its bracket; the update of
    :func:`~powermean.fe.profile_nb2_scalar.safeguarded_update` is selected for the
    block at once; the index is reconstructed; the trial is evaluated freshly, and
    only then is the update applied and counted; the trial tightens the bracket,
    which is kept with its ordered other-block inputs.  A refusal anywhere raises
    ``FloatingPointError``: the block does not complete and no update of the visit
    is applied or counted.
    """
    encoded = problem.grouping.blocks[block]
    alpha = coefficients[block]
    sums, errors = level_equations(problem, index, block)
    scalar.add("current", encoded.n_levels)
    unfinished = np.abs(errors) > tolerance
    if not unfinished.any():
        return coefficients, index, stored
    lower, upper, context = _brackets(
        problem,
        coefficients,
        block,
        stored=stored,
        unfinished=unfinished,
        tolerance=tolerance,
        scalar=scalar,
    )
    curvatures = level_curvatures(problem, index, block)
    lower, upper = tightened(lower, upper, alpha, sums, unfinished)
    update = safeguarded_update(
        alpha, sums, curvatures, lower=lower, upper=upper, mask=unfinished
    )
    candidate = replaced_block(coefficients, block, update.values)
    candidate_index = reconstruct_index(problem, candidate)
    trial_sums, _ = level_equations(problem, candidate_index, block)
    scalar.add("proposed", int(unfinished.sum()))
    for name, mask in (
        ("newton", update.newton),
        ("midpoint", update.midpoint),
        ("endpoint_moves", update.endpoint),
        ("closed_form", update.closed_form),
    ):
        scalar.add(name, int(mask.sum()))
    lower, upper = tightened(lower, upper, update.values, trial_sums, unfinished)
    return candidate, candidate_index, Bracket(lower, upper, context)


# ------------------------------------------------------------------- the loop


def _sweep(
    problem: ProfileProblem,
    point: _Point,
    brackets: list[Bracket | None],
    *,
    tolerance: float,
    counters: _Counters,
    scalar: _ScalarCounters,
) -> _Point:
    """One ordered sweep from ``point``: a visit of every block at the current index.

    As the power sweep: the finite state after the sweep without a check, or, on a
    refused block, the last complete finite state with the failure named, keeping
    the completed check of ``point`` only while no block has completed.  The
    brackets of the blocks are updated in place for the next sweep.
    """
    coefficients, index, check = point.coefficients, point.index, point.check
    counters.count("sweeps_attempted")
    for block, encoded in enumerate(problem.grouping.blocks):
        counters.count("blocks_attempted")
        try:
            coefficients, index, brackets[block] = _visit(
                problem,
                coefficients,
                index,
                block,
                stored=brackets[block],
                tolerance=tolerance,
                scalar=scalar,
            )
        except FloatingPointError as error:
            return _Point(
                coefficients, index, check, f"block {encoded.name!r}: {error}"
            )
        check = None
        counters.count("blocks_completed")
    counters.count("sweeps_completed")
    return _Point(coefficients, index, None, "")


def profile_nb2(
    problem: ProfileProblem,
    *,
    tolerance: float,
    max_sweeps: int,
    state: ProfileState | None = None,
) -> ProfileNB2Result:
    """Profile the blocks of an NB2 ``problem`` by ordered safeguarded level updates.

    Parameters
    ----------
    problem : ProfileProblem
        The checked outcomes, base index, weights, totals and exact NB2 variance
        (:func:`prepare_profile_nb2`).
    tolerance : float
        The requested maximum absolute weighted level equation over every original
        block and level; positive and finite; equality passes.
    max_sweeps : int
        The budget of complete sweeps, a positive integer, 1 included.
    state : ProfileState, optional
        The starting coefficients, bound to the same grouping; the cold start if
        omitted.  Recorded as ``start`` in the result, never modified.

    Returns
    -------
    ProfileNB2Result
        The accepted profile result (index, state, status, reason, equations and
        block-level work) and the scalar work; unsuccessful statuses keep the last
        complete finite state.

    Raises
    ------
    ValueError
        A problem whose variance is not the exact NB2 family, malformed controls,
        or a state bound to another grouping.
    FloatingPointError
        A starting index that is not representable: no candidate exists.
    """
    if type(problem.variance) is not NB2Variance:
        raise ValueError(NB2_VARIANCE)
    if problem.variance.kappa == 0:
        return _power_identity(problem, tolerance, max_sweeps, state)
    tol = _tolerance(tolerance)
    budget = _budget(max_sweeps)
    start = cold_state(problem.grouping) if state is None else _bound(state, problem)
    counters, scalar = _Counters(), _ScalarCounters()
    brackets: list[Bracket | None] = [None] * len(problem.grouping.blocks)
    point = _evaluate(
        problem,
        start.coefficients,
        reconstruct_index(problem, start.coefficients),
        counters,
    )

    def finished(point: _Point, status: str, reason: str) -> ProfileNB2Result:
        return ProfileNB2Result(
            _result(
                point,
                status,
                reason,
                problem=problem,
                start=start,
                controls=(tol, budget),
                counters=counters,
            ),
            scalar.work(),
        )

    while True:
        if point.check is None:
            return finished(point, RANGE, point.failure)
        if point.check.passing(tol):
            return finished(point, CONVERGED, "")
        if counters.values["sweeps_attempted"] + 1 > budget:
            reason = (
                f"maximum equation {point.check.maximum:.3e} exceeds the tolerance "
                f"{tol:.3e} after {counters.values['sweeps_completed']} sweeps, "
                f"the budget of {budget}"
            )
            return finished(point, BUDGET, reason)
        moved = _sweep(
            problem, point, brackets, tolerance=tol, counters=counters, scalar=scalar
        )
        if moved.failure:
            return finished(moved, RANGE, moved.failure)
        point = _evaluate(problem, moved.coefficients, moved.index, counters)
