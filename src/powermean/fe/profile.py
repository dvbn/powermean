"""Internal power-family profiling of the fixed-effect blocks: state, sweeps, status.

The inner profile only, for any positive number of ordered blocks.  No outer slope
Newton, Schur complement, line search or tightening schedule, no NB2 (the dedicated
:mod:`powermean.fe.profile_nb2` handles it, and this entry refuses an NB2 problem
before any check, stationary or not; the shared record admits the exact NB2
family, this loop does not), no sample
deletion, separation, rank, default, covariance, certificate or public path; a
completed profile is not a full-joint certificate.  The arithmetic is
:mod:`powermean.fe.profile_arithmetic`; the controls are ``profile_tol`` 2e-11 and
``profile_max_sweeps`` 20000 of :class:`powermean.solve.absorbed.AbsorbedOptions`,
unchanged, taken as explicit arguments (positive finite real, positive integer; 1
valid) and never chosen here.

State.  :class:`ProfileState` is one owned read-only coefficient
vector per block, on the transformed scale, bound to the grouping (positions,
block names, codes, level counts and the ordered level labels; a relabeling that keeps
the partition is another grouping).  Starting values are the caller's choice and the
supplied start is recorded in the result; no observation-level index is a warm state,
no mutable alias exists, and nothing is cached by slopes: changed outcomes, weights or
base are a new :class:`~powermean.fe.profile_arithmetic.ProfileProblem`, whose support,
totals, sums, index and equations are formed afresh.  The work runs on local copies;
a failure never modifies the caller's state.

Loop.  The index is reconstructed from the start, the fresh
equations of every original block are evaluated there, and an initially passing check
returns with zero sweeps.  Otherwise each sweep visits the blocks in declared order,
each block adding its increment at the current index and the index being
reconstructed before the next block; after EVERY complete sweep the fresh check of all
original blocks runs at the one returned index, the last allowed sweep included,
before exhaustion is declared.  The sole accuracy predicate is ``max |e_g|`` at most
the requested tolerance, equality passing: no coefficient displacement, last-block
check, scale multiplier, curvature normalization or roundoff allowance (an
allowance ``tol + 32 eps mean(a + b)`` is not taken).  One
block has an exact scalar update algebraically, not an automatic completion: its
fresh check decides like any other.  A zero step or an unchanged coefficient proves
nothing.

Numerator reuse.  At EXACT ``tau = 1`` (the float ``1.0`` of an exact
:class:`~powermean.fe.profile_arithmetic.ProfileProblem`) a block's step is
:func:`~powermean.fe.profile_arithmetic.invariant_block_step`: the numerator level
log sums of a visit are held by the call for the block's next visit, where they stand
for that reduction only if its fresh inputs are bitwise the same.  Every check, sum,
difference, update, reconstruction and fresh check, their order, the refusals and the
counters are those of the original loop; nothing is held between calls.  Every other
``tau``, a subclass record and a replaced ``block_step`` of this module keep
:func:`~powermean.fe.profile_arithmetic.block_step`.

Work.  ``sweeps_attempted`` is counted before a sweep's first block,
``blocks_attempted`` before a block's sums, ``blocks_completed`` after its coefficient
and index are finite, ``sweeps_completed`` after all blocks, ``checks_attempted`` and
``checks_completed`` around each fresh check.  These count block-level work, not
scalar operations.

Outcomes.  Malformed controls or a state bound to another grouping raise built-in
``ValueError``; a starting index that is not representable raises
``FloatingPointError`` (no candidate exists).  Otherwise the result carries
``converged``, ``budget`` (the allowance ended with the last fresh check above the
tolerance) or ``range`` (a refused operation of the checked arithmetic, including a
lost required quantity), with the reason, the requested controls, the counters, the
last complete finite coefficient and index state, and the signed equations of the
last completed check at that state when one exists (a first block that refuses leaves
the checked start as that state; a completed block does not).  Nothing is retried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe.absorb import _budget, _tolerance
from powermean.fe.absorb_numeric import checked_sum
from powermean.fe.groups import Block, Grouping
from powermean.fe.profile_arithmetic import (
    VARIANCE,
    Coefficients,
    EquationCheck,
    NumeratorSums,
    ProfileReductions,
    ProfileSum,
    ProfileProblem,
    _profile_reductions,
    block_step,
    fresh_equations,
    invariant_block_step,
    invariant_denominator_block_step,
    reconstruct_index,
    updated_coefficients,
)
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray, PowerVariance

STATUSES = ("converged", "budget", "range")
CONVERGED, BUDGET, RANGE = STATUSES
BOUND = "the state is bound to another grouping"
COEFFICIENTS = (
    "coefficients must be one finite real vector per block, one value per level"
)
# The step this loop was written against: a replaced ``block_step`` of this module,
# the fault-injection seam of the tests, keeps the original loop.
_ACCEPTED_STEP = block_step
_ACCEPTED_RECONSTRUCT = reconstruct_index
_ACCEPTED_EQUATIONS = fresh_equations
_ACCEPTED_INVARIANT_STEP = invariant_block_step
_ACCEPTED_DENOMINATOR_STEP = invariant_denominator_block_step
_ACCEPTED_REDUCTIONS = _profile_reductions
# The numerator sums one call holds, by block position (module, numerator reuse).
Numerators = dict[int, NumeratorSums | None]


# ---------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class ProfileState:
    """The block coefficients on the transformed scale and their exact binding.

    Attributes
    ----------
    grouping : Grouping
        The accepted grouping the coefficients belong to: its positions, block names,
        codes, level counts and ordered level labels are the row and level identity.
    coefficients : tuple of arrays of float
        One vector per block, one finite value per level, read-only and owned.

    Notes
    -----
    The state is the coefficients and nothing else: no index, residual, flag,
    tolerance or weight.  A direct construction is checked and owns its arrays.
    """

    grouping: Grouping
    coefficients: Coefficients

    def __post_init__(self) -> None:
        """Check the binding and the coefficients; store owned read-only arrays."""
        if not isinstance(self.grouping, Grouping):
            raise ValueError("grouping must be a Grouping record")
        given = tuple(self.coefficients)
        if len(given) != len(self.grouping.blocks):
            raise ValueError(COEFFICIENTS)
        owned: list[FloatArray] = []
        for block, alpha in zip(self.grouping.blocks, given, strict=True):
            if np.iscomplexobj(alpha):
                raise ValueError(COEFFICIENTS)
            array: FloatArray = np.array(alpha, dtype=np.float64, copy=True)
            if array.shape != (block.n_levels,) or not np.isfinite(array).all():
                raise ValueError(COEFFICIENTS)
            owned.append(immutable_array(array))
        object.__setattr__(self, "coefficients", tuple(owned))


def cold_state(grouping: Grouping) -> ProfileState:
    """The cold start: every coefficient zero, bound to ``grouping``."""
    return ProfileState(
        grouping, tuple(np.zeros(block.n_levels) for block in grouping.blocks)
    )


@dataclass(frozen=True)
class Work:
    """The block-level work of one call, attempted and completed apart (exact)."""

    sweeps_attempted: int = 0
    sweeps_completed: int = 0
    blocks_attempted: int = 0
    blocks_completed: int = 0
    checks_attempted: int = 0
    checks_completed: int = 0


@dataclass(frozen=True, eq=False)
class ProfileResult:
    """The outcome of one profile: index, state, start, status, equations, work.

    Attributes
    ----------
    index : array of float
        The reconstructed index of ``state``, every row, read-only and owned.
    state : ProfileState
        The last complete finite coefficient state.
    start : ProfileState
        The supplied starting state, as recorded.
    status : str
        ``converged``, ``budget`` or ``range``.
    reason : str
        Empty on convergence; otherwise what stopped the profile.
    tolerance : float
        The requested maximum absolute weighted level equation.
    max_sweeps : int
        The requested budget of complete sweeps.
    errors : tuple of arrays of float or None
        The signed ``e_g`` of every block at ``index`` from the last completed fresh
        check at that state; ``None`` when no check completed there.
    maximum : float or None
        ``max |e_g|`` of ``errors``.
    work : Work
        The counters.
    """

    index: FloatArray
    state: ProfileState
    start: ProfileState
    status: str
    reason: str
    tolerance: float
    max_sweeps: int
    errors: tuple[FloatArray, ...] | None
    maximum: float | None
    work: Work

    def __post_init__(self) -> None:
        """Check the status and the consistency of the equations; own the arrays."""
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        object.__setattr__(self, "index", immutable_array(self.index))
        if self.errors is None:
            if self.maximum is not None:
                raise ValueError("maximum must be None without errors")
            return
        if self.maximum is None:
            raise ValueError("maximum must be the largest absolute error")
        object.__setattr__(
            self, "errors", EquationCheck(tuple(self.errors), self.maximum).errors
        )

    @property
    def converged(self) -> bool:
        """Whether the fresh check passed at the returned index."""
        return self.status == CONVERGED


# ----------------------------------------------------------------- the loop


class _Counters:
    """The mutable counters of one call; frozen into :class:`Work` at the end."""

    __slots__ = ("values", "reductions")

    def __init__(self) -> None:
        """Start every counter at zero."""
        self.values = dict.fromkeys(Work.__dataclass_fields__, 0)
        try:
            native_slot = (
                type(self) is _REDUCTION_COUNTERS
                and vars(_REDUCTION_COUNTERS).get("reductions") is _REDUCTION_SLOT
            )
            if native_slot:
                _REDUCTION_SLOT.__set__(self, None)
            if native_slot and _profile_reductions is _ACCEPTED_REDUCTIONS:
                _REDUCTION_SLOT.__set__(self, _profile_reductions())
        except MemoryError:
            if type(self) is _REDUCTION_COUNTERS:
                _REDUCTION_SLOT.__set__(self, None)
            # Only optional guard/state setup; no required kernel ran.

    def count(self, name: str) -> None:
        """Count one unit of ``name``."""
        self.values[name] += 1

    def work(self) -> Work:
        """The frozen record."""
        return Work(**self.values)


_REDUCTION_COUNTERS = _Counters
_REDUCTION_SLOT = vars(_Counters)["reductions"]


def _local_reductions(counters: _Counters) -> ProfileReductions | None:
    """No new getter/callback reads on a foreign or modified counter record."""
    try:
        if (type(counters) is _REDUCTION_COUNTERS
                and vars(_REDUCTION_COUNTERS).get("reductions") is _REDUCTION_SLOT):
            try:
                return _REDUCTION_SLOT.__get__(counters, _REDUCTION_COUNTERS)
            except AttributeError:
                pass  # An initializer that issued no local reduction state.
    except MemoryError:
        if type(counters) is _REDUCTION_COUNTERS:
            _REDUCTION_SLOT.__set__(counters, None)
        # Release only optional state before the original numerical fallback.
    return None


@dataclass(frozen=True, eq=False)
class _Point:
    """A finite coefficient state, its index and its fresh check, if one completed."""

    coefficients: Coefficients
    index: FloatArray
    check: EquationCheck | None
    failure: str


def _bound(state: ProfileState, problem: ProfileProblem) -> ProfileState:
    """``state`` if it is bound exactly to the grouping of ``problem``.

    The binding is the positions, the block names and, per block, the level count,
    the codes and the ORDERED level labels (the identity the encoder keys the levels
    by): a relabeling that keeps the partition is another grouping.
    """
    mine, theirs = state.grouping, problem.grouping
    if (
        mine is theirs
        and type(mine) is Grouping
        and not mine.positions.flags.writeable
        and all(
            type(b) is Block and not b.codes.flags.writeable for b in mine.blocks
        )
    ):
        return state
    same = (
        mine.names == theirs.names
        and np.array_equal(mine.positions, theirs.positions)
        and all(
            a.n_levels == b.n_levels
            and np.array_equal(a.codes, b.codes)
            and a.levels == b.levels
            for a, b in zip(mine.blocks, theirs.blocks, strict=True)
        )
    )
    if not same:
        raise ValueError(BOUND)
    return state


def _evaluate(
    problem: ProfileProblem,
    coefficients: Coefficients,
    index: FloatArray,
    counters: _Counters,
) -> _Point:
    """The fresh check of every original block at ``index``; its failure if any."""
    counters.count("checks_attempted")
    try:
        reductions = _local_reductions(counters)
        check = (
            fresh_equations(problem, index, _sums=reductions[0],
                            _inputs=reductions[2])
            if reductions is not None and fresh_equations is _ACCEPTED_EQUATIONS
            else fresh_equations(problem, index)
        )
    except FloatingPointError as error:
        return _Point(coefficients, index, None, f"fresh check: {error}")
    counters.count("checks_completed")
    return _Point(coefficients, index, check, "")


def _numerators(problem: ProfileProblem) -> Numerators | None:
    """The store of held invariant block sums at EXACT ``tau = 1`` or ``tau = 2``.

    Only an exact :class:`~powermean.fe.profile_arithmetic.ProfileProblem` of the
    exact power family whose ``tau`` is the float ``1.0`` or ``2.0`` qualifies, each
    read only once its record's exact type is known, so that no hook runs; a subclass
    and every other ``tau`` keep the original loop.
    """
    if type(problem) is not ProfileProblem:
        return None
    variance = problem.variance
    if type(variance) is not PowerVariance:
        return None
    tau = variance.tau
    if type(tau) is not float:
        return None
    if tau == 1.0:
        inv_num: Numerators | None = getattr(problem, "_inv_num", None)
        return {} if inv_num is None else inv_num
    if tau == 2.0:
        inv_den: Numerators | None = getattr(problem, "_inv_den", None)
        return {} if inv_den is None else inv_den
    return None


def _step(
    problem: ProfileProblem, q: FloatArray, block: int, numerators: Numerators | None,
    *, sums: ProfileSum | None = None,
) -> FloatArray:
    """The step of ``block`` at ``q``, with the call's held invariant sums if any.

    :func:`~powermean.fe.profile_arithmetic.block_step`, unless the call has a store
    and this module's ``block_step`` is the accepted one; then
    :func:`~powermean.fe.profile_arithmetic.invariant_block_step` at ``tau = 1`` or
    :func:`~powermean.fe.profile_arithmetic.invariant_denominator_block_step` at
    ``tau = 2``, the same step, whose returned sums the store holds for the block's
    next visit.
    """
    if numerators is None or block_step is not _ACCEPTED_STEP:
        return (
            block_step(problem, q, block, _sums=sums)
            if sums is not None and block_step is _ACCEPTED_STEP
            else block_step(problem, q, block)
        )
    variance = problem.variance
    if type(variance) is PowerVariance and variance.tau == 2.0:
        if (sums is not None
                and invariant_denominator_block_step is _ACCEPTED_DENOMINATOR_STEP):
            step, numerators[block] = invariant_denominator_block_step(
                problem, q, block, numerators.get(block), _sums=sums
            )
        else:
            step, numerators[block] = invariant_denominator_block_step(
                problem, q, block, numerators.get(block)
            )
        return step
    if sums is not None and invariant_block_step is _ACCEPTED_INVARIANT_STEP:
        step, numerators[block] = invariant_block_step(
            problem, q, block, numerators.get(block), _sums=sums
        )
    else:
        step, numerators[block] = invariant_block_step(
            problem, q, block, numerators.get(block)
        )
    return step


_ACCEPTED_PRIVATE_STEP = _step


def _sweep(
    problem: ProfileProblem,
    point: _Point,
    counters: _Counters,
    numerators: Numerators | None,
) -> _Point:
    """One ordered sweep from ``point``: each block's increment at the current index.

    Returns the finite state after the sweep without a check, or, on a refused
    block, the last complete finite state with the failure named; that state keeps
    the completed check of ``point`` only while no block has completed, since the
    check belongs to that state alone.  The counters record what was attempted and
    what completed; ``numerators`` is the call's store of held numerator sums, or
    ``None`` (:func:`_step`).
    """
    coefficients, index, check = point.coefficients, point.index, point.check
    counters.count("sweeps_attempted")
    blocks = problem.grouping.blocks
    fast_reconstruct = (
        type(problem) is ProfileProblem and reconstruct_index is _ACCEPTED_RECONSTRUCT
    )
    prefix = problem.base
    for block, encoded in enumerate(blocks):
        counters.count("blocks_attempted")
        try:
            reductions = _local_reductions(counters)
            if reductions is not None and _step is _ACCEPTED_PRIVATE_STEP:
                step = _step(problem, index, block, numerators, sums=reductions[1])
            else:
                step = _step(problem, index, block, numerators)
            candidate = updated_coefficients(coefficients, block, step)
            if fast_reconstruct:
                next_prefix = checked_sum(
                    prefix, candidate[block][encoded.codes], "reconstructed index"
                )
                candidate_index = next_prefix
                for later_block, later_alpha in zip(
                    blocks[block + 1 :], candidate[block + 1 :], strict=True
                ):
                    candidate_index = checked_sum(
                        candidate_index,
                        later_alpha[later_block.codes],
                        "reconstructed index",
                    )
            else:
                next_prefix = prefix
                candidate_index = reconstruct_index(problem, candidate)
        except FloatingPointError as error:
            return _Point(
                coefficients, index, check, f"block {encoded.name!r}: {error}"
            )
        prefix = next_prefix
        coefficients, index, check = candidate, candidate_index, None
        counters.count("blocks_completed")
    counters.count("sweeps_completed")
    return _Point(coefficients, index, None, "")


def _result(
    point: _Point,
    status: str,
    reason: str,
    *,
    problem: ProfileProblem,
    start: ProfileState,
    controls: tuple[float, int],
    counters: _Counters,
) -> ProfileResult:
    """The result record of a point."""
    check = point.check
    return ProfileResult(
        index=point.index,
        state=ProfileState(problem.grouping, point.coefficients),
        start=start,
        status=status,
        reason=reason,
        tolerance=controls[0],
        max_sweeps=controls[1],
        errors=None if check is None else check.errors,
        maximum=None if check is None else check.maximum,
        work=counters.work(),
    )


def profile(
    problem: ProfileProblem,
    *,
    tolerance: float,
    max_sweeps: int,
    state: ProfileState | None = None,
) -> ProfileResult:
    """Profile the blocks of ``problem`` by ordered exact block updates.

    Parameters
    ----------
    problem : ProfileProblem
        The checked outcomes, base index, weights, totals and variance
        (:func:`~powermean.fe.profile_arithmetic.prepare_profile`).
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
    ProfileResult
        The index and state of the returned point, its status, reason, equations
        and work; unsuccessful statuses keep the last complete finite state.

    Raises
    ------
    ValueError
        Malformed controls, a state bound to another grouping, or a problem whose
        variance is not the exact power family (an NB2 problem is refused here
        before any check, stationary or not; its solver is
        :func:`powermean.fe.profile_nb2.profile_nb2`).
    FloatingPointError
        A starting index that is not representable: no candidate exists.
    """
    if type(problem.variance) is not PowerVariance:
        raise ValueError(VARIANCE)
    tol = _tolerance(tolerance)
    budget = _budget(max_sweeps)
    start = cold_state(problem.grouping) if state is None else _bound(state, problem)
    counters = _Counters()
    point = _evaluate(
        problem,
        start.coefficients,
        reconstruct_index(problem, start.coefficients),
        counters,
    )
    numerators = _numerators(problem)
    while True:
        if point.check is None:
            return _result(
                point,
                RANGE,
                point.failure,
                problem=problem,
                start=start,
                controls=(tol, budget),
                counters=counters,
            )
        if point.check.passing(tol):
            return _result(
                point,
                CONVERGED,
                "",
                problem=problem,
                start=start,
                controls=(tol, budget),
                counters=counters,
            )
        if counters.values["sweeps_attempted"] + 1 > budget:
            reason = (
                f"maximum equation {point.check.maximum:.3e} exceeds the tolerance "
                f"{tol:.3e} after {counters.values['sweeps_completed']} sweeps, "
                f"the budget of {budget}"
            )
            return _result(
                point,
                BUDGET,
                reason,
                problem=problem,
                start=start,
                controls=(tol, budget),
                counters=counters,
            )
        moved = _sweep(problem, point, counters, numerators)
        if moved.failure:
            return _result(
                moved,
                RANGE,
                moved.failure,
                problem=problem,
                start=start,
                controls=(tol, budget),
                counters=counters,
            )
        point = _evaluate(problem, moved.coefficients, moved.index, counters)
