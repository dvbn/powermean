"""Checked arithmetic of the power-family profile: inputs, log sums, steps, equations.

Exact block updates with log-sum-exp; the controls ``profile_tol`` 2e-11 and
``profile_max_sweeps`` 20000 are taken as explicit arguments by
:mod:`powermean.fe.profile`.  This module holds the primitives only.  The state, the
sweeps, the budget and the statuses are :mod:`powermean.fe.profile`.  The shared
checked record, the variance-neutral reconstruction and the fresh equations admit the
exact NB2 family, with the power factory :func:`prepare_profile`, the power log inputs
and the power loop kept power-only by explicit refusals; the NB2 factory and solver
are :mod:`powermean.fe.profile_nb2`.

Notation.  ``Z >= 0`` is the already-transformed outcome, ``b`` the finite base index
(``X gamma + omega o``, the transformed offset included: no omega, outcome power,
exposure or slope enters here), ``a >= 0`` the raw observation weights, ``tau`` the
power of the exact :class:`~powermean.model.variance.PowerVariance` (its internal
domain ``[0, 3]`` inherited, no profiler-only restriction), block ``k`` of the
grouping has codes ``c_k`` and ``L_k`` levels, ``alpha_k`` is its coefficient vector,
and the authoritative index is the ordered reconstruction

    q = ((b + alpha_1[c_1]) + alpha_2[c_2]) + ...     every row, zero weights included.

Block update, at the CURRENT ``q``, for each level ``g``:

    A_g = LSE(log a_i + log Z_i + (1 - tau) q_i : a_i > 0, Z_i > 0)
    B_g = LSE(log a_i + (2 - tau) q_i : a_i > 0)
    alpha_k,g <- fl(alpha_k,g + (A_g - B_g))             an INCREMENT

The replacement formula ``alpha_k,g = A_g - B_g`` uses the index
WITHOUT block ``k``; the two agree algebraically, and only the increment is formed
here.  Each raw weight enters both sums once; it is not a curvature weight.  Absent
terms (a zero weight, a zero outcome in ``A``) are negative infinity, an exact zero
term, never a floor.  ``LSE`` is the reduction
:func:`powermean.fe.reductions.group_logsumexp`: the maximum of the level taken out,
the exponentials of the differences summed in row order, the log of the sum added
back.  The maximum term of
every level is ``exp(0) = 1``, so the sum is finite and positive: a smaller normalized
exponential that underflows to zero, or a shifted difference that overflows to
negative infinity from two finite ordered logs, is a vanishing tail of a present
level, allowed.  Eligible log inputs, every level sum, the step, the updated
coefficient and the reconstructed index must be finite, and a required exponent
product ``(1 - tau) q_i`` or ``(2 - tau) q_i`` whose two factors are nonzero must not
round to zero on an eligible row (an exact-zero factor at ``tau = 1`` or ``tau = 2``,
a zero index and a nonzero subnormal product are not losses); otherwise the operation
refuses with ``FloatingPointError`` and nothing is clipped, floored or recentered.

Numerator reuse at EXACT ``tau = 1``.  There ``(1 - tau) q_i`` is a zero, so ``v_A``
is ``log a_i + log Z_i`` on its eligible rows at every finite ``q`` and ``A_g`` does
not move between the visits of a block.  :func:`invariant_block_step`, the step of
the power loop at that ``tau``, still forms :func:`log_inputs` with every check, the
denominator sums and the difference at each visit, in the order of
:func:`block_step`; it takes the numerator sums of an earlier visit of the same call
(:class:`NumeratorSums`) only when this visit's ``v_A``, codes and level count are
bitwise the inputs those sums were formed from.  That is a fresh equality of values,
whose sums the deterministic reduction then shares, not an ownership claim; other
inputs, a record that is not plain NumPy data or a replaced reduction run the call
as :func:`block_step` does.  :func:`block_step` itself is unchanged.

Fresh equations, at one authoritative ``q`` on the rows of
positive weight, through the unchanged row kernel
:func:`powermean.model.rows.evaluate_rows`:

    r_i = Z_i exp((1 - tau) q_i) - exp((2 - tau) q_i)     the kernel score, its branches
    d_g = sum a_i,   S_g = sum a_i r_i,   e_g = S_g / d_g  in row order, per level

Two reuse qualifications: (a) this full ``q`` includes
the rows of zero weight, which :class:`powermean.systems.joint.JointSystem` leaves out
of its index arithmetic; the two range domains are not claimed identical, and the
rows of zero weight enter no sum and no equation here; (b) ``evaluate_rows`` also
forms both curvatures and refuses when any of its outputs is not representable, so
an equation check can refuse by the range of a quantity it does not use.  That is a
conservative refusal of this caller, not a score-only representability promise.

Required-quantity guards, distinct from ordinary rounding: the row kernel lets a
genuine final underflow return a zero score, and this caller does not change that
kernel; but for THIS check a zero is insufficient evidence of a zero equation.  An
active score that is exactly zero where ``Z = 0`` or ``log Z != q`` (the nonzero
branches of the kernel), a product
``a_i r_i`` that is zero while both factors are nonzero, a quotient ``S_g / d_g`` that
is zero while ``S_g`` is nonzero, and any nonfinite product, sum or quotient refuse.
A zero score on the kernel's ``log Z == q`` branch, a nonzero subnormal and an exact
cancellation zero pass: no generic inexact-underflow trap is imposed, and nothing is
pre-normalized.  These guards do not promise that every finite mathematical profile
is computable, and they add no row-kernel behavior, tolerance or accuracy bound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb_numeric import (
    PreparedWeights,
    checked_difference,
    checked_sum,
    prepare_weights,
)
from powermean.fe.groups import Block, Grouping
from powermean.fe import reductions as _reductions
from powermean.fe.reductions import group_logsumexp, group_sum, immutable_array
from powermean.model.order import IndexArray
from powermean.model.rows import RowEvaluation, evaluate_criterion, evaluate_rows
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance

Coefficients = tuple[FloatArray, ...]
ProfileSum = Callable[[FloatArray, IndexArray, int, int], FloatArray]
ProfileInput = Callable[[FloatArray], Callable[[IndexArray, int, int], FloatArray] | None]
ProfileReductions = tuple[ProfileSum, ProfileSum, ProfileInput]
BoolArray = np.ndarray[tuple[int], np.dtype[np.bool_]]
RowCacheSlot = tuple[FloatArray, RowEvaluation | None, FloatArray | None]

VARIANCE = "variance must be a PowerVariance"
FAMILY = "variance must be a PowerVariance or an NB2Variance"
OUTCOMES = "z must be a real vector with one finite nonnegative value per row"
BASE = "base must be a real vector with one finite value per row"
SUPPORT = (
    "every level of every block needs a row of positive weight and positive outcome"
)
PREPARED = "prepared must be a PreparedWeights record"
LOGS = "log_weights and log_outcomes must be the logs of the weights and of z"

_EVALUATE_ROWS = evaluate_rows
_EVALUATE_CRITERION = evaluate_criterion


# ------------------------------------------------------------------ the problem


def _outcomes(z: ArrayLike, n_rows: int) -> FloatArray:
    """A finite nonnegative vector of ``n_rows`` doubles, or ``ValueError``."""
    if np.iscomplexobj(z):
        raise ValueError(OUTCOMES)
    found: FloatArray = np.array(z, dtype=np.float64, copy=True)
    if found.shape != (n_rows,) or not np.isfinite(found).all() or (found < 0).any():
        raise ValueError(OUTCOMES)
    return found


def _base(base: ArrayLike, n_rows: int) -> FloatArray:
    """A finite vector of ``n_rows`` doubles, or ``ValueError``."""
    if np.iscomplexobj(base):
        raise ValueError(BASE)
    found: FloatArray = np.array(base, dtype=np.float64, copy=True)
    if found.shape != (n_rows,) or not np.isfinite(found).all():
        raise ValueError(BASE)
    return found


def _logs(weights: FloatArray, z: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``log a`` and ``log Z``, negative infinity at a zero: absent terms."""
    with np.errstate(divide="ignore"):
        return np.log(weights), np.log(z)


def _support(grouping: Grouping, supported: BoolArray) -> None:
    """Every level of every block has a row with ``a > 0`` and ``Z > 0`` (by masks)."""
    for block in grouping.blocks:
        found = np.bincount(block.codes[supported], minlength=block.n_levels)
        if (found == 0).any():
            raise ValueError(SUPPORT)


def _immutable_backing(array: np.ndarray) -> bool:
    """Whether ``array`` is a read-only C-contiguous array backed by full ``bytes``."""
    return (
        not array.flags.writeable
        and array.flags.c_contiguous
        and type(array.base) is bytes
        and len(array.base) == array.nbytes
    )


def _immutable_bool(mask: BoolArray) -> BoolArray:
    """An owned read-only boolean array backed by ``bytes``."""
    view: BoolArray = np.frombuffer(mask.tobytes(), dtype=np.bool_).reshape(mask.shape)
    view.flags.writeable = False
    return view


def _detached_bool(mask: BoolArray) -> BoolArray:
    """An O(1) read-only boolean view with detached array metadata."""
    view: BoolArray = mask.view()
    view.flags.writeable = False
    return view


def _detached_float(values: FloatArray) -> FloatArray:
    """An O(1) read-only float64 view with detached array metadata."""
    view: FloatArray = values.view()
    view.flags.writeable = False
    return view


def _init_problem_caches(
    problem: ProfileProblem, *, shared: ProfileProblem | None = None
) -> None:
    """Attach invariant row masks, log sums and single-slot evaluation state."""
    if shared is not None:
        for attr in (
            "_active_mask",
            "_all_active",
            "_eligible_a",
            "_all_eligible_a",
            "_log_wz",
            "_last_eval",
            "_inv_num",
            "_inv_den",
            "_nb2_means",
        ):
            object.__setattr__(problem, attr, getattr(shared, attr))
        return
    active: BoolArray = _immutable_bool(problem.prepared.weights > 0)
    all_active = bool(active.all())
    eligible_a: BoolArray = _immutable_bool(active & (problem.z > 0))
    all_eligible_a = bool(eligible_a.all())
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        log_wz: FloatArray = immutable_array(
            problem.log_weights + problem.log_outcomes
        )
    variance = problem.variance
    inv_num: dict[int, NumeratorSums | None] | None = None
    inv_den: dict[int, NumeratorSums | None] | None = None
    if type(variance) is PowerVariance and type(variance.tau) is float:
        if variance.tau == 1.0:
            inv_num = {}
        elif variance.tau == 2.0:
            inv_den = {}
    object.__setattr__(problem, "_active_mask", active)
    object.__setattr__(problem, "_all_active", all_active)
    object.__setattr__(problem, "_eligible_a", eligible_a)
    object.__setattr__(problem, "_all_eligible_a", all_eligible_a)
    object.__setattr__(problem, "_log_wz", log_wz)
    object.__setattr__(problem, "_last_eval", [None])
    object.__setattr__(problem, "_inv_num", inv_num)
    object.__setattr__(problem, "_inv_den", inv_den)
    object.__setattr__(problem, "_nb2_means", {})


def _build_profile_problem(
    prepared: PreparedWeights,
    outcomes: FloatArray,
    base: FloatArray,
    variance: PowerVariance | NB2Variance,
    log_weights: FloatArray,
    log_outcomes: FloatArray,
) -> ProfileProblem:
    """Construct a checked ``ProfileProblem`` without duplicating array checks."""
    active: BoolArray = prepared.weights > 0
    supported: BoolArray = active & (outcomes > 0)
    _support(prepared.grouping, supported)
    problem = object.__new__(ProfileProblem)
    object.__setattr__(problem, "prepared", prepared)
    object.__setattr__(problem, "z", immutable_array(outcomes))
    object.__setattr__(problem, "base", immutable_array(base))
    object.__setattr__(problem, "variance", variance)
    object.__setattr__(problem, "log_weights", immutable_array(log_weights))
    object.__setattr__(problem, "log_outcomes", immutable_array(log_outcomes))
    _init_problem_caches(problem)
    return problem


@dataclass(frozen=True, eq=False)
class ProfileProblem:
    """One profiling problem: weights with totals, outcomes, base index, variance.

    Attributes
    ----------
    prepared : PreparedWeights
        The raw weights and the level totals ``d_g`` of the grouping
        (:func:`powermean.fe.absorb_numeric.prepare_weights`): finite nonnegative
        weights, a finite positive total at every level.
    z : array of float
        The transformed outcome, finite and nonnegative, read-only and owned.
    base : array of float
        The base index ``b``, finite, read-only and owned.
    variance : PowerVariance or NB2Variance
        The exact built-in family, by exact type (the exact
        :class:`~powermean.model.variance.NB2Variance` is admitted; a subclass or
        another object is refused).  The power sums, step and loop of this module and of
        :mod:`powermean.fe.profile` are power-only and refuse an NB2 problem; the
        NB2 factory and solver are :mod:`powermean.fe.profile_nb2`.
    log_weights, log_outcomes : array of float
        ``log a`` and ``log Z``, negative infinity at a zero, read-only and owned.

    Notes
    -----
    Constructed by :func:`prepare_profile` (power only) or by the NB2 factory.  A
    direct construction is checked: the arrays as stated, the logs those of the
    weights and the outcomes, and every level of every block supported by a row of
    positive weight and positive outcome.  Changed outcomes, weights or base are a
    new record; nothing is cached.
    """

    prepared: PreparedWeights
    z: FloatArray
    base: FloatArray
    variance: PowerVariance | NB2Variance
    log_weights: FloatArray
    log_outcomes: FloatArray

    def __post_init__(self) -> None:
        """Check the inputs and the support; store owned read-only arrays."""
        if type(self.variance) not in (PowerVariance, NB2Variance):
            raise ValueError(FAMILY)
        if not isinstance(self.prepared, PreparedWeights):
            raise ValueError(PREPARED)
        n_rows = self.prepared.n_rows
        z, base = _outcomes(self.z, n_rows), _base(self.base, n_rows)
        log_weights, log_outcomes = _logs(self.prepared.weights, z)
        given = (np.asarray(self.log_weights), np.asarray(self.log_outcomes))
        if not (
            np.array_equal(given[0], log_weights)
            and np.array_equal(given[1], log_outcomes)
        ):
            raise ValueError(LOGS)
        _support(self.prepared.grouping, (self.prepared.weights > 0) & (z > 0))
        object.__setattr__(self, "z", immutable_array(z))
        object.__setattr__(self, "base", immutable_array(base))
        object.__setattr__(self, "log_weights", immutable_array(log_weights))
        object.__setattr__(self, "log_outcomes", immutable_array(log_outcomes))
        _init_problem_caches(self)

    def _with_base(
        self,
        base: ArrayLike,
        *,
        variance: PowerVariance | NB2Variance | None = None,
    ) -> ProfileProblem:
        """A ``ProfileProblem`` with validated ``base`` sharing invariant state."""
        target_variance = self.variance if variance is None else variance
        if type(target_variance) not in (PowerVariance, NB2Variance):
            raise ValueError(FAMILY)
        if (
            base is self.base
            and _immutable_backing(self.base)
            and target_variance == self.variance
        ):
            return self
        if (
            type(base) is np.ndarray
            and base.dtype == np.float64
            and base.shape == (self.n_rows,)
            and _immutable_backing(base)
        ):
            if not np.isfinite(base).all():
                raise ValueError(BASE)
            checked = immutable_array(base)
        else:
            checked = immutable_array(_base(base, self.n_rows))
        problem = object.__new__(ProfileProblem)
        object.__setattr__(problem, "prepared", self.prepared)
        object.__setattr__(problem, "z", self.z)
        object.__setattr__(problem, "base", checked)
        object.__setattr__(problem, "variance", target_variance)
        object.__setattr__(problem, "log_weights", self.log_weights)
        object.__setattr__(problem, "log_outcomes", self.log_outcomes)
        if variance is None or target_variance == self.variance:
            _init_problem_caches(problem, shared=self)
        else:
            _init_problem_caches(problem)
        return problem

    @property
    def grouping(self) -> Grouping:
        """The accepted grouping of the blocks."""
        return self.prepared.grouping

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return self.prepared.n_rows

    @property
    def active(self) -> BoolArray:
        """The rows of positive weight: the only rows of any sum or equation."""
        mask: BoolArray | None = getattr(self, "_active_mask", None)
        if mask is not None:
            return _detached_bool(mask)
        found: BoolArray = self.prepared.weights > 0
        return found


def prepare_profile(
    grouping: Grouping,
    z: ArrayLike,
    base: ArrayLike,
    weights: ArrayLike,
    *,
    variance: PowerVariance,
) -> ProfileProblem:
    """The checked problem of a grouping, outcomes, base index, weights and variance.

    Parameters
    ----------
    grouping : Grouping
        The accepted grouping (:func:`powermean.fe.groups.encode_blocks`).
    z : array_like
        Transformed outcome, one finite nonnegative value per row.
    base : array_like
        Base index ``b``, one finite value per row, the transformed offset included.
    weights : array_like
        Raw observation weights, one finite nonnegative value per row; every level of
        every block needs a finite positive total (``ValueError`` otherwise, the
        refusal of :func:`~powermean.fe.absorb_numeric.prepare_weights`).
    variance : PowerVariance
        The exact power family; another type is refused, NB2 included (the NB2
        factory is :func:`powermean.fe.profile_nb2.prepare_profile_nb2`).

    ``ValueError`` for malformed inputs and for a level without a row of positive
    weight and positive outcome: a refusal of this inner calculation, not a sample
    deletion, an all-zero-level policy, an existence theorem or a rank decision.
    """
    if type(variance) is not PowerVariance:
        raise ValueError(VARIANCE)
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


# ----------------------------------------------------------- index and update


def reconstruct_index(
    problem: ProfileProblem, coefficients: Coefficients
) -> FloatArray:
    """``q = ((b + alpha_1[c_1]) + alpha_2[c_2]) + ...`` in block order, every row.

    The authoritative index of a coefficient state: gathers added to the base in the
    declared order, each sum range-checked (``FloatingPointError`` on overflow or an
    invalid operation).  No separately drifting incremental index exists.
    """
    q = problem.base
    for block, alpha in zip(problem.grouping.blocks, coefficients, strict=True):
        q = checked_sum(q, alpha[block.codes], "reconstructed index")
    return q


def log_inputs(problem: ProfileProblem, q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``v_A = log a + log Z + (1 - tau) q`` and ``v_B = log a + (2 - tau) q``.

    Absent terms are negative infinity: rows of zero weight in both, rows of zero
    outcome in ``v_A``.  Every eligible term must be finite, and a required exponent
    product ``(1 - tau) q`` or ``(2 - tau) q`` of two nonzero factors that rounded to
    zero on an eligible row is a lost quantity (``FloatingPointError`` in both cases;
    exact-zero factors, a zero index and nonzero subnormal products pass).  Power
    only: an NB2 problem is refused with ``ValueError``, since ``tau`` has no meaning
    there.
    """
    if type(problem.variance) is not PowerVariance:
        raise ValueError(VARIANCE)
    tau = problem.variance.tau
    plain_problem = type(problem) is ProfileProblem
    eligible_a = (
        problem._eligible_a if plain_problem else (problem.active & (problem.z > 0))
    )
    eligible_b = problem._active_mask if plain_problem else problem.active
    all_eligible_a = plain_problem and problem._all_eligible_a
    all_eligible_b = plain_problem and problem._all_active
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        product_a = (1 - tau) * q
        product_b = (2 - tau) * q
        log_wz = (
            problem._log_wz
            if plain_problem
            else (problem.log_weights + problem.log_outcomes)
        )
        numerator = log_wz + product_a
        denominator = problem.log_weights + product_b
    if not all_eligible_a:
        numerator[~eligible_a] = -np.inf
    if not all_eligible_b:
        denominator[~eligible_b] = -np.inf
    finite_a = (
        np.isfinite(numerator).all()
        if all_eligible_a
        else np.isfinite(numerator[eligible_a]).all()
    )
    finite_b = (
        np.isfinite(denominator).all()
        if all_eligible_b
        else np.isfinite(denominator[eligible_b]).all()
    )
    if not (finite_a and finite_b):
        raise FloatingPointError("a log input of the block sums is not finite")
    # The required exponent products: a product of two nonzero factors that rounded
    # to zero on an eligible row is a lost quantity, not the permitted tail of a level
    # sum; a zero factor (tau = 1 in A, tau = 2 in B), a zero index and a nonzero
    # subnormal product are not.  Exact unit factors (+1.0 and -1.0) never round a
    # nonzero IEEE 754 binary64 value to zero.
    for factor, product, eligible, all_eligible in (
        (1 - tau, product_a, eligible_a, all_eligible_a),
        (2 - tau, product_b, eligible_b, all_eligible_b),
    ):
        if factor not in (0.0, 1.0, -1.0):
            lost = (
                ((q != 0) & (product == 0)).any()
                if all_eligible
                else (eligible & (q != 0) & (product == 0)).any()
            )
            if lost:
                raise FloatingPointError("a required exponent product rounded to zero")
    return numerator, denominator


def level_log_sums(values: FloatArray, codes: IndexArray, n_levels: int) -> FloatArray:
    """``L_g = m_g + log sum_i exp(v_i - m_g)`` per level, each finite.

    The reduction :func:`~powermean.fe.reductions.group_logsumexp`: the maximum
    shift, tails that underflow to zero
    allowed, an invalid difference refused.  A level whose sum is not finite, or
    without a finite term, refuses with ``FloatingPointError``.
    """
    sums = group_logsumexp(values, codes, n_levels)
    if not np.isfinite(sums).all():
        raise FloatingPointError("a level log sum is not finite")
    return sums


def block_step(
    problem: ProfileProblem, q: FloatArray, block: int,
    *, _sums: ProfileSum | None = None,
) -> FloatArray:
    """``A_g - B_g`` for every level of ``block`` at the current index ``q``.

    The increment is to be ADDED to the block's current coefficients, since ``q``
    already contains them.  ``FloatingPointError`` when a log input, a level sum or
    the difference is not finite.
    """
    numerator, denominator = log_inputs(problem, q)
    encoded = problem.grouping.blocks[block]
    a = (level_log_sums(numerator, encoded.codes, encoded.n_levels) if _sums is None
         else _sums(numerator, encoded.codes, encoded.n_levels, block))
    b = (level_log_sums(denominator, encoded.codes, encoded.n_levels) if _sums is None
         else _sums(denominator, encoded.codes, encoded.n_levels, block))
    return checked_difference(a, b, "block step")


# The accepted reductions whose call a held numerator replaces: a replaced one, a
# seam of the tests, keeps every call of block_step.
_LEVEL_LOG_SUMS = level_log_sums
_GROUP_LOGSUMEXP = group_logsumexp
_GROUP_SUM = group_sum
_PREPARED_REDUCTIONS = _reductions._prepared_reductions
_LOG_INPUTS = log_inputs


def _profile_reductions() -> ProfileReductions | None:
    """One profile call's empty issuer; never consume caller-held admission state."""
    if _reductions._prepared_reductions is not _PREPARED_REDUCTIONS:
        return None
    sums, logs, inputs = _PREPARED_REDUCTIONS()

    def grouped(values, codes, count, slot):
        if group_sum is _GROUP_SUM and _reductions.group_sum is _GROUP_SUM:
            return sums(values, codes, count, slot)
        return group_sum(values, codes, count)

    def logged(values, codes, count, slot):
        if (level_log_sums is _LEVEL_LOG_SUMS
                and group_logsumexp is _GROUP_LOGSUMEXP
                and _reductions.group_logsumexp is _GROUP_LOGSUMEXP):
            try:
                prepared = (inputs(values, logarithmic=True)
                            if log_inputs is _LOG_INPUTS else None)
            except MemoryError:
                prepared = None
            found = (logs(values, codes, count, slot) if prepared is None
                     else prepared(codes, count, slot))
            if not np.isfinite(found).all():
                raise FloatingPointError("a level log sum is not finite")
            return found
        return level_log_sums(values, codes, count)

    def channel(values):
        try:
            if (required_product is _REQUIRED_PRODUCT and group_sum is _GROUP_SUM
                    and _reductions.group_sum is _GROUP_SUM):
                return inputs(values)
        except MemoryError:
            pass
        return None

    return grouped, logged, channel


@dataclass(frozen=True, eq=False)
class NumeratorSums:
    """The numerator level log sums of one block and the inputs they were formed from.

    Attributes
    ----------
    values : array of float
        A private copy of the log inputs ``v_A`` given to :func:`level_log_sums`.
    codes : array of int
        A private copy of the block's codes given with them.
    n_levels : int
        The block's level count given with them.
    sums : array of float
        ``A_g``, what :func:`level_log_sums` returned for these inputs.

    Notes
    -----
    Formed by :func:`invariant_block_step` at a visit of a block and held by one
    profile call for its next visit of that block; never shared between calls.
    """

    values: FloatArray
    codes: IndexArray
    n_levels: int
    sums: FloatArray


def _plain(values: object, encoded: object) -> bool:
    """Whether ``level_log_sums(values, encoded.codes, encoded.n_levels)`` runs no hook.

    Exact NumPy vectors and an exact :class:`~powermean.fe.groups.Block`, read in
    that order, and the accepted reductions: skipping that call then changes nothing
    but its cost.  The arguments are typed as ``object`` because this is the run-time
    check of their types.
    """
    return (
        level_log_sums is _LEVEL_LOG_SUMS
        and group_logsumexp is _GROUP_LOGSUMEXP
        and type(values) is np.ndarray
        and values.dtype == np.float64
        and values.ndim == 1
        and type(encoded) is Block
        and type(encoded.codes) is np.ndarray
        and encoded.codes.dtype == np.intp
        and type(encoded.counts) is np.ndarray
    )


def _same_inputs(held: NumeratorSums, values: FloatArray, encoded: Block) -> bool:
    """Whether ``values``, the codes and the level count are bitwise ``held``'s."""
    return (
        encoded.n_levels == held.n_levels
        and (
            (encoded.codes is held.codes and _immutable_backing(encoded.codes))
            or np.array_equal(encoded.codes, held.codes)
        )
        and (
            (
                _immutable_backing(values)
                and _immutable_backing(held.values)
                and (values is held.values or values.base is held.values.base)
            )
            or np.array_equal(values.view(np.uint64), held.values.view(np.uint64))
        )
    )


def invariant_block_step(
    problem: ProfileProblem, q: FloatArray, block: int, held: NumeratorSums | None,
    *, _sums: ProfileSum | None = None,
) -> tuple[FloatArray, NumeratorSums | None]:
    """:func:`block_step`, with the numerator sums of ``held`` where its inputs recur.

    The step of the power loop at EXACT ``tau = 1`` (module, numerator reuse):
    :func:`log_inputs` at the current ``q`` with every check, then the numerator
    sums, the denominator sums and the checked difference, as :func:`block_step`
    forms them.  The numerator sums of ``held`` stand for the call of
    :func:`level_log_sums` only on plain data (:func:`_plain`) whose ``v_A``, codes
    and level count are bitwise those of ``held``; otherwise that call runs, and its
    inputs and sums are held for the next visit (``None`` on data that is not plain).
    With ``held`` from an earlier call of this function, or ``None``, the step and
    every refusal are those of :func:`block_step`, at any ``tau``.

    Returns
    -------
    tuple
        The step, and the numerator sums to hold for the next visit of ``block``.
    """
    numerator, denominator = log_inputs(problem, q)
    encoded = problem.grouping.blocks[block]
    if (
        held is not None
        and _plain(numerator, encoded)
        and _same_inputs(held, numerator, encoded)
    ):
        a = held.sums
    else:
        a = (level_log_sums(numerator, encoded.codes, encoded.n_levels) if _sums is None
             else _sums(numerator, encoded.codes, encoded.n_levels, block))
        held = None
        if _plain(numerator, encoded):
            codes = (
                encoded.codes
                if _immutable_backing(encoded.codes)
                else encoded.codes.copy()
            )
            held = NumeratorSums(numerator.copy(), codes, encoded.n_levels, a)
    b = (level_log_sums(denominator, encoded.codes, encoded.n_levels) if _sums is None
         else _sums(denominator, encoded.codes, encoded.n_levels, block))
    return checked_difference(a, b, "block step"), held


def invariant_denominator_block_step(
    problem: ProfileProblem, q: FloatArray, block: int, held: NumeratorSums | None,
    *, _sums: ProfileSum | None = None,
) -> tuple[FloatArray, NumeratorSums | None]:
    """:func:`block_step`, with the denominator sums of ``held`` where inputs recur.

    At EXACT ``tau = 2`` (GPML), ``(2 - tau) q_i`` is zero on every finite ``q``, so
    ``v_B`` is ``log a_i`` on active rows and ``B_g`` is invariant across visits when
    ``v_B``, codes and level count match ``held`` bit for bit.  Forms
    :func:`log_inputs`, the numerator sums ``a``, the denominator sums ``b`` and the
    checked difference in the exact order of :func:`block_step`.
    """
    numerator, denominator = log_inputs(problem, q)
    encoded = problem.grouping.blocks[block]
    a = (level_log_sums(numerator, encoded.codes, encoded.n_levels) if _sums is None
         else _sums(numerator, encoded.codes, encoded.n_levels, block))
    if (
        held is not None
        and _plain(denominator, encoded)
        and _same_inputs(held, denominator, encoded)
    ):
        b = held.sums
    else:
        b = (level_log_sums(denominator, encoded.codes, encoded.n_levels) if _sums is None
             else _sums(denominator, encoded.codes, encoded.n_levels, block))
        held = None
        if _plain(denominator, encoded):
            codes = (
                encoded.codes
                if _immutable_backing(encoded.codes)
                else encoded.codes.copy()
            )
            held = NumeratorSums(denominator.copy(), codes, encoded.n_levels, b)
    return checked_difference(a, b, "block step"), held


def updated_coefficients(
    coefficients: Coefficients, block: int, step: FloatArray
) -> Coefficients:
    """Fresh coefficient tuple with ``alpha_block + step``, range-checked."""
    found = list(coefficients)
    found[block] = checked_sum(coefficients[block], step, "updated coefficients")
    return tuple(found)


# ------------------------------------------------------------ fresh equations


def _representable(operation: Callable[[], FloatArray], what: str) -> FloatArray:
    """Run ``operation`` with overflow, invalid and zero division raised, named."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            return operation()
        except FloatingPointError as error:
            raise FloatingPointError(f"{what}: {error}") from error


def _same_q(a: FloatArray, b: FloatArray) -> bool:
    """Whether two float64 vectors share the exact same IEEE 754 bit pattern."""
    if (
        _immutable_backing(a)
        and _immutable_backing(b)
        and (a is b or a.base is b.base)
    ):
        return True
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def _cached_rows(problem: ProfileProblem, q: FloatArray) -> RowEvaluation:
    """``evaluate_rows(problem.z, q, variance=problem.variance)`` with 1-slot reuse."""
    if (
        type(problem) is ProfileProblem
        and problem._all_active
        and evaluate_rows is _EVALUATE_ROWS
        and type(q) is np.ndarray
        and q.dtype == np.float64
        and q.shape == (problem.n_rows,)
    ):
        slot: RowCacheSlot | None = problem._last_eval[0]
        if slot is not None and _same_q(q, slot[0]):
            if slot[1] is not None:
                return slot[1]
            rows = evaluate_rows(problem.z, q, variance=problem.variance)
            problem._last_eval[0] = (slot[0], rows, slot[2])
            return rows
        rows = evaluate_rows(problem.z, q, variance=problem.variance)
        key = immutable_array(q)
        problem._last_eval[0] = (key, rows, None)
        return rows
    return evaluate_rows(problem.z, q, variance=problem.variance)


def _cached_criterion(problem: ProfileProblem, q: FloatArray) -> FloatArray:
    """``evaluate_criterion(problem.z, q, variance=...)`` with 1-slot reuse."""
    if (
        type(problem) is ProfileProblem
        and problem._all_active
        and evaluate_criterion is _EVALUATE_CRITERION
        and type(q) is np.ndarray
        and q.dtype == np.float64
        and q.shape == (problem.n_rows,)
    ):
        slot: RowCacheSlot | None = problem._last_eval[0]
        if slot is not None and _same_q(q, slot[0]):
            if slot[2] is not None:
                return _detached_float(slot[2])
            values = evaluate_criterion(problem.z, q, variance=problem.variance)
            problem._last_eval[0] = (slot[0], slot[1], values)
            return _detached_float(values)
        values = evaluate_criterion(problem.z, q, variance=problem.variance)
        key = immutable_array(q)
        problem._last_eval[0] = (key, None, values)
        return _detached_float(values)
    return evaluate_criterion(problem.z, q, variance=problem.variance)


def active_scores(problem: ProfileProblem, q: FloatArray) -> FloatArray:
    """The row scores ``r_i`` on the rows of positive weight, with the score guard.

    :func:`~powermean.model.rows.evaluate_rows`, unchanged, on the active rows at the
    index ``q``; it may refuse by the range of any of its outputs (qualification (b)
    of the module).  A score that is exactly zero off the kernel's ``log Z == q``
    branch is a lost required quantity and refuses with ``FloatingPointError``.
    """
    plain_problem = type(problem) is ProfileProblem
    if (
        plain_problem
        and problem._all_active
        and type(q) is np.ndarray
        and q.dtype == np.float64
        and q.shape == (problem.n_rows,)
    ):
        z, index = problem.z, q
        score = _cached_rows(problem, q).score
    else:
        active = problem._active_mask if plain_problem else problem.active
        z, index = problem.z[active], q[active]
        score = evaluate_rows(z, index, variance=problem.variance).score
    return _checked_active_score(z, index, score)


def _checked_active_score(
    z: FloatArray, index: FloatArray, score: FloatArray,
) -> FloatArray:
    """Keep the required-zero predicate on either fresh source of the same rows."""
    zero_score = score == 0
    if zero_score.any():
        z_zero = z[zero_score]
        if (z_zero <= 0).any() or (np.log(z_zero) != index[zero_score]).any():
            raise FloatingPointError("a required score rounded to zero")
    return _detached_float(score)


def required_product(weights: FloatArray, score: FloatArray) -> FloatArray:
    """``a_i r_i`` row by row; a zero product of two nonzero factors refuses."""
    products = _representable(lambda: np.multiply(weights, score), "weighted score")
    if ((products == 0) & (weights != 0) & (score != 0)).any():
        raise FloatingPointError("a required weighted score rounded to zero")
    return products


_REQUIRED_PRODUCT = required_product


def required_quotient(sums: FloatArray, totals: FloatArray) -> FloatArray:
    """``S_g / d_g`` per level; a zero quotient of a nonzero sum refuses."""
    quotient = _representable(lambda: np.divide(sums, totals), "level equation")
    if ((quotient == 0) & (sums != 0)).any():
        raise FloatingPointError("a required level equation rounded to zero")
    return quotient


@dataclass(frozen=True, eq=False)
class EquationCheck:
    """The signed equations ``e_g`` of every block at one index, and their maximum.

    Attributes
    ----------
    errors : tuple of arrays of float
        One vector per block, ``e_g`` in level order, read-only and owned.
    maximum : float
        ``max |e_g|`` over every block and level.
    """

    errors: tuple[FloatArray, ...]
    maximum: float

    def __post_init__(self) -> None:
        """Own the errors, each finite; the maximum must be theirs."""
        owned = tuple(immutable_array(e) for e in self.errors)
        if not owned or any(
            e.ndim != 1 or not e.size or not np.isfinite(e).all() for e in owned
        ):
            raise ValueError("errors must be one finite vector per block")
        if self.maximum != max(float(np.max(np.abs(e))) for e in owned):
            raise ValueError("maximum must be the largest absolute error")
        object.__setattr__(self, "errors", owned)

    def passing(self, tolerance: float) -> bool:
        """Whether ``max |e_g| <= tolerance``; equality passes."""
        return self.maximum <= tolerance


def fresh_equations(
    problem: ProfileProblem, q: FloatArray, *, _sums: ProfileSum | None = None,
    _inputs: ProfileInput | None = None,
) -> EquationCheck:
    """All ``e_g = S_g / d_g`` of every ORIGINAL block at the one index ``q``.

    The scores of :func:`active_scores`, the products of :func:`required_product`,
    the row-order level sums of the accepted :func:`~powermean.fe.reductions
    .group_sum` (overflow and invalid refused) and the quotients of
    :func:`required_quotient` by the prepared raw totals.  A refusal anywhere raises
    ``FloatingPointError`` and no check exists for that index; nothing is replaced
    by zero or by a passing value.
    """
    return _equations(problem, q, None, _sums, _inputs)


def _fresh_equations_from_rows(
    problem: ProfileProblem, q: FloatArray, rows: RowEvaluation,
) -> EquationCheck:
    """Consume this certificate invocation's fresh rows at its admitted all-active q.

    The certificate authenticates the original producers and matches response,
    family, weights and index before selecting this private composition. Nothing
    is read from or written to a profile/solver row cache.
    """
    return _equations(problem, q, rows, None)


def _equations(
    problem: ProfileProblem, q: FloatArray, rows: RowEvaluation | None,
    sums_for: ProfileSum | None, inputs: ProfileInput | None = None,
) -> EquationCheck:
    """The one ordered equation assembly, with every product/sum/quotient check."""
    plain_problem = type(problem) is ProfileProblem
    all_active = plain_problem and problem._all_active
    active = problem._active_mask if plain_problem else problem.active
    weights = (
        problem.prepared.weights if all_active else problem.prepared.weights[active]
    )
    score = (active_scores(problem, q) if rows is None
             else _checked_active_score(problem.z, q, rows.score))
    products = required_product(weights, score)
    errors: list[FloatArray] = []
    prepared = None
    for slot, (block, totals) in enumerate(zip(
        problem.grouping.blocks, problem.prepared.totals, strict=True
    )):
        codes = block.codes if all_active else block.codes[active]
        count = block.n_levels
        if slot == 0 and inputs is not None:
            try:
                prepared = inputs(products)
            except MemoryError:
                prepared = None
        sums = (prepared(codes, count, slot) if prepared is not None else
                group_sum(products, codes, count) if sums_for is None else
                sums_for(products, codes, count, slot))
        errors.append(required_quotient(sums, totals))
    return EquationCheck(tuple(errors), max(float(np.max(np.abs(e))) for e in errors))
