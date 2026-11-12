"""The profiled joint system with absorbed effects: problem, points, equations, Schur.

The profiled organization: every block is profiled at the slopes, the gradient
``X'r`` is taken at the profiled effects, and the Newton system is formed by weighted
absorption of the regressors with the curvature weights.  The controls are taken here
as explicit primitive arguments and never chosen.  The organization covers any number
of blocks, NB2, weights and offsets, with no two-block restriction, coefficient-only
cache, roundoff allowance or broad exception catch.  This module is internal:
``powermean`` exports nothing of it.

Notation.  ``z = Y^omega >= 0`` is the transformed outcome, ``X`` the active scaled
slopes (``n`` by ``K``, ``K >= 0``) of coordinates ``theta``, ``o`` the transformed
offset (``omega`` times the offset), ``a > 0`` the raw observation weights of total
``A``, and the blocks those of the grouping.  At the slopes ``theta``:

    base             b = X theta + o                    the offset enters once
    profiled index   q = b + sum_k alpha_k[c_k]           every block profiled
    criterion        L/A = sum_i a_i l(q_i; z_i) / A
    true gradient    -X'(a r)/A at the same q
    direction        X~: X absorbed with the weights a c (observed) or a w
                     (expected), and the K-by-K matrix X~' diag(a c) X~ (or a w)

The profile routine is used unchanged: :func:`profile` for the power
family and :func:`profile_nb2` for NB2, each from a recorded starting state, at the
requested tolerance and sweep budget, returning ``converged``, ``budget`` or ``range``
with its work.  The criterion and gradient are formed in the order of operations of the
carried ``LinearModelLoss`` (the weight times the row term, the sum or the division by
the weight total, then ``X'``), so that they reproduce the values of the solver's
pointwise loss at the same index.  The expected curvature changes the direction only:
the criterion, the true gradient and the profile equations are the same for both.  The
weighted absorption is the projection of :mod:`powermean.fe.absorb`, from a cold
start unless a direction hands :func:`_absorb_slopes` its warm slope state; a level
of a block whose curvature total is not finite and positive has no Newton system and
refuses with ``FloatingPointError``: no floor, ridge or individual curvature division
(a zero row at ``tau = 2`` legitimately has no curvature; its level keeps a positive
total through its positive rows).  Nothing is cached; a changed outcome, weight, base or
control is a new evaluation.

A completed absorption is kept whatever follows: when its Schur matrix is
not representable, :func:`absorb_slopes` returns the absorption's own record with no
matrix and the range failure named, so that the work and status of the absorber survive
the refusal; :func:`absorb_design` refuses as before.  No scaled substitute, clipping or
other matrix is formed.  The private :func:`_absorb_slopes` is the same absorption that
also returns the fresh record of the weights it prepared, for the full-RHS solve of the
same fitted-index direction only.  Given a direction's warm slope state (the owned
coefficients of :func:`_slope_state`, bound to the grouping and the ordered slope
columns), it starts that absorption there: the absorber checks the binding, rebuilds
``X - lift(a)`` and checks it afresh with the weights this call prepared, at this call's
tolerance.  The public :func:`absorb_slopes` and :func:`absorb_design` stay cold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb import AbsorptionResult, AbsorptionState, absorb
from powermean.fe.absorb_numeric import (
    PreparedWeights,
    _from_checked_weights,
    group_sums,
    prepare_weights,
)
from powermean.fe.groups import Grouping
from powermean.fe.profile import ProfileResult, ProfileState, profile
from powermean.fe.profile_arithmetic import (
    ProfileProblem,
    _cached_criterion,
    _cached_rows,
    prepare_profile,
)
from powermean.fe.profile_nb2 import ScalarWork, prepare_profile_nb2, profile_nb2
from powermean.fe.reductions import immutable_array
from powermean.model.rows import evaluate_criterion, evaluate_rows
from powermean.data.numeric_fit import NumericFit
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance

Family = PowerVariance | NB2Variance
SCOPE = "the profiled solver covers PowerVariance with 1 <= tau <= 2 and NB2Variance"
SLOPES = "theta must hold one finite value per slope column"
SCHUR = "the Schur matrix is not representable in float64"

_ACCEPTED_PREPARE_PROFILE = prepare_profile
_ACCEPTED_PREPARE_NB2 = prepare_profile_nb2
_ACCEPTED_PREPARE_WEIGHTS = prepare_weights
_ACCEPTED_CRITERION = evaluate_criterion
_ACCEPTED_ROWS = evaluate_rows


def _real(value: ArrayLike, name: str, shape: tuple[int, ...]) -> FloatArray:
    """An owned finite real array of ``shape``, or ``ValueError``."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    found: FloatArray = np.array(value, dtype=np.float64, copy=True)
    if found.shape != shape or not np.isfinite(found).all():
        raise ValueError(f"{name} must be finite, of shape {shape}")
    return found


def _build_initial(problem: ProfiledProblem, base: FloatArray) -> ProfileProblem:
    """The freshly constructed profile problem of the family at ``base``."""
    variance = problem.variance
    if isinstance(variance, PowerVariance):
        return prepare_profile(
            problem.grouping, problem.z, base, problem.weights, variance=variance
        )
    return prepare_profile_nb2(
        problem.grouping, problem.z, base, problem.weights, variance=variance
    )


@dataclass(frozen=True, eq=False)
class ProfiledProblem:
    """The profiled joint problem of one sample: arrays, blocks and working variance.

    Attributes
    ----------
    z : array of float
        The transformed outcome, finite and nonnegative, one value per row.
    design : array of float
        The active scaled slopes ``X``, ``n`` by ``K``; ``K = 0`` for effects only.
    offset : array of float
        The transformed offset ``o``, one finite value per row.
    weights : array of float
        The raw observation weights ``a``, finite and positive, one per row.
    grouping : Grouping
        The blocks of the rows.
    variance : PowerVariance or NB2Variance
        ``tau`` in ``[1, 2]``, or NB2: the convex scope of the solver.
    columns : tuple of str
        The distinct names of the slope columns: the identity of absorbed columns.

    Notes
    -----
    The arrays are stored as owned read-only copies.  Every level of every block must
    hold a row of positive outcome, which the sample construction guarantees; the
    profile factory checks it here, once, and refuses otherwise with ``ValueError``,
    as it refuses malformed arrays.
    """

    z: FloatArray
    design: FloatArray
    offset: FloatArray
    weights: FloatArray
    grouping: Grouping
    variance: Family
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check every field and the support of every level; own the arrays."""
        variance = self.variance
        if not (
            type(variance) is NB2Variance
            or (type(variance) is PowerVariance and 1.0 <= variance.tau <= 2.0)
        ):
            raise ValueError(SCOPE)
        if not isinstance(self.grouping, Grouping):
            raise ValueError("grouping must be a Grouping record")
        n = self.grouping.n_rows
        design = np.asarray(self.design)
        if design.ndim != 2:
            raise ValueError("design must be a matrix with one row per observation")
        columns = tuple(self.columns)
        if len(columns) != design.shape[1] or len(set(columns)) != len(columns):
            raise ValueError("provide one distinct column name per slope")
        found = {
            "z": _real(self.z, "z", (n,)),
            "design": _real(design, "design", (n, len(columns))),
            "offset": _real(self.offset, "offset", (n,)),
            "weights": _real(self.weights, "weights", (n,)),
        }
        if (found["z"] < 0).any() or (found["weights"] <= 0).any():
            raise ValueError("z must be nonnegative and the weights positive")
        for name, value in found.items():
            object.__setattr__(self, name, immutable_array(value))
        object.__setattr__(self, "columns", columns)
        initial = _prepared(self, self.offset)
        object.__setattr__(self, "_initial", initial)

    @classmethod
    def _from_numeric(cls, numeric: NumericFit, variance: Family) -> ProfiledProblem:
        """Read one admitted operation's numeric owner; public construction is unchanged."""
        if not (type(variance) is NB2Variance or
                (type(variance) is PowerVariance and 1.0 <= variance.tau <= 2.0)):
            raise ValueError(SCOPE)
        result = object.__new__(cls)
        for name, value in (
            ("z", numeric.response), ("design", numeric.design),
            ("offset", numeric.offset), ("weights", numeric.weights),
            ("grouping", numeric._grouping), ("variance", variance),
            ("columns", numeric.columns),
        ):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_initial", _build_initial(result, result.offset))
        object.__setattr__(result, "_numeric_owner", numeric)
        return result

    def _fresh_certificate_problem(self) -> ProfiledProblem:
        """Share fixed inputs, never the solver's evaluated rows or final equations."""
        result = object.__new__(ProfiledProblem)
        for name in ("z", "design", "offset", "weights"):
            object.__setattr__(result, name, getattr(self, name).view())
        for name in ("grouping", "variance", "columns"):
            object.__setattr__(result, name, getattr(self, name))
        initial: ProfileProblem = getattr(self, "_initial")
        # A different immutable header guarantees _with_base constructs a new
        # prepared record. Constant family terms may be shared; q-dependent
        # evaluation state is local to this fresh certificate.
        fresh = initial._with_base(initial.base.view())
        object.__setattr__(fresh, "_last_eval", [None])
        object.__setattr__(fresh, "_nb2_means", {})
        object.__setattr__(result, "_initial", fresh)
        owner = getattr(self, "_numeric_owner", None)
        if owner is not None:
            object.__setattr__(result, "_numeric_owner", owner)
        return result

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return int(self.z.size)

    @property
    def n_slopes(self) -> int:
        """The number of slope columns ``K``."""
        return int(self.design.shape[1])


def _prepared(problem: ProfiledProblem, base: FloatArray) -> ProfileProblem:
    """The accepted profile problem of the family at ``base`` (its checks included)."""
    if (
        type(problem) is ProfiledProblem
        and prepare_profile is _ACCEPTED_PREPARE_PROFILE
        and prepare_profile_nb2 is _ACCEPTED_PREPARE_NB2
    ):
        initial: ProfileProblem | None = getattr(problem, "_initial", None)
        if initial is not None:
            return initial._with_base(base)
    return _build_initial(problem, base)


def _base(problem: ProfiledProblem, coefficients: FloatArray) -> FloatArray:
    """``X theta + o`` of checked slopes; ``FloatingPointError`` if unrepresentable."""
    with np.errstate(over="ignore", invalid="ignore"):
        base: FloatArray = problem.design @ coefficients + problem.offset
    if not np.isfinite(base).all():
        raise FloatingPointError("the base index is not representable in float64")
    return immutable_array(base)


def base_index(problem: ProfiledProblem, theta: ArrayLike) -> FloatArray:
    """``b = X theta + o``: the slopes' index with the transformed offset, once.

    ``ValueError`` for slopes that are not one finite value per column;
    ``FloatingPointError`` for a base that is not representable in float64.
    """
    return _base(problem, _real(theta, "theta", (problem.n_slopes,)))


@dataclass(frozen=True, eq=False)
class PointProfile:
    """One profile of every block at the slopes ``theta``, whatever its status.

    Attributes
    ----------
    theta : array of float
        The slopes, owned and read-only.
    base : array of float
        ``X theta + o``, owned and read-only.
    result : ProfileResult
        The accepted record: index, state, start, status, equations and block work.
    scalar : ScalarWork or None
        The NB2 scalar work; ``None`` for the power family.
    """

    theta: FloatArray
    base: FloatArray
    result: ProfileResult
    scalar: ScalarWork | None

    def __post_init__(self) -> None:
        """Own the slopes."""
        object.__setattr__(self, "theta", immutable_array(self.theta))

    @property
    def converged(self) -> bool:
        """Whether the fresh check of every block passed at the returned index."""
        return self.result.converged

    @property
    def index(self) -> FloatArray:
        """The reconstructed index of the returned state, every row."""
        return self.result.index

    @property
    def state(self) -> ProfileState:
        """The block coefficients of the returned state."""
        return self.result.state


def profile_point(
    problem: ProfiledProblem,
    theta: ArrayLike,
    *,
    tolerance: float,
    max_sweeps: int,
    state: ProfileState | None = None,
) -> PointProfile:
    """Profile every block at the slopes ``theta`` from ``state`` (cold if omitted).

    The accepted routine of the family, unchanged, on the base of :func:`base_index`;
    its statuses, refusals and work are returned as they are.  ``FloatingPointError``
    when the base or the starting index is not representable (no profile ran).
    """
    coefficients = _real(theta, "theta", (problem.n_slopes,))
    base = _base(problem, coefficients)
    prepared = _prepared(problem, base)
    if type(problem.variance) is PowerVariance:
        result = profile(
            prepared, tolerance=tolerance, max_sweeps=max_sweeps, state=state
        )
        return PointProfile(coefficients, base, result, None)
    found = profile_nb2(
        prepared, tolerance=tolerance, max_sweeps=max_sweeps, state=state
    )
    return PointProfile(coefficients, base, found.profile, found.scalar)


@dataclass(frozen=True, eq=False)
class PointEquations:
    """The criterion, true gradient and curvature weights at one profiled index.

    Attributes
    ----------
    loss : float
        ``L/A = sum_i a_i l_i / A``.
    gradient : array of float
        ``-X'(a r)/A``, one value per slope column.
    curvature : array of float
        ``a c``, the weighted observed curvature of every row.
    information : array of float
        ``a w``, the weighted information of every row.
    """

    loss: float
    gradient: FloatArray
    curvature: FloatArray
    information: FloatArray


def point_equations(problem: ProfiledProblem, index: ArrayLike) -> PointEquations:
    """The criterion, gradient and curvature weights of the rows at ``index``.

    The order of operations of the carried ``LinearModelLoss``: the weight times the
    row criterion summed and divided by ``A``; the weight times ``-r`` divided by
    ``A``, then ``X'``.  ``FloatingPointError`` where a row or an aggregate is not
    representable (nothing is clipped).
    """
    q = _real(index, "index", (problem.n_rows,))
    initial: ProfileProblem | None = (
        getattr(problem, "_initial", None)
        if type(problem) is ProfiledProblem
        else None
    )
    if (
        initial is not None
        and evaluate_criterion is _ACCEPTED_CRITERION
        and evaluate_rows is _ACCEPTED_ROWS
    ):
        values = _cached_criterion(initial, q)
        rows = _cached_rows(initial, q)
    else:
        values = evaluate_criterion(problem.z, q, variance=problem.variance)
        rows = evaluate_rows(problem.z, q, variance=problem.variance)
    weights = problem.weights
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.sum(weights)
        loss = (weights * values).sum() / total
        gradient: FloatArray = problem.design.T @ ((weights * -rows.score) / total)
        curvature: FloatArray = weights * rows.observed_curvature
        information: FloatArray = weights * rows.fisher_curvature
    found = (gradient, curvature, information)
    if not math.isfinite(loss) or not all(np.isfinite(a).all() for a in found):
        raise FloatingPointError("the row equations are not representable in float64")
    return PointEquations(float(loss), *(immutable_array(a) for a in found))


@dataclass(frozen=True, eq=False)
class Absorbed:
    """The slopes absorbed with curvature weights: the Schur state of one index.

    Attributes
    ----------
    weights : array of float
        The weights of the projection, ``a c`` or ``a w``, owned and read-only.
    result : AbsorptionResult
        The accepted record: the absorbed design ``X~`` (its residual), status and work.
    matrix : array of float or None
        ``X~' diag(weights) X~``, the K-by-K Schur complement of the effects, when the
        absorption converged and that matrix is representable; ``None`` otherwise.
    failure : str
        The range failure of the Schur matrix after a converged absorption
        (:data:`SCHUR`), empty when there was none; ``result`` is then the completed
        absorption, its status and work as returned.
    """

    weights: FloatArray
    result: AbsorptionResult
    matrix: FloatArray | None
    failure: str = ""


def _prepare_curvature(problem: ProfiledProblem, weights: ArrayLike) -> PreparedWeights:
    """Check the curvature weights and their positive level totals in one pass."""
    found: FloatArray = np.array(weights, dtype=np.float64, copy=True)
    if found.shape != (problem.n_rows,):
        raise ValueError("provide one curvature weight per row")
    if not np.isfinite(found).all() or (found < 0).any():
        raise FloatingPointError("the curvature weights are not finite and nonnegative")
    totals: list[FloatArray] = []
    for block in problem.grouping.blocks:
        level_totals = group_sums(found, block.codes, block.n_levels)
        if not (level_totals > 0).all():
            raise FloatingPointError(
                f"a level of block {block.name!r} has no positive curvature total"
            )
        totals.append(immutable_array(level_totals))
    return _from_checked_weights(
        problem.grouping, immutable_array(found), tuple(totals)
    )


def curvature_totals(problem: ProfiledProblem, weights: ArrayLike) -> FloatArray:
    """The checked curvature weights, refused unless every level total is positive.

    ``ValueError`` for weights of another shape.  ``FloatingPointError`` for a weight
    that is not finite and nonnegative, a level total that is not representable (the
    accepted checked sum), or a level whose total is not positive, naming its block:
    that Newton system has no solution, and nothing is floored.
    """
    return _prepare_curvature(problem, weights).weights


_ACCEPTED_CURVATURE_TOTALS = curvature_totals


def absorb_slopes(
    problem: ProfiledProblem,
    weights: ArrayLike,
    *,
    tolerance: float,
    max_sweeps: int,
    acceleration: bool,
) -> Absorbed:
    """Absorb the slopes with the curvature ``weights``; their Schur matrix if it can.

    The weighted absorption from a cold start.  ``ValueError`` without a slope
    column; ``FloatingPointError`` for a level without a positive curvature total (no
    absorption ran).  The absorption's statuses and work are returned as they are, and
    kept whatever follows: the matrix is formed only when it converged, and when that
    matrix is not representable the completed absorption is returned with no matrix and
    ``failure`` :data:`SCHUR`.
    """
    absorbed, _ = _absorb_slopes(
        problem,
        weights,
        tolerance=tolerance,
        max_sweeps=max_sweeps,
        acceleration=acceleration,
    )
    return absorbed


def _absorb_slopes(
    problem: ProfiledProblem,
    weights: ArrayLike,
    *,
    tolerance: float,
    max_sweeps: int,
    acceleration: bool,
    state: AbsorptionState | None = None,
) -> tuple[Absorbed, PreparedWeights]:
    """:func:`absorb_slopes`, with the one record of ``weights`` that it prepared.

    The same work in the same order: ``ValueError`` without a slope column, then the
    existing chain, :func:`curvature_totals` and the accepted
    :func:`~powermean.fe.absorb_numeric.prepare_weights` of its checked copy, each
    refusal raised where it was; the absorption with that record and, when it
    converged, the Schur matrix with the record's own weights.  The record is formed
    from this call's ``weights`` alone and kept nowhere: the fitted-index direction
    hands it to its full-RHS solve, which then prepares nothing again.

    ``state`` is a direction's warm slope state (:func:`_slope_state`), handed to the
    accepted absorber as its start: the absorber refuses a state bound to another
    grouping or other columns (``ValueError``, no cold substitute) and checks the
    rebuilt candidate afresh with this record at ``tolerance``.  A seed is eligible
    only if every freshly checked current weight is strictly positive.  Otherwise
    the call is the original cold calculation, with every row retained.  This scan
    follows the original weight validation and precedes any seed reconstruction;
    there is no retry after an absorption failure.
    """
    if not problem.n_slopes:
        raise ValueError("there is no slope column to absorb")
    if (
        curvature_totals is _ACCEPTED_CURVATURE_TOTALS
        and prepare_weights is _ACCEPTED_PREPARE_WEIGHTS
    ):
        prepared = _prepare_curvature(problem, weights)
    else:
        prepared = prepare_weights(problem.grouping, curvature_totals(problem, weights))
    found = prepared.weights
    if state is not None and not _positive_slope_curvature(found):
        state = None
    if state is None:
        result = absorb(
            problem.design,
            prepared,
            problem.columns,
            tolerance=tolerance,
            max_sweeps=max_sweeps,
            acceleration=acceleration,
        )
    else:
        result = absorb(
            problem.design,
            prepared,
            problem.columns,
            tolerance=tolerance,
            max_sweeps=max_sweeps,
            acceleration=acceleration,
            state=state,
        )
    if not result.converged:
        return Absorbed(found, result, None), prepared
    residual = result.residual
    with np.errstate(over="ignore", invalid="ignore"):
        matrix: FloatArray = residual.T @ (found[:, None] * residual)
    if not np.isfinite(matrix).all():
        return Absorbed(found, result, None, SCHUR), prepared
    return Absorbed(found, result, immutable_array(matrix)), prepared


def _positive_slope_curvature(weights: FloatArray) -> bool:
    """Warm eligibility on freshly validated current weights, counting every row."""
    return bool((weights > 0.0).all())


def _slope_state(result: AbsorptionResult) -> AbsorptionState:
    """The warm slope state of a converged slope absorption: its owned coefficients.

    A new checked :class:`~powermean.fe.absorb.AbsorptionState` of the result's block
    coefficients, with their binding (the grouping and the ordered slope columns), in
    arrays of its own; nothing else of the result (residual, status, tolerance, work
    or weights) is kept.
    """
    state = result.state
    return AbsorptionState(state.grouping, state.columns, state.coefficients)


def absorb_design(
    problem: ProfiledProblem,
    weights: ArrayLike,
    *,
    tolerance: float,
    max_sweeps: int,
    acceleration: bool,
) -> Absorbed:
    """Absorb the slopes with the curvature ``weights`` (cold start).

    ``ValueError`` without a slope column; ``FloatingPointError`` for a level without a
    positive curvature total (no absorption ran) and for a Schur matrix that is not
    representable (:func:`absorb_slopes` keeps that completed absorption).  The
    absorption's statuses and work are returned as they are; the matrix is formed only
    when it converged.
    """
    absorbed = absorb_slopes(
        problem,
        weights,
        tolerance=tolerance,
        max_sweeps=max_sweeps,
        acceleration=acceleration,
    )
    if absorbed.failure:
        raise FloatingPointError(absorbed.failure)
    return absorbed
