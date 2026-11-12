"""Support-only weighted least squares of a working outcome on ``B = [X, F]``.

Internal module.  The iterative rectifier
(:mod:`powermean.solve.support_rectifier`) projects its working outcome ``u`` with its
auxiliary weights ``w``; nothing here is the statistical WLS, a
collinearity or omission policy, or a sample rule.

Absorption.  ``[u, X / s]`` is projected through the accepted
:func:`powermean.fe.absorb.absorb` with ``w`` and the explicit controls of
:class:`ProjectionControls` (tolerance, sweep budget, the accepted acceleration); any
status other than ``converged`` ends the projection with status ``absorption``.  The
column scales ``s`` are powers of two with ``max_i |x_ij| / s_j`` in ``[1/2, 1)``, and
``s_j = 1`` for a zero column: LAPACK's ``xGEEQUB`` scales by powers of the radix so
that scaling and unscaling round nothing.
Without blocks the columns are taken as they are.

Slopes.  ``A = sqrt(w) X~`` and ``b = sqrt(w) u~``.  The rank is the number of singular
values of ``A`` above ``eps max(n, p) sigma_max``; a
column-pivoted QR ``A P = Q R`` supplies the basic coordinates, its first ``r`` pivots,
whose coefficients solve ``R_rr c = Q_r' b`` by back substitution while the nonbasic
coordinates are zero.  ``||A - Q_r Q_r' A||_2 <= cutoff`` is checked for every input
column (``||A||_2`` at ``r = 0``); a failed range check, a zero basic pivot or a failed
decomposition is ``rank_uncertain``.  No Gram inverse, pseudoinverse, ridge, changed
cutoff or other solver is used, no column is omitted from any equation, and a
rank-redundant representation still yields the unique fitted projection without a claim
of identified coefficients.

Refinement of ``u``.  The absorber's weighted level means bound
``sum_g w u~`` only by ``tol sum_g w``, about ``K`` times what the rectifier's dual
needs.  Given the dual's ``e``, the ``u`` column of the converged joint absorption must
meet the requirement ``R_e``: on every level of every block, ``s = sum w u~`` and
``m = sum |w u~|`` have ``|s| / m <= e`` (the dual's own ratio) or
``|s| <= (G + n_g + 2) 2^-52 (sum w (|u| + sum_b |a_b|) + m)``, a floor of at least
twice a first-order bound of the rounding error of ``s``'s own evaluation.  That the
correctly rounded exact projection meets it is a first-order rationale for normal-range
arithmetic, not a guarantee on every sample: a floor that underflows leaves ``R_e``
unevaluable even at an exact zero sum (a subnormal ``u``), and meeting ``R_e`` implies
nothing about the absolute tolerance, which that projection already misses at ``|u|``
near ``2^50``.  Unmet, one more absorber call refines the column alone, warm-started
from the joint call's coefficients with ``R_e`` as its requirement and at most
``c (D + 1)`` sweeps (``D`` levels, ``c = 2`` for the accelerated walk of several
blocks, else 1) within the per-call budget, so that a projection's absorber work is at
most the joint call's plus that call's; each evaluation of ``R_e`` is further work,
counted apart.  Its candidate replaces the column only when ``R_e`` holds there afresh.
Otherwise (unmet, unevaluable or an unsuccessful call) the joint values stand, exactly
as without a requirement, and the rechecks below decide.

Reconstruction and recheck.  The block coefficients are ``a_u - sum_j c_j a_xj`` per
block, the slopes ``beta = c / s``; the fitted values ``f = X beta + lift(a)`` are
reconstructed by :func:`powermean.solve.support_witness.reconstruct` and ``r = u - f``.
Every weighted level mean ``sum_g w r / sum_g w`` and every slope equation ``sum w x_j r
/ sum w |x_j|`` of every original slope, basic or not, must be at most the absolute
tolerance; an identically zero slope needs an exact zero numerator.  A failure is
``wls_uncertain``, an unrepresentable quantity ``range``: numerical outcomes, never a
separation verdict.  Elementwise steps are range-checked, products run with every
floating-point condition raised, and decomposition outputs must be finite; a slope
ratio is the dual check's range-checked quotient
(:func:`powermean.solve.support_witness.relative`), so a nonzero numerator whose
quotient is lost is ``range``, never an exact zero, as a level mean already is.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, cast

import numpy as np

from powermean.fe.absorb import AbsorptionResult, AbsorptionState, absorb
from powermean.fe.absorb_numeric import (
    Coefficients,
    PreparedWeights,
    candidate,
    checked_difference,
    checked_product,
    checked_quotient,
    checked_sum,
    group_sums,
    lift,
    prepare_weights,
)
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.support_problem import SupportProblem, indices
from powermean.solve.support_witness import (
    Effects,
    column_sums,
    reconstruct,
    relative,
)

EPSILON = float(np.finfo(np.float64).eps)
Status = Literal["solved", "absorption", "rank_uncertain", "wls_uncertain", "range"]


class Qr(Protocol):
    """``scipy.linalg.qr`` with column pivoting and the economic factors."""

    def __call__(
        self, a: FloatArray, /, *, mode: str, pivoting: bool, check_finite: bool
    ) -> tuple[FloatArray, FloatArray, IndexArray]:
        """``Q``, ``R`` and the pivots ``P`` of ``a[:, P] = Q R``."""
        ...


class SolveTriangular(Protocol):
    """``scipy.linalg.solve_triangular`` on an upper triangular matrix."""

    def __call__(
        self, a: FloatArray, b: FloatArray, /, *, lower: bool, check_finite: bool
    ) -> FloatArray:
        """The solution of ``a x = b`` by back substitution."""
        ...


class Linalg(Protocol):
    """The objects of ``scipy.linalg`` that this module uses."""

    qr: Qr
    solve_triangular: SolveTriangular


linalg = cast("Linalg", importlib.import_module("scipy.linalg"))


@dataclass(frozen=True)
class ProjectionControls:
    """The explicit controls of one support projection.

    ``tolerance`` is the absolute weighted-mean tolerance of the absorption and of
    every equation recheck; ``max_sweeps`` the sweep budget of one absorber call,
    which also caps the refinement call's (module docstring); ``acceleration`` the
    accepted absorption acceleration, recorded.
    """

    tolerance: float
    max_sweeps: int
    acceleration: bool

    def __post_init__(self) -> None:
        """Positive finite tolerance, positive integer budget, a Boolean flag."""
        if isinstance(self.tolerance, bool) or not isinstance(self.tolerance, float):
            raise ValueError("tolerance must be a positive finite float")
        if not (np.isfinite(self.tolerance) and self.tolerance > 0):
            raise ValueError("tolerance must be a positive finite float")
        if isinstance(self.max_sweeps, bool) or not isinstance(self.max_sweeps, int):
            raise ValueError("max_sweeps must be a positive integer")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be a positive integer")
        if not isinstance(self.acceleration, bool):
            raise ValueError("acceleration must be True or False")


@dataclass(frozen=True, eq=False)
class Refinement:
    """The requirement ``R_e`` on the ``u`` column and its refinement call, if any.

    ``reason`` is ``joint`` (``R_e`` holds at the joint candidate, no call), ``met``
    (it holds afresh at the call's candidate, adopted), ``unmet`` (the call converged
    without it), ``unevaluable`` (an evaluation refused by range) or the call's own
    unsuccessful status (``budget``, ``range``, ``breakdown``); ``adopted`` is whether
    the call's candidate replaced the column (``met`` only); ``budget`` the call's
    sweep budget (0 without a call); ``evaluations`` every evaluation of ``R_e``;
    ``result`` the call's absorber result (``None`` without a call).
    """

    reason: str
    adopted: bool
    budget: int
    evaluations: int
    result: AbsorptionResult | None = None


@dataclass(frozen=True, eq=False)
class Projection:
    """One support projection: its status, decomposition, coefficients and rechecks.

    ``scales`` are the column scales ``s``; ``absorption`` the absorber's result of the
    joint call (none without blocks); ``singular_values``, ``cutoff``, ``rank``,
    ``pivots`` and ``range_residual`` the decomposition of ``A``; ``slopes`` (``beta``,
    original coordinates, nonbasic zero), ``effects``, ``fitted`` and ``residual`` the
    reconstruction; ``level_means`` and ``slope_ratios`` the rechecked equations;
    ``refinement`` the requirement on ``u`` (``None`` without a dual tolerance or
    without blocks).
    """

    status: Status
    reason: str
    scales: FloatArray
    absorption: AbsorptionResult | None = None
    singular_values: FloatArray | None = None
    cutoff: float | None = None
    rank: int | None = None
    pivots: IndexArray | None = None
    range_residual: float | None = None
    slopes: FloatArray | None = None
    effects: Effects | None = None
    fitted: FloatArray | None = None
    residual: FloatArray | None = None
    level_means: tuple[FloatArray, ...] = field(default=())
    slope_ratios: FloatArray | None = None
    refinement: Refinement | None = None

    @property
    def solved(self) -> bool:
        """Whether the projection and every recheck passed."""
        return self.status == "solved"

    @property
    def basic(self) -> IndexArray | None:
        """The basic slope coordinates: the first ``rank`` pivots."""
        if self.pivots is None or self.rank is None:
            return None
        return indices(self.pivots[: self.rank])


def scales(design: FloatArray) -> FloatArray:
    """Per column the power of two ``s``, ``max |x| / s`` in ``[1/2, 1)``; 1 if zero.

    The exponent is capped at 1023, the largest power of two that is a double, so a
    column whose largest magnitude is at least ``2^1023`` keeps a ratio below 2.
    """
    largest = np.max(np.abs(design), axis=0) if design.size else np.zeros(0)
    exponents = np.minimum(np.frexp(largest)[1], 1023)
    return immutable_array(np.where(largest == 0, 1.0, np.ldexp(1.0, exponents)))


@dataclass(frozen=True, eq=False)
class _Absorbed:
    """``u~``, ``X~ / s`` and the block coefficients of ``[u, X / s]`` (columns)."""

    u: FloatArray
    x: FloatArray
    coefficients: tuple[FloatArray, ...]
    result: AbsorptionResult | None
    prepared: PreparedWeights | None = None


def _absorbed(
    problem: SupportProblem,
    values: FloatArray,
    weights: FloatArray,
    controls: ProjectionControls,
) -> _Absorbed:
    """Absorb every column of ``values`` (``[u, X / s]``), or take it as is."""
    if problem.grouping is None:
        return _Absorbed(values[:, 0], values[:, 1:], (), None)
    names = ("u", *(f"x{j}" for j in range(problem.n_slopes)))
    prepared = prepare_weights(problem.grouping, weights)
    result = absorb(
        values,
        prepared,
        names,
        tolerance=controls.tolerance,
        max_sweeps=controls.max_sweeps,
        acceleration=controls.acceleration,
    )
    found = result.residual
    coefficients = result.state.coefficients
    return _Absorbed(found[:, 0], found[:, 1:], coefficients, result, prepared)


def _settled(
    grouping: Grouping,
    u: FloatArray,
    weights: FloatArray,
    coefficients: Coefficients,
    dual_tolerance: float,
) -> bool:
    """Whether ``R_e`` holds at the ``u`` column's block coefficients (one column).

    The candidate is the absorber's own ``u - lift(a)``; per level ``s`` and ``m`` are
    the dual's column sums of ``w u~`` and their ratio its range-checked quotient; the
    floor is ``(G + n_g + 2) 2^-52`` times the level sum of ``w (|u| + sum_b |a_b|)``
    plus ``m``.  ``FloatingPointError`` when a quantity is not representable.
    """
    residual = candidate(u[:, None], coefficients, grouping)[:, 0]
    weighted = checked_product(weights, residual, "requirement product")
    magnitude = lift(tuple(np.abs(a) for a in coefficients), grouping)[:, 0]
    terms = checked_sum(np.abs(u), magnitude, "requirement magnitude")
    terms = checked_product(weights, terms, "requirement magnitude")
    blocks = grouping.blocks
    for block in blocks:
        sums = column_sums(weighted, block.codes, block.n_levels)
        mass = group_sums(terms, block.codes, block.n_levels)
        factor = np.ldexp((len(blocks) + 2 + block.counts).astype(np.float64), -52)
        floor = checked_sum(mass, sums.denominators, "requirement floor")
        floor = checked_product(factor, floor, "requirement floor")
        level = (sums.ratios <= dual_tolerance) | (np.abs(sums.numerators) <= floor)
        if not level.all():
            return False
    return True


def _refined(
    problem: SupportProblem,
    u: FloatArray,
    weights: FloatArray,
    controls: ProjectionControls,
    *,
    absorbed: _Absorbed,
    dual_tolerance: float,
) -> tuple[_Absorbed, Refinement]:
    """``R_e`` at the converged joint candidate and, unmet, one refinement call.

    The call absorbs ``u`` alone from the joint coefficients of its column with ``R_e``
    as requirement, the same tolerance and acceleration and ``c (D + 1)`` sweeps at
    most; an unevaluable requirement asks nothing more.  Its candidate replaces the
    ``u`` column only when ``R_e`` holds there afresh; otherwise ``absorbed`` stands.
    """
    grouping, prepared = problem.grouping, absorbed.prepared
    if grouping is None or prepared is None:
        raise ValueError("a refinement needs the joint absorption of the blocks")
    start = tuple(a[:, :1] for a in absorbed.coefficients)
    count = [0]

    def settled(coefficients: Coefficients) -> bool:
        count[0] += 1
        return _settled(grouping, u, weights, coefficients, dual_tolerance)

    def requirement(coefficients: Coefficients) -> bool:
        try:
            return settled(coefficients)
        except FloatingPointError:
            return True

    try:
        met = settled(start)
    except FloatingPointError:
        return absorbed, Refinement("unevaluable", False, 0, count[0])
    if met:
        return absorbed, Refinement("joint", False, 0, count[0])
    blocks = grouping.blocks
    cost = 2 if controls.acceleration and len(blocks) > 1 else 1
    budget = min(controls.max_sweeps, cost * (sum(b.n_levels for b in blocks) + 1))
    try:
        result = absorb(
            u[:, None],
            prepared,
            ("u",),
            tolerance=controls.tolerance,
            max_sweeps=budget,
            acceleration=controls.acceleration,
            state=AbsorptionState(grouping, ("u",), start),
            requirement=requirement,
        )
    except FloatingPointError:
        return absorbed, Refinement("range", False, budget, count[0])
    if not result.converged:
        return absorbed, Refinement(result.status, False, budget, count[0], result)
    try:
        met = settled(result.state.coefficients)
    except FloatingPointError:
        return absorbed, Refinement("unevaluable", False, budget, count[0], result)
    if not met:
        return absorbed, Refinement("unmet", False, budget, count[0], result)
    coefficients = tuple(
        np.column_stack([refined[:, 0], joint[:, 1:]])
        for refined, joint in zip(
            result.state.coefficients, absorbed.coefficients, strict=True
        )
    )
    adopted = replace(absorbed, u=result.residual[:, 0], coefficients=coefficients)
    return adopted, Refinement("met", True, budget, count[0], result)


@dataclass(frozen=True, eq=False)
class _Basis:
    """The decomposition of ``A`` and, when it passes, the basic coefficients."""

    singular_values: FloatArray
    cutoff: float
    rank: int
    pivots: IndexArray
    range_residual: float
    coordinates: FloatArray | None
    reason: str
    status: Status = "solved"


def _basis(a: FloatArray, b: FloatArray) -> _Basis:
    """SVD rank, pivoted QR, the all-column range check and the basic solve.

    A failed range check or zero pivot is ``rank_uncertain`` and nonfinite basic
    coefficients ``range``, the decomposition kept; ``FloatingPointError`` for an
    unrepresentable product and ``LinAlgError`` from a failed decomposition.
    """
    n, p = a.shape
    singular: FloatArray = np.asarray(
        np.linalg.svd(a, compute_uv=False), dtype=np.float64
    )
    cutoff = EPSILON * max(n, p) * float(singular[0]) if singular.size else 0.0
    rank = int(np.count_nonzero(singular > cutoff))
    q, r, pivots = linalg.qr(a, mode="economic", pivoting=True, check_finite=False)
    basis = q[:, :rank]
    # finite inputs: an overflowing or invalid product raises FloatingPointError
    with np.errstate(over="raise", invalid="raise"):
        leftover = a - basis @ (basis.T @ a)
        rhs = basis.T @ b
    residual = float(np.linalg.norm(leftover, 2)) if leftover.size else 0.0
    found = _Basis(singular, cutoff, rank, indices(pivots), residual, None, "")
    uncertain = replace(found, status="rank_uncertain")
    if not residual <= cutoff:
        return replace(uncertain, reason="a column is outside the basic range")
    block = r[:rank, :rank]
    if (np.diag(block) == 0).any():
        return replace(uncertain, reason="a basic pivot is exactly zero")
    coordinates = np.zeros(p)
    if rank:
        solution = linalg.solve_triangular(block, rhs, lower=False, check_finite=False)
        coordinates[found.pivots[:rank]] = solution
    if not np.isfinite(coordinates).all():
        return replace(found, status="range", reason="basic coefficients not finite")
    return replace(found, coordinates=coordinates)


def _effects(absorbed: _Absorbed, coordinates: FloatArray) -> Effects:
    """``a_u - sum_j c_j a_xj`` per block, the columns in their order, checked."""
    found = []
    for block in absorbed.coefficients:
        a = block[:, 0]
        for j, c in enumerate(coordinates):
            term = checked_product(block[:, 1 + j], np.array([c]), "effect product")
            a = checked_difference(a, term, "recovered effects")
        found.append(immutable_array(a))
    return tuple(found)


def _rechecks(
    problem: SupportProblem, weights: FloatArray, residual: FloatArray
) -> tuple[tuple[FloatArray, ...], FloatArray]:
    """Every weighted level mean of ``r`` and every slope equation ratio, checked."""
    weighted = checked_product(weights, residual, "weighted residual")
    blocks = () if problem.grouping is None else problem.grouping.blocks
    means = tuple(
        immutable_array(
            checked_quotient(
                group_sums(weighted, b.codes, b.n_levels),
                group_sums(weights, b.codes, b.n_levels),
                "level mean",
            )
        )
        for b in blocks
    )
    if not problem.n_slopes:
        return means, immutable_array(np.zeros(0))
    single = indices(np.zeros(problem.n_rows))
    design = problem.design
    num = group_sums(checked_product(design, weighted[:, None]), single, 1)[0]
    den = group_sums(checked_product(np.abs(design), weights[:, None]), single, 1)[0]
    return means, immutable_array(relative(np.abs(num), den))


def _reconstructed(
    problem: SupportProblem,
    u: FloatArray,
    weights: FloatArray,
    *,
    absorbed: _Absorbed,
    found: Projection,
    coordinates: FloatArray,
) -> Projection:
    """The coefficients, the fresh reconstruction and the rechecked equations."""
    effects = _effects(absorbed, coordinates)
    slopes = checked_quotient(coordinates, found.scales, "slopes")
    fitted = reconstruct(problem, slopes, effects)
    residual = checked_difference(u, fitted, "support residual")
    means, ratios = _rechecks(problem, weights, residual)
    return replace(
        found,
        slopes=immutable_array(slopes),
        effects=effects,
        fitted=immutable_array(fitted),
        residual=immutable_array(residual),
        level_means=means,
        slope_ratios=ratios,
    )


def project(
    problem: SupportProblem,
    u: FloatArray,
    weights: FloatArray,
    controls: ProjectionControls,
    *,
    dual_tolerance: float | None = None,
) -> Projection:
    """The support projection of ``u`` on ``[X, F]`` with auxiliary weights ``w``.

    Parameters
    ----------
    problem : SupportProblem
        The current sample: ``X`` and the grouping are read, nothing is modified.
    u : array of float
        The working outcome, one finite value per row.
    weights : array of float
        The positive auxiliary weights, one per row (penalties, not the statistical
        weights).
    controls : ProjectionControls
        The explicit absorption tolerance, budget and acceleration.
    dual_tolerance : float, optional
        The ``e`` of the dual that ``w r`` will meet: with blocks, the ``u`` column
        must then meet ``R_e`` (module docstring); none by default, the projection
        without a requirement.

    Returns
    -------
    Projection
        ``solved`` when every recheck passes; otherwise the failed stage
        (``absorption``, ``rank_uncertain``, ``wls_uncertain``, ``range``) with what
        was computed.  ``ValueError`` for malformed ``u``, weights or tolerance.
    """
    s = scales(problem.design)
    shape = (problem.n_rows,)
    if u.shape != shape or weights.shape != shape or not np.isfinite(u).all():
        raise ValueError("u and the weights must have one finite value per row")
    if not (np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("the auxiliary weights must be positive and finite")
    if dual_tolerance is not None and not (
        isinstance(dual_tolerance, float) and 0 < dual_tolerance < 1
    ):
        raise ValueError("the dual tolerance must be a float in (0, 1)")
    try:
        x = checked_quotient(problem.design, s[None, :], "scaled design")
        absorbed = _absorbed(problem, np.column_stack([u, x]), weights, controls)
    except FloatingPointError as error:
        return Projection("range", str(error), s)
    found = Projection("solved", "", s, absorption=absorbed.result)
    if absorbed.result is not None and not absorbed.result.converged:
        reason = f"{absorbed.result.status}: {absorbed.result.reason}"
        return replace(found, status="absorption", reason=reason)
    if dual_tolerance is not None and absorbed.result is not None:
        absorbed, refinement = _refined(
            problem,
            u,
            weights,
            controls,
            absorbed=absorbed,
            dual_tolerance=dual_tolerance,
        )
        found = replace(found, refinement=refinement)
    try:
        coordinates = np.zeros(0)
        if problem.n_slopes:
            root = np.sqrt(weights)
            a = checked_product(root[:, None], absorbed.x, "weighted design")
            basis = _basis(a, checked_product(root, absorbed.u, "weighted outcome"))
            found = replace(
                found,
                singular_values=immutable_array(basis.singular_values),
                cutoff=basis.cutoff,
                rank=basis.rank,
                pivots=basis.pivots,
                range_residual=basis.range_residual,
            )
            if basis.coordinates is None:
                return replace(found, status=basis.status, reason=basis.reason)
            coordinates = basis.coordinates
        found = _reconstructed(
            problem, u, weights, absorbed=absorbed, found=found, coordinates=coordinates
        )
    except FloatingPointError as error:
        return replace(found, status="range", reason=str(error))
    except np.linalg.LinAlgError as error:
        return replace(found, status="rank_uncertain", reason=str(error))
    return _verdict(found, controls.tolerance)


def _verdict(found: Projection, tolerance: float) -> Projection:
    """``wls_uncertain`` unless every level mean and slope ratio is within tolerance."""
    failed = []
    if any((np.abs(means) > tolerance).any() for means in found.level_means):
        failed.append("level")
    if found.slope_ratios is not None and (found.slope_ratios > tolerance).any():
        failed.append("slope")
    if not failed:
        return found
    reason = " and ".join(failed) + " equations exceed the tolerance"
    return replace(found, status="wls_uncertain", reason=reason)
