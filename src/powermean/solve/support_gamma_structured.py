"""A structured solver for the two recession programs of the GPML diagnostic.

The programs are those of :mod:`powermean.solve.support_gamma_lp` (LP1, ``negative``:
minimize ``t'v`` subject to ``A_P v >= 0`` and the box; LP2, ``zero_sum``: minimize
``-p'v`` under the same and ``t'v = 0``), and every report this module returns is
checked afresh by the unchanged witnesses of
:mod:`powermean.solve.support_gamma_witness`.  Only the optimizer differs.

For an ordinary mixed sample, :func:`calibrated_reports` proposes both reports
using positive-row loadings calibrated to the full weighted column totals. The
caller checks both reports before admitting either one.

Reduction.  Choose one block ``E`` whose every level is an LP coordinate.  Write a
coordinate vector as ``(u, a)``, ``a`` the levels of ``E`` and ``u`` every other
coordinate (slopes and the kept levels of the other blocks), and ``b_j`` the row of
``A`` on ``u``.  For fixed ``u`` the constraints on the level ``g`` of ``E`` are
``a_g >= -b_j'u`` on the positive rows ``P_g`` of ``g``, and ``a_g`` enters ``t'v``
with ``n_g = t_g > 0``; the least ``a_g`` is ``max_{P_g} (-b_j'u)``, and

    f(u) = c'u - sum_g n_g min_{j in P_g} b_j'u,        c = t on u,

is the least objective.  ``f`` is the support function of the polytope

    K = { c - sum_j lambda_j b_j : lambda >= 0, sum_{j in P_g} lambda_j = n_g },

so LP1 has a negative value (a ``negative_sum_recession``) exactly when ``0`` is not in
``K``, and the dual multipliers of LP1 at ``v = 0`` are exactly the ``lambda`` that
represent ``0``.  LP2 has a positive value (a ``zero_sum_recession``) exactly when
``0`` is on the relative boundary of ``K``: its dual at ``v = 0`` is ``kappa lambda -
rho_P`` with ``nu = -kappa`` for a representing ``lambda`` that is positive on every
row of ``P``.  A level of ``E`` without a positive row makes LP1 unbounded below in
``a_g`` up to the box (its indicator direction).

Method.  The minimum-norm point of ``K`` by Wolfe's algorithm: the linear oracle over
``K`` picks, in every level of ``E``, a positive row of largest ``b_j'x``, in one pass
over the rows; the corral has at most ``dim u + 1`` vertices, each a selection of one
row per level.  A representation of ``0`` gives LP1's dual; a representation of
``-eps r`` (``r`` the point of ``K`` of the uniform multipliers) for some ``eps > 0``
gives a positive one, hence LP2's dual.  A minimum-norm point away from ``0`` gives
LP1's direction ``u = -x`` with the least ``a``.

What is returned.  :func:`structured_call` returns an
:class:`~powermean.solve.support_gamma_lp.LPReport` of the requested program, filled
as a usable optimizer report (status 0, the point, the objective, the four marginal
blocks), or ``None`` when the reduction does not apply or the method does not settle
within its budget; the caller then runs the HiGHS program.  Nothing here certifies:
the report is evidence only through the fresh primal and dual checks.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from powermean.solve.support_gamma_lp import (
    LPData,
    LPReport,
    Marginal,
    Stage,
    _positions,
    stage_objective,
)
from powermean.solve.support_gamma_problem import GammaProblem
from powermean.solve.support_gamma_witness import dual_check, primal_witness
from powermean.solve.support_lp import LPControls

FloatArray = np.ndarray
IndexArray = np.ndarray

if TYPE_CHECKING:

    class _Optimize(Protocol):
        def nnls(
            self, matrix: FloatArray, target: FloatArray, /, *, maxiter: int
        ) -> tuple[FloatArray, float]: ...


# Representations: a residual at most ZERO times the scale of K is a representation of
# the target; a gap at most GAP times |x|^2 ends Wolfe's major cycle.
ZERO = 1e-15
GAP = 1e-10
# Targets -eps r tried in turn for LP2's positive representation.
DEPTHS = (0.1, 0.01, 0.5, 1e-3, 1e-6)
# An iterate within NEAR of the target is tried as a representation, then completed
# by :func:`finish`.
NEAR = 1e-10
# Major cycles of one minimum-norm computation, at most; a computation whose |x| has
# not fallen by STALL in WINDOW cycles stops unsettled.
CYCLES = 5000
STALL, WINDOW = 1e-3, 50
# Dimension limit for dense corral algebra and ordinary Gram calibration.
MAX_DIMENSION = 400


@dataclass(frozen=True, eq=False)
class Reduced:
    """The reduced data of one sample: the rows of ``P`` on ``u``, the levels of ``E``.

    ``outer`` are the LP indices of ``u``; ``levels`` the LP index of every level of
    ``E``; ``group`` the level of ``E`` of each positive row; ``slopes`` the scaled
    slope values of the positive rows on the kept slopes, with ``slope_at`` their
    places in ``u``; ``codes`` and ``code_at`` the other blocks' level codes of the
    positive rows and the place in ``u`` of each level (``-1`` for an omitted level).
    """

    k: int
    outer: IndexArray
    levels: IndexArray
    n: FloatArray
    c: FloatArray
    group: IndexArray
    slopes: FloatArray
    slope_at: IndexArray
    codes: tuple[IndexArray, ...]
    code_at: tuple[IndexArray, ...]
    empty: IndexArray

    @property
    def m(self) -> int:
        """The dimension of ``u``."""
        return int(self.outer.size)

    def scores(self, u: FloatArray) -> FloatArray:
        """``b_j'u`` for every positive row ``j``."""
        found = (
            self.slopes @ u[self.slope_at]
            if self.slope_at.size
            else np.zeros(self.group.size)
        )
        for codes, at in zip(self.codes, self.code_at, strict=True):
            gathered = np.where(at >= 0, u[np.maximum(at, 0)], 0.0)
            found = found + gathered[codes]
        return found

    def vertex(self, selection: IndexArray) -> FloatArray:
        """``c - sum_g n_g b_{selection(g)}``: the vertex of a selection of rows."""
        found = self.c.copy()
        if self.slope_at.size:
            found[self.slope_at] -= self.n @ self.slopes[selection]
        for codes, at in zip(self.codes, self.code_at, strict=True):
            sums = np.bincount(codes[selection], weights=self.n, minlength=at.size)
            kept = at >= 0
            np.subtract.at(found, at[kept], sums[kept])
        return found

    def best(self, scores: FloatArray, *, largest: bool) -> IndexArray:
        """In every level of ``E``, the first positive row of largest (least) score."""
        signed = scores if largest else -scores
        top = np.full(self.n.size, -np.inf)
        np.maximum.at(top, self.group, signed)
        rows = np.flatnonzero(signed == top[self.group])
        chosen = np.empty(self.n.size, dtype=np.intp)
        chosen[self.group[rows[::-1]]] = rows[::-1]
        return chosen

    def value(self, u: FloatArray) -> float:
        """``f(u)``: the least LP1 objective over ``a`` at the outer part ``u``."""
        scores = self.scores(u)
        low = np.full(self.n.size, np.inf)
        np.minimum.at(low, self.group, scores)
        return float(self.c @ u - self.n @ low)


def reduce(problem: GammaProblem, data: LPData) -> Reduced | None:
    """The reduction on the largest block whose every level is a coordinate, or None."""
    grouping = problem.grouping
    if grouping is None or not grouping.blocks:
        return None
    positions, p = _positions(problem), problem.n_slopes
    offsets = np.cumsum([p] + [b.n_levels for b in grouping.blocks])[:-1]
    eligible = [
        i
        for i, b in enumerate(grouping.blocks)
        if (positions[offsets[i] : offsets[i] + b.n_levels] >= 0).all()
    ]
    if not eligible:
        return None
    e = max(eligible, key=lambda i: grouping.blocks[i].n_levels)
    levels = positions[offsets[e] : offsets[e] + grouping.blocks[e].n_levels]
    k = problem.n_coordinates
    outer = np.setdiff1d(np.arange(k), levels)
    place = np.full(k, -1, dtype=np.intp)
    place[outer] = np.arange(outer.size)
    t = np.asarray(stage_objective(data, "negative"))
    positive = problem.positive
    kept = np.flatnonzero(positions[:p] >= 0)
    codes, code_at = [], []
    for i, block in enumerate(grouping.blocks):
        if i == e:
            continue
        at = positions[offsets[i] : offsets[i] + block.n_levels]
        codes.append(np.asarray(block.codes)[positive])
        code_at.append(np.where(at >= 0, place[np.maximum(at, 0)], -1))
    group = np.asarray(grouping.blocks[e].codes)[positive]
    counts = np.bincount(group, minlength=grouping.blocks[e].n_levels)
    return Reduced(
        k,
        outer,
        levels,
        t[levels],
        t[outer],
        group,
        np.asarray(data.scaled)[np.ix_(positive, kept)],
        place[positions[kept]],
        tuple(codes),
        tuple(code_at),
        np.flatnonzero(counts == 0),
    )


@dataclass(frozen=True, eq=False)
class Corral:
    """Wolfe's corral: vertices, their selections and weights, and the point ``x``."""

    points: FloatArray
    selections: tuple[IndexArray, ...]
    weights: FloatArray
    x: FloatArray
    iterations: int
    settled: bool


def _affine(points: FloatArray) -> FloatArray:
    """The weights, summing to one, of the least-norm affine combination of the rows."""
    r = points.shape[0]
    gram = points @ points.T
    system = np.zeros((r + 1, r + 1))
    system[:r, :r], system[:r, r], system[r, :r] = gram, 1.0, 1.0
    rhs = np.zeros(r + 1)
    rhs[r] = 1.0
    solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return solution[:r]


def _minor(
    points: FloatArray,
    selections: tuple[IndexArray, ...],
    weights: FloatArray,
) -> tuple[FloatArray, tuple[IndexArray, ...], FloatArray]:
    """Wolfe's minor cycle: move to the affine minimizer, dropping vertices."""
    while True:
        alpha = _affine(points)
        if (alpha > 1e-15).all():
            return points, selections, alpha
        down = alpha <= 1e-15
        steps = weights[down] / (weights[down] - alpha[down])
        theta = float(np.min(steps)) if steps.size else 1.0
        weights = theta * alpha + (1 - theta) * weights
        keep = weights > 1e-15
        points = points[keep]
        selections = tuple(s for s, k in zip(selections, keep, strict=True) if k)
        weights = weights[keep] / weights[keep].sum()


def _barycentric(points: FloatArray) -> FloatArray | None:
    """The weights of the origin in a full simplex of ``dim + 1`` points, or None."""
    r = points.shape[0]
    if r != points.shape[1] + 1:
        return None
    system = np.vstack([points.T, np.ones(r)])
    rhs = np.zeros(r)
    rhs[-1] = 1.0
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None


def wolfe(
    reduced: Reduced,
    target: FloatArray,
    budget: int,
    start: Corral | None = None,
) -> Corral:
    """The point of ``K`` nearest ``target`` (Wolfe's minimum-norm point of K - target).

    ``start`` is a corral of ``K`` itself (target 0) to begin from.
    """
    selections: tuple[IndexArray, ...]
    if start is None:
        selection = reduced.best(reduced.scores(np.zeros(reduced.m)), largest=True)
        points = (reduced.vertex(selection) - target)[None, :]
        selections, weights = (selection,), np.ones(1)
    else:
        points = start.points - target
        selections, weights = start.selections, start.weights
    scale = max(float(np.abs(reduced.c).max()), float(reduced.n.max()), 1.0)
    best, since = np.inf, 0
    for iteration in range(1, budget + 1):
        x = weights @ points
        size = float(np.abs(x).max())
        if size <= ZERO * scale:
            return Corral(points, selections, weights, x, iteration, True)
        best, since = (size, 0) if size < (1 - STALL) * best else (best, since + 1)
        if since > WINDOW:
            break
        inside = _barycentric(points)
        if inside is not None and (inside > 0).all():
            x = inside @ points
            return Corral(
                points,
                selections,
                inside,
                x,
                iteration,
                float(np.abs(x).max()) <= ZERO * scale,
            )
        selection = reduced.best(reduced.scores(x), largest=True)
        v = reduced.vertex(selection) - target
        size = max(float(x @ x), float(np.linalg.norm(x) * np.linalg.norm(v)))
        known = any(np.array_equal(selection, s) for s in selections)
        if known or float(x @ x - x @ v) <= GAP * size:
            return Corral(points, selections, weights, x, iteration, True)
        points = np.vstack([points, v])
        selections = (*selections, selection)
        weights = np.append(weights, 0.0)
        points, selections, weights = _minor(points, selections, weights)
    return Corral(points, selections, weights, weights @ points, iteration, False)


def finish(reduced: Reduced, corral: Corral, target: FloatArray) -> Corral:
    """Complete a near representation of ``target`` to a full-dimensional one.

    Wolfe's iterate stalls at rounding level when the corral spans only a face: the
    oracle vertices of both signs of every direction orthogonal to the corral's affine
    hull are added, and nonnegative least squares weights them (the weights summing
    to one through a heavily weighted row).  The result is a corral of the vertices
    with positive weight.
    """
    points, selections = corral.points, list(corral.selections)
    centred = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=True)
    rank = int((singular > 1e-10 * max(float(singular[0]), 1e-300)).sum())
    extra = []
    for direction in vt[rank:]:
        for sign in (1.0, -1.0):
            chosen = reduced.best(reduced.scores(sign * direction), largest=True)
            extra.append((reduced.vertex(chosen) - target, chosen))
    if extra:
        points = np.vstack([points, *[e[0][None, :] for e in extra]])
        selections += [e[1] for e in extra]
    weight = 1e3 * max(float(np.abs(points).max()), 1.0)
    matrix = np.vstack([points.T, weight * np.ones(points.shape[0])])
    rhs = np.concatenate([np.zeros(points.shape[1]), [weight]])
    nnls = cast("_Optimize", import_module("scipy.optimize")).nnls
    found = nnls(matrix, rhs, maxiter=50 * matrix.shape[1])[0]
    keep = found > 0
    if not keep.any():
        return corral
    weights = found[keep] / found[keep].sum()
    kept = points[keep]
    return Corral(
        kept,
        tuple(s for s, k in zip(selections, keep, strict=True) if k),
        weights,
        weights @ kept,
        corral.iterations + len(extra),
        True,
    )


def multipliers(reduced: Reduced, corral: Corral) -> FloatArray:
    """``lambda`` on the rows of ``P`` from the corral: ``n_g`` times the weights."""
    found = np.zeros(reduced.group.size)
    for selection, w in zip(corral.selections, corral.weights, strict=True):
        np.add.at(found, selection, w * reduced.n)
    return found


def refine(
    reduced: Reduced,
    corral: Corral,
    target: FloatArray,
    rounds: int = 3,
) -> Corral:
    """Correct the corral's weights by the residual of their multipliers, row by row.

    The residual ``c - sum_j lambda_j b_j - target`` is formed from the rows (not from
    the vertices), and the barycentric correction of the vertices removes it; the
    weights keep summing to one.  The vertices and selections are unchanged.
    """
    points, weights = corral.points, corral.weights.copy()
    system = np.vstack([points.T, np.ones(points.shape[0])])
    for _ in range(rounds):
        lam = multipliers(
            reduced,
            Corral(
                points,
                corral.selections,
                weights,
                corral.x,
                corral.iterations,
                corral.settled,
            ),
        )
        residual = represented(reduced, lam) - target
        rhs = np.concatenate([-residual, [1.0 - float(weights.sum())]])
        weights = weights + np.linalg.lstsq(system, rhs, rcond=None)[0]
    if (weights < 0).any():
        return corral
    return Corral(
        points,
        corral.selections,
        weights,
        weights @ points,
        corral.iterations,
        corral.settled,
    )


def uniform(reduced: Reduced) -> FloatArray:
    """``lambda`` spread evenly: ``n_g / |P_g|`` on every positive row of ``g``."""
    counts = np.bincount(reduced.group, minlength=reduced.n.size)
    return cast("FloatArray", (reduced.n / counts)[reduced.group])


def represented(reduced: Reduced, lam: FloatArray) -> FloatArray:
    """``c - sum_j lambda_j b_j`` on ``u``: the residual of a multiplier set."""
    found = reduced.c.copy()
    if reduced.slope_at.size:
        found[reduced.slope_at] -= lam @ reduced.slopes
    for codes, at in zip(reduced.codes, reduced.code_at, strict=True):
        sums = np.bincount(codes, weights=lam, minlength=at.size)
        kept = at >= 0
        np.subtract.at(found, at[kept], sums[kept])
    return found


# ------------------------------------------------------------------- reports


def _report(
    stage: Stage,
    controls: LPControls,
    rows: int,
    point: FloatArray,
    reported: float,
    *,
    marginals: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    iterations: int,
    message: str = "structured: minimum-norm point of the reduced polytope",
) -> LPReport:
    """A usable report of the program: status 0, the point, objective and marginals."""
    return LPReport(
        stage,
        controls,
        rows,
        0,
        status=0,
        success=True,
        message=message,
        iterations=int(iterations),
        reported=float(reported),
        failure="",
        point=np.asarray(point, dtype=np.float64),
        marginals=cast(
            "Marginal", tuple(np.asarray(m, dtype=np.float64) for m in marginals)
        ),
    )


def calibration_supported(problem: GammaProblem) -> bool:
    """Whether the ordinary mixed sample can try the dense calibration."""
    if problem.grouping is not None or problem.coordinates is not None:
        return False
    positive, k = problem.positive, problem.n_coordinates
    return positive.size != problem.n_rows and 0 < k <= min(
        positive.size, MAX_DIMENSION
    )


def calibrated_reports(
    problem: GammaProblem, data: LPData, controls: LPControls
) -> tuple[LPReport, LPReport] | None:
    """Propose both duals from one positive representation of the full total."""
    if not calibration_supported(problem):
        return None
    positive, k = problem.positive, problem.n_coordinates
    rho_p = data.positive_weights
    try:
        with np.errstate(all="raise"):
            a = data.scaled[positive]
            gram = a.T @ (rho_p[:, None] * a)
            correction = np.linalg.solve(gram, data.total - data.positive_total)
            loading = rho_p * (1.0 + a @ correction)
            if not (np.isfinite(loading) & (loading > 0)).all():
                return None
            # A_P' loading = t. For LP2, A_P'(kappa * loading - rho_p)
            # plus (-kappa) * t gives -p. The factor of two leaves sign margin.
            kappa = 2.0 * np.max(rho_p / loading)
            boundary = rho_p - kappa * loading
    except (FloatingPointError, np.linalg.LinAlgError):
        return None
    zero = np.zeros(k)
    message = "structured: positive loadings calibrated to the full total"
    first = _report(
        "negative",
        controls,
        positive.size,
        zero,
        0.0,
        marginals=(-loading, np.zeros(0), zero, zero),
        iterations=0,
        message=message,
    )
    second = _report(
        "zero_sum",
        controls,
        positive.size + 1,
        zero,
        0.0,
        marginals=(boundary, np.array([-kappa]), zero, zero),
        iterations=0,
        message=message,
    )
    return first, second


def _direction(reduced: Reduced, u: FloatArray) -> FloatArray:
    """The LP point of an outer ``u``: the least ``a``, scaled into the box."""
    scores = reduced.scores(u)
    low = np.full(reduced.n.size, np.inf)
    np.minimum.at(low, reduced.group, scores)
    v = np.zeros(reduced.k)
    v[reduced.outer], v[reduced.levels] = u, -low
    return v / float(np.max(np.abs(v)))


def _near(reduced: Reduced, corral: Corral) -> bool:
    """Whether the corral's point is within rounding of its target."""
    scale = max(float(np.abs(reduced.c).max()), float(reduced.n.max()), 1.0)
    return float(np.abs(corral.x).max()) <= NEAR * scale


def representations(
    reduced: Reduced, corral: Corral, target: FloatArray
) -> Iterator[Corral]:
    """Corrals representing ``target``: Wolfe's own, then its completion.

    Nothing is yielded when the corral's point is not within rounding of the target.
    Wolfe's corral comes first when it settled; :func:`finish` completes it when it
    spans only a face.  Each is refined; the caller keeps the first that checks.
    """
    if not _near(reduced, corral):
        return
    if corral.settled:
        yield refine(reduced, corral, target)
    finished = finish(reduced, corral, target)
    if finished is not corral and finished.settled:
        yield refine(reduced, finished, target)


def _negative(
    problem: GammaProblem,
    data: LPData,
    controls: LPControls,
    reduced: Reduced,
) -> LPReport | None:
    """LP1: a representation of 0 (dual) or the direction of the minimum-norm point."""
    rows, k = problem.positive.size, reduced.k
    t = np.asarray(stage_objective(data, "negative"))
    zeros = np.zeros(k)
    none = (np.zeros(rows), np.zeros(0), zeros, zeros)
    if reduced.empty.size:
        v = np.zeros(k)
        v[reduced.levels[reduced.empty]] = -1.0
        return _report(
            "negative", controls, rows, v, float(t @ v), marginals=none, iterations=0
        )
    corral = wolfe(reduced, np.zeros(reduced.m), min(controls.max_iterations, CYCLES))
    for found in representations(reduced, corral, np.zeros(reduced.m)):
        report = _report(
            "negative",
            controls,
            rows,
            zeros,
            0.0,
            marginals=(-multipliers(reduced, found), np.zeros(0), zeros, zeros),
            iterations=corral.iterations,
        )
        if _evidence(problem, controls, "negative", report):
            return report
    if not corral.settled or _near(reduced, corral):
        return None
    v = _direction(reduced, -corral.x)
    return _report(
        "negative",
        controls,
        rows,
        v,
        float(t @ v),
        marginals=none,
        iterations=corral.iterations,
    )


def positive_multipliers(
    reduced: Reduced, budget: int
) -> Iterator[tuple[FloatArray, int]]:
    """Representations of 0 positive on every row of ``P``, by depth.

    Yields ``(lambda, iterations)``: the mixtures of the uniform multipliers with a
    representation of ``-eps r``, for each ``eps`` of :data:`DEPTHS` in turn.  The
    major cycles of all the computations share ``budget``.
    """
    spread = uniform(reduced)
    r = represented(reduced, spread)
    base = wolfe(reduced, np.zeros(reduced.m), budget)
    used = base.iterations
    if _near(reduced, base):
        base = finish(reduced, base, np.zeros(reduced.m))
    if not base.settled:
        return
    for eps in DEPTHS:
        if used >= budget:
            return
        corral = wolfe(reduced, -eps * r, budget - used, base)
        used += corral.iterations
        for found in representations(reduced, corral, -eps * r):
            yield (multipliers(reduced, found) + eps * spread) / (1.0 + eps), used


def _zero_sum(
    problem: GammaProblem,
    data: LPData,
    controls: LPControls,
    reduced: Reduced,
) -> LPReport | None:
    """LP2: the dual at ``v = 0`` of the first positive representation that checks."""
    rho_p = np.asarray(data.positive_weights)
    rows, k = problem.positive.size + 1, reduced.k
    zeros = np.zeros(k)
    for lam, used in positive_multipliers(
        reduced, min(controls.max_iterations, CYCLES)
    ):
        kappa = float(np.max(rho_p / lam)) * (1.0 + 1e-6)
        report = _report(
            "zero_sum",
            controls,
            rows,
            zeros,
            0.0,
            marginals=(-(kappa * lam - rho_p), np.array([-kappa]), zeros, zeros),
            iterations=used,
        )
        if _evidence(problem, controls, "zero_sum", report):
            return report
    return None


def _evidence(
    problem: GammaProblem,
    controls: LPControls,
    stage: Stage,
    report: LPReport,
) -> bool:
    """Whether the report already passes the fresh check its verdict will rest on."""
    # This path receives the structured builder's complete report.
    point = cast("tuple[FloatArray, float]", (report.point, report.reported))
    if np.any(report.point != 0):
        return primal_witness(problem, controls, stage, *point).kind is not None
    return dual_check(
        problem, controls, stage, point, cast("Marginal", report.marginals)
    ).passed


def structured_call(
    problem: GammaProblem,
    data: LPData,
    stage: Stage,
    controls: LPControls,
) -> LPReport | None:
    """The report of ``stage`` by the structured method, or None to run the LP.

    A report is returned only when the fresh check its verdict rests on (the primal
    witness of a direction, the dual check of multipliers at ``v = 0``) passes on it.
    With no slope left (``m = 0``) it declines, as above its largest dimension: the
    minimum-norm iteration has no coordinate, and the program answers as before.
    """
    reduced = reduce(problem, data)
    if (
        reduced is None
        or not problem.positive.size
        or not 0 < reduced.m <= MAX_DIMENSION
    ):
        return None
    report = (_negative if stage == "negative" else _zero_sum)(
        problem, data, controls, reduced
    )
    if report is None or not _evidence(problem, controls, stage, report):
        return None
    return report
