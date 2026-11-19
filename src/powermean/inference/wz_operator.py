"""Published CWP11/21 p.23 WZ pair covariance on present cells (design 03.5.4 / 04.7).

SUM coordinates over the ACTUAL retained rows. A directed pair g=(i,j) owns only
its present dates (T_g>=1); no P*T padding and no balanced reshape. P is the
number of present pairs. With the expected pair information
H_g=diag(mu_g)-mu_g mu_g'/sum(mu_g), A=sum_g Xtilde_g' H_g Xtilde_g and
C=sum_g D_g' H_g D_g. W_N=A/P and W_N^(phi)=C/P, so
Lev_g=H_g Xtilde_g A^-1 Xtilde_g' + H_g D_g C^-1 D_g' and
V_U=P/(P-1) A^-1 sum_g[Xtilde_g' solve(I-Lev_g,S_g) S_g' Xtilde_g] A^-1.
The inverse acts on the LEFT of S S'; there is no square root or second inverse.
The analytically symmetric information and final covariance use the accepted
range-safe symmetric mean; the one-sided meat remains separately intact.

D is an implicit, identified nuisance design, NOT a coefficient reduction map.
Each row carries its first-interaction (origin-date) and second-interaction
(destination-date) level. H_g annihilates pair constants, so D's identified
columns are the levels whose within-pair-differenced indicators are the
canonical leading (pivot) columns of the exact integer echelon of those
differences (wz_binding.Coordinates: the pair-cancelled admission's exact
quotient, or the same canonical set by the same exact elimination). They are
independent and span every direction H sees, so C is positive definite in
exact arithmetic on every present-cell topology: missing dates, unequal origin
and destination sets, countries absent at a date and patterned deficiency.
Lev and Xtilde do not depend on that choice. There is no reference gauge,
pseudoinverse, ridge or numerical rank cutoff; every factor keeps its gates.

Work. C is assembled from pair-local H_g entries (sum of T_g^2 per pair, never
d column scans of all rows). Xtilde uses ONE multi-RHS solve. Leverage blocks
gather C^-1 entries from ONE inverse of the same factor, in bounded batches of
pairs; only each pair's I-Lev keeps its own existing equilibrated LU gate.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.inference.numeric import symmetric_part
from powermean.inference.ssc import direct_errors
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.factorization import (
    Factor,
    SingularJacobianError,
    factorize,
    symmetric_cholesky,
)
from powermean.solve.options import SolverOptions

IndexArray = NDArray[np.intp]

# Matrix entries per vectorized batch of pair-local work. This bounds transient
# batch buffers only; it selects no tolerance, rank, sample or estimator.
CHUNK = 1 << 16

# Batched pair solves (``_pair_solves``). A block is taken as passing the gate
# without its own LAPACK estimate only when its exact reciprocal condition
# exceeds the threshold by this factor; ``ESTIMATE_SLACK`` is the relative
# rounding allowance when exact values are compared with LAPACK estimates.
GATE_GUARD = 16.0
ESTIMATE_SLACK = 1e-6
# The batched LU and the per-block LU round differently, by about the condition
# number times 1e-17 in a solution.  Only blocks whose exact reciprocal condition
# exceeds this bound are solved together, so that the two paths agree to 1e-9 or
# better; a batch with any other block takes the per-block path.
BATCH_RCOND = 1e-6


@dataclass(frozen=True)
class Sizes:
    """Actual present-cell sizes that determine the operator's buffers and work.

    ``nuisance`` is the identified dimension, or before identification its
    structural upper bound (exact on complete panels; see wz_binding.layout).
    ``pair_entries``/``pair_cubes`` sum T_g^2/T_g^3 over pairs with T_g>=2;
    ``country_entries``/``country_cubes`` sum T_c^2/T_c^3 over the countries of
    both roles, T_c being a country's present dates in that role.
    """

    n: int
    k: int
    pairs: int
    levels: int
    nuisance: int
    periods: int
    longest: int
    pair_entries: int
    pair_cubes: int
    country_entries: int
    country_cubes: int


@dataclass(frozen=True)
class Reservation:
    """Per-call explicit numeric buffers and array-work bound, not an OS cap.

    Native BLAS/LAPACK private workspace and Python object/label overhead are
    outside the numeric-buffer accounting, as in upstream AttemptBounds.
    No dense n*d, n*levels, n*n, P*T or cubic tensor is allocated. Dense C,
    its LU and C^-1 cost d*d each; a too-large plan refuses before capture.
    """

    bytes: int
    work: int
    n: int
    k: int
    nuisance: int
    max_pair_rows: int

    def require(self, bounds: AttemptBounds) -> None:
        """Reserve the ENTIRE call before copies, diagnostics or computations."""
        if type(bounds) is not AttemptBounds:
            raise TypeError("bounds must be the existing AttemptBounds")
        if self.bytes > bounds.max_bytes or self.work > bounds.max_work:
            raise ValueError("WZ covariance exceeds explicit byte/array-work bounds")


def _batch(sizes: Sizes) -> int:
    """Largest pair-local batch in entries: never above the panel's own entries."""
    return max(min(CHUNK, sizes.pair_entries), sizes.longest * sizes.longest)


def _plan(sizes: Sizes, *, extra_bytes: int = 0, extra_work: int = 0) -> Reservation:
    """Words of the present-cell algorithm's explicit buffers, all counted live.

    Index/layout: role, date, level, pair and coordinate codes, the pair sort,
    their owned copies and sort temporaries (24n), offsets and level maps.
    Rows/Hessian/slopes: owned x, y, mu, sorted mu/pi/pair index, H x, D z,
    Xtilde and its owned copy (8nk+4n). Dense: C, the symmetric/curvature gate
    temporaries, the LU, C^-1 and its solve temporaries (9d^2; counting these
    helpers' temporaries gives about 7d^2 alive at once, not a measurement),
    D' H x and its solve (4d(k+1)).
    Pair batches: coordinates, gathers, H/Lev blocks, X_g and A^-1 X_g'
    ((24+4k) batch entries). Work: scans/sort, pair-local assembly and
    gathers, one small LU/rcond per pair, Cholesky+LU+inverse (about 4d^3)
    and multi-RHS solves; doubled for headroom, not tuned to a fixture.
    """
    n, k, p, d = sizes.n, sizes.k, sizes.pairs, sizes.nuisance
    batch = _batch(sizes)
    slots = (
        24 * n
        + 8 * n * k
        + 4 * n
        + 8 * p
        + 4 * sizes.levels
        + 9 * d * d
        + 4 * d * (k + 1)
        + (24 + 4 * k) * batch
        + 16 * k * k
        + 8 * k
    )
    visits = (
        64 * n * (k + 1)
        + 8 * n * max(1, n.bit_length())
        + 16 * sizes.pair_entries * (k + 4)
        + 8 * sizes.pair_cubes
        + 256 * p
        + 4 * d**3
        + 32 * d * d * (k + 2)
        + 8 * n * k * (k + 4)
        + 8 * p * k * k
    )
    return Reservation(
        8 * slots + extra_bytes, 2 * visits + extra_work, n, k, d, sizes.longest
    )


@dataclass(frozen=True)
class Rows:
    """Internal immutable captured current rows; built only by adapter admission.

    ``pair_order`` sorts rows by pair then date; ``pair_offsets`` (P+1) delimit
    each pair's present rows. ``nuisance_first``/``nuisance_second`` are each
    row's identified coordinates (-1 for a level outside the identified set).
    """

    x: FloatStorage
    y: FloatStorage
    mu: FloatStorage
    origins: IndexStorage
    destinations: IndexStorage
    dates: IndexStorage
    pair_order: IndexStorage
    pair_offsets: IndexStorage
    nuisance_first: IndexStorage
    nuisance_second: IndexStorage
    origin_count: int
    destination_count: int
    periods: int
    pairs: int
    dimension: int
    names: tuple[str, ...]
    binding_sha256: str
    route: str
    identification: str
    sizes: Sizes


@dataclass(frozen=True)
class Information:
    """Same-point Xtilde and slope factor shared by covariance and coefficient.

    Valid only for rows whose binding digest (x, y, mu, role/date codes, pair
    order and identified coordinates) and rcond_min are identical. Fit-scoped:
    it is handed from a covariance to the same fit's coefficient and never
    stored in a result, request, option or module state.
    """

    binding_sha256: str
    rcond_min: float
    xtilde: FloatStorage
    slope: Factor
    nuisance_rcond: float


@dataclass(frozen=True)
class Provenance:
    """Counts and factor ownership; no coefficient correction factor or ssc meat."""

    source: str
    route: str
    binding_sha256: str
    origins: int
    destinations: int
    periods: int
    pairs: int
    observations: int
    nuisance_dimension: int
    identification: str
    slope_names: tuple[str, ...]
    factor_numerator: int
    factor_denominator: int
    factor_owner: str
    requested_ssc: str
    rcond_min: float
    nuisance_rcond: float
    slope_rcond: float
    minimum_pair_rcond: float
    reservation: Reservation
    status: str = "numerical_and_independent_validation_pending"


@dataclass(frozen=True)
class PairCovariance:
    """Detached immutable matrices; raw meat and one-sided meat retained apart.

    Properties create fresh metadata on immutable bytes. Raw is the unadjusted
    pair sandwich WITHOUT any ssc factor. Adjusted owns P/(P-1) exactly once.
    """

    _raw: FloatStorage
    _adjusted: FloatStorage
    _raw_meat: FloatStorage
    _one_sided_meat: FloatStorage
    _raw_se: FloatStorage
    _adjusted_se: FloatStorage
    _unscaled_adjusted: FloatStorage
    provenance: Provenance

    @property
    def raw(self) -> FloatArray:
        """Return detached ordinary covariance without an ssc multiplier."""
        return self._raw.view()

    @property
    def adjusted(self) -> FloatArray:
        """Return detached covariance with the sole published pair factor."""
        return self._adjusted.view()

    @property
    def unscaled_adjusted(self) -> FloatArray:
        """Return WZ covariance before its sole published scalar factor."""
        return self._unscaled_adjusted.view()

    @property
    def raw_meat(self) -> FloatArray:
        """Return the unadjusted pair-score meat."""
        return self._raw_meat.view()

    @property
    def one_sided_meat(self) -> FloatArray:
        """Return the retained unsymmetrized one-sided meat."""
        return self._one_sided_meat.view()

    @property
    def raw_se(self) -> FloatArray:
        """Return standard errors of the ordinary covariance."""
        return self._raw_se.view()

    @property
    def adjusted_se(self) -> FloatArray:
        """Return standard errors of the adjusted covariance."""
        return self._adjusted_se.view()


def _segments(counts: IndexArray) -> IndexArray:
    """Concatenated ``arange(c)`` for every count ``c``."""
    total = int(counts.sum())
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    found: IndexArray = np.arange(total, dtype=np.intp) - starts
    return found


def component_labels(count: int, u: IndexArray, v: IndexArray) -> IndexArray:
    """Smallest vertex of each vertex's component, by fe.connectivity's rule.

    Minimum-label propagation over the explicit edges (u, v) with pointer
    jumping: exact integer work that terminates as labels only decrease.
    """
    label = np.arange(count, dtype=np.intp)
    while u.size:
        candidate = label.copy()
        np.minimum.at(candidate, u, label[v])
        np.minimum.at(candidate, v, label[u])
        while True:
            jumped = candidate[candidate]
            if np.array_equal(jumped, candidate):
                break
            candidate = jumped
        if np.array_equal(candidate, label):
            break
        label = candidate
    return label


def pair_entries(
    starts: IndexArray, sizes: IndexArray, limit: int = CHUNK
) -> Iterator[tuple[IndexArray, IndexArray]]:
    """Ordered row pairs (r, s) inside each pair with T_g>=2, in bounded batches.

    Indices are positions in pair order. A pair is never split across batches;
    a pair larger than ``limit`` entries forms its own batch. Singleton pairs
    are skipped: their H_g, score and leverage are exactly zero.
    """
    multi = np.flatnonzero(sizes >= 2)
    if not multi.size:
        return
    ends = np.cumsum(sizes[multi] ** 2)
    lo = 0
    while lo < multi.size:
        base = 0 if lo == 0 else int(ends[lo - 1])
        hi = max(lo + 1, int(np.searchsorted(ends, base + limit, side="right")))
        group = multi[lo:hi]
        size, start = sizes[group], starts[group]
        rows = np.repeat(start, size) + _segments(size)
        repeat = np.repeat(size, size)
        yield (
            np.repeat(rows, repeat),
            np.repeat(np.repeat(start, size), repeat) + _segments(repeat),
        )
        lo = hi


def _finite(value: FloatArray) -> FloatArray:
    if not np.isfinite(value).all():
        raise FloatingPointError("WZ covariance intermediate is not finite")
    return value


def _curvature(matrix: FloatArray, what: str) -> None:
    """Existing Cholesky curvature gate, refused as the existing typed class.

    LAPACK potrf's raw LinAlgError is the only exception translated, at this one
    call; no threshold, ridge or repair is added.
    """
    try:
        symmetric_cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise SingularJacobianError(
            f"WZ {what} information is not positive definite"
        ) from error


class PairHessian:
    """H_g on each pair's present rows, in pair order, without a global H."""

    def __init__(self, rows: Rows) -> None:
        self.order = rows.pair_order.view()
        offsets = rows.pair_offsets.view()
        self.starts, self.sizes = offsets[:-1], np.diff(offsets)
        self.pair = np.repeat(np.arange(self.sizes.size, dtype=np.intp), self.sizes)
        self.mu = rows.mu.view()[self.order]
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            self.total = np.add.reduceat(self.mu, self.starts)
            if not np.isfinite(self.total).all() or np.any(self.total <= 0):
                raise FloatingPointError(
                    "pair fitted-mean total is not representable positive"
                )
            self.pi = self.mu / self.total[self.pair]

    def sums(self, values: FloatArray) -> FloatArray:
        """Pair sums of values given in pair order."""
        found: FloatArray = np.add.reduceat(values, self.starts, axis=0)
        return found

    def action(self, v: FloatArray) -> FloatArray:
        """Apply expected pair information to (n,) or (n,k) original-order values."""
        ordered = v[self.order]
        if v.ndim == 1:
            mean = self.sums(self.pi * ordered)[self.pair]
            image = self.mu * (ordered - mean)
        else:
            mean = self.sums(self.pi[:, None] * ordered)[self.pair]
            image = self.mu[:, None] * (ordered - mean)
        out = np.empty(v.shape, dtype=np.float64)
        out[self.order] = image
        return _finite(out)


class Nuisance:
    """Implicit identified D and D' on original rows; no row-by-level matrix."""

    def __init__(self, rows: Rows) -> None:
        self.first = rows.nuisance_first.view()
        self.second = rows.nuisance_second.view()
        self.dimension = rows.dimension
        self._a = np.flatnonzero(self.first >= 0)
        self._b = np.flatnonzero(self.second >= 0)

    def matvec(self, z: FloatArray) -> FloatArray:
        """Apply D to (d,) or (d,k) coordinates."""
        out = np.zeros((self.first.size, *z.shape[1:]), dtype=np.float64)
        out[self._a] = z[self.first[self._a]]
        out[self._b] += z[self.second[self._b]]
        return out

    def rmatvec(self, v: FloatArray) -> FloatArray:
        """Apply D' to (n,) or (n,k) row values."""
        columns = v[:, None] if v.ndim == 1 else v
        out = np.empty((self.dimension, columns.shape[1]), dtype=np.float64)
        a, b = self.first[self._a], self.second[self._b]
        for column in range(columns.shape[1]):
            out[:, column] = np.bincount(
                a, weights=columns[self._a, column], minlength=self.dimension
            ) + np.bincount(
                b, weights=columns[self._b, column], minlength=self.dimension
            )
        return out[:, 0] if v.ndim == 1 else out


def _nuisance_information(h: PairHessian, nuisance: Nuisance) -> FloatArray:
    """C = sum_g sum_{r,s in g} H_g[r,s] (e_a_r+e_b_r)(e_a_s+e_b_s)', pair-locally."""
    d = nuisance.dimension
    first, second = nuisance.first[h.order], nuisance.second[h.order]
    c = np.zeros(d * d, dtype=np.float64)
    for rr, ss in pair_entries(h.starts, h.sizes):
        weight = h.mu[rr] * ((rr == ss) - h.pi[ss])
        for left, right in (
            (first, first),
            (first, second),
            (second, first),
            (second, second),
        ):
            u, v = left[rr], right[ss]
            keep = (u >= 0) & (v >= 0)
            np.add.at(c, u[keep] * d + v[keep], weight[keep])
    return c.reshape(d, d)


def _information(
    rows: Rows,
    h: PairHessian,
    nuisance: Nuisance,
    options: SolverOptions,
    *,
    inverse: bool,
) -> tuple[Information, FloatArray | None]:
    """Assemble and gate C and A once at the current point; C^-1 only on request."""
    x = rows.x.view()
    c = symmetric_part(_finite(_nuisance_information(h, nuisance)))
    _curvature(c, "nuisance")  # Actual curvature gate; no threshold/ridge/repair.
    cf = factorize(c, options.rcond_min)
    del c
    # ONE multi-RHS solve residualizes every slope on the identified nuisance.
    fitted = nuisance.matvec(cf.solve(nuisance.rmatvec(h.action(x))))
    xtilde = _finite(x - fitted)
    del fitted
    a = symmetric_part(_finite(xtilde.T @ h.action(xtilde)))
    _curvature(a, "slope")
    af = factorize(a, options.rcond_min)
    full = (
        _finite(cf.solve(np.eye(nuisance.dimension, dtype=np.float64)))
        if inverse
        else None
    )
    shared = Information(
        rows.binding_sha256,
        options.rcond_min,
        FloatStorage.store(xtilde),
        af,
        cf.rcond,
    )
    return shared, full


def information(
    rows: Rows, options: SolverOptions, shared: Information | None
) -> Information:
    """Reuse same-point information for identical rows, else form it afresh."""
    if (
        shared is not None
        and shared.binding_sha256 == rows.binding_sha256
        and shared.rcond_min == options.rcond_min
    ):
        return shared
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        found, _ = _information(
            rows, PairHessian(rows), Nuisance(rows), options, inverse=False
        )
    return found


def _nuisance_blocks(
    inverse: FloatArray, first: IndexArray, second: IndexArray
) -> FloatArray:
    """Batched D_g C^-1 D_g' gathered from ONE inverse, for (B, m) coordinates."""
    blocks = np.zeros((*first.shape, first.shape[-1]), dtype=np.float64)
    for left in (first, second):
        for right in (first, second):
            keep = (left[:, :, None] >= 0) & (right[:, None, :] >= 0)
            entry = inverse[
                np.maximum(left, 0)[:, :, None], np.maximum(right, 0)[:, None, :]
            ]
            blocks += np.where(keep, entry, 0.0)
    return blocks


def _pair_solves_each(
    left: FloatArray, score: FloatArray, rcond_min: float, minimum: float
) -> tuple[FloatArray, float]:
    """Each pair's own equilibrated LU, gate and solve, one block at a time."""
    corrected = np.empty_like(score)
    for b in range(left.shape[0]):
        factor = factorize(left[b], rcond_min)
        minimum = min(minimum, factor.rcond)
        corrected[b] = factor.solve(score[b])
    return corrected, minimum


def _pair_solves(
    left: FloatArray, score: FloatArray, rcond_min: float, minimum: float
) -> tuple[FloatArray, float]:
    """``solve(I-Lev_g, S_g)`` for a batch of pairs of one size, same gate.

    The gate of every block stays ``factorize``: equilibration, LU with partial
    pivoting, LAPACK's 1-norm estimate and ``rcond > rcond_min``. This function
    only avoids calling it block by block where its verdict is already known.

    The blocks are equilibrated together with ``factorize``'s own arithmetic.
    From the inverse of each equilibrated block, ``exact`` is its reciprocal
    1-norm condition number. LAPACK's estimate bounds the norm of the inverse
    from below, so its reciprocal condition is never below ``exact`` beyond
    rounding: a block with ``exact > GATE_GUARD * rcond_min`` passes the gate.
    Blocks that pass it and are conditioned better than ``BATCH_RCOND`` are solved
    together by LU with partial pivoting of the same equilibrated matrix. A batch
    with any other block (a zero or non-finite scale, a singular block, ``exact``
    within the guard of the threshold or not above ``BATCH_RCOND``) is handed
    whole to ``_pair_solves_each``, which decides, raises and reports exactly as
    before.

    The reported minimum is LAPACK's estimate as before: ``factorize`` is run
    on the block of smallest ``exact`` and on every block whose ``exact`` does
    not exceed the smallest estimate found so far, which are the only blocks
    that can attain the minimum of the estimates.
    """
    if left.shape[0] == 0:
        return np.empty_like(score), minimum
    absolute = np.abs(left)
    scale = np.sqrt(np.maximum(absolute.max(axis=1), absolute.max(axis=2)))
    if not np.isfinite(scale).all() or np.any(scale == 0):
        return _pair_solves_each(left, score, rcond_min, minimum)
    scaled = left / scale[:, :, None] / scale[:, None, :]
    try:
        with np.errstate(all="ignore"):
            inverse = np.linalg.inv(scaled)
            norms = np.abs(scaled).sum(axis=1).max(axis=1) * np.abs(inverse).sum(
                axis=1
            ).max(axis=1)
            exact = 1.0 / norms
    except np.linalg.LinAlgError:
        return _pair_solves_each(left, score, rcond_min, minimum)
    del inverse
    floor = max(GATE_GUARD * rcond_min, BATCH_RCOND)
    if not np.isfinite(exact).all() or np.any(exact <= floor):
        return _pair_solves_each(left, score, rcond_min, minimum)
    # Every block passes the gate. The estimates that can attain the minimum:
    for b in np.argsort(exact, kind="stable").tolist():
        if exact[b] > minimum * (1.0 + ESTIMATE_SLACK):
            break
        minimum = min(minimum, factorize(left[b], rcond_min).rcond)
    solved = np.linalg.solve(scaled, (score / scale)[:, :, None])[:, :, 0]
    corrected: FloatArray = _finite(solved / scale)
    return corrected, minimum


def _pair_meats(
    h: PairHessian,
    nuisance: Nuisance,
    inverse: FloatArray,
    shared: Information,
    *,
    y: FloatArray,
    options: SolverOptions,
) -> tuple[FloatArray, FloatArray, float]:
    """Raw and one-sided adjusted meats over pairs with T_g>=2, batched by size."""
    xtilde = shared.xtilde.view()[h.order]
    k = xtilde.shape[1]
    ys = y[h.order]
    first, second = nuisance.first[h.order], nuisance.second[h.order]
    raw, adjusted, minimum = np.zeros((k, k)), np.zeros((k, k)), 1.0
    groups = np.flatnonzero(h.sizes >= 2)
    groups = groups[np.argsort(h.sizes[groups], kind="stable")]
    lengths, starts, counts = np.unique(
        h.sizes[groups], return_index=True, return_counts=True
    )
    for m, start, count in zip(
        lengths.tolist(), starts.tolist(), counts.tolist(), strict=True
    ):
        same_length = groups[start : start + count]
        step = max(1, CHUNK // (m * m))
        identity = np.eye(m, dtype=np.float64)
        for lo in range(0, same_length.size, step):
            at = h.starts[same_length[lo : lo + step], None] + np.arange(
                m, dtype=np.intp
            )
            mu, pi, yg, xg = h.mu[at], h.pi[at], ys[at], xtilde[at]
            score = yg - pi * yg.sum(axis=1, keepdims=True)
            solved = shared.slope.solve(xg.reshape(-1, k).T).T.reshape(xg.shape)
            block = xg @ np.swapaxes(solved, 1, 2) + _nuisance_blocks(
                inverse, first[at], second[at]
            )
            leverage = mu[:, :, None] * (block - pi[:, None, :] @ block)
            left = _finite(identity - leverage)
            # ONE-SIDED inverse before S S'.
            corrected_score, minimum = _pair_solves(
                left, score, options.rcond_min, minimum
            )
            ordinary = np.einsum("bm,bmk->bk", score, xg)
            corrected = np.einsum("bm,bmk->bk", corrected_score, xg)
            raw += ordinary.T @ ordinary
            adjusted += corrected.T @ ordinary
    return raw, adjusted, minimum


def evaluate(
    rows: Rows,
    options: SolverOptions,
    bounds: AttemptBounds,
    *,
    ssc: str,
    plan: Reservation,
) -> PairCovariance:
    """Internal admitted operator; no fit, root solve or coefficient-bias call."""
    return evaluate_information(rows, options, bounds, ssc=ssc, plan=plan)[0]


def evaluate_information(
    rows: Rows,
    options: SolverOptions,
    bounds: AttemptBounds,
    *,
    ssc: str,
    plan: Reservation,
) -> tuple[PairCovariance, Information]:
    """The covariance and its same-point information, under the caller's plan.

    Adapters are the entry points. Every factor uses options.rcond_min,
    including each pair's I-Lev; singular or ill-conditioned blocks refuse with
    the existing typed error, without an invented ceiling or fallback.
    """
    plan.require(bounds)
    y = rows.y.view()
    k, p = len(rows.names), rows.pairs
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        h, nuisance = PairHessian(rows), Nuisance(rows)
        shared, inverse = _information(rows, h, nuisance, options, inverse=True)
        assert inverse is not None
        raw_meat, adjusted_meat, minimum_rcond = _pair_meats(
            h, nuisance, inverse, shared, y=y, options=options
        )
        del inverse
        af = shared.slope
        raw_unsymmetric = af.solve(af.solve(raw_meat).T).T
        adjusted_unsymmetric = af.solve(af.solve(adjusted_meat).T).T
        raw = symmetric_part(_finite(raw_unsymmetric))
        # All averages have canceled; this is the sole mandatory P/(P-1).
        adjusted = symmetric_part(_finite(adjusted_unsymmetric * (p / (p - 1))))
        order = OrderScale(1.0, np.ones(k))  # Raw original slopes: mapping once.
        raw_se, adjusted_se = direct_errors(raw, order), direct_errors(adjusted, order)
        provenance = Provenance(
            "Weidner-Zylkin CWP11/21 (2021-03-06), PDF25/printed23; eqs13,16",
            rows.route,
            rows.binding_sha256,
            rows.origin_count,
            rows.destination_count,
            rows.periods,
            p,
            rows.x.shape[0],
            rows.dimension,
            rows.identification,
            rows.names,
            p,
            p - 1,
            "published_WZ_only",
            ssc,
            options.rcond_min,
            shared.nuisance_rcond,
            af.rcond,
            minimum_rcond,
            plan,
        )
        result = PairCovariance(
            FloatStorage.store(raw),
            FloatStorage.store(adjusted),
            FloatStorage.store(raw_meat),
            FloatStorage.store(adjusted_meat),
            FloatStorage.store(raw_se),
            FloatStorage.store(adjusted_se),
            FloatStorage.store(symmetric_part(_finite(adjusted_unsymmetric))),
            provenance,
        )
    return result, shared
