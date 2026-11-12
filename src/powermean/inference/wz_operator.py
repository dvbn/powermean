"""Published CWP11/21 p.23 WZ pair covariance.

SUM coordinates: P=N^2 or N(N-1) actual pairs, A=sum Xtilde' H Xtilde, C=sum D' H D.
W_N=A/P and W_N^(phi)=C/P, so Lev=H Xtilde A^-1 Xtilde' + H D C^-1 D'.
V_U=P/(P-1) A^-1 sum[Xtilde' solve(I-Lev,S) S' Xtilde] A^-1.
The inverse acts on the LEFT of S S'; there is no square root or second inverse.
The analytically symmetric information and final covariance use the accepted
range-safe symmetric mean; the one-sided meat remains separately intact.

D is an implicit, identified nuisance design, NOT a coefficient reduction map.
For balanced directed complete or off-diagonal panels: set the last date's
effects to zero (pair constants absorb them), and the last destination's effects
to zero at every other date. The remaining (2N-1)(T-1) columns are independent
for N>=3: u_it+v_jt=0 for all i!=j, and also i=j on complete support, forces u
constant and v its negative; v_ref=0 fixes both. Self-pairs only add equations.
This is a coordinate gauge, not a pseudoinverse or a new numerical rank policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.inference.numeric import symmetric_part
from powermean.inference.ssc import direct_errors
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.factorization import Factor, factorize, symmetric_cholesky
from powermean.solve.options import SolverOptions

IndexArray = NDArray[np.intp]


@dataclass(frozen=True)
class Reservation:
    """Per-call explicit numeric buffers and array-work bound, not an OS cap.

    Native BLAS/LAPACK private workspace and Python object/label overhead are
    outside the numeric-buffer accounting, as in upstream AttemptBounds.
    No dense n*d, n*levels, n*n or cubic tensor is allocated. Dense C costs d*d;
    its LU is reserved, and a too-large C refuses before admission or snapshots.
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


def _plan(
    n: int,
    k: int,
    d: int,
    t: int,
    sweeps: int,
    *,
    extra_bytes: int = 0,
    extra_work: int = 0,
) -> Reservation:
    # Copies, immutable staging, source rebinding, recovery and outward results.
    slots = (
        1024 * n * (k + 1)
        + 32 * d * d
        + 64 * d * (k + t + 1)
        + 64 * t * t
        + 64 * k * k
        + 128 * n
        + 64 * d
    )
    # Native factor.solve's explicit scaling is covered; LAPACK itself is native.
    visits = (
        1024 * (sweeps + 1) * n * (k + 1)
        + 256 * (d + k + 1) * n * (k + 1)
        + 256 * n * (d * d + d * k + t * t + k * t + k * k + 1)
    )
    return Reservation(
        8 * slots + 256 * n + extra_bytes, visits + extra_work, n, k, d, t
    )


@dataclass(frozen=True)
class Rows:
    """Internal immutable captured current rows; built only by adapter admission."""

    x: FloatStorage
    y: FloatStorage
    mu: FloatStorage
    origins: IndexStorage
    destinations: IndexStorage
    dates: IndexStorage
    pair_order: IndexStorage
    nodes: int
    periods: int
    pairs: int
    names: tuple[str, ...]
    binding_sha256: str
    route: str


@dataclass(frozen=True)
class Provenance:
    """Counts and factor ownership; no coefficient correction factor or ssc meat."""

    source: str
    route: str
    binding_sha256: str
    nodes: int
    periods: int
    pairs: int
    observations: int
    nuisance_dimension: int
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


class Nuisance:
    """Implicit D and D' in original positional order; no row-by-level matrix."""

    def __init__(self, rows: Rows) -> None:
        self.i, self.j, self.t = (
            rows.origins.view(),
            rows.destinations.view(),
            rows.dates.view(),
        )
        self.n, self.m = rows.nodes, rows.periods - 1
        self.dimension = (2 * self.n - 1) * self.m
        self.active = self.t < self.m

    def matvec(self, v: FloatArray) -> FloatArray:
        """Apply the original implicit nuisance design to a coordinate vector."""
        out = np.zeros(self.t.size, dtype=np.float64)
        a = self.active
        out[a] = v[self.i[a] * self.m + self.t[a]]
        b = a & (self.j < self.n - 1)
        out[b] += v[self.n * self.m + self.j[b] * self.m + self.t[b]]
        return out

    def rmatvec(self, v: FloatArray) -> FloatArray:
        """Apply the original transpose nuisance action to row values."""
        a = self.active
        out = np.bincount(
            self.i[a] * self.m + self.t[a], weights=v[a], minlength=self.n * self.m
        )
        b = a & (self.j < self.n - 1)
        tail = np.bincount(
            self.j[b] * self.m + self.t[b],
            weights=v[b],
            minlength=(self.n - 1) * self.m,
        )
        return np.concatenate((out, tail))


def _finite(value: FloatArray) -> FloatArray:
    if not np.isfinite(value).all():
        raise FloatingPointError("WZ covariance intermediate is not finite")
    return value


class PairHessian:
    """H_g=diag(mu_g)-mu_g mu_g'/sum(mu_g), applied without global H."""

    def __init__(self, rows: Rows) -> None:
        self.mu = rows.mu.view()
        self.order = rows.pair_order.view()
        self.t, self.p = rows.periods, rows.pairs
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            self.total = self.mu[self.order].reshape(self.p, self.t).sum(axis=1)
            if not np.isfinite(self.total).all() or np.any(self.total <= 0):
                raise FloatingPointError(
                    "pair fitted-mean total is not representable positive"
                )
            self.pi = self.mu[self.order].reshape(self.p, self.t) / self.total[:, None]

    def action(self, v: FloatArray) -> FloatArray:
        """Apply expected pair information in original positional order."""
        block = v[self.order].reshape(self.p, self.t)
        mean = np.sum(self.pi * block, axis=1)
        out = np.empty_like(v)
        out[self.order] = self.mu[self.order] * (block - mean[:, None]).ravel()
        return _finite(out)

    def block(self, g: int) -> FloatArray:
        """Materialize one reserved expected-information pair block."""
        at = self.order[g * self.t : (g + 1) * self.t]
        mu = self.mu[at]
        return np.diag(mu) - np.outer(mu, self.pi[g])


def _information(
    x: FloatArray, nuisance: Nuisance, h: PairHessian, options: SolverOptions
) -> tuple[Factor, FloatArray, Factor]:
    """Assemble and factor the same current nuisance and slope information."""
    d, k = nuisance.dimension, x.shape[1]
    c, unit = np.empty((d, d)), np.zeros(d)
    for column in range(d):
        unit[column] = 1.0
        c[:, column] = nuisance.rmatvec(h.action(nuisance.matvec(unit)))
        unit[column] = 0.0
    c = symmetric_part(_finite(c))
    symmetric_cholesky(c)  # Actual curvature gate; no threshold/ridge/repair.
    cf = factorize(c, options.rcond_min)
    xtilde = np.empty_like(x)
    for column in range(k):
        rhs = nuisance.rmatvec(h.action(x[:, column]))
        fitted = nuisance.matvec(cf.solve(rhs))
        xtilde[:, column] = x[:, column] - fitted
    a = np.empty((k, k))
    for column in range(k):
        a[:, column] = xtilde.T @ h.action(xtilde[:, column])
    a = symmetric_part(_finite(a))
    symmetric_cholesky(a)
    af = factorize(a, options.rcond_min)
    return cf, xtilde, af


def evaluate(
    rows: Rows,
    options: SolverOptions,
    bounds: AttemptBounds,
    *,
    ssc: str,
    plan: Reservation,
) -> PairCovariance:
    """Internal admitted operator; no fit, root solve or coefficient-bias call.

    Adapters are the entry points. C is assembled by implicit D'/H/D actions,
    streaming one nuisance column at a time. Only a reserved T*T pair leverage
    block is formed. Every factor uses options.rcond_min, including I-Lev;
    singular or ill-conditioned blocks refuse, without an invented ceiling.
    """
    plan.require(bounds)
    x, y = rows.x.view(), rows.y.view()
    k, t, p = x.shape[1], rows.periods, rows.pairs
    nuisance, h = Nuisance(rows), PairHessian(rows)
    d, n = nuisance.dimension, x.shape[0]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        cf, xtilde, af = _information(x, nuisance, h, options)
        raw_meat, adjusted_meat = np.zeros((k, k)), np.zeros((k, k))
        seed = np.zeros(n)
        minimum_rcond = 1.0
        for group in range(p):
            at = h.order[group * t : (group + 1) * t]
            xg, yg = xtilde[at], y[at]
            score = yg - h.pi[group] * yg.sum()
            hg = h.block(group)
            # D_g C^-1 D_g' is streamed: NOT an n*d or global hat matrix.
            nuisance_block = np.empty((t, t))
            for column, row in enumerate(at):
                seed[row] = 1.0
                dual = nuisance.rmatvec(seed)  # D', never coefficient reduction.
                image = nuisance.matvec(cf.solve(dual))
                nuisance_block[:, column] = image[at]
                seed[row] = 0.0
            slope_block = xg @ af.solve(xg.T)
            leverage = hg @ (slope_block + nuisance_block)
            left = factorize(np.eye(t) - leverage, options.rcond_min)
            minimum_rcond = min(minimum_rcond, left.rcond)
            adjusted_score = left.solve(score)  # ONE-SIDED inverse before S S'.
            ordinary = xg.T @ score
            corrected = xg.T @ adjusted_score
            raw_meat += np.outer(ordinary, ordinary)
            adjusted_meat += np.outer(corrected, ordinary)
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
            rows.nodes,
            t,
            p,
            n,
            d,
            rows.names,
            p,
            p - 1,
            "published_WZ_only",
            ssc,
            options.rcond_min,
            cf.rcond,
            af.rcond,
            minimum_rcond,
            plan,
        )
        return PairCovariance(
            FloatStorage.store(raw),
            FloatStorage.store(adjusted),
            FloatStorage.store(raw_meat),
            FloatStorage.store(adjusted_meat),
            FloatStorage.store(raw_se),
            FloatStorage.store(adjusted_se),
            FloatStorage.store(symmetric_part(_finite(adjusted_unsymmetric))),
            provenance,
        )
