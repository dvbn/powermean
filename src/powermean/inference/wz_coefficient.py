"""Published WZ coefficient bias, CWP11/21 Section 3.4 and Appendix A.2.1.

At the uncorrected PPML root, A=sum Xtilde' Hbar Xtilde and
bias=N/(N-1) solve(A,U). U sums the signed origin/destination contractions.
H=Y(diag(p)-pp') is observed, Hbar=m(diag(p)-pp') is expected, and
G[x]=-DH[x]. The formal likelihood derivative resolves the incompatible
two-equal-index entries of the printed mixed-component appendix display.
No covariance output, BBP correction, nuisance correction or optimizer is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe._cancel_storage import FloatStorage
from powermean.inference.numeric import symmetric_part
from powermean.inference.wz_operator import (
    IndexArray,
    Nuisance,
    PairHessian,
    Reservation,
    Rows,
    _finite,
    _information,
    _plan,
)
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.factorization import factorize
from powermean.solve.options import SolverOptions


def workspace(n: int, k: int, nodes: int, t: int) -> tuple[int, int]:
    """Additional whole-call numeric buffers/work, added BEFORE row capture.

    The shared plan separately reserves source validation, snapshots, recovery
    and expected-information assembly/solves. These extra slots reserve pair
    indexing, Q/M/K/L/F, congruences and factor/solve temporaries, T*K pair
    products, and detached triplets/report mapping. Only one group's workspace
    is live. Work includes 2N contrast factorizations (T^3 arithmetic, never
    T^3 storage), both passes over each role's pairs and all slope products.
    Constants bound explicit arrays/visits conservatively; native private
    workspace and Python label overhead follow Reservation's existing scope.
    """
    slots = 16 * n * (k + 1) + 96 * t * t + 48 * t * (k + 1) + 32 * k * k + 32 * nodes
    work = (
        128 * n * (t * t + t * k + k + 1)
        + 256 * nodes * (t**3 + t * t * (k + 1))
        + 128 * n * (nodes + 1)
    )
    return 8 * slots, work


def contrasts(t: int) -> FloatArray:
    """Fixed orthonormal Helmert columns spanning the KNOWN T-1 subspace."""
    q = np.zeros((t, t - 1), dtype=np.float64)
    for column in range(t - 1):
        a = column + 1
        scale = np.sqrt(float(a * (a + 1)))
        q[:a, column] = 1.0 / scale
        q[a, column] = -a / scale
    return q


def _score(y: FloatArray, p: FloatArray) -> FloatArray:
    return y - y.sum() * p


def _group(
    groups: IndexArray,
    h: PairHessian,
    y: FloatArray,
    xtilde: FloatArray,
    q: FloatArray,
    options: SolverOptions,
) -> tuple[FloatArray, float]:
    """One group: -sum S'L H x + 1/2 sum tr(G[x] L K L)."""
    t, k = h.t, xtilde.shape[1]
    expected, meat = np.zeros((t, t)), np.zeros((t, t))
    for g in groups:
        at = h.order[g * t : (g + 1) * t]
        score = _score(y[at], h.pi[g])
        expected += h.block(int(g))
        meat += np.outer(score, score)
    identified = symmetric_part(_finite(q.T @ expected @ q))
    factor = factorize(identified, options.rcond_min)
    # Factor ONCE per origin/destination. Lift all RHSs in one solve, then
    # reuse L for the group meat and every pair/slope; no generic pseudoinverse.
    lifted = q @ factor.solve(q.T)
    f = symmetric_part(_finite(lifted @ meat @ lifted.T))
    total = np.zeros(k)
    for g in groups:
        at = h.order[g * t : (g + 1) * t]
        p, yg = h.pi[g], y[at]
        observed_total = yg.sum()
        a = lifted @ _score(yg, p)
        # Observed H uses Y even when the fitted and observed totals nearly agree.
        first = -observed_total * p * (a - p @ a)
        v = 2.0 * (f @ p) - np.diag(f)
        # tr(G[x]F)=Y*(p*(x-p'x))'*(2Fp-diag(F)), G=-DH.
        third = observed_total * p * (v - p @ v)
        total += xtilde[at].T @ (first + 0.5 * third)
    return _finite(total), factor.rcond


@dataclass(frozen=True)
class CoefficientBias:
    """Detached bias in the INPUT slope coordinates, before one OrderScale map."""

    bias: FloatStorage
    binding_sha256: str
    route: str
    nodes: int
    periods: int
    names: tuple[str, ...]
    rcond_min: float
    nuisance_rcond: float
    slope_rcond: float
    minimum_group_rcond: float
    reservation: Reservation


def evaluate(
    rows: Rows,
    options: SolverOptions,
    bounds: AttemptBounds,
    *,
    plan: Reservation,
) -> CoefficientBias:
    """Evaluate admitted rows under a fresh FULL coefficient reservation.

    The caller must have added workspace() before layout/capture. Recompute its
    minimum here so an old covariance reservation cannot authorize this call.
    """
    extra = workspace(rows.x.shape[0], len(rows.names), rows.nodes, rows.periods)
    minimum = _plan(
        rows.x.shape[0],
        len(rows.names),
        (2 * rows.nodes - 1) * (rows.periods - 1),
        rows.periods,
        0,
        extra_bytes=extra[0],
        extra_work=extra[1],
    )
    if plan.bytes < minimum.bytes or plan.work < minimum.work:
        raise ValueError("WZ coefficient requires its complete additional reservation")
    plan.require(bounds)
    x, y = rows.x.view(), rows.y.view()
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        h, nuisance = PairHessian(rows), Nuisance(rows)
        cf, xtilde, af = _information(x, nuisance, h, options)
        q, total = contrasts(rows.periods), np.zeros(x.shape[1])
        minimum_rcond = 1.0
        for role in (rows.origins, rows.destinations):
            codes = role.view()[h.order[:: h.t]]
            order = np.argsort(codes, kind="stable")
            counts = np.bincount(codes, minlength=rows.nodes)
            start = 0
            for count in counts:
                stop = start + int(count)
                contribution, rcond = _group(
                    order[start:stop], h, y, xtilde, q, options
                )
                total += contribution
                minimum_rcond = min(minimum_rcond, rcond)
                start = stop
        # Sole coefficient normalization. Covariance separately owns P/(P-1).
        bias = _finite((rows.nodes / (rows.nodes - 1)) * af.solve(total))
        return CoefficientBias(
            FloatStorage.store(bias),
            rows.binding_sha256,
            rows.route,
            rows.nodes,
            rows.periods,
            rows.names,
            options.rcond_min,
            cf.rcond,
            af.rcond,
            minimum_rcond,
            plan,
        )
