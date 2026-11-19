"""Published WZ coefficient bias on present cells, CWP11/21 Section 3.4, App. A.2.1.

At the uncorrected PPML root, A=sum Xtilde' Hbar Xtilde and
bias=solve(A, No/(No-1) U_o + Nd/(Nd-1) U_d), with the separate counts of
present origins No and destinations Nd (owner ruling 29); when they agree this
is the published single N/(N-1). U_role sums, over that role's countries, the
signed contractions -sum S'L H x + 1/2 sum tr(G[x] L K L).
H=Y(diag(p)-pp') is observed, Hbar=m(diag(p)-pp') is expected, and
G[x]=-DH[x]. The formal likelihood derivative resolves the incompatible
two-equal-index entries of the printed mixed-component appendix display.

Country contrasts. A country's expected information E_c sums its pairs' Hbar_g
over the dates at which it is present in that role; absent dates are left out.
Its null space is spanned by the indicators of the connected components of
those dates, two dates being joined when one pair of the country is present at
both. Orthonormal Helmert columns inside each component (a one-date component
has none) span exactly its range, so L_c=Q(Q'E_cQ)^-1 Q' is formed once per
country under the existing factor gate. On a complete panel Q is the fixed T-1
Helmert basis. No generic pseudoinverse is used.
No covariance output, BBP correction, nuisance correction or optimizer is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe._cancel_storage import FloatStorage
from powermean.inference.numeric import symmetric_part
from powermean.inference.wz_operator import (
    IndexArray,
    Information,
    PairHessian,
    Reservation,
    Rows,
    Sizes,
    _batch,
    _finite,
    _plan,
    component_labels,
    information,
    pair_entries,
)
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.factorization import factorize
from powermean.solve.options import SolverOptions


def workspace(sizes: Sizes) -> tuple[int, int]:
    """Additional whole-call numeric buffers/work, added BEFORE row capture.

    The shared plan separately reserves source validation, snapshots and the
    same-point information. These extra words reserve both roles' country,
    node, local-date and flat codes (8n), per-row a, Fp, v, first/third terms
    and scores (8n), pair-entry batches with both roles' indices (12 batch
    entries), both roles' flat E, K, L and F blocks (4 country entries),
    per-country Q/factor temporaries (8T^2) and slope products. Work covers
    two passes over pair entries, one T_c^3 factorization and products per
    country and the slope contractions; doubled for headroom.
    """
    n, k, t = sizes.n, sizes.k, sizes.periods
    batch = _batch(sizes)
    slots = (
        16 * n + 12 * batch + 4 * sizes.country_entries + 8 * t * t + 8 * k * (k + 1)
    )
    work = 2 * (
        32 * n
        + 24 * sizes.pair_entries
        + 16 * sizes.country_cubes
        + 8 * sizes.country_entries
        + 8 * n * k
    )
    return 8 * slots, work


def contrasts(t: int) -> FloatArray:
    """Fixed orthonormal Helmert columns spanning the KNOWN t-1 subspace."""
    q = np.zeros((t, t - 1), dtype=np.float64)
    for column in range(t - 1):
        a = column + 1
        scale = np.sqrt(float(a * (a + 1)))
        q[:a, column] = 1.0 / scale
        q[a, column] = -a / scale
    return q


def component_contrasts(component: IndexArray) -> FloatArray:
    """Helmert columns within each date component, members in date order.

    ``component`` labels a country's present dates; a component of one date
    contributes no column. A single component is exactly ``contrasts(t)``.
    """
    t = component.size
    columns = [
        _embedded(t, members)
        for members in (
            np.flatnonzero(component == value) for value in np.unique(component)
        )
        if members.size >= 2
    ]
    return np.hstack(columns) if columns else np.zeros((t, 0), dtype=np.float64)


def _embedded(t: int, members: IndexArray) -> FloatArray:
    block = np.zeros((t, members.size - 1), dtype=np.float64)
    block[members] = contrasts(int(members.size))
    return block


@dataclass(frozen=True)
class _Role:
    """One role's countries on rows in pair order; flat T_c*T_c blocks."""

    flat: IndexArray
    local: IndexArray
    base: IndexArray
    periods: IndexArray
    nodes: IndexArray
    components: IndexArray
    complete: bool = False


def _complete_role(
    codes: IndexArray, count: int, dates: IndexArray, h: PairHessian, periods: int
) -> _Role | None:
    """Use the common Helmert basis after proving every country's date coverage."""
    if (
        periods < 2
        or not h.sizes.size
        or codes.shape != dates.shape
        or dates.size != h.sizes.size * periods
        or h.starts[0] != 0
        or not np.all(h.sizes == periods)
        or not np.all(dates.reshape(-1, periods) == np.arange(periods))
    ):
        return None
    by_pair = codes.reshape(-1, periods)
    ends = by_pair[:, 0]
    if (
        int(ends.min()) < 0
        or int(ends.max()) >= count
        or not np.all(by_pair == ends[:, None])
        or not np.bincount(ends, minlength=count).all()
    ):
        return None
    # Every pair joins all dates of its country into one component. The dense
    # node and block numbering is the country-major order of the general case.
    nodes = np.arange(count + 1, dtype=np.intp) * periods
    base = nodes * periods
    flat = base[codes] + dates * periods
    labels = np.repeat(nodes[:-1], periods)
    return _Role(flat, dates, base, np.diff(nodes), nodes, labels, complete=True)


def _role(
    codes: IndexArray, count: int, dates: IndexArray, h: PairHessian, periods: int
) -> _Role:
    """Country-local present dates and their pair-connected components."""
    full = _complete_role(codes, count, dates, h, periods)
    if full is not None:
        return full
    present, node = np.unique(codes * periods + dates, return_inverse=True)
    node = node.astype(np.intp).reshape(codes.shape)
    nodes = np.searchsorted(
        present, np.arange(count + 1, dtype=np.intp) * periods
    ).astype(np.intp)
    country_periods = np.diff(nodes)
    base = np.concatenate(([0], np.cumsum(country_periods * country_periods))).astype(
        np.intp
    )
    local = node - nodes[codes]
    later = np.ones(node.size, dtype=bool)
    later[h.starts] = False
    joined = np.flatnonzero(later)
    labels = component_labels(int(present.size), node[joined - 1], node[joined])
    flat = base[codes] + local * country_periods[codes]
    return _Role(flat, local, base, country_periods, nodes, labels)


def _blocks(
    h: PairHessian, score: FloatArray, roles: tuple[_Role, ...]
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Each role's flat E_c (expected Hbar) and K_c (score meat), pair-locally."""
    expected = [np.zeros(int(role.base[-1]), dtype=np.float64) for role in roles]
    meat = [np.zeros(int(role.base[-1]), dtype=np.float64) for role in roles]
    for rr, ss in pair_entries(h.starts, h.sizes):
        weight = h.mu[rr] * ((rr == ss) - h.pi[ss])
        product = score[rr] * score[ss]
        for index, role in enumerate(roles):
            at = role.flat[rr] + role.local[ss]
            np.add.at(expected[index], at, weight)
            np.add.at(meat[index], at, product)
    return expected, meat


def _lift(
    role: _Role, expected: FloatArray, meat: FloatArray, options: SolverOptions
) -> tuple[FloatArray, FloatArray, float]:
    """L_c=Q(Q'E_cQ)^-1 Q' and F_c=L_c K_c L_c', factored ONCE per country."""
    lifted, f = np.zeros_like(expected), np.zeros_like(expected)
    minimum = 1.0
    shared = contrasts(int(role.periods[0])) if role.complete else None
    for country in range(role.periods.size):
        t = int(role.periods[country])
        lo, hi = int(role.base[country]), int(role.base[country + 1])
        q = shared
        if q is None:
            q = component_contrasts(
                role.components[role.nodes[country] : role.nodes[country + 1]]
            )
        if not q.shape[1]:
            continue
        identified = symmetric_part(_finite(q.T @ expected[lo:hi].reshape(t, t) @ q))
        factor = factorize(identified, options.rcond_min)
        minimum = min(minimum, factor.rcond)
        # Lift all RHSs in one solve; no generic pseudoinverse.
        block = q @ factor.solve(q.T)
        lifted[lo:hi] = block.ravel()
        f[lo:hi] = symmetric_part(
            _finite(block @ meat[lo:hi].reshape(t, t) @ block.T)
        ).ravel()
    return lifted, f, minimum


def _contraction(
    role: _Role,
    lifted: FloatArray,
    f: FloatArray,
    h: PairHessian,
    terms: tuple[FloatArray, FloatArray, FloatArray],
) -> FloatArray:
    """-sum S'L H x + 1/2 sum tr(G[x] L K L) over one role's pairs."""
    score, observed, xtilde = terms
    n = score.size
    a, fp = np.zeros(n), np.zeros(n)
    for rr, ss in pair_entries(h.starts, h.sizes):
        at = role.flat[rr] + role.local[ss]
        np.add.at(a, rr, lifted[at] * score[ss])
        np.add.at(fp, rr, f[at] * h.pi[ss])
    # Observed H uses Y even when the fitted and observed totals nearly agree.
    first = -observed * h.pi * (a - h.sums(h.pi * a)[h.pair])
    # tr(G[x]F)=Y*(p*(x-p'x))'*(2Fp-diag(F)), G=-DH.
    v = 2.0 * fp - f[role.flat + role.local]
    third = observed * h.pi * (v - h.sums(h.pi * v)[h.pair])
    return _finite(xtilde.T @ (first + 0.5 * third))


@dataclass(frozen=True)
class CoefficientBias:
    """Detached bias in the INPUT slope coordinates, before one OrderScale map."""

    bias: FloatStorage
    binding_sha256: str
    route: str
    origins: int
    destinations: int
    periods: int
    pairs: int
    names: tuple[str, ...]
    identification: str
    rcond_min: float
    nuisance_rcond: float
    slope_rcond: float
    minimum_group_rcond: float
    reservation: Reservation
    shared_information: bool


def evaluate(
    rows: Rows,
    options: SolverOptions,
    bounds: AttemptBounds,
    *,
    plan: Reservation,
    shared: Information | None = None,
) -> CoefficientBias:
    """Evaluate admitted rows under a fresh FULL coefficient reservation.

    The caller must have added workspace() before layout/capture. Recompute its
    minimum here so an old covariance reservation cannot authorize this call.
    ``shared`` is the same fit's covariance information at this very point; it
    is used only when its binding digest and rcond_min equal these rows'.
    """
    extra = workspace(rows.sizes)
    minimum = _plan(rows.sizes, extra_bytes=extra[0], extra_work=extra[1])
    if plan.bytes < minimum.bytes or plan.work < minimum.work:
        raise ValueError("WZ coefficient requires its complete additional reservation")
    plan.require(bounds)
    y = rows.y.view()
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        h = PairHessian(rows)
        found = information(rows, options, shared)
        ordered = y[h.order]
        observed = h.sums(ordered)[h.pair]
        terms = (ordered - h.pi * observed, observed, found.xtilde.view()[h.order])
        dates, t = rows.dates.view()[h.order], rows.periods
        roles = tuple(
            _role(codes.view()[h.order], count, dates, h, t)
            for codes, count in (
                (rows.origins, rows.origin_count),
                (rows.destinations, rows.destination_count),
            )
        )
        expected, meat = _blocks(h, terms[0], roles)
        totals, minimum_rcond = [], 1.0
        for index, role in enumerate(roles):
            lifted, f, rcond = _lift(role, expected[index], meat[index], options)
            minimum_rcond = min(minimum_rcond, rcond)
            totals.append(_contraction(role, lifted, f, h, terms))
        no, nd = rows.origin_count, rows.destination_count
        af = found.slope
        # Sole coefficient normalizations. Covariance separately owns P/(P-1).
        if no == nd:
            bias = _finite((no / (no - 1)) * af.solve(totals[0] + totals[1]))
        else:
            bias = _finite(
                af.solve((no / (no - 1)) * totals[0] + (nd / (nd - 1)) * totals[1])
            )
        return CoefficientBias(
            FloatStorage.store(bias),
            rows.binding_sha256,
            rows.route,
            no,
            nd,
            rows.periods,
            rows.pairs,
            rows.names,
            rows.identification,
            options.rcond_min,
            found.nuisance_rcond,
            af.rcond,
            minimum_rcond,
            plan,
            found is shared,
        )
