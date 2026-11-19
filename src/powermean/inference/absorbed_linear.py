"""Fixed-effect log-OLS by absorption: the count of the effects and the named HC.

Owner's rulings 5, 52, 55, 63 and 65; ``HC_DESIGN_NOTE.md``, Sections 2.5 and 4.3
(candidate D1).  Private.  The fit is the weighted least squares of ``log y`` on the
slopes and the indicators of every level of every block; it is solved by absorption
(:mod:`powermean.solve.linear_absorbed`) and holds no exact basis and no
certificate.  This module supplies what the covariance needs beside the within
solve.

Coordinates (:func:`indicator_coordinates`).  An identified set of indicator
coordinates on given rows: every level of the block with most levels (the lead
block, whose Gram block is diagonal) and, of the other blocks, a set of levels whose
indicators complete a basis of the indicator space.

* One block: its levels.
* Two blocks: one level of the second block is left out per connected component of
  the bipartite graph of the two blocks (``scipy.sparse.csgraph``): the rank is
  ``L1 + L2 - C``, exact, as in ``reghdfe``, ``fixest`` and ``pyfixest``.
* Three or more blocks: the levels of the other blocks are selected by the pivoted
  Cholesky factorization (LAPACK ``dpstrf``) of their Gram matrix after the
  elimination of the lead block, formed with unit weights on the rows given and
  equilibrated (each indicator scaled to unit norm, ``1 / sqrt(n_j)``): a level is
  left out when its pivot, the squared distance of its indicator from the span of
  the lead block and of the levels already kept, as a share of the level's rows, is
  at most 1e-9 (:data:`powermean.data.collinearity.PIVOT_TOL`).  This coordinate
  count was already relative before ruling 92: its unit-norm inputs and pivoted
  Cholesky are unchanged.  It is not the formula-order slope-omission rule.  Without
  the scaling the rounding of a dependent pivot grows with the rows of the levels
  (about ``1e-14`` to ``1e-12`` times the largest diagonal) and passes that tolerance
  from about ``1e5`` rows per level; scaled, it stays of the order of the machine
  epsilon.  The rank of an indicator design does not depend on positive weights, so
  the weights never enter this decision.

The rank is ``sum`` of the kept coordinates: the rank of the indicator design, equal
for every order in which the blocks are written.

One function forms them (:func:`exact_coordinates`): the robust, clustered and named HC
requests of a fit, at fit time and post fit, and the saturation decision read them
there, with the fit's own store, so that a fit forms them once for each set of rows.

Which count (owner's ruling 80, :func:`exact_requested`).  By default the count of the
absorbed effects is the default count of ``fixest`` 0.13 (``ssc(K.fixef = "nested",
K.exact = FALSE)``), which ``pyfixest`` 0.70 also uses (``df_k = k + sum(levels) -
(n_fe - 1)``, the blocks nested in a cluster dimension left out): the levels of every
counted block less one per counted block after the first (:func:`fixest_count`), an
upper bound on the rank, with no rank formed.  ``ssc=pm.ssc(...,
absorbed_count="exact")`` asks for the exact rank of ruling 5 (:func:`exact_count`).
HC2 to HC5 form the coordinates for their leverage whatever the count, and their
records keep the exact rank, which their trace check reads; no factor of theirs reads
a count.

Count (:func:`exact_count`).  Under ``ssc="regression"`` the count of the absorbed
parameters is the exact rank (ruling 5): the blocks nested in a requested cluster
dimension are not counted (decision D76, unchanged), the count of the others is the
rank of their indicators, less one when a block is nested.  With one or two counted
blocks this is the count of :func:`powermean.inference.absorbed_rank.absorbed_count`.

Named HC (:func:`linear_preflight`, :func:`linear_composition`).  The leverage is
that of the dummy-variable regression,

    h_i = a_i d_i' (D'AD)^+ d_i = a_i [ e_i' (E'AE)^+ e_i + xt_i' (Xt'A Xt)^-1 xt_i ],

computed by :func:`powermean.inference.block_leverage.eliminated_forms` on the
coordinates above (one Cholesky factor of the reduced matrix, no solve per row); the
trace ``sum f_i h_i`` must round to ``K = slopes + rank``, the factor of HC1 is
``N / (N - K)``.  Singleton rows, the multipliers and the note for hat values close to
one are those of :mod:`powermean.inference.absorbed_block_hc`.  A computed hat value
above ``1 - sqrt(eps)`` (``UNSTABLE_ABOVE``: the criterion of ``sandwich::meatHC`` and
of the hat check of ``fixest``) is a hat value of one: rounding decides on which side
of one a row of exact leverage one falls, so HC2 to HC5 then hold a missing covariance,
with that note.  HC0 and HC1 compute no leverage.
"""

from __future__ import annotations

import hashlib
import importlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.collinearity import PIVOT_TOL
from powermean.data.inference_requests import HCMethod
from powermean.errors import CovarianceWarning, SpecificationError
from powermean.inference.absorbed_block_hc import BlockPreflight, singleton_rows
from powermean.inference.absorbed_constant import ConstantRow
from powermean.inference.absorbed_rank import (
    AbsorbedCount,
    BlockCount,
    IntegerArray,
    _nesting,
    _numbered,
)
from powermean.inference.block_leverage import (
    _eliminated,
    _incidence,
    _reduced_gram,
    eliminated_forms,
)
from powermean.inference.bread import Bread
from powermean.inference.contributions import statistical_count
from powermean.inference.hc import (
    LEVERAGE_METHODS,
    HCCovariance,
    Multipliers,
    _method,
    _unstable,
    adjusted_estimate,
    missing_covariance,
    multipliers,
)
from powermean.inference.leverage import Leverage, _frequency, _multiplicity, _traced
from powermean.inference.ssc import (
    ResolvedConvention,
    apply_convention,
    convention_counts,
    scaled_covariance,
)
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array

_lapack = cast("Any", importlib.import_module("scipy.linalg.lapack"))
_sparse = cast("Any", importlib.import_module("scipy.sparse"))
_csgraph = cast("Any", importlib.import_module("scipy.sparse.csgraph"))


@dataclass(frozen=True, eq=False)
class Coordinates:
    """An identified set of indicator coordinates of the blocks on given rows.

    Attributes
    ----------
    rows : array of int
        The rows the coordinates were formed on, increasing.
    lead : array of int
        Per row of the sample, the coordinate of its level of the lead block;
        ``-1`` outside ``rows``.
    others : tuple of array of int
        Per remaining block, the coordinate in ``0 .. order - 1`` of the row's
        level, ``-1`` where the level is left out and outside ``rows``.
    order : int
        The kept coordinates of the remaining blocks.
    levels, kept : tuple of int
        Per block, in the order given: its levels on ``rows`` and how many of them
        are coordinates.  Only the sum of ``kept`` is a property of the design; its
        split between the blocks is that of the basis chosen.
    """

    rows: IndexArray
    lead: IndexArray
    others: tuple[IndexArray, ...]
    order: int
    levels: tuple[int, ...]
    kept: tuple[int, ...]

    @property
    def rank(self) -> int:
        """The rank of the indicator design on ``rows``."""
        return int(sum(self.kept))


def _references(
    first: IntegerArray, left: int, second: IntegerArray, right: int
) -> Any:
    """One level of the second block per connected component of the two blocks."""
    size = left + right
    graph = _sparse.csr_array(
        (np.ones(first.size, dtype=np.int8), (first, second + left)),
        shape=(size, size),
    )
    _, label = _csgraph.connected_components(graph, directed=False)
    # Every level is present on the rows, so every component holds a level of each
    # block; the first level of the second block met in a component is left out.
    _, reference = np.unique(label[left:], return_index=True)
    return reference


def _selected(
    lead: IndexArray, others: list[IndexArray], sizes: list[int]
) -> IndexArray:
    """The levels of the other blocks kept by the pivoted Cholesky factorization."""
    order = int(sum(sizes))
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    columns = [codes + int(start) for codes, start in zip(others, offsets, strict=True)]
    rows = _eliminated(
        np.ones(lead.size), np.asarray(lead, dtype=np.intp),
        np.zeros((lead.size, 0)), columns, order,
    )  # fmt: skip
    gram = _reduced_gram(rows, _incidence(rows))
    # Equilibration (module docstring): unit-norm indicators; every level is present
    # on the rows, so each count is at least one.
    scale = 1.0 / np.sqrt(np.bincount(np.concatenate(columns), minlength=order))
    gram *= scale[:, None]
    gram *= scale[None, :]
    if float(gram.diagonal().max()) <= PIVOT_TOL:
        # LAPACK tests its tolerance from the second pivot on: no level adds to the
        # lead block when the largest first pivot is already below it.
        return np.zeros(0, dtype=np.intp)
    _, pivots, rank, info = _lapack.dpstrf(gram, lower=1, tol=PIVOT_TOL, overwrite_a=1)
    if info < 0:
        raise ValueError("the Gram matrix of the fixed effects cannot be factorized")
    kept: IndexArray = np.sort(np.asarray(pivots[: int(rank)], dtype=np.intp) - 1)
    return kept


def indicator_coordinates(
    codes: Sequence[IntegerArray], rows: IndexArray, n_rows: int
) -> Coordinates:
    """The coordinates of the blocks on ``rows`` (module docstring).

    Parameters
    ----------
    codes : sequence of array of int
        The level of every row of the sample in each block, at least one block.
    rows : array of int
        The rows of the request, increasing, at least one.
    n_rows : int
        The rows of the sample.
    """
    if not codes:
        raise ValueError("provide at least one block")
    numbered = [_numbered(np.asarray(block)[rows]) for block in codes]
    levels = [size for _, size in numbered]
    chosen = int(np.argmax(levels))
    rest = [index for index in range(len(codes)) if index != chosen]
    lead_codes = numbered[chosen][0]
    position: list[IntegerArray] = []
    if len(rest) == 1:
        values, size = numbered[rest[0]]
        found = np.ones(size, dtype=bool)
        found[_references(lead_codes, levels[chosen], values, size)] = False
        position.append(np.where(found, np.cumsum(found) - 1, -1))
    elif rest:
        sizes = [levels[index] for index in rest]
        kept = _selected(lead_codes, [numbered[index][0] for index in rest], sizes)
        every = np.full(int(sum(sizes)), -1, dtype=np.intp)
        every[kept] = np.arange(kept.size)
        start = 0
        for size in sizes:
            position.append(every[start : start + size])
            start += size
    lead = np.full(n_rows, -1, dtype=np.intp)
    lead[rows] = lead_codes
    others: list[IndexArray] = []
    counted = [0] * len(codes)
    counted[chosen] = levels[chosen]
    for index, held in zip(rest, position, strict=True):
        column = np.full(n_rows, -1, dtype=np.intp)
        column[rows] = held[numbered[index][0]]
        others.append(column)
        counted[index] = int(np.count_nonzero(held >= 0))
    order = int(sum(counted)) - levels[chosen]
    return Coordinates(
        frozen_index(rows), lead, tuple(others), order, tuple(levels), tuple(counted)
    )


def exact_count(
    blocks: Sequence[tuple[str, IntegerArray]],
    clusters: Sequence[IntegerArray],
    rows: IndexArray,
    coordinates: Coordinates | None = None,
) -> AbsorbedCount:
    """The count of the absorbed parameters of one request by the exact rank.

    Parameters
    ----------
    blocks : sequence of (str, array)
        The name and the codes of each block on every row of the sample.
    clusters : sequence of array
        The codes of each requested cluster dimension on ``rows`` (none for a robust
        or named HC request).
    rows : array of int
        The rows of the request.
    coordinates : Coordinates, optional
        The coordinates of every block on the same rows, where the caller holds
        them; used when no block is nested.

    Returns
    -------
    AbsorbedCount
        ``absorbed`` is the rank of the indicators of the blocks that are not nested
        in a requested dimension, less one when a block is nested;
        ``nested_adjustment`` as decision D76.  The split of the redundant levels
        between the blocks is that of the basis chosen; ``pairwise`` is empty.
    """
    names = [name for name, _ in blocks]
    numbered = [_numbered(np.asarray(codes)[rows]) for _, codes in blocks]
    witnesses = _nesting(numbered, list(clusters))
    counted = [j for j, witness in enumerate(witnesses) if witness is None]
    kept: dict[int, int] = {}
    if counted:
        if coordinates is None or len(counted) != len(blocks):
            coordinates = indicator_coordinates(
                [numbered[j][0] for j in counted],
                np.arange(rows.size, dtype=np.intp),
                int(rows.size),
            )
        kept = dict(zip(counted, coordinates.kept, strict=True))
    nested = len(counted) != len(blocks)
    records: list[BlockCount] = []
    for j, name in enumerate(names):
        levels = numbered[j][1]
        witness = witnesses[j]
        if witness is not None:
            records.append(BlockCount(name, levels, levels, witness, True))
            continue
        redundant = levels - kept[j] + (1 if nested and j == counted[0] else 0)
        records.append(BlockCount(name, levels, redundant, None, True))
    absorbed = sum(record.counted for record in records if record.nested_in is None)
    return AbsorbedCount(tuple(records), (), int(absorbed), int(nested))


def within_influence(
    within: FloatArray, weights: FloatArray, scores: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The slope influence rows and the slope map of the within regression.

    ``within`` is ``Xt``, the slopes after the absorption of the effects with the
    weights ``a``; ``scores`` the weighted residuals ``b_i = a_i e_i``.  Returns
    ``u_i = (Xt'A Xt)^-1 xt_i b_i`` and ``W`` with rows ``(Xt'A Xt)^-1 xt_i a_i``.
    The linear equation has one curvature: the observed and the expected bread are
    the same matrix.

    Raises
    ------
    FloatingPointError
        A value that is not representable in float64.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weighted = weights[:, None] * within
        solved = np.linalg.solve(within.T @ weighted, within.T).T
        influence = scores[:, None] * solved
        slope_map = weights[:, None] * solved
    if not (np.isfinite(influence).all() and np.isfinite(slope_map).all()):
        raise FloatingPointError("the slope influence is not representable in float64")
    return influence, slope_map


def within_constant(
    design: FloatArray,
    weights: FloatArray,
    scores: FloatArray,
    influence: FloatArray,
) -> ConstantRow:
    """The private reporting constant's influence row, from the within regression.

    The constant of :mod:`powermean.inference.absorbed_constant` is the weighted
    mean of the total absorbed effect, ``c = sum_i a_i (E alpha)_i / sum_i a_i``,
    and its influence row is ``c_i = b_i d_i' H^-1 l`` with ``H = D'AD`` and
    ``l = D'a / A - (xbar, 0)``, ``A = sum a``, ``xbar = X'a / A``.  The constant
    vector lies in the span of the indicators, so ``d_i' H^-1 D'a = 1``; and the
    slope rows of ``H^-1 D'`` are those of the within regression.  Hence

        c_i = b_i / A - xbar' u_i,

    ``u_i`` the slope influence rows of :func:`within_influence`: no basis and no
    factor of ``H``.  ``design`` is ``X`` in the coordinates of ``influence``.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total = float(np.sum(weights))
        mean = (weights @ design) / total
        row = scores / total - influence @ mean
    if not (np.isfinite(total) and total > 0 and np.isfinite(row).all()):
        raise FloatingPointError("the constant influence is not representable")
    return ConstantRow(
        functional=_immutable(np.zeros(0)),
        total=total,
        map=_immutable(np.zeros(0)),
        influence=_immutable(row),
        factor=None,
    )


def exact_coordinates(
    codes: Sequence[IntegerArray],
    rows: IndexArray,
    n_rows: int,
    memo: dict[bytes, Coordinates] | None = None,
) -> Coordinates:
    """:func:`indicator_coordinates` of ``codes`` on ``rows``, once per fit.

    ``memo`` is the fit's own store (a retained envelope's), keyed by a digest of the
    rows and the codes: the coordinates of the same rows of the same blocks are formed
    once; ``None`` forms them afresh.  The one place where the exact count of ruling 5
    is formed for fixed-effect log-OLS.
    """
    if memo is None:
        return indicator_coordinates(codes, rows, n_rows)
    key = _coordinates_key(codes, rows, n_rows)
    found = memo.get(key)
    if found is None:
        found = indicator_coordinates(codes, rows, n_rows)
        memo[key] = found
    return found


def exact_requested(convention: ResolvedConvention) -> bool:
    """Whether the convention in force asks for the exact count (ruling 80)."""
    return convention.values.absorbed_count == "exact"


def rank_formed(method: str | None, convention: ResolvedConvention) -> bool:
    """Whether a request forms the exact rank of the indicators (ruling 80).

    A named HC method with leverage forms the exact coordinates, and so does a
    request whose convention asks for the exact count; the rank in the counts of any
    other request is ``fixest``'s count.  ``method`` is ``None`` for a robust or
    cluster request.
    """
    return method in LEVERAGE_METHODS or exact_requested(convention)


def fixest_count(
    blocks: Sequence[tuple[str, IntegerArray]],
    clusters: Sequence[IntegerArray],
    rows: IndexArray,
) -> AbsorbedCount:
    """The default count of ``fixest`` and ``pyfixest`` (module docstring, ruling 80).

    The blocks nested in a requested cluster dimension are left out, as
    :func:`exact_count` does (decision D76); every other block counts its levels on
    ``rows``, less one for every such block after the first.  The first counted block
    loses one level when a block is nested, as in :func:`exact_count`, so that
    ``K_factor`` is ``fixest``'s ``K``.  Only the first block's count is exact.
    """
    names = [name for name, _ in blocks]
    numbered = [_numbered(np.asarray(codes)[rows]) for _, codes in blocks]
    witnesses = _nesting(numbered, list(clusters))
    nested = any(witness is not None for witness in witnesses)
    records: list[BlockCount] = []
    first = True
    for j, name in enumerate(names):
        levels = numbered[j][1]
        witness = witnesses[j]
        if witness is not None:
            records.append(BlockCount(name, levels, levels, witness, True))
            continue
        redundant = (1 if nested else 0) if first else 1
        records.append(BlockCount(name, levels, redundant, None, first))
        first = False
    absorbed = sum(count.counted for count in records if count.nested_in is None)
    return AbsorbedCount(tuple(records), (), int(absorbed), int(nested))


def fixest_rank(blocks: Sequence[tuple[str, IntegerArray]], rows: IndexArray) -> int:
    """``fixest``'s count of the indicators of every block on ``rows``.

    Its ``K`` less the slopes: the levels less one per block after the first.
    """
    levels = [_numbered(np.asarray(codes)[rows])[1] for _, codes in blocks]
    return int(sum(levels) - (len(levels) - 1))


def remember_coordinates(
    memo: dict[bytes, Coordinates],
    codes: Sequence[IntegerArray],
    rows: IndexArray,
    n_rows: int,
    coordinates: Coordinates,
) -> None:
    """Store coordinates formed before the fit's store existed (fit-time HC)."""
    memo.setdefault(_coordinates_key(codes, rows, n_rows), coordinates)


def _coordinates_key(
    codes: Sequence[IntegerArray], rows: IndexArray, n_rows: int
) -> bytes:
    """A digest of the rows, the sample size and the codes of every block."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(rows, dtype=np.intp).tobytes())
    digest.update(int(n_rows).to_bytes(8, "little"))
    for block in codes:
        digest.update(np.asarray(block, dtype=np.int64).tobytes())
    return digest.digest()


def linear_preflight(
    n_slopes: int,
    blocks: Sequence[tuple[str, IntegerArray]],
    weights: FloatArray,
    method: HCMethod,
    convention: ResolvedConvention,
    *,
    frequency: bool,
    memo: dict[bytes, Coordinates] | None = None,
) -> tuple[BlockPreflight, Coordinates | None]:
    """The counts and refusals of a named HC request on fixed-effect log-OLS.

    The record is the one of a joint fit
    (:class:`~powermean.inference.absorbed_block_hc.BlockPreflight`): the singleton
    rows are no observations of the request, ``N`` and the rank exclude them.  The
    rank is ``n_slopes`` plus that of :func:`indicator_coordinates` on the rows of
    the request.

    Raises
    ------
    SpecificationError
        ``specification.no_residual_dof`` or ``specification.conflicting``.
    """
    found = _method(method)
    a = _finite_array(weights, "weights")
    f = _multiplicity(a, frequency=_frequency(frequency))
    codes = [np.asarray(values) for _, values in blocks]
    singletons = singleton_rows(codes, f)
    kept = f > 0
    kept[singletons] = False
    rows = frozen_index(np.flatnonzero(kept))
    if not rows.size:
        raise SpecificationError(
            "specification.no_residual_dof",
            {"method": found, "count": 0, "rank": int(n_slopes)},
        )
    count = statistical_count(a[rows], frequency=frequency)
    coordinates: Coordinates | None = None
    if rank_formed(found, convention):
        coordinates = exact_coordinates(codes, rows, int(a.size), memo)
    if exact_requested(convention):
        assert coordinates is not None
        rank = int(n_slopes) + coordinates.rank
        absorbed = exact_count(blocks, (), rows, coordinates)
    else:
        # Ruling 80: fixest's count in HC1's K; the leverage methods' records keep
        # the exact rank of their coordinates, which their trace check reads.
        rank = int(n_slopes) + (
            coordinates.rank if coordinates is not None else fixest_rank(blocks, rows)
        )
        absorbed = fixest_count(blocks, (), rows)
    applied = apply_convention(
        convention,
        convention_counts(count, int(n_slopes), rank, absorbed),
        kind="robust",
        method=found,
    )
    return (
        BlockPreflight(
            found, applied, rows, singletons, count, rank, frequency, int(a.size),
            absorbed,
        ),
        coordinates,
    )  # fmt: skip


def linear_leverage(
    design: FloatArray,
    weights: FloatArray,
    preflight: BlockPreflight,
    coordinates: Coordinates,
) -> Leverage:
    """The per-replica hats of the dummy-variable regression on the request's rows.

    Raises
    ------
    ValueError
        A reduced matrix without Cholesky factor, or a trace that does not round to
        the rank of the preflight.
    """
    kind = _frequency(preflight.frequency)
    gram = np.zeros(weights.shape[0])
    gram[preflight.rows] = weights[preflight.rows]
    f = _multiplicity(gram, frequency=kind)
    forms = eliminated_forms(
        gram, coordinates.lead, design, coordinates.others, coordinates.order
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        hat = np.where(f > 0, forms / np.where(f > 0, f, 1.0), 0.0)
    trace, p = _traced(hat, f, preflight.rank, "full-design")
    return Leverage(
        hat=_immutable(hat),
        gram_weights=_immutable(gram),
        multiplicity=_immutable(f),
        frequency=kind,
        trace=trace,
        rank=preflight.rank,
        p=p,
        geometry="expected",
        slope_hat=None,
        nuisance_hat=None,
        slope_map=None,
        nuisance=None,
        diagonal=None,
    )


def linear_composition(
    design: FloatArray,
    weights: FloatArray,
    influence: FloatArray,
    preflight: BlockPreflight,
    coordinates: Coordinates | None,
    *,
    bread: Bread = "observed",
) -> HCCovariance:
    """The named HC covariance of fixed-effect log-OLS (module docstring).

    ``design`` is ``X`` on every row of the sample, ``weights`` the raw weights,
    ``influence`` the rows of :func:`within_influence` at the fit's point, and
    ``preflight`` with ``coordinates`` the request's own (:func:`linear_preflight`
    of the same blocks, weights and kind).  ``bread`` is the label of the request:
    the linear equation has one curvature and both breads are the same matrix.
    """
    if type(preflight) is not BlockPreflight or not (
        type(coordinates) is Coordinates
        or (coordinates is None and preflight.method not in LEVERAGE_METHODS)
    ):
        raise TypeError("preflight and coordinates must be those of linear_preflight")
    matrix = _finite_array(design, "design")
    a = _finite_array(weights, "weights")
    n = preflight.n_rows
    if matrix.ndim != 2 or matrix.shape[0] != n or a.shape != (n,):
        raise ValueError("the preflight belongs to other rows")
    if influence.shape != matrix.shape:
        raise ValueError("influence must hold one row per row of the design")
    found, active = preflight.method, preflight.rows
    if preflight.singletons.size:
        warnings.warn(
            CovarianceWarning(
                "covariance.singletons_excluded",
                {
                    "method": found,
                    "count": int(preflight.singletons.size),
                    "kept": preflight.count,
                },
            ),
            stacklevel=2,
        )
    leverage: Leverage | None = None
    weighted: Multipliers | None = None
    unstable = undefined = 0
    values = np.ones(active.size)
    if found in LEVERAGE_METHODS:
        assert coordinates is not None
        leverage = linear_leverage(matrix, a, preflight, coordinates)
        # The rows above UNSTABLE_ABOVE are noted once and count as hat values of
        # one (module docstring): the same test, so every such row is noted.
        unstable = _unstable(found, leverage)
        undefined = unstable
        if not undefined:
            weighted = multipliers(found, leverage, preflight.count)
            values = weighted.values
    estimate = adjusted_estimate(
        influence[active], a[active], values, frequency=preflight.frequency
    )
    scaled = scaled_covariance(estimate, preflight.applied)
    return HCCovariance(
        found,
        missing_covariance(scaled.covariance) if undefined else scaled.covariance,
        estimate,
        preflight.applied,
        leverage,
        weighted,
        bread,
        "absorbed",
        _immutable(influence[active]),
        dropped=int(preflight.singletons.size),
        unstable=unstable,
        undefined=undefined,
    )


__all__ = [
    "Coordinates",
    "exact_coordinates",
    "exact_count",
    "exact_requested",
    "fixest_count",
    "fixest_rank",
    "indicator_coordinates",
    "linear_composition",
    "linear_leverage",
    "linear_preflight",
    "rank_formed",
    "remember_coordinates",
    "within_constant",
    "within_influence",
]
