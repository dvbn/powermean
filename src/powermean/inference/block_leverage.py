"""Exact full-design leverage by block elimination: no solve per row.

Owner's rulings 55, 63 and 65 (13 November 2026); ``HC_DESIGN_NOTE.md``, Section 5.1.
Private.  The quantity is the one of :mod:`powermean.inference.leverage`: with the
full design ``D = [X, E_J]`` (slopes and the kept indicator coordinates of the
effects) and the Gram weights ``g``,

    q_i = g_i d_i' (D' diag(g) D)^-1 d_i,

the diagonal of the hat of the system the fit solves; the per-replica leverage is
``q_i / f_i`` (:func:`block_leverage`).

Elimination.  The kept coordinates of one block (the lead block: the block with most
kept coordinates) have a diagonal Gram block, since a row has one level per block.
They are eliminated in closed form.  With ``c(i)`` the lead coordinate of row ``i``
(none where its level is a reference of the basis), ``G_c = sum_{j in c} g_j``,
``pi_i = g_i / G_c(i)`` (zero for a row without lead coordinate), ``z_i`` the other
coordinates of the row (its slopes and the kept indicators of the other blocks) and
``zt_i = z_i - sum_{j in c(i)} pi_j z_j``,

    M = sum_i g_i zt_i zt_i',        q_i = pi_i + g_i zt_i' M^-1 zt_i,

the partitioned inverse, exact in real arithmetic for any choice of the lead block
and any basis ``J`` of the indicator space.  ``M`` is dense of order
``K + |J| - |J_lead|``: one Cholesky factor and its inverse (LAPACK ``dpotrf``,
``dpotri``; the diagonal of a hat needs entries of the inverse, not a solve).

No matrix with one row per observation and one column per coordinate is formed.  The
indicator part of ``zt_i`` is ``f_i - fbar_c(i)``: ``f_i`` has at most one unit entry
per block, and ``fbar_c`` is supported on the cells of the lead group (its distinct
other-block coordinates).  ``M`` is assembled from the sparse incidence of the rows
and of the cells; the quadratic forms are gathered from ``M^-1`` per row and per
pair of cells of one lead group (work ``sum_c (cells of c)^2``), the pairs taken
``PAIRS_PER_GATHER`` at a time (a bound on a transient array, not a tolerance: no
value depends on it).

Failure.  A reduced matrix that is not numerically positive definite has no Cholesky
factor: ``ValueError``.  An allocation that fails raises the ordinary
``MemoryError``; there is no budget of the package (ruling 65).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from powermean.fe.absorb_numeric import checked_product
from powermean.fe.certificate_basis import CertificateBasis
from powermean.inference.leverage import (
    Leverage,
    _frequency,
    _multiplicity,
    _rows,
    _traced,
)
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result

# The pairs of cells gathered together from the inverse: a bound on the memory of
# one gather, not a tolerance; each pair is read on its own.
PAIRS_PER_GATHER = 1 << 20

_linalg = cast("Any", importlib.import_module("scipy.linalg"))
_sparse = cast("Any", importlib.import_module("scipy.sparse"))


def _group_sums(group: IndexArray, values: FloatArray, size: int) -> FloatArray:
    """The sums of the columns of ``values`` within the groups."""
    found = np.empty((size, values.shape[1]))
    for j in range(values.shape[1]):
        found[:, j] = np.bincount(group, weights=values[:, j], minlength=size)
    return found


def _cells(
    group: IndexArray, share: FloatArray, coordinates: list[IndexArray], order: int
) -> tuple[IndexArray, IndexArray, FloatArray, list[IndexArray]]:
    """The distinct (lead group, coordinate) cells, sorted by group, then coordinate.

    Returns the group and the coordinate of each cell, its value
    ``sum pi_i`` over its rows, and per block the cell of every row (``-1`` where the
    row has none: no lead coordinate, or a reference level of that block).
    """
    keys: list[IndexArray] = []
    rows: list[IndexArray] = []
    for column in coordinates:
        held = np.flatnonzero((column >= 0) & (share > 0))
        keys.append(group[held] * order + column[held])
        rows.append(held)
    stacked = np.concatenate(keys) if keys else np.zeros(0, dtype=np.intp)
    cells, inverse = np.unique(stacked, return_inverse=True)
    weights = np.concatenate([share[held] for held in rows]) if rows else np.zeros(0)
    values = np.bincount(inverse, weights=weights, minlength=cells.size)
    of_row: list[IndexArray] = []
    start = 0
    for held in rows:
        found = np.full(group.size, -1, dtype=np.intp)
        found[held] = inverse[start : start + held.size]
        start += held.size
        of_row.append(found)
    return cells // order, cells % order, values, of_row


def _reduced_inverse(reduced: FloatArray) -> FloatArray:
    """The inverse of the reduced Gram matrix by its Cholesky factor, symmetric."""
    factor, info = _linalg.lapack.dpotrf(reduced, lower=1, overwrite_a=1)
    if info:
        raise ValueError(
            "the reduced Gram matrix of the full design is not positive definite: "
            "the design is not of full column rank on the rows of positive weight"
        )
    inverse, info = _linalg.lapack.dpotri(factor, lower=1, overwrite_c=1)
    if info:
        raise ValueError("the reduced Gram matrix of the full design has no inverse")
    lower = np.tril(inverse)
    lower += np.tril(inverse, -1).T
    return cast("FloatArray", lower)


def _cell_forms(
    cell_group: IndexArray,
    cell_column: IndexArray,
    values: FloatArray,
    block: FloatArray,
) -> FloatArray:
    """``sum_l P[c_j, c_l] value_l`` over the cells ``l`` of the group of cell ``j``."""
    count = cell_group.size
    found = np.zeros(count)
    if not count:
        return found
    sizes = np.bincount(cell_group)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    size_of = sizes[cell_group]
    start_of = starts[cell_group]
    cumulative = np.cumsum(size_of)
    first = 0
    while first < count:
        base = int(cumulative[first - 1]) if first else 0
        last = int(np.searchsorted(cumulative, base + PAIRS_PER_GATHER, side="right"))
        last = min(max(last, first + 1), count)
        m = size_of[first:last]
        ends = np.cumsum(m)
        offsets = ends - m
        own = np.repeat(np.arange(first, last), m)
        other = np.repeat(start_of[first:last] - offsets, m) + np.arange(int(ends[-1]))
        terms = block[cell_column[own], cell_column[other]] * values[other]
        found[first:last] = np.add.reduceat(terms, offsets)
        first = last
    return found


@dataclass(frozen=True, eq=False)
class _Reduced:
    """The rows of positive weight after the elimination of the lead block."""

    g: FloatArray  # Gram weights
    x: FloatArray  # slopes, centered within the lead groups
    group: IndexArray  # lead group of each row; the last group holds the rows without
    share: FloatArray  # pi_i, zero in the last group
    total: FloatArray  # G_c
    others: list[IndexArray]  # coordinates of the remaining blocks
    cell_group: IndexArray
    cell_column: IndexArray
    cell_value: FloatArray
    cell_of: list[IndexArray]
    order: int


def _eliminated(
    g: FloatArray,
    code: IndexArray,
    design: FloatArray,
    others: list[IndexArray],
    order: int,
) -> _Reduced:
    """Center the slopes within the lead groups and list the cells."""
    size = int(code.max()) + 2
    group = np.where(code >= 0, code, size - 1)
    total = np.bincount(group, weights=g, minlength=size)
    share = np.where(code >= 0, g / total[group], 0.0)
    x = np.array(design, dtype=np.float64)
    x -= _group_sums(group, share[:, None] * x, size)[group]
    cell_group, cell_column, cell_value, cell_of = _cells(group, share, others, order)
    return _Reduced(
        g, x, group, share, total, others, cell_group, cell_column, cell_value,
        cell_of, order,
    )  # fmt: skip


def _incidence(rows: _Reduced) -> tuple[Any, IndexArray, IndexArray]:
    """The sparse incidence of the rows on the remaining coordinates, its entries."""
    held = [np.flatnonzero(column >= 0) for column in rows.others]
    row = np.concatenate(held)
    column = np.concatenate([c[h] for c, h in zip(rows.others, held, strict=True)])
    shape = (rows.g.size, rows.order)
    return (
        _sparse.csr_array((np.ones(row.size), (row, column)), shape=shape),
        row,
        column,
    )


def _reduced_gram(rows: _Reduced, incidence: Any | None) -> FloatArray:
    """``M = sum_i g_i zt_i zt_i'``, assembled without the centered indicators."""
    k, order = rows.x.shape[1], rows.order
    weighted = rows.g[:, None] * rows.x
    reduced = np.zeros((k + order, k + order))
    reduced[:k, :k] = rows.x.T @ weighted
    if incidence is None:
        return reduced
    matrix, row, column = incidence
    scaled = _sparse.csr_array((rows.g[row], (row, column)), shape=matrix.shape)
    reduced[k:, k:] = (matrix.T @ scaled).toarray()
    size = rows.total.size
    cells = (rows.cell_group, rows.cell_column)
    means = _sparse.csr_array((rows.cell_value, cells), shape=(size, order))
    scaled = _sparse.csr_array(
        (rows.cell_value * rows.total[rows.cell_group], cells), shape=(size, order)
    )
    reduced[k:, k:] -= (means.T @ scaled).toarray()
    reduced[k:, :k] = matrix.T @ weighted
    reduced[:k, k:] = reduced[k:, :k].T
    return reduced


def _indicator_forms(rows: _Reduced, matrix: Any, inverse: FloatArray) -> FloatArray:
    """``2 xt_i' P_xf ft_i + ft_i' P_ff ft_i`` per row, gathered from the inverse."""
    k = rows.x.shape[1]
    size = rows.total.size
    block = inverse[k:, k:]
    found = np.zeros(rows.g.size)
    if k:
        y = np.asarray(matrix @ np.ascontiguousarray(inverse[k:, :k]))
        y -= _group_sums(rows.group, rows.share[:, None] * y, size)[rows.group]
        found += 2.0 * np.einsum("ij,ij->i", rows.x, y)
    for first, a in enumerate(rows.others):
        for b in rows.others[first:]:
            both = np.flatnonzero((a >= 0) & (b >= 0))
            found[both] += (1.0 if b is a else 2.0) * block[a[both], b[both]]
    form = _cell_forms(rows.cell_group, rows.cell_column, rows.cell_value, block)
    for cell in rows.cell_of:
        held = np.flatnonzero(cell >= 0)
        found[held] -= 2.0 * form[cell[held]]
    within = np.bincount(
        rows.cell_group, weights=rows.cell_value * form, minlength=size
    )
    return found + within[rows.group]


def eliminated_forms(
    gram: FloatArray,
    lead: IndexArray,
    design: FloatArray,
    coordinates: tuple[IndexArray, ...],
    order: int,
    *,
    basis: FloatArray | None = None,
) -> FloatArray:
    """``g_i d_i' (D' diag(g) D)^-1 d_i`` per row, by elimination of the lead block.

    Parameters
    ----------
    gram : array
        The Gram weights ``g_i``, nonnegative; a row of zero weight is no row of the
        system and its form is zero.
    lead : array of int
        The eliminated coordinate of each row, ``-1`` for a row without one.  The
        rows of one coordinate are its group.
    design : array
        ``X``, one row per row and ``K`` columns (``K = 0`` allowed).
    coordinates : tuple of array of int
        Per remaining block, the coordinate of each row in ``0 .. order - 1``, ``-1``
        for a row without one (a reference level).  A row has at most one coordinate
        per block.
    order : int
        The number of remaining indicator coordinates.
    basis : array, optional
        ``T``, ``K + order`` rows and full column rank: the identified coordinates
        of ``z = (x, remaining indicators)`` where these columns are not independent
        (a cancelled fit on its spectral basis).  The forms are then those of the
        design ``z T``, gathered from ``T (T' M T)^-1 T'``; the hat does not depend
        on the basis of that column space.

    Returns
    -------
    array
        The forms, one per row given.

    Raises
    ------
    ValueError
        A reduced Gram matrix that is not positive definite (the design is not of
        full column rank on the rows of positive weight).
    """
    forms = np.zeros(gram.shape[0])
    kept = np.flatnonzero(gram > 0)
    if not kept.size:
        return forms
    rows = _eliminated(
        gram[kept], lead[kept], design[kept], [c[kept] for c in coordinates], order
    )
    incidence = _incidence(rows) if order else None
    reduced = _reduced_gram(rows, incidence)
    if basis is None:
        inverse = _reduced_inverse(reduced)
    else:
        if basis.ndim != 2 or basis.shape[0] != reduced.shape[0]:
            raise ValueError("basis must have one row per coordinate")
        inverse = basis @ _reduced_inverse(basis.T @ reduced @ basis) @ basis.T
    k = rows.x.shape[1]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        found: FloatArray = np.einsum("ij,jk,ik->i", rows.x, inverse[:k, :k], rows.x)
        if incidence is not None:
            found += _indicator_forms(rows, incidence[0], inverse)
        forms[kept] = rows.share + rows.g * found
    return _finite_result(forms, "leverage")


def _lead_block(basis: CertificateBasis) -> int:
    """The block with most kept coordinates (the first of them)."""
    kept = [int(np.count_nonzero(position >= 0)) for position in basis.positions]
    return int(np.argmax(kept))


def _compact(position: IndexArray, start: int) -> tuple[IndexArray, int]:
    """The kept levels of one block numbered from ``start``; ``-1`` at a reference."""
    found = np.full(position.shape[0], -1, dtype=np.intp)
    kept = np.flatnonzero(position >= 0)
    found[kept] = start + np.arange(kept.size)
    return found, int(kept.size)


def block_leverage(
    design: FloatArray,
    basis: CertificateBasis,
    weights: FloatArray,
    fisher: FloatArray,
    *,
    frequency: bool,
    dropped: IndexArray | None = None,
) -> Leverage:
    """The per-replica hats of the full design ``[X, E_J]`` by block elimination.

    The hats of the expected-information geometry on the full nuisance basis,
    computed as the module docstring states.

    Parameters
    ----------
    design : array
        ``X``, the scaled slopes on every row of the grouping.
    basis : CertificateBasis
        The kept nuisance coordinates ``J`` of the same rows.
    weights : array
        The raw weights ``a`` of the rows, nonnegative.
    fisher : array
        The expected row curvature ``w_i``, positive.
    frequency : bool
        Whether the weights are frequency counts.
    dropped : array of int, optional
        Rows of leverage exactly one that are no observations of the request
        (singleton rows).  The hats are computed on the full design, in which each
        of them spans one dimension; the record then leaves them out: zero hat and
        zero multiplicity, and a rank lowered by their number.

    Returns
    -------
    Leverage
        The hats, the trace and ``p``; the parts and the provenance of the solves of
        the earlier route are ``None``.

    Raises
    ------
    ValueError
        Misaligned inputs, a design that is not identified, or a trace that does not
        round to its rank.
    FloatingPointError
        A value that is not representable in float64.
    """
    kind = _frequency(frequency)
    matrix, a, w = _rows(design, weights, fisher)
    if type(basis) is not CertificateBasis:
        raise TypeError("basis must be a CertificateBasis")
    if basis.grouping.n_rows != matrix.shape[0]:
        raise ValueError("the basis belongs to other rows")
    g = checked_product(a, w, "Gram weight")
    f = _multiplicity(a, frequency=kind)
    blocks = basis.grouping.blocks
    chosen = _lead_block(basis)
    lead = np.zeros(matrix.shape[0], dtype=np.intp)
    coordinates: list[IndexArray] = []
    order = 0
    for index, (block, position) in enumerate(
        zip(blocks, basis.positions, strict=True)
    ):
        if index == chosen:
            lead = _compact(position, 0)[0][block.codes]
            continue
        compact, count = _compact(position, order)
        coordinates.append(compact[block.codes])
        order += count
    forms = eliminated_forms(g, lead, matrix, tuple(coordinates), order)
    with np.errstate(divide="ignore", invalid="ignore"):
        hat = np.where(f > 0, forms / np.where(f > 0, f, 1.0), 0.0)
    rank = int(matrix.shape[1]) + basis.n_kept
    if dropped is not None and dropped.size:
        f = np.array(f)
        f[dropped] = 0.0
        hat[dropped] = 0.0
        rank -= int(dropped.size)
    trace, p = _traced(hat, f, rank, "full-design")
    return Leverage(
        hat=_immutable(hat),
        gram_weights=_immutable(g),
        multiplicity=_immutable(f),
        frequency=kind,
        trace=trace,
        rank=rank,
        p=p,
        geometry="expected",
        slope_hat=None,
        nuisance_hat=None,
        slope_map=None,
        nuisance=None,
        diagonal=None,
    )


__all__ = ["PAIRS_PER_GATHER", "block_leverage", "eliminated_forms"]
