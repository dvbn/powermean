"""Request-specific convention counts of absorbed effects: pairwise redundancy, nesting.

The ``reghdfe`` degrees-of-freedom table: redundancies exact for one and two blocks,
bounded pairwise for three or more, and whether a block is nested in a cluster
variable.  Internal module: the convention count ``df_a`` that the ``"regression"``
preset and the custom ``K*`` strings use.  It is NOT the exact full-design rank ``K``
of the certificate (``powermean.fe.certificate_basis``: ``K = n_slopes + |J|``), which
no cluster request changes, and NOT the structural bounds of
:func:`powermean.fe.connectivity.indicator_rank`, whose upper bound
``sum L - (G - 1) C`` is not the pairwise count.

The count, in the retained order of the blocks, for the cluster dimensions of ONE
request (none for a robust request):

1. A block is nested when every one of its levels maps into one level of at least one
   requested dimension: on the aligned active rows, the pairs (level, cluster code) of
   that dimension are as many as the levels.  Names prove nothing; the actual partition
   is tested.  A nested block contributes no coordinate to the count (``redundant =
   levels``) and takes no part in the pairwise comparisons.
2. The first surviving block has redundancy 0 without any nested block and 1 with one:
   with a nested block the source removes the intercept of the first surviving block.
3. Each later surviving block ``j`` has redundancy ``max(1, max_{i < j} C_ij)`` over the
   earlier surviving blocks ``i``, ``C_ij`` the connected components of the bipartite
   graph of the pair (:func:`pairwise_components`): exact for the second surviving
   block (the two-block rank), a bound from the third on.
4. ``absorbed = sum_j (levels_j - redundant_j)`` over the surviving blocks, and
   ``nested_adjustment`` is one when any block is nested, zero otherwise (
   ``K_reference = n_slopes + absorbed`` and
   ``K_factor = K_reference + nested_adjustment``,
   the latter for the clustered factor only; both formed by
   :func:`powermean.inference.ssc.convention_counts`).

The kernel works on code arrays alone: a post-fit request constructs no
:class:`~powermean.fe.groups.Block` or grouping (the retained-state guard forbids those
fit-time entries), and the levels of a block are those present on the rows given (the
active rows of the request), numbered by ``numpy.unique``.  The components of a pair
are found by the minimum-label propagation with pointer jumping of
:mod:`powermean.fe.connectivity`, written here on the two code vectors.  Every array
is read, nothing is modified, and
the result owns its tuples.

Within one count operation each block is numbered once. Nesting uses per-level
minimum/maximum cluster labels without a floating-point pair table, and the same
numbered vectors feed the original pairwise graph propagation. No count is reused
across requests, active samples or fits. Exact raw uint64 labels are never narrowed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntegerArray = NDArray[np.integer[Any]]


@dataclass(frozen=True)
class BlockCount:
    """The convention count of one block.

    Attributes
    ----------
    name : str
        The name of the block.
    levels : int
        Its levels on the rows of the request.
    redundant : int
        The levels not counted: every level of a nested block; 0 or 1 for the first
        surviving block; ``max(1, max C_ij)`` for a later surviving block.
    nested_in : int or None
        The position, in the request, of the first dimension in which the block is
        nested; ``None`` for a surviving block.
    exact : bool
        Whether the redundancy is exact (a nested block, the first two surviving
        blocks) or a pairwise bound (the third surviving block onward).
    """

    name: str
    levels: int
    redundant: int
    nested_in: int | None
    exact: bool

    @property
    def counted(self) -> int:
        """``levels - redundant``: the coordinates the block adds to the count."""
        return self.levels - self.redundant


@dataclass(frozen=True)
class AbsorbedCount:
    """The convention count ``df_a`` of the blocks for one request.

    Attributes
    ----------
    blocks : tuple of BlockCount
        One record per block, in the retained order.
    pairwise : tuple of (int, int, int)
        ``(i, j, C_ij)`` for each ordered pair ``i < j`` of SURVIVING blocks, by their
        positions among all blocks: the comparisons the count made.
    absorbed : int
        ``sum_j (levels_j - redundant_j)`` over the surviving blocks.
    nested_adjustment : int
        One when any block is nested in a requested dimension, zero otherwise.
    """

    blocks: tuple[BlockCount, ...]
    pairwise: tuple[tuple[int, int, int], ...]
    absorbed: int
    nested_adjustment: int

    @property
    def nested(self) -> tuple[int, ...]:
        """The positions of the nested blocks."""
        return tuple(
            k for k, block in enumerate(self.blocks) if block.nested_in is not None
        )


def _codes(values: object, name: str, rows: int | None) -> IntegerArray:
    """A one-dimensional integer array of nonnegative codes, one per row."""
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if rows is not None and values.shape != (rows,):
        raise ValueError(f"{name} must hold one code per row of the request")
    if np.any(values < 0):
        raise ValueError(f"{name} contains a missing label (a negative code)")
    found: IntegerArray = values
    return found


def _numbered(codes: IntegerArray) -> tuple[IntegerArray, int]:
    """The codes renumbered ``0 .. L - 1`` over the levels present, and ``L``."""
    levels, inverse = np.unique(codes, return_inverse=True)
    if type(codes) is np.ndarray and codes.dtype == np.dtype(np.intp) and np.array_equal(
        levels, np.arange(levels.size, dtype=np.intp)
    ):
        # Fit-owned canonical codes already are the numbered representation.
        # Do not retain another N-vector per block merely to relabel it identically.
        return codes, int(levels.size)
    found: IntegerArray = np.asarray(inverse, dtype=np.intp).reshape(-1)
    return found, int(levels.size)


def nested_in(block: IntegerArray, cluster: IntegerArray) -> bool:
    """Whether every level of ``block`` maps into one level of ``cluster``.

    Parameters
    ----------
    block, cluster : array
        One nonnegative integer code per row, aligned; levels need not be numbered
        contiguously.

    Returns
    -------
    bool
        True when no level of ``block`` meets two codes of ``cluster``: the distinct
        (level, code) pairs are as many as the distinct levels.  The converse (the
        cluster nested in the block) does not make the block nested.
    """
    codes = _codes(block, "block codes", None)
    groups = _codes(cluster, "cluster codes", codes.size)
    if type(codes) is not np.ndarray or type(groups) is not np.ndarray:
        # Preserve the original accepted subclass dispatch/callback boundary.
        pairs = np.unique(np.column_stack([codes, groups]), axis=0)
        return int(pairs.shape[0]) == int(np.unique(codes).size)
    numbered, levels = _numbered(codes)
    return _nested_numbered(numbered, levels, groups)


def pairwise_components(first: IntegerArray, second: IntegerArray) -> int:
    """``C_ij``: the connected components of the bipartite graph of two blocks.

    The levels of both blocks are the vertices and each row joins its two levels.
    Minimum-label propagation with pointer jumping, as
    :func:`powermean.fe.connectivity.components` finds the components of a grouping:
    the labels never increase and strictly decrease somewhere until no edge joins two
    labels; at the end two vertices share a label exactly when a path joins them.
    """
    u, left = _numbered(_codes(first, "block codes", None))
    v, right = _numbered(_codes(second, "block codes", u.size))
    return _pairwise_numbered(u, left, v, right)


def _nested_numbered(
    block: IntegerArray, levels: int, cluster: IntegerArray
) -> bool:
    """Each present level has one cluster iff its minimum equals its maximum.

    Block codes are numbered by unique, never by a narrowing signed cast. Cluster
    labels keep their original unsigned/signed dtype, including uint64 above 2**63.
    Every numbered level occurs, so initial sentinels cannot create a witness.
    """
    low = np.full(levels, np.iinfo(cluster.dtype).max, dtype=cluster.dtype)
    high = np.zeros(levels, dtype=cluster.dtype)
    np.minimum.at(low, block, cluster)
    np.maximum.at(high, block, cluster)
    return bool(np.array_equal(low, high))


def _pairwise_numbered(
    u: IntegerArray, left: int, second: IntegerArray, right: int
) -> int:
    """The original graph propagation on already numbered, aligned code vectors."""
    v = second + left
    label: IntegerArray = np.arange(left + right, dtype=np.intp)
    while True:
        candidate = label.copy()
        np.minimum.at(candidate, u, label[v])
        np.minimum.at(candidate, v, label[u])
        while True:
            jumped: IntegerArray = candidate[candidate]
            if np.array_equal(jumped, candidate):
                break
            candidate = jumped
        if np.array_equal(candidate, label):
            return int(np.unique(label).size)
        label = candidate


def _nesting(
    blocks: Sequence[tuple[IntegerArray, int]], clusters: Sequence[IntegerArray]
) -> list[int | None]:
    """For each block, the first requested dimension that nests it, or ``None``."""
    found: list[int | None] = []
    for codes, levels in blocks:
        witness = next(
            (d for d, cluster in enumerate(clusters)
             if _nested_numbered(codes, levels, cluster)), None
        )
        found.append(witness)
    return found



def _general_nesting(
    blocks: Sequence[IntegerArray], clusters: Sequence[IntegerArray]
) -> list[int | None]:
    """The original caller/subclass entry order; no new numeric subclass actions."""
    return [
        next((d for d, cluster in enumerate(clusters) if nested_in(codes, cluster)), None)
        for codes in blocks
    ]


def absorbed_count(
    blocks: Sequence[tuple[str, IntegerArray]],
    clusters: Sequence[IntegerArray] = (),
) -> AbsorbedCount:
    """The convention count of the blocks for the requested cluster dimensions.

    Parameters
    ----------
    blocks : sequence of (str, array)
        The name and the codes of each block on the rows of the request, in the
        retained order; at least one block, all on the same rows.
    clusters : sequence of array, optional
        The codes of each requested dimension on the same rows, in the order of the
        request, repetitions kept; empty for a robust request.

    Returns
    -------
    AbsorbedCount
        The per-block counts, the pairwise comparisons, ``df_a`` and the adjustment.

    Raises
    ------
    ValueError
        No block, codes that are not aligned nonnegative integer vectors, or a block
        name repeated.
    """
    if not blocks:
        raise ValueError("provide at least one block")
    names = [name for name, _ in blocks]
    if len(set(names)) != len(names):
        raise ValueError("FE names must be unique")
    rows = int(np.asarray(blocks[0][1]).size)
    codes = [_codes(values, name, rows) for name, values in blocks]
    dimensions = [_codes(cluster, "cluster codes", rows) for cluster in clusters]
    plain = all(type(values) is np.ndarray for values in (*codes, *dimensions))
    numbered = [_numbered(values) for values in codes] if plain else None
    witnesses = (_general_nesting(codes, dimensions) if numbered is None
                 else _nesting(numbered, dimensions))
    def block(j: int) -> tuple[IntegerArray, int]:
        return _numbered(codes[j]) if numbered is None else numbered[j]

    def pair(i: int, j: int) -> int:
        if numbered is None:
            return pairwise_components(codes[i], codes[j])
        earlier, size = numbered[i]
        values, levels = numbered[j]
        return _pairwise_numbered(earlier, size, values, levels)

    return _count_fold(names, witnesses, block, pair)


def _count_fold(
    names: Sequence[str], witnesses: Sequence[int | None],
    block: Callable[[int], tuple[IntegerArray, int]],
    pair: Callable[[int, int], int],
) -> AbsorbedCount:
    """One ordered convention fold; rank and request nesting remain distinct."""
    any_nested = any(witness is not None for witness in witnesses)
    counts: list[BlockCount] = []
    pairwise: list[tuple[int, int, int]] = []
    survivors: list[int] = []
    for j, name in enumerate(names):
        _, levels = block(j)
        witness = witnesses[j]
        if witness is not None:
            counts.append(BlockCount(name, levels, levels, witness, True))
            continue
        if not survivors:
            redundant = 1 if any_nested else 0
        else:
            found = [1]
            for i in survivors:
                shared = pair(i, j)
                pairwise.append((i, j, shared))
                found.append(shared)
            redundant = max(found)
        exact = len(survivors) < 2
        counts.append(BlockCount(name, levels, redundant, None, exact))
        survivors.append(j)
    absorbed = sum(count.counted for count in counts if count.nested_in is None)
    return AbsorbedCount(tuple(counts), tuple(pairwise), int(absorbed), int(any_nested))


def _prepared_count(
    blocks: Sequence[tuple[str, IntegerArray]], levels: tuple[int, ...],
    joint: int, pairs: tuple[tuple[int, int, int], ...],
    clusters: Sequence[IntegerArray],
) -> AbsorbedCount:
    """Consume only the live owned producer's fresh complete-row graph counts.

    This private data fold is not an admission boundary. The interface issues it
    only immediately after the original owned basis callback; partial populations,
    foreign producers and changed count hooks keep absorbed_count.
    """
    rows = int(blocks[0][1].size)
    dimensions = [_codes(c, "cluster codes", rows) for c in clusters]
    numbered = [(values, size) for (_, values), size in zip(
        blocks, levels, strict=True
    )]
    witnesses = _nesting(numbered, dimensions)
    counts = {(i, j): count for i, j, count in pairs}

    def pair(i: int, j: int) -> int:
        return joint if len(blocks) == 2 else counts[i, j]

    return _count_fold([name for name, _ in blocks], witnesses,
                       numbered.__getitem__, pair)


_COUNT_BLOCK = BlockCount
_COUNT_RECORD = AbsorbedCount
_COUNT_CALLS = (
    absorbed_count, _codes, _numbered, nested_in, pairwise_components,
    _nesting, _general_nesting, _nested_numbered, _pairwise_numbered, _count_fold,
    BlockCount, BlockCount.__init__, BlockCount.counted,
    AbsorbedCount, AbsorbedCount.__init__, _prepared_count,
)


def _prepared_count_original() -> bool:
    """Constant-size callback identity check, never a row validity replay."""
    if BlockCount is not _COUNT_BLOCK or AbsorbedCount is not _COUNT_RECORD:
        return False
    current = (
        absorbed_count, _codes, _numbered, nested_in, pairwise_components,
        _nesting, _general_nesting, _nested_numbered, _pairwise_numbered, _count_fold,
        BlockCount, BlockCount.__init__, BlockCount.counted,
        AbsorbedCount, AbsorbedCount.__init__, _prepared_count,
    )
    return all(a is b for a, b in zip(current, _COUNT_CALLS, strict=True))


__all__ = [
    "AbsorbedCount",
    "BlockCount",
    "absorbed_count",
    "nested_in",
    "pairwise_components",
]

