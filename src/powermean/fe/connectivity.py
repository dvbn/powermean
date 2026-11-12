"""Connectivity of the fixed-effect blocks: components, indicator rank and its bounds.

Redundancies are exact for one and two blocks (connected components of the bipartite
graph) and bounded pairwise for three or more, as in ``reghdfe``; effects are
normalized by reference levels within connected components.  Connectivity is new code,
not a port.  A disconnected two-block design needs the rank of its connected
components, not ``sum(levels) - 1``.

The graph.  The levels of all blocks are the vertices; each row joins its level in the
first block to its level in every other block, so the levels of one row form a
connected star and the components of this graph are those of the usual bipartite graph
for two blocks and of its generalization for more.  Components are numbered in the order
in which the rows first reach them: a deterministic numbering, invariant to the labels
of the levels and to the names of the blocks.  Nothing is sorted in place, no dense
indicator matrix is formed in production, and no factorization runs; the components are
found by minimum-label propagation over the edges with pointer jumping, on NumPy arrays
(:func:`_vertex_labels`), which terminates since the labels decrease until no edge joins
two labels, and which is exact integer work.

The indicator rank.  Let ``D_g`` be the ``n`` by ``L_g`` matrix of all level indicators
of block ``g``, ``D = [D_1 ... D_G]`` and ``C`` the number of components.

- One block: ``rank D = L_1``, and ``C = L_1``.
- Two blocks: ``rank D = L_1 + L_2 - C`` exactly.  A null vector ``x`` of ``D``
  satisfies ``x[a_i] + x[b_i] = 0`` on every row ``i``, so along any path of a
  component its values alternate between one constant and its negative: the null space
  is spanned by the ``C`` vectors that are ``+1`` on the first-block levels of a
  component and ``-1`` on its second-block levels, which are independent.
- Three or more blocks: connectivity alone does not identify the rank (two blocks that
  encode the same partition are one dependency per level that no graph sees), so only
  proven bounds are reported.  Lower bound: the rank of any pair of blocks,
  ``L_i + L_j - C_ij`` with ``C_ij`` the components of that pair, since a matrix has at
  least the rank of a subset of its columns; the largest over the pairs.  Upper bound:
  ``sum L_g - (G - 1) C``, since for each component ``K`` and each block ``g >= 2``
  the vector ``+1`` on the first-block levels of ``K`` and ``-1`` on the block-``g``
  levels of ``K`` is a null vector of ``D`` (every row of ``K`` meets one level of
  each), and these ``(G - 1) C`` vectors are independent: the vector of ``(K, g)`` is
  the only one of them that is nonzero on the block-``g`` levels of ``K``, so a
  vanishing combination has a zero coefficient on each.  Their supports are not
  disjoint, the vectors of one component sharing its first-block levels; the
  independence rests on the exclusive block-``g`` coordinates alone.
  Where the two bounds coincide the rank is determined; where they do not, ``exact`` is
  ``None`` and no number is invented.

The intercept.  The constant lies in the span of every block (``D_g 1 = 1``), so adding
an explicit intercept column changes no rank: ``rank [1, D] = rank D``.  Counted next to
an explicit intercept, the blocks contribute ``rank D - 1`` centered indicator columns
(the fixed-effect slopes beyond one common level); the two counts are reported apart.
No reference level, omission, QR threshold or normalization is chosen here: the
components say how many normalizations each pair of blocks needs and which levels lie
together, and a later stage decides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from powermean.fe.groups import ALIGNED, Block, _index, _vectors
from powermean.model.order import IndexArray


def _aligned(blocks: Sequence[Block]) -> tuple[Block, ...]:
    """The blocks as a nonempty tuple on the same rows."""
    found = tuple(blocks)
    if not found:
        raise ValueError("provide at least one block")
    if any(block.n_rows != found[0].n_rows for block in found):
        raise ValueError(ALIGNED)
    return found


def _offsets(blocks: Sequence[Block]) -> IndexArray:
    """The vertex offset of each block: the levels of block ``g`` start there."""
    counts = [block.n_levels for block in blocks]
    offsets: IndexArray = np.cumsum([0, *counts[:-1]], dtype=np.intp)
    return offsets


def _vertex_labels(blocks: Sequence[Block]) -> IndexArray:
    """The smallest vertex of the component of each vertex (minimum-label propagation).

    Each round takes, for every vertex, the minimum of its label and of the labels of
    its neighbors, then follows the labels to their roots (pointer jumping).  The labels
    never increase and strictly decrease somewhere until no edge joins two labels, so
    the rounds terminate; at the end two vertices carry the same label exactly when
    a path joins them, since a vertex's label is always a vertex of its own component.
    """
    offsets = _offsets(blocks)
    label: IndexArray = np.arange(int(offsets[-1]) + blocks[-1].n_levels, dtype=np.intp)
    if len(blocks) == 1:
        return label
    u = np.concatenate([blocks[0].codes] * (len(blocks) - 1))
    v = np.concatenate(
        [
            block.codes + offset
            for block, offset in zip(blocks[1:], offsets[1:], strict=True)
        ]
    )
    while True:
        candidate = label.copy()
        np.minimum.at(candidate, u, label[v])
        np.minimum.at(candidate, v, label[u])
        while True:
            jumped: IndexArray = candidate[candidate]
            if np.array_equal(jumped, candidate):
                break
            candidate = jumped
        if np.array_equal(candidate, label):
            return label
        label = candidate


@dataclass(frozen=True, eq=False)
class Components:
    """The connected components of the levels of the blocks.

    Attributes
    ----------
    of_rows : array of int
        The component of each row, numbered ``0 .. n_components - 1`` in the order in
        which the rows first reach each component; read-only, owned.
    of_levels : tuple of array of int
        For each block, the component of each level; read-only, owned.
    row_counts : array of int
        The number of rows of each component; read-only, owned.
    level_counts : tuple of array of int
        For each block, the number of its levels in each component, at least one;
        read-only, owned.

    Notes
    -----
    Constructed by :func:`components`.  A direct construction is checked against
    these statements, whatever sequences and integer arrays are given, and stores
    owned tuples of owned index arrays.
    """

    of_rows: IndexArray
    of_levels: tuple[IndexArray, ...]
    row_counts: IndexArray
    level_counts: tuple[IndexArray, ...]

    def __post_init__(self) -> None:
        """Check the partition and the counts; store owned tuples and arrays."""
        of_rows, *of_levels = _vectors(
            (self.of_rows, *self.of_levels),
            "components must be nonempty integer vectors",
        )
        if not of_levels:
            raise ValueError("components need at least one block")
        found, first = np.unique(of_rows, return_index=True)
        n = found.size
        if found[0] != 0 or found[-1] != n - 1 or not np.all(np.diff(first) > 0):
            raise ValueError("components must be numbered by first occurrence")
        if any(part.min() < 0 or part.max() >= n for part in of_levels):
            raise ValueError("a level's component is outside [0, n_components)")
        counts = [np.bincount(part, minlength=n) for part in (of_rows, *of_levels)]
        if any(count.min() < 1 for count in counts[1:]):
            raise ValueError("every component must hold a level of every block")
        given = [np.asarray(part) for part in (self.row_counts, *self.level_counts)]
        if len(given) != len(counts) or any(
            not np.array_equal(a, b) for a, b in zip(given, counts, strict=True)
        ):
            raise ValueError("counts must be the rows and the levels of each component")
        object.__setattr__(self, "of_rows", _index(of_rows))
        object.__setattr__(self, "of_levels", tuple(map(_index, of_levels)))
        object.__setattr__(self, "row_counts", _index(counts[0]))
        object.__setattr__(self, "level_counts", tuple(map(_index, counts[1:])))

    @property
    def n_components(self) -> int:
        """The number of components."""
        return int(self.row_counts.size)


def components(blocks: Sequence[Block]) -> Components:
    """The connected components of the levels of aligned blocks.

    Parameters
    ----------
    blocks : sequence of Block
        At least one block, all on the same rows.

    Returns
    -------
    Components
        The component of each row and of each level, numbered by first occurrence
        along the rows, with the counts of rows and of levels per component.

    Raises
    ------
    ValueError
        No block, or blocks of unequal length.
    """
    found = _aligned(blocks)
    offsets = _offsets(found)
    label = _vertex_labels(found)
    root_rows = label[found[0].codes]
    roots, first = np.unique(root_rows, return_index=True)
    rank = np.empty(roots.size, dtype=np.intp)
    rank[np.argsort(first, kind="stable")] = np.arange(roots.size)
    of_rows = rank[np.searchsorted(roots, root_rows)]
    of_vertices = rank[np.searchsorted(roots, label)]
    of_levels = tuple(
        of_vertices[offset : offset + block.n_levels]
        for block, offset in zip(found, offsets, strict=True)
    )
    return Components(
        of_rows,
        of_levels,
        np.bincount(of_rows, minlength=roots.size),
        tuple(np.bincount(part, minlength=roots.size) for part in of_levels),
    )


@dataclass(frozen=True)
class IndicatorRank:
    """The rank of the full indicator matrix of the blocks, exact or bounded.

    Attributes
    ----------
    levels : tuple of int
        The number of levels of each block.
    n_components : int
        The components of the graph of all blocks.
    pairwise_components : tuple of (int, int, int)
        For three or more blocks, ``(i, j, C_ij)`` for each pair ``i < j``; empty
        otherwise, the components of one or two blocks being ``n_components``.
    lower, upper : int
        Proven bounds on ``rank [D_1 ... D_G]``; equal for one and two blocks.
    """

    levels: tuple[int, ...]
    n_components: int
    pairwise_components: tuple[tuple[int, int, int], ...]
    lower: int
    upper: int

    @property
    def exact(self) -> int | None:
        """The rank where the bounds coincide; ``None`` where they leave a gap."""
        return self.lower if self.lower == self.upper else None

    @property
    def redundant(self) -> tuple[int, int]:
        """Bounds on the redundant levels ``sum(levels) - rank``: ``(lower, upper)``."""
        total = sum(self.levels)
        return total - self.upper, total - self.lower

    @property
    def with_intercept(self) -> tuple[int, int]:
        """Bounds on ``rank [1, D]``: those of ``rank D``, the constant in its span."""
        return self.lower, self.upper

    @property
    def beyond_intercept(self) -> tuple[int, int]:
        """Bounds on the centered indicator columns next to an explicit intercept.

        ``rank D - 1``: the fixed-effect slopes beyond one common level, which is what
        the blocks add to a design that already has an intercept.
        """
        return self.lower - 1, self.upper - 1


def indicator_rank(blocks: Sequence[Block]) -> IndicatorRank:
    """The rank of the full indicator matrix of aligned blocks, exact or bounded.

    Parameters
    ----------
    blocks : sequence of Block
        At least one block, all on the same rows.

    Returns
    -------
    IndicatorRank
        Exact for one block (``L_1``) and two (``L_1 + L_2 - C``); for three or more,
        the lower bound ``max_{i<j} (L_i + L_j - C_ij)`` and the upper bound
        ``sum L_g - (G - 1) C``, with ``exact`` set only where they coincide.

    Raises
    ------
    ValueError
        No block, or blocks of unequal length.
    """
    found = _aligned(blocks)
    levels = tuple(block.n_levels for block in found)
    joint = components(found).n_components
    return _rank_from_counts(found, levels, joint)


def _rank_from_counts(
    found: tuple[Block, ...], levels: tuple[int, ...], joint: int
) -> IndicatorRank:
    """The original rank arithmetic, with all changing pairwise graphs intact."""
    if len(found) == 1:
        return IndicatorRank(levels, joint, (), levels[0], levels[0])
    if len(found) == 2:
        rank = levels[0] + levels[1] - joint
        return IndicatorRank(levels, joint, (), rank, rank)
    pairwise = tuple(
        (i, j, components((found[i], found[j])).n_components)
        for i in range(len(found))
        for j in range(i + 1, len(found))
    )
    return _rank_from_pairs(levels, joint, pairwise, len(found))


def _rank_from_pairs(
    levels: tuple[int, ...], joint: int,
    pairwise: tuple[tuple[int, int, int], ...], blocks: int,
) -> IndicatorRank:
    """The same bounds arithmetic over the freshly computed pair counts."""
    lower = max(levels[i] + levels[j] - c for i, j, c in pairwise)
    upper = sum(levels) - (blocks - 1) * joint
    return IndicatorRank(levels, joint, pairwise, lower, upper)


def _indicator_rank_prepared(
    blocks: tuple[Block, ...], parts: Components
) -> IndicatorRank:
    """Consume this private producer's already computed joint components."""
    levels = tuple(block.n_levels for block in blocks)
    return _rank_from_counts(blocks, levels, parts.n_components)


def _indicator_rank_counted(
    blocks: tuple[Block, ...], parts: Components
) -> IndicatorRank:
    """Only the live owned-basis producer consumes unexported pair counts here.

    The caller keeps its full joint Components and checks original producers.
    Every declared level is represented in its admitted blocks. Every component
    therefore meets a row and the first block, so distinct terminal vertex roots
    count exactly the components that components() numbers by first row.
    Nothing here replaces its fresh graph propagation or any rank bound.
    """
    levels = tuple(block.n_levels for block in blocks)
    joint = parts.n_components
    if len(blocks) <= 2:
        return _rank_from_counts(blocks, levels, joint)
    pairwise = tuple(
        (i, j, int(np.unique(_vertex_labels((blocks[i], blocks[j]))).size))
        for i in range(len(blocks))
        for j in range(i + 1, len(blocks))
    )
    return _rank_from_pairs(levels, joint, pairwise, len(blocks))
