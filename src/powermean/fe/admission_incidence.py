"""Exact sparse quotient of retained incidence by cancelled indicators.

Let A be the within-group row-difference operator: for each non-head row subtract
its group's first row. ker(A)=col(F_C)=ker(M_C), because every cancelled group is
nonempty. Hence rank(A F_R)=rank(M_C F_R)=rank([F_C,F_R])-G_C, by exact row
operations on the joint incidence. This identity concerns exact FE structure;
it says nothing about floating projection error or operational slope rank.

Only differences of sparse retained code rows enter certificate_basis._reduced,
the accepted exact integer elimination. No cancelled columns enter its echelon.
Its canonical rational original-column witnesses are reused and checked against
EVERY difference row. Group heads require no difference; no sample row is dropped.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator
from itertools import pairwise
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from powermean.fe._cancel_storage import IndexArray
from powermean.fe.admission_records import WorkLog
from powermean.fe.cancel import CancelStructure
from powermean.fe.certificate_basis import (
    Row,
    Witness,
    _canonical,
    _column_witnesses,
    _primitive,
    _reduced,
)

# One block part of a difference row, as its (column, value) items.
Part = tuple[tuple[int, int], ...]
BoolArray = NDArray[np.bool_]

# SciPy publishes no type information; its sparse modules are reached untyped.
_sparse: Any = importlib.import_module("scipy.sparse")
_csgraph: Any = importlib.import_module("scipy.sparse.csgraph")


class QuotientRank(NamedTuple):
    """Exact rank, original-level span witnesses and reproducible integer rows."""

    rank: int
    block_ranks: tuple[int, ...]
    annihilated_levels: tuple[tuple[bool, ...], ...]
    kept: tuple[int, ...]
    witnesses: tuple[Witness, ...]
    echelon: tuple[tuple[tuple[int, int], ...], ...]


def _differences(structure: CancelStructure) -> Iterator[Row]:
    codes = structure.retained_sorted_codes
    offsets: list[int] = []
    offset = 0
    for size in structure.block_sizes:
        offsets.append(offset)
        offset += size
    segments = structure.segments
    for start, stop in pairwise(segments):
        for row in range(int(start) + 1, int(stop)):
            found: Row = {}
            for block, first in zip(codes, offsets, strict=True):
                here, head = first + int(block[row]), first + int(block[start])
                if here != head:
                    found[here], found[head] = 1, -1
            yield found


# The largest magnitude of an int64 product sum taken by the array verification.
_INT64_ROOM = 2**62


def _difference_codes(
    structure: CancelStructure,
) -> tuple[int, list[tuple[IndexArray, IndexArray]]]:
    """The number of difference rows and, per block, their columns ``(here, head)``.

    The rows are those of :func:`_differences`, in its order: every row of a group
    after its first, against the group's first row. A block's part of the row is
    ``e_here - e_head``, or nothing where the two are equal.
    """
    segments = structure.segments
    counts = np.diff(segments)
    rows = np.arange(int(segments[-1]), dtype=np.intp)
    heads = np.repeat(segments[:-1], counts)
    rows, heads = rows[rows != heads], heads[rows != heads]
    found = []
    offset = 0
    for codes, size in zip(
        structure.retained_sorted_codes, structure.block_sizes, strict=True
    ):
        found.append(
            (
                (offset + codes[rows]).astype(np.intp),
                (offset + codes[heads]).astype(np.intp),
            )
        )
        offset += size
    return int(rows.size), found


def _forest(left: IndexArray, right: IndexArray, vertices: int) -> BoolArray:
    """Which edges, taken in order, join two parts not yet joined by earlier ones.

    This is the union-find of :func:`_joined_keys` over the whole sequence: edge
    ``r`` is new exactly when no path of edges before it joins its ends. With the
    weight ``r + 1`` the weights are distinct, so the minimum spanning forest is
    unique and is what Kruskal's rule builds, edge by edge in that order, keeping
    an edge exactly when its ends are in different components: those edges. A
    repeated edge is never new after its first occurrence, which alone is kept in
    the graph.
    """
    new = np.zeros(left.size, dtype=bool)
    if not left.size:
        return new
    low, high = np.minimum(left, right), np.maximum(left, right)
    _, first = np.unique(low.astype(np.int64) * vertices + high, return_index=True)
    weights = (first + 1).astype(np.float64)
    rows, cols = low[first].astype(np.int32), high[first].astype(np.int32)
    graph = _sparse.coo_array(
        (weights, (rows, cols)), shape=(vertices, vertices)
    ).tocsr()
    kept = _csgraph.minimum_spanning_tree(graph).tocoo()
    new[kept.data.astype(np.intp) - 1] = True
    return new


def _spanned_rows(
    codes: list[tuple[IndexArray, IndexArray]],
    present: list[BoolArray],
    m: int,
    columns: int,
) -> BoolArray:
    """Rows that :func:`_linked` (two blocks) or :func:`_joined` (one) declare spanned.

    Two blocks: a row with both parts is spanned when its parts are already joined;
    a part is keyed by its ``(here, head)`` columns, distinct across blocks. One
    block: a row with a part is spanned when its two levels are already joined. Any
    other row, and every row with another number of blocks, is not declared spanned.
    """
    spanned = np.zeros(m, dtype=bool)
    if len(codes) == 2:
        both = np.flatnonzero(present[0] & present[1])
        if both.size:
            keys = [
                here[both].astype(np.int64) * columns + head[both]
                for here, head in codes
            ]
            _, ids = np.unique(np.concatenate(keys), return_inverse=True)
            ids = ids.astype(np.intp)
            left, right = ids[: both.size], ids[both.size :]
            spanned[both] = ~_forest(left, right, int(ids.max()) + 1)
    elif len(codes) == 1:
        here, head = codes[0]
        rows = np.flatnonzero(present[0])
        spanned[rows] = ~_forest(here[rows], head[rows], columns)
    return spanned


class _Charges:
    """Work counts kept locally and charged to a log at once, with the same totals.

    Every operation is begun before it runs and completed after, as the log's own
    calls would be; a log only sums them, so charging the totals once, also after a
    failure, leaves the same counts. Only names that were begun are charged.
    """

    def __init__(self) -> None:
        self.begun: dict[str, int] = {}
        self.completed: dict[str, int] = {}

    def begin(self, name: str, units: int = 1) -> None:
        """Count units begun."""
        self.begun[name] = self.begun.get(name, 0) + units

    def complete(self, name: str, units: int = 1) -> None:
        """Count units completed."""
        self.completed[name] = self.completed.get(name, 0) + units

    def charge(self, work: WorkLog) -> None:
        """Add the totals to ``work``."""
        for name, units in self.begun.items():
            work.begin(name, units)
        for name, units in self.completed.items():
            work.complete(name, units)


def _root[K](links: dict[K, K], part: K) -> K:
    """The representative of ``part``'s component, compressing its path."""
    root = part
    while (up := links.get(root, root)) != root:
        root = up
    while part != root:
        links[part], part = root, links[part]
    return root


def _linked(links: dict[Part, Part] | None, parts: list[Row]) -> bool:
    """Whether a row with two block parts ``u + v`` lies in the span of earlier rows.

    With exactly two retained blocks, every earlier row whose two parts are
    nonzero joins its block-0 part to its block-1 part in ``links``. If ``u`` and
    ``v`` are already joined, a path ``u = w_0, w_1, ..., w_m = v`` of earlier rows
    alternates between the blocks, so ``m`` is odd and ``u + v`` is the
    alternating sum ``(w_0 + w_1) - (w_1 + w_2) + ... + (w_{m-1} + w_m)``. A row
    of any other shape is never declared spanned. The row then joins ``u`` and
    ``v``; ``links`` is None with any other number of blocks.
    """
    if links is None or not all(parts):
        return False
    return _joined_keys(links, *(tuple(part.items()) for part in parts))


def _joined_keys[K](links: dict[K, K], first: K, second: K) -> bool:
    """Whether ``first`` and ``second`` are joined in ``links``; join them if not."""
    first, second = _root(links, first), _root(links, second)
    if first == second:
        return True
    links[first] = second
    return False


def _joined(levels: dict[int, int] | None, row: Row) -> bool:
    """Whether a one-block row ``e_a - e_b`` lies in the span of earlier rows.

    With exactly one retained block, every nonempty difference row is ``e_a - e_b``
    for two levels ``a`` and ``b``, and every earlier one joins its two levels in
    ``levels``. If ``a`` and ``b`` are already joined, a path ``a = l_0, l_1, ...,
    l_m = b`` of earlier rows gives ``e_a - e_b`` as the sum of the
    ``e_{l_i} - e_{l_{i+1}}``, each an earlier row up to sign. A row of any other
    shape is never declared spanned. The row then joins ``a`` and ``b``; ``levels``
    is None with any other number of blocks.
    """
    if levels is None or len(row) != 2 or sorted(row.values()) != [-1, 1]:
        return False
    return _joined_keys(levels, *row)


def _insert(
    row: Row, pivots: dict[int, Row], work: WorkLog, *, spanned: bool = False
) -> None:
    """Reduce ``row`` by ``pivots`` and keep a nonzero remainder as a new pivot.

    A ``spanned`` row is known to lie in the span of the rows already reduced,
    which is the span of ``pivots``; ``_reduced`` takes such a row to the empty
    row, so that reduction is counted as before and its empty result is used.
    """
    charges = _Charges()
    try:
        _kept_insert(row, pivots, charges, spanned=spanned)
    finally:
        charges.charge(work)


def _kept_insert(
    row: Row,
    pivots: dict[int, Row],
    charges: _Charges,
    *,
    spanned: bool,
    reducer: _Reducer | None = None,
) -> None:
    """:func:`_insert`'s reduction, charging its counted operations to ``charges``.

    With a ``reducer`` of ``pivots`` the remainder is its :meth:`_Reducer.reduced`,
    which is proportional to that of ``_reduced``, so the primitive row kept is
    the same.
    """
    charges.begin("exact_reduce")
    if spanned:
        reduced: Row = {}
    elif reducer is None:
        reduced = _reduced(row, pivots)
    else:
        reduced = reducer.reduced(row)
    charges.complete("exact_reduce")
    if reduced:
        charges.begin("exact_normalize")
        normalized = _primitive(dict(reduced))
        charges.complete("exact_normalize")
        pivot = min(normalized)
        pivots[pivot] = normalized
        if reducer is not None:
            reducer.added(pivot)


class _Reducer:
    """Exact reduction of integer rows by a growing echelon, with remembered columns.

    For an echelon whose rows each have the entry one at their pivot, the
    remainder of a row ``v``, the vector of ``v + span(pivots)`` that is zero at
    every pivot column, is unique and integral, and linear in ``v``: it is the sum
    of ``v_c`` times the remainder of ``e_c``. That of a non-pivot column is
    ``e_c``; that of a pivot ``p`` with row ``u`` is the remainder of ``e_p - u``,
    whose columns all exceed ``p``. Each pivot's remainder is remembered with the
    number of pivots it was formed against; when more have been added since, the
    columns of it that have become pivots are replaced by their own remainders.
    ``_reduced`` takes the same row along the pivots one at a time and divides by
    common factors on the way, so its remainder is this one times a nonzero
    rational, and :func:`_primitive` makes both the same row. After a pivot whose
    entry is not one, every reduction is ``_reduced``'s.
    """

    def __init__(self, pivots: dict[int, Row]) -> None:
        self.pivots = pivots
        self.unit = all(row[p] == 1 for p, row in pivots.items())
        self.columns: dict[int, tuple[int, Row]] = {}

    def added(self, pivot: int) -> None:
        """Record a new pivot row of the echelon."""
        if self.pivots[pivot][pivot] != 1:
            self.unit = False

    def reduced(self, row: Row) -> Row:
        """The remainder of ``row``, zero at every pivot column."""
        if not self.unit:
            return _reduced(row, self.pivots)
        pivots = self.pivots
        out: Row = {}
        for c, value in row.items():
            if c in pivots:
                for j, w in self._column(c).items():
                    out[j] = out.get(j, 0) + value * w
            else:
                out[c] = out.get(c, 0) + value
        return {c: value for c, value in out.items() if value}

    def _column(self, pivot: int) -> Row:
        """The remainder of ``e_pivot`` against the current pivots."""
        pivots, columns, now = self.pivots, self.columns, len(self.pivots)
        stack = [pivot]
        while stack:
            c = stack[-1]
            held = columns.get(c)
            if held is not None and held[0] == now:
                stack.pop()
                continue
            base = (
                {j: -w for j, w in pivots[c].items() if j != c}
                if held is None
                else held[1]
            )
            waiting = [
                j
                for j in base
                if j in pivots and (j not in columns or columns[j][0] != now)
            ]
            if waiting:
                stack.extend(waiting)
                continue
            columns[c] = (now, self._composed(base))
            stack.pop()
        return columns[pivot][1]

    def _composed(self, base: Row) -> Row:
        """``base`` with each pivot column replaced by its current remainder.

        Every pivot column of ``base`` already holds its remainder against the
        current pivots in ``self.columns``; the other columns are kept.
        """
        pivots, columns = self.pivots, self.columns
        out: Row = {}
        for j, w in base.items():
            if j in pivots:
                for i, x in columns[j][1].items():
                    out[i] = out.get(i, 0) + w * x
            else:
                out[j] = out.get(j, 0) + w
        return {i: x for i, x in out.items() if x}


def _skip(counts: list[int], rows: int, entries: int, blocks: int) -> None:
    """Count ``rows`` completed rows that no reduction reaches, as the loop would."""
    counts[0] += rows
    counts[1] += rows
    counts[2] += entries
    counts[3] += entries
    counts[4] += rows * (1 + blocks)


def _row_plan(
    codes: list[tuple[IndexArray, IndexArray]], m: int, width: int
) -> tuple[NDArray[np.int64], BoolArray, list[BoolArray]]:
    """Each row's entries, whether it is declared spanned, and its block inserts.

    A part enters its block's echelon at its first row, unless, with one block,
    that row is spanned, where the part is the row and is not inserted.
    """
    present = [here != head for here, head in codes]
    entries = (
        2 * np.sum(present, axis=0, dtype=np.int64)
        if codes
        else np.zeros(m, dtype=np.int64)
    )
    spanned = _spanned_rows(codes, present, m, width)
    inserts = []
    for (here, head), mine in zip(codes, present, strict=True):
        first = np.zeros(m, dtype=bool)
        rows = np.flatnonzero(mine)
        keys = here[rows].astype(np.int64) * width + head[rows]
        first[rows[np.unique(keys, return_index=True)[1]]] = True
        inserts.append(first & ~spanned if len(codes) == 1 else first)
    return entries, spanned, inserts


def _eliminate(
    structure: CancelStructure,
    pivots: dict[int, Row],
    per_block: list[dict[int, Row]],
    work: WorkLog,
) -> None:
    """The elimination loop of :func:`quotient_rank`, on integer codes.

    The same rows reach :func:`_reduced` in the same order, with the same spans
    declared by :func:`_linked` (two blocks) and :func:`_joined` (one block), and
    the same work is counted; a part is keyed by its ``(here, head)`` columns in
    place of its items, which identify it the same way. Which rows are declared
    spanned, and where a part first occurs, is found for all rows at once
    (:func:`_spanned_rows`); only the rows that reach a reduction are visited,
    and the rows between are counted together, as completed rows with their
    entries and their spanned reductions, which is what the loop counts for them.
    """
    blocks = len(per_block)
    charges = _Charges()
    m, codes = _difference_codes(structure)
    entries, spanned, inserts = _row_plan(codes, m, sum(structure.block_sizes))
    reach = ~spanned
    for insert in inserts:
        reach |= insert
    visited = np.flatnonzero(reach)
    # Before each visited row, the rows skipped since the last one are charged at
    # once: each completes, with its entries and 1 + blocks spanned reductions.
    totals = np.concatenate([[0], np.cumsum(entries)]).tolist()
    parts = [
        (here[visited].tolist(), head[visited].tolist(), insert[visited].tolist())
        for (here, head), insert in zip(codes, inserts, strict=True)
    ]
    declared = spanned[visited].tolist()
    main = _Reducer(pivots)
    reducers = [_Reducer(echelon) for echelon in per_block]
    # Rows begun and completed, entries begun and completed, spanned reductions.
    counts = [0, 0, 0, 0, 0]
    done = 0
    try:
        for at, r in enumerate(visited.tolist()):
            _skip(counts, r - done, totals[r] - totals[done], blocks)
            row_entries = totals[r + 1] - totals[r]
            counts[0] += 1
            counts[2] += row_entries
            if declared[at]:
                counts[4] += 1
            else:
                row: Row = {}
                for here, head, _ in parts:
                    if here[at] != head[at]:
                        row[here[at]], row[head[at]] = 1, -1
                _kept_insert(row, pivots, charges, spanned=False, reducer=main)
            for (here, head, insert), echelon, own in zip(
                parts, per_block, reducers, strict=True
            ):
                if insert[at]:
                    _kept_insert(
                        {here[at]: 1, head[at]: -1},
                        echelon,
                        charges,
                        spanned=False,
                        reducer=own,
                    )
                else:
                    counts[4] += 1
            counts[3] += row_entries
            counts[1] += 1
            done = r + 1
        _skip(counts, m - done, totals[m] - totals[done], blocks)
    finally:
        if counts[0]:
            charges.begin("exact_difference_row", counts[0])
            charges.complete("exact_difference_row", counts[1])
            charges.begin("exact_input_entry", counts[2])
            charges.complete("exact_input_entry", counts[3])
            charges.begin("exact_reduce", counts[4])
            charges.complete("exact_reduce", counts[4])
        charges.charge(work)


def _charge_witness(
    work: WorkLog, rows: int, terms: int, *, completed: int | None = None
) -> None:
    """Charge witness rows and any nonzero witness terms to ``work``."""
    work.begin("exact_witness_row", rows)
    if terms:
        work.begin("exact_witness_term", terms)
        work.complete("exact_witness_term", terms)
    work.complete("exact_witness_row", rows if completed is None else completed)


def _verified_arrays(
    structure: CancelStructure,
    inverse: dict[int, list[tuple[int, int]]],
    scale: int,
    work: WorkLog,
) -> bool:
    """:func:`_verify`'s loop as one exact integer product, where it fits in int64.

    With ``A`` the difference rows and ``M[c, t]`` the scaled value of column
    ``c`` in the witness of ``t``, each row's sums are its row of ``A M``, so the
    check is ``A M = L A``. Every sum is at most the largest absolute row sum of
    ``M`` times a row's entries, which must stay below ``_INT64_ROOM``; ``False``
    otherwise, and the loop runs. A row that fails is the first failing row of the
    loop, which is then charged and refused as the loop does.
    """
    m, codes = _difference_codes(structure)
    n_columns = sum(structure.block_sizes)
    entries = [(np.asarray(here), np.asarray(head)) for here, head in codes]
    if m == 0:
        return True
    if not entries:
        _charge_witness(work, m, 0)
        return True
    pairs = [(c, t, v) for c, items in inverse.items() for t, v in items]
    largest = max((abs(v) for _, _, v in pairs), default=0)
    if 2 * (len(entries) + 1) * max(largest, scale) >= _INT64_ROOM:
        return False
    rows, cols, vals = [], [], []
    for here, head in entries:
        moved = np.flatnonzero(here != head)
        rows += [moved, moved]
        cols += [here[moved], head[moved]]
        vals += [
            np.ones(moved.size, dtype=np.int64),
            -np.ones(moved.size, dtype=np.int64),
        ]
    a = _sparse.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(m, n_columns),
        dtype=np.int64,
    )
    if pairs:
        c, t, v = (np.array(x, dtype=np.int64) for x in zip(*pairs, strict=True))
    else:
        c = t = v = np.zeros(0, dtype=np.int64)
    w = _sparse.csr_array((v, (c, t)), shape=(n_columns, n_columns), dtype=np.int64)
    per_column = np.bincount(c, minlength=n_columns).astype(np.int64)
    terms = abs(a) @ per_column
    defect = a @ w - scale * a
    defect.eliminate_zeros()
    failing = np.flatnonzero(np.diff(defect.indptr) > 0)
    if failing.size:
        first = int(failing[0])
        _charge_witness(work, first + 1, int(terms[: first + 1].sum()), completed=first)
        raise ValueError("quotient witnesses do not reproduce original levels")
    _charge_witness(work, m, int(terms.sum()))
    return True


def _row_sums(
    row: Row, inverse: dict[int, list[tuple[int, int]]]
) -> tuple[dict[int, int], int]:
    """Scaled witness combination and term count for one difference ``row``."""
    sums: dict[int, int] = {}
    terms = 0
    for column, multiple in row.items():
        for target, value in inverse.get(column, ()):
            terms += 1
            sums[target] = sums.get(target, 0) + multiple * value
    return sums, terms


def _verify(
    structure: CancelStructure,
    kept: tuple[int, ...],
    witnesses: tuple[Witness, ...],
    work: WorkLog,
) -> None:
    """Check every difference row against the witnesses, in exact integers.

    Every rational ``n / d`` is ``n (L / d) / L`` for the least common multiple
    ``L`` of the denominators, so each row's sums times ``L`` are integers, and a
    sum equals an entry exactly when its integer equals the entry times ``L``:
    the verdict of the same sums in ``Fraction``. The work counts are unchanged.
    """
    scale = math.lcm(*(d for witness in witnesses for _, _, d in witness))
    inverse: dict[int, list[tuple[int, int]]] = {}
    for column, witness in enumerate(witnesses):
        for position, numerator, denominator in witness:
            inverse.setdefault(kept[position], []).append(
                (column, numerator * (scale // denominator))
            )
    if _verified_arrays(structure, inverse, scale, work):
        return
    for row in _differences(structure):
        sums, terms = _row_sums(row, inverse)
        bad = any(sums.get(c, 0) != v * scale for c, v in row.items()) or any(
            v != row.get(c, 0) * scale for c, v in sums.items()
        )
        _charge_witness(work, 1, terms, completed=0 if bad else 1)
        if bad:
            raise ValueError("quotient witnesses do not reproduce original levels")


def _back_step(
    row: Row,
    pivot: int,
    *,
    place: dict[int, int],
    slot: dict[int, int],
    values: NDArray[np.int64],
    largest: NDArray[np.int64],
) -> bool:
    """One step of :func:`_integer_witnesses`' back substitution, at ``pivot``.

    Forms ``values[place[pivot]]`` from the pivot's ``row`` and the rows of later
    pivots, already formed, and records its largest magnitude in ``largest``.
    ``reach`` bounds every partial sum of every value formed, so no ``int64``
    operation overflows; False, with the step unfinished, where it reaches
    ``_INT64_ROOM``.
    """
    found = values[place[pivot]]
    reach = 0
    for j, value in row.items():
        if j == pivot:
            continue
        size = 1 if j in slot else int(largest[place[j]])
        reach += abs(value) * size
        if reach >= _INT64_ROOM:
            return False
        if j in slot:
            found[slot[j]] += value
        elif size:
            found -= value * values[place[j]]
    largest[place[pivot]] = int(np.abs(found).max(initial=0))
    return True


def _integer_witnesses(
    pivots: dict[int, Row], kept: list[int], count: int
) -> tuple[Witness, ...] | None:
    """The canonical witnesses of every column, where they are integers, or None.

    Column ``c``'s witness ``t`` solves ``R_J t = R_c`` over the echelon rows ``R``:
    ``R_J`` is upper triangular in pivot order, so ``t`` is unique and back
    substitution over ``reversed(kept)`` gives it, as in ``_dependent``. Where every
    pivot entry is one, each step only subtracts integer multiples, so ``t`` is
    integral and is formed here for all dependent columns at once in ``int64``,
    every value bounded below ``_INT64_ROOM`` as it is formed. None where a pivot
    entry is not one or the bound fails: the rational substitution then decides.
    """
    place = {c: i for i, c in enumerate(kept)}
    others = [c for c in range(count) if c not in place]
    if any(pivots[p][p] != 1 for p in kept):
        return None
    slot = {c: i for i, c in enumerate(others)}
    values = np.zeros((len(kept), len(others)), dtype=np.int64)
    largest = np.zeros(len(kept), dtype=np.int64)
    for p in reversed(kept):
        if not _back_step(
            pivots[p], p, place=place, slot=slot, values=values, largest=largest
        ):
            return None
    witnesses: list[Witness] = []
    for c in range(count):
        if c in place:
            witnesses.append(((place[c], 1, 1),))
            continue
        column = values[:, slot[c]]
        witnesses.append(
            tuple((int(i), int(column[i]), 1) for i in np.flatnonzero(column))
        )
    return tuple(witnesses)


def _canonical_witnesses(
    pivots: dict[int, Row], kept: list[int], count: int
) -> tuple[Witness, ...]:
    """Every column's canonical witness: in integers where they are, else rational."""
    found = _integer_witnesses(pivots, kept, count)
    if found is not None:
        return found
    columns = _column_witnesses(pivots, kept, count)
    return tuple(_canonical(columns[c]) for c in range(count))


def quotient_rank(structure: CancelStructure, work: WorkLog) -> QuotientRank:
    """Eliminate exact sparse difference rows and verify every original column."""
    pivots: dict[int, Row] = {}
    per_block: list[dict[int, Row]] = [{} for _ in structure.block_sizes]
    boundaries = [0]
    for size in structure.block_sizes:
        boundaries.append(boundaries[-1] + size)
    # With one block the part is the row and its echelon receives the same rows as
    # ``pivots``, so a row spanned there is spanned here (in :func:`_eliminate`).
    _eliminate(structure, pivots, per_block, work)
    kept = tuple(sorted(pivots))
    witnesses = work.run(
        "exact_column_witnesses",
        lambda: _canonical_witnesses(pivots, list(kept), boundaries[-1]),
    )
    _verify(structure, kept, witnesses, work)
    annihilated = tuple(
        tuple(not witnesses[c] for c in range(start, stop))
        for start, stop in pairwise(boundaries)
    )
    return QuotientRank(
        len(kept),
        tuple(map(len, per_block)),
        annihilated,
        kept,
        witnesses,
        tuple(tuple(sorted(pivots[p].items())) for p in kept),
    )
