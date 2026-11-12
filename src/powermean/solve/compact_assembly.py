"""Scatter the existing checked FE Hessian entries into disjoint numeric partitions.

No formula or summation is replaced: the ordinary geometry assembly emits the same
already checked row-order sums. This sink retains D/C/B/G instead of concatenating
full joint row/column/value arrays. Block selection uses the admitted exact basis,
not a balanced-panel formula or a new rank/reference decision.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array

from powermean.fe import absorb_numeric as _numeric
from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.fe.absorb_numeric import group_sums
from powermean.fe.certificate_basis import CertificateBasis
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_pair_layout import native_pair_inputs, native_pair_layout
from powermean.solve.compact_matrix import (
    CompactMatrix,
    CscStorage,
    Entries,
    MatrixParts,
    canonical_partition,
)


def kept_block(basis: CertificateBasis, n_slopes: int) -> tuple[int, int] | None:
    """The existing largest contiguous categorical block, deterministic tie order."""
    selected, count = None, 0
    for position in basis.positions:
        kept = position[position >= 0]
        if kept.size <= count:
            continue
        first, stop = int(kept[0]), int(kept[-1]) + 1
        if np.array_equal(kept, np.arange(first, stop, dtype=np.intp)):
            selected, count = (n_slopes + first, n_slopes + stop), int(kept.size)
    return selected


type _Selection = slice | NDArray[np.bool_] | None


class CompactEntries:
    """An assembly sink owning no full joint matrix or indicator array."""

    def __init__(self, size: int, block: tuple[int, int]) -> None:
        self.size, self.block = size, block
        self.diagonal = np.zeros(block[1] - block[0])
        self.seen = np.zeros(self.diagonal.size, dtype=np.bool_)
        self.left: list[Entries] = []
        self.right: list[Entries] = []
        self.remainder: list[Entries] = []

    def add(self, rows: IndexArray, cols: IndexArray, values: FloatArray) -> None:
        """Scatter each unique already-summed entry, retaining exact zero decisions."""
        self._scatter(rows, cols, values)

    def _scatter(
        self, rows: IndexArray, cols: IndexArray, values: FloatArray,
    ) -> tuple[FloatArray, IndexArray, tuple[_Selection, ...]]:
        """Classify once for both the numeric partitions and optional source slots."""
        rows, cols = np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)
        values = np.asarray(values, dtype=np.float64)
        start, stop = self.block
        rb, cb = (rows >= start) & (rows < stop), (cols >= start) & (cols < stop)
        ra, ca = bool(np.all(rb)), bool(np.all(cb))
        rn, cn = not np.any(rb), not np.any(cb)
        whole = slice(None)
        aligned = (rows.ndim == cols.ndim == values.ndim == 1
                   and rows.shape == cols.shape == values.shape)
        selections: tuple[_Selection, ...]
        if aligned and ra and ca:
            selections = (whole, None, None, None)
        elif aligned and rn and ca:
            selections = (None, whole, None, None)
        elif aligned and ra and cn:
            selections = (None, None, whole, None)
        elif aligned and rn and cn:
            selections = (None, None, None, whole)
        else:
            selections = (rb & cb, ~rb & cb, rb & ~cb, ~rb & ~cb)
        diagonal = selections[0]
        positions = np.zeros(0, dtype=np.intp)
        if diagonal is not None:
            if np.any(rows[diagonal] != cols[diagonal]):
                raise ValueError("the categorical principal block is not diagonal")
            positions = rows[diagonal] - start
            if np.any(self.seen[positions]):
                raise ValueError("an ordered entry of the Hessian was summed twice")
            self.diagonal[positions], self.seen[positions] = values[diagonal], True
        count = stop - start
        for selected, target, row_inside, col_inside in zip(
            selections[1:], (self.left, self.right, self.remainder),
            (False, True, False), (True, False, False), strict=True,
        ):
            if selected is not None:
                rr, cc = rows[selected], cols[selected]
                rr = rr - start if row_inside else rr - np.where(rr >= stop, count, 0)
                cc = cc - start if col_inside else cc - np.where(cc >= stop, count, 0)
                target.append((rr, cc,
                               values.copy() if isinstance(selected, slice)
                               else values[selected]))
        return values, positions, selections

    def _matrix(self, batches: list[Entries], shape: tuple[int, int]) -> CscStorage:
        """Canonicalize one rectangular partition, never the full joint shape."""
        rows = np.concatenate([np.zeros(0, dtype=np.intp), *(e[0] for e in batches)])
        cols = np.concatenate([np.zeros(0, dtype=np.intp), *(e[1] for e in batches)])
        values = np.concatenate([np.zeros(0), *(e[2] for e in batches)])
        return canonical_partition((rows, cols, values), shape)

    def finish(self) -> CompactMatrix:
        """Own the exact disjoint partition bytes and release construction batches."""
        start, stop = self.block
        m, r = stop - start, self.size - (stop - start)
        remaining = np.concatenate((np.arange(start), np.arange(stop, self.size))).astype(np.intp)
        parts = MatrixParts(
            self.block, IndexStorage.store(remaining), FloatStorage.store(self.diagonal),
            IndexStorage.store(np.flatnonzero(self.diagonal != 0).astype(np.intp)),
            self._matrix(self.left, (r, m)), self._matrix(self.right, (m, r)),
            self._matrix(self.remainder, (r, r)),
        )
        self.left.clear()
        self.right.clear()
        self.remainder.clear()
        return CompactMatrix(parts)


class JointLayout:
    """Immutable pair incidence of an admitted basis, reused only inside one fit.

    Weight-dependent sums retain the original row order and checked reducer.
    No incidence matrix or numerical verdict is stored. A temporary combined key
    is used only when its native-integer range is proved before multiplication.
    """

    def __init__(self, basis: CertificateBasis, n_slopes: int) -> None:
        self.basis, self.n_slopes = basis, n_slopes
        self.block = kept_block(basis, n_slopes)
        self._structure: _JointStructure | None = None
        self.pairs: list[tuple[IndexStorage, IndexStorage, IndexStorage, int]] = []
        blocks = basis.grouping.blocks
        for g, first in enumerate(blocks):
            for h in range(g + 1, len(blocks)):
                layout = None
                if (native_pair_layout is _PAIR_LAYOUT
                        and native_pair_inputs is _PAIR_INPUTS
                        and _numeric._input_original is _PAIR_ORIGINAL
                        and _PAIR_ORIGINAL()
                        and native_pair_inputs(blocks, first, h)):
                    layout = _combined_pair_layout(first.codes, blocks[h].codes)
                native = layout is not None
                if layout is None:
                    keys = np.column_stack((first.codes, blocks[h].codes))
                    layout = native_pair_layout(keys, first.codes, blocks[h].codes)
                    layout = (np.unique(keys, axis=0, return_inverse=True)
                              if layout is None else layout)
                pairs, inverse = layout
                rows = basis.positions[g][pairs[:, 0]]
                cols = basis.positions[h][pairs[:, 1]]
                inverse = inverse.reshape(-1).astype(np.intp)
                # Only adjacent equal incidences: at most one linear comparison
                # per pair, never a quadratic search over all block-pair layouts.
                previous = self.pairs[-1] if self.pairs else None
                shared = (previous[0] if native and previous is not None
                          and previous[3] == len(pairs)
                          and np.array_equal(inverse, previous[0].view()) else
                          IndexStorage.store(inverse))
                self.pairs.append((
                    shared, IndexStorage.store(rows), IndexStorage.store(cols), len(pairs),
                ))

    def add_pairs(
        self, curvature: FloatArray, entries: CompactEntries, *, shared: bool = False,
    ) -> None:
        """Fresh checked pair sums; only the owned consumer shares equal incidence."""
        batches = (self._pair_values(curvature, shared=True) if shared
                   else self._pair_values(curvature))
        for rows, cols, sums in batches:
            kept = (rows >= 0) & (cols >= 0)
            rr, cc = self.n_slopes + rows[kept], self.n_slopes + cols[kept]
            entries.add(rr, cc, sums[kept])
            entries.add(cc, rr, sums[kept])

    def _pair_values(
        self, curvature: FloatArray, *, shared: bool = False,
    ) -> Iterator[tuple[IndexArray, IndexArray, FloatArray]]:
        """Reuse only a same-call sum of the exact same immutable incidence."""
        previous = None
        sums: FloatArray
        for inverse, left, right, count in self.pairs:
            if not shared or inverse is not previous:
                sums = group_sums(curvature, inverse.view(), count)
            previous = inverse
            yield left.view(), right.view(), sums


def _combined_pair_layout(
    left: IndexArray, right: IndexArray,
) -> tuple[IndexArray, IndexArray] | None:
    """Canonical pair tuples from a temporary injective key, or the old fallback."""
    if not (type(left) is np.ndarray and type(right) is np.ndarray
            and left.dtype == right.dtype == np.dtype(np.intp)
            and left.ndim == right.ndim == 1 and left.shape == right.shape
            and left.size and left.flags.c_contiguous and right.flags.c_contiguous):
        return None
    if int(left.min()) < 0 or int(right.min()) < 0:
        return None
    largest, width = int(left.max()), int(right.max()) + 1
    limit = np.iinfo(np.intp).max
    if width > limit or largest > (limit - (width - 1)) // width:
        return None
    keys, inverse = np.unique(left * width + right, return_inverse=True)
    return np.column_stack((keys // width, keys % width)), inverse.reshape(-1)


_PAIR_LAYOUT = native_pair_layout
_PAIR_INPUTS = native_pair_inputs
_PAIR_ORIGINAL = _numeric._input_original


@dataclass(frozen=True, eq=False)
class _CscRoute:
    """Potential canonical slots, without numeric values or caller headers."""

    shape: tuple[int, int]
    sources: IndexStorage
    pointers: IndexStorage
    rows: IndexStorage

    @classmethod
    def compile(
        cls, batches: list[Entries], sources: list[IndexArray],
        shape: tuple[int, int],
    ) -> _CscRoute | None:
        """Sort potential slots once; unknown duplicates keep generic zero rules."""
        rows = np.concatenate([np.zeros(0, dtype=np.intp), *(b[0] for b in batches)])
        cols = np.concatenate([np.zeros(0, dtype=np.intp), *(b[1] for b in batches)])
        slots = np.concatenate([np.zeros(0, dtype=np.intp), *sources])
        native = (
            all(type(a) is np.ndarray and a.dtype == np.dtype(np.intp)
                and a.ndim == 1 for a in (rows, cols, slots))
            and rows.shape == cols.shape == slots.shape and rows.size > 0
            and type(shape) is tuple and len(shape) == 2
            and all(type(d) is int for d in shape)
            and 0 < min(shape) <= max(shape) <= np.iinfo(np.int32).max
            and int(cols.min()) >= 0 and int(cols.max()) < shape[1]
        )
        if native:
            pointers = np.concatenate(([0], np.cumsum(
                np.bincount(cols, minlength=shape[1]),
            )))
            # Compressed construction/sort never coalesces duplicate entries or
            # interprets source slots as numerical Hessian values.
            order = np.argsort(cols, kind="stable")
            matrix = csc_array((slots[order], rows[order], pointers), shape=shape)
            matrix.sort_indices()
            rows, slots = matrix.indices, matrix.data
            cols = cols[order]
        else:
            order = np.lexsort((rows, cols))
            rows, cols, slots = rows[order], cols[order], slots[order]
        if np.any((np.diff(rows) == 0) & (np.diff(cols) == 0)):
            return None
        if not native:
            pointers = np.concatenate(([0], np.cumsum(
                np.bincount(cols, minlength=shape[1]),
            )))
        return cls(shape, IndexStorage.store(np.asarray(slots, dtype=np.intp)),
                   IndexStorage.store(np.asarray(pointers, dtype=np.intp)),
                   IndexStorage.store(np.asarray(rows, dtype=np.intp)))

    def pack(self, values: FloatArray) -> CscStorage:
        """Fresh RAW-zero selection; retain full structure only for all live slots."""
        selected = values[self.sources.view()]
        keep = selected != 0
        if np.all(keep):
            return CscStorage(self.shape, self.pointers, self.rows,
                              FloatStorage.store(selected))
        counts = np.concatenate(([0], np.cumsum(keep)))
        pointers = np.asarray(counts[self.pointers.view()], dtype=np.intp)
        return CscStorage(self.shape, IndexStorage.store(pointers),
                          IndexStorage.store(self.rows.view()[keep]),
                          FloatStorage.store(selected[keep]))


@dataclass(frozen=True, eq=False)
class _JointStructure:
    """One operation's immutable route, never a weighted eligibility verdict."""

    block: tuple[int, int]
    remaining: IndexStorage
    diagonal: IndexStorage
    partitions: tuple[_CscRoute, _CscRoute, _CscRoute]
    mixed: tuple[tuple[IndexStorage, IndexStorage], ...]
    pairs: tuple[IndexStorage, ...]
    value_count: int

    def pack(self, values: FloatArray) -> CompactMatrix:
        """Form fresh owned numeric partitions in the original FULL coordinates."""
        diagonal = values[self.diagonal.view()]
        left, right, remainder = (route.pack(values) for route in self.partitions)
        return CompactMatrix(MatrixParts(
            self.block, self.remaining, FloatStorage.store(diagonal),
            IndexStorage.store(np.flatnonzero(diagonal != 0).astype(np.intp)),
            left, right, remainder,
        ))


class _ValueEntries:
    """Fresh value workspace in the qualified emitter's original batch order."""

    def __init__(self, structure: _JointStructure) -> None:
        """Allocate only changing values; no coordinate arrays or sorting."""
        self.values = np.empty(structure.value_count)
        self.offset = 0

    def add(self, values: FloatArray) -> None:
        """Copy one fresh checked batch; never retain a changing reducer buffer."""
        stop = self.offset + values.size
        self.values[self.offset:stop] = values
        self.offset = stop

    def finish(self, structure: _JointStructure) -> CompactMatrix:
        """Only a complete original sequence supplies the current numeric matrix."""
        assert self.offset == self.values.size
        return structure.pack(self.values)


class _PreparingEntries(CompactEntries):
    """First original assembly, recording potential routes including current zeros."""

    def __init__(self, size: int, block: tuple[int, int]) -> None:
        """Keep original scatter/error order while recording one-time slot sources."""
        super().__init__(size, block)
        self.value_batches: list[FloatArray] = []
        self.sources: tuple[list[IndexArray], ...] = ([], [], [])
        self.diagonal_sources: IndexArray | None = np.full(
            self.diagonal.size, -1, dtype=np.intp,
        )
        self.offset = 0

    def add(self, rows: IndexArray, cols: IndexArray, values: FloatArray) -> None:
        """Perform the original checked scatter before recording structural facts."""
        values, positions, selections = self._scatter(rows, cols, values)
        if not self.sources:
            return
        try:
            self._record(values, positions, selections)
        except MemoryError:
            self._discard()

    def _discard(self) -> None:
        """Release every optional recording owner, keeping baseline batches."""
        self.sources = ()
        self.value_batches.clear()
        self.diagonal_sources = None

    def _record(
        self, values: FloatArray, positions: IndexArray,
        selections: tuple[_Selection, ...],
    ) -> None:
        """Route the same batch selections, never classify or gather coordinates again."""
        slots = self.offset + np.arange(values.size, dtype=np.intp)
        self.offset += values.size
        self.value_batches.append(values)
        diagonal_sources = cast("IndexArray", self.diagonal_sources)
        if selections[0] is not None:
            diagonal_sources[positions] = slots[selections[0]]
        for selected, target in zip(selections[1:], self.sources, strict=True):
            if selected is not None:
                target.append(slots[selected])

    def prepare(
        self, layout: JointLayout,
    ) -> tuple[CompactMatrix, _JointStructure | None]:
        """Build actual potential structure after the original fresh reductions."""
        prepared = None
        if self.sources and np.all(self.seen):
            try:
                prepared = self._prepare_route(layout)
            except MemoryError:
                prepared = None
        if prepared is not None:
            return prepared
        self._discard()
        return self.finish(), None

    def _prepare_route(
        self, layout: JointLayout,
    ) -> tuple[CompactMatrix, _JointStructure] | None:
        """Prepare only optional structure; allocation refusal keeps original data."""
        start, stop = self.block
        m, r = stop - start, self.size - (stop - start)
        partitions = tuple(
            _CscRoute.compile(batches, sources, shape)
            for batches, sources, shape in zip(
                (self.left, self.right, self.remainder), self.sources,
                ((r, m), (m, r), (r, r)), strict=True,
            )
        )
        if any(part is None for part in partitions):
            return None
        mixed: list[tuple[IndexStorage, IndexStorage]] = []
        for position in layout.basis.positions:
            kept = np.flatnonzero(position >= 0)
            mixed.append((IndexStorage.store(kept),
                          IndexStorage.store(layout.n_slopes + position[kept])))
        pairs = tuple(IndexStorage.store(np.flatnonzero(
            (left.view() >= 0) & (right.view() >= 0),
        )) for _, left, right, _ in layout.pairs)
        remaining = np.concatenate((np.arange(start), np.arange(stop, self.size)))
        structure = _JointStructure(
            self.block, IndexStorage.store(remaining.astype(np.intp)),
            IndexStorage.store(cast("IndexArray", self.diagonal_sources)),
            cast("tuple[_CscRoute, _CscRoute, _CscRoute]", partitions),
            tuple(mixed), pairs, self.offset,
        )
        values = np.concatenate([np.zeros(0), *self.value_batches])
        return structure.pack(values), structure


def original_entries(layout: JointLayout) -> bool:
    """Changed sink/pair callbacks keep their original dynamic assembly contract."""
    return (type(layout) is JointLayout
            and all(globals()[name] is value for name, value in _ENTRY_PRODUCERS)
            and all(getattr(cls, name) is value for cls, name, value in _ENTRY_METHODS)
            and _CscRoute.__dict__["compile"] is _ROUTE_COMPILE
            and not any(name in layout.__dict__
                        for name in ("add_pairs", "_pair_values")))


_ENTRY_PRODUCERS = tuple((name, globals()[name]) for name in (
    "CompactEntries", "JointLayout", "canonical_partition", "kept_block",
    "group_sums", "native_pair_layout", "native_pair_inputs", "_combined_pair_layout",
    "csc_array", "_CscRoute", "_JointStructure",
    "_ValueEntries", "_PreparingEntries",
))
_ENTRY_METHODS = tuple((cls, name, getattr(cls, name)) for cls, names in (
    (CompactEntries, ("__init__", "add", "_scatter", "_matrix", "finish")),
    (JointLayout, ("__init__", "add_pairs", "_pair_values")),
    (_CscRoute, ("pack",)), (_JointStructure, ("pack",)),
    (_ValueEntries, ("__init__", "add", "finish")),
    (_PreparingEntries, (
        "__init__", "add", "_discard", "_record", "prepare", "_prepare_route",
    )),
) for name in names)
_ROUTE_COMPILE = _CscRoute.__dict__["compile"]
