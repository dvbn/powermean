"""Partitioned FE geometry without a full joint CSC assembly.

The exact categorical
principal block is diagonal. Raw entries retain their row-order sums. Ordinary
and curvature views retain raw partition storage plus scales, not three full
joint matrices. Full CSC materialization is a separate, lossless rich operation.
No factor, provider, sample graph or caller ndarray is retained here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.sparse_matrix import SparseMatrix

MatrixKind = Literal["raw", "ordinary", "curvature"]
Entries = tuple[IndexArray, IndexArray, FloatArray]


@dataclass(frozen=True)
class CscStorage:
    """Immutable bytes and shape of one rectangular canonical partition."""

    shape: tuple[int, int]
    pointers: IndexStorage
    rows: IndexStorage
    values: FloatStorage

    @classmethod
    def of(cls, matrix: SparseMatrix) -> CscStorage:
        """Own checked canonical partition buffers without retaining array headers."""
        return cls(matrix.shape, IndexStorage.store(matrix.indptr),
                   IndexStorage.store(matrix.indices), FloatStorage.store(matrix.data))

    @property
    def matrix(self) -> SparseMatrix:
        """Fresh private headers on immutable partition bytes."""
        return SparseMatrix(self.shape, self.pointers.view(), self.rows.view(),
                            self.values.view())

    @property
    def nnz(self) -> int:
        """The stored-entry count, including intentional ordinary underflow zeros."""
        return self.values.shape[0]


def canonical_partition(entries: Entries, shape: tuple[int, int], *,
                        omit_zeros: bool = True) -> CscStorage:
    """Sort distinct entries by column then row, preserving every summed value."""
    rows, cols, values = entries
    if omit_zeros:
        keep = values != 0
        rows, cols, values = rows[keep], cols[keep], values[keep]
    order = np.lexsort((rows, cols))
    rows, cols, values = rows[order], cols[order], values[order]
    if np.any((np.diff(rows) == 0) & (np.diff(cols) == 0)):
        raise ValueError("an ordered entry of the Hessian was summed twice")
    pointers = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=shape[1]))))
    matrix = SparseMatrix(shape, np.asarray(pointers, dtype=np.intp),
                          np.asarray(rows, dtype=np.intp), np.asarray(values, dtype=np.float64))
    return CscStorage.of(matrix)


@dataclass(frozen=True)
class MatrixParts:
    """D, C, B, G for the original full coordinate order, never a reduced equation."""

    block: tuple[int, int]
    remaining: IndexStorage
    diagonal: FloatStorage
    stored_diagonal: IndexStorage
    left: CscStorage
    right: CscStorage
    remainder: CscStorage

    @property
    def dimension(self) -> int:
        """The full joint dimension."""
        return self.diagonal.shape[0] + self.remainder.shape[0]

    @property
    def nnz(self) -> int:
        """All stored entries of the implicit FULL matrix."""
        return (self.stored_diagonal.view().size + self.left.nnz
                + self.right.nnz + self.remainder.nnz)


def partition_entries(parts: MatrixParts) -> tuple[Entries, ...]:
    """Map each disjoint partition back to original full row/column coordinates."""
    start, _ = parts.block
    remaining = parts.remaining.view()
    diagonal = parts.stored_diagonal.view()
    left, right, remainder = (p.matrix for p in
                              (parts.left, parts.right, parts.remainder))
    return (
        (start + diagonal, start + diagonal, parts.diagonal.view()[diagonal]),
        (remaining[left.indices], start + left.columns, left.data),
        (start + right.indices, remaining[right.columns], right.data),
        (remaining[remainder.indices], remaining[remainder.columns], remainder.data),
    )


def materialize_parts(parts: MatrixParts) -> SparseMatrix:
    """One detached full CSC, only for an actual explicit/rich consumer."""
    entries = partition_entries(parts)
    rows = np.concatenate([e[0] for e in entries])
    cols = np.concatenate([e[1] for e in entries])
    values = np.concatenate([e[2] for e in entries])
    # Ordinary equilibration retains H's pattern even if an entry underflows.
    return canonical_partition((rows, cols, values), (parts.dimension, parts.dimension),
                               omit_zeros=False).matrix


@dataclass(frozen=True, eq=False)
class CompactMatrix:
    """Raw byte-owned partitions and a lossless ordinary/curvature view.

    The optional working cache is per matrix, hence per fresh point. It contains
    only immutable partition bytes, never a factor. Retained copies omit it and
    keep raw bytes/scales; rich access remains a fresh detached materialization.
    """

    raw: MatrixParts
    kind: MatrixKind = "raw"
    scaling: FloatStorage | None = None
    _working: list[MatrixParts] = field(default_factory=list, repr=False)

    @property
    def parts(self) -> MatrixParts:
        """The checked current view, formed once within this point's work."""
        if self.kind == "raw":
            return self.raw
        if not self._working:
            from powermean.solve.compact_scaling import scaled_parts

            assert self.scaling is not None
            self._working.append(scaled_parts(self.raw, self.scaling.view(), self.kind))
        return self._working[0]

    @property
    def shape(self) -> tuple[int, int]:
        """The FULL joint shape, not the factored remainder's shape."""
        return self.raw.dimension, self.raw.dimension

    @property
    def nnz(self) -> int:
        """The current full stored-entry count without a full matrix."""
        return self.parts.nnz

    def materialize(self) -> SparseMatrix:
        """Produce fresh full CSC buffers when explicitly required."""
        return materialize_parts(self.parts)

    @property
    def indptr(self) -> IndexArray:
        """Compatibility exposure for a genuine explicit consumer."""
        return self.materialize().indptr

    @property
    def indices(self) -> IndexArray:
        """Compatibility exposure for a genuine explicit consumer."""
        return self.materialize().indices

    @property
    def data(self) -> FloatArray:
        """Compatibility exposure for a genuine explicit consumer."""
        return self.materialize().data

    @property
    def columns(self) -> IndexArray:
        """Compatibility full CSC columns, formed only on demand."""
        return self.materialize().columns

    def scaled(self, scale: FloatArray, kind: MatrixKind) -> CompactMatrix:
        """Bind owned scales without constructing another full-sized matrix."""
        return CompactMatrix(self.raw, kind, FloatStorage.store(scale))

    def detached(self) -> CompactMatrix:
        """A passive copy with no working graph, factor or shared mutable header."""
        return CompactMatrix(self.raw, self.kind, self.scaling)

    def binding(self) -> tuple[object, ...]:
        """Exact immutable value binding, without serializing a full CSC graph."""
        p = self.raw
        matrices = tuple((m.shape, m.pointers.data, m.rows.data, m.values.data)
                         for m in (p.left, p.right, p.remainder))
        return (p.block, p.remaining.data, p.diagonal, p.stored_diagonal.data,
                matrices, self.kind, self.scaling)


def matrix_nnz(matrix: SparseMatrix | CompactMatrix) -> int:
    """Count actual FULL stored entries without forcing implicit materialization."""
    return matrix.nnz if isinstance(matrix, CompactMatrix) else int(matrix.data.size)
