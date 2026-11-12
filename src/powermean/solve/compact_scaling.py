"""Exact equilibration, symmetric curvature and norms on partitioned FE storage.

The order is the existing geometry contract: ordinary (H_ij/s_i)/s_j;
curvature hplus first, then scales in original lower/upper coordinate order.
Column sums retain original full CSC row order, not partition accumulation order.
"""

from __future__ import annotations

import importlib
import math
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve import compact_matrix as matrices
from powermean.solve.compact_matrix import (
    CscStorage,
    MatrixKind,
    MatrixParts,
    canonical_partition,
    partition_entries,
)
from powermean.solve.sparse_matrix import SparseMatrix


def _finite(values: FloatArray, quantity: str = "Hessian") -> FloatArray:
    """Keep the original scaled-Hessian representability refusal."""
    if not np.isfinite(values).all():
        raise FloatingPointError(f"the scaled {quantity} is not representable")
    return values


def _mean(first: FloatArray, second: FloatArray) -> FloatArray:
    """The existing range-safe pair mean, with no tolerance or repair."""
    with np.errstate(over="ignore", under="ignore"):
        total = first + second
        found: FloatArray = np.where(np.isfinite(total), total / 2,
                                    first / 2 + second / 2)
    return found


def _ordinary(matrix: CscStorage, rows: IndexArray, cols: IndexArray,
              scale: FloatArray) -> CscStorage:
    """Scale one partition, keeping its exact stored structure and operation order."""
    source = matrix.matrix
    with np.errstate(all="ignore"):
        values = (source.data / scale[rows[source.indices]]) / scale[cols[source.columns]]
    return CscStorage(matrix.shape, matrix.pointers, matrix.rows,
                      FloatStorage.store(_finite(values)))


class _Csc(Protocol):
    """The existing SciPy CSC operations used for structural pairing only."""

    data: NDArray[np.int64] | FloatArray
    indices: NDArray[np.integer]
    indptr: NDArray[np.integer]
    shape: tuple[int, int]

    def __add__(self, other: _Csc) -> _Csc:
        """Union distinct positive slot tags, never add Hessian values."""
        ...

    def transpose(self) -> _Csc:
        """Transpose canonical structure."""
        ...

    def tocsc(self) -> _Csc:
        """Convert a transposed structure by the existing sparse kernel."""
        ...

    def sort_indices(self) -> None:
        """Canonical row order inside each column."""
        ...

    def eliminate_zeros(self) -> None:
        """Remove exact numerical zeros only after the finite check."""
        ...


class _Sparse(Protocol):
    """The existing sparse constructor, with integer or float data."""

    def csc_array(self, arg: object, /, *, shape: tuple[int, int]) -> _Csc:
        """Build a private working CSC from its three buffers."""
        ...


_sparse = cast("_Sparse", importlib.import_module("scipy.sparse"))
_CANONICAL_PARTITION = canonical_partition
_COLUMNS = vars(SparseMatrix)["columns"]
_STORAGE_OF = vars(CscStorage)["of"]


def _stored(matrix: _Csc) -> CscStorage:
    """Use the same public-value constructors on already canonical buffers."""
    return matrices.CscStorage.of(matrices.SparseMatrix(
        matrix.shape, np.asarray(matrix.indptr, dtype=np.intp),
        np.asarray(matrix.indices, dtype=np.intp),
        np.asarray(matrix.data, dtype=np.float64),
    ))


def _paired(left: CscStorage, right: CscStorage, rows: IndexArray,
            cols: IndexArray, scale: FloatArray) -> tuple[CscStorage, CscStorage]:
    """Align canonical CSC slots; keep the exact safe mean and full-coordinate order."""
    first, second = left.matrix, right.matrix
    if (type(left) is not CscStorage or type(right) is not CscStorage
            or type(first) is not SparseMatrix or type(second) is not SparseMatrix
            or canonical_partition is not _CANONICAL_PARTITION
            or vars(SparseMatrix).get("columns") is not _COLUMNS
            or vars(CscStorage).get("of") is not _STORAGE_OF
            or matrices.CscStorage is not CscStorage
            or matrices.SparseMatrix is not SparseMatrix
            or right.shape != (left.shape[1], left.shape[0])):
        return _paired_entries(left, right, first, second, rows, cols, scale)
    n_first, n_second = first.data.size, second.data.size
    stride = n_first + 1
    if stride > (1 << 63) - 1 or n_first + stride * n_second > (1 << 63) - 1:
        return _paired_entries(left, right, first, second, rows, cols, scale)
    # Every tag is positive, so the union retains even explicit zero-valued slots.
    # At most one slot from either canonical input occupies a union coordinate.
    a = _sparse.csc_array((
        np.arange(1, n_first + 1, dtype=np.int64), first.indices, first.indptr,
    ), shape=left.shape)
    b = _sparse.csc_array((
        stride * np.arange(1, n_second + 1, dtype=np.int64),
        second.indices, second.indptr,
    ), shape=right.shape)
    union = a + b.transpose().tocsc()
    del a, b
    union.sort_indices()
    tags = cast("NDArray[np.int64]", union.data)
    left_slot, right_slot = tags % stride, tags // stride
    x, y = np.zeros(tags.size), np.zeros(tags.size)
    present = left_slot != 0
    x[present] = first.data[left_slot[present] - 1]
    present = right_slot != 0
    y[present] = second.data[right_slot[present] - 1]
    del left_slot, right_slot, present
    row = np.asarray(union.indices, dtype=np.intp)
    col = np.repeat(np.arange(left.shape[1], dtype=np.intp), np.diff(union.indptr))
    lo, hi = np.minimum(rows[row], cols[col]), np.maximum(rows[row], cols[col])
    with np.errstate(all="ignore"):
        means = (_mean(x, y) / scale[lo]) / scale[hi]
    means = _finite(means, "curvature")
    del x, y, tags, row, col, lo, hi
    # Do not elide zeros before division: a stored 0/0 must still refuse.
    forward = _sparse.csc_array(
        (means, union.indices, union.indptr), shape=left.shape,
    )
    del means, union
    forward.eliminate_zeros()
    stored = _stored(forward)
    backward = forward.transpose().tocsc()
    backward.sort_indices()
    return stored, _stored(backward)


def _paired_entries(
    left: CscStorage, right: CscStorage, first: SparseMatrix, second: SparseMatrix,
    rows: IndexArray, cols: IndexArray, scale: FloatArray,
) -> tuple[CscStorage, CscStorage]:
    """Original coordinate producer for foreign callbacks or unencodable slot tags."""
    r = np.concatenate((first.indices, second.columns))
    c = np.concatenate((first.columns, second.indices))
    values = np.concatenate((first.data, second.data))
    side = np.arange(values.size) >= first.data.size
    order = np.lexsort((r, c))
    r, c, values, side = r[order], c[order], values[order], side[order]
    starts = np.ones(r.size, dtype=np.bool_)
    starts[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    groups = np.cumsum(starts) - 1
    x, y = np.zeros(int(starts.sum())), np.zeros(int(starts.sum()))
    x[groups[~side]], y[groups[side]] = values[~side], values[side]
    row, col = r[starts], c[starts]
    lo, hi = np.minimum(rows[row], cols[col]), np.maximum(rows[row], cols[col])
    with np.errstate(all="ignore"):
        means = (_mean(x, y) / scale[lo]) / scale[hi]
    means = _finite(means, "curvature")
    forward = canonical_partition((row, col, means), left.shape)
    backward = canonical_partition((col, row, means), right.shape)
    return forward, backward


def scaled_parts(raw: MatrixParts, scale: FloatArray, kind: MatrixKind) -> MatrixParts:
    """One independent ordinary or curvature view of the exact raw partitions."""
    start, stop = raw.block
    kept, remaining = np.arange(start, stop, dtype=np.intp), raw.remaining.view()
    diagonal = raw.diagonal.view()
    if kind == "ordinary":
        with np.errstate(all="ignore"):
            d = (diagonal / scale[kept]) / scale[kept]
        left = _ordinary(raw.left, remaining, kept, scale)
        right = _ordinary(raw.right, kept, remaining, scale)
        remainder = _ordinary(raw.remainder, remaining, remaining, scale)
        stored = raw.stored_diagonal
    elif kind == "curvature":
        with np.errstate(all="ignore"):
            d = (_mean(diagonal, diagonal) / scale[kept]) / scale[kept]
        left, right = _paired(raw.left, raw.right, remaining, kept, scale)
        remainder, _ = _paired(raw.remainder, raw.remainder, remaining, remaining, scale)
        stored = IndexStorage.store(np.flatnonzero(d != 0).astype(np.intp))
    else:
        raise ValueError("scale an ordinary or curvature view")
    quantity = "Hessian" if kind == "ordinary" else "curvature"
    return MatrixParts(raw.block, raw.remaining, FloatStorage.store(_finite(d, quantity)),
                       stored, left, right, remainder)


def equilibration_parts(parts: MatrixParts) -> FloatArray:
    """sqrt(max(rowmax(abs(H)), colmax(abs(H)))) on the FULL coordinate system."""
    rowmax, colmax = np.zeros(parts.dimension), np.zeros(parts.dimension)
    for rows, cols, values in partition_entries(parts):
        absolute = np.abs(values)
        np.maximum.at(rowmax, rows, absolute)
        np.maximum.at(colmax, cols, absolute)
    scale = np.sqrt(np.maximum(rowmax, colmax))
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError("zero row/column in the Hessian")
    return immutable_array(scale)


def one_norm_parts(parts: MatrixParts) -> float:
    """The largest full CSC column sum, in each column's original row order."""
    start, _ = parts.block
    diagonal, left, right, remainder = partition_entries(parts)
    sums = np.zeros(parts.dimension)
    ordered = (
        (remainder, remainder[0] < start),
        (right, np.ones(right[2].size, dtype=np.bool_)),
        (remainder, remainder[0] >= start),
        (left, left[0] < start),
        (diagonal, np.ones(diagonal[2].size, dtype=np.bool_)),
        (left, left[0] >= start),
    )
    with np.errstate(all="ignore"):
        for (_, cols, values), selected in ordered:
            np.add.at(sums, cols[selected], np.abs(values[selected]))
    norm = float(np.max(sums)) if sums.size else 0.0
    if not math.isfinite(norm):
        raise FloatingPointError("the one-norm of the scaled Hessian is not finite")
    return norm
