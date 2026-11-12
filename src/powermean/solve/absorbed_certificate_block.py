"""Analytical incidence-block factors with full-system actions and diagnostics.

The analytical system may use a different elimination order, not different
equations, thresholds or inference.
For A = [[D, B], [C, G]], D is one kept categorical block's exact diagonal.
Store C/D, B and D and factor only G - (C/D) B. No full block L/U or inverse
is materialized. Both adjoint actions include the eliminated right-hand side.
The ordinary and curvature matrices are prepared and factored independently.

Any number/topology of admitted blocks is allowed. A reduction is attempted
only if the selected principal block is diagonal and invertible. Ordinary
systems may use the directly constructed dense remainder when its accounted
array workspace fits the internal allowance. The existing stored-entry rule
remains for the sparse producer and for curvature/merit; these are optimization
eligibility tests, not new fit refusals or public resource limits. Other cases
use the original factor before attempting reduction. A failed reduced factor
never falls back.
Rank/support and full-system gates retain their full coordinate meaning.
Partitioned geometry supplies D/C/B/G directly; rich full matrices remain available
by lossless detached materialization. No full joint CSC is constructed merely to
slice these same partitions back out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from powermean.data.accounting import frozen_index
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_factor import (
    Calls,
    CscView,
    FactorKind,
    FactorRecord,
    FactorStatus,
    SuperLU,
    factor_record,
    factor_side,
    factorized,
    index_width,
    sparse,
)
from powermean.solve.absorbed_certificate_geometry import (
    FEGeometry,
    SparseMatrix,
    scaled_curvature,
)
from powermean.solve.absorbed_dense_factor import (
    DENSE_WORKSPACE_BYTES,
    DenseCholesky,
    DenseLU,
    dense_cholesky,
    dense_factorized,
    schur_workspace_bytes,
    workspace_bytes,
)
from powermean.solve.compact_matrix import CompactMatrix, CscStorage, matrix_nnz

Block = tuple[int, int]


class _Csc(CscView, Protocol):
    """Sparse operations used to extract and multiply the small block system."""

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        ...

    def transpose(self) -> _Csc:
        """The transpose for an adjoint action."""
        ...

    def __getitem__(self, key: object) -> _Csc:
        """A detached sparse submatrix."""
        ...

    def __matmul__(self, other: object) -> object:
        """A sparse matrix or dense RHS product."""
        ...

    def __sub__(self, other: object) -> _Csc:
        """A sparse matrix difference."""
        ...

    def sum_duplicates(self) -> None:
        """Canonicalize duplicate stored entries."""
        ...

    def sort_indices(self) -> None:
        """Sort rows in each column."""
        ...

    def toarray(self, order: str = "F") -> FloatArray:
        """A private dense representation with the requested storage order."""
        ...

    def eliminate_zeros(self) -> None:
        """Remove exactly zero entries, without a numerical threshold."""
        ...


def elimination_block(geometry: FEGeometry) -> Block | None:
    """Largest contiguous kept categorical block, with deterministic tie order.

    Exact-rank/gauge selection already occurred in the admitted basis. Dropped
    reference and dependent levels stay dropped; no topology or rank is inferred.
    """
    selected = None
    count = 0
    for position in geometry.basis.positions:
        kept = position[position >= 0]
        if kept.size <= count:
            continue
        first, stop = int(kept[0]), int(kept[-1]) + 1
        if np.array_equal(kept, np.arange(first, stop, dtype=np.intp)):
            selected = (geometry.n_slopes + first, geometry.n_slopes + stop)
            count = int(kept.size)
    return selected


@dataclass(frozen=True)
class _Parts:
    """Only the live diagonal/couplings and small unfactored remainder."""

    block: Block
    remaining: IndexArray
    diagonal: FloatArray
    left: _Csc
    right: _Csc
    remainder: _Csc
    dense_workspace: int = 0


def _parts(
    matrix: SparseMatrix | CompactMatrix, block: Block | None, *,
    ordinary: bool = False,
) -> _Parts | None:
    """Check structural/resource eligibility before any reduced factor attempt."""
    if isinstance(matrix, CompactMatrix):
        return _compact_parts(matrix, block, ordinary=ordinary)
    if block is None or matrix.shape[0] != matrix.shape[1]:
        return None
    n = matrix.shape[0]
    start, stop = block
    if not 0 <= start < stop <= n:
        return None
    r = n - (stop - start)
    workspace = (schur_workspace_bytes(
        n, r, matrix.data.size, matrix.data.size - (stop - start), compact=False,
    ) if ordinary else 0)
    direct = ordinary and workspace <= DENSE_WORKSPACE_BYTES
    if not direct and r * r > matrix.data.size:
        return None
    lo, hi = matrix.indptr[start], matrix.indptr[stop]
    rows = matrix.indices[lo:hi]
    cols = np.repeat(np.arange(start, stop), np.diff(matrix.indptr[start : stop + 1]))
    inside = (rows >= start) & (rows < stop)
    if np.any(rows[inside] != cols[inside]):
        return None
    full = cast(
        "_Csc",
        sparse.csc_array(
            (matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape
        ),
    )
    diagonal = np.array(full.diagonal()[start:stop], dtype=np.float64)
    if np.any(diagonal == 0) or not np.isfinite(diagonal).all():
        return None
    remaining = np.concatenate((np.arange(start), np.arange(stop, n))).astype(np.intp)
    columns = full[:, remaining]
    right, remainder = columns[start:stop, :], columns[remaining, :]
    left = full[:, start:stop][remaining, :]
    with np.errstate(all="ignore"):
        left.data[:] /= np.repeat(diagonal, np.diff(left.indptr))
    return _Parts(
        block, remaining, diagonal, left, right, remainder,
        workspace if direct else 0,
    )


def _compact_parts(
    matrix: CompactMatrix, block: Block | None, *, ordinary: bool = False,
) -> _Parts | None:
    """Select the producer from raw metadata before allocating scaled partitions."""
    raw = matrix.raw
    if block != raw.block:
        return _parts(matrix.materialize(), block, ordinary=ordinary)
    n, r = matrix.shape[0], raw.remainder.shape[0]
    # Only ordinary scaling shares the raw sparse pattern used by this bound.
    ordinary = ordinary and matrix.kind != "curvature"
    workspace = (schur_workspace_bytes(
        n, r, raw.nnz, raw.left.nnz + raw.right.nnz, compact=True,
    ) if ordinary else 0)
    direct = ordinary and workspace <= DENSE_WORKSPACE_BYTES
    if ordinary and not direct and r * r > raw.nnz:
        return None
    parts = matrix.parts
    remaining, diagonal = parts.remaining.view(), parts.diagonal.view()
    if not ordinary and r * r > matrix.nnz:
        return None
    if np.any(diagonal == 0) or not np.isfinite(diagonal).all():
        return None

    def scipy_partition(storage: CscStorage) -> _Csc:
        """Make only the private partition buffers needed by the existing action."""
        stored = storage.matrix
        return cast("_Csc", sparse.csc_array(
            (np.array(stored.data, copy=True), np.array(stored.indices, copy=True),
             np.array(stored.indptr, copy=True)), shape=stored.shape))

    left, right, remainder = (
        scipy_partition(p) for p in (parts.left, parts.right, parts.remainder)
    )
    with np.errstate(all="ignore"):
        left.data[:] /= np.repeat(diagonal, np.diff(left.indptr))
    return _Parts(
        parts.block, remaining, diagonal, left, right, remainder,
        workspace if direct else 0,
    )


def _product(matrix: _Csc, values: FloatArray) -> FloatArray:
    """A sparse vector or matrix action, with its original RHS shape."""
    return np.asarray(matrix @ values, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class BlockFactor:
    """A full operator backed by one diagonal block and a reduced factor.

    L/U nonzero diagnostics refer to the implicit full block factors. Only the
    reduced factor and the two sparse couplings are materialized; provenance
    separately records their counts and the actual reduced splu call. Current
    dense factor records its own call and leaves L/U counts unavailable.
    Ordinary dense LU supplies final/certificate ordinary actions; eligible
    curvature uses the Cholesky verdict and whitening action.
    """

    block: Block
    remaining: IndexArray
    diagonal: FloatArray
    left: _Csc
    right: _Csc
    reduced: SuperLU | DenseLU | DenseCholesky | None
    reduced_record: FactorRecord | None
    perm_r: NDArray[np.intc]
    perm_c: NDArray[np.intc]

    def solve(self, rhs: FloatArray, trans: str = "N") -> FloatArray:
        """Full A or A.T solve for a vector or any matrix RHS, including K=0."""
        rhs = np.asarray(rhs, dtype=np.float64)
        n = self.diagonal.size + self.remaining.size
        if rhs.ndim not in (1, 2) or rhs.shape[0] != n:
            raise ValueError("the block factor RHS has another full dimension")
        if trans not in ("N", "T"):
            raise ValueError("the block factor supports N and T actions")
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            return np.empty_like(rhs)
        start, stop = self.block
        b = rhs[start:stop]
        d = self.diagonal if rhs.ndim == 1 else self.diagonal[:, None]
        eliminated = b / d
        out = np.empty_like(rhs)
        if self.reduced is None:
            out[start:stop] = eliminated
            return out
        reduced_rhs = rhs[self.remaining] - (
            _product(self.left, b)
            if trans == "N"
            else _product(self.right.transpose(), eliminated)
        )
        x = self.reduced.solve(reduced_rhs, trans=trans)
        out[self.remaining] = x
        out[start:stop] = (
            (b - _product(self.right, x)) / d
            if trans == "N"
            else eliminated - _product(self.left.transpose(), x)
        )
        return out

    def curvature_reason(self) -> str:
        """Full implicit-factor finite/pivot checks, not reduced SPD alone.

        A returned dense Cholesky has no off-diagonal pivot. Its implicit LDL
        diagonal is recorded in natural order; dpotrf failures never get here.
        """
        if not np.array_equal(self.perm_r, self.perm_c):
            return "a threshold-zero off-diagonal pivot (perm_r != perm_c)"
        arrays = [self.diagonal, self.left.data, self.right.data]
        record = self.reduced_record
        if isinstance(self.reduced, DenseCholesky):
            arrays.append(self.reduced.values)
        elif isinstance(self.reduced, DenseLU):
            raise TypeError("the ordinary dense LU supplies no curvature test")
        elif self.reduced is not None:
            arrays.extend(
                (
                    factor_side(self.reduced, "L").data,
                    factor_side(self.reduced, "U").data,
                )
            )
        if not all(np.isfinite(values).all() for values in arrays):
            return "the curvature factor holds a value that is not finite"
        if np.any(self.diagonal <= 0) or (
            record is not None
            and (record.diagonal is None or np.any(record.diagonal <= 0))
        ):
            return "the curvature is not positive definite: a nonpositive pivot"
        return ""

    def whitened(self, rhs: FloatArray, calls: Calls) -> FloatArray:
        """Full information score, including every eliminated off-root score."""
        start, stop = self.block
        first = rhs[start:stop] / np.sqrt(self.diagonal)
        if self.reduced is None:
            return first
        if isinstance(self.reduced, DenseCholesky):
            residual = rhs[self.remaining] - _product(self.left, rhs[start:stop])
            return np.concatenate((first, calls.whitening(self.reduced, residual)))
        if isinstance(self.reduced, DenseLU):
            raise TypeError("the ordinary dense LU supplies no whitened score")
        record = self.reduced_record
        assert record is not None and record.perm_r is not None
        assert record.diagonal is not None
        residual = rhs[self.remaining] - _product(self.left, rhs[start:stop])
        permuted = np.empty_like(residual)
        permuted[record.perm_r] = residual
        second = calls.triangular(factor_side(self.reduced, "L"), permuted)
        second /= np.sqrt(record.diagonal)
        return np.concatenate((first, second))


def _remainder(parts: _Parts, kind: FactorKind) -> SparseMatrix:
    """Sparse Schur complement; curvature mirrors its range-safe pair mean."""
    schur = parts.remainder - (parts.left @ parts.right)
    schur.sum_duplicates()
    schur.sort_indices()
    schur.eliminate_zeros()
    if not np.isfinite(schur.data).all():
        raise FloatingPointError("the Schur complement is not representable")
    r = parts.remaining.size
    matrix = SparseMatrix(
        (r, r),
        np.asarray(schur.indptr, dtype=np.intp),
        np.asarray(schur.indices, dtype=np.intp),
        np.asarray(schur.data, dtype=np.float64),
    )
    return scaled_curvature(matrix, np.ones(r)) if kind != "ordinary" else matrix


def _dense_remainder(parts: _Parts) -> FloatArray:
    """Direct ordinary Schur buffer, using the SAME sparse product accumulation.

    Adapted from the delivered dense-final producer. Neither a sparse Schur
    subtraction nor its sorting/zero removal/scatter is needed. The product and
    its dense conversion die before in-place LU. Curvature keeps _remainder.
    """
    product = cast("_Csc", parts.left @ parts.right)
    dense = np.asarray(parts.remainder.toarray(order="F"), dtype=np.float64)
    with np.errstate(all="ignore"):
        dense -= np.asarray(product.toarray(order="F"), dtype=np.float64)
    del product
    if not np.isfinite(dense).all():
        raise FloatingPointError("the Schur complement is not representable")
    # Match sparse eliminate_zeros followed by the existing zero-filled scatter.
    dense[dense == 0] = 0.0
    return dense


def _record(
    parts: _Parts, kind: FactorKind, reduced: FactorRecord | None, nnz: int, *,
    dense: bool = False,
) -> FactorRecord:
    """Full implicit diagnostics plus truthful reduced-call/resource provenance."""
    start, stop = parts.block
    m, r = stop - start, int(parts.remaining.size)
    n = m + r
    row, col = np.empty(n, dtype=np.intp), np.empty(n, dtype=np.intp)
    row[start:stop] = col[start:stop] = np.arange(m)
    if (
        reduced is not None
        and reduced.perm_r is not None
        and reduced.perm_c is not None
    ):
        row[parts.remaining] = m + reduced.perm_r
        col[parts.remaining] = m + reduced.perm_c
    detail = MappingProxyType(
        {
            "factorization": "diagonal_schur",
            "eliminated_range": parts.block,
            "full_dimension": n,
            "reduced_dimension": r,
            "left_nnz": int(parts.left.nnz),
            "right_nnz": int(parts.right.nnz),
            "reduced_matrix_nnz": nnz,
            "remainder_entry_bound": r * r,
            "schur_product_terms": int(
                np.dot(
                    np.diff(parts.left.indptr).astype(np.int64),
                    np.bincount(parts.right.indices, minlength=m).astype(np.int64),
                )
            ),
            "reduced_L_nnz": None if reduced is None else reduced.l_nnz,
            "reduced_U_nnz": None if reduced is None else reduced.u_nnz,
            "splu": None if reduced is None or dense else reduced.call,
            **({("dense_cholesky" if kind == "curvature" else "dense_lu"): reduced.call}
               if dense and reduced is not None else {}),
            **({"dense_workspace_bytes": parts.dense_workspace,
                "dense_workspace_allowance": DENSE_WORKSPACE_BYTES}
               if dense and parts.dense_workspace else {}),
        }
    )
    status: FactorStatus = "factored" if reduced is None else reduced.status
    reason = "" if reduced is None else reduced.reason
    record = factor_record(kind, status, reason, detail)
    if dense and reduced is not None:
        record = replace(record, options=reduced.options)
    if status != "factored":
        return record
    l_nnz = m + int(parts.left.nnz)
    u_nnz = m + int(parts.right.nnz)
    if reduced is not None:
        l_nnz += int(reduced.l_nnz or 0)
        u_nnz += int(reduced.u_nnz or 0)
    diagonal = None
    if kind != "ordinary":
        tail = np.zeros(0) if reduced is None else reduced.diagonal
        assert tail is not None
        diagonal = immutable_array(np.concatenate((parts.diagonal, tail)))
    return replace(
        record,
        perm_r=frozen_index(row),
        perm_c=frozen_index(col),
        l_nnz=None if dense else l_nnz,
        u_nnz=None if dense else u_nnz,
        diagonal=diagonal,
    )


def block_factorized(
    matrix: SparseMatrix | CompactMatrix, kind: FactorKind, block: Block | None
) -> tuple[FactorRecord, SuperLU | None]:
    """Reduce only structurally eligible systems; preserve every failed attempt."""
    refusal = index_width(matrix.shape[0], matrix_nnz(matrix))
    if refusal:
        return factor_record(kind, "refused", refusal), None
    parts = _parts(matrix, block, ordinary=kind == "ordinary")
    if parts is None:
        return factorized(matrix.materialize() if isinstance(matrix, CompactMatrix) else matrix, kind)
    return _factor_parts(
        parts, kind,
        dense_entry_budget=matrix_nnz(matrix) if kind in ("ordinary", "curvature") else 0,
    )


def _factor_parts(
    parts: _Parts, kind: FactorKind, *, dense_entry_budget: int = 0,
) -> tuple[FactorRecord, SuperLU | None]:
    """Factor an eligible partition; choose storage before a factor, never after.

    The ordinary direct producer is selected before partition allocation.
    Otherwise the existing sparse producer and dense-buffer bound remain.
    Curvature may use Cholesky only within that same bound, after the unchanged
    ordered symmetrization of its sparse remainder. Merit stays sparse.
    A failed selected factor is never retried.
    """
    if not (
        np.isfinite(parts.left.data).all() and np.isfinite(parts.right.data).all()
    ):
        reason = "the block coupling is not representable"
        return factor_record(kind, "failed", reason), None
    reduced_record, reduced, nnz = None, None, 0
    dense = False
    if parts.remaining.size:
        dense = kind == "ordinary" and parts.dense_workspace > 0
        if dense:
            try:
                values = _dense_remainder(parts)
            except FloatingPointError as error:
                return factor_record(kind, "failed", str(error)), None
            nnz = int(np.count_nonzero(values))
            reduced_record, reduced = dense_factorized(values)
        else:
            try:
                remainder = _remainder(parts, kind)
            except FloatingPointError as error:
                return factor_record(kind, "failed", str(error)), None
            nnz = int(remainder.data.size)
            dense = (kind in ("ordinary", "curvature") and dense_entry_budget > 0
                     and workspace_bytes(remainder) <= 8 * dense_entry_budget)
            if dense:
                if kind == "curvature":
                    reduced_record, reduced = dense_cholesky(remainder)
                else:
                    reduced_record, reduced = dense_factorized(remainder)
            else:
                reduced_record, reduced = factorized(remainder, kind)
    record = _record(parts, kind, reduced_record, nnz, dense=dense)
    if record.status != "factored":
        return record, None
    assert record.perm_r is not None and record.perm_c is not None
    factor = BlockFactor(
        parts.block,
        parts.remaining,
        parts.diagonal,
        parts.left,
        parts.right,
        reduced,
        reduced_record,
        record.perm_r.astype(np.intc),
        record.perm_c.astype(np.intc),
    )
    return record, factor
