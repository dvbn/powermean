"""Dense factors of admitted ordinary and curvature Schur systems.

The partial-pivoting backend also serves the final observed projection and
the certificate's ordinary actions, and the dense-final producer is adapted to it.
A Cholesky curvature factor is used within the existing stored-entry
and dense-workspace bounds; merit and ineligible curvature remain sparse.
The direct producer is selected by an internal route-local array-workspace bound,
not a memory limit set by the caller or a whole-process/native-allocator RSS
guarantee.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from powermean.data.accounting import frozen_index
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_factor import FactorRecord, factor_record
from powermean.solve.sparse_matrix import SparseMatrix


class _Lapack(Protocol):
    """The SciPy LAPACK double-precision factor and action interfaces used here."""

    def dgetrf(
        self, a: FloatArray, *, overwrite_a: bool,
    ) -> tuple[FloatArray, NDArray[np.intc], int]:
        """Partial-pivoting LU, overwriting the owned Fortran buffer."""
        ...

    def dgetrs(
        self, lu: FloatArray, piv: NDArray[np.intc], b: FloatArray, *,
        trans: int, overwrite_b: bool,
    ) -> tuple[FloatArray, int]:
        """Normal or transpose solve, overwriting only the owned RHS copy."""
        ...

    def dpotrf(
        self, a: FloatArray, *, lower: int, clean: int, overwrite_a: bool,
    ) -> tuple[FloatArray, int]:
        """Lower Cholesky, overwriting the owned Fortran buffer."""
        ...

    def dpotrs(
        self, c: FloatArray, b: FloatArray, *, lower: int, overwrite_b: bool,
    ) -> tuple[FloatArray, int]:
        """Symmetric solve, overwriting only the owned RHS copy."""
        ...

    def dtrtrs(
        self, a: FloatArray, b: FloatArray, *, lower: int, trans: int,
        unitdiag: int, overwrite_b: bool,
    ) -> tuple[FloatArray, int]:
        """Lower nonunit triangular solve for the curvature whitening."""
        ...


lapack = cast("_Lapack", importlib.import_module("scipy.linalg.lapack"))


def workspace_bytes(matrix: SparseMatrix) -> int:
    """Bound live remainder storage before choosing a dense remainder backend.

    The sparse Schur product has already been released by the unchanged producer.
    Count the surviving CSC, one float64 factor buffer and its finite-check mask.
    Sixteen intp-length vectors conservatively cover pivots, permutations and their
    detached record copies. No product, second dense matrix or full L/U coexists.
    This bound is compared with the existing full stored-entry budget in bytes;
    ineligibility keeps the original sparse factor, not a new fit refusal.
    """
    size = matrix.shape[0]
    sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    vectors = 16 * size * np.dtype(np.intp).itemsize
    return int(sparse_bytes + 9 * size * size + vectors)


# The ordinary producer's internal eligibility allowance, not a fit refusal.
DENSE_WORKSPACE_BYTES = 128 * 2**20


def schur_workspace_bytes(
    dimension: int, reduced: int, entries: int, couplings: int, *, compact: bool,
) -> int:
    """Bound accounted live arrays before ordinary partition/product allocation.

    Use worst-case eight-byte sparse indices. Compact ordinary scaling shares
    raw pointers/indices; raw storage, scaled values, writable partitions and
    both scale buffers require 40*entries + 16*m + 56*r + 48 bytes. For a full
    CSC, 64*entries + 64*n also covers extraction columns and structural masks.
    Multiplication allows two worst-case product CSCs, converted operand index
    arrays and row work arrays. Dense subtraction holds one product and two
    dense buffers; both temporaries are released before in-place LU. Factor
    checks and full permutation/provenance vectors are a separate peak.
    Existing fit arrays outside this route and native allocator overhead are
    not a promised RSS bound. All arithmetic here uses shape/nnz metadata only.
    """
    n, r, e = int(dimension), int(reduced), int(entries)
    m = n - r
    fixed = (40 * e + 16 * m + 56 * r + 48 if compact
             else 64 * e + 64 * n)
    product = 16 * r * r + 8 * (r + 1)
    converted_indices = 8 * int(couplings) + 8 * (n + 2)
    partition_work = 16 * e + 16 * (n + 1)
    product_work = 2 * product + converted_indices + 16 * r
    dense_work = product + 16 * r * r
    factor_work = 9 * r * r + 48 * n + 128 * r
    return fixed + max(partition_work, product_work, dense_work, factor_work)


Remainder = SparseMatrix | FloatArray


def _dense_buffer(matrix: Remainder) -> FloatArray:
    """Scatter canonical CSC or consume private owned Fortran float64 storage.

    Only the direct producer supplies an ndarray to the factor. Its buffer may
    be overwritten by LAPACK; a non-owned/non-Fortran input is copied first.
    """
    if isinstance(matrix, np.ndarray):
        if (matrix.dtype != np.float64 or not matrix.flags.f_contiguous
                or not matrix.flags.owndata or not matrix.flags.writeable):
            return np.array(matrix, dtype=np.float64, order="F", copy=True)
        return matrix
    values = np.zeros(matrix.shape, dtype=np.float64, order="F")
    for column in range(matrix.shape[0]):
        first, stop = matrix.indptr[column:column + 2]
        values[matrix.indices[first:stop], column] = matrix.data[first:stop]
    return values


@dataclass(frozen=True, eq=False)
class DenseLU:
    """Owned packed LU and pivots; only the ordinary N/T action is exposed."""

    lu: FloatArray
    piv: NDArray[np.intc]
    perm_r: NDArray[np.intc]
    perm_c: NDArray[np.intc]

    def solve(self, rhs: FloatArray, trans: str = "N") -> FloatArray:
        """Solve A or A.T for a vector or matrix RHS without modifying it."""
        values = np.asarray(rhs, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.lu.shape[0]:
            raise ValueError("the dense remainder RHS has another dimension")
        if trans not in ("N", "T"):
            raise ValueError("the dense remainder supports N and T actions")
        if not np.isfinite(values).all():
            raise FloatingPointError("the dense remainder RHS is not finite")
        if values.ndim == 2 and values.shape[1] == 0:
            return np.empty_like(values)
        one = values.ndim == 1
        work = np.array(values[:, None] if one else values, order="F", copy=True)
        found, info = lapack.dgetrs(
            self.lu, self.piv, work, trans=0 if trans == "N" else 1,
            overwrite_b=True,
        )
        if info != 0 or not np.isfinite(found).all():
            raise FloatingPointError("the dense remainder solve is not representable")
        return found[:, 0] if one else found


def _checked_cholesky_rhs(
    rhs: FloatArray, size: int, trans: str,
) -> FloatArray:
    """Validate the dense symmetric action without modifying its input."""
    values = np.asarray(rhs, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != size:
        raise ValueError("the dense remainder RHS has another dimension")
    if trans not in ("N", "T"):
        raise ValueError("the dense remainder supports N and T actions")
    if not np.isfinite(values).all():
        raise FloatingPointError("the dense remainder RHS is not finite")
    return values


@dataclass(frozen=True, eq=False)
class DenseCholesky:
    """Owned lower factor R of a symmetric remainder A = R R'.

    R^-1 v has squared norm v' A^-1 v. Its squared diagonal gives the implicit
    LDL pivots in the natural remainder order, not historical SuperLU pivots.
    """

    lower: FloatArray

    @property
    def diagonal(self) -> FloatArray:
        """The pivots of the implicit LDL factor, as a fresh array."""
        with np.errstate(all="ignore"):
            found: FloatArray = np.diagonal(self.lower) ** 2
        return found

    @property
    def values(self) -> FloatArray:
        """The factor buffer, with the unused upper triangle cleaned to zero."""
        return self.lower

    def solve(self, rhs: FloatArray, trans: str = "N") -> FloatArray:
        """Solve the symmetric A or A.T without modifying a vector/matrix RHS."""
        values = _checked_cholesky_rhs(rhs, self.lower.shape[0], trans)
        if values.ndim == 2 and values.shape[1] == 0:
            return np.empty_like(values)
        one = values.ndim == 1
        work = np.array(values[:, None] if one else values, order="F", copy=True)
        found, info = lapack.dpotrs(self.lower, work, lower=1, overwrite_b=True)
        if info != 0 or not np.isfinite(found).all():
            raise FloatingPointError("the dense remainder solve is not representable")
        return found[:, 0] if one else found

    def whiten(self, rhs: FloatArray) -> FloatArray:
        """R^-1 rhs for one vector, by one lower triangular solve."""
        values = _checked_cholesky_rhs(rhs, self.lower.shape[0], "N")
        if values.ndim != 1:
            raise ValueError("the whitening action takes one vector")
        work = np.array(values[:, None], order="F", copy=True)
        found, info = lapack.dtrtrs(
            self.lower, work, lower=1, trans=0, unitdiag=0, overwrite_b=True,
        )
        if info != 0:
            raise FloatingPointError("the whitening solve is not representable")
        return np.asarray(found[:, 0], dtype=np.float64)


def _row_permutation(piv: NDArray[np.intc]) -> NDArray[np.intc]:
    """Input-to-output scatter permutation, matching the existing factor records."""
    order = np.arange(piv.size, dtype=np.intc)
    for first, second in enumerate(piv):
        order[first], order[second] = order[second], order[first]
    permutation = np.empty_like(order)
    permutation[order] = np.arange(piv.size, dtype=np.intc)
    return permutation


def dense_factorized(matrix: Remainder) -> tuple[FactorRecord, DenseLU | None]:
    """Factor one admitted ordinary remainder; preserve failed attempts.

    The direct producer transfers its private buffer. CSC input retains its
    canonical scatter. Neither path forms full L/U for optional diagnostics.
    """
    size = matrix.shape[0]
    if matrix.shape != (size, size) or size == 0:
        raise ValueError("the dense remainder must be nonempty and square")
    entries = matrix if isinstance(matrix, np.ndarray) else matrix.data
    if not np.isfinite(entries).all():
        reason = "the Schur complement is not representable"
        return factor_record("ordinary", "failed", reason), None
    values = _dense_buffer(matrix)
    call = MappingProxyType({
        "routine": "scipy.linalg.lapack.dgetrf", "shape": matrix.shape,
        "overwrite_a": True,
    })
    options = MappingProxyType({"pivoting": "partial", "packed_lu": True})
    lu, piv, info = lapack.dgetrf(values, overwrite_a=True)
    record = replace(factor_record("ordinary", "factored", "", call), options=options)
    if info != 0 or not np.isfinite(lu).all():
        reason = (f"dgetrf argument {-info} is invalid" if info < 0 else
                  "the dense remainder is exactly singular" if info > 0 else
                  "the dense remainder factor is not finite")
        return replace(record, status="failed", reason=reason), None
    row, col = _row_permutation(piv), np.arange(size, dtype=np.intc)
    record = replace(
        record, perm_r=frozen_index(np.asarray(row, dtype=np.intp)),
        perm_c=frozen_index(np.asarray(col, dtype=np.intp)),
    )
    return record, DenseLU(lu, piv, row, col)


def dense_cholesky(
    matrix: SparseMatrix,
) -> tuple[FactorRecord, DenseCholesky | None]:
    """Factor the existing symmetrized remainder; preserve a failed attempt.

    Adapted from the delivered dense-final code. dpotrf decides positive
    definiteness in natural order. A failed leading minor is recorded, never
    retried with sparse LDL; borderline verdicts may differ from that ordering.
    """
    size = matrix.shape[0]
    if matrix.shape != (size, size) or size == 0:
        raise ValueError("the dense remainder must be nonempty and square")
    if not np.isfinite(matrix.data).all():
        reason = "the Schur complement is not representable"
        return factor_record("curvature", "failed", reason), None
    values = _dense_buffer(matrix)
    call: dict[str, object] = {
        "routine": "scipy.linalg.lapack.dpotrf", "shape": matrix.shape,
        "lower": True, "clean": True, "overwrite_a": True,
    }
    options = MappingProxyType({"pivoting": "none", "packed_cholesky": True})
    lower, info = lapack.dpotrf(values, lower=1, clean=1, overwrite_a=True)
    if info != 0:
        if info > 0:
            call["leading_minor"] = int(info)
        reason = (f"dpotrf argument {-info} is invalid" if info < 0 else
                  "the curvature is not positive definite: a nonpositive pivot")
        record = factor_record("curvature", "failed", reason, MappingProxyType(call))
        return replace(record, options=options), None
    factor = DenseCholesky(lower)
    diagonal = factor.diagonal
    if not (np.isfinite(factor.values).all() and np.isfinite(diagonal).all()):
        record = factor_record(
            "curvature", "failed",
            "the curvature factor holds a value that is not finite",
            MappingProxyType(call),
        )
        return replace(record, options=options), None
    identity = frozen_index(np.arange(size, dtype=np.intp))
    record = replace(
        factor_record("curvature", "factored", "", MappingProxyType(call)),
        options=options, perm_r=identity, perm_c=identity,
        diagonal=immutable_array(diagonal),
    )
    return record, factor
