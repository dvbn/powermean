"""Bounded full-joint PPML direction in the admitted two-block FE gauge."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

import numpy as np

from powermean.fe.absorb_numeric import checked_product, checked_quotient
from powermean.fe.certificate_basis import CertificateBasis, embed, lift
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_factor import index_width, linalg, sparse
from powermean.solve.absorbed_certificate_geometry import moment
from powermean.solve.absorbed_dense_factor import (
    DENSE_WORKSPACE_BYTES,
    schur_workspace_bytes,
)
from powermean.solve.compact_matrix import CompactMatrix


class _CG(Protocol):
    def __call__(
        self, operator: object, right: FloatArray, /, *, rtol: float, maxiter: int
    ) -> tuple[FloatArray, int]: ...


_cg = cast("_CG", import_module("scipy.sparse.linalg").cg)
_EPS = float(np.finfo(np.float64).eps)
# Bound the extra work before returning to the direct path.
_MAX_ITERATIONS = 128


def _expensive_schur(raw: CompactMatrix, n_slopes: int) -> bool:
    """Require the second FE block itself to exceed the compact entry budget."""
    parts = raw.raw
    remaining: int = parts.remainder.shape[0]
    entries: int = parts.nnz
    return (remaining - n_slopes) ** 2 > entries and schur_workspace_bytes(
        parts.dimension,
        remaining,
        entries,
        parts.left.nnz + parts.right.nnz,
        compact=True,
    ) > DENSE_WORKSPACE_BYTES


def _jacobi(raw: CompactMatrix) -> FloatArray:
    """Read square roots of the checked joint diagonal in implicit coordinates."""
    parts = raw.raw
    tail = parts.remainder.matrix
    matrix = sparse.csc_array((tail.data, tail.indices, tail.indptr), shape=tail.shape)
    diagonal = np.empty(parts.dimension)
    start, stop = parts.block
    diagonal[start:stop] = parts.diagonal.view()
    diagonal[parts.remaining.view()] = matrix.diagonal()
    if not np.isfinite(diagonal).all() or (diagonal < np.finfo(np.float64).tiny).any():
        raise FloatingPointError("the joint Jacobi scale is outside the normal range")
    return np.sqrt(diagonal)


def _row_coordinates(
    basis: CertificateBasis, n_slopes: int, size: int
) -> tuple[IndexArray, IndexArray]:
    """Map the two blocks to their canonical coordinates, with a zero reference."""
    rows: list[IndexArray] = []
    for block, positions in zip(basis.grouping.blocks, basis.positions, strict=True):
        kept = positions[block.codes]
        rows.append(np.where(kept >= 0, n_slopes + kept, size))
    first, second = rows
    return first, second


def _operator(
    design: FloatArray,
    basis: CertificateBasis,
    curvature: FloatArray,
    scale: FloatArray,
) -> object:
    """Apply R'R, for R = sqrt(C) B diag(scale)^-1, by row gathers and sums."""
    size, k = scale.size, design.shape[1]
    first, second = _row_coordinates(basis, k, size)
    root = np.sqrt(curvature)
    padded: FloatArray = np.append(scale, 1.0)
    first_scale = checked_quotient(root, padded[first])
    second_scale = checked_quotient(root, padded[second])
    slopes = checked_quotient(checked_product(root[:, None], design), scale[:k])

    def matvec(values: FloatArray) -> FloatArray:
        coefficients: FloatArray = np.append(values, 0.0)
        q = slopes @ coefficients[:k]
        q += first_scale * coefficients[first] + second_scale * coefficients[second]
        found = np.bincount(first, first_scale * q, minlength=size + 1)
        found += np.bincount(second, second_scale * q, minlength=size + 1)
        found[:k] = slopes.T @ q
        if not np.isfinite(found).all():
            raise FloatingPointError("the joint matrix action is not finite")
        return found[:size]

    return linalg.LinearOperator((size, size), matvec=matvec, dtype=np.float64)


def iterated_direction(
    design: FloatArray,
    basis: CertificateBasis,
    raw: CompactMatrix,
    curvature: FloatArray,
    score: FloatArray,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray] | None:
    """Try bounded CG and recompute every kept equation from the returned row step."""
    k = design.shape[1]
    if not _expensive_schur(raw, k) or index_width(raw.shape[0], raw.nnz):
        return None
    scale = _jacobi(raw)
    right = checked_quotient(moment(design, score, basis), scale)
    peak = float(np.max(np.abs(right)))
    if not peak:
        return None
    operator = _operator(design, basis, curvature, scale)
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        values, status = _cg(
            operator, right / peak, rtol=32 * _EPS, maxiter=_MAX_ITERATIONS
        )
    if status or not np.isfinite(values).all():
        return None
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        step = values * peak / scale
        change = design @ step[:k] + lift(basis, embed(basis, step[k:]))
    if not np.isfinite(step).all() or not np.isfinite(change).all():
        return None
    # Use checked original row products and sums, independent of the CG matvec and
    # its recurrence. Scale only after reducing the signed score and fitted action.
    achieved = moment(design, checked_product(curvature, change), basis)
    error = checked_quotient(achieved, scale) - right
    if float(np.max(np.abs(error))) > min(tolerance, 512 * _EPS) * peak:
        return None
    return step, change
