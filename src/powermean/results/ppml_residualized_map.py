"""Canonical ALL-level witnesses, checked products and normwise map predicates.

Simple projection ``projection_map_v2``.
Witnesses sum every block/level term in declared order, including
references and dependent levels. Stored transpose products preserve CSC entry order
and both mixed orientations. Checked binary64 arithmetic retains exact zeros and
representable subnormals and refuses invalid, overflow and inexact underflow.
Backward checks are runtime predicates, never forward-accuracy certificates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.fe.absorb_numeric import (
    checked_product,
    group_sums,
)
from powermean.fe.certificate_basis import CertificateBasis, Witness, _witness_proof
from powermean.fe.certificate_witness import WitnessBlock
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.ppml_standard_records import (
    BudgetRefusal,
    MapCheck,
    holds,
    normwise_bound,
)
from powermean.solve.absorbed_certificate_geometry import SparseMatrix

NOT_FINITE = "a value of the predicate is not finite"


class BudgetExceededError(Exception):
    """A reservation that does not fit the remaining units of a ledger."""

    def __init__(self, refusal: BudgetRefusal) -> None:
        super().__init__(
            f"{refusal.action} needs {refusal.required_s} S and {refusal.required_e} E"
            f" units; {refusal.remaining_s} S and {refusal.remaining_e} E remain"
        )
        self.refusal = refusal


# ------------------------------------------------------ canonical coordinates


@dataclass(frozen=True, eq=False)
class Coordinates:
    """``T`` (``p x K``, owned) and its witness terms, the inexact ratios counted."""

    values: FloatArray
    terms: int
    inexact: int


def _ratio(numerator: int, denominator: int) -> tuple[float, bool]:
    """``n/d`` correctly rounded, and whether it is exact; refused off range."""
    try:
        found = numerator / denominator
    except OverflowError as error:
        raise FloatingPointError(f"witness ratio: {error}") from error
    if not math.isfinite(found) or (found == 0.0 and numerator != 0):
        raise FloatingPointError("witness ratio: not representable")
    return found, Fraction(found) == Fraction(numerator, denominator)


def witness_coordinates(
    basis: CertificateBasis, coefficients: tuple[FloatArray, ...]
) -> Coordinates:
    """``T`` of every block coefficient by the witnesses (module docstring).

    ``coefficients`` holds one ``L_b x K`` array per block of the basis's grouping, in
    order; ``ValueError`` for another layout, ``FloatingPointError`` on range.
    """
    blocks = basis.grouping.blocks
    proof = _witness_proof(basis)
    if len(coefficients) != len(blocks) or (
        len(basis.witnesses) if proof is None else len(proof.blocks)
    ) != len(blocks):
        raise ValueError("one coefficient array and witness tuple per block")
    places: list[int] = []
    ratios: list[float] = []
    gathered: list[FloatArray] = []
    inexact = 0
    for block, levels, values in zip(
        blocks, basis.witnesses if proof is None else proof.blocks, coefficients,
        strict=True,
    ):
        if values.ndim != 2 or values.shape[0] != block.n_levels:
            raise ValueError("block coefficients must be one row per level")
        if (len(levels) if proof is None else cast("WitnessBlock", levels).size
                ) != block.n_levels:
            raise ValueError("one witness per level")
        rows: list[int] = []
        witness_terms = (
            cast("tuple[Witness, ...]", levels) if proof is None
            else cast("WitnessBlock", levels).iter_witnesses()
        )
        for level, witness in enumerate(witness_terms):
            for position, numerator, denominator in witness:
                ratio, exact = _ratio(numerator, denominator)
                places.append(position)
                ratios.append(ratio)
                rows.append(level)
                inexact += not exact
        gathered.append(values[np.asarray(rows, dtype=np.intp)])
    weights = np.asarray(ratios, dtype=np.float64)[:, None]
    terms = checked_product(weights, np.concatenate(gathered), "witness term")
    codes = np.asarray(places, dtype=np.intp)
    found = group_sums(terms, codes, basis.n_kept)
    return Coordinates(found, len(places), inexact)


# --------------------------------------------------------- checked products


@dataclass(frozen=True, eq=False)
class SparseBlock:
    """The stored entries of one block of a canonical CSC matrix, in storage order.

    ``rows`` and ``columns`` are coordinates within the block; ``shape`` is its own.
    """

    rows: IndexArray
    columns: IndexArray
    data: FloatArray
    shape: tuple[int, int]


def stored_block(
    matrix: SparseMatrix, first_row: int, columns: tuple[int, int]
) -> SparseBlock:
    """The stored entries of the rows from ``first_row`` and the columns ``columns``."""
    low, high = columns
    rows, cols = matrix.indices, matrix.columns
    keep = (rows >= first_row) & (cols >= low) & (cols < high)
    return SparseBlock(
        frozen_index(rows[keep] - first_row),
        frozen_index(cols[keep] - low),
        immutable_array(matrix.data[keep]),
        (matrix.shape[0] - first_row, high - low),
    )


def transpose_action(block: SparseBlock, values: FloatArray) -> FloatArray:
    """``M' v`` of stored entries (module docstring); ``FloatingPointError`` on range.

    ``values`` is a vector or a matrix with one row per row of the block; the result
    has one row per column of the block, an exact zero where it stores nothing.
    """
    count = block.shape[1]
    if not block.data.size:
        return np.zeros((count, *values.shape[1:]))
    entries = block.data if values.ndim == 1 else block.data[:, None]
    terms = checked_product(entries, values[block.rows], "transpose product term")
    return group_sums(terms, block.columns, count)


def cross(coordinates: FloatArray, values: FloatArray) -> FloatArray:
    """``T U``: per column of ``U``, the checked ``T[i, k] U[k, j]`` in ``k`` order."""
    single = np.zeros(coordinates.shape[1], dtype=np.intp)
    columns: list[FloatArray] = []
    for column in range(values.shape[1]):
        terms = checked_product(coordinates, values[:, column], "cross product term")
        columns.append(group_sums(np.ascontiguousarray(terms.T), single, 1)[0])
    return np.column_stack(columns)


def _largest(values: FloatArray) -> float:
    """The largest absolute entry (exact), ``0`` for no entry."""
    return float(np.max(np.abs(values), initial=0.0))


# --------------------------------------------------------------- predicates


def normwise(
    name: str,
    stage: int,
    residual: FloatArray,
    solution: FloatArray,
    rhs: FloatArray,
    *,
    norm: float,
    tolerance: float,
) -> tuple[MapCheck, ...]:
    """One normwise backward predicate per column (module docstring).

    ``FloatingPointError`` for a value that is not finite or a bound off range.
    """
    found: list[MapCheck] = []
    for column in range(residual.shape[1]):
        e, x = _largest(residual[:, column]), _largest(solution[:, column])
        b = _largest(rhs[:, column])
        if not all(map(math.isfinite, (e, x, b, norm))):
            raise FloatingPointError(f"{name}: {NOT_FINITE}")
        bound = normwise_bound(tolerance, norm, x, b)
        values = (e, norm, x, b, tolerance, bound, holds(e, bound))
        found.append(MapCheck(name, stage, column, *values))
    return tuple(found)
