"""Exact recession witnesses for the cancelled design.

The floating-point LP locates a face. Binary input slopes and candidate coefficients
are rational numbers. Center their products with integer arithmetic, including all
zero-outcome rows, to check every positive-row sign exactly. If the LP point misses
its face through rounding, solve its active equalities over the rationals and check
the repaired direction exactly. Failure to repair certifies nothing.

No denominator cutoff, rank tolerance, or iteration budget is used. The existing LP
feasibility tolerance selects candidate binding rows; it never accepts a sign.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

import numpy as np

from powermean.fe._cancel_storage import IndexArray
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray
from powermean.solve.support_gamma import BASELINE

Row = dict[int, Fraction]
Basis = dict[int, tuple[Row, Fraction]]


def _products(
    design: CenteredDesign,
    coefficients: list[Fraction],
    slopes: FloatArray | None = None,
) -> tuple[list[int], int]:
    """Numerators of centered products; row denominators are count[g] * common."""
    structure = design.structure
    common = math.lcm(*(value.denominator for value in coefficients))
    integers = [
        value.numerator * (common // value.denominator) for value in coefficients
    ]
    source = design.centered_slopes if slopes is None else slopes
    columns: dict[int, list[tuple[int, int]]] = {}
    exponent = 0
    for column in range(design.n_slopes):
        if not integers[column]:
            continue
        pairs = [float(value).as_integer_ratio() for value in source[:, column]]
        columns[column] = pairs
        largest_denominator = max(den for _, den in pairs)
        exponent = max(exponent, largest_denominator.bit_length() - 1)
    values = [0] * structure.n_rows
    for column, pairs in columns.items():
        coefficient = integers[column]
        for index, (numerator, denominator) in enumerate(pairs):
            values[index] += (numerator * coefficient) << (
                exponent - (denominator.bit_length() - 1)
            )
    for codes, start, size in zip(
        structure.retained_sorted_codes,
        design.block_starts,
        structure.block_sizes,
        strict=True,
    ):
        levels = integers[start : start + size]
        if not any(levels):
            continue
        levels = [value << exponent for value in levels]
        for index, code in enumerate(codes):
            values[index] += levels[int(code)]
    centered = [0] * structure.n_rows
    for start, count in zip(
        structure.starts.tolist(), structure.counts.tolist(), strict=True
    ):
        total = sum(values[start : start + count])
        for index in range(start, start + count):
            centered[index] = count * values[index] - total
    return centered, common << exponent


def signs(
    design: CenteredDesign,
    direction: FloatArray,
    positive: IndexArray,
    *,
    slopes: FloatArray | None = None,
) -> tuple[bool, bool]:
    """Whether the exact centered positive products have either strict sign."""
    products, _ = _products(
        design, [Fraction(float(value)) for value in direction], slopes
    )
    return (
        any(products[int(index)] > 0 for index in positive),
        any(products[int(index)] < 0 for index in positive),
    )


class _Rows:
    """Sparse rational rows, with group means constructed on demand."""

    def __init__(
        self, design: CenteredDesign, slopes: FloatArray | None = None
    ) -> None:
        self.design = design
        self.slopes = design.centered_slopes if slopes is None else slopes
        self.means: dict[int, Row] = {}

    def row(self, index: int) -> Row:
        """One exactly centered row of source slopes and retained indicators."""
        design = self.design
        structure = design.structure
        group = int(structure.sorted_codes[index])
        if group not in self.means:
            self.means[group] = self._mean(group)
        result = {column: -value for column, value in self.means[group].items()}
        for column in range(design.n_slopes):
            result[column] = result.get(column, Fraction()) + Fraction(
                float(self.slopes[index, column])
            )
        for codes, start in zip(
            structure.retained_sorted_codes, design.block_starts, strict=True
        ):
            column = start + int(codes[index])
            result[column] = result.get(column, Fraction()) + 1
        return {column: value for column, value in result.items() if value}

    def _mean(self, group: int) -> Row:
        design = self.design
        structure = design.structure
        start = int(structure.starts[group])
        count = int(structure.counts[group])
        result = {}
        for column in range(design.n_slopes):
            result[column] = (
                sum(
                    (
                        Fraction(float(value))
                        for value in self.slopes[start : start + count, column]
                    ),
                    Fraction(),
                )
                / count
            )
        for codes, first in zip(
            structure.retained_sorted_codes, design.block_starts, strict=True
        ):
            counts = Counter(codes[start : start + count].tolist())
            for code, frequency in counts.items():
                result[first + int(code)] = Fraction(frequency, count)
        return {column: value for column, value in result.items() if value}


def _insert(basis: Basis, row: Row, right: Fraction) -> bool:
    """Insert an exact equality, returning false if it is inconsistent."""
    while row:
        pivot = min(row)
        coefficient = row[pivot]
        if pivot not in basis:
            basis[pivot] = (
                {column: value / coefficient for column, value in row.items()},
                right / coefficient,
            )
            return True
        previous, value = basis[pivot]
        for column, entry in previous.items():
            updated = row.get(column, Fraction()) - coefficient * entry
            if updated:
                row[column] = updated
            else:
                row.pop(column, None)
        right -= coefficient * value
    return not right


def witness(
    design: CenteredDesign,
    direction: FloatArray,
    positive: IndexArray,
    *,
    candidate_h: FloatArray | None = None,
    slopes: FloatArray | None = None,
) -> bool:
    """Check an exact direction, optionally repairing the LP's candidate face."""
    coefficients = [Fraction(float(value)) for value in direction]
    products, common = _products(design, coefficients, slopes)
    if all(products[int(index)] >= 0 for index in positive) and any(
        products[int(index)] > 0 for index in positive
    ):
        return True
    if candidate_h is None:
        return False
    anchor = int(positive[np.argmax(candidate_h[positive])])
    if products[anchor] <= 0:
        return False
    scale = float(np.max(np.abs(candidate_h)))
    active = positive[np.abs(candidate_h[positive]) <= BASELINE.feasibility * scale]
    rows = _Rows(design, slopes)
    basis: Basis = {}
    for index in active:
        if not _insert(basis, rows.row(int(index)), Fraction()):
            return False
    if not _insert(basis, rows.row(anchor), Fraction(1)):
        return False
    group = int(design.structure.sorted_codes[anchor])
    multiplier = Fraction(
        common * int(design.structure.counts[group]), products[anchor]
    )
    repaired = [value * multiplier for value in coefficients]
    for pivot in sorted(basis, reverse=True):
        row, right = basis[pivot]
        repaired[pivot] = right - sum(
            (
                value * repaired[column]
                for column, value in row.items()
                if column != pivot
            ),
            Fraction(),
        )
    products, _ = _products(design, repaired, slopes)
    return all(products[int(index)] >= 0 for index in positive) and any(
        products[int(index)] > 0 for index in positive
    )
