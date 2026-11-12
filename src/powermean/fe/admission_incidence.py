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

from collections.abc import Iterator
from fractions import Fraction
from itertools import pairwise
from typing import NamedTuple

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


def _insert(row: Row, pivots: dict[int, Row], work: WorkLog) -> None:
    reduced = work.run("exact_reduce", lambda: _reduced(row, pivots))
    if reduced:
        normalized = work.run(
            "exact_normalize",
            lambda: _primitive(dict(reduced)),
        )
        pivots[min(normalized)] = normalized


def _verify(
    structure: CancelStructure,
    kept: tuple[int, ...],
    witnesses: tuple[Witness, ...],
    work: WorkLog,
) -> None:
    inverse: dict[int, list[tuple[int, Fraction]]] = {}
    for column, witness in enumerate(witnesses):
        for position, numerator, denominator in witness:
            inverse.setdefault(kept[position], []).append(
                (column, Fraction(numerator, denominator))
            )
    for row in _differences(structure):
        work.begin("exact_witness_row")
        sums: dict[int, Fraction] = {}
        for column, multiple in row.items():
            for target, value in inverse.get(column, ()):
                work.begin("exact_witness_term")
                sums[target] = sums.get(target, Fraction(0)) + multiple * value
                work.complete("exact_witness_term")
        if any(sums.get(c, 0) != v for c, v in row.items()) or any(
            v != row.get(c, 0) for c, v in sums.items()
        ):
            raise ValueError("quotient witnesses do not reproduce original levels")
        work.complete("exact_witness_row")


def quotient_rank(structure: CancelStructure, work: WorkLog) -> QuotientRank:
    """Eliminate exact sparse difference rows and verify every original column."""
    pivots: dict[int, Row] = {}
    per_block: list[dict[int, Row]] = [{} for _ in structure.block_sizes]
    boundaries = [0]
    for size in structure.block_sizes:
        boundaries.append(boundaries[-1] + size)
    for row in _differences(structure):
        work.begin("exact_difference_row")
        work.begin("exact_input_entry", len(row))
        _insert(row, pivots, work)
        for block, (start, stop) in enumerate(pairwise(boundaries)):
            _insert(
                {c: v for c, v in row.items() if start <= c < stop},
                per_block[block],
                work,
            )
        work.complete("exact_input_entry", len(row))
        work.complete("exact_difference_row")
    kept = tuple(sorted(pivots))
    columns = work.run(
        "exact_column_witnesses",
        lambda: _column_witnesses(pivots, list(kept), boundaries[-1]),
    )
    witnesses = tuple(_canonical(columns[c]) for c in range(boundaries[-1]))
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
