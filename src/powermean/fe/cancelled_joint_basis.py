"""Exact joint FE basis lifted from the same fit's admitted cancelled quotient."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, cast

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.fe.certificate_basis import (
    CertificateBasis,
    _first_rows,
    _packed_coordinates,
    _proof_bytes,
)
from powermean.fe.certificate_witness import (
    FactoredRows,
    PackedHeads,
    SparseValues,
    WitnessBlock,
    WitnessContext,
    WitnessProof,
)
from powermean.fe.connectivity import _indicator_rank_counted, components

if TYPE_CHECKING:
    from powermean.fe.admission_incidence import QuotientRank
    from powermean.fe.cancel import CancelStructure
    from powermean.fe.certificate_basis import Echelon, Witness
    from powermean.fe.groups import Grouping


def _basis_from_cancelled_quotient(
    grouping: Grouping, structure: CancelStructure, quotient: QuotientRank
) -> CertificateBasis | None:
    """Lift the admitted difference-row quotient on the same full joint rows.

    Subtract the first row of each cancelled group from its other rows. The
    leading incidence block then has an identity on the heads and zeros on
    differences. Thus the joint kept columns are all heads followed by the
    retained pivots. For a dependent retained column with quotient witness t,
    its joint head coefficients are its head incidence minus the kept head
    incidences times t. FactoredRows supplies these terms only on demand.
    """
    blocks = grouping.blocks
    levels = tuple(block.n_levels for block in blocks)
    n_heads = structure.n_groups
    if levels != (n_heads, *structure.block_sizes) or not np.array_equal(
        blocks[0].codes, structure.cancel_codes
    ):
        return None
    order = structure.order
    if any(
        not np.array_equal(block.codes[order], codes)
        for block, codes in zip(
            blocks[1:], structure.retained_sorted_codes, strict=True
        )
    ):
        return None
    boundaries = np.cumsum((0, *levels), dtype=np.intp)
    columns = tuple(
        frozen_index(np.arange(start, stop, dtype=np.intp))
        for start, stop in pairwise(boundaries)
    )
    pivots = tuple(n_heads + c for c in quotient.kept)
    count = int(boundaries[-1])
    selected, positions = _packed_coordinates(columns, n_heads, pivots, count)
    first = order[structure.starts]
    anchors = np.column_stack(
        [
            column[block.codes[first]]
            for column, block in zip(columns[1:], blocks[1:], strict=True)
        ]
    )
    heads = PackedHeads(
        n_heads,
        (np.arange(n_heads + 1, dtype=np.intp) * (len(blocks) - 1)).tobytes(),
        anchors.tobytes(),
        np.full(anchors.size, -1, dtype=np.intp).tobytes(),
        (),
        tuple(
            SparseValues(
                tuple(n_heads + column for column, _ in row),
                tuple(value for _, value in row),
            )
            for row in quotient.echelon
        ),
    )
    # Transpose the existing sparse quotient witnesses into the retained rows
    # consumed by the joint coordinate action. No head-by-column map is formed.
    kept = set(pivots)
    dependent = tuple(c for c in range(n_heads, count) if c not in kept)
    tails: list[list[tuple[int, int, int]]] = [[] for _ in pivots]
    for column in dependent:
        for position, numerator, denominator in quotient.witnesses[column - n_heads]:
            tails[position].append((column, numerator, denominator))
    factors = FactoredRows(
        heads,
        pivots,
        tuple(
            SparseValues(
                tuple(column for column, _, _ in row),
                tuple(n for _, n, _ in row),
                tuple(d for _, _, d in row) if any(d != 1 for _, _, d in row) else (),
            )
            for row in tails
        ),
        tuple(range(n_heads, selected.size)),
    )
    parts = components(blocks)
    rank = _indicator_rank_counted(blocks, parts)
    if not rank.lower <= selected.size <= rank.upper:
        return None
    context = WitnessContext(
        tuple(_proof_bytes(c) for c in columns),
        tuple(_proof_bytes(c) for c in parts.of_levels),
        tuple(_proof_bytes(p) for p in positions),
        dependent,
        (),
        factors,
    )
    witnesses: list[WitnessBlock] = []
    start = 0
    for block, level_columns, place in zip(blocks, columns, positions, strict=True):
        missing = tuple(np.flatnonzero(place < 0).tolist())
        witnesses.append(
            WitnessBlock(
                block.n_levels,
                start,
                missing,
                tuple(int(level_columns[level]) for level in missing),
                context,
                len(witnesses),
            )
        )
        start += block.n_levels - len(missing)
    proof = WitnessProof(tuple(witnesses), heads)
    return CertificateBasis(
        grouping,
        "elimination",
        rank,
        tuple(frozen_index(np.empty(0, dtype=np.intp)) for _ in blocks[1:]),
        columns,
        selected,
        positions,
        cast("tuple[tuple[Witness, ...], ...]", proof),
        cast("Echelon", proof),
        int(_first_rows(grouping).size),
    )
