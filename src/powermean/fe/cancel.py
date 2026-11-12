"""Owned layout and ordinary centering for one cancelled fixed-effect block.

Codes use the package's
first-occurrence label convention, including arbitrary hashable scalar labels.
Row identity is positional, independent of those labels. No row is deleted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.raw import Column
from powermean.fe._cancel_storage import (
    BuilderRecord,
    IndexArray,
    IndexStorage,
    finite_product,
)
from powermean.fe.groups import Labels, _positions, encode_block
from powermean.model.variance import FloatArray


@dataclass(frozen=True, eq=False, init=False)
class CancelStructure(BuilderRecord):
    """Immutable codes/segments, built by :func:`cancel_structure`.

    Numeric properties return new views on immutable bytes. Label properties
    return detached tuples using the existing ``Column`` ownership convention.
    Sorted rows are an internal reduction order; output rows remain positional.
    """

    __slots__ = (
        "_block_sizes",
        "_cancel_codes",
        "_cancel_levels",
        "_counts",
        "_inverse",
        "_order",
        "_positions",
        "_retained_codes",
        "_retained_levels",
        "_segments",
        "_sorted_codes",
    )

    _cancel_levels: Column
    _retained_levels: tuple[Column, ...]
    _cancel_codes: IndexStorage
    _order: IndexStorage
    _inverse: IndexStorage
    _segments: IndexStorage
    _counts: IndexStorage
    _sorted_codes: IndexStorage
    _retained_codes: tuple[IndexStorage, ...]
    _positions: IndexStorage
    _block_sizes: tuple[int, ...]

    @property
    def cancel_levels(self) -> tuple[object, ...]:
        """Cancelled labels, in package code order, detached on each access."""
        return tuple(self._cancel_levels.values.tolist())

    @property
    def retained_levels(self) -> tuple[tuple[object, ...], ...]:
        """Labels of every retained block, in its own code order."""
        return tuple(tuple(column.values.tolist()) for column in self._retained_levels)

    @property
    def cancel_codes(self) -> IndexArray:
        """Cancelled codes in input row order."""
        return self._cancel_codes.view()

    @property
    def order(self) -> IndexArray:
        """Input row index of each sorted row."""
        return self._order.view()

    @property
    def inverse(self) -> IndexArray:
        """Sorted row index of each input row."""
        return self._inverse.view()

    @property
    def positions(self) -> IndexArray:
        """Original sample positions in input row order."""
        return self._positions.view()

    @property
    def segments(self) -> IndexArray:
        """Group pointers, including the final row count."""
        return self._segments.view()

    @property
    def starts(self) -> IndexArray:
        """First sorted position of every cancelled group."""
        return self.segments[:-1]

    @property
    def counts(self) -> IndexArray:
        """Number of rows in each cancelled group."""
        return self._counts.view()

    @property
    def sorted_codes(self) -> IndexArray:
        """Cancelled codes in segment order."""
        return self._sorted_codes.view()

    @property
    def retained_sorted_codes(self) -> tuple[IndexArray, ...]:
        """One sorted code vector per retained block, with no block cap."""
        return tuple(codes.view() for codes in self._retained_codes)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Full number of levels in each retained block."""
        return self._block_sizes

    @property
    def n_rows(self) -> int:
        """Number of rows, including singleton and all-zero groups."""
        return self.positions.size

    @property
    def n_groups(self) -> int:
        """Number of cancelled groups."""
        return self.counts.size


def cancel_structure(
    cancel_labels: Labels,
    retained_labels: Sequence[Labels] = (),
    *,
    positions: ArrayLike | None = None,
) -> CancelStructure:
    """Encode aligned labels once; retain no caller-owned arrays.

    Parameters
    ----------
    cancel_labels : array_like
        Nonmissing hashable labels of the single cancelled block.
    retained_labels : sequence of array_like
        Zero or more aligned retained blocks; all their levels are kept.
    positions : array_like, optional
        Distinct nonnegative sample positions, defaulting to input row indices.

    Returns
    -------
    CancelStructure
        Owned layout. Labels follow ``fe.groups``' domain and refusals.
    """
    cancelled = encode_block(cancel_labels, name="cancelled")
    retained = tuple(
        encode_block(labels, name=f"retained{j}")
        for j, labels in enumerate(retained_labels)
    )
    if any(block.n_rows != cancelled.n_rows for block in retained):
        raise ValueError("all FE arrays must have the same number of rows")
    # These permutations are generated here, not supplied by a caller. Avoid a
    # second sort/ownership pass through a separate SortedRows record.
    order = np.argsort(cancelled.codes, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    segments = np.concatenate(([0], np.cumsum(cancelled.counts)))
    record = object.__new__(CancelStructure)
    object.__setattr__(record, "_cancel_levels", cancelled._levels)
    object.__setattr__(record, "_retained_levels", tuple(b._levels for b in retained))
    object.__setattr__(record, "_cancel_codes", IndexStorage.store(cancelled.codes))
    object.__setattr__(record, "_order", IndexStorage.store(order))
    object.__setattr__(record, "_inverse", IndexStorage.store(inverse))
    object.__setattr__(record, "_segments", IndexStorage.store(segments))
    object.__setattr__(record, "_counts", IndexStorage.store(cancelled.counts))
    object.__setattr__(
        record, "_sorted_codes", IndexStorage.store(cancelled.codes[order])
    )
    object.__setattr__(
        record,
        "_retained_codes",
        tuple(IndexStorage.store(b.codes[order]) for b in retained),
    )
    object.__setattr__(
        record,
        "_positions",
        IndexStorage.store(_positions(positions, cancelled.n_rows)),
    )
    object.__setattr__(record, "_block_sizes", tuple(b.n_levels for b in retained))
    return record


def segment_sum(structure: CancelStructure, values: FloatArray) -> FloatArray:
    """Sum sorted rows over cancelled groups along axis zero."""
    with np.errstate(over="raise", invalid="raise"):
        return finite_product(np.add.reduceat(values, structure.starts, axis=0))


def center_sorted(structure: CancelStructure, values: FloatArray) -> FloatArray:
    """Apply ordinary, unweighted ``M_C`` to sorted row vectors or slope matrices."""
    counts = structure.counts if values.ndim == 1 else structure.counts[:, None]
    with np.errstate(over="raise", invalid="raise"):
        means = segment_sum(structure, values) / counts
        return finite_product(values - means[structure.sorted_codes])


def to_sorted(structure: CancelStructure, values: FloatArray) -> FloatArray:
    """Rearrange input rows into stable segment order."""
    return values[structure.order]


def to_original(structure: CancelStructure, values: FloatArray) -> FloatArray:
    """Return sorted values to their input positional rows."""
    return values[structure.inverse]
