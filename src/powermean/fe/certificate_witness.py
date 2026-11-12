"""Lossless immutable witness storage for one admitted exact FE construction.

Kept units and reference combinations are demanded views. Historical primitive
heads and dependent rational maps are detached immutable values, never rebuilt
from final tails. This stores proof values, not a rank/support verdict, mutable
array header, sample or provider.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from fractions import Fraction
import math
from typing import Callable, NamedTuple, cast

import numpy as np

Witness = tuple[tuple[int, int, int], ...]


class SparseValues(NamedTuple):
    """Ordered arbitrary-width integer values, with optional rational denominators."""

    positions: tuple[int, ...]
    numerators: tuple[int, ...]
    denominators: tuple[int, ...] = ()

    def items(self) -> Iterator[tuple[int, int]]:
        """Stream an integer proof row without allocating its outward pair tuples."""
        return zip(self.positions, self.numerators, strict=True)

    def canonical(self) -> Witness:
        """The exact promised ordered rational triples on actual demand."""
        if not self.denominators:
            return tuple((p, n, 1) for p, n in self.items())
        return tuple(zip(
            self.positions, self.numerators, self.denominators, strict=True
        ))

    def at(self, column: int, _fraction: type[Fraction] = Fraction) -> int | Fraction:
        """One exact coefficient, without expanding the other entries."""
        index = bisect_left(self.positions, column)
        if index == len(self.positions) or self.positions[index] != column:
            return 0
        numerator = self.numerators[index]
        return (numerator if not self.denominators else
                _fraction(numerator, self.denominators[index]))


class HeadRecipe(NamedTuple):
    """An original incidence anchor and immutable tail versions at its birth."""

    pivot: int
    anchor: tuple[int, ...]
    birth: tuple[tuple[int, SparseValues], ...]

    def at(self, column: int) -> int | Fraction:
        """The unnormalized birth coefficient, with its unit pivot implicit."""
        if column == self.pivot:
            return 1
        value: int | Fraction = int(column in self.anchor)
        for pivot, row in self.birth:
            if column == pivot:
                value -= 1
            value -= row.at(column)
        return value

    def expanded(
        self, _gcd: Callable[..., int] = math.gcd,
        _lcm: Callable[..., int] = math.lcm,
    ) -> SparseValues:
        """Reconstruct the SAME historical primitive row only on actual demand."""
        found: dict[int, int | Fraction] = dict.fromkeys(self.anchor, 1)
        found[self.pivot] = 1
        for pivot, row in self.birth:
            found.pop(pivot, None)
            for column in row.positions:
                value = found.get(column, 0) - row.at(column)
                if value:
                    found[column] = value
                else:
                    found.pop(column, None)
        scale = _lcm(*(value.denominator for value in found.values()))
        integers = {column: int(value * scale) for column, value in found.items()}
        divisor = _gcd(*integers.values())
        if integers[min(integers)] < 0:
            divisor = -divisor
        columns = tuple(sorted(integers))
        return SparseValues(columns, tuple(integers[c] // divisor for c in columns))


class PackedHeads(NamedTuple):
    """Dense head prefix, with original anchors and immutable birth versions.

    No per-level Python record exists until an actual outward demand. The
    captured HeadRecipe class makes that demand a value view, not a new call to
    a possibly rebound constructor. Historical versions are never final tails.
    """

    count: int
    offsets: bytes
    anchors: bytes
    births: bytes
    versions: tuple[SparseValues, ...]
    later: tuple[SparseValues, ...] = ()

    def head(self, index: int, _recipe: type[HeadRecipe] = HeadRecipe,
             _buffer=np.frombuffer, _index=np.intp) -> HeadRecipe:
        offsets = _buffer(self.offsets, dtype=_index)
        low, high = int(offsets[index]), int(offsets[index + 1])
        anchors = _buffer(self.anchors, dtype=_index)[low:high]
        births = _buffer(self.births, dtype=_index)[low:high]
        return tuple.__new__(_recipe, (index, tuple(map(int, anchors)), tuple(
            (int(column), self.versions[int(version)])
            for column, version in zip(anchors, births, strict=True) if version >= 0
        )))


class RetainedRows(NamedTuple):
    """Exact retained-only action, emitted with the fresh generated proof.

    Kept additions stay a separate vector phase. Nonkept terms are in original
    block/level order, each witness in retained-coordinate order. These immutable
    values are not admission: only the live producer's authenticated reader can
    offer them to the direct numerical consumer.
    """

    minimum: int
    kept_levels: tuple[bytes, ...]
    kept_positions: tuple[bytes, ...]
    levels: tuple[tuple[int, ...], ...]
    terms: tuple[tuple[Witness, ...], ...]
    norm: Fraction


class HeadRows(NamedTuple):
    """Operational head anchors in the existing immutable input topology.

    Only the first source row of each head is retained. No primitive row,
    birth-version, or historical proof object is constructed by this owner.
    """

    count: int
    first: bytes
    codes: tuple[bytes, ...]
    columns: tuple[bytes, ...]

    def anchor(self, index: int, _buffer=np.frombuffer, _index=np.intp) -> tuple[int, ...]:
        row = int(_buffer(self.first, dtype=_index)[index])
        return tuple(column for codes, columns in zip(
            self.codes[1:], self.columns[1:], strict=True
        ) if (column := int(_buffer(columns, dtype=_index)[
            int(_buffer(codes, dtype=_index)[row])
        ])) >= 0)


class FactoredRows(NamedTuple):
    """Canonical head anchors and final later rows, never expanded column maps."""

    heads: tuple[HeadRecipe, ...] | PackedHeads | HeadRows
    tail_columns: tuple[int, ...]
    tails: tuple[SparseValues, ...]
    tail_positions: tuple[int, ...]

    def dependent(self, column: int, minimum: int = 0) -> Witness:
        """Demand one column of A_D - A_Q T and T, in canonical J order."""
        found: list[tuple[int, int, int]] = []
        count = (self.heads.count if type(self.heads) is PackedHeads or type(self.heads) is HeadRows
                 else len(self.heads))
        for position in range(minimum, count):
            if type(self.heads) is HeadRows:
                anchor = self.heads.anchor(position)
            else:
                head = (self.heads.head(position) if type(self.heads) is PackedHeads
                        else self.heads[position])
                anchor = head.anchor
            value: int | Fraction = int(column in anchor)
            for pivot in anchor:
                index = bisect_left(self.tail_columns, pivot)
                if index < len(self.tail_columns) and self.tail_columns[index] == pivot:
                    value -= self.tails[index].at(column)
            if value:
                found.append((position, value.numerator, value.denominator))
        for position, row in zip(self.tail_positions, self.tails, strict=True):
            if position >= minimum and (value := row.at(column)):
                found.append((position, value.numerator, value.denominator))
        return tuple(found)


class WitnessContext(NamedTuple):
    """Detached numeric topology and dependent maps for lossless reference recipes."""

    columns: tuple[bytes, ...]
    components: tuple[bytes, ...]
    positions: tuple[bytes, ...]
    dependent_columns: tuple[int, ...]
    dependent_values: tuple[SparseValues, ...]
    factors: FactoredRows | None = None

    def dependent(self, column: int, minimum: int = 0) -> Witness:
        """Existing stored columns or the same exact factored column on demand."""
        if self.factors is not None:
            return self.factors.dependent(column, minimum)
        index = bisect_left(self.dependent_columns, column)
        return tuple(term for term in self.dependent_values[index].canonical()
                     if term[0] >= minimum)

    def _add(self, total: dict[int, int | Fraction], position: int,
             column: int, multiple: int, minimum: int = 0) -> None:
        """The original exact sparse addition, including immediate zero cancellation."""
        if position >= 0:
            if position < minimum:
                return
            if value := total.get(position, 0) + multiple:
                total[position] = value
            else:
                total.pop(position, None)
            return
        for p, n, d in self.dependent(column, minimum):
            term = n if d == 1 else Fraction(n, d)
            if value := total.get(p, 0) + multiple * term:
                total[p] = value
            else:
                total.pop(p, None)

    def references(
        self, block: int, levels: Sequence[int], minimum: int = 0,
    ) -> dict[int, Witness]:
        """Compose only requested reference values, with one ordered pass per block.

        First-block additions precede later-block subtractions, exactly as the
        eager canonical producer. Batch demand avoids a scan per component on
        disconnected inputs; no result is cached on the proof or another fit.
        """
        components = np.frombuffer(self.components[block], dtype=np.intp)
        totals: dict[int, dict[int, int | Fraction]] = {
            int(components[level]): {} for level in levels
        }
        reference = {int(components[level]): level for level in levels}
        first = np.frombuffer(self.components[0], dtype=np.intp)
        first_positions = np.frombuffer(self.positions[0], dtype=np.intp)
        first_columns = np.frombuffer(self.columns[0], dtype=np.intp)
        positions = np.frombuffer(self.positions[block], dtype=np.intp)
        columns = np.frombuffer(self.columns[block], dtype=np.intp)
        for level, component in enumerate(first):
            if int(component) in totals:
                self._add(totals[int(component)], int(first_positions[level]),
                          int(first_columns[level]), 1, minimum)
        for level, component in enumerate(components):
            key = int(component)
            if key in totals and level != reference[key]:
                self._add(totals[key], int(positions[level]), int(columns[level]),
                          -1, minimum)
        return {
            level: tuple((p, value.numerator, value.denominator)
                         for p, value in sorted(totals[int(components[level])].items())
                         if value)
            for level in levels
        }


class WitnessBlock(NamedTuple):
    """A block's nonkept exceptions, with canonical kept coordinates implicit."""

    size: int
    start: int
    levels: tuple[int, ...]
    values: tuple[Witness | SparseValues | int | None, ...]
    context: WitnessContext | None = None
    block: int = 0

    def witness(self, level: int) -> Witness:
        """One exact original-level witness, without expanding other units."""
        index = bisect_left(self.levels, level)
        if index < len(self.levels) and self.levels[index] == level:
            value = self.values[index]
            if type(value) is SparseValues:
                return value.canonical()
            if type(value) is int:
                assert self.context is not None
                return self.context.dependent(value)
            if value is None:
                assert self.context is not None
                return self.context.references(self.block, (level,))[level]
            return value
        return ((self.start + level - index, 1, 1),)

    def materialize(self) -> tuple[Witness, ...]:
        """Complete canonical tuple for an actual rich/public consumer."""
        requested = [level for level, value in zip(self.levels, self.values, strict=True)
                     if value is None]
        references = ({} if not requested or self.context is None
                      else self.context.references(self.block, requested))
        found: list[Witness] = []
        index, position = 0, self.start
        for level in range(self.size):
            if index < len(self.levels) and self.levels[index] == level:
                value = self.values[index]
                found.append(value.canonical() if type(value) is SparseValues
                             else self.context.dependent(value)
                             if type(value) is int and self.context is not None
                             else references[level] if value is None else value)
                index += 1
            else:
                found.append(((position, 1, 1),))
                position += 1
        return tuple(found)

    def iter_selected(self, levels: Sequence[int]) -> Iterator[Witness]:
        """Demand one batch of numerical nonzero levels, not all other witnesses."""
        requested = []
        for level in levels:
            index = bisect_left(self.levels, level)
            if (index < len(self.levels) and self.levels[index] == level
                    and self.values[index] is None):
                requested.append(level)
        references = ({} if not requested or self.context is None
                      else self.context.references(self.block, requested))
        for level in levels:
            yield references[level] if level in references else self.witness(level)

    def iter_witnesses(self) -> Iterator[Witness]:
        """Stream all declared terms when a numerical consumer really needs them."""
        if (type(self) is not WitnessBlock
                or WitnessBlock.witness is not _BLOCK_WITNESS
                or WitnessBlock.__getattribute__ is not _BLOCK_ACCESS):
            for level in range(self.size):
                yield self.witness(level)
            return
        requested = [level for level, value in zip(self.levels, self.values, strict=True)
                     if value is None]
        references = ({} if not requested or self.context is None
                      else self.context.references(self.block, requested))
        for level in range(self.size):
            yield references[level] if level in references else self.witness(level)


class WitnessProof(NamedTuple):
    """Canonical exceptions/echelon, detached from every mutable producer map."""

    blocks: tuple[WitnessBlock, ...]
    echelon: tuple[tuple[tuple[int, int], ...] | SparseValues | HeadRecipe, ...] | PackedHeads
    retained: RetainedRows | None = None

    def witness(self, block: int, level: int) -> Witness:
        """Read only the original-level map actually needed by an action."""
        return self.blocks[block].witness(level)

    def materialize(self) -> tuple[tuple[Witness, ...], ...]:
        """Expose every promised level as ordinary immutable tuple values."""
        return tuple(block.materialize() for block in self.blocks)

    def materialize_echelon(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Expose historical integer rows as the unchanged exact outward tuples."""
        if type(self.echelon) is PackedHeads:
            return tuple(tuple(self.echelon.head(index).expanded().items())
                         for index in range(self.echelon.count)) + tuple(
                tuple(row.items()) for row in self.echelon.later
            )
        return tuple(tuple(row.expanded().items()) if type(row) is HeadRecipe else
                     tuple(row.items()) if type(row) is SparseValues else row
                     for row in self.echelon)


_BLOCK_WITNESS = WitnessBlock.witness
_BLOCK_ACCESS = WitnessBlock.__getattribute__


class OperationalBasis(NamedTuple):
    """One numerical layout/action, not a certificate presentation graph.

    The immutable topology bytes already belong to this fit. Final tails and
    ordered actions are numerical data; historical primitive rows and generic
    WitnessBlock/Proof records are deliberately absent.
    """

    columns: tuple[bytes, ...]
    components: tuple[bytes, ...]
    positions: tuple[bytes, ...]
    factors: FactoredRows
    retained: RetainedRows | None = None

    def dependent(self, column: int, minimum: int = 0) -> Witness:
        return self.factors.dependent(column, minimum)

    _add = WitnessContext._add
    references = WitnessContext.references

    def audit(self, kernels: tuple, _buffer=np.frombuffer, _index=np.intp) -> tuple[
            tuple[tuple[Witness, ...], ...],
            tuple[tuple[tuple[int, int], ...], ...]]:
        """Adapt immutable values through the one existing exact audit kernels.

        The caller supplies the package's captured original kernels/primitives.
        No fit, provider, profile, absorption, verdict or numerical capability is
        consumed, and no result is cached on this value-only record.
        """
        reduce, column_witnesses, dependent, witnesses, canonical, reference, primitives, flat = kernels[:8]
        if len(kernels) > 8:
            try:
                replayed = kernels[8](self)
            except (MemoryError, TypeError, ValueError, IndexError, OverflowError):
                replayed = None  # Malformed/optional views keep the original checked route.
            if replayed is not None:
                return cast("tuple", replayed)
        heads = self.factors.heads
        assert type(heads) is HeadRows
        columns = tuple(_buffer(raw, dtype=_index) for raw in self.columns)
        codes = tuple(_buffer(raw, dtype=_index) for raw in heads.codes)
        parts = tuple(_buffer(raw, dtype=_index) for raw in self.components)
        pivots: dict[int, dict[int, int]] = {}
        for values in zip(*codes, strict=True):
            row = reduce(dict.fromkeys((int(col[int(code)]) for col, code in
                         zip(columns, values, strict=True) if col[int(code)] >= 0), 1),
                         pivots, _primitives=primitives)
            if row:
                pivots[min(row)] = row
        kept = sorted(pivots)
        count = sum(int((column >= 0).sum()) for column in columns)
        maps = column_witnesses(pivots, kept, count,
                                _audit=(dependent, primitives[5]))
        views = witnesses(None, None, columns, maps,
                          _audit=(parts, tuple(len(c) for c in columns),
                                  flat, canonical, reference))
        return views, tuple(tuple(sorted(pivots[p].items())) for p in kept)
