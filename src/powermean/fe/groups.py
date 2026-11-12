"""Grouping of the fixed-effect blocks: levels, codes, counts, interactions and sorting.

A block is a partition of the observations into levels, encoded before it is
estimated, cancelled or differenced.  The preparation layer works on arrays and
integer codes.

A block is encoded by the first occurrence of its labels (the level of the first row is
code 0, the next new label code 1, and so on).  Nothing here handles a pandas index,
refuses duplicated row labels, builds a dense dummy matrix or a pivoted QR, or imports
pandas.  The encoding gives the levels by first occurrence, the codes, the counts and
the names, and refuses a missing or unhashable label, an empty or multidimensional
array and blocks of unequal length.

Row identity.  Every row keeps its exact original position (``Grouping.positions``), a
distinct nonnegative integer supplied by the caller (:mod:`powermean.data.sample`) or
defaulting to the row order ``0 .. n - 1``.  Labels of the user, of the rows as of the
levels, may repeat and are never a key here: levels are labels, codes are integers,
positions are the identity.  Encoding reorders nothing and deletes nothing: a singleton
level, a level of zero outcomes and a row of zero weight are encoded like any other
(steps 5 to 8 of the sample construction are not done here).  A missing label is
``None`` or a value unequal to itself, NaN of any type or NaT (:func:`_is_missing`);
an unhashable label is refused with it.

Exactness.  The labels are read as the Python objects given, through an object array,
never through an inferred numeric array: an integer beyond ``2**53`` beside a float
stays that integer, and two such integers that differ by one are two levels.  Labels
that Python holds equal, as ``1``, ``1.0`` and ``True``, are one level.  An
interaction is encoded from tuples of the codes of its components, by
first occurrence, without a mixed-radix product or a dense indicator matrix.  Labels
whose exact types decide the outcome of the loop over the rows, primitive labels and
typed integer arrays, and interactions of exact blocks are encoded in bulk instead,
with the same codes and the same first labels (:mod:`powermean.fe.primitive_codes`);
every other label keeps the loop, with its reads, calls and refusals.

Ownership.  Every array of a record is an owned read-only index array in bytes of its
own (:func:`powermean.data.accounting.frozen_index`), from a factory as from a direct
construction, which checks the invariants the record states; the levels are kept in a
:class:`~powermean.data.raw.Column`, which copies a mutable label and hands out fresh
copies.  A later change of a caller's input, or of an object handed out, reaches no
record.  Sorted rows are metadata (:func:`sorted_rows`); the reductions of
:mod:`powermean.fe.reductions` sum the rows in their original order.  Refusals are
built-in ``ValueError`` with a plain message; their translation into the messages of
the interface is interface work.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from powermean.data import raw as _raw
from powermean.data.accounting import frozen_index
from powermean.data.raw import AnyArray, Column, _object_array, _object_column
from powermean.fe import primitive_codes as _primitive_codes
from powermean.fe.primitive_codes import (
    distinct_plain,
    first_rows,
    key_codes,
    plain_codes,
    plain_kind,
    tuple_codes,
)
from powermean.model.order import IndexArray

# The labels of a block: an array, or a sequence of the Python objects given.
Labels = ArrayLike | Sequence[object]
ARRAYS = "FE arrays must be nonempty and one-dimensional"
ALIGNED = "all FE arrays must have the same number of rows"
LABELS = "FE labels must be nonmissing hashable scalars"
MISSING = "FE labels must not be missing"
BLOCKS = "provide at least one FE array and one name per array"


def _index(values: ArrayLike) -> IndexArray:
    """An owned read-only index array of ``values``, which must fit the index type."""
    array = np.asarray(values)
    if array.size and array.max() > np.iinfo(np.intp).max:
        raise ValueError("an index exceeds the range of the index type")
    return frozen_index(np.asarray(array, dtype=np.intp))


def _vectors(arrays: Sequence[ArrayLike], message: str) -> tuple[AnyArray, ...]:
    """The arrays as nonempty one-dimensional integer arrays, or ``ValueError``."""
    found = tuple(np.asarray(array) for array in arrays)
    if any(a.ndim != 1 or not a.size or a.dtype.kind not in "iu" for a in found):
        raise ValueError(message)
    return found


# ---------------------------------------------------------------------- the records


@dataclass(frozen=True, eq=False)
class Block:
    """One encoded block: the levels by first occurrence, the code and count of each.

    Attributes
    ----------
    name : str
        The name of the block, unique within a :class:`Grouping`.
    codes : array of int
        One code per row, ``0 .. n_levels - 1``, the levels numbered in the order of
        their first occurrence; read-only, owned.
    counts : array of int
        The number of rows of each level, at least one; read-only, owned.

    Notes
    -----
    Constructed by :func:`encode_block`, :func:`encode_blocks` and :func:`interaction`.
    A direct construction is checked: a column of distinct nonmissing hashable levels,
    codes numbering every level by first occurrence, and their counts.
    """

    name: str
    codes: IndexArray
    counts: IndexArray
    _levels: Column = field(repr=False)

    def __post_init__(self) -> None:
        """Check the levels, the codes and the counts; store owned index arrays."""
        if not isinstance(self._levels, Column):
            raise ValueError("levels must be a Column of labels")
        n_levels = self._levels.size
        if _distinct(self._levels) != n_levels:
            raise ValueError("levels must be distinct labels")
        (codes,) = _vectors((self.codes,), "codes must be a nonempty integer vector")
        if codes.min() < 0 or codes.max() >= n_levels:
            raise ValueError("group code is outside [0, n_levels)")
        if not _numbered(codes, n_levels):
            raise ValueError("codes must number the levels by first occurrence")
        counts = np.bincount(codes, minlength=n_levels)
        if not np.array_equal(np.asarray(self.counts), counts):
            raise ValueError("counts must be the number of rows of each level")
        object.__setattr__(self, "codes", _index(codes))
        object.__setattr__(self, "counts", _index(counts))

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return int(self.codes.size)

    @property
    def n_levels(self) -> int:
        """The number of levels."""
        return int(self.counts.size)

    @property
    def levels(self) -> tuple[object, ...]:
        """The label of each level, in the order of the codes; fresh copies."""
        if type(self._levels) is Column and type(self._levels._stored) is tuple:
            return cast("tuple[object, ...]", self._levels.stored)
        return tuple(self._levels.values.tolist())


@dataclass(frozen=True, eq=False)
class Grouping:
    """The encoded blocks of an estimation and the original positions of its rows.

    Attributes
    ----------
    blocks : tuple of Block
        The blocks, in the order given, with unique names and the same number of rows;
        an owned tuple, whatever sequence was given.
    positions : array of int
        The exact original position of each row, distinct nonnegative integers,
        read-only and owned: the identity of the rows, which no label replaces.
    """

    blocks: tuple[Block, ...]
    positions: IndexArray

    def __post_init__(self) -> None:
        """Check the blocks and the positions; store them owned."""
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError(BLOCKS)
        if any(not isinstance(block, Block) for block in blocks):
            raise ValueError("blocks must be Block records")
        names = [block.name for block in blocks]
        if len(set(names)) != len(names):
            raise ValueError("FE names must be unique")
        n_rows = blocks[0].n_rows
        if any(block.n_rows != n_rows for block in blocks):
            raise ValueError(ALIGNED)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "positions", _positions(self.positions, n_rows))

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return int(self.positions.size)

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the blocks, in their order."""
        return tuple(block.name for block in self.blocks)


@dataclass(frozen=True, eq=False)
class SortedRows:
    """The rows of one block sorted by code, stably, with the way back.

    Attributes
    ----------
    order : array of int
        ``order[k]`` is the row at sorted position ``k``: the rows of level 0 first, in
        their original order, then those of level 1, and so on.
    inverse : array of int
        ``inverse[row]`` is the sorted position of ``row``; ``order[inverse]`` is the
        identity, and so is ``inverse[order]``.
    segments : array of int
        ``n_levels + 1`` pointers: the rows of level ``l`` are
        ``order[segments[l]:segments[l + 1]]``.

    Notes
    -----
    Metadata on the rows, read-only and owned; the original positions of the rows in
    sorted order are ``grouping.positions[order]``.  Nothing is reordered in place.  A
    direct construction is checked: inverse permutations, increasing pointers from 0
    to the number of rows, and the rows of every segment in their original order.
    The checks compare neighbors and never subtract them, since the subtraction of the
    unsigned arrays that :func:`_vectors` accepts wraps, and they establish the bounds
    of the pointers before a pointer indexes anything.
    """

    order: IndexArray
    inverse: IndexArray
    segments: IndexArray

    def __post_init__(self) -> None:
        """Check the three arrays against each other; store owned index arrays."""
        order, inverse, segments = _vectors(
            (self.order, self.inverse, self.segments),
            "sorted rows must be nonempty integer vectors",
        )
        rows = np.arange(order.size)
        if not (
            np.array_equal(np.sort(order), rows)
            and np.array_equal(np.sort(inverse), rows)
            and np.array_equal(order[inverse], rows)
        ):
            raise ValueError("order and inverse must be inverse permutations")
        # Neighbors are compared, not subtracted: an unsigned difference wraps and is
        # never below zero.  Pointers from 0 to n_rows that increase lie in
        # [0, n_rows], the interior ones in [1, n_rows - 1], so the indexing below is
        # in bounds and the subtraction of one from an interior pointer cannot wrap.
        if (
            segments[0] != 0
            or segments[-1] != rows.size
            or (segments[1:] <= segments[:-1]).any()
        ):
            raise ValueError("segments must be increasing pointers from 0 to n_rows")
        within = np.ones(order.size - 1, dtype=bool)
        within[segments[1:-1] - 1] = False  # the steps from one segment to the next
        if (order[1:] <= order[:-1])[within].any():
            raise ValueError("the rows of a level must keep their original order")
        object.__setattr__(self, "order", _index(order))
        object.__setattr__(self, "inverse", _index(inverse))
        object.__setattr__(self, "segments", _index(segments))


# --------------------------------------------------------------------- the encoding


def _label_array(labels: Labels) -> AnyArray:
    """The labels as a nonempty one-dimensional object array, the objects as given."""
    array: AnyArray = np.asarray(labels, dtype=object)
    if array.ndim != 1 or not len(array):
        raise ValueError(ARRAYS)
    return array


def _is_missing(value: object) -> bool:
    """``None``, or a value unequal to itself: NaN of any type, NaT (the reference).

    ``operator.ne``, since the linter reads a name compared with itself as a typo; an
    array raises ``ValueError`` here and ``pd.NA`` ``TypeError``, refused by the caller.
    """
    return value is None or bool(operator.ne(value, value))


def _encode(elements: Sequence[object]) -> tuple[list[int], list[object]]:
    """Codes and distinct elements by first occurrence; the reference's refusals."""
    lookup: dict[Any, int] = {}
    categories: list[object] = []
    encoded: list[int] = []
    for value in elements:
        try:
            if _is_missing(value):
                raise ValueError(MISSING)
            hash(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(LABELS) from exc
        if value not in lookup:
            lookup[value] = len(categories)
            categories.append(value)
        encoded.append(lookup[value])
    return encoded, categories


def _distinct(levels: Column) -> int:
    """The number of distinct labels of ``levels``, as :func:`_encode` counts them.

    The plain stored labels of an exact ``Column`` are counted by a set
    (:func:`~powermean.fe.primitive_codes.distinct_plain`); any other column is encoded
    from fresh copies, refusing a missing or an unhashable label.
    """
    found = distinct_plain(levels._stored) if type(levels) is Column else None
    return len(_encode(levels.values.tolist())[1]) if found is None else found


def _numbered(codes: AnyArray, n_levels: int) -> bool:
    """Whether codes in ``[0, n_levels)`` number all the levels by first occurrence.

    The first row of each code without a sort where the codes are exact ``intp``
    (:func:`~powermean.fe.primitive_codes.first_rows`), else by ``numpy.unique``: every
    level occurs and the first rows increase with the codes.
    """
    first = first_rows(codes)
    if first is not None:
        return bool(first.size == n_levels)
    found, first = np.unique(codes, return_index=True)
    return bool(found.size == n_levels and np.all(np.diff(first) > 0))


def _block(
    name: str, encoded: Sequence[int] | IndexArray, categories: Sequence[object]
) -> Block:
    """The block of encoded codes and their levels."""
    codes = np.asarray(encoded, dtype=np.intp)
    counts = np.bincount(codes, minlength=len(categories))
    return Block(str(name), codes, counts, _object_column(categories))


def encode_block(labels: Labels, *, name: str = "fe0") -> Block:
    """Encode one block by the first occurrence of its labels.

    Parameters
    ----------
    labels : array_like
        One label per row, nonempty and one-dimensional, read as the objects given;
        ``ValueError`` otherwise, or for a missing or an unhashable label.
    name : str, default "fe0"
        The name of the block.

    Notes
    -----
    A nonempty one-dimensional integer array, whose labels are its values as Python
    integers, is encoded by :func:`~powermean.fe.primitive_codes.key_codes`, its
    levels the values of the first rows; labels of one plain kind by
    :func:`~powermean.fe.primitive_codes.plain_codes`; any other labels by the loop.
    """
    if (
        type(labels) is np.ndarray
        and labels.ndim == 1
        and labels.size > 0
        and labels.dtype.kind in "iu"
    ):
        codes, first = key_codes((labels,))
        return _block(name, codes, labels[first].tolist())
    elements = _label_array(labels).tolist()
    if plain_kind(elements) is not None:
        return _block(name, *plain_codes(elements))
    return _block(name, *_encode(elements))


def _positions(positions: ArrayLike | None, n_rows: int) -> IndexArray:
    """The original positions, distinct and nonnegative; ``None`` is the row order."""
    if positions is None:
        return _index(np.arange(n_rows))
    array = np.asarray(positions)
    if array.dtype.kind not in "iu" or array.shape != (n_rows,):
        raise ValueError("positions must be an integer vector with one value per row")
    if array.min() < 0:
        raise ValueError("positions must be nonnegative")
    if np.unique(array).size != array.size:
        raise ValueError("positions must be distinct: one original position per row")
    return _index(array)


def encode_blocks(
    groups: Mapping[str, Labels] | Sequence[Labels],
    *,
    names: Sequence[str] | None = None,
    positions: ArrayLike | None = None,
) -> Grouping:
    """Encode the blocks of an estimation, in their order, with the rows' positions.

    Parameters
    ----------
    groups : mapping of str to array_like, or sequence of array_like
        One array of labels per block, aligned on the rows: a mapping of names to
        arrays, or a sequence of arrays named by ``names``, or ``fe0, fe1, ...``.
    names : sequence of str, optional
        The names of the blocks of a sequence; not with a mapping.
    positions : array_like, optional
        The exact original position of each row, distinct nonnegative integers; the
        row order ``0 .. n - 1`` if omitted.

    ``ValueError`` for no block, a name per array missing, repeated names, arrays of
    unequal length, an invalid array, label or position.
    """
    if isinstance(groups, Mapping):
        if names is not None:
            raise ValueError("names must not be supplied with a group mapping")
        group_names, values = tuple(map(str, groups)), tuple(groups.values())
    else:
        values = tuple(groups)
        group_names = (
            tuple(names)
            if names is not None
            else tuple(f"fe{j}" for j in range(len(values)))
        )
    if not values or len(group_names) != len(values):
        raise ValueError(BLOCKS)
    if len(set(group_names)) != len(group_names):
        raise ValueError("FE names must be unique")
    arrays = [_label_array(labels) for labels in values]
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError(ALIGNED)
    blocks = tuple(
        _block(name, *_encode(array.tolist()))
        for name, array in zip(group_names, arrays, strict=True)
    )
    return Grouping(blocks, _positions(positions, len(arrays[0])))


def interaction(components: Sequence[Block], *, name: str | None = None) -> Block:
    """Encode the interaction of blocks from tuples of their codes, by first occurrence.

    Parameters
    ----------
    components : sequence of Block
        At least two blocks on the same rows, in the order of the interaction;
        ``ValueError`` otherwise.  The levels of the result are tuples of the levels
        of the components, in that order, numbered by first occurrence.
    name : str, optional
        The name of the interaction; the names of the components joined by ``^`` if
        omitted.

    Notes
    -----
    Components of the exact type ``Block`` whose codes are ``intp`` vectors within their
    levels are encoded in bulk (:func:`~powermean.fe.primitive_codes.tuple_codes`), with
    the codes and levels of the loop; any other components take the loop.
    """
    blocks = tuple(components)
    if len(blocks) < 2:
        raise ValueError("an interaction needs at least two blocks")
    if any(block.n_rows != blocks[0].n_rows for block in blocks):
        raise ValueError(ALIGNED)
    levels = [block.levels for block in blocks]
    if all(type(block) is Block for block in blocks):
        bulk = tuple_codes([block.codes for block in blocks], levels)
        if bulk is not None:
            return _block(_joined(blocks, name), *bulk)
    lookup: dict[tuple[int, ...], int] = {}
    categories: list[object] = []
    encoded: list[int] = []
    for key in zip(*(block.codes.tolist() for block in blocks), strict=True):
        code = lookup.get(key)
        if code is None:
            code = len(categories)
            lookup[key] = code
            categories.append(
                tuple(labels[k] for labels, k in zip(levels, key, strict=True))
            )
        encoded.append(code)
    return _block(_joined(blocks, name), encoded, categories)


def _joined(blocks: tuple[Block, ...], name: str | None) -> str:
    """An interaction's name: ``name``, else its components' names joined by ``^``."""
    return "^".join(block.name for block in blocks) if name is None else name


def sorted_rows(block: Block) -> SortedRows:
    """The stable order of the rows of ``block`` by code, its inverse and its segments.

    Read-only and owned; the rows of each level keep their original order within it.
    """
    order = np.argsort(block.codes, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    segments = np.concatenate(([0], np.cumsum(block.counts)))
    return SortedRows(order, inverse, segments)


def _owned_block(
    name: str, codes: IndexArray, categories: Sequence[object],
) -> Block:
    """Compose one actual native encoding, with its first and only counts.

    Only _owned_derived calls this after native immutable input admission and
    original-producer checks. The encoder already established first occurrence,
    nonmissing distinct levels and exact in-range codes; public Block and _block
    keep all their ordinary construction/ownership checks.
    """
    result = object.__new__(Block)
    for key, value in (
        ("name", name), ("codes", frozen_index(codes)),
        ("counts", frozen_index(np.bincount(codes, minlength=len(categories)))),
        ("_levels", _object_column(categories)),
    ):
        object.__setattr__(result, key, value)
    return result


def _owned_interaction(blocks: Sequence[Block], name: str) -> Block | None:
    """The existing exact interaction encoder on freshly produced native blocks."""
    made = tuple_codes(
        [block.codes for block in blocks],
        [block._levels._stored for block in blocks],
    )
    return None if made is None else _owned_block(name, *made)


def _owned_encoding_original() -> bool:
    """Changed omitted constructors/helpers retain the checked encoding route."""
    return (
        all(globals()[name] is value for name, value in _OWNED_ENCODING_GLOBALS)
        and all(vars(owner).get(name) is value
                for owner, name, value in _OWNED_ENCODING_METHODS)
        and _primitive_codes.key_codes is key_codes
        and _raw._owned_objects is _OWNED_OBJECTS
        and _raw._flat_immutable_tuples is _FLAT_IMMUTABLE
        and _raw._IMMUTABLE_TYPES is _IMMUTABLE_TYPES
    )


_OWNED_ENCODING_GLOBALS = tuple(
    (name, globals()[name]) for name in (
        "Block", "Column", "_block", "_index", "_vectors", "_distinct",
        "_numbered", "_positions", "_object_column", "frozen_index", "key_codes",
        "tuple_codes",
        "plain_codes", "plain_kind", "distinct_plain", "first_rows",
        "_owned_block", "_owned_interaction", "_owned_encoding_original",
    )
)
_OWNED_ENCODING_METHODS = tuple(
    (owner, name, vars(owner).get(name)) for owner, names in (
        (Block, ("__new__", "__init__", "__post_init__", "__getattribute__",
                 "__getattr__", "name", "codes", "counts", "_levels",
                 "n_rows", "n_levels", "levels")),
        (Column, ("__getattribute__", "__getattr__", "stored", "values",
                  "size", "dtype", "take", "_stored")),
    ) for name in names
)
_OWNED_OBJECTS = _raw._owned_objects
_FLAT_IMMUTABLE = _raw._flat_immutable_tuples
_IMMUTABLE_TYPES = _raw._IMMUTABLE_TYPES
