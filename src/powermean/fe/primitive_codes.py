"""First-occurrence codes of exact primitive labels and of integer keys, in bulk.

:mod:`powermean.fe.groups` encodes a
block by the first occurrence of its labels with a loop over the rows (``_encode``),
which reads every label as the object given and refuses a missing or an unhashable one.
Where the exact types of the labels decide that loop's outcome, the functions here give
the same codes and the same first labels with built-in dictionary and NumPy operations
over the whole vector; every other input stays with the loop, which its callers keep.

Primitive labels (:func:`plain_kind`).  An exact list or tuple whose items all have one
exact type among ``bool``, ``int``, ``float`` and ``str``, compared by identity, so that
no method of an item or of its class runs and a subclass is not its base; no float is
NaN, the missing label of the loop.  Such labels are never missing, always hashable, and
equal by built-in equality: the loop neither refuses them nor calls caller code.  Their
equality is kept as it is (:func:`plain_codes`): no label is converted to a NumPy type,
a float or a fixed-width string, so ``2**64 + 1`` is a level of its own, ``0.0`` and
``-0.0`` are one level whose label is the first of them, and a string and that string
with a trailing NUL character are two.  Tuples of such labels, one plain kind at each
place, are the levels of an interaction (:func:`plain_tuples`); a set counts either
(:func:`distinct_plain`), as the loop counts the levels of a block.

Integer keys (:func:`key_codes`).  The rows of one or more integer vectors (the labels
of a typed integer array, the source codes of a derived block, the codes of the
components of an interaction) are grouped by a stable lexicographic sort, which keeps
the rows of each distinct tuple in row order, so the first of each run is its first
row; the runs are numbered by that row.  No mixed-radix product is formed, so no number
of levels can overflow, and no indicator matrix.  An interaction's levels are then the
tuples of its components' labels at those first rows (:func:`tuple_codes`).

First rows (:func:`first_rows`).  Codes numbered by first occurrence have the first row
of each code where their running maximum rises: ``numpy.unique(codes,
return_index=True)[1]`` found without a sort, for exact ``intp`` vectors that are so
numbered, and ``None`` for any other value, whose caller keeps the sort.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from itertools import repeat
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from powermean.model.order import IndexArray

# The exact types of primitive labels; a subclass, as NumPy's ``str_``, is not one.
PLAIN_LABELS: tuple[type, ...] = (bool, int, float, str)


def plain_kind(items: Sequence[object]) -> type | None:
    """The one exact type of :data:`PLAIN_LABELS` of all items, no NaN; else ``None``.

    Only an exact nonempty ``list`` or ``tuple``, whose iteration and length are not a
    caller's; types are read with ``type`` and compared by identity.
    """
    if (type(items) is not list and type(items) is not tuple) or not items:
        return None
    kind = type(items[0])
    if not any(kind is t for t in PLAIN_LABELS):
        return None
    if not all(map(operator.is_, map(type, items), repeat(kind))):
        return None
    if kind is float and any(map(math.isnan, cast("Sequence[float]", items))):
        return None
    return kind


def plain_tuples(items: Sequence[object]) -> bool:
    """Whether the items are exact tuples of one nonzero length, one plain kind a place.

    Only an exact nonempty ``list`` or ``tuple`` of exact ``tuple`` items; each place,
    the items' labels at that position, has a :func:`plain_kind`.
    """
    if (type(items) is not list and type(items) is not tuple) or not items:
        return False
    if not all(map(operator.is_, map(type, items), repeat(tuple))):
        return False
    found = cast("Sequence[tuple[object, ...]]", items)
    if len(set(map(len, found))) != 1 or not found[0]:
        return False
    return all(plain_kind(place) is not None for place in zip(*found, strict=True))


def distinct_plain(items: object) -> int | None:
    """The number of distinct items of a tuple of plain labels or tuples, or ``None``.

    Items of one :func:`plain_kind`, or :func:`plain_tuples`, are counted by a set, by
    their own hash and equality; ``None`` for any other value.
    """
    if type(items) is not tuple or (
        plain_kind(items) is None and not plain_tuples(items)
    ):
        return None
    return len(set(cast("tuple[object, ...]", items)))


def plain_codes(items: Sequence[object]) -> tuple[IndexArray, list[object]]:
    """Codes and first labels by first occurrence of labels of one :func:`plain_kind`.

    ``dict.fromkeys`` keeps the first of equal labels, with its type, in the order of
    the rows; each row's code is the place of its label there.  The labels' own hash
    and equality decide, as in the loop of :mod:`powermean.fe.groups`.
    """
    levels = list(dict.fromkeys(items))
    place = dict(zip(levels, range(len(levels)), strict=True))
    codes = np.fromiter(map(place.__getitem__, items), dtype=np.intp, count=len(items))
    return codes, levels


def key_codes(
    keys: Sequence[NDArray[np.integer[Any]]],
) -> tuple[IndexArray, IndexArray]:
    """Codes by first occurrence of the rows' tuples of keys, and each code's first row.

    ``keys`` are nonempty one-dimensional integer vectors of one length.  The stable
    ``numpy.lexsort`` puts equal tuples together in row order; a run starts where any
    key changes, its first row is its first element, and runs are numbered by that row.
    """
    order = np.lexsort(tuple(keys))
    starts = np.zeros(order.size, dtype=np.bool_)
    starts[0] = True
    for key in keys:
        ranked = key[order]
        starts[1:] |= ranked[1:] != ranked[:-1]
    heads = order[starts]
    by_row = np.argsort(heads)
    rank = np.empty(heads.size, dtype=np.intp)
    rank[by_row] = np.arange(heads.size, dtype=np.intp)
    codes = np.empty(order.size, dtype=np.intp)
    codes[order] = rank[np.cumsum(starts) - 1]
    return codes, heads[by_row]


def tuple_codes(
    keys: Sequence[object], levels: Sequence[Sequence[object]]
) -> tuple[IndexArray, list[tuple[object, ...]]] | None:
    """The codes and levels of the rows' tuples of component codes, or ``None``.

    ``keys`` are the components' codes and ``levels`` their labels.  For exact nonempty
    one-dimensional ``intp`` keys within their levels: the codes of the tuples by first
    occurrence (:func:`key_codes`), and each level the tuple of the components' labels
    at its first row, as a loop over the rows would form it there.  ``None`` otherwise.
    """
    if not all(
        type(key) is np.ndarray
        and key.dtype == np.intp
        and key.ndim == 1
        and key.size > 0
        and int(key.min()) >= 0
        and int(key.max()) < len(labels)
        for key, labels in zip(keys, levels, strict=True)
    ):
        return None
    vectors = cast("list[IndexArray]", list(keys))
    codes, first = key_codes(vectors)
    heads = [key[first].tolist() for key in vectors]
    places = [
        [labels[k] for k in head] for labels, head in zip(levels, heads, strict=True)
    ]
    return codes, list(zip(*places, strict=True))


def first_rows(codes: object) -> IndexArray | None:
    """``numpy.unique(codes, return_index=True)[1]`` of first-occurrence codes, or None.

    Only an exact nonempty one-dimensional ``intp`` array with no code below zero, whose
    codes where the running maximum rises are ``0, 1, 2, ...``: then every code up to
    the maximum occurs, first at the row where the maximum reaches it, and those rows in
    code order are what the stable sort of ``numpy.unique`` returns.  ``None`` for any
    other value, whose caller sorts.
    """
    if (
        type(codes) is not np.ndarray
        or codes.dtype != np.intp
        or codes.ndim != 1
        or not codes.size
    ):
        return None
    vector = cast("IndexArray", codes)
    rises = np.empty(vector.size, dtype=np.bool_)
    rises[0] = True
    np.greater(vector[1:], np.maximum.accumulate(vector)[:-1], out=rises[1:])
    first = np.flatnonzero(rises)
    if vector.min() < 0 or not np.array_equal(vector[first], np.arange(first.size)):
        return None
    return first
