"""The sources of the blocks of the ordered sample and the comparison of two groupings.

Internal module: no pandas, Formulaic or ``solve``.  :func:`same_bits` is the comparison
of two arrays by shape and bytes that this module and
:mod:`powermean.data.sample_fe_state` use; that module still provides every name defined
here, the very same object, and this one imports neither it nor a later stage.

Sources of the blocks.  The labels of the blocks are owned copies
(:class:`~powermean.data.raw.Column`): each access hands out fresh copies, so a label
whose equality is its identity never equals another reading of itself.  Such labels are
admitted by the grouping and stay admitted here through their actual construction
instead.  A :class:`Source` keeps the owned label columns of every component, the very
records of the prepared effects, and the rows it encodes; :class:`Derived` blocks are
the encoding of those columns on some of its rows, with the source level of each level,
checked against the source's own codes.  Two groupings are one level map
(:func:`same_blocks`) when they are the same record, when they derive from one source on
the same rows with the same source levels, or when their labels are equal as the
encoding equates them; never by identity, bytes, fingerprint or string of a label.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.data import fe_component_storage as _storage
from powermean.data.accounting import frozen_index
from powermean.data.fe_component_storage import ComponentRun, matching_runs, runs_for
from powermean.data.raw import Column, _object_array, _OwnedMapping
from powermean.data.retained import _exact_index, _inverse
from powermean.fe.groups import (
    Block,
    Grouping,
    _block,
    _owned_block,
    _owned_encoding_original,
    _owned_interaction,
    encode_block,
    interaction,
)
from powermean.fe.primitive_codes import key_codes, plain_codes, plain_kind
from powermean.model.order import IndexArray

type KeyEncoder = Callable[
    [Sequence[NDArray[np.integer[Any]]]], tuple[IndexArray, IndexArray]
]


def same_bits(a: NDArray[Any], b: NDArray[Any]) -> bool:
    """Whether two arrays have one shape and the same bytes."""
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------- the blocks and their source


def _encoded(
    terms: tuple[tuple[str, ...], ...], columns: Mapping[str, Column], rows: IndexArray
) -> tuple[Block, ...]:
    """The encoding of every term on ``rows``: one name, or ``a^b``.

    Each component by :func:`~powermean.fe.groups.encode_block` on its owned labels of
    these rows, an interaction by :func:`~powermean.fe.groups.interaction`; codes by
    first occurrence on these rows.  Every term reads each of its components afresh from
    the live column, where ``take`` would read it (:func:`_component`): a caller's label
    callback, run while an earlier component or term is encoded, may have replaced a
    column, so no read serves another term.
    """
    blocks: list[Block] = []
    for term in terms:
        parts = [_component(columns[c], rows, c) for c in term]
        name = "^".join(term)
        blocks.append(parts[0] if len(parts) == 1 else interaction(parts, name=name))
    return tuple(blocks)


def _component(column: Column, rows: IndexArray, name: str) -> Block:
    """``encode_block(column.take(rows).values, name=name)``: the same block.

    Built from the codes and first labels that :func:`_stored_codes` reads from the
    column's storage at this call; any other column takes ``take``, ``values`` and
    ``encode_block``.
    """
    found = _stored_codes(column, rows)
    if found is None:
        return encode_block(column.take(rows).values, name=name)
    return _block(name, *found)


def _stored_codes(
    column: Column, rows: IndexArray
) -> tuple[IndexArray, list[object]] | None:
    """The codes and first labels of ``column`` on ``rows``, or ``None``.

    Read from the storage of an exact ``Column``: a tuple whose labels on these rows are
    of one plain kind (:func:`~powermean.fe.primitive_codes.plain_kind`), the very
    objects that ``take`` and ``values`` hand out, as they are their own copies; or a
    typed integer vector, whose labels are its values.  Encoded as
    :func:`~powermean.fe.groups.encode_block` encodes those labels.  ``None`` for no
    row and any other column.
    """
    if type(column) is not Column or not rows.size:
        return None
    labels = column._stored
    if type(labels) is tuple:
        items = list(map(labels.__getitem__, rows.tolist()))
        return plain_codes(items) if plain_kind(items) is not None else None
    if (
        type(labels) is not np.ndarray
        or labels.ndim != 1
        or labels.dtype.kind not in "iu"
    ):
        return None
    values = labels[rows]
    codes, first = key_codes((values,))
    return codes, values[first].tolist()


@dataclass(frozen=True, eq=False, init=False)
class Source:
    """The owned labels of every block and the rows they are encoded on: a source.

    ``terms`` are the fixed-effect terms in formula order, a name or the two names of
    ``a^b``; ``columns`` the owned label :class:`~powermean.data.raw.Column` of each
    component, the prepared effects' own records, kept by reference; ``nobs`` the
    number of input rows; ``rows`` the rows encoded, exact distinct indices below
    ``nobs``; ``blocks`` their accepted encoding on ``rows``, derived, never an
    argument.  ``ValueError`` for another term, a column missing, extra or of other
    rows, or rows that are not exact distinct indices.
    """

    terms: tuple[tuple[str, ...], ...]
    columns: Mapping[str, Column]
    nobs: int
    rows: IndexArray
    blocks: tuple[Block, ...] = field(init=False, repr=False)
    _component_runs: dict[str, ComponentRun] | None = field(init=False, repr=False)

    def __init__(
        self,
        terms: Sequence[Sequence[str]],
        columns: Mapping[str, Column],
        nobs: int,
        rows: ArrayLike,
    ) -> None:
        """Check the terms, the owned columns and the rows; encode the blocks."""
        found = tuple(tuple(term) for term in terms)
        if not found or any(len(term) not in (1, 2) for term in found):
            raise ValueError("a fixed-effect term is one name or an interaction of two")
        names = tuple(dict.fromkeys(name for term in found for name in term))
        exact = isinstance(nobs, int) and not isinstance(nobs, bool)
        owned = exact and set(columns) == set(names)
        if not owned or not all(
            isinstance(columns[name], Column) and columns[name].size == nobs
            for name in names
        ):
            raise ValueError("provide exactly one column per component of the terms")
        encoded = _exact_index(rows, "rows of the blocks", upper=nobs)
        if np.unique(encoded).size != encoded.size:
            raise ValueError("the rows of the blocks must be distinct")
        kept = MappingProxyType({name: columns[name] for name in names})
        object.__setattr__(self, "terms", found)
        object.__setattr__(self, "columns", kept)
        object.__setattr__(self, "nobs", nobs)
        object.__setattr__(self, "rows", encoded)
        runs = runs_for(kept)
        blocks = _encoded(found, kept, encoded)
        object.__setattr__(self, "blocks", blocks)
        # These are the complete original columns used by this first encoding,
        # not level-code compression or a remembered eligibility assertion.
        object.__setattr__(self, "_component_runs",
                           runs if matching_runs(runs, kept) else None)

    def __getattribute__(self, name: str) -> Any:
        """Keep original stable logical fields without forcing numeric readers."""
        if type(name) is str and name in ("columns", "_component_runs"):
            state = object.__getattribute__(self, "__dict__")
            if "_native_columns" in state:
                if name == "columns" and state.get("columns") is None:
                    columns = MappingProxyType(_storage.object_columns(
                        state["_native_columns"], state["_native_runs"],
                    ))
                    object.__setattr__(self, "columns", columns)
                    object.__setattr__(self, "_object_columns", columns)
                elif name == "_component_runs" and state.get("_component_runs") is None:
                    object.__setattr__(self, "_component_runs", runs_for(self.columns))
        return object.__getattribute__(self, name)

    def __getstate__(self) -> Any:
        """Preserve ordinary state; copies of native records expose logical fields."""
        state = object.__getstate__(self)
        if type(self) is not _SOURCE_RECORD or "_native_columns" not in vars(self):
            return state
        state = dict(vars(self), columns=self.columns, _component_runs=self._component_runs)
        return {name: value for name, value in state.items()
                if name not in ("_native_columns", "_native_runs", "_object_columns")}


def _native_source_parts(
    source: Source,
) -> tuple[Mapping[str, Column], dict[str, ComponentRun]] | None:
    """The actual internal native owner, not a public field or a trust assertion."""
    if type(source) is not Source or not _owned_source_original():
        return None
    state = vars(source)
    columns, runs = state.get("_native_columns"), state.get("_native_runs")
    if (type(columns) is not MappingProxyType
            or not matching_runs(runs, columns)):
        return None
    shown = state.get("columns")
    if shown is not None:
        if shown is not state.get("_object_columns"):
            return None
        if shown.keys() != columns.keys() or any(
            shown[name] is not runs[name].object_column() for name in columns
        ):
            return None
    elif "_object_columns" in state:
        return None
    return columns, runs


def _ordinals(block: Block, source: Block, at: IndexArray) -> IndexArray:
    """The source level of each level of ``block``, whose rows are ``at`` in ``source``.

    The source codes of those rows recoded by first occurrence (the accepted
    :func:`~powermean.fe.groups.encode_block`, as :func:`restrict` recodes); they must
    be the codes of ``block`` bit for bit: the checked relation of a derived block.
    """
    made = encode_block(source.codes[at], name=block.name)
    if not same_bits(made.codes, block.codes):
        raise ValueError("derived blocks must be their source's levels on their rows")
    return frozen_index(np.asarray(made.levels, dtype=np.intp))


@dataclass(frozen=True, eq=False, init=False)
class Derived(Grouping):
    """The blocks of a :class:`Source` on some of its rows, with their source levels.

    ``source`` is the source; ``positions`` exact distinct rows of it; the blocks are
    the encoding of the source's owned columns on those rows (codes by first occurrence
    there, each level the label of its first row, as the source's own encoding of these
    rows would give); ``ordinals`` holds per block the source level of each level
    (:func:`_ordinals`).  Blocks and ordinals are derived, never arguments, also by
    ``dataclasses.replace``.  ``ValueError`` for rows that are not the source's or an
    encoding that is not the source's partition there.
    """

    blocks: tuple[Block, ...] = field(init=False)
    source: Source
    ordinals: tuple[IndexArray, ...] = field(init=False)

    def __init__(self, source: Source, positions: ArrayLike) -> None:
        """Derive the blocks of ``source`` on ``positions`` and their source levels."""
        if not isinstance(source, Source):
            raise ValueError("derived blocks need their Source")
        rows = _exact_index(positions, "rows of the blocks", upper=source.nobs)
        place = np.full(source.nobs, -1, dtype=np.intp)
        place[source.rows] = np.arange(source.rows.size, dtype=np.intp)
        at = place[rows]
        if (at < 0).any() or np.unique(rows).size != rows.size:
            raise ValueError("derived blocks are on distinct rows of their source")
        blocks = source.blocks
        ordinals = tuple(
            frozen_index(np.arange(b.n_levels, dtype=np.intp)) for b in blocks
        )
        if not same_bits(rows, source.rows):
            native = (_native_source_parts(source)
                      if type(source) is Source
                      and "_native_columns" in object.__getattribute__(source, "__dict__")
                      and _native_source_parts is _NATIVE_SOURCE_PARTS else None)
            blocks = None
            if native is not None:
                try:
                    blocks = _owned_encoded(
                        source.terms, native[0], rows, runs=native[1],
                    )
                except MemoryError:
                    pass
            if blocks is None:
                blocks = _encoded(source.terms, source.columns, rows)
            ordinals = tuple(
                _ordinals(block, origin, at)
                for block, origin in zip(blocks, source.blocks, strict=True)
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "positions", rows)
        object.__setattr__(self, "ordinals", ordinals)
        super().__post_init__()


def _one_source(a: Grouping, b: Grouping) -> bool:
    """Whether two groupings derive from one source with the same source levels.

    The same :class:`Source`, or sources of the very same owned label columns, terms,
    input rows and encoded rows, and equal ordinals: then the labels of each level are
    copies of one owned label.  Never a comparison of the labels themselves.
    """
    if not isinstance(a, Derived) or not isinstance(b, Derived):
        return False
    s, t = a.source, b.source
    same = s is t or (
        s.terms == t.terms
        and s.nobs == t.nobs
        and same_bits(s.rows, t.rows)
        and all(s.columns[name] is t.columns[name] for name in s.columns)
    )
    return same and all(map(same_bits, a.ordinals, b.ordinals))


def restrict(source: Grouping, positions: IndexArray, nobs: int) -> Grouping:
    """The blocks of ``source`` on its rows ``positions``, of ``nobs`` input rows.

    A :class:`Derived` source gives the :class:`Derived` blocks of its own source on
    those rows: its checked relation.  Another ``source`` is on increasing rows; each
    row's level there is recoded by first occurrence by
    :func:`~powermean.fe.groups.encode_block`, and each block keeps the source's labels
    of those levels in that order: the original level map.
    """
    if isinstance(source, Derived):
        return Derived(source.source, positions)
    rows = _inverse(source.positions, nobs)[positions]
    blocks = []
    for block in source.blocks:
        made = encode_block(block.codes[rows], name=block.name)
        labels = block.levels
        chosen = [labels[code] for code in np.asarray(made.levels).tolist()]
        blocks.append(
            Block(made.name, made.codes, made.counts, Column(_object_array(chosen)))
        )
    return Grouping(tuple(blocks), positions)


def same_blocks(a: Grouping | None, b: Grouping | None) -> bool:
    """The same blocks of the same rows, their original level labels included.

    Names in formula order, positions, codes and counts bit for bit, and one level map:
    the same record, grouping or block (one owned map is itself); blocks derived from
    one source with the same source levels (:class:`Derived`); or each block's ordered
    level labels equal as the encoding equates labels: Python equality of the hashable
    labels, an interaction's tuples element by element (``1``, ``1.0`` and ``True`` are
    one level).  No label identity, bytes, fingerprint or string coercion; independent
    copies of a label equal only by identity are not one.
    """
    if a is None or b is None:
        return a is None and b is None
    same = a.names == b.names and same_bits(a.positions, b.positions)
    same = same and all(
        same_bits(x.codes, y.codes) and same_bits(x.counts, y.counts)
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )
    if not same:
        return False
    if a is b or _one_source(a, b):
        return True
    return all(
        x is y or x.levels == y.levels for x, y in zip(a.blocks, b.blocks, strict=True)
    )


def _owned_encoded(
    terms: tuple[tuple[str, ...], ...], columns: Mapping[str, Column], rows: IndexArray,
    *, runs: dict[str, ComponentRun] | None = None,
    encoder: KeyEncoder | None = None,
) -> tuple[Block, ...] | None:
    """Native encodings once, retaining generic reads for every unknown label."""
    if runs is not None:
        return _native_encoded(terms, columns, rows, runs, encoder=encoder)
    blocks: list[Block] = []
    for term in terms:
        parts = []
        for name in term:
            made = _stored_codes(columns[name], rows)
            if made is None:
                return None
            parts.append(_owned_block(name, *made))
        block = (parts[0] if len(parts) == 1
                 else _owned_interaction(parts, "^".join(term)))
        if block is None:
            return None
        blocks.append(block)
    return tuple(blocks)


def _native_encoded(
    terms: tuple[tuple[str, ...], ...], columns: Mapping[str, Column],
    rows: IndexArray, runs: dict[str, ComponentRun],
    *, encoder: KeyEncoder | None = None,
) -> tuple[Block, ...] | None:
    """Encode once per integral component and term in this actual owned sample.

    The interface may supply its exact hash encoder after native-owner admission.
    Codes/counts and actual first-row labels are still the sole shared source for
    closure, numeric solving and retained views; no encoder or live owner escapes.
    """
    if not rows.size:
        return None
    uses = {name: sum(term.count(name) for term in terms) for name in runs}
    held: dict[str, Block] = {}
    before = {name: run._presented for name, run in runs.items()}
    blocks: list[Block] = []
    try:
        for term in terms:
            parts = []
            for name in term:
                part = held.get(name)
                if part is None:
                    values = cast("np.ndarray", columns[name]._stored)[rows]
                    codes, first = (key_codes if encoder is None else encoder)((values,))
                    part = _owned_block(name, codes, runs[name].labels(rows[first]))
                uses[name] -= 1
                if uses[name]:
                    held[name] = part
                else:
                    held.pop(name, None)
                parts.append(part)
            if len(parts) == 1:
                block = parts[0]
            elif encoder is None:
                block = _owned_interaction(parts, "^".join(term))
            else:
                codes, first = encoder([part.codes for part in parts])
                labels = [part._levels._stored for part in parts]
                heads = [part.codes[first].tolist() for part in parts]
                places = [[values[k] for k in head]
                          for values, head in zip(labels, heads, strict=True)]
                block = _owned_block("^".join(term), codes,
                                     list(zip(*places, strict=True)))
            if block is None:
                break
            blocks.append(block)
        else:
            return tuple(blocks)
    except MemoryError:
        for name, run in runs.items():
            object.__setattr__(run, "_presented", before[name])
        raise
    for name, run in runs.items():
        object.__setattr__(run, "_presented", before[name])
    return None


def _owned_derived(
    terms: tuple[tuple[str, ...], ...], columns: Mapping[str, Column],
    nobs: int, positions: IndexArray, *, _runs: dict[str, ComponentRun] | None = None,
    _encoder: KeyEncoder | None = None,
) -> Derived | None:
    """One actual owned first encoding; Source/Derived public admission stays intact.

    The selected rows are checked once before encoding. Original native encoders
    establish the block invariants; the identical initial derived rows need no
    inverse/gather, second/third uniqueness sort or second block construction.
    Fresh headers share only immutable index bytes. No caller-supplied proof or
    remembered eligibility is consumed.
    """
    if not _owned_source_original() or not _owned_input(terms, columns, nobs):
        return None
    if type(positions) is not np.ndarray or positions.dtype != np.intp:
        return None
    items = cast("_OwnedMapping[Column]", columns)._items
    runs = runs_for(items) if _runs is None else _runs
    if not matching_runs(runs, items):
        return None
    rows = _exact_index(positions, "rows of the blocks", upper=nobs)
    if not bool(np.all(rows[1:] > rows[:-1])) and np.unique(rows).size != rows.size:
        raise ValueError("the rows of the blocks must be distinct")
    blocks = (_owned_encoded(terms, items, rows) if _runs is None
              else _owned_encoded(terms, items, rows, runs=runs, encoder=_encoder))
    if blocks is None:
        return None
    origin = object.__new__(Source)
    for name, value in (
        ("terms", terms), ("columns", MappingProxyType({
            name: items[name] for name in dict.fromkeys(
                name for term in terms for name in term
            )
        })),
        ("nobs", nobs), ("rows", rows), ("blocks", blocks), ("_component_runs", runs),
    ):
        object.__setattr__(origin, name, value)
    if _runs is not None:
        object.__setattr__(origin, "columns", None)
        object.__setattr__(origin, "_component_runs", None)
        object.__setattr__(origin, "_native_columns", MappingProxyType(dict(items)))
        object.__setattr__(origin, "_native_runs", runs)
    result = object.__new__(Derived)
    for name, value in (
        ("source", origin), ("blocks", blocks), ("positions", rows.view()),
        ("ordinals", tuple(frozen_index(np.arange(b.n_levels, dtype=np.intp))
                           for b in blocks)),
    ):
        object.__setattr__(result, name, value)
    return result


def _owned_input(
    terms: tuple[tuple[str, ...], ...], columns: Mapping[str, Column], nobs: int,
) -> bool:
    """Admit only actual exact containers; unknown/stale fields keep old checks."""
    if (type(terms) is not tuple or not terms or type(nobs) is not int
            or type(columns) is not _OwnedMapping or type(columns._items) is not dict):
        return False
    if any(type(term) is not tuple or len(term) not in (1, 2)
           or any(type(name) is not str for name in term) for term in terms):
        return False
    items = cast("_OwnedMapping[Column]", columns)._items
    if any(type(name) is not str for name in items):
        return False
    names = tuple(dict.fromkeys(name for term in terms for name in term))
    return (
        set(items) == set(names)
        and len({"^".join(term) for term in terms}) == len(terms)
        and all(type(column) is Column and column.size == nobs
                for column in items.values())
    )


def _owned_source_original() -> bool:
    """One existing-style identity boundary, not per-row reauthentication."""
    return (
        all(globals()[name] is value for name, value in _OWNED_SOURCE_GLOBALS)
        and all(vars(owner).get(name) is value
                for owner, name, value in _OWNED_SOURCE_METHODS)
        and _storage.original_producers is _STORAGE_ORIGINAL
        and _storage.original_producers()
        and _owned_encoding_original()
    )


_OWNED_SOURCE_GLOBALS = tuple(
    (name, globals()[name]) for name in (
        "Source", "Derived", "Grouping", "Column", "_OwnedMapping", "MappingProxyType",
        "_encoded",
        "_component", "_stored_codes", "_ordinals", "same_bits", "_exact_index",
        "_inverse", "frozen_index", "_block", "encode_block", "interaction",
        "plain_codes", "plain_kind", "key_codes", "runs_for", "matching_runs",
        "_owned_block", "_owned_interaction", "_owned_encoding_original",
        "_owned_encoded", "_owned_derived", "_owned_input", "_owned_source_original",
        "_native_encoded", "_native_source_parts",
    )
)
_OWNED_SOURCE_METHODS = tuple(
    (owner, name, vars(owner).get(name)) for owner, names in (
        (Source, ("__new__", "__init__", "__getattribute__", "__getattr__", "__getstate__",
                  "terms", "columns", "nobs", "rows", "blocks", "_component_runs")),
        (Derived, ("__new__", "__init__", "__getattribute__", "__getattr__",
                   "source", "blocks", "positions", "ordinals")),
        (Grouping, ("__new__", "__init__", "__post_init__",
                    "__getattribute__", "__getattr__", "blocks", "positions")),
        (_OwnedMapping, ("__getattribute__", "__getattr__", "__iter__",
                         "__getitem__", "__len__")),
    ) for name in names
)
_STORAGE_ORIGINAL = _storage.original_producers
_NATIVE_SOURCE_PARTS = _native_source_parts
_SOURCE_RECORD = Source
