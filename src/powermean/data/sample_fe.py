"""The fixed-effect steps of the sample: the labels of the blocks and steps 5 and 6.

Internal module: accounting on arrays, integer codes and records below the coordinator
:mod:`powermean.interface.sample`, which alone calls the diagnostics; no pandas,
Formulaic or ``solve``.  The ledger and the completed sample are
:mod:`powermean.data.sample_fe_state`, whose names this module still provides.

Blocks.  :class:`EffectColumns` owns the raw labels of every component of every term
(``a``, or the interaction ``a^b``: one block).  They are encoded by
:mod:`powermean.fe.groups` on given positions, as objects, never joined or coerced to
strings; :meth:`EffectColumns.fixed` encodes them once on the rows of steps 3 and 4 and
maps each input row to its first-occurrence code there, a level's identity on every
later subset (:class:`FixedEffects` admits exactly those codes).  Counts of levels are
not the rank of three or more blocks.  The encoded grouping is derived from its source,
these owned columns on those rows (:class:`~powermean.data.sample_fe_state.Derived`), so
that the blocks of a later subset keep their relation to it.

Step 5 (``reghdfe.mata`` 143-180 and 2683-2767, a convention; no code copied).
:func:`singleton_closure` visits the blocks cyclically in formula order; a visit removes
every row of its block's singleton levels, attributed to that block; ``remaining``
starts at the number of blocks, falls by one after a visit without removal and is reset
to that number minus one after a removal; the closure stops at zero or when no row is
current.  A singleton level has one physical row, whose exact count is one with
frequency weights (a replica total of one; counts are positive integers after step 4).
Weight values never decide.

Step 6.  :func:`level_separation` removes at once every row of every level, of any
block, with no current row of positive RAW outcome (indicators, never sums or a
transformed outcome: at ``omega = 0`` an outcome of one is positive), attributed to the
first block whose level holding it is such a level, and records each block's levels.
:func:`close` alternates steps 5 and 6 to their fixed point; ``keep_singletons`` skips
step 5 only.  Both steps first check that the blocks are encoded on every current row;
their records own exact rows and blocks, and a closure's ledger codes every row it
removed with that removal's first cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from powermean.data import fe_component_storage as _storage
from powermean.data.accounting import frozen_index
from powermean.data.raw import Column, _OwnedMapping
from powermean.data.retained import _exact_index
from powermean.data.sample_fe_source import KeyEncoder, _owned_derived
from powermean.data.sample_fe_state import (
    FE_REASONS,
    FIXED_EFFECT,
    LARGEST,
    NO_BLOCK,
    SEPARATED,
    SINGLETON,
    Derived,
    FESample,
    RowLedger,
    Source,
    final_sample,
    ledger_of,
    replicas_of,
)
from powermean.fe.groups import Grouping
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

# The names of this module and of the state in data/sample_fe_state.py.
__all__ = [
    "FE_REASONS",
    "FIXED_EFFECT",
    "NO_BLOCK",
    "OUTSIDE",
    "SEPARATED",
    "SINGLETON",
    "Closure",
    "ClosurePass",
    "EffectColumns",
    "FESample",
    "FixedEffects",
    "LevelSeparation",
    "RowLedger",
    "Visit",
    "close",
    "final_sample",
    "ledger_of",
    "level_separation",
    "replicas_of",
    "singleton_closure",
]
OUTSIDE = -1  # the code of a row outside the rows that the blocks were encoded on


def _row_codes(grouping: Grouping, rows: int) -> tuple[IndexArray, ...]:
    """Each block's code of every input row: on the grouping's rows, else ``-1``."""
    found = []
    for block in grouping.blocks:
        codes = np.full(rows, OUTSIDE, dtype=np.intp)
        codes[grouping.positions] = block.codes
        found.append(frozen_index(codes))
    return tuple(found)


@dataclass(frozen=True, eq=False)
class FixedEffects:
    """The blocks encoded on the rows of steps 3 and 4, and each input row's code.

    ``codes`` holds per block one code per input row, ``-1`` outside those rows;
    ``ValueError`` unless they are exactly the grouping's on its rows.
    """

    grouping: Grouping
    codes: tuple[IndexArray, ...]

    def __post_init__(self) -> None:
        """Own exact row codes; refuse codes that are not the grouping's."""
        grouping, given = self.grouping, tuple(self.codes)
        if not isinstance(grouping, Grouping) or len(given) != len(grouping.blocks):
            raise ValueError("fixed effects are a Grouping and its row codes")
        found = tuple(
            _exact_index(codes, "row codes", lower=OUTSIDE, upper=LARGEST)
            for codes in given
        )
        rows = found[0].size
        covered = all(codes.size == rows for codes in found)
        if (
            not covered
            or int(grouping.positions.max()) >= rows
            or not all(map(np.array_equal, found, _row_codes(grouping, rows)))
        ):
            raise ValueError("the row codes must be the grouping's on its rows")
        object.__setattr__(self, "codes", found)

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return int(self.codes[0].size)


@dataclass(frozen=True, eq=False, init=False)
class EffectColumns:
    """The raw labels of the fixed-effect terms on every input row, owned.

    ``terms`` are the terms in formula order, one name or the two names of ``a^b``;
    ``columns`` the labels of each component, one per input row, as given (an object
    column keeps copies of its elements, :class:`Column`); ``rows`` the number of input
    rows.  ``ValueError`` for no term, a term of another length, or a column that is
    missing, misaligned or named by no term.

    ``columns`` is the record's :class:`~powermean.data.raw._OwnedMapping`: the exact
    ``dict`` that the constructor formed, carried as the record's current data and
    handed out read-only through that one object.
    """

    terms: tuple[tuple[str, ...], ...]
    columns: Mapping[str, Column]
    rows: int

    def __init__(
        self,
        terms: Sequence[Sequence[str]],
        columns: Mapping[str, ArrayLike | Column],
        rows: int,
    ) -> None:
        """Check the terms and store each component column owned."""
        found = tuple(tuple(term) for term in terms)
        if not found or any(len(term) not in (1, 2) for term in found):
            raise ValueError("a fixed-effect term is one name or an interaction of two")
        names = tuple(dict.fromkeys(name for term in found for name in term))
        if set(columns) != set(names):
            raise ValueError("provide exactly one column per component of the terms")
        stored: dict[str, Column] = {}
        for name in names:
            given = columns[name]
            column = given if isinstance(given, Column) else Column(given, name=name)
            if column.size != rows:
                raise ValueError(f"effects column {name!r} must have one value per row")
            stored[name] = column
        object.__setattr__(self, "terms", found)
        object.__setattr__(self, "columns", _OwnedMapping(stored))
        object.__setattr__(self, "rows", rows)

    @classmethod
    def _from_native(
        cls, terms: tuple[tuple[str, ...], ...],
        columns: Mapping[str, Column], rows: int,
    ) -> EffectColumns | None:
        """Compose the actual integral owner; no public validity assertion."""
        if cls is not EffectColumns or not _fixed_original():
            return None
        native = _storage.native_columns(columns)
        if native is None:
            return None
        found, runs = native
        if (set(found) != {name for term in terms for name in term}
                or any(column.size != rows for column in found.values())):
            return None
        result = object.__new__(cls)
        for name, value in (
            ("terms", terms), ("rows", rows), ("columns", None),
            ("_native_columns", _OwnedMapping(found)), ("_native_runs", runs),
        ):
            object.__setattr__(result, name, value)
        return result

    def __getattribute__(self, name: str) -> Any:
        """The original stable object mapping, materialized only when requested."""
        if type(name) is str and name == "columns":
            state = object.__getattribute__(self, "__dict__")
            if state.get("columns") is None and "_native_columns" in state:
                found = _storage.object_columns(
                    state["_native_columns"], state["_native_runs"],
                )
                object.__setattr__(self, "columns", _OwnedMapping(found))
        return object.__getattribute__(self, name)

    def __getstate__(self) -> Any:
        """Copies retain ordinary state; native backing never escapes as fields."""
        state = object.__getstate__(self)
        if type(self) is not _EFFECT_RECORD or "_native_columns" not in vars(self):
            return state
        state = dict(vars(self), columns=self.columns)
        return {name: value for name, value in state.items()
                if name not in ("_native_columns", "_native_runs")}

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the blocks, in formula order: ``a``, or ``a^b``."""
        return tuple("^".join(term) for term in self.terms)

    def encode(self, positions: IndexArray) -> Grouping:
        """The grouping of every block on the rows ``positions``, in order.

        Each component by :func:`~powermean.fe.groups.encode_block`, an interaction by
        :func:`~powermean.fe.groups.interaction`; codes by first occurrence on these
        rows, their original positions kept.  The grouping is the
        :class:`~powermean.data.sample_fe_state.Derived` blocks of its own
        :class:`~powermean.data.sample_fe_state.Source`, these owned columns on these
        rows, so that blocks derived from it later keep their relation.
        """
        source = Source(self.terms, self.columns, self.rows, positions)
        return Derived(source, positions)

    def fixed(self, positions: IndexArray) -> FixedEffects:
        """:meth:`encode` on ``positions``, with every input row's code (``-1`` off)."""
        grouping = self.encode(positions)
        return FixedEffects(grouping, _row_codes(grouping, self.rows))


# ------------------------------------------------------------------ steps 5 and 6


def _rows(values: object, what: str) -> IndexArray:
    """Owned exact rows: nonnegative integers, strictly increasing."""
    found = _exact_index(values, what, upper=LARGEST)
    if (found[1:] <= found[:-1]).any():
        raise ValueError(f"{what} must increase strictly")
    return found


@dataclass(frozen=True, eq=False)
class Visit:
    """One visit of step 5: the block and the rows that it removed (owned)."""

    block: int
    removed: IndexArray

    def __post_init__(self) -> None:
        """Own exact rows; ``ValueError`` for a block that is not an exact index."""
        (block,) = _exact_index([self.block], "a visit's block", upper=LARGEST)
        object.__setattr__(self, "block", int(block))
        object.__setattr__(self, "removed", _rows(self.removed, "removed rows"))


@dataclass(frozen=True, eq=False)
class LevelSeparation:
    """Step 6: the rows removed, each one's first block, every block's levels.

    ``levels`` holds per block the codes of its levels without a positive outcome.
    """

    removed: IndexArray
    blocks: IndexArray
    levels: tuple[IndexArray, ...]

    def __post_init__(self) -> None:
        """Own exact rows, blocks and levels; one first block per removed row."""
        removed = _rows(self.removed, "removed rows")
        blocks = _exact_index(self.blocks, "first blocks", upper=LARGEST)
        if blocks.size != removed.size:
            raise ValueError("one first block per removed row")
        levels = tuple(_rows(level, "levels") for level in self.levels)
        object.__setattr__(self, "removed", removed)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True, eq=False)
class ClosurePass:
    """Step 5 to its fixed point, then step 6 (``None`` when no row remained)."""

    visits: tuple[Visit, ...]
    separation: LevelSeparation | None

    def __post_init__(self) -> None:
        """Own the visits as a tuple; ``ValueError`` for another record."""
        visits = tuple(self.visits)
        separation = self.separation
        if not all(isinstance(visit, Visit) for visit in visits) or not (
            separation is None or isinstance(separation, LevelSeparation)
        ):
            raise ValueError("a pass holds visits and a level separation or None")
        object.__setattr__(self, "visits", visits)


@dataclass(frozen=True, eq=False)
class Closure:
    """The passes of steps 5 and 6 to their fixed point, and the ledger after them.

    ``ValueError`` unless the ledger codes every removed row with that removal's first
    cause (step 5 and its visit's block, step 6 and its first block), once.
    """

    passes: tuple[ClosurePass, ...]
    ledger: RowLedger

    def __post_init__(self) -> None:
        """Own the passes; the ledger codes each removal with its first cause."""
        passes = tuple(self.passes)
        if not isinstance(self.ledger, RowLedger) or not all(
            isinstance(found, ClosurePass) for found in passes
        ):
            raise ValueError("a closure holds passes and a ledger")
        rows: list[int] = []
        causes: list[tuple[int, int]] = []
        for found in passes:
            for visit in found.visits:
                rows += visit.removed.tolist()
                causes += [(SINGLETON, visit.block)] * visit.removed.size
            separation = found.separation
            if separation is not None:
                rows += separation.removed.tolist()
                causes += [(FIXED_EFFECT, b) for b in separation.blocks.tolist()]
        ledger = self.ledger
        once = len(set(rows)) == len(rows) and all(r < ledger.nobs for r in rows)
        coded = [(int(ledger.codes[r]), int(ledger.blocks[r])) for r in rows if once]
        if not once or coded != causes:
            raise ValueError("the ledger must code each removed row with its cause")
        object.__setattr__(self, "passes", passes)


def _aligned(fixed: FixedEffects, ledger: RowLedger) -> None:
    """``ValueError`` unless the blocks are encoded on every current input row."""
    if fixed.nobs != ledger.nobs or (fixed.codes[0][ledger.positions] < 0).any():
        raise ValueError("the blocks and the ledger must be of the same input rows")


def _singletons(
    codes: IndexArray, current: IndexArray, replicas: IndexArray | None, levels: int
) -> IndexArray:
    """The current rows of the singleton levels of one block."""
    found = codes[current]
    lone = np.bincount(found, minlength=levels)[found] == 1
    if replicas is not None:
        # One physical row is one replica only when its exact count is one.
        lone &= replicas[current] == 1
    return frozen_index(current[lone])


def singleton_closure(
    fixed: FixedEffects, ledger: RowLedger
) -> tuple[tuple[Visit, ...], RowLedger]:
    """Step 5 to its fixed point by the cyclic traversal of the blocks."""
    _aligned(fixed, ledger)
    blocks = fixed.grouping.blocks
    count = len(blocks)
    visits: list[Visit] = []
    remaining, g = count, 0
    while remaining and ledger.count:
        found = _singletons(
            fixed.codes[g], ledger.positions, ledger.replicas, blocks[g].n_levels
        )
        visits.append(Visit(g, found))
        if found.size:
            ledger = ledger.exclude(found, SINGLETON, np.full(found.size, g))
            remaining = count - 1
        else:
            remaining -= 1
        g = (g + 1) % count
    return tuple(visits), ledger


def level_separation(
    fixed: FixedEffects, ledger: RowLedger, outcome: FloatArray
) -> tuple[LevelSeparation, RowLedger]:
    """Step 6 once: the union of the levels with no positive raw ``outcome``.

    ``outcome`` is the raw outcome of every input row, read on the current rows.
    """
    _aligned(fixed, ledger)
    current = ledger.positions
    positive = outcome[current] > 0
    first = np.full(ledger.nobs, NO_BLOCK, dtype=np.intp)
    levels = []
    for g, block in enumerate(fixed.grouping.blocks):
        codes = fixed.codes[g][current]
        present = np.bincount(codes, minlength=block.n_levels)
        positives = np.bincount(codes[positive], minlength=block.n_levels)
        empty = (present > 0) & (positives == 0)
        levels.append(frozen_index(np.flatnonzero(empty)))
        first[current[empty[codes] & (first[current] == NO_BLOCK)]] = g
    removed = frozen_index(np.flatnonzero(first != NO_BLOCK))
    separation = LevelSeparation(removed, frozen_index(first[removed]), tuple(levels))
    if removed.size:
        ledger = ledger.exclude(removed, FIXED_EFFECT, first[removed])
    return separation, ledger


def close(
    fixed: FixedEffects | None,
    ledger: RowLedger,
    outcome: FloatArray,
    *,
    keep_singletons: bool,
) -> Closure:
    """Steps 5 and 6 to their joint fixed point; nothing without blocks."""
    passes: list[ClosurePass] = []
    while fixed is not None:
        visits: tuple[Visit, ...] = ()
        if not keep_singletons:
            visits, ledger = singleton_closure(fixed, ledger)
        if not ledger.count:
            passes.append(ClosurePass(visits, None))
            break
        separation, ledger = level_separation(fixed, ledger, outcome)
        passes.append(ClosurePass(visits, separation))
        if not separation.removed.size or not ledger.count:
            break
    return Closure(tuple(passes), ledger)

def _native_parts(
    effects: EffectColumns,
) -> tuple[_OwnedMapping[Column], dict[str, _storage.ComponentRun]] | None:
    """Actual same-record native owners and any already requested object fields."""
    if type(effects) is not EffectColumns or not _fixed_original():
        return None
    state = vars(effects)
    if set(state) != {"terms", "columns", "rows", "_native_columns", "_native_runs"}:
        return None
    native, runs = state.get("_native_columns"), state.get("_native_runs")
    if (type(native) is not _OwnedMapping
            or not _storage.matching_runs(runs, native._items)):
        return None
    shown = state.get("columns")
    if shown is not None and (
        type(shown) is not _OwnedMapping or shown._items.keys() != native._items.keys()
        or any(shown._items[name] is not runs[name].object_column()
               for name in native._items)
    ):
        return None
    return native, runs


def _owned_fixed(
    effects: EffectColumns, positions: IndexArray, *, _encoder: KeyEncoder | None = None,
) -> FixedEffects:
    """Compose this native producer's exact grouping and full-input codes once."""
    if type(effects) is not EffectColumns or not _fixed_original():
        return effects.fixed(positions)
    native = _native_parts(effects)
    if native is None:
        if set(vars(effects)) != {"terms", "columns", "rows"}:
            return effects.fixed(positions)
        grouping = _owned_derived(effects.terms, effects.columns, effects.rows, positions)
    else:
        columns, runs = native
        grouping = None
        try:
            grouping = _owned_derived(
                effects.terms, columns, effects.rows, positions, _runs=runs,
                _encoder=_encoder,
            )
        except MemoryError:
            pass
        if grouping is None:
            for run in runs.values():
                run.discard_presentation()
    if grouping is None:
        return effects.fixed(positions)
    result = object.__new__(FixedEffects)
    object.__setattr__(result, "grouping", grouping)
    object.__setattr__(result, "codes", _row_codes(grouping, effects.rows))
    return result


def _fixed_original() -> bool:
    """Unknown constructor/encoding hooks keep their original reads and errors."""
    return (
        all(globals()[name] is value for name, value in _FIXED_GLOBALS)
        and all(vars(owner).get(name) is value
                for owner, name, value in _FIXED_METHODS)
        and _storage.original_producers is _STORAGE_ORIGINAL
        and _STORAGE_ORIGINAL()
    )


_FIXED_GLOBALS = tuple(
    (name, globals()[name]) for name in (
        "FixedEffects", "EffectColumns", "Source", "Derived", "Grouping",
        "Column", "_OwnedMapping",
        "_row_codes", "_exact_index",
        "frozen_index", "_owned_derived", "_owned_fixed", "_fixed_original",
        "_native_parts",
    )
)
_FIXED_METHODS = tuple(
    (owner, name, vars(owner).get(name)) for owner, names in (
        (FixedEffects, ("__new__", "__init__", "__post_init__",
                        "__getattribute__", "__getattr__", "grouping", "codes")),
        (EffectColumns, ("__new__", "__init__", "__getattribute__", "__getattr__",
                         "__getstate__", "_from_native",
                         "encode", "fixed", "names", "terms", "columns", "rows")),
        (_OwnedMapping, ("__new__", "__init__")),
    ) for name in names
)

_STORAGE_ORIGINAL = _storage.original_producers
_EFFECT_RECORD = EffectColumns
