"""Detached fixed-effect block and grouping snapshots and grouping identity map.

This module defines :func:`_owned_components`, :class:`RetainedBlockSnapshot`,
:class:`RetainedGroupingSnapshot`, :func:`detach_grouping`, and
:class:`GroupingIdentityMap`.  Every independent ``.levels``, ``.levels_array``, and
``.columns`` getter call freshly clones nested mutable label objects.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from powermean.data.fe_component_storage import ComponentRun, original_producers
from powermean.data.raw import AnyArray, _object_array
from powermean.data.sample_fe_source import Derived
from powermean.fe.connectivity import Components
from powermean.fe.groups import Grouping
from powermean.model.order import IndexArray
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import GroupingKey, PartitionKey
from powermean.results.retained_labels import (
    RetainedLabelBatch,
    _copy_free_labels,
    _copy_with_memo,
    _stored_raw_elements,
)

__all__ = [
    "GroupingIdentityMap",
    "GroupingKey",
    "RetainedBlockSnapshot",
    "RetainedGroupingSnapshot",
    "_owned_components",
    "_view_over_labels",
    "detach_grouping",
]


def _owned_components(comp: Components) -> Components:
    """Create a deeply owned ``Components`` with fresh array objects."""
    return Components(
        of_rows=owned_array(comp.of_rows),
        of_levels=tuple(owned_array(arr) for arr in comp.of_levels),
        row_counts=owned_array(comp.row_counts),
        level_counts=tuple(owned_array(arr) for arr in comp.level_counts),
    )


@dataclass(frozen=True, init=False)
class RetainedBlockSnapshot:
    """Detached snapshot of one fixed-effect ``Block`` without re-encoding labels."""

    name: str
    _codes: IndexArray
    _counts: IndexArray
    _levels_tuple: tuple[object, ...]

    def __init__(
        self,
        name: str,
        codes: IndexArray,
        counts: IndexArray,
        levels_tuple: Sequence[object],
    ) -> None:
        """Own block codes, counts, and materialized level representatives."""
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "_codes", owned_array(codes))
        object.__setattr__(self, "_counts", owned_array(counts))
        object.__setattr__(self, "_levels_tuple", tuple(levels_tuple))

    @property
    def codes(self) -> IndexArray:
        """Return a fresh owned copy of the block level codes."""
        return owned_array(self._codes)

    @property
    def counts(self) -> IndexArray:
        """Return a fresh owned copy of the block level counts."""
        return owned_array(self._counts)

    @property
    def n_rows(self) -> int:
        """Return the number of rows in the block."""
        return int(self._codes.size)

    @property
    def n_levels(self) -> int:
        """Return the number of levels in the block."""
        return int(self._counts.size)

    @property
    def levels(self) -> tuple[object, ...]:
        """Return a fresh tuple of cloned level representatives for this access."""
        return _copy_with_memo(self._levels_tuple, {})

    @property
    def levels_array(self) -> AnyArray:
        """Return a fresh read-only object array of cloned level representatives."""
        return _object_array(list(_copy_with_memo(self._levels_tuple, {})))

    def rematerialize(
        self, memo: dict[int, Any] | None = None
    ) -> RetainedBlockSnapshot:
        """Return a fresh block snapshot copied with one local ``memo``."""
        active_memo: dict[int, Any] = {} if memo is None else memo
        return RetainedBlockSnapshot(
            self.name,
            self._codes,
            self._counts,
            _copy_with_memo(self._levels_tuple, active_memo),
        )


@dataclass(frozen=True, init=False)
class RetainedGroupingSnapshot:
    """Detached snapshot of a ``Grouping`` or ``Derived`` partition."""

    blocks: tuple[RetainedBlockSnapshot, ...]
    terms: tuple[tuple[str, ...], ...] | None
    partition_key: PartitionKey
    _positions: IndexArray
    _columns_tuple: Mapping[str, tuple[object, ...]] | None
    _column_dtypes: Mapping[str, np.dtype[Any]] | None
    _ordinals: tuple[IndexArray, ...] | None
    _component_runs: Mapping[str, ComponentRun] | None

    def __init__(
        self,
        blocks: Sequence[RetainedBlockSnapshot],
        positions: IndexArray,
        *,
        partition_key: PartitionKey = 0,
        terms: Sequence[Sequence[str]] | None = None,
        columns_tuple: Mapping[str, Sequence[object]] | None = None,
        column_dtypes: Mapping[str, np.dtype[Any]] | None = None,
        ordinals: Sequence[IndexArray] | None = None,
    ) -> None:
        """Own blocks, positions, optional derived component columns, and ordinals."""
        object.__setattr__(self, "blocks", tuple(blocks))
        object.__setattr__(self, "partition_key", int(partition_key))
        object.__setattr__(self, "_positions", owned_array(positions))
        object.__setattr__(
            self,
            "terms",
            None if terms is None else tuple(tuple(t) for t in terms),
        )
        object.__setattr__(
            self,
            "_columns_tuple",
            None
            if columns_tuple is None
            else MappingProxyType({k: tuple(v) for k, v in columns_tuple.items()}),
        )
        object.__setattr__(
            self,
            "_column_dtypes",
            None if column_dtypes is None else MappingProxyType(dict(column_dtypes)),
        )
        object.__setattr__(
            self,
            "_ordinals",
            None if ordinals is None else tuple(owned_array(o) for o in ordinals),
        )
        object.__setattr__(self, "_component_runs", None)

    @classmethod
    def _from_component_runs(
        cls, blocks: Sequence[RetainedBlockSnapshot], positions: IndexArray, *,
        partition_key: PartitionKey, terms: tuple[tuple[str, ...], ...] | None,
        runs: Mapping[str, ComponentRun], ordinals: Sequence[IndexArray] | None,
    ) -> RetainedGroupingSnapshot:
        """Fresh grouping headers over private already-owned complete components."""
        result = cls(blocks, positions, partition_key=partition_key, terms=terms,
                     column_dtypes={name: run.dtype for name, run in runs.items()},
                     ordinals=ordinals)
        object.__setattr__(result, "_component_runs", MappingProxyType(dict(runs)))
        return result

    def _stored_component_tuples(
        self,
    ) -> Mapping[str, tuple[object, ...]] | None:
        """Expose full component tuples only to an actual outward materializer."""
        runs = self._component_runs
        if runs is None:
            return self._columns_tuple
        result: dict[str, tuple[object, ...]] = {}
        for name, run in runs.items():
            result[name] = run.objects()
        return result

    @property
    def positions(self) -> IndexArray:
        """Return a fresh owned copy of the grouping row positions."""
        return owned_array(self._positions)

    @property
    def ordinals(self) -> tuple[IndexArray, ...] | None:
        """Return fresh owned copies of the block level ordinals, or ``None``."""
        if self._ordinals is None:
            return None
        return tuple(owned_array(o) for o in self._ordinals)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the ordered block names."""
        return tuple(b.name for b in self.blocks)

    @property
    def n_rows(self) -> int:
        """Return the number of rows in the grouping."""
        return int(self._positions.size)

    @property
    def n_levels(self) -> tuple[int, ...]:
        """Return the level count of each block."""
        return tuple(b.n_levels for b in self.blocks)

    @property
    def columns(self) -> dict[str, AnyArray] | None:
        """Return fresh read-only component column arrays with cloned object labels."""
        if self._columns_tuple is None and self._component_runs is None:
            return None
        out: dict[str, AnyArray] = {}
        dtypes = self._column_dtypes or {}
        memo: dict[int, Any] = {}
        columns = self._stored_component_tuples()
        assert columns is not None
        for name, tup in columns.items():
            dt = dtypes.get(name)
            if dt is not None and dt.kind != "O":
                out[name] = owned_array(np.asarray(list(tup), dtype=dt))
            else:
                copied = _copy_with_memo(tup, memo)
                out[name] = _object_array(list(copied))
        return out

    @property
    def _copy_free(self) -> bool:
        """Whether every owned level/component has only immutable built-in leaves."""
        if self._component_runs is not None:
            return (original_producers()
                    and _copy_with_memo is _RUN_COPY
                    and _copy_free_labels is _RUN_COPY_FREE
                    and all(_copy_free_labels(b._levels_tuple) for b in self.blocks)
                    and all(type(run) is ComponentRun and run.bound()
                            for run in self._component_runs.values()))
        return all(_copy_free_labels(b._levels_tuple) for b in self.blocks) and (
            self._columns_tuple is None
            or all(_copy_free_labels(v) for v in self._columns_tuple.values())
        )

    def rematerialize(
        self, memo: dict[int, Any] | None = None
    ) -> RetainedGroupingSnapshot:
        """Materialize a fresh grouping view in one pass using one fresh local memo."""
        active_memo: dict[int, Any] = {} if memo is None else memo
        new_blocks = tuple(b.rematerialize(active_memo) for b in self.blocks)
        runs = (_rematerialized_runs(self) if memo is None
                and _rematerialized_runs is _RUN_REMATERIALIZE else None)
        if runs is not None:
            return RetainedGroupingSnapshot._from_component_runs(
                new_blocks, self._positions, partition_key=self.partition_key,
                terms=self.terms, runs=runs, ordinals=self._ordinals,
            )
        stored_columns = self._stored_component_tuples()
        new_cols = (
            None if stored_columns is None else
            {name: _copy_with_memo(values, active_memo)
             for name, values in stored_columns.items()}
        )
        return RetainedGroupingSnapshot(
            new_blocks,
            self._positions,
            partition_key=self.partition_key,
            terms=self.terms,
            columns_tuple=new_cols,
            column_dtypes=self._column_dtypes,
            ordinals=self._ordinals,
        )


def _rematerialized_runs(
    grouping: RetainedGroupingSnapshot,
) -> dict[str, ComponentRun] | None:
    """Detach actual native run headers without forcing full object presentation.

    Only a fresh local memo is used. Explicit caller memos and unknown records,
    callbacks or current bindings retain the original materializer. Optional clone
    failure unwinds this frame/memo before the required old construction resumes.
    """
    if (type(grouping) is not RetainedGroupingSnapshot
            or not all(globals()[name] is value for name, value in _MATERIAL_GLOBALS)
            or not all(vars(owner).get(name) is value
                       for owner, name, value in _MATERIAL_METHODS)
            or copy.deepcopy is not _MATERIAL_DEEPCOPY):
        return None
    runs, dtypes = grouping._component_runs, grouping._column_dtypes
    if runs is None or dtypes is None:
        return None
    if (type(runs) is not MappingProxyType or type(dtypes) is not MappingProxyType
            or grouping._columns_tuple is not None
            or type(grouping.blocks) is not tuple
            or any(type(block) is not RetainedBlockSnapshot
                   or type(block._levels_tuple) is not tuple for block in grouping.blocks)
            or any(type(name) is not str for name in (*runs, *dtypes))
            or runs.keys() != dtypes.keys()
            or type(grouping.partition_key) is not int):
        return None
    terms = grouping.terms
    if terms is not None and (
        type(terms) is not tuple or any(type(term) is not tuple
            or any(type(name) is not str for name in term) for term in terms)
    ):
        return None
    if (any(type(run) is not ComponentRun or run.dtype is not np.dtype(object)
            or dtypes[name] is not run.dtype or run._column is not None
            or type(run._stored) is not np.ndarray or run._stored.dtype.kind not in "iu"
            for name, run in runs.items()) or not grouping._copy_free):
        return None
    try:
        memo: dict[int, Any] = {}
        return {name: copy.deepcopy(run, memo) for name, run in runs.items()}
    except MemoryError:
        return None


def detach_grouping(
    grouping: Grouping | RetainedGroupingSnapshot,
    *,
    partition_key: PartitionKey = 0,
    memo: dict[int, Any] | None = None,
) -> RetainedGroupingSnapshot:
    """Detach an admitted ``Grouping`` or rematerialize a snapshot."""
    if isinstance(grouping, RetainedGroupingSnapshot):
        return grouping.rematerialize(memo)
    active_memo: dict[int, Any] = {} if memo is None else memo
    blocks = tuple(
        RetainedBlockSnapshot(
            b.name,
            b.codes,
            b.counts,
            _copy_with_memo(_stored_raw_elements(b._levels), active_memo),
        )
        for b in grouping.blocks
    )
    if isinstance(grouping, Derived):
        src = grouping.source
        cols = {
            k: _copy_with_memo(_stored_raw_elements(c), active_memo)
            for k, c in src.columns.items()
        }
        dtypes = {k: c.dtype for k, c in src.columns.items()}
        return RetainedGroupingSnapshot(
            blocks,
            grouping.positions,
            partition_key=partition_key,
            terms=src.terms,
            columns_tuple=cols,
            column_dtypes=dtypes,
            ordinals=grouping.ordinals,
        )
    return RetainedGroupingSnapshot(
        blocks, grouping.positions, partition_key=partition_key
    )


_RUN_DETACH = detach_grouping
_RUN_ELEMENTS = _stored_raw_elements
_RUN_COPY = _copy_with_memo
_RUN_COPY_FREE = _copy_free_labels

# The same small producer-identity boundary as the run/owned capture path.
# Do not activate constructors, getters, copiers or full materializers to qualify.
_RUN_REMATERIALIZE = _rematerialized_runs
_MATERIAL_DEEPCOPY = copy.deepcopy
_MATERIAL_GLOBALS = tuple((name, globals()[name]) for name in (
    "copy", "ComponentRun", "RetainedGroupingSnapshot", "RetainedBlockSnapshot",
    "MappingProxyType", "original_producers", "owned_array",
    "_copy_with_memo", "_copy_free_labels", "_rematerialized_runs",
))
_MATERIAL_METHODS = tuple(
    (owner, name, vars(owner).get(name)) for owner, names in (
        (RetainedGroupingSnapshot, (
            "__new__", "__init__", "__getattribute__", "__getattr__",
            "_from_component_runs", "_stored_component_tuples", "_copy_free",
        )),
        (RetainedBlockSnapshot, (
            "__new__", "__init__", "__getattribute__", "__getattr__",
            "rematerialize", "_levels_tuple",
        )),
        (ComponentRun, (
            "__getattribute__", "__getattr__", "__deepcopy__", "__slots__",
        )),
    ) for name in names
)


def _view_over_labels(
    grouping: Grouping,
    levels: tuple[tuple[object, ...], ...],
    *,
    partition_key: GroupingKey,
    terms: tuple[tuple[str, ...], ...] | None,
    columns: dict[str, tuple[object, ...]] | None,
    dtypes: dict[str, np.dtype[Any]] | None,
    ordinals: tuple[IndexArray, ...] | None = None,
) -> RetainedGroupingSnapshot:
    """The internally owned view of an admitted ``grouping`` over given label copies.

    The sibling of :func:`detach_grouping` for the retained partition
    (:class:`~powermean.results.absorbed_partitions.RetainedPartitionSnapshot`):
    codes, counts and positions are owned here, once, from the live grouping; the
    level and component labels are the partition batch's own copies, shared with
    this view and with nothing outward, since every outward exposure of the view
    copies them anew (:meth:`RetainedGroupingSnapshot.rematerialize`).
    """
    blocks = tuple(
        RetainedBlockSnapshot(b.name, b.codes, b.counts, lvls)
        for b, lvls in zip(grouping.blocks, levels, strict=True)
    )
    return RetainedGroupingSnapshot(
        blocks,
        grouping.positions,
        partition_key=partition_key,
        terms=terms,
        columns_tuple=columns,
        column_dtypes=dtypes,
        ordinals=ordinals,
    )


def _view_over_component_runs(
    grouping: Grouping, levels: tuple[tuple[object, ...], ...],
    runs: Mapping[str, ComponentRun], *, partition_key: PartitionKey,
    terms: tuple[tuple[str, ...], ...] | None,
    ordinals: Sequence[IndexArray] | None = None,
) -> RetainedGroupingSnapshot:
    """Internal grouping view, without converting whole component runs to tuples."""
    blocks = tuple(
        RetainedBlockSnapshot(block.name, block.codes, block.counts, values)
        for block, values in zip(grouping.blocks, levels, strict=True)
    )
    return RetainedGroupingSnapshot._from_component_runs(
        blocks, grouping.positions, partition_key=partition_key,
        terms=terms, runs=runs, ordinals=ordinals,
    )


class GroupingIdentityMap:
    """Temporary per-handoff identity map allocating one ``GroupingKey`` per object."""

    def __init__(self) -> None:
        """Initialize an empty grouping identity map."""
        self._by_id: dict[int, GroupingKey] = {}
        self._snapshots: dict[int, RetainedGroupingSnapshot] = {}
        self._refs: list[object] = []
        self._next_key: GroupingKey = 0

    def key_for(self, grouping: Grouping) -> GroupingKey:
        """Return the stable ``GroupingKey`` for ``grouping``, allocated once."""
        gid = id(grouping)
        if gid in self._by_id:
            return self._by_id[gid]
        key = self._next_key
        self._next_key += 1
        self._by_id[gid] = key
        self._refs.append(grouping)
        return key

    def register_detached(
        self, grouping: Grouping, snapshot: RetainedGroupingSnapshot
    ) -> RetainedGroupingSnapshot:
        """Cache ``snapshot`` for ``grouping`` if not already cached."""
        gid = id(grouping)
        self.key_for(grouping)
        if gid not in self._snapshots:
            self._snapshots[gid] = snapshot
        return self._snapshots[gid]

    def register(self, grouping: Grouping) -> RetainedGroupingSnapshot:
        """Reuse immutable built-in labels; preserve ordinary custom rematerialization.

        The snapshot is the map's internally owned record, the same object on every
        call for one ``grouping``: the retained records assembled from it store it
        privately and rematerialize it on every outward exposure. Custom labels retain
        the original fresh memo and callback sequence on each internal retrieval too.
        """
        gid = id(grouping)
        if gid in self._snapshots:
            snapshot = self._snapshots[gid]
            return snapshot if snapshot._copy_free else snapshot.rematerialize()
        key = self.key_for(grouping)
        batch = (
            RetainedLabelBatch._from_runs(grouping, grouping)
            if (type(self) is GroupingIdentityMap
                and detach_grouping is _RUN_DETACH
                and _stored_raw_elements is _RUN_ELEMENTS
                and _copy_with_memo is _RUN_COPY
                and _copy_free_labels is _RUN_COPY_FREE)
            else None
        )
        if batch is not None and type(grouping) is Derived:
            levels, _, runs = batch._run_views()
            snap = _view_over_component_runs(
                grouping, levels, runs, partition_key=key,
                terms=grouping.source.terms, ordinals=grouping.ordinals,
            )
        else:
            snap = detach_grouping(grouping, partition_key=key)
        self._snapshots[gid] = snap
        return snap

    def clear_anchors(self) -> None:
        """Release strong references to original grouping objects after capture."""
        self._refs.clear()
