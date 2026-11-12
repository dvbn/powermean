"""Detached fixed-effect partition, compound-label view, and connectivity snapshots.

This module defines :func:`validate_partition_lineage`, :class:`PartitionLabelsView`,
:class:`RetainedPartitionSnapshot` (including :meth:`compound_labels`),
:class:`RetainedTopologySnapshot`, and :class:`PartitionIdentityMap`, and re-exports
block/grouping symbols from :mod:`powermean.results.absorbed_groupings`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from powermean.data.raw import AnyArray, _object_array
from powermean.data.sample_fe_source import Derived
from powermean.fe.connectivity import (
    Components,
    IndicatorRank,
    components,
    indicator_rank,
)
from powermean.fe.groups import Block, Grouping
from powermean.model.order import IndexArray
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    GroupingKey,
    RetainedBlockSnapshot,
    RetainedGroupingSnapshot,
    _owned_components,
    _view_over_component_runs,
    _view_over_labels,
    detach_grouping,
)
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import PartitionKey
from powermean.results.retained_labels import (
    RetainedLabelBatch,
    _copy_with_memo,
    _stored_raw_elements,
    build_label_batch,
)

_RUN_BATCH = build_label_batch
_RUN_LABEL_VIEW = _view_over_labels

__all__ = [
    "GroupingIdentityMap",
    "GroupingKey",
    "PartitionIdentityMap",
    "PartitionLabelsView",
    "RetainedBlockSnapshot",
    "RetainedGroupingSnapshot",
    "RetainedPartitionSnapshot",
    "RetainedTopologySnapshot",
    "_copy_with_memo",
    "_owned_components",
    "_stored_raw_elements",
    "build_label_batch",
    "detach_grouping",
    "validate_partition_lineage",
]


def validate_partition_lineage(
    source: Grouping, grouping: Grouping, nobs: int
) -> tuple[IndexArray, ...]:
    """Validate and copy or derive final->source level ordinals before detachment."""
    if isinstance(grouping, Derived):
        return tuple(owned_array(o) for o in grouping.ordinals)
    place = np.full(nobs, -1, dtype=np.intp)
    place[source.positions] = np.arange(source.positions.size, dtype=np.intp)
    at = place[grouping.positions]
    if (at < 0).any():
        raise ValueError("grouping positions are not a subset of source positions")
    found: list[IndexArray] = []
    for blk, orig in zip(grouping.blocks, source.blocks, strict=True):
        rev = np.arange(blk.codes.size - 1, -1, -1, dtype=np.intp)
        first = np.empty(blk.n_levels, dtype=np.intp)
        first[blk.codes[rev]] = rev
        ordinal: IndexArray = orig.codes[at[first]]
        if not np.array_equal(ordinal[blk.codes], orig.codes[at]) or not np.all(
            ordinal[:-1] < ordinal[1:]
        ):
            raise ValueError(
                "grouping block codes are not an order-preserving restriction"
            )
        found.append(owned_array(ordinal))
    return tuple(found)


def _materialize_columns_dict(
    comp_tuples: dict[str, tuple[object, ...]],
    comp_dtypes: dict[str, np.dtype[Any]],
) -> dict[str, AnyArray]:
    """Build a fresh mapping of read-only column arrays from materialized tuples."""
    out: dict[str, AnyArray] = {}
    for name, tup in comp_tuples.items():
        dt = comp_dtypes.get(name)
        if dt is not None and dt.kind != "O":
            out[name] = owned_array(np.asarray(list(tup), dtype=dt))
        else:
            out[name] = _object_array(list(tup))
    return out


@dataclass(frozen=True)
class PartitionLabelsView:
    """Single passive caller-owned compound label view sharing one fresh local memo."""

    names: tuple[str, ...]
    source_columns: dict[str, AnyArray] | None
    final_columns: dict[str, AnyArray] | None
    source_levels: tuple[tuple[object, ...], ...]
    final_levels: tuple[tuple[object, ...], ...]
    source_levels_array: tuple[AnyArray, ...]
    final_levels_array: tuple[AnyArray, ...]
    component_columns: dict[str, AnyArray] | None
    ordinals: tuple[IndexArray, ...]

    @property
    def grouping_columns(self) -> dict[str, AnyArray] | None:
        """Alias for ``final_columns``."""
        return self.final_columns

    @property
    def grouping_levels(self) -> tuple[tuple[object, ...], ...]:
        """Alias for ``final_levels``."""
        return self.final_levels

    @property
    def grouping_levels_array(self) -> tuple[AnyArray, ...]:
        """Alias for ``final_levels_array``."""
        return self.final_levels_array


@dataclass(frozen=True, init=False)
class RetainedPartitionSnapshot:
    """Detached source and final partition with one shared label batch and lineage.

    ``_source_view`` and ``_final_view`` are the two groupings as the retained
    assembly consumes them internally: owned once at detachment, over the batch's own
    label copies, and shared by every record assembled from this partition, each of
    which stores them privately and copies them anew on outward exposure.  The outward
    views are :meth:`compound_view` and :meth:`compound_labels`, fresh on every call.
    """

    key: PartitionKey
    source_key: GroupingKey
    grouping_key: GroupingKey
    nobs: int
    names: tuple[str, ...]
    terms: tuple[tuple[str, ...], ...] | None
    is_derived: bool
    batch: RetainedLabelBatch
    _ordinals: tuple[IndexArray, ...]
    _source_positions: IndexArray
    _final_positions: IndexArray
    _source_codes: tuple[IndexArray, ...]
    _source_counts: tuple[IndexArray, ...]
    _final_codes: tuple[IndexArray, ...]
    _final_counts: tuple[IndexArray, ...]
    _source_view: RetainedGroupingSnapshot
    _final_view: RetainedGroupingSnapshot

    def __init__(
        self,
        source: Grouping,
        grouping: Grouping,
        nobs: int,
        *,
        key: PartitionKey = 0,
        source_key: GroupingKey | None = None,
        grouping_key: GroupingKey | None = None,
    ) -> None:
        """Detach ``(source, grouping)`` into one shared batch and lineage ordinals."""
        ords = validate_partition_lineage(source, grouping, nobs)
        batch = build_label_batch(source, grouping)
        derived = isinstance(source, Derived)
        terms = source.source.terms if isinstance(source, Derived) else None
        s_key = int(key) if source_key is None else int(source_key)
        g_key = int(key) if grouping_key is None else int(grouping_key)
        src_levels, fin_levels, components = batch._stored_slices()
        columns = components if derived else None
        dtypes = batch._dtypes if derived else None
        src_view = _view_over_labels(
            source,
            src_levels,
            partition_key=s_key,
            terms=terms,
            columns=columns,
            dtypes=dtypes,
        )
        fin_view = _view_over_labels(
            grouping,
            fin_levels,
            partition_key=g_key,
            terms=terms,
            columns=columns,
            dtypes=dtypes,
            ordinals=ords,
        )
        for attr, val in (
            ("key", int(key)),
            ("source_key", s_key),
            ("grouping_key", g_key),
            ("nobs", int(nobs)),
            ("names", grouping.names),
            ("terms", terms),
            ("is_derived", derived),
            ("batch", batch),
            ("_ordinals", ords),
            ("_source_positions", src_view._positions),
            ("_final_positions", fin_view._positions),
            ("_source_codes", tuple(b._codes for b in src_view.blocks)),
            ("_source_counts", tuple(b._counts for b in src_view.blocks)),
            ("_final_codes", tuple(b._codes for b in fin_view.blocks)),
            ("_final_counts", tuple(b._counts for b in fin_view.blocks)),
            ("_source_view", src_view),
            ("_final_view", fin_view),
        ):
            object.__setattr__(self, attr, val)

    def __getstate__(self) -> Any:
        """Export declared rich state without carrying a private owner recipe.

        Generic/subclass copying retains object.__getstate__, including slots and
        existing reduction hooks. Only this exact private native view expands.
        """
        state = object.__getstate__(self)
        if (type(self) is RetainedPartitionSnapshot
                and "_numeric_partition" in vars(self)):
            state = dict(vars(self))
            for field in fields(self):
                state[field.name] = getattr(self, field.name)
            for name in ("_numeric_partition",):
                state.pop(name, None)
            # The requested legacy fields now ARE the owned state. Future rich
            # copies do not reconstruct another full-row projection.
            vars(self).update(state)
            for name in tuple(vars(self)):
                if name not in state:
                    vars(self).pop(name)
        return state

    @classmethod
    def _from_numeric(
        cls, source: Grouping, grouping: Grouping, nobs: int, *,
        key: PartitionKey, grouping_map: GroupingIdentityMap,
    ) -> RetainedPartitionSnapshot:
        """Lossless view of PRIVATE owned topology; no eager compound label batch.

        Grouping consumers need code/level views, not the all-input compound view
        or a second connectivity/rank calculation. Those detailed public fields
        retain their original constructor when actually requested.
        """
        result = object.__new__(cls)
        for name, value in (
            ("key", key), ("source_key", grouping_map.key_for(source)),
            ("grouping_key", grouping_map.key_for(grouping)), ("nobs", nobs),
            ("names", grouping.names),
            ("terms", source.source.terms if type(source) is Derived else None),
            ("is_derived", type(source) is Derived),
            ("_numeric_partition", (source, grouping, grouping_map)),
        ):
            object.__setattr__(result, name, value)
        return result

    def __getattribute__(self, name: str) -> Any:
        """Resolve exact-native presentation fields; generic lookup stays unchanged."""
        if type(self) is RetainedPartitionSnapshot and type(name) is str:
            state = object.__getattribute__(self, "__dict__")
            if ("_numeric_partition" in state and name not in state and name in {
                "batch", "_ordinals", "_source_positions", "_final_positions",
                "_source_codes", "_source_counts", "_final_codes", "_final_counts",
                "_source_view", "_final_view",
            }):
                source, grouping, groupings = self._numeric_partition
                if name == "_source_view":
                    return groupings.register(source)
                if name == "_final_view":
                    return groupings.register(grouping)
                materialized = RetainedPartitionSnapshot(
                    source, grouping, self.nobs, key=self.key,
                    source_key=self.source_key, grouping_key=self.grouping_key,
                )
                vars(self).update(vars(materialized))
                vars(self).pop("_numeric_partition", None)
        return object.__getattribute__(self, name)

    @classmethod
    def _from_owned_components(
        cls, source: Grouping, grouping: Grouping, nobs: int, *,
        key: PartitionKey, source_key: GroupingKey, grouping_key: GroupingKey,
    ) -> RetainedPartitionSnapshot | None:
        """Lossless private composition after original owned-order admission."""
        if (cls is not RetainedPartitionSnapshot
                or build_label_batch is not _RUN_BATCH
                or _view_over_labels is not _RUN_LABEL_VIEW):
            return None
        batch = RetainedLabelBatch._from_runs(source, grouping)
        if batch is None or type(source) is not Derived:
            return None
        ords = validate_partition_lineage(source, grouping, nobs)
        src_levels, fin_levels, runs = batch._run_views()
        terms = source.source.terms
        src_view = _view_over_component_runs(
            source, src_levels, runs, partition_key=source_key, terms=terms,
        )
        fin_view = _view_over_component_runs(
            grouping, fin_levels, runs, partition_key=grouping_key,
            terms=terms, ordinals=ords,
        )
        result = object.__new__(cls)
        for attr, val in (
            ("key", int(key)), ("source_key", source_key),
            ("grouping_key", grouping_key), ("nobs", int(nobs)),
            ("names", grouping.names), ("terms", terms), ("is_derived", True),
            ("batch", batch), ("_ordinals", ords),
            ("_source_positions", src_view._positions),
            ("_final_positions", fin_view._positions),
            ("_source_codes", tuple(b._codes for b in src_view.blocks)),
            ("_source_counts", tuple(b._counts for b in src_view.blocks)),
            ("_final_codes", tuple(b._codes for b in fin_view.blocks)),
            ("_final_counts", tuple(b._counts for b in fin_view.blocks)),
            ("_source_view", src_view), ("_final_view", fin_view),
        ):
            object.__setattr__(result, attr, val)
        return result

    @property
    def ordinals(self) -> tuple[IndexArray, ...]:
        """Return fresh owned copies of the block level ordinals."""
        return tuple(owned_array(o) for o in self._ordinals)

    def compound_view(
        self, memo: dict[int, Any] | None = None
    ) -> tuple[
        RetainedGroupingSnapshot, RetainedGroupingSnapshot, tuple[IndexArray, ...]
    ]:
        """Materialize source and final groupings in one pass with one local memo."""
        view_memo: dict[int, Any] = {} if memo is None else memo
        src_slices, fin_slices, comp_tuples, comp_dtypes = self.batch.materialize(
            view_memo
        )
        src_blocks = tuple(
            RetainedBlockSnapshot(name, c_cod, c_cnt, lvls)
            for name, c_cod, c_cnt, lvls in zip(
                self.names,
                self._source_codes,
                self._source_counts,
                src_slices,
                strict=True,
            )
        )
        fin_blocks = tuple(
            RetainedBlockSnapshot(name, c_cod, c_cnt, lvls)
            for name, c_cod, c_cnt, lvls in zip(
                self.names,
                self._final_codes,
                self._final_counts,
                fin_slices,
                strict=True,
            )
        )
        cols = comp_tuples if self.is_derived else None
        dtypes = comp_dtypes if self.is_derived else None
        src_grp = RetainedGroupingSnapshot(
            src_blocks,
            self._source_positions,
            partition_key=self.source_key,
            terms=self.terms,
            columns_tuple=cols,
            column_dtypes=dtypes,
        )
        fin_grp = RetainedGroupingSnapshot(
            fin_blocks,
            self._final_positions,
            partition_key=self.grouping_key,
            terms=self.terms,
            columns_tuple=cols,
            column_dtypes=dtypes,
            ordinals=self._ordinals,
        )
        return src_grp, fin_grp, self.ordinals

    def compound_labels(self) -> PartitionLabelsView:
        """Return one caller-owned passive label view sharing one fresh local memo."""
        view_memo: dict[int, Any] = {}
        src_slices, fin_slices, comp_tuples, comp_dtypes = self.batch.materialize(
            view_memo
        )
        src_arrays = tuple(_object_array(list(lvls)) for lvls in src_slices)
        fin_arrays = tuple(_object_array(list(lvls)) for lvls in fin_slices)
        if self.is_derived:
            src_cols: dict[str, AnyArray] | None = _materialize_columns_dict(
                comp_tuples, comp_dtypes
            )
            fin_cols: dict[str, AnyArray] | None = _materialize_columns_dict(
                comp_tuples, comp_dtypes
            )
            comp_cols: dict[str, AnyArray] | None = _materialize_columns_dict(
                comp_tuples, comp_dtypes
            )
        else:
            src_cols, fin_cols, comp_cols = None, None, None
        return PartitionLabelsView(
            names=self.names,
            source_columns=src_cols,
            final_columns=fin_cols,
            source_levels=src_slices,
            final_levels=fin_slices,
            source_levels_array=src_arrays,
            final_levels_array=fin_arrays,
            component_columns=comp_cols,
            ordinals=self.ordinals,
        )


@dataclass(frozen=True, init=False)
class RetainedTopologySnapshot:
    """Owned bipartite connectivity and indicator rank of a retained partition."""

    names: tuple[str, ...]
    n_rows: int
    n_levels: tuple[int, ...]
    rank: IndicatorRank
    _source_ordinals: tuple[IndexArray, ...] | None
    _components: Components

    def __init__(
        self,
        blocks: tuple[Block, ...],
        n_rows: int,
        *,
        source_ordinals: tuple[IndexArray, ...] | None = None,
    ) -> None:
        """Compute and own connectivity components and indicator rank on ingress."""
        ords = (
            None
            if source_ordinals is None
            else tuple(owned_array(o) for o in source_ordinals)
        )
        for attr, val in (
            ("names", tuple(b.name for b in blocks)),
            ("n_rows", int(n_rows)),
            ("n_levels", tuple(b.n_levels for b in blocks)),
            ("rank", indicator_rank(blocks)),
            ("_source_ordinals", ords),
            ("_components", _owned_components(components(blocks))),
        ):
            object.__setattr__(self, attr, val)

    @property
    def source_ordinals(self) -> tuple[IndexArray, ...] | None:
        """Return fresh owned copies of the source-to-final level ordinals."""
        if self._source_ordinals is None:
            return None
        return tuple(owned_array(o) for o in self._source_ordinals)

    @property
    def components(self) -> Components:
        """Return a fresh deeply owned copy of the :class:`Components`."""
        return _owned_components(self._components)


class PartitionIdentityMap:
    """Temporary per-handoff identity map from original ``(source, grouping, nobs)``."""

    def __init__(self, *, grouping_map: GroupingIdentityMap | None = None) -> None:
        """Initialize an empty partition identity map with a separate grouping map."""
        self.grouping_map = (
            GroupingIdentityMap() if grouping_map is None else grouping_map
        )
        self._by_pair: dict[tuple[int, int, int], RetainedPartitionSnapshot] = {}
        self._refs: list[object] = []
        self._next_key: PartitionKey = 0

    def capture(
        self, source: Grouping, grouping: Grouping, nobs: int
    ) -> RetainedPartitionSnapshot:
        """Return the ``RetainedPartitionSnapshot`` for ``(source, grouping, nobs)``."""
        pair_id = (id(source), id(grouping), int(nobs))
        if pair_id in self._by_pair:
            return self._by_pair[pair_id]
        key = self._next_key
        self._next_key += 1
        src_key = self.grouping_map.key_for(source)
        fin_key = self.grouping_map.key_for(grouping)
        snap = RetainedPartitionSnapshot(
            source,
            grouping,
            nobs,
            key=key,
            source_key=src_key,
            grouping_key=fin_key,
        )
        return self._remember(source, grouping, pair_id, snap)

    def _capture_numeric(
        self, source: Grouping, grouping: Grouping, nobs: int,
    ) -> RetainedPartitionSnapshot:
        """Register the owned topology view once, without forcing label exports."""
        pair_id = (id(source), id(grouping), int(nobs))
        if pair_id in self._by_pair:
            return self._by_pair[pair_id]
        snapshot = RetainedPartitionSnapshot._from_numeric(
            source, grouping, nobs, key=self._next_key,
            grouping_map=self.grouping_map,
        )
        self._next_key += 1
        self._by_pair[pair_id] = snapshot
        self._refs.extend((source, grouping))
        return snapshot

    def _capture_owned_components(
        self, source: Grouping, grouping: Grouping, nobs: int,
    ) -> RetainedPartitionSnapshot:
        """The existing normal-order admission consumes its source's full runs."""
        if (type(self) is not PartitionIdentityMap
                or type(self.grouping_map) is not GroupingIdentityMap):
            return self.capture(source, grouping, nobs)
        pair_id = (id(source), id(grouping), int(nobs))
        if pair_id in self._by_pair:
            return self._by_pair[pair_id]
        snap = RetainedPartitionSnapshot._from_owned_components(
            source, grouping, nobs, key=self._next_key,
            source_key=self.grouping_map.key_for(source),
            grouping_key=self.grouping_map.key_for(grouping),
        )
        if snap is None:
            return self.capture(source, grouping, nobs)
        self._next_key += 1
        return self._remember(source, grouping, pair_id, snap)

    def _remember(
        self, source: Grouping, grouping: Grouping,
        pair_id: tuple[int, int, int], snap: RetainedPartitionSnapshot,
    ) -> RetainedPartitionSnapshot:
        """Keep the original registration and one-memo custom materialization order."""
        self._by_pair[pair_id] = snap
        # Only copy-free labels skip the original compound materialization. Custom
        # labels retain its callback order, copy depth and one-memo alias semantics.
        if snap.batch._copy_free:
            src_view, fin_view = snap._source_view, snap._final_view
        else:
            src_view, fin_view, _ = snap.compound_view()
        self.grouping_map.register_detached(source, src_view)
        self.grouping_map.register_detached(grouping, fin_view)
        self._refs.extend((source, grouping))
        return snap

    def key_for_grouping(self, grouping: Grouping) -> GroupingKey:
        """Return or assign a stable ``GroupingKey`` for ``grouping``."""
        return self.grouping_map.key_for(grouping)

    def register_grouping(self, grouping: Grouping) -> RetainedGroupingSnapshot:
        """Return cached or freshly detached ``RetainedGroupingSnapshot``."""
        return self.grouping_map.register(grouping)

    def clear_anchors(self) -> None:
        """Release strong references to original partition and grouping objects."""
        self._refs.clear()
        self.grouping_map.clear_anchors()
