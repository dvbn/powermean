"""Narrow label batch storage and per-view memo materialization for FE partitions.

This module owns :class:`RetainedLabelBatch` and :func:`build_label_batch`, copying
source-block level representatives, final-block level representatives, and derived
component-column representatives with one memo on ingress and materializing each
complete outward compound view in a single pass with one fresh local memo, without
storing consumable memos or exposing raw mutable items.
"""

from __future__ import annotations

import copy
import importlib
import operator
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import compress, repeat
from typing import Any, cast

import numpy as np

from powermean.data.fe_component_storage import (
    ComponentRun,
    matching_runs,
    original_producers,
    run_for,
)
from powermean.data.raw import _IMMUTABLE_TYPES, Column
from powermean.data.sample_fe_source import (
    Derived, Source, _native_source_parts, same_bits,
)
from powermean.fe.groups import Grouping
from powermean.formula.formulaic_api import formulaic
from powermean.model.order import IndexArray
from powermean.results.owned import owned_array

_contrasts_mod = cast("Any", importlib.import_module("formulaic.transforms.contrasts"))
_CONTRASTS_STATE: type[Any] | None = getattr(_contrasts_mod, "ContrastsState", None)
_CUSTOM_CONTRASTS: type[Any] = _contrasts_mod.CustomContrasts
_POLY_CONTRASTS: type[Any] = _contrasts_mod.PolyContrasts
# The ids of the exact immutable built-in types.  Those types live as long as the
# interpreter and ``el`` keeps its own type alive, so ``id(type(el))`` is one of these
# ids exactly when ``type(el)`` is one of :data:`_IMMUTABLE_TYPES`.  The ids are plain
# integers: membership runs no method of ``el``, of its class or of a metaclass.
_IMMUTABLE_TYPE_IDS = frozenset(map(id, _IMMUTABLE_TYPES))
# The same ids with that of the exact ``tuple``, the one container of a copy-free label.
_LABEL_TYPE_IDS = _IMMUTABLE_TYPE_IDS | {id(tuple)}
# The labels of a longer tuple are read in bulk, a window at a time: a label batch is a
# few long runs of one exact type each (block levels, then component columns), so most
# windows take one identity pass.  A tuple of at most ``_SHORT_LABELS`` labels, such as
# an interaction label, is cheaper to read label by label.
_LABEL_WINDOW = 4096
_SHORT_LABELS = 8


def _same_opt_bits(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Return True if two optional arrays are both None or match on ``same_bits``."""
    return (a is None and b is None) or (
        a is not None and b is not None and same_bits(a, b)
    )


def _same_spec_state(a: Any, b: Any) -> bool:
    """Return True if two Formulaic state values match without replay."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same_spec_state(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(
            _same_spec_state(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, np.ndarray):
        return (
            a.shape == b.shape
            and a.dtype == b.dtype
            and (
                same_bits(a, b)
                if a.dtype.kind != "O"
                else all(
                    _same_spec_state(x, y)
                    for x, y in zip(a.tolist(), b.tolist(), strict=True)
                )
            )
        )
    if isinstance(a, float | np.floating):
        return same_bits(np.asarray(a), np.asarray(b))
    if _CONTRASTS_STATE is not None and isinstance(a, _CONTRASTS_STATE):
        fields: tuple[str, ...] = ("contrasts", "levels")
    elif isinstance(a, _CUSTOM_CONTRASTS):
        if hasattr(a, "names") != hasattr(b, "names"):
            return False
        fields = (
            ("contrasts", "contrast_names", "names")
            if hasattr(a, "names")
            else ("contrasts", "contrast_names")
        )
    elif isinstance(a, _POLY_CONTRASTS):
        fields = ("scores",)
    else:
        return bool(a == b)
    return all(
        hasattr(a, f)
        and hasattr(b, f)
        and _same_spec_state(getattr(a, f), getattr(b, f))
        for f in fields
    )


def _same_design_spec(spec_a: object, spec_b: object) -> bool:
    """Return True if two fitted ``formulaic.ModelSpec`` objects match passively."""
    if (
        type(spec_a) is not formulaic.ModelSpec
        or type(spec_b) is not formulaic.ModelSpec
    ):
        return False
    s_a, s_b = cast("Any", spec_a), cast("Any", spec_b)
    attrs_a = (
        s_a.formula,
        s_a.materializer,
        s_a.ensure_full_rank,
        s_a.na_action,
        s_a.output,
        s_a.cluster_by,
        s_a.structure,
        dict(s_a.term_slices),
    )
    attrs_b = (
        s_b.formula,
        s_b.materializer,
        s_b.ensure_full_rank,
        s_b.na_action,
        s_b.output,
        s_b.cluster_by,
        s_b.structure,
        dict(s_b.term_slices),
    )
    return attrs_a == attrs_b and _same_spec_state(
        (s_a.materializer_params, s_a.transform_state, s_a.encoder_state),
        (s_b.materializer_params, s_b.transform_state, s_b.encoder_state),
    )


def _one_exact_type(items: Sequence[object], types: Collection[object]) -> type | None:
    """Return the one exact type, among ``types``, of all ``items``, or ``None``.

    Types are read with ``type`` and compared by identity only, so no method of an
    item or of its class runs; a subclass is not its base.  Only an exact ``tuple``
    or ``list`` has a single-pass stable view; any other sequence or no items give
    ``None``.
    """
    if (type(items) is not tuple and type(items) is not list) or not items:
        return None
    kind = type(items[0])
    if not any(kind is t for t in types):
        return None
    return kind if all(map(operator.is_, map(type, items), repeat(kind))) else None


def _copy_scalar_with_memo(el: object, memo: dict[int, Any]) -> object:
    """Copy one element using ``memo``, keeping immutable built-in scalars as-is."""
    return (
        el if any(type(el) is t for t in _IMMUTABLE_TYPES) else copy.deepcopy(el, memo)
    )


def _windowed_copy_free(items: tuple[object, ...]) -> bool:
    """:func:`_copy_free_labels` of a longer exact tuple, ``_LABEL_WINDOW`` at a time.

    The exact type of every label is read with ``type`` and compared by identity or by
    its id: one identity pass decides a window of one exact immutable built-in type,
    the set of the ids of its labels' exact types any other window.  The labels that
    this leaves open (the exact tuples of a window, or all its labels if it holds one
    of another type) are then checked one by one, in order, as the label-by-label walk
    checks them, so the check ends at the label where that walk ends.  No method of a
    label, of its class or of a metaclass runs.
    """
    for start in range(0, len(items), _LABEL_WINDOW):
        window = items[start : start + _LABEL_WINDOW]
        kind = type(window[0])
        kinds = (
            {id(kind)}
            if all(map(operator.is_, map(type, window), repeat(kind)))
            else set(map(id, map(type, window)))
        )
        if kinds <= _IMMUTABLE_TYPE_IDS:
            continue
        labels = (
            compress(window, map(operator.is_, map(type, window), repeat(tuple)))
            if kinds <= _LABEL_TYPE_IDS
            else window
        )
        for item in labels:
            if id(type(item)) not in _IMMUTABLE_TYPE_IDS and not (
                type(item) is tuple and _copy_free_labels(item)
            ):
                return False
    return True


def _copy_free_labels(items: tuple[object, ...]) -> bool:
    """Whether these exact built-in labels have no deepcopy callbacks or mutability.

    Interaction labels may be tuples of the same scalar leaves. Custom scalars,
    subclasses and mutable containers retain the ordinary copy path, without running
    any of their methods to decide which path to take.

    The current labels are read on every call.  An exact tuple of more than
    ``_SHORT_LABELS`` labels is read a window at a time (:func:`_windowed_copy_free`);
    a shorter one, or any other container, takes the label-by-label walk below.  Both
    check the exact tuples among the labels one by one, in order, with this function.
    """
    if type(items) is tuple and len(items) > _SHORT_LABELS:
        return _windowed_copy_free(items)
    return all(
        id(type(item)) in _IMMUTABLE_TYPE_IDS
        or (type(item) is tuple and _copy_free_labels(item))
        for item in items
    )


def _copy_with_memo(
    items: Sequence[object], memo: dict[int, Any]
) -> tuple[object, ...]:
    """Copy ``items`` using ``memo``, keeping immutable built-in scalars as-is.

    Items all of one immutable built-in type are each their own copy, which leaves
    ``memo`` untouched: a new tuple of the items themselves.  Other items are read in
    one pass: an item of an exact immutable built-in type (:data:`_IMMUTABLE_TYPE_IDS`)
    is kept directly, and any other item falls back to :func:`_copy_scalar_with_memo`.
    """
    if _one_exact_type(items, _IMMUTABLE_TYPES) is not None:
        return tuple(iter(items))
    return tuple(
        el if id(type(el)) in _IMMUTABLE_TYPE_IDS else _copy_scalar_with_memo(el, memo)
        for el in items
    )


def _stored_raw_elements(col: Column) -> tuple[object, ...]:
    """Read stored elements from ``col`` without running ``Column.values`` copy."""
    return (
        col._stored if isinstance(col._stored, tuple) else tuple(col._stored.tolist())
    )


@dataclass(frozen=True, init=False)
class RetainedLabelBatch:
    """Narrowly owned batch of source labels, final labels, and component provenance."""

    component_names: tuple[str, ...]
    _items: tuple[object, ...]
    _source_indices: tuple[IndexArray, ...]
    _final_indices: tuple[IndexArray, ...]
    _component_indices: tuple[IndexArray, ...]
    _component_dtypes: tuple[np.dtype[Any], ...]
    _source_runs: tuple[ComponentRun, ...] | None
    _final_runs: tuple[ComponentRun, ...]
    _component_runs: tuple[ComponentRun, ...]

    def __init__(
        self,
        *,
        component_names: tuple[str, ...],
        items: Sequence[object],
        source_indices: Sequence[IndexArray],
        final_indices: Sequence[IndexArray],
        component_indices: Sequence[IndexArray],
        component_dtypes: Sequence[np.dtype[Any]],
    ) -> None:
        """Store owned index arrays and private copied label representatives."""
        object.__setattr__(self, "component_names", tuple(component_names))
        object.__setattr__(self, "_items", tuple(items))
        object.__setattr__(
            self, "_source_indices", tuple(owned_array(idx) for idx in source_indices)
        )
        object.__setattr__(
            self, "_final_indices", tuple(owned_array(idx) for idx in final_indices)
        )
        object.__setattr__(
            self,
            "_component_indices",
            tuple(owned_array(idx) for idx in component_indices),
        )
        object.__setattr__(self, "_component_dtypes", tuple(component_dtypes))
        object.__setattr__(self, "_source_runs", None)
        object.__setattr__(self, "_final_runs", ())
        object.__setattr__(self, "_component_runs", ())

    @classmethod
    def _from_runs(
        cls, source: Grouping, grouping: Grouping,
    ) -> RetainedLabelBatch | None:
        """Compose full immutable runs only for actual owned-order capture.

        No public constructor accepts a run or an alleged copy-free flag. Source
        and final level representatives are read independently from their actual
        blocks; the complete original component columns are never compressed by
        numeric equality, source level codes or final/source ordinals.
        """
        if (cls is not RetainedLabelBatch or type(source) is not Derived
                or type(grouping) not in (Grouping, Derived)
                or not original_producers() or _copy_with_memo is not _RUN_COPY
                or _copy_free_labels is not _RUN_COPY_FREE
                or _stored_raw_elements is not _RUN_ELEMENTS
                or _native_source_parts is not _RUN_NATIVE_PARTS):
            return None
        origin = source.source
        if type(origin) is not Source:
            return None
        native = _native_source_parts(origin)
        if native is None:
            columns = origin.columns
            runs = getattr(origin, "_component_runs", None)
        else:
            columns, runs = native
        if not matching_runs(runs, columns):
            return None
        source_runs = tuple(run_for(block._levels) for block in source.blocks)
        final_runs = tuple(run_for(block._levels) for block in grouping.blocks)
        if any(run is None for run in (*source_runs, *final_runs)):
            return None
        source_runs = cast("tuple[ComponentRun, ...]", source_runs)
        final_runs = cast("tuple[ComponentRun, ...]", final_runs)
        result = object.__new__(cls)
        for name, value in (
            ("component_names", tuple(runs)), ("_items", ()),
            ("_source_indices", ()), ("_final_indices", ()),
            ("_component_indices", ()),
            ("_component_dtypes", tuple(run.dtype for run in runs.values())),
            ("_source_runs", source_runs), ("_final_runs", final_runs),
            ("_component_runs", tuple(runs.values())),
        ):
            object.__setattr__(result, name, value)
        return result

    def _run_indices(self, part: int) -> tuple[IndexArray, ...]:
        """Original logical flat indices, generated only on actual detailed access."""
        assert self._source_runs is not None
        groups = (self._source_runs, self._final_runs, self._component_runs)
        offset = sum(run.size for group in groups[:part] for run in group)
        result: list[IndexArray] = []
        for run in groups[part]:
            result.append(owned_array(np.arange(offset, offset + run.size,
                                                dtype=np.intp)))
            offset += run.size
        return tuple(result)

    def _run_views(
        self,
    ) -> tuple[
        tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...],
        dict[str, ComponentRun],
    ]:
        """Only small level tuples plus full-column run handles, never row gathers."""
        assert self._source_runs is not None
        def levels(run: ComponentRun) -> tuple[object, ...]:
            return run.objects()

        return (
            tuple(levels(run) for run in self._source_runs),
            tuple(levels(run) for run in self._final_runs),
            dict(zip(self.component_names, self._component_runs, strict=True)),
        )

    @property
    def source_indices(self) -> tuple[IndexArray, ...]:
        """Return fresh owned index slices for each source block."""
        if self._source_runs is not None:
            return self._run_indices(0)
        return tuple(owned_array(idx) for idx in self._source_indices)

    @property
    def final_indices(self) -> tuple[IndexArray, ...]:
        """Return fresh owned index slices for each final block."""
        if self._source_runs is not None:
            return self._run_indices(1)
        return tuple(owned_array(idx) for idx in self._final_indices)

    @property
    def component_indices(self) -> tuple[IndexArray, ...]:
        """Return fresh owned index slices for each derived component column."""
        if self._source_runs is not None:
            return self._run_indices(2)
        return tuple(owned_array(idx) for idx in self._component_indices)

    def _slices(
        self, items: tuple[object, ...]
    ) -> tuple[
        tuple[tuple[object, ...], ...],
        tuple[tuple[object, ...], ...],
        dict[str, tuple[object, ...]],
    ]:
        """The source, final and component slices of ``items`` by the stored indices."""
        take = items.__getitem__

        def gather(idx: IndexArray) -> tuple[object, ...]:
            if type(idx) is np.ndarray and idx.ndim == 1 and idx.dtype.kind in "iu":
                return tuple(map(take, idx.tolist()))
            return tuple(items[int(i)] for i in idx.tolist())

        src = tuple(gather(idx) for idx in self._source_indices)
        fin = tuple(gather(idx) for idx in self._final_indices)
        comp = {
            name: gather(idx)
            for name, idx in zip(
                self.component_names, self._component_indices, strict=True
            )
        }
        return src, fin, comp

    def _stored_slices(
        self,
    ) -> tuple[
        tuple[tuple[object, ...], ...],
        tuple[tuple[object, ...], ...],
        dict[str, tuple[object, ...]],
    ]:
        """The slices of the stored items themselves, uncopied.

        Private to the internally owned grouping views of
        :class:`~powermean.results.absorbed_partitions.RetainedPartitionSnapshot`, which
        this batch belongs to: the items are the batch's own copies, made once at its
        construction with one memo, and those views hold them privately and copy them
        anew with one memo on every outward exposure (:meth:`materialize` and the views'
        ``rematerialize``).  Nothing of this is handed outward.
        """
        if self._source_runs is not None:
            src, fin, runs = self._run_views()
            return src, fin, {name: run.objects() for name, run in runs.items()}
        return self._slices(self._items)

    @property
    def _copy_free(self) -> bool:
        """Check the actual owned labels, not a cached eligibility assertion."""
        if self._source_runs is not None:
            return (original_producers()
                    and _copy_with_memo is _RUN_COPY
                    and _copy_free_labels is _RUN_COPY_FREE
                    and _stored_raw_elements is _RUN_ELEMENTS
                    and all(type(run) is ComponentRun and run.bound()
                            for group in (self._source_runs, self._final_runs,
                                          self._component_runs) for run in group))
        return _copy_free_labels(self._items)

    def materialize(
        self, memo: dict[int, Any] | None = None
    ) -> tuple[
        tuple[tuple[object, ...], ...],
        tuple[tuple[object, ...], ...],
        dict[str, tuple[object, ...]],
        dict[str, np.dtype[Any]],
    ]:
        """Materialize one complete compound view using one fresh local memo."""
        active_memo: dict[int, Any] = {} if memo is None else memo
        if self._source_runs is not None:
            src, fin, components = self._stored_slices()
            return (
                tuple(_copy_with_memo(items, active_memo) for items in src),
                tuple(_copy_with_memo(items, active_memo) for items in fin),
                {name: _copy_with_memo(items, active_memo)
                 for name, items in components.items()},
                self._dtypes,
            )
        src, fin, comp = self._slices(_copy_with_memo(self._items, active_memo))
        dtypes = dict(zip(self.component_names, self._component_dtypes, strict=True))
        return src, fin, comp, dtypes

    @property
    def _dtypes(self) -> dict[str, np.dtype[Any]]:
        """The dtype of every component column, by name (internal views)."""
        return dict(zip(self.component_names, self._component_dtypes, strict=True))


_RUN_COPY = _copy_with_memo
_RUN_COPY_FREE = _copy_free_labels
_RUN_ELEMENTS = _stored_raw_elements
_RUN_NATIVE_PARTS = _native_source_parts


def build_label_batch(
    source: Grouping,
    grouping: Grouping,
    *,
    memo: dict[int, Any] | None = None,
) -> RetainedLabelBatch:
    """Extract stored representatives from ``source`` and ``grouping`` into a batch."""
    raw_items: list[object] = []
    src_indices: list[IndexArray] = []
    fin_indices: list[IndexArray] = []
    comp_names: list[str] = []
    comp_indices: list[IndexArray] = []
    comp_dtypes: list[np.dtype[Any]] = []

    for blk in source.blocks:
        elems = _stored_raw_elements(blk._levels)
        start = len(raw_items)
        raw_items.extend(elems)
        src_indices.append(owned_array(np.arange(start, len(raw_items), dtype=np.intp)))

    for blk in grouping.blocks:
        elems = _stored_raw_elements(blk._levels)
        start = len(raw_items)
        raw_items.extend(elems)
        fin_indices.append(owned_array(np.arange(start, len(raw_items), dtype=np.intp)))

    if isinstance(source, Derived):
        for name, col in source.source.columns.items():
            elems = _stored_raw_elements(col)
            start = len(raw_items)
            raw_items.extend(elems)
            comp_names.append(name)
            comp_indices.append(
                owned_array(np.arange(start, len(raw_items), dtype=np.intp))
            )
            comp_dtypes.append(col.dtype)

    active_memo: dict[int, Any] = {} if memo is None else memo
    return RetainedLabelBatch(
        component_names=tuple(comp_names),
        items=_copy_with_memo(raw_items, active_memo),
        source_indices=tuple(src_indices),
        final_indices=tuple(fin_indices),
        component_indices=tuple(comp_indices),
        component_dtypes=tuple(comp_dtypes),
    )
