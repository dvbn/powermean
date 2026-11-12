"""Private immutable component runs shared by FE encoding and owned capture.

Runs retain every original row and its actual already-owned scalar/tuple objects.
They never compress by label equality or replace source/final representatives.
The ingress checks genuine immutable storage once; reuse checks the current
Column storage/header binding. Public and callback-bearing construction keeps its
existing ordinary ownership path. No live frame, operation or caller memo is held.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Mapping
from itertools import repeat
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from powermean.data.raw import _IMMUTABLE_TYPES, Column

_IMMUTABLE_IDS = frozenset(map(id, _IMMUTABLE_TYPES))


def _immutable_items(items: tuple[object, ...]) -> bool:
    """Inspect exact immutable leaves without invoking any label operation."""
    if not items:
        return True
    kind = type(items[0])
    if id(kind) in _IMMUTABLE_IDS and all(
        map(operator.is_, map(type, items), repeat(kind))
    ):
        return True
    return all(
        id(type(item)) in _IMMUTABLE_IDS
        or (type(item) is tuple and _immutable_items(item))
        for item in items
    )


def _immutable_array(array: np.ndarray) -> bool:
    """Only a plain one-dimensional primitive view rooted in immutable bytes."""
    if (type(array) is not np.ndarray or array.ndim != 1
            or array.dtype.kind not in "biufcUS" or array.dtype.fields is not None
            or array.dtype.metadata is not None or array.flags.writeable):
        return False
    root: object = array
    while type(root) is np.ndarray:
        if root.flags.writeable:
            return False
        root = root.base
    return type(root) is bytes


def _array_header(array: np.ndarray) -> tuple[object, ...]:
    """The immutable relevant header description, including strides and offset."""
    return (array.shape, array.strides, array.dtype,
            array.__array_interface__["data"][0])


class ComponentRun:
    """One immutable storage run with its original owner/header description.

    Construction is private to ``_run`` after the real storage admission. The
    record cannot replace its owner/view/dtype/shape fields. Object storage
    remains the original exact immutable tuple, including aliases and Python
    types; numeric storage has a separate header on the same immutable bytes.
    """

    __slots__ = ("_dtype", "_header", "_owner", "_stored", "_presented", "_column")

    _owner: tuple[object, ...] | np.ndarray
    _stored: tuple[object, ...] | np.ndarray
    _dtype: np.dtype[Any]
    _header: tuple[object, ...] | None
    _presented: Mapping[int, object]
    _column: tuple[Column, tuple[object, ...]] | None

    def __new__(cls, *_args: object) -> ComponentRun:
        """No public constructor accepts an alleged eligibility assertion."""
        raise TypeError("component runs are made from admitted Column storage")

    def __setattr__(self, _name: str, _value: object) -> None:
        """Factory-owned immutable fields, not an alleged tuple proof payload."""
        raise AttributeError("component runs are immutable")

    def __deepcopy__(self, memo: dict[int, Any]) -> ComponentRun:
        """Detach this outward run and its headers, preserving graph memo aliases.

        Numeric bytes are shared only while genuinely immutable; writable or
        foreign storage retains ordinary deepcopy. Tuple leaves and existing
        memo entries keep the same compound aliases as the enclosing detacher.
        No live source/Column owner or mutable array header is returned as self.
        """
        if id(self) in memo:
            return cast("ComponentRun", memo[id(self)])
        result = object.__new__(ComponentRun)
        memo[id(self)] = result
        for name in self.__slots__:
            value = object.__getattribute__(self, name)
            if name == "_presented":
                object.__setattr__(result, name, MappingProxyType({
                    row: (memo.get(id(item), item) if id(type(item)) in _IMMUTABLE_IDS
                          else copy.deepcopy(item, memo))
                    for row, item in value.items()
                }))
                continue
            if (id(value) not in memo and type(value) is np.ndarray
                    and _immutable_array(value)):
                memo[id(value)] = value.view()
            object.__setattr__(result, name, copy.deepcopy(value, memo))
        return result

    @property
    def values(self) -> tuple[object, ...] | np.ndarray:
        """Actual immutable storage; private numeric consumers do not export it."""
        stored = self._stored
        return (stored.view() if type(stored) is np.ndarray else
                cast("tuple[object, ...]", stored))

    @property
    def dtype(self) -> np.dtype[Any]:
        """The unchanged original component dtype."""
        return self._dtype

    @property
    def size(self) -> int:
        """The full original row count, not just the active source rows."""
        return len(self.values)

    def bound(self) -> bool:
        """Check the actual immutable storage/header relation, with no row walk."""
        if type(self) is not ComponentRun or not hasattr(self, "_owner"):
            return False
        current, values = self._owner, self._stored
        presentation = self._column
        if presentation is not None and (
            type(presentation[0]) is not _COLUMN_KIND
            or presentation[0]._stored is not presentation[1]
        ):
            return False
        if type(current) is tuple:
            return (values is current and self.dtype == np.dtype(object)
                    and self._header is None)
        header = self._header
        return (type(current) is np.ndarray and type(values) is np.ndarray
                and _array_header(current) == header
                and _array_header(values) == header
                and (current.dtype == self.dtype or (
                    self.dtype == np.dtype(object) and current.dtype.kind in "iu"
                ))
                and _immutable_array(current) and _immutable_array(values))

    def matches(self, column: Column) -> bool:
        """Bind current storage/header, not a remembered verdict on mutable state."""
        return (type(column) is Column and self.bound()
                and column._stored is self._owner
                and (column.dtype == self.dtype or (
                    self.dtype == np.dtype(object) and column.dtype.kind in "iu"
                )))

    def labels(self, rows: np.ndarray) -> list[object]:
        """Present these ACTUAL original rows, not equal-level representatives.

        Only the integral formula owner uses this method. Level construction asks
        for a subset; retain those actual immutable objects until a full public
        Column is requested, so source/final/interaction memo aliases agree.
        """
        presentation = self._column
        indices = rows.tolist()
        if presentation is not None:
            return [presentation[0]._stored[row] for row in indices]
        values = cast("np.ndarray", self._stored)[rows].tolist()
        prior = self._presented
        if not prior:
            # This first presentation has no prior identities to reconcile. The
            # same first-row objects serve source blocks and later rich exports.
            if indices:
                object.__setattr__(self, "_presented", MappingProxyType(
                    dict(zip(indices, values, strict=True))
                ))
            return values
        added = {}
        result = []
        for row, value in zip(indices, values, strict=True):
            actual = prior.get(row, value)
            result.append(actual)
            if row not in prior:
                added[row] = actual
        if added:
            object.__setattr__(self, "_presented",
                               MappingProxyType({**prior, **added}))
        return result

    def object_column(self) -> Column:
        """One record-owned logical object Column, only on actual presentation.

        Its integers were always immutable Python objects in the old ingress.
        Build the same plain tuple directly, without a live constructor/copy hook
        that the old mapping getter never called. Generic Column APIs stay intact.
        """
        presentation = self._column
        if presentation is not None:
            return presentation[0]
        values = cast("np.ndarray", self._stored).tolist()
        for row, value in self._presented.items():
            values[row] = value
        items = tuple(values)
        column = object.__new__(_COLUMN_KIND)
        object.__setattr__(column, "_stored", items)
        object.__setattr__(self, "_column", (column, items))
        object.__setattr__(self, "_presented", MappingProxyType({}))
        return column

    def objects(self) -> tuple[object, ...]:
        """Logical component objects at the existing detailed export boundary."""
        values = self.values
        if type(values) is tuple:
            return values
        if self.dtype == np.dtype(object):
            return cast("tuple[object, ...]", self.object_column()._stored)
        return tuple(values.tolist())

    def discard_presentation(self) -> None:
        """Abandon optional first-build representatives before old construction."""
        if self._column is None:
            object.__setattr__(self, "_presented", MappingProxyType({}))


def _run(column: Column) -> ComponentRun | None:
    """Admit actual owned storage once; unknown hooks keep their old path."""
    if type(column) is not Column:
        return None
    stored = column._stored
    if type(stored) is tuple:
        if not _immutable_items(stored):
            return None
        values: tuple[object, ...] | np.ndarray = stored
    elif type(stored) is np.ndarray and _immutable_array(stored):
        values = stored.view()
    else:
        return None
    header = None if type(stored) is tuple else _array_header(stored)
    result = object.__new__(ComponentRun)
    for name, value in (("_owner", stored), ("_stored", values),
                        ("_dtype", column.dtype), ("_header", header),
                        ("_presented", MappingProxyType({})), ("_column", None)):
        object.__setattr__(result, name, value)
    return result


def native_columns(
    columns: Mapping[str, Column],
) -> tuple[dict[str, Column], dict[str, ComponentRun]] | None:
    """Fresh headers on actual immutable integral owners, no full object columns."""
    if not original_producers():
        return None
    found: dict[str, Column] = {}
    runs: dict[str, ComponentRun] = {}
    for name, column in columns.items():
        if type(column) is not Column:
            return None
        stored = column._stored
        if not _immutable_array(stored) or stored.dtype.kind not in "iu":
            return None
        owned = object.__new__(_COLUMN_KIND)
        object.__setattr__(owned, "_stored", stored.view())
        run = _run(owned)
        assert run is not None
        object.__setattr__(run, "_dtype", np.dtype(object))
        found[name], runs[name] = owned, run
    return found, runs


def object_columns(
    columns: Mapping[str, Column], runs: Mapping[str, ComponentRun],
) -> dict[str, Column]:
    """Exact logical fields; replaced/foreign storage keeps current checked reads."""
    return {
        name: (runs[name].object_column() if runs[name].matches(column)
               else Column(np.asarray(column.values, dtype=object)))
        for name, column in columns.items()
    }


def run_for(column: Column) -> ComponentRun | None:
    """One actual owned run, or the original callback-bearing construction route."""
    return _run(column) if original_producers() else None


def runs_for(columns: Mapping[str, Column]) -> dict[str, ComponentRun] | None:
    """One source's complete runs, never a cross-fit registry or caller cache."""
    if not original_producers():
        return None
    result: dict[str, ComponentRun] = {}
    for name, column in columns.items():
        if type(name) is not str:
            return None
        run = _run(column)
        if run is None:
            return None
        result[name] = run
    return result


def matching_runs(
    runs: Mapping[str, ComponentRun] | None, columns: Mapping[str, Column],
) -> bool:
    """Require the actual same full-column owners and unchanged current headers."""
    return (original_producers() and type(runs) is dict
            and type(columns) in (dict, MappingProxyType)
            and all(type(name) is str for name in runs)
            and all(type(name) is str for name in columns)
            and runs.keys() == columns.keys()
            and all(type(run) is ComponentRun and run.matches(columns[name])
                    for name, run in runs.items()))


def original_producers() -> bool:
    """Changed private producers retain the original checked construction route."""
    return (all(globals()[name] is value for name, value in _PRODUCERS)
            and ComponentRun.__new__ is _RUN_NEW
            and ComponentRun.__setattr__ is _RUN_SET
            and vars(ComponentRun).get("values") is _RUN_VALUES
            and vars(ComponentRun).get("dtype") is _RUN_DTYPE
            and vars(ComponentRun).get("size") is _RUN_SIZE
            and ComponentRun.bound is _RUN_BOUND
            and ComponentRun.matches is _RUN_MATCHES
            and ComponentRun.labels is _RUN_LABELS
            and ComponentRun.object_column is _RUN_OBJECT_COLUMN
            and ComponentRun.objects is _RUN_OBJECTS
            and ComponentRun.discard_presentation is _RUN_DISCARD
            and Column is _COLUMN_KIND
            and vars(Column).get("dtype") is _COLUMN_DTYPE
            and all(vars(Column).get(name) is original
                    for name, original in _COLUMN_METHODS))


_PRODUCERS = tuple(
    (name, globals()[name])
    for name in (
        "ComponentRun", "_immutable_items", "_immutable_array", "_array_header",
        "_run", "run_for", "runs_for", "matching_runs", "native_columns",
        "object_columns",
    )
)
_RUN_NEW = ComponentRun.__new__
_RUN_SET = ComponentRun.__setattr__
_RUN_VALUES = vars(ComponentRun)["values"]
_RUN_DTYPE = vars(ComponentRun)["dtype"]
_RUN_SIZE = vars(ComponentRun)["size"]
_RUN_BOUND = ComponentRun.bound
_RUN_MATCHES = ComponentRun.matches
_COLUMN_DTYPE = vars(Column)["dtype"]

_RUN_LABELS = ComponentRun.labels
_RUN_OBJECT_COLUMN = ComponentRun.object_column
_RUN_OBJECTS = ComponentRun.objects
_RUN_DISCARD = ComponentRun.discard_presentation
_COLUMN_KIND = Column

_COLUMN_METHODS = tuple((name, vars(Column).get(name)) for name in (
    "__new__", "__init__", "__getattribute__", "__getattr__",
    "stored", "values", "size", "take",
))
