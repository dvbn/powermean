"""The compact values of a primitive capture, as the fresh live sample check reads them.

Internal to the primitive snapshot route of
:class:`~powermean.interface.retained_context.PerFitPreparationContext`; the general
route, the package default, uses nothing here.

A :class:`~powermean.results.primitive_snapshot.PrimitiveCapture` is bytes, plain
tuples and plain scalars.  :class:`CompactCapture` is that capture as its one fresh
consumer, :func:`~powermean.results.primitive_sample_check.check_compact_sample`,
compares it with a live sample: every numerical buffer by its shape and its bytes, so
no array of the capture is serialized again on a check; every object column by its
plain tuple and its recorded kind (:class:`_PlainColumn`), so direct tuple comparisons
do not rebuild a column or rescan the capture's kind; every numeric column by dtype,
shape, bytes and one read-only view (:class:`_TypedColumn`). It is formed once per fit
(:meth:`CompactCapture.of`): immutable data derived from the immutable capture, held
by the context's :class:`~powermean.results.primitive_parts.PrimitiveParts` and shared
by nothing outward. Numerical arrays view immutable bytes. Object labels remain in
immutable tuples: an object ndarray's read-only flag cannot make its owner immutable.
Consumers needing an array operand form fresh working arrays from those tuples; those
arrays are derivatives, never retained comparison authority. This allocation remains
inside fresh validation and hence inside any complete-fit measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from powermean._prepare.inputs import Columns
from powermean.data.accounting import CodeArray
from powermean.data.raw import AnyArray, _object_array
from powermean.model.order import BoolArray
from powermean.results.owned import owned_array
from powermean.results.primitive_snapshot import (
    PrimitiveCapture,
    _BaseEnc,
    _Buffer,
    _ColumnEnc,
    _EffectsEnc,
    _Plain,
)

__all__ = ["CompactCapture"]

# The exact plain type of a recorded kind (:data:`primitive_snapshot._PLAIN_NAMES`).
_KINDS: dict[str, type] = {"bool": bool, "int": int, "float": float, "str": str}


@dataclass(frozen=True, eq=False)
class _PlainColumn:
    """One object column of the capture: its plain kind and immutable label tuple.

    ``kind`` is the recorded name of the one exact plain type of the labels
    (:data:`_KINDS`); ``labels`` is the capture's own immutable tuple. ``array`` forms
    disposable working storage, never an owner shared with a future comparison.
    """

    kind: str
    labels: tuple[object, ...]

    @property
    def array(self) -> AnyArray:
        """A fresh working operand; mutating it cannot change the stored tuple."""
        return _object_array(self.labels)

    @property
    def plain_type(self) -> type:
        """The exact plain type of the labels."""
        return _KINDS[self.kind]


@dataclass(frozen=True, eq=False)
class _TypedColumn:
    """One numeric column of the capture: its dtype, shape, bytes and one view."""

    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    data: bytes
    view: AnyArray


type _CompactColumn = _PlainColumn | _TypedColumn


def _compact_column(enc: _ColumnEnc) -> _CompactColumn:
    """The compact form of one encoded column."""
    if type(enc) is _Plain:
        return _PlainColumn(enc.kind, enc.labels)
    buffer = cast("_Buffer", enc)
    return _TypedColumn(
        np.dtype(buffer.dtype), buffer.shape, buffer.data, buffer.view()
    )


def _values_of(column: _CompactColumn) -> AnyArray:
    """The array a ``Column`` of the capture would hand out: the generic operand."""
    return (
        column.array
        if type(column) is _PlainColumn
        else cast("_TypedColumn", column).view
    )


@dataclass(frozen=True, eq=False)
class _CompactEffects:
    """The effects of the capture: terms, names, rows and compact component columns."""

    terms: tuple[tuple[str, ...], ...]
    names: tuple[str, ...]
    rows: int
    columns: dict[str, _CompactColumn]
    column_names: tuple[str, ...]
    column_set: frozenset[str]

    @classmethod
    def of(cls, enc: _EffectsEnc) -> _CompactEffects:
        """The compact effects of an encoded ``EffectColumns``."""
        columns = {name: _compact_column(col) for name, col in enc.columns}
        return cls(
            terms=enc.terms,
            names=tuple("^".join(term) for term in enc.terms),
            rows=enc.rows,
            columns=columns,
            column_names=tuple(columns),
            column_set=frozenset(columns),
        )


@dataclass(frozen=True, eq=False)
class CompactCapture:
    """The compact values of one capture as the live sample check reads them.

    Formed once per context by :meth:`of`; immutable data derived from the immutable
    capture and shared by nothing outward.

    Attributes
    ----------
    base : _BaseEnc
        The encoded base: buffers by dtype string, shape and bytes.
    columns : Columns
        The columns record of the preparation.
    labels_size, nobs : int
        The number of row labels and of input rows.
    y, design, offset, exposure, complete : array
        Read-only views of the buffers that the check gathers at the sample's rows.
    excluded_rows, excluded_codes : array
        The rows that steps 1 and 2 excluded and the base's codes of those rows.
    missing : tuple
        Per missing group: its name, its dtype and its buffer.
    missing_names : tuple of str
        The group names, in the capture's order.
    excluded : tuple
        The dtype and the buffer of the union of the groups.
    weights : tuple or None
        The compact weights column and its kind, or ``None`` without weights.
    metadata : dict
        The compact metadata columns, by name, in the capture's order.
    metadata_names, metadata_set : tuple, frozenset
        The metadata names, in order and as a set.
    effects : _CompactEffects or None
        The compact effects, or ``None`` without a fixed-effect part.
    """

    base: _BaseEnc
    columns: Columns
    labels_size: int
    nobs: int
    y: AnyArray
    design: AnyArray
    offset: AnyArray | None
    exposure: AnyArray | None
    complete: BoolArray
    excluded_rows: BoolArray
    excluded_codes: CodeArray
    missing: tuple[tuple[str, np.dtype[Any], _Buffer], ...]
    missing_names: tuple[str, ...]
    excluded: tuple[np.dtype[Any], _Buffer]
    weights: tuple[_CompactColumn, str] | None
    metadata: dict[str, _CompactColumn]
    metadata_names: tuple[str, ...]
    metadata_set: frozenset[str]
    effects: _CompactEffects | None

    @classmethod
    def of(cls, capture: PrimitiveCapture) -> CompactCapture:
        """The compact values of ``capture``."""
        if type(capture) is not PrimitiveCapture:
            raise TypeError("capture must be a PrimitiveCapture")
        base = capture.base
        complete = cast("BoolArray", base.complete.view())
        excluded_rows = cast("BoolArray", owned_array(~complete))
        codes = cast("CodeArray", base.codes.view())
        metadata = {name: _compact_column(col) for name, col in base.metadata}
        return cls(
            base=base,
            columns=capture.columns_record(),
            labels_size=capture.labels.size,
            nobs=int(base.codes.shape[0]),
            y=base.y.view(),
            design=base.design.view(),
            offset=None if base.offset is None else base.offset.view(),
            exposure=None if base.exposure is None else base.exposure.view(),
            complete=complete,
            excluded_rows=excluded_rows,
            excluded_codes=cast("CodeArray", owned_array(codes[excluded_rows])),
            missing=tuple(
                (group, np.dtype(mask.dtype), mask)
                for group, mask in base.missing_groups
            ),
            missing_names=tuple(group for group, _ in base.missing_groups),
            excluded=(np.dtype(base.missing_excluded.dtype), base.missing_excluded),
            weights=None
            if base.weights is None
            else (_compact_column(base.weights[0]), base.weights[1]),
            metadata=metadata,
            metadata_names=tuple(metadata),
            metadata_set=frozenset(metadata),
            effects=None
            if capture.effects is None
            else _CompactEffects.of(capture.effects),
        )


def _column_values(column: _CompactColumn) -> tuple[AnyArray, type | None]:
    """``_label_values`` of the capture's column: its array and its plain kind."""
    if type(column) is _PlainColumn:
        return column.array, column.plain_type
    return cast("_TypedColumn", column).view, None
