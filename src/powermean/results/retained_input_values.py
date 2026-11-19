"""Owned input-value helpers, result-local keys, and RetainedInputSnapshot.

``design/02_computation.md``, Sections 2 and 7; ``plan/evidence/M1.15_contract.txt``,
items 1, 3 and 4; ``brief/layout_C_ownership.txt`` (P01).
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast, overload

import numpy as np
import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.dependence import ColumnProvenance, DependenceProvenance
from powermean._prepare.inputs import Columns
from powermean._prepare.owned import _owned_frame, _owned_index
from powermean.data.design import ScaledDesign
from powermean.data.raw import (
    AlignedColumns,
    AnyArray,
    Column,
    RawInputs,
    RawWeights,
    _object_array,
    _object_column,
)
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_source import same_bits
from powermean.data.sample_fe_state import RowLedger
from powermean.formula.frame import Design, ModelFrame
from powermean.model.order import IndexArray, Selection, TransformedProblem
from powermean.model.order_scale import OrderScale
from powermean.results.owned import _owned_missing, owned_array
from powermean.results.retained_labels import (
    RetainedLabelBatch,
    _copy_scalar_with_memo,
    _copy_with_memo,
    _one_exact_type,
    _stored_raw_elements,
    build_label_batch,
)

InputKey = int
SampleKey = int
PartitionKey = int
GroupingKey = int
PointKey = int
ProblemKey = int

# The exact types of plain labels (:func:`_plain_kind`); a subclass is not one of them.
_PLAIN_TYPES = frozenset({bool, int, float, str})

__all__ = [
    "GroupingKey",
    "InputKey",
    "PartitionKey",
    "PointKey",
    "ProblemKey",
    "RetainedInputSnapshot",
    "RetainedLabelBatch",
    "SampleKey",
    "_copy_column",
    "_copy_scalar_with_memo",
    "_copy_with_memo",
    "_is_missing_scalar",
    "_label_values",
    "_owned_column",
    "_owned_column_provenance",
    "_owned_column_values",
    "_owned_dependence",
    "_owned_dependence_provenance",
    "_owned_effect_columns",
    "_owned_fixed_effects",
    "_owned_missing",
    "_owned_model_frame",
    "_owned_prepared_frame",
    "_owned_raw_inputs",
    "_owned_row_ledger",
    "_owned_sample_base",
    "_owned_scaled_design",
    "_owned_transformed_problem",
    "_plain_kind",
    "_same_column_values",
    "_stored_raw_elements",
    "_tuple_arrays",
    "build_label_batch",
    "owned_order_scale",
]


def _tuple_arrays[ArrayT: AnyArray](arrays: Sequence[ArrayT]) -> tuple[ArrayT, ...]:
    """Return a tuple of fresh owned copies of ``arrays``."""
    return tuple(cast("ArrayT", owned_array(a)) for a in arrays)


@overload
def owned_order_scale(order: OrderScale) -> OrderScale: ...
@overload
def owned_order_scale(order: None) -> None: ...
def owned_order_scale(order: OrderScale | None) -> OrderScale | None:
    """Return a fresh :class:`OrderScale` with an owned scale vector, or ``None``."""
    return None if order is None else OrderScale(order.omega, owned_array(order.scale))


def _is_missing_scalar(value: object) -> bool:
    """Return True if ``value`` is a missing marker or nonfinite float."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if type(value) is np.longdouble:
        return not bool(np.isfinite(value))
    if isinstance(value, float | np.floating):
        return not math.isfinite(float(value))
    if isinstance(value, np.datetime64 | np.timedelta64):
        return bool(np.isnat(value))
    return bool(operator.ne(value, value))


def _plain_kind(values: Sequence[Any]) -> type | None:
    """Return the one exact type of ``_PLAIN_TYPES`` of all ``values``, or ``None``.

    Floats must be finite: no value of a kind is missing for :func:`_is_missing_scalar`,
    and ``==`` between values of one kind runs no caller code and is exact.  A mixture
    of types, a subclass, any other value and an empty sequence have no kind; types are
    compared by identity (:func:`_one_exact_type`).
    """
    kind = _one_exact_type(values, _PLAIN_TYPES)
    if kind is float and not all(map(math.isfinite, values)):
        return None
    return kind


def _label_values(col: Column) -> tuple[AnyArray, type | None]:
    """Return ``col.values`` and the plain kind of its labels, or ``None``.

    Labels of one plain kind (:func:`_plain_kind`) are immutable built-in values, their
    own copies, so they are read from the column's storage: the very objects that
    ``col.values`` hands out, without copying them.  Any other column is ``col.values``.
    """
    if type(col) is Column:
        stored = col._stored
        if type(stored) is tuple and (kind := _plain_kind(stored)) is not None:
            return np.fromiter(stored, dtype=object, count=len(stored)), kind
    return col.values, None


def _same_column_values(left: AnyArray, right: AnyArray) -> bool:
    """Compare two 1D arrays element by element, treating missing markers as equal.

    Two object vectors of one plain kind (:func:`_plain_kind`) have no missing value and
    one type throughout, so they are compared as one list.
    """
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind != "O":
        return same_bits(left, right)
    l_list, r_list = left.tolist(), right.tolist()
    kind = _plain_kind(l_list) if left.ndim == 1 else None
    if kind is not None and kind is _plain_kind(r_list):
        return bool(l_list == r_list)
    for a, b in zip(l_list, r_list, strict=True):
        if _is_missing_scalar(a) and _is_missing_scalar(b):
            continue
        if type(a) is not type(b) or a != b:
            return False
    return True


def _owned_column_values(col: Column, memo: dict[int, Any] | None = None) -> AnyArray:
    """Return a fresh array of column values with copied object elements when needed."""
    if isinstance(col._stored, tuple):
        copied = _copy_with_memo(col._stored, {} if memo is None else memo)
        return _object_array(list(copied))
    return owned_array(col.values)


def _copy_column(col: Column, memo: dict[int, Any] | None = None) -> Column:
    """Create a fresh ``Column`` with a distinct numeric array or copied objects."""
    if type(col) is Column and type(col._stored) is tuple:
        copied = _copy_with_memo(col._stored, {} if memo is None else memo)
        return _object_column(copied)
    return Column(_owned_column_values(col, memo))


_owned_column = _copy_column


def _owned_raw_inputs(inputs: RawInputs) -> RawInputs:
    """Create a deeply owned ``RawInputs`` with fresh numeric and object columns."""
    weights: RawWeights | None = None
    if inputs.weights is not None:
        w_vals: Any = _owned_column_values(inputs.weights.column)
        weights = RawWeights(w_vals, inputs.weights.kind)
    memo: dict[int, Any] = {}
    meta: dict[str, Any] = {
        k: _copy_column(c, memo).values for k, c in inputs.metadata.columns.items()
    }
    return RawInputs(
        y=owned_array(inputs.y),
        design=owned_array(inputs.design),
        column_names=inputs.column_names,
        weights=weights,
        offset=None if inputs.offset is None else owned_array(inputs.offset),
        exposure=None if inputs.exposure is None else owned_array(inputs.exposure),
        metadata=meta,
    )


def _owned_sample_base(base: SampleBase) -> SampleBase:
    """Create a deeply owned ``SampleBase`` with fresh array objects for all masks."""
    return SampleBase(
        inputs=_owned_raw_inputs(base.inputs),
        subset=None if base.subset is None else owned_array(base.subset),
        missing=_owned_missing(base.missing),
        complete=owned_array(base.complete),
        negative_weight=owned_array(base.negative_weight),
        zero_weight=owned_array(base.zero_weight),
        codes=owned_array(base.codes),
    )


def _owned_model_frame(frame: ModelFrame) -> ModelFrame:
    """Create a deeply owned ``ModelFrame`` including its boolean invalid masks."""
    copied = _owned_frame(frame)
    d = copied.design
    design = Design(
        d.matrix,
        d.spec,
        d.terms,
        d.variables,
        d.labels,
        owned_array(d.referenced_invalid),
        owned_array(d.evaluated_invalid),
        d.formulaic_version,
        d._contrast_spec,
    )
    return ModelFrame(
        copied.parts, copied.outcome, owned_array(copied.outcome_invalid), design
    )


def _owned_effect_columns(effects: EffectColumns | None) -> EffectColumns | None:
    """Create a deeply owned ``EffectColumns`` keeping cross-column label identity."""
    if effects is None:
        return None
    memo: dict[int, Any] = {}
    cols = {k: _copy_column(c, memo) for k, c in effects.columns.items()}
    return EffectColumns(terms=effects.terms, columns=cols, rows=effects.rows)


_owned_fixed_effects = _owned_effect_columns


def _owned_column_provenance(
    p: ColumnProvenance, memo: dict[int, Any]
) -> ColumnProvenance:
    """Create a deeply owned ``ColumnProvenance`` using ``memo`` for object labels."""
    vals = (
        _object_array(
            list(_copy_with_memo(_stored_raw_elements(p._column_record), memo))
        )
        if p._column_record.dtype.kind == "O"
        else owned_array(p._column_record.values)
    )
    return ColumnProvenance(
        name=p.name,
        roles=p.roles,
        used_in_fit=p.used_in_fit,
        source_dtype=p.source_dtype,
        pandas_kind=p.pandas_kind,
        missing_mask=owned_array(p.missing_mask),
        values=vals,
    )


def _owned_dependence_provenance(
    dependence: DependenceProvenance | None,
) -> DependenceProvenance | None:
    """Create a deeply owned ``DependenceProvenance`` with fresh column records."""
    if dependence is None:
        return None
    memo: dict[int, Any] = {}
    cols = {k: _owned_column_provenance(p, memo) for k, p in dependence.columns.items()}
    return DependenceProvenance(
        used=dependence.used,
        columns=cols,
        used_missing_union=owned_array(dependence.used_missing_union),
    )


_owned_dependence = _owned_dependence_provenance


def _owned_prepared_frame(prepared: PreparedFrame) -> PreparedFrame:
    """Create a deeply owned ``PreparedFrame`` isolating all arrays, masks, and axes."""
    return PreparedFrame(
        frame=_owned_model_frame(prepared.frame),
        base=_owned_sample_base(prepared.base),
        labels=_owned_index(prepared.labels),
        columns=prepared.columns,
        subset_missing=prepared.subset_missing,
        effects=_owned_effect_columns(prepared.effects),
        dependence=_owned_dependence_provenance(prepared.dependence),
    )


def _owned_row_ledger(ledger: RowLedger) -> RowLedger:
    """Create a deeply owned ``RowLedger`` with fresh array objects."""
    return RowLedger(
        codes=owned_array(ledger.codes),
        blocks=owned_array(ledger.blocks),
        positions=owned_array(ledger.positions),
        retained_index=owned_array(ledger.retained_index),
        missing=_owned_missing(ledger.missing),
        replicas=None if ledger.replicas is None else owned_array(ledger.replicas),
    )


def _owned_transformed_problem(problem: TransformedProblem) -> TransformedProblem:
    """Create a deeply owned ``TransformedProblem`` with fresh array objects."""
    s = problem.selection
    return replace(
        problem,
        response=owned_array(problem.response),
        offset=owned_array(problem.offset),
        selection=Selection(
            owned_array(s.mask),
            owned_array(s.positions),
            owned_array(s.removed),
            s.nobs,
            s.positive_only,
        ),
    )


def _owned_scaled_design(design: ScaledDesign | None) -> ScaledDesign | None:
    """Create a deeply owned ``ScaledDesign`` or ``None``."""
    if design is None:
        return None
    return ScaledDesign(
        owned_array(design.matrix),
        owned_array(design.scale),
        tuple(design.column_names),
    )


@dataclass(frozen=True, init=False)
class RetainedInputSnapshot:
    """One shared owned snapshot of the full-frame ``SampleBase`` and preparation.

    The clean internal fit path transfers one already isolated working capture
    through _from_owned_prepared; no duplicate baseline or snapshot is constructed.
    Its outward accessors are the same fresh-copy accessors below.
    Other constructors own copies of the base and optional frame parts.
    """

    key: InputKey
    nobs: int
    columns: Columns | None
    subset_missing: int
    _dependence: DependenceProvenance | None
    _base: SampleBase
    _frame: ModelFrame | None
    _labels: pd.Index[Any] | None
    _effects: EffectColumns | None

    def __init__(
        self,
        base: SampleBase,
        *,
        key: InputKey = 0,
        prepared: PreparedFrame | None = None,
        labels: pd.Index[Any] | None = None,
    ) -> None:
        """Own the ``SampleBase`` and optional ``PreparedFrame`` parts once."""
        owned_base = _owned_sample_base(base)
        lbls = (
            _owned_index(prepared.labels)
            if prepared is not None
            else (None if labels is None else _owned_index(labels))
        )
        for attr, val in (
            ("key", int(key)),
            ("nobs", owned_base.nobs),
            ("columns", None if prepared is None else prepared.columns),
            ("subset_missing", 0 if prepared is None else prepared.subset_missing),
            (
                "_dependence",
                None
                if prepared is None
                else _owned_dependence_provenance(prepared.dependence),
            ),
            ("_base", owned_base),
            (
                "_frame",
                None if prepared is None else _owned_model_frame(prepared.frame),
            ),
            ("_labels", lbls),
            (
                "_effects",
                None if prepared is None else _owned_effect_columns(prepared.effects),
            ),
        ):
            object.__setattr__(self, attr, val)

    @classmethod
    def from_prepared(
        cls, prepared: PreparedFrame, *, key: InputKey = 0
    ) -> RetainedInputSnapshot:
        """Construct a :class:`RetainedInputSnapshot` from a ``PreparedFrame``."""
        return cls(prepared.base, key=key, prepared=prepared)

    @classmethod
    def _from_owned_prepared(
        cls, prepared: PreparedFrame, *, key: InputKey = 0
    ) -> RetainedInputSnapshot:
        """Use private fit-owned parts; every outward accessor still detaches."""
        snap = object.__new__(cls)
        for name, value in (
            ("key", int(key)),
            ("nobs", prepared.base.nobs),
            ("columns", prepared.columns),
            ("subset_missing", prepared.subset_missing),
            ("_dependence", prepared.dependence),
            ("_base", prepared.base),
            ("_frame", prepared._frame),
            ("_labels", prepared._labels),
            ("_effects", prepared.effects),
        ):
            object.__setattr__(snap, name, value)
        return snap

    @property
    def dependence(self) -> DependenceProvenance | None:
        """Return a fresh deeply owned ``DependenceProvenance``, or ``None``."""
        return _owned_dependence_provenance(self._dependence)

    @property
    def base(self) -> SampleBase:
        """Return a fresh deeply owned ``SampleBase``."""
        result = _owned_sample_base(self._base)
        if self._dependence is not None:
            metadata = dict(result.inputs.metadata.columns)
            for name, column in self._dependence.columns.items():
                if name not in metadata:
                    metadata[name] = Column(column.values)
            object.__setattr__(
                result.inputs, "metadata", AlignedColumns(metadata, self.nobs)
            )
        return result

    @property
    def frame(self) -> ModelFrame | None:
        """Return a fresh deeply owned ``ModelFrame``, or ``None``."""
        return None if self._frame is None else _owned_model_frame(self._frame)

    @property
    def labels(self) -> pd.Index[Any]:
        """Return a fresh owned copy of the full-frame user row ``Index``."""
        if self._labels is None:
            return pd.Index(range(self.nobs))
        return _owned_index(self._labels)

    @property
    def effects(self) -> EffectColumns | None:
        """Return a fresh deeply owned ``EffectColumns``, or ``None``."""
        return _owned_effect_columns(self._effects)

    def labels_at(self, positions: IndexArray) -> pd.Index[Any]:
        """Return a fresh owned user row index sliced at ``positions``."""
        return self.labels.take(positions)

    def metadata_at(self, positions: IndexArray) -> AlignedColumns:
        """Select owned metadata columns at ``positions`` without rebuilding inputs."""
        metadata = dict(self._base.inputs.metadata.take(positions).columns)
        if self._dependence is not None:
            for name, column in self._dependence.columns.items():
                if name not in metadata:
                    metadata[name] = Column(column.take(positions))
        return AlignedColumns(metadata, positions.size)
