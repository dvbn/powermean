"""Owned preparation storage and callback-free fit dispatch.

Public normalization and sample/scientific kernels retain their error decisions.
This module recognizes passive input state without calling foreign objects, then
captures it once.  It does not encode/replay the formula, cache validation across
public calls, or verify the implementation by reconstructing a second graph.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.dependence import ColumnProvenance, DependenceProvenance, UsedDependence
from powermean._prepare.inputs import Columns
from powermean.data.accounting import MissingAccount
from powermean.data.raw import AlignedColumns, Column, RawInputs, RawWeights, _OwnedMapping
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns, _native_parts
from powermean.formula.frame import Design, ModelFrame
from powermean.formula.grammar import FormulaParts, Instruments
from powermean.results.primitive_snapshot import (
    _ENUMS, _ETS, _EVALUATED_FACTOR, _FACTOR, _MODEL_SPEC, _SCOPED_FACTOR,
    _SCOPED_TERM, _SIMPLE_FORMULA, _SIMPLE_TERMS, _TERM, _TOKEN, _VARIABLE,
)
from powermean.results.retained_labels import _copy_free_labels
from powermean.results.retained_input_values import (
    _owned_dependence_provenance, _owned_effect_columns, _owned_index,
    _owned_model_frame, _owned_sample_base,
)

_NATIVE_PARTS = _native_parts

_RECORDS = (
    PreparedFrame, Columns, SampleBase, RawInputs, RawWeights, AlignedColumns,
    Column, MissingAccount, EffectColumns, ColumnProvenance,
    DependenceProvenance, UsedDependence, FormulaParts, Instruments, ModelFrame, Design,
)
_SCALARS = (
    bool, int, float, complex, str, bytes,
    np.bool_, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.complex64, np.complex128,
    np.str_, np.bytes_,
)
_DTYPE_TYPES = tuple(type(np.dtype(code)) for code in
                     ("?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
                      "f2", "f4", "f8", "c8", "c16", "S1", "U1", "M8", "m8", "O"))
# The existing fitted-spec allowlist, inspected rather than serialized. In
# particular, transform-state arrays are checked by dtype, not copied to bytes.
# Unknown skeletons (including older Formula factories) keep the general route.
_SPEC_RECORDS = (
    (_MODEL_SPEC, ("formula", "materializer", "materializer_params",
                   "ensure_full_rank", "na_action", "output", "cluster_by",
                   "structure", "transform_state", "encoder_state")),
    (_SIMPLE_FORMULA, (_SIMPLE_TERMS, "ordering")),
    (_FACTOR, ("expr", "_eval_method", "_kind", "metadata", "token")),
    (_TOKEN, ("token", "_kind", "source", "source_start", "source_end")),
    (_TERM, ("factors",)),
    (_SCOPED_TERM, ("factors", "scale")),
    (_SCOPED_FACTOR, ("factor", "reduced")),
    (_EVALUATED_FACTOR, ("factor", "values", "variables")),
    (_ETS, ("term", "scoped_terms", "columns")),
    (_VARIABLE, ("roles", "source")),
)
_SPEC_ENUMS = tuple(_ENUMS.values())


def _plain_dtype(dtype: object) -> bool:
    return any(type(dtype) is kind for kind in _DTYPE_TYPES) and (
        dtype.fields is None and dtype.metadata is None
    )


def _passive_values(values: tuple[object, ...]) -> bool:
    """Whether retained state has no caller equality/deepcopy/extension hooks.

    This is eligibility, not a validity gate: a foreign value keeps the existing
    ordered route.  Only exact package records, native arrays/pandas containers,
    immutable label leaves and the already accepted passive fitted-spec allowlist
    are traversed.  The fitted specification is inspected, never retrained.
    """
    completed: set[int] = set()
    active: set[int] = set()

    def passive(value: Any) -> bool:
        kind = type(value)
        if value is None or any(kind is t for t in _SCALARS):
            return True
        ident = id(value)
        if ident in completed:
            return True
        if ident in active:
            return False
        active.add(ident)
        try:
            if kind is np.ndarray:
                ok = _plain_dtype(value.dtype) and (
                    value.dtype.kind != "O" or all(passive(v) for v in value.flat)
                )
            elif kind is slice:
                # Formulaic caches term_slices; reject components with index hooks.
                ok = all(part is None or type(part) is int
                         for part in (value.start, value.stop, value.step))
            elif kind is tuple or kind is list:
                ok = all(passive(v) for v in value)
            elif kind is dict:
                ok = all(passive(k) and passive(v) for k, v in value.items())
            elif kind is set or kind is frozenset:
                ok = all(passive(v) for v in value)
            elif kind is _OwnedMapping:
                items = object.__getattribute__(value, "_items")
                ok = type(items) is dict and passive(items)
            elif kind is pd.Index or kind is pd.RangeIndex:
                stored = object.__getattribute__(value, "_data")
                ok = (type(stored) is np.ndarray and _plain_dtype(stored.dtype)
                      and stored.dtype == value.dtype and passive(stored)
                      and passive(value.name))
            elif kind is pd.Series:
                ok = (_plain_dtype(value.dtype) and passive(value.attrs)
                      and passive(value.name) and passive(value.index)
                      and passive(value.to_numpy(copy=False)))
            elif kind is pd.DataFrame:
                ok = (all(_plain_dtype(dt) for dt in value.dtypes)
                      and passive(value.attrs) and passive(value.index)
                      and passive(value.columns)
                      and passive(value.to_numpy(copy=False)))
            elif any(kind is enum for enum in _SPEC_ENUMS):
                ok = passive(object.__getattribute__(value, "value"))
            elif any(kind is record for record, _ in _SPEC_RECORDS):
                names = next(names for record, names in _SPEC_RECORDS if kind is record)
                state = getattr(value, "__dict__", None)
                ok = (all(passive(object.__getattribute__(value, name)) for name in names)
                      and (state is None or (type(state) is dict and passive(state))))
            elif any(kind is record for record in _RECORDS):
                ok = all(passive(object.__getattribute__(value, field.name))
                         for field in fields(kind))
            else:
                ok = False
        finally:
            active.remove(ident)
        if ok:
            completed.add(ident)
        return ok

    return all(passive(value) for value in values)


def own_prepared_once(prepared: PreparedFrame) -> PreparedFrame:
    """Capture passive prepared parts exactly once, without accessor recopying.

    Private ownership constructor: its fields are newly owned by the existing
    copying primitives.  Bypassing the public PreparedFrame constructor prevents
    its second model-frame copy; this never adopts an external mutable buffer.
    """
    owned = object.__new__(PreparedFrame)
    for name, value in (
        ("columns", prepared.columns),
        ("base", _owned_sample_base(prepared.base)),
        ("subset_missing", prepared.subset_missing),
        ("effects", _owned_effect_columns(prepared.effects)),
        ("dependence", _owned_dependence_provenance(prepared.dependence)),
        ("_frame", _owned_model_frame(prepared._frame)),
        ("_labels", _owned_index(prepared._labels)),
    ):
        object.__setattr__(owned, name, value)
    return owned

def passive_preparation(prepared: PreparedFrame) -> bool:
    """The unchanged public/foreign preparation eligibility predicate."""
    return type(prepared) is PreparedFrame and _passive_values((prepared,))


def normalized_numeric_inputs(prepared: PreparedFrame) -> bool:
    """Eligibility for a fresh NORMAL producer's single owned capture.

    This does not declare the old whole-frame passive predicate true. Only fields
    consumed by the numeric route and a callback-free fitted specification are
    considered; ordinary public constructors keep their original whole-graph path.
    Unknown metadata or fitted-spec callbacks retain that original path.
    """
    if type(prepared) is not PreparedFrame:
        return False
    if not _passive_values((prepared._frame.design.spec,)):
        return False
    labels = prepared._labels
    if type(labels) is not pd.RangeIndex:
        if type(labels) is not pd.Index:
            return False
        stored = object.__getattribute__(labels, "_data")
        if type(stored) is not np.ndarray:
            return False
        if stored.dtype.kind == "O" and not _copy_free_labels(tuple(stored)):
            return False
    columns = list(prepared.base.inputs.metadata.columns.values())
    if prepared.effects is not None:
        native = (_native_parts(prepared.effects)
                  if _native_parts is _NATIVE_PARTS else None)
        effect_columns = prepared.effects.columns if native is None else native[0]
        columns.extend(effect_columns.values())
    for column in columns:
        if type(column) is not Column:
            return False
        values = column._stored
        if type(values) is tuple:
            if not _copy_free_labels(values):
                return False
        elif not (type(values) is np.ndarray and values.dtype.kind != "O"
                  and _plain_dtype(values.dtype)):
            return False
    return _copy_free_labels(tuple(labels.names))


def normalized_fit_frame(template: PreparedFrame) -> PreparedFrame:
    """Isolate operation records, sharing only the private fitted model description.

    The model description is never consumed as a mutable public object by the
    numerical path. All public .frame accesses still make detached owned copies.
    Numeric/sample/effect/dependence headers retain their accepted owning copies.
    """
    frame = object.__new__(PreparedFrame)
    for name, value in (
        ("columns", template.columns), ("base", _owned_sample_base(template.base)),
        ("subset_missing", template.subset_missing),
        ("effects", _owned_effect_columns(template.effects)),
        ("dependence", _owned_dependence_provenance(template.dependence)),
        ("_frame", template._frame), ("_labels", _owned_index(template._labels)),
    ):
        object.__setattr__(frame, name, value)
    return frame

def _single_fit_provenance(dependence: DependenceProvenance | None) -> bool:
    """Preserve copy hooks outside the normal predicate's metadata/effect columns."""
    if dependence is None:
        return True
    if type(dependence) is not DependenceProvenance or type(
        dependence._columns
    ) is not _OwnedMapping:
        return False
    for name, provenance in dependence._columns.items():
        if type(provenance) is not ColumnProvenance:
            return False
        descriptors = (
            name, provenance.name, provenance.roles, provenance.used_in_fit,
            provenance.source_dtype, provenance.pandas_kind,
        )
        if not _passive_values(descriptors):
            return False
        # Inspect already owned original components without expanding a pair.
        columns = (provenance._retained_sources
                   if "_retained_sources" in vars(provenance)
                   else (provenance._column_record,))
        for column in columns:
            if type(column) is not Column:
                return False
            values = column._stored
            if type(values) is tuple:
                if not _copy_free_labels(values):
                    return False
            elif not (type(values) is np.ndarray and values.dtype.kind != "O"
                      and _plain_dtype(values.dtype)):
                return False
    return _passive_values((dependence.used,))


def single_fit_frame(prepared: PreparedFrame) -> PreparedFrame | None:
    """Transfer fresh prepare_frame-owned rows into one unexposed direct fit.

    Only api._fit requests this lifetime, after the unchanged normal producer
    eligibility. Public prepare/refits never transfer. The producer already owns
    these records and buffers; no outward accessor returns their live headers.
    Preserve original copying for unknown hooks, and keep the original model and
    label detachment. There is no reusable preparation or cross-fit mutable state.
    """
    effects = prepared.effects
    if effects is not None and type(effects) is not EffectColumns:
        return None
    native = (_native_parts(effects) if effects is not None
              and _native_parts is _NATIVE_PARTS else None)
    effect_columns = (None if effects is None else
                      effects.columns if native is None else native[0])
    descriptors = (
        prepared.columns, prepared.base.inputs.column_names,
        tuple(prepared.base.inputs.metadata.columns),
        None if effects is None else (effects.terms, tuple(effect_columns)),
        prepared.base.inputs.weights,
    )
    if not _passive_values(descriptors) or not _single_fit_provenance(
        prepared.dependence
    ):
        return None
    owned = object.__new__(PreparedFrame)
    for name, value in (
        ("columns", prepared.columns), ("base", prepared.base),
        ("subset_missing", prepared.subset_missing), ("effects", effects),
        ("dependence", prepared.dependence),
        ("_frame", _owned_model_frame(prepared._frame)),
        ("_labels", _owned_index(prepared._labels)),
    ):
        object.__setattr__(owned, name, value)
    return owned

