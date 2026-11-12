"""Primitive-only immutable per-fit capture of a prepared frame (internal).

Opt-in through ``PerFitPreparationContext(..., primitive_snapshot=True)``; the package
default path imports no behavior from here.

Representation.  :class:`PrimitiveCapture` holds the complete prepared frame as owned
immutable Python values and nothing else: every numerical buffer as ``bytes`` with its
exact dtype string and shape (:class:`_Buffer`), every object column as a tuple of
labels of one exact plain kind with that kind recorded (:class:`_Plain`), every pandas
axis as a range or as such a buffer or tuple with its name (:class:`_RangeAxis`,
:class:`_ArrayAxis`), every scalar as a type-tagged tuple, and the fitted
``formulaic.ModelSpec`` as a tagged tuple tree of those leaves (:func:`_encode_node`).
No pandas object, NumPy array, dictionary, list, record or Formulaic object is
retained, so no caller alias and no later edit of the original graph can reach the
capture, and no read-only flag, identity, hash or prior success is relied on.

Recognition.  A frame is *eligible* exactly when this encoding succeeds over a closed
allowlist of exact types (:func:`primitive_ineligibility`): the concrete records of
the preparation, one-dimensional non-object NumPy storage whose dtype holds no object
(``dtype.hasobject``), object storage of one exact plain kind, exact ``RangeIndex`` or
``Index`` axes with plain names, pandas objects without ``attrs`` and with default
flags, and a spec whose skeleton is Formulaic's own ``SimpleFormula`` (``Formula``
before Formulaic 1.1), ``Term``, ``Factor``, ``Token``, ``EncodedTermStructure``,
``ScopedTerm``, ``ScopedFactor``, ``EvaluatedFactor`` and ``Variable`` with passive
state (plain scalars, NumPy scalars and non-object arrays, lists, tuples,
string-keyed dictionaries, Formulaic's enumerations).  Anything else (a caller's
class or subclass, a nested caller object however it copies, a contrast state
object, a structured or object dtype, an extension index) is *ineligible* with
the reason recorded, and the frame keeps the general route unchanged.
Recognition reads types with ``type`` and compares them by identity: it
runs no method of a caller's object.  That holds for every leaf, the keys of every
mapping and every scalar field of a record alike: a mapping field of the four records
that hold one (``MissingAccount.groups``, ``AlignedColumns.columns``,
``EffectColumns.columns``, ``DependenceProvenance._columns``) is read only as the
exact ``dict`` it is or carries, the record's current data: the
:class:`~powermean.data.raw._OwnedMapping` that the records' constructors form, whose
carried object is checked to be an exact ``dict`` before it is read, or an exact
``dict`` itself (:func:`_enc_items`).  No read of a ``mappingproxy`` can tell a
caller's mapping behind it from a dict without running its code, so a proxy, like a
mapping of any other class or an owned mapping carrying anything but an exact dict,
is refused unread; nothing is inferred from the identity of any object.  The keys and
the record's text fields (weight kinds, column names, dtype texts, row counts,
enumeration values) are checked to be of the exact plain type before any of them is
compared, formatted or retained, and a subclass of ``str`` or ``int`` is refused as a
caller's class is.  A spec dictionary is read as an exact ``dict``
(:func:`_enc_mapping`).

Materialization.  :meth:`PrimitiveCapture.prepared_frame` and its part methods build a
fresh, detached ``PreparedFrame`` (or part) through the public constructors on every
call, with fresh pandas objects, fresh arrays and a fresh spec: nothing mutable of the
result is shared with the capture, with the original or with a previous materialization
(read-only arrays view the capture's immutable bytes, as the owned arrays view their own
immutable bytes).  Nothing fitted is learned or evaluated again: the spec is rebuilt
from its recorded skeleton and state.  When the capture is formed,
:meth:`PrimitiveCapture.capture` materializes it once and verifies that the
materialization re-encodes to the identical capture; the context then verifies that
materialization against the original under the frame comparison.  Otherwise the frame
takes the general route with the reason recorded.

Consumers.  The context that owns a capture materializes the parts that its internal
consumers read once, over the capture's immutable bytes
(:class:`~powermean.results.primitive_parts.PrimitiveParts`): a live ``FESample`` is
validated against those parts on every authentication
(:meth:`~powermean.results.primitive_parts.PrimitiveParts.check_sample`); a live frame
presented at an unscoped entry is validated by re-encoding it and comparing with the
capture (:meth:`PrimitiveCapture.check_frame`).  Neither keeps anything between calls.
The consumer that binds or checks dependence metadata reads the base, the labels and
the dependence provenance and nothing else, as one :class:`MetadataParts` of those
parts: no model frame, effect columns or ``PreparedFrame`` is rebuilt for it.
:meth:`PrimitiveCapture.metadata_parts` materializes those three parts afresh.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.dependence import (
    ColumnProvenance,
    DependenceProvenance,
    UsedDependence,
)
from powermean._prepare.inputs import Columns
from powermean.data.accounting import MissingAccount
from powermean.data.raw import (
    AlignedColumns,
    AnyArray,
    Column,
    RawInputs,
    RawWeights,
    _object_array,
    _OwnedMapping,
)
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.data.weights import WEIGHT_KINDS
from powermean.formula.frame import Design, ModelFrame
from powermean.formula.grammar import FormulaParts, Instruments
from powermean.results.absorbed_sample_binding import UNBOUND
from powermean.results.retained_input_values import _PLAIN_TYPES, _plain_kind

GENERAL = "general"
PRIMITIVE = "primitive"

__all__ = [
    "GENERAL",
    "PRIMITIVE",
    "MetadataParts",
    "PrimitiveCapture",
    "primitive_ineligibility",
]

# ------------------------------------------------------------------ Formulaic types
# The exact Formulaic classes of a fitted spec's skeleton, imported as the accepted
# :mod:`powermean.results.retained_labels` imports the contrast classes.
_parser_types = cast("Any", importlib.import_module("formulaic.parser.types"))
_FACTOR: type[Any] = _parser_types.Factor
_TERM: type[Any] = _parser_types.Term
_TOKEN: type[Any] = _parser_types.Token
_mat_types = cast("Any", importlib.import_module("formulaic.materializers.types"))
_SCOPED_TERM: type[Any] = _mat_types.ScopedTerm
_SCOPED_FACTOR: type[Any] = _mat_types.ScopedFactor
_EVALUATED_FACTOR: type[Any] = _mat_types.EvaluatedFactor
_ETS: type[Any] = cast(
    "Any", importlib.import_module("formulaic.materializers.base")
).EncodedTermStructure
_spec_mod = cast("Any", importlib.import_module("formulaic.model_spec"))
_MODEL_SPEC: type[Any] = _spec_mod.ModelSpec
_FORMULA: type[Any] = cast("Any", importlib.import_module("formulaic")).Formula
# The formula of a fitted spec from Formulaic 1.1 on: the ``Formula`` factory builds
# this exact class for a formula without structure, the only kind a ``ModelSpec``
# admits.  Formulaic 1.0 has no such class (``None``, which is no object's type) and
# builds an exact ``Formula``.  The whole state of a ``SimpleFormula`` is its instance
# dictionary of exactly these two fields: the private plain list of its terms and its
# ordering.
_SIMPLE_FORMULA: type[Any] | None = getattr(
    importlib.import_module("formulaic.formula"), "SimpleFormula", None
)
_SIMPLE_TERMS = "_SimpleFormula__terms"
_SIMPLE_FIELDS = frozenset({_SIMPLE_TERMS, "ordering"})
_variables_mod = cast("Any", importlib.import_module("formulaic.utils.variables"))
_VARIABLE: type[Any] = _variables_mod.Variable
_DEFAULT_PARSER: type[Any] = cast(
    "Any", importlib.import_module("formulaic.parser.parser")
).DefaultFormulaParser
# The enumerations of the skeleton and state, by a tag; a value is encoded by its tag
# and its ``value`` and rebuilt by the enumeration's own lookup.
_ENUMS: dict[str, type[Any]] = {
    "NAAction": _spec_mod.NAAction,
    "ClusterBy": _spec_mod.ClusterBy,
    "OrderingMethod": cast(
        "Any", importlib.import_module("formulaic.formula")
    ).OrderingMethod,
    "EvalMethod": _FACTOR.EvalMethod,
    "Kind": _FACTOR.Kind,
    "TokenKind": _TOKEN.Kind,
    "Role": _VARIABLE.Role,
}
_ENUM_TAGS: dict[type[Any], str] = {cls: tag for tag, cls in _ENUMS.items()}
# The exact NumPy scalar types of a passive state value; each has one dtype.
_NP_SCALARS: tuple[type[Any], ...] = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)
# The dtype kinds of buffers: Booleans, integers, floats, complex, timedeltas,
# datetimes, bytes and unicode; never objects (``kind == "O"``) or structured/void
# storage (``kind == "V"``), whose dtype string would not rebuild the fields.
_BUFFER_KINDS = "biufcmMSU"
# The dtype kinds of a pandas axis or outcome with NumPy storage.
_AXIS_KINDS = "biuf"
_PLAIN_NAMES = {bool: "bool", int: "int", float: "float", str: "str"}

type _Node = tuple[Any, ...]
type _Scalar = tuple[Any, ...]
type _ColumnEnc = _Buffer | _Plain
type _Axis = _RangeAxis | _ArrayAxis


class _IneligibleError(Exception):
    """The frame is not eligible; ``args[0]`` is the recorded reason."""


# ---------------------------------------------------------------------- leaves


def _enc_scalar(value: object, what: str) -> _Scalar:
    """Encode ``None`` or one exact plain scalar with its type; floats finite."""
    if value is None:
        return ("none",)
    kind = type(value)
    if not any(kind is t for t in _PLAIN_TYPES):
        raise _IneligibleError(f"{what} is not a plain scalar")
    if kind is float:
        number = cast("float", value)
        if not math.isfinite(number):
            raise _IneligibleError(f"{what} is not finite")
        return ("float", number.hex())
    return (_PLAIN_NAMES[kind], value)


def _dec_scalar(node: _Scalar) -> object:
    """Rebuild the scalar of :func:`_enc_scalar`."""
    tag = node[0]
    if tag == "none":
        return None
    if tag == "float":
        return float.fromhex(node[1])
    return node[1]


@dataclass(frozen=True)
class _Buffer:
    """One NumPy buffer as immutable bytes with its exact dtype string and shape."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def of(cls, array: object, what: str, *, kinds: str = _BUFFER_KINDS) -> _Buffer:
        """Encode an exact ``ndarray`` of a buffer dtype; others are ineligible."""
        if type(array) is not np.ndarray:
            raise _IneligibleError(f"{what} is not a plain ndarray")
        arr = cast("AnyArray", array)
        if arr.dtype.hasobject or arr.dtype.kind not in kinds:
            raise _IneligibleError(f"{what} is not plain numeric storage")
        return cls(arr.dtype.str, tuple(int(n) for n in arr.shape), arr.tobytes())

    def view(self) -> AnyArray:
        """A fresh read-only array object over the immutable bytes (no copy)."""
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)

    def copy(self) -> AnyArray:
        """A fresh writable array in bytes of its own."""
        return self.view().copy()


@dataclass(frozen=True)
class _Plain:
    """Object storage as a tuple of labels of one exact plain kind, kind recorded."""

    kind: str
    labels: tuple[object, ...]

    @classmethod
    def of(cls, labels: object, what: str) -> _Plain:
        """Encode an exact tuple of labels of one plain kind (:func:`_plain_kind`)."""
        if type(labels) is not tuple:
            raise _IneligibleError(f"{what} is not plain storage")
        kind = _plain_kind(labels)
        if kind is None:
            raise _IneligibleError(f"{what} is not plain storage")
        return cls(_PLAIN_NAMES[kind], cast("tuple[object, ...]", labels))

    def array(self) -> AnyArray:
        """A fresh read-only object array of the labels."""
        return _object_array(self.labels)


def _enc_column(col: object, what: str) -> _ColumnEnc:
    """Encode an exact ``Column``: its plain buffer or its plain label tuple."""
    if type(col) is not Column:
        raise _IneligibleError(f"{what} is not plain storage")
    stored = col._stored
    if type(stored) is tuple:
        return _Plain.of(stored, what)
    if (
        type(stored) is not np.ndarray
        or stored.ndim != 1
        or stored.dtype.hasobject
        or stored.dtype.kind not in _BUFFER_KINDS
    ):
        raise _IneligibleError(f"{what} is not plain storage")
    return _Buffer.of(stored, what)


def _dec_column_values(enc: _ColumnEnc) -> AnyArray:
    """The values a ``Column`` constructor rebuilds the encoded column from."""
    return enc.array() if type(enc) is _Plain else cast("_Buffer", enc).view()


def _enc_axis(index: object, what: str, reason: str) -> _Axis:
    """Encode an exact ``RangeIndex`` or ``Index`` with a plain name.

    ``reason`` is recorded for any other index (a ``MultiIndex``, a categorical,
    datetime or extension index, an index of labels that are not of one plain kind).
    """
    kind = type(index)
    if kind is not pd.RangeIndex and kind is not pd.Index:
        raise _IneligibleError(reason)
    idx = cast("pd.Index[Any]", index)
    name = _enc_scalar(idx.name, f"{what} name")
    if kind is pd.RangeIndex:
        rng = cast("pd.RangeIndex", index)
        return _RangeAxis(int(rng.start), int(rng.stop), int(rng.step), name)
    try:
        if not isinstance(idx.dtype, np.dtype):
            raise _IneligibleError(reason)
        if idx.dtype.kind == "O":
            return _ArrayAxis(_Plain.of(tuple(idx.tolist()), what), name)
        return _ArrayAxis(_Buffer.of(idx.to_numpy(), what, kinds=_AXIS_KINDS), name)
    except _IneligibleError:
        raise _IneligibleError(reason) from None


@dataclass(frozen=True)
class _RangeAxis:
    """An exact ``RangeIndex`` by its range and plain name."""

    start: int
    stop: int
    step: int
    name: _Scalar

    @property
    def size(self) -> int:
        """The number of labels."""
        return len(range(self.start, self.stop, self.step))

    def index(self) -> pd.Index[Any]:
        """A fresh ``RangeIndex``."""
        return pd.RangeIndex(
            self.start, self.stop, self.step, name=_dec_scalar(self.name)
        )


@dataclass(frozen=True)
class _ArrayAxis:
    """An exact ``Index`` by its plain labels or buffer and plain name."""

    values: _ColumnEnc
    name: _Scalar

    @property
    def size(self) -> int:
        """The number of labels."""
        values = self.values
        return (
            len(values.labels)
            if type(values) is _Plain
            else cast("_Buffer", values).shape[0]
        )

    def index(self) -> pd.Index[Any]:
        """A fresh ``Index`` of the recorded dtype, labels and name."""
        name = _dec_scalar(self.name)
        if type(self.values) is _Plain:
            return pd.Index(self.values.array(), dtype=object, name=name)
        return pd.Index(cast("_Buffer", self.values).copy(), name=name)


def _check_pandas_object(obj: pd.Series[Any] | pd.DataFrame, what: str) -> None:
    """Refuse pandas metadata the materialization does not reproduce."""
    if obj.attrs or not obj.flags.allows_duplicate_labels:
        raise _IneligibleError(f"{what} carries attrs or non-default flags")


# ------------------------------------------------------------- the fitted spec


def _enc_enum(value: object, what: str) -> _Node:
    """Encode one member of a registered Formulaic enumeration by its plain value."""
    # Hashing a caller's class can run its metaclass; use identity only.
    tag = next((tag for cls, tag in _ENUM_TAGS.items() if type(value) is cls), None)
    if tag is None:
        raise _IneligibleError(f"{what} is not a Formulaic enumeration")
    return ("enum", tag, _enc_scalar(cast("Any", value).value, f"{what} value"))


def _enc_items(mapping: object, what: str) -> tuple[tuple[str, Any], ...]:
    """The items of a record's mapping field, read from its exact ``dict``.

    ``mapping`` is the field as the record presents it now, and is read only as the
    exact ``dict`` it is or carries: an :class:`_OwnedMapping` (what the four records'
    constructors form) whose carried object is an exact ``dict``, or an exact ``dict``
    itself (an account given a caller's plain dict).  Every read is then the code of
    ``dict`` and of nothing else, and the items read are the current contents that
    the field presents to every reader.  A mapping of any other type, a
    ``mappingproxy`` among them, whose reads would run the code of whatever is behind
    it, and an owned mapping carrying anything but an exact dict, are ineligible
    unread; a key of any other type is ineligible before it is compared, formatted or
    retained.  Nothing is inferred from the identity of any object.
    """
    kind = type(mapping)
    if kind is _OwnedMapping:
        items = cast("_OwnedMapping[Any]", mapping)._items
        if type(items) is not dict:
            raise _IneligibleError(f"{what} is not backed by a plain dict")
    elif kind is dict:
        items = cast("dict[Any, Any]", mapping)
    else:
        raise _IneligibleError(f"{what} is not a plain mapping")
    pairs = tuple(items.items())
    if any(type(key) is not str for key, _ in pairs):
        raise _IneligibleError(f"{what} has a key that is not a str")
    return pairs


def _enc_mapping(value: Mapping[Any, Any], what: str) -> _Node:
    """Encode an exact ``dict`` with string keys, in insertion order."""
    if type(value) is not dict:
        raise _IneligibleError(f"{what} is not a plain dict")
    items = []
    for key, item in value.items():
        if type(key) is not str:
            raise _IneligibleError(f"{what} has a key that is not a str")
        items.append((key, _encode_node(item, f"{what}[{key!r}]")))
    return ("dict", tuple(items))


def _enc_token(token: object) -> _Node:
    """Encode a ``Token`` of a factor, or ``None``."""
    if token is None:
        return ("none",)
    if type(token) is not _TOKEN:
        raise _IneligibleError("a factor token is not a Formulaic Token")
    tok = cast("Any", token)
    kind = ("none",) if tok._kind is None else _enc_enum(tok._kind, "token kind")
    return (
        "Token",
        _enc_scalar(tok.token, "token text"),
        kind,
        _enc_scalar(tok.source, "token source"),
        _enc_scalar(tok.source_start, "token start"),
        _enc_scalar(tok.source_end, "token end"),
    )


def _enc_factor(factor: object, what: str) -> _Node:
    """Encode an exact ``Factor``: expression, evaluation, kind, metadata, token."""
    if type(factor) is not _FACTOR:
        raise _IneligibleError(f"{what} is not a Formulaic Factor")
    fac = cast("Any", factor)
    return (
        "Factor",
        _enc_scalar(fac.expr, f"{what} expr"),
        _enc_enum(fac._eval_method, f"{what} eval_method"),
        _enc_enum(fac._kind, f"{what} kind"),
        _enc_mapping(fac.metadata, f"{what} metadata"),
        _enc_token(fac.token),
    )


def _enc_term(term: object, what: str) -> _Node:
    """Encode an exact ``Term`` as its factors, in order."""
    if type(term) is not _TERM or type(cast("Any", term).factors) is not tuple:
        raise _IneligibleError(f"{what} is not a Formulaic Term")
    return (
        "Term",
        tuple(
            _enc_factor(f, f"{what} factor {k}")
            for k, f in enumerate(cast("Any", term).factors)
        ),
    )


def _enc_variables(variables: object, what: str) -> _Node:
    """Encode an optional set of ``Variable`` names with their roles and source."""
    if variables is None:
        return ("none",)
    if type(variables) is not set:
        raise _IneligibleError(f"{what} is not a set of Variables")
    encoded = []
    for var in variables:
        if type(var) is not _VARIABLE or type(var.roles) is not set:
            raise _IneligibleError(f"{what} is not a set of Variables")
        roles = tuple(sorted(_enc_enum(r, f"{what} role") for r in var.roles))
        encoded.append((str(var), roles, _enc_scalar(var.source, f"{what} source")))
    return ("variables", tuple(sorted(encoded)))


def _enc_scoped_term(scoped: object, what: str) -> _Node:
    """Encode an exact ``ScopedTerm`` of ``ScopedFactor`` of ``EvaluatedFactor``."""
    if (
        type(scoped) is not _SCOPED_TERM
        or type(cast("Any", scoped).factors) is not tuple
    ):
        raise _IneligibleError(f"{what} is not a Formulaic ScopedTerm")
    factors = []
    for k, sf in enumerate(cast("Any", scoped).factors):
        where = f"{what} scoped factor {k}"
        if type(sf) is not _SCOPED_FACTOR or type(sf.factor) is not _EVALUATED_FACTOR:
            raise _IneligibleError(f"{where} is not a Formulaic ScopedFactor")
        ev = sf.factor
        if ev.values is not None:
            raise _IneligibleError(f"{where} still holds evaluated values")
        factors.append(
            (
                "ScopedFactor",
                _enc_factor(ev.factor, where),
                _enc_variables(ev.variables, f"{where} variables"),
                _enc_scalar(sf.reduced, f"{where} reduced"),
            )
        )
    return (
        "ScopedTerm",
        tuple(factors),
        _enc_scalar(cast("Any", scoped).scale, f"{what} scale"),
    )


def _enc_structure(structure: object) -> _Node:
    """Encode the optional list of ``EncodedTermStructure`` of a fitted spec."""
    if structure is None:
        return ("none",)
    if type(structure) is not list:
        raise _IneligibleError("spec structure is not a list")
    encoded = []
    for k, st in enumerate(structure):
        what = f"spec structure {k}"
        if (
            type(st) is not _ETS
            or type(st.scoped_terms) is not list
            or type(st.columns) is not list
        ):
            raise _IneligibleError(f"{what} is not a Formulaic EncodedTermStructure")
        encoded.append(
            (
                "EncodedTermStructure",
                _enc_term(st.term, f"{what} term"),
                tuple(
                    _enc_scoped_term(sc, f"{what} scoped term {j}")
                    for j, sc in enumerate(st.scoped_terms)
                ),
                tuple(_enc_scalar(c, f"{what} column") for c in st.columns),
            )
        )
    return ("structure", tuple(encoded))


def _enc_formula_ordering(value: object) -> _Node:
    """Encode only the ordering enumeration the trusted formula factory accepts."""
    ordering = _enc_enum(value, "spec formula ordering")
    if type(value) is not _ENUMS["OrderingMethod"]:
        raise _IneligibleError(
            "spec formula ordering is not a Formulaic OrderingMethod"
        )
    return ordering


def _enc_simple_formula(formula: object) -> _Node:
    """Encode an exact ``SimpleFormula``: its plain list of terms and its ordering.

    The instance dictionary, the whole state, is read only as the exact ``dict`` of
    exactly the two fields of ``_SIMPLE_FIELDS``, keyed by exact strings, and the
    terms only as an exact ``list``; each is checked before what it holds is read,
    and the ordering before the list is iterated.  An instance with any other state
    (an added attribute, another container of the terms) or with no dictionary is
    ineligible unread.  The node is the ``Formula`` node of root terms and ordering,
    from which the ``Formula`` factory rebuilds an exact ``SimpleFormula`` of fresh
    terms with that ordering, parsing nothing; :meth:`PrimitiveCapture.capture`
    verifies that it re-encodes to the same node, which fails for a list that its
    ordering would reorder.
    """
    state: Any = getattr(formula, "__dict__", None)
    keys = tuple(state) if type(state) is dict else ()
    if any(type(key) is not str for key in keys) or frozenset(keys) != _SIMPLE_FIELDS:
        raise _IneligibleError("spec formula is not a plain SimpleFormula")
    terms = state[_SIMPLE_TERMS]
    if type(terms) is not list:
        raise _IneligibleError("spec formula is not a plain SimpleFormula")
    ordering = _enc_formula_ordering(state["ordering"])
    return (
        "Formula",
        tuple(_enc_term(t, f"spec formula term {k}") for k, t in enumerate(terms)),
        ordering,
    )


def _enc_formula(formula: object) -> _Node:
    """Encode an exact ``Formula`` of root terms with its ordering.

    From Formulaic 1.1 the formula of a fitted spec is an exact ``SimpleFormula``
    (:func:`_enc_simple_formula`).  Formulaic 1.0 forms an exact ``Formula``, whose
    two parsers are Formulaic's default parsers of a parsed string, not state of the
    fit, and are not recorded; the rebuilt formula gets the same defaults, which
    :meth:`PrimitiveCapture.capture` verifies as equal.
    """
    if type(formula) is _SIMPLE_FORMULA:
        return _enc_simple_formula(formula)
    if type(formula) is not _FORMULA:
        raise _IneligibleError("spec formula is not a Formulaic Formula")
    fm = cast("Any", formula)
    keys = tuple(fm._structure) if type(fm._structure) is dict else ()
    if (
        fm._metadata is not None
        or len(keys) != 1
        or type(keys[0]) is not str
        or keys[0] != "root"
        or type(fm._structure["root"]) is not list
        or type(fm._parser) is not _DEFAULT_PARSER
        or type(fm._nested_parser) is not _DEFAULT_PARSER
    ):
        raise _IneligibleError("spec formula is not a plain root formula")
    return (
        "Formula",
        tuple(
            _enc_term(t, f"spec formula term {k}")
            for k, t in enumerate(fm._structure["root"])
        ),
        _enc_enum(fm._ordering, "spec formula ordering"),
    )


def _enc_spec(spec: object) -> _Node:
    """Encode an exact fitted ``formulaic.ModelSpec`` passively, field by field."""
    if type(spec) is not _MODEL_SPEC:
        raise _IneligibleError("design spec is not a formulaic.ModelSpec")
    s = cast("Any", spec)
    return (
        "ModelSpec",
        _enc_formula(s.formula),
        _enc_scalar(s.materializer, "spec materializer"),
        ("none",)
        if s.materializer_params is None
        else _enc_mapping(s.materializer_params, "spec materializer_params"),
        _enc_scalar(s.ensure_full_rank, "spec ensure_full_rank"),
        _enc_enum(s.na_action, "spec na_action"),
        _enc_scalar(s.output, "spec output"),
        _enc_enum(s.cluster_by, "spec cluster_by"),
        _enc_structure(s.structure),
        _enc_mapping(s.transform_state, "spec transform_state"),
        _enc_mapping(s.encoder_state, "spec encoder_state"),
    )


def _encode_node(value: object, what: str) -> _Node:
    """Encode one passive state value of a fitted spec; anything else is ineligible."""
    kind = type(value)
    if value is None or any(kind is t for t in _PLAIN_TYPES):
        return _enc_scalar(value, what)
    if kind is dict:
        return _enc_mapping(cast("dict[Any, Any]", value), what)
    if kind is list or kind is tuple:
        items = tuple(
            _encode_node(v, f"{what}[{k}]")
            for k, v in enumerate(cast("tuple[Any, ...]", value))
        )
        return ("list" if kind is list else "tuple", items)
    if kind is np.ndarray:
        buf = _Buffer.of(value, what)
        return ("ndarray", buf.dtype, buf.shape, buf.data)
    if any(kind is t for t in _NP_SCALARS):
        scalar = cast("np.generic", value)
        return ("npscalar", scalar.dtype.str, scalar.tobytes())
    if any(kind is cls for cls in _ENUM_TAGS):
        return _enc_enum(value, what)
    # Even reading a caller's class name can invoke its metaclass.
    raise _IneligibleError(f"{what} holds a value of an unsupported type")


def _decode_node(node: _Node) -> Any:
    """Rebuild the fresh value of :func:`_encode_node` and the spec encoders."""
    tag = node[0]
    if tag in _SCALAR_TAGS:
        return _dec_scalar(node)
    return _DECODERS[tag](node)


def _dec_variables(node: _Node) -> set[Any]:
    """Rebuild the set of ``Variable`` of :func:`_enc_variables`."""
    return {
        _VARIABLE(name, roles={_decode_node(r) for r in roles}, source=_dec_scalar(src))
        for name, roles, src in node[1]
    }


def _dec_scoped_factor(node: _Node) -> Any:
    """Rebuild a ``ScopedFactor`` of an ``EvaluatedFactor`` without values."""
    evaluated = _EVALUATED_FACTOR(_decode_node(node[1]), None, _decode_node(node[2]))
    return _SCOPED_FACTOR(evaluated, reduced=_dec_scalar(node[3]))


def _dec_model_spec(node: _Node) -> Any:
    """Rebuild the fitted ``ModelSpec`` of :func:`_enc_spec`."""
    return _MODEL_SPEC(
        formula=_decode_node(node[1]),
        materializer=_dec_scalar(node[2]),
        materializer_params=_decode_node(node[3]),
        ensure_full_rank=_dec_scalar(node[4]),
        na_action=_decode_node(node[5]),
        output=_dec_scalar(node[6]),
        cluster_by=_decode_node(node[7]),
        structure=_decode_node(node[8]),
        transform_state=_decode_node(node[9]),
        encoder_state=_decode_node(node[10]),
    )


_SCALAR_TAGS = frozenset({"none", "bool", "int", "float", "str"})
# One rebuilding function per tag of :func:`_encode_node` and the spec encoders; each
# calls the public constructor of the value with freshly rebuilt parts.
_DECODERS: dict[str, Any] = {
    "dict": lambda node: {k: _decode_node(v) for k, v in node[1]},
    "list": lambda node: [_decode_node(v) for v in node[1]],
    "tuple": lambda node: tuple(_decode_node(v) for v in node[1]),
    "ndarray": lambda node: (
        np.frombuffer(node[3], dtype=np.dtype(node[1])).reshape(node[2]).copy()
    ),
    "npscalar": lambda node: np.frombuffer(node[2], dtype=np.dtype(node[1]))[0],
    "enum": lambda node: _ENUMS[node[1]](_dec_scalar(node[2])),
    "Token": lambda node: _TOKEN(
        _dec_scalar(node[1]),
        kind=_decode_node(node[2]),
        source=_dec_scalar(node[3]),
        source_start=_dec_scalar(node[4]),
        source_end=_dec_scalar(node[5]),
    ),
    "Factor": lambda node: _FACTOR(
        _dec_scalar(node[1]),
        eval_method=_decode_node(node[2]),
        kind=_decode_node(node[3]),
        metadata=_decode_node(node[4]),
        token=_decode_node(node[5]),
    ),
    "Term": lambda node: _TERM(factors=[_decode_node(f) for f in node[1]]),
    "variables": _dec_variables,
    "ScopedFactor": _dec_scoped_factor,
    "ScopedTerm": lambda node: _SCOPED_TERM(
        factors=tuple(_decode_node(f) for f in node[1]), scale=_dec_scalar(node[2])
    ),
    "EncodedTermStructure": lambda node: _ETS(
        _decode_node(node[1]),
        [_decode_node(sc) for sc in node[2]],
        [_dec_scalar(c) for c in node[3]],
    ),
    "structure": lambda node: [_decode_node(st) for st in node[1]],
    "Formula": lambda node: _FORMULA(
        [_decode_node(t) for t in node[1]], _ordering=_decode_node(node[2])
    ),
    "ModelSpec": _dec_model_spec,
}


# ------------------------------------------------------------------ the parts


def _enc_strings(values: object, what: str) -> tuple[str, ...]:
    """An exact tuple of exact strings."""
    if type(values) is not tuple or any(type(v) is not str for v in values):
        raise _IneligibleError(f"{what} is not a tuple of str")
    return cast("tuple[str, ...]", values)


def _enc_str(value: object, what: str) -> str:
    """An exact string."""
    if type(value) is not str:
        raise _IneligibleError(f"{what} is not a str")
    return value


def _enc_opt_str(value: object, what: str) -> str | None:
    """``None`` or an exact string."""
    if value is not None and type(value) is not str:
        raise _IneligibleError(f"{what} is not a str")
    return value


def _enc_columns(columns: object) -> tuple[Any, ...]:
    """Encode an exact ``Columns`` record as its plain fields."""
    if type(columns) is not Columns:
        raise _IneligibleError("columns are not a concrete Columns")
    c = columns
    kind = _enc_str(c.weight_kind, "weight kind")
    if kind not in WEIGHT_KINDS:
        raise _IneligibleError("weight kind is not known")
    return (
        _enc_opt_str(c.weights, "weights column"),
        kind,
        _enc_opt_str(c.offset, "offset column"),
        _enc_opt_str(c.exposure, "exposure column"),
        _enc_opt_str(c.cluster, "cluster column"),
        _enc_strings(c.metadata, "metadata columns"),
    )


def _enc_parts(parts: object) -> tuple[Any, ...]:
    """Encode exact ``FormulaParts`` (and ``Instruments``) as plain fields."""
    if type(parts) is not FormulaParts:
        raise _IneligibleError("frame parts are not concrete FormulaParts")
    p = parts
    inst = p.instruments
    if inst is not None and type(inst) is not Instruments:
        raise _IneligibleError("instruments are not concrete Instruments")
    for text in (p.formula, p.first_part, p.outcome, p.regressors):
        if type(text) is not str:
            raise _IneligibleError("frame parts are not plain text")
    if type(p.fixed_effects) is not tuple:
        raise _IneligibleError("fixed-effect terms are not a tuple")
    return (
        p.formula,
        p.first_part,
        p.outcome,
        p.regressors,
        tuple(_enc_strings(term, "fixed-effect term") for term in p.fixed_effects),
        None
        if inst is None
        else (
            _enc_opt_str(inst.endogenous, "endogenous"),
            _enc_opt_str(inst.excluded, "excluded"),
        ),
    )


def _dec_parts(enc: tuple[Any, ...]) -> FormulaParts:
    """Rebuild the ``FormulaParts`` of :func:`_enc_parts`."""
    inst = enc[5]
    return FormulaParts(
        enc[0],
        enc[1],
        enc[2],
        enc[3],
        enc[4],
        None if inst is None else Instruments(inst[0], inst[1]),
    )


@dataclass(frozen=True)
class _BaseEnc:
    """An exact ``SampleBase``: its raw inputs, masks and codes."""

    y: _Buffer
    design: _Buffer
    column_names: tuple[str, ...]
    weights: tuple[_ColumnEnc, str] | None
    offset: _Buffer | None
    exposure: _Buffer | None
    metadata: tuple[tuple[str, _ColumnEnc], ...]
    subset: _Buffer | None
    missing_groups: tuple[tuple[str, _Buffer], ...]
    missing_excluded: _Buffer
    complete: _Buffer
    negative_weight: _Buffer
    zero_weight: _Buffer
    codes: _Buffer

    @classmethod
    def of(cls, base: object) -> _BaseEnc:
        """Encode an exact ``SampleBase`` of concrete records and plain storage."""
        if type(base) is not SampleBase:
            raise _IneligibleError("base is not a concrete SampleBase")
        b = base
        inputs = b.inputs
        if type(inputs) is not RawInputs or type(inputs.metadata) is not AlignedColumns:
            raise _IneligibleError(
                "inputs are not concrete RawInputs with AlignedColumns"
            )
        if type(b.missing) is not MissingAccount:
            raise _IneligibleError("missing account is not a concrete MissingAccount")
        weights = None
        if inputs.weights is not None:
            w = inputs.weights
            if type(w) is not RawWeights:
                raise _IneligibleError("weights are not a plain RawWeights column")
            w_kind = _enc_str(w.kind, "weights kind")
            if w_kind not in WEIGHT_KINDS:
                raise _IneligibleError("weights are not a plain RawWeights column")
            weights = (_enc_column(w.column, "weights"), w_kind)
        return cls(
            y=_Buffer.of(inputs.y, "y"),
            design=_Buffer.of(inputs.design, "design"),
            column_names=_enc_strings(inputs.column_names, "column names"),
            weights=weights,
            offset=None
            if inputs.offset is None
            else _Buffer.of(inputs.offset, "offset"),
            exposure=None
            if inputs.exposure is None
            else _Buffer.of(inputs.exposure, "exposure"),
            metadata=tuple(
                (name, _enc_column(col, f"metadata column {name!r}"))
                for name, col in _enc_items(inputs.metadata.columns, "metadata columns")
            ),
            subset=None if b.subset is None else _Buffer.of(b.subset, "subset"),
            missing_groups=tuple(
                (group, _Buffer.of(mask, f"mask {group!r}"))
                for group, mask in _enc_items(b.missing.groups, "missing groups")
            ),
            missing_excluded=_Buffer.of(b.missing.excluded, "missing excluded"),
            complete=_Buffer.of(b.complete, "complete"),
            negative_weight=_Buffer.of(b.negative_weight, "negative weight"),
            zero_weight=_Buffer.of(b.zero_weight, "zero weight"),
            codes=_Buffer.of(b.codes, "codes"),
        )

    def aligned_columns(self) -> AlignedColumns:
        """Fresh owned aligned metadata columns."""
        return AlignedColumns(
            {name: Column(_dec_column_values(enc)) for name, enc in self.metadata},
            int(self.y.shape[0]),
        )

    def sample_base(self) -> SampleBase:
        """A fresh ``SampleBase`` through the public constructors."""
        weights = None
        if self.weights is not None:
            enc, kind = self.weights
            weights = RawWeights(_dec_column_values(enc), cast("Any", kind))
        inputs = RawInputs(
            y=self.y.view(),
            design=self.design.view(),
            column_names=self.column_names,
            weights=weights,
            offset=None if self.offset is None else self.offset.view(),
            exposure=None if self.exposure is None else self.exposure.view(),
            metadata=self.aligned_columns().columns,
        )
        return SampleBase(
            inputs=inputs,
            subset=None if self.subset is None else self.subset.view(),
            missing=MissingAccount._owned(
                {group: buf.view() for group, buf in self.missing_groups},
                self.missing_excluded.view(),
            ),
            complete=self.complete.view(),
            negative_weight=self.negative_weight.view(),
            zero_weight=self.zero_weight.view(),
            codes=self.codes.view(),
        )


@dataclass(frozen=True)
class _FrameEnc:
    """An exact ``ModelFrame``: outcome, fitted design and masks."""

    parts: tuple[Any, ...]
    outcome: _Buffer
    outcome_name: _Scalar
    outcome_axis: _Axis
    outcome_invalid: _Buffer
    matrix: _Buffer
    matrix_axis: _Axis
    matrix_columns: _Axis
    spec: _Node
    terms: tuple[tuple[str, tuple[str, ...]], ...]
    variables: tuple[str, ...]
    labels: _Axis
    referenced_invalid: _Buffer
    evaluated_invalid: _Buffer
    formulaic_version: str
    contrast_spec: _Node

    @classmethod
    def of(cls, frame: object) -> _FrameEnc:
        """Encode an exact ``ModelFrame`` with a plain outcome, matrix and spec."""
        if type(frame) is not ModelFrame:
            raise _IneligibleError("frame is not a concrete ModelFrame")
        f = frame
        d = f.design
        if type(d) is not Design:
            raise _IneligibleError("design is not a concrete Design")
        if type(f.outcome) is not pd.Series or type(d.matrix) is not pd.DataFrame:
            raise _IneligibleError("matrix or outcome is not a concrete pandas object")
        _check_pandas_object(f.outcome, "outcome")
        _check_pandas_object(d.matrix, "matrix")
        if not isinstance(f.outcome.dtype, np.dtype):
            raise _IneligibleError("outcome dtype is not plain numeric")
        if any(
            not isinstance(dt, np.dtype) or dt != np.dtype(np.float64)
            for dt in d.matrix.dtypes
        ):
            raise _IneligibleError("matrix dtype is not plain numeric")
        if type(d.terms) is not tuple or any(
            type(t) is not tuple or len(t) != 2 or type(t[0]) is not str
            for t in d.terms
        ):
            raise _IneligibleError("design terms are not plain")
        if type(d.formulaic_version) is not str:
            raise _IneligibleError("formulaic version is not a str")
        return cls(
            parts=_enc_parts(f.parts),
            outcome=_Buffer.of(f.outcome.to_numpy(), "outcome", kinds=_AXIS_KINDS),
            outcome_name=_enc_scalar(f.outcome.name, "outcome name"),
            outcome_axis=_enc_axis(
                f.outcome.index, "outcome index", "outcome index is not a plain Index"
            ),
            outcome_invalid=_Buffer.of(f.outcome_invalid, "outcome mask"),
            matrix=_Buffer.of(d.matrix.to_numpy(dtype=np.float64), "matrix"),
            matrix_axis=_enc_axis(
                d.matrix.index, "matrix index", "matrix index is not a plain Index"
            ),
            matrix_columns=_enc_axis(
                d.matrix.columns,
                "matrix columns",
                "matrix columns is not a plain Index",
            ),
            spec=_enc_spec(d.spec),
            terms=tuple((t[0], _enc_strings(t[1], "term columns")) for t in d.terms),
            variables=_enc_strings(d.variables, "design variables"),
            labels=_enc_axis(
                d.labels, "design labels", "design labels is not a plain Index"
            ),
            referenced_invalid=_Buffer.of(d.referenced_invalid, "referenced mask"),
            evaluated_invalid=_Buffer.of(d.evaluated_invalid, "evaluated mask"),
            formulaic_version=d.formulaic_version,
            contrast_spec=_enc_spec(d._contrast_spec),
        )

    def model_frame(self) -> ModelFrame:
        """A fresh ``ModelFrame`` with fresh pandas objects and a fresh spec."""
        outcome = pd.Series(
            self.outcome.copy(),
            index=self.outcome_axis.index(),
            name=_dec_scalar(self.outcome_name),
        )
        matrix = pd.DataFrame(
            self.matrix.copy(),
            index=self.matrix_axis.index(),
            columns=self.matrix_columns.index(),
        )
        return ModelFrame(
            parts=_dec_parts(self.parts),
            outcome=outcome,
            outcome_invalid=self.outcome_invalid.view(),
            design=Design(
                matrix=matrix,
                spec=_decode_node(self.spec),
                terms=self.terms,
                variables=self.variables,
                labels=self.labels.index(),
                referenced_invalid=self.referenced_invalid.view(),
                evaluated_invalid=self.evaluated_invalid.view(),
                formulaic_version=self.formulaic_version,
                _contrast_spec=_decode_node(self.contrast_spec),
            ),
        )


@dataclass(frozen=True)
class _EffectsEnc:
    """Exact ``EffectColumns``: terms, plain component columns, rows."""

    terms: tuple[tuple[str, ...], ...]
    columns: tuple[tuple[str, _ColumnEnc], ...]
    rows: int

    @classmethod
    def of(cls, effects: object) -> _EffectsEnc:
        """Encode exact ``EffectColumns`` of plain columns."""
        if type(effects) is not EffectColumns:
            raise _IneligibleError("effects are not concrete EffectColumns")
        e = effects
        if type(e.rows) is not int:
            raise _IneligibleError("effects rows is not a plain int")
        if type(e.terms) is not tuple:
            raise _IneligibleError("effects terms are not a plain tuple")
        return cls(
            terms=tuple(_enc_strings(t, "effect term") for t in e.terms),
            columns=tuple(
                (name, _enc_column(col, f"effects column {name!r}"))
                for name, col in _enc_items(e.columns, "effects columns")
            ),
            rows=e.rows,
        )

    def effect_columns(self) -> EffectColumns:
        """Fresh ``EffectColumns`` through the public constructor."""
        return EffectColumns(
            terms=self.terms,
            columns={
                name: Column(_dec_column_values(enc)) for name, enc in self.columns
            },
            rows=self.rows,
        )


@dataclass(frozen=True)
class _DependenceEnc:
    """Exact ``DependenceProvenance``: the request, plain columns, the union mask."""

    used: tuple[Any, ...]
    columns: tuple[
        tuple[str, tuple[str, ...], bool, str, str, _Buffer, _ColumnEnc], ...
    ]
    used_missing_union: _Buffer

    @classmethod
    def of(cls, dependence: object) -> _DependenceEnc:
        """Encode exact ``DependenceProvenance`` of plain ``ColumnProvenance``."""
        if type(dependence) is not DependenceProvenance:
            raise _IneligibleError("dependence is not a concrete DependenceProvenance")
        dep = dependence
        used = dep.used
        if type(used) is not UsedDependence:
            raise _IneligibleError("used dependence is not concrete")
        coords = used.coordinates
        if coords is not None and (type(coords) is not tuple or len(coords) != 2):
            raise _IneligibleError("coordinates are not a plain pair")
        columns = []
        # The private mapping: the public ``columns`` rebuilds a dict, hashing keys.
        for name, prov in _enc_items(dep._columns, "dependence columns"):
            what = f"dependence column {name!r}"
            if type(prov) is not ColumnProvenance or type(prov.used_in_fit) is not bool:
                raise _IneligibleError(f"{what} is not plain storage")
            if _enc_str(prov.name, f"{what} name") != name:
                raise _IneligibleError(f"{what} is not named by its key")
            columns.append(
                (
                    name,
                    _enc_strings(prov.roles, f"{what} roles"),
                    prov.used_in_fit,
                    _enc_str(prov.source_dtype, f"{what} dtype"),
                    _enc_str(prov.pandas_kind, f"{what} kind"),
                    _Buffer.of(prov._missing_mask, f"{what} mask"),
                    _enc_column(prov._column_record, what),
                )
            )
        return cls(
            used=(
                _enc_strings(used.clusters, "clusters"),
                _enc_opt_str(used.time, "time"),
                _enc_opt_str(used.unit, "unit"),
                None
                if coords is None
                else (
                    _enc_opt_str(coords[0], "latitude"),
                    _enc_opt_str(coords[1], "longitude"),
                ),
                _enc_strings(used.retained, "retained"),
            ),
            columns=tuple(columns),
            used_missing_union=_Buffer.of(dep.used_missing_union, "dependence union"),
        )

    def dependence(self) -> DependenceProvenance:
        """Fresh ``DependenceProvenance`` through the public constructors."""
        clusters, time, unit, coords, retained = self.used
        return DependenceProvenance(
            used=UsedDependence(clusters, time, unit, coords, retained),
            columns={
                name: ColumnProvenance(
                    name=name,
                    roles=cast("Any", roles),
                    used_in_fit=used_in_fit,
                    source_dtype=source_dtype,
                    pandas_kind=pandas_kind,
                    missing_mask=mask.view(),
                    values=_dec_column_values(enc),
                )
                for name, roles, used_in_fit, source_dtype, pandas_kind, mask, enc in (
                    self.columns
                )
            },
            used_missing_union=self.used_missing_union.view(),
        )


@dataclass(frozen=True)
class MetadataParts:
    """The parts of a prepared frame that dependence metadata binding reads.

    Binding and checking dependence metadata read the sample base (its row count, raw
    weights and aligned metadata columns), the user's row labels and the dependence
    provenance of a frame, and never its model frame or effect columns.  This record
    carries exactly those three parts: from a frame as that frame hands them out
    (:meth:`of_frame`), or fresh from a capture
    (:meth:`PrimitiveCapture.metadata_parts`), where materializing a complete
    ``PreparedFrame`` for the consumer would rebuild what it does not read.  It is not
    a frame and claims nothing about one.

    Parameters
    ----------
    base : SampleBase
        The order-neutral state of the sample, with its raw inputs.
    labels : pandas.Index
        The user's row labels, in input order, one per input row.
    dependence : DependenceProvenance, optional
        Fit-used and retained-only column provenance; ``None`` without any.
    """

    base: SampleBase
    labels: pd.Index[Any]
    dependence: DependenceProvenance | None

    def __post_init__(self) -> None:
        """Refuse any part that is not of its concrete type."""
        if type(self.base) is not SampleBase:
            raise TypeError("base must be a concrete SampleBase")
        if not isinstance(self.labels, pd.Index):
            raise TypeError("labels must be a pandas Index")
        if (
            self.dependence is not None
            and type(self.dependence) is not DependenceProvenance
        ):
            raise TypeError("dependence must be a concrete DependenceProvenance")

    @classmethod
    def of_frame(cls, prepared: PreparedFrame) -> MetadataParts:
        """The parts of a concrete ``PreparedFrame``, as the frame hands them out.

        The base and the dependence provenance are the frame's own records; the
        labels are the fresh owned copy of :attr:`PreparedFrame.labels`.
        """
        if type(prepared) is not PreparedFrame:
            raise TypeError("prepared must be a concrete PreparedFrame")
        return cls(
            base=prepared.base, labels=prepared.labels, dependence=prepared.dependence
        )


# --------------------------------------------------------------- the capture


@dataclass(frozen=True)
class PrimitiveCapture:
    """The complete prepared frame as owned immutable values (module docstring).

    Formed by :meth:`capture`; equality is structural over bytes, tuples and plain
    scalars, which :meth:`check_frame` uses to validate a live frame.
    """

    labels: _Axis
    columns: tuple[Any, ...]
    subset_missing: int
    base: _BaseEnc
    frame: _FrameEnc
    effects: _EffectsEnc | None
    dependence: _DependenceEnc | None

    @classmethod
    def encode(cls, prepared: object) -> PrimitiveCapture:
        """Encode ``prepared`` exactly, or raise :class:`_IneligibleError` with why."""
        if type(prepared) is not PreparedFrame:
            raise _IneligibleError("prepared is not a concrete PreparedFrame")
        p = prepared
        if type(p.subset_missing) is not int:
            raise _IneligibleError("subset_missing is not a plain int")
        return cls(
            labels=_enc_axis(
                p._labels, "row labels", "row labels are not a plain Index"
            ),
            columns=_enc_columns(p.columns),
            subset_missing=p.subset_missing,
            base=_BaseEnc.of(p.base),
            frame=_FrameEnc.of(p._frame),
            effects=None if p.effects is None else _EffectsEnc.of(p.effects),
            dependence=None
            if p.dependence is None
            else _DependenceEnc.of(p.dependence),
        )

    @classmethod
    def capture(
        cls, prepared: PreparedFrame
    ) -> tuple[PrimitiveCapture, PreparedFrame] | str:
        """Capture ``prepared`` and materialize it once, exactly.

        Returns the capture and its fresh materialization, which re-encodes to the
        identical capture (the caller compares that materialization with the original
        under the accepted frame comparison and may then adopt it as private detached
        storage: it shares nothing with the capture or the original); or the reason the
        frame keeps the general route.
        """
        try:
            captured = cls.encode(prepared)
            detached = captured.prepared_frame()
            if cls.encode(detached) != captured:
                return "primitive materialization is not exact"
        except _IneligibleError as error:
            return str(error.args[0])
        return captured, detached

    # -- materialization ------------------------------------------------------

    def columns_record(self) -> Columns:
        """A fresh ``Columns`` record."""
        return Columns(*self.columns)

    def labels_index(self) -> pd.Index[Any]:
        """A fresh owned row ``Index``."""
        return self.labels.index()

    def sample_base(self) -> SampleBase:
        """A fresh ``SampleBase``."""
        return self.base.sample_base()

    def model_frame(self) -> ModelFrame:
        """A fresh ``ModelFrame``."""
        return self.frame.model_frame()

    def effect_columns(self) -> EffectColumns | None:
        """Fresh ``EffectColumns``, or ``None``."""
        return None if self.effects is None else self.effects.effect_columns()

    def dependence_provenance(self) -> DependenceProvenance | None:
        """Fresh ``DependenceProvenance``, or ``None``."""
        return None if self.dependence is None else self.dependence.dependence()

    def prepared_frame(self) -> PreparedFrame:
        """A fresh, detached ``PreparedFrame`` through its constructor."""
        return PreparedFrame(
            frame=self.model_frame(),
            labels=self.labels_index(),
            columns=self.columns_record(),
            base=self.sample_base(),
            subset_missing=self.subset_missing,
            effects=self.effect_columns(),
            dependence=self.dependence_provenance(),
        )

    def metadata_parts(self) -> MetadataParts:
        """Fresh base, labels and dependence provenance, as one :class:`MetadataParts`.

        The three parts that dependence metadata binding reads, each materialized as
        :meth:`sample_base`, :meth:`labels_index` and :meth:`dependence_provenance`
        materialize it, detached from the capture, the original and every other
        materialization alike; the model frame, the effect columns and the
        ``PreparedFrame`` constructor, none of which that binding reads, are not run.
        """
        return MetadataParts(
            base=self.sample_base(),
            labels=self.labels_index(),
            dependence=self.dependence_provenance(),
        )

    # -- validation of live objects -------------------------------------------
    # A live sample is validated by the once-materialized parts of the capture
    # (:class:`~powermean.results.primitive_parts.PrimitiveParts`); a live frame by
    # re-encoding it here.

    def matches_frame(self, frame: object) -> bool:
        """Whether the live ``frame`` encodes to exactly this capture."""
        try:
            return PrimitiveCapture.encode(frame) == self
        except _IneligibleError:
            return False

    def check_frame(self, frame: object) -> None:
        """Raise ``ValueError`` unless the live ``frame`` encodes to this capture."""
        if not self.matches_frame(frame):
            raise ValueError(
                f"PreparedFrame contents do not match issued baseline ({UNBOUND})"
            )

    def matches_base(self, base: object) -> bool:
        """Whether the live ``base`` encodes to exactly this capture's base."""
        try:
            return _BaseEnc.of(base) == self.base
        except _IneligibleError:
            return False


def primitive_ineligibility(prepared: object) -> str | None:
    """Return why ``prepared`` cannot take the primitive route, or ``None`` if it can.

    Conservative by construction: the exact encoding of the module docstring is
    attempted and its first refusal is the reason.  The text is recorded by the
    context that served the general route instead.
    """
    try:
        PrimitiveCapture.encode(prepared)
    except _IneligibleError as error:
        return str(error.args[0])
    return None
