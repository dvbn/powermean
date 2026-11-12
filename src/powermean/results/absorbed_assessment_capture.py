"""Owned replicas of the inputs of one absorbed point certificate (internal).

Part of the internal within-fit full-assessment handoff, with the encoding and
materialization discipline of :mod:`powermean.results.primitive_snapshot`.  Only the
modules of the handoff (:mod:`powermean.results.absorbed_assessment_handoff`) import
this module. Normal joint composition uses that handoff. WZ adds its source, numerical
options and independent bounds to the same closed capture, sharing the replica's current
grouping through the existing memo rather than retaining the caller's grouping.

What a certificate reads.  :func:`~powermean.solve.absorbed_certificate.certify_point`
of a profiled or fitted-index point, with the built-in provider and family, reads:

- the sample: numeric record, transformed problem, ledger, blocks with their codes,
  counts and ordered level labels, scaled design, active columns, reasons and order;
- the family, the point record and every record it holds;
- the absorbed options, the four certificate controls and the covariance request.

It reads neither ``FESample.base`` nor ``FESample.source``, nor the ``source`` and
``ordinals`` of a :class:`~powermean.data.sample_fe_source.Derived` grouping: no module
it reaches names them.  Those four fields are not captured.

Capture (:func:`capture`).  The roots are encoded in order, with one memo, into a tuple
of tokens built only of ``bytes``, ``str``, ``int``, ``bool``, ``None`` and tuples of
these:

- a float by its eight bytes, so that a NaN and a signed zero keep their bits; a NumPy
  scalar of a closed list of types by its dtype and bytes;
- a tuple or list of one exact type among ``bool``, ``int``, ``str`` and ``bytes`` by
  that type and its items, any other by its length and items;
- an array of the exact class ``ndarray``, of a plain dtype (Booleans, integers, floats,
  complex, times, bytes, text) with items of nonzero width, by its dtype string, shape,
  strides, contiguity, write flag and bytes;
- an exact ``dict`` with ``str`` keys by its keys in order and its values; an
  :class:`~powermean.data.raw._OwnedMapping` by the exact ``dict`` it carries;
- a record of the closed list :data:`RECORDS` by its position there and its fields, read
  from the instance's own storage, an exact ``dict`` with ``str`` keys;
- an object met again by the position of its first visit.

Types are read with ``type`` and compared by identity, so no method of a caller's object
runs and no caller's class is hashed.  Any other object leaves the inputs uncapturable
(``None``): a caller's class or subclass, an array subclass, an object or structured
dtype, a ``mappingproxy`` (whose contents cannot be read without running what is behind
it), a set.  A capture keeps no input object: its tokens are values, and the strings and
integers among them are immutable.

Replica (:func:`replicate`).  The tokens are materialized into new objects in the order
of the visits:

- every record of :data:`RECORDS` is a new instance of its exact class holding its
  encoded fields (``object.__new__``: no constructor runs, nothing is recomputed);
- every array is a new read-only array over the capture's own immutable bytes, as
  :func:`powermean.results.owned.owned_array` keeps its copy;
- every float and NumPy scalar is rebuilt from its bytes;
- every tuple, list, dict and owned mapping is a new one;
- every alias is the same new object.

The fields that are not captured are absent, so reading one raises ``AttributeError``:
no value that the capture does not hold can be read.  A replica is admitted only when it
encodes again to the very tokens it was materialized from.  It shares no record, array
or container with the inputs, so no later change of them, through any object, reaches
anything it holds.  ``prefix`` reuses an earlier replica of the first roots for the same
positions, when this capture begins with the tokens of that replica's capture.

Nothing here is a cache or a proof that a value cannot change.  Two captures are equal
exactly when the same values were read in the same structure, bit for bit.  No identity,
hash, marker, write flag or frozen status is evidence.
"""

from __future__ import annotations

import operator
import struct
from dataclasses import dataclass, fields
from itertools import repeat
from typing import Any, cast

import numpy as np

from powermean.data.accounting import MissingAccount
from powermean.data.design import ScaledDesign
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inputs import RegressionData
from powermean.data.raw import Column, _OwnedMapping
from powermean.data.request_values import OMITTED, Omitted
from powermean.data.sample_fe_source import Derived
from powermean.data.sample_fe_state import FESample, RowLedger
from powermean.data.weights import ObservationWeights
from powermean.fe.absorb import AbsorptionResult, AbsorptionState
from powermean.fe.connectivity import IndicatorRank
from powermean.fe.groups import Block, Grouping
from powermean.fe.profile import ProfileResult, ProfileState, Work
from powermean.fe.profile_nb2 import ScalarWork
from powermean.inference.wz_binding import WZSource
from powermean.inference.wz_covariance import WZControls
from powermean.model.order import Selection, TransformedProblem
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_direction import IndexPoint, Normalization, Recovery
from powermean.solve.assessment import CertificateControls
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.options import SolverOptions
from powermean.systems.profiled import PointProfile

Token = tuple[object, ...]

# The records whose fields are encoded: the classes of the inputs of a nonlinear point
# certificate, each exact (a subclass is another class and is not encoded).
RECORDS: tuple[type[Any], ...] = (
    FESample,
    RowLedger,
    MissingAccount,
    RegressionData,
    ObservationWeights,
    TransformedProblem,
    Selection,
    Grouping,
    Derived,
    Block,
    Column,
    ScaledDesign,
    PowerVariance,
    NB2Variance,
    PointProfile,
    ProfileState,
    ProfileResult,
    Work,
    ScalarWork,
    IndexPoint,
    Recovery,
    Normalization,
    AbsorptionResult,
    AbsorptionState,
    IndicatorRank,
    AbsorbedOptions,
    CertificateControls,
    CovarianceRequest,
    ConventionValues,
    WZSource,
    WZControls,
    AttemptBounds,
    SolverOptions,
)
# The fields that no certificate reads (module docstring), by record.
SKIPPED: tuple[tuple[type[Any], frozenset[str]], ...] = (
    (FESample, frozenset({"base", "source"})),
    (Derived, frozenset({"source", "ordinals"})),
)
# The dtype kinds of an encoded array: never objects ("O") or structured storage ("V").
_KINDS = "biufcmMSU"
# The exact NumPy scalar types of a field, each encoded by its dtype and bytes.
_SCALARS: tuple[type[Any], ...] = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
)
_PLAIN = ("bool", "int", "str", "bytes")


class _UncapturableError(Exception):
    """The inputs hold an object outside the closed encoding; ``args[0]`` says what."""


def _plain(kind: type[Any]) -> bool:
    """Whether ``kind`` is exactly ``bool``, ``int``, ``str`` or ``bytes``."""
    return kind is bool or kind is int or kind is str or kind is bytes


def _position(kind: type[Any]) -> int | None:
    """The position of ``kind`` in :data:`RECORDS`, by identity; ``None`` otherwise."""
    return next((i for i, record in enumerate(RECORDS) if kind is record), None)


def _names(kind: type[Any]) -> tuple[str, ...]:
    """The encoded fields of a record of :data:`RECORDS`, in their declared order."""
    skipped = next((names for record, names in SKIPPED if kind is record), frozenset())
    return tuple(f.name for f in fields(kind) if f.name not in skipped)


@dataclass(frozen=True, eq=False)
class Capture:
    """The tokens of some roots, with the tokens and visits so far after each root.

    ``ends[i]`` is the number of tokens and ``counts[i]`` the number of objects visited
    (records, arrays, tuples, lists, dicts, owned mappings) once root ``i`` is encoded.
    """

    tokens: tuple[Token, ...]
    ends: tuple[int, ...]
    counts: tuple[int, ...]

    def prefix(self, roots: int) -> Capture:
        """The capture of the first ``roots`` roots, which these tokens begin with."""
        end = self.ends[roots - 1]
        return Capture(self.tokens[:end], self.ends[:roots], self.counts[:roots])

    def rest(self, roots: int) -> tuple[Token, ...]:
        """The tokens of the roots after the first ``roots``."""
        return self.tokens[self.ends[roots - 1] :]


class _Encoder:
    """The tokens and the memo of one capture."""

    def __init__(self) -> None:
        self.tokens: list[Token] = []
        self.memo: dict[int, int] = {}

    def visit(self, value: object) -> None:
        """Encode ``value`` (module docstring)."""
        kind = type(value)
        if value is None:
            self.tokens.append(("none",))
        elif kind is Omitted:
            self.tokens.append(("omitted",))
        elif kind is float:
            self.tokens.append(("float", struct.pack("<d", value)))
        elif _plain(kind):
            self.tokens.append((kind.__name__, value))
        elif any(kind is scalar for scalar in _SCALARS):
            scalar = cast("np.generic", value)
            self.tokens.append(("scalar", scalar.dtype.str, scalar.tobytes()))
        else:
            self._shared(value, kind)

    def _shared(self, value: object, kind: type[Any]) -> None:
        """An object met before by its first position, else encoded and numbered."""
        seen = self.memo.get(id(value))
        if seen is not None:
            self.tokens.append(("seen", seen))
            return
        self.memo[id(value)] = len(self.memo)
        if kind is tuple or kind is list:
            self._sequence(value, "tuple" if kind is tuple else "list")
        elif kind is np.ndarray:
            self._array(value)
        elif kind is dict:
            self._mapping(value)
        elif kind is _OwnedMapping:
            carried: object = cast("_OwnedMapping[Any]", value)._items
            if type(carried) is not dict:
                raise _UncapturableError("an owned mapping that carries no exact dict")
            self.tokens.append(("owned",))
            self.visit(carried)
        else:
            self._record(value, kind)

    def _sequence(self, value: Any, name: str) -> None:
        """A tuple or list: one exact plain type and its items, else item by item."""
        if value:
            first = type(value[0])
            if _plain(first) and all(
                map(operator.is_, map(type, value), repeat(first))
            ):
                # A new tuple: ``tuple`` of an exact tuple is that very object.
                self.tokens.append((name, first.__name__, (*value,)))
                return
        self.tokens.append((name, len(value)))
        for item in value:
            self.visit(item)

    def _array(self, array: Any) -> None:
        """An exact ``ndarray`` of a plain dtype by its layout and bytes."""
        dtype = array.dtype
        if dtype.hasobject or dtype.kind not in _KINDS or not dtype.itemsize:
            raise _UncapturableError("an array of object, structured or empty items")
        flags = array.flags
        layout = (flags.c_contiguous, flags.f_contiguous, flags.writeable)
        found = ("ndarray", dtype.str, array.shape, array.strides, *layout)
        self.tokens.append((*found, array.tobytes()))

    def _mapping(self, mapping: Any) -> None:
        """An exact ``dict`` with ``str`` keys: its keys in order, then its values."""
        items = tuple(mapping.items())
        keys = tuple(key for key, _ in items)
        if not all(map(operator.is_, map(type, keys), repeat(str))):
            raise _UncapturableError("a mapping key that is not a str")
        self.tokens.append(("dict", keys))
        for _, item in items:
            self.visit(item)

    def _record(self, record: object, kind: type[Any]) -> None:
        """A record of :data:`RECORDS` by its position and fields, from its storage."""
        position = _position(kind)
        if position is None:
            raise _UncapturableError("an object outside the closed encoding")
        storage = vars(record)
        if type(storage) is not dict or not all(
            map(operator.is_, map(type, storage), repeat(str))
        ):
            raise _UncapturableError("a record whose storage is not a plain dict")
        names = _names(kind)
        if any(name not in storage for name in names):
            raise _UncapturableError("a record without its fields")
        self.tokens.append(("record", position, names))
        for name in names:
            self.visit(storage[name])


def capture(roots: tuple[object, ...]) -> Capture | None:
    """The capture of ``roots`` (module docstring); ``None`` when uncapturable.

    Parameters
    ----------
    roots : tuple
        The objects to encode, in order, with one memo.

    Returns
    -------
    Capture or None
        The tokens, and the tokens and visits so far after each root; ``None`` for an
        object outside the closed encoding.
    """
    encoder = _Encoder()
    ends: list[int] = []
    counts: list[int] = []
    try:
        for root in roots:
            encoder.visit(root)
            ends.append(len(encoder.tokens))
            counts.append(len(encoder.memo))
    except _UncapturableError:
        return None
    return Capture(tuple(encoder.tokens), tuple(ends), tuple(counts))


# ------------------------------------------------------------------ the replica


@dataclass(frozen=True, eq=False)
class Replica:
    """The new objects of a capture: its roots and every object, by position."""

    capture: Capture
    roots: tuple[object, ...]
    objects: tuple[object, ...]

    def prefix(self, roots: int) -> Replica:
        """The replica of the first ``roots`` roots, the same objects."""
        count = self.capture.counts[roots - 1]
        found = self.capture.prefix(roots)
        return Replica(found, self.roots[:roots], self.objects[:count])


_PENDING = object()  # a position whose object is still being formed


class _Materializer:
    """The new objects of a capture's tokens, numbered as the encoder visited them."""

    def __init__(self, tokens: tuple[Token, ...], at: int, objects: list[object]):
        self.tokens = tokens
        self.at = at
        self.objects = objects

    def next(self) -> object:
        """The value of the next token and of the tokens that it holds."""
        token = self.tokens[self.at]
        self.at += 1
        tag = token[0]
        if tag == "none":
            return None
        if tag == "omitted":
            return OMITTED
        if tag == "float":
            return struct.unpack("<d", cast("bytes", token[1]))[0]
        if tag in _PLAIN:
            return token[1]
        if tag == "scalar":
            dtype = np.dtype(cast("str", token[1]))
            return np.frombuffer(cast("bytes", token[2]), dtype=dtype)[0]
        if tag == "seen":
            found = self.objects[cast("int", token[1])]
            if found is _PENDING:
                raise _UncapturableError("an object that holds itself")
            return found
        position = len(self.objects)
        self.objects.append(_PENDING)
        value = self._shared(tag, token)
        self.objects[position] = value
        return value

    def _shared(self, tag: object, token: Token) -> object:
        """A new tuple, list, array, dict, owned mapping or record."""
        if tag in ("tuple", "list"):
            items = (
                list(cast("tuple[object, ...]", token[2]))
                if len(token) == 3
                else [self.next() for _ in range(cast("int", token[1]))]
            )
            return tuple(items) if tag == "tuple" else items
        if tag == "ndarray":
            dtype, shape = np.dtype(cast("str", token[1])), cast("tuple[int]", token[2])
            return np.frombuffer(cast("bytes", token[7]), dtype=dtype).reshape(shape)
        if tag == "dict":
            keys = cast("tuple[str, ...]", token[1])
            return {key: self.next() for key in keys}
        if tag == "owned":
            carried = self.next()
            if type(carried) is not dict:
                raise _UncapturableError("an owned mapping that carries no exact dict")
            return _OwnedMapping(cast("dict[str, Any]", carried))
        kind = RECORDS[cast("int", token[1])]
        names = cast("tuple[str, ...]", token[2])
        if names != _names(kind):
            raise _UncapturableError("a record of other fields")
        record = object.__new__(kind)
        for name in names:
            object.__setattr__(record, name, self.next())
        return record


def replicate(captured: Capture, *, prefix: Replica | None = None) -> Replica | None:
    """The replica of ``captured`` (module docstring); ``None`` unless exact.

    Parameters
    ----------
    captured : Capture
        The capture to materialize.
    prefix : Replica, optional
        The replica of the first roots of an earlier capture, reused for the same
        positions when ``captured`` begins with exactly that capture's tokens; the
        replica then holds those very tokens, which the prefix's arrays view.

    Returns
    -------
    Replica or None
        The new roots and objects, which encode again to the tokens of ``captured``;
        ``None`` when they do not, or when ``captured`` does not begin with ``prefix``.
    """
    roots: list[object] = []
    objects: list[object] = []
    at = 0
    if prefix is not None:
        given = prefix.capture
        count = len(given.ends)
        if (
            len(captured.ends) <= count
            or captured.ends[:count] != given.ends
            or captured.counts[:count] != given.counts
            or captured.tokens[: given.ends[-1]] != given.tokens
        ):
            return None
        roots, objects, at = list(prefix.roots), list(prefix.objects), given.ends[-1]
        # The equal tokens of the prefix are its own: the bytes that its replica views.
        tokens = given.tokens + captured.tokens[at:]
        captured = Capture(tokens, captured.ends, captured.counts)
    materializer = _Materializer(captured.tokens, at, objects)
    try:
        while len(roots) < len(captured.ends):
            roots.append(materializer.next())
    except _UncapturableError:
        return None
    found = Replica(captured, tuple(roots), tuple(materializer.objects))
    again = capture(found.roots)
    if again is None or again.tokens != captured.tokens:
        return None
    return found
