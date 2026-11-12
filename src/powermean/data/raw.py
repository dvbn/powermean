"""The raw inputs of the sample construction: every input row, before any accounting.

:class:`RawInputs` holds the arrays of every input row, aligned by position and in input
order, as the interface evaluated them and before any row is excluded: the raw outcome,
the design trained by Formulaic on the full frame, the raw weights, the offset or the
exposure, and the aligned metadata that later stages need (cluster labels, time, unit or
location information).  A value may be missing or infinite where the masks of the
interface say so: the record checks shapes and alignment, never values, and selects
nothing.  :func:`powermean.data.sample.build_sample` applies the steps of the sample
construction to it.

Two boundaries hold here.  First, the raw record is not the validated numeric record:
:class:`~powermean.data.inputs.RegressionData` holds finite values only and would refuse
a row that the accounting excludes first, so it is formed on the retained rows alone.
Second, the raw weights keep their original objects: a frequency count arrives as the
Python or NumPy integer that the user gave, in an object array where the column holds
such objects, and is never converted to a double before the exact check of
:class:`~powermean.data.weights.ObservationWeights` on the retained rows.  A ``2**53 +
1`` on a row that the subset excludes is therefore never seen by that check, and the
same count on a retained row is still refused.  A list of Python numbers is kept as the
objects given, never inferred into a narrower type: ``[2**53 + 1, 0.0]`` would otherwise
become doubles, and the first count would round to ``2**53`` before any check could see
it.

Ownership (:class:`Column`).  A plain array (integers, doubles, Booleans, strings,
dates) is kept in immutable bytes of its own, :func:`powermean.model.rows._immutable`
for doubles and the same bytes with their own type otherwise, which NumPy refuses to
make writable again.  An object array has no plain bytes, and any holder of an array
that owns its memory can set its write flag again; and its elements are objects, which a
caller can change in place where they are mutable (a label that is an instance of a
class of the caller, say), through the object given or through one handed out.  So an
object column keeps its own copies of the elements (:func:`_owned_objects`: an element
of an immutable built-in type as it is, every other element its ``copy.deepcopy``, with
one memo for the column, so that an object that occurs twice is one copy that occurs
twice), and every access, through :attr:`Column.values`, :attr:`Column.stored` or a
selection, forms a fresh read-only array, or tuple, of fresh copies of them.  An array
or an element that a caller got and changed is the caller's own: the record is
unchanged, and so is every record derived from it.  A later change of the caller's
input, of an array or of an element, reaches no record either.  What is kept is the type
and the value of each element, and the equivalences within a column; not the identity of
the caller's objects.  Booleans are not weights (the exact-count check refuses them, and
R and Stata take numeric weights only), and neither are strings, dates or complex
values.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from dataclasses import dataclass
from itertools import chain, repeat
from types import MappingProxyType
from typing import Any, Never, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.data.inputs import _column_names
from powermean.data.weights import WEIGHT_KINDS, WeightKind
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray

AnyArray = NDArray[Any]
# The kinds of NumPy arrays that may hold weights: integers, unsigned integers, floats
# and objects (exact Python integers, fractions, or a column with missing markers).
WEIGHT_KINDS_OF_ARRAYS = "iufO"
# The built-in types whose instances nothing can change: an element of one of them is
# its own copy.  Exact types, so that a subclass is copied like any other object.
_IMMUTABLE_TYPES = frozenset({int, float, complex, bool, str, bytes, type(None)})


def _elements(values: ArrayLike, name: str) -> AnyArray:
    """``values`` as a one-dimensional array: an array as it is, a sequence as objects.

    A list or a tuple becomes an object array of the objects given, so that NumPy
    infers no narrower type for it: a Python integer beyond ``2**53`` beside a float
    stays that integer.  A nested list is not one-dimensional and is refused.
    """
    array: AnyArray
    if isinstance(values, np.ndarray):
        array = values
    elif isinstance(values, list | tuple):
        array = np.array(values, dtype=object)
    else:
        array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _object_array(elements: Sequence[object]) -> AnyArray:
    """A fresh read-only object array of ``elements``, each the object as it is."""
    if type(elements) is tuple or type(elements) is list:
        # Object fromiter stores each item, including a tuple label, as one object;
        # sequence broadcasting would instead expand nested labels. Exact built-in
        # sequences have no caller-defined iteration or length callbacks.
        array = np.fromiter(elements, dtype=object, count=len(elements))
    else:
        array = np.empty(len(elements), dtype=object)
        for position, element in enumerate(elements):
            array[position] = element
    array.setflags(write=False)
    return array


def _flat_immutable_tuples(elements: Sequence[object]) -> bool:
    """The whole sequence consists of exact tuples of immutable scalar types.

    The caller admits an exact list or tuple. Prove tuple elements before
    flattening, then exact built-in metaclasses before hashed membership: neither
    scan may invoke a caller's iteration, hash or equality callback. Batch
    iterators avoid one Python helper and generator per label; the two component
    scans keep constant auxiliary space and no state beyond this call.
    """
    if not all(map(operator.is_, map(type, elements), repeat(tuple))):
        return False
    tuples = cast("Sequence[tuple[object, ...]]", elements)
    kinds = map(type, chain.from_iterable(tuples))
    if not all(map(operator.is_, map(type, kinds), repeat(type))):
        return False
    return all(
        map(_IMMUTABLE_TYPES.__contains__, map(type, chain.from_iterable(tuples)))
    )


def _owned_objects(elements: Sequence[object]) -> tuple[object, ...]:
    """Copies of ``elements`` that share nothing mutable with them.

    An element of an immutable built-in type (:data:`_IMMUTABLE_TYPES`: ``int``,
    ``float``, ``complex``, ``bool``, ``str``, ``bytes``, ``None``) is itself, since
    nothing can change it; every other element is its :func:`copy.deepcopy`, of the
    same type and value, with one memo for the whole sequence, so that an object that
    occurs twice becomes one copy that occurs twice and two equal objects two equal
    copies.  An object whose copy is itself by its own definition stays itself (the
    missing markers of pandas, a ``Fraction``, a NumPy scalar's value in a fresh
    scalar); an object that cannot be copied raises the error of ``copy.deepcopy``,
    since a column cannot own what it cannot copy.

    A fresh same-call identity check of a stable built-in sequence can establish
    that every element has one exact immutable type, or is an exact flat tuple of
    the immutable scalar types. Deepcopy already returns each such tuple unchanged.
    Only then is a fresh outer tuple formed directly. The whole sequence is checked
    before copying: a custom copier elsewhere could populate the shared memo for a
    tuple. Nested tuples and all other values retain the original copying loop.
    """
    if (type(elements) is tuple or type(elements) is list) and elements:
        kind = type(elements[0])
        if any(kind is t for t in _IMMUTABLE_TYPES) and all(
            map(operator.is_, map(type, elements), repeat(kind))
        ):
            return tuple(iter(elements))
        if kind is tuple and _flat_immutable_tuples(elements):
            return tuple(iter(elements))
    memo: dict[int, Any] = {}
    return tuple(
        element if type(element) in _IMMUTABLE_TYPES else copy.deepcopy(element, memo)
        for element in elements
    )


@dataclass(frozen=True, eq=False, init=False)
class Column:
    """One column of the input rows, owned, handed out read-only.

    Parameters
    ----------
    values : array_like
        One value per row: an array of any type, kept as its type; a list or a tuple,
        kept as the objects given, never inferred into a narrower type.  The elements
        of an object column are copied (:func:`_owned_objects`).
    name : str, default "values"
        The name of the values in the message of a refusal.

    Attributes
    ----------
    stored : array or tuple
        The retained form: the plain array in immutable bytes of its own, or, for an
        object column, a fresh tuple of fresh copies of its elements on every access.

    Raises
    ------
    ValueError
        Values that are not one-dimensional.

    Notes
    -----
    :attr:`values` is the plain array itself, which NumPy cannot make writable again,
    or a fresh read-only array of fresh copies of the objects, formed on every access,
    as :attr:`stored` and :meth:`take` are.  An array that a caller got and made
    writable, and an element that it got and changed, are the caller's own; the
    column is unchanged, and a later change of the values or of the objects given
    reaches it no more.  The copies keep the type and the value of each element and
    the equivalences within the column (an object that occurs twice is one copy that
    occurs twice); they are not the caller's objects.
    """

    _stored: AnyArray | tuple[object, ...]

    def __init__(self, values: ArrayLike, *, name: str = "values") -> None:
        """Keep the values in their owned form."""
        array = _elements(values, name)
        stored: AnyArray | tuple[object, ...]
        if array.dtype.kind == "O":
            stored = _owned_objects(array.tolist())
        else:
            stored = np.frombuffer(array.tobytes(), dtype=array.dtype)
        object.__setattr__(self, "_stored", stored)

    @property
    def stored(self) -> AnyArray | tuple[object, ...]:
        """The retained form: the plain array, or a fresh tuple of fresh copies."""
        if isinstance(self._stored, tuple):
            return _owned_objects(self._stored)
        return self._stored

    @property
    def values(self) -> AnyArray:
        """The values, read-only: the plain array, or a fresh array of fresh copies."""
        if isinstance(self._stored, tuple):
            return _object_array(_owned_objects(self._stored))
        return self._stored

    @property
    def size(self) -> int:
        """The number of rows."""
        return len(self._stored)

    @property
    def dtype(self) -> np.dtype[Any]:
        """The NumPy type of the values; ``object`` for an object column."""
        if isinstance(self._stored, tuple):
            return np.dtype(object)
        return self._stored.dtype

    def take(self, positions: IndexArray) -> Column:
        """The column of the rows ``positions``, in that order, owned.

        The elements of an object column are copied once more for the new column.
        """
        if isinstance(self._stored, tuple):
            chosen = [self._stored[k] for k in positions.tolist()]
            return _object_column(chosen)
        return Column(self._stored[positions])


def _object_column(elements: Sequence[object]) -> Column:
    """Own one object per element without an intermediate object array.

    Exact lists and tuples already define the one-dimensional element sequence
    of Column(_object_array(elements)). Capture it before any deepcopy
    callback, then run the same fresh per-column ownership operation. The
    ndarray and its tolist were only reference containers; neither ran
    element callbacks. Other sequences retain that original route.
    """
    if type(elements) is not list and type(elements) is not tuple:
        return Column(_object_array(elements))
    captured = tuple(elements)
    stored = _owned_objects(captured)
    result = object.__new__(Column)
    object.__setattr__(result, "_stored", stored)
    return result


# A dict may defer ordering to an arbitrary reflected result from the other operand.
type _Comparison = Callable[[object, object], object]


class _InPlaceUnion(Protocol):
    """The runtime mappingproxy refusal slot, absent from its typing stub."""

    def __ior__(self, other: object, /) -> Never:
        """Delegate the immutable proxy's in-place union refusal."""
        ...


class _OwnedMapping[V](Mapping[str, V]):
    """A record's owned mapping: the exact ``dict`` it carries, handed out read-only.

    The four records that hold a mapping of columns or masks (:class:`AlignedColumns`,
    :class:`~powermean.data.sample_fe.EffectColumns`,
    :class:`~powermean._prepare.dependence.DependenceProvenance`, and
    :class:`~powermean.data.accounting.MissingAccount` by its internal owned route)
    put the exact ``dict`` that their constructor formed inside one of these and
    present that very object as their mapping field.  The ``dict`` is the record's
    current data and this object is its one projection: every read through it (an
    item, the names, the length, the key, item and value views, equality, a copy, the
    reversed names) is a read of that ``dict`` and of nothing else; a caller cannot set
    or delete an item, nor rebind the ``dict`` through an attribute.  The private
    per-fit snapshot (:mod:`powermean.results.primitive_snapshot`) reads the carried
    ``dict`` after checking its exact type and the exact type of every key and value,
    and infers nothing from the identity of this object or of the ``dict``: a mapping
    of any other class in a record's field, or another object in the place of the
    ``dict``, is refused unread and read by the general route as it always was.

    Mapping operations retain the previous ``mappingproxy`` dispatch, including
    caller hooks in comparisons, unions and key lookups. The temporary proxies
    used for display and operators are not retained or consulted by recognition.
    """

    __slots__ = ("_items",)

    _items: dict[str, V]

    def __init__(self, items: dict[str, V]) -> None:
        """Carry ``items``, an exact ``dict``, as it is; another type is refused."""
        if type(items) is not dict:
            raise TypeError("an owned mapping carries an exact dict")
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse: the carried dict is bound once, by the constructor."""
        raise AttributeError(f"{type(self).__name__} is read-only")

    def __delattr__(self, name: str) -> None:
        """Refuse: the carried dict is bound once, by the constructor."""
        raise AttributeError(f"{type(self).__name__} is read-only")

    def __reduce__(self) -> str | tuple[Any, ...]:
        """Refuse pickling and copying, as a ``mappingproxy`` does."""
        return MappingProxyType(self._items).__reduce__()

    def __getitem__(self, key: str) -> V:
        """The value of ``key``."""
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        """The keys, in the order of the dict."""
        return iter(self._items)

    def __reversed__(self) -> Iterator[str]:
        """The keys, in the reversed order of the dict."""
        return reversed(self._items)

    def __len__(self) -> int:
        """The number of items."""
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` is a key of the dict."""
        return key in self._items

    @staticmethod
    def _operand(other: Any) -> Any:
        """Unwrap another owned mapping for the old two-proxy operator dispatch."""
        return other._items if type(other) is _OwnedMapping else other

    def __eq__(self, other: object) -> Any:
        """The dict's equality, preserving a caller's reflected result."""
        return operator.eq(self._items, self._operand(other))

    def __ne__(self, other: object) -> Any:
        """The dict's inequality, not a negation of a caller's equality."""
        return operator.ne(self._items, self._operand(other))

    def __lt__(self, other: object) -> Any:
        """Delegate the comparison or its refusal as the previous proxy did."""
        return cast(_Comparison, operator.lt)(self._items, self._operand(other))

    def __le__(self, other: object) -> Any:
        """Delegate the comparison or its refusal as the previous proxy did."""
        return cast(_Comparison, operator.le)(self._items, self._operand(other))

    def __gt__(self, other: object) -> Any:
        """Delegate the comparison or its refusal as the previous proxy did."""
        return cast(_Comparison, operator.gt)(self._items, self._operand(other))

    def __ge__(self, other: object) -> Any:
        """Delegate the comparison or its refusal as the previous proxy did."""
        return cast(_Comparison, operator.ge)(self._items, self._operand(other))

    def __hash__(self) -> int:
        """Delegate the unhashable-dict refusal, as a ``mappingproxy`` does."""
        return hash(self._items)

    def __repr__(self) -> str:
        """The same display as the previous read-only mapping."""
        return repr(MappingProxyType(self._items))

    def __str__(self) -> str:
        """The dict's display, as a ``mappingproxy`` prints."""
        return str(self._items)

    def get(self, key: str, default: Any = None, /) -> Any:
        """Look up the key without swallowing a caller's ``KeyError``."""
        return self._items.get(key, default)

    def keys(self) -> KeysView[str]:
        """The dict's own view of the keys."""
        return self._items.keys()

    def items(self) -> ItemsView[str, V]:
        """The dict's own view of the items."""
        return self._items.items()

    def values(self) -> ValuesView[V]:
        """The dict's own view of the values."""
        return self._items.values()

    def copy(self) -> dict[str, V]:
        """A fresh dict of the items, as a ``mappingproxy`` copies."""
        return self._items.copy()

    def __or__(self, other: object) -> Any:
        """The proxy's union dispatch, including a caller's reflected operator."""
        return MappingProxyType(self._items).__or__(self._operand(other))

    def __ror__(self, other: object) -> Any:
        """The proxy's reflected union dispatch, including a caller's operator."""
        return MappingProxyType(self._items).__ror__(self._operand(other))

    def __ior__(self, other: object) -> Any:
        """Refuse, as a ``mappingproxy`` does: the mapping is read-only."""
        return cast(_InPlaceUnion, MappingProxyType(self._items)).__ior__(other)


@dataclass(frozen=True, eq=False, init=False)
class AlignedColumns(Mapping[str, AnyArray]):
    """Aligned columns of the rows, by name, each owned and handed out read-only.

    Parameters
    ----------
    columns : mapping of str to array_like or Column
        One column per name, with one value per row; a :class:`Column` is kept as it
        is.
    rows : int
        The number of rows that every column must have.

    Attributes
    ----------
    columns : mapping of str to Column
        The owned columns, by name, in the order given; read-only.
    rows : int
        The number of rows.

    Raises
    ------
    ValueError
        A key that is not a nonempty string; a column that is not one-dimensional or
        not of ``rows`` values.

    Notes
    -----
    ``aligned[name]`` is :attr:`Column.values`: the plain array, or a fresh read-only
    array of fresh copies of the objects.  The record compares by identity.

    :attr:`columns` is the record's :class:`_OwnedMapping`: the exact ``dict`` that
    the constructor formed, carried as the record's current data and handed out
    read-only through that one object.
    """

    columns: Mapping[str, Column]
    rows: int

    def __init__(self, columns: Mapping[str, ArrayLike | Column], rows: int) -> None:
        """Check each column and store it owned."""
        stored: dict[str, Column] = {}
        for key, values in columns.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metadata keys must be nonempty strings")
            if isinstance(values, Column):
                column = values
            else:
                column = Column(values, name=f"metadata[{key!r}]")
            if column.size != rows:
                raise ValueError(f"metadata[{key!r}] must have one value per input row")
            stored[key] = column
        object.__setattr__(self, "columns", _OwnedMapping(stored))
        object.__setattr__(self, "rows", rows)

    def __getitem__(self, key: str) -> AnyArray:
        """The values of the column ``key``, read-only."""
        return self.columns[key].values

    def __iter__(self) -> Iterator[str]:
        """The names, in the order given."""
        return iter(self.columns)

    def __len__(self) -> int:
        """The number of columns."""
        return len(self.columns)

    def __eq__(self, other: object) -> bool:
        """Identity: two records of equal columns are two inputs."""
        return self is other

    def __hash__(self) -> int:
        """The hash of the identity."""
        return id(self)

    def take(self, positions: IndexArray) -> AlignedColumns:
        """The columns of the rows ``positions``, in that order: a new owned record."""
        return AlignedColumns(
            {key: column.take(positions) for key, column in self.columns.items()},
            int(positions.size),
        )


def _real_rows(values: ArrayLike | None, rows: int, name: str) -> FloatArray | None:
    """A real vector of ``rows`` doubles, missing and infinite values allowed."""
    if values is None:
        return None
    if np.iscomplexobj(np.asarray(values)):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (rows,):
        raise ValueError(f"{name} must have one value per input row")
    return _immutable(array)


@dataclass(frozen=True, eq=False, init=False)
class RawWeights:
    """The raw weights of every input row, of one explicit kind, objects kept.

    Parameters
    ----------
    values : array_like
        One weight per input row: an array of integers, floats or objects (exact
        Python integers, fractions, or the missing markers of a column), kept as its
        type; a list or a tuple, kept as the objects given (:class:`Column`).  Never
        converted or normalized.
    kind : {"analytic", "frequency", "probability"}
        The kind of the weights, as the caller states it; never inferred.

    Attributes
    ----------
    column : Column
        The owned weights.
    kind : {"analytic", "frequency", "probability"}
        The kind, as given.

    Raises
    ------
    ValueError
        An unknown kind; values that are not one-dimensional; a Boolean, string, date
        or complex array.  An object that is not a real number is refused by the sample
        construction on the rows that the masks keep, where it is read.
    """

    column: Column
    kind: WeightKind

    def __init__(self, values: ArrayLike, kind: WeightKind) -> None:
        """Check the kind and the type of the array, and store the values owned."""
        chosen: object = kind
        if chosen not in WEIGHT_KINDS:
            raise ValueError("weight kind must be analytic, frequency or probability")
        column = Column(values, name="weights")
        if column.dtype.kind not in WEIGHT_KINDS_OF_ARRAYS:
            raise ValueError(
                "weights must be an array of integers, floats or real objects"
            )
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "kind", kind)

    @property
    def values(self) -> AnyArray:
        """The weights, read-only, as given (:attr:`Column.values`)."""
        return self.column.values

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return self.column.size


@dataclass(frozen=True, eq=False, init=False)
class RawInputs:
    """The aligned arrays of every input row, before the sample construction.

    Parameters
    ----------
    y : array_like
        The raw outcome, one real value per input row; missing or infinite where the
        outcome mask of the interface says so.
    design : array_like
        The design trained on the full frame, one row per input row and at least one
        column; missing or infinite where the masks say so.
    column_names : sequence of str, optional
        The names of the columns, unique and nonempty; ``x0``, ``x1``, ... if omitted.
    weights : RawWeights, optional
        The raw weights, aligned with ``y``.
    offset : array_like, optional
        The offset ``o`` of the log power mean, one value per input row.  Not together
        with ``exposure``.
    exposure : array_like, optional
        The exposure ``e``, one value per input row; its logarithm is formed by the
        numeric record on the retained rows.  Not together with ``offset``.
    metadata : mapping of str to array_like, optional
        Aligned columns kept for later stages (cluster labels, time, unit, location),
        one value per input row, of any type, each kept as given
        (:class:`AlignedColumns`).

    Raises
    ------
    ValueError
        Arrays that are not real or not aligned; no row or no column; both an offset
        and an exposure; malformed names or metadata.
    TypeError
        ``weights`` that is not a ``RawWeights``.

    Notes
    -----
    The record is frozen, compares by identity and owns read-only copies.  Nothing is
    validated beyond shape and type: values are the accounting's business.
    """

    y: FloatArray
    design: FloatArray
    column_names: tuple[str, ...]
    weights: RawWeights | None
    offset: FloatArray | None
    exposure: FloatArray | None
    metadata: AlignedColumns

    def __init__(
        self,
        y: ArrayLike,
        design: ArrayLike,
        *,
        column_names: Sequence[str] | None = None,
        weights: RawWeights | None = None,
        offset: ArrayLike | None = None,
        exposure: ArrayLike | None = None,
        metadata: Mapping[str, ArrayLike | Column] | None = None,
    ) -> None:
        """Check shapes, alignment and types, and store every array owned."""
        if np.iscomplexobj(np.asarray(y)) or np.iscomplexobj(np.asarray(design)):
            raise ValueError("y and design must be real-valued")
        outcome = np.asarray(y, dtype=np.float64)
        matrix = np.asarray(design, dtype=np.float64)
        if outcome.ndim != 1 or not outcome.size:
            raise ValueError("y must be a nonempty vector")
        rows = int(outcome.size)
        if matrix.ndim != 2 or matrix.shape[0] != rows:
            raise ValueError("design must have one row per input row")
        if not matrix.shape[1]:
            raise ValueError("design must have at least one column")
        if offset is not None and exposure is not None:
            raise ValueError("specify offset or exposure, not both")
        if weights is not None:
            if type(weights) is not RawWeights:
                raise TypeError("weights must be a RawWeights record")
            if weights.nobs != rows:
                raise ValueError("weights must have one value per input row")
        stored: dict[str, object] = {
            "y": _immutable(outcome),
            "design": _immutable(matrix),
            "column_names": _column_names(column_names, int(matrix.shape[1])),
            "weights": weights,
            "offset": _real_rows(offset, rows, "offset"),
            "exposure": _real_rows(exposure, rows, "exposure"),
            "metadata": AlignedColumns({} if metadata is None else metadata, rows),
        }
        for key, value in stored.items():
            object.__setattr__(self, key, value)

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return int(self.y.size)

    @property
    def nparams(self) -> int:
        """The number of columns of the design."""
        return int(self.design.shape[1])
