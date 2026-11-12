"""Range-checked primitives of the weighted absorption: means, walks, lifts, products.

These primitives serve weighted alternating projections on all columns at once, with
no dense indicator matrix.  Nothing here decides a budget, a stopping or a status: the
orchestration is :mod:`powermean.fe.absorb` and the accelerated recurrence
:mod:`powermean.fe.absorb_acceleration`.

Notation.  ``Y`` or ``z`` is an ``n``-by-``k`` matrix of values, ``w`` the row weights,
block ``b`` of the grouping (:mod:`powermean.fe.groups`) has codes ``c_b``
and ``L_b`` levels, its level totals are ``d_b[g] = sum_{c_b(i) = g} w_i``, and a
block coefficient array ``a_b`` is ``L_b``-by-``k``.  The weighted mean of block ``b``
is ``M_b(z)[g, j] = sum_{c_b(i) = g} w_i z_ij / d_b[g]`` and its projection
``Q_b z = z - M_b(z)[c_b, :]``.  A walk applies ``Q`` along an order of blocks and
accumulates the subtracted means as block coefficients, whose lift
``lift(a) = sum_b a_b[c_b, :]`` is the subtracted component in real arithmetic.

Weights.  Nonnegative finite row weights with a finite strictly positive total at
every level of every block (:func:`prepare_weights`).  A row of zero weight is kept
in its position, enters every product as an exact zero and is never a divisor; no
weight is floored, clipped, normalized or dropped.  Positive totals make each block
projection well defined; they are not a certificate of full weighted rank or of
unique fitted values on zero-weight rows.

Range.  Every arithmetic operation of this module runs under
``numpy.errstate(all="raise")``: an invalid operation, an overflow, a division by
zero or an inexact underflow raises ``FloatingPointError`` and no later finite value
erases it.  IEEE binary64 and NumPy signal underflow only for a result that is tiny
AND inexact, so an exactly representable subnormal product or quotient passes and a
nonzero product rounded to zero refuses (for instance ``w = 2^-1074``,
``z = 1/2``); exact zeros and exact cancellations pass.  Additions never produce an
inexact tiny result in binary64, so the checked scope of the reductions is overflow
and invalid, which the reduction reports.  No floor, bound, rescaling or substitute
tolerance exists here.

Order of operations, documented for the comparisons.  Means: the products ``w_i
z_ij``, then the sequential row-order sum of each level through ``group_sum`` (the
``np.add.at`` order), then the division by the total.  Walk: for each visited
block, the means of the current rows, the coefficients increased by them, the rows
decreased by their gathered means.  Lift: the gathered coefficients added in block
order.  Inner products: ``(a_ij b_ij) w_i`` summed in row order through the same
reduction.  These are the adapted mechanics, documented, not a claim of bitwise
parity with any source routine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.groups import Block, Grouping
from powermean.fe import reductions as _reductions
from powermean.fe.reductions import (
    _codes,
    _integer,
    _values,
    group_sum,
    immutable_array,
)
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

Coefficients = tuple[FloatArray, ...]
BoolArray = np.ndarray[tuple[int], np.dtype[np.bool_]]

WEIGHTS = "weights must be a real vector with one finite nonnegative value per row"
TOTALS = "every level of every block must have a finite positive total weight"
GROUPING = "grouping must be a Grouping record"


# ------------------------------------------------------------- checked arithmetic


def _checked(operation: Callable[[], FloatArray], what: str) -> FloatArray:
    """Run ``operation`` with every floating-point condition raised, named by ``what``.

    Invalid, overflow, division by zero and inexact underflow stop the evaluation
    with ``FloatingPointError``; the message names the operation and the condition.
    """
    with np.errstate(all="raise"):
        try:
            return operation()
        except FloatingPointError as error:
            raise FloatingPointError(f"{what}: {error}") from error


def checked_product(a: FloatArray, b: FloatArray, what: str = "product") -> FloatArray:
    """``a * b`` with NumPy broadcasting, every element range-checked."""
    return _checked(lambda: np.multiply(a, b), what)


def checked_sum(a: FloatArray, b: FloatArray, what: str = "sum") -> FloatArray:
    """``a + b`` with NumPy broadcasting, range-checked (overflow, invalid)."""
    return _checked(lambda: np.add(a, b), what)


def checked_difference(
    a: FloatArray, b: FloatArray, what: str = "difference"
) -> FloatArray:
    """``a - b`` with NumPy broadcasting, range-checked (overflow, invalid)."""
    return _checked(lambda: np.subtract(a, b), what)


def checked_quotient(
    a: FloatArray, b: FloatArray, what: str = "quotient"
) -> FloatArray:
    """``a / b`` with NumPy broadcasting, every element range-checked."""
    return _checked(lambda: np.divide(a, b), what)


def group_sums(values: FloatArray, codes: IndexArray, n_levels: int) -> FloatArray:
    """The sequential row-order sums of ``values`` by level, through ``group_sum``.

    The accepted reduction of :mod:`powermean.fe.reductions` raises on an overflow
    or an invalid sum; here it also runs with underflow raised, which an addition
    cannot signal, so that the reduction and the surrounding products share one
    error state.  ``values`` is a vector or a matrix with one row per code.

    Plain double one-column matrices use the same checked row-order ``add.at``
    on their owned column view. Generic inputs retain the original conversion
    and callback path, and vectors and wider matrices retain ``group_sum``.
    """
    if (
        type(values) is np.ndarray
        and values.dtype == np.float64
        and values.ndim == 2
        and values.shape[1] == 1
        and type(codes) is np.ndarray
        and codes.dtype.kind in "iu"
        and type(n_levels) is int
    ):
        return _checked(lambda: _column_sum(values, codes, n_levels), "group sum")
    return _checked(lambda: group_sum(values, codes, n_levels), "group sum")


def _column_sum(values: FloatArray, codes: IndexArray, count: int) -> FloatArray:
    """Keep the checks and copies, executing an owned single column as a vector."""
    a = _values(values)
    count = _integer(count, "count")
    c = _codes(codes, len(a), count)
    return _column_sum_rows(a, c, count)


def _column_sum_rows(a: FloatArray, c: IndexArray, count: int) -> FloatArray:
    """The unchanged column/vector refusal-replay kernel, after input lowering."""
    out = np.zeros((count, *a.shape[1:]))
    with np.errstate(over="raise", invalid="raise"):
        if a.ndim == 2 and a.shape[1] == 1:
            try:
                np.add.at(out[:, 0], c, a[:, 0])
            except FloatingPointError as error:
                # Replay only a refusal on owned arrays to retain the installed
                # matrix diagnostic. Never return a successful replayed answer.
                out.fill(0.0)
                np.add.at(out, c, a)
                raise error
        else:
            np.add.at(out, c, a)
    if not np.isfinite(out).all():
        raise FloatingPointError("group sums are not representable")
    return immutable_array(out)


# -------------------------------------------------------------- prepared weights


def _zero_code_vector(n_rows: int) -> IndexArray:
    """An owned read-only vector of ``n_rows`` zero integer codes."""
    zeros: IndexArray = np.zeros(n_rows, dtype=np.intp)
    zeros.flags.writeable = False
    return zeros


@dataclass(frozen=True, eq=False)
class PreparedWeights:
    """The row weights and the level totals of every block of a grouping.

    Attributes
    ----------
    grouping : Grouping
        The accepted grouping whose blocks the totals follow; its arrays are owned
        and read-only already.
    weights : array of float
        One finite nonnegative weight per row, read-only and owned.
    totals : tuple of arrays of float
        ``d_b``, one finite strictly positive total per level of each block,
        read-only and owned.

    Notes
    -----
    Constructed by :func:`prepare_weights`, which checks one detached copy of the
    weights and sums it by level once.  A direct construction is checked the same
    way, and the totals it is given must equal those checked sums: they are compared,
    never trusted, and the record keeps its own.  Changed weights are a new record;
    nothing weight-dependent survives outside it.
    """

    grouping: Grouping
    weights: FloatArray
    totals: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        """Check the weights and the given totals; store owned read-only copies."""
        if not isinstance(self.grouping, Grouping):
            raise ValueError(GROUPING)
        weights, totals = _checked_weights(self.weights, self.grouping)
        given = tuple(self.totals)
        if len(given) != len(totals) or any(
            not np.array_equal(np.asarray(found), expected, equal_nan=False)
            for found, expected in zip(given, totals, strict=True)
        ):
            raise ValueError("totals must be the checked sums of the weights by level")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "totals", totals)
        object.__setattr__(
            self, "_zero_codes", _zero_code_vector(self.grouping.n_rows)
        )

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return int(self.weights.size)

    @property
    def n_blocks(self) -> int:
        """The number of blocks."""
        return len(self.grouping.blocks)


def _weights(weights: ArrayLike, n_rows: int) -> FloatArray:
    """A finite nonnegative vector of ``n_rows`` doubles, or ``ValueError``."""
    if np.iscomplexobj(weights):
        raise ValueError(WEIGHTS)
    found: FloatArray = np.array(weights, dtype=np.float64, copy=True)
    if found.shape != (n_rows,) or not np.isfinite(found).all() or (found < 0).any():
        raise ValueError(WEIGHTS)
    return found


def _totals(weights: FloatArray, grouping: Grouping) -> tuple[FloatArray, ...]:
    """The checked level totals of every block, each finite and strictly positive."""
    totals = tuple(
        group_sums(weights, block.codes, block.n_levels) for block in grouping.blocks
    )
    if any(not np.isfinite(d).all() or (d <= 0).any() for d in totals):
        raise ValueError(TOTALS)
    return totals


def _checked_weights(
    weights: ArrayLike, grouping: Grouping
) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    """One detached copy of the weights, checked, its totals, then a read-only copy.

    The weights are converted once, from the caller's object, into a copy of their
    own; the checks and the level sums read that copy, which is then kept read-only
    in bytes of its own.  Nothing the caller holds is read again.
    """
    found = _weights(weights, grouping.n_rows)
    totals = _totals(found, grouping)
    return immutable_array(found), totals


def _from_checked_weights(
    grouping: Grouping, owned: FloatArray, totals: tuple[FloatArray, ...]
) -> PreparedWeights:
    """Assemble ``PreparedWeights`` from already-validated owned weights and totals."""
    if not isinstance(grouping, Grouping):
        raise ValueError(GROUPING)
    prepared = object.__new__(PreparedWeights)
    object.__setattr__(prepared, "grouping", grouping)
    object.__setattr__(prepared, "weights", owned)
    object.__setattr__(prepared, "totals", totals)
    object.__setattr__(prepared, "_zero_codes", _zero_code_vector(grouping.n_rows))
    return prepared


def prepare_weights(grouping: Grouping, weights: ArrayLike) -> PreparedWeights:
    """The weights and the level totals of ``grouping``, checked and owned.

    The one checked construction of the record: the weights checked on a detached
    copy and summed by level once, each refusal raised where it was raised before
    (the type of the grouping is checked where the record is formed).  The record is
    assembled from these values rather than by a direct construction, which would
    check and sum the same copy again only to compare it with itself.

    Parameters
    ----------
    grouping : Grouping
        The accepted grouping of the blocks (:func:`powermean.fe.groups.encode_blocks`).
    weights : array_like
        One finite nonnegative weight per row; ``ValueError`` otherwise, or when a
        level of a block has no positive total weight (every row of zero weight).

    Raises
    ------
    FloatingPointError
        A total that is not representable.
    """
    owned, totals = _checked_weights(weights, grouping)
    return _from_checked_weights(grouping, owned, totals)


# ---------------------------------------------------------- means, walks, lifts


def group_means(
    z: FloatArray, block: int, prepared: PreparedWeights, *, _inputs=None,
) -> FloatArray:
    """``M_b(z)``: the weighted means of every column of ``z`` by level of block ``b``.

    The one range-checked mean path of the task: the weighted products, their
    sequential row-order sums by level and the division by the level totals, each
    refusing an invalid, overflowing or inexactly underflowing result.

    Parameters
    ----------
    z : array of float
        One row per observation, any number of columns.
    block : int
        The index of the block in the grouping.
    prepared : PreparedWeights
        The weights and totals.

    Returns
    -------
    array of float
        ``L_b`` rows, one column per column of ``z``.
    """
    encoded = prepared.grouping.blocks[block]
    products = checked_product(prepared.weights[:, None], z, "weighted product")
    try:
        consume = None if _inputs is None else _inputs(products)
    except MemoryError:
        consume = None  # Optional lowering only, before the required sum.
    sums = (group_sums(products, encoded.codes, encoded.n_levels) if consume is None
            else consume(encoded.codes, encoded.n_levels, block))
    return checked_quotient(sums, prepared.totals[block][:, None], "weighted mean")


class Attempted:
    """A count of block operations begun: the attempted work of one walk or check.

    A block visit or a block-mean evaluation is attempted when its first operation
    begins, before that arithmetic can refuse, so an operation that refuses by range
    is counted.  The record is local to one operator application, sweep or fresh
    check and is never stored; the work record of the absorption reports its count
    beside the reserved budget units (exact visits and diagnostic passes).
    """

    __slots__ = ("count",)

    def __init__(self) -> None:
        """Start at zero."""
        self.count = 0

    def begin(self) -> None:
        """Count one block operation as attempted."""
        self.count += 1


def forward_order(n_blocks: int) -> tuple[int, ...]:
    """The forward sweep ``0, ..., G - 1``: every block once."""
    return tuple(range(n_blocks))


def symmetric_order(n_blocks: int) -> tuple[int, ...]:
    """The literal symmetric walk ``0, ..., G - 1, G - 2, ..., 0``: ``2G - 1`` visits.

    For one block it is the single visit ``(0,)``.
    """
    return (*range(n_blocks), *range(n_blocks - 2, -1, -1))


def zero_coefficients(grouping: Grouping, n_columns: int) -> Coefficients:
    """One ``L_b``-by-``k`` array of zeros per block: the cold coefficients."""
    return tuple(np.zeros((block.n_levels, n_columns)) for block in grouping.blocks)


def _all_positive_zeros(coefficients: Coefficients, grouping: Grouping) -> bool:
    """Whether ``coefficients`` is a plain tuple of positive-zero block matrices."""
    blocks = grouping.blocks
    if type(coefficients) is not tuple or len(coefficients) != len(blocks):
        return False
    columns = -1
    for block, a in zip(blocks, coefficients, strict=True):
        if (
            type(a) is not np.ndarray
            or a.dtype != np.float64
            or a.ndim != 2
            or a.shape[0] != block.n_levels
            or a.any()
            or np.signbit(a).any()
        ):
            return False
        if columns < 0:
            columns = a.shape[1]
        elif a.shape[1] != columns:
            return False
    return columns >= 0


def walk(
    z: FloatArray,
    prepared: PreparedWeights,
    order: Sequence[int],
    attempted: Attempted | None = None,
    *, _inputs=None,
) -> tuple[FloatArray, Coefficients, int]:
    """Apply ``Q_b`` along ``order``, accumulating the subtracted means by block.

    For each visited block: the means of the current rows, added to that block's
    coefficients; the rows decreased by their gathered means.  Returns the rows after
    the walk, the coefficients whose lift is the subtracted component in real
    arithmetic (the operator ``z - T z`` for the symmetric order), and the number of
    visits.  Every operation is range-checked; ``attempted``, when given, counts each
    visit as it begins, so that a refused visit is reported as attempted.
    """
    grouping = prepared.grouping
    rows = z
    delta = list(zero_coefficients(grouping, z.shape[1]))
    for block in order:
        if attempted is not None:
            attempted.begin()
        means = (group_means(rows, block, prepared, _inputs=_inputs)
                 if _inputs is not None and group_means is _INPUT_CALLS[0]
                 else group_means(rows, block, prepared))
        delta[block] = checked_sum(delta[block], means, "accumulated means")
        gathered = means[grouping.blocks[block].codes]
        rows = checked_difference(rows, gathered, "projected rows")
    return rows, tuple(delta), len(order)


def lift(coefficients: Coefficients, grouping: Grouping) -> FloatArray:
    """``sum_b a_b[c_b, :]`` by indexed gathers added in block order, range-checked."""
    if _all_positive_zeros(coefficients, grouping):
        return np.zeros((grouping.n_rows, coefficients[0].shape[1]), dtype=np.float64)
    blocks = grouping.blocks
    found = coefficients[0][blocks[0].codes]
    checked_finite = False
    for block, a in zip(blocks[1:], coefficients[1:], strict=True):
        if (
            type(a) is np.ndarray
            and a.dtype == np.float64
            and a.shape == (block.n_levels, found.shape[1])
            and not a.any()
            and not np.signbit(a).any()
        ):
            if not checked_finite:
                if not np.isfinite(found).all():
                    raise FloatingPointError("lifted component is not representable")
                checked_finite = True
            zero_mask = found == 0.0
            if not zero_mask.any() or not np.signbit(found[zero_mask]).any():
                continue
        found = checked_sum(found, a[block.codes], "lifted component")
        checked_finite = True
    return found


def candidate(
    values: FloatArray, coefficients: Coefficients, grouping: Grouping
) -> FloatArray:
    """The authoritative residual ``Y - lift(a)``, range-checked."""
    if (
        type(values) is np.ndarray
        and values.dtype == np.float64
        and values.ndim == 2
        and values.shape[0] == grouping.n_rows
        and _all_positive_zeros(coefficients, grouping)
        and values.shape[1] == coefficients[0].shape[1]
        and np.isfinite(values).all()
    ):
        return values.copy()
    return checked_difference(values, lift(coefficients, grouping), "candidate")


def weighted_inner(
    prepared: PreparedWeights, a: FloatArray, b: FloatArray
) -> FloatArray:
    """``<a, b>_w`` per column: the row-order sum of ``(a_ij b_ij) w_i``, checked.

    Both products are range-checked; the sum runs through the same reduction as the
    means, every row in one level.
    """
    products = checked_product(a, b, "inner product term")
    weighted = checked_product(products, prepared.weights[:, None], "weighted term")
    codes: IndexArray | None = getattr(prepared, "_zero_codes", None)
    if codes is None:
        codes = _zero_code_vector(prepared.n_rows)
    sums = group_sums(weighted, codes, 1)
    found: FloatArray = sums[0]
    return found


# ------------------------------------------------------------ coefficient tuples


def select_columns(coefficients: Coefficients, columns: IndexArray) -> Coefficients:
    """The columns ``columns`` of every block array (an exact gather)."""
    return tuple(a[:, columns] for a in coefficients)


def replaced_columns(
    coefficients: Coefficients, columns: IndexArray, updated: Coefficients
) -> Coefficients:
    """Fresh block arrays with the columns ``columns`` replaced by ``updated``."""
    found: list[FloatArray] = []
    for full, part in zip(coefficients, updated, strict=True):
        copy: FloatArray = np.array(full, dtype=np.float64, copy=True)
        copy[:, columns] = part
        found.append(copy)
    return tuple(found)


def scaled(coefficients: Coefficients, scalars: FloatArray) -> Coefficients:
    """Each block array with every column multiplied by its scalar, checked."""
    return tuple(
        checked_product(a, scalars, "scaled coefficients") for a in coefficients
    )


def added(a: Coefficients, b: Coefficients) -> Coefficients:
    """Block by block ``a_b + b_b``, checked."""
    return tuple(
        checked_sum(x, y, "combined coefficients") for x, y in zip(a, b, strict=True)
    )


def subtracted(a: Coefficients, b: Coefficients) -> Coefficients:
    """Block by block ``a_b - b_b``, checked."""
    return tuple(
        checked_difference(x, y, "combined coefficients")
        for x, y in zip(a, b, strict=True)
    )


# ----------------------------------------------------------------- fresh check


@dataclass(frozen=True, eq=False)
class FreshCheck:
    """The weighted group means of one candidate over every block, level and column.

    Attributes
    ----------
    column_errors : array of float
        Per column, the maximum absolute weighted mean over every block and level,
        read-only and owned.
    maximum : float
        The maximum over the columns.
    """

    column_errors: FloatArray
    maximum: float

    def __post_init__(self) -> None:
        """Store the errors owned; the maximum must be theirs."""
        errors = immutable_array(self.column_errors)
        if errors.ndim != 1 or not errors.size or not np.isfinite(errors).all():
            raise ValueError("column errors must be a finite vector")
        if self.maximum != float(errors.max()):
            raise ValueError("maximum must be the largest column error")
        object.__setattr__(self, "column_errors", errors)

    def passing(self, tolerance: float) -> BoolArray:
        """Per column, whether the error is at most ``tolerance``; equality passes."""
        found: BoolArray = self.column_errors <= tolerance
        return found


def fresh_check(
    z: FloatArray, prepared: PreparedWeights, attempted: Attempted | None = None,
    *, _inputs=None,
) -> FreshCheck:
    """All ``M_b(z)`` from the same ``z``, current weights and totals.

    Every block is evaluated through :func:`group_means`; a range failure in any of
    them raises ``FloatingPointError`` and no check exists for that candidate.
    ``attempted``, when given, counts each block evaluation as it begins, so that the
    evaluation that refuses is reported as attempted and the blocks never reached are
    not.
    """
    errors = np.zeros(z.shape[1])
    try:
        if _inputs is not None and not (type(z) is np.ndarray and z.dtype == np.float64
                and (attempted is None or type(attempted) is _INPUT_CLASSES[3])
                and _input_original()):
            _inputs = None
    except MemoryError:
        _inputs = None
    consume = None
    for block in range(prepared.n_blocks):
        if attempted is not None:
            attempted.begin()
        if _inputs is not None and (block == 0 or consume is not None):
            encoded = prepared.grouping.blocks[block]
            if block == 0:
                products = checked_product(
                    prepared.weights[:, None], z, "weighted product"
                )
                try:
                    consume = _inputs(products)
                except MemoryError:
                    consume = None  # Product already computed; never retry it.
            sums = (group_sums(products, encoded.codes, encoded.n_levels)
                    if consume is None else
                    consume(encoded.codes, encoded.n_levels, block))
            means = checked_quotient(
                sums, prepared.totals[block][:, None], "weighted mean"
            )
            if consume is None:
                # Optional lowering declined/failed: the first original product
                # has served its mean. Do not keep it beside later generic rows.
                del products
        else:
            means = group_means(z, block, prepared)
        errors = np.maximum(errors, np.max(np.abs(means), axis=0))
    return FreshCheck(errors, float(errors.max()))


def _input_sum(a, codes, count):
    """Keep the existing one-column/replay or matrix kernel and checked context."""
    kernel = (_column_sum_rows if a.ndim == 2 and a.shape[1] == 1
              else _reductions._sum_rows)
    return _checked(lambda: kernel(a, codes, count), "group sum")


_INPUT_CLASSES = PreparedWeights, Grouping, Block, Attempted
_INPUT_METHODS = tuple(
    (cls, name, vars(cls).get(name))
    for cls, names in (
        (PreparedWeights, ("__getattribute__", "__getattr__", "n_rows", "n_blocks",
                           "grouping", "weights", "totals")),
        (Grouping, ("__getattribute__", "__getattr__", "blocks")),
        (Block, ("__getattribute__", "__getattr__", "n_levels", "codes", "counts")),
        (Attempted, ("__getattribute__", "__getattr__", "begin")),
    ) for name in names
)
_INPUT_CALLS = (
    group_means, group_sums, _column_sum, _column_sum_rows, _checked,
    checked_product, checked_quotient, _values, _integer, _codes, immutable_array,
    _input_sum, np.multiply, group_sum, _reductions._prepared_reductions,
)


def _input_original():
    """Constant-size native producer check; no foreign class/getter is consulted."""
    try:
        if not (PreparedWeights is _INPUT_CLASSES[0]
                and Grouping is _INPUT_CLASSES[1] and Block is _INPUT_CLASSES[2]
                and Attempted is _INPUT_CLASSES[3]):
            return False
        return all(vars(cls).get(name) is original
                   for cls, name, original in _INPUT_METHODS) and all(a is b for a, b in zip((
            group_means, group_sums, _column_sum, _column_sum_rows, _checked,
            checked_product, checked_quotient, _values, _integer, _codes, immutable_array,
            _input_sum, np.multiply, group_sum, _reductions._prepared_reductions,
        ), _INPUT_CALLS, strict=True))
    except MemoryError:
        return False


def _input_reductions(prepared):
    """One absorption's code-only reducer store, no values or numerical verdicts.

    Only immutable native weight/group records qualify. No requirement callback
    is present on this route. Later fresh buffers belong to their immediate
    original checked_product invocation, not to caller-supplied array flags.
    """
    try:
        if type(prepared) is not _INPUT_CLASSES[0] or not _input_original():
            return None
        grouping, weights = prepared.grouping, prepared.weights
        if (type(grouping) is not Grouping or type(grouping.blocks) is not tuple
                or any(type(b) is not Block for b in grouping.blocks)
                or type(weights) is not np.ndarray or weights.dtype != np.float64
                or weights.dtype.metadata is not None or weights.ndim != 1
                or weights.flags.writeable or not weights.flags.c_contiguous):
            return None
        def backed(a, dtype):
            if (type(a) is not np.ndarray or a.dtype != dtype
                    or a.dtype.metadata is not None or a.ndim != 1
                    or a.flags.writeable or not a.flags.c_contiguous):
                return False
            owner = a.base
            while type(owner) is np.ndarray:
                if (owner.dtype != dtype or owner.dtype.metadata is not None
                        or owner.flags.writeable or not owner.flags.c_contiguous
                        or owner.nbytes != a.nbytes):
                    return False
                owner = owner.base
            return type(owner) is bytes and len(owner) == a.nbytes

        if (not backed(weights, np.float64) or type(prepared.totals) is not tuple
                or len(prepared.totals) != len(grouping.blocks)
                or any(not backed(b.codes, np.int64)
                       or not backed(b.counts, np.int64)
                       or not backed(d, np.float64)
                       for b, d in zip(grouping.blocks, prepared.totals, strict=True))):
            return None
        _, _, inputs = _reductions._prepared_reductions(
            sum_call=group_sums, sum_rows=_input_sum
        )

        def channel(values):
            try:
                return inputs(values) if _input_original() else None
            except MemoryError:
                return None

        return channel
    except MemoryError:
        # Optional lifetime setup only. No weighted product or reduction ran.
        return None
