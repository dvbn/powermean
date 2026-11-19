"""Group sums and group log-sum-exp by integer codes, copied from the pyiols backends.

Bounded port of the NumPy branches of ``group_sum`` and ``group_logsumexp`` of
``backends.py`` of pyiols (source S6 of ``design/06_reuse_map.md``), with the three
checks they call, ``_integer``, ``_codes`` and ``_values``, and the ``immutable_array``
of ``_numeric.py`` that closes them (decision D66: a bounded numerical reduction copied
from S6 keeps its arithmetic and is verified bit for bit on the same machine;
``tools/verify_ports_groups.py``; ``PROVENANCE.md``, Section 20).  Names, checks,
messages, error states and arithmetic are those of the source. Vectors and matrix
columns use ``np.bincount``, which adds each group's values in their original row
order from positive zero. Nonfinite sums replay ``np.add.at``. Log sums still use
the group maximum and sum the shifted exponentials in row order.

Two adaptations, documented here and in ``PROVENANCE.md``, and checked by the tool.

- No native or automatic dispatch.  The source resolves ``backend="auto"`` to a compiled
  kernel when one is importable and to NumPy otherwise; this foundation is NumPy only,
  so the keyword and the resolution are removed and each function is the source's NumPy
  branch, which is what ``backend="python"`` selects in the source.  A compiled kernel,
  if one is measured to be needed, arrives behind an explicit adapter that records
  which implementation ran (``design/11``, Section 5, rules 10 and 16).
- Storage.  :func:`immutable_array` keeps its copy in bytes of its own
  (:func:`powermean.model.rows._immutable`), as :mod:`powermean.inference.numeric`
  already does for the same source function, so that the write flag of the result
  cannot be set again; the source clears the flag of an ordinary copy, which a caller
  can set back.  The values are the same doubles.

What the source does and this port keeps: the values are a nonempty real vector or
matrix of finite doubles (log values may be negative infinity, an exact zero term);
the count is an integer at least one within the range of ``int64``; the codes are an
integer vector aligned with the rows, each in ``[0, count)``; an unobserved code has a
zero sum and a log sum of negative infinity, as has a group of zero terms only; a sum
that is not representable is a ``FloatingPointError``, never an infinity; a finite
negative difference in the log sums that overflows to negative infinity has the
exponential zero, which is not an overflow of a required positive sum; a group whose
range exceeds the largest double returns its maximum.  Nothing is clipped and no
code is dropped.  Refusals are the built-in exceptions of the source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray

IntArray = np.ndarray[tuple[int], np.dtype[np.int64]]

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol, TypeVar, overload

    _Values_contra = TypeVar("_Values_contra", bound=ArrayLike, contravariant=True)
    _Codes_contra = TypeVar("_Codes_contra", bound=ArrayLike, contravariant=True)

    _CheckedCall = Callable[[_Values_contra, _Codes_contra, int], FloatArray]
    _NativeRows = Callable[[FloatArray, IntArray, int], FloatArray]
    _CheckedReduction = Callable[[_Values_contra, _Codes_contra, int, int], FloatArray]
    _Consumer = Callable[[_Codes_contra, int, int], FloatArray]
    _CodeBinding = tuple[bytes, tuple[int, ...], tuple[int, ...], int]

    class _InputLowering(Protocol[_Values_contra, _Codes_contra]):
        def __call__(
            self, values: _Values_contra, *, logarithmic: bool = False
        ) -> _Consumer[_Codes_contra] | None: ...

    _PreparedReductions = tuple[
        _CheckedReduction[_Values_contra, _Codes_contra],
        _CheckedReduction[_Values_contra, _Codes_contra],
        _InputLowering[_Values_contra, _Codes_contra],
    ]


def immutable_array(value: ArrayLike) -> FloatArray:
    """An owned read-only copy of a real array of doubles (S6 ``immutable_array``).

    The copy holds bytes of its own, whose write flag cannot be set again (storage
    adaptation; the values are those of the source's copy).

    Parameters
    ----------
    value : array_like
        The values, real.

    Returns
    -------
    array
        The copy, in the shape of the values.
    """
    if (
        type(value) is np.ndarray
        and value.dtype == np.float64
        and not value.flags.writeable
        and value.flags.c_contiguous
    ):
        # The NumPy stub omits the immutable buffer owner of frombuffer views.
        owner: object = value.base
        if type(owner) is bytes and len(owner) == value.nbytes:
            view: FloatArray = np.frombuffer(owner, dtype=np.float64).reshape(
                value.shape
            )
            view.flags.writeable = False
            return view
    if np.iscomplexobj(value):
        raise ValueError("value must be real-valued")
    return _immutable(np.asarray(value, dtype=np.float64))


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    """An integer at least ``minimum`` in the range of ``int64`` (S6 ``_integer``)."""
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or value < minimum
        or value > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} must be an integer >= {minimum} within int64 range")
    return int(value)


def _codes(value: ArrayLike, n: int, count: int) -> IntArray:
    """An ``int64`` copy of integer codes aligned with ``n`` rows, in ``[0, count)``."""
    a = np.asarray(value)
    if a.dtype.kind not in "iu" or a.shape != (n,):
        raise ValueError("codes must be an integer vector aligned with values")
    if np.any(a < 0) or np.any(a >= count):
        raise ValueError("group code is outside [0, count)")
    codes: IntArray = np.array(a, dtype=np.int64, order="C", copy=True)
    return codes


def _values(value: ArrayLike, *, log: bool = False) -> FloatArray:
    """A C-ordered copy of a nonempty real vector or matrix of finite doubles.

    With ``log``, negative infinity is also valid: it is an exact zero term.
    """
    if np.iscomplexobj(value):
        raise ValueError("values must be real-valued")
    a: FloatArray = np.array(value, dtype=np.float64, order="C", copy=True)
    if a.ndim not in (1, 2) or not a.size:
        raise ValueError("values must be a nonempty vector or matrix")
    valid = (a < np.inf) if log else np.isfinite(a)
    if not valid.all():
        raise ValueError(
            "values must be finite (log sums may include negative infinity)"
        )
    return a


def group_sum(values: ArrayLike, codes: ArrayLike, count: int) -> FloatArray:
    """Sum rows by group; unobserved groups have zero sums (S6 ``group_sum``).

    Parameters
    ----------
    values : array_like
        A nonempty real vector, or a matrix with one row per observation, finite.
    codes : array_like
        One integer code per row, in ``[0, count)``.
    count : int
        The number of groups, at least one.

    Returns
    -------
    array
        One row per group, read-only and owned: the sequential sum of the rows of the
        group in their order, zero for a group without rows.

    Raises
    ------
    ValueError
        Invalid values, codes or count.
    FloatingPointError
        A sum that is not representable.
    """
    a = _values(values)
    count = _integer(count, "count")
    c = _codes(codes, len(a), count)
    return _sum_rows(a, c, count)


_BINCOUNT = np.bincount


def _sum_rows(a: FloatArray, c: IntArray, count: int) -> FloatArray:
    """The row-order sums after input preparation; a matrix one column at a time.

    For each column, ``np.bincount`` adds the rows of every group in their order to a
    positive zero, the additions of ``np.add.at``, so the sums are the same doubles.
    A sum that is not finite is discarded and the unchanged ``np.add.at`` arithmetic
    runs on the same input, so that a refusal and its message are the source's.
    """
    if a.ndim == 2:
        found = _column_sums(a, c, count)
        if np.isfinite(found).all():
            return immutable_array(found)
    elif np.bincount is _BINCOUNT and c.dtype == np.intp:
        with np.errstate(over="ignore", invalid="ignore"):
            found = np.bincount(c, weights=a, minlength=count)
        if np.isfinite(found).all():
            return immutable_array(found)
    return _sum_rows_at(a, c, count)


def _column_sums(a: FloatArray, c: IntArray, count: int) -> FloatArray:
    """Per column, the row-order sums of every group by ``np.bincount``, unchecked."""
    out = np.empty((count, a.shape[1]))
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(a.shape[1]):
            out[:, j] = np.bincount(c, weights=a[:, j], minlength=count)
    return out


def _sum_rows_at(a: FloatArray, c: IntArray, count: int) -> FloatArray:
    """The unchanged row-order arithmetic after input preparation."""
    out = np.zeros((count, *a.shape[1:]))
    with np.errstate(over="raise", invalid="raise"):
        np.add.at(out, c, a)
    # The source's check after its compiled kernel, which returns an infinity where
    # a sum overflows; np.add.at raises under the error state above before any sum
    # of finite rows can be infinite, so this check is kept with the copy, unreached.
    if not np.isfinite(out).all():
        raise FloatingPointError("group sums are not representable")
    return immutable_array(out)


def group_logsumexp(values: ArrayLike, codes: ArrayLike, count: int) -> FloatArray:
    """Stable group log sums; empty/all-zero-log groups return negative infinity.

    S6 ``group_logsumexp``: the maximum of each group is taken out, the exponentials
    of the differences are summed in the order of the rows, and the log of a positive
    sum is added back.

    Parameters
    ----------
    values : array_like
        A nonempty real vector, or a matrix with one row per observation, of finite
        doubles or negative infinity.
    codes : array_like
        One integer code per row, in ``[0, count)``.
    count : int
        The number of groups, at least one.

    Returns
    -------
    array
        One row per group, read-only and owned: the log of the sum of the exponentials
        of the group, negative infinity for a group without rows or of negative
        infinities only.

    Raises
    ------
    ValueError
        Invalid values, codes or count.
    FloatingPointError
        An invalid operation in the sums.
    """
    a = _values(values, log=True)
    count = _integer(count, "count")
    c = _codes(codes, len(a), count)
    return _log_sum_rows(a, c, count)


def _log_sum_rows(a: FloatArray, c: IntArray, count: int) -> FloatArray:
    """The unchanged flat-row maximum, exponential, sum and logarithm."""
    maxima = np.full((count, *a.shape[1:]), -np.inf)
    np.maximum.at(maxima, c, a)
    # A finite negative difference may overflow to -inf; its exponential
    # is correctly zero. This is not overflow of a required positive sum.
    if np.isfinite(maxima).all():
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            terms = np.exp(a - maxima[c])
    else:
        terms = np.zeros_like(a)
        shifted = maxima[c]
        active = np.isfinite(shifted)
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            terms[active] = np.exp(a[active] - shifted[active])
    # Every shifted term is in [0, 1] and each sum is bounded by its row count.
    if np.bincount is _BINCOUNT and c.dtype == np.intp:
        sums = (
            np.bincount(c, weights=terms, minlength=count)
            if a.ndim == 1
            else _column_sums(terms, c, count)
        )
    else:
        sums = np.zeros_like(maxima)
        np.add.at(sums, c, terms)
    out = maxima.copy()
    positive = sums > 0
    with np.errstate(over="raise", invalid="raise"):
        out[positive] += np.log(sums[positive])
    return immutable_array(out)


def _input_values(value: object, *, log: bool = False) -> FloatArray | None:
    """Validate an immediate native producer buffer without making another copy.

    Only private arithmetic producers use this entry. Public/foreign input lowering
    still uses ``_values``. The buffer is not retained by an invocation or admitted
    on the strength of a writable/readonly flag: its original producer is checked
    by the consuming module, and this complete native header is checked here.
    """
    if not (
        type(value) is np.ndarray
        and value.dtype == np.float64
        and value.dtype.metadata is None
        and value.flags.owndata
        and value.flags.c_contiguous
        and value.ndim in (1, 2)
    ):
        return None
    if not value.size:
        raise ValueError("values must be a nonempty vector or matrix")
    valid = (value < np.inf) if log else np.isfinite(value)
    if not valid.all():
        raise ValueError(
            "values must be finite (log sums may include negative infinity)"
        )
    return value


_PREPARED_ORIGINALS = (
    group_sum,
    group_logsumexp,
    _values,
    _integer,
    _codes,
    _sum_rows,
    _log_sum_rows,
    _input_values,
)


if TYPE_CHECKING:

    @overload
    def _prepared_reductions(
        *, sum_call: None = None, sum_rows: _NativeRows | None = None
    ) -> _PreparedReductions[ArrayLike, ArrayLike]: ...

    @overload
    def _prepared_reductions[Values: ArrayLike, Codes: ArrayLike](
        *, sum_call: _CheckedCall[Values, Codes], sum_rows: _NativeRows | None = None
    ) -> _PreparedReductions[Values, Codes]: ...


def _prepared_reductions[Values: ArrayLike, Codes: ArrayLike](
    *,
    sum_call: _CheckedCall[Values, Codes] | None = None,
    sum_rows: _NativeRows | None = None,
) -> _PreparedReductions[Values, Codes]:
    """Local reducers and immediate input lowering; real successful calls admit codes.

    Slots are original block positions, so replacement releases the old binding.
    Entries retain only an immutable input byte owner and a detached code header,
    never values, numerical answers, eligibility flags or a serialized authority.
    A changed helper, mutable input or changed complete header keeps the original
    checked call. The first two entries preserve the checked generic lowering.
    The third prepares one genuine fresh internal channel for immediate ordered
    consumers; its closure alone holds that channel, never this code-only store.
    ``sum_call/sum_rows`` select absorption's existing column/replay arithmetic.
    """
    admitted: dict[
        int, tuple[bytes, tuple[int, ...], tuple[int, ...], int, IntArray]
    ] = {}

    def original() -> bool:
        try:
            return all(
                current is accepted
                for current, accepted in zip(
                    (
                        group_sum,
                        group_logsumexp,
                        _values,
                        _integer,
                        _codes,
                        _sum_rows,
                        _log_sum_rows,
                        _input_values,
                    ),
                    _PREPARED_ORIGINALS,
                    strict=True,
                )
            )
        except MemoryError:
            admitted.clear()
            return False

    def binding(values: object, codes: object, count: object) -> _CodeBinding | None:
        if not (
            type(values) is np.ndarray
            and values.dtype == np.float64
            and values.dtype.metadata is None
            and values.ndim in (1, 2)
            and values.size
            and type(codes) is np.ndarray
            and codes.dtype == np.int64
            and codes.dtype.metadata is None
            and codes.ndim == 1
            and codes.shape == (len(values),)
            and not codes.flags.writeable
            and codes.flags.c_contiguous
            and type(count) is int
        ):
            return None
        # The native base chain can terminate in an owner other than an array.
        owner: object = codes.base
        while type(owner) is np.ndarray:
            if (
                owner.dtype != np.int64
                or owner.dtype.metadata is not None
                or owner.flags.writeable
                or not owner.flags.c_contiguous
                or owner.nbytes != codes.nbytes
            ):
                return None
            owner = owner.base
        if type(owner) is not bytes or len(owner) != codes.nbytes:
            return None
        return owner, codes.shape, codes.strides, count

    def checked_binding(
        slot: int, values: ArrayLike, codes: ArrayLike, count: int
    ) -> tuple[_CodeBinding | None, IntArray | None]:
        # Optional preparation must release its frame before a required fallback.
        try:
            key = binding(values, codes, count)
            held = admitted.get(slot)
            if (
                key is not None
                and held is not None
                and key[0] is held[0]
                and key[1:] == held[1:4]
            ):
                return key, held[4]
            admitted.pop(slot, None)
            return key, None
        except MemoryError:
            admitted.clear()
            return None, None

    def remember(
        slot: int, key: _CodeBinding, values: ArrayLike, codes: ArrayLike, count: int
    ) -> None:
        try:
            after = binding(values, codes, count)
            if (
                after is not None
                and after[0] is key[0]
                and after[1:] == key[1:]
                and original()
            ):
                view: IntArray = np.frombuffer(key[0], dtype=np.int64).reshape(key[1])
                admitted[slot] = (*key, view)
        except MemoryError:
            admitted.pop(slot, None)

    def reduce(
        values: Values,
        codes: Codes,
        count: int,
        slot: int,
        *,
        logarithmic: bool,
        prepared_values: FloatArray | None = None,
    ) -> FloatArray:
        public: _CheckedCall[Values, Codes] = (
            group_logsumexp
            if logarithmic
            else group_sum
            if sum_call is None
            else sum_call
        )
        if not original():
            admitted.clear()
            return public(values, codes, count)
        key, prepared = checked_binding(slot, values, codes, count)
        if prepared is not None:
            a = (
                _values(values, log=logarithmic)
                if prepared_values is None
                else prepared_values
            )
            return (
                _log_sum_rows(a, prepared, count)
                if logarithmic
                else (_sum_rows if sum_rows is None else sum_rows)(a, prepared, count)
            )
        result = public(values, codes, count)
        if key is not None:
            remember(slot, key, values, codes, count)
        return result

    def sums(values: Values, codes: Codes, count: int, slot: int) -> FloatArray:
        return reduce(values, codes, count, slot, logarithmic=False)

    def logs(values: Values, codes: Codes, count: int, slot: int) -> FloatArray:
        return reduce(values, codes, count, slot, logarithmic=True)

    def inputs(values: Values, *, logarithmic: bool = False) -> _Consumer[Codes] | None:
        # Optional lowering only: required public reductions/kernels below never
        # run inside this allocation fallback. Numerical ValueError/FPE propagate.
        try:
            if not original():
                return None
            a = _input_values(values, log=logarithmic)
            if a is None:
                return None

            # Keep typing metadata out of each successfully lowered closure.
            def consume(codes, count, slot):
                # type: (Codes, int, int) -> FloatArray
                return reduce(
                    values,
                    codes,
                    count,
                    slot,
                    logarithmic=logarithmic,
                    prepared_values=a,
                )

            return consume
        except MemoryError:
            admitted.clear()
            return None

    return sums, logs, inputs
