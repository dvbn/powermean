"""Group sums and group log-sum-exp by integer codes.

``group_sum`` and ``group_logsumexp``, with the three checks they call, ``_integer``,
``_codes`` and ``_values``, and the ``immutable_array`` that closes them.  The sums use
``np.add.at`` in the order of the rows, never a sort, a ``reduceat`` or a fused sum,
so that a group sum is the sequential sum of its rows in their original order, and
the log sums are formed around the maximum of each group.

- No native or automatic dispatch: the reductions are NumPy only.
- Storage.  :func:`immutable_array` keeps its copy in bytes of its own
  (:func:`powermean.model.rows._immutable`), as :mod:`powermean.inference.numeric`
  does, so that the write flag of the result cannot be set again.

The values are a nonempty real vector or matrix of finite doubles (log values may be
negative infinity, an exact zero term); the count is an integer at least one within
the range of ``int64``; the codes are an integer vector aligned with the rows, each in
``[0, count)``; an unobserved code has a zero sum and a log sum of negative infinity,
as has a group of zero terms only; a sum that is not representable is a
``FloatingPointError``, never an infinity; a finite negative difference in the log
sums that overflows to negative infinity has the exponential zero, which is not an
overflow of a required positive sum; a group whose range exceeds the largest double
returns its maximum.  Nothing is clipped and no code is dropped.  Refusals are
built-in exceptions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray

IntArray = np.ndarray[tuple[int], np.dtype[np.int64]]


def immutable_array(value: ArrayLike) -> FloatArray:
    """An owned read-only copy of a real array of doubles.

    The copy holds bytes of its own, whose write flag cannot be set again.

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
        and type(value.base) is bytes
        and len(value.base) == value.nbytes
    ):
        view: FloatArray = np.frombuffer(value.base, dtype=np.float64).reshape(
            value.shape
        )
        view.flags.writeable = False
        return view
    if np.iscomplexobj(value):
        raise ValueError("value must be real-valued")
    return _immutable(np.asarray(value, dtype=np.float64))


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    """An integer at least ``minimum`` in the range of ``int64``."""
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
    """Sum rows by group; unobserved groups have zero sums.

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


def _sum_rows(a: FloatArray, c: IntArray, count: int) -> FloatArray:
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

    The maximum of each group is taken out, the exponentials of the differences are
    summed in the order of the rows, and the log of a positive sum is added back.

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
    sums = np.zeros_like(maxima)
    np.add.at(sums, c, terms)
    out = maxima.copy()
    positive = sums > 0
    with np.errstate(over="raise", invalid="raise"):
        out[positive] += np.log(sums[positive])
    return immutable_array(out)


def _input_values(value, *, log=False):
    """Validate an immediate native producer buffer without making another copy.

    Only private arithmetic producers use this entry. Public/foreign input lowering
    still uses ``_values``. The buffer is not retained by an invocation or admitted
    on the strength of a writable/readonly flag: its original producer is checked
    by the consuming module, and this complete native header is checked here.
    """
    if not (type(value) is np.ndarray and value.dtype == np.float64
            and value.dtype.metadata is None and value.flags.owndata
            and value.flags.c_contiguous and value.ndim in (1, 2)):
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
    group_sum, group_logsumexp, _values, _integer, _codes,
    _sum_rows, _log_sum_rows, _input_values,
)


def _prepared_reductions(*, sum_call=None, sum_rows=None):
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
    admitted: dict[int, tuple[bytes, tuple[int, ...], tuple[int, ...], int, IntArray]] = {}

    def original() -> bool:
        try:
            return all(current is accepted for current, accepted in zip((
                group_sum, group_logsumexp, _values, _integer, _codes,
                _sum_rows, _log_sum_rows, _input_values,
            ), _PREPARED_ORIGINALS, strict=True))
        except MemoryError:
            admitted.clear()
            return False

    def binding(values, codes, count):
        if not (
            type(values) is np.ndarray and values.dtype == np.float64
            and values.dtype.metadata is None and values.ndim in (1, 2)
            and values.size and type(codes) is np.ndarray
            and codes.dtype == np.int64 and codes.dtype.metadata is None
            and codes.ndim == 1 and codes.shape == (len(values),)
            and not codes.flags.writeable and codes.flags.c_contiguous
            and type(count) is int
        ):
            return None
        owner = codes.base
        while type(owner) is np.ndarray:
            if (owner.dtype != np.int64 or owner.dtype.metadata is not None
                    or owner.flags.writeable or not owner.flags.c_contiguous
                    or owner.nbytes != codes.nbytes):
                return None
            owner = owner.base
        if type(owner) is not bytes or len(owner) != codes.nbytes:
            return None
        return owner, codes.shape, codes.strides, count

    def checked_binding(slot, values, codes, count):
        # Optional preparation must release its frame before a required fallback.
        try:
            key = binding(values, codes, count)
            held = admitted.get(slot)
            if (key is not None and held is not None and key[0] is held[0]
                    and key[1:] == held[1:4]):
                return key, held[4]
            admitted.pop(slot, None)
            return key, None
        except MemoryError:
            admitted.clear()
            return None, None

    def remember(slot, key, values, codes, count):
        try:
            after = binding(values, codes, count)
            if (after is not None and after[0] is key[0]
                    and after[1:] == key[1:] and original()):
                view = np.frombuffer(key[0], dtype=np.int64).reshape(key[1])
                admitted[slot] = (*key, view)
        except MemoryError:
            admitted.pop(slot, None)

    def reduce(values, codes, count, slot, *, logarithmic, prepared_values=None):
        public = (group_logsumexp if logarithmic else
                  group_sum if sum_call is None else sum_call)
        if not original():
            admitted.clear()
            return public(values, codes, count)
        key, prepared = checked_binding(slot, values, codes, count)
        if prepared is not None:
            a = (_values(values, log=logarithmic) if prepared_values is None
                 else prepared_values)
            return (_log_sum_rows(a, prepared, count) if logarithmic
                    else (_sum_rows if sum_rows is None else sum_rows)(
                        a, prepared, count))
        result = public(values, codes, count)
        if key is not None:
            remember(slot, key, values, codes, count)
        return result

    def sums(values, codes, count, slot):
        return reduce(values, codes, count, slot, logarithmic=False)

    def logs(values, codes, count, slot):
        return reduce(values, codes, count, slot, logarithmic=True)

    def inputs(values, *, logarithmic=False):
        # Optional lowering only: required public reductions/kernels below never
        # run inside this allocation fallback. Numerical ValueError/FPE propagate.
        try:
            if not original():
                return None
            a = _input_values(values, log=logarithmic)
            if a is None:
                return None

            def consume(codes, count, slot):
                return reduce(values, codes, count, slot, logarithmic=logarithmic,
                              prepared_values=a)

            return consume
        except MemoryError:
            admitted.clear()
            return None

    return sums, logs, inputs
