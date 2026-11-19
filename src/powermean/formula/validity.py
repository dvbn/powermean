"""Missing and non-finite values of a model frame, marked row by row.

``design/02_computation.md``, Section 2, step 2: a row leaves the sample for a missing
or non-finite value of a variable that it uses, reported by variable group.  This
module marks rows and removes none; task M0.20 applies the steps of Section 2 in their
order.  A raw value is invalid when pandas reads it as missing (``NaN``, ``None``,
``NaT``, ``pd.NA``) or when it is a floating-point infinity, in a float or complex
column or as a float in an object column; integers, Booleans, strings, categories and
dates are never infinite.  An evaluated value is invalid when it is not finite.  A mask
is a NumPy Boolean array with one entry per row, in input order; :func:`frozen` makes
the owned, read-only copy that a record keeps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

BoolArray = npt.NDArray[np.bool_]


def values_invalid(values: pd.Series[Any]) -> BoolArray:
    """Rows where a raw column is missing or infinite.

    Parameters
    ----------
    values : pandas.Series
        A column of the data, of any dtype.

    Returns
    -------
    numpy.ndarray of bool
        True where the value is missing or a floating-point infinity.
    """
    missing = np.asarray(values.isna(), dtype=np.bool_)
    raw = values.to_numpy()
    if raw.dtype.kind in "fc":
        return missing | np.asarray(np.isinf(raw), dtype=np.bool_)
    if raw.dtype.kind == "O":
        infinite = [_infinite_float(value) for value in raw]
        return missing | np.asarray(infinite, dtype=np.bool_)
    return missing


def raw_invalid_count(values: npt.NDArray[Any]) -> int:
    """Count invalid retained labels when only the raw column is available.

    Keep the original array dtype: complex infinities invalidate a complex
    column, while in an object column only floating-point infinities do.
    Pandas recognizes missing scalar objects, including Decimal NaNs, and
    treats tuple-valued labels as individual objects.
    """
    kind = values.dtype.kind
    if kind in "fc":
        return int(np.count_nonzero(~np.isfinite(values)))
    if kind in "Mm":
        return int(np.count_nonzero(np.isnat(values)))
    if kind in "biuSU":
        return 0
    if kind == "O" and pd.api.types.infer_dtype(values, skipna=False) in (
        "string",
        "bytes",
        "boolean",
        "integer",
    ):
        # These label types cannot contain floating-point infinity.
        return int(np.count_nonzero(pd.isna(values)))
    series = pd.Series(values, dtype=values.dtype, copy=False)
    return int(np.count_nonzero(values_invalid(series)))


def columns_invalid(frame: pd.DataFrame, names: Iterable[str]) -> BoolArray:
    """Rows where any of the named columns of ``frame`` is missing or infinite.

    Parameters
    ----------
    frame : pandas.DataFrame
        The data, one row per input row.
    names : iterable of str
        The columns to read; none gives no invalid row.
    """
    invalid = np.zeros(len(frame), dtype=np.bool_)
    for name in names:
        invalid |= values_invalid(frame[name])
    return invalid


def design_invalid(matrix: pd.DataFrame) -> BoolArray:
    """Rows where a value of an evaluated matrix is not finite (NaN or infinite).

    Parameters
    ----------
    matrix : pandas.DataFrame
        A numeric model matrix; a missing value of a nullable column counts as NaN.
    """
    values = matrix.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.asarray(~np.isfinite(values).all(axis=1), dtype=np.bool_)


def unretained(positions: npt.ArrayLike, count: int) -> BoolArray:
    """Rows of ``range(count)`` that are not among the retained ``positions``.

    Parameters
    ----------
    positions : array_like of int
        The positions that a materialization kept.
    count : int
        The number of input rows.
    """
    absent = np.ones(count, dtype=np.bool_)
    absent[np.asarray(positions, dtype=np.intp)] = False
    return absent


def frozen(mask: BoolArray) -> BoolArray:
    """An owned copy of ``mask`` whose write flag cannot be set again.

    The copy lives in an immutable ``bytes`` object, so NumPy refuses both a write and
    a request to make it writable.
    """
    return np.frombuffer(mask.tobytes(), dtype=np.bool_)


def _infinite_float(value: object) -> bool:
    """Whether a value of an object column is a floating-point infinity."""
    if type(value) is np.longdouble:
        return bool(np.isinf(value))
    return isinstance(value, float | np.floating) and math.isinf(value)
