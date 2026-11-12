"""The fitted conditional power mean, from the owned fitted right-hand side.

``predict(type="power_mean")`` returns the fitted conditional power mean ``exp(X'beta +
effects)``, the conditional mean of ``Y`` only at ``omega = 1``, and the label says so.
The index of the log power mean is ``X'beta + offset``, the offset with unit
coefficient; an exposure ``e`` is the offset ``log e``.

    power mean    exp(X beta + o)     o the raw log-power-mean offset, log e for an
                                      exposure, formed once; never omega X beta nor
                                      omega o, which belong to the transformed problem

Without data the prediction is formed on the retained rows of the fit, from the original
unscaled trained design and the raw offset of the sample's numeric record.  With data,
the fitted right-hand side of the formula is replayed on its rows by
:func:`powermean.formula.frame.replay_design`, with the categorical levels, contrasts
and stateful-transform states (a centering, a polynomial, a spline basis) that it
learned on the training frame, with the explicit empty context, and never learned again;
the offset or the exposure is read from the column of the same name, as the preparation
read it.  The replayed columns are checked to be the fitted ones, in their order.  A row
is missing where the replayed design says so, by the two masks that the preparation
reads for the training rows: a data column that the regressors read is missing or
infinite on it (``referenced_invalid``: a Boolean transform such as ``I(x > 0)`` hides a
missing or infinite ``x`` behind a ``False``, and a missing category is coded as the
baseline), or an evaluated factor is null by Formulaic's own check or a value of the
matrix is not finite (``evaluated_invalid``).  Such a row has a missing prediction and
keeps its position; no row is dropped or reordered, and no coding is invented for it.
Formulaic's own coding of a category outside the learned levels is kept, its
``DataMismatchWarning`` included, which reaches the caller.  A row whose offset is
missing has a missing prediction as well.  An exposure that is not strictly positive is
refused, as the numeric record refuses it.  An offset or exposure given as an expression
is replayed the same way from its own fitted design on the original columns of the data
(:func:`expression_offset`): one column, the same masks, the logarithm of an exposure
formed once; no internal column of the caller is required.

A prediction uses the reported coefficients ``beta`` of the point.  Where the index is
finite and its exponential is not representable in float64, because it overflows or
because a positive value rounds to zero (an index below about ``-745.13``), the
``FloatingPointError`` of the other layers is raised: a representable subnormal power
mean is kept as it is, and nothing is clipped, floored or replaced by zero.  The
caller's data are read by position and never modified; the retained design is never
handed out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from powermean.formula.frame import Design, replay_design
from powermean.model.variance import FloatArray


def _doubles(column: pd.Series[Any], name: str) -> FloatArray:
    """A column as doubles, NaN where missing; a non-numeric column is refused."""
    try:
        values = column.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {name!r} must be numeric") from error
    return np.asarray(values, dtype=np.float64)


def _column(data: pd.DataFrame, name: str) -> pd.Series[Any]:
    """The one column ``name`` of ``data``."""
    if name not in data.columns:
        raise ValueError(f"the data have no column {name!r}")
    column = data[name]
    if not isinstance(column, pd.Series):
        raise ValueError(f"{name!r} names more than one column of the data")
    return column


def offset_of(
    data: pd.DataFrame, *, offset: str | None, exposure: str | None
) -> FloatArray:
    """The raw log-power-mean offset of the rows of ``data``: ``o``, or ``log e``.

    Zero without an offset or exposure column; NaN where the column is missing.
    """
    rows = len(data)
    if offset is not None:
        return _doubles(_column(data, offset), offset)
    if exposure is None:
        return np.zeros(rows)
    values = _doubles(_column(data, exposure), exposure)
    present = ~np.isnan(values)
    if np.any(values[present] <= 0):
        raise ValueError(f"exposure {exposure!r} must be strictly positive")
    logged = np.full(rows, np.nan)
    logged[present] = np.log(values[present])  # formed once; finite for a double > 0
    return logged


def expression_offset(
    design: Design, data: pd.DataFrame, *, exposure: bool
) -> FloatArray:
    """The raw offset of the rows of ``data`` from a replayed expression.

    Parameters
    ----------
    design : Design
        The fitted design of the expression, of exactly one column, as the
        preparation evaluated it on the training rows.
    data : pandas.DataFrame
        The rows on which the expression is replayed, read by position, from the
        original columns that the expression names; never modified.
    exposure : bool
        Whether the expression is an exposure, whose logarithm is the offset, formed
        once; otherwise the offset itself.

    Returns
    -------
    array
        ``o``, or ``log e``, one value per row; NaN where the replayed expression is
        missing (a column it reads is missing or infinite, or its value is not
        finite).

    Raises
    ------
    ValueError
        A replayed design with other columns than the one of the fit, or an
        exposure that is not strictly positive on a row where it is present.
    """
    evaluated = replay_design(design, data)
    if evaluated.columns != design.columns or len(design.columns) != 1:
        raise ValueError("the replayed expression has not the column of the fit")
    values = np.array(
        evaluated.matrix.to_numpy(dtype=np.float64, na_value=np.nan)[:, 0],
        dtype=np.float64,
    )
    missing = evaluated.referenced_invalid | evaluated.evaluated_invalid
    values[missing] = np.nan
    if not exposure:
        return values
    present = ~np.isnan(values)
    if np.any(values[present] <= 0):
        raise ValueError(f"exposure {design.columns[0]!r} must be strictly positive")
    logged = np.full(values.shape, np.nan)
    # The logarithm once, as for an exposure column; finite for a positive double.
    logged[present] = np.log(values[present])
    return logged


def replayed_matrix(design: Design, data: pd.DataFrame) -> FloatArray:
    """The fitted right-hand side replayed on ``data``, as doubles, NaN where missing.

    The columns must be the fitted ones, in their order; the spec is reused and
    learns nothing (:func:`powermean.formula.frame.replay_design`).  A row that the
    replayed design marks invalid, in a raw column that the regressors read or in an
    evaluated factor (module docstring), is NaN throughout, at its position.
    """
    replayed = replay_design(design, data)
    if replayed.columns != design.columns:
        raise ValueError("the replayed design has not the columns of the fit")
    matrix = np.array(
        replayed.matrix.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64
    )
    invalid = replayed.referenced_invalid | replayed.evaluated_invalid
    matrix[invalid] = np.nan
    return matrix


def power_mean(matrix: FloatArray, beta: FloatArray, offset: FloatArray) -> FloatArray:
    """``exp(X beta + o)``, NaN where a row is missing (module docstring).

    Parameters
    ----------
    matrix : array
        ``X``, one row per prediction and one column per coefficient; NaN allowed.
    beta : array
        The reported coefficients, finite.
    offset : array
        The raw offset ``o`` of each row; NaN allowed.

    Returns
    -------
    array
        The fitted power means.

    Raises
    ------
    ValueError
        Shapes that do not align.
    FloatingPointError
        A finite index whose exponential is not representable, because it overflows
        or because it rounds to zero (the exponential of a finite index is positive,
        so a zero is a genuine final underflow, not a value); or a finite row whose
        index is not.
    """
    if matrix.ndim != 2 or matrix.shape[1] != beta.shape[0]:
        raise ValueError("the design must have one column per coefficient")
    if offset.shape != (matrix.shape[0],):
        raise ValueError("the offset must have one value per row of the design")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        index: FloatArray = matrix @ beta + offset
        fitted: FloatArray = np.exp(index)
    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(offset)
    if not np.isfinite(index[complete]).all():
        raise FloatingPointError("a prediction index is not representable in float64")
    if not np.isfinite(fitted[complete]).all():
        raise FloatingPointError("a fitted power mean is not representable in float64")
    if np.any(fitted[complete] == 0):
        raise FloatingPointError(
            "a fitted power mean is not representable in float64: it rounds to zero"
        )
    fitted[~complete] = np.nan
    return fitted
