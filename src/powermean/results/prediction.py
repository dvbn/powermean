"""The fitted conditional power mean, from the owned fitted right-hand side.

``DESIGN.md``, Section 4.4 (``predict(type="power_mean")`` returns the fitted
conditional power mean ``exp(X'beta + effects)``, the conditional mean of ``Y`` only
at ``omega = 1``, and the label says so); ``design/01_statistical_core.md``, Section
3.1 (the index of the log power mean is ``X'beta + offset``, the offset with unit
coefficient; an exposure ``e`` is the offset ``log e``); decisions D41 and D56.

    power mean    exp(X beta + o)     o the raw log-power-mean offset, log e for an
                                      exposure, formed once; never omega X beta nor
                                      omega o, which belong to the transformed problem

Without data the prediction is formed on the retained rows of the fit, from the
original unscaled trained design and the raw offset of the sample's numeric record.
With data, the fitted right-hand side of the formula is replayed on its rows by
:func:`powermean.formula.frame.replay_prediction_design`, with fitted levels, contrasts
and stateful transforms (a centering, polynomial or spline basis) that it
learned on the training frame, with the explicit empty context, and never learned
again; the offset or the exposure is read from the column of the same name, as the
preparation read it.  The replayed columns are checked to be the fitted ones, in their
order.  A row is missing where the replayed design says so, by the two masks that the
preparation reads for the training rows (``design/02``, Section 2, step 2): a data
column that the regressors read is missing or infinite on it (``referenced_invalid``:
a Boolean transform such as ``I(x > 0)`` hides a missing or infinite ``x`` behind a
``False``, and a missing category is coded as the baseline), or an evaluated factor is
null by Formulaic's own check or a value of the matrix is not finite
(``evaluated_invalid``).  Such a row has a missing prediction and keeps its position.
A categorical value outside the fitted levels raises a package specification error.
A missing offset also gives a missing prediction. An exposure must be strictly
positive.  An offset or exposure given as an expression (task
M0.22) is replayed the same way from its own fitted design on the original columns
of the data (:func:`expression_offset`): one column, the same masks, the logarithm
of an exposure formed once; no internal column of the caller is required.

A prediction uses the reported coefficients ``beta`` of the point.  Where the index is
finite and its exponential is not representable in float64, because it overflows or
because a positive value rounds to zero (an index below about ``-745.13``), the
``FloatingPointError`` of the other layers is raised: a representable subnormal
power mean is kept as it is, and nothing is clipped, floored or replaced by zero
(decisions D49 and D54).  The caller's data are read by position and never modified;
the retained design is never handed out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from powermean import messages
from powermean.errors import SpecificationError
from powermean.formula.frame import Design, replay_prediction_design
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results import naming

PredictionKind = Literal["power_mean", "fixed_effects", "residual"]
PREDICTION_KINDS: tuple[PredictionKind, ...] = (
    "power_mean",
    "fixed_effects",
    "residual",
)
DEFAULT_PREDICTION: PredictionKind = "power_mean"
PREDICTION_KEYWORDS: tuple[str, ...] = ("type",)


class PredictionOptions(TypedDict, total=False):
    """Choose fitted power means, normalized FE levels, or response residuals."""

    type: PredictionKind


type Prediction = pd.Series[float] | dict[str, pd.Series[float]]


def prediction_kind(options: Mapping[str, object]) -> PredictionKind:
    """Validate the shared public predict options."""
    for key in options:
        if key not in PREDICTION_KEYWORDS:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": "keyword of predict",
                    "key": key,
                    "valid": messages.format_keys(PREDICTION_KEYWORDS),
                },
            )
    kind = options.get("type", DEFAULT_PREDICTION)
    for valid in PREDICTION_KINDS:
        if kind == valid:
            return valid
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "type",
            "value": kind,
            "valid": messages.format_keys(PREDICTION_KINDS),
        },
    )


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
    evaluated = replay_prediction_design(design, data)
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
    learns nothing. A row marked invalid in a raw regressor column or in an
    evaluated factor (module docstring), is NaN throughout, at its position.
    """
    replayed = replay_prediction_design(design, data)
    if replayed.columns != design.columns:
        raise ValueError("the replayed design has not the columns of the fit")
    matrix = np.array(
        replayed.matrix.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64
    )
    invalid = replayed.referenced_invalid | replayed.evaluated_invalid
    matrix[invalid] = np.nan
    return matrix


def prediction_data(data: pd.DataFrame | None, kind: PredictionKind) -> None:
    """Only fitted means can take a replacement covariate frame."""
    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data is not None and kind != "power_mean":
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "data",
                "value": "new covariates",
                "valid": "None for " + kind,
            },
        )


def retained_rows(original: IndexArray, retained: IndexArray) -> slice | IndexArray:
    """Locate the final rows in the original order sample after FE pruning."""
    if original.size == retained.size:
        return slice(None)
    rows = np.searchsorted(original, retained)
    if np.any(rows == original.size) or not np.array_equal(original[rows], retained):
        raise ValueError("the fitted rows are not a subset of the order sample")
    return rows


def slope_index(matrix: FloatArray, beta: FloatArray) -> FloatArray:
    """Evaluate the slope index, preserving incomplete rows and checking range."""
    complete = np.isfinite(matrix).all(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        index = matrix @ beta
    if not np.isfinite(index[complete]).all():
        raise FloatingPointError("a prediction index is not representable in float64")
    index[~complete] = np.nan
    return index


def check_fitted_groups(
    data: pd.DataFrame,
    labels: pd.Index[Any],
    blocks: Iterable[tuple[tuple[str, ...], Iterable[object], IndexArray]],
) -> None:
    """When FE identifiers are provided, verify their original row membership."""
    if not data.index.equals(labels):
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "data",
                "value": "a different row index",
                "valid": "the retained index in fitting order",
            },
        )
    if not data.columns.is_unique:
        raise ValueError("counterfactual data must have unique column names")
    for names, levels, codes in blocks:
        for position, name in enumerate(names):
            if name not in data:
                continue
            values = (
                levels if len(names) == 1 else (level[position] for level in levels)
            )
            expected = pd.Index(list(values), dtype=object, tupleize_cols=False).take(
                codes
            )
            supplied = pd.Index(_column(data, name), dtype=object, tupleize_cols=False)
            if not expected.equals(supplied):
                raise SpecificationError(
                    "specification.invalid_value",
                    {
                        "argument": name,
                        "value": "different fixed-effect identifiers",
                        "valid": "the fitted identifiers in retained row order",
                    },
                )


def counterfactual_index(
    matrix: FloatArray,
    columns: tuple[str, ...],
    beta: pd.Series[float],
    original: FloatArray,
    original_columns: tuple[str, ...],
    *,
    q: FloatArray,
    omega: float,
    zero: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Update the fitted index without losing resolved changes to cancellation."""
    names = tuple(beta.index)
    matrix = _active_columns(matrix, columns, names)
    original = _active_columns(original, original_columns, names)
    coefficients = beta.to_numpy()
    multiplier = 1.0 if omega == 0 else omega
    if zero is None:
        return _adjust_index(q, matrix, original, coefficients, multiplier)
    found = q + boundary_delta(matrix)
    found[~zero] = _adjust_index(
        q[~zero], matrix[~zero], original[~zero], coefficients, multiplier
    )
    return found


def _active_columns(
    matrix: FloatArray, columns: tuple[str, ...], names: tuple[str, ...]
) -> FloatArray:
    return (
        matrix
        if columns == names
        else matrix[:, [columns.index(name) for name in names]]
    )


def _adjust_index(
    q: FloatArray,
    changed: FloatArray,
    original: FloatArray,
    beta: FloatArray,
    multiplier: float,
) -> FloatArray:
    """Apply the slope changes with exact fallback when error can be amplified."""
    complete = np.isfinite(changed).all(axis=1)
    limits = np.finfo(np.float64)
    with np.errstate(all="ignore"):
        difference = changed - original
        index = q + multiplier * (difference @ beta)
        magnitude = np.abs(q) + abs(multiplier) * (np.abs(difference) @ np.abs(beta))
        # Use the ratio of the absolute contributions to the output scale to
        # detect cancellation independently of the number of columns.
        cancellation = magnitude / 256 > np.maximum(abs(multiplier), np.abs(index))
    # Near the float64 limit, even a finite intermediate may have rounded
    # toward zero when the exact final result would overflow.
    uncertain = complete & (
        ~np.isfinite(index) | ~(magnitude <= limits.max / 2) | cancellation
    )
    if abs(multiplier) < limits.tiny:
        # A scaled change can round away before the addition even when the
        # final index should round up by one subnormal step.
        uncertain |= complete & np.any(difference != 0, axis=1)
    if uncertain.any():
        weights = [Fraction(value) for value in beta]
        scale = Fraction(multiplier)
        try:
            for row in np.flatnonzero(uncertain):
                index[row] = float(
                    Fraction(q[row])
                    + scale
                    * sum(
                        (Fraction(a) - Fraction(b)) * weight
                        for a, b, weight in zip(
                            changed[row], original[row], weights, strict=True
                        )
                    )
                )
        except OverflowError as error:
            raise FloatingPointError(
                "a prediction index is not representable in float64"
            ) from error
    index[~complete] = np.nan
    return index


def boundary_delta(matrix: FloatArray) -> FloatArray:
    """A known zero stays at zero when its replacement regressors are supplied."""
    return np.where(np.isfinite(matrix).all(axis=1), 0.0, np.nan)


def exp_index(index: FloatArray, *, boundary: bool = False) -> FloatArray:
    """Exponentiate finite rows, allowing -inf only for a known zero boundary."""
    finite = np.isfinite(index)
    if np.isposinf(index).any() or (not boundary and np.isneginf(index).any()):
        raise FloatingPointError("a prediction index is not representable in float64")
    with np.errstate(over="ignore", under="ignore"):
        fitted = np.exp(index)
    if not np.isfinite(fitted[finite]).all():
        raise FloatingPointError("a fitted power mean is not representable in float64")
    if np.any(finite & (fitted == 0)):
        raise FloatingPointError(
            "a fitted power mean is not representable in float64: it rounds to zero"
        )
    return fitted


def fitted_series(
    q: FloatArray,
    labels: pd.Index[Any],
    omega: float,
    *,
    boundary: bool = False,
    counterfactual: bool = False,
) -> pd.Series[float]:
    """A fitted mean from the transformed or log-outcome index."""
    with np.errstate(over="ignore", invalid="ignore"):
        index = q if omega == 0 else q / omega
    if np.any(np.isfinite(q) & ~np.isfinite(index)):
        raise FloatingPointError("a prediction index is not representable in float64")
    series = pd.Series(
        exp_index(index, boundary=boundary),
        index=labels.copy(deep=True),
        name="power_mean",
    )
    series.attrs.update(label=naming.prediction_label(omega), omega=omega)
    if counterfactual:
        series.attrs["counterfactual"] = True
    return series


def residual_series(
    q: FloatArray,
    response: FloatArray,
    labels: pd.Index[Any],
    omega: float,
    *,
    boundary: bool = False,
) -> pd.Series[float]:
    """Observed minus fitted outcome on the estimating equation's scale."""
    fitted = q if omega == 0 else exp_index(q, boundary=boundary)
    with np.errstate(over="raise", invalid="raise"):
        residual = response - fitted
    series = pd.Series(residual, index=labels.copy(deep=True), name="residual")
    scale = (
        "log outcome" if omega == 0 else "outcome" if omega == 1 else "power outcome"
    )
    series.attrs.update(label=f"response residual on the {scale} scale", omega=omega)
    return series


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
