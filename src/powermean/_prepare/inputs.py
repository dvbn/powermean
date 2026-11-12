"""Input column extraction and subset-mask helpers for frame preparation.

These helpers belong to steps 1 and 2 of the sample construction.  The module
provides :class:`Columns`, :func:`_column`, :func:`_values`, :func:`_doubles`,
:func:`_nan`, :func:`_object_subset`, :func:`_subset_mask`,
:func:`_check_same_data`, and :func:`_inputs`, extracted from
:mod:`powermean._prepare.compose`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from powermean.data.raw import AnyArray, RawInputs, RawWeights
from powermean.data.weights import WeightKind
from powermean.formula import validity
from powermean.formula.frame import ModelFrame


@dataclass(frozen=True)
class Columns:
    """The data columns that the sample construction reads beside the formula.

    Parameters
    ----------
    weights : str, optional
        The column of the weights, of the kind ``weight_kind``.
    weight_kind : {"analytic", "frequency", "probability"}, default "analytic"
        The kind of the weights, as the caller states it.
    offset : str, optional
        The column of the offset ``o``.  Not together with ``exposure``.
    exposure : str, optional
        The column of the exposure ``e``.  Not together with ``offset``.
    cluster : str, optional
        The cluster variable, a used variable of step 2, kept as metadata.
    metadata : tuple of str, default ()
        Other columns kept aligned for later stages (time, unit, location); their
        values exclude no row here.
    """

    weights: str | None = None
    weight_kind: WeightKind = "analytic"
    offset: str | None = None
    exposure: str | None = None
    cluster: str | None = None
    metadata: tuple[str, ...] = ()


def _column(data: pd.DataFrame, name: str) -> pd.Series[Any]:
    """The one column ``name`` of ``data``, by position."""
    if name not in data.columns:
        raise KeyError(f"{name!r} is not a column of the data")
    column = data[name]
    if not isinstance(column, pd.Series):
        raise ValueError(f"{name!r} names more than one column of the data")
    return column


def _values(column: pd.Series[Any]) -> AnyArray:
    """The values of a column as pandas holds them, exact, never narrowed.

    A NumPy-backed column gives its array, type kept; a column of an extension type
    gives the objects that pandas holds, with its own missing marker (module
    docstring).
    """
    if isinstance(column.dtype, np.dtype):
        return column.to_numpy()
    return column.to_numpy(dtype=object)


def _doubles(column: pd.Series[Any]) -> Any:
    """The values of a column as doubles, NaN where missing."""
    return column.to_numpy(dtype=np.float64, na_value=np.nan)


def _nan(value: object) -> bool:
    """Whether a value of an object column is a floating-point NaN."""
    return isinstance(value, float) and math.isnan(value)


def _object_subset(values: list[object]) -> tuple[Any, int]:
    """A subset of objects: Booleans, and missing markers taken as false."""
    mask = np.zeros(len(values), dtype=np.bool_)
    missing = 0
    for position, value in enumerate(values):
        if isinstance(value, bool | np.bool_):
            mask[position] = bool(value)
        elif value is None or value is pd.NA or _nan(value):
            missing += 1
        else:
            raise ValueError("subset must be Boolean")
    return mask, missing


def _subset_mask(
    subset: pd.Series[Any] | ArrayLike | None, data: pd.DataFrame
) -> tuple[Any, int]:
    """The subset as a positional Boolean mask, and how many conditions were missing."""
    if subset is None:
        return None, 0
    if not isinstance(subset, pd.Series):
        array = np.asarray(subset)
        if array.ndim != 1 or array.shape[0] != len(data):
            raise ValueError("subset must have one value per row of the data")
        if array.dtype.kind != "b":
            raise ValueError("subset must be Boolean")
        return array.astype(np.bool_), 0
    if not subset.index.equals(data.index):
        raise ValueError("subset must carry the labels of the data, in their order")
    if subset.dtype.kind == "b":
        missing = int(np.count_nonzero(np.asarray(subset.isna(), dtype=np.bool_)))
        return subset.to_numpy(dtype=np.bool_, na_value=False), missing
    if subset.dtype.kind == "O":
        return _object_subset(subset.tolist())
    raise ValueError("subset must be Boolean")


def _check_same_data(frame: ModelFrame, data: pd.DataFrame) -> None:
    """``data`` has the rows and the labels of the frame's data."""
    if len(data) != frame.outcome.size or not data.index.equals(frame.design.labels):
        raise ValueError("data must be the data of the frame: same rows, same labels")


def _inputs(
    frame: ModelFrame, data: pd.DataFrame, columns: Columns
) -> tuple[RawInputs, dict[str, Any]]:
    """The raw inputs and the missing masks by group, from the frame and the data."""
    design = frame.design
    missing: dict[str, Any] = {
        "outcome": frame.outcome_invalid,
        "regressors": design.referenced_invalid | design.evaluated_invalid,
    }
    weights = None
    if columns.weights is not None:
        column = _column(data, columns.weights)
        weights = RawWeights(_values(column), columns.weight_kind)
        missing["weights"] = validity.values_invalid(column)
    if columns.offset is not None and columns.exposure is not None:
        raise ValueError("specify offset or exposure, not both")
    offset: Any = None
    exposure: Any = None
    # An exposure is the offset in another form: both belong to the group "offset".
    if columns.offset is not None:
        column = _column(data, columns.offset)
        offset, missing["offset"] = _doubles(column), validity.values_invalid(column)
    if columns.exposure is not None:
        column = _column(data, columns.exposure)
        exposure, missing["offset"] = _doubles(column), validity.values_invalid(column)
    metadata: dict[str, Any] = {}
    if columns.cluster is not None:
        column = _column(data, columns.cluster)
        missing["clusters"] = validity.values_invalid(column)
        metadata[columns.cluster] = _values(column)
    for name in columns.metadata:
        metadata[name] = _values(_column(data, name))
    inputs = RawInputs(
        _doubles(frame.outcome),
        design.matrix.to_numpy(dtype=np.float64, na_value=np.nan),
        column_names=design.columns,
        weights=weights,
        offset=offset,
        exposure=exposure,
        metadata=metadata,
    )
    return inputs, missing
