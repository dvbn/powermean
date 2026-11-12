"""Offset and exposure expressions, and the cluster interaction ``a^b``.

``offset`` and ``exposure`` take a column or an expression.  A cluster term can be an
interaction ``a^b``, which clusters on the observed combinations of the two
variables.

An offset or an exposure names a column of the data, which the preparation
reads as it is (:mod:`powermean._prepare.compose`), or, where no column has that
name, a Formulaic expression of the columns of the data: ``log(dist)``,
``np.log(dist)``, ``I(x > 0)``.  :func:`evaluate_expression` evaluates it with the
Formulaic adapter, once, on every row of the data and with the explicit
empty context (no variable of a caller is read; no ``eval`` and no grammar of this
package), as the one-sided formula ``0 + expression``, so that no intercept is added,
and keeps the fitted ``ModelSpec`` of that evaluation as a
:class:`~powermean.formula.frame.Design` of exactly one column: its matrix on the
training rows and its masks of the rows whose value is missing (a column that the
expression reads is missing or infinite, or the evaluated value is not finite), which
the preparation reads as missing values of the group ``offset``.  An expression of
several columns (a categorical variable, ``a + b``), a formula of several parts, or
one that Formulaic cannot parse is refused.  The evaluated column is handed to the
preparation under Formulaic's name of the expression, which is also the name that the
result records; a prediction replays the fitted design on the caller's original
columns (:func:`powermean.results.prediction.expression_offset`), so that no caller
has to form an internal column.  The exposure is logged exactly once, by the numeric
record of the sample on the training rows and by the replay on prediction rows.

A cluster ``a^b`` is the observed combination of the two columns ``a`` and ``b``:
:func:`interaction_column` forms one object column of the pairs ``(a_i, b_i)`` of
the exact values that the data hold, missing where either is missing or infinite
(:mod:`powermean.formula.validity`), so that the preparation excludes such rows as
missing clusters and the result's recomputation numbers the pairs by first
appearance.  The column is retained under the requested name ``a^b``; ``a + b`` is a
multiway cluster, not an interaction.  The data are
read by position and never modified: the columns are added to the working copy of
the interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from powermean.errors import SpecificationError
from powermean.formula import materialization, validity
from powermean.formula.formulaic_api import formulaic, formulaic_errors
from powermean.formula.frame import Design, _design, _working_frame
from powermean.formula.grammar import _split

EXPRESSION = (
    "a column of the data, or a Formulaic expression of its columns that "
    "evaluates to one numeric column"
)


def _invalid(argument: str, expression: str) -> SpecificationError:
    """The refusal of an expression that is not one numeric column."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": argument, "value": expression, "valid": EXPRESSION},
    )


def evaluate_expression(expression: str, data: pd.DataFrame, argument: str) -> Design:
    """Evaluate ``expression`` on every row of ``data`` as one numeric column.

    Parameters
    ----------
    expression : str
        A Formulaic expression of the columns of ``data``.
    data : pandas.DataFrame
        The data, read by position and never modified.
    argument : str
        ``"offset"`` or ``"exposure"``, the argument named by a refusal.

    Returns
    -------
    Design
        The fitted design of the one-sided formula ``0 + expression``: one column,
        its values on every input row, its masks and its fitted spec.

    Raises
    ------
    SpecificationError
        ``specification.invalid_value``: Formulaic cannot parse the expression, it
        has several parts or a top-level ``~`` or ``|`` (the grammar's own split, so
        that a formula is never evaluated as an expression), or it evaluates to
        other than one column.
    """
    try:
        parts = [_split(expression, separator, expression)[1:] for separator in "~|"]
    except SpecificationError as error:  # unbalanced quotes or brackets
        raise _invalid(argument, expression) from error
    if any(parts):
        raise _invalid(argument, expression)  # a formula of parts, not an expression
    working = _working_frame(data)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = materialization.materialize_formula(
                f"0 + {expression}",
                working,
            )
    except formulaic_errors.FormulaSyntaxError as error:
        raise _invalid(argument, expression) from error
    spec = getattr(matrix, "model_spec", None)
    if not isinstance(spec, formulaic.ModelSpec):
        raise _invalid(argument, expression)
    design = _design(spec, matrix, working, data.index)
    if len(design.columns) != 1:
        raise _invalid(argument, expression)
    return design


def expression_values(design: Design) -> Any:
    """The evaluated column of an expression design as doubles, NaN where missing."""
    values = design.matrix.to_numpy(dtype=np.float64, na_value=np.nan)[:, 0]
    column = np.array(values, dtype=np.float64)
    column[design.referenced_invalid | design.evaluated_invalid] = np.nan
    return column


def interaction_names(name: str) -> tuple[str, str] | None:
    """The two column names of a cluster ``a^b``, or ``None`` for a plain name."""
    if "^" not in name:
        return None
    parts = [part.strip() for part in name.split("^")]
    if len(parts) != 2 or not all(parts):
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "vcov",
                "value": {"cluster": name},
                "valid": "one cluster variable, or the interaction a^b of two",
            },
        )
    return parts[0], parts[1]


def _batch_pairs(left: Any, right: Any) -> Any:
    """Form exact tuple objects from aligned plain arrays without Python indexing."""
    return np.fromiter(zip(left, right, strict=True), dtype=object, count=len(left))


def interaction_column(data: pd.DataFrame, first: str, second: str) -> Any:
    """The observed combinations ``(a_i, b_i)`` of two columns, as an object column.

    Parameters
    ----------
    data : pandas.DataFrame
        The data, read by position.
    first, second : str
        The two columns.

    Returns
    -------
    numpy.ndarray
        One tuple of the exact values per row, ``None`` where either value is
        missing or infinite.

    Raises
    ------
    KeyError
        A name that is not a column of the data.
    """
    columns = []
    for name in (first, second):
        if name not in data.columns:
            raise KeyError(f"{name!r} is not a column of the data")
        column = data[name]
        if not isinstance(column, pd.Series):
            raise ValueError(f"{name!r} names more than one column of the data")
        columns.append(column)
    invalid = validity.values_invalid(columns[0]) | validity.values_invalid(columns[1])
    left = columns[0].to_numpy(dtype=object)
    right = columns[1].to_numpy(dtype=object)
    # Plain arrays have callback-free iteration with the same scalar objects as
    # indexing. Keep the old rowwise path for missing rows or foreign containers;
    # it never constructs tuples for excluded entries or bypasses their hooks.
    if (
        type(data) is pd.DataFrame
        and all(type(column) is pd.Series for column in columns)
        and type(left) is np.ndarray
        and type(right) is np.ndarray
        and type(invalid) is np.ndarray
        and left.dtype == right.dtype == np.dtype(object)
        and invalid.dtype == np.dtype(np.bool_)
        and left.shape == right.shape == invalid.shape == (len(data),)
        and not invalid.any()
    ):
        return _batch_pairs(left, right)
    pairs = np.empty(len(data), dtype=object)
    for position in range(len(data)):
        pairs[position] = (
            None if invalid[position] else (left[position], right[position])
        )
    return pairs
