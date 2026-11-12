"""The model frame of a formula's first part, evaluated by Formulaic on every row.

A thin adapter to Formulaic's documented public API, not a formula engine.  The first
part, ``outcome ~ regressors``, goes as written to the explicit native pandas
materializer bridge (:mod:`powermean.formula.materialization`), which evaluates it
with an explicit
empty context, so that no variable of a caller or of a module is ever read, explicit
pandas output, and ``na_action="ignore"``, so that every row is kept; the fitted
``ModelSpec`` of the right-hand side is replayed and checked for ``na_action="drop"``
positions through the same materializer-instance ``get_model_matrix`` API with the same
empty context and ``materializer="pandas"`` recorded on the fitted spec.  The
outcome stays in the formula so that Formulaic reads the regressors as it reads any
two-sided formula.  Formulaic owns their semantics: transformations, categorical
encoding and contrasts, interactions, polynomials and splines, and the names and order
of terms and columns.  Its structural contrast coding is not the package's
collinearity resolution.  The outcome is the raw data column that Formulaic reads for it
(:mod:`powermean.formula.grammar`).  Formulaic is called through the typed views of
:mod:`powermean.formula.formulaic_api`.

Evaluation is not sample selection.  The data
are copied into an owned working frame indexed by position, so rows are never aligned
by label and duplicate labels are kept.  A record keeps every input row in order, with
the row labels, the raw outcome and three masks, kept apart for the sample
accounting, which alone applies ``subset``, missing values, the domain and the weights:

- ``ModelFrame.outcome_invalid``: the raw outcome is missing or infinite;
- ``Design.referenced_invalid``: a data column that the regressors read is missing or
  infinite, read raw, because a Boolean transform can hide a missing value and a
  missing category is encoded as all-zero contrasts, the row of the baseline;
- ``Design.evaluated_invalid``: an evaluated factor is null by Formulaic's own null
  check, or a value of the model matrix is not finite, because a transform can make
  a missing or infinite value from finite data.

The data columns that the regressors read, ``Design.variables``, follow how Formulaic
evaluates each factor of the fitted spec, by its ``Factor.eval_method``.  A factor that
Formulaic looks up reads the column of its name, whole: `` `x.mean` `` reads the column
``x.mean``.  A factor that Formulaic evaluates as Python reads the columns that the
expression names within backticks, which Formulaic binds to variables first, and the
columns of the names that Python loads from the mapping in which Formulaic evaluates
the expression (:mod:`powermean.formula.names`): the data, the empty context and the
transforms, as the locals of an evaluation with empty globals.  An attribute is not a
name, so a chain reads its first name, whose attributes Python takes from the value:
``x.fillna(0)`` and ``x - x.mean()`` read ``x``, never a column named ``x.fillna`` or
``x.mean``, and ``np.log(x)`` reads ``x``.  A name is a column where Python loads it
from the mapping, not where it is bound or global: ``x.map(lambda v: v * 2)`` and
``[v * 2 for v in x]`` read ``x`` alone, and ``v + x.map(lambda v: v * 2)`` also reads
``v``; ``x.map(lambda v: abs(v))`` calls the builtin ``abs``, as a name in a lambda's
body never reaches the mapping, while ``abs + x.map(lambda v: abs(v))`` reads the
column ``abs`` outside the lambda; ``(z := x) + z`` reads ``x`` alone, the stored
``z``, while ``(z > 0) + (z := x)`` reads the column ``z`` before the store.  The
expression is read, never evaluated again.  Formulaic's own list of variables names a
chain whole (``x.fillna``) and counts the names bound within the expression among
them, so it does not tell which columns Python reads, and it is not used.  With the
empty context, a name loaded from the mapping that is not a column is one of
Formulaic's transforms or a builtin.

Formulaic's null check is read from a second materialization of the same fitted spec
with ``na_action="drop"`` on the positional frame (Formulaic 1.0.0 has no ``drop_rows``
argument): only the positions that it keeps are used, as bookkeeping of the rows that
Formulaic would drop.  It learns nothing, its matrix is not the design, and no row is
removed or filled.

The fitted ``ModelSpec`` of the right-hand side is kept with the version of Formulaic
that fitted it.  :func:`replay_design` evaluates it on other rows with the categorical
levels and the stateful-transform states (centering, polynomial and spline bases) that
it learned, never learning them again; a category outside the learned levels, unseen
or missing, follows Formulaic's own behavior, its ``DataMismatchWarning`` included,
which reaches the caller.  Nothing promises that a spec survives a change of
Formulaic's version.

Numerical precaution: a transform such as ``log`` of zero or of a negative number
returns ``-inf`` or NaN with NumPy's floating-point warning of a division by zero or of
an invalid value.  These two conditions, and only they, are ignored while Formulaic
evaluates (``numpy.errstate``); the rows that they reach are in ``evaluated_invalid``.
Every other warning, an overflow included, reaches the caller.

A malformed formula raises the :class:`~powermean.errors.SpecificationError` of
:func:`powermean.formula.grammar.malformed`, also for Formulaic's
``FormulaSyntaxError`` while it parses the first part and for Formulaic's own
multi-part syntax inside the first part.  Every other error, such as Formulaic's
``FactorEvaluationError`` for a name that is not a column, propagates unchanged.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from powermean.formula import materialization, names, validity
from powermean.formula.formulaic_api import Spec, formulaic, formulaic_errors
from powermean.formula.grammar import FormulaParts, malformed, parse_formula

# The installed Formulaic, recorded with every materialization.
FORMULAIC_VERSION = importlib.metadata.version("formulaic")


@dataclass(frozen=True, eq=False)
class Design:
    """The right-hand side of a first part, evaluated on every input row by position.

    Attributes
    ----------
    matrix : pandas.DataFrame
        Formulaic's model matrix with pandas output, owned: one row per input row in
        input order, indexed by position, with Formulaic's columns in Formulaic's
        order.  No row is removed and no value is filled.
    spec : formulaic.ModelSpec
        The fitted ModelSpec of the right-hand side that the matrix comes from.
    terms : tuple of (str, tuple of str)
        Each term of the spec, by Formulaic's name, with its columns, in order.
    variables : tuple of str
        The data columns that the regressors read, sorted (module docstring).
    labels : pandas.Index
        The labels of the input rows, in order, duplicates kept.
    referenced_invalid, evaluated_invalid : numpy.ndarray of bool
        Read-only masks of the input rows (module docstring).
    formulaic_version : str
        The version of Formulaic that evaluated the matrix.
    _contrast_spec : formulaic.ModelSpec
        A separate owned snapshot of fitted encoding/transform metadata. Contrast
        clones its state per call; ordinary replay/prediction never receives it.
    """

    matrix: pd.DataFrame
    spec: Spec
    terms: tuple[tuple[str, tuple[str, ...]], ...]
    variables: tuple[str, ...]
    labels: pd.Index[Any]
    referenced_invalid: validity.BoolArray
    evaluated_invalid: validity.BoolArray
    formulaic_version: str
    _contrast_spec: Spec = field(repr=False)

    @property
    def columns(self) -> tuple[str, ...]:
        """The names of the columns of the matrix, in order."""
        return tuple(str(name) for name in self.matrix.columns)


@dataclass(frozen=True, eq=False)
class ModelFrame:
    """A formula evaluated on every row of its data, before any sample selection.

    Attributes
    ----------
    parts : FormulaParts
        The parsed formula, with its fixed-effect and instrument parts as metadata.
    outcome : pandas.Series
        The raw outcome column, owned and indexed by position, values and dtype
        unchanged.
    outcome_invalid : numpy.ndarray of bool
        Read-only mask of the rows whose raw outcome is missing or infinite.
    design : Design
        The right-hand side of the first part.
    """

    parts: FormulaParts
    outcome: pd.Series[Any]
    outcome_invalid: validity.BoolArray
    design: Design


def model_frame(formula: str, data: pd.DataFrame) -> ModelFrame:
    """Evaluate the first part of ``formula`` on every row of ``data``.

    Parameters
    ----------
    formula : str
        A formula of the grammar of :mod:`powermean.formula.grammar`.
    data : pandas.DataFrame
        The data, read by position and never modified.

    Returns
    -------
    ModelFrame
        Every input row, with its raw outcome, its row of the design and the masks of
        missing and non-finite values; no row is excluded.

    Raises
    ------
    SpecificationError
        ``specification.invalid_value`` for a malformed formula (module docstring).
    """
    parts = parse_formula(formula)
    working = _working_frame(data)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            matrices = materialization.materialize_formula(parts.first_part, working)
    except formulaic_errors.FormulaSyntaxError as error:
        raise malformed(formula) from error
    regressors = getattr(matrices, "rhs", None)
    spec = getattr(regressors, "model_spec", None)
    # Formulaic's own parts inside the first part give no single fitted spec.
    if not isinstance(spec, formulaic.ModelSpec):
        raise malformed(formula)
    outcome = working[parts.outcome].copy(deep=True)
    return ModelFrame(
        parts=parts,
        outcome=outcome,
        outcome_invalid=validity.frozen(validity.values_invalid(outcome)),
        design=_design(spec, regressors, working, data.index),
    )


def replay_design(design: Design, data: pd.DataFrame) -> Design:
    """Evaluate a fitted right-hand side on the rows of ``data``, learning nothing.

    Parameters
    ----------
    design : Design
        A design of :func:`model_frame` or of this function, whose spec is reused.
    data : pandas.DataFrame
        The rows to evaluate, read by position and never modified.

    Returns
    -------
    Design
        The design of these rows with the learned categorical levels, contrasts and
        stateful-transform states of ``design.spec``, never learned again from
        ``data``; its masks are those of these rows.
    """
    working = _working_frame(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = materialization.materialize_spec(design.spec, working)
    return _design(design.spec, matrix, working, data.index)


def _working_frame(data: pd.DataFrame) -> pd.DataFrame:
    """An owned copy of ``data`` indexed by position, so no row is matched by label."""
    working = data.copy(deep=True)
    working.index = pd.RangeIndex(len(working))
    return working


def replay_contrast_design(design: Design, data: pd.DataFrame) -> Design:
    """One contrast-only row replay, preserving fitted state and null diagnostics.

    Ordinary fit, prepare, prediction and replay_design retain their own paths.
    The private fitted metadata snapshot is cloned by the materializer, so neither
    successful nor failed contrast evaluation can write the result's fitted state.
    """
    working = _working_frame(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = materialization.materialize_contrast(design._contrast_spec, working)
    table = cast("pd.DataFrame", matrix).copy(deep=True)
    invalid = np.ones(len(working), dtype=bool)
    invalid[table.index.to_numpy(dtype=np.intp)] = validity.design_invalid(table)
    return replace(
        design,
        matrix=table,
        labels=data.index.copy(deep=True),
        referenced_invalid=validity.frozen(
            validity.columns_invalid(working, design.variables)
        ),
        evaluated_invalid=validity.frozen(invalid),
    )


def _design(
    spec: Spec, matrix: object, working: pd.DataFrame, labels: pd.Index[Any]
) -> Design:
    """The record of the model matrix that ``spec`` evaluated on ``working``."""
    table = cast("pd.DataFrame", matrix).copy(deep=True)
    columns = tuple(str(name) for name in table.columns)
    terms = tuple(
        (str(term), columns[place]) for term, place in spec.term_slices.items()
    )
    variables = _columns_read(spec, working.columns)
    referenced = validity.columns_invalid(working, variables)
    with np.errstate(divide="ignore", invalid="ignore"):
        kept = materialization.materialize_spec(spec, working, na_action="drop")
    dropped = validity.unretained(cast("pd.DataFrame", kept).index, len(table))
    return Design(
        matrix=table,
        spec=spec,
        terms=terms,
        variables=variables,
        labels=labels.copy(deep=True),
        referenced_invalid=validity.frozen(referenced),
        evaluated_invalid=validity.frozen(validity.design_invalid(table) | dropped),
        formulaic_version=FORMULAIC_VERSION,
        _contrast_spec=materialization.owned_spec(spec),
    )


def _columns_read(spec: Spec, columns: pd.Index[Any]) -> tuple[str, ...]:
    """The data columns that the factors of ``spec`` read, sorted (module docstring)."""
    read: set[str] = set()
    for term in spec.term_slices:
        for factor in term.factors:
            if factor.eval_method.value == "lookup":
                read.add(factor.expr)
            elif factor.eval_method.value == "python":
                read |= names.python_names(factor.expr)
    return tuple(sorted(name for name in read if name in columns))
