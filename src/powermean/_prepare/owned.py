"""Owned pandas Index, Series, DataFrame, and ModelFrame copy helpers.

These helpers belong to steps 1 and 2 of the sample construction.  The module
provides :func:`_holds_objects`, :func:`_owned_names`, :func:`_owned_multiindex`,
:func:`_owned_index`, :func:`_owned_series`, :func:`_owned_table`, and
:func:`_owned_frame`, extracted from :mod:`powermean._prepare.compose`.
"""

from __future__ import annotations

import copy
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from powermean.data.raw import _owned_objects
from powermean.formula.frame import Design, ModelFrame


def _holds_objects(index: pd.Index[Any]) -> bool:
    """Whether ``index`` is of object dtype, so that its labels may be mutable."""
    return isinstance(index.dtype, np.dtype) and index.dtype.kind == "O"


def _owned_names(index: pd.Index[Any]) -> list[Hashable]:
    """The names of ``index``, owned: one for an Index, one per level of a MultiIndex.

    Each name is any hashable object, mutable in place where it is an instance of a
    class of the caller's; ``names`` hands them out as they are.  The copy is
    ``copy.deepcopy`` of the list of names, with one memo for the list: the rule of
    :func:`powermean.data.raw._owned_objects` (a name of an immutable built-in type,
    a string, an integer or ``None`` say, is itself, since ``deepcopy`` returns such
    an object as it is; every other name its deep copy, of the same type and value;
    a name that cannot be copied raises the error of ``deepcopy``), written on the
    list so that the names keep their declared type.
    """
    names: list[Hashable] = list(index.names)
    return copy.deepcopy(names)


def _owned_multiindex(index: pd.MultiIndex) -> pd.MultiIndex:
    """A MultiIndex formed anew on owned parts (module docstring).

    Each level is owned by :func:`_owned_index`, of its kind, dtype, labels, order
    and categories; the codes are copied, the ``-1`` of a missing label included;
    the names are owned by :func:`_owned_names`; nothing is cached.  pandas keeps an
    Index given as a level as the Index it is (the "sequence of arrays" of its
    documentation; a list of labels would be inferred into a level of another kind),
    which its stubs, declaring the levels as sequences of labels, do not say: the
    owned levels are passed as the sequence of Indexes they are, declared without
    narrowing.
    """
    levels: Sequence[Any] = [_owned_index(level) for level in index.levels]
    codes = [level_codes.tolist() for level_codes in index.codes]
    return pd.MultiIndex(levels=levels, codes=codes, names=_owned_names(index))


def _owned_index(index: pd.Index[Any]) -> pd.Index[Any]:
    """A copy of ``index`` that shares no mutable label, name, buffer or cache with it.

    Parameters
    ----------
    index : pandas.Index
        The user's row labels, of any kind; an axis of the outcome or of the matrix.

    Returns
    -------
    pandas.Index
        The same type, names, dtype, labels, duplicates, order, levels and
        categories, whose object labels and names are copies by the rule of
        :func:`powermean.data.raw._owned_objects` (module docstring): a RangeIndex
        from its range, an object Index label by label, a MultiIndex by
        :func:`_owned_multiindex`, a CategoricalIndex by its owned categories, codes
        and order; any other Index by ``copy(deep=True)``; each with its owned name
        (:func:`_owned_names`).
    """
    if isinstance(index, pd.MultiIndex):
        return _owned_multiindex(index)
    (name,) = _owned_names(index)
    if isinstance(index, pd.RangeIndex):
        # A copy of a RangeIndex, deep or not, shares the cache that holds its array.
        return pd.RangeIndex(index.start, index.stop, index.step, name=name)
    if isinstance(index, pd.CategoricalIndex):
        # The Categorical that the Index wraps: its codes and order, on its owned
        # categories (a copy of a Categorical keeps the categories Index, buffer
        # included).
        categorical = pd.Categorical(index)
        values = pd.Categorical.from_codes(
            categorical.codes.tolist(),
            categories=_owned_index(categorical.categories),
            ordered=categorical.ordered,
        )
        return pd.CategoricalIndex(values, name=name)
    if _holds_objects(index):
        elements = _owned_objects(index.tolist())
        # An object array, element by element, so that pandas infers nothing from the
        # elements: a label that is a tuple stays a label and makes no MultiIndex.
        array = np.empty(len(elements), dtype=object)
        for position, element in enumerate(elements):
            array[position] = element
        return pd.Index(array, dtype=object, name=name)
    # The copy of the array; the name set to the owned one, whatever the copy did
    # with the original's (not documented).
    copied = index.copy(deep=True)
    copied.name = name
    return copied


def _owned_series(series: pd.Series[Any]) -> pd.Series[Any]:
    """A deep copy of ``series`` whose row axis is owned (:func:`_owned_index`)."""
    copied = series.copy(deep=True)
    copied.index = _owned_index(series.index)
    return copied


def _owned_table(table: pd.DataFrame) -> pd.DataFrame:
    """A deep copy of ``table`` whose two axes are owned (:func:`_owned_index`)."""
    copied = table.copy(deep=True)
    copied.index = _owned_index(table.index)
    copied.columns = _owned_index(table.columns)
    return copied


def _owned_frame(frame: ModelFrame) -> ModelFrame:
    """A copy of ``frame`` that shares nothing mutable with it.

    The outcome and the design matrix are deep copies with owned axes
    (:func:`_owned_series`, :func:`_owned_table`); the labels are owned copies
    (:func:`_owned_index`); the fitted spec is its deep copy, with the states that it
    learned; the parsed parts, the terms, the variables, the read-only masks and the
    version are immutable and shared.
    """
    design = frame.design
    return ModelFrame(
        parts=frame.parts,
        outcome=_owned_series(frame.outcome),
        outcome_invalid=frame.outcome_invalid,
        design=Design(
            matrix=_owned_table(design.matrix),
            spec=copy.deepcopy(design.spec),
            terms=design.terms,
            variables=design.variables,
            labels=_owned_index(design.labels),
            referenced_invalid=design.referenced_invalid,
            evaluated_invalid=design.evaluated_invalid,
            formulaic_version=design.formulaic_version,
            _contrast_spec=copy.deepcopy(design._contrast_spec),
        ),
    )
