"""The raw labels of a formula's fixed-effect terms, read from its data by position.

A missing or non-finite value of a variable that a row uses excludes the row, reported
by variable group, effects included (step 2 of the sample construction).  A
fixed-effect term is a name or ``a^b``, the interaction of two classifications.
Internal module: :func:`powermean._prepare.compose.prepare_frame` calls
:func:`read_effects` only for a formula with a fixed-effect part.

:func:`read_effects` reads, by name, the column of every component of every parsed
term, both components of an interaction included and each component once, from the
data of the frame, by position.  Its values are read as the objects that pandas holds
(``Series.to_numpy(dtype=object)``): an integer beyond ``2**53``, a nullable integer
beside a missing one, a category, a date with a time zone stay exactly themselves, as
the grouping (:mod:`powermean.fe.groups`) compares labels as objects.  No
label is joined, coerced to a string or packed into a pair of integers.  A missing or
non-finite value of any component marks its row in the mask of the group ``effects``
(:func:`powermean.formula.validity.values_invalid`), before any grouping; a column
that no term names is not read, and the user's row labels are never data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from powermean._prepare.dependence import (
    ColumnProvenance, DependenceProvenance, _retained_original,
)
from powermean.data.sample_fe import EffectColumns, _fixed_original
from powermean.formula import validity
from powermean.formula.grammar import FixedEffects


def components(terms: FixedEffects) -> tuple[str, ...]:
    """The distinct component names of the terms, in order of first appearance."""
    return tuple(dict.fromkeys(name for term in terms for name in term))


def _component(data: pd.DataFrame, name: str) -> pd.Series[Any]:
    """The one column ``name`` of ``data``: ``KeyError`` if absent, else unique."""
    if name not in data.columns:
        raise KeyError(f"{name!r} is not a column of the data")
    column = data[name]
    if not isinstance(column, pd.Series):
        raise ValueError(f"{name!r} names more than one column of the data")
    return column


def read_effects(
    data: pd.DataFrame, terms: FixedEffects
) -> tuple[EffectColumns, validity.BoolArray]:
    """The raw labels of every component and the mask of the group ``effects``.

    Parameters
    ----------
    data : pandas.DataFrame
        The data of the frame, read by position.
    terms : tuple of tuple of str
        The parsed fixed-effect terms (:attr:`FormulaParts.fixed_effects`), nonempty.

    Returns
    -------
    EffectColumns
        The owned labels of every component, one per input row.
    numpy.ndarray of bool
        True where a component of the row is missing or non-finite.

    Raises
    ------
    KeyError
        A component that is not a column of the data.
    ValueError
        A name of several columns, or no term.
    """
    invalid = np.zeros(len(data), dtype=np.bool_)
    columns: dict[str, Any] = {}
    for name in components(terms):
        column = _component(data, name)
        invalid |= validity.values_invalid(column)
        columns[name] = column.to_numpy(dtype=object)
    return EffectColumns(terms, columns, len(data)), invalid


def _native_effects(
    data: pd.DataFrame, terms: FixedEffects, provenance: DependenceProvenance,
) -> tuple[EffectColumns, validity.BoolArray] | None:
    """Use the same already owned originals, not another whole object lowering."""
    if (not _native_effects_original()
            or type(provenance) is not DependenceProvenance):
        return None
    columns = {}
    for name in components(terms):
        prior = provenance.columns.get(name)
        if type(prior) is not ColumnProvenance:
            return None
        state = vars(prior)
        sources = state.get("_retained_sources")
        column = (state.get("_column_record") if sources is None
                  else sources[0] if type(sources) is tuple and len(sources) == 1
                  else None)
        if column is None:
            return None
        columns[name] = column
    try:
        effects = EffectColumns._from_native(terms, columns, len(data))
    except MemoryError:
        return None
    if effects is None:
        return None
    invalid = np.zeros(len(data), dtype=np.bool_)
    for name in components(terms):
        invalid |= validity.values_invalid(_component(data, name))
    return effects, invalid


_NATIVE_GLOBALS = tuple((name, globals()[name]) for name in (
    "EffectColumns", "ColumnProvenance", "DependenceProvenance",
    "components", "_component", "_retained_original", "_fixed_original",
))
_NATIVE_INVALID = validity.values_invalid


def _native_effects_original() -> bool:
    """Existing constant-size producer guards, never a row eligibility cache."""
    return (all(globals().get(name) is original for name, original in _NATIVE_GLOBALS)
            and validity.values_invalid is _NATIVE_INVALID
            and _retained_original() and _fixed_original())
