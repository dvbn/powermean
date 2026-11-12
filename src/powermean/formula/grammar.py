"""The outer grammar of a formula, split into its parts before anything is evaluated.

    outcome ~ regressors | fixed effects | endogenous ~ instruments

A formula has one to three parts, separated by ``|``.  The
first part, ``outcome ~ regressors``, is required: its outcome is one data column,
written untransformed, and its regressors are Formulaic syntax, passed on as written
(transformations, categorical expansion, interactions, polynomials and splines;
:mod:`powermean.formula.frame`).  A later part without ``~`` holds the fixed effects:
terms joined by ``+``, each a name or ``a^b``, the interaction of two classifications.
A part with ``~`` holds the endogenous regressors and the excluded instruments; it is
the last part, and the second when there are no fixed effects (``y ~ x | d ~ z``).  The
fixed-effect and instrument parts are parsed as metadata only; nothing here evaluates
them.

Only top-level characters carry this grammar.  A character is at top level when it is
outside quotes (``'...'`` and ``"..."``), outside backticks (a column name) and outside
balanced ``()``, ``[]`` and ``{}``; within quotes and backticks a backslash keeps the
next character from closing them.  So ``I((x > 0) | (x < -1))``, a backticked column
``a|b`` and a string that holds ``~`` stay inside their part, as Formulaic reads them.
A name, for the outcome and the fixed effects, is a Python identifier other than a
keyword, or a column name within backticks, read as Formulaic's tokenizer reads it: the
name is the text between the backticks, and a backslash there stays in the name with
the character that it keeps, a backtick included.

A malformed formula raises :class:`~powermean.errors.SpecificationError` with the
existing identifier ``specification.invalid_value``: the argument ``formula``, the
formula as given and the valid forms of :data:`FORMS`.  This module imports the
standard library and the foundation only.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from itertools import pairwise

from powermean.errors import SpecificationError
from powermean.messages import format_keys

# The four forms of a formula, listed by the error of a malformed formula.
FORMS: tuple[str, ...] = (
    "outcome ~ regressors",
    "outcome ~ regressors | fixed effects",
    "outcome ~ regressors | endogenous ~ instruments",
    "outcome ~ regressors | fixed effects | endogenous ~ instruments",
)
FixedEffects = tuple[tuple[str, ...], ...]
_QUOTES = "'\"`"
_CLOSING = {"(": ")", "[": "]", "{": "}"}


@dataclass(frozen=True)
class Instruments:
    """The part ``endogenous ~ instruments``, each side as written (metadata only)."""

    endogenous: str
    excluded: str


@dataclass(frozen=True)
class FormulaParts:
    """A formula split into its parts; nothing is evaluated.

    Attributes
    ----------
    formula : str
        The formula as given.
    first_part : str
        ``outcome ~ regressors`` as written, which Formulaic evaluates.
    outcome : str
        The name of the outcome column, without its backticks; a backslash within them
        stays in the name, as Formulaic reads it.
    regressors : str
        The right-hand side of the first part as written, without outer spaces.
    fixed_effects : tuple of tuple of str
        One tuple of names per fixed-effect term, in order: ``("a",)`` for ``a`` and
        ``("a", "b")`` for ``a^b``; empty without fixed effects.
    instruments : Instruments or None
        The instrument part, or ``None`` without one.
    """

    formula: str
    first_part: str
    outcome: str
    regressors: str
    fixed_effects: FixedEffects
    instruments: Instruments | None


def malformed(formula: str) -> SpecificationError:
    """The error that refuses a malformed ``formula``, with the valid forms."""
    fields = {"argument": "formula", "value": formula, "valid": format_keys(FORMS)}
    return SpecificationError("specification.invalid_value", fields)


def parse_formula(formula: str) -> FormulaParts:
    """Split ``formula`` into its parts.

    Parameters
    ----------
    formula : str
        ``outcome ~ regressors``, optionally followed by ``| fixed effects`` and by
        ``| endogenous ~ instruments``.

    Returns
    -------
    FormulaParts
        The parts; the regressors and the instrument sides are kept as written.

    Raises
    ------
    SpecificationError
        ``specification.invalid_value`` for a malformed formula: an unbalanced quote,
        backtick or delimiter; more than three parts, or an empty one; a first part
        without exactly one top-level ``~`` or without regressors; an outcome that is
        not one name; a fixed-effect term that is neither a name nor ``a^b``; a later
        part that is neither fixed effects nor ``endogenous ~ instruments`` in its
        place.
    """
    first, *rest = _split(formula, "|", formula)
    if len(rest) > 2 or not all(part.strip() for part in (first, *rest)):
        raise malformed(formula)
    outcome, regressors = _first_part(first, formula)
    fixed_effects, instruments = _later_parts(rest, formula)
    return FormulaParts(
        formula=formula,
        first_part=first,
        outcome=outcome,
        regressors=regressors,
        fixed_effects=fixed_effects,
        instruments=instruments,
    )


def _first_part(part: str, formula: str) -> tuple[str, str]:
    """The outcome name and the regressors of ``outcome ~ regressors``."""
    sides = _split(part, "~", formula)
    if len(sides) != 2 or not sides[1].strip():
        raise malformed(formula)
    return _column_name(sides[0], formula), sides[1].strip()


def _later_parts(
    parts: list[str], formula: str
) -> tuple[FixedEffects, Instruments | None]:
    """The fixed effects and the instruments of the parts after the first."""
    instruments = _instruments(parts[-1], formula) if parts else None
    effects = parts[:-1] if instruments is not None else parts
    if len(effects) > 1:
        raise malformed(formula)
    fixed_effects = _fixed_effects(effects[0], formula) if effects else ()
    return fixed_effects, instruments


def _instruments(part: str, formula: str) -> Instruments | None:
    """The part ``endogenous ~ instruments``; ``None`` without a top-level ``~``."""
    sides = _split(part, "~", formula)
    if len(sides) == 1:
        return None
    if len(sides) != 2 or not all(side.strip() for side in sides):
        raise malformed(formula)
    return Instruments(endogenous=sides[0].strip(), excluded=sides[1].strip())


def _fixed_effects(part: str, formula: str) -> FixedEffects:
    """The terms of a fixed-effect part: names joined by ``+``, ``a^b`` for two."""
    terms: list[tuple[str, ...]] = []
    for term in _split(part, "+", formula):
        names = _split(term, "^", formula)
        if len(names) > 2:
            raise malformed(formula)
        terms.append(tuple(_column_name(name, formula) for name in names))
    return tuple(terms)


def _column_name(text: str, formula: str) -> str:
    """The column that ``text`` names: an identifier, or a name within backticks."""
    name = text.strip()
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    if len(name) > 2 and name[0] == "`" and name[-1] == "`":
        return _quoted(name[1:-1], formula)
    raise malformed(formula)


def _quoted(text: str, formula: str) -> str:
    """The name whose text between backticks is ``text``: that text, as written.

    Formulaic's tokenizer keeps a backslash and the character after it in the name; the
    backslash only stops that character, a backtick included, from closing the name.
    ``text`` comes from a balanced part (:func:`_top_level`), so it cannot end inside an
    escape; a backtick that no backslash keeps separates two names and is refused.
    """
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "`":
            raise malformed(formula)
    return text


def _split(text: str, separator: str, formula: str) -> list[str]:
    """The pieces of ``text`` between its top-level occurrences of ``separator``."""
    top = _top_level(text, formula)
    cuts = [at for at, char in enumerate(text) if char == separator and top[at]]
    bounds = [-1, *cuts, len(text)]
    return [text[start + 1 : end] for start, end in pairwise(bounds)]


def _top_level(text: str, formula: str) -> list[bool]:
    """Whether each character of ``text`` is at top level; refuse unbalanced text."""
    top: list[bool] = []
    awaited: list[str] = []  # the closing characters still expected, innermost last
    escaped = False
    for char in text:
        top.append(not awaited)
        if awaited and awaited[-1] in _QUOTES:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == awaited[-1]:
                awaited.pop()
        elif char in _QUOTES:
            awaited.append(char)
        elif char in _CLOSING:
            awaited.append(_CLOSING[char])
        elif char in ")]}" and (not awaited or awaited.pop() != char):
            raise malformed(formula)
    if awaited:
        raise malformed(formula)
    return top
