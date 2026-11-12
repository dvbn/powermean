"""The value rules and the serialization shared by the internal request records.

The records of :mod:`powermean.data.inference_requests` and
:mod:`powermean.data.inference_conventions` check their fields with the functions below
and serialize themselves through :func:`serialized` and :func:`settings`.  Nothing here
reads data, resolves a convention or computes.

Omitted and explicit settings.  A setting that a request omits holds :data:`OMITTED`,
not a value: an omitted lag or cutoff is left to its automatic rule, and an omitted
step, units or distance takes the default, which the ``resolved_*`` properties of the
request return; an explicit lag or cutoff of zero is a value.  ``None`` is never an
omission: it is refused wherever the domain excludes it, ``lag=None`` included.

Values.  A name is a nonempty string, stored as given.  An integer setting accepts a
number of a type registered as ``numbers.Integral`` and stores the exact ``int``; the
cutoff accepts a number of a type registered as ``numbers.Real`` and stores the nearest
``float``, ``-0.0`` as ``0.0``.  Booleans, NumPy durations, strings and floats for an
integer are refused, never converted: the rules of :mod:`powermean.solve.options`, which
this layer does not import.  A value outside its domain raises ``SpecificationError``
(``specification.invalid_value``) whose ``argument`` is the key in its public spelling;
a forbidden combination raises ``specification.conflicting``.

Serialization.  ``to_dict`` returns a new mapping of plain data, the type of the record
under ``"type"`` and only the settings given, sequences as lists; ``from_dict`` reads
it back, an absent setting as omitted, so that the round trip keeps presence.  What
``from_dict`` reads must be a mapping: anything else, a list of pairs included, is
refused (``specification.invalid_value``, argument ``record``) before any key is read,
as the solver records of :mod:`powermean.solve.absorbed` refuse it, so that no
repeated key of a sequence of pairs is silently reduced to its last value.  A mapping
of another type raises ``ValueError``; a key that names no field raises
``SpecificationError`` (``specification.unknown_option``).  A JSON text with a
repeated key is a matter of the decoder that reads it, which keeps one value before
any mapping reaches ``from_dict``; nothing here detects it.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import fields
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from powermean import messages
from powermean.errors import SpecificationError

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# The texts that fill the field ``valid`` of ``specification.invalid_value``; each
# reads after "the valid values are".
COLUMN_NAMES = "nonempty column names"
COORDINATES = "nonempty column names, latitude then longitude"
DIMENSIONS = "lists or tuples of one or more nonempty column names"
RETAINED = "lists or tuples of nonempty column names"
NONNEGATIVE_INTEGERS = "nonnegative integers"
POSITIVE_INTEGERS = "positive integers"
NONNEGATIVE_REALS = "nonnegative finite real numbers"
CONFIRMATION = "True"
RECORD_MAPPINGS = "mappings of field names to values"
# The second term of the conflict of units given without a cutoff.
AUTOMATIC_CUTOFF = "the automatic cutoff"

# Refused before the numeric test, as in powermean.solve.options: booleans are not
# numbers here, and NumPy registers its durations as integers.
_NOT_NUMBERS = (bool, np.bool_, np.timedelta64)


class Omitted(Enum):
    """The marker of a setting that a request omits: an absence, not a value."""

    OMITTED = "omitted"


OMITTED: Final = Omitted.OMITTED


# ------------------------------------------------------------------- the checks


def invalid(argument: str, value: object, valid: str) -> SpecificationError:
    """The refusal of a value outside the domain of its setting."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": argument, "value": value, "valid": valid},
    )


def choice(argument: str, value: object, choices: tuple[str, ...]) -> str:
    """The member of ``choices`` equal to ``value``, which must be a string."""
    if isinstance(value, str):
        for candidate in choices:
            if value == candidate:
                return candidate
    raise invalid(argument, value, messages.format_keys(choices))


def column_name(argument: str, value: object, valid: str = COLUMN_NAMES) -> str:
    """A nonempty name, as given; a name of blanks only is empty."""
    if isinstance(value, str) and value.strip():
        return value
    raise invalid(argument, value, valid)


def column_names(
    argument: str, value: object, valid: str, minimum: int
) -> tuple[str, ...]:
    """At least ``minimum`` nonempty names, in their order, repetitions kept."""
    if isinstance(value, list | tuple) and len(value) >= minimum:
        items: list[object] = list(value)
        if all(isinstance(item, str) and item.strip() for item in items):
            return tuple(str(item) for item in items)
    raise invalid(argument, value, valid)


def exact_integer(argument: str, value: object, lower: int, valid: str) -> int:
    """An integer of any registered type, at least ``lower``, as an exact ``int``."""
    if not isinstance(value, _NOT_NUMBERS) and isinstance(value, Integral):
        number = operator.index(value)
        if number >= lower:
            return number
    raise invalid(argument, value, valid)


def cutoff(value: object) -> float:
    """The nearest ``float`` of a nonnegative real number with a finite double.

    ``float`` raises ``OverflowError`` for an integer or a fraction beyond the largest
    double, which has no finite ``float``; ``-0.0`` is stored as ``0.0``.
    """
    if not isinstance(value, _NOT_NUMBERS) and isinstance(value, Real):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if math.isfinite(number) and number >= 0.0:
            return number + 0.0
    raise invalid("cutoff", value, NONNEGATIVE_REALS)


def given(value: object) -> bool:
    """Whether a setting was given: anything but :data:`OMITTED`, ``None`` included."""
    return not isinstance(value, Omitted)


# ------------------------------------------------------------ the serialization


def plain(value: object) -> object:
    """The plain data of a stored setting: a tuple of names as a new list."""
    return list(value) if isinstance(value, tuple) else value


def serialized(name: str, values: dict[str, object]) -> dict[str, object]:
    """The type of a record and the settings given, as a new mapping of plain data."""
    record: dict[str, object] = {"type": name}
    for key, value in values.items():
        if given(value):
            record[key] = plain(value)
    return record


def settings(record: object, name: str, valid: tuple[str, ...]) -> dict[str, Any]:
    """The settings of a serialized record of type ``name``, each key a field.

    Raises ``SpecificationError`` (``specification.invalid_value``, argument
    ``record``) for anything that is not a mapping, before any key is read; then
    ``ValueError`` for a mapping of another type; then ``SpecificationError``
    (``specification.unknown_option``) for the first key, in the order of the mapping,
    that names no field.
    """
    if not isinstance(record, Mapping):
        raise invalid("record", record, RECORD_MAPPINGS)
    data: dict[str, Any] = dict(record)
    if data.pop("type", None) != name:
        raise ValueError(f"not a {name} record")
    for key in data:
        if key not in valid:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": f"{name} field",
                    "key": key,
                    "valid": messages.format_keys(valid),
                },
            )
    return data


def field_names(record: type[DataclassInstance]) -> tuple[str, ...]:
    """The names of the fields of a record type, in their order."""
    return tuple(field.name for field in fields(record))
