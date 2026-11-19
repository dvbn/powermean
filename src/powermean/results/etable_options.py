"""The keywords of ``etable``: one typed record, validated at the entry.

Each keyword is a field of :class:`EtableOptions` with its default; :func:`collect`
refuses an unknown key with ``specification.unknown_option``, which lists the valid
ones, and a value outside its domain with ``specification.invalid_value`` (a closed
list of values) or ``specification.table_argument`` (any other domain).

==================  =================================================================
``names``           the column headings; default ``(1)``, ``(2)``, ...
``scale``           ``"coefficient"`` (default) or ``"effect"``, ``100 (exp(b) - 1)``
``uncertainty``     ``"se"`` (default), ``"ci"`` or ``"p"``: the second row of a term
``level``           confidence level of the intervals; default 0.95
``vcov``            a covariance request passed to ``with_vcov`` of every result,
                    which validates it; default none (each result's own covariance)
``stars``           ``True`` (``0.10``, ``0.05``, ``0.01``), ``False``, or up to
                    three p-value thresholds in decreasing order
``keep``            exact term names to print, in this order
``drop``            exact term names to leave out
``labels``          printed labels by term or fixed-effect name
``digits``          decimals of the printed numbers; default 3
``stats``           statistics of the last block, among ``"nobs"`` and ``"zeros"``;
                    default observations
``title``           the title of the table; default none
``notes``           notes added after those that the table writes
==================  =================================================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from itertools import pairwise
from typing import Any, Literal, TypedDict, cast

from powermean import messages
from powermean.errors import SpecificationError

Scale = Literal["coefficient", "effect"]
Uncertainty = Literal["se", "ci", "p"]

SCALES: tuple[str, ...] = ("coefficient", "effect")
UNCERTAINTIES: tuple[str, ...] = ("se", "ci", "p")
STATISTICS: tuple[str, ...] = ("nobs", "zeros")
DEFAULT_STARS = (0.10, 0.05, 0.01)
MAX_DIGITS = 15


class EtableKeywords(TypedDict, total=False):
    """The keywords of ``etable``, for the type checker."""

    names: Sequence[str] | None
    scale: str
    uncertainty: str
    level: float
    vcov: object
    stars: bool | Sequence[float]
    keep: Sequence[str] | None
    drop: Sequence[str]
    labels: Mapping[str, str] | None
    digits: int
    stats: Sequence[str]
    title: str
    notes: Sequence[str]


@dataclass(frozen=True)
class EtableOptions:
    """The resolved keywords of ``etable`` (module docstring).

    ``stars`` holds the thresholds in decreasing order, empty for no stars.
    """

    names: tuple[str, ...] | None = None
    scale: Scale = "coefficient"
    uncertainty: Uncertainty = "se"
    level: float = 0.95
    vcov: object = None
    stars: tuple[float, ...] = DEFAULT_STARS
    keep: tuple[str, ...] | None = None
    drop: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    digits: int = 3
    stats: tuple[str, ...] = ("nobs",)
    title: str = ""
    notes: tuple[str, ...] = ()


def refuse(argument: str, value: object, expected: str) -> SpecificationError:
    """The refusal of a value outside the domain of its argument."""
    return SpecificationError(
        "specification.table_argument",
        {"argument": argument, "value": value, "expected": expected},
    )


def _strings(name: str, value: object) -> tuple[str, ...]:
    expected = "a sequence of strings"
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise refuse(name, value, expected)
    if not all(isinstance(item, str) for item in value):
        raise refuse(name, value, expected)
    return tuple(cast("Sequence[str]", value))


def _optional_strings(name: str, value: object) -> tuple[str, ...] | None:
    return None if value is None else _strings(name, value)


def _choice(name: str, value: object, valid: tuple[str, ...]) -> str:
    if isinstance(value, str) and value in valid:
        return value
    raise SpecificationError(
        "specification.invalid_value",
        {"argument": name, "value": value, "valid": messages.format_keys(valid)},
    )


def _statistics(name: str, value: object) -> tuple[str, ...]:
    return tuple(_choice(name, item, STATISTICS) for item in _strings(name, value))


def _stars(name: str, value: object) -> tuple[float, ...]:
    if value is True:
        return DEFAULT_STARS
    if value is False:
        return ()
    expected = "True, False, or up to three p-value thresholds in decreasing order"
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise refuse(name, value, expected)
    levels = [float(p) for p in value if isinstance(p, float) and 0.0 < p < 1.0]
    decreasing = all(a > b for a, b in pairwise(levels))
    if len(levels) != len(value) or not decreasing or len(levels) > len(DEFAULT_STARS):
        raise refuse(name, value, expected)
    return tuple(levels)


def _level(name: str, value: object) -> float:
    if isinstance(value, float) and 0.0 < value < 1.0:
        return value
    raise refuse(name, value, "a float strictly between zero and one")


def _digits(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise refuse(name, value, f"an integer between 0 and {MAX_DIGITS}")
    if not 0 <= value <= MAX_DIGITS:
        raise refuse(name, value, f"an integer between 0 and {MAX_DIGITS}")
    return value


def _title(name: str, value: object) -> str:
    if isinstance(value, str):
        return value
    raise refuse(name, value, "a string")


def _labels(name: str, value: object) -> Mapping[str, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        return dict(cast("Mapping[str, str]", value))
    raise refuse(name, value, "a mapping from term names to printed labels")


_VALIDATORS: Mapping[str, Callable[[str, object], object]] = {
    "names": _optional_strings,
    "scale": lambda name, value: _choice(name, value, SCALES),
    "uncertainty": lambda name, value: _choice(name, value, UNCERTAINTIES),
    "level": _level,
    "vcov": lambda _name, value: value,
    "stars": _stars,
    "keep": _optional_strings,
    "drop": _strings,
    "labels": _labels,
    "digits": _digits,
    "stats": _statistics,
    "title": _title,
    "notes": _strings,
}


def collect(options: Mapping[str, object]) -> EtableOptions:
    """Validate the keywords of ``etable`` and resolve the defaults.

    Raises
    ------
    SpecificationError
        An unknown keyword (``specification.unknown_option``) or a value outside
        its domain (``specification.invalid_value``,
        ``specification.table_argument``).
    """
    valid = tuple(item.name for item in fields(EtableOptions))
    # Each value below is validated for its field; the record is built from them.
    resolved: dict[str, Any] = {}
    for name, value in options.items():
        if name not in valid:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": "keyword of etable",
                    "key": name,
                    "valid": messages.format_keys(valid),
                },
            )
        resolved[name] = _VALIDATORS[name](name, value)
    return EtableOptions(**resolved)
