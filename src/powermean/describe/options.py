"""The keywords of the preliminary tables: one typed record, validated at the entry.

Each keyword is a field of :class:`TableOptions` with its default.  A table function
declares the keywords that it accepts (:data:`SUMMARY_KEYWORDS` and the three
others); :func:`collect` refuses any other key with ``specification.unknown_option``,
which lists the valid ones, and a value outside its domain with
``specification.invalid_value`` (a closed list of values) or
``specification.table_argument`` (any other domain).

==================  =================================================================
``probs``           quantile levels in ``[0, 1]``; default those of the R package RCT
``orders``          mean orders of the outcome table; default ``(1,)``
``control``         the reference arm; default the first value in sorted order
``bins``            cut a continuous treatment into this many quantile groups
``continuous``      balance regression of the dose itself on the variables
``cluster``         one column: clustered assignment or sampling
``fe``              one classification, or ``"a^b"``: strata absorbed
``weights``         one column of analytic or probability weights
``by``              one column: repeat the table within its levels
``vcov``            ``"robust"`` or, in the balance regression, ``"classical"``
``ssc``             ``"stata"``, ``"none"`` or ``"regression"``
``complete``        restrict every row to the rows complete in all variables
``expand``          expand categorical columns to indicators
``positive_only``   compute orders at or below zero on positive outcomes
``level``           confidence level of the intervals of the outcome table
``sample``          a fitted result: use the rows of its estimation sample
``labels``          printed labels by column name
``digits``          decimals of the printed real numbers
==================  =================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypedDict, cast

from powermean import messages
from powermean.errors import SpecificationError

Preset = Literal["stata", "none", "regression"]
Covariance = Literal["robust", "classical"]

DEFAULT_PROBS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
PRESETS: tuple[str, ...] = ("stata", "none", "regression")
COVARIANCES: tuple[str, ...] = ("robust", "classical")


class TableKeywords(TypedDict, total=False):
    """The keywords of the four table functions, for the type checker."""

    probs: Sequence[float]
    orders: Sequence[float]
    control: object
    bins: int | None
    continuous: bool
    cluster: str | None
    fe: str | None
    weights: str | None
    by: str | None
    vcov: str
    ssc: str
    complete: bool
    expand: bool
    positive_only: bool
    level: float
    sample: object
    labels: Mapping[str, str] | None
    digits: int


@dataclass(frozen=True)
class TableOptions:
    """The resolved keywords of a table call (module docstring)."""

    probs: tuple[float, ...] = DEFAULT_PROBS
    orders: tuple[float, ...] = (1.0,)
    control: object = None
    bins: int | None = None
    continuous: bool = False
    cluster: str | None = None
    fe: str | None = None
    weights: str | None = None
    by: str | None = None
    vcov: Covariance = "robust"
    ssc: Preset = "stata"
    complete: bool = False
    expand: bool = False
    positive_only: bool = False
    level: float = 0.95
    sample: object = None
    labels: Mapping[str, str] = field(default_factory=dict)
    digits: int = 2


_COMMON = ("weights", "by", "sample", "labels", "digits")
SUMMARY_KEYWORDS = (*_COMMON, "probs", "complete", "expand")
BALANCE_KEYWORDS = (
    *_COMMON,
    "control",
    "bins",
    "cluster",
    "fe",
    "ssc",
    "complete",
    "expand",
)
REGRESSION_KEYWORDS = (
    *_COMMON,
    "control",
    "bins",
    "continuous",
    "cluster",
    "fe",
    "vcov",
    "ssc",
    "expand",
)
OUTCOME_KEYWORDS = (
    *_COMMON,
    "orders",
    "control",
    "bins",
    "cluster",
    "ssc",
    "positive_only",
    "level",
)


def refuse(argument: str, value: object, expected: str) -> SpecificationError:
    """The refusal of a value outside the domain of its argument."""
    return SpecificationError(
        "specification.table_argument",
        {"argument": argument, "value": value, "expected": expected},
    )


def _reals(name: str, value: object, expected: str) -> tuple[float, ...]:
    """A nonempty sequence of finite real numbers, as a tuple of floats."""
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise refuse(name, value, expected)
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise refuse(name, value, expected)
        if not math.isfinite(item):
            raise refuse(name, value, expected)
        numbers.append(float(item))
    return tuple(numbers)


def _probs(value: object) -> tuple[float, ...]:
    expected = "a nonempty sequence of numbers between 0 and 1"
    numbers = _reals("probs", value, expected)
    if any(not 0.0 <= p <= 1.0 for p in numbers):
        raise refuse("probs", value, expected)
    return numbers


def _orders(value: object) -> tuple[float, ...]:
    return _reals("orders", value, "a nonempty sequence of finite mean orders")


def _column(name: str, value: object) -> str | None:
    if value is None or (isinstance(value, str) and value):
        return value
    raise refuse(name, value, "the name of one column, or None")


def _flag(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise refuse(name, value, "True or False")


def _choice(name: str, value: object, valid: tuple[str, ...]) -> str:
    if isinstance(value, str) and value in valid:
        return value
    raise SpecificationError(
        "specification.invalid_value",
        {"argument": name, "value": value, "valid": messages.format_keys(valid)},
    )


def _bins(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise refuse("bins", value, "an integer of at least 2, or None")
    return value


def _level(value: object) -> float:
    if isinstance(value, float) and 0.0 < value < 1.0:
        return value
    raise refuse("level", value, "a float strictly between zero and one")


def _digits(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
        raise refuse("digits", value, "an integer between 0 and 15")
    return value


def _labels(value: object) -> Mapping[str, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        return dict(cast("Mapping[str, str]", value))
    raise refuse("labels", value, "a mapping from column names to printed labels")


def _resolved(name: str, value: object) -> object:
    """The validated value of one keyword."""
    if name in ("cluster", "fe", "weights", "by"):
        return _column(name, value)
    if name in ("continuous", "complete", "expand", "positive_only"):
        return _flag(name, value)
    validators = {
        "probs": _probs,
        "orders": _orders,
        "bins": _bins,
        "level": _level,
        "digits": _digits,
        "labels": _labels,
    }
    if name in validators:
        return validators[name](value)
    if name == "vcov":
        return _choice(name, value, COVARIANCES)
    if name == "ssc":
        return _choice(name, value, PRESETS)
    return value


def collect(
    options: Mapping[str, object],
    function: str,
    allowed: tuple[str, ...],
    *,
    digits: int,
) -> TableOptions:
    """Validate the keywords of a call of ``function`` and resolve the defaults.

    Parameters
    ----------
    options : mapping
        The keywords as given.
    function : str
        The name of the public function, for the message.
    allowed : tuple of str
        The keywords that the function accepts.
    digits : int
        The default number of decimals of the function.

    Raises
    ------
    SpecificationError
        An unknown keyword (``specification.unknown_option``); a value outside its
        domain (``specification.invalid_value``, ``specification.table_argument``);
        ``vcov="classical"`` with ``cluster`` (``specification.conflicting``).
    """
    known = {item.name for item in fields(TableOptions)}
    # Each value below is validated for its field; the record is built from them.
    resolved: dict[str, Any] = {"digits": digits}
    for name, value in options.items():
        if name not in allowed or name not in known:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": f"keyword of {function}",
                    "key": name,
                    "valid": messages.format_keys(allowed),
                },
            )
        resolved[name] = _resolved(name, value)
    if resolved.get("vcov") == "classical" and resolved.get("cluster") is not None:
        raise SpecificationError(
            "specification.conflicting",
            {"first": "vcov='classical'", "second": "cluster"},
        )
    if resolved.get("continuous") and resolved.get("bins") is not None:
        raise SpecificationError(
            "specification.conflicting", {"first": "continuous=True", "second": "bins"}
        )
    return TableOptions(**resolved)
