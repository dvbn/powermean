"""Options of the certificate and of the convex Newton solver, as one typed record.

The thirteen options: the four thresholds of the certificate and the nine settings of
the convex Newton solver, with their numeric domains.  Each is a field of
:class:`SolverOptions` with its default, its valid values (:data:`DOMAINS`) and one
sentence of documentation:

    option            default      valid values
    tol               1e-8         positive finite real numbers
    tol_secondary     1e-6         positive finite real numbers
    coef_tol          1e-6         positive finite real numbers
    rcond_min         1e-13        real numbers strictly between 0 and 1
    newton_provider   'bundled'    'bundled', 'sklearn'
    gradient_tol      1e-10        positive finite real numbers
    max_iter          100000       positive integers
    max_evaluations   1000000      positive integers
    armijo            2**-11       real numbers strictly between 0 and 1
    backtrack_factor  0.5          real numbers strictly between 0 and 1
    max_backtracks    21           positive integers
    retry             1            the integers 0 and 1
    curvature         'observed'   'expected', 'observed'

The other groups of options (absorption, root Newton, continuation, GMM, IOLS, dense
systems, threads) are not in this record; their keys are unknown here.
:func:`resolve_options` turns a flat mapping into the record, and
:meth:`SolverOptions.to_dict` returns the resolved values by name, for the provenance
of a result.

Values.  A real option accepts a number of a type registered as ``numbers.Real``
(Python and NumPy integers and floats, ``fractions.Fraction``) and stores the
``float`` nearest to it.  An integer option accepts a number of a type registered as
``numbers.Integral`` (Python and NumPy integers) and stores it exactly, as an
``int``.  Booleans, NumPy durations, complex numbers, strings (numeric or not),
arrays of any shape, ``decimal.Decimal`` and a float for an integer option are
refused, never converted or truncated.  The stored value must satisfy the domain: a
number without a finite ``float`` (``10**400``), or whose nearest ``float`` leaves the
domain (a positive fraction that rounds to zero, a fraction below 1 that rounds to 1),
is refused like any other value outside it.  Nothing is clipped, no budget is capped,
no ordering of the tolerances or relation between the budgets is imposed, and the
conditioning safeguard cannot be disabled.

Errors, from :mod:`powermean.errors` with the messages of :mod:`powermean.messages`.
A value outside its domain raises ``SpecificationError`` with the identifier
``specification.invalid_value`` and the fields ``argument`` (the option), ``value``
(as supplied) and ``valid``: the text of its domain in the table above, one constant
of this module per numeric domain, or the values of a categorical option through
:func:`powermean.messages.format_keys`.  A key that names no option raises
``SpecificationError`` with the identifier ``specification.unknown_option``,
``kind="solver option"``, the ``key`` as supplied and every valid key in ``valid``.
The record validates its fields when it is constructed, directly or through
:func:`resolve_options`, so that neither path bypasses the other.

Not decided here: how a top-level ``fit(tol=...)`` combines with ``solver_options``;
whether an installed scikit-learn may serve as the provider ``"sklearn"`` (the record
checks the name only); and the bread of the covariance, since ``curvature="expected"``
selects Fisher scoring in the Newton system, not the inference option.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, Protocol, get_args

import numpy as np

from powermean import messages
from powermean.errors import SpecificationError

# The texts that fill the field ``valid`` of ``specification.invalid_value``: one per
# numeric domain, and one for a value that is not a mapping.  Each
# reads after "the valid values are".
POSITIVE_REALS = "positive finite real numbers"
UNIT_INTERVAL = "real numbers strictly between 0 and 1"
POSITIVE_INTEGERS = "positive integers"
RETRY_COUNTS = "the integers 0 and 1"
OPTION_MAPPINGS = "mappings of solver option names to values"
# The field ``kind`` of ``specification.unknown_option``.
KIND = "solver option"

# The values of the two categorical options.
Provider = Literal["bundled", "sklearn"]
Curvature = Literal["observed", "expected"]

# Refused before the numeric test, because booleans are not numeric settings:
# Python's booleans are integers, and NumPy's are not registered as numbers at all.
# NumPy registers its durations, a subclass of its signed integers, as integers, but
# a duration is neither a budget nor a tolerance.
_NOT_NUMBERS = (bool, np.bool_, np.timedelta64)


def _real(value: object) -> float | None:
    """The finite ``float`` nearest to a real number, or ``None``.

    For the supported types ``float`` returns the nearest double.  It raises
    ``OverflowError`` for an integer or a fraction beyond the largest double, which
    has no finite ``float``, and neither has an infinite or NaN float.
    """
    if isinstance(value, _NOT_NUMBERS) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    """An integer of any registered type as an exact ``int``, or ``None``."""
    if isinstance(value, _NOT_NUMBERS) or not isinstance(value, Integral):
        return None
    return operator.index(value)


class Domain(Protocol):
    """The valid values of an option, and the text that names them in an error."""

    @property
    def valid(self) -> str:
        """The field ``valid`` of ``specification.invalid_value``."""
        ...

    def resolve(self, value: object) -> float | int | str | None:
        """The value to store for ``value``, or ``None`` if it is not valid."""
        ...


@dataclass(frozen=True)
class RealInterval:
    """Finite real numbers strictly above ``lower`` and, if given, below ``upper``."""

    lower: float
    upper: float | None
    valid: str

    def resolve(self, value: object) -> float | None:
        """The nearest ``float``, if it lies strictly inside the interval."""
        number = _real(value)
        if number is None or number <= self.lower:
            return None
        if self.upper is not None and number >= self.upper:
            return None
        return number


@dataclass(frozen=True)
class IntegerRange:
    """Integers from ``lower`` to ``upper`` inclusive, unbounded above without one."""

    lower: int
    upper: int | None
    valid: str

    def resolve(self, value: object) -> int | None:
        """The exact ``int``, if it lies in the range."""
        number = _integer(value)
        if number is None or number < self.lower:
            return None
        if self.upper is not None and number > self.upper:
            return None
        return number


@dataclass(frozen=True)
class Choices:
    """One of the strings ``values``, compared exactly."""

    values: tuple[str, ...]

    @property
    def valid(self) -> str:
        """The values, sorted and quoted by :func:`powermean.messages.format_keys`."""
        return messages.format_keys(self.values)

    def resolve(self, value: object) -> str | None:
        """The member of ``values`` equal to ``value``, which must be a string."""
        if isinstance(value, str):
            for choice in self.values:
                if value == choice:
                    return choice
        return None


_POSITIVE_REAL = RealInterval(lower=0.0, upper=None, valid=POSITIVE_REALS)
_UNIT = RealInterval(lower=0.0, upper=1.0, valid=UNIT_INTERVAL)
_POSITIVE_INTEGER = IntegerRange(lower=1, upper=None, valid=POSITIVE_INTEGERS)

# The domain of each option, read whenever a record is validated.
DOMAINS: Mapping[str, Domain] = MappingProxyType(
    {
        "tol": _POSITIVE_REAL,
        "tol_secondary": _POSITIVE_REAL,
        "coef_tol": _POSITIVE_REAL,
        "rcond_min": _UNIT,
        "newton_provider": Choices(values=get_args(Provider)),
        "gradient_tol": _POSITIVE_REAL,
        "max_iter": _POSITIVE_INTEGER,
        "max_evaluations": _POSITIVE_INTEGER,
        "armijo": _UNIT,
        "backtrack_factor": _UNIT,
        "max_backtracks": _POSITIVE_INTEGER,
        "retry": IntegerRange(lower=0, upper=1, valid=RETRY_COUNTS),
        "curvature": Choices(values=get_args(Curvature)),
    }
)


def _resolved(name: str, value: object) -> float | int | str:
    """The value stored for option ``name``; a value outside its domain is an error."""
    domain = DOMAINS[name]
    resolved = domain.resolve(value)
    if resolved is None:
        raise SpecificationError(
            "specification.invalid_value",
            {"argument": name, "value": value, "valid": domain.valid},
        )
    return resolved


@dataclass(frozen=True, kw_only=True)
class SolverOptions:
    """The options of the certificate and of the convex Newton solver.

    Keyword arguments only.  Each value is validated against :data:`DOMAINS` and
    stored as resolved: a ``float`` for a real option, an ``int`` for an integer
    option, the exact string for a categorical one.  A value outside its domain
    raises ``SpecificationError`` (``specification.invalid_value``).

    Parameters
    ----------
    tol : float, default 1e-8
        Tolerance of the certificate on the scaled score and the index correction; a
        positive finite real.
    tol_secondary : float, default 1e-6
        Secondary level of the same two criteria, met by a solution whose status is
        ``verified_secondary``; a positive finite real.
    coef_tol : float, default 1e-6
        Largest Newton correction of a reported coefficient that the certificate
        accepts, in units of its standard error; a positive finite real.
    rcond_min : float, default 1e-13
        Threshold that the reciprocal condition number of the factorized system, in
        scaled coordinates, must exceed for the certificate; strictly between 0 and 1.
    newton_provider : {'bundled', 'sklearn'}, default 'bundled'
        Copy of scikit-learn's Newton iteration that runs: the files carried by the
        package or the installed scikit-learn; naming one does not
        verify it.
    gradient_tol : float, default 1e-10
        Native stopping tolerance of the Newton iteration, its gradient criterion; a
        positive finite real.
    max_iter : int, default 100000
        Budget of Newton iterations; a positive integer.
    max_evaluations : int, default 1000000
        Budget of evaluations of the system; a positive integer.
    armijo : float, default 2**-11
        Constant of the Armijo sufficient-decrease condition of the line search;
        strictly between 0 and 1.
    backtrack_factor : float, default 0.5
        Factor by which a rejected step is contracted; strictly between 0 and 1.
    max_backtracks : int, default 21
        Largest number of trial steps in one line search; a positive integer.
    retry : int, default 1
        Retries of the same method when a native stop misses the certificate but has
        improved the score; 0 or 1.
    curvature : {'observed', 'expected'}, default 'observed'
        Curvature of the Newton system: observed, or expected for Fisher scoring; it
        does not select the bread of the covariance.
    """

    tol: float = 1e-8
    tol_secondary: float = 1e-6
    coef_tol: float = 1e-6
    rcond_min: float = 1e-13
    newton_provider: Provider = "bundled"
    gradient_tol: float = 1e-10
    max_iter: int = 100000
    max_evaluations: int = 1000000
    armijo: float = 2.0**-11
    backtrack_factor: float = 0.5
    max_backtracks: int = 21
    retry: int = 1
    curvature: Curvature = "observed"

    def __post_init__(self) -> None:
        """Validate every option, in the order of the fields, and store it resolved."""
        for option in fields(self):
            value = getattr(self, option.name)
            object.__setattr__(self, option.name, _resolved(option.name, value))

    def to_dict(self) -> dict[str, float | int | str]:
        """The resolved options by name, in the order of the fields, as Python scalars.

        The defaults convert to JSON.  A budget is the exact ``int`` given, whatever
        its size, and :mod:`json` writes an integer only within the interpreter's
        limit on the digits of an integer converted to text
        (:func:`sys.get_int_max_str_digits`).
        """
        return {option.name: getattr(self, option.name) for option in fields(self)}


# The valid keys of a mapping of options: the fields of the record, in their order.
OPTION_NAMES: tuple[str, ...] = tuple(option.name for option in fields(SolverOptions))
_KEYS = Choices(values=OPTION_NAMES)


def resolve_options(options: object) -> SolverOptions:
    """The record of a flat mapping of options: its values over the defaults.

    Parameters
    ----------
    options : mapping
        Option names to values, owned by the caller: each value is read once, and
        the mapping is neither modified nor retained.  Absent options keep their
        defaults; the mapping is flat, so a nested mapping is a value, not a group.

    Returns
    -------
    SolverOptions
        The validated record, in which each value of the mapping replaces the default
        of its option and of no other.

    Raises
    ------
    SpecificationError
        ``specification.invalid_value`` with ``argument="solver_options"`` if
        ``options`` is not a mapping; ``specification.unknown_option`` for the first
        key, in the order of the mapping, that names no option, before any value is
        validated; then ``specification.invalid_value`` for the first invalid value,
        in the order of the fields.

    Notes
    -----
    ``options`` is typed as ``object`` because this function is the run-time check
    of what a caller passes: its first test is a type test.
    """
    if not isinstance(options, Mapping):
        raise SpecificationError(
            "specification.invalid_value",
            {"argument": "solver_options", "value": options, "valid": OPTION_MAPPINGS},
        )
    # Values of any type, by option: the construction of the record validates them.
    changes: dict[str, Any] = {}
    for key, value in options.items():
        name = _KEYS.resolve(key)
        if name is None:
            raise SpecificationError(
                "specification.unknown_option",
                {"kind": KIND, "key": key, "valid": _KEYS.valid},
            )
        changes[name] = value
    return SolverOptions(**changes)
