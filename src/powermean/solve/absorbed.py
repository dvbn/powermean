"""Options of the absorbed Newton organizations, and the record machinery of options.

Internal records.  Nothing in the package builds or reads them:
:class:`~powermean.solve.options.SolverOptions`,
:func:`~powermean.solve.options.resolve_options` and ``solver_defaults`` keep their
thirteen options, whose resolver refuses these keys as unknown; no absorbed
organization is a default.  Each option is a field with its default, its valid values
and one sentence of documentation:

    record           option                  default     valid values
    AbsorbedOptions  absorbed                (required)  'index', 'profiled'
                     absorb_tol_start        1e-4        positive finite real numbers
                     absorb_tol_final        2e-11       positive finite real numbers
                     absorb_max_sweeps       20000       positive integers
                     absorb_acceleration     True        True or False
                     profile_tol             2e-11       positive finite real numbers
                     profile_max_sweeps      20000       positive integers

The options of the optional no-FE IOLS solver are the record of
:mod:`powermean.solve.iols_options`, which validates and reads its mapping with the
machinery below: :class:`Setting`, the valid values of an option of any stored type;
:class:`Flag`, a Boolean by type; :func:`resolved`, one value against its domain; and
:func:`changes`, a flat mapping of options read once.

The absorbed organization has no default: the record names it explicitly.

Domains: positive finite tolerances, positive integer budgets, a Boolean for
``absorb_acceleration``;
no ordering of the tolerances or relation between the budgets is imposed.  The domain
records and the value rules of :mod:`powermean.solve.options` are reused: a real
option stores the nearest finite ``float``, an integer option the exact ``int``, and
booleans, durations, strings and a float for an integer are refused.  The Boolean
accepts ``True`` and ``False`` only, as the interface reads a flag.

Not recorded, so that their keys are unknown here: ``threads``; the tighter diagnostic
tolerances of an iteration-triggered existence reanalysis; the root Newton,
continuation, GMM and dense options.

Errors, with the messages of :mod:`powermean.messages`: a value outside its domain
raises ``SpecificationError`` (``specification.invalid_value``) with the option as
``argument``; through ``from_dict``, a key that names no option of the record raises
``specification.unknown_option`` with ``kind="solver option"`` and every valid key,
and something other than a mapping ``specification.invalid_value`` with
``argument="solver_options"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Literal, Protocol, get_args

from powermean.errors import SpecificationError
from powermean.solve.options import (
    KIND,
    OPTION_MAPPINGS,
    POSITIVE_INTEGERS,
    POSITIVE_REALS,
    Choices,
    IntegerRange,
    RealInterval,
)

Organization = Literal["profiled", "index"]
BasisBackend = Literal["python", "cpp"]
# The field ``valid`` of a refused Boolean, as the interface words it.
BOOLEANS = "True or False"


class Setting(Protocol):
    """The valid values of an option, and the text that names them in an error.

    The protocol of :class:`powermean.solve.options.Domain`, whose stored value may be
    of any type: a scalar, a Boolean, or the tuple of a grid.
    """

    @property
    def valid(self) -> str:
        """The field ``valid`` of ``specification.invalid_value``."""
        ...

    def resolve(self, value: object) -> object:
        """The value to store for ``value``, or ``None`` if it is not valid."""
        ...


@dataclass(frozen=True)
class Flag:
    """``True`` or ``False``, by type: a number or a NumPy Boolean is not a flag."""

    valid: str = BOOLEANS

    def resolve(self, value: object) -> bool | None:
        """The Boolean itself, or ``None`` if ``value`` is not a ``bool``."""
        return value if isinstance(value, bool) else None


POSITIVE_REAL = RealInterval(lower=0.0, upper=None, valid=POSITIVE_REALS)
POSITIVE_INTEGER = IntegerRange(lower=1, upper=None, valid=POSITIVE_INTEGERS)

# The domain of each option of the record, read whenever a record is validated.
ABSORBED_DOMAINS: Mapping[str, Setting] = MappingProxyType(
    {
        "absorbed": Choices(values=get_args(Organization)),
        "absorb_tol_start": POSITIVE_REAL,
        "absorb_tol_final": POSITIVE_REAL,
        "absorb_max_sweeps": POSITIVE_INTEGER,
        "absorb_acceleration": Flag(),
        "profile_tol": POSITIVE_REAL,
        "profile_max_sweeps": POSITIVE_INTEGER,
        "basis_backend": Choices(values=get_args(BasisBackend)),
    }
)


def resolved(domains: Mapping[str, Setting], name: str, value: object) -> object:
    """The value stored for option ``name``; a value outside its domain is an error."""
    domain = domains[name]
    stored = domain.resolve(value)
    if stored is None:
        raise SpecificationError(
            "specification.invalid_value",
            {"argument": name, "value": value, "valid": domain.valid},
        )
    return stored


def changes(options: object, names: tuple[str, ...]) -> dict[str, Any]:
    """The values of a flat mapping of options, by option, before their validation.

    Raises ``SpecificationError``: ``specification.invalid_value`` if ``options`` is
    not a mapping; ``specification.unknown_option`` for the first key, in the order of
    the mapping, that names no option of ``names``.  The mapping is read once and
    neither modified nor retained.
    """
    if not isinstance(options, Mapping):
        raise SpecificationError(
            "specification.invalid_value",
            {"argument": "solver_options", "value": options, "valid": OPTION_MAPPINGS},
        )
    keys = Choices(values=names)
    given: dict[str, Any] = {}
    for key, value in options.items():
        name = keys.resolve(key)
        if name is None:
            raise SpecificationError(
                "specification.unknown_option",
                {"kind": KIND, "key": key, "valid": keys.valid},
            )
        given[name] = value
    return given


@dataclass(frozen=True, kw_only=True)
class AbsorbedOptions:
    """The options of the joint solver with absorbed effects.

    Keyword arguments only; ``absorbed`` is required.  Each value is validated against
    :data:`ABSORBED_DOMAINS` and stored as resolved.

    Parameters
    ----------
    absorbed : {'profiled', 'index'}
        The organization of the absorbed Newton solver, named explicitly: Newton on the
        criterion with the effects profiled, or in the fitted index with the effects
        absorbed, as ``ppmlhdfe`` does; no default.
    absorb_tol_start : float, default 1e-4
        Inner tolerance of the weighted absorption at the first outer iteration; a
        positive finite real.
    absorb_tol_final : float, default 2e-11
        Tolerance of the final tight absorption before the certificate; a positive
        finite real.
    absorb_max_sweeps : int, default 20000
        Budget of sweeps of one absorption; a positive integer.
    absorb_acceleration : bool, default True
        Whether the alternating projections are accelerated.
    profile_tol : float, default 2e-11
        Tolerance of the effect equations of a profile; a positive finite real.
    profile_max_sweeps : int, default 20000
        Budget of sweeps of one profile; a positive integer.
    basis_backend : {'python', 'cpp'}, default 'python'
        Exact FE basis implementation. Explicit cpp requires the built accelerator;
        representation declines use Python exact arithmetic and are recorded.
    """

    absorbed: Organization
    absorb_tol_start: float = 1e-4
    absorb_tol_final: float = 2e-11
    absorb_max_sweeps: int = 20000
    absorb_acceleration: bool = True
    profile_tol: float = 2e-11
    profile_max_sweeps: int = 20000
    basis_backend: BasisBackend = "python"

    def __post_init__(self) -> None:
        """Validate every option, in the order of the fields, and store it resolved."""
        for option in fields(self):
            value = getattr(self, option.name)
            stored = resolved(ABSORBED_DOMAINS, option.name, value)
            object.__setattr__(self, option.name, stored)

    def to_dict(self) -> dict[str, object]:
        """The resolved options by name, in the order of the fields, as a new dict."""
        return {option.name: getattr(self, option.name) for option in fields(self)}

    @classmethod
    def from_dict(cls, options: object) -> AbsorbedOptions:
        """The record of a flat mapping of options: its values over the defaults.

        The mapping must name ``absorbed``: without it the construction raises
        ``TypeError``, since the organization has no default.
        """
        return cls(**changes(options, tuple(ABSORBED_DOMAINS)))
