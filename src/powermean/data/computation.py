"""The resolved numerical settings of one solve: the solver and its options.

:class:`Computation` holds the resolved numerical settings of one solve: the solver,
``"newton"`` (the damped Newton algorithm with two providers) or optional no-FE
``"iols"``, and ``options``, the snapshot of a
:class:`~powermean.solve.options.SolverOptions` that the composing caller has resolved
with :func:`powermean.solve.options.resolve_options` and serialized with its
``to_dict``.  The record stores an owned read-only copy of that mapping and checks its
representation only: a nonempty mapping of names to Python floats, integers or strings,
each float finite, so that it converts to JSON and back without loss.  It defines no
default, schema or domain of a solver option and imports nothing of the solve stage (a
stage of the pipeline imports earlier stages only): a serialized mapping is not evidence
that it was validated, so a caller passes the ``options`` of a record read back through
``resolve_options`` before it uses them, and an unknown or invalid option is refused
there, by the one table of defaults and domains.  The record holds no initialization,
reference backend, pilot or certificate threshold: the start of a solve is an argument
of the interface, the provider an option, and the thresholds of the certificate are
options too.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, get_args

from powermean import messages
from powermean.errors import SpecificationError

Solver = Literal["newton", "iols"]
OptionValue = float | int | str | tuple[float, ...]


def _solver(value: object) -> Solver:
    """``"newton"`` or ``"iols"``; anything else is invalid."""
    if type(value) is str:
        if value == "newton":
            return "newton"
        if value == "iols":
            return "iols"
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "solver",
            "value": value,
            "valid": messages.format_keys(get_args(Solver)),
        },
    )


def _is_scalar(value: object) -> bool:
    """A Python ``float`` that is finite, an ``int`` or a ``str``, by exact type."""
    if type(value) is float:
        return math.isfinite(value)
    return type(value) in (int, str)


def _is_float_grid(value: object) -> bool:
    """A nonempty list or tuple of finite Python ``float`` values."""
    if (
        not isinstance(value, list | tuple)
        or type(value) not in (list, tuple)
        or not value
    ):
        return False
    return all(type(item) is float and math.isfinite(item) for item in value)


def _snapshot(
    options: object, *, solver: Solver = "newton"
) -> Mapping[str, OptionValue]:
    """An owned read-only copy of a resolved mapping of options.

    The representation of a resolved record, nothing of its policy: a nonempty mapping
    whose keys are strings and whose values are finite Python floats, integers or
    strings (or a nonempty float grid for ``solver="iols"``).
    """
    if not isinstance(options, Mapping) or not options:
        raise TypeError("options must be a nonempty mapping of resolved solver options")
    copied: dict[str, OptionValue] = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise ValueError(
                "options must map option names to finite floats, integers or strings"
            )
        if _is_scalar(value):
            assert isinstance(value, float | int | str)
            copied[key] = value
        elif solver == "iols" and _is_float_grid(value):
            assert isinstance(value, list | tuple)
            copied[key] = tuple(float(item) for item in value)
        else:
            raise ValueError(
                "options must map option names to finite floats, integers or strings"
            )
    return MappingProxyType(copied)


@dataclass(frozen=True)
class Computation:
    """The resolved numerical settings of one solve.

    Parameters
    ----------
    options : mapping
        The resolved options, ``resolve_options(...).to_dict()`` of the composing
        caller; required.  Stored as an owned read-only mapping, so that a later change
        of the caller's mapping does not reach the record.
    solver : {"newton", "iols"}, default "newton"
        The selected solver. This storage record does not execute it.
    iols_backend_explicit : bool or None, default None
        Backend-presence metadata for IOLS snapshots, separate from the nine solver
        options. None on a Newton record; a legacy IOLS snapshot defaults to True.

    Raises
    ------
    TypeError
        ``options`` is not a nonempty mapping.
    ValueError
        A key that is not a string, or a value that is not a finite Python float, an
        integer or a string.
    SpecificationError
        An unknown solver (``specification.invalid_value``).

    Notes
    -----
    The record does not know the names, defaults or domains of the options: the
    caller resolves them before, and again after reading a record back.
    """

    options: Mapping[str, OptionValue]
    solver: Solver = "newton"
    # Presence metadata, not a solver option. Legacy complete IOLS mappings are strict.
    iols_backend_explicit: bool | None = None

    def __post_init__(self) -> None:
        """Check the solver and the representation of the options; store them owned."""
        resolved_solver = _solver(self.solver)
        object.__setattr__(self, "solver", resolved_solver)
        presence = self.iols_backend_explicit
        if presence is not None and type(presence) is not bool:
            raise ValueError("iols_backend_explicit must be a Boolean or None")
        if resolved_solver == "newton" and presence is not None:
            raise ValueError("IOLS backend presence is only valid for IOLS")
        if resolved_solver == "iols" and presence is None:
            object.__setattr__(self, "iols_backend_explicit", True)
        object.__setattr__(
            self, "options", _snapshot(self.options, solver=resolved_solver)
        )

    def to_dict(self) -> dict[str, object]:
        """The type, the solver and a fresh copy of the options."""
        return {
            "type": "Computation",
            "solver": self.solver,
            **(
                {"iols_backend_explicit": self.iols_backend_explicit}
                if self.solver == "iols"
                else {}
            ),
            "options": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.options.items()
            },
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Computation:
        """The record of :meth:`to_dict`; its options are a snapshot to resolve again.

        Raises ``ValueError`` for a record of another type and ``KeyError`` without
        ``options``.
        """
        data: dict[str, Any] = dict(record)
        if data.pop("type", None) != "Computation":
            raise ValueError("not a Computation record")
        return cls(options=data.pop("options"), **data)
