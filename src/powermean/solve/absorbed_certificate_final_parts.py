"""Owned copies of the arrays bound by a candidate's final work.

Internal module: re-forming a candidate's final record so that it
retains no array that another holder can write, used by
:mod:`powermean.solve.absorbed_certificate_final`.

Each array of the records that do not own their arrays at construction (the final
profile's base, the point equations, the final rows, the absorbed weights and Schur
matrix, the recovery effects and normalization, the alarms) that is not read-only over
bytes of its own is replaced by such a copy, of the same dtype, shape and bytes.  A
record whose arrays are all closed, as every record of the accepted producers is, is
kept with its identity.  ``ValueError`` (:data:`INCOHERENT`) refuses an entry that is
not a numeric or boolean array.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import Field, replace
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from powermean.solve.absorbed_direction import Recovery
from powermean.solve.newton_absorbed import IndexFinal
from powermean.solve.profiled_indications import Alarm
from powermean.solve.profiled_loss import FinalWork

INCOHERENT = "the candidate's history is not coherent with the accepted final work"
AnyArray = NDArray[Any]
Own = Callable[[Any], Any]


class Dataclass(Protocol):
    """A dataclass instance, which ``dataclasses.replace`` re-forms."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def _closed(values: AnyArray) -> bool:
    """Whether ``values`` is read-only over immutable bytes that nothing can write."""
    if values.flags.writeable:
        return False
    base: object = values
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _owned(values: object) -> object:
    """An array, itself when closed, else a read-only copy over bytes of its own."""
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "biuf":
        raise ValueError(INCOHERENT)
    if _closed(values):
        return values
    stored: AnyArray = np.frombuffer(values.tobytes(), dtype=values.dtype)
    return stored.reshape(values.shape)


def _owned_field(value: object) -> object:
    """An optional array or tuple of arrays, owned; itself when all are closed."""
    if value is None:
        return None
    if isinstance(value, tuple):
        found = tuple(_owned(v) for v in value)
        same = all(a is b for a, b in zip(found, value, strict=True))
        return value if same else found
    return _owned(value)


def _reformed[Record: Dataclass](
    record: Record,
    arrays: tuple[str, ...] = (),
    nested: Mapping[str, Own] = MappingProxyType({}),
) -> Record:
    """``record`` with its named arrays and nested records owned, by replacement."""
    changes: dict[str, object] = {}
    for name in arrays:
        value = getattr(record, name)
        found = _owned_field(value)
        if found is not value:
            changes[name] = found
    for name, own in nested.items():
        value = getattr(record, name)
        found = None if value is None else own(value)
        if found is not value:
            changes[name] = found
    return replace(record, **changes) if changes else record


def _owned_alarm(alarm: Alarm) -> Alarm:
    """An alarm with its zero-row and chord diagnostics owned."""
    return _reformed(
        alarm,
        nested={
            "zero_rows": lambda z: _reformed(z, ("rows", "positions", "indexes")),
            "chord": lambda c: _reformed(c, ("before", "after")),
        },
    )


def _owned_recovery(recovery: Recovery) -> Recovery:
    """A recovery with its effects and normalization owned (``eta`` owns itself)."""
    return _reformed(
        recovery,
        ("effects",),
        {"normalization": lambda n: _reformed(n, ("effects", "references", "shifts"))},
    )


def owned_profiled(final: FinalWork) -> FinalWork:
    """A profiled final record with every array it binds owned (module docstring)."""
    return _reformed(
        final,
        nested={
            "point": lambda p: _reformed(p, ("base",)),
            "equations": lambda e: _reformed(
                e, ("gradient", "curvature", "information")
            ),
            "absorbed": lambda a: _reformed(a, ("weights", "matrix")),
            "alarm": _owned_alarm,
        },
    )


def owned_index(final: IndexFinal) -> IndexFinal:
    """An index final record with every array it binds owned (module docstring)."""
    return _reformed(
        final,
        nested={
            "point": lambda p: _reformed(p, nested={"recovered": _owned_recovery}),
            "rows": lambda r: _reformed(r, ("gradient", "curvature", "right")),
            "absorbed": lambda a: _reformed(a, ("weights", "matrix")),
            "alarm": _owned_alarm,
        },
    )
