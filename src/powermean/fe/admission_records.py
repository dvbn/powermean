"""Private immutable evidence and actual-work accounting for fit admission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import NamedTuple, TypeVar, cast

import numpy as np

from powermean.fe.geometry_errors import GeometryAdmissionError


class EvidenceArray(NamedTuple):
    """Array evidence as immutable bytes and detached shape/type metadata."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes


class EvidenceRecord(NamedTuple):
    """Named evidence fields, recursively detached from runtime records."""

    name: str
    fields: tuple[tuple[str, EvidenceValue], ...]


EvidenceValue = (
    None
    | bool
    | int
    | float
    | str
    | EvidenceArray
    | EvidenceRecord
    | tuple["EvidenceValue", ...]
)


def freeze_evidence(value: object) -> EvidenceValue:
    """Detach numerical diagnostic records without retaining rich runtime state."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        return EvidenceArray(value.dtype.str, value.shape, value.tobytes(order="C"))
    if isinstance(value, tuple | list):
        return tuple(freeze_evidence(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return EvidenceRecord(
            type(value).__name__,
            tuple(
                (field.name, freeze_evidence(cast(object, getattr(value, field.name))))
                for field in fields(value)
            ),
        )
    raise TypeError("unsupported admission evidence")


class WorkCount(NamedTuple):
    """Actual begun/completed calls; failed calls remain in the begun count."""

    name: str
    begun: int
    completed: int


@dataclass(frozen=True)
class AdmissionWork:
    """Immutable structural record; point evaluations have a separate lifetime."""

    counts: tuple[WorkCount, ...]


_Result = TypeVar("_Result")


class WorkLog:
    """One local construction log, never a cross-call cache."""

    def __init__(self) -> None:
        self._begun: dict[str, int] = {}
        self._completed: dict[str, int] = {}
        self.evidence: EvidenceValue = None

    def begin(self, name: str, units: int = 1) -> None:
        """Charge immediately before the named operation starts."""
        self._begun[name] = self._begun.get(name, 0) + units

    def complete(self, name: str, units: int = 1) -> None:
        """Record only completed work."""
        self._completed[name] = self._completed.get(name, 0) + units

    def run(self, name: str, action: Callable[[], _Result]) -> _Result:
        """Count an actual call, including one that raises."""
        self.begin(name)
        result = action()
        self.complete(name)
        return result

    def snapshot(self) -> AdmissionWork:
        """Detach counts without estimating unobserved internal operations."""
        return AdmissionWork(
            tuple(
                WorkCount(name, count, self._completed.get(name, 0))
                for name, count in sorted(self._begun.items())
            )
        )


class AdmissionError(GeometryAdmissionError):
    """Refusal with the actual partial work and completed evidence preserved."""

    def __init__(self, cause: str, message: str, work: WorkLog) -> None:
        """Attach immutable evidence; no failed operation is credited as complete."""
        super().__init__(cause, message)
        self.work = work.snapshot()
        self.evidence = work.evidence
