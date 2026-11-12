"""Small storage extensions for internal cancellation geometry records."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from powermean.fe._cancel_storage import BuilderRecord

BoolArray = NDArray[np.bool_]


class BoolStorage(NamedTuple):
    """Immutable boolean bytes, with a fresh outward view on every access."""

    data: bytes

    @classmethod
    def store(cls, values: BoolArray) -> BoolStorage:
        """Detach checked boolean values."""
        return cls(values.tobytes(order="C"))

    def view(self) -> BoolArray:
        """A fresh view backed by immutable bytes."""
        return np.frombuffer(self.data, dtype=np.bool_)


def _build_record(cls: type[BuilderRecord], **fields: object) -> BuilderRecord:
    """Assemble already validated fields using the existing BuilderRecord seam."""
    record = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(record, name, value)
    return record
