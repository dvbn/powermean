"""Immutable storage for the cancellation records, with detached array metadata.

Like ``model.rows._immutable``, numeric storage is backed by owned bytes. Each
access returns a fresh view, so changing an outward shape cannot change a record.
These internal buffers store already checked values; they certify no input.
"""

from __future__ import annotations

from typing import NamedTuple, Never, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.model.variance import FloatArray, _finite_array, _finite_result

IndexArray = NDArray[np.intp]


class FloatStorage(NamedTuple):
    """Owned doubles and immutable shape metadata."""

    data: bytes
    shape: tuple[int, ...]

    @classmethod
    def store(cls, array: FloatArray) -> FloatStorage:
        """Detach the data and shape, using the package's immutable-byte convention."""
        return cls(array.tobytes(order="C"), array.shape)

    def view(self) -> FloatArray:
        """A fresh read-only view of the stored doubles."""
        return np.frombuffer(self.data, dtype=np.float64).reshape(self.shape)


class IndexStorage(NamedTuple):
    """Owned one-dimensional integer codes or positions."""

    data: bytes

    @classmethod
    def store(cls, array: IndexArray) -> IndexStorage:
        """Detach a checked integer vector."""
        return cls(array.tobytes(order="C"))

    def view(self) -> IndexArray:
        """A fresh read-only view of the stored indices."""
        return np.frombuffer(self.data, dtype=np.intp)


class BuilderRecord:
    """Cancellation records are built by validated factories only.

    Copying and serialization are refused: rebuild at a real input boundary.
    Explicit slots on the frozen records avoid generated dataclass pickle paths.
    Deliberate private reflection or monkeypatching is outside this contract.
    """

    __slots__ = ()

    def __init__(self) -> None:
        raise TypeError("use the validated cancellation builder")

    def __copy__(self) -> Never:
        """Refuse reconstruction outside the builder."""
        raise TypeError("cancellation records do not support copying")

    def __deepcopy__(self, memo: dict[int, object]) -> Never:
        """Refuse reconstruction outside the builder."""
        del memo
        raise TypeError("cancellation records do not support copying")

    def __reduce__(self) -> Never:
        """Refuse pickle serialization."""
        raise TypeError("cancellation records do not support pickle serialization")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        """Refuse every pickle protocol."""
        del protocol
        raise TypeError("cancellation records do not support pickle serialization")

    def __getstate__(self) -> Never:
        """Refuse serialization through the effective class method."""
        raise TypeError("cancellation records do not support pickle serialization")

    def __setstate__(self, state: object) -> Never:
        """Refuse restoration, including a pre-existing pickle's BUILD path."""
        del state
        raise TypeError("cancellation records do not support pickle restoration")


def finite_vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    """Check a real finite vector at every external product or point boundary."""
    array = _finite_array(values, name)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def finite_product(values: FloatArray) -> FloatArray:
    """Refuse a nonrepresentable operator result without changing its method."""
    return _finite_result(values, "cancellation product")
