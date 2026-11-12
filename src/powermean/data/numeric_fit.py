"""Fit-owned numeric rows, independent of preparation and retained object graphs.

This representation is constructed from the already admitted, detached FINAL sample. Its
buffers and original-row positions serve solve, fresh certification and inference. It
holds no FESample, dataframe, formula specification, provider or source callback. Public
constructors and foreign handoffs do not acquire this internal ownership route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.fe.groups import Block, Grouping
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray


def _immutable_bytes(array: np.ndarray) -> bytes | None:
    """The existing complete immutable-byte owner, without copying or callbacks.

    A read-only flag or a view of a writable owner is never an ownership proof.
    Shape/dtype/strides live in the new storage, not in an externally held header.
    """
    owner: object = array
    while type(owner) is np.ndarray:
        owner = owner.base
    if (type(owner) is bytes and array.flags.c_contiguous
            and len(owner) == array.nbytes):
        return owner
    return None


def _owned_bytes(array: np.ndarray) -> bytes:
    """Reuse complete immutable bytes; otherwise preserve the original one copy."""
    owner = _immutable_bytes(array)
    return array.tobytes(order="C") if owner is None else owner


def _floats(array: FloatArray) -> FloatStorage:
    """Adopt already checked float64 values with detached shape metadata."""
    return FloatStorage(_owned_bytes(array), tuple(array.shape))


def _indices(array: IndexArray) -> IndexStorage:
    """Adopt already checked native index values with detached metadata."""
    return IndexStorage(_owned_bytes(array))


def _grouping_headers(source: Grouping) -> Grouping:
    """Detach mutable array metadata; admitted immutable label storage is shared."""
    blocks: list[Block] = []
    for block in source.blocks:
        owned = object.__new__(Block)
        for name, value in (("name", block.name), ("codes", _indices(block.codes).view()),
                            ("counts", _indices(block.counts).view()),
                            ("_levels", block._levels)):
            object.__setattr__(owned, name, value)
        blocks.append(owned)
    result = object.__new__(Grouping)
    object.__setattr__(result, "blocks", tuple(blocks))
    object.__setattr__(result, "positions", _indices(source.positions).view())
    return result


@dataclass(frozen=True, eq=False)
class NumericFit:
    """One immutable numerical representation of final-filtered fitting rows.

    This internal record is built only by from_owned_parts after actual sample
    admission. It deliberately contains no validity flag: changing numerical
    conditions and final original equations are still checked by their kernels.
    The grouping is an operation-private canonical view on the admitted snapshot's
    code bytes; no code or label is recomputed or selected by this representation.
    """

    _response: FloatStorage
    _design: FloatStorage
    _offset: FloatStorage
    _weights: FloatStorage
    _positions: IndexStorage
    _active: IndexStorage
    _scale: FloatStorage
    _grouping: Grouping
    columns: tuple[str, ...]
    omega: float
    frequency: bool

    @classmethod
    def from_owned_parts(cls, parts: NumericParts) -> NumericFit:
        """Transfer admitted numeric ownership; do not rerun a sample constructor."""
        return cls(_floats(parts.response), _floats(parts.design),
                   _floats(parts.offset), _floats(parts.weights),
                   _indices(parts.positions), _indices(parts.active),
                   _floats(parts.scale), _grouping_headers(parts.grouping), tuple(parts.columns),
                   parts.omega, parts.frequency)

    @property
    def response(self) -> FloatArray:
        """Detached read-only header on the immutable transformed response."""
        return self._response.view()

    @property
    def design(self) -> FloatArray:
        """Detached read-only header on the active scaled slope matrix."""
        return self._design.view()

    @property
    def offset(self) -> FloatArray:
        """Detached read-only header on the once-transformed offset."""
        return self._offset.view()

    @property
    def weights(self) -> FloatArray:
        """Raw observation weights, without renormalization or double application."""
        return self._weights.view()

    @property
    def positions(self) -> IndexArray:
        """Original final-row positions, never display-label identity."""
        return self._positions.view()

    @property
    def active(self) -> IndexArray:
        """Original active-column positions, without reselection."""
        return self._active.view()

    @property
    def scale(self) -> FloatArray:
        """The original active-column scales."""
        return self._scale.view()

    @property
    def grouping(self) -> Grouping:
        """Fresh code/position headers; no outward shape mutation reaches the owner."""
        return _grouping_headers(self._grouping)

    @property
    def count(self) -> int:
        """Number of final physical fitting rows."""
        return self._response.shape[0]

    @property
    def n_slopes(self) -> int:
        """Number of actual active slopes."""
        return self._design.shape[1]


@dataclass(frozen=True)
class NumericParts:
    """Already admitted internal components, not another public validation API."""

    response: FloatArray
    design: FloatArray
    offset: FloatArray
    weights: FloatArray
    positions: IndexArray
    active: IndexArray
    scale: FloatArray
    grouping: Grouping
    columns: tuple[str, ...]
    omega: float
    frequency: bool
