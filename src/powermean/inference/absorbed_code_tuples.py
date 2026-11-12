"""Freshly constructed, byte-bound reuse of absorbed nuisance code tuples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.fe.certificate_basis import CertificateBasis
from powermean.model.order import IndexArray


def _code_tuples(basis: CertificateBasis) -> tuple[IndexArray, IndexArray]:
    """The distinct kept-position tuples of the rows (``-1`` at a reference), inverse.

    The inverse maps every row to its tuple.
    """
    columns = [
        position[block.codes]
        for block, position in zip(basis.grouping.blocks, basis.positions, strict=True)
    ]
    stacked = np.column_stack(columns)
    distinct, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return (
        frozen_index(np.asarray(distinct, dtype=np.intp)).reshape(distinct.shape),
        frozen_index(np.asarray(inverse, dtype=np.intp).reshape(-1)),
    )


@dataclass(frozen=True, init=False)
class CachedCodeTuples:
    """Compute once from the actual basis; never accept caller-asserted tuples."""

    _binding: tuple[tuple[str, tuple[int, ...], bytes], ...]
    _distinct: bytes
    _shape: tuple[int, ...]
    _inverse: bytes

    def __init__(self, basis: CertificateBasis) -> None:
        """Own actual streamed code-tuple data in immutable bytes."""
        distinct, inverse = _code_tuples(basis)
        object.__setattr__(self, "_binding", _binding(basis))
        object.__setattr__(self, "_distinct", distinct.tobytes())
        object.__setattr__(self, "_shape", distinct.shape)
        object.__setattr__(self, "_inverse", inverse.tobytes())

    def for_basis(self, basis: CertificateBasis) -> tuple[IndexArray, IndexArray]:
        """Fresh array headers only after current basis byte equality."""
        if _binding(basis) != self._binding:
            raise ValueError("code tuples belong to another nuisance basis")
        distinct = np.frombuffer(self._distinct, dtype=np.intp).reshape(self._shape)
        inverse = np.frombuffer(self._inverse, dtype=np.intp)
        return distinct, inverse


def _binding(basis: CertificateBasis) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    """Actual codes and chosen nuisance coordinates, without a hash/flag cache."""
    arrays = (*[b.codes for b in basis.grouping.blocks], *basis.positions)
    return tuple((a.dtype.str, a.shape, a.tobytes()) for a in arrays)
