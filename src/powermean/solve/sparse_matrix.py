"""Passive canonical CSC storage shared by explicit and implicit FE geometry.

Design: 02 Computation, Section 7. This is the existing geometry record, separated
without changing its fields or column-order semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray


@dataclass(frozen=True, eq=False)
class SparseMatrix:
    """An owned canonical CSC matrix: indptr, sorted indices and data."""

    shape: tuple[int, int]
    indptr: IndexArray
    indices: IndexArray
    data: FloatArray

    @property
    def columns(self) -> IndexArray:
        """The column of every stored entry, in storage order."""
        found: IndexArray = np.repeat(
            np.arange(self.shape[1], dtype=np.intp), np.diff(self.indptr)
        )
        return found
