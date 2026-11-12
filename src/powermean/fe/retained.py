"""Full-column implicit ``B = M_C [X, F_R]`` products.

Only centered slopes (n by K), codes and segments are stored. Retained indicators
are gathered and reduced; no row-by-level matrix is formed, even temporarily.
Columns are slopes then every level of every retained block in package code order.
Identification, reference levels and rank decisions belong to the next layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe._cancel_storage import (
    BuilderRecord,
    FloatStorage,
    finite_product,
    finite_vector,
)
from powermean.fe.cancel import (
    CancelStructure,
    center_sorted,
    to_original,
    to_sorted,
)
from powermean.model.variance import FloatArray, _finite_array


@dataclass(frozen=True, eq=False, init=False)
class CenteredDesign(BuilderRecord):
    """Immutable full-column design built by :func:`centered_design`.

    All four product methods check new vectors afresh. Sorted methods are for
    callers already using the structure's segment order; unsorted methods use
    input positional order. The structure may be shared across fresh points.
    """

    __slots__ = ("_block_starts", "_n_columns", "_slopes", "structure")

    structure: CancelStructure
    _slopes: FloatStorage
    _block_starts: tuple[int, ...]
    _n_columns: int

    @property
    def centered_slopes(self) -> FloatArray:
        """The stored ``M_C X``, with rows sorted by cancelled group."""
        return self._slopes.view()

    @property
    def n_slopes(self) -> int:
        """Number of slope columns, possibly zero."""
        return self._slopes.shape[1]

    @property
    def block_starts(self) -> tuple[int, ...]:
        """First full-column coordinate of each retained block."""
        return self._block_starts

    @property
    def n_columns(self) -> int:
        """K plus all retained levels, including annihilated or collinear columns."""
        return self._n_columns

    def checked_coefficients(self, values: ArrayLike) -> FloatArray:
        """Check a fresh full-column coefficient vector."""
        return finite_vector(values, self.n_columns, "coefficient vector")

    def matvec(self, values: ArrayLike) -> FloatArray:
        """``B v`` at input positional rows."""
        return to_original(self.structure, self.matvec_sorted(values))

    def rmatvec(self, values: ArrayLike) -> FloatArray:
        """``B' u`` for a row vector in input positional order."""
        rows = finite_vector(values, self.structure.n_rows, "row vector")
        return self.rmatvec_sorted(to_sorted(self.structure, rows))

    def matvec_sorted(self, values: ArrayLike) -> FloatArray:
        """``B v`` in sorted order, by gathers and ordinary group centering."""
        coefficients = self.checked_coefficients(values)
        with np.errstate(over="raise", invalid="raise"):
            product = self.centered_slopes @ coefficients[: self.n_slopes]
            if self.structure.block_sizes:
                gathered = np.zeros(self.structure.n_rows)
                for codes, start, size in zip(
                    self.structure.retained_sorted_codes,
                    self.block_starts,
                    self.structure.block_sizes,
                    strict=True,
                ):
                    gathered += coefficients[start : start + size][codes]
                product += center_sorted(self.structure, gathered)
        return finite_product(product)

    def rmatvec_sorted(self, values: ArrayLike) -> FloatArray:
        """``B' u`` for sorted rows, with centered retained-block reductions."""
        rows = finite_vector(values, self.structure.n_rows, "sorted row vector")
        with np.errstate(over="raise", invalid="raise"):
            parts = [self.centered_slopes.T @ rows]
            if self.structure.block_sizes:
                centered = center_sorted(self.structure, rows)
                for codes, size in zip(
                    self.structure.retained_sorted_codes,
                    self.structure.block_sizes,
                    strict=True,
                ):
                    parts.append(np.bincount(codes, weights=centered, minlength=size))
        return finite_product(np.concatenate(parts))


def centered_design(structure: CancelStructure, slopes: ArrayLike) -> CenteredDesign:
    """Build ``M_C [X, F_R]`` without constructing any retained indicators.

    Parameters
    ----------
    structure : CancelStructure
        Validated owned layout from ``cancel_structure``.
    slopes : array_like
        Finite real n by K regressors in input row order. K may be zero if there
        are retained columns. No intercept, level or collinear column is removed.

    Returns
    -------
    CenteredDesign
        Owned centered slopes and a shared immutable layout.
    """
    if type(structure) is not CancelStructure:
        raise TypeError("structure must come from cancel_structure")
    array = _finite_array(slopes, "slopes")
    if array.ndim != 2 or array.shape[0] != structure.n_rows:
        raise ValueError(f"slopes must have shape ({structure.n_rows}, K)")
    size = array.shape[1]
    starts: list[int] = []
    for count in structure.block_sizes:
        starts.append(size)
        size += count
    if size == 0:
        raise ValueError("the design has no column")
    centered = center_sorted(structure, to_sorted(structure, array))
    record = object.__new__(CenteredDesign)
    object.__setattr__(record, "structure", structure)
    object.__setattr__(record, "_slopes", FloatStorage.store(centered))
    object.__setattr__(record, "_block_starts", tuple(starts))
    object.__setattr__(record, "_n_columns", size)
    return record
