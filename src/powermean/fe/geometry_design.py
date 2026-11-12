"""Owned coordinate maps B_id = BT.

T selects original slope columns and a retained complement. Omitted slopes are
set to zero. A kept coefficient is an original slope in this normalization; it
is separately identified in the unrestricted model only if its nullspace row
vanishes. T need not be perpendicular to ker(B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe._cancel_storage import (
    BuilderRecord,
    FloatStorage,
    IndexArray,
    IndexStorage,
    finite_product,
    finite_vector,
)
from powermean.fe.cancel import CancelStructure, to_original, to_sorted
from powermean.fe.geometry_gram import GeometryBudget, GeometryPasses
from powermean.fe.geometry_storage import BoolArray, BoolStorage
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray


@dataclass(frozen=True, eq=False, init=False)
class NumericalGeometry(BuilderRecord):
    """Owned ``B_id = B T`` and omitted-slope-zero normalization.

    slope_alias_matrix has shape (d_X, K): normalized beta = alias @ beta_X.
    slope_identified_mask has length K and marks numerical-nullspace rows
    with squared norm <= gram_rtol. It distinguishes a separately identified
    original coefficient from a normalized combination of dependent slopes.
    """

    __slots__ = (
        "_annihilated_blocks",
        "_annihilated_levels",
        "_basis_matrix",
        "_eigenvalues",
        "_gram",
        "_identified_gram",
        "_kept_slope_indices",
        "_null_basis",
        "_omitted_slope_indices",
        "_original_column_map",
        "_reduction_matrix",
        "_retained_eigenvalues",
        "_slope_alias_matrix",
        "_slope_identified_mask",
        "block_deficiencies",
        "block_ranks",
        "budget",
        "cross_block_redundancy",
        "design",
        "gap_min",
        "gap_ratio",
        "gram_rtol",
        "max_bytes",
        "nuisance_basis_kind",
        "passes",
        "retained_bytes",
        "retained_gap_ratio",
        "slope_omission_reasons",
        "threshold",
    )

    design: CenteredDesign
    _gram: FloatStorage
    _eigenvalues: FloatStorage
    _retained_eigenvalues: FloatStorage
    _null_basis: FloatStorage
    _basis_matrix: FloatStorage
    _reduction_matrix: FloatStorage
    _identified_gram: FloatStorage
    _kept_slope_indices: IndexStorage
    _omitted_slope_indices: IndexStorage
    slope_omission_reasons: tuple[str | None, ...]
    _original_column_map: IndexStorage
    _slope_alias_matrix: FloatStorage
    _slope_identified_mask: BoolStorage
    block_ranks: tuple[int, ...]
    block_deficiencies: tuple[int, ...]
    _annihilated_blocks: BoolStorage
    _annihilated_levels: tuple[BoolStorage, ...]
    cross_block_redundancy: int
    gram_rtol: float
    gap_min: float
    threshold: float
    gap_ratio: float
    retained_gap_ratio: float
    nuisance_basis_kind: str
    budget: GeometryBudget
    passes: GeometryPasses
    max_bytes: int
    retained_bytes: int

    @property
    def gram(self) -> FloatArray:
        """Detached view of the owned gram."""
        return self._gram.view()

    @property
    def eigenvalues(self) -> FloatArray:
        """Detached view of the owned eigenvalues."""
        return self._eigenvalues.view()

    @property
    def retained_eigenvalues(self) -> FloatArray:
        """Detached view of the owned retained eigenvalues."""
        return self._retained_eigenvalues.view()

    @property
    def null_basis(self) -> FloatArray:
        """Detached view of the owned null basis."""
        return self._null_basis.view()

    @property
    def basis_matrix(self) -> FloatArray:
        """Detached view of the owned basis matrix."""
        return self._basis_matrix.view()

    @property
    def reduction_matrix(self) -> FloatArray:
        """Detached view of the owned reduction matrix."""
        return self._reduction_matrix.view()

    @property
    def identified_gram(self) -> FloatArray:
        """Detached view of the owned identified gram."""
        return self._identified_gram.view()

    @property
    def kept_slope_indices(self) -> IndexArray:
        """Detached view of the owned kept slope indices."""
        return self._kept_slope_indices.view()

    @property
    def omitted_slope_indices(self) -> IndexArray:
        """Detached view of the owned omitted slope indices."""
        return self._omitted_slope_indices.view()

    @property
    def original_column_map(self) -> IndexArray:
        """Detached view of the owned original column map."""
        return self._original_column_map.view()

    @property
    def slope_alias_matrix(self) -> FloatArray:
        """Detached view of the owned slope alias matrix."""
        return self._slope_alias_matrix.view()

    @property
    def slope_identified_mask(self) -> BoolArray:
        """Detached view of the owned slope identified mask."""
        return self._slope_identified_mask.view()

    @property
    def annihilated_blocks(self) -> BoolArray:
        """Detached view of the owned annihilated blocks."""
        return self._annihilated_blocks.view()

    @property
    def annihilated_levels(self) -> tuple[BoolArray, ...]:
        """Detached view of the owned annihilated levels."""
        return tuple(values.view() for values in self._annihilated_levels)

    @property
    def identification_status(self) -> Literal["numerical_candidate"]:
        """A numerical candidate, never structural acceptance."""
        return "numerical_candidate"

    @property
    def structure(self) -> CancelStructure:
        """Underlying cancelled/retained code and segment structure."""
        return self.design.structure

    @property
    def n_slopes_full(self) -> int:
        """Number of input slope columns ``K`` in ``design``."""
        return self.design.n_slopes

    @property
    def n_slopes_identified(self) -> int:
        """Number of identified slope columns ``d_X``."""
        return int(self.kept_slope_indices.size)

    @property
    def n_retained_full(self) -> int:
        """Total represented retained levels ``L = sum_b L_b``."""
        return sum(self.design.structure.block_sizes)

    @property
    def n_retained_identified(self) -> int:
        """Identified retained nuisance dimension ``d_R = rank(M_C F_R)``."""
        return int(self.basis_matrix.shape[1] - self.kept_slope_indices.size)

    @property
    def n_retained_redundant(self) -> int:
        """Total redundant retained directions ``L - d_R`` removed by identification."""
        return self.n_retained_full - self.n_retained_identified

    @property
    def n_columns_full(self) -> int:
        """Full represented column count ``p = K + L`` of ``B``."""
        return self.design.n_columns

    @property
    def n_identified(self) -> int:
        """Identified coordinate dimension ``d = d_X + d_R = rank(B)``."""
        return int(self.basis_matrix.shape[1])

    @property
    def n_columns(self) -> int:
        """Active column count ``d`` of the identified operator ``B_id = B T``."""
        return self.n_identified

    @property
    def nullity(self) -> int:
        """Total rank deficiency ``p - d`` of the full-column representation."""
        return self.n_columns_full - self.n_identified

    @property
    def omitted_slopes(self) -> tuple[tuple[int, str], ...]:
        """Pairs ``(slope_index, reason)`` for every omitted original slope column."""
        reasons = self.slope_omission_reasons
        return tuple(
            (int(idx), str(reasons[int(idx)])) for idx in self.omitted_slope_indices
        )

    @property
    def annihilated_block_indices(self) -> tuple[int, ...]:
        """Indices of retained blocks completely annihilated by ``M_C``."""
        return tuple(int(idx) for idx in np.flatnonzero(self.annihilated_blocks))

    @staticmethod
    def _product(matrix: FloatArray, vector: FloatArray) -> FloatArray:
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(matrix @ vector)

    def checked_coefficients(self, v: ArrayLike) -> FloatArray:
        """Validate a fresh finite vector of length d."""
        return finite_vector(v, self.n_identified, "identified coefficient vector")

    def checked_full_coefficients(self, v_full: ArrayLike) -> FloatArray:
        """Validate a fresh finite vector of length p."""
        return self.design.checked_coefficients(v_full)

    def expand(self, v: ArrayLike) -> FloatArray:
        """Map identified coordinates ``v`` to full coordinates ``T v``."""
        return self._product(self.basis_matrix, self.checked_coefficients(v))

    def expand_transpose(self, u_full: ArrayLike) -> FloatArray:
        """Map a full covector of length p to T' u of length d."""
        return self._product(
            self.basis_matrix.T, self.checked_full_coefficients(u_full)
        )

    def reduce(self, v_full: ArrayLike) -> FloatArray:
        """Map full coordinates ``v_full`` to equivalent identified coordinates."""
        return self._product(
            self.reduction_matrix, self.checked_full_coefficients(v_full)
        )

    def reduce_transpose(self, v: ArrayLike) -> FloatArray:
        """Map an identified covector to ``P' v`` in full coefficient space."""
        return self._product(self.reduction_matrix.T, self.checked_coefficients(v))

    def project_full(self, v_full: ArrayLike) -> FloatArray:
        """Apply the oblique coefficient projection T P."""
        return self.expand(self.reduce(v_full))

    def identified_slopes(self, v: ArrayLike) -> FloatArray:
        """Return slopes under omitted-original-coefficients = 0 normalization.

        For an arbitrary full beta these are slope_alias_matrix @ beta_X.
        A normalized kept slope equals its unrestricted original coefficient
        only where slope_identified_mask is true; dependent omitted slopes
        can otherwise contribute to it. Changing nuisance basis preserves it.
        """
        return self.checked_coefficients(v)[: self.n_slopes_identified].copy()

    def full_slopes(self, v: ArrayLike, *, omitted_fill: float = np.nan) -> FloatArray:
        """Place normalized slopes at kept indices and ``omitted_fill`` elsewhere."""
        coeffs = self.checked_coefficients(v)
        result = np.full(self.n_slopes_full, float(omitted_fill), dtype=np.float64)
        kept = self.kept_slope_indices
        if kept.size:
            result[kept] = coeffs[: kept.size]
        return result

    def matvec_sorted(self, v: ArrayLike) -> FloatArray:
        """``B_id v = B (T v)`` in sorted segment row order, checking a fresh ``v``."""
        return self.design.matvec_sorted(self.expand(v))

    def rmatvec_sorted(self, u: ArrayLike) -> FloatArray:
        """``B_id' u = T' (B' u)`` for ``u`` in sorted segment row order."""
        return self.expand_transpose(self.design.rmatvec_sorted(u))

    def matvec(self, v: ArrayLike) -> FloatArray:
        """``B_id v`` at the original positional rows."""
        product = self.matvec_sorted(self.checked_coefficients(v))
        return to_original(self.structure, product)

    def rmatvec(self, u: ArrayLike) -> FloatArray:
        """``B_id' u`` for ``u`` given at the original positional rows."""
        values = finite_vector(u, self.structure.n_rows, "row vector")
        return self.rmatvec_sorted(to_sorted(self.structure, values))
