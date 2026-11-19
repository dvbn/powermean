"""Owned coordinate maps B_id = BT (design/03, Section 3.2).

T selects original slope columns and a retained complement. Omitted slopes are
set to zero. A kept coefficient is an original slope in this normalization; it
is separately identified in the unrestricted model only if its nullspace row
vanishes. T need not be perpendicular to ker(B).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

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
from powermean.fe.geometry_gram import GeometryBudget, GeometryPasses, reserve_geometry
from powermean.fe.geometry_storage import BoolArray, BoolStorage
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray

if TYPE_CHECKING:
    from powermean.fe.admission_incidence import QuotientRank
    from powermean.fe.admission_records import AdmissionWork

SpectrumProvenance = Literal["spectral_eigh", "certified_bounds"]
ThresholdKind = Literal["spectral_scale", "gershgorin_upper_bound"]
GapKind = Literal["spectral_ratio", "certified_lower_bound"]


class _ChoSolve(Protocol):
    def __call__(
        self,
        c_and_lower: tuple[FloatArray, bool],
        b: FloatArray,
        *,
        check_finite: bool = True,
    ) -> FloatArray: ...


_cho_solve = cast(_ChoSolve, importlib.import_module("scipy.linalg").cho_solve)


@dataclass(frozen=True, eq=False, init=False)
class NullProjection(BuilderRecord):
    """Immutable small-side Cholesky factor for null-orthogonal projection.

    Holds ``r``-by-``m`` non-annihilated dependent witnesses ``W`` and the lower
    Cholesky factor of ``I_r + W W'`` when ``r_side`` is true (``r <= m``) or of
    ``I_m + W' W`` when ``r_side`` is false (``m < r``).
    """

    __slots__ = ("_chol", "_dep_indices", "_kept_indices", "_witnesses", "r_side")

    _kept_indices: IndexStorage
    _dep_indices: IndexStorage
    _witnesses: FloatStorage
    _chol: FloatStorage
    r_side: bool

    @property
    def kept_indices(self) -> IndexArray:
        """Detached view of the full kept retained column indices ``k + J``."""
        return self._kept_indices.view()

    @property
    def dep_indices(self) -> IndexArray:
        """Detached view of the full dependent retained column indices ``k + D``."""
        return self._dep_indices.view()

    @property
    def witnesses(self) -> FloatArray:
        """Detached view of the ``r``-by-``m`` float64 witness matrix ``W``."""
        return self._witnesses.view()

    @property
    def chol(self) -> FloatArray:
        """Detached view of the ``min(r, m)``-square lower Cholesky factor."""
        return self._chol.view()


@dataclass(frozen=True, eq=False, init=False)
class SelectedProof(BuilderRecord):
    """Immutable fit-owned certificate of selected-columns Gram/operator bounds."""

    __slots__ = (
        "delta_full",
        "delta_ret",
        "full_min_low",
        "max_null_defect",
        "max_reconstruction_defect",
        "ret_min_low",
        "selected_min_low",
        "top_high",
    )

    top_high: float
    delta_full: float
    delta_ret: float
    full_min_low: float
    ret_min_low: float
    selected_min_low: float
    max_null_defect: float
    max_reconstruction_defect: float


@dataclass(frozen=True, eq=False, init=False)
class NumericalGeometry(BuilderRecord):
    """Owned ``B_id = B T`` and omitted-slope-zero normalization.

    slope_alias_matrix has shape (d_X, K): normalized beta = alias @ beta_X.
    slope_identified_mask has length K and marks numerical-nullspace rows
    with squared norm <= gram_rtol. It distinguishes a separately identified
    original coefficient from a normalized combination of dependent slopes.
    The selected-columns construction also holds the exact quotient it is built
    from and the columns that T selects. A spectral construction holds neither,
    or, where it decided after the selected one declined, the exact quotient that
    construction had already computed for the same structure.
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
        "_null_projection",
        "_omitted_slope_indices",
        "_original_column_map",
        "_quotient",
        "_reduction_matrix",
        "_retained_eigenvalues",
        "_selected_columns",
        "_selected_proof",
        "_slope_alias_matrix",
        "_slope_identified_mask",
        "block_deficiencies",
        "block_ranks",
        "budget",
        "cross_block_redundancy",
        "design",
        "gap_kind",
        "gap_min",
        "gap_ratio",
        "gram_rtol",
        "max_bytes",
        "nuisance_basis_kind",
        "passes",
        "retained_bytes",
        "retained_gap_ratio",
        "slope_omission_reasons",
        "spectrum_provenance",
        "threshold",
        "threshold_kind",
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
    spectrum_provenance: SpectrumProvenance
    threshold_kind: ThresholdKind
    gap_kind: GapKind
    budget: GeometryBudget
    passes: GeometryPasses
    max_bytes: int
    retained_bytes: int
    _quotient: tuple[QuotientRank, AdmissionWork] | None
    _selected_columns: IndexStorage | None
    _null_projection: NullProjection | None
    _selected_proof: SelectedProof | None

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
    def selected_columns(self) -> IndexArray | None:
        """The full columns that T selects, in coordinate order; None if T is not."""
        found = self._selected_columns
        return None if found is None else found.view()

    @property
    def exact_quotient(self) -> tuple[QuotientRank, AdmissionWork] | None:
        """The exact quotient of this geometry's structure, with its work; or None."""
        return self._quotient

    @property
    def selected_proof(self) -> SelectedProof | None:
        """Certified Gram and operator bounds for ``exact_selection``, or None."""
        return self._selected_proof

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
        """Map identified coordinates ``v`` to full coordinates ``T v``.

        Where T selects columns, ``v`` is placed at them: the same values.
        """
        values = self.checked_coefficients(v)
        columns = self.selected_columns
        if columns is None:
            return self._product(self.basis_matrix, values)
        full = np.zeros(self.n_columns_full, dtype=np.float64)
        full[columns] = values
        return full

    def expand_transpose(self, u_full: ArrayLike) -> FloatArray:
        """Map a full covector of length p to T' u of length d.

        Where T selects columns, those entries of ``u``: the same values.
        """
        values = self.checked_full_coefficients(u_full)
        columns = self.selected_columns
        if columns is None:
            return self._product(self.basis_matrix.T, values)
        return values[columns]

    def normalization_workspace(self) -> tuple[int, int]:
        """Incremental ``(bytes, work_units)`` for :meth:`normalized_full`."""
        proj = self._null_projection
        if proj is None:
            return 0, 0
        w_shape = proj._witnesses.shape
        r, m = w_shape[0], w_shape[1]
        s_min = proj._chol.shape[0]
        p = self.n_columns_full
        n_bytes = 8 * (4 * p + 4 * r + 4 * m + 4 * s_min)
        work = 128 * (p + 2 * r * m + 2 * s_min * s_min + r + m)
        return n_bytes, work

    def normalized_full(self, v: ArrayLike) -> FloatArray:
        """``T v``, its retained part taken orthogonal to the null space.

        A retained part is identified only up to the null space. The spectral bases
        hold it orthogonal to that space, their columns spanning its complement;
        on a selection ``T v`` is zero outside the selected columns, so the
        fit-owned small-side Cholesky factor removes the active null component in
        ``O(p + r m + min(r, m)^2)`` with 1D temporaries only. Unit null columns
        correspond to annihilated levels already zero in ``T v``.
        """
        proj = self._null_projection
        if self._selected_columns is None or proj is None:
            return self.expand(v)
        reserve_geometry(self.design, self.max_bytes)
        values = self.checked_coefficients(v)
        k = self.n_slopes_identified
        v_j = values[k:]
        w_mat, chol = proj.witnesses, proj.chol
        # Immutable storage is C-order. Its transpose is the same upper factor
        # in Fortran order, so POTRS needs no square layout-conversion buffer.
        # Both this fit-owned factor and the fresh coefficient vector are finite.
        with np.errstate(over="raise", invalid="raise"):
            if proj.r_side:
                u_r = np.asarray(
                    _cho_solve((chol.T, False), v_j, check_finite=False),
                    dtype=np.float64,
                )
                kept_vals = u_r
                dep_vals = w_mat.T @ u_r
            else:
                rhs_m = w_mat.T @ v_j
                rhs_m = finite_product(rhs_m)
                u_m = np.asarray(
                    _cho_solve((chol.T, False), rhs_m, check_finite=False),
                    dtype=np.float64,
                )
                kept_vals = v_j - w_mat @ u_m
                dep_vals = u_m
            full = np.zeros(self.n_columns_full, dtype=np.float64)
            if k:
                full[self.kept_slope_indices] = values[:k]
            full[proj.kept_indices] = kept_vals
            full[proj.dep_indices] = dep_vals
            return finite_product(full)

    def compress(self, full: FloatArray) -> FloatArray:
        """``T' M T`` for a p-by-p ``M``; its principal submatrix where T selects."""
        columns = self.selected_columns
        if columns is None:
            basis = self.basis_matrix
            return finite_product(basis.T @ full @ basis)
        return finite_product(full[np.ix_(columns, columns)])

    def gathered(self, matrix: FloatArray) -> tuple[FloatArray, IndexArray] | None:
        """``T M T'`` for reading at pairs of columns, where T selects; else None.

        Returns ``M`` with a zero row and column appended and the index of each
        full column in it: a selected column's coordinate, or the appended one.
        Entry ``(a, b)`` of :meth:`inflate` is then entry ``(index[a], index[b])``,
        with no p-by-p array formed.
        """
        columns = self.selected_columns
        if columns is None:
            return None
        d = self.n_identified
        padded = np.zeros((d + 1, d + 1), dtype=np.float64)
        padded[:d, :d] = matrix
        index = np.full(self.n_columns_full, d, dtype=np.intp)
        index[columns] = np.arange(d, dtype=np.intp)
        return finite_product(padded), index

    def inflate(self, matrix: FloatArray) -> FloatArray:
        """``T M T'`` for a d-by-d ``M``; at the selected columns where T selects."""
        columns = self.selected_columns
        if columns is None:
            basis = self.basis_matrix
            return finite_product(basis @ matrix @ basis.T)
        p = self.n_columns_full
        full = np.zeros((p, p), dtype=np.float64)
        full[np.ix_(columns, columns)] = matrix
        return finite_product(full)

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
