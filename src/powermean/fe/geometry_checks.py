"""Fresh original-column evidence for a numerical geometry candidate.

This is a veto at the caller's explicit controls, not a new rank algorithm.
Original-space null/reconstruction checks use gram_rtol directly, without the
Gram squaring or a looser cutoff. Exact raw group-constancy is checked from fresh
raw columns. As in data.collinearity_geometry, disagreement refuses rather than
changing the model. General structural admission still needs a separately
specified matrix-free direct-rank procedure; the M1 dense routines are not called.

A geometry on the exact quotient's pivot columns binds external raw inputs
afresh via :func:`_bound_raw` and verifies its immutable fit-owned construction
proof (:class:`SelectedProof`, recorded at construction on immutable
:class:`FloatStorage` bytes): discrete quotient consistency, bit-identical
identified Gram slice and exact witness maps, certified spectral/gap separation,
and operational reconstruction/null bounds covering witness conversion, gather
accumulation, and group centering at ``gram_rtol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.collinearity_numeric import norm
from powermean.data.sample_fe_source import same_bits
from powermean.fe._cancel_storage import BuilderRecord, IndexArray
from powermean.fe.admission_incidence import QuotientRank
from powermean.fe.cancel import center_sorted, to_sorted
from powermean.fe.geometry import _witness_maps
from powermean.fe.geometry_design import NumericalGeometry, SelectedProof
from powermean.fe.geometry_errors import GeometryAdmissionError
from powermean.fe.geometry_gram import reserve_geometry
from powermean.fe.geometry_rank import CLEAR_MARGIN, _original_constants, _spectral_rank
from powermean.fe.geometry_storage import BoolArray, BoolStorage, _build_record
from powermean.fe.groups import _positions
from powermean.model.variance import FloatArray, _finite_array

if TYPE_CHECKING:
    from powermean.data.collinearity_records import CollinearityAnalysis


class GeometryInputs(NamedTuple):
    """Untrusted raw slopes, row identities and original slope names, in input order.

    Raw slopes and positions must be ndarrays of the declared shapes.
    Constructing this request validates nothing. Every check/evaluation reads its
    current values afresh; no caller-supplied success flag exists. ``analysis`` is
    the fit's one collinearity analysis of these columns in their original unit, which
    decided the kept slopes (the fixed-effect dispatch); the admission of the slopes
    checks the geometry against it instead of running it again, or runs it when none
    is given.
    """

    raw_slopes: ArrayLike
    positions: ArrayLike
    source_columns: tuple[str, ...]
    analysis: CollinearityAnalysis | None = None

    @property
    def original_slopes(self) -> FloatArray | None:
        """Raw columns before policy-driven replacements, if available."""
        if self.analysis is None:
            return np.asarray(self.raw_slopes)
        problem = self.analysis.problem
        return None if problem is None else problem.design


@dataclass(frozen=True, eq=False, init=False)
class GeometryCheck(BuilderRecord):
    """Owned evidence of fresh veto checks, without structural acceptance."""

    __slots__ = (
        "_raw_constant",
        "max_null_defect",
        "max_reconstruction_defect",
        "n_center_sorted_calls",
        "n_eigh_calls",
        "n_matvec_sorted_calls",
        "n_raw_group_segments",
        "source_columns",
    )

    _raw_constant: BoolStorage
    source_columns: tuple[str, ...]
    max_null_defect: float
    max_reconstruction_defect: float
    n_matvec_sorted_calls: int
    n_eigh_calls: int
    n_center_sorted_calls: int
    n_raw_group_segments: int

    @property
    def raw_constant(self) -> BoolArray:
        """Raw exact group-constancy flags, freshly bound to the centered design."""
        return self._raw_constant.view()

    @property
    def identification_status(self) -> Literal["original_space_checked_candidate"]:
        """Successful veto checks leave a numerical candidate only."""
        return "original_space_checked_candidate"


def _bound_raw(geometry: NumericalGeometry, inputs: GeometryInputs) -> BoolArray:
    design = geometry.design
    structure = design.structure
    request = inputs.raw_slopes
    if not isinstance(request, np.ndarray) or request.dtype.kind not in "biuf":
        raise TypeError("raw slopes must be a real numeric ndarray")
    if request.shape != (structure.n_rows, design.n_slopes):
        raise ValueError("raw slopes must have shape (n_rows, n_slopes)")
    row_ids = inputs.positions
    if not isinstance(row_ids, np.ndarray) or row_ids.shape != (structure.n_rows,):
        raise ValueError("positions must be an ndarray with one value per row")
    raw = _finite_array(request, "raw slopes").copy()
    positions = _positions(row_ids, structure.n_rows)
    if not np.array_equal(positions, structure.positions):
        raise GeometryAdmissionError("row_binding", "positions differ from the design")
    names = inputs.source_columns
    if (
        not isinstance(names, tuple)
        or len(names) != design.n_slopes
        or any(type(name) is not str for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "source_columns must be distinct original names, one per slope"
        )
    sorted_raw = to_sorted(structure, raw)
    if not np.array_equal(center_sorted(structure, sorted_raw), design.centered_slopes):
        raise GeometryAdmissionError(
            "raw_binding", "M_C raw slopes differ from the design"
        )
    # Every row equals the first row of its group, for all groups at once: the
    # conjunction over groups of each group's own comparison.
    heads = sorted_raw[structure.starts][structure.sorted_codes]
    constants = np.all(sorted_raw == heads, axis=0)
    for index, reason in geometry.omitted_slopes:
        if reason == "centered_constant" and not constants[index]:
            raise GeometryAdmissionError(
                "raw_constancy", f"slope {index} is not exactly constant within groups"
            )
    omitted = geometry.omitted_slope_indices
    if (
        geometry.selected_columns is not None
        and omitted.size
        and not _original_constants(design, inputs.original_slopes, omitted)
    ):
        raise GeometryAdmissionError(
            "raw_constancy", "original omitted slopes vary within groups"
        )
    return constants


def _relative_defect(
    residual: FloatArray, scale: float, rtol: float, cause: str
) -> float:
    size = norm(residual, cause)
    if not np.isfinite(scale) or scale < 0.0:
        raise FloatingPointError("original-space check scale is not finite nonnegative")
    if size > rtol * scale:
        raise GeometryAdmissionError(
            cause, f"residual norm {size} exceeds {rtol} * {scale}"
        )
    return size / scale if scale > 0.0 else 0.0


def _original_checks(geometry: NumericalGeometry) -> tuple[float, float]:
    design = geometry.design
    p = design.n_columns
    column_norms = np.zeros(p)
    unit = np.zeros(p)
    max_reconstruction = 0.0
    for column in range(p):
        unit[column] = 1.0
        original = design.matvec_sorted(unit)
        column_norms[column] = norm(original, "original column")
        reconstructed = design.matvec_sorted(geometry.project_full(unit))
        defect = _relative_defect(
            original - reconstructed,
            column_norms[column],
            geometry.gram_rtol,
            "original_reconstruction",
        )
        max_reconstruction = max(max_reconstruction, defect)
        unit[column] = 0.0
    max_null = 0.0
    for direction in geometry.null_basis.T:
        scale = float(np.abs(direction) @ column_norms)
        defect = _relative_defect(
            design.matvec_sorted(direction), scale, geometry.gram_rtol, "original_null"
        )
        max_null = max(max_null, defect)
    return max_null, max_reconstruction


def _quotient_consistent(
    geometry: NumericalGeometry, quotient: QuotientRank, columns: IndexArray
) -> bool:
    """Whether ``geometry``'s discrete metadata agrees with ``quotient``."""
    k = geometry.n_slopes_full
    zero = (
        ~np.any(geometry.design.centered_slopes, axis=0)
        if geometry.omitted_slope_indices.size
        else np.zeros(k, dtype=bool)
    )
    kept, omitted = np.flatnonzero(~zero), np.flatnonzero(zero)
    expected = np.concatenate([kept, k + np.asarray(quotient.kept, dtype=np.intp)])
    annihilated = tuple(
        tuple(bool(x) for x in levels) for levels in geometry.annihilated_levels
    )
    return (
        np.array_equal(columns, expected)
        and geometry.n_identified == int(expected.size)
        and np.array_equal(geometry.kept_slope_indices, kept)
        and np.array_equal(geometry.omitted_slope_indices, omitted)
        and geometry.slope_omission_reasons
        == tuple("centered_constant" if v else None for v in zero)
        and geometry.block_ranks == quotient.block_ranks
        and annihilated == quotient.annihilated_levels
        and geometry.cross_block_redundancy == sum(quotient.block_ranks) - quotient.rank
    )


def _proof_valid(geometry: NumericalGeometry, proof: SelectedProof) -> bool:
    """Whether fit-owned :class:`SelectedProof` satisfies separation invariants."""
    scalars = (
        proof.top_high,
        proof.delta_full,
        proof.delta_ret,
        proof.full_min_low,
        proof.ret_min_low,
        proof.selected_min_low,
        proof.max_null_defect,
        proof.max_reconstruction_defect,
    )
    if not all(np.isfinite(val) and val >= 0.0 for val in scalars):
        return False
    req_gap = max(geometry.gap_min, CLEAR_MARGIN)
    return bool(
        proof.top_high > 0.0
        and proof.full_min_low > 0.0
        and proof.selected_min_low >= CLEAR_MARGIN * geometry.gram_rtol * proof.top_high
        and proof.delta_ret <= proof.delta_full
        and geometry.spectrum_provenance == "certified_bounds"
        and geometry.gap_ratio >= req_gap
        and geometry.retained_gap_ratio >= req_gap
    )


def _maps_consistent(
    geometry: NumericalGeometry, quotient: QuotientRank, columns: IndexArray
) -> bool:
    """Whether ``geometry``'s stored maps equal those of ``quotient``'s witnesses."""
    try:
        maps = _witness_maps(
            geometry.n_slopes_full, quotient, columns, geometry.n_columns_full
        )
    except OverflowError:
        return False
    stored = (geometry.basis_matrix, geometry.reduction_matrix, geometry.null_basis)
    return all(same_bits(a, b) for a, b in zip(maps, stored, strict=True))


def _selected_checks(geometry: NumericalGeometry) -> tuple[float, float]:
    """Verify immutable fit-owned selection proof and operator bounds at ``rtol``."""
    held = geometry.exact_quotient
    columns = geometry.selected_columns
    proof = geometry.selected_proof
    if (
        held is None
        or columns is None
        or type(proof) is not SelectedProof
        or not _quotient_consistent(geometry, held[0], columns)
        or not _proof_valid(geometry, proof)
    ):
        raise GeometryAdmissionError("fresh_rank", "Gram candidate rank disagrees")
    fresh = geometry.gram[np.ix_(columns, columns)]
    if not same_bits(fresh, geometry.identified_gram):
        raise GeometryAdmissionError("fresh_rank", "Gram candidate rank disagrees")
    if not _maps_consistent(geometry, held[0], columns):
        raise GeometryAdmissionError(
            "original_reconstruction",
            "maps differ from the exact quotient's witnesses",
        )
    rtol = geometry.gram_rtol
    for value, cause in (
        (proof.max_reconstruction_defect, "original_reconstruction"),
        (proof.max_null_defect, "original_null"),
    ):
        if not value <= rtol:
            raise GeometryAdmissionError(
                cause, f"certified operator bound {value} exceeds {rtol}"
            )
    return proof.max_null_defect, proof.max_reconstruction_defect


def check_geometry(
    geometry: NumericalGeometry, inputs: GeometryInputs, *, max_bytes: int
) -> GeometryCheck:
    """Bind current raw metadata and veto original-space rank disagreement.

    Parameters
    ----------
    geometry : NumericalGeometry
        Owned numerical maps, not structural rank authority.
    inputs : GeometryInputs
        Fresh raw columns and positional/original-column metadata.
    max_bytes : int
        Explicit prospective reservation for all new work, including raw copies.

    Returns
    -------
    GeometryCheck
        Checked candidate evidence. No general structural rank certificate.
    """
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must come from numerical_geometry")
    reserve_geometry(geometry.design, max_bytes)
    if type(inputs) is not GeometryInputs:
        raise TypeError("inputs must be a GeometryInputs request")
    selected = geometry.selected_columns is not None
    with np.errstate(over="raise", invalid="raise"):
        constants = _bound_raw(geometry, inputs)
        if selected:
            max_null, max_reconstruction = _selected_checks(geometry)
        else:
            _, _, zero, _, _ = _spectral_rank(
                geometry.gram, geometry.gram_rtol, geometry.gap_min, "fresh_full_gram"
            )
            if int(np.sum(~zero)) != geometry.n_identified:
                raise GeometryAdmissionError(
                    "fresh_rank", "Gram candidate rank disagrees"
                )
            max_null, max_reconstruction = _original_checks(geometry)
    products = 0 if selected else 2 * geometry.n_columns_full + geometry.nullity
    return cast(
        GeometryCheck,
        _build_record(
            GeometryCheck,
            _raw_constant=BoolStorage.store(constants),
            source_columns=tuple(inputs.source_columns),
            max_null_defect=max_null,
            max_reconstruction_defect=max_reconstruction,
            n_matvec_sorted_calls=products,
            n_eigh_calls=0 if selected else 1,
            n_center_sorted_calls=1
            + (products if geometry.structure.block_sizes else 0),
            n_raw_group_segments=geometry.structure.n_groups,
        ),
    )
