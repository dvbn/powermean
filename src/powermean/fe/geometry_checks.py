"""Fresh original-column evidence for a numerical geometry candidate.

This is a veto at the caller's explicit controls, not a new rank algorithm.
Original-space null/reconstruction checks use gram_rtol directly, without the
Gram squaring or a looser cutoff. Exact raw group-constancy is checked from fresh
raw columns. As in data.collinearity_geometry, disagreement refuses rather than
changing the model. General structural admission still needs a separately
specified matrix-free direct-rank procedure; the dense routines are not called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.collinearity_numeric import norm
from powermean.fe._cancel_storage import BuilderRecord
from powermean.fe.cancel import center_sorted, to_sorted
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_errors import GeometryAdmissionError
from powermean.fe.geometry_gram import reserve_geometry
from powermean.fe.geometry_rank import _spectral_rank
from powermean.fe.geometry_storage import BoolArray, BoolStorage, _build_record
from powermean.fe.groups import _positions
from powermean.model.variance import FloatArray, _finite_array


class GeometryInputs(NamedTuple):
    """Untrusted raw slopes, row identities and original slope names, in input order.

    Raw slopes and positions must be ndarrays of the declared shapes.
    Constructing this request validates nothing. Every check/evaluation reads its
    current values afresh; no caller-supplied success flag exists.
    """

    raw_slopes: ArrayLike
    positions: ArrayLike
    source_columns: tuple[str, ...]


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
    constants = np.ones(design.n_slopes, dtype=np.bool_)
    for start, stop in zip(
        structure.segments[:-1], structure.segments[1:], strict=True
    ):
        constants &= np.all(sorted_raw[start:stop] == sorted_raw[start], axis=0)
    for index, reason in geometry.omitted_slopes:
        if reason == "centered_constant" and not constants[index]:
            raise GeometryAdmissionError(
                "raw_constancy", f"slope {index} is not exactly constant within groups"
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
    with np.errstate(over="raise", invalid="raise"):
        constants = _bound_raw(geometry, inputs)
        _, _, zero, _, _ = _spectral_rank(
            geometry.gram, geometry.gram_rtol, geometry.gap_min, "fresh_full_gram"
        )
        if int(np.sum(~zero)) != geometry.n_identified:
            raise GeometryAdmissionError("fresh_rank", "Gram candidate rank disagrees")
        max_null, max_reconstruction = _original_checks(geometry)
    return cast(
        GeometryCheck,
        _build_record(
            GeometryCheck,
            _raw_constant=BoolStorage.store(constants),
            source_columns=tuple(inputs.source_columns),
            max_null_defect=max_null,
            max_reconstruction_defect=max_reconstruction,
            n_matvec_sorted_calls=2 * geometry.n_columns_full + geometry.nullity,
            n_eigh_calls=1,
            n_center_sorted_calls=1
            + (
                2 * geometry.n_columns_full + geometry.nullity
                if geometry.structure.block_sizes
                else 0
            ),
            n_raw_group_segments=geometry.structure.n_groups,
        ),
    )
