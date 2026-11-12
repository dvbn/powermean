"""Owned per-fit cancellation input and fresh in-flight identified equations.

The shared Newton bridge uses admission.geometry.design and its unchanged T/P;
its independently assessed points use CapturedFit.equations. No optimizer or row
engine is copied here. The bridge's exact PowerVariance(2)/unit-multiplier domain
remains separate from these general equation actions. No inference is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.collinearity_records import DiagnosticControls
from powermean.fe._cancel_storage import BuilderRecord, FloatStorage, finite_vector
from powermean.fe.admission import AdmissionInputs, AdmittedGeometry, admit_geometry
from powermean.fe.admission_budget import reserve_admission
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_storage import _build_record
from powermean.model.variance import FloatArray, Variance
from powermean.systems.identified import IdentifiedState, _identified_equations


class FitRows(NamedTuple):
    """Current transformed rows; no filtering, transformation or recentering."""

    z: ArrayLike
    centered_offset: ArrayLike | None
    obs_weights: ArrayLike | None


@dataclass(frozen=True, eq=False, init=False)
class CapturedFit(BuilderRecord):
    """Immutable owned fit problem, with structural work absent from point calls."""

    __slots__ = ("_offset", "_weights", "_z", "admission")

    admission: AdmittedGeometry
    _z: FloatStorage
    _offset: FloatStorage
    _weights: FloatStorage | None

    @property
    def z(self) -> FloatArray:
        """All transformed nonnegative outcomes in input row order."""
        return self._z.view()

    @property
    def centered_offset(self) -> FloatArray:
        """Owned already-centered transformed offset, added once by the core."""
        return self._offset.view()

    @property
    def obs_weights(self) -> FloatArray | None:
        """Owned positive multipliers, without new statistical semantics."""
        return None if self._weights is None else self._weights.view()

    def equations(self, theta: ArrayLike, *, variance: Variance) -> IdentifiedState:
        """Fresh point, callback, weights/curvature and actions; shared structure.

        No state, variance object or callback result is retained by this capture.
        Mutable variance implementations are consulted on every evaluation.
        Changed sample/outcome/offset/statistical weights require a fresh capture.
        """
        return _identified_equations(
            self.admission.geometry,
            self.z,
            theta,
            evidence=self.admission.check,
            variance=variance,
            centered_offset=self.centered_offset,
            obs_weights=self.obs_weights,
        )


def _rows(values: ArrayLike, n: int, name: str) -> FloatStorage:
    if not isinstance(values, np.ndarray) or values.shape != (n,):
        raise ValueError(f"{name} must be an ndarray with one value per row")
    return FloatStorage.store(finite_vector(values, n, name))


def capture_fit(
    geometry: NumericalGeometry,
    inputs: AdmissionInputs,
    rows: FitRows,
    *,
    controls: DiagnosticControls,
    mode: str,
    max_bytes: int,
) -> CapturedFit:
    """Fresh external fit entry: reserve, own rows, admit structure exactly once.

    Invoke inside the complete-fit timer. No supplied admitted object/flag is
    accepted. Later caller mutations cannot change the in-flight fit problem.
    Point failures never alter admission or an earlier accepted point state.
    """
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must come from numerical_geometry")
    reserve_admission(geometry.design, max_bytes)
    if type(rows) is not FitRows:
        raise TypeError("rows must be a FitRows request")
    n = geometry.structure.n_rows
    z = _rows(rows.z, n, "z")
    if np.any(z.view() < 0):
        raise ValueError("transformed z must be nonnegative")
    offset = _rows(
        np.zeros(n) if rows.centered_offset is None else rows.centered_offset,
        n,
        "centered_offset",
    )
    weights = (
        None if rows.obs_weights is None else _rows(rows.obs_weights, n, "obs_weights")
    )
    if weights is not None and np.any(weights.view() <= 0):
        raise ValueError(
            "obs_weights must be positive; zero-weight policy is unresolved"
        )
    admission = admit_geometry(
        geometry, inputs, controls=controls, mode=mode, max_bytes=max_bytes
    )
    return cast(
        CapturedFit,
        _build_record(
            CapturedFit,
            admission=admission,
            _z=z,
            _offset=offset,
            _weights=weights,
        ),
    )
