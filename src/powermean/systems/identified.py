"""Internal coordinate actions for the actual cancelled equations.

S_id=T'S, H_id v=T'H(Tv), H_id'v=T'H'(Tv). The full CancelledState remains
available for original-coordinate inference contributions. This is numerical
preparation; no optimizer, covariance policy or structural admission is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe._cancel_storage import BuilderRecord, FloatStorage, finite_product
from powermean.fe.geometry_checks import GeometryCheck, GeometryInputs, check_geometry
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_storage import _build_record
from powermean.model.variance import FloatArray, Variance
from powermean.systems.cancelled import (
    CancelledAccounting,
    CancelledRows,
    CancelledState,
    cancelled_equations,
)


@dataclass(frozen=True, eq=False, init=False)
class IdentifiedState(BuilderRecord):
    """Owned coordinate state with fresh raw binding and full original moments."""

    __slots__ = ("_score", "_theta", "design", "full_state", "geometry_check")

    design: NumericalGeometry
    geometry_check: GeometryCheck
    full_state: CancelledState
    _theta: FloatStorage
    _score: FloatStorage

    @property
    def theta(self) -> FloatArray:
        """Coordinate coefficients, with immutable backing and detached metadata."""
        return self._theta.view()

    @property
    def theta_full(self) -> FloatArray:
        """Full coefficients at the same current point."""
        return self.full_state.theta

    @property
    def score(self) -> FloatArray:
        """Unnormalized coordinate score T'S."""
        return self._score.view()

    def jac_vec(self, values: ArrayLike) -> FloatArray:
        """T'H(Tv), validating the new coordinate vector."""
        return self.design.expand_transpose(
            self.full_state.jac_vec(self.design.expand(values))
        )

    def jac_t_vec(self, values: ArrayLike) -> FloatArray:
        """T'H'(Tv), preserving the true nonsymmetric transpose."""
        return self.design.expand_transpose(
            self.full_state.jac_t_vec(self.design.expand(values))
        )

    def group_score_dot(self, values: ArrayLike) -> FloatArray:
        """Coordinate group-score contractions in cancelled code order."""
        return self.full_state.group_score_dot(self.design.expand(values))

    def group_jacobian_form(self, left: ArrayLike, right: ArrayLike) -> FloatArray:
        """Group forms (Tu)'H_g(Tv), without symmetry assumptions."""
        return self.full_state.group_jacobian_form(
            self.design.expand(left), self.design.expand(right)
        )

    def row_term_dot(self, values: ArrayLike) -> FloatArray:
        """Coordinate row-contribution contractions in input positional order."""
        return self.full_state.row_term_dot(self.design.expand(values))

    def group_scores(self) -> FloatArray:
        """Detached G-by-d coordinate contributions; full sparse scores remain held."""
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(
                np.asarray(
                    self.full_state.group_scores_sparse() @ self.design.basis_matrix,
                    dtype=np.float64,
                )
            )

    def rows(self) -> CancelledRows:
        """Detached original positional row quantities, without retransforming."""
        return self.full_state.rows()

    def accounting(self) -> CancelledAccounting:
        """Preserved singleton/all-zero group accounting, without deletion."""
        return self.full_state.accounting()


def identified_equations(
    geometry: NumericalGeometry,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    inputs: GeometryInputs,
    variance: Variance,
    centered_offset: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
) -> IdentifiedState:
    """Evaluate actual cancelled equations after fresh geometry veto checks.

    Parameters
    ----------
    geometry : NumericalGeometry
        Numerical complement with an explicit construction reservation.
    z : array_like
        Already transformed finite nonnegative outcomes, as in cancelled_equations.
    theta : array_like
        New coordinate point. It is detached before any variance callback.
    inputs : GeometryInputs
        Current raw columns, original names and row positions; checked afresh.
    variance : Variance
        Existing actual-package working variance, evaluated afresh by the core.
    centered_offset : array_like, optional
        Already centered transformed offset, added exactly once by the core.
    obs_weights : array_like, optional
        Existing positive multipliers, with no new weight policy.

    Returns
    -------
    IdentifiedState
        Coordinate actions plus full moments and checked-candidate evidence.
    """
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must come from numerical_geometry")
    evidence = check_geometry(geometry, inputs, max_bytes=geometry.max_bytes)
    return _identified_equations(
        geometry,
        z,
        theta,
        evidence=evidence,
        variance=variance,
        centered_offset=centered_offset,
        obs_weights=obs_weights,
    )


def _identified_equations(
    geometry: NumericalGeometry,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    evidence: GeometryCheck,
    variance: Variance,
    centered_offset: ArrayLike | None,
    obs_weights: ArrayLike | None,
) -> IdentifiedState:
    """Shared point construction for external checks and owned in-flight fits.

    Only the owned capture calls this private seam without fresh structural work.
    Point detachment, callbacks, rows, score and action semantics are shared.
    """
    point = FloatStorage.store(geometry.checked_coefficients(theta))
    full_state = cancelled_equations(
        geometry.design,
        z,
        geometry.expand(point.view()),
        variance=variance,
        centered_offset=centered_offset,
        obs_weights=obs_weights,
    )
    score = geometry.expand_transpose(full_state.score)
    return cast(
        IdentifiedState,
        _build_record(
            IdentifiedState,
            design=geometry,
            geometry_check=evidence,
            full_state=full_state,
            _theta=point,
            _score=FloatStorage.store(score),
        ),
    )
