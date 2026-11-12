"""Fresh-point cancelled and identified bias operators.

Wraps ``cancelled_equations`` and ``identified_equations`` at a fresh candidate
point, reserves numeric workspace before allocation, and provides tensor-free
analytic directional and adjoint second-derivative contractions as well as
after-cancellation cell score and cell derivative operators.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import finite_product
from powermean.fe.cancel import segment_sum, to_sorted
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.retained import CenteredDesign
from powermean.inference.correction._cancelled_operator import (
    CancelledBiasState,
    _second_zeta_sorted,
)
from powermean.inference.correction._inputs import (
    BiasOperatorRequest,
    capture_point,
    checked_array,
    point_variance,
)
from powermean.inference.correction._row_derivatives import (
    compute_bias_row_primitives,
)
from powermean.inference.correction.budget import (
    BiasBounds,
    OperatorWorkspacePlan,
    reserve_bias_operator,
)
from powermean.model.variance import Variance
from powermean.systems.cancelled import cancelled_equations
from powermean.systems.identified import IdentifiedState, identified_equations


@dataclass(frozen=True, slots=True)
class IdentifiedBiasState:
    """Fresh-point second-derivative and cell operator on identified ``R^d``."""

    geometry: NumericalGeometry
    identified_state: IdentifiedState
    cancelled_bias_state: CancelledBiasState
    workspace_plan: OperatorWorkspacePlan

    @property
    def theta(self) -> NDArray[np.float64]:
        """Detached identified coefficient vector in ``R^d``."""
        return self.identified_state.theta

    @property
    def score(self) -> NDArray[np.float64]:
        """Detached unnormalized identified score ``S(theta)`` in ``R^d``."""
        return self.identified_state.score

    @property
    def group_cancel_scale(self) -> NDArray[np.float64]:
        """Unweighted cancellation scale ``A_hat_g = n_g^-1 sum_{i in g} Z_i/m_i``."""
        return self.cancelled_bias_state.group_cancel_scale

    @property
    def fitted_mean(self) -> NDArray[np.float64]:
        """After-cancellation fitted mean ``M_hat_i`` in original row order."""
        return self.cancelled_bias_state.fitted_mean_original

    @property
    def cell_residual(self) -> NDArray[np.float64]:
        """After-cancellation cell residual ``e_hat_i`` in original row order."""
        return self.cancelled_bias_state.cell_residual_original

    def jac_vec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``H v`` on ``R^d`` with actual nonsymmetric ``H = -nabla S``."""
        return self.identified_state.jac_vec(values)

    def jac_t_vec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``H' v`` on ``R^d`` without assuming symmetry."""
        return self.identified_state.jac_t_vec(values)

    def hessian(self) -> NDArray[np.float64]:
        """Materialize the exact ``(d, d)`` nonsymmetric Jacobian ``H = -nabla S``."""
        n_id = self.geometry.n_identified
        matrix = np.zeros((n_id, n_id), dtype=np.float64)
        basis = np.zeros(n_id, dtype=np.float64)
        for col in range(n_id):
            basis[col] = 1.0
            matrix[:, col] = self.jac_vec(basis)
            basis[col] = 0.0
        return finite_product(matrix)

    def group_score_dot(self, values: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``psi_g' (T v)`` for each group ``g``."""
        return self.identified_state.group_score_dot(values)

    def group_jacobian_form(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``(T left)' H_g (T right)`` for each group ``g``."""
        return self.identified_state.group_jacobian_form(left, right)

    def second_derivative_vec(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``T' D^2 S(T theta)[T left, T right]`` in ``R^d``."""
        full = self.cancelled_bias_state.second_derivative_vec(
            self.geometry.expand(left), self.geometry.expand(right)
        )
        return self.geometry.expand_transpose(full)

    def group_second_derivative_form(
        self, adjoint: ArrayLike, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``(T adjoint)' D^2 psi_g[T left, T right]`` per group ``g``."""
        return self.cancelled_bias_state.group_second_derivative_form(
            self.geometry.expand(adjoint),
            self.geometry.expand(left),
            self.geometry.expand(right),
        )

    def contract_second_derivative_with_rows(
        self, matrix: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute ``sum_g nabla^2 psi_g : C`` and sorted row curvature ``zeta``."""
        n_id = self.geometry.n_identified
        mat = checked_array(matrix, (n_id, n_id), "matrix").copy()

        cb = self.cancelled_bias_state
        s = cb.design.structure
        zeta_total = np.zeros(s.n_rows, dtype=np.float64)
        basis = np.zeros(n_id, dtype=np.float64)
        for col in range(n_id):
            basis[col] = 1.0
            du = self.geometry.matvec_sorted(mat[:, col])
            dv = self.geometry.matvec_sorted(basis)
            basis[col] = 0.0
            zeta_col = _second_zeta_sorted(cb.state, cb.primitives, du, dv)
            with np.errstate(over="raise", invalid="raise"):
                zeta_total = finite_product(zeta_total + zeta_col)

        with np.errstate(over="raise", invalid="raise"):
            zeta_sums = segment_sum(s, zeta_total)
            d2_rho = finite_product(
                zeta_total - cb.state.pi * zeta_sums[s.sorted_codes]
            )
        return self.geometry.rmatvec_sorted(d2_rho), zeta_total

    def contract_second_derivative(self, matrix: ArrayLike) -> NDArray[np.float64]:
        """Compute ``sum_g nabla^2 psi_g : C`` in ``R^d`` without ``d^3`` arrays."""
        contracted, _ = self.contract_second_derivative_with_rows(matrix)
        return contracted

    def cell_score_dot(self, values: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``f_i(theta)' (T v)`` in original row order."""
        return self.cancelled_bias_state.cell_score_dot(self.geometry.expand(values))

    def cell_score_rmatvec(self, row_weights: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``sum_i row_weights_i T' f_i(theta)`` in ``R^d``."""
        return self.geometry.expand_transpose(
            self.cancelled_bias_state.cell_score_rmatvec(row_weights)
        )

    def cell_derivative_vec(
        self, row_weights: ArrayLike, values: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``sum_i row_weights_i T' nabla f_i(theta) (T v)`` in ``R^d``."""
        full = self.cancelled_bias_state.cell_derivative_vec(
            row_weights, self.geometry.expand(values)
        )
        return self.geometry.expand_transpose(full)

    def cell_derivative_form(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``(T left)' nabla f_i(theta) (T right)`` per row ``i``."""
        return self.cancelled_bias_state.cell_derivative_form(
            self.geometry.expand(left), self.geometry.expand(right)
        )


def evaluate_cancelled_bias_state(
    design: CenteredDesign,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    variance: Variance,
    centered_offset: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
    bounds: BiasBounds,
    cell: bool = False,
) -> CancelledBiasState:
    """Evaluate a fresh-point ``CancelledBiasState`` with prospective reservation."""
    if type(design) is not CenteredDesign:
        raise TypeError("design must be a CenteredDesign")
    plan = reserve_bias_operator(design, bounds=bounds)
    point = capture_point(
        z,
        theta,
        design.structure.n_rows,
        design.n_columns,
        offset=centered_offset,
        weights=obs_weights,
    )
    q = design.matvec_sorted(point.theta)
    if point.offset is not None:
        q = finite_product(q + to_sorted(design.structure, point.offset))
    family = point_variance(variance, q)
    state = cancelled_equations(
        design,
        point.z,
        point.theta,
        variance=family,
        centered_offset=point.offset,
        obs_weights=point.weights,
    )
    prim, cell_data = compute_bias_row_primitives(state, point, family, cell=cell)
    return CancelledBiasState(state, prim, cell_data, plan)


def evaluate_identified_bias_state(
    geometry: NumericalGeometry,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    inputs: GeometryInputs,
    variance: Variance,
    request: BiasOperatorRequest,
) -> IdentifiedBiasState:
    """Bind current raw geometry through accepted identified_equations, every call.

    Numerical geometry checks are vetoes, not structural rank/sampling admission.
    The caller supplies an uncorrected root; this evaluator does not certify one.
    """
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must be a NumericalGeometry")
    if type(request) is not BiasOperatorRequest or type(request.cell) is not bool:
        raise TypeError("request must be a BiasOperatorRequest with a bool cell option")
    plan = reserve_bias_operator(geometry, bounds=request.bounds)
    point = capture_point(
        z,
        theta,
        geometry.structure.n_rows,
        geometry.n_identified,
        offset=request.centered_offset,
        weights=request.obs_weights,
    )
    q = geometry.matvec_sorted(point.theta)
    if point.offset is not None:
        q = finite_product(q + to_sorted(geometry.structure, point.offset))
    family = point_variance(variance, q)
    id_state = identified_equations(
        geometry,
        point.z,
        point.theta,
        inputs=inputs,
        variance=family,
        centered_offset=point.offset,
        obs_weights=point.weights,
    )
    prim, cell_data = compute_bias_row_primitives(
        id_state.full_state, point, family, cell=request.cell
    )
    cb = CancelledBiasState(id_state.full_state, prim, cell_data, plan)
    return IdentifiedBiasState(geometry, id_state, cb, plan)
