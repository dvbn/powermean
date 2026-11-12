"""Full-coordinate implicit cancelled derivative actions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import finite_product, finite_vector
from powermean.fe.cancel import segment_sum, to_original, to_sorted
from powermean.fe.retained import CenteredDesign
from powermean.inference.correction._inputs import checked_array
from powermean.inference.correction._row_derivatives import (
    BiasRowPrimitives,
    CellRowPrimitives,
)
from powermean.inference.correction.budget import (
    OperatorWorkspacePlan,
)
from powermean.systems.cancelled import CancelledState


def _second_zeta_sorted(
    state: CancelledState,
    primitives: BiasRowPrimitives,
    du_sorted: NDArray[np.float64],
    dv_sorted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute uncentered sorted row curvature terms ``zeta_i(du, dv)``."""
    structure = state.design.structure
    du_sorted = checked_array(du_sorted, (structure.n_rows,), "du_sorted")
    dv_sorted = checked_array(dv_sorted, (structure.n_rows,), "dv_sorted")
    codes = structure.sorted_codes
    k_vec = state.k
    with np.errstate(over="raise", invalid="raise"):
        s_ku = segment_sum(structure, finite_product(k_vec * du_sorted))
        s_kv = segment_sum(structure, finite_product(k_vec * dv_sorted))
        coupled = finite_product(dv_sorted * s_ku[codes] + du_sorted * s_kv[codes])
        direct = finite_product(-primitives.k_dot * du_sorted * dv_sorted)
        return finite_product(direct + primitives.eta_weights * coupled)


@dataclass(frozen=True, slots=True)
class CancelledBiasState:
    """Fresh-point second-derivative and cell operator in cancelled coordinates."""

    state: CancelledState
    primitives: BiasRowPrimitives
    _cell: CellRowPrimitives | None
    workspace_plan: OperatorWorkspacePlan

    @property
    def cell_primitives(self) -> CellRowPrimitives:
        """Cell data exist only after an explicit cell-capable evaluation."""
        if self._cell is None:
            raise ValueError(
                "cell primitives were not requested; evaluate with cell=True"
            )
        return self._cell

    @property
    def design(self) -> CenteredDesign:
        """Underlying group-centered design ``B = M_C[X, F_R]``."""
        return self.state.design

    @property
    def group_cancel_scale(self) -> NDArray[np.float64]:
        """Unweighted cancellation scale ``A_hat_g = n_g^-1 sum_{i in g} Z_i/m_i``."""
        return self.cell_primitives.group_cancel_scale

    @property
    def fitted_mean_original(self) -> NDArray[np.float64]:
        """After-cancellation fitted mean ``M_hat_i = A_hat_{g(i)} m_hat_i``."""
        return to_original(self.design.structure, self.cell_primitives.fitted_mean)

    @property
    def cell_residual_original(self) -> NDArray[np.float64]:
        """After-cancellation cell residual ``e_hat_i = Z_i - M_hat_i``."""
        return to_original(self.design.structure, self.cell_primitives.cell_residual)

    def pi_centered_sorted(self, values: ArrayLike) -> NDArray[np.float64]:
        """Compute ``u_i' v = (B_i - Bbar_g)' v`` in sorted group order."""
        structure = self.design.structure
        image = self.design.matvec_sorted(values)
        with np.errstate(over="raise", invalid="raise"):
            means = segment_sum(structure, finite_product(self.state.pi * image))
            return finite_product(image - means[structure.sorted_codes])

    def group_adjoint_linear_weights(self, adjoint: ArrayLike) -> NDArray[np.float64]:
        """Compute sorted linear adjoint weights ``gamma_g(w)_i = -k_i * w_hat_i``."""
        w_hat = self.pi_centered_sorted(adjoint)
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(-self.state.k * w_hat)

    def second_derivative_sorted(
        self, du_sorted: NDArray[np.float64], dv_sorted: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate ``D^2 rho[du, dv]`` in sorted group order."""
        structure = self.design.structure
        zeta = _second_zeta_sorted(self.state, self.primitives, du_sorted, dv_sorted)
        with np.errstate(over="raise", invalid="raise"):
            zeta_sums = segment_sum(structure, zeta)
            return finite_product(
                zeta - self.state.pi * zeta_sums[structure.sorted_codes]
            )

    def second_derivative_vec(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``D^2 S(theta)[left, right]`` in full column coordinates."""
        du = self.design.matvec_sorted(left)
        dv = self.design.matvec_sorted(right)
        return self.design.rmatvec_sorted(self.second_derivative_sorted(du, dv))

    def group_second_derivative_form(
        self, adjoint: ArrayLike, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``w' D^2 psi_g(theta)[u, v]`` for each cancelled group ``g``."""
        w_hat = self.pi_centered_sorted(adjoint)
        du = self.design.matvec_sorted(left)
        dv = self.design.matvec_sorted(right)
        zeta = _second_zeta_sorted(self.state, self.primitives, du, dv)
        with np.errstate(over="raise", invalid="raise"):
            return segment_sum(self.design.structure, finite_product(w_hat * zeta))

    def cell_score_dot(self, values: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``f_i(theta)' v = lambda_i u_i' v`` in original row order."""
        u_dot_v = self.pi_centered_sorted(values)
        with np.errstate(over="raise", invalid="raise"):
            scores_sorted = finite_product(
                self.cell_primitives.lambda_weights * u_dot_v
            )
        return to_original(self.design.structure, scores_sorted)

    def cell_score_rmatvec(self, row_weights: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``sum_i row_weights_i f_i(theta)`` without an ``n x p`` matrix."""
        s = self.design.structure
        alpha = to_sorted(s, finite_vector(row_weights, s.n_rows, "row_weights"))
        with np.errstate(over="raise", invalid="raise"):
            mu = finite_product(self.cell_primitives.lambda_weights * alpha)
            mu_sums = segment_sum(s, mu)
            centered = finite_product(mu - self.state.pi * mu_sums[s.sorted_codes])
        return self.design.rmatvec_sorted(centered)

    def cell_derivative_vec(
        self, row_weights: ArrayLike, values: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``sum_i row_weights_i (nabla f_i(theta) v)`` in ``O(n)`` work."""
        s = self.design.structure
        p = self.cell_primitives
        alpha = to_sorted(s, finite_vector(row_weights, s.n_rows, "row_weights"))
        bv = self.design.matvec_sorted(values)
        with np.errstate(over="raise", invalid="raise"):
            mu = finite_product(p.lambda_weights * alpha)
            mu_group = segment_sum(s, mu)
            term1 = finite_product(p.u_lambda_weights * alpha)
            term2 = finite_product(
                mu_group[s.sorted_codes] * self.primitives.eta_weights
            )
            xi = finite_product((term1 + term2) * bv)
            xi_sums = segment_sum(s, xi)
            centered = finite_product(xi - self.state.pi * xi_sums[s.sorted_codes])
            return finite_product(-self.design.rmatvec_sorted(centered))

    def cell_derivative_form(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``left' nabla f_i(theta) right`` at each original row ``i``."""
        s = self.design.structure
        p = self.cell_primitives
        w_hat = self.pi_centered_sorted(left)
        bv = self.design.matvec_sorted(right)
        with np.errstate(over="raise", invalid="raise"):
            direct = finite_product(-p.u_lambda_weights * w_hat * bv)
            coupling = segment_sum(
                s, finite_product(self.primitives.eta_weights * w_hat * bv)
            )
            indirect = finite_product(-p.lambda_weights * coupling[s.sorted_codes])
            return to_original(s, finite_product(direct + indirect))
