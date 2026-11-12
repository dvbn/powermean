"""BBP after-cancellation cell analytical bias correction.

Evaluates the cell-level reduced-bias adjustment and cell-pair meat over an
explicit set of ordered dependent cell pairs ``Pairs`` (including the diagonal),
using the unweighted cancellation scale ``A_hat_g = n_g^-1 sum_{i in g} Z_i/m_i``
and actual nonsymmetric ``H = -nabla S(theta)``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import (
    FloatStorage,
    finite_product,
)
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_design import NumericalGeometry
from powermean.inference.cancelled import transpose_rhs
from powermean.inference.correction._inputs import BiasOperatorRequest
from powermean.inference.correction.budget import CellWorkspacePlan, reserve_cell_bias
from powermean.inference.correction.operator import (
    IdentifiedBiasState,
    evaluate_identified_bias_state,
)
from powermean.inference.correction.pairs import (
    CellPairs,
    pair_count,
    validate_cell_pairs,
)
from powermean.inference.correction.selector import (
    BiasEvalOptions,
    OrderMappedBias,
    ResolvedSelector,
    apply_single_order_mapping,
    resolve_selector,
    selector_size,
)
from powermean.inference.correction.short_group import _exact_symmetric
from powermean.model.variance import Variance
from powermean.solve.factorization import factorize


@dataclass(frozen=True, slots=True)
class CellBiasResult:
    """Detached outputs of BBP after-cancellation cell bias correction."""

    operator: IdentifiedBiasState
    selector: ResolvedSelector
    cell_pairs: CellPairs
    rcond: float
    mapped: OrderMappedBias
    workspace_plan: CellWorkspacePlan
    _hessian: FloatStorage
    _meat: FloatStorage
    _cov_theta: FloatStorage
    _adjoint_weights: FloatStorage
    _first_term: FloatStorage
    _second_term: FloatStorage
    _bias_theta: FloatStorage

    @property
    def uncorrected(self) -> NDArray[np.float64]:
        """Reported uncorrected coefficients ``L theta / omega``."""
        return self.mapped.uncorrected

    @property
    def bias(self) -> NDArray[np.float64]:
        """Reported cell bias estimate ``b_hat / omega``."""
        return self.mapped.bias

    @property
    def corrected(self) -> NDArray[np.float64]:
        """Reported bias-corrected coefficients ``(L theta - b_hat) / omega``."""
        return self.mapped.corrected

    @property
    def vcov(self) -> NDArray[np.float64]:
        """Reported first-order covariance at the uncorrected point."""
        return self.mapped.vcov

    @property
    def se(self) -> NDArray[np.float64]:
        """Reported first-order standard errors at the uncorrected point."""
        return self.mapped.se

    @property
    def bias_theta(self) -> NDArray[np.float64]:
        """Full identified bias vector ``H^-1 (v_1 + 0.5 v_2)`` in ``R^d``."""
        return self._bias_theta.view()

    @property
    def hessian(self) -> NDArray[np.float64]:
        """Actual nonsymmetric Jacobian ``H = -nabla S(theta)`` in ``R^{d x d}``."""
        return self._hessian.view()

    @property
    def meat(self) -> NDArray[np.float64]:
        """Unnormalized cell-pair meat ``K = sum_{(i,j) in Pairs} f_i f_j' e_i e_j``."""
        return self._meat.view()

    @property
    def cov_theta(self) -> NDArray[np.float64]:
        """Unscaled parameter sandwich ``C_theta = H^-1 K H^-T`` in ``R^{d x d}``."""
        return self._cov_theta.view()

    @property
    def adjoint_weights(self) -> NDArray[np.float64]:
        """Adjoint projection columns ``W = H^-T L'`` in ``R^{d x q}``."""
        return self._adjoint_weights.view()

    @property
    def first_term(self) -> NDArray[np.float64]:
        """First contraction ``sum_{(i,j) in Pairs} nabla f_i H^-1 f_j e_i e_j``."""
        return self._first_term.view()

    @property
    def second_term(self) -> NDArray[np.float64]:
        """Second contraction ``nabla^2 S : C_theta`` in ``R^d``."""
        return self._second_term.view()

    @property
    def group_cancel_scale(self) -> NDArray[np.float64]:
        """Unweighted cancellation scale ``A_hat_g = n_g^-1 sum_{i in g} Z_i/m_i``."""
        return self.operator.group_cancel_scale

    @property
    def fitted_mean(self) -> NDArray[np.float64]:
        """After-cancellation fitted mean ``M_hat_i = A_hat_{g(i)} m_hat_i``."""
        return self.operator.fitted_mean

    @property
    def cell_residual(self) -> NDArray[np.float64]:
        """After-cancellation cell residual ``e_hat_i = Z_i - M_hat_i``."""
        return self.operator.cell_residual


def _stream_cell_pair_contractions(
    op: IdentifiedBiasState, cell_pairs: CellPairs, h_inv: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stream ``K`` and ``sum_{(i,j) in Pairs} nabla f_i H^-1 f_j e_i e_j``."""
    n_id = op.geometry.n_identified
    n_obs = cell_pairs.n_obs
    left_idx, right_idx = cell_pairs.left_indices, cell_pairs.right_indices
    residuals = op.cell_residual

    meat = np.zeros((n_id, n_id), dtype=np.float64)
    first_term = np.zeros(n_id, dtype=np.float64)
    basis = np.zeros(n_id, dtype=np.float64)

    for col in range(n_id):
        basis[col] = 1.0
        f_col = op.cell_score_dot(basis)
        basis[col] = 0.0
        with np.errstate(over="raise", invalid="raise"):
            y_right = finite_product(f_col * residuals)[right_idx]
            pair_sums = np.bincount(left_idx, weights=y_right, minlength=n_obs)
            p_weights = finite_product(residuals * pair_sums)
        meat[:, col] = op.cell_score_rmatvec(p_weights)
        contrib = op.cell_derivative_vec(p_weights, h_inv[:, col])
        with np.errstate(over="raise", invalid="raise"):
            first_term = finite_product(first_term + contrib)

    if cell_pairs.is_symmetric:
        meat = _exact_symmetric(meat)
    return finite_product(meat), first_term


def bbp_cell_correction(
    geometry: NumericalGeometry,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    inputs: GeometryInputs,
    pairs: ArrayLike | CellPairs | None,
    variance: Variance,
    projection: ArrayLike | None = None,
    options: BiasEvalOptions,
) -> CellBiasResult:
    """Compute BBP after-cancellation cell bias correction and first-order vcov."""
    if type(options) is not BiasEvalOptions:
        raise TypeError("options must be a BiasEvalOptions")
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must be a NumericalGeometry")
    s = geometry.design.structure
    count = pair_count(pairs, s.n_rows)
    plan = reserve_cell_bias(
        geometry, selector_size(geometry, projection), count, bounds=options.bounds
    )
    cell_pairs = validate_cell_pairs(
        pairs,
        s.n_rows,
        require_symmetric=options.require_symmetric_pairs,
        bounds=options.bounds,
    )
    selector = resolve_selector(geometry, projection, bounds=options.bounds)
    if options.order.dimension != selector.n_selected:
        raise ValueError("order requires one scale per selected coefficient")
    op = evaluate_identified_bias_state(
        geometry,
        z,
        theta,
        inputs=inputs,
        variance=variance,
        request=BiasOperatorRequest(
            centered_offset=options.centered_offset,
            obs_weights=options.obs_weights,
            bounds=options.bounds,
            cell=True,
        ),
    )

    hessian = op.hessian()
    factor = factorize(hessian, options.rcond_min)
    n_id = geometry.n_identified
    h_inv = factor.solve(np.eye(n_id, dtype=np.float64))
    adjoint_weights = transpose_rhs(factor, selector.matrix)

    meat, first_term = _stream_cell_pair_contractions(op, cell_pairs, h_inv)
    h_inv_k = factor.solve(meat)
    cov_raw = finite_product(factor.solve(h_inv_k.T).T)
    cov_theta = _exact_symmetric(cov_raw) if cell_pairs.is_symmetric else cov_raw

    second_term = op.contract_second_derivative(cov_theta)
    with np.errstate(over="raise", invalid="raise"):
        combined_rhs = finite_product(first_term + 0.5 * second_term)
        bias_theta = factor.solve(combined_rhs)
        bias_gamma = finite_product(adjoint_weights.T @ combined_rhs)
        uncorrected_gamma = selector.project_vector(op.theta)
        corrected_gamma = finite_product(uncorrected_gamma - bias_gamma)
        vcov_raw = finite_product(adjoint_weights.T @ meat @ adjoint_weights)
        vcov_gamma = _exact_symmetric(vcov_raw) if cell_pairs.is_symmetric else vcov_raw

    mapped = apply_single_order_mapping(
        uncorrected_gamma, bias_gamma, corrected_gamma, vcov_gamma, order=options.order
    )

    return CellBiasResult(
        op,
        selector,
        cell_pairs,
        float(factor.rcond),
        mapped,
        plan,
        FloatStorage.store(hessian),
        FloatStorage.store(meat),
        FloatStorage.store(cov_theta),
        FloatStorage.store(adjoint_weights),
        FloatStorage.store(first_term),
        FloatStorage.store(second_term),
        FloatStorage.store(bias_theta),
    )
