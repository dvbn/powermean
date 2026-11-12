"""BBP short-group analytical bias correction.

Evaluates the reduced-bias adjustment for independent cancelled-group scores
using the actual nonsymmetric Jacobian ``H = -nabla S(theta)``, unnormalized
group scores ``psi_g``, unscaled ``C_theta = H^-1 K H^-T``, and a single
``OrderScale`` mapping at the reporting boundary.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import FloatStorage, finite_product
from powermean.fe.cancel import segment_sum
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_design import NumericalGeometry
from powermean.inference.cancelled import transpose_rhs
from powermean.inference.correction._inputs import BiasOperatorRequest
from powermean.inference.correction.budget import (
    ShortGroupWorkspacePlan,
    reserve_short_group_bias,
)
from powermean.inference.correction.operator import (
    IdentifiedBiasState,
    evaluate_identified_bias_state,
)
from powermean.inference.correction.selector import (
    BiasEvalOptions,
    OrderMappedBias,
    ResolvedSelector,
    apply_single_order_mapping,
    resolve_selector,
    selector_size,
)
from powermean.model.variance import Variance
from powermean.solve.factorization import factorize


def _exact_symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize an outer-product matrix without touching nonsymmetric H."""
    with np.errstate(over="ignore"):
        total = matrix + matrix.T
    halves = 0.5 * matrix + 0.5 * matrix.T
    return finite_product(np.where(np.isinf(total), halves, 0.5 * total))


@dataclass(frozen=True, slots=True)
class ShortGroupBiasResult:
    """Detached outputs of BBP short-group bias correction at a candidate point."""

    operator: IdentifiedBiasState
    selector: ResolvedSelector
    rcond: float
    mapped: OrderMappedBias
    workspace_plan: ShortGroupWorkspacePlan
    _hessian: FloatStorage
    _meat: FloatStorage
    _cov_theta: FloatStorage
    _adjoint_weights: FloatStorage
    _first_term: FloatStorage
    _second_term: FloatStorage
    _bias_theta: FloatStorage
    _p_row: FloatStorage
    _zeta_row: FloatStorage

    @property
    def uncorrected(self) -> NDArray[np.float64]:
        """Reported uncorrected coefficients ``L theta / omega``."""
        return self.mapped.uncorrected

    @property
    def bias(self) -> NDArray[np.float64]:
        """Reported bias estimate ``b_hat / omega``."""
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
    def uncorrected_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale uncorrected coefficients ``L theta``."""
        return self.mapped.uncorrected_transformed

    @property
    def bias_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale bias ``b_hat_gamma = L b_hat_theta``."""
        return self.mapped.bias_transformed

    @property
    def corrected_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale bias-corrected coefficients ``L theta - b_hat``."""
        return self.mapped.corrected_transformed

    @property
    def vcov_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale first-order covariance ``L C_theta L'``."""
        return self.mapped.vcov_transformed

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
        """Unnormalized group score outer product ``K = sum_g psi_g psi_g'``."""
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
        """First contraction vector ``sum_g nabla psi_g H^-1 psi_g`` in ``R^d``."""
        return self._first_term.view()

    @property
    def second_term(self) -> NDArray[np.float64]:
        """Second contraction vector ``sum_g nabla^2 psi_g : C_theta`` in ``R^d``."""
        return self._second_term.view()

    def group_contractions(
        self, selected_index: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Detached G-vectors for one selection, never a G-by-d table.

        These are the adjoint first/second contractions. Their sum
        gives that selected bias before order mapping. This accesses captured
        result data only and is never accepted as evidence for a new evaluation.
        """
        if (
            type(selected_index) is not int
            or not 0 <= selected_index < self.selector.n_selected
        ):
            raise ValueError("selected_index must index the resolved selection")
        return _group_adjoint_column(
            self.operator,
            self.adjoint_weights[:, selected_index],
            self._p_row.view(),
            self._zeta_row.view(),
        )


def _accumulate_meat_and_prow(
    op: IdentifiedBiasState, h_inv: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute unnormalized ``K = sum_g psi_g psi_g'`` and ``p_row = P_g rho_g``."""
    geometry = op.geometry
    cb = op.cancelled_bias_state
    structure = cb.design.structure
    n_id = geometry.n_identified
    meat = np.zeros((n_id, n_id), dtype=np.float64)
    p_row = np.zeros(structure.n_rows, dtype=np.float64)
    basis = np.zeros(n_id, dtype=np.float64)
    rho = cb.state.rho
    codes = structure.sorted_codes

    for col in range(n_id):
        basis[col] = 1.0
        dots = op.group_score_dot(basis)
        basis[col] = 0.0
        dots_row = dots[codes]
        with np.errstate(over="raise", invalid="raise"):
            weighted_rho = finite_product(rho * dots_row)
        meat[:, col] = geometry.rmatvec_sorted(weighted_rho)
        h_col_sorted = geometry.matvec_sorted(h_inv[:, col])
        with np.errstate(over="raise", invalid="raise"):
            p_row = finite_product(p_row + h_col_sorted * dots_row)

    return _exact_symmetric(finite_product(meat)), p_row


def _group_adjoint_column(
    op: IdentifiedBiasState,
    adjoint: NDArray[np.float64],
    p_row: NDArray[np.float64],
    zeta_row: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One selected adjoint's G-vector contractions, streaming implicit B."""
    cb = op.cancelled_bias_state
    full = op.geometry.expand(adjoint)
    gamma = cb.group_adjoint_linear_weights(full)
    centered = cb.pi_centered_sorted(full)
    with np.errstate(over="raise", invalid="raise"):
        first = segment_sum(cb.design.structure, finite_product(gamma * p_row))
        second = segment_sum(cb.design.structure, finite_product(centered * zeta_row))
    return first, second


def bbp_short_group_correction(
    geometry: NumericalGeometry,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    inputs: GeometryInputs,
    variance: Variance,
    projection: ArrayLike | None = None,
    options: BiasEvalOptions,
) -> ShortGroupBiasResult:
    """Compute BBP short-group coefficient bias correction and first-order vcov."""
    if type(options) is not BiasEvalOptions:
        raise TypeError("options must be a BiasEvalOptions")
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must be NumericalGeometry")
    s = geometry.design.structure
    plan = reserve_short_group_bias(
        geometry,
        selector_size(geometry, projection),
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
        ),
    )

    hessian = op.hessian()
    factor = factorize(hessian, options.rcond_min)
    n_id = geometry.n_identified
    h_inv = factor.solve(np.eye(n_id, dtype=np.float64))
    adjoint_weights = transpose_rhs(factor, selector.matrix)

    meat, p_row = _accumulate_meat_and_prow(op, h_inv)
    h_inv_k = factor.solve(meat)
    cov_theta = _exact_symmetric(finite_product(factor.solve(h_inv_k.T).T))

    cb = op.cancelled_bias_state
    with np.errstate(over="raise", invalid="raise"):
        kp = finite_product(cb.state.k * p_row)
        kp_sums = segment_sum(s, kp)
        first_sorted = finite_product(kp - cb.state.pi * kp_sums[s.sorted_codes])
        first_term = finite_product(-geometry.rmatvec_sorted(first_sorted))

    second_term, zeta_row = op.contract_second_derivative_with_rows(cov_theta)
    with np.errstate(over="raise", invalid="raise"):
        combined_rhs = finite_product(first_term + 0.5 * second_term)
        bias_theta = factor.solve(combined_rhs)
        bias_gamma = finite_product(adjoint_weights.T @ combined_rhs)
        uncorrected_gamma = selector.project_vector(op.theta)
        corrected_gamma = finite_product(uncorrected_gamma - bias_gamma)
        vcov_gamma = _exact_symmetric(
            finite_product(adjoint_weights.T @ meat @ adjoint_weights)
        )

    mapped = apply_single_order_mapping(
        uncorrected_gamma, bias_gamma, corrected_gamma, vcov_gamma, order=options.order
    )

    return ShortGroupBiasResult(
        op,
        selector,
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
        FloatStorage.store(p_row),
        FloatStorage.store(zeta_row),
    )
