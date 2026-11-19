"""BBP short-group analytical bias correction (design03 Sections 5.1-5.2).

Evaluates the reduced-bias adjustment for independent cancelled-group scores
using the actual nonsymmetric Jacobian ``H = -nabla S(theta)``, unnormalized
group scores ``psi_g``, unscaled ``C_theta = H^-1 K H^-T``, and a single
``OrderScale`` mapping at the reporting boundary.
"""

import importlib
from dataclasses import dataclass
from typing import NamedTuple, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import FloatStorage, finite_product
from powermean.fe.cancel import segment_sum
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_design import NumericalGeometry
from powermean.inference.cancelled import transpose_rhs
from powermean.inference.correction._cancelled_operator import (
    CancelledBiasState,
    gathers_apply,
    group_congruence,
    group_right_form,
)
from powermean.inference.correction._inputs import BiasOperatorRequest
from powermean.inference.correction._row_derivatives import certified_ppml_primitives
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
from powermean.solve.factorization import Factor, factorize
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import IdentifiedState


class _Trmm(Protocol):
    """``scipy.linalg.blas.dtrmm``: ``b op(a)`` with ``side=1``, ``a`` triangular."""

    def __call__(
        self,
        alpha: float,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        /,
        *,
        side: int,
        lower: int,
    ) -> NDArray[np.float64]:
        """The product, in a new array."""
        ...


_trmm = cast("_Trmm", importlib.import_module("scipy.linalg.blas").dtrmm)


def _sandwich(
    h_inv: NDArray[np.float64], meat: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``C = H^-1 K H^-T`` for the symmetric meat ``K``.

    Where ``K = L L'`` has a Cholesky factor, ``C = M M'`` with ``M = H^-1 L``: a
    Cholesky factor, one triangular product and one symmetric rank-``d`` product
    in place of the two general products, which are used otherwise. Both round to
    the same matrix: ``L L'`` reproduces ``K`` to rounding in each entry relative
    to ``sqrt(K_ii K_jj)``, the scale at which the products round. ``M M'`` is
    symmetric bit for bit.
    """
    if meat.size == 0:
        return (h_inv @ meat) @ h_inv.T
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            lower = np.linalg.cholesky(meat)
    except np.linalg.LinAlgError:
        return (h_inv @ meat) @ h_inv.T
    if not np.isfinite(lower).all():
        return (h_inv @ meat) @ h_inv.T
    product = np.ascontiguousarray(_trmm(1.0, lower, h_inv, side=1, lower=1))
    return product @ product.T


def _exact_symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize an outer-product matrix without touching nonsymmetric H."""
    with np.errstate(over="ignore"):
        total = matrix + matrix.T
    if np.isfinite(total).all():
        return finite_product(0.5 * total)
    halves = 0.5 * matrix + 0.5 * matrix.T
    return finite_product(np.where(np.isinf(total), halves, 0.5 * total))


class ShortGroupFitPoint(NamedTuple):
    """Call-local equations, actual LU and kept-slope adjoint from certification."""

    capture: CapturedFit
    state: IdentifiedState
    factor: Factor
    adjoint: NDArray[np.float64]


class ShortGroupReport(NamedTuple):
    """The correction outputs consumed by a normal fit."""

    uncorrected: NDArray[np.float64]
    bias: NDArray[np.float64]
    corrected: NDArray[np.float64]
    bias_theta: NDArray[np.float64]
    workspace_plan: ShortGroupWorkspacePlan


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

        These are the design03 5.2 adjoint first/second contractions. Their sum
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
    """Compute unnormalized ``K = sum_g psi_g psi_g'`` and ``p_row = P_g rho_g``.

    ``K = T' B' (sum_g rho_g rho_g') B T`` and
    ``p_row_i = sum_j (b_i' H^-1 b_j) rho_j`` over the rows ``j`` of the group
    of ``i`` are formed from the entries of each group where ``gathers_apply``
    holds, and by the loop over identified columns otherwise.
    """
    cb, basis = op.cancelled_bias_state, op.geometry.basis_matrix
    if not gathers_apply(cb, op.geometry.n_identified, basis, h_inv):
        return _meat_and_prow_by_columns(op, h_inv)
    rho = cb.state.rho
    with np.errstate(over="raise", invalid="raise"):
        outer = group_congruence(cb.design, None, rho, rho)
        meat = op.geometry.compress(outer)
        gathered = op.geometry.gathered(h_inv)
        if gathered is None:
            p_row = group_right_form(cb.design, op.geometry.inflate(h_inv), rho)
        else:
            p_row = group_right_form(cb.design, gathered[0], rho, gathered[1])
    return _exact_symmetric(meat), finite_product(p_row)


def _meat_and_prow_by_columns(
    op: IdentifiedBiasState, h_inv: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``K`` and ``p_row`` one identified column at a time, by design products."""
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
    report = _short_group_correction(
        geometry,
        z,
        theta,
        inputs=inputs,
        variance=variance,
        projection=projection,
        options=options,
        fit_point=None,
    )
    assert isinstance(report, ShortGroupBiasResult)
    return report


def _short_group_correction(
    geometry: NumericalGeometry,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    inputs: GeometryInputs,
    variance: Variance,
    projection: ArrayLike | None = None,
    options: BiasEvalOptions,
    fit_point: ShortGroupFitPoint | None,
) -> ShortGroupBiasResult | ShortGroupReport:
    """Use fit-owned artifacts only after the reporting caller binds their inputs."""
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
    if fit_point is None:
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
                admitted=options.admitted,
            ),
        )
        hessian = op.hessian()
        factor = factorize(hessian, options.rcond_min)
    else:
        state, factor = fit_point.state, fit_point.factor
        primitives = certified_ppml_primitives(state.full_state)
        cb = CancelledBiasState(state.full_state, primitives, None, plan.operator)
        op = IdentifiedBiasState(geometry, state, cb, plan.operator)
    n_id = geometry.n_identified
    h_inv = factor.solve(np.eye(n_id, dtype=np.float64))
    adjoint_weights = (
        transpose_rhs(factor, selector.matrix)
        if fit_point is None
        else fit_point.adjoint
    )

    meat, p_row = _accumulate_meat_and_prow(op, h_inv)
    # C = H^-1 K H^-T from the inverse already formed, in place of two solves with
    # d right-hand sides: the same matrix to rounding.
    with np.errstate(over="raise", invalid="raise"):
        cov_theta = _exact_symmetric(finite_product(_sandwich(h_inv, meat)))

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

    if fit_point is not None:
        return ShortGroupReport(
            mapped.uncorrected, mapped.bias, mapped.corrected, bias_theta, plan
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
