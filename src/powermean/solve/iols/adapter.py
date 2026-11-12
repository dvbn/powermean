"""No-FE direct fixed-tau IOLS adapter and column scaling."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.direct_route import (
    DirectNoFEBackend,
    IOLSDirectNoFEResult,
    solve_nofe_direct_fixed_tau,
)
from powermean.solve.iols.kernel import (
    IOLSKernelControl,
    _profile_intercept_adjustment,
)
from powermean.solve.iols.normalization import (
    TransformedNormalizationState,
    _gram_condition,
    apply_transformed_normalization,
    back_transform_beta,
    back_transform_residual,
    build_transformed_normalization_state,
)
from powermean.solve.iols.records import owned_value

ADAPTER_VERSION: Final = "bbp-iols-direct-adapter-2026-10-05.7-curvature-rerun"
CAP_FAILURE_REASONS: Final = frozenset({"phase2_evaluation_budget_exhausted"})
NUMERICAL_FAILURE_REASONS: Final = frozenset(
    {
        "not_converged",
        "invalid_phi_step",
        "criterion_stagnation",
        "criterion_explosion",
        "noncontractive",
        "post_convergence_residual",
        "post_convergence_noncontractive",
        "post_convergence_invalid_map",
        "q_convergence_invalid",
        "phase1_candidate_unusable",
        "empty_finite_rho_path",
        "tau2_anchor_failed",
        "tau2_anchor_missing_rho",
        "not_started",
    }
)
SUPPORT_FAILURE_REASONS: Final = frozenset({"tau2_boundary_separation"})
BACKEND_DETAIL_KEYS: Final = (
    "phase1_evaluations",
    "qhat_invalid_count",
    "kappa_monitor_max",
    "kappa_monitor_checks",
    "runtime_version",
    "nonconverged_iterate_policy",
    "phase2_attempts",
    "phase2_warm_restarts",
    "best_score_norm",
    "best_score_rho",
    "kappa_coordinate_row_max",
    "kappa_monitor_policy",
    "progress_monitor_policy",
    "post_check_policy",
    "post_convergence_invalid_map_count",
    "phase1_post_convergence_invalid_map_count",
    "rho_grid_policy",
    "rho_grid_hook_version",
    "rho_grid_count",
    "rho_grid_alpha0",
    "rho_grid_clipped",
    "rho_curvature_L0",
    "rho_curvature_lambda_min",
    "rho_curvature_wall_ns",
    "rho_curvature_pilot",
    "requested_backend",
    "preferred_backend",
    "actual_backend",
    "fallback_used",
    "availability_reason",
)


@dataclass(frozen=True)
class IOLSAdapterResult:
    """Result of the no-FE IOLS adapter."""

    theta: FloatArray | None
    intercept: float
    slopes: FloatArray
    slopes_star: FloatArray
    linear: FloatArray | None
    native_converged: bool
    native_status: str
    failure_reason: str | None
    failure_class: str
    budget_exhausted: bool
    phase1_iterations: int
    phase2_iterations: int
    rho_restarts: int
    requested_backend: str
    preferred_backend: str
    actual_backend: Literal["cpp", "numpy"]
    fallback_used: bool
    availability_reason: str
    normalization: TransformedNormalizationState
    diagnostics: Mapping[str, Any]
    direct_result: IOLSDirectNoFEResult | None = None

    def __post_init__(self) -> None:
        """Own mapped coefficients, fitted indices and nested diagnostics."""
        for name in ("theta", "slopes", "slopes_star", "linear", "diagnostics"):
            object.__setattr__(self, name, owned_value(getattr(self, name)))


def classify_failure_reason(reason: str | None) -> str:
    """Map a kernel failure reason to its coarse failure class."""
    if reason is None:
        return "unspecified_nonconvergence"
    if reason in CAP_FAILURE_REASONS:
        return "cap_or_refinement_limit"
    if reason in NUMERICAL_FAILURE_REASONS:
        return "numerical_nonconvergence"
    if reason in SUPPORT_FAILURE_REASONS:
        return "support_check"
    return "unrecognized_kernel_reason"


def _extract_nofe_slope_design(
    endog: npt.ArrayLike,
    exog_full: npt.ArrayLike,
    intercept_col_idx: int,
) -> tuple[FloatArray, FloatArray, list[int], int]:
    y: FloatArray = np.asarray(endog, dtype=np.float64)
    x_full: FloatArray = np.asarray(exog_full, dtype=np.float64)
    if y.ndim != 1 or x_full.ndim != 2:
        raise ValueError("exog_full must be 2-D and endog 1-D")
    n_obs, n_cols = x_full.shape
    if y.shape[0] != n_obs:
        raise ValueError("exog_full and endog row counts differ")
    if n_cols < 2:
        raise ValueError("exog_full must contain the intercept column and >= 1 slope")
    if n_obs <= n_cols:
        raise ValueError("need n_obs > n_cols")
    if intercept_col_idx < 0 or intercept_col_idx >= n_cols:
        raise ValueError("intercept_col_idx is out of bounds")
    if not np.all(x_full[:, intercept_col_idx] == 1.0):
        raise ValueError("intercept column must be all ones")

    slope_cols = [idx for idx in range(n_cols) if idx != intercept_col_idx]
    exog: FloatArray = x_full[:, slope_cols]
    constant = np.flatnonzero(np.ptp(exog, axis=0) == 0.0)
    if constant.size:
        raise ValueError(
            f"slope columns {[int(slope_cols[c]) for c in constant]} are constant "
            "(collinear with the implicit intercept)"
        )
    return y, exog, slope_cols, n_cols


def _build_adapter_diagnostics(
    direct_res: IOLSDirectNoFEResult,
    norm: TransformedNormalizationState,
    *,
    tau: float,
    slopes: FloatArray,
    beta_star: FloatArray,
    reason: str | None,
    native_converged: bool,
) -> dict[str, Any]:
    iteration = direct_res.iteration
    detail = direct_res.diagnostics
    backend_details = direct_res.backend_details
    diagnostics: dict[str, Any] = {
        "route": direct_res.backend,
        "tau_target": float(tau),
        "start_policy": "kernel_internal_log1p_ols_on_demeaned_x",
        "tau_fit": direct_res.tau_fit,
        "phase1_tau": direct_res.phase1_tau,
        "phase2_tau": direct_res.phase2_tau,
        "tau2_anchor_used": False,
        "tau_continuation_used": False,
        "iteration_failure_reason": reason,
        "final_rho": iteration["final_rho"],
        "final_criterion_star": iteration["final_criterion"],
        "anderson_m_final": iteration["anderson_m_final"],
        "native_iteration_counter_sum": int(iteration["iterations"]),
        "phase1_attempts": int(detail["phase1_attempts"]),
        "phase1_accepted": int(detail["phase1_accepted"]),
        "phase1_rejected": int(detail["phase1_rejected"]),
        "final_step_post_convergence_residual_star": detail[
            "post_convergence_residual"
        ],
        "final_step_slope_score_max_abs_per_n_star": detail[
            "post_convergence_score_residual"
        ],
        "final_step_intercept_score_abs_per_n": detail[
            "post_convergence_fe_score_residual"
        ],
        "final_step_score_validation_status": detail["score_validation_status"],
        "phase2_count_trustworthy": True,
        "normalization_condition_before": norm.condition_before,
        "normalization_condition_after": norm.condition_after,
        "back_transform_residual": back_transform_residual(slopes, beta_star, norm),
        "failure_class": (
            "none" if native_converged else classify_failure_reason(reason)
        ),
    }
    for key in BACKEND_DETAIL_KEYS:
        if key in backend_details:
            value = backend_details[key]
            if value is None or isinstance(value, bool | int | str):
                diagnostics[key] = value
            else:
                diagnostics[key] = float(value)
    trace = backend_details.get("phase2_attempt_trace")
    if trace is not None:
        diagnostics["phase2_attempt_trace"] = tuple(dict(item) for item in trace)
    return diagnostics


def _reconstruct_full_theta(
    y: FloatArray,
    exog: FloatArray,
    slopes: FloatArray,
    direct_res: IOLSDirectNoFEResult,
    *,
    tau: float,
    intercept_col_idx: int,
    slope_cols: list[int],
    diagnostics: dict[str, Any],
) -> tuple[FloatArray | None, FloatArray | None, float]:
    n_cols = len(slope_cols) + 1
    theta: FloatArray | None = None
    linear: FloatArray | None = None
    intercept = float("nan")
    if np.all(np.isfinite(slopes)) and np.isfinite(direct_res.intercept):
        base: FloatArray = exog @ slopes
        intercept = float(
            _profile_intercept_adjustment(endog=y, linear=base, tau=float(tau))
        )
        candidate_linear: FloatArray = base + intercept
        candidate_theta: FloatArray = np.empty(n_cols, dtype=np.float64)
        candidate_theta[intercept_col_idx] = intercept
        for slope_pos, col_idx in enumerate(slope_cols):
            candidate_theta[col_idx] = slopes[slope_pos]
        if np.all(np.isfinite(candidate_theta)) and np.all(
            np.isfinite(candidate_linear)
        ):
            theta = candidate_theta
            linear = candidate_linear
            if direct_res.linear is not None:
                diagnostics["fitted_index_consistency_max_abs"] = float(
                    np.max(np.abs(np.asarray(direct_res.linear) - candidate_linear))
                )
    return theta, linear, intercept


def run_iols_nofe(
    *,
    endog: npt.ArrayLike,
    exog_full: npt.ArrayLike,
    tau: float,
    control: IOLSKernelControl,
    backend: DirectNoFEBackend = "auto",
    rho_grid_policy: str = "curvature_step_grid",
    intercept_col_idx: int = 0,
    scale_columns: bool = True,
) -> IOLSAdapterResult:
    """Run the direct fixed-tau no-FE IOLS solve on ``exog_full``."""
    y, exog, slope_cols, _ = _extract_nofe_slope_design(
        endog, exog_full, intercept_col_idx
    )
    norm = build_transformed_normalization_state(
        exog,
        mode="scale" if scale_columns else "none",
    )
    exog_star = apply_transformed_normalization(exog, norm)
    core_control = dataclasses.replace(control, transformed_design_normalization="none")

    direct_res = solve_nofe_direct_fixed_tau(
        endog=y,
        exog_slopes=exog_star,
        tau=float(tau),
        backend=backend,
        control=core_control,
        rho_grid_policy=rho_grid_policy,
    )

    iteration = direct_res.iteration
    detail = direct_res.diagnostics
    backend_details = direct_res.backend_details
    reason: str | None = direct_res.failure_reason or iteration.get("failure_reason")
    native_converged = bool(direct_res.converged)
    beta_star: FloatArray = np.asarray(direct_res.beta, dtype=np.float64)
    slopes: FloatArray = np.asarray(
        back_transform_beta(beta_star, norm), dtype=np.float64
    )
    phase1_evals = int(backend_details.get("phase1_evaluations", 0))
    phase2_evals = int(detail["phase2_evaluations"])

    diagnostics = _build_adapter_diagnostics(
        direct_res,
        norm,
        tau=tau,
        slopes=slopes,
        beta_star=beta_star,
        reason=reason,
        native_converged=native_converged,
    )
    theta, linear, intercept = _reconstruct_full_theta(
        y,
        exog,
        slopes,
        direct_res,
        tau=tau,
        intercept_col_idx=intercept_col_idx,
        slope_cols=slope_cols,
        diagnostics=diagnostics,
    )

    status = "converged" if native_converged else f"direct_nonconvergence:{reason}"
    if theta is None:
        status += ":nonfinite_beta_or_index"
        diagnostics["failure_class"] = "nonfinite_beta_or_index"

    return IOLSAdapterResult(
        theta=theta,
        intercept=intercept,
        slopes=slopes,
        slopes_star=beta_star.copy(),
        linear=linear,
        native_converged=native_converged,
        native_status=status,
        failure_reason=reason,
        failure_class=str(diagnostics["failure_class"]),
        budget_exhausted=reason in CAP_FAILURE_REASONS,
        phase1_iterations=phase1_evals,
        phase2_iterations=phase2_evals,
        rho_restarts=int(iteration["restarts"]),
        requested_backend=direct_res.requested_backend,
        preferred_backend=direct_res.preferred_backend,
        actual_backend=direct_res.backend,
        fallback_used=direct_res.fallback_used,
        availability_reason=direct_res.availability_reason,
        normalization=norm,
        diagnostics=diagnostics,
        direct_result=direct_res,
    )


__all__ = [
    "ADAPTER_VERSION",
    "BACKEND_DETAIL_KEYS",
    "CAP_FAILURE_REASONS",
    "NUMERICAL_FAILURE_REASONS",
    "SUPPORT_FAILURE_REASONS",
    "IOLSAdapterResult",
    "TransformedNormalizationState",
    "_gram_condition",
    "apply_transformed_normalization",
    "back_transform_beta",
    "back_transform_residual",
    "build_transformed_normalization_state",
    "classify_failure_reason",
    "run_iols_nofe",
]
