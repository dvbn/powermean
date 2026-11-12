"""Direct fixed-tau no-FE IOLS solver kernel and dependency closure."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import (
    FloatArray,
    IntArray,
    exp_with_float64_overflow_guard,
)
from powermean.solve.iols.fe import demean, profile_fe
from powermean.solve.iols.kernel_grid import (
    _adapt_anderson_m,
    _criterion_stagnated,
    _delta_base_for_anchor,
    _direct_policy_rho_path,
    _direct_profiled_curvature_l0,
    _direct_progress_trace_fields,
    _direct_step_rho_grid,
    _effective_anderson_m,
    _focus_rho_path,
    _max_abs_change,
    _record_beta_path_entry,
    _record_criterion_progress,
    _rho_base_for_tau,
    _rho_path_for_linear_state,
    _windowed_contraction_rate,
)
from powermean.solve.iols.kernel_phase import _run_concentrated_fullrecover
from powermean.solve.iols.kernel_records import (
    _BETA_PATH_RETAINED,
    _DIRECT_KAPPA_WINDOW,
    _DIRECT_NOFE_INT_DIAGNOSTICS,
    DIRECT_KAPPA_MONITOR_POLICY,
    DIRECT_NOFE_FIXED_TAU_ALGORITHM,
    DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS,
    DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS,
    DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS,
    DIRECT_NOFE_FIXED_TAU_RESULT_KEYS,
    DIRECT_NOFE_FIXED_TAU_ROUTE,
    DIRECT_NOFE_FIXED_TAU_SCHEMA,
    DIRECT_NOFE_RUNTIME_VERSION,
    DIRECT_POST_CHECK_POLICY,
    DIRECT_PROGRESS_MONITOR_POLICY,
    DIRECT_QHAT_GUARD_POLICY_NUMPY,
    DIRECT_RESTART_START_POLICIES,
    DIRECT_RHO_GRID_HOOK_VERSION,
    DIRECT_RHO_GRID_POLICIES,
    FULLRECOVER_JOINT_FE_POLICY,
    FULLRECOVER_PROFILED_FE_POLICY,
    KERNEL_BACKEND_CPP,
    KERNEL_BACKEND_PYTHON_REFERENCE,
    KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL,
    KERNEL_FAILURE_CRITERION_EXPLOSION,
    KERNEL_FAILURE_CRITERION_STAGNATION,
    KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH,
    KERNEL_FAILURE_INVALID_PHI_STEP,
    KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED,
    KERNEL_FAILURE_MIN_DTAU_REACHED,
    KERNEL_FAILURE_NONCONTRACTIVE,
    KERNEL_FAILURE_NOT_CONVERGED,
    KERNEL_FAILURE_NOT_STARTED,
    KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE,
    KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED,
    KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED,
    KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP,
    KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE,
    KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL,
    KERNEL_FAILURE_Q_CONVERGENCE_INVALID,
    KERNEL_FAILURE_REASONS,
    KERNEL_FAILURE_TAU2_ANCHOR_FAILED,
    KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
    TAU2_BOUNDARY_DIRECTION_TOL,
    TAU2_BOUNDARY_FUTURE_POLICY,
    TAU2_BOUNDARY_SEPARATION_CODE,
    TAU2_BOUNDARY_SUPPORT_POLICY,
    TAU2_JOINT_HDFE_SUPPORT_CHECK,
    TAU2_RECOVERED_FE_SUPPORT_CHECK,
    TRANSFORMED_NORMALIZATION_ABS_BACKSTOP,
    TRANSFORMED_NORMALIZATION_REL_FLOOR,
    IOLSIterationDiagnostics,
    IOLSKernelControl,
    KernelTopology,
    ScoreValidationPolicy,
    _ConcentratedSolve,
    _control_to_cpp_mapping,
    _control_with_anderson_m,
    _direct_nofe_validate_control,
    _direct_nofe_validated_inputs,
    _FullrecoverContext,
    _FullrecoverInvariantState,
    _FullrecoverProblem,
    _transformed_normalization_mode,
)
from powermean.solve.iols.kernel_state import (
    _build_fullrecover_context,
    _demean_for_groups,
    _demean_singleton_intercept,
    _fullrecover_fe_policy,
    _fullrecover_invariant_state,
    _has_recovered_fe_support_scope,
    _initial_linear_state,
    _is_singleton_intercept_group,
    _logsumexp_all,
    _profile_intercept_adjustment,
    _profile_linear,
    _recover_fe_linear,
    _recovered_fe_tau2_boundary_separation,
    _uses_joint_hdfe_state,
)
from powermean.solve.iols.kernel_step import (
    _phi_step_fullrecover,
    _profiled_fe_score_residual,
    _profiled_slope_score_residual,
)


def _phase2_initialization_usable(
    *,
    context: _FullrecoverContext,
    beta: FloatArray,
    linear: FloatArray | None,
) -> bool:
    if not np.all(np.isfinite(beta)):
        return False
    if linear is None:
        linear_current = _initial_linear_state(context, beta)
    else:
        linear_current = np.asarray(linear, dtype=np.float64)
    if not np.all(np.isfinite(linear_current)):
        return False
    rho_path, _ = _rho_path_for_linear_state(
        linear=linear_current,
        tau=context.tau,
        rho_hint=None,
        control=context.control,
    )
    for rho in rho_path:
        _, _, valid = _phi_step_fullrecover(
            context=context,
            beta=beta,
            linear=linear_current,
            phase=1,
            param=float(rho),
        )
        if valid:
            return True
    return False


def _concentrated_fullrecover(
    problem: _FullrecoverProblem,
    *,
    beta_init: FloatArray | None = None,
    linear_init: FloatArray | None = None,
    rho_hint: float | None = None,
    score_validation_policy: ScoreValidationPolicy = "diagnostic",
    direct_nofe_fixed_tau: bool = False,
    direct_rho_grid_policy: str = "legacy",
) -> _ConcentratedSolve:
    def phi_step(
        *,
        context: _FullrecoverContext,
        beta: FloatArray,
        linear: FloatArray | None,
        phase: int,
        param: float,
    ) -> tuple[FloatArray, FloatArray | None, bool]:
        return _phi_step_fullrecover(
            context=context, beta=beta, linear=linear, phase=phase, param=param
        )

    def usable(
        *,
        context: _FullrecoverContext,
        beta: FloatArray,
        linear: FloatArray | None,
    ) -> bool:
        return _phase2_initialization_usable(context=context, beta=beta, linear=linear)

    return _run_concentrated_fullrecover(
        problem,
        beta_init=beta_init,
        linear_init=linear_init,
        rho_hint=rho_hint,
        score_validation_policy=score_validation_policy,
        direct_nofe_fixed_tau=direct_nofe_fixed_tau,
        direct_rho_grid_policy=direct_rho_grid_policy,
        callbacks=(phi_step, usable),
    )


def _solve_nofe_direct_fixed_tau(
    *,
    endog: npt.ArrayLike,
    exog_slopes: npt.ArrayLike,
    tau: float,
    control: IOLSKernelControl,
    rho_grid_policy: str = "legacy",
) -> dict[str, Any]:
    """Direct fixed-tau noFE iOLS solve (NumPy reference)."""
    if rho_grid_policy not in DIRECT_RHO_GRID_POLICIES:
        raise ValueError(f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}")
    y, x, tau_value = _direct_nofe_validated_inputs(endog, exog_slopes, tau)
    _direct_nofe_validate_control(control)
    groups: tuple[IntArray, ...] = (np.zeros(y.shape[0], dtype=np.int32),)
    counts = (1,)
    problem = _FullrecoverProblem(
        endog=y,
        exog=x,
        groups=groups,
        counts=counts,
        tau=tau_value,
        control=control,
    )
    solve = _concentrated_fullrecover(
        problem,
        beta_init=None,
        rho_hint=None,
        direct_nofe_fixed_tau=True,
        direct_rho_grid_policy=rho_grid_policy,
    )
    beta: FloatArray = np.array(solve.beta, dtype=np.float64, copy=True)
    intercept = float("nan")
    linear: FloatArray | None = None
    if np.all(np.isfinite(beta)):
        base: FloatArray = x @ beta
        intercept = _profile_intercept_adjustment(endog=y, linear=base, tau=tau_value)
        candidate: FloatArray = base + intercept
        if np.all(np.isfinite(candidate)):
            linear = candidate
    iteration = solve.iteration
    details = dict(iteration.details)
    diagnostics: dict[str, Any] = {}
    for key in DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS:
        if key in _DIRECT_NOFE_INT_DIAGNOSTICS:
            diagnostics[key] = int(details.get(key, 0))
        elif key == "score_validation_status":
            diagnostics[key] = str(details.get(key, "not_checked"))
        elif key == "score_warning_codes":
            diagnostics[key] = tuple(str(code) for code in details.get(key, ()))
        else:
            diagnostics[key] = float(details.get(key, float("nan")))
    return {
        "schema": DIRECT_NOFE_FIXED_TAU_SCHEMA,
        "route": DIRECT_NOFE_FIXED_TAU_ROUTE,
        "backend": "numpy",
        "kernel_backend": KERNEL_BACKEND_PYTHON_REFERENCE,
        "algorithm": DIRECT_NOFE_FIXED_TAU_ALGORITHM,
        "intercept_policy": "profiled_singleton_intercept",
        "tau_requested": tau_value,
        "tau_fit": tau_value,
        "phase1_tau": tau_value,
        "phase2_tau": tau_value,
        "tau2_anchor_used": False,
        "tau_continuation_used": False,
        "converged": bool(iteration.converged),
        "failure_reason": iteration.failure_reason,
        "beta": beta,
        "intercept": float(intercept),
        "linear": linear,
        "iteration": {
            "converged": bool(iteration.converged),
            "phase": str(iteration.phase),
            "iterations": int(iteration.iterations),
            "restarts": int(iteration.restarts),
            "final_rho": iteration.final_rho,
            "rho_path": tuple(float(value) for value in iteration.rho_path),
            "final_criterion": iteration.final_criterion,
            "anderson_m_initial": int(iteration.anderson_m_initial),
            "anderson_m_final": int(iteration.anderson_m_final),
            "failure_reason": iteration.failure_reason,
        },
        "diagnostics": diagnostics,
        "control": {
            key: getattr(control, key) for key in DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS
        },
        "backend_details": {
            **{
                key: value
                for key, value in details.items()
                if key not in DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS
            },
            **(
                {
                    "pilot_deltas": control._pilot_deltas,
                    "pilot_grid_input": "exact_vector",
                }
                if control._pilot_deltas is not None
                else {}
            ),
            "runtime_version": DIRECT_NOFE_RUNTIME_VERSION,
            "anderson_history_policy": "grows_with_adapted_m",
            "kappa_window_first_entry": 4,
            "qhat_check": DIRECT_QHAT_GUARD_POLICY_NUMPY,
        },
    }


__all__ = [
    "DIRECT_KAPPA_MONITOR_POLICY",
    "DIRECT_NOFE_FIXED_TAU_ALGORITHM",
    "DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS",
    "DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS",
    "DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS",
    "DIRECT_NOFE_FIXED_TAU_RESULT_KEYS",
    "DIRECT_NOFE_FIXED_TAU_ROUTE",
    "DIRECT_NOFE_FIXED_TAU_SCHEMA",
    "DIRECT_NOFE_RUNTIME_VERSION",
    "DIRECT_POST_CHECK_POLICY",
    "DIRECT_PROGRESS_MONITOR_POLICY",
    "DIRECT_QHAT_GUARD_POLICY_NUMPY",
    "DIRECT_RESTART_START_POLICIES",
    "DIRECT_RHO_GRID_HOOK_VERSION",
    "DIRECT_RHO_GRID_POLICIES",
    "FULLRECOVER_JOINT_FE_POLICY",
    "FULLRECOVER_PROFILED_FE_POLICY",
    "KERNEL_BACKEND_CPP",
    "KERNEL_BACKEND_PYTHON_REFERENCE",
    "KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL",
    "KERNEL_FAILURE_CRITERION_EXPLOSION",
    "KERNEL_FAILURE_CRITERION_STAGNATION",
    "KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH",
    "KERNEL_FAILURE_INVALID_PHI_STEP",
    "KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED",
    "KERNEL_FAILURE_MIN_DTAU_REACHED",
    "KERNEL_FAILURE_NONCONTRACTIVE",
    "KERNEL_FAILURE_NOT_CONVERGED",
    "KERNEL_FAILURE_NOT_STARTED",
    "KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE",
    "KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED",
    "KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED",
    "KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP",
    "KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE",
    "KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL",
    "KERNEL_FAILURE_Q_CONVERGENCE_INVALID",
    "KERNEL_FAILURE_REASONS",
    "KERNEL_FAILURE_TAU2_ANCHOR_FAILED",
    "KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO",
    "TAU2_BOUNDARY_DIRECTION_TOL",
    "TAU2_BOUNDARY_FUTURE_POLICY",
    "TAU2_BOUNDARY_SEPARATION_CODE",
    "TAU2_BOUNDARY_SUPPORT_POLICY",
    "TAU2_JOINT_HDFE_SUPPORT_CHECK",
    "TAU2_RECOVERED_FE_SUPPORT_CHECK",
    "TRANSFORMED_NORMALIZATION_ABS_BACKSTOP",
    "TRANSFORMED_NORMALIZATION_REL_FLOOR",
    "_BETA_PATH_RETAINED",
    "_DIRECT_KAPPA_WINDOW",
    "IOLSIterationDiagnostics",
    "IOLSKernelControl",
    "KernelTopology",
    "ScoreValidationPolicy",
    "_ConcentratedSolve",
    "_FullrecoverContext",
    "_FullrecoverInvariantState",
    "_FullrecoverProblem",
    "_adapt_anderson_m",
    "_build_fullrecover_context",
    "_concentrated_fullrecover",
    "_control_to_cpp_mapping",
    "_control_with_anderson_m",
    "_criterion_stagnated",
    "_delta_base_for_anchor",
    "_demean_for_groups",
    "_demean_singleton_intercept",
    "_direct_nofe_validate_control",
    "_direct_nofe_validated_inputs",
    "_direct_policy_rho_path",
    "_direct_profiled_curvature_l0",
    "_direct_progress_trace_fields",
    "_direct_step_rho_grid",
    "_effective_anderson_m",
    "_focus_rho_path",
    "_fullrecover_fe_policy",
    "_fullrecover_invariant_state",
    "_has_recovered_fe_support_scope",
    "_initial_linear_state",
    "_is_singleton_intercept_group",
    "_logsumexp_all",
    "_max_abs_change",
    "_phase2_initialization_usable",
    "_phi_step_fullrecover",
    "_profile_intercept_adjustment",
    "_profile_linear",
    "_profiled_fe_score_residual",
    "_profiled_slope_score_residual",
    "_record_beta_path_entry",
    "_record_criterion_progress",
    "_recover_fe_linear",
    "_recovered_fe_tau2_boundary_separation",
    "_rho_base_for_tau",
    "_rho_path_for_linear_state",
    "_solve_nofe_direct_fixed_tau",
    "_transformed_normalization_mode",
    "_uses_joint_hdfe_state",
    "_windowed_contraction_rate",
    "demean",
    "exp_with_float64_overflow_guard",
    "profile_fe",
]
