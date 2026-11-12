"""Outer-phase setup, diagnostic assembly, and concentrated fullrecover driver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.kernel_attempt import _run_attempt_loop
from powermean.solve.iols.kernel_grid import (
    _delta_base_for_anchor,
    _direct_policy_rho_path,
    _direct_profiled_curvature_l0,
    _rho_path_for_linear_state,
)
from powermean.solve.iols.kernel_loop import (
    UsableCallable,
    _check_tau2_support,
    _finish_attempt_or_jump_restart,
    _initialize_outer_state,
    _validate_concentrated_arguments,
    _verify_post_convergence,
)
from powermean.solve.iols.kernel_records import (
    DIRECT_KAPPA_MONITOR_POLICY,
    DIRECT_POST_CHECK_POLICY,
    DIRECT_PROGRESS_MONITOR_POLICY,
    DIRECT_QHAT_GUARD_POLICY_NUMPY,
    DIRECT_RHO_GRID_HOOK_VERSION,
    KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH,
    KERNEL_FAILURE_NOT_CONVERGED,
    KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE,
    IOLSIterationDiagnostics,
    ScoreValidationPolicy,
    _ConcentratedSolve,
    _FullrecoverContext,
    _FullrecoverProblem,
)
from powermean.solve.iols.kernel_state import (
    _build_fullrecover_context,
    _fullrecover_fe_policy,
    _initial_linear_state,
    _uses_joint_hdfe_state,
)
from powermean.solve.iols.kernel_step import (
    PhiStepCallable,
    _new_attempt_state,
    _OuterLoopState,
    _PhaseStepState,
)


def _prepare_outer_step(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    outer_index: int,
    *,
    beta_init: FloatArray | None,
) -> tuple[_PhaseStepState | None, _ConcentratedSolve | None]:
    if np.any(np.isnan(outer.beta)):
        outer.beta = outer.beta_initial.copy()
        outer.linear = outer.linear_initial.copy()
    beta_outer_initial = outer.beta.copy()
    linear_outer_initial = outer.linear.copy()

    if outer_index < len(outer.delta_grid):
        return (
            _PhaseStepState(
                phase=0,
                param=float(outer.delta_grid[outer_index]),
                eps_current=max(float(context.control.eps) * 10.0, 1e-5),
                stopcrit_current=min(
                    int(context.control.stopcrit),
                    int(context.control.max_phase1_iterations),
                ),
                rho_path=np.array([], dtype=np.float64),
                rho_details={},
                beta_outer_initial=beta_outer_initial,
                linear_outer_initial=linear_outer_initial,
                aa_m_final=int(context.control.anderson_m),
            ),
            None,
        )

    if beta_init is None:
        outer.beta = outer.phase1_accepted_beta.copy()
        outer.linear = outer.phase1_accepted_linear.copy()
    linear_for_rho = np.asarray(outer.linear, dtype=np.float64)
    if outer.direct_nofe_fixed_tau and outer.direct_rho_grid_policy != "legacy":
        if outer.direct_rho_grid_policy == "curvature_step_grid":
            outer.hook_curvature = _direct_profiled_curvature_l0(
                context=context,
                beta=outer.beta,
            )
        rho_path, rho_details = _direct_policy_rho_path(
            policy=outer.direct_rho_grid_policy,
            linear=linear_for_rho,
            tau=context.tau,
            control=context.control,
            curvature=outer.hook_curvature,
        )
        outer.hook_alpha0 = float(rho_details.pop("_rho_grid_alpha0"))
        outer.hook_alpha_min = float(rho_details.pop("_rho_grid_alpha_min"))
        outer.hook_clipped = bool(rho_details.pop("_rho_grid_clipped"))
        outer.hook_values = tuple(float(v) for v in rho_path)
    else:
        rho_path, rho_details = _rho_path_for_linear_state(
            linear=linear_for_rho,
            tau=context.tau,
            rho_hint=outer.rho_hint,
            control=context.control,
        )
    if rho_path.size == 0:
        rho_details.update(
            {
                "phase1_attempts": int(outer.phase1_attempts),
                "phase1_accepted": int(outer.phase1_accepted),
                "phase1_rejected": int(outer.phase1_rejected),
            }
        )
        if outer.direct_nofe_fixed_tau:
            rho_details["phase1_evaluations"] = int(outer.phase1_evaluations)
        return (
            None,
            _ConcentratedSolve(
                beta=np.asarray(outer.beta, dtype=np.float64),
                linear=np.asarray(outer.linear, dtype=np.float64),
                iteration=IOLSIterationDiagnostics(
                    converged=False,
                    phase="P2",
                    iterations=0,
                    restarts=0,
                    final_rho=None,
                    rho_path=(),
                    final_criterion=None,
                    anderson_m_initial=int(context.control.anderson_m),
                    anderson_m_final=int(context.control.anderson_m),
                    failure_reason=KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH,
                    details=rho_details,
                ),
            ),
        )
    return (
        _PhaseStepState(
            phase=1,
            param=float(rho_path[0]),
            eps_current=float(context.control.eps),
            stopcrit_current=int(context.control.stopcrit),
            rho_path=rho_path,
            rho_details=rho_details,
            beta_outer_initial=outer.beta.copy(),
            linear_outer_initial=outer.linear.copy(),
            aa_m_final=int(context.control.anderson_m),
        ),
        None,
    )


def _common_phase_details(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    *,
    include_phase1_evals: bool = False,
) -> dict[str, Any]:
    return {
        "phase1_attempts": int(outer.phase1_attempts),
        "phase1_accepted": int(outer.phase1_accepted),
        "phase1_rejected": int(outer.phase1_rejected),
        "fe_state_policy": _fullrecover_fe_policy(context.groups),
        "anderson_policy": (
            "disabled_for_joint_hdfe"
            if _uses_joint_hdfe_state(context.groups)
            else "beta_anderson"
        ),
        "qhat_last": float(step.qhat_last),
        "qhat_invalid_count": int(step.qhat_invalid_count),
        "kappa_monitor_last": float(step.kappa_monitor_last),
        "kappa_monitor_max": float(step.kappa_monitor_max),
        "kappa_monitor_checks": int(step.kappa_monitor_checks),
        **(
            {"phase1_evaluations": int(outer.phase1_evaluations)}
            if include_phase1_evals and outer.direct_nofe_fixed_tau
            else {}
        ),
        "phase2_evaluations": int(step.phase2_evaluations),
        "max_phase2_evaluations": int(context.control.max_phase2_evaluations),
        "last_finite_criterion": float(step.last_finite_criterion),
        "best_finite_criterion": float(step.best_finite_criterion),
        "post_convergence_residual": float(step.post_convergence_residual),
        "post_convergence_kappa_est": float(step.post_convergence_kappa_est),
        "post_convergence_score_residual": float(step.post_convergence_score_residual),
        "post_convergence_fe_score_residual": float(
            step.post_convergence_fe_score_residual
        ),
        "score_tolerance": float(step.score_tol),
        "score_validation_policy": str(outer.score_validation_policy),
        "score_validation_status": str(step.score_validation_status),
        "score_warning_codes": tuple(step.score_warning_codes),
        "score_warning_messages": tuple(step.score_warning_messages),
        "post_convergence_score_exceeds_tolerance": bool(step.score_exceeds_tolerance),
        "post_convergence_fe_score_exceeds_tolerance": bool(
            step.fe_score_exceeds_tolerance
        ),
    }


def _finalize_phase0_outer(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    converged: bool,
    usable_fn: UsableCallable,
) -> tuple[bool, Mapping[str, Any]]:
    outer.phase1_attempts += 1
    candidate_usable = bool(
        converged
        and usable_fn(
            context=context,
            beta=outer.beta,
            linear=outer.linear,
        )
    )
    if candidate_usable:
        outer.phase1_accepted += 1
        outer.phase1_accepted_beta = outer.beta.copy()
        outer.phase1_accepted_linear = outer.linear.copy()
    else:
        outer.phase1_rejected += 1
        outer.beta = outer.phase1_accepted_beta.copy()
        outer.linear = outer.phase1_accepted_linear.copy()
        if converged and step.failure_reason is None:
            step.failure_reason = KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE
    return (
        candidate_usable,
        _common_phase_details(context, outer, step, include_phase1_evals=True),
    )


def _finalize_phase1_outer(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    converged: bool,
) -> tuple[bool, Mapping[str, Any]]:
    if not converged and not outer.direct_nofe_fixed_tau:
        outer.beta = step.beta_outer_initial.copy()
        outer.linear = step.linear_outer_initial.copy()
    if outer.direct_nofe_fixed_tau:
        nonconverged_policy = "last_attempt_final_iterate"
        if outer.restart_from_best:
            nonconverged_policy = "best_score_iterate"
            if not converged and outer.best_score_beta is not None:
                outer.beta = outer.best_score_beta.copy()
                outer.linear = _initial_linear_state(context, outer.beta)
        step.rho_details.update(
            {
                "nonconverged_iterate_policy": nonconverged_policy,
                "phase1_evaluations": int(outer.phase1_evaluations),
                "direct_restart_start": str(context.control.direct_restart_start),
                "phase2_attempts": len(outer.phase2_attempt_trace),
                "phase2_warm_restarts": int(outer.phase2_warm_restarts),
                "best_score_norm": float(outer.best_score_norm),
                "best_score_rho": float(outer.best_score_rho),
                "phase2_attempt_trace": tuple(outer.phase2_attempt_trace),
                "kappa_coordinate_row_max": float(outer.kappa_coordinate_row_max),
                "kappa_monitor_policy": DIRECT_KAPPA_MONITOR_POLICY,
                "qhat_guard_policy": DIRECT_QHAT_GUARD_POLICY_NUMPY,
                "progress_monitor_policy": DIRECT_PROGRESS_MONITOR_POLICY,
                "post_check_policy": DIRECT_POST_CHECK_POLICY,
                "post_convergence_invalid_map_count": int(
                    outer.post_convergence_invalid_map_counts[1]
                ),
                "phase1_post_convergence_invalid_map_count": int(
                    outer.post_convergence_invalid_map_counts[0]
                ),
                "rho_grid_policy": str(outer.direct_rho_grid_policy),
                "rho_grid_hook_version": DIRECT_RHO_GRID_HOOK_VERSION,
                "rho_grid_count": len(step.rho_path),
                "rho_grid_values": outer.hook_values,
                "rho_grid_alpha0": outer.hook_alpha0,
                "rho_grid_alpha_min": outer.hook_alpha_min,
                "rho_grid_clipped": outer.hook_clipped,
                "rho_curvature_L0": (
                    None if outer.hook_curvature is None else outer.hook_curvature["L0"]
                ),
                "rho_curvature_lambda_min": (
                    None
                    if outer.hook_curvature is None
                    else outer.hook_curvature["lambda_min"]
                ),
                "rho_curvature_wall_ns": (
                    None
                    if outer.hook_curvature is None
                    else int(outer.hook_curvature["wall_ns"])
                ),
                "rho_curvature_pilot": (
                    None if outer.hook_curvature is None else "phase1_accepted_beta"
                ),
            }
        )
    step.rho_details.update(_common_phase_details(context, outer, step))
    return bool(converged), step.rho_details


def _run_concentrated_fullrecover(
    problem: _FullrecoverProblem,
    *,
    beta_init: FloatArray | None = None,
    linear_init: FloatArray | None = None,
    rho_hint: float | None = None,
    score_validation_policy: ScoreValidationPolicy = "diagnostic",
    direct_nofe_fixed_tau: bool = False,
    direct_rho_grid_policy: str = "legacy",
    callbacks: tuple[PhiStepCallable, UsableCallable],
) -> _ConcentratedSolve:
    phi_step_fn, usable_fn = callbacks
    _validate_concentrated_arguments(
        problem,
        beta_init=beta_init,
        score_validation_policy=score_validation_policy,
        direct_nofe_fixed_tau=direct_nofe_fixed_tau,
        direct_rho_grid_policy=direct_rho_grid_policy,
    )
    delta_grid = (
        _delta_base_for_anchor(problem.control)
        if beta_init is None
        else np.array([], dtype=np.float64)
    )
    context = _build_fullrecover_context(
        endog=problem.endog,
        exog=problem.exog,
        groups=problem.groups,
        counts=problem.counts,
        tau=problem.tau,
        control=problem.control,
        invariant_state=problem.invariant_state,
    )
    outer = _initialize_outer_state(
        context,
        delta_grid=delta_grid,
        beta_init=beta_init,
        linear_init=linear_init,
        rho_hint=rho_hint,
        score_validation_policy=score_validation_policy,
        direct_nofe_fixed_tau=direct_nofe_fixed_tau,
        direct_rho_grid_policy=direct_rho_grid_policy,
    )
    support_abort = _check_tau2_support(context, outer, beta_init=beta_init)
    if support_abort is not None:
        return support_abort

    for outer_index in range(outer.n_outer):
        step, early_exit = _prepare_outer_step(
            context, outer, outer_index, beta_init=beta_init
        )
        if early_exit is not None:
            return early_exit
        assert step is not None

        converged = False
        while not step.verified:
            attempt = _new_attempt_state(context, outer, step)
            _run_attempt_loop(context, outer, step, attempt, phi_step_fn)
            converged = _verify_post_convergence(
                context, outer, step, attempt, phi_step_fn
            )
            _finish_attempt_or_jump_restart(context, outer, step, attempt, converged)

        if step.phase == 0:
            diag_converged, diag_details = _finalize_phase0_outer(
                context, outer, step, converged, usable_fn
            )
        else:
            diag_converged, diag_details = _finalize_phase1_outer(
                context, outer, step, converged
            )

        outer.final_diag = IOLSIterationDiagnostics(
            converged=bool(diag_converged),
            phase="P1" if step.phase == 0 else "P2",
            iterations=int(step.iteration_count),
            restarts=int(step.restart_index),
            final_rho=None if step.phase == 0 else float(step.param),
            rho_path=tuple(
                float(value) for value in step.rho_path[: step.restart_index + 1]
            ),
            final_criterion=step.final_criterion,
            anderson_m_initial=int(context.control.anderson_m),
            anderson_m_final=int(step.aa_m_final),
            failure_reason=(
                None
                if diag_converged
                else step.failure_reason or KERNEL_FAILURE_NOT_CONVERGED
            ),
            details=diag_details,
        )
        if bool(diag_converged) and not _uses_joint_hdfe_state(context.groups):
            outer.linear = _initial_linear_state(context, outer.beta)

    return _ConcentratedSolve(
        beta=np.asarray(outer.beta, dtype=np.float64),
        linear=(
            np.asarray(outer.linear, dtype=np.float64)
            if np.all(np.isfinite(outer.linear))
            else None
        ),
        iteration=outer.final_diag,
    )
