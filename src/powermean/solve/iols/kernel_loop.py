"""Outer Phase-1/Phase-2 loop orchestration and post-convergence checks for IOLS."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.kernel_grid import (
    _direct_progress_trace_fields,
)
from powermean.solve.iols.kernel_records import (
    DIRECT_RHO_GRID_POLICIES,
    KERNEL_FAILURE_NOT_STARTED,
    KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP,
    KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE,
    KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL,
    TAU2_BOUNDARY_SEPARATION_CODE,
    IOLSIterationDiagnostics,
    ScoreValidationPolicy,
    _ConcentratedSolve,
    _FullrecoverContext,
    _FullrecoverProblem,
)
from powermean.solve.iols.kernel_state import (
    _demean_for_groups,
    _fullrecover_fe_policy,
    _initial_linear_state,
    _is_singleton_intercept_group,
    _recovered_fe_tau2_boundary_separation,
)
from powermean.solve.iols.kernel_step import (
    PhiStepCallable,
    _AttemptState,
    _OuterLoopState,
    _PhaseStepState,
    _profiled_fe_score_residual,
    _profiled_slope_score_residual,
)


class UsableCallable(Protocol):
    """Keyword contract for a Phase-2 initialization check."""

    def __call__(
        self,
        *,
        context: _FullrecoverContext,
        beta: FloatArray,
        linear: FloatArray | None,
    ) -> bool:
        """Check the supplied state without accepting arbitrary keywords."""
        ...


def _check_post_convergence_scores(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
) -> None:
    score_residual = _profiled_slope_score_residual(
        context=context,
        beta=outer.beta,
        linear=outer.linear if attempt.joint_hdfe else None,
    )
    step.score_tol = max(step.eps_current * 10.0, 1e-8)
    step.post_convergence_score_residual = float(score_residual)
    step.score_exceeds_tolerance = bool(
        not np.isfinite(score_residual) or score_residual > step.score_tol
    )
    if step.score_exceeds_tolerance:
        step.score_warning_codes.append("post_convergence_score_residual")
        step.score_warning_messages.append(
            "Slope score residual exceeds tolerance at this tau-path state."
        )
        step.score_validation_status = "diagnostic_warning"
    elif step.score_validation_status == "not_checked":
        step.score_validation_status = "passed"

    fe_score_residual = _profiled_fe_score_residual(
        context=context,
        beta=outer.beta,
        linear=outer.linear if attempt.joint_hdfe else None,
    )
    step.score_tol = max(step.eps_current * 10.0, 1e-8)
    step.post_convergence_fe_score_residual = float(fe_score_residual)
    step.fe_score_exceeds_tolerance = bool(
        not np.isfinite(fe_score_residual) or fe_score_residual > step.score_tol
    )
    if step.fe_score_exceeds_tolerance:
        step.score_warning_codes.append("post_convergence_fe_score_residual")
        step.score_warning_messages.append(
            "Recovered-FE score residual exceeds tolerance at this tau-path state."
        )
        step.score_validation_status = "diagnostic_warning"
    elif step.score_validation_status == "not_checked":
        step.score_validation_status = "passed"


def _verify_post_convergence(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    phi_step_fn: PhiStepCallable,
) -> bool:
    converged = (
        not np.isnan(attempt.criterion) and attempt.criterion <= step.eps_current
    )
    step.final_criterion = (
        None if np.isnan(attempt.criterion) else float(attempt.criterion)
    )
    if not converged:
        return False

    beta_check, linear_check, valid = phi_step_fn(
        context=context,
        beta=outer.beta,
        linear=outer.linear,
        phase=step.phase,
        param=step.param,
    )
    if valid:
        residual = float(np.max(np.abs(beta_check - outer.beta)))
        if attempt.joint_hdfe and linear_check is not None:
            residual = max(residual, float(np.max(np.abs(linear_check - outer.linear))))
        step.post_convergence_residual = float(residual)
        if residual > step.eps_current * 10.0:
            converged = False
            step.failure_reason = KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL
    elif outer.direct_nofe_fixed_tau:
        converged = False
        step.failure_reason = KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP
        outer.post_convergence_invalid_map_counts[step.phase] += 1
        step.post_convergence_residual = float("nan")
        step.post_convergence_kappa_est = float("nan")

    if converged and step.phase == 1:
        _check_post_convergence_scores(context, outer, step, attempt)

    if (
        converged
        and len(attempt.beta_path) >= 4
        and not (step.phase == 1 and abs(context.tau - 2.0) <= 1e-10)
    ):
        diffs: list[np.float64] = [
            np.max(np.abs(attempt.beta_path[-j] - attempt.beta_path[-j - 1]))
            for j in range(1, min(4, len(attempt.beta_path)))
        ]
        if diffs[1] > 1e-15 and diffs[2] > 1e-15:
            kappa_est = diffs[0] / diffs[1]
            step.post_convergence_kappa_est = float(kappa_est)
            if kappa_est >= 1.0:
                converged = False
                step.failure_reason = KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE
    return converged


def _finish_attempt_or_jump_restart(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    converged: bool,
) -> None:
    if outer.direct_nofe_fixed_tau and step.phase == 1:
        outer.phase2_attempt_trace.append(
            {
                "attempt": int(min(step.restart_index, len(step.rho_path) - 1)),
                "rho": float(step.param),
                "iterations": int(attempt.k),
                "evaluations": int(
                    step.phase2_evaluations - outer.attempt_evaluations_start
                ),
                "failure_reason": (
                    None
                    if converged
                    else (
                        "stopcrit_reached"
                        if not np.isnan(attempt.criterion)
                        and attempt.criterion > step.eps_current
                        else step.failure_reason
                    )
                ),
                "final_criterion": step.final_criterion,
                "best_criterion": float(
                    min(attempt.criterion_history)
                    if attempt.criterion_history
                    else float("nan")
                ),
                "kappa_last": float(step.kappa_monitor_last),
                "start_policy": outer.attempt_start_policy,
                **_direct_progress_trace_fields(
                    attempt.progress_history,
                    aa_m_start=attempt.attempt_aa_m_start,
                    aa_m_end=attempt.aa_m,
                    aa_accepted=attempt.attempt_aa_accepted,
                ),
            }
        )
    if (
        not converged
        and step.phase == 1
        and not step.phase2_budget_exhausted
        and step.restart_index < len(step.rho_path) - 1
    ):
        step.restart_index = min(
            step.restart_index + max(step.restart_index, 5),
            len(step.rho_path) - 1,
        )
        step.param = float(step.rho_path[step.restart_index])
        outer.beta = step.beta_outer_initial.copy()
        outer.linear = step.linear_outer_initial.copy()
        if outer.direct_nofe_fixed_tau:
            outer.attempt_start_policy = "phase1_start"
            if outer.restart_from_best and outer.best_score_beta is not None:
                outer.beta = outer.best_score_beta.copy()
                outer.linear = _initial_linear_state(context, outer.beta)
                outer.phase2_warm_restarts += 1
                outer.attempt_start_policy = "best_score_iterate"
            step.kappa_monitor_last = float("nan")
    else:
        step.verified = True
    step.aa_m_final = attempt.aa_m


def _validate_concentrated_arguments(
    context: _FullrecoverProblem,
    *,
    beta_init: FloatArray | None,
    score_validation_policy: ScoreValidationPolicy,
    direct_nofe_fixed_tau: bool,
    direct_rho_grid_policy: str,
) -> None:
    if score_validation_policy != "diagnostic":
        raise ValueError("score_validation_policy must be 'diagnostic'")
    if direct_rho_grid_policy not in DIRECT_RHO_GRID_POLICIES:
        raise ValueError(f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}")
    if direct_rho_grid_policy != "legacy" and not direct_nofe_fixed_tau:
        raise ValueError(
            "rho_grid_policy other than 'legacy' requires direct_nofe_fixed_tau."
        )
    if direct_nofe_fixed_tau and (
        beta_init is not None
        or not _is_singleton_intercept_group(context.groups, context.counts)
    ):
        raise ValueError(
            "Direct fixed-tau Phase 1 + Phase 2 is authorized only for the "
            "noFE singleton-intercept route without beta_init."
        )
    if (
        beta_init is None
        and abs(float(context.tau) - 2.0) > 1e-10
        and not direct_nofe_fixed_tau
    ):
        raise ValueError(
            "Non-anchor iOLS path steps require beta_init from the tau=2 "
            "Phase 1 + Phase 2 anchor."
        )


def _initialize_outer_state(
    context: _FullrecoverContext,
    *,
    delta_grid: FloatArray,
    beta_init: FloatArray | None,
    linear_init: FloatArray | None,
    rho_hint: float | None,
    score_validation_policy: ScoreValidationPolicy,
    direct_nofe_fixed_tau: bool,
    direct_rho_grid_policy: str,
) -> _OuterLoopState:
    if beta_init is not None:
        beta: FloatArray = beta_init.copy()
        linear: FloatArray = (
            np.asarray(linear_init, dtype=np.float64).copy()
            if linear_init is not None
            else _initial_linear_state(context, beta)
        )
        n_outer = 1
    else:
        y_init = _demean_for_groups(
            np.log(context.endog + 1.0),
            context.groups,
            context.counts,
            tol=float(context.control.fe_demean_tol),
            maxiter=int(context.control.fe_demean_maxiter),
        )
        beta = context.projection @ y_init
        linear = _initial_linear_state(context, beta)
        n_outer = len(delta_grid) + 1
    final_diag = IOLSIterationDiagnostics(
        converged=False,
        phase="not_started",
        iterations=0,
        restarts=0,
        final_rho=None,
        rho_path=(),
        final_criterion=None,
        anderson_m_initial=int(context.control.anderson_m),
        anderson_m_final=int(context.control.anderson_m),
        failure_reason=KERNEL_FAILURE_NOT_STARTED,
    )
    return _OuterLoopState(
        delta_grid=delta_grid,
        n_outer=n_outer,
        beta=beta,
        linear=linear,
        beta_initial=beta.copy(),
        linear_initial=linear.copy(),
        phase1_accepted_beta=beta.copy(),
        phase1_accepted_linear=linear.copy(),
        direct_nofe_fixed_tau=direct_nofe_fixed_tau,
        direct_rho_grid_policy=direct_rho_grid_policy,
        rho_hint=rho_hint,
        score_validation_policy=score_validation_policy,
        restart_from_best=(
            str(context.control.direct_restart_start) == "best_score_iterate"
        ),
        final_diag=final_diag,
    )


def _check_tau2_support(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    *,
    beta_init: FloatArray | None,
) -> _ConcentratedSolve | None:
    if beta_init is None and abs(float(context.tau) - 2.0) <= 1e-10:
        support_details = _recovered_fe_tau2_boundary_separation(context)
        if bool(support_details.get("separation_detected", False)):
            support_details.update(
                {
                    "phase1_attempts": 0,
                    "phase1_accepted": 0,
                    "phase1_rejected": 0,
                    "rho_candidate_count": 0,
                    "fe_state_policy": _fullrecover_fe_policy(context.groups),
                    "anderson_policy": "not_started_boundary_support_failure",
                    "failure_reason": TAU2_BOUNDARY_SEPARATION_CODE,
                }
            )
            return _ConcentratedSolve(
                beta=np.asarray(outer.beta, dtype=np.float64),
                linear=np.asarray(outer.linear, dtype=np.float64),
                iteration=IOLSIterationDiagnostics(
                    converged=False,
                    phase="support_check",
                    iterations=0,
                    restarts=0,
                    final_rho=None,
                    rho_path=(),
                    final_criterion=None,
                    anderson_m_initial=int(context.control.anderson_m),
                    anderson_m_final=0,
                    failure_reason=TAU2_BOUNDARY_SEPARATION_CODE,
                    details=support_details,
                ),
            )
    return None
