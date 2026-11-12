"""Inner fixed-point iteration, Anderson step, monitors, and post-check for IOLS."""

from __future__ import annotations

import numpy as np

from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.kernel_grid import (
    _adapt_anderson_m,
    _direct_progress_trace_fields,
    _record_beta_path_entry,
    _record_criterion_progress,
    _windowed_contraction_rate,
)
from powermean.solve.iols.kernel_records import (
    _DIRECT_KAPPA_WINDOW,
    KERNEL_FAILURE_INVALID_PHI_STEP,
    KERNEL_FAILURE_NONCONTRACTIVE,
    KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED,
    KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED,
    KERNEL_FAILURE_Q_CONVERGENCE_INVALID,
    _FullrecoverContext,
)
from powermean.solve.iols.kernel_state import _initial_linear_state
from powermean.solve.iols.kernel_step import (
    PhiStepCallable,
    _AttemptState,
    _OuterLoopState,
    _PhaseStepState,
)


def _consume_evaluation_budget(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    *,
    alias_exhausted: bool = False,
) -> bool:
    if step.phase == 1:
        step.phase2_evaluations += 1
        if step.phase2_evaluations > int(context.control.max_phase2_evaluations):
            attempt.criterion = np.nan
            step.failure_reason = (
                KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED
                if alias_exhausted
                else KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED
            )
            step.phase2_budget_exhausted = True
            return False
    else:
        outer.phase1_evaluations += 1
    return True


def _record_direct_step_score(
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    beta_phi: FloatArray,
) -> None:
    step_score_norm = float(np.linalg.norm(beta_phi - outer.beta)) * (
        1.0 + float(step.param)
    )
    if np.isfinite(step_score_norm) and step_score_norm < outer.best_score_norm:
        outer.best_score_norm = step_score_norm
        outer.best_score_rho = float(step.param)
        outer.best_score_beta = outer.beta.copy()
    attempt.plain_residual_maxabs = float(np.max(np.abs(beta_phi - outer.beta)))


def _try_anderson_step(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    iterates: tuple[FloatArray, FloatArray],
    *,
    phi_step_fn: PhiStepCallable,
) -> tuple[bool, bool]:
    beta0, beta_phi = iterates
    if attempt.aa_m > 0 and attempt.k > attempt.aa_m + 1:
        f_k: FloatArray = beta_phi - outer.beta
        attempt.aa_x_hist.append(outer.beta.copy())
        attempt.aa_f_hist.append(f_k.copy())
        if len(attempt.aa_f_hist) > attempt.aa_m + 1:
            attempt.aa_x_hist.pop(0)
            attempt.aa_f_hist.pop(0)

        m_cur = len(attempt.aa_f_hist) - 1
        if m_cur >= 1:
            f_mat = np.column_stack(
                [
                    attempt.aa_f_hist[-1] - attempt.aa_f_hist[-(j + 1)]
                    for j in range(1, m_cur + 1)
                ]
            )
            gamma, _, _, _ = np.linalg.lstsq(f_mat, attempt.aa_f_hist[-1], rcond=None)
            beta_aa = beta_phi.copy()
            for j in range(m_cur):
                phi_prev = attempt.aa_x_hist[-(j + 2)] + attempt.aa_f_hist[-(j + 2)]
                beta_aa += gamma[j] * (phi_prev - beta_phi)

            if np.all(np.isfinite(beta_aa)):
                if not _consume_evaluation_budget(
                    context, outer, step, attempt, alias_exhausted=True
                ):
                    return False, True
                beta_aa_phi, _, aa_valid = phi_step_fn(
                    context=context,
                    beta=beta_aa,
                    linear=None,
                    phase=step.phase,
                    param=step.param,
                )
                if aa_valid:
                    res_plain = np.linalg.norm(beta_phi - outer.beta)
                    res_aa = np.linalg.norm(beta_aa_phi - beta_aa)
                    if res_aa < res_plain * 1.1:
                        outer.beta = beta_aa
                        (
                            attempt.beta_path_entries,
                            attempt.kappa_row_max,
                        ) = _record_beta_path_entry(
                            attempt.beta_path,
                            outer.beta,
                            entries_seen=attempt.beta_path_entries,
                            kappa_row_max=attempt.kappa_row_max,
                        )
                        if attempt.k >= 2:
                            attempt.criterion = float(
                                np.max(np.abs(outer.beta - beta0))
                            )
                        attempt.attempt_aa_accepted += 1
                        return True, False
    elif attempt.aa_m > 0:
        f_k = beta_phi - outer.beta
        attempt.aa_x_hist.append(outer.beta.copy())
        attempt.aa_f_hist.append(f_k.copy())
    return False, False


def _advance_iterate_and_criterion(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    previous: tuple[FloatArray, FloatArray],
    *,
    phi_out: tuple[FloatArray, FloatArray | None],
    accepted_aa_step: bool,
) -> None:
    beta0, linear0 = previous
    beta_phi, linear_phi = phi_out
    if not accepted_aa_step:
        outer.beta = beta_phi
        if linear_phi is not None:
            outer.linear = linear_phi
        elif not attempt.joint_hdfe:
            outer.linear = _initial_linear_state(context, outer.beta)
        attempt.beta_path_entries, attempt.kappa_row_max = _record_beta_path_entry(
            attempt.beta_path,
            outer.beta,
            entries_seen=attempt.beta_path_entries,
            kappa_row_max=attempt.kappa_row_max,
        )
    if attempt.k >= 2:
        if not accepted_aa_step:
            beta_criterion = float(np.max(np.abs(outer.beta - beta0)))
            if attempt.joint_hdfe:
                linear_criterion = float(np.max(np.abs(outer.linear - linear0)))
                attempt.criterion = max(beta_criterion, linear_criterion)
            else:
                attempt.criterion = beta_criterion
        direct_progress = outer.direct_nofe_fixed_tau and step.phase == 1
        if direct_progress and np.isfinite(attempt.plain_residual_maxabs):
            attempt.progress_history.append(attempt.plain_residual_maxabs)
        (
            step.last_finite_criterion,
            step.best_finite_criterion,
            criterion_failure,
        ) = _record_criterion_progress(
            attempt.criterion_history,
            attempt.criterion,
            phase=step.phase,
            tau=context.tau,
            eps=step.eps_current,
            max_phase2_evaluations=int(context.control.max_phase2_evaluations),
            bounds=(step.last_finite_criterion, step.best_finite_criterion),
            stagnation_history=attempt.progress_history if direct_progress else None,
        )
        if criterion_failure is not None:
            attempt.criterion = np.nan
            step.failure_reason = criterion_failure


def _update_qhat_and_kappa_monitors(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
) -> None:
    if (
        attempt.k > 4
        and len(attempt.beta_path) >= 4
        and not (step.phase == 1 and abs(context.tau - 2.0) <= 1e-10)
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = np.abs(attempt.beta_path[-1] - attempt.beta_path[-2])
            d2 = np.abs(attempt.beta_path[-2] - attempt.beta_path[-3])
            d3 = np.abs(attempt.beta_path[-3] - attempt.beta_path[-4])
            qhat = np.nanmean(np.log(d1 / d2) / np.log(d2 / d3))
        step.qhat_last = float(qhat)
        if np.isinf(np.abs(qhat)) and attempt.criterion > step.eps_current * 10.0:
            step.qhat_invalid_count += 1
            if not outer.direct_nofe_fixed_tau:
                attempt.criterion = np.nan
                step.failure_reason = KERNEL_FAILURE_Q_CONVERGENCE_INVALID

    if (
        step.phase == 1
        and abs(context.tau - 2.0) > 1e-10
        and attempt.k >= 30
        and attempt.k % 10 == 0
        and attempt.beta_path_entries >= 4
    ):
        kappa_val = attempt.kappa_row_max
        if outer.direct_nofe_fixed_tau:
            if not np.isnan(attempt.kappa_row_max):
                outer.kappa_coordinate_row_max = (
                    float(attempt.kappa_row_max)
                    if np.isnan(outer.kappa_coordinate_row_max)
                    else max(
                        outer.kappa_coordinate_row_max, float(attempt.kappa_row_max)
                    )
                )
            kappa_val = _windowed_contraction_rate(
                attempt.progress_history,
                window=_DIRECT_KAPPA_WINDOW,
            )
        step.kappa_monitor_last = float(kappa_val)
        step.kappa_monitor_checks += 1
        if np.isfinite(kappa_val):
            step.kappa_monitor_max = (
                max(step.kappa_monitor_max, float(kappa_val))
                if np.isfinite(step.kappa_monitor_max)
                else float(kappa_val)
            )
        aa_m_new = _adapt_anderson_m(attempt.aa_m, kappa_val)
        if aa_m_new > attempt.aa_m:
            attempt.aa_m = aa_m_new
            attempt.kappa_bad_count = 0
        elif np.isfinite(kappa_val) and kappa_val >= 1.0:
            attempt.kappa_bad_count += 1
        else:
            attempt.kappa_bad_count = 0
        if attempt.kappa_bad_count >= 3:
            attempt.criterion = np.nan
            step.failure_reason = KERNEL_FAILURE_NONCONTRACTIVE


def _reset_attempt_for_in_loop_restart(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
) -> bool:
    step.restart_index += 1
    if (
        step.phase == 1
        and not step.phase2_budget_exhausted
        and step.restart_index < len(step.rho_path)
    ):
        if outer.direct_nofe_fixed_tau:
            outer.phase2_attempt_trace.append(
                {
                    "attempt": int(step.restart_index - 1),
                    "rho": float(step.param),
                    "iterations": int(attempt.k),
                    "evaluations": int(
                        step.phase2_evaluations - outer.attempt_evaluations_start
                    ),
                    "failure_reason": step.failure_reason,
                    "final_criterion": (
                        None
                        if np.isnan(step.last_finite_criterion)
                        else float(step.last_finite_criterion)
                    ),
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
        step.param = float(step.rho_path[step.restart_index])
        attempt.criterion = 100.0
        outer.beta = step.beta_outer_initial.copy()
        outer.linear = step.linear_outer_initial.copy()
        if outer.direct_nofe_fixed_tau:
            outer.attempt_start_policy = "phase1_start"
            if outer.restart_from_best and outer.best_score_beta is not None:
                outer.beta = outer.best_score_beta.copy()
                outer.linear = _initial_linear_state(context, outer.beta)
                outer.phase2_warm_restarts += 1
                outer.attempt_start_policy = "best_score_iterate"
            outer.attempt_evaluations_start = step.phase2_evaluations
            step.kappa_monitor_last = float("nan")
        attempt.k = 1
        attempt.criterion_history = []
        attempt.kappa_bad_count = 0
        attempt.beta_path = [outer.beta.copy()]
        attempt.beta_path_entries = 1
        attempt.kappa_row_max = float("nan")
        attempt.aa_x_hist = []
        attempt.aa_f_hist = []
        attempt.progress_history = []
        attempt.plain_residual_maxabs = float("nan")
        attempt.attempt_aa_m_start = attempt.aa_m
        attempt.attempt_aa_accepted = 0
        return True
    return False


def _run_attempt_loop(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
    attempt: _AttemptState,
    phi_step_fn: PhiStepCallable,
) -> None:
    while (
        attempt.criterion > step.eps_current
        and attempt.k < step.stopcrit_current
        and not np.isnan(attempt.criterion)
    ):
        beta0 = outer.beta.copy()
        linear0 = outer.linear.copy()
        if not _consume_evaluation_budget(context, outer, step, attempt):
            break
        beta_phi, linear_phi, valid = phi_step_fn(
            context=context,
            beta=outer.beta,
            linear=outer.linear,
            phase=step.phase,
            param=step.param,
        )
        if not valid:
            attempt.criterion = np.nan
            step.failure_reason = KERNEL_FAILURE_INVALID_PHI_STEP
            break
        if outer.direct_nofe_fixed_tau and step.phase == 1:
            _record_direct_step_score(outer, step, attempt, beta_phi)
        accepted_aa, stop_now = _try_anderson_step(
            context,
            outer,
            step,
            attempt,
            (beta0, beta_phi),
            phi_step_fn=phi_step_fn,
        )
        if stop_now:
            break
        _advance_iterate_and_criterion(
            context,
            outer,
            step,
            attempt,
            (beta0, linear0),
            phi_out=(beta_phi, linear_phi),
            accepted_aa_step=accepted_aa,
        )
        _update_qhat_and_kappa_monitors(context, outer, step, attempt)
        attempt.k += 1
        step.iteration_count = attempt.k
        if np.isnan(attempt.criterion) and not _reset_attempt_for_in_loop_restart(
            context, outer, step, attempt
        ):
            break
