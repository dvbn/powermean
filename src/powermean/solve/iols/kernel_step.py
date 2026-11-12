"""Concentrated fixed-point map, score residuals, and loop state records for IOLS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from powermean.solve.iols._numeric import (
    FloatArray,
    exp_with_float64_overflow_guard,
)
from powermean.solve.iols.kernel_grid import _effective_anderson_m
from powermean.solve.iols.kernel_records import (
    IOLSIterationDiagnostics,
    ScoreValidationPolicy,
    _FullrecoverContext,
)
from powermean.solve.iols.kernel_state import (
    _demean_for_groups,
    _profile_linear,
    _recover_fe_linear,
    _uses_joint_hdfe_state,
)


class PhiStepCallable(Protocol):
    """Keyword contract for the replaceable fixed-point map."""

    def __call__(
        self,
        *,
        context: _FullrecoverContext,
        beta: FloatArray,
        linear: FloatArray | None,
        phase: int,
        param: float,
    ) -> tuple[FloatArray, FloatArray | None, bool]:
        """Evaluate the map with the exact extracted helper arguments."""
        ...


@dataclass
class _OuterLoopState:
    """Mutable state across Phase 1 (delta grid) and Phase 2 (rho grid)."""

    delta_grid: FloatArray
    n_outer: int
    beta: FloatArray
    linear: FloatArray
    beta_initial: FloatArray
    linear_initial: FloatArray
    phase1_accepted_beta: FloatArray
    phase1_accepted_linear: FloatArray
    direct_nofe_fixed_tau: bool
    direct_rho_grid_policy: str
    rho_hint: float | None
    score_validation_policy: ScoreValidationPolicy
    restart_from_best: bool
    final_diag: IOLSIterationDiagnostics
    phase1_attempts: int = 0
    phase1_accepted: int = 0
    phase1_rejected: int = 0
    phase1_evaluations: int = 0
    best_score_norm: float = float("inf")
    best_score_rho: float = float("nan")
    best_score_beta: FloatArray | None = None
    phase2_warm_restarts: int = 0
    phase2_attempt_trace: list[dict[str, Any]] = field(default_factory=list)
    kappa_coordinate_row_max: float = float("nan")
    attempt_evaluations_start: int = 0
    attempt_start_policy: str = "phase1_start"
    post_convergence_invalid_map_counts: list[int] = field(
        default_factory=lambda: [0, 0]
    )
    hook_alpha0: float | None = None
    hook_alpha_min: float | None = None
    hook_clipped: bool | None = None
    hook_values: tuple[float, ...] | None = None
    hook_curvature: dict[str, Any] | None = None


@dataclass
class _PhaseStepState:
    """Mutable state for one outer phase (one delta pilot or the Phase-2 rho path)."""

    phase: int
    param: float
    eps_current: float
    stopcrit_current: int
    rho_path: FloatArray
    rho_details: dict[str, Any]
    beta_outer_initial: FloatArray
    linear_outer_initial: FloatArray
    aa_m_final: int
    restart_index: int = 0
    verified: bool = False
    phase2_evaluations: int = 0
    phase2_budget_exhausted: bool = False
    iteration_count: int = 0
    final_criterion: float | None = None
    failure_reason: str | None = None
    qhat_last: float = float("nan")
    qhat_invalid_count: int = 0
    kappa_monitor_last: float = float("nan")
    kappa_monitor_max: float = float("nan")
    kappa_monitor_checks: int = 0
    post_convergence_residual: float = float("nan")
    post_convergence_kappa_est: float = float("nan")
    post_convergence_score_residual: float = float("nan")
    post_convergence_fe_score_residual: float = float("nan")
    score_tol: float = float("nan")
    score_validation_status: str = "not_checked"
    score_warning_codes: list[str] = field(default_factory=list)
    score_warning_messages: list[str] = field(default_factory=list)
    score_exceeds_tolerance: bool = False
    fe_score_exceeds_tolerance: bool = False
    last_finite_criterion: float = float("nan")
    best_finite_criterion: float = float("nan")


@dataclass
class _AttemptState:
    """Mutable state for one fixed-point attempt at a single delta or rho value."""

    joint_hdfe: bool
    aa_m: int
    beta_path: list[FloatArray]
    attempt_aa_m_start: int
    k: int = 1
    criterion: float = 100.0
    criterion_history: list[float] = field(default_factory=list)
    kappa_bad_count: int = 0
    beta_path_entries: int = 1
    kappa_row_max: float = float("nan")
    aa_x_hist: list[FloatArray] = field(default_factory=list)
    aa_f_hist: list[FloatArray] = field(default_factory=list)
    progress_history: list[float] = field(default_factory=list)
    plain_residual_maxabs: float = float("nan")
    attempt_aa_accepted: int = 0


def _new_attempt_state(
    context: _FullrecoverContext,
    outer: _OuterLoopState,
    step: _PhaseStepState,
) -> _AttemptState:
    if outer.direct_nofe_fixed_tau and step.phase == 1:
        outer.attempt_evaluations_start = step.phase2_evaluations
    joint_hdfe = _uses_joint_hdfe_state(context.groups)
    aa_m = (
        0
        if joint_hdfe
        else _effective_anderson_m(context.control.anderson_m, step.phase, step.param)
    )
    return _AttemptState(
        joint_hdfe=joint_hdfe,
        aa_m=aa_m,
        beta_path=[outer.beta.copy()],
        attempt_aa_m_start=aa_m,
    )


def _phi_step_fullrecover(
    *,
    context: _FullrecoverContext,
    beta: FloatArray,
    linear: FloatArray | None,
    phase: int,
    param: float,
) -> tuple[FloatArray, FloatArray | None, bool]:
    joint_hdfe = _uses_joint_hdfe_state(context.groups)
    if joint_hdfe:
        if linear is None:
            return beta.copy(), None, False
        linear_current: FloatArray = np.asarray(linear, dtype=np.float64)
    else:
        linear_current = _profile_linear(
            endog=context.endog,
            exog=context.exog,
            beta=beta,
            groups=context.groups,
            counts=context.counts,
            tau=context.tau,
            maxiter=int(context.control.fe_profile_maxiter),
            tol=float(context.control.fe_profile_tol),
        )
    if not np.all(np.isfinite(linear_current)):
        return beta.copy(), linear, False

    endog = context.endog
    tau = context.tau
    mu = exp_with_float64_overflow_guard(linear_current)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u_ratio = endog / mu
        weights = exp_with_float64_overflow_guard((2.0 - tau) * linear_current)
        if phase == 0:
            delta = float(param)
            delta_tilde = delta / np.maximum(weights, 1e-300)
            pseudo: FloatArray = linear_current + np.log1p(
                (u_ratio - 1.0) / (1.0 + delta_tilde)
            )
        else:
            rho = float(param)
            log_denom = np.log1p(rho)
            positive = endog > 0.0
            positive_component: FloatArray = np.zeros_like(
                linear_current, dtype=np.float64
            )
            positive_log: FloatArray = np.full(
                linear_current.shape[0], -np.inf, dtype=np.float64
            )
            positive_log[positive] = (
                np.log(endog[positive])
                + (1.0 - tau) * linear_current[positive]
                - log_denom
            )
            positive_component[positive] = exp_with_float64_overflow_guard(
                positive_log[positive]
            )
            negative_component = exp_with_float64_overflow_guard(
                (2.0 - tau) * linear_current - log_denom
            )
            pseudo = linear_current + positive_component - negative_component
    if not np.all(np.isfinite(pseudo)):
        return beta.copy(), linear, False
    demeaned_pseudo = _demean_for_groups(
        pseudo,
        context.groups,
        context.counts,
        tol=float(context.control.fe_demean_tol),
        maxiter=int(context.control.fe_demean_maxiter),
    )
    beta_new: FloatArray = context.projection @ demeaned_pseudo
    if not np.all(np.isfinite(beta_new)):
        return beta.copy(), linear, False
    if not joint_hdfe:
        return np.asarray(beta_new, dtype=np.float64), None, True

    recovered = _recover_fe_linear(
        pseudo - context.exog @ beta_new,
        context.groups,
        context.counts,
        tol=float(context.control.fe_demean_tol),
        maxiter=int(context.control.fe_demean_maxiter),
    )
    linear_new: FloatArray = context.exog @ beta_new + recovered
    if not np.all(np.isfinite(linear_new)):
        return beta.copy(), linear, False
    return (
        np.asarray(beta_new, dtype=np.float64),
        np.asarray(linear_new, dtype=np.float64),
        True,
    )


def _profiled_slope_score_residual(
    *,
    context: _FullrecoverContext,
    beta: FloatArray,
    linear: FloatArray | None = None,
) -> float:
    if linear is None:
        linear_arr = _profile_linear(
            endog=context.endog,
            exog=context.exog,
            beta=beta,
            groups=context.groups,
            counts=context.counts,
            tau=context.tau,
            maxiter=int(context.control.fe_profile_maxiter),
            tol=float(context.control.fe_profile_tol),
        )
    else:
        linear_arr = np.asarray(linear, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_obs = context.endog * exp_with_float64_overflow_guard(
            (1.0 - context.tau) * linear_arr
        ) - exp_with_float64_overflow_guard((2.0 - context.tau) * linear_arr)
    if not np.all(np.isfinite(score_obs)):
        return float("inf")
    score = context.x_demeaned.T @ score_obs
    if not np.all(np.isfinite(score)):
        return float("inf")
    return float(np.max(np.abs(score)) / max(float(context.endog.shape[0]), 1.0))


def _profiled_fe_score_residual(
    *,
    context: _FullrecoverContext,
    beta: FloatArray,
    linear: FloatArray | None = None,
) -> float:
    if linear is None:
        linear_arr = _profile_linear(
            endog=context.endog,
            exog=context.exog,
            beta=beta,
            groups=context.groups,
            counts=context.counts,
            tau=context.tau,
            maxiter=int(context.control.fe_profile_maxiter),
            tol=float(context.control.fe_profile_tol),
        )
    else:
        linear_arr = np.asarray(linear, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_obs = context.endog * exp_with_float64_overflow_guard(
            (1.0 - context.tau) * linear_arr
        ) - exp_with_float64_overflow_guard((2.0 - context.tau) * linear_arr)
    if not np.all(np.isfinite(score_obs)):
        return float("inf")
    if len(context.groups) == 0:
        return abs(float(np.sum(score_obs))) / max(float(context.endog.shape[0]), 1.0)
    residual = 0.0
    for group, count in zip(context.groups, context.counts, strict=True):
        sums = np.bincount(group, weights=score_obs, minlength=count)
        if sums.size:
            residual = max(residual, float(np.max(np.abs(sums))))
    return residual / max(float(context.endog.shape[0]), 1.0)
