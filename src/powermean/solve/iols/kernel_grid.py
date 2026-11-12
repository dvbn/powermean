"""Phase-1 delta grids, Phase-2 rho grids, curvature pilot, and contraction monitors."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import (
    FloatArray,
    exp_with_float64_overflow_guard,
)
from powermean.solve.iols.kernel_records import (
    _BETA_PATH_RETAINED,
    DIRECT_RHO_GRID_POLICIES,
    KERNEL_FAILURE_CRITERION_EXPLOSION,
    KERNEL_FAILURE_CRITERION_STAGNATION,
    IOLSKernelControl,
    _FullrecoverContext,
)
from powermean.solve.iols.kernel_state import _profile_linear


def _effective_anderson_m(anderson_m: int, _phase: int, _param: float) -> int:
    return int(anderson_m)


def _adapt_anderson_m(aa_m: int, kappa_val: float) -> int:
    if not np.isfinite(kappa_val):
        return aa_m
    if kappa_val >= 0.95:
        return max(aa_m, 20)
    if kappa_val >= 0.85:
        return max(aa_m, 12)
    if kappa_val >= 0.70:
        return max(aa_m, 8)
    return aa_m


def _windowed_contraction_rate(history: list[float], window: int = 10) -> float:
    """Compute geometric-mean step contraction over the last ``window`` steps."""
    if window < 1 or len(history) < window + 1:
        return float("nan")
    current = float(history[-1])
    previous = float(history[-1 - window])
    if not (math.isfinite(current) and math.isfinite(previous)):
        return float("nan")
    if previous <= 0.0 or current < 0.0:
        return float("nan")
    return float((current / previous) ** (1.0 / float(window)))


def _direct_progress_trace_fields(
    progress_history: list[float],
    *,
    aa_m_start: int,
    aa_m_end: int,
    aa_accepted: int,
) -> dict[str, Any]:
    """Return runtime_v4 per-attempt progress-monitor fields."""
    return {
        "final_progress_residual": (
            float(progress_history[-1]) if progress_history else float("nan")
        ),
        "best_progress_residual": (
            float(min(progress_history)) if progress_history else float("nan")
        ),
        "anderson_m_start": int(aa_m_start),
        "anderson_m_end": int(aa_m_end),
        "anderson_accepted": int(aa_accepted),
    }


def _criterion_stagnated(
    history: list[float],
    *,
    eps: float,
    window: int = 50,
    min_iterations: int = 100,
    min_relative_drop: float = 0.01,
) -> bool:
    if len(history) < max(window * 2, min_iterations):
        return False
    current = float(history[-1])
    previous = float(history[-window])
    if not (np.isfinite(current) and np.isfinite(previous)):
        return False
    if current <= eps * 10.0:
        return False
    relative_drop = (previous - current) / max(abs(previous), eps)
    return bool(relative_drop < min_relative_drop)


def _record_criterion_progress(
    history: list[float],
    criterion: float,
    *,
    phase: int,
    tau: float,
    eps: float,
    max_phase2_evaluations: int,
    bounds: tuple[float, float],
    stagnation_history: list[float] | None = None,
) -> tuple[float, float, str | None]:
    last_fin, best_fin = bounds
    if not np.isfinite(criterion):
        return last_fin, best_fin, None
    criterion_float = float(criterion)
    history.append(criterion_float)
    last_fin = criterion_float
    best_fin = (
        min(float(best_fin), criterion_float)
        if np.isfinite(best_fin)
        else criterion_float
    )
    if phase == 1 and tau > 2.0 and len(history) >= 20:
        criterion_ref = history[-20]
        if (
            np.isfinite(criterion_ref)
            and criterion_ref > eps
            and criterion_float > max(criterion_ref * 100.0, 1e6)
        ):
            return last_fin, best_fin, KERNEL_FAILURE_CRITERION_EXPLOSION
    min_iters = min(max(int(max_phase2_evaluations) // 2500, 20), 200)
    window = min(max(min_iters // 2, 10), 50)
    if phase == 1 and _criterion_stagnated(
        history if stagnation_history is None else stagnation_history,
        eps=eps,
        window=window,
        min_iterations=min_iters,
        min_relative_drop=1e-4,
    ):
        return last_fin, best_fin, KERNEL_FAILURE_CRITERION_STAGNATION
    return last_fin, best_fin, None


def _focus_rho_path(rho_path: FloatArray, rho_hint: float | None = None) -> FloatArray:
    path: FloatArray = np.asarray(rho_path, dtype=np.float64)
    path = path[np.isfinite(path) & (path > 0.0)]
    if path.size == 0:
        return path
    if rho_hint is None or not np.isfinite(rho_hint) or rho_hint <= 0.0:
        return path

    hint = min(max(float(rho_hint), float(path[0])), float(path[-1]))
    focus_mult = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0], dtype=np.float64)
    ordered: list[float] = []
    seen: set[float] = set()
    focus = np.minimum(np.maximum(hint * focus_mult, path[0]), path[-1])
    for value in np.concatenate([focus, path]):
        finite_value = float(value)
        if not np.isfinite(finite_value) or finite_value <= 0.0 or finite_value in seen:
            continue
        seen.add(finite_value)
        ordered.append(finite_value)
    ordered_arr: FloatArray = np.asarray(ordered, dtype=np.float64)
    return ordered_arr


def _rho_base_for_tau(tau: float, control: IOLSKernelControl) -> FloatArray:
    _ = tau
    grid_size = max(int(control.rho_grid_size), 2)
    log_min = float(control.rho_log_min)
    log_max = float(control.rho_log_max)
    if not (np.isfinite(log_min) and np.isfinite(log_max)):
        raise ValueError("rho log-grid bounds must be finite")
    if log_max <= log_min:
        raise ValueError("rho_log_max must be greater than rho_log_min")
    res: FloatArray = np.exp(np.linspace(log_min, log_max, grid_size, dtype=np.float64))
    return res


def _delta_base_for_anchor(control: IOLSKernelControl) -> FloatArray:
    if control._pilot_deltas is not None:
        custom: FloatArray = np.array(
            control._pilot_deltas, dtype=np.float64, copy=True
        )
        return custom
    grid_size = max(int(control.delta_grid_size), 0)
    if grid_size == 0:
        empty: FloatArray = np.array([], dtype=np.float64)
        return empty
    log_min = float(control.delta_log_min)
    log_max = float(control.delta_log_max)
    if not (np.isfinite(log_min) and np.isfinite(log_max)):
        raise ValueError("delta log-grid bounds must be finite")
    if log_max <= log_min:
        raise ValueError("delta_log_max must be greater than delta_log_min")
    res: FloatArray = np.exp(np.linspace(log_min, log_max, grid_size, dtype=np.float64))
    return res


def _rho_path_for_linear_state(
    *,
    linear: FloatArray,
    tau: float,
    rho_hint: float | None,
    control: IOLSKernelControl,
) -> tuple[FloatArray, dict[str, Any]]:
    linear_arr = np.asarray(linear, dtype=np.float64)
    base = _rho_base_for_tau(tau, control)
    rho_path: FloatArray = np.unique(base[np.isfinite(base) & (base > 0.0)])
    rho_path = _focus_rho_path(rho_path, rho_hint=rho_hint)
    details: dict[str, Any] = {
        "rho_policy": "fixed_log_grid",
        "linear_min": float(np.nanmin(linear_arr)) if linear_arr.size else float("nan"),
        "linear_max": float(np.nanmax(linear_arr)) if linear_arr.size else float("nan"),
        "rho_candidate_count": int(rho_path.size),
        "rho_hint": None if rho_hint is None else float(rho_hint),
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
    }
    if rho_path.size:
        details.update(
            {
                "rho_min": float(rho_path[0]),
                "rho_max": float(rho_path[-1]),
            }
        )
    return rho_path, details


def _max_abs_change(left: FloatArray, right: FloatArray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _record_beta_path_entry(
    beta_path: list[FloatArray],
    beta: FloatArray,
    *,
    entries_seen: int,
    kappa_row_max: float,
) -> tuple[int, float]:
    """Append to the bounded beta path and fold in the newest kappa ratio row."""
    beta_path.append(beta.copy())
    entries_seen += 1
    if entries_seen >= 4:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.abs(beta_path[-1] - beta_path[-2]) / np.abs(
                beta_path[-2] - beta_path[-3]
            )
        row_mean = float(np.nanmean(ratios))
        if not np.isnan(row_mean):
            kappa_row_max = (
                row_mean if np.isnan(kappa_row_max) else max(kappa_row_max, row_mean)
            )
    if len(beta_path) > _BETA_PATH_RETAINED:
        del beta_path[0]
    return entries_seen, kappa_row_max


def _direct_step_rho_grid(
    alpha0: float, alpha_min: float
) -> tuple[tuple[float, ...], bool]:
    alpha0 = float(alpha0)
    alpha_min = float(alpha_min)
    if not (math.isfinite(alpha0) and alpha0 > 0.0):
        raise RuntimeError(f"direct_rho_grid_invalid: alpha0={alpha0!r}")
    if not (math.isfinite(alpha_min) and alpha_min > 0.0):
        raise RuntimeError(f"direct_rho_grid_invalid: alpha_min={alpha_min!r}")
    clipped = alpha0 <= alpha_min
    if clipped:
        alphas = [alpha_min]
    else:
        alphas = []
        a = alpha0
        while a > alpha_min:
            alphas.append(a)
            a = a * 0.5
        alphas.append(alpha_min)
    rho = tuple(1.0 / a - 1.0 for a in alphas)
    previous = None
    for value in rho:
        if not (math.isfinite(value) and value > 0.0):
            raise RuntimeError(f"direct_rho_grid_invalid: rho={value!r}")
        if previous is not None and not value > previous:
            raise RuntimeError("direct_rho_grid_invalid: not strictly increasing")
        previous = value
    return rho, clipped


def _profiled_curvature_weights(
    y: FloatArray, linear: FloatArray, tau: float, n: int
) -> tuple[FloatArray, float]:
    c1 = float(tau) - 1.0
    c2 = 2.0 - float(tau)
    h: FloatArray = np.zeros(n, dtype=np.float64)
    if c1 != 0.0:
        positive = y > 0.0
        h[positive] += c1 * exp_with_float64_overflow_guard(
            np.log(y[positive]) + (1.0 - float(tau)) * linear[positive]
        )
    if c2 != 0.0:
        h += c2 * exp_with_float64_overflow_guard((2.0 - float(tau)) * linear)
    if not np.all(np.isfinite(h)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_row_curvature")
    total = float(np.sum(h))
    if not (math.isfinite(total) and total > 0.0):
        raise RuntimeError("direct_rho_curvature_failed: nonpositive_curvature_sum")
    return h, total


def _direct_profiled_curvature_l0(
    *,
    context: _FullrecoverContext,
    beta: FloatArray,
) -> dict[str, Any]:
    """Compute max generalized eigenvalue of (profiled H, Z'Z/n) at ``beta``."""
    start_ns = time.perf_counter_ns()
    linear = _profile_linear(
        endog=context.endog,
        exog=context.exog,
        beta=beta,
        groups=context.groups,
        counts=context.counts,
        tau=context.tau,
        maxiter=int(context.control.fe_profile_maxiter),
        tol=float(context.control.fe_profile_tol),
    )
    linear_arr = np.asarray(linear, dtype=np.float64)
    if not np.all(np.isfinite(linear_arr)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_index")
    y = np.asarray(context.endog, dtype=np.float64)
    z = np.asarray(context.x_demeaned, dtype=np.float64)
    n = z.shape[0]
    h, total = _profiled_curvature_weights(y, linear_arr, context.tau, n)
    g = z.T @ h
    hess = ((z.T * h) @ z - np.outer(g, g) / total) / float(n)
    hess = 0.5 * (hess + hess.T)
    if not np.all(np.isfinite(hess)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_hessian")
    gram = (z.T @ z) / float(n)
    gram = 0.5 * (gram + gram.T)
    try:
        chol = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("direct_rho_curvature_failed: gram_cholesky_failed") from exc
    left = np.linalg.solve(chol, hess)
    cmat = np.linalg.solve(chol, left.T)
    cmat = 0.5 * (cmat + cmat.T)
    if not np.all(np.isfinite(cmat)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_hessian")
    try:
        eigenvalues = np.linalg.eigvalsh(cmat)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("direct_rho_curvature_failed: eigensolver_failed") from exc
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_eigenvalue")
    return {
        "L0": float(np.max(eigenvalues)),
        "lambda_min": float(np.min(eigenvalues)),
        "wall_ns": int(time.perf_counter_ns() - start_ns),
    }


def _direct_policy_rho_path(
    *,
    policy: str,
    linear: npt.ArrayLike,
    tau: float,
    control: IOLSKernelControl,
    curvature: dict[str, Any] | None,
) -> tuple[FloatArray, dict[str, Any]]:
    _ = tau
    alpha_max = 1.0 / (1.0 + math.exp(float(control.rho_log_min)))
    alpha_min = 1.0 / (1.0 + math.exp(float(control.rho_log_max)))
    if policy == "step_grid":
        alpha0 = alpha_max
    elif policy == "curvature_step_grid":
        if curvature is None:
            raise RuntimeError("direct_rho_grid_invalid: missing curvature")
        alpha0 = 1.0 / max(
            1.0 + math.exp(float(control.rho_log_min)), float(curvature["L0"])
        )
    else:
        raise ValueError(f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}")
    values, clipped = _direct_step_rho_grid(alpha0, alpha_min)
    rho_path: FloatArray = np.asarray(values, dtype=np.float64)
    linear_arr: FloatArray = np.asarray(linear, dtype=np.float64)
    details: dict[str, Any] = {
        "rho_policy": "direct_" + policy,
        "linear_min": float(np.nanmin(linear_arr)) if linear_arr.size else float("nan"),
        "linear_max": float(np.nanmax(linear_arr)) if linear_arr.size else float("nan"),
        "rho_candidate_count": int(rho_path.size),
        "rho_hint": None,
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
        "rho_min": float(rho_path[0]),
        "rho_max": float(rho_path[-1]),
        "_rho_grid_alpha0": float(alpha0),
        "_rho_grid_alpha_min": float(alpha_min),
        "_rho_grid_clipped": bool(clipped),
    }
    return rho_path, details
