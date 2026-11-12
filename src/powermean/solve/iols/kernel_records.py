"""Control records, diagnostic schemas, and input validation for the IOLS kernel."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import FloatArray, IntArray

KernelTopology = Literal["noFE", "fullcancel", "fullrecover", "mixed"]
ScoreValidationPolicy = Literal["diagnostic"]

TAU2_JOINT_HDFE_SUPPORT_CHECK: Final = "tau2_joint_hdfe_boundary_separation"
TAU2_RECOVERED_FE_SUPPORT_CHECK: Final = "tau2_recovered_fe_boundary_separation"
TAU2_BOUNDARY_SEPARATION_CODE: Final = "tau2_boundary_separation"
TAU2_BOUNDARY_SUPPORT_POLICY: Final = "fail_explicit"
TAU2_BOUNDARY_FUTURE_POLICY: Final = (
    "drop_separated_requires_theory_diagnostics_sample_accounting_covariance_mc"
)
TAU2_BOUNDARY_DIRECTION_TOL: Final = 1e-7
FULLRECOVER_JOINT_FE_POLICY: Final = "pure_hdfe_recover_all"
FULLRECOVER_PROFILED_FE_POLICY: Final = "analytical_profile"
KERNEL_BACKEND_PYTHON_REFERENCE: Final = "python_reference"
KERNEL_BACKEND_CPP: Final = "cpp"

KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL: Final = "configured_path_partial"
KERNEL_FAILURE_CRITERION_EXPLOSION: Final = "criterion_explosion"
KERNEL_FAILURE_CRITERION_STAGNATION: Final = "criterion_stagnation"
KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH: Final = "empty_finite_rho_path"
KERNEL_FAILURE_INVALID_PHI_STEP: Final = "invalid_phi_step"
KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED: Final = "max_path_evaluations_reached"
KERNEL_FAILURE_MIN_DTAU_REACHED: Final = "min_dtau_reached"
KERNEL_FAILURE_NONCONTRACTIVE: Final = "noncontractive"
KERNEL_FAILURE_NOT_CONVERGED: Final = "not_converged"
KERNEL_FAILURE_NOT_STARTED: Final = "not_started"
KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE: Final = "phase1_candidate_unusable"
KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED: Final = (
    "phase2_evaluation_budget_exhausted"
)
KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED: Final = (
    KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED
)
KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE: Final = (
    "post_convergence_noncontractive"
)
KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL: Final = "post_convergence_residual"
# runtime_v4 direct route only: the final verification map was invalid.
KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP: Final = "post_convergence_invalid_map"
KERNEL_FAILURE_Q_CONVERGENCE_INVALID: Final = "q_convergence_invalid"
KERNEL_FAILURE_TAU2_ANCHOR_FAILED: Final = "tau2_anchor_failed"
KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO: Final = "tau2_anchor_missing_rho"
TRANSFORMED_NORMALIZATION_REL_FLOOR: Final = 1e-12
TRANSFORMED_NORMALIZATION_ABS_BACKSTOP: Final = 1e-300
KERNEL_FAILURE_REASONS: Final[frozenset[str]] = frozenset(
    {
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
        KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED,
        KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE,
        KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL,
        KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP,
        KERNEL_FAILURE_Q_CONVERGENCE_INVALID,
        KERNEL_FAILURE_TAU2_ANCHOR_FAILED,
        KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
        TAU2_BOUNDARY_SEPARATION_CODE,
    }
)

DIRECT_RESTART_START_POLICIES: Final = ("phase1_start", "best_score_iterate")
DIRECT_NOFE_RUNTIME_VERSION: Final = "direct_nofe_runtime_v4_2026_10_03"
DIRECT_KAPPA_MONITOR_POLICY: Final = "windowed_plain_residual_ratio_10"
DIRECT_QHAT_GUARD_POLICY_NUMPY: Final = "advisory_numpy_reference_direct_route"
DIRECT_PROGRESS_MONITOR_POLICY: Final = (
    "plain_map_residual_maxabs_star_at_current_iterate"
)
DIRECT_POST_CHECK_POLICY: Final = "invalid_verification_map_clears_convergence"
DIRECT_NOFE_FIXED_TAU_SCHEMA: Final = "powermean.direct_nofe_fixed_tau.v1"
DIRECT_NOFE_FIXED_TAU_ROUTE: Final = "direct_nofe_fixed_tau"
DIRECT_NOFE_FIXED_TAU_ALGORITHM: Final = "iols_phase1_phase2_at_requested_tau"
DIRECT_NOFE_FIXED_TAU_RESULT_KEYS: Final = (
    "schema",
    "route",
    "backend",
    "kernel_backend",
    "algorithm",
    "intercept_policy",
    "tau_requested",
    "tau_fit",
    "phase1_tau",
    "phase2_tau",
    "tau2_anchor_used",
    "tau_continuation_used",
    "converged",
    "failure_reason",
    "beta",
    "intercept",
    "linear",
    "iteration",
    "diagnostics",
    "control",
    "backend_details",
)
DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS: Final = (
    "converged",
    "phase",
    "iterations",
    "restarts",
    "final_rho",
    "rho_path",
    "final_criterion",
    "anderson_m_initial",
    "anderson_m_final",
    "failure_reason",
)
DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS: Final = (
    "phase1_attempts",
    "phase1_accepted",
    "phase1_rejected",
    "phase2_evaluations",
    "last_finite_criterion",
    "best_finite_criterion",
    "post_convergence_residual",
    "post_convergence_kappa_est",
    "post_convergence_score_residual",
    "post_convergence_fe_score_residual",
    "score_tolerance",
    "score_validation_status",
    "score_warning_codes",
)
DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS: Final = (
    "anderson_m",
    "eps",
    "stopcrit",
    "max_phase1_iterations",
    "max_phase2_evaluations",
    "delta_log_min",
    "delta_log_max",
    "delta_grid_size",
    "rho_log_min",
    "rho_log_max",
    "rho_grid_size",
)
_DIRECT_NOFE_INT_DIAGNOSTICS: Final = frozenset(
    {"phase1_attempts", "phase1_accepted", "phase1_rejected", "phase2_evaluations"}
)
DIRECT_RHO_GRID_POLICIES: Final = ("legacy", "step_grid", "curvature_step_grid")
DIRECT_RHO_GRID_HOOK_VERSION: Final = "rho_grid_hook_v1_2026_10_05"
_DIRECT_KAPPA_WINDOW: Final = 10
_BETA_PATH_RETAINED: Final = 4


@dataclass(frozen=True)
class IOLSKernelControl:
    """Numerical controls for the package-owned iOLS kernel."""

    anderson_m: int = 5
    eps: float = 1e-6
    stopcrit: int = 10000
    tau_path_steps: int = 20
    min_dtau: float | None = None
    max_path_evaluations: int = 10000
    max_phase1_iterations: int = 10000
    max_phase2_evaluations: int = 50000
    fe_profile_maxiter: int = 10000
    fe_profile_tol: float = 1e-6
    fe_demean_maxiter: int = 10000
    fe_demean_tol: float = 1e-10
    delta_log_min: float = -5.0
    delta_log_max: float = 5.0
    delta_grid_size: int = 5
    rho_log_min: float = -7.0
    rho_log_max: float = 14.0
    rho_grid_size: int = 150
    transformed_design_normalization: Literal["none", "scale"] = "none"
    direct_restart_start: Literal["phase1_start", "best_score_iterate"] = "phase1_start"
    _pilot_deltas: tuple[float, ...] | None = None


@dataclass(frozen=True)
class IOLSIterationDiagnostics:
    """Iteration record for one kernel solve."""

    converged: bool
    phase: str
    iterations: int
    restarts: int
    final_rho: float | None
    rho_path: tuple[float, ...]
    final_criterion: float | None
    anderson_m_initial: int
    anderson_m_final: int
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ConcentratedSolve:
    """Concentrated fixed-tau solve result record."""

    beta: FloatArray
    linear: FloatArray | None
    iteration: IOLSIterationDiagnostics


@dataclass(frozen=True)
class _FullrecoverInvariantState:
    """Precomputed demeaned design and least-squares projection matrix."""

    x_demeaned: FloatArray
    projection: FloatArray


@dataclass(frozen=True)
class _FullrecoverProblem:
    """Inputs retained before validation, pilot-grid construction and projection."""

    endog: FloatArray
    exog: FloatArray
    groups: tuple[IntArray, ...]
    counts: tuple[int, ...]
    tau: float
    control: IOLSKernelControl
    invariant_state: _FullrecoverInvariantState | None = None


@dataclass(frozen=True)
class _FullrecoverContext:
    """Problem and control references sharing one solve's design projection."""

    endog: FloatArray
    exog: FloatArray
    groups: tuple[IntArray, ...]
    counts: tuple[int, ...]
    tau: float
    control: IOLSKernelControl
    x_demeaned: FloatArray
    projection: FloatArray


def _transformed_normalization_mode(control: IOLSKernelControl) -> str:
    mode = str(getattr(control, "transformed_design_normalization", "none"))
    if mode not in {"none", "scale"}:
        raise ValueError(
            "transformed_design_normalization must be one of 'none' or 'scale'"
        )
    return mode


def _control_with_anderson_m(
    control: IOLSKernelControl,
    *,
    anderson_m: int,
) -> IOLSKernelControl:
    values = {
        item.name: getattr(control, item.name) for item in fields(IOLSKernelControl)
    }
    values["anderson_m"] = int(anderson_m)
    return IOLSKernelControl(**values)


def _control_to_cpp_mapping(control: IOLSKernelControl) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "anderson_m": int(control.anderson_m),
        "eps": float(control.eps),
        "stopcrit": int(control.stopcrit),
        "tau_path_steps": int(control.tau_path_steps),
        "min_dtau": None if control.min_dtau is None else float(control.min_dtau),
        "max_path_evaluations": int(control.max_path_evaluations),
        "max_phase1_iterations": int(control.max_phase1_iterations),
        "max_phase2_evaluations": int(control.max_phase2_evaluations),
        "fe_demean_maxiter": int(control.fe_demean_maxiter),
        "fe_demean_tol": float(control.fe_demean_tol),
        "delta_log_min": float(control.delta_log_min),
        "delta_log_max": float(control.delta_log_max),
        "delta_grid_size": int(control.delta_grid_size),
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
        "direct_restart_start": str(control.direct_restart_start),
    }
    if control._pilot_deltas is not None:
        mapping["_pilot_deltas"] = list(control._pilot_deltas)
    return mapping


def _direct_nofe_validated_inputs(
    endog: npt.ArrayLike,
    exog_slopes: npt.ArrayLike,
    tau: float,
) -> tuple[FloatArray, FloatArray, float]:
    y: FloatArray = np.array(endog, dtype=np.float64, copy=True)
    x: FloatArray = np.array(exog_slopes, dtype=np.float64, copy=True, order="C")
    if y.ndim != 1:
        raise ValueError("endog must be one-dimensional")
    if x.ndim != 2:
        raise ValueError("exog_slopes must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("endog and exog_slopes must have the same number of rows")
    if x.shape[1] < 1:
        raise ValueError(
            "exog_slopes must have at least one slope column; the intercept is profiled"
        )
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("endog must be finite and nonnegative")
    if not np.any(y > 0.0):
        raise ValueError("endog must contain at least one positive value")
    if not np.all(np.isfinite(x)):
        raise ValueError("exog_slopes must be finite")
    for column in range(x.shape[1]):
        if np.max(x[:, column]) == np.min(x[:, column]):
            raise ValueError(
                f"exog_slopes column {column} is constant; pass slopes only, the "
                "intercept is profiled"
            )
    tau_value = float(tau)
    if not np.isfinite(tau_value) or tau_value < 0.0 or tau_value > 3.0:
        raise ValueError("tau must be finite and in [0, 3]")
    return y, x, tau_value


def _direct_nofe_validate_control(control: IOLSKernelControl) -> None:
    if not isinstance(control, IOLSKernelControl):
        raise TypeError("control must be an IOLSKernelControl")
    if _transformed_normalization_mode(control) != "none":
        raise ValueError(
            "direct fixed-tau route requires transformed_design_normalization='none'"
        )
    if control._pilot_deltas is not None and (
        not control._pilot_deltas
        or any(not math.isfinite(v) or v <= 0.0 for v in control._pilot_deltas)
    ):
        raise ValueError("_pilot_deltas must be a nonempty positive finite vector")
    if int(control.delta_grid_size) < 1:
        raise ValueError(
            "direct fixed-tau route requires delta_grid_size >= 1 so Phase 1 runs"
        )
    if int(control.anderson_m) < 0:
        raise ValueError("anderson_m must be nonnegative")
    if not np.isfinite(float(control.eps)) or float(control.eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if int(control.stopcrit) < 2 or int(control.max_phase1_iterations) < 2:
        raise ValueError("stopcrit and max_phase1_iterations must be at least 2")
    if int(control.max_phase2_evaluations) < 1:
        raise ValueError("max_phase2_evaluations must be at least 1")
    if str(control.direct_restart_start) not in DIRECT_RESTART_START_POLICIES:
        raise ValueError(
            f"direct_restart_start must be one of {DIRECT_RESTART_START_POLICIES}"
        )
