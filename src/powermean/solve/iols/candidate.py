"""Package solver entry and candidate record for the optional no-FE IOLS solver.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

import numpy as np

from powermean.errors import SolverCapabilityError
from powermean.model.rows import _immutable
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.adapter import IOLSAdapterResult, run_iols_nofe
from powermean.solve.iols.direct_route import IOLSDirectNoFEResult
from powermean.solve.iols.options import (
    IolsOptions,
    effective_backend_request,
    options_to_kernel_control,
    resolve_iols_options,
)
from powermean.solve.iols.records import owned_value
from powermean.solve.newton_trace import NativeReason, Timing
from powermean.solve.options import SolverOptions
from powermean.systems.joint import JointSystem

UNSUPPORTED_IOLS_SUFFIX = (
    "solver='iols' supports only no-FE Power(tau) models with tau in [1, 2], "
    "omega > 0, an intercept and at least one slope, n > K, and no weights, "
    "offsets, exposure, fixed effects, instruments, or user start "
    "(use solver='newton')"
)


@dataclass(frozen=True)
class _IolsRequestFlags:
    """Explicit unsupported-feature indicators checked before system inspection."""

    has_weights: bool = False
    has_offset: bool = False
    has_exposure: bool = False
    has_fixed_effects: bool = False
    has_instruments: bool = False
    has_user_start: bool = False


_DEFAULT_REQUEST_FLAGS = _IolsRequestFlags()


def refuse_iols(case: str) -> NoReturn:
    """Raise :class:`SolverCapabilityError` for an unsupported IOLS request."""
    raise SolverCapabilityError(
        "solver.unsupported_case",
        {"solver": "iols", "case": f"{case}; {UNSUPPORTED_IOLS_SUFFIX}"},
    )


def _refuse_unsupported_request(
    system: JointSystem,
    flags: _IolsRequestFlags,
) -> None:
    if flags.has_fixed_effects:
        refuse_iols("fixed effects")
    if flags.has_instruments:
        refuse_iols("instruments (i2SLS)")
    if flags.has_weights or not bool(np.all(system.weights == 1.0)):
        refuse_iols("weights")
    if flags.has_offset or flags.has_exposure or not bool(np.all(system.offset == 0.0)):
        refuse_iols("offsets or exposure")
    if flags.has_user_start:
        refuse_iols("user-supplied start values")


def _validate_intercept_and_dimensions(
    system: JointSystem,
    intercept: int | None,
) -> int:
    if intercept is None:
        refuse_iols("designs without an intercept")
    if not 0 <= intercept < system.dimension or not bool(
        np.all(system.design[:, intercept] == 1.0)
    ):
        refuse_iols("designs without a unit intercept column")
    if system.dimension < 2:
        refuse_iols("intercept-only designs (at least one slope is required)")
    if system.units.nobs <= system.dimension:
        refuse_iols(
            f"samples with n <= K (n={system.units.nobs}, K={system.dimension})"
        )
    return int(intercept)


def validate_iols_system(
    system: JointSystem,
    *,
    intercept: int | None,
    omega: float = 1.0,
    flags: _IolsRequestFlags = _DEFAULT_REQUEST_FLAGS,
) -> tuple[int, float]:
    """Validate that a system is within the scope of the no-FE IOLS solver."""
    _refuse_unsupported_request(system, flags)
    if not np.isfinite(omega) or omega <= 0.0:
        refuse_iols(
            f"nonpositive mean order omega={omega!r} (including log OLS at omega=0)"
        )
    if isinstance(system.variance, NB2Variance) or not isinstance(
        system.variance, PowerVariance
    ):
        refuse_iols("NB2 working variance")
    tau = float(system.variance.tau)
    if not 1.0 <= tau <= 2.0:
        refuse_iols(f"Power(tau={tau!r}) outside [1, 2]")
    intercept_col = _validate_intercept_and_dimensions(system, intercept)
    return intercept_col, tau


def _map_native_reason(adapter_res: IOLSAdapterResult) -> NativeReason:
    """Map an :class:`IOLSAdapterResult` outcome to :data:`NativeReason`."""
    if adapter_res.native_converged and adapter_res.theta is not None:
        return "native_stop"
    if adapter_res.budget_exhausted:
        return "evaluation_limit"
    if adapter_res.failure_reason in {"not_converged", "phase1_candidate_unusable"}:
        return "iteration_limit"
    if adapter_res.theta is None or adapter_res.failure_reason in {
        "invalid_phi_step",
        "post_convergence_invalid_map",
        "q_convergence_invalid",
    }:
        return "numerical_breakdown"
    return "numerical_breakdown"


@dataclass(frozen=True, eq=False)
class IolsCandidate:
    """The candidate of one no-FE IOLS solve, with its trace (uncertified)."""

    coefficients: FloatArray
    options: SolverOptions
    iols_options: IolsOptions
    reason: NativeReason
    timing: Timing
    outcome: FloatArray
    solver_design: FloatArray
    unscaled_design: FloatArray
    design_scale: FloatArray
    omega: float
    tau: float
    intercept: int
    requested_backend: str
    preferred_backend: Literal["cpp"]
    actual_backend: Literal["cpp", "numpy"]
    fallback_used: bool
    availability_reason: str
    adapter_result: IOLSAdapterResult
    direct_result: IOLSDirectNoFEResult

    def __post_init__(self) -> None:
        """Own the point and the inputs actually handed to the adapter."""
        for name in (
            "coefficients",
            "outcome",
            "solver_design",
            "unscaled_design",
            "design_scale",
        ):
            object.__setattr__(self, name, owned_value(getattr(self, name)))

    @property
    def iterations(self) -> int:
        """The source kernel's iteration counter, without Newton work counts."""
        return int(self.direct_result.iteration["iterations"])

    @property
    def phase1_iterations(self) -> int:
        """Pilot evaluations in Phase 1."""
        return self.adapter_result.phase1_iterations

    @property
    def phase2_iterations(self) -> int:
        """Exact-map evaluations in Phase 2."""
        return self.adapter_result.phase2_iterations

    @property
    def rho_restarts(self) -> int:
        """Step-grid restarts during Phase 2."""
        return self.adapter_result.rho_restarts

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """Full adapter and kernel diagnostic mapping."""
        return self.adapter_result.diagnostics


def solve_iols(
    system: JointSystem,
    *,
    intercept: int | None,
    iols_options: IolsOptions | Mapping[str, object] | None = None,
    cert_options: SolverOptions | None = None,
    unscaled_design: FloatArray | None = None,
    design_scale: FloatArray | None = None,
    omega: float = 1.0,
    flags: _IolsRequestFlags = _DEFAULT_REQUEST_FLAGS,
) -> IolsCandidate:
    """Solve a no-FE ``JointSystem`` with the direct fixed-tau IOLS solver."""
    intercept_col, tau = validate_iols_system(
        system,
        intercept=intercept,
        omega=omega,
        flags=flags,
    )
    resolved_iols = resolve_iols_options(iols_options)
    resolved_cert = SolverOptions() if cert_options is None else cert_options
    control = options_to_kernel_control(resolved_iols)
    backend_req = effective_backend_request(resolved_iols)

    if (unscaled_design is None) != (design_scale is None):
        raise ValueError("unscaled_design and design_scale must be supplied together")
    exog_full = np.asarray(
        system.design if unscaled_design is None else unscaled_design, dtype=np.float64
    )
    scale_vec = np.asarray(
        np.ones(system.dimension) if design_scale is None else design_scale,
        dtype=np.float64,
    )
    if (
        exog_full.shape != system.design.shape
        or scale_vec.shape != (system.dimension,)
        or not np.all(np.isfinite(scale_vec))
        or not np.all(scale_vec > 0.0)
        or scale_vec[intercept_col] != 1.0
        or not np.array_equal(exog_full / scale_vec, system.design)
    ):
        raise ValueError("the design and scale do not reproduce the solver coordinates")

    wall0, cpu0 = time.perf_counter(), time.process_time()
    adapter_res = run_iols_nofe(
        endog=system.z,
        exog_full=exog_full,
        tau=tau,
        control=control,
        backend=backend_req,
        rho_grid_policy="curvature_step_grid",
        intercept_col_idx=intercept_col,
        scale_columns=True,
    )
    timing = Timing(
        wall=time.perf_counter() - wall0,
        cpu=time.process_time() - cpu0,
    )

    assert adapter_res.direct_result is not None
    direct_res = adapter_res.direct_result
    theta_sys: FloatArray | None = None
    if adapter_res.theta is not None and bool(np.all(np.isfinite(adapter_res.theta))):
        mapped = np.asarray(adapter_res.theta * scale_vec, dtype=np.float64)
        mapped[intercept_col] = float(adapter_res.theta[intercept_col])
        if bool(np.all(np.isfinite(mapped))):
            theta_sys = _immutable(mapped)

    coefficients = (
        _immutable(np.full(system.dimension, np.nan, dtype=np.float64))
        if theta_sys is None
        else theta_sys
    )
    return IolsCandidate(
        coefficients=coefficients,
        options=resolved_cert,
        iols_options=resolved_iols,
        reason=_map_native_reason(adapter_res),
        timing=timing,
        outcome=system.z,
        solver_design=system.design,
        unscaled_design=exog_full,
        design_scale=scale_vec,
        omega=float(omega),
        tau=tau,
        intercept=intercept_col,
        requested_backend=adapter_res.requested_backend,
        preferred_backend="cpp",
        actual_backend=adapter_res.actual_backend,
        fallback_used=adapter_res.fallback_used,
        availability_reason=adapter_res.availability_reason,
        adapter_result=adapter_res,
        direct_result=direct_res,
    )
