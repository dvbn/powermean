"""NumPy reference kernel of the optional no-FE IOLS solver.

Re-exports the extracted direct fixed-tau no-FE IOLS reference implementation from
:mod:`powermean.solve.iols.kernel`.
"""

from __future__ import annotations

from powermean.solve.iols.kernel import (
    DIRECT_NOFE_FIXED_TAU_ALGORITHM,
    DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS,
    DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS,
    DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS,
    DIRECT_NOFE_FIXED_TAU_RESULT_KEYS,
    DIRECT_NOFE_FIXED_TAU_ROUTE,
    DIRECT_NOFE_FIXED_TAU_SCHEMA,
    DIRECT_RESTART_START_POLICIES,
    DIRECT_RHO_GRID_POLICIES,
    IOLSIterationDiagnostics,
    IOLSKernelControl,
    _concentrated_fullrecover,
    _ConcentratedSolve,
    _control_to_cpp_mapping,
    _direct_nofe_validate_control,
    _direct_nofe_validated_inputs,
    _direct_policy_rho_path,
    _direct_profiled_curvature_l0,
    _direct_step_rho_grid,
    _FullrecoverContext,
    _FullrecoverInvariantState,
    _phi_step_fullrecover,
    _profile_intercept_adjustment,
    _solve_nofe_direct_fixed_tau,
)

__all__ = [
    "DIRECT_NOFE_FIXED_TAU_ALGORITHM",
    "DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS",
    "DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS",
    "DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS",
    "DIRECT_NOFE_FIXED_TAU_RESULT_KEYS",
    "DIRECT_NOFE_FIXED_TAU_ROUTE",
    "DIRECT_NOFE_FIXED_TAU_SCHEMA",
    "DIRECT_RESTART_START_POLICIES",
    "DIRECT_RHO_GRID_POLICIES",
    "IOLSIterationDiagnostics",
    "IOLSKernelControl",
    "_ConcentratedSolve",
    "_FullrecoverContext",
    "_FullrecoverInvariantState",
    "_concentrated_fullrecover",
    "_control_to_cpp_mapping",
    "_direct_nofe_validate_control",
    "_direct_nofe_validated_inputs",
    "_direct_policy_rho_path",
    "_direct_profiled_curvature_l0",
    "_direct_step_rho_grid",
    "_phi_step_fullrecover",
    "_profile_intercept_adjustment",
    "_solve_nofe_direct_fixed_tau",
]
