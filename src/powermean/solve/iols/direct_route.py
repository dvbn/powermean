"""Direct fixed-tau no-FE iOLS route with C++ preference and NumPy backup."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from powermean.errors import SolverCapabilityError
from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.kernel import (
    DIRECT_NOFE_FIXED_TAU_ALGORITHM,
    DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS,
    DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS,
    DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS,
    DIRECT_NOFE_FIXED_TAU_RESULT_KEYS,
    DIRECT_NOFE_FIXED_TAU_ROUTE,
    DIRECT_NOFE_FIXED_TAU_SCHEMA,
    DIRECT_RHO_GRID_POLICIES,
    IOLSKernelControl,
    _control_to_cpp_mapping,
    _direct_nofe_validate_control,
    _direct_nofe_validated_inputs,
    _solve_nofe_direct_fixed_tau,
)
from powermean.solve.iols.records import owned_value, plain_value
from powermean.solve.iols.source import source_receipt

# The compiled kernel is not part of this version; the NumPy kernel is.
_NO_COMPILED_KERNEL = (
    "iols_backend='cpp': this version has no compiled kernel; "
    "omit iols_backend or pass iols_backend='numpy'"
)

DirectNoFEBackend = Literal["auto", "cpp", "numpy", "python_reference"]
DIRECT_NOFE_BACKENDS: Final = ("cpp", "numpy")
DIRECT_NOFE_REQUEST_BACKENDS: Final = ("auto", "cpp", "numpy", "python_reference")
NATIVE_IOLS_MODULE_NAME: Final = "powermean._native.iols._kernel_cpp"

_INT_DIAGNOSTICS: Final = frozenset(
    {"phase1_attempts", "phase1_accepted", "phase1_rejected", "phase2_evaluations"}
)
_INT_CONTROLS: Final = frozenset(
    {
        "anderson_m",
        "stopcrit",
        "max_phase1_iterations",
        "max_phase2_evaluations",
        "delta_grid_size",
        "rho_grid_size",
    }
)


@dataclass(frozen=True)
class IOLSBackendResolution:
    """Backend resolution preserving requested, preferred, and actual backends."""

    requested_backend: str
    preferred_backend: Literal["cpp"]
    actual_backend: Literal["cpp", "numpy"]
    fallback_used: bool
    availability_reason: str
    cpp_module: ModuleType | None = None


@dataclass(frozen=True)
class IOLSDirectNoFEResult:
    """Normalized result of a direct fixed-tau no-FE IOLS solve."""

    schema: str
    route: str
    backend: Literal["cpp", "numpy"]
    kernel_backend: str
    algorithm: str
    intercept_policy: str
    tau_requested: float
    tau_fit: float
    phase1_tau: float
    phase2_tau: float
    tau2_anchor_used: bool
    tau_continuation_used: bool
    converged: bool
    failure_reason: str | None
    beta: FloatArray
    intercept: float
    linear: FloatArray | None
    iteration: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    control: Mapping[str, Any]
    backend_details: Mapping[str, Any]
    _input_endog: FloatArray
    _input_slopes: FloatArray
    requested_backend: str = "auto"
    preferred_backend: str = "cpp"
    actual_backend: str = "cpp"
    fallback_used: bool = False
    availability_reason: str = "cpp_available"

    def __post_init__(self) -> None:
        """Detach kernel arrays and recursively own every diagnostic mapping."""
        for name in (
            "beta",
            "linear",
            "iteration",
            "diagnostics",
            "control",
            "backend_details",
            "_input_endog",
            "_input_slopes",
        ):
            object.__setattr__(self, name, owned_value(getattr(self, name)))

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable dictionary copy of the direct result payload."""
        out: dict[str, Any] = {
            "schema": self.schema,
            "route": self.route,
            "backend": self.backend,
            "kernel_backend": self.kernel_backend,
            "algorithm": self.algorithm,
            "intercept_policy": self.intercept_policy,
            "tau_requested": self.tau_requested,
            "tau_fit": self.tau_fit,
            "phase1_tau": self.phase1_tau,
            "phase2_tau": self.phase2_tau,
            "tau2_anchor_used": self.tau2_anchor_used,
            "tau_continuation_used": self.tau_continuation_used,
            "converged": self.converged,
            "failure_reason": self.failure_reason,
            "beta": self.beta.copy(),
            "intercept": self.intercept,
            "linear": None if self.linear is None else self.linear.copy(),
            "requested_backend": self.requested_backend,
            "preferred_backend": self.preferred_backend,
            "actual_backend": self.actual_backend,
            "fallback_used": self.fallback_used,
            "availability_reason": self.availability_reason,
        }
        for key in ("iteration", "diagnostics", "control", "backend_details"):
            out[key] = plain_value(getattr(self, key))
        return out


def resolve_iols_backend(
    backend: DirectNoFEBackend = "auto",
    *,
    module_name: str = NATIVE_IOLS_MODULE_NAME,
) -> IOLSBackendResolution:
    """Resolve the direct no-FE IOLS backend under the strict availability policy.

    Rules
    -----
    - Preferred backend is always ``"cpp"``.
    - Explicit ``"numpy"`` (or internal alias ``"python_reference"``) selects NumPy
      directly with ``availability_reason="explicit_numpy"``.
    - Explicit ``"cpp"`` requires the compiled module to import and expose
      ``solve_nofe_direct_fixed_tau``; any import failure is raised directly.
    - ``"auto"`` prefers ``"cpp"`` and falls back to ``"numpy"`` ONLY when the
      specific native module ``module_name`` is genuinely absent (no module spec
      present and ``ModuleNotFoundError.name == module_name``). Missing transitive
      dependencies, bad ABI, corrupted shared libraries, missing symbols, or
      arbitrary import exceptions are raised and never trigger fallback.
    """
    if not isinstance(backend, str) or backend not in DIRECT_NOFE_REQUEST_BACKENDS:
        raise ValueError(
            f"backend must be one of {DIRECT_NOFE_REQUEST_BACKENDS}, got {backend!r}"
        )

    if backend in {"numpy", "python_reference"}:
        return IOLSBackendResolution(
            requested_backend=backend,
            preferred_backend="cpp",
            actual_backend="numpy",
            fallback_used=False,
            availability_reason="explicit_numpy",
            cpp_module=None,
        )

    spec = importlib.util.find_spec(module_name)

    try:
        cpp_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        is_target_absent = (exc.name == module_name) and (spec is None)
        if backend == "auto" and is_target_absent:
            return IOLSBackendResolution(
                requested_backend="auto",
                preferred_backend="cpp",
                actual_backend="numpy",
                fallback_used=True,
                availability_reason=f"native_module_absent:{module_name}",
                cpp_module=None,
            )
        if backend == "cpp" and is_target_absent:
            raise SolverCapabilityError(
                "solver.unsupported_case",
                {"solver": "iols", "case": _NO_COMPILED_KERNEL},
            ) from exc
        raise

    if getattr(cpp_module, "direct_test_hooks_compiled", False):
        raise ImportError("native IOLS production backend contains test hooks")
    if not callable(getattr(cpp_module, "solve_nofe_direct_fixed_tau", None)):
        raise ImportError(
            f"{module_name} is missing required entry point "
            "'solve_nofe_direct_fixed_tau'"
        )

    return IOLSBackendResolution(
        requested_backend=backend,
        preferred_backend="cpp",
        actual_backend="cpp",
        fallback_used=False,
        availability_reason="cpp_available" if backend == "auto" else "explicit_cpp",
        cpp_module=cpp_module,
    )


def _validate_identity(
    raw: Mapping[str, Any], backend: str, tau_requested: float
) -> None:
    """Retain the research adapter's exact route, backend and tau checks."""
    if (
        raw["tau_requested"] != tau_requested
        or raw["schema"] != DIRECT_NOFE_FIXED_TAU_SCHEMA
        or raw["route"] != DIRECT_NOFE_FIXED_TAU_ROUTE
        or raw["algorithm"] != DIRECT_NOFE_FIXED_TAU_ALGORITHM
        or raw["intercept_policy"] != "profiled_singleton_intercept"
        or raw["backend"] != backend
        or raw["kernel_backend"] != ("cpp" if backend == "cpp" else "python_reference")
        or bool(raw["converged"]) != bool(raw["iteration"]["converged"])
        or raw["failure_reason"] != raw["iteration"]["failure_reason"]
        or raw["tau2_anchor_used"]
        or raw["tau_continuation_used"]
        or any(
            raw[name] != raw["tau_requested"]
            for name in ("tau_fit", "phase1_tau", "phase2_tau")
        )
    ):
        raise RuntimeError("direct no-FE result violates its fixed-tau route schema")


def _require_section_keys(
    section: Mapping[str, Any],
    required: tuple[str, ...],
    label: str,
) -> None:
    missing = set(required).difference(section.keys())
    if missing:
        raise RuntimeError(f"direct no-FE {label} missing keys: {sorted(missing)}")


def _normalize_diagnostics(raw_diag: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for key in DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS:
        if key in _INT_DIAGNOSTICS:
            diagnostics[key] = int(raw_diag[key])
        elif key == "score_validation_status":
            diagnostics[key] = str(raw_diag[key])
        elif key == "score_warning_codes":
            diagnostics[key] = tuple(str(code) for code in raw_diag[key])
        else:
            diagnostics[key] = float(raw_diag[key])
    return diagnostics


def _normalize(
    raw: Mapping[str, Any],
    *,
    resolution: IOLSBackendResolution,
    tau_requested: float,
    inputs: tuple[FloatArray, FloatArray],
) -> IOLSDirectNoFEResult:
    input_shape = inputs[1].shape
    _require_section_keys(raw, DIRECT_NOFE_FIXED_TAU_RESULT_KEYS, "result")
    _validate_identity(raw, resolution.actual_backend, tau_requested)
    raw_iteration = dict(raw["iteration"])
    _require_section_keys(
        raw_iteration, DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS, "iteration"
    )
    raw_diag = dict(raw["diagnostics"])
    _require_section_keys(
        raw_diag, DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS, "diagnostics"
    )
    raw_control = dict(raw["control"])
    _require_section_keys(raw_control, DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS, "control")

    beta: FloatArray = np.array(raw["beta"], dtype=np.float64, copy=True)
    if beta.shape != (input_shape[1],):
        raise RuntimeError("direct no-FE beta has the wrong dimension")
    beta.setflags(write=False)
    linear_raw = raw["linear"]
    linear: FloatArray | None
    if linear_raw is None:
        linear = None
    else:
        linear = np.array(linear_raw, dtype=np.float64, copy=True)
        if linear.shape != (input_shape[0],):
            raise RuntimeError("direct no-FE linear index has the wrong dimension")
        linear.setflags(write=False)

    final_rho = raw_iteration["final_rho"]
    final_criterion = raw_iteration["final_criterion"]
    failure_reason = raw["failure_reason"]
    iter_failure_reason = raw_iteration["failure_reason"]
    iteration = {
        "converged": bool(raw_iteration["converged"]),
        "phase": str(raw_iteration["phase"]),
        "iterations": int(raw_iteration["iterations"]),
        "restarts": int(raw_iteration["restarts"]),
        "final_rho": None if final_rho is None else float(final_rho),
        "rho_path": tuple(float(value) for value in raw_iteration["rho_path"]),
        "final_criterion": (
            None if final_criterion is None else float(final_criterion)
        ),
        "anderson_m_initial": int(raw_iteration["anderson_m_initial"]),
        "anderson_m_final": int(raw_iteration["anderson_m_final"]),
        "failure_reason": (
            None if iter_failure_reason is None else str(iter_failure_reason)
        ),
    }
    diagnostics = _normalize_diagnostics(raw_diag)
    control = {
        key: (
            int(raw_control[key]) if key in _INT_CONTROLS else float(raw_control[key])
        )
        for key in DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS
    }
    backend_name = resolution.actual_backend
    if raw["backend"] != backend_name:
        raise RuntimeError(
            "direct no-FE result backend differs from the selected backend"
        )
    backend_details = {
        **dict(raw["backend_details"]),
        "requested_backend": resolution.requested_backend,
        "preferred_backend": resolution.preferred_backend,
        "actual_backend": resolution.actual_backend,
        "fallback_used": bool(resolution.fallback_used),
        "availability_reason": resolution.availability_reason,
    }
    return IOLSDirectNoFEResult(
        schema=str(raw["schema"]),
        route=str(raw["route"]),
        backend=backend_name,
        kernel_backend=str(raw["kernel_backend"]),
        algorithm=str(raw["algorithm"]),
        intercept_policy=str(raw["intercept_policy"]),
        tau_requested=float(raw["tau_requested"]),
        tau_fit=float(raw["tau_fit"]),
        phase1_tau=float(raw["phase1_tau"]),
        phase2_tau=float(raw["phase2_tau"]),
        tau2_anchor_used=bool(raw["tau2_anchor_used"]),
        tau_continuation_used=bool(raw["tau_continuation_used"]),
        converged=bool(raw["converged"]),
        failure_reason=None if failure_reason is None else str(failure_reason),
        beta=beta,
        intercept=float(raw["intercept"]),
        linear=linear,
        iteration=iteration,
        diagnostics=diagnostics,
        control=control,
        backend_details=backend_details,
        _input_endog=inputs[0],
        _input_slopes=inputs[1],
        requested_backend=resolution.requested_backend,
        preferred_backend=resolution.preferred_backend,
        actual_backend=resolution.actual_backend,
        fallback_used=bool(resolution.fallback_used),
        availability_reason=resolution.availability_reason,
    )


def solve_nofe_direct_fixed_tau(
    endog: npt.ArrayLike,
    exog_slopes: npt.ArrayLike,
    tau: float,
    *,
    backend: DirectNoFEBackend = "auto",
    control: IOLSKernelControl | None = None,
    rho_grid_policy: str = "legacy",
) -> IOLSDirectNoFEResult:
    """Solve no-FE iOLS at the requested fixed tau without a tau=2 anchor."""
    if rho_grid_policy not in DIRECT_RHO_GRID_POLICIES:
        raise ValueError(
            f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}, "
            f"got {rho_grid_policy!r}"
        )
    effective_control = IOLSKernelControl() if control is None else control
    y, x, tau_value = _direct_nofe_validated_inputs(endog, exog_slopes, tau)
    _direct_nofe_validate_control(effective_control)

    resolution = resolve_iols_backend(backend)
    receipt = source_receipt(resolution.cpp_module)
    dispatched_inputs: tuple[FloatArray, FloatArray] = (owned_value(y), owned_value(x))
    if resolution.actual_backend == "numpy":
        raw = _solve_nofe_direct_fixed_tau(
            endog=y,
            exog_slopes=x,
            tau=tau_value,
            control=effective_control,
            rho_grid_policy=rho_grid_policy,
        )
        raw["backend_details"]["source_files"] = receipt
        return _normalize(
            raw,
            resolution=resolution,
            tau_requested=tau_value,
            inputs=dispatched_inputs,
        )
    assert resolution.cpp_module is not None
    raw = resolution.cpp_module.solve_nofe_direct_fixed_tau(
        y,
        x,
        tau_value,
        control=_control_to_cpp_mapping(effective_control),
        rho_grid_policy=rho_grid_policy,
    )
    raw["backend_details"]["source_files"] = receipt
    raw["backend_details"]["native_build"] = dict(resolution.cpp_module.backend_info())
    return _normalize(
        raw, resolution=resolution, tau_requested=tau_value, inputs=dispatched_inputs
    )


__all__ = [
    "DIRECT_NOFE_BACKENDS",
    "DIRECT_NOFE_FIXED_TAU_ALGORITHM",
    "DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS",
    "DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS",
    "DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS",
    "DIRECT_NOFE_FIXED_TAU_RESULT_KEYS",
    "DIRECT_NOFE_FIXED_TAU_ROUTE",
    "DIRECT_NOFE_FIXED_TAU_SCHEMA",
    "DIRECT_NOFE_REQUEST_BACKENDS",
    "DIRECT_RHO_GRID_POLICIES",
    "NATIVE_IOLS_MODULE_NAME",
    "DirectNoFEBackend",
    "IOLSBackendResolution",
    "IOLSDirectNoFEResult",
    "IOLSKernelControl",
    "resolve_iols_backend",
    "solve_nofe_direct_fixed_tau",
]
