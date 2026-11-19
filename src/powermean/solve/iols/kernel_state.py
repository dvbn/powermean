"""Profiled intercept, demeaning, fixed-point map, and score helpers for IOLS."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import (
    FloatArray,
    IntArray,
)
from powermean.solve.iols.kernel_records import (
    FULLRECOVER_JOINT_FE_POLICY,
    FULLRECOVER_PROFILED_FE_POLICY,
    TAU2_BOUNDARY_DIRECTION_TOL,
    TAU2_BOUNDARY_FUTURE_POLICY,
    TAU2_BOUNDARY_SUPPORT_POLICY,
    TAU2_JOINT_HDFE_SUPPORT_CHECK,
    TAU2_RECOVERED_FE_SUPPORT_CHECK,
    IOLSKernelControl,
    _FullrecoverContext,
    _FullrecoverInvariantState,
)


def _is_singleton_intercept_group(
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
) -> bool:
    if len(groups) != 1 or tuple(int(count) for count in counts) != (1,):
        return False
    group = np.asarray(groups[0], dtype=int)
    return bool(group.size == 0 or np.all(group == 0))


def _logsumexp_all(values: FloatArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("-inf")
    if np.any(np.isposinf(array)):
        return float("inf")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("-inf")
    max_value = float(np.max(finite))
    return max_value + float(np.log(np.sum(np.exp(finite - max_value))))


def _profile_intercept_adjustment(
    *,
    endog: npt.ArrayLike,
    linear: npt.ArrayLike,
    tau: float,
) -> float:
    endog_arr = np.asarray(endog, dtype=np.float64)
    linear_arr = np.asarray(linear, dtype=np.float64)
    positive = endog_arr > 0.0
    log_numerator_terms: FloatArray = np.full(
        linear_arr.shape[0], -np.inf, dtype=np.float64
    )
    log_numerator_terms[positive] = (
        np.log(endog_arr[positive]) + (1.0 - tau) * linear_arr[positive]
    )
    log_denominator_terms: FloatArray = (2.0 - tau) * linear_arr
    log_numerator = _logsumexp_all(log_numerator_terms)
    log_denominator = _logsumexp_all(log_denominator_terms)
    if np.isfinite(log_numerator) and np.isfinite(log_denominator):
        return float(log_numerator - log_denominator)
    return 0.0


def _demean_singleton_intercept(
    values: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> FloatArray:
    array: FloatArray = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.copy()
    if weights is None:
        # An overflow is a mean of inf; the kernel then refuses as before.
        with np.errstate(over="ignore", invalid="ignore"):
            unweighted: FloatArray = array - np.mean(array, axis=0)
        return unweighted

    weight_arr: FloatArray = np.asarray(weights, dtype=np.float64)
    if weight_arr.shape != (array.shape[0],):
        raise ValueError("weights and group must have the same shape")
    weight_sum = float(np.sum(weight_arr))
    if weight_sum <= 1e-15:
        return array.copy()
    if array.ndim == 1:
        mean_1d = float(np.sum(weight_arr * array) / weight_sum)
        res_1d: FloatArray = array - mean_1d
        return res_1d
    mean_2d: FloatArray = np.sum(weight_arr[:, None] * array, axis=0) / weight_sum
    res_2d: FloatArray = array - mean_2d
    return res_2d


def _profile_linear(
    *,
    endog: FloatArray,
    exog: FloatArray,
    beta: FloatArray,
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
    tau: float,
    maxiter: int = 10000,
    tol: float = 1e-6,
) -> FloatArray:
    _ = (tol, maxiter)
    base_linear: FloatArray = exog @ beta
    if len(groups) == 0:
        return base_linear
    if _is_singleton_intercept_group(groups, counts):
        adjusted: FloatArray = base_linear + _profile_intercept_adjustment(
            endog=endog,
            linear=base_linear,
            tau=tau,
        )
        return adjusted
    raise NotImplementedError(
        "Fixed-effect profiling is outside the no-FE direct route."
    )


def _demean_for_groups(
    values: npt.ArrayLike,
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
    *,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> FloatArray:
    _ = (tol, maxiter)
    if len(groups) == 0:
        copied: FloatArray = np.asarray(values, dtype=np.float64).copy()
        return copied
    if _is_singleton_intercept_group(groups, counts):
        return _demean_singleton_intercept(values)
    raise NotImplementedError(
        "Fixed-effect demeaning is outside the no-FE direct route."
    )


def _uses_joint_hdfe_state(groups: tuple[IntArray, ...]) -> bool:
    return len(groups) >= 2


def _fullrecover_fe_policy(groups: tuple[IntArray, ...]) -> str:
    if _uses_joint_hdfe_state(groups):
        return FULLRECOVER_JOINT_FE_POLICY
    return FULLRECOVER_PROFILED_FE_POLICY


def _has_recovered_fe_support_scope(
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
) -> bool:
    if len(groups) == 0:
        return False
    if len(groups) == 1 and counts == (1,):
        group = np.asarray(groups[0], dtype=int)
        return bool(group.size and not np.all(group == 0))
    return True


def _recover_fe_linear(
    values: npt.ArrayLike,
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
    *,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> FloatArray:
    values_arr: FloatArray = np.asarray(values, dtype=np.float64)
    if len(groups) == 0:
        zeros: FloatArray = np.zeros_like(values_arr, dtype=np.float64)
        return zeros
    diff: FloatArray = values_arr - _demean_for_groups(
        values_arr,
        groups,
        counts,
        tol=tol,
        maxiter=maxiter,
    )
    return diff


def _fixed_design_projection(x: FloatArray) -> FloatArray:
    """Solve the Gram normal equations with their original relative rank cutoff."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        gram = x.T @ x
        # On the ordinary path only the small Gram is inspected.
        small = gram.diagonal() < np.finfo(np.float64).tiny
        if not np.all(np.isfinite(gram)) or (small.any() and np.any(x[:, small])):
            largest = float(np.max(np.abs(x), initial=0.0))
            if 0.0 < largest < np.inf:
                shift = int(np.frexp(largest)[1])
                z = np.ldexp(x, -shift)
                # A mixed-scale design may need a smaller shift to keep its
                # smallest entries exact. Find the largest reversible shift.
                if shift > 0 and not np.array_equal(np.ldexp(z, shift), x):
                    low, high = 0, shift
                    while high - low > 1:
                        mid = (low + high) // 2
                        if np.array_equal(np.ldexp(np.ldexp(x, -mid), mid), x):
                            low = mid
                        else:
                            high = mid
                    shift = low
                    z = np.ldexp(x, -shift)
                scaled_gram = z.T @ z
                if shift and np.all(np.isfinite(scaled_gram)):
                    scaled = np.linalg.lstsq(scaled_gram, z.T, rcond=None)[0]
                    projection: FloatArray = np.ldexp(scaled, -shift)
                    if np.all(np.isfinite(projection)):
                        return projection
    ordinary: FloatArray = np.linalg.lstsq(gram, x.T, rcond=None)[0]
    return ordinary


def _fullrecover_invariant_state(
    *,
    exog: FloatArray,
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> _FullrecoverInvariantState:
    x_demeaned = _demean_for_groups(
        exog,
        groups,
        counts,
        tol=float(control.fe_demean_tol),
        maxiter=int(control.fe_demean_maxiter),
    )
    return _FullrecoverInvariantState(
        x_demeaned=x_demeaned,
        projection=_fixed_design_projection(x_demeaned),
    )


def _build_fullrecover_context(
    *,
    endog: FloatArray,
    exog: FloatArray,
    groups: tuple[IntArray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
    invariant_state: _FullrecoverInvariantState | None = None,
) -> _FullrecoverContext:
    state = (
        invariant_state
        if invariant_state is not None
        else _fullrecover_invariant_state(
            exog=exog,
            groups=groups,
            counts=counts,
            control=control,
        )
    )
    return _FullrecoverContext(
        endog=endog,
        exog=exog,
        groups=groups,
        counts=counts,
        tau=tau,
        control=control,
        x_demeaned=state.x_demeaned,
        projection=state.projection,
    )


def _recovered_fe_tau2_boundary_separation(
    context: _FullrecoverContext,
) -> dict[str, Any]:
    """Detect Gamma/tau=2 boundary separation for recovered-FE designs."""
    y = np.asarray(context.endog, dtype=np.float64)
    positive = y > 0.0
    nonpositive = ~positive
    positive_count = int(np.sum(positive))
    nonpositive_count = int(np.sum(nonpositive))
    joint = _uses_joint_hdfe_state(context.groups)
    support_check = (
        TAU2_JOINT_HDFE_SUPPORT_CHECK if joint else TAU2_RECOVERED_FE_SUPPORT_CHECK
    )
    support_scope = "joint_hdfe" if joint else "recovered_fe"
    details: dict[str, Any] = {
        "support_check": support_check,
        "support_check_scope": support_scope,
        "support_policy": TAU2_BOUNDARY_SUPPORT_POLICY,
        "future_support_policy": TAU2_BOUNDARY_FUTURE_POLICY,
        "positive_observations": positive_count,
        "nonpositive_observations": nonpositive_count,
        "separation_detected": False,
        "separation_certificate": (
            "exists v in span(X, recovered FE) with v[y>0] >= 0 and sum(v) <= -1"
        ),
        "direction_tolerance": TAU2_BOUNDARY_DIRECTION_TOL,
    }
    if not _has_recovered_fe_support_scope(
        context.groups, context.counts
    ) or not np.any(nonpositive):
        return details
    raise NotImplementedError(
        "Non-singleton recovered-FE boundary LP is outside the direct no-FE route."
    )


def _initial_linear_state(
    context: _FullrecoverContext,
    beta: FloatArray,
) -> FloatArray:
    if not _uses_joint_hdfe_state(context.groups):
        return _profile_linear(
            endog=context.endog,
            exog=context.exog,
            beta=beta,
            groups=context.groups,
            counts=context.counts,
            tau=context.tau,
            maxiter=int(context.control.fe_profile_maxiter),
            tol=float(context.control.fe_profile_tol),
        )
    base: FloatArray = context.exog @ np.asarray(beta, dtype=np.float64)
    target: FloatArray = np.log(np.asarray(context.endog, dtype=np.float64) + 1.0)
    recovered = _recover_fe_linear(
        target - base,
        context.groups,
        context.counts,
        tol=float(context.control.fe_demean_tol),
        maxiter=int(context.control.fe_demean_maxiter),
    )
    total: FloatArray = base + recovered
    return total
