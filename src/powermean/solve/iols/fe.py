"""Singleton-intercept and one-group centering/profiling helpers for no-FE IOLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import FloatArray, IntArray


@dataclass(frozen=True)
class _ProfileFEGroups:
    """Internal one- or two-way group specification for :func:`profile_fe`."""

    group_i: npt.ArrayLike | None
    group_t: npt.ArrayLike | None
    n_i: int | None
    n_t: int | None


def _as_float_array(values: npt.ArrayLike) -> FloatArray:
    arr: FloatArray = np.asarray(values, dtype=np.float64)
    return arr


def _as_group_array(group: npt.ArrayLike) -> IntArray:
    raw = np.asarray(group)
    if raw.ndim != 1:
        raise ValueError("FE group labels must be one-dimensional")
    if np.issubdtype(raw.dtype, np.integer):
        if raw.size and np.min(raw) < 0:
            raise ValueError("FE group labels must be non-negative integers")
        if raw.size and int(np.max(raw)) <= np.iinfo(np.int32).max:
            res32: IntArray = raw.astype(np.int32, copy=False)
            return res32
        res64: IntArray = raw.astype(np.int64, copy=False)
        return res64

    array: IntArray = np.asarray(raw, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("FE group labels must be one-dimensional")
    if array.size and np.min(array) < 0:
        raise ValueError("FE group labels must be non-negative integers")
    if array.size and int(np.max(array)) <= np.iinfo(np.int32).max:
        arr32: IntArray = array.astype(np.int32, copy=False)
        return arr32
    return array


def _resolve_n_groups(group: IntArray, n_groups: int | None = None) -> int:
    if n_groups is not None:
        return int(n_groups)
    return int(group.max()) + 1 if group.size else 0


def demean_oneway(
    values: npt.ArrayLike,
    group: npt.ArrayLike,
    n_groups: int | None = None,
    *,
    weights: npt.ArrayLike | None = None,
) -> FloatArray:
    """Within-group demean a vector or matrix, optionally with weights."""
    array = _as_float_array(values)
    group_arr = _as_group_array(group)
    if array.shape[0] != group_arr.shape[0]:
        raise ValueError("values and group must have the same number of rows")
    if array.ndim == 2 and array.shape[1] == 0:
        return array.copy()

    n_group = _resolve_n_groups(group_arr, n_groups)
    return _demean_oneway_prepared(
        array,
        group_arr,
        n_group,
        weights=weights,
    )


def _demean_oneway_prepared(
    array: FloatArray,
    group_arr: IntArray,
    n_group: int,
    *,
    weights: npt.ArrayLike | None = None,
) -> FloatArray:
    """Within-group demean for already validated arrays."""
    if weights is None:
        counts: FloatArray = np.bincount(group_arr, minlength=n_group).astype(
            np.float64
        )
        denom: FloatArray = np.maximum(counts, 1.0)
        if array.ndim == 1:
            sums: FloatArray | IntArray = np.bincount(
                group_arr, weights=array, minlength=n_group
            )
            res_1d: FloatArray = array - (sums / denom)[group_arr]
            return res_1d
        out: FloatArray = np.empty_like(array, dtype=np.float64)
        for column in range(array.shape[1]):
            col_sums: FloatArray | IntArray = np.bincount(
                group_arr,
                weights=array[:, column],
                minlength=n_group,
            )
            out[:, column] = array[:, column] - (col_sums / denom)[group_arr]
        return out

    weight_arr = _as_float_array(weights)
    if weight_arr.shape != group_arr.shape:
        raise ValueError("weights and group must have the same shape")
    weight_sum: FloatArray | IntArray = np.bincount(
        group_arr, weights=weight_arr, minlength=n_group
    )
    valid = weight_sum > 1e-15

    if array.ndim == 1:
        w_sums: FloatArray | IntArray = np.bincount(
            group_arr,
            weights=weight_arr * array,
            minlength=n_group,
        )
        means: FloatArray = np.zeros_like(weight_sum, dtype=np.float64)
        means[valid] = w_sums[valid] / weight_sum[valid]
        w_res_1d: FloatArray = array - means[group_arr]
        return w_res_1d

    w_out: FloatArray = np.empty_like(array, dtype=np.float64)
    for column in range(array.shape[1]):
        w_col_sums: FloatArray | IntArray = np.bincount(
            group_arr,
            weights=weight_arr * array[:, column],
            minlength=n_group,
        )
        col_means: FloatArray = np.zeros_like(weight_sum, dtype=np.float64)
        col_means[valid] = w_col_sums[valid] / weight_sum[valid]
        w_out[:, column] = array[:, column] - col_means[group_arr]
    return w_out


def demean(
    values: npt.ArrayLike,
    group_i: npt.ArrayLike | None = None,
    group_t: npt.ArrayLike | None = None,
    n_i: int | None = None,
    n_t: int | None = None,
    *,
    weights: npt.ArrayLike | None = None,
    tol: float = 1e-12,
    maxiter: int = 10000,
) -> FloatArray:
    """Dispatch singleton-intercept / one-group demeaning for the direct no-FE loop."""
    _ = (tol, maxiter)
    groups: list[npt.ArrayLike] = []
    n_groups: list[int | None] = []
    if group_i is not None:
        groups.append(group_i)
        n_groups.append(n_i)
    if group_t is not None:
        groups.append(group_t)
        n_groups.append(n_t)
    if not groups:
        return _as_float_array(values).copy()
    if len(groups) == 1:
        return demean_oneway(values, groups[0], n_groups[0], weights=weights)
    raise ValueError(
        "direct no-FE route supports at most one singleton-intercept group"
    )


def group_logsumexp(
    values: npt.ArrayLike,
    group: npt.ArrayLike,
    n_groups: int,
) -> FloatArray:
    """Groupwise log-sum-exp with empty-group protection."""
    values_arr = _as_float_array(values)
    group_arr = _as_group_array(group)
    return _group_logsumexp_prepared(values_arr, group_arr, int(n_groups))


def _group_logsumexp_prepared(
    values_arr: FloatArray,
    group_arr: IntArray,
    n_group: int,
) -> FloatArray:
    """Groupwise log-sum-exp for already validated arrays."""
    out: FloatArray = np.full(n_group, -np.inf, dtype=np.float64)
    if values_arr.size == 0 or n_group == 0:
        return out

    group_max: FloatArray = np.full(n_group, -np.inf, dtype=np.float64)
    np.maximum.at(group_max, group_arr, values_arr)
    posinf = np.isposinf(group_max)
    finite_groups = np.isfinite(group_max)
    valid_obs = finite_groups[group_arr] & np.isfinite(values_arr)
    if np.any(valid_obs):
        shifted: FloatArray = np.zeros_like(values_arr, dtype=np.float64)
        shifted[valid_obs] = np.exp(
            values_arr[valid_obs] - group_max[group_arr][valid_obs]
        )
        sums: FloatArray | IntArray = np.bincount(
            group_arr, weights=shifted, minlength=n_group
        )
        valid_sums = finite_groups & (sums > 0.0)
        out[valid_sums] = group_max[valid_sums] + np.log(sums[valid_sums])
    out[posinf] = np.inf
    return out


def _profile_one_effect(
    endog: FloatArray,
    partial_linear: FloatArray,
    group: IntArray,
    n_groups: int,
    tau: float,
) -> FloatArray:
    positive = endog > 0.0
    log_numerator_terms: FloatArray = np.full(
        partial_linear.shape[0], -np.inf, dtype=np.float64
    )
    log_numerator_terms[positive] = (
        np.log(endog[positive]) + (1.0 - tau) * partial_linear[positive]
    )
    log_denominator_terms: FloatArray = (2.0 - tau) * partial_linear
    log_numerator = _group_logsumexp_prepared(
        log_numerator_terms,
        group,
        int(n_groups),
    )
    log_denominator = _group_logsumexp_prepared(
        log_denominator_terms,
        group,
        int(n_groups),
    )

    effect: FloatArray = np.zeros(int(n_groups), dtype=np.float64)
    valid = np.isfinite(log_numerator) & np.isfinite(log_denominator)
    effect[valid] = log_numerator[valid] - log_denominator[valid]
    return effect


def profile_fe(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike,
    beta: npt.ArrayLike,
    groups: _ProfileFEGroups,
    *,
    tau: float = 2.0,
    maxiter: int = 10000,
    tol: float = 1e-10,
) -> tuple[FloatArray, FloatArray | None]:
    """Profile singleton-intercept / one-group FE by stable log-sum-exp update."""
    _ = (maxiter, tol)
    y = _as_float_array(endog)
    x = _as_float_array(exog)
    b = _as_float_array(beta)
    if y.ndim != 1:
        raise ValueError("endog must be one-dimensional")
    if x.ndim != 2:
        raise ValueError("exog must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("endog and exog must have the same row count")

    base_linear: FloatArray = x @ b
    group_i = groups.group_i
    group_t = groups.group_t
    n_i = groups.n_i
    n_t = groups.n_t
    if group_i is None and group_t is None:
        return np.zeros(0, dtype=np.float64), None
    if group_i is None:
        group_i = group_t
        n_i = n_t
        group_t = None
        n_t = None

    assert group_i is not None
    i_arr = _as_group_array(group_i)
    i_count = _resolve_n_groups(i_arr, n_i)
    if group_t is None:
        return _profile_one_effect(y, base_linear, i_arr, i_count, tau), None
    raise ValueError(
        "direct no-FE route supports at most one singleton-intercept group"
    )
