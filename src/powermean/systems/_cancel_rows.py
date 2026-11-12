"""Row arithmetic for cancellation.

Reuse the package's log components for its built-in variances: in particular the
small NB2 complement must survive an elasticity that rounds to two. Other Variance
implementations supply freshly checked ``log_value`` and ``elasticity`` arrays.
No outcome/order mapping, criterion or statistical weight policy is defined here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from powermean.fe._cancel_storage import IndexArray, finite_vector
from powermean.model.rows import _exp, _immutable, _log_components, _log_sum
from powermean.model.variance import (
    FloatArray,
    NB2Variance,
    PowerVariance,
    Variance,
    _finite_result,
)


class RowComponents(NamedTuple):
    """Current w, t and log coefficients of the two terms of k."""

    w: FloatArray
    t: FloatArray
    log_t: FloatArray
    log_w: FloatArray
    log_u: FloatArray
    sign_u: FloatArray
    log_v: FloatArray
    sign_v: FloatArray


def _components(
    z: FloatArray,
    q: FloatArray,
    variance: Variance,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Logs of m²/V, mz/V, e-1 and 2-e, with signs of the last two."""
    with np.errstate(divide="ignore"):
        log_z = np.log(z)
    if type(variance) in (PowerVariance, NB2Variance) and isinstance(
        variance, (PowerVariance, NB2Variance)
    ):
        log_w, log_t, log_u, sign_u, log_v, sign_v = _log_components(q, log_z, variance)
        return (
            log_w,
            log_t,
            np.broadcast_to(log_u, q.shape),
            np.full_like(q, sign_u),
            np.broadcast_to(log_v, q.shape),
            np.full_like(q, sign_v),
        )
    # Snapshot before callbacks; each receives a fresh view with detached metadata.
    snapshot = _immutable(q)
    e = finite_vector(variance.elasticity(snapshot.view()), q.size, "elasticity").copy()
    log_value = finite_vector(
        variance.log_value(snapshot.view()), q.size, "log variance"
    ).copy()
    with np.errstate(over="raise", invalid="raise", divide="ignore"):
        log_w = 2.0 * q - log_value
        log_t = log_z + q - log_value
        u, v = e - 1.0, 2.0 - e
        return (
            log_w,
            log_t,
            np.log(np.abs(u)),
            np.sign(u),
            np.log(np.abs(v)),
            np.sign(v),
        )


def row_components(
    z: FloatArray,
    q: FloatArray,
    log_weights: FloatArray,
    variance: Variance,
) -> RowComponents:
    """Evaluate w=a m²/V and t=a mz/V, refusing lost positive row terms."""
    if not isinstance(variance, Variance):
        raise TypeError("variance must implement the package Variance protocol")
    log_w, log_t, log_u, sign_u, log_v, sign_v = _components(z, q, variance)
    with np.errstate(over="raise", invalid="raise"):
        log_w = log_w + log_weights
        log_t = log_t + log_weights
    w, t = _exp(log_w), _exp(log_t)
    if np.any(w <= 0) or np.any(t[z > 0] <= 0):
        raise FloatingPointError("positive cancellation row term underflows float64")
    return RowComponents(w, t, log_t, log_w, log_u, sign_u, log_v, sign_v)


def group_residuals(
    rows: RowComponents,
    group_w: FloatArray,
    group_t: FloatArray,
    codes: IndexArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute pi, rho=t-pi T and k=(e-1)t+(2-e)(T/W)w on the log scale."""
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        log_total_t = np.log(group_t)[codes]
        log_total_w = np.log(group_w)[codes]
        log_pi = rows.log_w - log_total_w
        pi = rows.w / group_w[codes]
        if np.any(pi <= 0):
            raise FloatingPointError(
                "positive cancellation group share underflows float64"
            )
        ones = np.ones_like(pi)
        rho = _finite_result(
            _log_sum(rows.log_t, ones, log_pi + log_total_t, -ones),
            "cancelled residual",
        )
        k = _finite_result(
            _log_sum(
                rows.log_t + rows.log_u,
                rows.sign_u,
                log_total_t - log_total_w + rows.log_w + rows.log_v,
                rows.sign_v,
            ),
            "cancelled Jacobian weight",
        )
    return pi, rho, k
