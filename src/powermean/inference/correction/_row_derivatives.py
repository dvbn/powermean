"""Tensor-free D2S primitives at one captured point.

k_dot is the partial derivative of k at fixed a=T/W. NB2 uses both logistic
shares directly. Cell scale and lambda are deliberately separate: short-group
correction never needs a representable Z/m, lambda, or fitted mean.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from powermean.fe._cancel_storage import FloatStorage, finite_product
from powermean.fe.cancel import to_sorted
from powermean.inference.correction._inputs import PointInputs
from powermean.model.rows import _exp, _log_sum
from powermean.model.variance import NB2Variance, PowerVariance, Variance
from powermean.systems._cancel_rows import RowComponents, row_components
from powermean.systems.cancelled import CancelledState


@dataclass(frozen=True, slots=True)
class BiasRowPrimitives:
    """Owned sorted weights for the second derivative, without cell reporting."""

    _k_dot: FloatStorage
    _two_minus_e: FloatStorage
    _eta_weights: FloatStorage

    @property
    def k_dot(self) -> NDArray[np.float64]:
        """Partial dk/dq holding a_g fixed."""
        return self._k_dot.view()

    @property
    def two_minus_e(self) -> NDArray[np.float64]:
        """Complement computed without subtracting a rounded NB2 elasticity."""
        return self._two_minus_e.view()

    @property
    def eta_weights(self) -> NDArray[np.float64]:
        """(2-e) pi in sorted rows."""
        return self._eta_weights.view()


@dataclass(frozen=True, slots=True)
class CellRowPrimitives:
    """Owned data needed only by the explicitly requested cell formula."""

    _lambda_weights: FloatStorage
    _u_lambda_weights: FloatStorage
    _log_cancel_scale: FloatStorage
    _fitted_mean: FloatStorage
    _cell_residual: FloatStorage

    @property
    def lambda_weights(self) -> NDArray[np.float64]:
        """lambda=a*m/V in sorted rows; a denotes the equation multiplier here."""
        return self._lambda_weights.view()

    @property
    def u_lambda_weights(self) -> NDArray[np.float64]:
        """(e-1) lambda, retaining the small NB2 share."""
        return self._u_lambda_weights.view()

    @property
    def group_cancel_scale(self) -> NDArray[np.float64]:
        """Ordinary within-group average of Z/m, independent of the multipliers."""
        logs = self._log_cancel_scale.view()
        scale = _exp(logs)
        if np.any(np.isfinite(logs) & (scale == 0)):
            raise FloatingPointError("positive cancellation scale underflows float64")
        return scale

    @property
    def fitted_mean(self) -> NDArray[np.float64]:
        """Ahat_g*m_i, evaluated through the log product, never exp(q) alone."""
        return self._fitted_mean.view()

    @property
    def cell_residual(self) -> NDArray[np.float64]:
        """Z_i-Ahat_g*m_i in sorted order."""
        return self._cell_residual.view()


def _k_dot(
    rows: RowComponents,
    log_aw: NDArray[np.float64],
    q: NDArray[np.float64],
    variance: Variance,
) -> NDArray[np.float64]:
    """(e'-(e-1)^2)t+((2-e)^2-e')aw; signed log sums preserve cancellation."""
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        if type(variance) is PowerVariance:
            return _log_sum(
                rows.log_t + 2 * rows.log_u,
                -np.ones_like(q),
                log_aw + 2 * rows.log_v,
                np.ones_like(q),
            )
        if type(variance) is NB2Variance:
            # u'=u*v and v'=-u*v imply coefficients u(v-u), v(v-u).
            u, v = np.exp(rows.log_u), np.exp(rows.log_v)
            delta = v - u
            log_delta = np.log(np.abs(delta))
            return _log_sum(
                rows.log_t + rows.log_u + log_delta,
                np.sign(delta),
                log_aw + rows.log_v + log_delta,
                np.sign(delta),
            )
        e = variance.elasticity(q)
        ep = variance.elasticity_derivative(q)
        c1 = finite_product(ep - (e - 1.0) ** 2)
        c2 = finite_product((2.0 - e) ** 2 - ep)
        return _log_sum(
            rows.log_t + np.log(np.abs(c1)),
            np.sign(c1),
            log_aw + np.log(np.abs(c2)),
            np.sign(c2),
        )


def compute_bias_row_primitives(
    state: CancelledState, point: PointInputs, variance: Variance, *, cell: bool
) -> tuple[BiasRowPrimitives, CellRowPrimitives | None]:
    """Evaluate captured rows without another invocation of the original callback."""
    s = state.design.structure
    z = to_sorted(s, point.z)
    log_weights = (
        np.zeros(s.n_rows)
        if point.weights is None
        else np.log(to_sorted(s, point.weights))
    )
    rows = row_components(z, state.q, log_weights, variance)
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        log_aw = (
            np.log(state.group_t)[s.sorted_codes]
            - np.log(state.group_w)[s.sorted_codes]
            + rows.log_w
        )
        complement = finite_product(rows.sign_v * _exp(rows.log_v))
        kd = finite_product(_k_dot(rows, log_aw, state.q, variance))
        eta = finite_product(complement * state.pi)
    primitives = BiasRowPrimitives(
        FloatStorage.store(kd), FloatStorage.store(complement), FloatStorage.store(eta)
    )
    return primitives, _cell_primitives(state, z, rows) if cell else None


def _cell_primitives(
    state: CancelledState, z: NDArray[np.float64], rows: RowComponents
) -> CellRowPrimitives:
    """Cell Ahat is ordinary mean(Z/m); evaluate positive quantities on logs.

    The scale is retained on logs and exponentiated only if explicitly requested.
    Positive fitted means/lambda lost to underflow refuse. All-zero groups and
    singleton residuals are exactly zero.
    """
    s, q = state.design.structure, state.q
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        lam = _exp(rows.log_w - q)
        u_lam = finite_product(rows.sign_u * _exp(rows.log_u + rows.log_w - q))
        log_ratio = np.log(z) - q
        # Group log-sum-exp avoids overflowing Z/m or its sum, and retains
        # contributions smaller than a representable per-row divided term.
        log_scale = np.logaddexp.reduceat(log_ratio, s.starts) - np.log(s.counts)
        mean = _exp(log_scale[s.sorted_codes] + q)
        if np.any(np.isfinite(log_scale[s.sorted_codes]) & (mean == 0)):
            raise FloatingPointError("positive fitted cell mean underflows float64")
        if np.any(lam == 0):
            raise FloatingPointError("positive lambda underflows float64")
        singleton = s.counts[s.sorted_codes] == 1
        mean[singleton] = z[singleton]
        residual = finite_product(z - mean)
    return CellRowPrimitives(
        *(FloatStorage.store(x) for x in (lam, u_lam, log_scale, mean, residual))
    )
