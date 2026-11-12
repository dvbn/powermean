"""Selected-coefficient projections and single OrderScale output mapping.

Supports default kept-slope projection ``L = [I_{d_X}, 0]``, explicit 1D
identified-index selectors, and arbitrary 2D linear contrast matrices ``L`` on
the ``d`` identified parameters, followed by a single range-checked
``OrderScale`` mapping at the reporting boundary.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import (
    FloatStorage,
    IndexStorage,
    finite_product,
)
from powermean.fe.geometry_design import NumericalGeometry
from powermean.inference.correction._inputs import checked_array
from powermean.inference.correction.budget import BiasBounds, enforce
from powermean.inference.ssc import direct_errors
from powermean.model.order_scale import OrderScale

_VECTOR_NDIM = 1
_MATRIX_NDIM = 2


def check_rcond_min(rcond_min: float) -> float:
    """Validate reciprocal-condition threshold in ``(0, 1)``."""
    if (
        isinstance(rcond_min, bool)
        or not isinstance(rcond_min, (int, float))
        or not np.isfinite(rcond_min)
        or rcond_min <= 0.0
        or rcond_min >= 1.0
    ):
        raise ValueError("rcond_min must be a finite float in (0, 1)")
    return float(rcond_min)


@dataclass(frozen=True, slots=True)
class BiasEvalOptions:
    """Evaluation options for BBP short-group and cell bias corrections."""

    _order: OrderScale
    _centered_offset: FloatStorage | None
    _obs_weights: FloatStorage | None
    rcond_min: float
    require_symmetric_pairs: bool
    bounds: BiasBounds

    def __init__(
        self,
        *,
        order: OrderScale,
        centered_offset: ArrayLike | None = None,
        obs_weights: ArrayLike | None = None,
        rcond_min: float,
        require_symmetric_pairs: bool = True,
        bounds: BiasBounds,
    ) -> None:
        if type(order) is not OrderScale:
            raise TypeError("order must be an OrderScale")
        if type(bounds) is not BiasBounds:
            raise TypeError("bounds must be a BiasBounds")
        if type(require_symmetric_pairs) is not bool:
            raise TypeError("require_symmetric_pairs must be a bool")
        enforce(64 * order.dimension, 16 * order.dimension, bounds)
        if order.omega == 0.0:
            raise ValueError("BBP correction is not needed at omega=0")
        object.__setattr__(
            self, "_order", OrderScale(order.omega, np.asarray(order.scale).copy())
        )
        object.__setattr__(
            self,
            "_centered_offset",
            _option_vector(centered_offset, "centered_offset", bounds),
        )
        object.__setattr__(
            self, "_obs_weights", _option_vector(obs_weights, "obs_weights", bounds)
        )
        object.__setattr__(self, "rcond_min", check_rcond_min(rcond_min))
        object.__setattr__(self, "require_symmetric_pairs", require_symmetric_pairs)
        object.__setattr__(self, "bounds", bounds)

    @property
    def order(self) -> OrderScale:
        """A fresh view of the immutable selected-coordinate order map."""
        return OrderScale(self._order.omega, self._order.scale.copy())

    @property
    def centered_offset(self) -> NDArray[np.float64] | None:
        """Detached configured offset data; rebuild options to change the input."""
        return None if self._centered_offset is None else self._centered_offset.view()

    @property
    def obs_weights(self) -> NDArray[np.float64] | None:
        """Detached positive multipliers, with no sampling-weight interpretation."""
        return None if self._obs_weights is None else self._obs_weights.view()


def _option_vector(
    value: ArrayLike | None, name: str, bounds: BiasBounds
) -> FloatStorage | None:
    if value is None:
        return None
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise TypeError(f"{name} must be a 1D ndarray")
    enforce(32 * value.size, 8 * value.size, bounds)
    return FloatStorage.store(checked_array(value, value.shape, name))


@dataclass(frozen=True, slots=True)
class ResolvedSelector:
    """Validated selected-coefficient projection ``L`` of shape ``(q, d)``."""

    _matrix: FloatStorage
    _indices: IndexStorage | None
    kind: Literal["default_slopes", "indices", "matrix"]
    n_selected: int
    n_identified: int

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Detached read-only view of the ``(q, d)`` projection matrix ``L``."""
        return self._matrix.view()

    @property
    def indices(self) -> NDArray[np.intp] | None:
        """Detached read-only view of selected identified indices, if indexed."""
        return None if self._indices is None else self._indices.view()

    def project_vector(self, values: ArrayLike) -> NDArray[np.float64]:
        """Compute ``L @ values`` for an identified vector in ``R^d``."""
        vec = checked_array(values, (self.n_identified,), "values")
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(self.matrix @ vec)

    def project_covariance(self, cov_theta: ArrayLike) -> NDArray[np.float64]:
        """Compute ``L @ cov_theta @ L.T`` for a ``(d, d)`` covariance matrix."""
        matrix = checked_array(
            cov_theta, (self.n_identified, self.n_identified), "cov_theta"
        )
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(self.matrix @ matrix @ self.matrix.T)


@dataclass(frozen=True, slots=True)
class OrderMappedBias:
    """Detached outputs before and after applying ``OrderScale`` once."""

    _order: OrderScale
    _uncorrected_transformed: FloatStorage
    _bias_transformed: FloatStorage
    _corrected_transformed: FloatStorage
    _vcov_transformed: FloatStorage
    _uncorrected: FloatStorage
    _bias: FloatStorage
    _corrected: FloatStorage
    _vcov: FloatStorage
    _se: FloatStorage

    @property
    def order(self) -> OrderScale:
        """Detached order-map metadata."""
        return OrderScale(self._order.omega, self._order.scale.copy())

    @property
    def uncorrected_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale uncorrected coefficients ``L theta_hat``."""
        return self._uncorrected_transformed.view()

    @property
    def bias_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale bias ``b_hat_gamma``."""
        return self._bias_transformed.view()

    @property
    def corrected_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale corrected coefficients ``L theta_hat - b_hat``."""
        return self._corrected_transformed.view()

    @property
    def vcov_transformed(self) -> NDArray[np.float64]:
        """Transformed-scale covariance ``L C_theta L'``."""
        return self._vcov_transformed.view()

    @property
    def uncorrected(self) -> NDArray[np.float64]:
        """Reported uncorrected coefficients ``L theta_hat / omega``."""
        return self._uncorrected.view()

    @property
    def bias(self) -> NDArray[np.float64]:
        """Reported bias estimate ``b_hat_gamma / omega``."""
        return self._bias.view()

    @property
    def corrected(self) -> NDArray[np.float64]:
        """Reported bias-corrected coefficients ``(L theta_hat - b_hat) / omega``."""
        return self._corrected.view()

    @property
    def vcov(self) -> NDArray[np.float64]:
        """Reported first-order uncorrected covariance ``L C_theta L' / omega^2``."""
        return self._vcov.view()

    @property
    def se(self) -> NDArray[np.float64]:
        """Reported first-order standard errors at the uncorrected point."""
        return self._se.view()


def _selector_from_indices(
    raw: NDArray[np.generic], n_identified: int
) -> ResolvedSelector:
    """Construct a projection matrix from 1D integer parameter indices."""
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("1D selector must be an integer index array")
    if np.any(np.less(raw, 0)) or np.any(np.greater_equal(raw, n_identified)):
        raise ValueError(f"selector indices must lie in [0, {n_identified})")
    indices = np.array(raw, dtype=np.intp, copy=True)
    n_selected = int(indices.size)
    matrix: NDArray[np.float64] = np.zeros((n_selected, n_identified), dtype=np.float64)
    if n_selected > 0:
        matrix[np.arange(n_selected), indices] = 1.0
    return ResolvedSelector(
        FloatStorage.store(matrix),
        IndexStorage.store(indices.astype(np.intp)),
        "indices",
        n_selected,
        n_identified,
    )


def _selector_from_matrix(
    raw: NDArray[np.generic], n_identified: int
) -> ResolvedSelector:
    """Validate and store an explicit 2D projection matrix ``L``."""
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("2D projection must be a numeric array")
    matrix = checked_array(raw, raw.shape, "projection").copy()
    if matrix.shape[1] != n_identified:
        raise ValueError(
            f"projection must have {n_identified} columns, got {matrix.shape[1]}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("projection must contain only finite values")
    return ResolvedSelector(
        FloatStorage.store(matrix),
        None,
        "matrix",
        int(matrix.shape[0]),
        n_identified,
    )


def resolve_selector(
    geometry: NumericalGeometry,
    projection: ArrayLike | None = None,
    *,
    bounds: BiasBounds,
) -> ResolvedSelector:
    """Resolve a coefficient selector ``L`` on the identified space ``R^d``."""
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must be a NumericalGeometry")
    n_identified = geometry.n_identified
    selected = selector_size(geometry, projection)
    enforce(
        32 * (selected * n_identified + selected),
        8 * (selected * n_identified + selected),
        bounds,
    )
    if projection is None:
        matrix = np.zeros((geometry.n_slopes_identified, n_identified))
        idx = np.arange(geometry.n_slopes_identified)
        matrix[idx, idx] = 1.0
        indices = np.arange(geometry.n_slopes_identified, dtype=np.int64)
        return ResolvedSelector(
            FloatStorage.store(matrix),
            IndexStorage.store(indices.astype(np.intp)),
            "default_slopes",
            geometry.n_slopes_identified,
            n_identified,
        )
    raw = np.asarray(projection)
    if raw.dtype.kind == "b":
        raise TypeError("projection cannot be a boolean array")
    if raw.ndim == _VECTOR_NDIM:
        return _selector_from_indices(raw, n_identified)
    if raw.ndim == _MATRIX_NDIM:
        return _selector_from_matrix(raw, n_identified)
    raise ValueError("projection must be a 1D index array or 2D matrix")


def apply_single_order_mapping(
    uncorrected_gamma: NDArray[np.float64],
    bias_gamma: NDArray[np.float64],
    corrected_gamma: NDArray[np.float64],
    vcov_gamma: NDArray[np.float64],
    *,
    order: OrderScale,
) -> OrderMappedBias:
    """Apply ``OrderScale`` once to transformed-scale bias and covariance outputs."""
    if type(order) is not OrderScale:
        raise TypeError("order must be an OrderScale")
    if order.omega == 0 or order.dimension != uncorrected_gamma.size:
        raise ValueError(
            "order must have nonzero omega and one scale per selected coefficient"
        )
    uncorrected = order.coefficients(uncorrected_gamma)
    bias = order.coefficients(bias_gamma)
    corrected = order.coefficients(corrected_gamma)
    vcov = order.covariance(vcov_gamma)
    se = direct_errors(vcov_gamma, order)
    return OrderMappedBias(
        order,
        FloatStorage.store(uncorrected_gamma),
        FloatStorage.store(bias_gamma),
        FloatStorage.store(corrected_gamma),
        FloatStorage.store(vcov_gamma),
        FloatStorage.store(uncorrected),
        FloatStorage.store(bias),
        FloatStorage.store(corrected),
        FloatStorage.store(vcov),
        FloatStorage.store(se),
    )


def selector_size(geometry: NumericalGeometry, projection: ArrayLike | None) -> int:
    """Read projection dimensions before allocating any selector matrix."""
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must be NumericalGeometry")
    if projection is None:
        size = int(geometry.n_slopes_identified)
    else:
        if not isinstance(projection, np.ndarray) or projection.ndim not in (1, 2):
            raise TypeError("projection must be a 1D/2D ndarray")
        if projection.ndim == 2 and projection.shape[1] != geometry.n_identified:
            raise ValueError(
                "projection must have one column per identified coordinate"
            )
        size = int(projection.shape[0])
    if size == 0 or geometry.n_identified == 0:
        raise ValueError("correction requires a nonempty identified selection")
    return size
