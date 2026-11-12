"""Owned point inputs and single-evaluation custom variance callbacks.

These are local data snapshots, never structural proof or cross-call caches.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import FloatStorage, finite_vector
from powermean.inference.correction.budget import BiasBounds
from powermean.model.variance import NB2Variance, PowerVariance, Variance


def checked_array(
    values: ArrayLike, shape: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    """Check shape before any numeric conversion or copy at an isolated boundary."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be an ndarray with declared shape {shape}")
    if values.shape != shape or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def owned_vector(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    """Capture finite vector data on immutable bytes, with fresh view metadata."""
    return FloatStorage.store(checked_array(values, (size,), name)).view()


@dataclass(frozen=True, slots=True)
class PointInputs:
    """Inputs captured before the first user callback at the current point."""

    z: NDArray[np.float64]
    theta: NDArray[np.float64]
    offset: NDArray[np.float64] | None
    weights: NDArray[np.float64] | None


def capture_point(
    z: ArrayLike,
    theta: ArrayLike,
    n: int,
    d: int,
    *,
    offset: ArrayLike | None,
    weights: ArrayLike | None,
) -> PointInputs:
    """Detach current arrays so callbacks cannot change the equation being evaluated."""
    outcomes = owned_vector(z, n, "z")
    point = owned_vector(theta, d, "theta")
    off = None if offset is None else owned_vector(offset, n, "centered_offset")
    mult = None if weights is None else owned_vector(weights, n, "obs_weights")
    if np.any(outcomes < 0):
        raise ValueError("z must be nonnegative")
    if mult is not None and np.any(mult <= 0):
        raise ValueError(
            "obs_weights must be positive; zero-weight policy is unresolved"
        )
    return PointInputs(outcomes, point, off, mult)


@dataclass(frozen=True, slots=True)
class _PointVariance:
    """Three callback results bound to one exact index, used only during this call."""

    q: FloatStorage
    log_v: FloatStorage
    e: FloatStorage
    ep: FloatStorage

    def _check(self, q: ArrayLike) -> None:
        value = finite_vector(q, self.q.shape[0], "callback index")
        if not np.array_equal(value, self.q.view()):
            raise ValueError("point variance cannot be reused at another index")

    def log_value(self, log_mean: ArrayLike) -> NDArray[np.float64]:
        """Return the owned log variance at this point only."""
        self._check(log_mean)
        return self.log_v.view()

    def elasticity(self, log_mean: ArrayLike) -> NDArray[np.float64]:
        """Return the owned elasticity at this point only."""
        self._check(log_mean)
        return self.e.view()

    def elasticity_derivative(self, log_mean: ArrayLike) -> NDArray[np.float64]:
        """Return the owned elasticity derivative at this point only."""
        self._check(log_mean)
        return self.ep.view()

    def to_dict(self) -> dict[str, str | float]:
        """Describe the local data adapter without invoking user callbacks."""
        return {"type": "isolated_current_point"}


def point_variance(variance: Variance, q: NDArray[np.float64]) -> Variance:
    """Evaluate each custom callback once, capturing its result before the next call.

    Built-in exact types retain the accepted stable NB2/power arithmetic. User
    callbacks receive separate views backed by immutable bytes; shape mutation
    cannot alter the next input and data mutation raises the ordinary exception.
    """
    if not isinstance(variance, Variance):
        raise TypeError("variance must implement Variance")
    if type(variance) in (PowerVariance, NB2Variance):
        return variance
    snapshot = FloatStorage.store(q)
    log_v = FloatStorage.store(
        checked_array(variance.log_value(snapshot.view()), q.shape, "log variance")
    )
    e = FloatStorage.store(
        checked_array(variance.elasticity(snapshot.view()), q.shape, "elasticity")
    )
    ep = FloatStorage.store(
        checked_array(
            variance.elasticity_derivative(snapshot.view()),
            q.shape,
            "elasticity derivative",
        )
    )
    return _PointVariance(snapshot, log_v, e, ep)


class BiasOperatorRequest(NamedTuple):
    """Untrusted per-call options, freshly captured by the evaluation boundary.

    Like upstream GeometryInputs this request does not validate or own the caller
    arrays. No request or callback object is retained in an evaluated state.
    """

    bounds: BiasBounds
    centered_offset: ArrayLike | None = None
    obs_weights: ArrayLike | None = None
    cell: bool = False
