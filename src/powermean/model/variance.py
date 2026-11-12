"""Working variances of the transformed outcome, evaluated on the log scale.

A working variance ``V(m)`` weights the estimating equation; it is not an assumption
about the distribution of the outcome.  Every system receives a variance object and
evaluates it at the index ``q = log m``: ``log V``, its first derivative ``e`` (the
elasticity of ``V`` with respect to ``m``) and its second derivative.

Both shares of NB2 are evaluated from the smaller tail, never as one minus a rounded
share, so that at a log-odds of 40 the derivative ``exp(-40)`` is still representable
while the elasticity rounds to 2.

The internal ``PowerVariance`` class admits any finite real ``tau`` for full-family
estimation; the public restriction, ``tau`` in ``[1, 2]``, remains enforced at the
public interface boundary.  An unrepresentable log variance is a ``FloatingPointError``.
No mixture family is included.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.errors import SpecificationError
from powermean.messages import format_keys

FloatArray = NDArray[np.float64]


def _finite_scalar(value: object, name: str) -> float:
    """A finite real number as ``float``; booleans and non-reals are refused.

    The input is typed as ``object`` because the function is the run-time check of
    what the dataclass fields declare as ``float``: it is called on whatever was
    passed, and its first test is a type test.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if result == 0.0:
        return 0.0
    return result


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    """A real array of finite doubles."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_result(value: FloatArray, name: str) -> FloatArray:
    """The value, if every entry is representable in double precision."""
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} is not representable in float64")
    return value


def _logistic_shares(log_odds: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate both shares without subtracting a rounded share from one."""
    tail = np.exp(-np.abs(log_odds))
    small, large = tail / (1 + tail), 1 / (1 + tail)
    positive = log_odds >= 0
    return np.where(positive, large, small), np.where(positive, small, large)


@runtime_checkable
class Variance(Protocol):
    """Log variance and its two derivatives on the transformed log-mean scale."""

    def log_value(self, log_mean: ArrayLike) -> FloatArray:
        """``log V(m)`` at ``m = exp(q)``."""
        ...

    def elasticity(self, log_mean: ArrayLike) -> FloatArray:
        """``e = d log V / d log m`` at ``m = exp(q)``."""
        ...

    def elasticity_derivative(self, log_mean: ArrayLike) -> FloatArray:
        """Second derivative of ``log V(exp(q))`` with respect to ``q``."""
        ...

    def to_dict(self) -> dict[str, str | float]:
        """The family and its parameters, as values that JSON can hold."""
        ...


@dataclass(frozen=True)
class PowerVariance:
    """V(m)=m**tau for any finite real variance power ``tau``."""

    tau: float = 2.0

    def __post_init__(self) -> None:
        """Validate ``tau`` as a finite real scalar and store it as a ``float``."""
        tau = _finite_scalar(self.tau, "tau")
        object.__setattr__(self, "tau", tau)

    def log_value(self, log_mean: ArrayLike) -> FloatArray:
        """``tau q``; an overflow is reported, not clipped."""
        q = _finite_array(log_mean, "log_mean")
        with np.errstate(over="ignore", invalid="ignore"):
            return _finite_result(self.tau * q, "log variance")

    def elasticity(self, log_mean: ArrayLike) -> FloatArray:
        """``tau`` in every row."""
        return np.full_like(_finite_array(log_mean, "log_mean"), self.tau)

    def elasticity_derivative(self, log_mean: ArrayLike) -> FloatArray:
        """Zero in every row."""
        return np.zeros_like(_finite_array(log_mean, "log_mean"))

    def to_dict(self) -> dict[str, str | float]:
        """The family and ``tau``."""
        return {"type": "PowerVariance", "tau": self.tau}


@dataclass(frozen=True)
class NB2Variance:
    """V(m)=m+kappa*m**2, with kappa fixed during coefficient estimation."""

    kappa: float = 1.0

    def __post_init__(self) -> None:
        """Validate ``kappa`` and store it as a ``float``."""
        kappa = _finite_scalar(self.kappa, "kappa")
        if kappa < 0:
            raise ValueError("kappa must be nonnegative")
        object.__setattr__(self, "kappa", kappa)

    def _shares(self, q: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The quadratic share ``kappa m / (1 + kappa m)`` and its complement."""
        if self.kappa == 0:
            return np.zeros_like(q), np.ones_like(q)
        return _logistic_shares(np.log(self.kappa) + q)

    def log_value(self, log_mean: ArrayLike) -> FloatArray:
        """``q + log(1 + kappa m)``, through ``logaddexp`` so that no mean is formed."""
        q = _finite_array(log_mean, "log_mean")
        if self.kappa == 0:
            return q.copy()
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            value = q + np.logaddexp(0.0, np.log(self.kappa) + q)
        return _finite_result(value, "log variance")

    def elasticity(self, log_mean: ArrayLike) -> FloatArray:
        """``1 + kappa m / (1 + kappa m)``, between 1 and 2, from the log share."""
        q = _finite_array(log_mean, "log_mean")
        if self.kappa == 0:
            return np.ones_like(q)
        with np.errstate(over="ignore", under="ignore"):
            log_share = -np.logaddexp(0.0, -np.log(self.kappa) - q)
            value: FloatArray = 1.0 + np.exp(log_share)
        return value

    def elasticity_derivative(self, log_mean: ArrayLike) -> FloatArray:
        """The product of the quadratic share and its complement."""
        share, complement = self._shares(_finite_array(log_mean, "log_mean"))
        return share * complement

    def to_dict(self) -> dict[str, str | float]:
        """The family and ``kappa``."""
        return {"type": "NB2Variance", "kappa": self.kappa}


type CancelledFamily = PowerVariance | NB2Variance

_CANCELLED_DEFAULT = PowerVariance(2.0)


def resolve_cancelled_variance(
    variance: object = None,
    *,
    tau: object = None,
    default: CancelledFamily = _CANCELLED_DEFAULT,
) -> CancelledFamily:
    """Resolve ``tau`` or ``variance`` into a supported full-family variance object.

    Parameters
    ----------
    variance : Variance or None, default None
        Explicit working variance instance (``PowerVariance`` with any finite real
        ``tau`` or ``NB2Variance`` with ``kappa >= 0``).
    tau : float or None, default None
        Shorthand variance power for ``PowerVariance(tau)``; mutually exclusive with
        ``variance``.
    default : PowerVariance or NB2Variance, default PowerVariance(2.0)
        Default family returned when both ``variance`` and ``tau`` are ``None``.
    """
    if tau is not None and variance is not None:
        raise SpecificationError(
            "specification.conflicting", {"first": "tau", "second": "variance"}
        )
    if tau is not None:
        if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, Real):
            raise SpecificationError(
                "specification.invalid_value",
                {"argument": "tau", "value": tau, "valid": "finite real numbers"},
            )
        value = float(tau)
        if not np.isfinite(value):
            raise SpecificationError(
                "specification.invalid_value",
                {"argument": "tau", "value": tau, "valid": "finite real numbers"},
            )
        return PowerVariance(0.0 if value == 0.0 else value)
    if variance is None:
        return default
    if isinstance(variance, (PowerVariance, NB2Variance)):
        return variance
    if isinstance(variance, Variance):
        raise SpecificationError(
            "specification.unimplemented",
            {"feature": f"variance={type(variance).__name__}"},
        )
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "variance",
            "value": variance,
            "valid": format_keys(("PowerVariance", "NB2Variance")),
        },
    )
