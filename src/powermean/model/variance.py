"""Working variances of the transformed outcome, evaluated on the log scale.

A working variance ``V(m)`` weights the estimating equation; it is not an assumption
about the distribution of the outcome.  Every system receives a variance object and
evaluates it at the index ``q = log m``: ``log V``, its first derivative ``e`` (the
elasticity of ``V`` with respect to ``m``) and its second derivative.

Design: ``design/01_statistical_core.md``, Section 4.  Port of ``variance.py`` of
expreg (source S7 of ``design/06_reuse_map.md``): the protocol, the two families,
the validation of parameters and inputs, ``log_value``, ``elasticity`` and
``to_dict`` keep that source's structure, names and arithmetic.  The second
derivative ``elasticity_derivative`` and the share arithmetic ``_logistic_shares``
come from ``families.py`` of pyiols (source S6).  Both shares of NB2 are evaluated
from the smaller tail, never as one minus a rounded share, so that at a log-odds of
40 the derivative ``exp(-40)`` is still representable while the elasticity rounds to
2.  ``PROVENANCE.md`` records the sources and the changes.

The internal ``PowerVariance`` class admits any finite real ``tau`` for M2 full-family
estimation; the public restriction of the first milestone, ``tau`` in ``[1, 2]``,
remains enforced at the public interface boundary (decision D41).  An unrepresentable
log variance is a ``FloatingPointError``.  No mixture family is included.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
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


def _mean(z: FloatArray, weights: FloatArray | None) -> float:
    """Use NumPy's mean on ordinary inputs and recover its range failures.

    A lost weighted product or overflowing weight total can leave a finite mean.
    Detect the arithmetic failure at its source before accepting that result.
    """
    with np.errstate(over="raise", under="raise", invalid="ignore"):
        try:
            return float(np.average(z, weights=weights))
        except FloatingPointError:
            pass

    values = np.asarray(z, dtype=np.float64)
    mass = None if weights is None else np.asarray(weights, dtype=np.float64)
    if not np.isfinite(values).all() or (
        mass is not None and not np.isfinite(mass).all()
    ):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return float(np.average(values, weights=mass))

    if mass is not None:
        active = mass != 0
        values, mass = values[active], mass[active]
    with np.errstate(over="raise", under="raise"):
        try:
            scaled, exponent = _reversible_scale(values)
            scaled_mass = None
            if mass is not None:
                scaled_mass, _ = _reversible_scale(mass)
            centre = min(
                float(np.average(scaled, weights=scaled_mass)), float(np.max(scaled))
            )
            return float(np.ldexp(centre, exponent))
        except FloatingPointError:
            # Opposed scales can lose a tiny weight on a large outcome even
            # after rescaling both arrays. Round only the final ratio here.
            return _exact_mean(values, mass)


def _reversible_scale(values: FloatArray) -> tuple[FloatArray, int]:
    """Scale by a power of two only when every input survives exactly.

    Some NumPy implementations of ldexp return rounded subnormals or zero
    without raising underflow. Reversing the shift checks both cases.
    """
    exponent = int(np.frexp(np.max(np.abs(values)))[1])
    scaled = np.ldexp(values, -exponent)
    if not np.array_equal(np.ldexp(scaled, exponent), values):
        raise FloatingPointError("binary rescaling lost a weighted-mean input")
    return scaled, exponent


def _exact_mean(values: FloatArray, mass: FloatArray | None) -> float:
    """Recover the mean of finite doubles when common binary scales are insufficient."""
    if mass is None:
        return float(
            sum((Fraction(float(x)) for x in values), Fraction()) / values.size
        )
    numerator, denominator = Fraction(), Fraction()
    for value, weight in zip(values, mass, strict=True):
        w = Fraction(float(weight))
        numerator += Fraction(float(value)) * w
        denominator += w
    return float(numerator / denominator)


def score_unit(z: FloatArray, weights: FloatArray | None, variance: object) -> float:
    """The unit of the averaged score of a fit: ``mean(z) ** (2 - tau)``.

    Under a power variance the averaged score and the Newton decrement are homogeneous
    of degree ``2 - tau`` in the unit of the outcome ``z`` (degree 0 at ``tau = 2``).
    A rule that compares the score with a pure number divides it by this unit first,
    so that it decides the same whatever the unit.  1 for any other variance, and for
    a mean that is not positive and finite.
    """
    if type(variance) is not PowerVariance or variance.tau == 2:
        return 1.0
    mean = _mean(z, weights)
    if not (np.isfinite(mean) and mean > 0):
        return 1.0
    return float(mean ** (2.0 - variance.tau))


def cancelled_score_unit(z: FloatArray, weights: FloatArray | None) -> float:
    """The unit of the cancelled moment of a fit: ``mean(z)``.

    The cancelled moment ``S = B' rho``, ``rho = t - pi T_g`` with ``t = a m z / V``,
    and its negative Jacobian are linear in the outcome ``z`` whatever the working
    variance, because the cancelled effect, not the index, carries the level of the
    outcome.  1 for a mean that is not positive and finite.
    """
    mean = _mean(z, weights)
    if not (np.isfinite(mean) and mean > 0):
        return 1.0
    return mean


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
