"""Linear coefficient targets and their transformations, design/04 Sections 7/9.

For a fixed design contrast D, l = D'beta and var(l) = D'VD, including all
cross terms. Finite changes report expm1(l), with gradient exp(l)D. Local
semi-elasticities use the full design derivative D; elasticities use xD at an
explicitly supplied x. They are linear targets, not exponentiated finite changes.
V is already reported, selected and scaled; no SSC or order map is applied here.
"""

from __future__ import annotations

import importlib
import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, cast

import numpy as np

from powermean.inference.ssc import ReferenceDistribution
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

TargetKind = Literal["finite_change", "semi_elasticity", "elasticity"]
IntervalKind = Literal["transform", "delta"]


@dataclass(frozen=True, slots=True)
class ReportingPoint:
    """Actual reporting ownership; certification remains at the original root."""

    status: str
    coefficient: str = "certified uncorrected root"
    covariance: str = "certified uncorrected root"


@dataclass(frozen=True, slots=True)
class ReportingContext:
    """Small immutable numeric snapshot, detached from any retained fit graph.

    Metadata is an opaque immutable JSON string produced by the interface. All
    numeric coordinates are tuples, so neither shape edits nor writable wrappers
    can reach retained arrays. The actual normal/t reference carries its own df.
    """

    names: tuple[str, ...]
    beta: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    reference: ReferenceDistribution
    omega: float
    point: ReportingPoint
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        """Require one aligned reported coordinate system, without repairing V."""
        size = len(self.names)
        if len(set(self.names)) != size or len(self.beta) != size:
            raise ValueError("coefficient names and beta must be unique and aligned")
        if len(self.covariance) != size or any(
            len(row) != size for row in self.covariance
        ):
            raise ValueError("covariance must have the full reported coefficient basis")
        if not all(math.isfinite(value) for value in self.beta):
            raise FloatingPointError("the reported coefficient point is not finite")


@dataclass(frozen=True, slots=True)
class LinearTarget:
    """A fixed coefficient gradient in exactly the reporting context's order."""

    weights: tuple[float, ...]
    kind: TargetKind = "finite_change"

    def __post_init__(self) -> None:
        """Refuse undefined gradients and unknown target kinds."""
        if not all(math.isfinite(value) for value in self.weights):
            raise ValueError("target weights must be finite")
        if self.kind not in ("finite_change", "semi_elasticity", "elasticity"):
            raise ValueError("unknown coefficient target kind")


@dataclass(frozen=True, slots=True)
class Inference:
    """A scalar estimate with pointwise inference; zero ratios stay undefined."""

    estimate: float
    std_error: float
    statistic: float
    p_value: float
    conf_low: float
    conf_high: float
    status: str


@dataclass(frozen=True, slots=True)
class TargetEstimate:
    """The base linear target and its reported transformation/gradient."""

    linear: Inference
    reported: Inference
    gradient: tuple[float, ...]
    variance: float
    kind: TargetKind


class _StudentT(Protocol):
    """SciPy operations required for actual t-reference inference."""

    def ppf(self, quantile: float, df: int) -> float:
        """A Student-t quantile."""
        ...

    def sf(self, statistic: float, df: int) -> float:
        """A Student-t upper tail probability."""
        ...


def reference_values(
    reference: ReferenceDistribution, level: float, statistic: float
) -> tuple[float, float]:
    """Actual normal/t critical value and two-sided p-value, without guessing df."""
    if not isinstance(level, float) or not 0.0 < level < 1.0:
        raise ValueError("level must be a float strictly between zero and one")
    if reference.kind == "normal":
        return (
            statistics.NormalDist().inv_cdf((1.0 + level) / 2.0),
            math.erfc(abs(statistic) / math.sqrt(2.0)),
        )
    if reference.kind != "t" or reference.df is None or reference.df <= 0:
        raise ValueError("a Student-t reference needs positive degrees of freedom")
    student_t = cast("_StudentT", importlib.import_module("scipy.stats").t)
    return (
        float(student_t.ppf((1.0 + level) / 2.0, reference.df)),
        float(2.0 * student_t.sf(abs(statistic), reference.df)),
    )


def derivative_target(
    derivative: tuple[float, ...], *, evaluation_value: float | None = None
) -> LinearTarget:
    """Private local target from the full learned-design derivative.

    No evaluation value is selected implicitly. Supplying x returns the elasticity
    gradient xD; omitting it returns the semi-elasticity gradient D. For log(x),
    the supplied derivative includes 1/x, yielding beta when multiplied by x.
    """
    if evaluation_value is None:
        return LinearTarget(derivative, "semi_elasticity")
    if not math.isfinite(evaluation_value):
        raise ValueError("the elasticity evaluation value must be finite")
    return LinearTarget(
        tuple(evaluation_value * value for value in derivative), "elasticity"
    )


# Above this magnitude every binary64 operand is a multiple of 2**-340.
# Products of two or three such operands and their sums cannot be subnormal.
_MIN_CONTRACTION_OPERAND = 2.0**-288


def _has_tiny_operand(values: FloatArray) -> bool:
    magnitude = np.abs(values)
    return bool(np.any((magnitude < _MIN_CONTRACTION_OPERAND) & (magnitude != 0.0)))


def _exact_dot(values: FloatArray, weights: list[Fraction]) -> Fraction:
    """An unrounded sum of products of finite binary64 operands."""
    return sum(
        (
            Fraction(float(value)) * weight
            for value, weight in zip(values, weights, strict=True)
        ),
        Fraction(),
    )


def _bilinear(
    left: FloatArray,
    covariance: FloatArray,
    right: FloatArray,
    *,
    may_underflow: bool,
    overflow: Literal["ignore", "raise"] = "ignore",
) -> float:
    """Recover the active form before applying the caller's overflow policy."""
    failure: FloatingPointError | None = None
    with np.errstate(over=overflow, invalid="ignore", under="ignore"):
        try:
            value = float(left @ covariance @ right)
        except FloatingPointError as error:
            failure = error
            value = math.nan
    if (math.isfinite(value) and not may_underflow) or not np.isfinite(
        covariance
    ).all():
        if failure is not None:
            raise failure
        return value

    # BLAS can silently round away subnormal terms even in a finite contraction.
    # The finite weights and active entries are binary rationals. Preserve
    # their contributions and cancellation before rounding back to float.
    exact_right = [Fraction(float(x)) for x in right]
    exact = Fraction()
    for weight, row in zip(left, covariance, strict=True):
        exact += Fraction(float(weight)) * _exact_dot(row, exact_right)
    try:
        return float(exact)
    except OverflowError:
        if failure is not None:
            raise failure from None
        if overflow == "raise" and not math.isfinite(value):
            # Keep the existing unavailable effect covariance when BLAS did
            # not raise on a contraction that is genuinely out of range.
            return value
        return -math.inf if exact < 0 else math.inf


def _target_support(
    gradient: tuple[float, ...],
) -> tuple[IndexArray, FloatArray]:
    """Select just the nonzero reported coefficients a target touches."""
    weights = np.asarray(gradient, dtype=np.float64)
    active = np.flatnonzero(weights)
    return active, weights[active]


def _target_variance(
    context: ReportingContext,
    active: IndexArray,
    selected: FloatArray,
    *,
    overflow: Literal["ignore", "raise"] = "ignore",
) -> float:
    """Contract the covariance on the target's support, including cross terms."""
    covariance = np.asarray(context.covariance, dtype=np.float64).reshape(
        (len(context.names), len(context.names))
    )
    block = covariance[np.ix_(active, active)]
    return _bilinear(
        selected,
        block,
        selected,
        may_underflow=_has_tiny_operand(selected) or _has_tiny_operand(block),
        overflow=overflow,
    )


def _target_influence_rows(rows: FloatArray, gradient: FloatArray) -> FloatArray:
    """Apply the target gradient without losing finite rows to product overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        values = rows @ gradient
    failed = np.flatnonzero(~np.isfinite(values))
    if not failed.size:
        return values

    exact_gradient = [Fraction(float(weight)) for weight in gradient]
    unavailable = False
    for index in failed:
        row = rows[index]
        if not np.isfinite(row).all():
            unavailable = True
            continue
        try:
            values[index] = float(_exact_dot(row, exact_gradient))
        except OverflowError:
            unavailable = True
    if unavailable:
        # Replay the native diagnostics only when genuine nonfinite rows remain.
        # Their values and the caller's NumPy warning/error policy are unchanged.
        _ = rows @ gradient
    return values


def linear_moments(
    context: ReportingContext, target: LinearTarget
) -> tuple[float, float]:
    """D'beta and D'VD on the active support, with every cross-covariance.

    Exactly zero weights exclude unrelated undefined entries, rather than forming
    zero times NaN. Negative, zero and undefined target variances are not repaired.
    """
    if len(target.weights) != len(context.names):
        raise ValueError("a target must use the full reported coefficient basis")
    active, selected = _target_support(target.weights)
    beta = np.asarray(context.beta, dtype=np.float64)[active]
    with np.errstate(over="raise", invalid="ignore"):
        estimate = float(selected @ beta)
    if not math.isfinite(estimate):
        raise FloatingPointError("the linear coefficient target is not finite")
    return estimate, _target_variance(context, active, selected, overflow="raise")


def evaluate(
    context: ReportingContext,
    target: LinearTarget,
    *,
    level: float = 0.95,
    ci: IntervalKind = "transform",
) -> TargetEstimate:
    """Pointwise inference from the reported beta and V, without additional scaling.

    The statistic/p-value test the linear null D'beta=0. Default finite-change
    intervals transform its actual reference endpoints; delta intervals use the
    same critical value. Local targets keep linear intervals under either choice.
    """
    if ci not in ("transform", "delta"):
        raise ValueError("ci must be 'transform' or 'delta'")
    estimate, variance = linear_moments(context, target)
    error = (
        math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else math.nan
    )
    usable = math.isfinite(error) and error > 0.0
    statistic = estimate / error if usable else math.nan
    quantile, p_value = reference_values(context.reference, level, statistic)
    half = quantile * error if usable else math.nan
    status = "ok" if usable else ("degenerate" if error == 0.0 else "missing")
    linear = Inference(
        estimate, error, statistic, p_value, estimate - half, estimate + half, status
    )
    if target.kind != "finite_change":
        return TargetEstimate(linear, linear, target.weights, variance, target.kind)
    reported, multiplier = _finite_change(linear, quantile, ci)
    gradient = tuple(multiplier * value for value in target.weights)
    if not all(math.isfinite(value) for value in gradient):
        raise FloatingPointError("the transformed target gradient is out of range")
    return TargetEstimate(
        linear,
        reported,
        gradient,
        variance,
        target.kind,
    )


def _finite_change(
    linear: Inference, quantile: float, ci: IntervalKind
) -> tuple[Inference, float]:
    """expm1(l), exp(l)se(l), and transformed or delta endpoints (Section 9.5)."""
    # Fail on unrepresentable transformations; never manufacture a zero delta SE
    # from an underflowed exp(l) or silently report successful inference at infinity.
    with np.errstate(over="raise", invalid="ignore"):
        multiplier = float(np.exp(linear.estimate))
        estimate = float(np.expm1(linear.estimate))
        error = multiplier * linear.std_error
        if multiplier == 0.0 or (
            linear.status == "ok" and (error == 0.0 or not math.isfinite(error))
        ):
            raise FloatingPointError(
                "the transformed coefficient target is out of range"
            )
        if ci == "transform":
            low = float(np.expm1(linear.conf_low))
            high = float(np.expm1(linear.conf_high))
        else:
            half = quantile * error if linear.status == "ok" else math.nan
            low, high = estimate - half, estimate + half
    return (
        Inference(
            estimate, error, linear.statistic, linear.p_value, low, high, linear.status
        ),
        multiplier,
    )
