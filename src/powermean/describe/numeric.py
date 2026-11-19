"""The arithmetic of the preliminary tables, on arrays.

Functions of arrays that return arrays, scalars or small records; nothing here reads a
frame.

Notation: ``x`` a variable, ``w`` nonnegative weights (``None`` for none), ``codes``
integer group codes in ``0..groups-1``, ``design`` an ``n`` by ``k`` matrix.

Moments.  The mean is ``sum(w x) / sum(w)``.  The variance without weights divides
by ``n - 1``; with weights it is ``n/(n-1) sum(w (x - mean)^2) / sum(w)``, the
analytic-weight convention (weights rescaled to sum to ``n``), which equals the
unweighted variance at equal weights.

Quantiles.  Without weights, linear interpolation between order statistics, the
default of R's ``quantile`` and of NumPy.  With weights, the inverse of the weighted
empirical distribution function: ``Q(p) = min {x : F(x) >= p}`` with
``F(x) = sum(w 1[x_i <= x]) / sum(w)``, and ``Q(0)`` the smallest value.

Two-sample comparison.  The difference of two means with the variance
``s_a^2/n_a + s_0^2/n_0``, which is the HC2 variance of the regression on arm
indicators, and the Welch degrees of freedom
``(v_a + v_0)^2 / (v_a^2/(n_a-1) + v_0^2/(n_0-1))`` with ``v_d = s_d^2/n_d``.

Regression.  Weighted least squares ``beta = (X'WX)^{-1} X'W y``.  The sandwich is
``B M B`` with ``B = (X'WX)^{-1}`` and ``M = sum_c g_c g_c'``, where ``g_c`` is the
sum over cluster ``c`` (or the row itself) of ``w_i e_i x_i``.  The convention then
supplies one scalar factor and the reference distribution (:func:`convention`).  The
classical covariance of the balance regression is ``s^2 B`` with
``s^2 = sum(w e^2)/(n - K)``.

Strata.  Absorbing one classification replaces each column by its deviation from the
weighted mean of its stratum; by the Frisch-Waugh-Lovell theorem the coefficients
and residuals are those of the regression with one indicator per stratum.
"""

from __future__ import annotations

import importlib
import math
import statistics
import sys
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from typing import Literal, NamedTuple, Protocol, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
Preset = Literal["stata", "none", "regression"]
ReferenceKind = Literal["normal", "t"]


class _Tail(Protocol):
    """The two methods of a SciPy distribution that the tables use."""

    def sf(self, x: float, *shape: float) -> float:
        """The survival function."""
        ...

    def ppf(self, q: float, *shape: float) -> float:
        """The quantile function."""
        ...


def _distribution(name: str) -> _Tail:
    """A distribution of ``scipy.stats``, loaded when a table first needs it."""
    return cast("_Tail", getattr(importlib.import_module("scipy.stats"), name))


# ---------------------------------------------------------------------- moments


def mean(x: FloatArray, w: FloatArray | None) -> float:
    """The mean ``sum(w x) / sum(w)`` of ``x``; NaN on no value."""
    if x.size == 0:
        return math.nan
    if w is None:
        return float(x.mean())
    total = float(w.sum())
    try:
        with np.errstate(under="raise"):
            shares = w / total
    except FloatingPointError:
        # Combine the exponents of w, x and total before rounding. A
        # vanishing weight share can multiply a very large observation.
        weight_frac, weight_exp = np.frexp(w)
        value_frac, value_exp = np.frexp(x)
        total_frac, total_exp = math.frexp(total)
        terms = np.ldexp(
            weight_frac * value_frac / total_frac, weight_exp + value_exp - total_exp
        )
        return float(terms.sum())
    return float((shares * x).sum())


def mean_variance(x: FloatArray, w: FloatArray | None) -> tuple[float, float]:
    """The mean and the variance of ``x`` (module docstring); NaN where undefined."""
    center = mean(x, w)
    n = x.size
    if n < 2:
        return center, math.nan
    if w is None:
        return center, float(((x - center) ** 2).sum() / (n - 1))
    return center, float(n / (n - 1) * (w * (x - center) ** 2).sum() / w.sum())


def quantiles(
    x: FloatArray, probs: tuple[float, ...], w: FloatArray | None
) -> FloatArray:
    """The quantiles of ``x`` at ``probs`` (module docstring); NaN on no value."""
    if x.size == 0:
        return np.full(len(probs), np.nan)
    if w is None:
        return np.asarray(np.quantile(x, probs), dtype=np.float64)
    order = np.argsort(x, kind="stable")
    values = x[order]
    cumulative = np.cumsum(w[order])
    targets = np.array(probs) * cumulative[-1]
    positions = np.searchsorted(cumulative, targets, side="left")
    return np.asarray(values[np.minimum(positions, values.size - 1)], dtype=np.float64)


def share_and_positive_mean(x: FloatArray, w: FloatArray | None) -> tuple[float, float]:
    """The share of zeros of ``x`` and the mean of its positive values."""
    weights = np.ones(x.size) if w is None else w
    total = float(weights.sum())
    if total == 0.0:
        return math.nan, math.nan
    positive = x > 0.0
    share = float(weights[x == 0.0].sum() / total)
    mass = float(weights[positive].sum())
    positive_mean = mean(x[positive], weights[positive]) if mass else math.nan
    return share, positive_mean


# ------------------------------------------------------------ reference and tests


class Reference(NamedTuple):
    """A reference distribution: the normal, or ``t`` with its degrees of freedom."""

    kind: ReferenceKind
    df: float


NORMAL = Reference("normal", math.inf)


def two_sided_p(statistic: float, reference: Reference) -> float:
    """``P(|T| >= |statistic|)`` under the reference; NaN where undefined."""
    if math.isnan(statistic) or (reference.kind == "t" and not reference.df > 0):
        return math.nan
    if reference.kind == "normal":
        return math.erfc(abs(statistic) / math.sqrt(2.0))
    return 2.0 * float(_distribution("t").sf(abs(statistic), reference.df))


def critical_value(level: float, reference: Reference) -> float:
    """``q`` with ``P(|T| <= q) = level`` under the reference."""
    if reference.kind == "normal":
        return statistics.NormalDist().inv_cdf((1.0 + level) / 2.0)
    if not reference.df > 0:
        return math.nan
    return float(_distribution("t").ppf((1.0 + level) / 2.0, reference.df))


class Welch(NamedTuple):
    """The two-sample comparison with unequal variances."""

    difference: float
    std_error: float
    df: float
    p_value: float


def welch(
    moments_a: tuple[float, float, int], moments_0: tuple[float, float, int]
) -> Welch:
    """Compare two arms, each given as ``(mean, variance, n)`` (module docstring)."""
    mean_a, variance_a, n_a = moments_a
    mean_0, variance_0, n_0 = moments_0
    difference = mean_a - mean_0
    if min(n_a, n_0) < 2:
        return Welch(difference, math.nan, math.nan, math.nan)
    v_a, v_0 = variance_a / n_a, variance_0 / n_0
    std_error = math.sqrt(v_a + v_0)
    if std_error == 0.0:
        return Welch(difference, 0.0, math.nan, math.nan)
    df = (v_a + v_0) ** 2 / (v_a**2 / (n_a - 1) + v_0**2 / (n_0 - 1))
    p = two_sided_p(difference / std_error, Reference("t", df))
    return Welch(difference, std_error, df, p)


# ------------------------------------------------------------------- regression


def absorb(columns: FloatArray, codes: IntArray, w: FloatArray | None) -> FloatArray:
    """The columns of an ``n`` by ``k`` matrix minus their weighted stratum means."""
    weights = np.ones(codes.size) if w is None else w
    groups = int(codes.max()) + 1 if codes.size else 0
    mass = np.bincount(codes, weights=weights, minlength=groups)
    centered = np.empty_like(columns)
    for j in range(columns.shape[1]):
        sums = np.bincount(codes, weights=weights * columns[:, j], minlength=groups)
        centered[:, j] = columns[:, j] - (sums / mass)[codes]
    return centered


class LeastSquares(NamedTuple):
    """A weighted least-squares solve: ``beta``, ``(X'WX)^{-1}``, residuals, rank."""

    beta: FloatArray
    bread: FloatArray
    residuals: FloatArray
    rank: int


def least_squares(
    y: FloatArray, design: FloatArray, w: FloatArray | None
) -> LeastSquares:
    """Solve the weighted least-squares problem; the caller checks the rank.

    The rank is that of the weighted design by NumPy's documented rule
    (:func:`numpy.linalg.matrix_rank`).  When it is deficient the bread is the
    pseudo-inverse and the caller must not report inference.
    """
    root = np.ones(y.size) if w is None else np.sqrt(w)
    weighted = design * root[:, None]
    gram = weighted.T @ weighted
    rank = int(np.linalg.matrix_rank(weighted)) if design.size else 0
    bread = np.linalg.pinv(gram, hermitian=True)
    beta = bread @ (weighted.T @ (root * y))
    return LeastSquares(beta, bread, y - design @ beta, rank)


def sandwich(
    design: FloatArray,
    fit: LeastSquares,
    w: FloatArray | None,
    clusters: IntArray | None,
) -> FloatArray:
    """``B M B`` without a factor: rows, or their sums within clusters, as scores."""
    weights = np.ones(fit.residuals.size) if w is None else w
    scores = design * (weights * fit.residuals)[:, None]
    if clusters is not None:
        groups = int(clusters.max()) + 1
        summed = np.zeros((groups, design.shape[1]))
        np.add.at(summed, clusters, scores)
        scores = summed
    return np.asarray(fit.bread @ (scores.T @ scores) @ fit.bread, dtype=np.float64)


def classical(fit: LeastSquares, w: FloatArray | None, n: int, k: int) -> FloatArray:
    """``s^2 (X'WX)^{-1}`` with ``s^2 = sum(w e^2) / (n - k)``; NaN when ``n <= k``."""
    weights = np.ones(fit.residuals.size) if w is None else w
    if n <= k:
        return np.full_like(fit.bread, np.nan)
    scale = float((weights * fit.residuals**2).sum() / (n - k))
    return np.asarray(scale * fit.bread, dtype=np.float64)


class Counts(NamedTuple):
    """The counts of a convention: rows, clusters, and the two parameter counts.

    ``k_factor`` enters the scalar factor and ``k_reference`` the degrees of freedom
    of the reference; they differ only for strata nested in the clusters.  ``clusters``
    is zero without clustering.
    """

    n: int
    clusters: int
    k_factor: int
    k_reference: int


def parameter_counts(slopes: int, strata: int, *, nested: bool) -> tuple[int, int]:
    """``(k_factor, k_reference)`` of a regression with ``slopes`` slopes.

    Without strata the intercept counts as one parameter.  With strata each level
    counts, unless the strata are nested in the clusters: they are then not counted
    in the reference and one constant enters the factor.
    """
    if strata == 0:
        return slopes + 1, slopes + 1
    if nested:
        return slopes + 1, slopes
    return slopes + strata, slopes + strata


def convention(ssc: Preset, counts: Counts) -> tuple[float, Reference]:
    """The scalar factor and the reference of a preset.

    ``"stata"``: ``N/(N-1)``, or ``G/(G-1)`` with clusters; normal.  ``"none"``: 1;
    normal.  ``"regression"``: ``N/(N-K)``, or ``(N-1)/(N-K) G/(G-1)`` with clusters;
    ``t`` with ``N-K`` degrees of freedom, or ``min(G-1, N-K)`` with clusters.  A
    factor whose denominator is not positive is NaN.
    """
    n, g = counts.n, counts.clusters

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else math.nan

    if ssc == "none":
        return 1.0, NORMAL
    if ssc == "stata":
        return (ratio(g, g - 1) if g else ratio(n, n - 1)), NORMAL
    residual = n - counts.k_reference
    if g:
        factor = ratio(n - 1, n - counts.k_factor) * ratio(g, g - 1)
        return factor, Reference("t", float(min(g - 1, residual)))
    return ratio(n, n - counts.k_factor), Reference("t", float(residual))


class JointTest(NamedTuple):
    """A Wald test that several coefficients are zero.

    ``distribution`` is ``"chi2"`` (``statistic`` is the Wald quadratic form, with
    ``df_num`` degrees of freedom) or ``"F"`` (``statistic`` is the quadratic form
    divided by ``df_num``, with ``df_denom`` denominator degrees of freedom).
    """

    statistic: float
    df_num: int
    df_denom: float
    p_value: float
    distribution: Literal["chi2", "F"]


def joint_test(
    beta: FloatArray, covariance: FloatArray, reference: Reference
) -> JointTest:
    """Test ``beta = 0`` with its covariance: chi-square, or ``F`` under ``t``."""
    count = int(beta.size)
    kind: Literal["chi2", "F"] = "chi2" if reference.kind == "normal" else "F"
    df_denom = math.inf if kind == "chi2" else reference.df
    usable = count > 0 and bool(np.isfinite(covariance).all()) and df_denom > 0
    if not usable or int(np.linalg.matrix_rank(covariance)) < count:
        return JointTest(math.nan, count, df_denom, math.nan, kind)
    quadratic = float(beta @ np.linalg.solve(covariance, beta))
    if kind == "chi2":
        p = float(_distribution("chi2").sf(quadratic, count))
        return JointTest(quadratic, count, df_denom, p, kind)
    statistic = quadratic / count
    p = float(_distribution("f").sf(statistic, count, df_denom))
    return JointTest(statistic, count, df_denom, p, kind)


# ------------------------------------------------------------------ power means


class LogMoment(NamedTuple):
    """``log M`` of one arm and the influence of each of its rows on it."""

    log_moment: float
    influence: FloatArray
    center: float = math.nan
    log_relative: float = math.nan


def _exact_relative_power(ratio: Fraction, omega: Fraction) -> Fraction | None:
    """Recover exact rational roots of a binary64 ratio for a binary64 order."""
    if ratio in (0, 1):
        # The caller admits a zero outcome only at positive orders.
        return ratio
    numerator, denominator = ratio.as_integer_ratio()
    # A reduced binary64 order is p / 2**k. Its root is rational precisely
    # when both coprime parts are successive perfect squares k times.
    for _ in range(omega.denominator.bit_length() - 1):
        a, b = math.isqrt(numerator), math.isqrt(denominator)
        if a * a != numerator or b * b != denominator:
            return None
        numerator, denominator = a, b

    # Guard the size *before* exponentiation. With size >= 2, this bounds
    # each powered integer by 16385 bits and preserves exact binary shifts
    # through 8192. Larger powers retain the bounded Decimal calculation.
    size = max(numerator, denominator)
    if (size.bit_length() - 1) * abs(omega.numerator) > 8192:
        return None
    return Fraction(numerator, denominator) ** omega.numerator


def _relative_powers(y: FloatArray, omega: float) -> list[Fraction]:
    """Powers relative to the largest power, without binary64 range loss."""
    support, inverse = np.unique(y, return_inverse=True)
    base = float(support[-1] if omega > 0 else support[support > 0][0])
    if omega.is_integer() and abs(omega) <= 16:
        degree = int(omega)
        scale_power = Fraction(base) ** degree
        powers = [Fraction(float(v)) ** degree / scale_power for v in support]
    else:
        # Allocate enough digits for a binary64 subnormal correction even
        # after division by a tiny order. This bound is 350 to 674 digits.
        order = Decimal.from_float(omega)
        ctx = Context(prec=350 + max(0, -order.adjusted()), Emin=-2048, Emax=2048)
        powers = []
        binary_order = Fraction(omega)
        binary_base = Fraction(base)
        with localcontext(ctx):
            scale = Decimal.from_float(base)
            for value in support:
                exact = _exact_relative_power(
                    Fraction(float(value)) / binary_base, binary_order
                )
                if exact is None:
                    ratio = Decimal.from_float(float(value)) / scale
                    exact = Fraction(ratio**order)
                powers.append(exact)
    return [powers[i] for i in inverse]


def _wide_range_moment(
    y: FloatArray, logs: FloatArray, weights: FloatArray, omega: float
) -> LogMoment:
    """Center a wide power comparison on its largest weighted contribution."""
    powers = _relative_powers(y, omega)
    masses = [Fraction(float(w)) for w in weights]
    terms = [w * r for w, r in zip(masses, powers, strict=True)]
    total = sum(masses, Fraction())
    moment = sum(terms, Fraction()) / total
    anchor = max(range(len(terms)), key=terms.__getitem__)
    center = float(logs[anchor])
    # Each weighted term is at most the anchor term, so relative <= n.
    relative = moment / powers[anchor]
    gap = relative - 1
    if relative >= Fraction(1, 2):
        log_relative = math.log1p(float(gap))
    else:
        # Scale even a subnormal relative mean into [0.5, 2] for its logarithm.
        exponent = relative.denominator.bit_length() - relative.numerator.bit_length()
        log_relative = math.log(
            float(relative * (1 << exponent))
        ) - exponent * math.log(2)

    # Compute q_i (r_i - mean(r)) / (omega * mean(r)) before rounding.
    # In particular the largest weighted row can have a small nonzero score.
    denominator = total * moment * Fraction(omega)
    influence = np.empty_like(y)
    for i, (w, r) in enumerate(zip(masses, powers, strict=True)):
        score = w * (r - moment) / denominator
        influence[i] = float(score)
        if influence[i] == 0 and score:
            raise FloatingPointError(
                "nonzero row influence outside floating-point range"
            )
    if abs(log_relative) < sys.float_info.min:
        value = center + float(gap / Fraction(omega))
        return LogMoment(value, influence, value, 0.0)
    return LogMoment(center + log_relative / omega, influence, center, log_relative)


def _centered_range_moment(
    logs: FloatArray, weights: FloatArray, omega: float
) -> LogMoment:
    """Recover a geometric or near-zero-order moment after intermediate range loss."""
    center = float(logs[weights.argmax()])
    gains = logs - center
    if not gains.any():
        return LogMoment(center, np.zeros_like(logs), center, 0.0)

    # g_i = expm1(omega * d_i) / omega, or d_i at order zero.
    # When the argument is below epsilon, g_i rounds to d_i to working precision.
    with np.errstate(under="ignore"):
        arguments = omega * gains
    nonlinear = np.abs(arguments) > sys.float_info.epsilon
    gains[nonlinear] *= np.expm1(arguments[nonlinear]) / arguments[nonlinear]

    # Only on this range fallback, keep the binary weights and gains as exact
    # fractions through centering and division so even subnormal scores round once.
    masses = [Fraction(float(w)) for w in weights]
    values = [Fraction(float(g)) for g in gains]
    total = sum(masses, Fraction())
    weighted_sum = sum((w * g for w, g in zip(masses, values, strict=True)), Fraction())
    average = weighted_sum / total
    correction = Fraction(omega) * average
    denominator = total * (1 + correction)
    influence = np.empty_like(logs)
    for i, (w, g) in enumerate(zip(masses, values, strict=True)):
        score = w * (g - average) / denominator
        influence[i] = float(score)
        if influence[i] == 0.0 and score:
            raise FloatingPointError(
                "nonzero row influence outside floating-point range"
            )

    # log1p(omega * average) / omega tends to average. When their product
    # is subnormal it can lose the correction before division by the order.
    scaled = float(correction)
    if abs(scaled) < sys.float_info.min:
        value = center + float(average)
        return LogMoment(value, influence, value, 0.0)
    log_relative = math.log1p(scaled)
    return LogMoment(center + log_relative / omega, influence, center, log_relative)


def log_power_moment(y: FloatArray, omega: float, w: FloatArray | None) -> LogMoment:
    """``log M_omega`` of an arm, scaled by ``1/omega``, and its influence.

    For ``omega != 0`` the value is ``(1/omega) log(sum(w y^omega)/sum(w))`` and row
    ``i`` contributes ``w_i (y_i^omega - M) / (omega sum(w) M)``.  For ``omega = 0``
    the value is the weighted mean of ``log y`` and row ``i`` contributes
    ``w_i (log y_i - mean) / sum(w)``.  Powers are formed relative to the largest one,
    on the log scale, so that no power overflows; a zero outcome, which the caller
    admits only at a positive order, has power zero.  An arm without a positive outcome
    has moment zero, whose logarithm is undefined: the value is NaN.
    """
    weights = np.ones(y.size) if w is None else w
    total = float(weights.sum())
    positive = y > 0.0
    if total == 0.0 or not positive.any():
        return LogMoment(math.nan, np.full(y.size, np.nan))
    logs = np.zeros(y.size)
    logs[positive] = np.log(y[positive])
    if omega == 0.0:
        try:
            center = mean(logs, weights)
            return LogMoment(center, (weights / total) * (logs - center), center, 0.0)
        except FloatingPointError:
            return _centered_range_moment(logs, weights, omega)
    center = float(logs[positive].max() if omega > 0 else logs[positive].min())
    try:
        delta = np.where(positive, omega * (logs - center), -np.inf)
    except FloatingPointError:
        return (
            _centered_range_moment(logs, weights, omega)
            if positive.all() and abs(omega) * float(np.ptp(logs)) < 0.5
            else _wide_range_moment(y, logs, weights, omega)
        )
    # expm1/log1p retain the near-zero-order limit on positive support.
    if positive.all() and float(np.max(np.abs(delta))) < 0.5:
        try:
            q = weights / total
            log_relative = math.log1p(float((q * np.expm1(delta)).sum()))
            influence = q * np.expm1(delta - log_relative) / omega
        except FloatingPointError:
            return _centered_range_moment(logs, weights, omega)
    else:
        try:
            q = weights / total
            log_relative = math.log(float((q * np.exp(delta)).sum()))
            change = delta - log_relative
            influence = q * np.expm1(change) / omega
        except FloatingPointError:
            return _wide_range_moment(y, logs, weights, omega)
        # A row near the weighted mean loses its small score in the
        # subtraction of two rounded logs. The same cancellation can
        # erase a contrast against a constant arm at that row's outcome.
        # Limit the extra check to an extreme power normalization or a weight
        # share much smaller than the equal-weight share. An ordinary exact
        # mean row alone does not call for arbitrary-precision arithmetic.
        threshold = math.sqrt(sys.float_info.epsilon)
        if log_relative < math.log(threshold) or (
            w is not None and float(q.min()) < threshold / q.size
        ):
            uncertainty = threshold * max(1.0, abs(log_relative))
            if float(np.min(np.abs(change))) <= uncertainty:
                return _wide_range_moment(y, logs, weights, omega)
    value = center + log_relative / omega
    return LogMoment(value, influence, center, log_relative)
