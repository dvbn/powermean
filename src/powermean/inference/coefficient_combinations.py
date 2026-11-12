"""On-demand coefficient inference.

The context owns reported beta, the already scaled full V and the active normal/t
reference. These functions never recover rows, fit, correct beta or repair V.
Joint tests follow the rank-aware policy of
:mod:`powermean.inference.restriction_rank`: redundant restrictions count once
with the documented machine-precision rank rule shared by the generalized inverse
and the degrees of freedom; a discrepancy with a component outside the covariance
range has no p-value; empty or inconsistent requests refuse. An exact all-zero
covariance has its own status; negative or undefined covariance is missing.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np

from powermean.data.request_values import invalid
from powermean.effects.targets import (
    Inference,
    ReportingContext,
    reference_values,
)
from powermean.inference.restriction_rank import (
    consistent,
    formation_scale,
    negative_variance,
    numerical_rank,
    outside_range,
    quadratic_form,
    restriction_null,
    spectrum,
)
from powermean.inference.ssc import ReferenceDistribution
from powermean.model.variance import FloatArray

JointDistribution = Literal["chi2", "F"]

EXACT_ZERO = "exact zero restriction covariance"
INDEFINITE = "indefinite restriction covariance; no eigenvalue repair"
UNDEFINED = "undefined discrepancy or restriction covariance"
OUTSIDE_RANGE = (
    "the discrepancy has a component outside the covariance range; "
    "its variance is zero, so the joint statistic is undefined"
)


@dataclass(frozen=True, slots=True)
class ScalarEstimate:
    """Detached scalar inference and the full reported-coordinate derivative."""

    inference: Inference
    variance: float
    gradient: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RestrictionEstimate:
    """Full restriction discrepancy/covariance and its chi-square/F inference.

    ``requested`` rows were supplied, ``independent`` of them are numerically
    independent, and ``df_num`` is the effective count: the rank of ``R V R'``
    at ``cutoff``. ``unavailable_reason`` says why no statistic was formed.
    """

    difference: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    quadratic: float
    statistic: float
    p_value: float
    distribution: JointDistribution
    df_num: int
    df_denom: int | None
    status: str
    requested: int
    independent: int
    cutoff: float
    unavailable_reason: str | None

    @property
    def reduction(self) -> int:
        """Requested restrictions not counted: ``requested - df_num``."""
        return self.requested - self.df_num


@dataclass(frozen=True, slots=True)
class _Quadratic:
    """The Wald quadratic with its effective rank, cutoff, status and reason."""

    value: float
    rank: int
    cutoff: float
    status: str
    reason: str | None


class _JointTail(Protocol):
    """The SciPy upper-tail operation used for a joint reference."""

    def sf(self, value: float, *degrees: int) -> float:
        """Upper-tail probability at the supplied reference degrees of freedom."""
        ...


def scalar_estimate(
    context: ReportingContext,
    value: float,
    gradient: tuple[float, ...],
    *,
    level: float,
) -> ScalarEstimate:
    """Delta inference with the full gradient'V gradient and ordinary endpoints."""
    if not math.isfinite(value) or not all(math.isfinite(x) for x in gradient):
        raise FloatingPointError("the coefficient combination is not finite")
    variance = _variance(context, gradient)
    error = (
        math.sqrt(variance) if math.isfinite(variance) and variance >= 0 else math.nan
    )
    usable = math.isfinite(error) and error > 0.0
    statistic = value / error if usable else math.nan
    critical, p_value = reference_values(context.reference, level, statistic)
    half = critical * error if usable else math.nan
    status = "ok" if usable else ("degenerate" if error == 0.0 else "missing")
    return ScalarEstimate(
        Inference(value, error, statistic, p_value, value - half, value + half, status),
        variance,
        gradient,
    )


def _variance(context: ReportingContext, gradient: tuple[float, ...]) -> float:
    if len(gradient) != len(context.names):
        raise ValueError("a gradient must use the full reported coefficient basis")
    weights = np.asarray(gradient, dtype=np.float64)
    active = np.flatnonzero(weights)
    selected = weights[active]
    covariance = np.asarray(context.covariance, dtype=np.float64).reshape(
        (len(context.names), len(context.names))
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return float(selected @ covariance[np.ix_(active, active)] @ selected)


def restrictions(
    context: ReportingContext, r: object, q: object
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Validate an explicit real R and q: shape, finiteness, nonempty, consistent.

    An empty matrix, or one whose rows are all numerically zero, restricts
    nothing and refuses; a right-hand side that the dependent rows of R cannot
    reach together refuses. Both use the documented rank rule of
    :mod:`powermean.inference.restriction_rank`. Rows are never changed.
    """
    matrix = _real_array(r, "R")
    rhs = _real_array(q, "q")
    if matrix.ndim != 2 or matrix.shape[1] != len(context.names):
        raise invalid(
            "R", r, "a finite real m by k matrix in reported coefficient order"
        )
    if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
        raise invalid("q", q, "a finite real vector with one entry per restriction")
    if matrix.shape[0] == 0 or numerical_rank(matrix) == 0:
        raise invalid("R", r, "at least one restriction row that is not zero")
    if not consistent(matrix, rhs):
        raise invalid(
            "q", q, "a right-hand side consistent with the dependent rows of R"
        )
    return tuple(tuple(float(x) for x in row) for row in matrix), tuple(
        float(x) for x in rhs
    )


def _real_array(value: object, argument: str) -> FloatArray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise invalid(argument, value, "finite real numeric arrays")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise invalid(argument, value, "finite real numeric arrays") from error
    if not np.isfinite(result).all():
        raise invalid(argument, value, "finite real numeric arrays")
    return result


def joint_estimate(
    context: ReportingContext,
    rows: tuple[tuple[float, ...], ...],
    rhs: tuple[float, ...],
) -> RestrictionEstimate:
    """``Q = d' S^+ d`` over the retained directions of ``S = R V R'``.

    Each pair contracts only its own active supports, preserving unrelated NaNs.
    The retained directions, their count ``r`` and the generalized inverse share
    one machine-precision cutoff. Normal gives chi-square(r); t gives
    ``F(r, df) = Q / r`` with the fit's denominator degrees of freedom.
    """
    weights = np.asarray(rows, dtype=np.float64)
    difference, covariance = _restriction_moments(context, rows, rhs)
    scale = formation_scale(
        weights, np.asarray(context.beta, dtype=np.float64), np.asarray(rhs)
    )
    distribution: JointDistribution = (
        "chi2" if context.reference.kind == "normal" else "F"
    )
    row_null = restriction_null(weights)
    found = _quadratic(difference, covariance, scale, row_null, weights)
    usable = found.status == "ok"
    statistic = (
        (found.value / found.rank if distribution == "F" else found.value)
        if usable
        else math.nan
    )
    p_value = _joint_p(context.reference, statistic, found.rank) if usable else math.nan
    return RestrictionEstimate(
        tuple(float(x) for x in difference),
        tuple(tuple(float(x) for x in row) for row in covariance),
        found.value,
        statistic,
        p_value,
        distribution,
        found.rank,
        context.reference.df if distribution == "F" else None,
        found.status,
        len(rows),
        len(rows) - row_null.shape[1],
        found.cutoff,
        found.reason,
    )


def _restriction_moments(
    context: ReportingContext,
    rows: tuple[tuple[float, ...], ...],
    rhs: tuple[float, ...],
) -> tuple[FloatArray, FloatArray]:
    weights = np.asarray(rows, dtype=np.float64)
    beta = np.asarray(context.beta, dtype=np.float64)
    matrix = np.asarray(context.covariance, dtype=np.float64)
    active = [np.flatnonzero(row) for row in weights]
    difference = np.empty(len(rows), dtype=np.float64)
    covariance = np.empty((len(rows), len(rows)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for j, support in enumerate(active):
            difference[j] = weights[j, support] @ beta[support] - rhs[j]
            for k in range(j, len(rows)):
                other = active[k]
                value = (
                    weights[j, support]
                    @ matrix[np.ix_(support, other)]
                    @ weights[k, other]
                )
                covariance[j, k] = covariance[k, j] = value
    return difference, covariance


def _quadratic(
    difference: FloatArray,
    covariance: FloatArray,
    scale: float,
    row_null: FloatArray,
    rows: FloatArray,
) -> _Quadratic:
    """Status order: undefined, exact zero, indefinite, outside range, ok."""
    if not (np.isfinite(difference).all() and np.isfinite(covariance).all()):
        return _Quadratic(math.nan, 0, math.nan, "missing", UNDEFINED)
    if not np.any(covariance):
        return _Quadratic(math.nan, 0, 0.0, "degenerate", EXACT_ZERO)
    found = spectrum(covariance)
    if (
        found.rank == 0
        or not math.isfinite(found.cutoff)
        or not np.isfinite(found.values).all()
    ):
        return _Quadratic(math.nan, found.rank, found.cutoff, "missing", UNDEFINED)
    if found.indefinite or negative_variance(covariance):
        return _Quadratic(math.nan, found.rank, found.cutoff, "missing", INDEFINITE)
    if found.discarded.shape[1] and not math.isfinite(scale):
        return _Quadratic(math.nan, found.rank, found.cutoff, "missing", UNDEFINED)
    if outside_range(found, difference, scale, covariance, row_null, rows=rows):
        return _Quadratic(
            math.nan, found.rank, found.cutoff, "degenerate", OUTSIDE_RANGE
        )
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic = quadratic_form(found, difference)
    if not math.isfinite(quadratic):
        return _Quadratic(math.nan, found.rank, found.cutoff, "missing", UNDEFINED)
    return _Quadratic(quadratic, found.rank, found.cutoff, "ok", None)


def _joint_p(reference: ReferenceDistribution, statistic: float, count: int) -> float:
    if count < 1:
        raise ValueError("a joint reference needs at least one restriction")
    if reference.kind == "normal":
        distribution = cast("_JointTail", importlib.import_module("scipy.stats").chi2)
        return float(distribution.sf(statistic, count))
    if reference.kind != "t" or reference.df is None or reference.df <= 0:
        raise ValueError("a Student-t reference needs positive degrees of freedom")
    distribution = cast("_JointTail", importlib.import_module("scipy.stats").f)
    return float(distribution.sf(statistic, count, reference.df))
