"""Detached coefficient combinations and restriction reports.

Every call consumes one accepted immutable reporting context. Output records own
only coefficient-sized tuples; arrays, metadata and tables are fresh copies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from powermean.effects.targets import ReportingContext
from powermean.inference.coefficient_combinations import (
    RestrictionEstimate,
    ScalarEstimate,
    joint_estimate,
    restrictions,
    scalar_estimate,
)
from powermean.inference.coefficient_expressions import affine, equality, nonlinear
from powermean.model.variance import FloatArray
from powermean.results.effect_reporting import context_metadata

RANK_RULE = (
    "numpy.linalg.matrix_rank default: largest singular value times the largest "
    "matrix dimension times float64 eps, shared by the generalized inverse and df"
)


def _metadata(context: ReportingContext, kind: str, term: str) -> dict[str, object]:
    metadata = context_metadata(context)
    metadata.pop("population")
    metadata.update(target_kind=kind, term=term, units="coefficient expression")
    return metadata


@dataclass(frozen=True, slots=True)
class CombinationResult:
    """One scalar coefficient function with ordinary delta-method inference."""

    context: ReportingContext = field(repr=False)
    numeric: ScalarEstimate
    term: str
    kind: Literal["lincom", "nlcom"]
    level: float

    @property
    def estimate(self) -> float:
        """Value at the actual reported coefficient point."""
        return self.numeric.inference.estimate

    @property
    def std_error(self) -> float:
        """Square root of the full delta covariance, without any second factor."""
        return self.numeric.inference.std_error

    @property
    def statistic(self) -> float:
        """Normal/t statistic for the scalar function equal to zero."""
        return self.numeric.inference.statistic

    @property
    def p_value(self) -> float:
        """Two-sided p-value under the active reference."""
        return self.numeric.inference.p_value

    @property
    def interval(self) -> tuple[float, float]:
        """Ordinary pointwise interval on the scalar function's own scale."""
        return self.numeric.inference.conf_low, self.numeric.inference.conf_high

    @property
    def gradient(self) -> FloatArray:
        """Detached analytic gradient in full reported coefficient order."""
        return np.array(self.numeric.gradient, dtype=np.float64)

    @property
    def metadata(self) -> dict[str, object]:
        """Fresh active covariance, point, reference and interval metadata."""
        metadata = _metadata(self.context, self.kind, self.term)
        metadata.update(
            level=self.level,
            ci="linear" if self.kind == "lincom" else "delta",
            selection=list(self.context.names),
        )
        return metadata

    def to_dict(self) -> dict[str, object]:
        """Fresh scalar row, following the accepted coefficient field discipline."""
        row = self.numeric.inference
        return {
            "term": self.term,
            "estimate": row.estimate,
            "std_error": row.std_error,
            "statistic": row.statistic,
            "p_value": row.p_value,
            "conf_low": row.conf_low,
            "conf_high": row.conf_high,
            "inference": row.status,
            "reference": self.context.reference.kind,
            "reference_df": self.context.reference.df,
            "units": "coefficient expression",
            "unavailable_reason": None,
        }

    def to_frame(self) -> pd.DataFrame:
        """Fresh one-row table with detached metadata."""
        table = pd.DataFrame([self.to_dict()])
        table.attrs.update(self.metadata)
        return table


@dataclass(frozen=True, slots=True)
class RestrictionResult:
    """A joint Wald restriction report, without a fabricated scalar interval.

    ``df_num`` is the effective restriction count: the machine-precision rank
    of ``R V R'`` shared with the generalized inverse. The requested count, the
    number of independent rows and the reduction are reported beside it.
    """

    context: ReportingContext = field(repr=False)
    numeric: RestrictionEstimate
    rows: tuple[tuple[float, ...], ...]
    rhs: tuple[float, ...]
    term: str

    @property
    def statistic(self) -> float:
        """Q under normal, Q/r under the active t convention, r the effective rank."""
        return self.numeric.statistic

    @property
    def p_value(self) -> float:
        """Upper tail of chi-square(r) or F(r, df); NaN when no statistic exists."""
        return self.numeric.p_value

    @property
    def difference(self) -> FloatArray:
        """Detached R beta - q at the reported coefficient point."""
        return np.array(self.numeric.difference, dtype=np.float64)

    @property
    def covariance(self) -> FloatArray:
        """Detached full R V R', without a PSD adjustment."""
        return np.array(self.numeric.covariance, dtype=np.float64)

    @property
    def metadata(self) -> dict[str, object]:
        """Fresh restriction, covariance, reference and point metadata."""
        metadata = _metadata(self.context, "wald", self.term)
        metadata.update(
            R=[list(row) for row in self.rows],
            q=list(self.rhs),
            distribution=self.numeric.distribution,
            df_num=self.numeric.df_num,
            df_denom=self.numeric.df_denom,
            requested=self.numeric.requested,
            independent=self.numeric.independent,
            reduction=self.numeric.reduction,
            rank_cutoff=self.numeric.cutoff,
            rank_rule=RANK_RULE,
            unavailable_reason=self.numeric.unavailable_reason,
        )
        return metadata

    def to_dict(self) -> dict[str, object]:
        """Fresh scalar test row with explicit requested and effective counts."""
        return {
            "term": self.term,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "wald_quadratic": self.numeric.quadratic,
            "distribution": self.numeric.distribution,
            "df_num": self.numeric.df_num,
            "df_denom": self.numeric.df_denom,
            "requested": self.numeric.requested,
            "reduction": self.numeric.reduction,
            "inference": self.numeric.status,
            "reference": self.context.reference.kind,
            "reference_df": self.context.reference.df,
            "unavailable_reason": self.numeric.unavailable_reason,
        }

    def to_frame(self) -> pd.DataFrame:
        """Fresh one-row table with detached metadata."""
        table = pd.DataFrame([self.to_dict()])
        table.attrs.update(self.metadata)
        return table


def linear_result(
    context: ReportingContext, expression: str, *, level: float
) -> CombinationResult:
    """Affine combination with full c'Vc and ordinary normal/t inference."""
    target = affine(expression, context.names)
    value = math.fsum(
        weight * beta for weight, beta in zip(target.weights, context.beta, strict=True)
    )
    numeric = scalar_estimate(
        context, value + target.constant, target.weights, level=level
    )
    return CombinationResult(context, numeric, expression, "lincom", level)


def nonlinear_result(
    context: ReportingContext, expression: str, *, level: float
) -> CombinationResult:
    """Smooth scalar value and analytic full gradient at reported beta."""
    target = nonlinear(expression, context.names, context.beta)
    numeric = scalar_estimate(context, target.value, target.gradient, level=level)
    return CombinationResult(context, numeric, expression, "nlcom", level)


def restriction_result(
    context: ReportingContext, r: object, q: object
) -> RestrictionResult:
    """Explicit joint restrictions on the active reported coefficient basis.

    Redundant rows count once; an untestable discrepancy component yields a
    degenerate report without a p-value; empty or inconsistent requests refuse.
    """
    rows, rhs = restrictions(context, r, q)
    return RestrictionResult(
        context, joint_estimate(context, rows, rhs), rows, rhs, "wald"
    )


def equality_result(context: ReportingContext, expression: str) -> RestrictionResult:
    """One affine equality, including its nonzero constant when supplied."""
    target = equality(expression, context.names)
    rows, rhs = restrictions(context, (target.weights,), (-target.constant,))
    return RestrictionResult(
        context, joint_estimate(context, rows, rhs), rows, rhs, expression
    )
