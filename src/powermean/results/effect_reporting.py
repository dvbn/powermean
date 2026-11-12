"""Separate coefficient-target records/tables.

This is a result-layer presentation adapter over the small numeric effects engine.
It neither changes coefficient tables nor implements the later shared export API.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from powermean.data.request_values import invalid
from powermean.effects.targets import (
    IntervalKind,
    LinearTarget,
    ReportingContext,
    ReportingPoint,
    TargetEstimate,
    evaluate,
    reference_values,
)
from powermean.errors import SpecificationError
from powermean.formula.frame import Design
from powermean.inference.ssc import ReferenceDistribution
from powermean.model.variance import FloatArray
from powermean.results.effect_design import coefficient_target, simple_columns
from powermean.results.naming import mean_label, order_text

InfluenceProvider = Callable[[], pd.DataFrame]


def context_snapshot(
    names: tuple[str, ...],
    beta: FloatArray,
    covariance: FloatArray,
    reference: ReferenceDistribution,
    omega: float,
    *,
    point: ReportingPoint,
    metadata: Mapping[str, object],
) -> ReportingContext:
    """Bind already reported numeric coordinates and small metadata exactly once."""
    return ReportingContext(
        names,
        tuple(float(value) for value in beta),
        tuple(tuple(float(value) for value in row) for row in covariance),
        reference,
        omega,
        point,
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")),
    )


def target_label(omega: float) -> str:
    """The target's symbol and conditional power-mean functional, for any fit order."""
    symbol = {1.0: "theta_N/NATE", 0.0: "theta_L", -1.0: "theta_H"}.get(
        omega, f"theta_omega (omega={order_text(omega)})"
    )
    return f"{symbol}: proportional effect on the conditional {mean_label(omega)}"


def context_metadata(context: ReportingContext) -> dict[str, object]:
    """Fresh, recursively detached metadata for each object/table access."""
    metadata = cast("dict[str, object]", json.loads(context.metadata_json))
    metadata.update(
        omega=context.omega,
        reference=context.reference.kind,
        reference_df=context.reference.df,
        status=context.point.status,
        coefficient_point=context.point.coefficient,
        covariance_point=context.point.covariance,
        population="conditional power mean",
        units="proportional",
        coverage="pointwise",
    )
    return metadata


@dataclass(frozen=True, slots=True)
class EffectResult:
    """A detached finite-change target with optional on-demand influence.

    Scalar inference, gradient, order and contrast are immutable. ``metadata`` and
    influence output are fresh on each access. The provider is not called until
    influence is explicitly requested; it uses the original retained capability.
    """

    context: ReportingContext = field(repr=False)
    numeric: TargetEstimate
    term: str
    contrast: tuple[float, float]
    level: float
    ci: IntervalKind
    _influence_provider: InfluenceProvider | None = field(default=None, repr=False)

    @property
    def estimate(self) -> float:
        """Proportional finite change expm1(D'beta)."""
        return self.numeric.reported.estimate

    @property
    def std_error(self) -> float:
        """Delta standard error exp(D'beta)se(D'beta)."""
        return self.numeric.reported.std_error

    @property
    def interval(self) -> tuple[float, float]:
        """Pointwise transformed or delta interval."""
        return self.numeric.reported.conf_low, self.numeric.reported.conf_high

    @property
    def label(self) -> str:
        """The conditional power-mean target, not a causal identification claim."""
        return target_label(self.context.omega)

    @property
    def order(self) -> float:
        """Actual fitted mean order."""
        return self.context.omega

    @property
    def population(self) -> str:
        """The target conditions on the other regressors and effects."""
        return "conditional power mean"

    @property
    def gradient(self) -> FloatArray:
        """A fresh coefficient gradient on the reported target scale."""
        return np.array(self.numeric.gradient, dtype=np.float64)

    @property
    def metadata(self) -> dict[str, object]:
        """Detached covariance, reference, point, contrast and interval metadata."""
        metadata = context_metadata(self.context)
        metadata.update(
            label=self.label,
            term=self.term,
            contrast=list(self.contrast),
            level=self.level,
            ci=self.ci,
            target_kind=self.numeric.kind,
        )
        return metadata

    def to_dict(self) -> dict[str, object]:
        """Fresh scalar row for the selected-target table."""
        row = self.numeric.reported
        return {
            "term": self.term,
            "target": self.label,
            "contrast": self.contrast,
            "estimate": row.estimate,
            "std_error": row.std_error,
            "statistic": row.statistic,
            "p_value": row.p_value,
            "conf_low": row.conf_low,
            "conf_high": row.conf_high,
            "inference": row.status,
            "log_contrast": self.numeric.linear.estimate,
            "log_std_error": self.numeric.linear.std_error,
            "reference": self.context.reference.kind,
            "reference_df": self.context.reference.df,
            "units": "proportional",
            "unavailable_reason": None,
        }

    def influence(self) -> pd.Series[float]:
        """Target gradient times genuinely retained coefficient contributions.

        Raises the existing discarded-state/certification refusal from the provider.
        Ordinary target output never calls this method or reconstructs primitives.
        """
        if self._influence_provider is None:
            raise SpecificationError(
                "specification.influence_discarded", {"request": "target influence"}
            )
        source = self._influence_provider()
        if tuple(source.columns) != self.context.names:
            raise ValueError("target influence and coefficient basis differ")
        values = source.to_numpy(dtype=np.float64) @ self.gradient
        return pd.Series(
            np.array(values, dtype=np.float64, copy=True),
            index=source.index.copy(deep=True),
            name=self.term,
        )


def effect_result(
    context: ReportingContext,
    columns: dict[str, str],
    term: str,
    *,
    contrast: tuple[float, float] | None = None,
    level: float = 0.95,
    ci: IntervalKind = "transform",
    influence_provider: InfluenceProvider | None = None,
) -> EffectResult:
    """Build one settled simple-regressor target through the common engine."""
    if ci not in ("transform", "delta"):
        raise invalid("ci", ci, "'transform' or 'delta'")
    target, endpoints = coefficient_target(context, columns, term, contrast)
    numeric = evaluate(context, target, level=level, ci=ci)
    return EffectResult(
        context, numeric, term, endpoints, level, ci, influence_provider
    )


def effects_table(
    context: ReportingContext,
    design: Design,
    terms: Sequence[str] | None,
    *,
    level: float = 0.95,
    ci: IntervalKind = "transform",
) -> pd.DataFrame:
    """Selected regressor targets, or every coefficient's exp(beta)-1 convenience.

    Without a selection, the table holds every reported coordinate. A coordinate
    transformation is not a full regressor contrast: categorical, interacted, nonlinear
    and intercept coordinates remain labeled as coordinates, with no claimed regressor
    endpoints. Selected names retain their standalone one-unit semantics and
    ambiguity/omission refusals.
    """
    reference_values(context.reference, level, math.nan)
    if ci not in ("transform", "delta"):
        raise invalid("ci", ci, "'transform' or 'delta'")
    if terms is not None and (
        isinstance(terms, str) or not isinstance(terms, Sequence)
    ):
        raise invalid("terms", terms, "a sequence of selected regressor names")
    requested = list(context.names if terms is None else terms)
    if any(not isinstance(term, str) for term in requested):
        raise invalid("terms", requested, "a sequence of selected regressor names")
    columns = {} if terms is None else simple_columns(design)
    rows: list[dict[str, object]] = []
    for term in requested:
        if terms is None:
            rows.append(_coefficient_row(context, term, level=level, ci=ci))
        else:
            rows.append(
                effect_result(context, columns, term, level=level, ci=ci).to_dict()
            )
    table = pd.DataFrame(rows, columns=_COLUMNS)
    table.attrs.update(context_metadata(context))
    table.attrs.update(
        level=level,
        ci=ci,
        selection=list(requested),
        panel=(
            "all coefficient transformations"
            if terms is None
            else "selected regressor targets"
        ),
        interpretation=(
            "exp(beta)-1 for each reported coefficient coordinate; "
            "not a standalone regressor change"
            if terms is None
            else "finite changes in standalone direct level regressors"
        ),
    )
    return table


_COLUMNS = (
    "term",
    "target",
    "contrast",
    "estimate",
    "std_error",
    "statistic",
    "p_value",
    "conf_low",
    "conf_high",
    "inference",
    "log_contrast",
    "log_std_error",
    "reference",
    "reference_df",
    "units",
    "unavailable_reason",
)


def _coefficient_row(
    context: ReportingContext,
    term: str,
    *,
    level: float,
    ci: IntervalKind,
) -> dict[str, object]:
    """The coordinate transformation, distinct from a regressor change."""
    target = LinearTarget(tuple(1.0 if name == term else 0.0 for name in context.names))
    numeric = evaluate(context, target, level=level, ci=ci)
    row = EffectResult(context, numeric, term, (0.0, 1.0), level, ci).to_dict()
    row.update(
        target=f"{target_label(context.omega)}; coefficient coordinate {term!r}",
        contrast=None,
    )
    return row
