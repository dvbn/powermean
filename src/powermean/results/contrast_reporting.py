"""Explicit learned-design contrasts.

Only the two supplied raw RHS rows are replayed. The common design-target engine
owns validation/alignment and the effects engine owns inference. A contrast-only
bridge replay uses the Design's separate fitted encoding/transform metadata,
local per-call state and one null-aware materialization. It unconditionally refuses
levels outside the fitted categorical encoding, including numeric inputs, without
changing warning filters, fitting, prediction or ordinary replay behavior. FE and
offsets remain fixed; no level, basis, average or endpoint is guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermean.data.request_values import invalid
from powermean.effects.targets import (
    IntervalKind,
    ReportingContext,
    TargetEstimate,
    evaluate,
)
from powermean.errors import SpecificationError
from powermean.formula.frame import Design, replay_contrast_design
from powermean.model.variance import FloatArray
from powermean.results.effect_design import replayed_target
from powermean.results.effect_reporting import (
    InfluenceProvider,
    context_metadata,
    target_label,
)

RawRow = tuple[tuple[str, object], ...]


def _raw_row(value: Mapping[str, object], variables: tuple[str, ...]) -> RawRow:
    """Own one complete scalar RHS row before Formulaic evaluates its values."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise invalid("configuration", value, "a mapping of raw RHS names to scalars")
    if set(value) != set(variables):
        raise invalid(
            "configuration names",
            tuple(value),
            f"exactly the raw RHS names {variables!r}",
        )
    if any(not pd.api.types.is_scalar(value[name]) for name in variables):
        raise invalid("configuration values", value, "scalar raw RHS values")
    return tuple((name, deepcopy(value[name])) for name in variables)


@dataclass(frozen=True, slots=True)
class ContrastResult:
    """A detached conditional finite change between two explicit raw RHS rows.

    Numeric coordinates are immutable. Raw configurations, gradients, metadata,
    tables and influence output are fresh on access. The influence provider is
    called only by an explicit influence request, using its retained capability.
    """

    context: ReportingContext = field(repr=False)
    numeric: TargetEstimate
    _before: RawRow = field(repr=False)
    _after: RawRow = field(repr=False)
    learned_columns: tuple[str, ...]
    level: float
    ci: IntervalKind
    _influence_provider: InfluenceProvider | None = field(default=None, repr=False)

    @property
    def estimate(self) -> float:
        """Proportional finite change expm1(Delta'beta)."""
        return self.numeric.reported.estimate

    @property
    def std_error(self) -> float:
        """Delta standard error from the full active already-scaled covariance."""
        return self.numeric.reported.std_error

    @property
    def interval(self) -> tuple[float, float]:
        """Pointwise transformed or delta interval under the actual reference."""
        return self.numeric.reported.conf_low, self.numeric.reported.conf_high

    @property
    def label(self) -> str:
        """Conditional power-mean functional at the actual fitted order."""
        return target_label(self.context.omega)

    @property
    def order(self) -> float:
        """Actual fitted mean order, with NATE at order one."""
        return self.context.omega

    @property
    def population(self) -> str:
        """The existing conditional power-mean interpretation."""
        return "conditional power mean"

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Full reported coefficient order of the target gradient."""
        return self.context.names

    @property
    def gradient(self) -> FloatArray:
        """Fresh exp(Delta'beta)Delta in the full reported coefficient basis."""
        return np.array(self.numeric.gradient, dtype=np.float64)

    @property
    def before(self) -> dict[str, object]:
        """Detached raw starting configuration, without inferred dose endpoints."""
        return deepcopy(dict(self._before))

    @property
    def after(self) -> dict[str, object]:
        """Detached raw ending configuration."""
        return deepcopy(dict(self._after))

    @property
    def contrast(self) -> str:
        """Explicit direction and components held fixed."""
        return "X(after) - X(before); fixed effects and offsets held fixed"

    @property
    def metadata(self) -> dict[str, object]:
        """Detached configurations, coordinate order, inference and point metadata."""
        metadata = context_metadata(self.context)
        metadata.update(
            label=self.label,
            contrast=self.contrast,
            before=self.before,
            after=self.after,
            coefficient_names=list(self.coefficient_names),
            learned_columns=list(self.learned_columns),
            level=self.level,
            ci=self.ci,
            target_kind=self.numeric.kind,
            fixed_effects="held fixed",
            offset="held fixed",
        )
        return metadata

    def to_dict(self) -> dict[str, object]:
        """Fresh scalar target row with truthful raw configurations."""
        row = self.numeric.reported
        return {
            "target": self.label,
            "contrast": self.contrast,
            "before": self.before,
            "after": self.after,
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

    def to_frame(self) -> pd.DataFrame:
        """Fresh one-row target table with detached metadata."""
        table = pd.DataFrame([self.to_dict()])
        table.attrs.update(self.metadata)
        return table

    def influence(self) -> pd.Series[float]:
        """Full target gradient times genuinely retained coefficient contributions."""
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
            name="regressor configuration contrast",
        )


def contrast_result(
    context: ReportingContext,
    design: Design,
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    level: float = 0.95,
    ci: IntervalKind = "transform",
    influence_provider: InfluenceProvider | None = None,
) -> ContrastResult:
    """Replay two explicit rows and evaluate through the shared effects engine."""
    if ci not in ("transform", "delta"):
        raise invalid("ci", ci, "'transform' or 'delta'")
    start = _raw_row(before, design.variables)
    end = _raw_row(after, design.variables)
    rows = pd.DataFrame([dict(start), dict(end)], columns=list(design.variables))
    target = replayed_target(context, design, rows, replay=replay_contrast_design)
    numeric = evaluate(context, target, level=level, ci=ci)
    return ContrastResult(
        context, numeric, start, end, design.columns, level, ci, influence_provider
    )
