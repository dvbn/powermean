"""Centered-SD slope column scaling and back-transformation for no-FE IOLS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from powermean.solve.iols._numeric import FloatArray
from powermean.solve.iols.kernel_records import (
    TRANSFORMED_NORMALIZATION_ABS_BACKSTOP,
    TRANSFORMED_NORMALIZATION_REL_FLOOR,
)
from powermean.solve.iols.records import owned_value


@dataclass(frozen=True)
class TransformedNormalizationState:
    """State of the centered-SD slope column scaling."""

    requested: bool
    enabled: bool
    status: str
    basis: str = "std"
    scale: FloatArray | None = None
    condition_before: float | None = None
    condition_after: float | None = None
    scale_min: float | None = None
    scale_max: float | None = None
    scale_floor: float | None = None
    transformed_rank: int | None = None

    def __post_init__(self) -> None:
        """Keep the column scales in immutable, detached storage."""
        object.__setattr__(self, "scale", owned_value(self.scale))


def _gram_condition(values: npt.ArrayLike) -> float | None:
    matrix: FloatArray = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return None
    gram = matrix.T @ matrix
    if not np.all(np.isfinite(gram)):
        return float("inf")
    try:
        return float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        return float("inf")


def build_transformed_normalization_state(
    exog_slopes: npt.ArrayLike,
    *,
    mode: Literal["none", "scale"] = "scale",
) -> TransformedNormalizationState:
    """Build the centered-SD column scaling state for slope columns."""
    if mode not in {"none", "scale"}:
        raise ValueError(
            "transformed_design_normalization must be one of 'none' or 'scale'"
        )
    if mode == "none":
        return TransformedNormalizationState(
            requested=False,
            enabled=False,
            status="disabled",
        )
    exog: FloatArray = np.asarray(exog_slopes, dtype=np.float64)
    if exog.ndim != 2 or exog.shape[1] == 0:
        return TransformedNormalizationState(
            requested=True,
            enabled=False,
            status="no_slope_columns",
        )
    transformed: FloatArray = exog - np.mean(exog, axis=0)
    if transformed.shape != exog.shape or not np.all(np.isfinite(transformed)):
        raise ValueError("transformed_design_normalization_failed")

    raw_scale: FloatArray = np.std(transformed, axis=0)
    positive_scale = raw_scale[raw_scale > 0.0]
    max_scale = float(np.max(positive_scale)) if positive_scale.size else 0.0
    scale_floor = max(
        TRANSFORMED_NORMALIZATION_REL_FLOOR * max_scale,
        TRANSFORMED_NORMALIZATION_ABS_BACKSTOP,
    )
    near_zero = (raw_scale > 0.0) & (raw_scale < scale_floor)
    if np.any(near_zero):
        raise ValueError(
            "S_singular: transformed-design scale is below the relative floor"
        )

    scale: FloatArray = np.where(raw_scale == 0.0, 1.0, raw_scale)
    transformed_star = transformed / scale
    return TransformedNormalizationState(
        requested=True,
        enabled=True,
        status="enabled",
        basis="std",
        scale=np.asarray(scale, dtype=np.float64),
        condition_before=_gram_condition(transformed),
        condition_after=_gram_condition(transformed_star),
        scale_min=float(np.min(scale)) if scale.size else None,
        scale_max=float(np.max(scale)) if scale.size else None,
        scale_floor=float(scale_floor),
        transformed_rank=int(np.linalg.matrix_rank(transformed)),
    )


def apply_transformed_normalization(
    exog_slopes: npt.ArrayLike,
    state: TransformedNormalizationState,
) -> FloatArray:
    """Scale slope columns (`exog / scale`, no mean shift)."""
    if not state.enabled or state.scale is None:
        raw: FloatArray = np.asarray(exog_slopes, dtype=np.float64)
        return raw
    scaled: FloatArray = np.asarray(exog_slopes, dtype=np.float64) / state.scale
    return scaled


def back_transform_beta(
    beta_star: npt.ArrayLike,
    state: TransformedNormalizationState,
) -> FloatArray:
    """Map scaled slope coefficients back to original slope coordinates."""
    values: FloatArray = np.asarray(beta_star, dtype=np.float64)
    if not state.enabled or state.scale is None:
        return values.copy()
    res: FloatArray = values / state.scale
    return res


def back_transform_residual(
    beta: npt.ArrayLike,
    beta_star: npt.ArrayLike,
    state: TransformedNormalizationState,
) -> float | None:
    """Maximum absolute discrepancy between ``beta`` and ``beta_star / scale``."""
    if not state.enabled or state.scale is None:
        return None
    canonical: FloatArray = np.asarray(beta, dtype=np.float64)
    expected: FloatArray = np.asarray(beta_star, dtype=np.float64) / state.scale
    if canonical.shape != expected.shape:
        return float("inf")
    return float(np.max(np.abs(canonical - expected))) if canonical.size else 0.0
