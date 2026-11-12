"""Selection and frozen learned-design contrasts for coefficient targets.

A direct level regressor has a constant design derivative only if no other fitted term
reads it. Interactions, transformations and categorical coding require a full
learned-design contrast. This module invents no selectors or formula grammar. Internal
full-row and derivative records support those targets.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from powermean.data.request_values import invalid
from powermean.effects.targets import (
    LinearTarget,
    ReportingContext,
    derivative_target,
)
from powermean.errors import SpecificationError
from powermean.formula.frame import Design, replay_design
from powermean.formula.names import python_names


@dataclass(frozen=True, slots=True)
class DesignGradient:
    """Private full learned-column contrast or analytic design derivative.

    All fitted columns must be declared, including any omitted from the reported
    coefficient basis. A nonzero loading on an omitted column is refused.
    """

    columns: tuple[str, ...]
    values: tuple[float, ...]


def simple_columns(design: Design) -> dict[str, str]:
    """Unambiguous direct level regressors and their single learned columns.

    Inspect fitted factors once, with the same name-reading rules as preparation.
    No data rows, new formula fit, or coefficient-name transform heuristics.
    """
    affected: dict[str, set[str]] = {}
    direct: dict[str, tuple[str, str]] = {}
    columns_by_term = dict(design.terms)
    for term in design.spec.term_slices:
        term_name = str(term)
        for factor in term.factors:
            method = factor.eval_method.value
            reads = (
                {factor.expr}
                if method == "lookup"
                else python_names(factor.expr)
                if method == "python"
                else set()
            )
            for variable in reads.intersection(design.variables):
                affected.setdefault(variable, set()).add(term_name)
        columns = columns_by_term[term_name]
        if len(term.factors) == 1 and len(columns) == 1:
            factor = term.factors[0]
            if factor.eval_method.value == "lookup" and columns[0] == factor.expr:
                direct[factor.expr] = (term_name, columns[0])
    return {
        variable: column
        for variable, (term, column) in direct.items()
        if affected.get(variable) == {term}
    }


def needs_design(term: str) -> SpecificationError:
    """Refuse an ambiguous regressor effect without inventing a public selector."""
    return SpecificationError(
        "specification.unsupported_inference",
        {
            "method": "effect",
            "case": (
                f"{term!r} requires an explicit full learned-design contrast; "
                "a standalone direct level regressor is required by this selector"
            ),
        },
    )


def numeric_contrast(contrast: tuple[float, float] | None) -> tuple[float, float]:
    """The one-unit default, or an explicit finite numeric (d0,d1)."""
    if contrast is None:
        return 0.0, 1.0
    if not isinstance(contrast, tuple) or len(contrast) != 2:
        raise invalid("contrast", contrast, "a pair (d0, d1) of finite real values")
    if any(not _finite_real(value) for value in contrast):
        raise invalid("contrast", contrast, "a pair (d0, d1) of finite real values")
    return float(contrast[0]), float(contrast[1])


def _finite_real(value: object) -> bool:
    """Validate runtime endpoint types before the annotated numeric boundary."""
    return (
        not isinstance(value, bool | np.bool_)
        and isinstance(value, Real)
        and math.isfinite(value)
    )


def coefficient_target(
    context: ReportingContext,
    columns: dict[str, str],
    term: str,
    contrast: tuple[float, float] | None,
) -> tuple[LinearTarget, tuple[float, float]]:
    """A simple level finite change, with truthful absence and ambiguity refusals."""
    if not isinstance(term, str) or term not in columns:
        raise needs_design(term)
    column = columns[term]
    if column not in context.names:
        raise invalid("term", term, "a retained, non-omitted coefficient")
    start, end = numeric_contrast(contrast)
    weights = tuple(end - start if name == column else 0.0 for name in context.names)
    return LinearTarget(weights), (start, end)


def aligned_gradient(
    context: ReportingContext, design: Design, gradient: DesignGradient
) -> tuple[float, ...]:
    """Map a full learned design gradient to reported names; never drop a loading."""
    if gradient.columns != design.columns or len(gradient.values) != len(
        design.columns
    ):
        raise invalid(
            "design contrast", gradient.columns, "the full learned column order"
        )
    if not set(context.names).issubset(gradient.columns):
        raise invalid(
            "coefficient basis", context.names, "columns of the fitted design"
        )
    values = dict(zip(gradient.columns, gradient.values, strict=True))
    if any(value != 0.0 for name, value in values.items() if name not in context.names):
        raise invalid(
            "design contrast", gradient.values, "zero loadings on omitted columns"
        )
    return tuple(values[name] for name in context.names)


def replayed_target(
    context: ReportingContext,
    design: Design,
    configurations: pd.DataFrame,
    *,
    replay: Callable[[Design, pd.DataFrame], Design] | None = None,
) -> LinearTarget:
    """Private finite difference between two complete caller-specified raw rows.

    Row 1 minus row 0 uses fitted encoding, state and column order. Categorical
    differences and interactions therefore use the same full-gradient engine.
    """
    if not isinstance(configurations, pd.DataFrame) or len(configurations) != 2:
        raise invalid("configurations", "shape", "exactly two complete raw design rows")
    if len(configurations.columns) != len(design.variables) or set(
        configurations.columns
    ) != set(design.variables):
        raise invalid(
            "configurations",
            tuple(configurations.columns),
            f"exactly the raw RHS names {design.variables!r}",
        )
    materialized = (replay_design if replay is None else replay)(design, configurations)
    if (
        materialized.columns != design.columns
        or materialized.referenced_invalid.any()
        or materialized.evaluated_invalid.any()
    ):
        raise invalid("configurations", "invalid", "finite rows in the learned design")
    matrix = materialized.matrix.to_numpy(dtype=np.float64)
    gradient = DesignGradient(design.columns, tuple((matrix[1] - matrix[0]).tolist()))
    return LinearTarget(aligned_gradient(context, design, gradient))


def local_design_target(
    context: ReportingContext,
    design: Design,
    derivative: DesignGradient,
    *,
    evaluation_value: float | None = None,
) -> LinearTarget:
    """Private analytic full learned-design derivative, optionally multiplied by x.

    The derivative must be defined independently in the fitted column basis,
    including all nonlinear/interaction terms. No derivative is guessed from names
    and no evaluation row or value is chosen by this function.
    """
    return derivative_target(
        aligned_gradient(context, design, derivative), evaluation_value=evaluation_value
    )
