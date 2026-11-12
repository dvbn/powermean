"""The preparation of a public fit: the frame, the sample base and the domain, once.

``pm.prepare`` builds the sample and design once and returns an object whose
``.fit(omega=, tau=)`` reuses them: it caches everything that does not depend on
``(omega, tau)``, the sample included (for a common outcome domain).
:func:`prepare_problem` runs the preparation boundary exactly once for a formula and
its data:

1. the formula is parsed; instruments are not available. Fixed effects are
   prepared with their order-neutral owned base;
2. the offset or exposure is a column, or an expression evaluated once on every
   row (:mod:`powermean.interface.expressions`); each requested cluster dimension is
   a column, or the interaction ``a^b`` formed from two; each evaluated column joins
   an owned working copy of the data, which the caller's data frame never sees.  One
   dimension is the plain cluster column; the dimensions
   of a request of several (``"a + b"``) are fit-used dependence columns of the
   preparation (:class:`~powermean._prepare.dependence.UsedDependence`),
   whose missing values exclude their rows at step 2 as a cluster variable's do and
   whose raw values are retained aligned with the rows, for the fit and for later
   requests;
3. the first part is evaluated by Formulaic on every row of the working copy, with
   its transformations and categories trained on the full input
   (:func:`powermean.formula.frame.model_frame`), before any exclusion;
4. the sample base of steps 1, 2 and 4 of the sample construction is built once
   (:func:`powermean._prepare.compose.prepare_frame`): subset, missing values by
   group, weights, with every input row retained or excluded for one reason.

:class:`PreparedProblem` is the frozen state that every fit of the problem reuses:
the owned prepared frame with its fitted right-hand side, the outcome domain
(``positive_only``, fixed here so that every mean order of the problem shares one
domain), the default inference and the design of an offset or exposure expression.
The sample of a mean order is derived from the base by the order rule
(:meth:`~powermean._prepare.compose.PreparedFrame.for_order`): with
``positive_only`` every order keeps the same rows; without it, an order at or below
zero on an outcome with zeros is refused by ``SampleError``, and no
row is selected anew.  The caller's data, subset and options are read once and
never retained or modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from powermean._prepare.compose import (
    Columns,
    PreparedFrame,
    _native_retained_original,
    _prepare_frame,
    prepare_frame,
)
from powermean._prepare.dependence import UsedDependence
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import ClusterRequest
from powermean.data.request_values import OMITTED, Omitted
from powermean.data.retained import OrderSample
from powermean.data.sample_fe import _fixed_original
from powermean.data.specs import Inference
from powermean.errors import SpecificationError
from powermean.formula.frame import Design, model_frame
from powermean.formula.grammar import parse_formula
from powermean.interface.expressions import (
    evaluate_expression,
    expression_values,
    interaction_column,
    interaction_names,
)
from powermean.interface.fe_bias_options import BiasRequest
from powermean.interface.fe_options import FEBread, FEVcovRequest
from powermean.interface.numeric_execution import _numeric_ownership_original
from powermean.interface.options import Preparation, Solver
from powermean.interface.retained_preparation import (
    RetainedPreparation,
    used_dependence_from_requests,
)
from powermean.results.retained_input_values import RetainedInputSnapshot
from powermean.solve.iols.candidate import refuse_iols


@dataclass(frozen=True, eq=False)
class PreparedProblem:
    """The prepared state of one formula on one data frame (module docstring).

    Attributes
    ----------
    frame : PreparedFrame
        The owned prepared frame and the order-neutral base of the sample.
    positive_only : bool
        The outcome domain of every fit of the problem.
    inference : Inference
        The default inference of the fits, whose cluster dimensions the preparation
        retained.
    offset_design : Design or None
        The fitted design of the offset or exposure expression, or ``None`` for a
        column or no offset.
    retained_preparation : RetainedPreparation or None
        The retained preparation coordinator when the formula has fixed effects.
    fe_defaults : tuple
        The default FE inference specification captured at preparation time.
    """

    frame: PreparedFrame
    positive_only: bool
    inference: Inference
    offset_design: Design | None
    retained_preparation: RetainedPreparation | None = None
    fe_defaults: tuple[
        FEVcovRequest,
        str | ConventionValues | Omitted,
        FEBread,
        bool,
        str | Literal[True] | None,
    ] = ("robust", OMITTED, "observed", False, None)
    bias_correction: BiasRequest = None
    _numeric_source: RetainedInputSnapshot | None = field(default=None, init=False, repr=False)

    @property
    def has_fixed_effects(self) -> bool:
        """Whether the formula has a fixed-effects part."""
        return self.retained_preparation is not None

    def sample(self, omega: float) -> OrderSample:
        """The sample of the mean order ``omega`` on the frozen domain."""
        return self.frame.for_order(omega, positive_only=self.positive_only)


def _check_inputs(formula: object, data: object) -> pd.DataFrame:
    """The formula is a string and the data a data frame."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    return data


def _check_parts(formula: str, *, solver: Solver = "newton") -> bool:
    """Refuse instruments and detect the implemented FE route."""
    parts = parse_formula(formula)
    if solver == "iols" and parts.fixed_effects:
        refuse_iols("fixed effects")
    if parts.instruments is not None:
        if solver == "iols":
            refuse_iols("instruments (i2SLS)")
        raise SpecificationError(
            "specification.unimplemented",
            {"feature": "instruments"},
        )
    return bool(parts.fixed_effects)


def _offset_column(
    working: pd.DataFrame, name: str, argument: str
) -> tuple[str, Design | None]:
    """A column as it is, or an expression evaluated and added to the working copy."""
    if name in working.columns:
        return name, None
    design = evaluate_expression(name, working, argument)
    column = design.columns[0]
    if column in working.columns:
        raise ValueError(
            f"the expression {name!r} evaluates to the column {column!r}, "
            "which is a column of the data"
        )
    working[column] = expression_values(design)
    return column, design


def _cluster_column(working: pd.DataFrame, name: str) -> str:
    """A cluster column as it is, or the interaction ``a^b`` added to the copy."""
    if name in working.columns:
        return name
    names = interaction_names(name)
    if names is None:
        return name
    working[name] = interaction_column(working, *names)
    return name


def _ordinary_columns(
    working: pd.DataFrame,
    preparation: Preparation,
    offset: str | None,
    exposure: str | None,
) -> tuple[Columns, UsedDependence | None]:
    """The columns read beside the formula of a fit without fixed effects.

    Every requested cluster dimension, in order and once, formed in the working copy
    where it is an interaction (module docstring, step 2): one dimension is the
    plain cluster column, several the fit-used dependence columns.
    """
    names: dict[str, None] = {}
    for name in preparation.inference.dimensions:
        names[_cluster_column(working, name)] = None
    dimensions = tuple(names)
    columns = Columns(
        weights=preparation.weights,
        weight_kind=preparation.weight_kind,
        offset=offset,
        exposure=exposure,
        cluster=dimensions[0] if len(dimensions) == 1 else None,
    )
    if len(dimensions) < 2:
        return columns, None
    return columns, UsedDependence(clusters=dimensions)


_RETAIN_INTERACTION = interaction_column
_RETAIN_CLUSTER = _cluster_column
_RETAIN_FRAME = prepare_frame
_RETAIN_NUMERIC_FRAME = _prepare_frame
_RETAIN_MODEL = model_frame
_RETAIN_NAMES = interaction_names


def _numeric_retained(
    data: pd.DataFrame, used: UsedDependence, regressors: str,
) -> dict[str, tuple[str, ...]] | None:
    """Original native columns only; unknown hooks keep ordered eager preparation.

    This is called only by the unexposed direct-fit producer, before model/sample
    creation. Literal interaction-named columns always precede derived pairs.
    There is no stored eligibility flag and no final-level reconstruction.
    """
    if (not _fixed_original() or not _native_retained_original()
            or not _numeric_ownership_original()
            or model_frame is not _RETAIN_MODEL
            or interaction_names is not _RETAIN_NAMES
            or type(regressors) is not str
            or type(used.retained) is not tuple or type(used.clusters) is not tuple
            or not all(type(name) is str for name in (*used.clusters, *used.retained))
            or prepare_frame is not _RETAIN_FRAME
            or _prepare_frame is not _RETAIN_NUMERIC_FRAME
            or type(data) is not pd.DataFrame
            or interaction_column is not _RETAIN_INTERACTION
            or _cluster_column is not _RETAIN_CLUSTER):
        return None
    # Unknown formula/data callbacks keep the original early interaction order.
    # This admits numeric first parts generally, not one FE topology or family.
    tokens = tuple(part.strip() for part in regressors.replace("-", "+").split("+"))
    if (data.attrs or type(data.index) is not pd.RangeIndex
            or (data.index.name is not None and type(data.index.name) is not str)
            or (data.columns.name is not None and type(data.columns.name) is not str)
            or not all(type(name) is str for name in data.columns)
            or not all(isinstance(dtype, np.dtype) and dtype.kind in "biuf"
                       and dtype.fields is None and dtype.metadata is None
                       for dtype in data.dtypes)
            or any(token not in ("", "0", "1") and (
                not token.isidentifier() or token not in data.columns
            ) for token in tokens)):
        return None
    result: dict[str, tuple[str, ...]] = {}
    for name in used.retained:
        if name in used.clusters:
            continue
        if name in data.columns:
            terms = (name,)
        else:
            parts = tuple(part.strip() for part in name.split("^"))
            if len(parts) != 2 or not all(parts):
                return None
            terms = parts
        for term in terms:
            if term not in data.columns:
                return None
            column = data[term]
            if (type(column) is not pd.Series or not isinstance(column.dtype, np.dtype)
                    or column.dtype.kind not in "biuf"
                    or column.dtype.fields is not None
                    or column.dtype.metadata is not None):
                return None
        result[name] = tuple(terms)
    return result


def prepare_problem(
    formula: str,
    data: pd.DataFrame,
    preparation: Preparation,
    *,
    solver: Solver = "newton",
    _single_fit: bool = False,
) -> PreparedProblem:
    """Prepare ``formula`` on ``data`` once (module docstring).

    ``_single_fit`` is private composition of api._fit's unexposed producer.
    Public prepare leaves it off; no public keyword selects this lifetime.

    Parameters
    ----------
    formula : str
        A formula of the grammar of :mod:`powermean.formula.grammar`, with its first
        part only.
    data : pandas.DataFrame
        The data, read by position and never modified.
    preparation : Preparation
        The normalized sample keywords.

    Returns
    -------
    PreparedProblem
        The frozen state that every fit of the problem reuses.

    Raises
    ------
    TypeError
        A formula that is not a string, or data that are not a data frame.
    SpecificationError
        A malformed formula, fixed effects or instruments, an
        expression that is not one numeric column.
    KeyError, ValueError, SampleError
        The refusals of the preparation: a column that the data do not
        have, a subset that is not Boolean or not aligned, both an offset and an
        exposure, a negative outcome, no complete row.
    """
    frame_data = _check_inputs(formula, data)
    has_fe = _check_parts(formula, solver=solver)
    if solver == "iols":
        if preparation.weights is not None:
            refuse_iols("weights")
        if preparation.offset is not None or preparation.exposure is not None:
            refuse_iols("offsets or exposure")
    working = frame_data.copy(deep=True)
    offset = exposure = None
    design: Design | None = None
    if preparation.offset is not None:
        offset, design = _offset_column(working, preparation.offset, "offset")
    if preparation.exposure is not None:
        exposure, design = _offset_column(working, preparation.exposure, "exposure")
    if not has_fe:
        columns, used = _ordinary_columns(working, preparation, offset, exposure)
        frame = model_frame(formula, working)
        subset: Any = preparation.subset
        prepared = prepare_frame(
            frame, working, columns=columns, subset=subset, dependence=used
        )
        return PreparedProblem(
            frame=prepared,
            positive_only=preparation.positive_only,
            inference=preparation.inference,
            offset_design=design,
        )
    c_req = (
        preparation.fe_vcov if isinstance(preparation.fe_vcov, ClusterRequest) else None
    )
    fe_parts = parse_formula(formula)
    fe_retain_names = tuple(
        dict.fromkeys(
            (
                *preparation.retain.names,
                *(name for term in fe_parts.fixed_effects for name in term),
                *("^".join(term) for term in fe_parts.fixed_effects),
            )
        )
    )
    used_dep = used_dependence_from_requests(
        cluster=None,
        cluster_request=c_req,
        retain=fe_retain_names,
    )
    retained = (
        _numeric_retained(working, used_dep, fe_parts.regressors)
        if (_single_fit and type(frame_data) is pd.DataFrame
            and type(formula) is str) else None
    )
    for dep_name in (*used_dep.clusters, *used_dep.retained):
        if retained is None or dep_name not in retained:
            _cluster_column(working, dep_name)
    fe_columns = Columns(
        weights=preparation.weights,
        weight_kind=preparation.weight_kind,
        offset=offset,
        exposure=exposure,
    )
    fe_frame = model_frame(formula, working)
    fe_subset: Any = preparation.subset
    if retained is None:
        fe_prepared = prepare_frame(
            fe_frame, working, columns=fe_columns, subset=fe_subset, dependence=used_dep
        )
    else:
        fe_prepared = _prepare_frame(
            fe_frame, working, columns=fe_columns, subset=fe_subset,
            dependence=used_dep, _retained=retained,
        )
    retained_prep = RetainedPreparation._from_normalized(
        fe_prepared,
        positive_only=preparation.positive_only,
        used_dependence=used_dep,
        cluster_request=c_req,
        offset_design=design,
        single_fit=_single_fit,
    )
    result = PreparedProblem(
        frame=fe_prepared,
        positive_only=preparation.positive_only,
        inference=preparation.inference,
        offset_design=design,
        retained_preparation=retained_prep,
        fe_defaults=(
            preparation.fe_vcov,
            preparation.ssc,
            preparation.bread,
            preparation.explicit_vcov,
            preparation.cancel,
        ),
        bias_correction=preparation.bias_correction,
    )
    # This is the actual detached input owner of the normal normalization producer,
    # not a trust boolean on an arbitrary externally supplied preparation graph.
    source = retained_prep.context._normal_origin or retained_prep.input_snapshot
    object.__setattr__(retained_prep.context, "_normal_origin", source)
    object.__setattr__(result, "_numeric_source", source)
    return result
