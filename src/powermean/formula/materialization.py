"""Explicit native pandas materialization bridge across Formulaic and pandas versions.

``DESIGN.md``, Section 4.2; ``design/06_reuse_map.md``, the rows of ``formula/``;
decisions D51 and D56; task M1.49.  Formulaic 1.0.0 registers ``PandasMaterializer``
only for ``pandas.core.frame.DataFrame`` and detects categorical series only for
``object`` and ``CategoricalDtype``.  Under pandas 3, ``DataFrame.__module__`` is
``"pandas"`` and default string series have dtype ``"str"`` (``na_value=nan``), while
nullable ``"string"`` series have ``na_value=pd.NA``.  Moreover, ``ModelSpec``
canonicalizes any non-string materializer argument to ``REGISTER_NAME`` (``"pandas"``)
during initialization, so passing an unregistered subclass through ``model_matrix``
sugar loses the subclass on evaluation and replay.

This module defines a non-registering ``PandasMaterializer`` subclass (it sets no
``REGISTER_NAME`` attribute in its own class dictionary, leaving Formulaic's
``REGISTERED_NAMES`` and ``REGISTERED_INPUTS`` untouched while inheriting
``REGISTER_NAME = "pandas"`` on fitted specs) that extends ``_is_categorical`` for
pandas 3's ``"str"`` dtype while preserving formula-visible ``Series`` objects, input
dtypes and nullable ``"string"`` behavior.  All four materialization entry points (first
part evaluation, learned-spec replay, ``na_action="drop"`` accounting and
offset/exposure expression evaluation) call the public materializer-instance
``get_model_matrix`` method with an explicit empty context.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from powermean.errors import SpecificationError
from powermean.formula.formulaic_api import (
    EvaluatedFactor,
    Factor,
    FactorKind,
    FormulaMaterializer,
    PandasMaterializer,
    Spec,
    override,
)


def _is_pandas_str(values: object) -> bool:
    """Whether ``values`` is a pandas series or categorical with ``"str"`` dtype."""
    return isinstance(values, (pd.Series, pd.Categorical)) and values.dtype == "str"


@dataclass(frozen=True, slots=True)
class CategoricalEncoding:
    """The fitted categorical kind and immutable levels of one factor."""

    kind: FactorKind
    levels: tuple[Hashable, ...]


class _PandasMaterializer(PandasMaterializer):
    """Non-registering ``PandasMaterializer`` recognizing pandas 3 ``"str"``."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        context: Mapping[str, object],
        learned: Mapping[str, CategoricalEncoding] | None = None,
        argument: Literal["configurations", "data"] = "configurations",
    ) -> None:
        """Use the fitted categorical encoding for the named input when supplied."""
        super().__init__(data, context=context)
        self._learned = learned
        self._argument = argument

    @override(force=True)
    def _evaluate_factor(
        self, factor: Factor, spec: Spec, drop_rows: set[int]
    ) -> EvaluatedFactor:
        """Preserve a fitted categorical kind even when both raw rows are numeric."""
        learned = None if self._learned is None else self._learned.get(factor.expr)
        if learned is not None:
            # Formulaic's pooled evaluation spec need not hold encoder_state.
            factor = copy(factor)
            factor.kind = learned.kind
        evaluated = super()._evaluate_factor(factor, spec, drop_rows)
        if learned is not None and factor.eval_method.value == "lookup":
            # Formulaic 1.0 determines rank from the incoming dtype. A fitted
            # categorical column still spans the intercept on numeric inputs.
            evaluated.metadata.spans_intercept = True
        return evaluated

    @override(force=True)
    def _encode_evaled_factor(
        self,
        factor: EvaluatedFactor,
        spec: Spec,
        drop_rows: Sequence[int],
        reduced_rank: bool = False,
    ) -> dict[str, object]:
        """Refuse unlearned values using the fitted kind, never an input dtype."""
        learned = None if self._learned is None else self._learned.get(factor.expr)
        if learned is not None:
            factor = _checked_factor(factor, learned, argument=self._argument)
        return super()._encode_evaled_factor(factor, spec, drop_rows, reduced_rank)

    @override
    def _is_categorical(self, values: object) -> bool:
        """Whether ``values`` is categorical, including pandas 3 ``"str"``."""
        return _is_pandas_str(values) or super()._is_categorical(values)

    @override(force=True)
    def _encode_categorical(
        self, values: object, *args: object, **kwargs: object
    ) -> object:
        """Encode ``values`` into contrasts after unwrapping ``FactorValues``."""
        return super()._encode_categorical(
            getattr(values, "__wrapped__", values), *args, **kwargs
        )


def _materializer(data: pd.DataFrame) -> PandasMaterializer:
    """Bind the explicit native pandas materializer to ``data`` with empty context."""
    materializer_cls = FormulaMaterializer.for_materializer(_PandasMaterializer)
    return materializer_cls(data, context={})


def materialize_formula(formula: str, data: pd.DataFrame) -> object:
    """Evaluate ``formula`` on ``data`` with explicit pandas materialization."""
    return _materializer(data).get_model_matrix(
        formula,
        output="pandas",
        na_action="ignore",
    )


def materialize_spec(
    spec: Spec, data: pd.DataFrame, *, na_action: str | None = None
) -> object:
    """Evaluate a fitted ``spec`` on ``data`` without learning new state."""
    materializer = _materializer(data)
    if na_action is None:
        return materializer.get_model_matrix(spec)
    return materializer.get_model_matrix(spec, na_action=na_action)


def owned_spec(spec: Spec) -> Spec:
    """Own only fitted encoding/transform metadata, never a sample or row graph."""
    return spec.update(
        encoder_state=deepcopy(dict(spec.encoder_state)),
        transform_state=deepcopy(dict(spec.transform_state)),
    )


def _checked_factor(
    factor: EvaluatedFactor,
    learned: CategoricalEncoding,
    *,
    argument: Literal["configurations", "data"],
) -> EvaluatedFactor:
    """Check observed labels before Formulaic encodes them against its fitted levels."""
    data = pd.Series(factor.values.__wrapped__)
    observed = data.dropna() if argument == "data" else data
    levels = pd.CategoricalDtype(list(learned.levels)).categories
    codes = levels.get_indexer(pd.Index(observed))
    if (codes < 0).any():
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": argument,
                "value": {
                    factor.expr: observed.iloc[(codes < 0).nonzero()[0]].tolist()
                },
                "valid": "levels of the fitted categorical encoding",
            },
        )
    if argument == "data" and isinstance(data.dtype, pd.CategoricalDtype):
        # pandas can refuse a category listed only in the new dtype. At this
        # point every observed label belongs to the fitted encoding.
        values = type(factor.values)(
            data.cat.set_categories(levels), metadata=factor.metadata
        )
        return factor.replace(values=values)
    return factor


def materialize_fitted(
    spec: Spec, data: pd.DataFrame, *, argument: Literal["configurations", "data"]
) -> object:
    """Replay the fitted state once with categorical checks and null-row positions."""
    local = owned_spec(spec)
    levels = {
        expr: CategoricalEncoding(
            kind, tuple(cast("Iterable[Hashable]", state["categories"]))
        )
        for expr, (kind, state) in local.encoder_state.items()
        if kind.value == "categorical" and "categories" in state
    }
    materializer = _PandasMaterializer(
        data, context={}, learned=levels, argument=argument
    )
    return materializer.get_model_matrix(local, na_action="drop")
