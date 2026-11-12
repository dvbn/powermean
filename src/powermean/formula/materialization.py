"""Explicit native pandas materialization bridge across Formulaic and pandas versions.

Formulaic 1.0.0 registers ``PandasMaterializer``
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
from typing import cast

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
    ) -> None:
        """Bind an optional contrast-only frozen categorical policy to this call."""
        super().__init__(data, context=context)
        self._learned = learned

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
        return super()._evaluate_factor(factor, spec, drop_rows)

    @override(force=True)
    def _encode_evaled_factor(
        self,
        factor: EvaluatedFactor,
        spec: Spec,
        drop_rows: Sequence[int],
        reduced_rank: bool = False,
    ) -> dict[str, object]:
        """Refuse unlearned values using the fitted kind, never an input dtype."""
        if self._learned is not None:
            learned = self._learned.get(factor.expr)
            if learned is not None:
                outside = _unlearned(factor.values, learned.levels)
                if outside:
                    raise SpecificationError(
                        "specification.invalid_value",
                        {
                            "argument": "configurations",
                            "value": {factor.expr: outside},
                            "valid": "levels of the fitted categorical encoding",
                        },
                    )
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


def _unlearned(values: object, levels: tuple[Hashable, ...]) -> list[object]:
    """Values without a fitted level, using the encoder's categorical index."""
    data = pd.Series(getattr(values, "__wrapped__", values))
    codes = pd.CategoricalDtype(list(levels)).categories.get_indexer(pd.Index(data))
    return [
        value
        for value, code in zip(data.tolist(), codes.tolist(), strict=True)
        if code < 0
    ]


def materialize_contrast(spec: Spec, data: pd.DataFrame) -> object:
    """Replay two rows once, with local fitted state and null-row accounting.

    The supplied spec is the Design's separate fit-time metadata snapshot, never
    the live spec used by fit/predict. Drop accounting and the model matrix share
    one Formulaic evaluation. The caller refuses a dropped or nonfinite row.
    """
    local = owned_spec(spec)
    levels = {
        expr: CategoricalEncoding(
            kind, tuple(cast("Iterable[Hashable]", state["categories"]))
        )
        for expr, (kind, state) in local.encoder_state.items()
        if kind.value == "categorical" and "categories" in state
    }
    materializer = _PandasMaterializer(data, context={}, learned=levels)
    return materializer.get_model_matrix(local, na_action="drop")
