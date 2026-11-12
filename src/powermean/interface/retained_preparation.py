"""Order-neutral retained preparation record and dependence request extraction.

This module defines
:func:`used_dependence_from_requests` and :class:`RetainedPreparation` backed by an
isolated :class:`~powermean.interface.retained_context.PerFitPreparationContext`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.dependence import (
    DependenceProvenance,
    UsedDependence,
)
from powermean.data.inference_conventions import RetainedColumns
from powermean.data.inference_requests import (
    ClusterRequest,
    ConleyRequest,
    HacRequest,
)
from powermean.data.request_values import given
from powermean.data.retained import OrderSample
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.formula.frame import Design, ModelFrame
from powermean.interface.retained_context import (
    PerFitPreparationContext,
)
from powermean.interface.retained_outcomes import (
    admit_sample_outcome as _admit_sample_outcome,
)
from powermean.interface.sample import (
    SampleOutcome,
    SampleRequest,
    prepare_sample,
)
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.owned import owned_design
from powermean.results.retained_input_values import (
    InputKey,
    RetainedInputSnapshot,
    _owned_dependence_provenance,
    _owned_effect_columns,
    _owned_model_frame,
    _owned_sample_base,
)

__all__ = [
    "RetainedPreparation",
    "used_dependence_from_requests",
]


def used_dependence_from_requests(
    *,
    cluster: str | None = None,
    cluster_request: ClusterRequest | None = None,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    retain: RetainedColumns | Sequence[str] | None = None,
) -> UsedDependence:
    """Form :class:`UsedDependence` from fit-time covariance and ``retain`` requests."""
    clusters: list[str] = [] if cluster is None else [cluster]
    if cluster_request is not None:
        clusters.extend(d for d in cluster_request.dimensions if d not in clusters)
    time_col = None if hac_request is None else hac_request.time
    unit_col = (
        str(hac_request.unit)
        if hac_request is not None and given(hac_request.unit)
        else None
    )
    coords = (
        None
        if conley_request is None
        else (conley_request.latitude, conley_request.longitude)
    )
    retained_names = (
        retain.names
        if isinstance(retain, RetainedColumns)
        else (() if retain is None else RetainedColumns(tuple(retain)).names)
    )
    return UsedDependence(tuple(clusters), time_col, unit_col, coords, retained_names)


@dataclass(frozen=True, init=False)
class RetainedPreparation:
    """Owned order-neutral preparation state for no-FE and FE fits."""

    input_key: InputKey
    positive_only: bool
    used_dependence: UsedDependence
    retained_columns: RetainedColumns
    cluster_request: ClusterRequest | None
    hac_request: HacRequest | None
    conley_request: ConleyRequest | None
    input_snapshot: RetainedInputSnapshot
    _context: PerFitPreparationContext
    _offset_design: Design | None

    def __init__(
        self,
        prepared: PreparedFrame,
        *,
        positive_only: bool = False,
        used_dependence: UsedDependence | None = None,
        cluster_request: ClusterRequest | None = None,
        hac_request: HacRequest | None = None,
        conley_request: ConleyRequest | None = None,
        offset_design: Design | None = None,
        primitive_snapshot: bool = False,
    ) -> None:
        """Store an isolated :class:`PerFitPreparationContext` and detached snapshot.

        ``primitive_snapshot`` is passed through to the context unchanged (internal
        opt-in route; default off).  On that route
        the outward parts below are fresh materializations of the context's immutable
        capture.
        """
        ctx = PerFitPreparationContext(
            prepared,
            used_dependence=used_dependence,
            cluster_request=cluster_request,
            hac_request=hac_request,
            conley_request=conley_request,
            primitive_snapshot=primitive_snapshot,
            clean_fit=not primitive_snapshot,
        )
        object.__setattr__(self, "input_key", ctx.input_key)
        object.__setattr__(self, "positive_only", bool(positive_only))
        object.__setattr__(self, "used_dependence", ctx.used_dependence)
        object.__setattr__(
            self, "retained_columns", RetainedColumns(ctx.used_dependence.retained)
        )
        object.__setattr__(self, "cluster_request", ctx.cluster_request)
        object.__setattr__(self, "hac_request", ctx.hac_request)
        object.__setattr__(self, "conley_request", ctx.conley_request)
        object.__setattr__(self, "input_snapshot", ctx.input_snapshot)
        object.__setattr__(self, "_context", ctx)
        object.__setattr__(
            self,
            "_offset_design",
            None if offset_design is None else owned_design(offset_design),
        )

    @classmethod
    def _from_normalized(
        cls, prepared: PreparedFrame, *, positive_only: bool,
        used_dependence: UsedDependence, cluster_request: ClusterRequest | None,
        offset_design: Design | None,
        single_fit: bool = False,
    ) -> RetainedPreparation:
        """Single owned boundary for prepare_problem's fresh canonical producer.

        Direct construction and explicit primitive snapshots remain unchanged.
        Unsupported callback-bearing state is served by the original constructor.
        Only the synchronous api._fit composition requests single_fit transfer;
        reusable public prepare keeps its original independent capture.
        """
        ctx = PerFitPreparationContext._from_normalized(
            prepared, used_dependence=used_dependence, cluster_request=cluster_request,
            single_fit=single_fit,
        )
        if ctx is None:
            return cls(prepared, positive_only=positive_only,
                       used_dependence=used_dependence, cluster_request=cluster_request,
                       offset_design=offset_design)
        result = object.__new__(cls)
        for name, value in (
            ("input_key", ctx.input_key), ("positive_only", bool(positive_only)),
            ("used_dependence", ctx.used_dependence),
            ("retained_columns", RetainedColumns(ctx.used_dependence.retained)),
            ("cluster_request", cluster_request), ("hac_request", None),
            ("conley_request", None), ("input_snapshot", ctx.input_snapshot),
            ("_context", ctx),
            ("_offset_design", None if offset_design is None else owned_design(offset_design)),
        ):
            object.__setattr__(result, name, value)
        return result

    def _for_fit(self) -> RetainedPreparation:
        """Bind one call-local execution context; never modify this preparation."""
        ctx = self._context.for_fit()
        if ctx is self._context:
            return self
        result = object.__new__(RetainedPreparation)
        for name in self.__dataclass_fields__:
            object.__setattr__(result, name, getattr(self, name))
        object.__setattr__(result, "_context", ctx)
        object.__setattr__(result, "input_snapshot", ctx.input_snapshot)
        return result

    @property
    def context(self) -> PerFitPreparationContext:
        """Return the underlying :class:`PerFitPreparationContext`."""
        return self._context

    @property
    def prepared(self) -> PreparedFrame:
        """Return a fresh owned PreparedFrame."""
        return self._context.template_frame

    @property
    def frame(self) -> ModelFrame:
        """Return a fresh owned ModelFrame."""
        capture = self._context._capture
        if capture is not None:
            return capture.model_frame()
        return _owned_model_frame(self._context._owned_template().frame)

    @property
    def base(self) -> SampleBase:
        """Return a fresh owned SampleBase."""
        capture = self._context._capture
        if capture is not None:
            return capture.sample_base()
        return _owned_sample_base(self._context._owned_template().base)

    @property
    def labels(self) -> pd.Index[Any]:
        """Return a fresh owned row Index."""
        capture = self._context._capture
        if capture is not None:
            return capture.labels_index()
        return self._context._owned_template().labels

    @property
    def effects(self) -> EffectColumns | None:
        """Return a fresh owned EffectColumns or None."""
        capture = self._context._capture
        if capture is not None:
            return capture.effect_columns()
        return _owned_effect_columns(self._context._owned_template().effects)

    @property
    def dependence(self) -> DependenceProvenance | None:
        """Return a fresh owned DependenceProvenance or None."""
        capture = self._context._capture
        if capture is not None:
            return capture.dependence_provenance()
        return _owned_dependence_provenance(self._context._owned_template().dependence)

    @property
    def offset_design(self) -> Design | None:
        """Return a fresh owned offset/exposure Design or None."""
        return (
            None if self._offset_design is None else owned_design(self._offset_design)
        )

    def for_order(
        self, omega: float, *, positive_only: bool | None = None
    ) -> OrderSample:
        """Derive a no-FE :class:`OrderSample` on a fresh owned frame copy."""
        pos = self.positive_only if positive_only is None else positive_only
        return self._context.template_frame.for_order(omega, positive_only=pos)

    def owns_prepared(self, other: object) -> bool:
        """Return True if ``other`` is owned by this preparation's context."""
        return self._context.owns_prepared(other)

    def prepare_sample(self, request: SampleRequest) -> SampleOutcome:
        """Run the accepted seven-group FE sample coordinator on an isolated frame."""
        return prepare_sample(self._context.new_fit_frame(), request)

    def admit_outcome(
        self,
        outcome: object,
        *,
        sample_snapshot: RetainedSampleSnapshot | None = None,
    ) -> RetainedOutcomeSnapshot:
        """Admit a concrete ``SampleOutcome`` produced from this preparation."""
        return _admit_sample_outcome(
            outcome, context=self._context, sample_snapshot=sample_snapshot
        )
