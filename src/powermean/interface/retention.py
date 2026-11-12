"""Internal prepared ownership, dependence declarations, and concrete outcome admission.

This module
coordinates ``prepare_retained_problem`` and delegates :class:`RetainedPreparation` to
:mod:`powermean.interface.retained_preparation` and concrete outcome, trace, trigger,
and metadata binding to :mod:`powermean.interface.retained_admission` and
:mod:`powermean.interface.retained_binding`.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from numpy.typing import ArrayLike

from powermean._prepare.compose import (
    Columns,
    PreparedFrame,
    prepare_frame,
)
from powermean._prepare.dependence import UsedDependence
from powermean.data.inference_requests import (
    ClusterRequest,
    ConleyRequest,
    HacRequest,
)
from powermean.data.sample_fe_state import FESample
from powermean.data.weights import ObservationWeights
from powermean.formula.frame import Design, model_frame
from powermean.interface.expressions import (
    evaluate_expression,
    expression_values,
    interaction_column,
    interaction_names,
)
from powermean.interface.retained_binding import (
    SampleTarget,
    _bind_metadata_parts,
    _bind_owned_metadata_parts,
)
from powermean.interface.retained_binding import (
    bind_sample_metadata as _bind_sample_metadata,
)
from powermean.interface.retained_context import (
    PerFitPreparationContext,
    authenticate_unscoped_frame,
)
from powermean.interface.retained_metadata import (
    BoundDependence,
    RetainedColumnMetadata,
)
from powermean.interface.retained_outcomes import (
    admit_sample_outcome as _admit_sample_outcome,
)
from powermean.interface.retained_preparation import (
    RetainedPreparation,
    used_dependence_from_requests,
)
from powermean.interface.retained_traces import (
    admit_recorded_trace as _admit_recorded_trace,
)
from powermean.interface.retained_traces import (
    admit_trigger_record as _admit_trigger_record,
)
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_trace import (
    RetainedTraceSnapshot,
    RetainedTriggerSnapshot,
)
from powermean.results.primitive_snapshot import PRIMITIVE, MetadataParts
from powermean.results.retained_input_values import (
    InputKey,
    SampleKey,
)

__all__ = [
    "BoundDependence",
    "PerFitPreparationContext",
    "RetainedColumnMetadata",
    "RetainedPreparation",
    "admit_recorded_trace",
    "admit_sample_outcome",
    "admit_trigger_record",
    "bind_sample_metadata",
    "prepare_retained_problem",
    "used_dependence_from_requests",
]


def _unwrap_context(
    preparation: RetainedPreparation | PerFitPreparationContext | None,
) -> PerFitPreparationContext | None:
    """Extract :class:`PerFitPreparationContext` from ``preparation``."""
    if preparation is None:
        return None
    if type(preparation) is RetainedPreparation:
        return preparation._context
    if type(preparation) is PerFitPreparationContext:
        return preparation
    raise TypeError(
        "preparation must be a RetainedPreparation or PerFitPreparationContext"
    )


def bind_sample_metadata(
    prepared: RetainedPreparation | PerFitPreparationContext | PreparedFrame,
    sample: SampleTarget,
    *,
    weights: ObservationWeights | None = None,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    input_key: InputKey | None = None,
    sample_key: SampleKey | None = None,
) -> BoundDependence:
    """Bind retained dependence metadata to ``sample`` from ``prepared``."""
    if isinstance(prepared, RetainedPreparation):
        if type(prepared) is not RetainedPreparation:
            raise TypeError("prepared must be a concrete RetainedPreparation")
        ctx = prepared._context
    elif isinstance(prepared, PerFitPreparationContext):
        if type(prepared) is not PerFitPreparationContext:
            raise TypeError("prepared must be a concrete PerFitPreparationContext")
        ctx = prepared
    else:
        authenticate_unscoped_frame(
            prepared, sample=sample if type(sample) is FESample else None
        )
        return _bind_sample_metadata(
            prepared,
            sample,
            weights=weights,
            hac_request=hac_request,
            conley_request=conley_request,
            input_key=input_key,
            sample_key=sample_key,
        )
    matched = ctx.matching_frame(sample) if type(sample) is FESample else None
    parts: MetadataParts | PreparedFrame
    if ctx.snapshot_route == PRIMITIVE:
        # The live frame is identified and the live sample validated against the
        # once-materialized parts of the immutable capture; the consumer reads the
        # three parts it needs of those parts, not a rebuilt frame.
        if matched is not None:
            ctx.authenticate_frame(matched, sample=cast("FESample", sample))
        parts = ctx._materialized_metadata()
    else:
        source = ctx._owned_template() if matched is None else matched
        ctx.authenticate_frame(
            source,
            sample=sample if matched is not None and type(sample) is FESample else None,
        )
        parts = source
    # Only the internal NumericExecution has admitted these exact final rows
    # and copy-free source columns. Public/foreign/primitive routes remain checked.
    operation = ctx._operation
    if (operation is not None and operation.numeric is not None
            and type(sample) is FESample and type(parts) is PreparedFrame):
        owned = operation.maps.sample_map._owned_snapshot(sample)
        final_key = sample_key if sample_key is not None else (
            None if owned is None else owned.key
        )
        return _bind_owned_metadata_parts(
            parts, sample, weights=weights,
            hac_request=ctx.hac_request if hac_request is None else hac_request,
            conley_request=ctx.conley_request if conley_request is None else conley_request,
            input_key=ctx.input_key if input_key is None else input_key,
            sample_key=final_key,
        )
    return _bind_metadata_parts(
        parts,
        sample,
        weights=weights,
        hac_request=ctx.hac_request if hac_request is None else hac_request,
        conley_request=ctx.conley_request if conley_request is None else conley_request,
        input_key=ctx.input_key if input_key is None else input_key,
        sample_key=sample_key,
    )


def admit_sample_outcome(
    outcome: object,
    *,
    preparation: RetainedPreparation | PerFitPreparationContext | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
) -> RetainedOutcomeSnapshot:
    """Admit a ``SampleOutcome`` and detach it into ``RetainedOutcomeSnapshot``."""
    return _admit_sample_outcome(
        outcome,
        context=_unwrap_context(preparation),
        sample_snapshot=sample_snapshot,
    )


def admit_recorded_trace(
    trace: object,
    *,
    preparation: RetainedPreparation | PerFitPreparationContext | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
) -> RetainedTraceSnapshot:
    """Admit a ``RecordedTrace`` and detach it into ``RetainedTraceSnapshot``."""
    return _admit_recorded_trace(
        trace,
        context=_unwrap_context(preparation),
        sample_snapshot=sample_snapshot,
    )


def admit_trigger_record(
    record: object,
    *,
    preparation: RetainedPreparation | PerFitPreparationContext | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
) -> RetainedTriggerSnapshot:
    """Admit a ``TriggerRecord`` and detach it into ``RetainedTriggerSnapshot``."""
    return _admit_trigger_record(
        record,
        context=_unwrap_context(preparation),
        sample_snapshot=sample_snapshot,
    )


def _resolve_offset_or_exposure(
    working: pd.DataFrame, columns: Columns
) -> tuple[Columns, Design | None]:
    """Evaluate offset or exposure expression onto ``working`` if needed."""
    if columns.offset is not None and columns.exposure is not None:
        raise ValueError("offset and exposure cannot both be given")
    off_col, exp_col, design = columns.offset, columns.exposure, None
    for attr, role in ((off_col, "offset"), (exp_col, "exposure")):
        if attr is not None and attr not in working.columns:
            design = evaluate_expression(attr, working, role)
            col_name = design.columns[0]
            if col_name in working.columns:
                raise ValueError(f"expression {attr!r} collides with {col_name!r}")
            working[col_name] = expression_values(design)
            if role == "offset":
                off_col = col_name
            else:
                exp_col = col_name
    return (
        Columns(
            columns.weights,
            columns.weight_kind,
            off_col,
            exp_col,
            columns.cluster,
            columns.metadata,
        ),
        design,
    )


def prepare_retained_problem(
    formula: str,
    data: pd.DataFrame,
    *,
    columns: Columns | None = None,
    subset: ArrayLike | None = None,
    used_dependence: UsedDependence | None = None,
    positive_only: bool = False,
    vcov_request: ClusterRequest | HacRequest | ConleyRequest | None = None,
    primitive_snapshot: bool = False,
) -> RetainedPreparation:
    """Prepare a formula and dependence declarations once on an owned working copy.

    ``primitive_snapshot`` (default off) is passed through to
    :class:`RetainedPreparation`: the internal opt-in route; the route taken is
    recorded on the context.
    """
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    working = data.copy(deep=True)
    resolved_cols, off_design = _resolve_offset_or_exposure(
        working, Columns() if columns is None else columns
    )
    c_req = vcov_request if isinstance(vcov_request, ClusterRequest) else None
    h_req = vcov_request if isinstance(vcov_request, HacRequest) else None
    sp_req = vcov_request if isinstance(vcov_request, ConleyRequest) else None
    resolved_used = used_dependence or used_dependence_from_requests(
        cluster=resolved_cols.cluster,
        cluster_request=c_req,
        hac_request=h_req,
        conley_request=sp_req,
        retain=resolved_cols.metadata,
    )
    for name in (
        *([] if resolved_cols.cluster is None else [resolved_cols.cluster]),
        *resolved_used.clusters,
        *resolved_used.retained,
    ):
        if name not in working.columns:
            parts = interaction_names(name)
            if parts is not None:
                working[name] = interaction_column(working, *parts)
    frame = model_frame(formula, working)
    prepared = prepare_frame(
        frame, working, columns=resolved_cols, subset=subset, dependence=resolved_used
    )
    return RetainedPreparation(
        prepared,
        positive_only=positive_only,
        used_dependence=resolved_used,
        cluster_request=c_req,
        hac_request=h_req,
        conley_request=sp_req,
        offset_design=off_design,
        primitive_snapshot=primitive_snapshot,
    )

