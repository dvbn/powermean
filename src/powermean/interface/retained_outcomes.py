"""Concrete coordinator outcome validation and lowering into detached snapshots.

This module validates concrete
:class:`~powermean.interface.sample.SampleOutcome` trees and lowers them into
:class:`~powermean.results.absorbed_trace.RetainedOutcomeSnapshot` records while
preserving shared and distinct original diagnostic problem identities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from powermean._prepare.compose import PreparedFrame
from powermean.data.retained import OrderSample
from powermean.data.sample_fe_state import FESample
from powermean.interface.retained_context import (
    PerFitPreparationContext,
    _AdmissionMaps,
    authenticate_unscoped_frame,
)
from powermean.interface.sample import (
    SampleOutcome,
    SampleRequest,
    same_ledger,
)
from powermean.interface.sample_records import (
    CollinearityInvocation,
    RecessionInvocation,
    Round,
    SupportInvocation,
)
from powermean.results.absorbed_outcome_records import (
    RetainedCollinearityInvocation,
    RetainedOutcomeSnapshot,
    RetainedRecessionInvocation,
    RetainedRound,
    RetainedSampleRequest,
    RetainedSupportInvocation,
    build_sample_request_snapshot,
)
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.retained_input_values import RetainedInputSnapshot
from powermean.results.retained_inputs import ProblemIdentityMap

if TYPE_CHECKING:
    from powermean.interface.numeric_execution import NumericExecution

__all__ = [
    "_lower_outcome_tree",
    "_lower_round",
    "_lower_sample_request",
    "_validate_outcome_diagnostics",
    "_validate_outcome_tree",
    "admit_sample_outcome",
]


def _validate_outcome_diagnostics(outcome: SampleOutcome) -> None:
    """Validate concrete round, recession, collinearity, and sample types on outcome."""
    for r in outcome.rounds:
        if type(r) is not Round:
            raise TypeError("every outcome round must be a Round")
        if r.support is not None and type(r.support) is not SupportInvocation:
            raise TypeError("round.support must be a SupportInvocation")
    if (
        outcome.recession is not None
        and type(outcome.recession) is not RecessionInvocation
    ):
        raise TypeError("outcome.recession must be a RecessionInvocation")
    if (
        outcome.collinearity is not None
        and type(outcome.collinearity) is not CollinearityInvocation
    ):
        raise TypeError("outcome.collinearity must be a CollinearityInvocation")
    if outcome.status == "complete" and (
        type(outcome.sample) is not FESample
        or not same_ledger(outcome.ledger, outcome.sample.ledger)
    ):
        raise ValueError("complete outcome must hold a matching FESample")


def _validate_outcome_tree(
    outcome: object,
    *,
    context: PerFitPreparationContext | None = None,
    prepared: PreparedFrame | None = None,
) -> SampleOutcome:
    """Validate concrete ``SampleOutcome`` types, ownership, and baseline."""
    if type(outcome) is not SampleOutcome:
        raise TypeError("outcome must be a SampleOutcome")
    if context is not None and context._operation is not None:
        context.authenticate_frame(
            outcome.prepared,
            sample=outcome.sample if outcome.status == "complete" else None,
        )
        return outcome
    if type(outcome.request) is not SampleRequest:
        raise TypeError("outcome.request must be a SampleRequest")
    if context is not None:
        if not context.owns_prepared(outcome.prepared):
            raise ValueError(
                "outcome.prepared is not the exact PreparedFrame owned by preparation"
            )
    elif prepared is not None:
        authenticate_unscoped_frame(prepared)
        if outcome.prepared is not prepared:
            raise ValueError(
                "outcome.prepared is not the exact PreparedFrame owned by preparation"
            )
    _validate_outcome_diagnostics(outcome)
    samp = outcome.sample if outcome.status == "complete" else None
    if context is not None:
        context.authenticate_frame(outcome.prepared, sample=samp)
    else:
        authenticate_unscoped_frame(outcome.prepared, sample=samp)
    if outcome.origin is not None:
        if outcome.origin.prepared is not outcome.prepared:
            raise ValueError("outcome.origin.prepared must match outcome.prepared")
        _validate_outcome_tree(outcome.origin, context=context, prepared=prepared)
    return outcome


def _lower_sample_request(request: SampleRequest) -> RetainedSampleRequest:
    """Lower a concrete ``SampleRequest`` into a :class:`RetainedSampleRequest`."""
    if type(request) is not SampleRequest:
        raise TypeError("request must be a SampleRequest")
    return build_sample_request_snapshot(
        omega=request.omega,
        positive_only=request.positive_only,
        variance=request.variance,
        keep_singletons=request.keep_singletons,
        support=request.support,
        recession=request.recession,
        collinearity=request.collinearity,
        mode=request.mode,
    )


def _lower_round(
    round_rec: Round,
    *,
    problem_map: ProblemIdentityMap,
    partition_map: PartitionIdentityMap,
) -> RetainedRound:
    """Lower a concrete coordinator ``Round`` into a :class:`RetainedRound`."""
    sup = (
        None
        if round_rec.support is None
        else RetainedSupportInvocation(
            problem=round_rec.support.problem,
            controls=round_rec.support.controls,
            result=round_rec.support.result,
            removed=round_rec.support.removed,
            problem_map=problem_map,
            partition_map=partition_map,
        )
    )
    return RetainedRound(closure=round_rec.closure, support=sup)


def _lower_outcome_tree(
    outcome: SampleOutcome,
    *,
    maps: _AdmissionMaps,
    input_snapshot: RetainedInputSnapshot | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
    owned_order: OrderSample | None = None,
) -> RetainedOutcomeSnapshot:
    """Lower a validated ``SampleOutcome`` chain into detached snapshots."""
    oid = id(outcome)
    cached = maps.outcome_memo.get(oid)
    if cached is not None:
        return cached
    in_snap = (
        maps.input_map.register_snapshot(input_snapshot, prepared=outcome.prepared)
        if input_snapshot is not None
        else maps.input_map.register(outcome.prepared)
    )
    orig_snap = (
        None
        if outcome.origin is None
        else _lower_outcome_tree(
            outcome.origin,
            maps=maps,
            input_snapshot=in_snap,
        )
    )
    rounds = tuple(
        _lower_round(r, problem_map=maps.problem_map, partition_map=maps.partition_map)
        for r in outcome.rounds
    )
    rec = (
        None
        if outcome.recession is None
        else RetainedRecessionInvocation(
            problem=outcome.recession.problem,
            controls=outcome.recession.controls,
            result=outcome.recession.result,
            problem_map=maps.problem_map,
            partition_map=maps.partition_map,
        )
    )
    coll = (
        None
        if outcome.collinearity is None
        else RetainedCollinearityInvocation(
            problem=outcome.collinearity.problem,
            controls=outcome.collinearity.controls,
            mode=outcome.collinearity.mode,
            analysis=outcome.collinearity.analysis,
            problem_map=maps.problem_map,
            partition_map=maps.partition_map,
        )
    )
    samp_snap = sample_snapshot
    if outcome.sample is None:
        samp_snap = None
    elif samp_snap is None:
        samp_snap = (
            maps.sample_map.register(
                outcome.sample,
                input_snapshot=in_snap,
                partition_map=maps.partition_map,
            )
            if owned_order is None
            else maps.sample_map._capture_owned_order(
                outcome.sample, owned_order,
                input_snapshot=in_snap,
                partition_map=maps.partition_map,
            )
        )
    snap = RetainedOutcomeSnapshot(
        input_snapshot=in_snap,
        request=_lower_sample_request(outcome.request),
        status=outcome.status,
        _ledger=outcome.ledger,
        rounds=rounds,
        recession=rec,
        collinearity=coll,
        sample=samp_snap,
        origin=orig_snap,
    )
    maps.outcome_memo[oid] = snap
    maps._outcome_refs.append(outcome)
    return snap


def _complete_numeric_outcome(
    snapshot: RetainedOutcomeSnapshot,
    details: tuple[SampleOutcome, _AdmissionMaps],
) -> None:
    """Export diagnostics only when requested, with the SAME identity maps.

    The completed private coordinator owns every original diagnostic. Numeric
    solving and finalization never need to lower these into a second problem
    graph. Foreign/refused/filtered routes keep the original eager admission.
    """
    outcome, maps = details
    rounds = tuple(_lower_round(
        row, problem_map=maps.problem_map, partition_map=maps.partition_map
    ) for row in outcome.rounds)
    rec = outcome.recession
    recession = None if rec is None else RetainedRecessionInvocation(
        problem=rec.problem, controls=rec.controls, result=rec.result,
        problem_map=maps.problem_map, partition_map=maps.partition_map,
    )
    col = outcome.collinearity
    collinearity = None if col is None else RetainedCollinearityInvocation(
        problem=col.problem, controls=col.controls, mode=col.mode,
        analysis=col.analysis, problem_map=maps.problem_map,
        partition_map=maps.partition_map,
    )
    origin = None if outcome.origin is None else _lower_outcome_tree(
        outcome.origin, maps=maps, input_snapshot=snapshot.input_snapshot,
    )
    for name, value in (
        ("rounds", rounds), ("recession", recession),
        ("collinearity", collinearity), ("origin", origin),
    ):
        object.__setattr__(snapshot, name, value)
    vars(snapshot).pop("_numeric_details", None)


def admit_sample_outcome(
    outcome: object,
    *,
    context: PerFitPreparationContext | None = None,
    prepared: PreparedFrame | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
    maps: _AdmissionMaps | None = None,
) -> RetainedOutcomeSnapshot:
    """Admit a concrete ``SampleOutcome`` and lower it into a detached snapshot."""
    return _admit_sample_outcome(
        outcome, context=context, prepared=prepared,
        sample_snapshot=sample_snapshot, maps=maps,
    )


def _admit_owned_sample_outcome(
    outcome: SampleOutcome, *, context: PerFitPreparationContext,
    order_sample: OrderSample, execution: NumericExecution | None = None,
) -> RetainedOutcomeSnapshot:
    """Consume the fresh private order producer, never a public supplied digest.

    Only the unexposed normal operation produced both records. Unknown/public
    preparation routes keep ordinary fresh capture. The snapshot itself checks
    raw-data identity: any final row change needs a new canonical fingerprint.
    """
    operation = context._operation
    if (execution is not None and operation is not None
            and execution.preparation.context is context
            and execution._sample is not None
            and execution._sample is outcome.sample
            and operation.numeric is execution.numeric):
        sample = outcome.sample
        assert sample is not None
        maps = operation.maps
        source = context.authenticate_frame(outcome.prepared, sample=sample)
        maps.input_map.register_snapshot(source, prepared=outcome.prepared)
        detached = maps.sample_map._capture_numeric(
            sample, order_sample, execution.numeric,
            input_snapshot=source, partition_map=maps.partition_map,
        )
        result = RetainedOutcomeSnapshot._from_numeric(
            input_snapshot=source, request=_lower_sample_request(outcome.request),
            sample=detached, details=(outcome, maps, _complete_numeric_outcome),
        )
        maps.outcome_memo[id(outcome)] = result
        maps._outcome_refs.append(outcome)
        return result
    owned = (order_sample if context._operation is not None
             and context._normal_origin is not None else None)
    return _admit_sample_outcome(outcome, context=context, owned_order=owned)


def _admit_sample_outcome(
    outcome: object,
    *,
    context: PerFitPreparationContext | None = None,
    prepared: PreparedFrame | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
    maps: _AdmissionMaps | None = None,
    owned_order: OrderSample | None = None,
) -> RetainedOutcomeSnapshot:
    """Preserve admission/capture order for generic and private compositions alike."""
    checked = _validate_outcome_tree(outcome, context=context, prepared=prepared)
    operation_maps = (
        context._operation.maps
        if context is not None and context._operation is not None else None
    )
    resolved_maps = maps if maps is not None else (
        operation_maps if operation_maps is not None else _AdmissionMaps()
    )
    samp = checked.sample if checked.status == "complete" else None
    in_snap = (
        context.authenticate_frame(checked.prepared, sample=samp)
        if context is not None
        else authenticate_unscoped_frame(checked.prepared, sample=samp)
    )
    try:
        return _lower_outcome_tree(
            checked,
            maps=resolved_maps,
            input_snapshot=in_snap,
            sample_snapshot=sample_snapshot,
            owned_order=owned_order,
        )
    finally:
        if maps is None and operation_maps is None:
            resolved_maps.clear_anchors()
