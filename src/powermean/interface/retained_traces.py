"""Recorded trace and trigger record admission and lowering into detached snapshots.

This module lowers concrete
:class:`~powermean.interface.sample_records.RecordedEvent`,
:class:`~powermean.interface.sample_records.RecordedTrace`, and
:class:`~powermean.solve.profiled_trigger.TriggerRecord` objects into detached
``powermean.results`` snapshots, capturing every invoked, returned, and witness problem
from its own concrete original without substituting parent event problems and wiring
trigger points through the shared
:class:`~powermean.results.absorbed_points.PointIdentityMap`.
"""

from __future__ import annotations

from typing import cast

from powermean._prepare.compose import PreparedFrame
from powermean.data.collinearity import CollinearityAnalysis
from powermean.data.sample_fe_state import FESample
from powermean.interface.retained_context import (
    PerFitPreparationContext,
    _AdmissionMaps,
    authenticate_unscoped_frame,
)
from powermean.interface.retained_outcomes import (
    _lower_outcome_tree,
    _lower_sample_request,
    _validate_outcome_tree,
    admit_sample_outcome,
)
from powermean.interface.sample import SampleOutcome, SampleRequest
from powermean.interface.sample_records import (
    RawSupportResult,
    RecordedEvent,
    RecordedTrace,
)
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_trace import (
    DetachedDiagnosticProblem,
    DetachedDiagnosticResult,
    RetainedEventSnapshot,
    RetainedTraceSnapshot,
    RetainedTriggerSnapshot,
    snapshot_diagnostic_problem,
    snapshot_trigger,
)
from powermean.results.retained_collinearity import snapshot_collinearity_analysis
from powermean.results.retained_diagnostic_problems import (
    RetainedCollinearityProblem,
    RetainedGammaProblem,
    RetainedSupportProblem,
)
from powermean.results.retained_gamma import snapshot_gamma_result
from powermean.results.retained_inputs import ProblemIdentityMap
from powermean.results.retained_support import (
    RawSupportFields,
    snapshot_raw_support,
    snapshot_support_result,
)
from powermean.solve.profiled_trigger import TriggerRecord
from powermean.solve.support import SupportResult
from powermean.solve.support_gamma import GammaResult

__all__ = [
    "_lower_event_result",
    "_lower_recorded_event",
    "admit_recorded_trace",
    "admit_trigger_record",
]


def _lower_event_result(
    result: object,
    *,
    problem_snapshot: DetachedDiagnosticProblem | None = None,
    problem_map: ProblemIdentityMap,
    partition_map: PartitionIdentityMap,
) -> DetachedDiagnosticResult | None:
    """Lower a recorded diagnostic result or ``RawSupportResult`` without replay."""
    if result is None:
        return None
    if type(result) is SupportResult:
        return snapshot_support_result(
            result,
            problem_snapshot=cast("RetainedSupportProblem | None", problem_snapshot),
            problem_map=problem_map,
            partition_map=partition_map,
        )
    if type(result) is RawSupportResult:
        return snapshot_raw_support(
            RawSupportFields(
                problem=result.problem,
                controls=result.controls,
                method=result.method,
                fe=result.fe,
                lp=result.lp,
                ir=result.ir,
                witnesses=result.witnesses,
                dual=result.dual,
                completion=result.completion,
                reason=result.reason,
            ),
            problem_snapshot=cast("RetainedSupportProblem | None", problem_snapshot),
            problem_map=problem_map,
            partition_map=partition_map,
        )
    if type(result) is GammaResult:
        return snapshot_gamma_result(
            result,
            problem_snapshot=cast("RetainedGammaProblem | None", problem_snapshot),
            problem_map=problem_map,
            partition_map=partition_map,
        )
    if type(result) is CollinearityAnalysis:
        return snapshot_collinearity_analysis(
            result,
            problem_snapshot=cast(
                "RetainedCollinearityProblem | None", problem_snapshot
            ),
            problem_map=problem_map,
            partition_map=partition_map,
        )
    raise TypeError(f"unexpected recorded event result type {type(result)!r}")


def _lower_recorded_event(
    event: RecordedEvent,
    *,
    problem_map: ProblemIdentityMap,
    partition_map: PartitionIdentityMap,
) -> RetainedEventSnapshot:
    """Lower one concrete ``RecordedEvent`` into a :class:`RetainedEventSnapshot`.

    Detaches ``event.problem`` and ``event.result`` from their own concrete originals
    via ``problem_map`` without substituting ``event.problem`` for ``result.problem``.
    """
    if type(event) is not RecordedEvent:
        raise TypeError("every trace event must be a RecordedEvent")
    prob_snap = snapshot_diagnostic_problem(
        event.problem, problem_map=problem_map, partition_map=partition_map
    )
    res_snap = _lower_event_result(
        event.result,
        problem_snapshot=None,
        problem_map=problem_map,
        partition_map=partition_map,
    )
    return RetainedEventSnapshot(
        kind=event.kind,
        stage=event.stage,
        problem=prob_snap,
        controls=event.controls,
        result=res_snap,
        _closure=event.closure,
        _removed=event.removed,
        mode=event.mode,
        unowned=event.unowned,
    )


def admit_recorded_trace(
    trace: object,
    *,
    context: PerFitPreparationContext | None = None,
    prepared: PreparedFrame | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
    maps: _AdmissionMaps | None = None,
) -> RetainedTraceSnapshot:
    """Admit a ``RecordedTrace`` and lower it into ``RetainedTraceSnapshot``."""
    if type(trace) is not RecordedTrace:
        raise TypeError("trace must be a RecordedTrace")
    if type(trace.request) is not SampleRequest:
        raise TypeError("trace.request must be a SampleRequest")
    if type(trace.start) is not FESample:
        raise TypeError("trace.start must be an FESample")
    origin_out = _validate_outcome_tree(
        trace.origin, context=context, prepared=prepared
    )
    if origin_out.sample is not trace.start:
        raise ValueError("trace.start must be trace.origin.sample")
    final_out: SampleOutcome | None = None
    if trace.outcome is not None:
        final_out = _validate_outcome_tree(
            trace.outcome, context=context, prepared=prepared
        )
        if final_out.origin is not origin_out:
            raise ValueError("trace.outcome.origin must be trace.origin")

    resolved_maps = maps if maps is not None else _AdmissionMaps()
    in_snap = (
        context.authenticate_frame(origin_out.prepared, sample=trace.start)
        if context is not None
        else authenticate_unscoped_frame(origin_out.prepared, sample=trace.start)
    )
    try:
        orig_snap = _lower_outcome_tree(
            origin_out,
            maps=resolved_maps,
            input_snapshot=in_snap,
            sample_snapshot=sample_snapshot,
        )
        assert orig_snap.sample is not None
        events = tuple(
            _lower_recorded_event(
                ev,
                problem_map=resolved_maps.problem_map,
                partition_map=resolved_maps.partition_map,
            )
            for ev in trace.events
        )
        out_snap = (
            None
            if final_out is None
            else _lower_outcome_tree(
                final_out,
                maps=resolved_maps,
                input_snapshot=orig_snap.input_snapshot,
            )
        )
        fail_type = None if trace.failure is None else type(trace.failure).__name__
        fail_msg = None if trace.failure is None else str(trace.failure)
        return RetainedTraceSnapshot(
            origin=orig_snap,
            start=orig_snap.sample,
            request=_lower_sample_request(trace.request),
            events=events,
            outcome=out_snap,
            unfinished=trace.unfinished,
            failure_type=fail_type,
            failure_message=fail_msg,
        )
    finally:
        if maps is None:
            resolved_maps.clear_anchors()


def admit_trigger_record(
    record: object,
    *,
    context: PerFitPreparationContext | None = None,
    prepared: PreparedFrame | None = None,
    sample_snapshot: RetainedSampleSnapshot | None = None,
    maps: _AdmissionMaps | None = None,
) -> RetainedTriggerSnapshot:
    """Admit a ``TriggerRecord`` and lower it into ``RetainedTriggerSnapshot``."""
    if type(record) is not TriggerRecord:
        raise TypeError("record must be a concrete TriggerRecord")
    resolved_maps = maps if maps is not None else _AdmissionMaps()
    try:
        trace_snap = admit_recorded_trace(
            record.trace,
            context=context,
            prepared=prepared,
            sample_snapshot=sample_snapshot,
            maps=resolved_maps,
        )
        samp_snap = (
            sample_snapshot
            if sample_snapshot is not None
            else (
                trace_snap.start
                if record.request.sample is record.trace.start
                else resolved_maps.sample_map.register(
                    record.request.sample,
                    partition_map=resolved_maps.partition_map,
                )
            )
        )
        out_snap = (
            None
            if record.outcome is None
            else (
                trace_snap.outcome
                if record.outcome is record.trace.outcome
                else admit_sample_outcome(
                    record.outcome,
                    context=context,
                    prepared=prepared,
                    maps=resolved_maps,
                )
            )
        )
        return snapshot_trigger(
            record,
            sample_snapshot=samp_snap,
            trace_snapshot=trace_snap,
            outcome_snapshot=out_snap,
            problem_map=resolved_maps.problem_map,
            partition_map=resolved_maps.partition_map,
            point_map=resolved_maps.point_map,
        )
    finally:
        if maps is None:
            resolved_maps.clear_anchors()
