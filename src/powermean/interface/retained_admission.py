"""Concrete interface-layer admission and detachment for outcomes, traces, and fits.

This module coordinates interface-level admission across optional
preparation, ``SampleOutcome``, ``RecordedTrace``, ``TriggerRecord``, ``Estimator``,
and ``BoundDependence`` and re-exports context, outcome, and trace helpers from
:mod:`powermean.interface.retained_context`,
:mod:`powermean.interface.retained_outcomes`, and
:mod:`powermean.interface.retained_traces`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace

from powermean._prepare.compose import PreparedFrame
from powermean.data.sample_fe_source import Derived
from powermean.data.sample_fe_state import FESample, same_bits
from powermean.data.specs import Estimator
from powermean.interface.retained_binding import _check_dependence_against_parts
from powermean.interface.retained_context import (
    PerFitPreparationContext,
    _AdmissionMaps,
    _check_frame_matches_baseline,
    _IssuedFitFrame,
    authenticate_unscoped_frame,
    authenticate_unscoped_sample,
)
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_outcomes import (
    _lower_outcome_tree,
    _lower_round,
    _lower_sample_request,
    _validate_outcome_diagnostics,
    _validate_outcome_tree,
    admit_sample_outcome,
)
from powermean.interface.retained_traces import (
    _lower_event_result,
    _lower_recorded_event,
    admit_recorded_trace,
    admit_trigger_record,
)
from powermean.interface.sample import SampleOutcome
from powermean.results.absorbed_binding import (
    AdmittedAbsorbedGraph,
    _consume_fit_audit,
    admit_absorbed_graph,
)
from powermean.results.absorbed_groupings import GroupingIdentityMap
from powermean.results.absorbed_history import _candidate_inner_and_start
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_points import PointIdentityMap
from powermean.results.absorbed_retained import RetainedAdmissionReceipt
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    SampleIdentityMap,
)
from powermean.results.absorbed_sample_binding import (
    UNBOUND,
    check_prepared_matches_sample,
)
from powermean.results.absorbed_trace import RetainedTriggerSnapshot
from powermean.results.primitive_snapshot import PRIMITIVE, MetadataParts
from powermean.results.retained_input_values import (
    RetainedInputSnapshot,
    SampleKey,
    _same_column_values,
)
from powermean.solve.absorbed_certificate_history import CandidateHistory
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_root_records import AbsorbedRootCandidate
from powermean.solve.profiled_trigger import TriggerRecord

__all__ = [
    "AdmittedInterfaceGraph",
    "PerFitPreparationContext",
    "_AdmissionMaps",
    "_IssuedFitFrame",
    "_check_frame_matches_baseline",
    "_lower_event_result",
    "_lower_outcome_tree",
    "_lower_recorded_event",
    "_lower_round",
    "_lower_sample_request",
    "_validate_outcome_diagnostics",
    "_validate_outcome_tree",
    "admit_interface_graph",
    "admit_recorded_trace",
    "admit_sample_outcome",
    "admit_trigger_record",
    "authenticate_unscoped_frame",
    "authenticate_unscoped_sample",
    "check_dependence_matches_sample",
    "check_estimator_matches_candidate",
]


def check_estimator_matches_candidate(
    estimator: object, sample: FESample, history: CandidateHistory
) -> Estimator:
    """Validate that a concrete ``Estimator`` matches the candidate and sample."""
    if type(estimator) is not Estimator:
        raise TypeError("estimator must be a concrete Estimator")
    if estimator.mean.omega != sample.omega:
        raise ValueError("estimator.mean.omega does not match sample.omega")
    if history.route == "linear":
        if sample.omega != 0.0:
            raise ValueError("linear route requires omega == 0")
    elif estimator.variance != history.variance:
        raise ValueError("estimator.variance does not match candidate variance")
    return estimator


def _resolve_dependence_parts(
    sample: FESample,
    context: PerFitPreparationContext | None,
    prepared: PreparedFrame | None,
) -> MetadataParts | PreparedFrame:
    """Resolve and authenticate the frame parts for dependence comparison.

    An explicit ``prepared`` that the context did not issue (a caller's helper copy,
    or any frame without a context) takes the reference unscoped path on every route,
    and is read in the original validation order.  On a primitive snapshot route, an
    explicit ``prepared`` issued by the context, or the issued frame matching
    ``sample``, is identified and ``sample`` is validated afresh against the
    once-materialized parts of the immutable capture; the parts returned for
    comparison are the base, labels and dependence of those parts, never the live
    frame and never a rebuilt frame.  On the general route the authenticated live
    frame's parts are compared, as before.
    """
    if prepared is not None and not (
        context is not None
        and context.snapshot_route == PRIMITIVE
        and context.owns_prepared(prepared)
    ):
        authenticate_unscoped_frame(prepared, sample=sample)
        return prepared
    if context is not None:
        matched = context.matching_frame(sample) if prepared is None else prepared
        if context.snapshot_route == PRIMITIVE:
            if matched is not None:
                context.authenticate_frame(matched, sample=sample)
            return context._materialized_metadata()
        frame = context._owned_template() if matched is None else matched
        context.authenticate_frame(
            frame, sample=sample if matched is not None else None
        )
        return frame
    raise ValueError(UNBOUND)


def check_dependence_matches_sample(
    dependence: object,
    sample: FESample,
    *,
    context: PerFitPreparationContext | None = None,
    prepared: PreparedFrame | None = None,
    sample_key: SampleKey | None = None,
) -> BoundDependence:
    """Validate that ``dependence`` is bound to ``sample`` and the per-fit frame."""
    if type(dependence) is not BoundDependence:
        raise TypeError("dependence must be a concrete BoundDependence")
    if context is not None and context._operation is not None:
        # Produced by this operation's metadata binder: an internal handoff.
        context.authenticate_sample(sample)
        return dependence
    raw_w = sample.base.inputs.weights
    exp_kind = None if raw_w is None else raw_w.kind
    exp_freqs = (
        tuple(int(v) for v in raw_w.values[sample.positions].tolist())
        if exp_kind == "frequency" and raw_w is not None
        else None
    )
    if (
        dependence.nobs != sample.ledger.nobs
        or dependence.count != sample.count
        or not same_bits(dependence.positions, sample.positions)
        or dependence.weight_kind != exp_kind
        or dependence.frequency_counts != exp_freqs
    ):
        raise ValueError(UNBOUND)
    for name, col in sample.metadata.columns.items():
        meta = dependence.columns.get(name)
        if (
            meta is None
            or meta.name != name
            or not _same_column_values(meta.values, col.values)
        ):
            raise ValueError(UNBOUND)
    parts = _resolve_dependence_parts(sample, context, prepared)
    _check_dependence_against_parts(
        dependence,
        sample,
        parts,
        hac_request=None if context is None else context.hac_request,
        conley_request=None if context is None else context.conley_request,
        input_key=None if context is None else context.input_key,
        sample_key=sample_key,
    )
    return dependence


@dataclass(frozen=True)
class AdmittedInterfaceGraph:
    """Admitted lower graph and detached interface snapshots sharing one key space.

    ``trigger_snapshot`` is the admitted trigger record of the fit: for a
    selected-root fit, whose root stages call no trigger, its convex initializer's.
    """

    admitted_graph: AdmittedAbsorbedGraph
    input_snapshot: RetainedInputSnapshot | None
    sample_snapshot: RetainedSampleSnapshot
    outcome_snapshot: RetainedOutcomeSnapshot | None
    trigger_snapshot: RetainedTriggerSnapshot | None
    dependence: BoundDependence | None
    point_map: PointIdentityMap | None = None
    sample_map: SampleIdentityMap | None = None
    grouping_map: GroupingIdentityMap | None = None

    @property
    def receipt(self) -> RetainedAdmissionReceipt:
        """Return the admission receipt."""
        return self.admitted_graph.receipt


def _authenticate_prepared_links(prep_frame: PreparedFrame, sample: FESample) -> None:
    """Verify original object links between ``prep_frame`` and ``sample``."""
    msg = (
        "outcome.prepared is not the exact PreparedFrame owned by preparation "
        f"({UNBOUND})"
    )
    if prep_frame.base is not sample.base:
        raise ValueError(msg)
    if isinstance(sample.source, Derived):
        if prep_frame.effects is None:
            raise ValueError(UNBOUND)
        src_cols = sample.source.source.columns
        eff_cols = prep_frame.effects.columns
        if set(src_cols) != set(eff_cols) or any(
            src_cols[key] is not eff_cols[key] for key in eff_cols
        ):
            raise ValueError(msg)


def _resolve_interface_preparation(
    preparation: object,
    outcome: object,
    sample: FESample,
) -> tuple[
    PerFitPreparationContext | None,
    PreparedFrame | None,
    SampleOutcome | None,
    RetainedInputSnapshot | None,
]:
    """Validate optional preparation and outcome originals against ``sample``."""
    if (type(preparation) is PerFitPreparationContext
            and preparation._operation is not None):
        frame, snapshot = preparation.authenticate_sample(sample)
        if outcome is not None and (
            outcome.prepared is not frame or outcome.status != "complete"
            or outcome.sample is not sample
        ):
            raise ValueError(UNBOUND)
        return preparation, frame, outcome, snapshot
    context: PerFitPreparationContext | None = None
    prep_frame: PreparedFrame | None = None
    if preparation is not None:
        if type(preparation) is PerFitPreparationContext:
            context = preparation
        elif type(preparation) is PreparedFrame:
            prep_frame = preparation
        else:
            raise TypeError(
                "preparation must be a RetainedPreparation or PreparedFrame"
            )

    checked_outcome: SampleOutcome | None = None
    if outcome is not None:
        if type(outcome) is not SampleOutcome:
            raise TypeError("outcome must be a concrete SampleOutcome")
        if outcome.status != "complete" or outcome.sample is not sample:
            raise ValueError("outcome.sample is not the candidate's FESample")
        checked_outcome = _validate_outcome_tree(
            outcome, context=context, prepared=prep_frame
        )
        if prep_frame is None:
            prep_frame = checked_outcome.prepared
    elif context is not None:
        prep_frame, _ = context.authenticate_sample(sample)

    in_snap: RetainedInputSnapshot | None = None
    if prep_frame is not None:
        in_snap = _authenticate_resolved_frame(context, prep_frame, sample)
    else:
        authenticate_unscoped_sample(sample)

    return context, prep_frame, checked_outcome, in_snap


def _authenticate_resolved_frame(
    context: PerFitPreparationContext | None,
    prep_frame: PreparedFrame,
    sample: FESample,
) -> RetainedInputSnapshot | None:
    """Authenticate the resolved ``prep_frame`` and ``sample`` at this entry.

    General route (and no context): the live frame is what consumers read, so it is
    compared against its evidence and then against the sample here as well.  Primitive
    route: consumers read the once-materialized parts of the capture, which
    ``authenticate_frame`` validated the sample against; the live frame is identified,
    not re-read.
    """
    _authenticate_prepared_links(prep_frame, sample)
    if context is None:
        in_snap = authenticate_unscoped_frame(prep_frame, sample=sample)
        check_prepared_matches_sample(prep_frame, sample)
        return in_snap
    in_snap = context.authenticate_frame(prep_frame, sample=sample)
    if context.snapshot_route != PRIMITIVE:
        check_prepared_matches_sample(prep_frame, sample)
    return in_snap


def _bind_candidate_point_context(
    maps: _AdmissionMaps,
    lower: AdmittedAbsorbedGraph,
    samp_snap: RetainedSampleSnapshot,
) -> None:
    """Bind candidate sample, variance, and inner controls onto ``maps.point_map``."""
    hist = lower.audit.history
    inner_ctrl, _ = _candidate_inner_and_start(
        hist.candidate,
        hist.route,
        grouping_snapshot=samp_snap._final_grouping,
        grouping_map=maps.grouping_map,
    )
    maps.point_map.bind_context(
        sample_snapshot=samp_snap,
        variance=hist.variance,
        inner=inner_ctrl,
    )


def admit_interface_graph(
    audit: object,
    *,
    preparation: PerFitPreparationContext | PreparedFrame | None = None,
    outcome: SampleOutcome | None = None,
    dependence: BoundDependence | None = None,
    estimator: Estimator | None = None,
    request: CovarianceRequest | None = None,
    keep_influence: bool = True,
) -> AdmittedInterfaceGraph:
    """Admit concrete interface and lower objects and detach shared snapshots.

    A call-local owned operation consumes its own scientific audit and reuses its
    existing snapshot maps. No caller-supplied graph uses that execution path.
    Otherwise this function is one external entry, which
    :meth:`PerFitPreparationContext._primitive_entry` marks: on a context's primitive
    snapshot route every authentication within it validates the live sample it
    presents afresh against the immutable capture.
    """
    operation = (preparation._operation
                 if type(preparation) is PerFitPreparationContext else None)
    lower = (
        _consume_fit_audit(
            audit, request=request, keep_influence=keep_influence,
            sample_map=(operation.maps.sample_map
                        if operation.numeric is not None else None),
        )
        if operation is not None else
        admit_absorbed_graph(audit, request=request, keep_influence=keep_influence)
    )
    with (
        preparation._primitive_entry()
        if type(preparation) is PerFitPreparationContext
        else nullcontext()
    ):
        return _admit_interface_graph(
            lower,
            preparation=preparation,
            outcome=outcome,
            dependence=dependence,
            estimator=estimator,
        )


def _admit_interface_graph(
    lower: AdmittedAbsorbedGraph,
    *,
    preparation: PerFitPreparationContext | PreparedFrame | None,
    outcome: SampleOutcome | None,
    dependence: BoundDependence | None,
    estimator: Estimator | None,
) -> AdmittedInterfaceGraph:
    """Admit the interface objects of ``lower`` and detach shared snapshots."""
    sample = lower.sample
    context, prep_frame, checked_outcome, auth_snap = _resolve_interface_preparation(
        preparation, outcome, sample
    )
    if estimator is not None:
        check_estimator_matches_candidate(estimator, sample, lower.audit.history)
    updated_receipt = replace(
        lower.receipt,
        same_preparation=prep_frame is not None,
        same_outcome_sample=checked_outcome is not None,
    )
    lower = replace(lower, receipt=updated_receipt)

    maps = (context._operation.maps
            if context is not None and context._operation is not None
            else _AdmissionMaps())
    in_snap: RetainedInputSnapshot | None = None
    if auth_snap is not None:
        in_snap = maps.input_map.register_snapshot(
            auth_snap, base=sample.base, prepared=prep_frame
        )
    elif prep_frame is not None:
        in_snap = maps.input_map.register(prep_frame)

    samp_snap = maps.sample_map.register(
        sample, input_snapshot=in_snap, partition_map=maps.partition_map
    )
    _bind_candidate_point_context(maps, lower, samp_snap)

    bound_dep: BoundDependence | None = None
    if dependence is not None:
        checked_dep = check_dependence_matches_sample(
            dependence,
            sample,
            context=context,
            prepared=prep_frame,
            sample_key=samp_snap.key,
        )
        bound_dep = checked_dep.with_keys(
            input_key=None if in_snap is None else in_snap.key,
            sample_key=samp_snap.key,
        )

    out_snap = (
        None
        if checked_outcome is None
        else _lower_outcome_tree(
            checked_outcome,
            maps=maps,
            input_snapshot=in_snap,
            sample_snapshot=samp_snap,
        )
    )
    trig = _fit_trigger(lower.audit.history)
    trig_snap = (
        None
        if trig is None
        else admit_trigger_record(
            trig,
            context=context,
            prepared=prep_frame,
            sample_snapshot=samp_snap,
            maps=maps,
        )
    )
    return AdmittedInterfaceGraph(
        admitted_graph=lower,
        input_snapshot=in_snap,
        sample_snapshot=samp_snap,
        outcome_snapshot=out_snap,
        trigger_snapshot=trig_snap,
        dependence=bound_dep,
        point_map=maps.point_map,
        sample_map=maps.sample_map,
        grouping_map=maps.grouping_map,
    )


def _fit_trigger(history: CandidateHistory) -> TriggerRecord | None:
    """The trigger record of the fit: a selected-root fit's convex initializer's."""
    candidate = history.candidate
    if type(candidate) is not AbsorbedRootCandidate:
        return history.trigger
    audit = candidate.initializer.audit
    return None if audit is None else audit.history.trigger
