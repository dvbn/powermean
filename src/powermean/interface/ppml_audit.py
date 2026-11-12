"""The internal joint-PPML entry with an explicit standard or full audit mode.

Nothing in the package imports this module and no public name exports it: it is reached
by an explicit import of :func:`solve_ppml_private` only, and nothing activates it.

Eligibility, refused before any solve: ``audit_mode`` exactly ``"standard"`` or
``"full"`` (no default), Boolean ``keep_influence`` and ``validate`` (``validate`` in
standard mode only); a ``RetainedPreparation``; a complete ``SampleOutcome`` of that
preparation whose ``FESample`` has fixed-effect blocks; exact ``PowerVariance`` with
``tau == 1.0`` and order exactly ``1.0`` (no tolerance, no NB2 alias); exact
``SolverOptions`` and ``AbsorbedOptions``; the built-in ``AbsorbedScale`` of an exact
robust or one-cluster ``CovarianceRequest`` whose codes are bound to the sample's final
rows; the fit's sample, family, absorbed options, request, controls and native gradient
control ``gradient_tol`` capturable and materialized exactly by the closed capture
(:mod:`powermean.results.absorbed_assessment_capture`).  ``TypeError`` for a record of
another type, ``SpecificationError`` (``specification.invalid_value``) otherwise.  The
Newton provider keeps its own admission, inside the solver.  The trigger is the entry's
own (:class:`_Coordinator`: the recorded recomposition, ``SampleRecorder`` and
``reprepare_recorded``), never a caller's callback; the admission of its outcome is the
coordinator's own ``admitted_outcome``.

Full mode, ``policy="full_source229"``: the unchanged composition of the
within-fit handoff, ``AssessmentHandoff`` as the solver's assessor, its audit (fresh
history admission, the selected point's owned or fresh certificate), then
``retain_absorbed`` of an ``AbsorbedHandoff``.  The result is labelled, never judged by
a standard predicate: ``standard_status="not_run"``, the audit status from the
certificate's level.

Standard mode, ``policy="ppml_standard_v1"``: the same solver and retry sequence with
the :class:`~powermean.results.ppml_standard_assessment.StandardAssessor`; the selection
and standard convergence; the requested inference at the converged selected point
(:mod:`powermean.results.ppml_standard_inference`); with ``validate=True``, before the
retention, the separately timed unchanged ``certify_point`` of an owned same-point
capture; the binding, narrow admission and detachment of
:mod:`powermean.results.ppml_standard_retained`, with the interface admission
of the preparation, outcome, dependence, estimator and trigger.  On the index route
the entry's exact ``StandardAssessor`` is registered as the index solver's private
pair, the assessor itself and its opening-retry callback
(``newton_absorbed._OpeningDispatch``),
and the fit's records carry ``policy="ppml_standard_index_opening_v2"`` while its
complete readings keep the point tag ``ppml_standard_v1``.  The profiled route, full
mode and every other assessor are passed to the solver unchanged.

The equation-aware stop.  On that same index route
in standard mode, and nowhere else, the pair is registered inside the exact private
adapter ``newton_absorbed._EquationStop`` with the entry's own probe log
(``newton_absorbed.ProbeLog``), one per call: every attempt of the solve, a retry's and
``retry=0``'s included, then stops natively only when the two native formulas and the
fresh fixed-effect equations pass (:mod:`powermean.solve.newton_absorbed`).  Whether the
stop ran is the log's activity, never its length; its receipt must be that of the
solve's own candidate, with the entry's captured ``tol`` and ``gradient_tol``, and its
probes are bound and detached into the envelope (``ppml_standard_retained
.detach_probes``), refused fits' included, and counted apart in the work.  The fit, its
selection, envelope and provenance then carry
``policy="ppml_standard_index_equation_stop_v3"``; complete readings keep
``ppml_standard_v1`` and opening-retry readings their partial tag.  The selection, the
standard convergence, the requested inference and the retention are unchanged.

The explicit residualized alternative.
:func:`solve_ppml_residualized_private` takes the same eight arguments
as :func:`solve_ppml_private` and fixes the requested inference method to
``residualized``; it refuses ``audit_mode`` other than ``"standard"`` before any solve,
and shares the admission (``_admitted`` with its private ``method`` keyword, carried in
``_Entry``), the coordinator, the solver, the assessor and the selection unchanged.
Only after the selection does the standard inference dispatch on the method:
``full_joint`` (the reference and default) runs
:func:`~powermean.results.ppml_standard_inference.standard_inference`,
``residualized`` runs
:func:`~powermean.results.ppml_residualized_inference.residualized_inference`.  No
method is selected automatically, and no inference falls back to, refits in or retries
the other method.  The provenance names the method; ``validate=True`` keeps its
separate same-point full audit after the inference, which nothing reads.

Statuses: ``standard_status`` (converged, refused, unavailable, not_run),
``inference_status`` (available, degenerate, refused, unavailable, not_applicable) and
``audit_status`` (not_run, verified, verified_secondary, failed, unavailable) are kept
apart; ``status`` is ``standard_converged``, ``standard_degenerate_inference``,
``standard_inference_refused``, ``standard_refused`` or ``standard_unavailable`` in
standard mode and ``full_<audit_status>`` in full mode, all private.
``standard_accepted`` and ``full_audit_accepted`` are ``None`` when not run.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from powermean.data.inference_requests import ClusterRequest
from powermean.data.sample_fe_state import FESample
from powermean.data.specs import Estimator, PowerMean
from powermean.errors import SpecificationError
from powermean.inference.absorbed import AbsorbedScale
from powermean.interface.retained_admission import (
    _resolve_interface_preparation,
    check_dependence_matches_sample,
    check_estimator_matches_candidate,
)
from powermean.interface.retained_context import _AdmissionMaps
from powermean.interface.retained_covariance import (
    PostFitCovarianceSpec,
    _request_label,
)
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_outcomes import _lower_outcome_tree
from powermean.interface.retained_postfit import (
    AbsorbedHandoff,
    RetainedAbsorbedResult,
    retain_absorbed,
)
from powermean.interface.retained_preparation import RetainedPreparation
from powermean.interface.retained_traces import admit_trigger_record
from powermean.interface.retention import bind_sample_metadata
from powermean.interface.sample import (
    SampleOutcome,
    admitted_outcome,
    reprepare_recorded,
)
from powermean.interface.sample_records import SampleRecorder
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.absorbed_assessment_binding import Anchors
from powermean.results.absorbed_assessment_capture import Replica, capture, replicate
from powermean.results.absorbed_assessment_handoff import FIT, AssessmentHandoff
from powermean.results.absorbed_handoff import check_cluster_request_rows
from powermean.results.absorbed_history import (
    RetainedHistorySnapshot,
    _candidate_inner_and_start,
    snapshot_candidate_history,
)
from powermean.results.absorbed_owned import RetainedCertificateSnapshot
from powermean.results.absorbed_point_records import RetainedPointSnapshot
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_scale_cache import RetainedInitialScaleCache
from powermean.results.ppml_residualized_inference import residualized_inference
from powermean.results.ppml_standard_assessment import (
    StandardAssessor,
    StandardSelection,
    standard_anchors,
)
from powermean.results.ppml_standard_inference import (
    NO_SLOPES,
    NOT_CONVERGED,
    same_point_audit,
    standard_error_status,
    standard_inference,
)
from powermean.results.ppml_standard_records import (
    FULL_POLICY,
    METHODS,
    OMITTED_CHECKS,
    AuditStatus,
    InferenceMethod,
    InferenceStatus,
    OmittedCheck,
    Policy,
    PPMLProvenance,
    PPMLWork,
    RetainedSamePointAudit,
    RetainedStandardEvidence,
    RetainedStandardInference,
    RetainedStandardSelection,
    RetainedStopProbes,
    StandardFailure,
    StandardInference,
    StandardStatus,
    fit_policy,
)
from powermean.results.ppml_standard_retained import (
    BoundStandard,
    StandardRetainedEnvelope,
    admit_standard,
    bound_standard,
    compose_standard_envelope,
    detach_probes,
)
from powermean.solve import newton_absorbed, newton_profiled
from powermean.solve.absorbed_certificate import Level
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.assessment import CertificateControls, certificate_controls
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.newton_trace import Timing
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_trigger import TraceView, TriggerRequest
from powermean.systems.profiled import PointProfile

AuditMode = Literal["standard", "full"]
MODES: tuple[AuditMode, ...] = ("standard", "full")
CALLER = "powermean.interface.ppml_audit.solve_ppml_private"
RESIDUALIZED_CALLER = "powermean.interface.ppml_audit.solve_ppml_residualized_private"
OWNERSHIP = "closed capture replica (results.absorbed_assessment_capture)"
UNBOUND = "the standard evidence does not bind to the admitted records"


def _invalid(argument: str, value: object, valid: str) -> SpecificationError:
    """The refusal ``specification.invalid_value`` of one argument."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": argument, "value": value, "valid": valid},
    )


# ---------------------------------------------------------------- the entry


@dataclass(frozen=True, eq=False)
class _Entry:
    """The admitted inputs of one call, read once; ``fit`` the entry's replica.

    ``probes`` is the call's own log of the private equation-aware stop, which only the
    standard index route registers and its one solve starts (module docstring);
    ``method`` the requested inference method, ``full_joint`` but at the alternative
    entry.
    """

    mode: AuditMode
    preparation: RetainedPreparation
    outcome: SampleOutcome
    sample: FESample
    variance: PowerVariance
    options: SolverOptions
    absorbed: AbsorbedOptions
    scale: AbsorbedScale
    request: CovarianceRequest
    fit: Replica
    keep_influence: bool
    validate: bool
    probes: newton_absorbed.ProbeLog
    method: InferenceMethod = "full_joint"


def _mode(audit_mode: object, keep_influence: object, validate: object) -> AuditMode:
    """The requested mode; its flags are Booleans, ``validate`` standard only."""
    if type(audit_mode) is not str or audit_mode not in MODES:
        raise _invalid("audit_mode", audit_mode, "'standard' or 'full'")
    for name, flag in (("keep_influence", keep_influence), ("validate", validate)):
        if type(flag) is not bool:
            raise _invalid(name, flag, "True or False")
    if validate and audit_mode == "full":
        raise _invalid("validate", validate, "False in full mode, which audits")
    return audit_mode


def _method(method: object, mode: AuditMode) -> InferenceMethod:
    """The requested inference method; the residualized one in standard mode only."""
    if type(method) is not str or method not in METHODS:
        raise _invalid("method", method, "'full_joint' or 'residualized'")
    if method == "residualized" and mode != "standard":
        valid = "'standard' for the residualized inference"
        raise _invalid("audit_mode", mode, valid)
    return method


def _records(
    preparation: object,
    outcome: object,
    options: object,
    absorbed: object,
    scale: object,
) -> tuple[FESample, CovarianceRequest]:
    """The exact records of the entry, the outcome's sample and the request."""
    if type(preparation) is not RetainedPreparation:
        raise TypeError("preparation must be a RetainedPreparation")
    if type(outcome) is not SampleOutcome:
        raise TypeError("outcome must be a SampleOutcome")
    if type(options) is not SolverOptions or type(absorbed) is not AbsorbedOptions:
        raise TypeError("options must be a SolverOptions and an AbsorbedOptions")
    request = getattr(scale, "request", None) if type(scale) is AbsorbedScale else None
    if type(request) is not CovarianceRequest:
        name = type(scale).__name__
        raise _invalid("scale", name, "the built-in AbsorbedScale of a request")
    sample = outcome.sample
    if outcome.status != "complete" or type(sample) is not FESample:
        raise _invalid("outcome", outcome.status, "a complete outcome")
    if sample.grouping is None:
        raise _invalid("outcome", "no fixed effects", "a sample with blocks")
    if not preparation.owns_prepared(outcome.prepared):
        raise _invalid("outcome", "a foreign outcome", "an outcome of the preparation")
    return sample, request


def _family(outcome: SampleOutcome, sample: FESample) -> PowerVariance:
    """The exact family of PPML at the order one, from the outcome's own request."""
    variance = outcome.request.variance
    if type(variance) is not PowerVariance or variance.tau != 1.0:
        raise _invalid("variance", repr(variance), "PowerVariance(tau=1.0), PPML")
    if outcome.request.omega != 1.0 or sample.omega != 1.0:
        raise _invalid("omega", sample.omega, "exactly 1.0, joint PPML")
    return variance


def _bound_rows(request: CovarianceRequest, sample: FESample) -> None:
    """A cluster request's codes are those of the sample's final rows."""
    if request.kind != "cluster":
        return
    codes, positions = request.codes, request.positions
    if (
        codes is None
        or positions is None
        or codes.shape != (sample.count,)
        or not np.array_equal(positions, sample.positions)
    ):
        raise _invalid("scale", request.name, "cluster codes of the final rows")


def _admitted(
    preparation: object,
    outcome: object,
    options: object,
    absorbed: object,
    scale: object,
    *,
    flags: tuple[object, object, object],
    method: InferenceMethod = "full_joint",
) -> _Entry:
    """The admitted entry (module docstring), or its refusal before any solve.

    ``method`` is the private requested inference method, carried in the entry.
    """
    mode = _mode(*flags)
    requested = _method(method, mode)
    sample, request = _records(preparation, outcome, options, absorbed, scale)
    found = cast("SampleOutcome", outcome)
    variance = _family(found, sample)
    _bound_rows(request, sample)
    solver, inner = cast("SolverOptions", options), cast("AbsorbedOptions", absorbed)
    roots = (sample, variance, inner, request, certificate_controls(solver))
    captured = capture((*roots, solver.gradient_tol))
    fit = None if captured is None else replicate(captured)
    if fit is None:
        raise _invalid("outcome", "inputs outside the closed capture", "capturable")
    return _Entry(
        mode,
        cast("RetainedPreparation", preparation),
        found,
        sample,
        variance,
        solver,
        inner,
        cast("AbsorbedScale", scale),
        request,
        fit,
        cast("bool", flags[1]),
        cast("bool", flags[2]),
        newton_absorbed.ProbeLog(),
        requested,
    )


class _Coordinator:
    """The entry's own trigger: one recorded recomposition of the origin (docstring)."""

    def __init__(self, origin: SampleOutcome) -> None:
        self._origin = origin

    def __call__(self, request: TriggerRequest, /) -> TraceView:
        """The frozen trace of the call, a propagated failure included."""
        recorder = SampleRecorder(self._origin)
        try:
            reprepare_recorded(
                self._origin,
                support=request.support,
                recession=request.recession,
                recorder=recorder,
            )
        except Exception as error:  # the trace keeps it: the Trigger protocol's record
            return recorder.failed(error)
        return recorder.snapshot()


def _solve(
    entry: _Entry, assess: StandardAssessor | AssessmentHandoff
) -> ProfiledCandidate | IndexCandidate:
    """The organization's unchanged solver with the entry's trigger and assessor.

    On the index route in standard mode the exact ``StandardAssessor`` is registered as
    the solver's private pair with its opening callback, held by the private
    equation-aware stop with the entry's own probe log (module docstring).
    """
    trigger, parts = _Coordinator(entry.outcome), (entry.sample, entry.variance)
    if entry.absorbed.absorbed == "profiled":
        return newton_profiled.solve(
            *parts,
            entry.options,
            entry.absorbed,
            trigger=trigger,
            admit=admitted_outcome,
            assess=assess,
        )
    index: newton_absorbed.IndexAssessor = assess
    if entry.mode == "standard" and type(assess) is StandardAssessor:
        pair = newton_absorbed._OpeningDispatch(assess, assess.opening)
        index = newton_absorbed._EquationStop(pair, entry.probes)
    return newton_absorbed.solve(
        *parts,
        entry.options,
        entry.absorbed,
        trigger=trigger,
        admit=admitted_outcome,
        assess=index,
    )


def _probes(
    entry: _Entry, candidate: ProfiledCandidate | IndexCandidate
) -> RetainedStopProbes | None:
    """The bound, detached probes of the fit's equation-aware stop, else ``None``.

    The log's activity, not its length, says that the stop ran; its receipt must be
    that of the solve of ``candidate``, and its bounds the entry's captured ``tol`` and
    ``gradient_tol`` (``ValueError`` otherwise).
    """
    log = entry.probes
    if not log.active:
        return None
    controls, gradient_tol = entry.fit.roots[FIT - 1], entry.fit.roots[FIT]
    captured = (cast("CertificateControls", controls).tol, cast("float", gradient_tol))
    return detach_probes(log.receipt(candidate), candidate, captured)


def _estimator(variance: PowerVariance) -> Estimator:
    """The estimator of joint PPML."""
    return Estimator(mean=PowerMean(omega=1.0), variance=variance)


def _dependence(entry: _Entry) -> BoundDependence:
    """The retained dependence metadata of the sample, bound by the accepted route."""
    sample = entry.sample
    return bind_sample_metadata(entry.preparation, sample, weights=sample.data.weights)


def _provenance(
    entry: _Entry,
    candidate: ProfiledCandidate | IndexCandidate,
    consumed: tuple[bool, str],
    policy: Policy,
) -> PPMLProvenance:
    """What ran, for either mode, with the policy of the fit (module docstring)."""
    caller = CALLER if entry.method == "full_joint" else RESIDUALIZED_CALLER
    return PPMLProvenance(
        caller=caller,
        policy=policy,
        organization=entry.absorbed.absorbed,
        request=entry.request.identity(),
        snapshot_route=entry.preparation.context.snapshot_route,
        ownership=OWNERSHIP,
        consumed=consumed[0],
        reason=consumed[1],
        provider=candidate.provider,
        provider_version=candidate.provider_version,
        retry=entry.options.retry,
        inference_method=entry.method,
    )


def _level(level: Level | None, formed: bool) -> AuditStatus:
    """The audit status of a certificate's level; ``unavailable`` without one."""
    if not formed:
        return "unavailable"
    if level == "strict":
        return "verified"
    return "verified_secondary" if level == "secondary" else "failed"


# ---------------------------------------------------------------- full mode


def _full(entry: _Entry, started: float, eligibility: float) -> PPMLAuditResult:
    """The unchanged full composition, labelled (module docstring)."""
    handoff = AssessmentHandoff(entry.absorbed, entry.scale)
    candidate = _solve(entry, handoff)
    begun = time.perf_counter()
    audit, receipt = handoff.audit(candidate, scale=entry.scale)
    audited = time.perf_counter()
    retained = retain_absorbed(
        AbsorbedHandoff(
            audit=audit,
            preparation=entry.preparation,
            outcome=entry.outcome,
            dependence=_dependence(entry),
            estimator=_estimator(entry.variance),
            request=entry.request,
            keep_influence=entry.keep_influence,
        )
    )
    retention = time.perf_counter() - audited
    certificate = retained.certificate
    status = _level(retained.level, certificate is not None)
    cache = retained.initial_scale
    inference: tuple[InferenceStatus, str] = ("unavailable", "no scale was formed")
    if retained.empty:
        inference = ("not_applicable", NO_SLOPES)
    elif cache is not None:
        inference = standard_error_status(cache.standard_errors)
    failures = (
        ()
        if certificate is None
        else tuple(
            StandardFailure(gate, reason) for gate, reason in certificate.errors.items()
        )
    )
    work = PPMLWork(
        total=time.perf_counter() - started,
        validation_extra=None,
        eligibility=eligibility,
        solve=candidate.timing,
        assessments=candidate.assessments,
        assessment=candidate.assessment_timing,
        capture=receipt.capture,
        selection=audited - begun,
        readings=None,
        inference=None,
        retention=retention,
        provider_calls=None,
        actions=None,
    )
    return PPMLAuditResult(
        audit_mode="full",
        policy=FULL_POLICY,
        status=f"full_{status}",
        standard_status="not_run",
        inference_status=inference[0],
        audit_status=status,
        standard_accepted=None,
        full_audit_accepted=None if certificate is None else retained.accepted,
        standard_evidence=None,
        selection=None,
        full_audit=certificate,
        inference_evidence=cache,
        history=retained.history,
        work=work,
        omitted_checks=(),
        failures=failures,
        provenance=_provenance(
            entry, candidate, (receipt.consumed, receipt.reason), FULL_POLICY
        ),
        validation=None,
        _retained=retained,
        _dependence=retained.dependence,
    )


# ------------------------------------------------------------ standard mode


def _inferred(
    selection: StandardSelection, method: InferenceMethod = "full_joint"
) -> StandardInference:
    """The inference at the selected point: computed only after convergence.

    Dispatched on the entry's requested ``method`` only after the unchanged selection
    (module docstring); nothing selects, retries or falls back to another method.
    """
    reading, replica = selection.reading, selection.replica
    if not selection.history.sample.active.size:
        return StandardInference(
            "not_applicable", NO_SLOPES, requested=method, executed=method
        )
    if selection.status != "converged" or reading is None or replica is None:
        return StandardInference(
            "unavailable", NOT_CONVERGED, requested=method, executed=method
        )
    request = standard_anchors(replica).request
    if method == "residualized":
        return residualized_inference(reading, request)
    return standard_inference(reading, request)


def _validated(
    entry: _Entry, selection: StandardSelection
) -> RetainedSamePointAudit | None:
    """The separately timed same-point full audit, when requested (docstring)."""
    if not entry.validate:
        return None
    reading = selection.reading
    selected = None if reading is None else reading.point
    keep = entry.keep_influence
    found = (selection.capture, selected, entry.request)
    return same_point_audit(*found, keep_influence=keep)


def _detached(
    entry: _Entry,
    selection: StandardSelection,
    bound: BoundStandard | None,
    inference: StandardInference,
    probes: RetainedStopProbes | None,
) -> tuple[StandardRetainedEnvelope, BoundDependence]:
    """The accepted interface admission and the detached envelope (docstring).

    The steps of the accepted ``admit_interface_graph`` after its lower admission, in
    its order, then the accepted history snapshot and the standard composition, with
    the fit's bound stop probes.
    """
    sample, history = entry.sample, selection.history
    context = entry.preparation.context
    ctx, prepared, outcome, auth = _resolve_interface_preparation(
        context, entry.outcome, sample
    )
    check_estimator_matches_candidate(_estimator(entry.variance), sample, history)
    maps = _AdmissionMaps()
    try:
        inputs = None
        if auth is not None:
            inputs = maps.input_map.register_snapshot(
                auth, base=sample.base, prepared=prepared
            )
        elif prepared is not None:
            inputs = maps.input_map.register(prepared)
        snapshot = maps.sample_map.register(
            sample, input_snapshot=inputs, partition_map=maps.partition_map
        )
        inner, _ = _candidate_inner_and_start(
            history.candidate,
            history.route,
            grouping_snapshot=snapshot._final_grouping,
            grouping_map=maps.grouping_map,
        )
        maps.point_map.bind_context(
            sample_snapshot=snapshot, variance=history.variance, inner=inner
        )
        checked = check_dependence_matches_sample(
            _dependence(entry),
            sample,
            context=ctx,
            prepared=prepared,
            sample_key=snapshot.key,
        )
        dependence = checked.with_keys(
            input_key=None if inputs is None else inputs.key, sample_key=snapshot.key
        )
        lowered = None
        if outcome is not None:
            lowered = _lower_outcome_tree(
                outcome, maps=maps, input_snapshot=inputs, sample_snapshot=snapshot
            )
        trigger = None
        if history.trigger is not None:
            trigger = admit_trigger_record(
                history.trigger,
                context=ctx,
                prepared=prepared,
                sample_snapshot=snapshot,
                maps=maps,
            )
        retained = snapshot_candidate_history(
            history,
            sample_snapshot=snapshot,
            trigger_snapshot=trigger,
            point_map=maps.point_map,
            sample_map=maps.sample_map,
            grouping_map=maps.grouping_map,
        )
        envelope = compose_standard_envelope(
            selection,
            bound,
            inference,
            snapshots=(snapshot, lowered, retained),
            maps=(maps.point_map, maps.sample_map, maps.grouping_map),
            keep_influence=entry.keep_influence,
            probes=probes,
        )
    finally:
        maps.clear_anchors()
    return envelope, dependence


def _retained(
    entry: _Entry,
    selection: StandardSelection,
    inference: StandardInference,
    candidate: ProfiledCandidate | IndexCandidate,
) -> tuple[StandardRetainedEnvelope, BoundDependence]:
    """The bound, admitted and detached standard fit; ``ValueError`` if unbound.

    The stop probes are bound to ``candidate``, the solve's own (:func:`_probes`).
    """
    history, bound = selection.history, None
    reading, replica = selection.reading, selection.replica
    if reading is not None and replica is not None:
        point = cast("PointProfile | IndexPoint", history.point)
        live = Anchors(
            history.sample, entry.variance, point, entry.request, selection.controls
        )
        bound = bound_standard(reading, inference, standard_anchors(replica), live)
        if bound is None:
            raise ValueError(UNBOUND)
        admit_standard(bound, history, entry.absorbed, entry.request)
    probes = _probes(entry, candidate)
    with entry.preparation.context._primitive_entry():
        return _detached(entry, selection, bound, inference, probes)


def _top(standard: StandardStatus, inference: InferenceStatus) -> str:
    """The private top-level status of a standard fit (module docstring).

    A converged fit is ``standard_converged`` only with available inference or the
    empty restriction of ``K = 0``.
    """
    if standard == "unavailable":
        return "standard_unavailable"
    if standard != "converged":
        return "standard_refused"
    if inference == "degenerate":
        return "standard_degenerate_inference"
    if inference == "refused":
        return "standard_inference_refused"
    if inference == "unavailable":
        return "standard_unavailable"
    return "standard_converged"


def _audited(validation: RetainedSamePointAudit | None) -> AuditStatus:
    """The status of the separately executed same-point audit, if requested."""
    if validation is None:
        return "not_run"
    return _level(validation.level, not validation.raised)


def _standard(entry: _Entry, started: float, eligibility: float) -> PPMLAuditResult:
    """The standard composition (module docstring)."""
    assessor = StandardAssessor(entry.fit, entry.request, entry.absorbed, entry.options)
    candidate = _solve(entry, assessor)
    begun = time.perf_counter()
    selection = assessor.select(candidate)
    selected = time.perf_counter()
    inference = _inferred(selection, entry.method)
    inferred = time.perf_counter()
    validation = _validated(entry, selection)
    extra = None if validation is None else time.perf_counter() - inferred
    retaining = time.perf_counter()
    envelope, dependence = _retained(entry, selection, inference, candidate)
    finished = time.perf_counter()
    failures = selection.failures
    if inference.status in ("refused", "degenerate"):
        failures = (*failures, StandardFailure("inference", inference.reason))
    probes = envelope.probes
    work = PPMLWork(
        total=finished - started - (extra or 0.0),
        validation_extra=extra,
        eligibility=eligibility,
        solve=candidate.timing,
        assessments=candidate.assessments,
        assessment=candidate.assessment_timing,
        capture=assessor.capture,
        selection=selected - begun,
        readings=assessor.readings,
        inference=inferred - selected,
        retention=finished - retaining,
        provider_calls=inference.provider_calls,
        actions=inference.actions,
        stop_probes=None if probes is None else len(probes.probes),
        stop_probe=None if probes is None else Timing(probes.wall, probes.cpu),
    )
    consumed = (selection.consumed, selection.reason)
    policy = fit_policy(entry.absorbed.absorbed, stop=probes is not None)
    return PPMLAuditResult(
        audit_mode="standard",
        policy=policy,
        status=_top(selection.status, inference.status),
        standard_status=selection.status,
        inference_status=inference.status,
        audit_status=_audited(validation),
        standard_accepted=selection.status == "converged",
        full_audit_accepted=(
            None if validation is None or validation.raised else validation.accepted
        ),
        standard_evidence=envelope.evidence,
        selection=envelope.selection,
        full_audit=validation,
        inference_evidence=envelope.inference,
        history=envelope.history,
        work=work,
        omitted_checks=OMITTED_CHECKS,
        failures=failures,
        provenance=_provenance(entry, candidate, consumed, policy),
        validation=validation,
        _retained=envelope,
        _dependence=dependence,
    )


# ------------------------------------------------------- post-fit requests


def _can_match(
    initial: RetainedInitialScaleCache | None, vcov: PostFitCovarianceSpec
) -> bool:
    """Whether ``vcov`` could name the initial request (the full facade's rule)."""
    if initial is None:
        return False
    found = initial.request
    if vcov is None:
        return True
    if vcov == "robust":
        return found.kind == "robust"
    if type(vcov) is CovarianceRequest:
        return (found.kind, found.name) == (vcov.kind, vcov.name)
    if type(vcov) is ClusterRequest and len(vcov.dimensions) == 1:
        return (found.kind, found.name) == ("cluster", vcov.dimensions[0])
    if isinstance(vcov, Mapping) and set(vcov) == {"cluster"}:
        return (found.kind, found.name) == ("cluster", vcov["cluster"])
    return False


def _cluster(vcov: PostFitCovarianceSpec) -> str:
    """The one cluster dimension that a post-fit request names."""
    if type(vcov) is ClusterRequest:
        if len(vcov.dimensions) != 1:
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": "multi-way cluster covariance"},
            )
        return vcov.dimensions[0]
    if isinstance(vcov, Mapping):
        name = vcov.get("cluster")
        if set(vcov) != {"cluster"} or not isinstance(name, str):
            raise ValueError(
                "vcov mapping must be {'cluster': <column_name>} with a string column"
            )
        return name
    raise TypeError(f"unsupported post-fit covariance request {vcov!r}")


def _normalized(
    envelope: StandardRetainedEnvelope,
    dependence: BoundDependence | None,
    vcov: PostFitCovarianceSpec,
) -> CovarianceRequest | None:
    """A post-fit request as a ``CovarianceRequest``, as the full facade forms it."""
    if vcov is None:
        return None
    if vcov == "robust":
        return CovarianceRequest("robust")
    if type(vcov) is CovarianceRequest:
        check_cluster_request_rows(vcov, envelope.sample)
        return vcov
    name = _cluster(vcov)
    if dependence is None:
        raise SpecificationError(
            "specification.cluster_not_retained", {"name": name, "retained": "none"}
        )
    codes = dependence.cluster_codes(name).codes
    return CovarianceRequest("cluster", name, codes, envelope.sample.positions)


# ------------------------------------------------------------------ result


@dataclass(frozen=True, eq=False)
class PPMLAuditResult:
    """The private result of one joint PPML fit under an explicit audit mode.

    The statuses and records of the module docstring.  ``standard_evidence``,
    ``selection`` and ``inference_evidence`` (standard mode) or ``full_audit`` (the
    full mode's certificate snapshot, or the standard mode's same-point validation)
    are detached records; the retained fit itself is private, read through the
    methods below with the capability, missing, stale and discard refusals of the
    accepted retained results.  No ``accepted`` or ``verified`` alias exists.
    """

    audit_mode: AuditMode
    policy: Policy
    status: str
    standard_status: StandardStatus
    inference_status: InferenceStatus
    audit_status: AuditStatus
    standard_accepted: bool | None
    full_audit_accepted: bool | None
    standard_evidence: RetainedStandardEvidence | None
    selection: RetainedStandardSelection | None
    full_audit: RetainedCertificateSnapshot | RetainedSamePointAudit | None
    inference_evidence: RetainedStandardInference | RetainedInitialScaleCache | None
    history: RetainedHistorySnapshot
    work: PPMLWork
    omitted_checks: tuple[OmittedCheck, ...]
    failures: tuple[StandardFailure, ...]
    provenance: PPMLProvenance
    validation: RetainedSamePointAudit | None
    _retained: StandardRetainedEnvelope | RetainedAbsorbedResult
    _dependence: BoundDependence | None

    @property
    def keep_influence(self) -> bool:
        """Whether the influence rows were retained (``keep_influence``)."""
        return self._retained.keep_influence

    @property
    def sample(self) -> RetainedSampleSnapshot:
        """The detached final sample snapshot."""
        return self._retained.sample

    @property
    def selected_point(self) -> RetainedPointSnapshot | None:
        """The detached selected point snapshot (``q``, effects, slopes)."""
        return self._retained.selected_point

    @property
    def full_result(self) -> RetainedAbsorbedResult | None:
        """The unchanged retained result of full mode, ``None`` in standard mode."""
        found = self._retained
        return found if type(found) is RetainedAbsorbedResult else None

    @property
    def standard_result(self) -> StandardRetainedEnvelope | None:
        """The detached standard envelope, ``None`` in full mode."""
        found = self._retained
        return found if type(found) is StandardRetainedEnvelope else None

    @property
    def stop_probes(self) -> RetainedStopProbes | None:
        """The detached probes of the private equation-aware stop, if it ran."""
        found = self._retained
        return found.probes if type(found) is StandardRetainedEnvelope else None

    def coef(self) -> FloatArray:
        """A fresh copy of the reported slope coefficients of the selected point."""
        return self._retained.coef()

    def scale_evaluation(
        self, vcov: PostFitCovarianceSpec = None
    ) -> RetainedInitialScaleCache:
        """The initial covariance cache, or a new request on retained influence rows."""
        found = self._retained
        if type(found) is RetainedAbsorbedResult:
            return found.scale_evaluation(vcov)
        envelope = cast("StandardRetainedEnvelope", found)
        label = _request_label(vcov)
        if not _can_match(envelope.initial_scale, vcov):
            envelope.require_influence(label)
        request = _normalized(envelope, self._dependence, vcov)
        return envelope.scale_evaluation(request, label=label)

    def se(self, vcov: PostFitCovarianceSpec = None) -> FloatArray:
        """A fresh copy of the direct reported standard errors."""
        found = self._retained
        if vcov is None:
            return found.se()
        return self.scale_evaluation(vcov).standard_errors

    def vcov(self, vcov: PostFitCovarianceSpec = None) -> FloatArray:
        """A fresh copy of the reported covariance matrix."""
        found = self._retained
        if vcov is None:
            return found.vcov()
        return self.scale_evaluation(vcov).require_reported_covariance()

    def influence(self) -> FloatArray | None:
        """A fresh copy of the observed influence rows ``u_i``."""
        return self._retained.influence()

    def scores(self) -> FloatArray | None:
        """A fresh copy of the raw weighted scores ``b``."""
        return self._retained.scores()

    def expected_information(self) -> SparseMatrix | None:
        """The expected joint Hessian, assembled on demand before discard."""
        return self._retained.expected_information()

    def same_point_audit(self) -> RetainedSamePointAudit:
        """The same-point full audit run before discard; a later request refuses."""
        if self.validation is not None:
            return self.validation
        if not self.keep_influence:
            raise SpecificationError(
                "specification.influence_discarded",
                {"request": "a same-point full audit"},
            )
        raise SpecificationError(
            "specification.unsupported_inference",
            {
                "method": "A same-point full audit after the fit",
                "case": "a fit without validate=True, whose capture is not kept",
            },
        )


# -------------------------------------------------------------------- entry


def solve_ppml_private(
    preparation: RetainedPreparation,
    outcome: SampleOutcome,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    scale: AbsorbedScale,
    *,
    audit_mode: AuditMode,
    keep_influence: bool = True,
    validate: bool = False,
) -> PPMLAuditResult:
    """Fit joint PPML on an admitted FE sample under an explicit audit mode.

    Parameters
    ----------
    preparation : RetainedPreparation
        The existing preparation that issued ``outcome``.
    outcome : SampleOutcome
        The complete outcome of the accepted sample coordinator.
    options : SolverOptions
        The solver and certificate options, read once and unmodified.
    absorbed : AbsorbedOptions
        The absorbed options; ``absorbed`` is ``"index"`` or ``"profiled"``.
    scale : AbsorbedScale
        The built-in provider of the requested robust or one-cluster covariance.
    audit_mode : {"standard", "full"}
        Required, no default: the standard reading or the unchanged full audit.
    keep_influence : bool, default True
        Whether the influence rows and ``W`` are retained (semantic discard if not).
    validate : bool, default False
        Standard mode only: run the separately timed same-point full audit.

    Returns
    -------
    PPMLAuditResult
        The private result with its separate statuses (module docstring).

    Raises
    ------
    TypeError, SpecificationError
        An input of this entry is refused before any solve (module docstring); the
        solver's own refusals (``newton_provider``) are unchanged.
    """
    started = time.perf_counter()
    flags = (audit_mode, keep_influence, validate)
    entry = _admitted(preparation, outcome, options, absorbed, scale, flags=flags)
    eligibility = time.perf_counter() - started
    if entry.mode == "full":
        return _full(entry, started, eligibility)
    return _standard(entry, started, eligibility)


def solve_ppml_residualized_private(
    preparation: RetainedPreparation,
    outcome: SampleOutcome,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    scale: AbsorbedScale,
    *,
    audit_mode: AuditMode,
    keep_influence: bool = True,
    validate: bool = False,
) -> PPMLAuditResult:
    """Fit joint PPML in standard mode with the residualized inference.

    The arguments, admission, solve, selection and retention of
    :func:`solve_ppml_private`; the requested inference method is fixed to
    ``residualized`` (module docstring), refused with any ``audit_mode`` but
    ``"standard"`` before any solve.

    Parameters
    ----------
    preparation : RetainedPreparation
        The existing preparation that issued ``outcome``.
    outcome : SampleOutcome
        The complete outcome of the accepted sample coordinator.
    options : SolverOptions
        The solver and certificate options, read once and unmodified.
    absorbed : AbsorbedOptions
        The absorbed options; ``absorbed`` is ``"index"`` or ``"profiled"``.
    scale : AbsorbedScale
        The built-in provider of the requested robust or one-cluster covariance.
    audit_mode : {"standard", "full"}
        Required, no default; only ``"standard"`` is admitted here.
    keep_influence : bool, default True
        Whether the influence rows and ``W`` are retained (semantic discard if not).
    validate : bool, default False
        Run the separately timed same-point full audit after the inference.

    Returns
    -------
    PPMLAuditResult
        The private result with its separate statuses (module docstring).

    Raises
    ------
    TypeError, SpecificationError
        An input of this entry is refused before any solve (module docstring); the
        solver's own refusals (``newton_provider``) are unchanged.
    """
    started = time.perf_counter()
    flags = (audit_mode, keep_influence, validate)
    entry = _admitted(
        preparation,
        outcome,
        options,
        absorbed,
        scale,
        flags=flags,
        method="residualized",
    )
    eligibility = time.perf_counter() - started
    return _standard(entry, started, eligibility)
