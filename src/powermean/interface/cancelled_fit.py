"""Private full-family cancellation lifecycle for any selected fixed-effect block."""

from __future__ import annotations

import copy
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, cast

import numpy as np

from powermean.errors import (
    ConvergenceError,
    CovarianceWarning,
    CoverageWarning,
    ExistenceError,
    NumericalWarning,
    SpecificationError,
)
from powermean.fe import cancel as _cancel
from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.admission import AdmissionInputs
from powermean.fe.admission_records import WorkLog
from powermean.fe.cancel import CancelStructure, cancel_structure
from powermean.fe.geometry import numerical_geometry
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_storage import BoolStorage, _build_record
from powermean.fe.groups import (
    _CancelledLabelPartition,
    _owned_encoding_original,
    _positions,
    encode_block,
)
from powermean.fe.retained import centered_design
from powermean.inference.cancelled import reserve_inference, slope_rhs
from powermean.inference.convention import ClusterCodes
from powermean.inference.correction.short_group import ShortGroupFitPoint
from powermean.inference.ssc import AppliedConvention
from powermean.inference.vcov import CovarianceWarning as PortWarning
from powermean.inference.wz_covariance import (
    WZControls,
    reported,
    retained_covariance_record,
)
from powermean.inference.wz_operator import Information, PairCovariance
from powermean.interface.cancelled_selection import (
    CancelledBlockInputs,
    CancelledFitControls,
    CancelledFitPolicy,
    CancelledSelection,
    NamedFEBlock,
    align_cancelled_reporting,
    resolve_policy_variance,
    resolve_selection,
    select_cancelled_block,
)
from powermean.interface.cancelled_tau1_origin import (
    reduced_tau1_origin,
    tau1_coordinator,
)
from powermean.interface.fe_bias import apply_cancelled_wz_bias
from powermean.model.order_scale import OrderScale
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    PowerVariance,
    Variance,
)
from powermean.results.cancelled_inference import (
    request_convention,
    same_point_inference,
)
from powermean.results.cancelled_retained import (
    CancelledReporting,
    RetainedCancelledEnvelope,
    compute_cancelled_accounting,
    prepare_cancelled_reporting,
)
from powermean.results.retained_hc import RetainedHC, cancelled_point_hc
from powermean.results.retained_inference import InferenceRequest, RetainedInference
from powermean.results.wz_bias import WZBias
from powermean.solve.assessment import Assessment, certificate_controls
from powermean.solve.cancelled_attempt import run_cancelled_attempts
from powermean.solve.cancelled_certificate import (
    CertifiedCancelledPoint,
    certified_again,
    certify_cancelled,
    saturated_geometry,
)
from powermean.solve.cancelled_existence import Existence, recession
from powermean.solve.cancelled_operator import cancelled_problem, coordinate_map
from powermean.solve.cancelled_records import (
    CancelledCertificate,
    CoverageKind,
    MeritMetric,
    RootExecutionBounds,
    RootInitializerRecord,
    RootOptions,
    RootStageRecord,
    RouteKind,
    Tau1SolveResult,
    reserve_certificate,
    resolve_root_options,
    route_for_variance,
)
from powermean.solve.cancelled_root import solve_cancelled_root
from powermean.solve.cancelled_tau1 import Tau1ExecutionBounds, solve_cancelled_tau1
from powermean.solve.factorization import Factor
from powermean.solve.newton_retry import Attempts
from powermean.solve.newton_trace import Timing
from powermean.solve.options import SolverOptions
from powermean.solve.reported import ReportedScale
from powermean.solve.status import Status
from powermean.systems.admitted import CapturedFit, FitRows, capture_fit
from powermean.systems.identified import IdentifiedState

FitInputs = AdmissionInputs | CancelledSelection | CancelledBlockInputs | GeometryInputs
_CANCEL_STRUCTURE = cancel_structure
_TAU1_COORDINATOR = tau1_coordinator

__all__ = [
    "CancelledBlockInputs",
    "CancelledFitControls",
    "CancelledFitPolicy",
    "CancelledReporting",
    "CancelledSelection",
    "NamedFEBlock",
    "align_cancelled_reporting",
    "fit_cancelled",
    "fit_cancelled_private",
    "select_cancelled_block",
]


def _convention_unless_saturated(
    geometry: NumericalGeometry, request: InferenceRequest
) -> AppliedConvention | None:
    """The request's convention before the solve, or ``None`` for a saturated fit.

    A convention whose factor or reference needs residual degrees of freedom is
    refused (``specification.no_residual_dof``) on any fit, except a saturated one
    (:func:`~powermean.solve.cancelled_certificate.saturated_geometry`): its
    certificate calls no provider, so this convention is never applied, and every
    covariance of the fit is missing (:mod:`powermean.results.saturated`).
    """
    try:
        return request_convention(geometry, request)
    except SpecificationError as error:
        if error.identifier == "specification.no_residual_dof" and saturated_geometry(
            geometry
        ):
            return None
        raise


# The classes of the notes that describe the covariance of one candidate
# (``powermean.results.cancelled_rows.NOTES``, which is loaded on request only).
NOTES = (CovarianceWarning, PortWarning, CoverageWarning)


@dataclass(slots=True)
class ShortGroupHandoff:
    """Temporary output for the normal call's post-selection BBP correction."""

    point: ShortGroupFitPoint | None = None

    def take(self) -> ShortGroupFitPoint | None:
        """Release the handoff's reference as soon as correction consumes it."""
        point, self.point = self.point, None
        return point

    def select(
        self, provider: _ScaleCapture, state: IdentifiedState | None, *, accepted: bool
    ) -> None:
        """Share only the accepted outer state's own factor and adjoint."""
        self.point = None
        scale = provider.short_group_scale
        if (
            accepted
            and state is not None
            and scale is not None
            and scale[0] is state._theta
            and provider.capture is not None
        ):
            self.point = ShortGroupFitPoint(provider.capture, state, scale[1], scale[2])


@dataclass
class _ScaleCapture:
    """One certification's inference output; reset before every fresh certificate."""

    order: OrderScale
    request: InferenceRequest
    work: WorkLog
    applied: AppliedConvention | None  # None for a saturated fit, never called then
    latest: RetainedInference | None = None
    rhs: FloatStorage | None = None
    wz: WZControls | None = None
    capture: CapturedFit | None = None
    # This fit's last WZ covariance and same-point information (owner D1/D2):
    # reused only for identical freshly captured rows; dies with the fit.
    wz_shared: tuple[PairCovariance, Information] | None = None
    # The scale's notes at the latest candidate, issued once at the fit.
    held: list[warnings.WarningMessage] = field(default_factory=list)
    share_short_group: bool = False
    short_group_scale: tuple[FloatStorage, Factor, FloatArray] | None = None

    def reset(self) -> None:
        """Clear previous inference state before a new certification attempt."""
        self.latest = None
        self.rhs = None
        self.short_group_scale = None

    def __call__(self, state: IdentifiedState, factor: Factor, /) -> ReportedScale:
        if self.request.adjustment == "wz":
            return self._wz_scale(state, factor)
        rhs = slope_rhs(state, factor, self.work)
        # Saturated certification returns before it invokes a scale provider.
        applied = cast("AppliedConvention", self.applied)
        if self.request.multiway_path:
            result = same_point_inference(
                state,
                rhs,
                self.order,
                self.request,
                applied,
                work=self.work,
                record=RetainedInference,
                rows=(self.capture, self.held),
            )
        else:
            # A one-way candidate's notes (the coverage warning of a dimension that
            # crosses the groups) are held as a multiway candidate's are: the fit
            # issues those of its reported point once, as a post-fit request does.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = same_point_inference(
                    state,
                    rhs,
                    self.order,
                    self.request,
                    applied,
                    work=self.work,
                    record=RetainedInference,
                    rows=(self.capture, self.held),
                )
            self.held[:] = [item for item in caught if isinstance(item.message, NOTES)]
            for item in caught:
                if not isinstance(item.message, NOTES):
                    warnings.warn(item.message, stacklevel=2)
        self.latest, self.rhs = result, FloatStorage.store(rhs)
        if self.share_short_group:
            # A retry can discard this evaluation and all its row arrays.
            self.short_group_scale = (state._theta, factor, rhs)
        k = state.design.n_slopes_identified
        return ReportedScale(
            gamma=state.theta[:k],
            order=self.order,
            standard_errors=result.standard_errors,
        )

    def _wz_scale(self, state: IdentifiedState, factor: Factor) -> ReportedScale:
        """Evaluate WZ before this candidate's SE-dependent coefficient gate."""
        wz, capture = self.wz, self.capture
        if wz is None or capture is None:
            raise ValueError("WZ requires the current original source and capture")
        previous = self.wz_shared
        self.wz_shared = None
        applied = cast("AppliedConvention", self.applied)
        shared = self.work.run(
            "wz_covariance",
            lambda: wz.cancelled_information(
                state,
                capture,
                self.request.clusters[0],
                ssc=applied.convention.preset or "custom",
                previous=previous,
            ),
        )
        self.wz_shared = shared
        result = shared[0]
        # Keep the uncorrected first-order W for ordinary inference and BBP.
        rhs = slope_rhs(state, factor, self.work)
        errors, matrix, failure = reported(result, self.order)
        self.latest = RetainedInference(
            self.request,
            applied,
            None,
            "observed",
            retained_covariance_record(result, applied),
            result.adjusted,
            errors,
            matrix,
            failure,
            state.theta,
            False,
            state.design.structure.n_groups,
            state.design.structure.n_rows,
            state.design.structure.n_rows,
            wz=result,
        )
        self.rhs = FloatStorage.store(rhs)
        return ReportedScale(
            gamma=state.theta[: self.order.dimension],
            order=self.order,
            standard_errors=errors,
        )


@dataclass(frozen=True)
class _RouteExecution:
    """Internal result bundle from executing one family route."""

    route: RouteKind
    final: CertifiedCancelledPoint
    status: Status
    history: tuple[CancelledCertificate, ...]
    attempts: Attempts | None = None
    root_selection: CoverageKind = "unique_root"
    merit_metric: MeritMetric | None = None
    initializer: RootInitializerRecord | None = None
    continuation_used: bool = False
    root_stages: tuple[RootStageRecord, ...] = ()
    tau1_result: Tau1SolveResult | None = None
    root_options: RootOptions | None = None
    # tau = 2 only, after a solve that does not certify (ruling 71); None otherwise.
    existence: Existence | None = None


def _issue_held(result: RetainedCancelledEnvelope, provider: _ScaleCapture) -> None:
    """Issue the covariance notes of the reported point once, at the fit's caller."""
    if not result.accepted or provider.latest is None:
        return
    for note in provider.held:
        warnings.warn(note.message, stacklevel=3)


def _issue_merit(result: RetainedCancelledEnvelope, policy: CancelledFitPolicy) -> None:
    """Note once a merit metric that the caller stated and the zero start replaced.

    :func:`~powermean.solve.cancelled_root.solve_cancelled_root` replaces the Gamma
    metric by Gram at its zero start whatever was requested, so that the solve does
    not depend on whether a factor forms by rounding (ruling 68); a caller who
    stated the metric is told so.  A metric left at its default is replaced
    without a note, as the record of the effective root options shows it.
    """
    given = policy.merit_metric_given or (
        isinstance(policy.root_options, Mapping)
        and "merit_metric" in policy.root_options
    )
    initializer, effective = result.initializer, result.merit_metric
    if not given or initializer is None or initializer.kind != "zero_start":
        return
    requested = resolve_root_options(policy.root_options).merit_metric
    if effective is None or effective == requested:
        return
    warn = NumericalWarning(
        "numerical.merit_metric_replaced",
        {"requested": requested},
        status=result.status,
        certificate=result.certificate,
    )
    warnings.warn(warn, stacklevel=3)


def _report(
    result: RetainedCancelledEnvelope,
    policy: CancelledFitPolicy,
    provider: _ScaleCapture,
    final: CertifiedCancelledPoint,
) -> RetainedCancelledEnvelope:
    if result.status == "verified_secondary" and policy.report_secondary:
        tols = {
            "tol_secondary": result.options.tol_secondary,
            "tol": result.options.tol,
        }
        warn = NumericalWarning(
            "numerical.secondary_convergence",
            tols,
            status=result.status,
            certificate=result.certificate,
        )
        warnings.warn(warn, stacklevel=3)
    if not result.accepted and policy.on_failure == "raise":
        if result.status == "no_finite_root":
            raise ExistenceError(
                "existence.no_finite_root",
                status=result.status,
                last_state=result,
                certificate=result.certificate,
            )
        raise ConvergenceError(
            "convergence.not_certified",
            status=result.status,
            last_state=result,
            certificate=result.certificate,
        )
    if policy.short_group is not None:
        policy.short_group.select(provider, final.state, accepted=result.accepted)
    return result


def _execute_tau2_route(
    capture: CapturedFit,
    controls: CancelledFitControls,
    options: SolverOptions,
    *,
    provider: _ScaleCapture,
    work: WorkLog,
    start: FloatArray | None,
) -> _RouteExecution:
    geom = capture.admission.geometry
    operator = coordinate_map(
        geom.design,
        geom.basis_matrix,
        geom.reduction_matrix,
        identity_atol=controls.identity_atol,
        bounds=controls.attempt,
    )
    problem = cancelled_problem(
        operator,
        capture.z,
        capture.centered_offset,
        variance=PowerVariance(2),
        obs_weights=capture.obs_weights,
    )
    initial = np.zeros(geom.n_identified) if start is None else start
    history: list[CancelledCertificate] = []
    made: list[CertifiedCancelledPoint] = []
    cert_options, cb = certificate_controls(options), controls.certificate

    def fresh(point: FloatArray) -> CertifiedCancelledPoint:
        current = certified_again(made, point)
        if current is None:
            current = certify_cancelled(
                capture, point, cert_options, bounds=cb, reported=provider, work=work
            )
            made.append(current)
        history.append(current.certificate)
        return current

    def assess(point: FloatArray) -> Assessment:
        return fresh(point).certificate.assessment()

    attempts = run_cancelled_attempts(problem, initial, options, assess=assess)
    selected = attempts.attempts[attempts.retained]
    final = fresh(selected.state.coefficients)
    status = final.certificate.status(selected.reason)
    # Ruling 71: a solve that does not certify ends no_finite_root where the
    # cancelled equation has a recession direction (solve.cancelled_existence).
    existence: Existence | None = None
    if status not in ("verified", "verified_secondary"):
        existence = work.run(
            "existence_program",
            lambda: recession(
                geom.design, capture.z, raw_slopes=capture.admission.raw_slopes
            ),
        )
        if existence == "no_finite_root":
            status = "no_finite_root"
    return _RouteExecution(
        "convex_tau2",
        final,
        status,
        tuple(history),
        attempts,
        root_options=None,
        existence=existence,
    )


def _execute_route(
    capture: CapturedFit,
    target_variance: CancelledFamily,
    controls: CancelledFitControls,
    options: SolverOptions,
    *,
    policy: CancelledFitPolicy,
    provider: _ScaleCapture,
    work: WorkLog,
    reduced_tau1: bool,
) -> _RouteExecution:
    route, root_sel = route_for_variance(target_variance)
    if route == "convex_tau2":
        return _execute_tau2_route(
            capture, controls, options, provider=provider, work=work, start=policy.start
        )
    if route == "absorbed_tau1":
        reduced_origin = (
            partial(
                reduced_tau1_origin,
                capture,
                target_variance,
                diagnostic=controls.diagnostic,
                mode=controls.collinear,
                work=work,
            )
            if reduced_tau1 and tau1_coordinator is _TAU1_COORDINATOR
            else None
        )

        t1 = solve_cancelled_tau1(
            capture,
            target_variance,
            options,
            absorbed=policy.absorbed_options,
            bounds=Tau1ExecutionBounds(
                controls.certificate,
                lambda sample, family: tau1_coordinator(
                    sample,
                    family,
                    diagnostic=controls.diagnostic,
                    mode=controls.collinear,
                    work=work,
                ),
                reduced_origin,
            ),
            reported=provider,
            work=work,
            start=policy.start,
        )
        return _RouteExecution(
            "absorbed_tau1",
            t1.final,
            t1.status,
            t1.history,
            tau1_result=t1,
            root_options=None,
        )
    ro = resolve_root_options(policy.root_options)
    rb = RootExecutionBounds(
        controls.certificate, controls.attempt, controls.identity_atol
    )
    rr = solve_cancelled_root(
        capture,
        target_variance,
        options,
        ro,
        bounds=rb,
        reported=provider,
        work=work,
        start=policy.start,
    )
    return _RouteExecution(
        "root_newton",
        rr.final,
        rr.status,
        rr.history,
        None,
        root_sel,
        rr.merit_metric,
        rr.initializer,
        rr.continuation_used,
        rr.stages,
        root_options=rr.root_options,
    )


@dataclass(frozen=True, eq=False)
class _PreparedCancelledLayout:
    """One public dispatch's layout and the exact immutable labels it encoded."""

    inputs: AdmissionInputs
    structure: CancelStructure

    def bind(self, current: AdmissionInputs) -> CancelStructure | None:
        """Reuse only on the same ordered owned partitions and positional rows."""
        before = (self.inputs.cancelled, *self.inputs.retained)
        after = (current.cancelled, *current.retained)
        positions = current.slopes.positions
        if (
            type(self.structure) is CancelStructure
            and cancel_structure is _CANCEL_STRUCTURE
            and _cancel.cancel_structure is _CANCEL_STRUCTURE
            and vars(_cancel).get("encode_block") is encode_block
            and vars(_cancel).get("_positions") is _positions
            and _owned_encoding_original()
            and len(before) == len(after)
            and all(
                type(old) is _CancelledLabelPartition and old is new
                for old, new in zip(before, after, strict=True)
            )
            and type(positions) is np.ndarray
            and positions.dtype == np.dtype(np.intp)
            and np.array_equal(positions, self.structure.positions)
        ):
            return self.structure
        return None


def _fit_structure(
    inputs: AdmissionInputs, prepared: _PreparedCancelledLayout | None, work: WorkLog
) -> CancelStructure:
    """Bind a matching layout or build it from the selected partitions."""
    if prepared is not None:
        bound = work.run("layout_binding", lambda: prepared.bind(inputs))
        if bound is not None:
            return bound
    return work.run(
        "layout",
        lambda: cancel_structure(
            inputs.cancelled, inputs.retained, positions=inputs.slopes.positions
        ),
    )


def fit_cancelled_private(
    inputs: FitInputs,
    rows: FitRows,
    reporting: CancelledReporting,
    controls: CancelledFitControls,
    options: SolverOptions,
    *,
    policy: CancelledFitPolicy,
    variance: Variance | float | None = None,
    cancel: str | Literal[True] | None = None,
    _prepared_layout: _PreparedCancelledLayout | None = None,
) -> RetainedCancelledEnvelope:
    """Run the cancelled fit. Public dispatch reports its own secondary warning."""
    began, work = (time.perf_counter(), time.process_time()), WorkLog()
    if type(policy) is not CancelledFitPolicy:
        raise TypeError("use an explicit CancelledFitPolicy")
    keep_influence, on_failure = policy.keep_influence, policy.on_failure
    if type(keep_influence) is not bool or on_failure not in ("raise", "return"):
        raise ValueError(
            "use explicit keep_influence and on_failure='raise' or 'return'"
        )
    if type(controls) is not CancelledFitControls or type(options) is not SolverOptions:
        raise TypeError("use explicit CancelledFitControls and existing SolverOptions")
    target_variance = work.run(
        "resolve_variance", lambda: resolve_policy_variance(policy, variance)
    )
    adm, aligned_rep, sel_mode, block_order, cancel_idx = work.run(
        "resolve_selection",
        lambda: resolve_selection(inputs, reporting, work=work, cancel=cancel),
    )
    structure = _fit_structure(adm, _prepared_layout, work)
    design = work.run(
        "centered_design", lambda: centered_design(structure, adm.slopes.raw_slopes)
    )
    g_rtol, gap, g_bytes = controls.gram_rtol, controls.gap_min, controls.geometry_bytes
    geometry = work.run(
        "candidate",
        lambda: numerical_geometry(
            design,
            gram_rtol=g_rtol,
            gap_min=gap,
            max_bytes=g_bytes,
            nuisance_basis="exact_selection",
            original_slopes=adm.slopes.original_slopes,
        ),
    )
    d_ctrl, c_mode, a_bytes = (
        controls.diagnostic,
        controls.collinear,
        controls.admission_bytes,
    )
    capture = work.run(
        "capture_fit",
        lambda: capture_fit(
            geometry, adm, rows, controls=d_ctrl, mode=c_mode, max_bytes=a_bytes
        ),
    )
    order, labels, full_scales = prepare_cancelled_reporting(geometry, aligned_rep)
    default_cluster = ClusterCodes(aligned_rep.cancelled_name, structure.cancel_codes)
    bound = (
        copy.deepcopy(policy.request)
        if policy.request is not None
        else InferenceRequest("cluster", (default_cluster,))
    )
    if type(bound) is not InferenceRequest:
        raise TypeError("request must be the actual InferenceRequest")
    applied = _convention_unless_saturated(geometry, bound)
    # The covariance request does not enter the solve (the requested covariance is
    # computed once per call, at the accepted point).  A multiway request is signed
    # and can leave a coefficient without a usable standard error at a candidate,
    # the Gamma initializer of the root route included; its certificates read the
    # route's own one-way scale on the cancelled block, as the fit without ``vcov``
    # does, and the request is evaluated once at the final point below.
    gate, gate_applied = bound, applied
    if bound.multiway_path:
        gate = InferenceRequest(
            "cluster",
            (default_cluster,),
            ssc=bound.ssc,
            bread=bound.bread,
            inherited=bound.inherited,
        )
        gate_applied = _convention_unless_saturated(geometry, gate)
    reserve_certificate(geometry, controls.certificate)
    reserve_inference(geometry, controls.inference)
    provider = _ScaleCapture(
        order,
        gate,
        work,
        gate_applied,
        wz=policy.wz,
        capture=capture,
        share_short_group=policy.short_group is not None,
    )
    ex = _execute_route(
        capture,
        target_variance,
        controls,
        options,
        policy=policy,
        provider=provider,
        work=work,
        reduced_tau1=(
            _prepared_layout is not None
            and structure is _prepared_layout.structure
            and order.omega == 1.0
            and policy.root_options is None
        ),
    )
    final, cert = ex.final, ex.final.certificate
    if gate is not bound and provider.latest is not None and provider.rhs is not None:
        # A reported scale and RHS carry the accepted identified-state branch.
        try:
            provider.latest = same_point_inference(
                cast("IdentifiedState", final.state),
                provider.rhs.view(),
                order,
                bound,
                cast("AppliedConvention", applied),
                work=work,
                record=RetainedInference,
                rows=(capture, provider.held),
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            provider.latest = None
            ex = replace(ex, status="solver_error")
    named: RetainedHC | None = None
    if (
        policy.hc is not None
        and ex.status in ("verified", "verified_secondary")
        and provider.rhs is not None
    ):
        full_names = capture.admission.check.source_columns
        named = cancelled_point_hc(
            cast("IdentifiedState", final.state),
            provider.rhs.view(),
            order,
            policy.hc,
            tuple(full_names[j] for j in geometry.kept_slope_indices),
            keep_influence=keep_influence,
            cancelled_index=cancel_idx,
        )
    correction: WZBias | None = None
    if policy.wz_coefficient is not None:
        correction = (
            apply_cancelled_wz_bias(
                final,
                capture,
                order,
                policy.wz_coefficient,
                default_cluster,
                shared=None if provider.wz_shared is None else provider.wz_shared[1],
            )
            if ex.status in ("verified", "verified_secondary")
            else WZBias(default_cluster.name, None)
        )
    provider.wz_shared = None
    accounting = work.run(
        "retained_accounting",
        lambda: compute_cancelled_accounting(capture, final.state),
    )
    result = cast(
        RetainedCancelledEnvelope,
        _build_record(
            RetainedCancelledEnvelope,
            geometry=geometry,
            slope_names_full=capture.admission.check.source_columns,
            quotient=capture.admission.quotient,
            slope_evidence=capture.admission.slope_evidence,
            admission_work=capture.admission.work,
            options=options,
            keep_influence=keep_influence,
            status=ex.status,
            timing=Timing(0.0, 0.0),
            work=work.snapshot(),
            _point=FloatStorage.store(cert.gamma),
            _order=order,
            _user_labels=labels,
            _certificate=cert,
            _history=ex.history,
            _attempts=ex.attempts,
            _request=bound,
            _initial=provider.latest,
            _hc=named,
            _capture=capture if keep_influence else None,
            _state=final.state if keep_influence else None,
            _rhs=provider.rhs if keep_influence else None,
            _inference_bounds=controls.inference,
            _accounting=accounting,
            _full_scales=full_scales,
            _zero_outcomes=BoolStorage.store(capture.z == 0),
            cancelled_name=aligned_rep.cancelled_name,
            retained_names=aligned_rep.retained_names,
            block_order=block_order,
            cancelled_index=cancel_idx,
            selection_mode=sel_mode,
            variance=target_variance,
            executed_variance=cert.variance,
            route=ex.route,
            root_selection=ex.root_selection,
            merit_metric=ex.merit_metric,
            _root_options=ex.root_options,
            _initializer=ex.initializer,
            continuation_used=ex.continuation_used,
            _root_stages=ex.root_stages,
            _tau1_result=ex.tau1_result,
            _wz_bias=correction,
            existence=ex.existence,
        ),
    )
    elapsed = Timing(time.perf_counter() - began[0], time.process_time() - began[1])
    object.__setattr__(result, "timing", elapsed)
    _issue_held(result, provider)
    _issue_merit(result, policy)
    return _report(result, policy, provider, final)


fit_cancelled = fit_cancelled_private
