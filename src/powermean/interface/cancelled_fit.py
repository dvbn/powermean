"""Private full-family cancellation lifecycle for any selected fixed-effect block."""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from powermean.errors import ConvergenceError, NumericalWarning
from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.admission import AdmissionInputs
from powermean.fe.admission_records import WorkLog
from powermean.fe.cancel import cancel_structure
from powermean.fe.geometry import numerical_geometry
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_storage import BoolStorage, _build_record
from powermean.fe.retained import centered_design
from powermean.inference.cancelled import reserve_inference, slope_rhs
from powermean.inference.convention import ClusterCodes
from powermean.inference.ssc import AppliedConvention
from powermean.inference.wz_covariance import (
    WZControls,
    reported,
    retained_covariance_record,
)
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
from powermean.interface.cancelled_tau1_origin import tau1_coordinator
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
from powermean.results.retained_inference import InferenceRequest, RetainedInference
from powermean.results.wz_bias import WZBias
from powermean.solve.assessment import Assessment, certificate_controls
from powermean.solve.cancelled_attempt import run_cancelled_attempts
from powermean.solve.cancelled_certificate import (
    CertifiedCancelledPoint,
    certify_cancelled,
)
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


@dataclass
class _ScaleCapture:
    """One certification's inference output; reset before every fresh certificate."""

    order: OrderScale
    request: InferenceRequest
    work: WorkLog
    applied: AppliedConvention
    latest: RetainedInference | None = None
    rhs: FloatStorage | None = None
    wz: WZControls | None = None
    capture: CapturedFit | None = None

    def reset(self) -> None:
        """Clear previous inference state before a new certification attempt."""
        self.latest = None
        self.rhs = None

    def __call__(self, state: IdentifiedState, factor: Factor, /) -> ReportedScale:
        if self.request.adjustment == "wz":
            return self._wz_scale(state, factor)
        rhs = slope_rhs(state, factor, self.work)
        result = same_point_inference(
            state,
            rhs,
            self.order,
            self.request,
            self.applied,
            work=self.work,
            record=RetainedInference,
        )
        self.latest, self.rhs = result, FloatStorage.store(rhs)
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
        result = self.work.run(
            "wz_covariance",
            lambda: wz.cancelled(
                state,
                capture,
                self.request.clusters[0],
                ssc=self.applied.convention.preset or "custom",
            ),
        )
        # Keep the uncorrected first-order W for ordinary inference and BBP.
        rhs = slope_rhs(state, factor, self.work)
        errors, matrix, failure = reported(result, self.order)
        self.latest = RetainedInference(
            self.request,
            self.applied,
            None,
            "observed",
            retained_covariance_record(result, self.applied),
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


def _report(
    result: RetainedCancelledEnvelope, on_failure: str
) -> RetainedCancelledEnvelope:
    if result.status == "verified_secondary":
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
    if not result.accepted and on_failure == "raise":
        raise ConvergenceError(
            "convergence.not_certified",
            status=result.status,
            last_state=result,
            certificate=result.certificate,
        )
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
    cert_options, cb = certificate_controls(options), controls.certificate

    def fresh(point: FloatArray) -> CertifiedCancelledPoint:
        current = certify_cancelled(
            capture, point, cert_options, bounds=cb, reported=provider, work=work
        )
        history.append(current.certificate)
        return current

    def assess(point: FloatArray) -> Assessment:
        return fresh(point).certificate.assessment()

    attempts = run_cancelled_attempts(problem, initial, options, assess=assess)
    selected = attempts.attempts[attempts.retained]
    final = fresh(selected.state.coefficients)
    status = final.certificate.status(selected.reason)
    return _RouteExecution(
        "convex_tau2", final, status, tuple(history), attempts, root_options=None
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
) -> _RouteExecution:
    route, root_sel = route_for_variance(target_variance)
    if route == "convex_tau2":
        return _execute_tau2_route(
            capture, controls, options, provider=provider, work=work, start=policy.start
        )
    if route == "absorbed_tau1":
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
        ro.merit_metric,
        rr.initializer,
        rr.continuation_used,
        rr.stages,
        root_options=rr.root_options,
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
) -> RetainedCancelledEnvelope:
    """Execute one complete full-family, unit-multiplier private cancelled fit."""
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
    structure = work.run(
        "layout",
        lambda: cancel_structure(
            adm.cancelled, adm.retained, positions=adm.slopes.positions
        ),
    )
    design = work.run(
        "centered_design", lambda: centered_design(structure, adm.slopes.raw_slopes)
    )
    g_rtol, gap, g_bytes = controls.gram_rtol, controls.gap_min, controls.geometry_bytes
    geometry = work.run(
        "candidate",
        lambda: numerical_geometry(
            design, gram_rtol=g_rtol, gap_min=gap, max_bytes=g_bytes
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
    applied = request_convention(geometry, bound)
    reserve_certificate(geometry, controls.certificate)
    reserve_inference(geometry, controls.inference)
    provider = _ScaleCapture(order, bound, work, applied, wz=policy.wz, capture=capture)
    ex = _execute_route(
        capture,
        target_variance,
        controls,
        options,
        policy=policy,
        provider=provider,
        work=work,
    )
    final, cert = ex.final, ex.final.certificate
    correction: WZBias | None = None
    if policy.wz_coefficient is not None:
        correction = (
            apply_cancelled_wz_bias(
                final, capture, order, policy.wz_coefficient, default_cluster
            )
            if ex.status in ("verified", "verified_secondary")
            else WZBias(default_cluster.name, None)
        )
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
        ),
    )
    elapsed = Timing(time.perf_counter() - began[0], time.process_time() - began[1])
    object.__setattr__(result, "timing", elapsed)
    return _report(result, on_failure)


fit_cancelled = fit_cancelled_private
