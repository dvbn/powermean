"""Exact tau=1 and NB2(kappa=0) cancelled solve via the joint-absorbed engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.fe.admission_records import WorkLog
from powermean.fe.profile import ProfileState
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    NB2Variance,
    PowerVariance,
)
from powermean.solve.absorbed_direction import IndexStart
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.cancelled_certificate import (
    CancelledScaleProvider,
    certify_cancelled,
)
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.cancelled_records import (
    CancelledCertificate,
    Tau1OriginError,
    Tau1SolveResult,
    resolve_absorbed_options,
)
from powermean.solve.cancelled_tau1_bridge import (
    _build_bridge_layout,
    _effects_to_identified,
    _index_point_to_identified,
    _Tau1Context,
    _warm_effects_from_start,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import (
    IndexCandidate,
    IndexWholePoint,
)
from powermean.solve.newton_absorbed import (
    solve as solve_absorbed,
)
from powermean.solve.newton_profiled import (
    FullPoint,
    ProfiledCandidate,
)
from powermean.solve.newton_profiled import (
    solve as solve_profiled,
)
from powermean.solve.newton_trace import NativeReason
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_trigger import Admission, Trigger
from powermean.solve.starts import Start
from powermean.solve.status import NATIVE_STATUS
from powermean.systems.admitted import CapturedFit

__all__ = ["Tau1SolveResult", "resolve_absorbed_options", "solve_cancelled_tau1"]


@dataclass(frozen=True)
class Tau1ExecutionBounds:
    """Certificate resources and an interface-owned genuine-origin factory."""

    certificate: AttemptBounds
    coordinator: Callable[[FESample, CancelledFamily], tuple[Trigger, Admission]] | None


def _validate_tau1_family(variance: CancelledFamily) -> PowerVariance | NB2Variance:
    """Verify that ``variance`` is ``PowerVariance(1.0)`` or ``NB2Variance(0.0)``."""
    if isinstance(variance, PowerVariance) and variance.tau == 1.0:
        return variance
    if isinstance(variance, NB2Variance) and variance.kappa == 0.0:
        return variance
    raise ValueError("cancelled_tau1 requires PowerVariance(1.0) or NB2Variance(0.0)")


def _map_native_reason(reason: str) -> NativeReason:
    """Map an absorbed/profiled operative reason to ``NativeReason``."""
    if reason in ("native_stop", "profile_only", "no_progress"):
        return "native_stop"
    if reason in ("iteration_limit", "inner_limit"):
        return "iteration_limit"
    if reason == "evaluation_limit":
        return "evaluation_limit"
    if reason in ("line_search_failure", "line_search_failed"):
        return "line_search_failure"
    if reason == "factorization_failure":
        return "factorization_failure"
    if reason == "non_descent_direction":
        return "non_descent_direction"
    return "numerical_breakdown"


def _candidate_evaluations(
    candidate: IndexCandidate | ProfiledCandidate | None, cert_count: int
) -> int:
    """Count inner solver and fresh certificate evaluations performed."""
    if candidate is None:
        return cert_count
    iter_evals = sum(
        0 if att.iteration is None else att.iteration.evaluations.total
        for att in candidate.attempts
    )
    return max(iter_evals, len(candidate.work.profiles)) + cert_count


def _assess_theta(
    ctx: _Tau1Context, theta: FloatArray, cert_controls: CertificateControls
) -> Assessment:
    """Certify ``theta`` on ``ctx.capture`` and record its certificate."""
    certified = certify_cancelled(
        ctx.capture,
        theta,
        cert_controls,
        bounds=ctx.bounds,
        reported=ctx.reported,
        work=ctx.work,
        variance=ctx.family,
    )
    ctx.history.append(certified.certificate)
    return certified.certificate.assessment()


def _run_index_organization(
    ctx: _Tau1Context,
) -> tuple[IndexCandidate, FloatArray | None]:
    """Return the evaluated canonical point, or None without substituting a start."""
    geometry, grouping = ctx.capture.admission.geometry, ctx.layout.sample.grouping
    assert grouping is not None

    def assess_index(pt: IndexWholePoint, ctrl: CertificateControls, /) -> Assessment:
        return _assess_theta(ctx, _index_point_to_identified(ctx, pt.point), ctrl)

    idx_start: IndexStart | None = None
    if ctx.start is not None and geometry.n_slopes_identified > 0:
        scaled, warm = _warm_effects_from_start(geometry, ctx.layout, ctx.fallback)
        idx_start = IndexStart(scaled, ProfileState(grouping, warm))

    assess_fn = assess_index if ctx.options.retry == 1 else None
    s_in, fam, opts, ab = ctx.layout.sample, ctx.family, ctx.options, ctx.absorbed
    candidate = ctx.work.run(
        "tau1_absorbed_index_solve",
        lambda: solve_absorbed(
            s_in,
            fam,
            opts,
            ab,
            trigger=ctx.trigger,
            admit=ctx.admit,
            assess=assess_fn,
            start=idx_start,
        ),
    )
    ctx.candidate = candidate
    # Only the canonical point reader is guarded; coordinate-mapping errors are not.
    try:
        point = candidate.point().point
    except ValueError:
        return candidate, None
    return candidate, _index_point_to_identified(ctx, point)


def _run_profiled_organization(
    ctx: _Tau1Context,
) -> tuple[ProfiledCandidate, FloatArray | None]:
    """Return the evaluated canonical point, or None without substituting a start."""
    geometry = ctx.capture.admission.geometry

    def assess_prof(pt: FullPoint, ctrl: CertificateControls, /) -> Assessment:
        p = pt.point
        theta = _effects_to_identified(
            geometry, ctx.layout, p.theta, p.state.coefficients
        )
        return _assess_theta(ctx, theta, ctrl)

    p_start: Start | None = None
    if ctx.start is not None and geometry.n_slopes_identified > 0:
        scaled, _ = _warm_effects_from_start(geometry, ctx.layout, ctx.fallback)
        p_start = Start(scaled, "user")

    assess_fn = assess_prof if ctx.options.retry == 1 else None
    s_in, fam, opts, ab = ctx.layout.sample, ctx.family, ctx.options, ctx.absorbed
    candidate = ctx.work.run(
        "tau1_absorbed_profiled_solve",
        lambda: solve_profiled(
            s_in,
            fam,
            opts,
            ab,
            trigger=ctx.trigger,
            admit=ctx.admit,
            assess=assess_fn,
            start=p_start,
        ),
    )
    ctx.candidate = candidate
    # Its documented ValueError means that no retained evaluated point is available.
    try:
        sp = candidate.point().point
    except ValueError:
        return candidate, None
    return candidate, _effects_to_identified(
        geometry, ctx.layout, sp.theta, sp.state.coefficients
    )


def solve_cancelled_tau1(
    capture: CapturedFit,
    variance: CancelledFamily,
    options: SolverOptions,
    *,
    absorbed: AbsorbedOptions | Mapping[str, Any] | None = None,
    bounds: AttemptBounds | Tau1ExecutionBounds,
    reported: CancelledScaleProvider,
    work: WorkLog,
    start: FloatArray | None = None,
) -> Tau1SolveResult:
    """Solve the exact tau=1 / NB2(0) route via the configured joint-absorbed solver."""
    family, controls = _validate_tau1_family(variance), certificate_controls(options)
    execution = (
        bounds
        if isinstance(bounds, Tau1ExecutionBounds)
        else Tau1ExecutionBounds(bounds, None)
    )
    cb, coordinator = execution.certificate, execution.coordinator
    ab = resolve_absorbed_options(absorbed)
    org: Literal["index", "profiled"] = ab.absorbed
    geom = capture.admission.geometry
    fb = (
        np.zeros(geom.n_identified)
        if start is None
        else geom.checked_coefficients(start)
    )
    history: list[CancelledCertificate] = []
    candidate: IndexCandidate | ProfiledCandidate | None = None
    iters, rows = 0, 0
    ctx: _Tau1Context | None = None
    message = ""
    no_evaluated_point = False
    origin_failure: object | None = None
    try:
        layout = work.run("tau1_bridge_sample", lambda: _build_bridge_layout(capture))
        rows = int(layout.active_rows.size)
        if coordinator is None:
            raise ValueError("tau=1 requires a genuine inner coordinator origin")
        trigger, admit = coordinator(layout.sample, family)
        ctx = _Tau1Context(
            capture,
            layout,
            family,
            options,
            ab,
            cb,
            reported,
            work,
            fb,
            trigger,
            admit,
            start,
            history,
        )
        runner = (
            _run_index_organization if org == "index" else _run_profiled_organization
        )
        candidate, evaluated = runner(ctx)
        iters = candidate.iterations
        native_reason = _map_native_reason(candidate.reason)
        no_evaluated_point = evaluated is None
        if no_evaluated_point:
            message = candidate.attempts[candidate.retained].message
        theta = fb if evaluated is None else evaluated
    except (ValueError, FloatingPointError) as error:
        message = f"{type(error).__name__}: {error}"
        if isinstance(error, Tau1OriginError):
            origin_failure = error.outcome
        candidate = None if ctx is None else ctx.candidate
        iters = 0 if candidate is None else candidate.iterations
        theta, native_reason = fb, "numerical_breakdown"

    certified = certify_cancelled(
        capture,
        theta,
        controls,
        bounds=cb,
        reported=reported,
        work=work,
        variance=family,
    )
    history.append(certified.certificate)
    evals = _candidate_evaluations(candidate, len(history))
    if no_evaluated_point:
        status = NATIVE_STATUS[native_reason]
    elif message:
        status = "solver_error"
    else:
        status = certified.certificate.status(native_reason)
    return Tau1SolveResult(
        certified,
        tuple(history),
        status,
        iters,
        evals,
        org,
        candidate,
        rows,
        message,
        origin_failure,
    )
