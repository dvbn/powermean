"""Nonsymmetric root-Newton solver for general cancelled working variances.

The attempt, the stage retry and the continuation schedule are the shared kernel of
:mod:`powermean.solve.root_kernel`, step for step; this module supplies the cancelled
certification of stage points and keeps the solver's records and its order of work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.admission_records import WorkLog
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    NB2Variance,
    PowerVariance,
)
from powermean.solve.assessment import Assessment, certificate_controls
from powermean.solve.cancelled_certificate import (
    CancelledScaleProvider,
    CertifiedCancelledPoint,
    build_merit_cholesky,
    certify_cancelled,
    failed_root_result,
    prepare_root_initializer,
)
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.cancelled_records import (
    CONTINUATION_POLICIES,
    MERIT_METRICS,
    CancelledCertificate,
    ContinuationPolicy,
    MeritMetric,
    RootAttemptRecord,
    RootExecutionBounds,
    RootInitializerRecord,
    RootOptions,
    RootSolveResult,
    RootStageRecord,
    resolve_root_options,
)
from powermean.solve.cancelled_root_attempt import (
    _RootBudget,
    _RootContext,
    _run_root_attempt,
    root_kernel,
)
from powermean.solve.newton_trace import NativeReason
from powermean.solve.options import SolverOptions
from powermean.solve.root_kernel import (
    RootRun,
    StageOutcome,
    run_continuation,
    run_root_stage,
)
from powermean.solve.status import Status
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import IdentifiedState

__all__ = [
    "CONTINUATION_POLICIES",
    "MERIT_METRICS",
    "ContinuationPolicy",
    "MeritMetric",
    "RootAttemptRecord",
    "RootExecutionBounds",
    "RootInitializerRecord",
    "RootOptions",
    "RootSolveResult",
    "RootStageRecord",
    "resolve_root_options",
    "solve_cancelled_root",
]

# The ``tau`` of the cancelled solver's certified initializer, its Gamma start.
GAMMA_ORIGIN = 2.0


def _certify_stage_point(
    ctx: _RootContext, stage_variance: CancelledFamily, point: FloatArray
) -> CertifiedCancelledPoint:
    """Certify a stage point on ``stage_variance`` with the fixed merit metric."""
    certified = certify_cancelled(
        ctx.capture,
        point,
        ctx.controls,
        bounds=ctx.bounds,
        reported=ctx.reported,
        work=ctx.work,
        variance=stage_variance,
        merit_cholesky=ctx.merit_cholesky,
    )
    ctx.history.append(certified.certificate)
    return certified


@dataclass(frozen=True, eq=False)
class _StageCertifier:
    """The cancelled certification of stage points, as the shared kernel reads it."""

    ctx: _RootContext

    def certify(
        self, variance: CancelledFamily, point: FloatArray
    ) -> CertifiedCancelledPoint:
        """Certify ``point`` on ``variance``, recorded in the solve's history."""
        return _certify_stage_point(self.ctx, variance, point)

    def accepted(self, certified: CertifiedCancelledPoint) -> bool:
        """Whether the original point passes all gates at either level."""
        return certified.certificate.accepted

    def assessment(self, certified: CertifiedCancelledPoint) -> Assessment:
        """The existing retry arithmetic of the certificate."""
        return certified.certificate.assessment()

    def status(
        self, certified: CertifiedCancelledPoint, native: NativeReason
    ) -> Status:
        """The existing status precedence of the certificate."""
        return certified.certificate.status(native)

    def point(self, certified: CertifiedCancelledPoint) -> FloatArray:
        """The identified point that the certificate assessed."""
        return certified.certificate.gamma


def _root_run(ctx: _RootContext) -> RootRun[IdentifiedState, CertifiedCancelledPoint]:
    """The shared kernel's run over the live budgets and history of ``ctx``."""

    def attempt(
        variance: CancelledFamily, point: FloatArray, max_iter: int
    ) -> RootAttemptRecord:
        return _run_root_attempt(ctx, variance, point, max_iter=max_iter)

    def stage(
        variance: CancelledFamily, point: FloatArray
    ) -> StageOutcome[CertifiedCancelledPoint]:
        record = _run_root_stage(ctx, variance, point)
        return StageOutcome(
            record.variance,
            record.attempts,
            record.retained,
            record.certified,
            record.status,
            record.iterations,
            record.evaluations,
        )

    return RootRun(
        root_kernel(ctx),
        ctx.iter_budget,
        _StageCertifier(ctx),
        ctx.initializer_certified,
        attempt,
        stage,
    )


def _stage_record(outcome: StageOutcome[CertifiedCancelledPoint]) -> RootStageRecord:
    """The cancelled record of one stage outcome, field for field."""
    return RootStageRecord(
        outcome.variance,
        outcome.attempts,
        outcome.retained,
        outcome.certified,
        outcome.status,
        outcome.iterations,
        outcome.evaluations,
    )


def _run_root_stage(
    ctx: _RootContext, stage_variance: CancelledFamily, start_theta: FloatArray
) -> RootStageRecord:
    """Run one root-Newton stage (including optional retry) and certify."""
    return _stage_record(run_root_stage(_root_run(ctx), stage_variance, start_theta))


def _run_continuation_fallback(
    ctx: _RootContext,
    target_variance: CancelledFamily,
    theta_start: FloatArray,
    direct: RootStageRecord,
) -> tuple[tuple[RootStageRecord, ...], RootStageRecord, bool]:
    """Execute the lazy continuation schedule from ``theta_start`` when applicable."""
    opened = StageOutcome(
        direct.variance,
        direct.attempts,
        direct.retained,
        direct.certified,
        direct.status,
        direct.iterations,
        direct.evaluations,
    )
    stages, final, used = run_continuation(
        _root_run(ctx), target_variance, theta_start, opened, origin=GAMMA_ORIGIN
    )
    records = tuple(
        direct if stage is opened else _stage_record(stage) for stage in stages
    )
    position = next(index for index, stage in enumerate(stages) if stage is final)
    return records, records[position], used


def _validate_root_family(variance: CancelledFamily) -> None:
    """Refuse families that belong to the convex tau=2 or absorbed tau=1 routes."""
    if isinstance(variance, PowerVariance) and variance.tau in (1.0, 2.0):
        raise ValueError("tau in {1, 2} belongs to convex/absorbed routes")
    if isinstance(variance, NB2Variance) and variance.kappa == 0.0:
        raise ValueError("NB2Variance(0.0) belongs to the tau=1 absorbed route")


def solve_cancelled_root(
    capture: CapturedFit,
    variance: CancelledFamily,
    options: SolverOptions,
    root_options: RootOptions | Mapping[str, Any] | None = None,
    *,
    bounds: AttemptBounds | RootExecutionBounds,
    reported: CancelledScaleProvider,
    work: WorkLog,
    start: FloatArray | None = None,
) -> RootSolveResult:
    """Solve the general cancelled equation ``S(theta) = 0`` via root Newton."""
    _validate_root_family(variance)
    ro = resolve_root_options(root_options)
    eb = (
        bounds
        if isinstance(bounds, RootExecutionBounds)
        else RootExecutionBounds(certificate=bounds)
    )
    cb, ctrl = eb.certificate, certificate_controls(options)
    history: list[CancelledCertificate] = []
    init_rec = prepare_root_initializer(
        capture, options, eb, reported=reported, work=work, history=history, start=start
    )
    merit_chol: FloatArray | None = None
    if init_rec.kind != "gamma_tau2" or init_rec.certified:
        try:
            merit_chol = build_merit_cholesky(
                capture, init_rec.point, ro.merit_metric, work
            )
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            merit_chol = None
    if merit_chol is None:
        fc, fr = failed_root_result(
            capture,
            variance,
            init_rec,
            controls=ctrl,
            bounds=cb,
            reported=reported,
            work=work,
            root_options=ro,
        )
        history.append(fc.certificate)
        return replace(fr, history=tuple(history))

    ms = FloatStorage.store(merit_chol)
    rem_ev = _RootBudget(max(0, options.max_evaluations - init_rec.evaluations))
    rem_it = _RootBudget(max(0, options.max_iter - init_rec.iterations))
    ctx = _RootContext(
        capture,
        ms.view(),
        options,
        ro,
        ctrl,
        cb,
        reported,
        rem_ev,
        rem_it,
        init_rec.certified,
        work,
        history,
    )
    direct = _run_root_stage(ctx, variance, init_rec.point)
    stages, final_st, cont = _run_continuation_fallback(
        ctx, variance, init_rec.point, direct
    )
    final = final_st.certified
    if cont and stages[-1].variance != variance:
        final = _certify_stage_point(ctx, variance, final.certificate.gamma)
    final_status = final.certificate.status(final_st.attempts[final_st.retained].native)
    r_it = sum(st.iterations for st in stages)
    r_ev = sum(st.evaluations for st in stages)
    t_it, t_ev = init_rec.iterations + r_it, init_rec.evaluations + r_ev
    return RootSolveResult(
        final,
        tuple(history),
        final_status,
        ro.merit_metric,
        replace(ro),
        ms,
        init_rec,
        stages,
        cont,
        t_it,
        t_ev,
        r_it,
        r_ev,
    )
