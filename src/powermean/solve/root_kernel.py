"""The shared root-Newton kernel: one attempt, one stage with its retry, continuation.

The iteration of the cancelled solver, extracted unchanged so
that every system without a usable criterion runs the same one: the cancelled equation
outside ``tau in {1, 2}`` (:mod:`powermean.solve.cancelled_root`) and the joint
equation outside ``tau in [1, 2]`` (:mod:`powermean.solve.joint_root`).  Nothing here
knows a system.  A :class:`RootKernel` names three seams:

    equations   the fresh state of the system at a point under one stage variance,
                with its unnormalized moment ``S``
    direction   the Newton step ``delta`` of ``H delta = S`` at a state, ``H`` the
                actual negative Jacobian, by the system's own factorization
    whiten      ``u = L_M^-1 S``, ``||u||`` and ``phi = ||u||^2 / 2`` for the fixed
                positive-definite merit metric ``M = L_M L_M'``

and a :class:`RootCertifier` names the independent certification of a stage point.

One attempt (:func:`run_root_attempt`): the opening evaluation; then, until the native
criterion ``max |S| <= gradient_tol`` holds or ``max_iter`` steps are spent, the
direction and an Armijo backtracking on ``phi``, its change formed as
``(u_t - u)'(u_t + u) / 2`` to limit cancellation, with the constant ``root_armijo``,
the contraction ``backtrack_factor`` and at most ``root_max_backtracks`` trials, each
trial charged to the evaluation budget before it is made.  A trial whose evaluation is
not representable is contracted, never accepted.  The native reasons are those of
:mod:`powermean.solve.newton_trace`; a native stop certifies nothing.

One stage (:func:`run_root_stage`): an attempt with the iterations left, its fresh
certification, and the single retry of the convex solver's rule
(:mod:`powermean.solve.newton_retry`) when the attempt stopped natively short of the
certificate with a finite error and improved on the start; the retained point is
certified afresh before its status is composed.

Continuation (:func:`run_continuation`): when the direct stage on the target fails,
the policy is ``"fallback"``, the initializer was certified, the target is a power
variance and budgets remain, the stage variances of
:func:`~powermean.solve.cancelled_records.iter_continuation_variances` lead from the
initializer's own ``origin`` toward the target, each stage from the last certified
point, stopping at the first failed stage or spent budget.  The schedule and the order
of the work are those of the cancelled solver, whose origin is ``tau = 2``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.admission_records import WorkLog
from powermean.model.variance import CancelledFamily, FloatArray, PowerVariance
from powermean.solve.assessment import Assessment
from powermean.solve.cancelled_records import (
    RootAttemptRecord,
    RootOptions,
    iter_continuation_variances,
)
from powermean.solve.factorization import SingularJacobianError
from powermean.solve.newton_retry import improved, verdict
from powermean.solve.newton_trace import NativeReason
from powermean.solve.options import SolverOptions
from powermean.solve.quantities import euclidean_norm
from powermean.solve.status import Status

# ``u``, ``||u||`` and ``phi`` of a whitened moment.
Whitened = tuple[FloatArray, float, float]


class RootState(Protocol):
    """A fresh state of a system at one point: the point and its moment ``S``."""

    @property
    def theta(self) -> FloatArray:
        """The coefficients of the point, in the coordinates of the solve."""
        ...

    @property
    def score(self) -> FloatArray:
        """The unnormalized moment ``S`` at the point."""
        ...


@dataclass
class RootBudget:
    """Remaining root work, separate from convex pointwise-loss call kinds."""

    limit: int
    used: int = 0


@dataclass(frozen=True)
class RootKernel[StateT: RootState]:
    """The seams of one root solve (module docstring), its controls and its budget.

    ``budget`` is the evaluation budget shared by every attempt of the solve; it is
    charged before each evaluation is made.  ``work`` counts the operations.
    """

    equations: Callable[[FloatArray, CancelledFamily], StateT]
    direction: Callable[[StateT], FloatArray]
    whiten: Callable[[FloatArray], Whitened]
    options: SolverOptions
    root_options: RootOptions
    budget: RootBudget
    work: WorkLog


class RootCertifier[CertifiedT](Protocol):
    """The independent certification of a stage point, and what a stage reads of it."""

    def certify(self, variance: CancelledFamily, point: FloatArray) -> CertifiedT:
        """The fresh certificate of ``point`` on the equation of ``variance``."""
        ...

    def accepted(self, certified: CertifiedT) -> bool:
        """Whether the certificate passes at either level."""
        ...

    def assessment(self, certified: CertifiedT) -> Assessment:
        """The retry reading of the certificate."""
        ...

    def status(self, certified: CertifiedT, native: NativeReason) -> Status:
        """The status of the certificate beside the native reason of its attempt."""
        ...

    def point(self, certified: CertifiedT) -> FloatArray:
        """The point that the certificate assessed."""
        ...


@dataclass(frozen=True)
class RootRun[StateT: RootState, CertifiedT]:
    """One root solve, with optional canonical system-adapter call-through seams.

    The cancelled adapter preserves its existing attempt/stage mutation boundaries;
    those wrappers delegate back to this one shared numerical implementation.
    """

    kernel: RootKernel[StateT]
    iter_budget: RootBudget
    certifier: RootCertifier[CertifiedT]
    initializer_certified: bool
    attempt: Callable[[CancelledFamily, FloatArray, int], RootAttemptRecord] | None = (
        None
    )
    stage: Callable[[CancelledFamily, FloatArray], StageOutcome[CertifiedT]] | None = (
        None
    )


@dataclass(frozen=True, eq=False)
class StageOutcome[CertifiedT]:
    """Trace and certification of one variance stage (direct or continuation)."""

    variance: CancelledFamily
    attempts: tuple[RootAttemptRecord, ...]
    retained: int
    certified: CertifiedT
    status: Status
    iterations: int = 0
    evaluations: int = 0


# ------------------------------------------------------------------- one attempt


def converged[StateT: RootState](kernel: RootKernel[StateT], score: FloatArray) -> bool:
    """Apply native gradient_tol; final certificate tolerances are independent."""
    grad_inf = float(np.max(np.abs(score))) if score.size else 0.0
    return grad_inf <= kernel.options.gradient_tol


def _line_search_step[StateT: RootState](
    kernel: RootKernel[StateT],
    stage_variance: CancelledFamily,
    theta: FloatArray,
    delta: FloatArray,
    *,
    u_curr: FloatArray,
    phi_curr: float,
) -> tuple[StateT | None, FloatArray, float, int, NativeReason | None]:
    """Run difference-based Armijo backtracking from ``theta`` along ``delta``."""
    step_size, backtracks = 1.0, 0
    for _ in range(kernel.root_options.root_max_backtracks):
        if kernel.budget.used >= kernel.budget.limit:
            return None, u_curr, phi_curr, backtracks, "evaluation_limit"
        kernel.budget.used += 1
        backtracks += 1
        theta_trial = theta + step_size * delta

        def evaluate_trial(point: FloatArray = theta_trial) -> StateT:
            return kernel.equations(point, stage_variance)

        try:
            trial_state = kernel.work.run("root_equations", evaluate_trial)
            u_trial, _norm_trial, phi_trial = kernel.whiten(trial_state.score)
            delta_phi = 0.5 * float(np.dot(u_trial - u_curr, u_trial + u_curr))
        except (FloatingPointError, ValueError):
            step_size *= kernel.options.backtrack_factor
            continue

        rhs = -2.0 * kernel.root_options.root_armijo * step_size * phi_curr
        if delta_phi <= rhs or converged(kernel, trial_state.score):
            return trial_state, u_trial, phi_trial, backtracks, None
        step_size *= kernel.options.backtrack_factor
    return None, u_curr, phi_curr, backtracks, "line_search_failure"


def _stop_record(
    opening: FloatStorage,
    theta: FloatArray,
    native: NativeReason,
    counts: tuple[int, int, int],
    merit: tuple[float, float],
    *,
    msg: str,
) -> RootAttemptRecord:
    """Construct an immutable ``RootAttemptRecord``."""
    pt, (it, ev, bt), (m0, m1) = FloatStorage.store(theta), counts, merit
    return RootAttemptRecord(opening, pt, native, it, ev, bt, m0, m1, msg)


def run_root_attempt[StateT: RootState](
    kernel: RootKernel[StateT],
    stage_variance: CancelledFamily,
    start_theta: FloatArray,
    *,
    max_iter: int,
) -> RootAttemptRecord:
    """Execute one damped nonsymmetric root-Newton attempt on ``stage_variance``."""
    opening = FloatStorage.store(start_theta)
    theta, nan_pair = opening.view(), (float("nan"), float("nan"))
    if kernel.budget.used >= kernel.budget.limit:
        msg = "evaluation limit reached before opening evaluation"
        return _stop_record(
            opening, theta, "evaluation_limit", (0, 0, 0), nan_pair, msg=msg
        )
    evals_start = kernel.budget.used
    kernel.budget.used += 1
    try:
        state = kernel.work.run(
            "root_equations", lambda: kernel.equations(theta, stage_variance)
        )
        u_curr, _norm_curr, phi_curr = kernel.whiten(state.score)
    except (FloatingPointError, ValueError) as err:
        counts = (0, kernel.budget.used - evals_start, 0)
        return _stop_record(
            opening, theta, "numerical_breakdown", counts, nan_pair, msg=str(err)
        )

    initial_merit, total_bt = phi_curr, 0
    for step_idx in range(max_iter + 1):
        merit_pair = (initial_merit, phi_curr)
        if converged(kernel, state.score):
            counts = (step_idx, kernel.budget.used - evals_start, total_bt)
            return _stop_record(
                opening, theta, "native_stop", counts, merit_pair, msg="root tol met"
            )
        if step_idx == max_iter:
            break
        try:
            delta = kernel.direction(state)
        except (SingularJacobianError, FloatingPointError, ValueError) as err:
            reason: NativeReason = (
                "factorization_failure"
                if isinstance(err, SingularJacobianError)
                else "numerical_breakdown"
            )
            counts = (step_idx, kernel.budget.used - evals_start, total_bt)
            return _stop_record(
                opening, theta, reason, counts, merit_pair, msg=str(err)
            )
        next_state, u_next, phi_next, bt, stop_reason = _line_search_step(
            kernel, stage_variance, theta, delta, u_curr=u_curr, phi_curr=phi_curr
        )
        total_bt += bt
        if stop_reason is not None or next_state is None:
            ls_reason: NativeReason = (
                "line_search_failure" if stop_reason is None else stop_reason
            )
            counts = (step_idx, kernel.budget.used - evals_start, total_bt)
            msg = f"line search stopped with {ls_reason}"
            return _stop_record(opening, theta, ls_reason, counts, merit_pair, msg=msg)
        theta, state, u_curr, phi_curr = next_state.theta, next_state, u_next, phi_next
        _norm_curr = euclidean_norm(u_curr)

    counts = (max_iter, kernel.budget.used - evals_start, total_bt)
    return _stop_record(
        opening,
        theta,
        "iteration_limit",
        counts,
        (initial_merit, phi_curr),
        msg="root_max_iter reached",
    )


# --------------------------------------------------------------------- one stage


def _should_retry[StateT: RootState, CertifiedT](
    run: RootRun[StateT, CertifiedT],
    stage_variance: CancelledFamily,
    start_theta: FloatArray,
    first: RootAttemptRecord,
    first_cert: CertifiedT,
    *,
    stage_max_iter: int,
) -> tuple[bool, bool]:
    """Whether a stage attempt warrants its single retry; whether the start was read."""
    kernel, certifier = run.kernel, run.certifier
    can_retry = (
        kernel.options.retry == 1
        and not certifier.accepted(first_cert)
        and first.native == "native_stop"
        and first.iterations < stage_max_iter
        and run.iter_budget.used < run.iter_budget.limit
        and kernel.budget.used < kernel.budget.limit
    )
    if not can_retry:
        return False, False
    first_assess = certifier.assessment(first_cert)
    if not first_assess.finite_error:
        return False, False
    init_cert = certifier.certify(stage_variance, start_theta)
    return improved(first_assess, certifier.assessment(init_cert)), True


def run_root_stage[StateT: RootState, CertifiedT](
    run: RootRun[StateT, CertifiedT],
    stage_variance: CancelledFamily,
    start_theta: FloatArray,
) -> StageOutcome[CertifiedT]:
    """Run one root-Newton stage (including its optional retry) and certify it."""
    kernel, certifier = run.kernel, run.certifier
    rem_global = max(0, run.iter_budget.limit - run.iter_budget.used)
    stage_max_iter = min(kernel.root_options.root_max_iter, rem_global)
    first = _attempt(run, stage_variance, start_theta, stage_max_iter)
    run.iter_budget.used += first.iterations
    first_cert = certifier.certify(stage_variance, first.point)
    retry, assessed_start = _should_retry(
        run,
        stage_variance,
        start_theta,
        first,
        first_cert,
        stage_max_iter=stage_max_iter,
    )
    if not retry:
        if assessed_start:
            first_cert = certifier.certify(stage_variance, first.point)
        status = certifier.status(first_cert, first.native)
        it, ev = first.iterations, first.evaluations
        return StageOutcome(stage_variance, (first,), 0, first_cert, status, it, ev)

    first_assess = certifier.assessment(first_cert)
    rem_iter = min(
        stage_max_iter - first.iterations,
        max(0, run.iter_budget.limit - run.iter_budget.used),
    )
    second = _attempt(run, stage_variance, first.point, rem_iter)
    run.iter_budget.used += second.iterations
    second_cert = certifier.certify(stage_variance, second.point)
    _, keep_first = verdict(first_assess, certifier.assessment(second_cert))
    idx = 0 if keep_first else 1
    chosen_cert = first_cert if keep_first else second_cert
    chosen_att = first if keep_first else second
    if keep_first:
        chosen_cert = certifier.certify(stage_variance, chosen_att.point)
    status = certifier.status(chosen_cert, chosen_att.native)
    it = first.iterations + second.iterations
    ev = first.evaluations + second.evaluations
    return StageOutcome(
        stage_variance, (first, second), idx, chosen_cert, status, it, ev
    )


def _attempt[StateT: RootState, CertifiedT](
    run: RootRun[StateT, CertifiedT],
    variance: CancelledFamily,
    point: FloatArray,
    max_iter: int,
) -> RootAttemptRecord:
    """Keep a system adapter's canonical attempt seam, without copying the kernel."""
    if run.attempt is not None:
        return run.attempt(variance, point, max_iter)
    return run_root_attempt(run.kernel, variance, point, max_iter=max_iter)


# ------------------------------------------------------------------ continuation


def run_continuation[StateT: RootState, CertifiedT](
    run: RootRun[StateT, CertifiedT],
    target_variance: CancelledFamily,
    theta_start: FloatArray,
    direct: StageOutcome[CertifiedT],
    *,
    origin: float,
) -> tuple[tuple[StageOutcome[CertifiedT], ...], StageOutcome[CertifiedT], bool]:
    """The lazy continuation schedule from ``theta_start``, when it applies.

    Returns every stage run (the direct one first), the stage of the target variance
    (the direct one unless a continuation stage reached the target) and whether any
    continuation stage ran.
    """
    kernel, certifier = run.kernel, run.certifier
    if (
        certifier.accepted(direct.certified)
        or kernel.root_options.continuation != "fallback"
        or not run.initializer_certified
        or not isinstance(target_variance, PowerVariance)
        or kernel.budget.used >= kernel.budget.limit
        or run.iter_budget.used >= run.iter_budget.limit
    ):
        return (direct,), direct, False

    max_stages = min(
        kernel.budget.limit - kernel.budget.used,
        max(1, run.iter_budget.limit - run.iter_budget.used),
    )
    stages: list[StageOutcome[CertifiedT]] = [direct]
    final_stage, curr_theta = direct, theta_start
    for step_var in iter_continuation_variances(
        target_variance,
        kernel.root_options.continuation_step,
        max_stages,
        origin=origin,
    ):
        if (
            kernel.budget.used >= kernel.budget.limit
            or run.iter_budget.used >= run.iter_budget.limit
        ):
            break
        stage = (
            run_root_stage(run, step_var, curr_theta)
            if run.stage is None
            else run.stage(step_var, curr_theta)
        )
        stages.append(stage)
        if step_var == target_variance:
            final_stage = stage
        if not certifier.accepted(stage.certified):
            break
        curr_theta = certifier.point(stage.certified)
    return tuple(stages), final_stage, len(stages) > 1
