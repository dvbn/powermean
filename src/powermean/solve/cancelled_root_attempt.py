"""One nonsymmetric H delta=S attempt with fixed-SPD difference Armijo merit.

Equilibrated LU, explicit native gradient control and bounded
backtracking; fresh final certification belongs to the stage orchestrator.  The
iteration is the shared kernel of :mod:`powermean.solve.root_kernel`, step for step;
this module supplies its cancelled seams: the fresh identified equations of the
capture, the materialized actual negative Jacobian solved by equilibrated LU, and the
dense whitening by the fixed merit factor, with the work labels of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe._cancel_storage import finite_product
from powermean.fe.admission_records import WorkLog
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
)
from powermean.solve.assessment import CertificateControls
from powermean.solve.cancelled_certificate import (
    CancelledScaleProvider,
    materialize_hessian,
    whitened_score,
)
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.cancelled_records import (
    CancelledCertificate,
    RootAttemptRecord,
    RootOptions,
)
from powermean.solve.factorization import factorize
from powermean.solve.options import SolverOptions
from powermean.solve.root_kernel import (
    RootBudget,
    RootKernel,
    Whitened,
    run_root_attempt,
)
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import IdentifiedState

# The name under which the cancelled stages construct their budgets.
_RootBudget = RootBudget


@dataclass(frozen=True)
class _RootContext:
    """Shared solver context for root-Newton attempts and stages."""

    capture: CapturedFit
    merit_cholesky: FloatArray
    options: SolverOptions
    root_options: RootOptions
    controls: CertificateControls
    bounds: AttemptBounds
    reported: CancelledScaleProvider
    budget: RootBudget
    iter_budget: RootBudget
    initializer_certified: bool
    work: WorkLog
    history: list[CancelledCertificate]


def _newton_direction(ctx: _RootContext, state: IdentifiedState) -> FloatArray:
    """Compute the nonsymmetric root-Newton step ``H delta = S``."""
    hessian = materialize_hessian(state, ctx.work, "root_hessian_materialization")
    factor = ctx.work.run("root_lu", lambda: factorize(hessian, ctx.options.rcond_min))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return finite_product(
            ctx.work.run("root_direction_solve", lambda: factor.solve(state.score))
        )


def root_kernel(ctx: _RootContext) -> RootKernel[IdentifiedState]:
    """The shared kernel with the cancelled seams of ``ctx`` and its live budget."""

    def equations(theta: FloatArray, variance: CancelledFamily) -> IdentifiedState:
        return ctx.capture.equations(theta, variance=variance)

    def direction(state: IdentifiedState) -> FloatArray:
        return _newton_direction(ctx, state)

    def whiten(score: FloatArray) -> Whitened:
        return whitened_score(score, ctx.merit_cholesky, ctx.work)

    return RootKernel(
        equations,
        direction,
        whiten,
        ctx.options,
        ctx.root_options,
        ctx.budget,
        ctx.work,
    )


def _run_root_attempt(
    ctx: _RootContext,
    stage_variance: CancelledFamily,
    start_theta: FloatArray,
    *,
    max_iter: int,
) -> RootAttemptRecord:
    """Execute one damped nonsymmetric root-Newton attempt on ``stage_variance``."""
    return run_root_attempt(
        root_kernel(ctx), stage_variance, start_theta, max_iter=max_iter
    )
