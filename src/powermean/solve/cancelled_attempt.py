"""Internal cancelled attempts using the existing solve-wide retry rule (02 5.1).

This is an attempt primitive. Geometry, support, fresh certificate and inference
are caller gates. The supplied assessor is called by newton_retry, never replaced
by the provider's native test or by a success marker. Its evaluations are separate
from the optimization budget, just as in the existing joint solve.
"""

from __future__ import annotations

from collections.abc import Callable

from powermean.fe._cancel_storage import finite_vector
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray
from powermean.solve.assessment import Assessment
from powermean.solve.cancelled_operator import CancelledProblem
from powermean.solve.newton_convex import resolve_provider, run_cancelled_attempt
from powermean.solve.newton_retry import Attempts, run_attempts
from powermean.solve.newton_trace import NewtonAttempt
from powermean.solve.options import SolverOptions
from powermean.solve.point_loss import Budget


def run_cancelled_attempts(
    problem: CancelledProblem,
    coefficients: FloatArray,
    options: SolverOptions,
    *,
    assess: Callable[[FloatArray], Assessment] | None,
) -> Attempts:
    """Resolve the existing provider and run actual same-allowance attempts.

    Parameters
    ----------
    problem : CancelledProblem
        Owned private unit-multiplier tau=2 problem with supplied coordinates.
    coefficients : ndarray
        Start, detached once before the driver or assessor can read it.
    options : SolverOptions
        Existing controls, passed unchanged. No tolerance or budget is retuned.
    assess : callable or None
        Fresh independent original-system assessment required when retry=1.
        This primitive assigns no status and implements no certificate policy.

    Returns
    -------
    Attempts
        Existing retry record, including each last accepted state and assessment.
    """
    if type(problem) is not CancelledProblem or type(options) is not SolverOptions:
        raise TypeError("use a validated CancelledProblem and SolverOptions")
    problem.operator.plan.require(problem.operator.bounds)
    start = _immutable(
        finite_vector(coefficients, problem.operator.shape[1], "identified start")
    )
    provider = resolve_provider(options.newton_provider)
    budget = Budget(options.max_evaluations)

    def attempt(values: FloatArray, max_iter: int) -> NewtonAttempt:
        return run_cancelled_attempt(
            provider, problem, values, options, max_iter=max_iter, budget=budget
        )

    return run_attempts(
        attempt,
        assess,
        start,
        retry=options.retry,
        max_iter=options.max_iter,
        evaluations_remain=lambda: budget.used < budget.limit,
    )
