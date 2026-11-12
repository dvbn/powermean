"""The candidate of the convex Newton solver and the trace of its attempts.

A solver returns a candidate and its trace, and makes no
claim of convergence.  The provenance of a
result records the provider, the resolved options, the start and the trace (iterations,
line-search trials, retries, wall time).

A native stop is the solver's own test, the gradient criterion and the Newton decrement
of the carried iteration; its reason is a diagnostic of the attempt and never a status
of the certificate, which only the independent certificate assigns.  The reasons are:

    native_stop            the iteration met its own stopping test
    iteration_limit        the iterations allowed to the solve are spent
    evaluation_limit       the evaluations allowed to the solve are spent
    line_search_failure    no trial step of a line search was acceptable
    factorization_failure  the Newton system was singular or ill-conditioned, or the
                           pointwise curvature was mostly negative
    non_descent_direction  the Newton step was not a descent direction
    numerical_breakdown    a value of the iteration was not representable

At the three failures, scikit-learn would hand the problem to L-BFGS; the adapter runs
no other optimizer, so :attr:`NewtonCandidate.fallback_triggers` counts triggers and
no fallback is ever executed.  Each attempt keeps its last accepted state: the
coefficients, with the provider's raw prediction ``X gamma`` (without the offset), the
loss and its gradient at them, never the coefficients of a refused trial.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from powermean.model.variance import FloatArray
from powermean.solve.assessment import Assessment
from powermean.solve.options import Provider, SolverOptions
from powermean.solve.point_loss import Call
from powermean.solve.starts import Start

NativeReason = Literal[
    "native_stop",
    "iteration_limit",
    "evaluation_limit",
    "line_search_failure",
    "factorization_failure",
    "non_descent_direction",
    "numerical_breakdown",
]
# The stops at which scikit-learn's iteration calls its L-BFGS fallback.
FALLBACK_TRIGGERS: frozenset[NativeReason] = frozenset(
    {"line_search_failure", "factorization_failure", "non_descent_direction"}
)


@dataclass(frozen=True)
class Timing:
    """Elapsed (``wall``) and processor (``cpu``) seconds."""

    wall: float
    cpu: float


@dataclass(frozen=True)
class EvaluationCount:
    """The calls of the pointwise loss, by kind."""

    loss: int
    loss_gradient: int
    gradient_hessian: int
    gradient: int

    @property
    def total(self) -> int:
        """The calls of every kind, charged to the budget ``max_evaluations``."""
        return self.loss + self.loss_gradient + self.gradient_hessian + self.gradient

    @classmethod
    def spent(
        cls, before: Mapping[Call, int], after: Mapping[Call, int]
    ) -> EvaluationCount:
        """The calls made between two readings of ``Budget.calls``."""
        return cls(
            loss=after["loss"] - before["loss"],
            loss_gradient=after["loss_gradient"] - before["loss_gradient"],
            gradient_hessian=after["gradient_hessian"] - before["gradient_hessian"],
            gradient=after["gradient"] - before["gradient"],
        )


class Tally:
    """The counts of one attempt as it runs, kept when an exception ends it.

    ``steps`` are the accepted Newton steps, ``line_searches`` those started,
    ``trials`` the trial steps evaluated, of which ``refused`` were not representable;
    ``loss_evaluated`` holds once the loss of the start is known, and
    ``gradient_evaluated`` once the derivatives have been evaluated.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.line_searches = 0
        self.trials = 0
        self.refused = 0
        self.loss_evaluated = False
        self.gradient_evaluated = False


@dataclass(frozen=True, eq=False)
class AcceptedState:
    """The last accepted state of an attempt, read-only.

    ``coefficients`` are the last accepted coefficients, or the start of the attempt
    when no step was accepted.  ``raw_prediction`` is the provider's ``X gamma`` at
    them, ``loss`` and ``gradient`` the averaged criterion ``L/A`` and its gradient
    ``-S/A`` there (:mod:`powermean.solve.point_loss`).  Each is ``None`` when the
    attempt stopped before evaluating it: the loss and the raw prediction when its
    first evaluation failed, the gradient when that of the derivatives failed.
    """

    coefficients: FloatArray
    raw_prediction: FloatArray | None
    loss: float | None
    gradient: FloatArray | None


@dataclass(frozen=True, eq=False)
class NewtonAttempt:
    """One run of the iteration, from ``start`` to its native reason.

    ``iterations`` are the accepted Newton steps; ``line_searches`` the line searches
    started and ``trials`` the trial steps evaluated in them, of which
    ``refused_trials`` had a value that was not representable and were refused as an
    infinite loss would be; ``evaluations`` the calls of the pointwise loss during the
    attempt; ``warnings`` the diagnostics that the provider issued, as
    ``"Category: text"``; ``message`` the text of the exception that ended it, or
    ``""``; ``timing`` the time of the attempt, assessments excluded; ``assessment``
    the independent assessment of its state, when the retry needed one.
    """

    start: FloatArray
    state: AcceptedState
    reason: NativeReason
    message: str
    iterations: int
    line_searches: int
    trials: int
    refused_trials: int
    evaluations: EvaluationCount
    warnings: tuple[str, ...]
    timing: Timing
    assessment: Assessment | None = None


@dataclass(frozen=True, eq=False)
class NewtonCandidate:
    """The candidate of one solve, with its trace; not a certified solution.

    ``attempts`` holds one attempt, or two when the retry ran; ``retained`` is the index
    of the attempt whose state is the candidate (the first, when the second made it
    worse).  ``start``, ``options`` (all thirteen resolved values), ``provider`` and
    ``provider_version`` (``None`` for the bundled files) say how it was computed.
    ``initial_assessment`` is the assessment of the start, made only when the retry
    needed its score; ``no_measurable_progress`` says that the second attempt did not
    reduce the common error below 0.99 times that of the first.  ``assessments`` and
    ``assessment_timing`` are the independent assessments of the retry, made and timed
    apart from the iteration; ``timing`` is the time of the whole solve, assessments
    included.
    """

    start: Start
    options: SolverOptions
    provider: Provider
    provider_version: str | None
    attempts: tuple[NewtonAttempt, ...]
    retained: int
    initial_assessment: Assessment | None
    no_measurable_progress: bool
    assessments: int
    assessment_timing: Timing
    timing: Timing

    @property
    def state(self) -> AcceptedState:
        """The state of the retained attempt."""
        return self.attempts[self.retained].state

    @property
    def coefficients(self) -> FloatArray:
        """The candidate coefficients, those of the retained attempt."""
        return self.state.coefficients

    @property
    def iterations(self) -> int:
        """The accepted Newton steps of every attempt, charged to ``max_iter``."""
        return sum(attempt.iterations for attempt in self.attempts)

    @property
    def evaluations(self) -> EvaluationCount:
        """The calls of the pointwise loss in every attempt, by kind."""
        counts = [attempt.evaluations for attempt in self.attempts]
        return EvaluationCount(
            loss=sum(count.loss for count in counts),
            loss_gradient=sum(count.loss_gradient for count in counts),
            gradient_hessian=sum(count.gradient_hessian for count in counts),
            gradient=sum(count.gradient for count in counts),
        )

    @property
    def fallback_triggers(self) -> int:
        """The attempts that ended where scikit-learn starts L-BFGS; none started it."""
        return sum(attempt.reason in FALLBACK_TRIGGERS for attempt in self.attempts)
