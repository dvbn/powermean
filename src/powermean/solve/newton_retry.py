"""The attempts of one convex Newton solve: the first, and the retry.

One retry of the same method when a native stop misses the certificate but has
improved the score; its cost is recorded.  The rule has two constants, which are not
options:

- a second attempt runs only when ``retry`` is 1, the first attempt stopped natively,
  its independent assessment is not accepted, its common error is finite, its score is
  below ``RETRY_FACTOR`` (0.99) times the score of the start, floored at
  ``SCORE_FLOOR`` (``1e-300``), and iterations and evaluations remain;
- it is the same method, with the same provider and options, from the coefficients of
  the first, and it receives what remains of the budgets of the solve, not new ones;
- unless the second assessment is accepted, the second attempt made no measurable
  progress when its common error is not finite or not below 0.99 times that of the
  first; the first point is then kept if, in addition, the second error is larger or
  not finite.

The assessments are those of :mod:`powermean.solve.assessment`, made only where the
rule reads one: the first attempt's after a native stop, the start's when the score
must be compared, the second attempt's after it.  With ``retry`` 0 none is made.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import NamedTuple, Protocol

from powermean.model.variance import FloatArray
from powermean.solve.assessment import Assessment
from powermean.solve.newton_trace import NewtonAttempt

# A retry needs the score of the first attempt below 0.99 times that
# of the start, and a second attempt that does not bring the common error below 0.99
# times the first's made no measurable progress.
RETRY_FACTOR = 0.99
# The score of the start is at least 1e-300, so that a start that is
# an exact root still has a positive score to compare with.
SCORE_FLOOR = 1e-300


def improved(first: Assessment, initial: Assessment) -> bool:
    """The measurable improvement: score below 0.99 times that of the start.

    The score of the start is floored at ``SCORE_FLOOR`` with Python's ``max``,
    which keeps a NaN score, and no score is then below it.
    """
    return first.score < RETRY_FACTOR * max(initial.score, SCORE_FLOOR)


def verdict(first: Assessment, second: Assessment) -> tuple[bool, bool]:
    """After the second attempt: (no measurable progress, keep the first point).

    An accepted second point is kept, with progress.  Otherwise there is no measurable
    progress when the second error is not finite or not below 0.99 times the first,
    and the first point is kept when the second error is also larger or not finite.
    """
    if second.accepted:
        return False, False
    error, previous = second.error, first.error
    if math.isfinite(error) and error < RETRY_FACTOR * previous:
        return False, False
    return True, previous < error or not math.isfinite(error)


class AttemptRunner(Protocol):
    """One attempt of the iteration from ``coefficients``, with ``max_iter`` steps."""

    def __call__(self, coefficients: FloatArray, max_iter: int, /) -> NewtonAttempt:
        """The attempt, with its last accepted state and native reason."""
        ...


class Attempts(NamedTuple):
    """The attempts of a solve, the one retained, and what the rule read."""

    attempts: tuple[NewtonAttempt, ...]
    retained: int
    initial: Assessment | None
    no_measurable_progress: bool


def run_attempts(
    run: AttemptRunner,
    assess: Callable[[FloatArray], Assessment] | None,
    start: FloatArray,
    *,
    retry: int,
    max_iter: int,
    evaluations_remain: Callable[[], bool],
) -> Attempts:
    """The first attempt and, if the retry rule allows it, the second.

    Parameters
    ----------
    run : AttemptRunner
        Runs one attempt of the same method; the budget of evaluations is its own and
        spans both attempts.
    assess : callable or None
        The independent assessment of a point.  Required with ``retry`` 1, since the
        rule cannot be decided without it, and not called with ``retry`` 0.
    start : array
        The start of the first attempt.
    retry : int
        The option ``retry``: 0 or 1.
    max_iter : int
        The iterations allowed to the whole solve.
    evaluations_remain : callable
        Whether the budget of evaluations has a call left.

    Returns
    -------
    Attempts
        One or two attempts, each carrying the assessment that the rule made of it.

    Raises
    ------
    ValueError
        ``retry`` is 1 and no assessment is given; nothing is run.
    """
    if retry == 1 and assess is None:
        raise ValueError("retry=1 needs an independent assessment of the candidate")
    first = run(start, max_iter)
    if assess is None or retry == 0 or first.reason != "native_stop":
        return Attempts((first,), 0, None, False)
    assessed = assess(first.state.coefficients)
    first = replace(first, assessment=assessed)
    budgets_remain = first.iterations < max_iter and evaluations_remain()
    if assessed.accepted or not assessed.finite_error or not budgets_remain:
        return Attempts((first,), 0, None, False)
    initial = assess(start)
    if not improved(assessed, initial):
        return Attempts((first,), 0, initial, False)
    second = run(first.state.coefficients, max_iter - first.iterations)
    reassessed = assess(second.state.coefficients)
    second = replace(second, assessment=reassessed)
    no_progress, keep_first = verdict(assessed, reassessed)
    return Attempts((first, second), 0 if keep_first else 1, initial, no_progress)
