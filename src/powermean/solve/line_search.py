# line_search adapts NewtonSolver.line_search of scikit-learn, which carries:
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# The full notice is powermean/_vendor/SKLEARN_COPYING.

"""The backtracking line search of scikit-learn's Newton iteration, with its options.

A full step first, contraction until the Armijo condition holds, a bounded number of
trials, and near roundoff acceptance on a smaller gradient, scikit-learn's safeguard;
the options ``armijo``, ``backtrack_factor`` and ``max_backtracks`` have the defaults
``2**-11``, ``0.5`` and 21.
``NewtonSolver.line_search`` of the carried files (scikit-learn's code,
BSD-3-Clause, with the attribution above and the notice
``powermean/_vendor/SKLEARN_COPYING``) fixes its constants in its body;
:func:`line_search` is that method with the three options in their place, and its
arithmetic in its order:

    trial t            coef_old + t coef_newton,  raw_prediction + t (X coef_newton)
    sufficient decrease    loss(t) - loss_old <= t * armijo * (gradient' coef_newton)
    near roundoff          |loss(t) - loss_old| <= 16 eps |loss_old|  and
                           sum |gradient(t)| < sum |gradient_old|
    refused trial          t *= backtrack_factor, at most max_backtracks trials

``eps`` is the spacing of the doubles at one, of the dtype of the loss, as in the
source; the step starts at one, which multiplies as scikit-learn's integer one does.  At
the defaults every operation is scikit-learn's, so that the line search is the carried
one bit for bit, which the tests check.  The raw prediction ``X coef_newton`` of the
displacement holds no offset: the offset enters the index once, in the pointwise loss.

Two changes:

- the state that the solver holds (coefficients, raw prediction, loss and gradient) is
  replaced by a trial only when the trial is accepted; a search that refuses every
  trial, or that an exception interrupts, leaves the last accepted state, where the
  source left the last refused trial;
- a trial that the pointwise loss cannot evaluate, because a value is not a double,
  raises ``FloatingPointError`` and is refused.

The search is written on the attributes of the solver (:class:`SearchState`), so that
it needs no class of a provider; the adapter supplies the evaluation of a trial
(:class:`TrialLoss`) and the sum of the absolute gradient (:class:`AbsoluteSum`).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np

from powermean.model.variance import FloatArray
from powermean.solve.newton_trace import Tally
from powermean.solve.options import SolverOptions


class SearchState(Protocol):
    """The attributes of a ``NewtonSolver`` that the line search reads and writes."""

    coef: FloatArray
    coef_old: FloatArray
    coef_newton: FloatArray
    raw_prediction: FloatArray
    loss_value: np.float64
    loss_value_old: np.float64
    gradient: FloatArray
    gradient_old: FloatArray
    gradient_times_newton: np.float64
    use_fallback_lbfgs_solve: bool


class TrialLoss(Protocol):
    """The loss and its gradient at a trial: ``LinearModelLoss.loss_gradient``."""

    def __call__(
        self, coef: FloatArray, raw: FloatArray, /
    ) -> tuple[np.float64, FloatArray]:
        """The averaged loss and gradient at ``coef``, of raw prediction ``raw``."""
        ...


class AbsoluteSum(Protocol):
    """``sum |gradient|``, as scikit-learn's line search forms it."""

    def __call__(self, gradient: FloatArray, /) -> float:
        """The sum of the absolute values of ``gradient``."""
        ...


class Trial(NamedTuple):
    """A trial point, with its loss and gradient."""

    coef: FloatArray
    raw: FloatArray
    loss: np.float64
    gradient: FloatArray


def _trial(
    state: SearchState,
    step: float,
    direction: FloatArray,
    evaluate: TrialLoss,
    tally: Tally,
) -> Trial | None:
    """The trial at ``step``, as the source forms it, or ``None`` if refused.

    A trial that ``evaluate`` cannot represent is refused, as an infinite loss is, and
    counted with the trials evaluated; one that an exception interrupts is not.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        coef = state.coef_old + step * state.coef_newton
        raw = state.raw_prediction + step * direction
    try:
        loss, gradient = evaluate(coef, raw)
    except FloatingPointError:
        tally.trials += 1
        tally.refused += 1
        return None
    tally.trials += 1
    return Trial(coef, raw, loss, gradient)


def _acceptable(
    state: SearchState,
    trial: Trial,
    bound: float,
    eps: float,
    absolute_sum: AbsoluteSum,
) -> bool:
    """The two tests of scikit-learn: sufficient decrease, else the roundoff safeguard.

    ``loss - loss_old <= bound``; else, within ``eps |loss_old|`` of roundoff, a
    strictly smaller ``sum |gradient|``.  A NaN passes neither, as in scikit-learn.
    scikit-learn forms ``sum |gradient_old|`` once per search and keeps it; it is formed
    again here, the same reduction of the same array, hence the same double.
    """
    loss_improvement = trial.loss - state.loss_value_old
    if loss_improvement <= bound:
        return True
    tiny_loss = np.abs(state.loss_value_old * eps)
    if np.abs(loss_improvement) <= tiny_loss:
        sum_abs_grad_old = absolute_sum(state.gradient_old)
        return absolute_sum(trial.gradient) < sum_abs_grad_old
    return False


def line_search(
    state: SearchState,
    direction: FloatArray,
    evaluate: TrialLoss,
    options: SolverOptions,
    tally: Tally,
    *,
    absolute_sum: AbsoluteSum,
) -> bool:
    """One backtracking line search along the Newton step; ``True`` if a step is taken.

    Parameters
    ----------
    state : SearchState
        The provider's solver after ``inner_solve``: its accepted coefficients, raw
        prediction, loss and gradient, the Newton step and its product with the
        gradient.
    direction : array
        The raw prediction of the Newton step, ``X coef_newton``, without offset.
    evaluate : TrialLoss
        The averaged loss and gradient at a trial.
    options : SolverOptions
        ``armijo``, ``backtrack_factor`` and ``max_backtracks`` are read.
    tally : Tally
        Counts the search, its trials and the trials refused.
    absolute_sum : AbsoluteSum
        ``sum |gradient|`` of the safeguard near roundoff.

    Returns
    -------
    bool
        Whether a trial was accepted, which then replaced the state.  Otherwise the
        state is the accepted one, its references ``_old`` set to it, and the failure
        flag ``use_fallback_lbfgs_solve`` is set, as in the source.
    """
    eps = 16 * float(np.finfo(np.result_type(state.loss_value)).eps)
    armijo_term = options.armijo * state.gradient_times_newton
    state.coef_old = state.coef
    state.loss_value_old = state.loss_value
    state.gradient_old = state.gradient
    tally.line_searches += 1
    step = 1.0
    for _ in range(options.max_backtracks):
        trial = _trial(state, step, direction, evaluate, tally)
        bound = step * armijo_term
        if trial is not None and _acceptable(state, trial, bound, eps, absolute_sum):
            state.coef, state.raw_prediction = trial.coef, trial.raw
            state.loss_value, state.gradient = trial.loss, trial.gradient
            return True
        step *= options.backtrack_factor
    state.use_fallback_lbfgs_solve = True
    return False
