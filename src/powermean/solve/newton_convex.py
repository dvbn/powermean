# _iterate adapts the loop of NewtonSolver.solve of scikit-learn, which carries:
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# The full notice is powermean/_vendor/SKLEARN_COPYING.

"""The Newton adapter of the convex joint system: scikit-learn's damped Newton.

The joint system without effects,
with a power variance of ``tau`` in ``[1, 2]`` or NB2, is solved from one start
by scikit-learn's Newton-Cholesky iteration.  This is the one module of the package
that imports :mod:`powermean._vendor`.

Providers.  One damped Newton algorithm, whose iteration comes from one of
two providers: ``"bundled"``, the default, is the ``NewtonCholeskySolver`` that the
package carries; ``"sklearn"`` is the ``NewtonCholeskySolver`` of the
installed scikit-learn, admitted only for a version in
:data:`VERIFIED_SKLEARN_VERSIONS`.  Both wrap the pointwise loss in the carried
``LinearModelLoss``.  Its curvature warning counts the negative pointwise hessians,
where that of scikit-learn 1.9.1 also counts the zero curvature that the zero rows of
the power family at ``tau = 2`` legitimately have; the installed library is used as it
is, and its ``LinearModelLoss`` is not.  A missing or unlisted scikit-learn is refused,
never replaced, and the bundled path imports nothing of scikit-learn; no provider
falls back on the other.

The iteration.  The provider's solver supplies ``setup``, ``update_gradient_hessian``,
``inner_solve`` (``scipy.linalg.solve(H, -g, check_finite=False, assume_a="sym")``,
whatever the name of the class, then the descent test), ``check_convergence`` (the
gradient criterion ``gradient_tol`` and the Newton decrement) and ``finalize``; the
carried ``LinearModelLoss`` averages the pointwise loss of
:mod:`powermean.solve.point_loss`.  The adapter supplies the loop of
``NewtonSolver.solve`` (scikit-learn's code, BSD-3-Clause, with the attribution above
and the notice ``powermean/_vendor/SKLEARN_COPYING``), statement for statement, and the
line search of :mod:`powermean.solve.line_search`, that of scikit-learn with the options
``armijo``, ``backtrack_factor`` and ``max_backtracks``; neither issues scikit-learn's
warning of an iteration limit or of a failed search, which are the reasons of the
attempt.  At the defaults an attempt that stops natively is the carried ``solve``, bit
for bit, as the tests check.  Three changes:

- no fallback: where scikit-learn hands a failure to L-BFGS, the attempt stops with
  its reason; the provider's ``solve``, which holds that fallback in scikit-learn, is
  never called, so no other optimizer runs, whatever the provider;
- the last accepted state: the line search replaces the state only by an accepted
  trial, so that a refused, failed or interrupted search keeps the last accepted step;
- a trial whose value is not representable is refused: where the package's rows raise
  ``FloatingPointError``, the trial is refused as the Armijo test refuses an infinite
  loss.

Budgets, retry and result.  ``max_iter`` counts accepted Newton steps and
``max_evaluations`` the calls of the pointwise loss, each over the whole solve.
The retry is that of :mod:`powermean.solve.newton_retry`, decided with the independent
assessment of :mod:`powermean.solve.assessment`, to which the controls of the
certificate go.  An attempt ends with a native reason of
:mod:`powermean.solve.newton_trace`; two exceptions, the evaluation limit and a
value that is not representable, end it too, and no other is caught.  The result is a
candidate with its trace, never a claim of convergence: only the certificate decides
whether it is a solution.

The profiled organization.
:func:`run_profiled_attempt` drives the same iteration, line search
and providers with the profiled linear loss of
:mod:`powermean.solve.profiled_linear_loss` (whose class this module still provides),
one object for both providers, built here with the provider's carried
``LinearModelLoss`` and the carried ``sandwich_dot`` handed over explicitly, so that
this module remains the only one that imports the carried files; and with
:class:`ProfiledEvaluation`, the whole-point state of the attempt
(:mod:`powermean.solve.profiled_loss`).  The profiled loss charges each evaluation to
the solve's one budget before its work.  Three hooks, each skipped on
the route without effects: :func:`search` brackets a line search (``begin``,
then ``end`` with the
accepted trial or ``None``), and :func:`_iterate` calls ``started`` after the setup and
``stepped`` after each native convergence test.  The step, search and acceptance
arithmetic is unchanged.  Two exceptions end a profiled attempt besides those two:
:class:`InnerLimitError` (an inner sweep allowance spent) and :class:`IndicationError`
(an existence indication).

The fitted-index organization.  :func:`run_index_attempt` drives the same
loop, line search and providers with an :class:`IndexEvaluation`, the whole-point state
of the attempt (:mod:`powermean.solve.absorbed_loss`), to which this module hands the
provider's carried ``LinearModelLoss`` and the carried ``sandwich_dot``
(:meth:`IndexEvaluation.provider_loss`), so that it remains the only module that imports
the carried files.  The provider's ``setup``, ``update_gradient_hessian``,
``inner_solve`` and ``finalize`` run at their usual places on the slopes' Schur view (K
and K-by-K storage); the loop calls the evaluation's representation seams where the
index route differs, and the other routes skip them: after the setup (the
start's full index replaces the slope-only prediction), after the inner solve (the full
direction and its derivative and decrement, the provider's refusal kept), in place of
the search on the slopes (the existing :func:`~powermean.solve.line_search.line_search`
along the full direction, unchanged) and of the provider's convergence method (the two
existing formulas on the full score and the unhalved origin decrement).  Without a slope
column the provider's inner solve, a factorization of an empty system, is skipped
explicitly; the provider's convergence method never runs on this route, whose native
test is the adapter's.  The other routes are unchanged.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import time
import warnings
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, cast

import numpy as np

from powermean import messages
from powermean._vendor import _scipy, vendor_linear_loss, vendor_newton
from powermean._vendor._typing import BaseLoss
from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import finite_vector
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.assessment import AssessmentLog, Assessor, certificate_controls
from powermean.solve.cancelled_loss import CancelledLinearLoss
from powermean.solve.cancelled_operator import CancelledProblem, ImplicitNewtonProblem
from powermean.solve.line_search import AbsoluteSum, line_search
from powermean.solve.newton_retry import run_attempts
from powermean.solve.newton_trace import (
    AcceptedState,
    EvaluationCount,
    NativeReason,
    NewtonAttempt,
    NewtonCandidate,
    Tally,
    Timing,
)
from powermean.solve.options import Provider, SolverOptions
from powermean.solve.point_loss import Budget, EvaluationLimitError, PointLoss
from powermean.solve.profiled_linear_loss import (
    Admissions,
    PointEvaluation,
    ProfiledLinearLoss,
    Sandwich,
)
from powermean.solve.starts import Start
from powermean.systems.joint import JointSystem

# The names of this module, and those of the profiled linear loss defined in
# solve/profiled_linear_loss.py, whose import route from here is kept.
__all__ = [
    "BUNDLED",
    "SKLEARN_DISTRIBUTION",
    "SKLEARN_NEWTON",
    "VERIFIED_SKLEARN_VERSIONS",
    "Hooks",
    "IndexEvaluation",
    "IndicationError",
    "InnerLimitError",
    "LinearLoss",
    "NewtonProblem",
    "NewtonProvider",
    "ProfiledEvaluation",
    "ProfiledIteration",
    "ProfiledLinearLoss",
    "ProfiledStop",
    "Solver",
    "Transaction",
    "absolute_sum",
    "installed_provider",
    "installed_version",
    "resolve_provider",
    "run_attempt",
    "run_index_attempt",
    "run_profiled_attempt",
    "search",
    "solve",
]

Solver = vendor_newton.NewtonCholeskySolver
LinearLoss = vendor_linear_loss.LinearModelLoss

SKLEARN_DISTRIBUTION = "scikit-learn"
SKLEARN_NEWTON = "sklearn.linear_model._glm._newton_solver"
# The versions of scikit-learn whose NewtonCholeskySolver, with the carried linear loss,
# reproduced the bundled provider bit for bit on all 24 cases of the parity of the
# providers on the package's loss: 1.4.2, the minimum declared version, and 1.9.1.
# No range is implied, and no other version is listed; the parity test checks a
# listed version whenever it is installed.
VERIFIED_SKLEARN_VERSIONS: frozenset[str] = frozenset({"1.4.2", "1.9.1"})


# ---------------------------------------------------------------------- providers


@dataclass(frozen=True)
class NewtonProvider:
    """The Newton class of one provider, the linear loss that it drives, their origin.

    ``solver`` is the ``NewtonCholeskySolver`` whose iteration runs; ``linear_loss``
    wraps the pointwise loss, the carried ``LinearModelLoss`` for both providers.
    ``version`` is that of the installed scikit-learn, ``None`` for the bundled files.
    """

    name: Provider
    version: str | None
    solver: type[Solver]
    linear_loss: type[LinearLoss]


BUNDLED = NewtonProvider("bundled", None, Solver, LinearLoss)


def installed_version() -> str | None:
    """The version of the installed scikit-learn, from its metadata, or ``None``."""
    try:
        return importlib.metadata.version(SKLEARN_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_provider() -> NewtonProvider:
    """The iteration of the installed scikit-learn, without the gate of the versions.

    Its ``NewtonCholeskySolver``, imported by name, with the carried
    ``LinearModelLoss`` of the bundled provider: only the iteration comes from
    scikit-learn, whose own linear loss, which its module imports, is not used.  For
    the parity test; the adapter reaches it through :func:`resolve_provider` only.
    The class is typed as the bundled one, whose interface the parity test checks on
    each version that the gate admits.
    """
    version = importlib.metadata.version(SKLEARN_DISTRIBUTION)
    newton = importlib.import_module(SKLEARN_NEWTON)
    solver = cast("type[Solver]", newton.NewtonCholeskySolver)
    return NewtonProvider("sklearn", version, solver, LinearLoss)


def resolve_provider(
    name: Provider, *, verified: frozenset[str] = VERIFIED_SKLEARN_VERSIONS
) -> NewtonProvider:
    """The provider of the option ``newton_provider``, or its refusal.

    ``"sklearn"`` needs an installed scikit-learn of a version in ``verified``;
    otherwise ``SpecificationError`` (``specification.invalid_value``) names the one
    provider available, nothing is substituted and nothing of scikit-learn imported.
    """
    if name == "bundled":
        return BUNDLED
    version = installed_version()
    if version is None or version not in verified:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "newton_provider",
                "value": name,
                "valid": messages.format_keys(["bundled"]),
            },
        )
    return installed_provider()


# ------------------------------------------------------------------- the iteration


class NewtonProblem(NamedTuple):
    """The arrays of the iteration: the rows of positive weight of the system.

    ``weights`` are those rows' weights, which :func:`solve` always passes, or ``None``.
    """

    design: FloatArray
    z: FloatArray
    weights: FloatArray | None


def absolute_sum(gradient: FloatArray) -> float:
    """``sum |g|`` by SciPy's norm of order one, as in scikit-learn's line search.

    SciPy refuses a gradient that is not finite with ``ValueError``; it raises
    ``FloatingPointError`` here, reported as a numerical breakdown.
    """
    if not np.isfinite(gradient).all():
        raise FloatingPointError("gradient is not representable in float64")
    return _scipy.linalg.norm(gradient, ord=1)


def search(
    solver: Solver,
    problem: NewtonProblem,
    options: SolverOptions,
    tally: Tally,
    *,
    transaction: Transaction | None = None,
) -> bool:
    """The line search of :mod:`powermean.solve.line_search` on the provider's solver.

    The raw prediction of the Newton step and the loss of a trial are those of the
    provider's ``LinearModelLoss``, called as scikit-learn's line search calls it.  With
    a ``transaction`` (the profiled route) the search is bracketed: ``begin`` before its
    first trial, ``end`` after it with the accepted coefficients, or ``None`` when every
    trial was refused or an exception interrupted it; nothing is caught.
    """

    def evaluate(coef: FloatArray, raw: FloatArray) -> tuple[np.float64, FloatArray]:
        return solver.linear_loss.loss_gradient(
            coef=coef,
            X=problem.design,
            y=problem.z,
            sample_weight=problem.weights,
            l2_reg_strength=solver.l2_reg_strength,
            n_threads=solver.n_threads,
            raw_prediction=raw,
        )

    _, _, direction = solver.linear_loss.weight_intercept_raw(
        solver.coef_newton, problem.design
    )
    if transaction is None:
        return line_search(
            solver, direction, evaluate, options, tally, absolute_sum=absolute_sum
        )
    transaction.begin()
    accepted = False
    try:
        accepted = line_search(
            solver, direction, evaluate, options, tally, absolute_sum=absolute_sum
        )
    finally:
        transaction.end(solver.coef if accepted else None)
    return accepted


def _inner_failure(solver: Solver) -> NativeReason:
    """The reason of a failed Newton step.

    An ascending step is ``non_descent_direction``; a singular system or a curvature
    mostly negative, ``factorization_failure``.  Without a new step, the product of the
    gradient and the step is that of the last accepted one, not positive, or absent.
    """
    product: float = getattr(solver, "gradient_times_newton", 0.0)
    return "non_descent_direction" if product > 0 else "factorization_failure"


def _step(
    solver: Solver, problem: NewtonProblem, index: IndexEvaluation | None
) -> NativeReason | None:
    """The Newton step, or the reason that refuses it.

    The provider's inner solve and its refusal, as the source runs them; on the index
    route the inner solve is skipped without a slope column, and the full direction is
    completed after it.
    """
    design, z, weights = problem
    if index is None or index.factorized:
        solver.inner_solve(X=design, y=z, sample_weight=weights)
        if solver.use_fallback_lbfgs_solve:
            return _inner_failure(solver)
    return None if index is None else index.completed(solver)


def _test(
    solver: Solver, problem: NewtonProblem, index: IndexEvaluation | None
) -> FloatArray:
    """The native convergence test; the accepted gradient that the hooks receive.

    The provider's method on the route without effects and the profiled route; the index
    route's two formulas on its full score otherwise.
    """
    if index is None:
        design, z, weights = problem
        solver.check_convergence(X=design, y=z, sample_weight=weights)
        return solver.gradient
    solver.converged = index.converged(solver)
    return index.full_gradient()


def _iterate(
    solver: Solver,
    problem: NewtonProblem,
    options: SolverOptions,
    tally: Tally,
    *,
    hooks: Hooks | None = None,
    index: IndexEvaluation | None = None,
) -> NativeReason:
    """The loop of ``NewtonSolver.solve``, with :func:`line_search` and no fallback.

    ``hooks`` (the profiled and index routes) are called after the setup and after
    each native convergence test, and bracket each line search; ``None`` on the route
    without effects.  ``index`` (the index route) supplies the representation seams
    of :class:`IndexEvaluation`; ``None`` on the other routes.
    """
    design, z, weights = problem
    solver.setup(X=design, y=z, sample_weight=weights)
    tally.loss_evaluated = True
    if index is not None:
        index.opened(solver)
    if hooks is not None:
        hooks.started(float(solver.loss_value))
    solver.iteration = 1
    solver.converged = False
    solver.use_fallback_lbfgs_solve = False
    failure: NativeReason | None = None
    while solver.iteration <= solver.max_iter and not solver.converged:
        solver.use_fallback_lbfgs_solve = False
        solver.update_gradient_hessian(X=design, y=z, sample_weight=weights)
        tally.gradient_evaluated = True
        failure = _step(solver, problem, index)
        if failure is not None:
            break
        if index is None:
            accepted = search(solver, problem, options, tally, transaction=hooks)
        else:
            accepted = index.search(solver, options, tally, absolute_sum=absolute_sum)
        if not accepted:
            failure = "line_search_failure"
            break
        tally.steps += 1
        gradient = _test(solver, problem, index)
        if hooks is not None:
            hooks.stepped(bool(solver.converged), gradient)
        solver.iteration += 1
    solver.iteration -= 1
    solver.finalize(X=design, y=z, sample_weight=weights)
    if failure is not None:
        return failure
    return "native_stop" if solver.converged else "iteration_limit"


def _guarded(
    solver: Solver,
    problem: NewtonProblem,
    options: SolverOptions,
    tally: Tally,
    *,
    hooks: Hooks | None = None,
    index: IndexEvaluation | None = None,
) -> tuple[NativeReason, str]:
    """The reason of the attempt, with the text of the exception that ended it, if any.

    Only two exceptions end an attempt: the evaluation limit and a value that is
    not representable.  Any other exception is not caught.
    """
    try:
        return _iterate(solver, problem, options, tally, hooks=hooks, index=index), ""
    except EvaluationLimitError as error:
        return "evaluation_limit", str(error)
    except FloatingPointError as error:
        return "numerical_breakdown", str(error)


def _state(solver: Solver, tally: Tally) -> AcceptedState:
    """The state that the solver holds, each part once it has been evaluated."""
    raw = _immutable(solver.raw_prediction) if tally.loss_evaluated else None
    loss = float(solver.loss_value) if tally.loss_evaluated else None
    gradient = _immutable(solver.gradient) if tally.gradient_evaluated else None
    return AcceptedState(_immutable(solver.coef), raw, loss, gradient)


def run_attempt(
    provider: NewtonProvider,
    base_loss: BaseLoss,
    problem: NewtonProblem,
    coefficients: FloatArray,
    options: SolverOptions,
    *,
    max_iter: int,
    budget: Budget,
) -> NewtonAttempt:
    """One attempt of the iteration from ``coefficients``, at most ``max_iter`` steps.

    The solver is built from a copy of the coefficients, the
    provider's linear loss (the carried ``LinearModelLoss``) of ``base_loss``
    without an intercept of its own (the design holds it), no penalty,
    ``tol=gradient_tol``, one thread.  NumPy does not
    report overflow or invalid operations, and the provider's warnings are
    recorded in the attempt.  ``budget`` is the one that ``base_loss`` charges.
    """
    linear_loss = provider.linear_loss(base_loss=base_loss, fit_intercept=False)
    solver = provider.solver(
        coef=np.array(coefficients, dtype=np.float64),
        linear_loss=linear_loss,
        l2_reg_strength=0.0,
        tol=options.gradient_tol,
        max_iter=max_iter,
        n_threads=1,
        verbose=0,
    )
    tally = Tally()
    before = dict(budget.calls)
    wall, cpu = time.perf_counter(), time.process_time()
    recorder = warnings.catch_warnings(record=True)
    with recorder as caught, np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("always")
        reason, message = _guarded(solver, problem, options, tally)
    shown = tuple(f"{type(item.message).__name__}: {item.message}" for item in caught)
    return NewtonAttempt(
        start=_immutable(coefficients),
        state=_state(solver, tally),
        reason=reason,
        message=message,
        iterations=tally.steps,
        line_searches=tally.line_searches,
        trials=tally.trials,
        refused_trials=tally.refused,
        evaluations=EvaluationCount.spent(before, budget.calls),
        warnings=shown,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
    )


def run_cancelled_attempt(
    provider: NewtonProvider,
    problem: CancelledProblem,
    coefficients: FloatArray,
    options: SolverOptions,
    *,
    max_iter: int,
    budget: Budget,
) -> NewtonAttempt:
    """Run one internal implicit unit-weight tau=2 attempt in the existing loop.

    Parameters
    ----------
    provider : NewtonProvider
        Caller-resolved existing provider; no alternate optimizer or fallback.
    problem : CancelledProblem
        Owned rows, explicit supplied coordinates and checked byte/work reservation.
    coefficients : ndarray
        Start in the supplied identified coordinates, copied for the provider.
    options : SolverOptions
        Existing resolved controls, forwarded unchanged.
    max_iter : int
        Remaining accepted-step allowance, at most options.max_iter.
    budget : Budget
        Solve-wide allowance, shared by retry and charged before implicit products.

    Returns
    -------
    NewtonAttempt
        Last accepted state and actual trace, never a fit or certificate.

    Notes
    -----
    The casts bridge frozen vendor annotations, not numerical representations.
    On the audited single-output path setup passes X only to our raw/loss methods;
    update passes it to gradient_hessian; inner_solve and native stop use provider
    buffers, and finalize only handles inactive multinomial branches. Search calls
    our raw helper and loss_gradient. No other X access or dense wrapper runs.
    Provider allocations are float64 (d,) and (d,d), from the coefficient length.
    """
    if type(problem) is not CancelledProblem or type(options) is not SolverOptions:
        raise TypeError("use a validated CancelledProblem and SolverOptions")
    if options.curvature != "observed":
        raise ValueError(
            "cancelled convex attempts require observed criterion curvature"
        )
    if type(budget) is not Budget or not 0 <= budget.limit <= options.max_evaluations:
        raise ValueError(
            "use the shared Budget within the resolved evaluation allowance"
        )
    if type(max_iter) is not int or not 0 < max_iter <= options.max_iter:
        raise ValueError("max_iter must be the remaining accepted-step allowance")
    operator = problem.operator
    operator.plan.require(operator.bounds)
    checked = finite_vector(coefficients, operator.shape[1], "identified start")
    linear = CancelledLinearLoss(problem, budget)
    carrier = ImplicitNewtonProblem(operator, problem.z, None)
    solver = provider.solver(
        coef=np.array(checked, dtype=np.float64),
        linear_loss=cast("LinearLoss", linear),
        l2_reg_strength=0.0,
        tol=options.gradient_tol,
        max_iter=max_iter,
        n_threads=1,
        verbose=0,
    )
    tally = Tally()
    before = dict(budget.calls)
    wall, cpu = time.perf_counter(), time.process_time()
    with (
        warnings.catch_warnings(record=True) as caught,
        np.errstate(over="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("always")
        reason, message = _guarded(
            solver, cast("NewtonProblem", carrier), options, tally
        )
    shown = tuple(f"{type(item.message).__name__}: {item.message}" for item in caught)
    return NewtonAttempt(
        start=_immutable(coefficients),
        state=_state(solver, tally),
        reason=reason,
        message=message,
        iterations=tally.steps,
        line_searches=tally.line_searches,
        trials=tally.trials,
        refused_trials=tally.refused,
        evaluations=EvaluationCount.spent(before, budget.calls),
        warnings=shown,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
    )


# ------------------------------------------------------------------------- solve


def _check(system: JointSystem, start: Start, options: SolverOptions) -> None:
    """The preconditions of :func:`solve`, each refused with a built-in error."""
    if type(system) is not JointSystem:
        raise TypeError("system must be a JointSystem")
    if type(start) is not Start:
        raise TypeError("start must be a Start")
    if type(options) is not SolverOptions:
        raise TypeError("options must be a SolverOptions")
    variance = system.variance
    if isinstance(variance, PowerVariance) and not 1.0 <= variance.tau <= 2.0:
        raise ValueError("the convex Newton solver needs tau in [1, 2] or NB2")
    if start.coefficients.shape != (system.dimension,):
        raise ValueError("start must have one value per column of the design")


def solve(
    system: JointSystem,
    start: Start,
    options: SolverOptions,
    *,
    assess: Assessor | None,
) -> NewtonCandidate:
    """Solve the convex joint system from ``start``: a candidate and its trace.

    Parameters
    ----------
    system : JointSystem
        The joint system without effects, with a power variance of ``tau`` in
        ``[1, 2]`` or an NB2 variance; its rows of positive weight are solved.
    start : Start
        One start (:mod:`powermean.solve.starts`), recorded with the candidate.
    options : SolverOptions
        The resolved options, all used: the provider, the native tolerance, the
        budgets, the line search, the retry and the curvature by the iteration, the
        controls of the certificate by the assessor only.
    assess : Assessor or None
        The independent assessment that the retry needs (the certificate):
        required with ``retry=1``, never called with ``retry=0``, and never
        replaced by the solver's own test.

    Returns
    -------
    NewtonCandidate
        The candidate and its trace; no claim of convergence.

    Raises
    ------
    SpecificationError
        ``newton_provider="sklearn"`` without a verified installed scikit-learn.
    TypeError, ValueError
        A precondition does not hold, or ``retry=1`` comes without ``assess``.
    """
    _check(system, start, options)
    wall, cpu = time.perf_counter(), time.process_time()
    provider = resolve_provider(options.newton_provider)
    budget = Budget(options.max_evaluations)
    point = PointLoss(
        offset=system.active_offset,
        variance=system.variance,
        curvature=options.curvature,
        budget=budget,
    )
    problem = NewtonProblem(
        system.active_design, system.active_z, system.active_weights
    )
    log: AssessmentLog | None = None
    if assess is not None:
        log = AssessmentLog(assess, system, certificate_controls(options))

    def attempt(coefficients: FloatArray, max_iter: int) -> NewtonAttempt:
        return run_attempt(
            provider,
            point,
            problem,
            coefficients,
            options,
            max_iter=max_iter,
            budget=budget,
        )

    outcome = run_attempts(
        attempt,
        log,
        start.coefficients,
        retry=options.retry,
        max_iter=options.max_iter,
        evaluations_remain=lambda: budget.used < budget.limit,
    )
    assessed = Timing(0.0, 0.0) if log is None else Timing(log.wall, log.cpu)
    return NewtonCandidate(
        start=start,
        options=options,
        provider=provider.name,
        provider_version=provider.version,
        attempts=outcome.attempts,
        retained=outcome.retained,
        initial_assessment=outcome.initial,
        no_measurable_progress=outcome.no_measurable_progress,
        assessments=0 if log is None else log.count,
        assessment_timing=assessed,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
    )


# ------------------------------------------------ the profiled organization


class InnerLimitError(RuntimeError):
    """A profile or absorption of a profiled point spent its sweeps.

    It stops the attempt: neither a refused trial nor a
    new allowance.
    """


class IndicationError(RuntimeError):
    """An existence indication stopped a profiled attempt.

    Raised by :meth:`Hooks.started` or :meth:`Hooks.stepped`; the operative stop of the
    attempt is then not a native stop.
    """


class Transaction(Protocol):
    """The bracket of one line search of the profiled iteration."""

    def begin(self) -> None:
        """A line search starts from the accepted point."""
        ...

    def end(self, coef: FloatArray | None, /) -> None:
        """It accepted the trial ``coef``, or none: refused or interrupted."""
        ...


class Hooks(Transaction, Protocol):
    """The points of the profiled iteration at which the attempt may stop."""

    def started(self, loss: float, /) -> None:
        """The start is evaluated, with its averaged loss."""
        ...

    def stepped(self, converged: bool, gradient: FloatArray, /) -> None:
        """A step was accepted and the native test run; the accepted gradient."""
        ...


class ProfiledEvaluation(Hooks, PointEvaluation, Protocol):
    """The profiled points of an attempt (:mod:`powermean.solve.profiled_loss`).

    The points that the profiled linear loss evaluates
    (:class:`~powermean.solve.profiled_linear_loss.PointEvaluation`) and the hooks of
    the iteration.
    """


ProfiledStop = Literal["inner_limit", "indication"]


@dataclass(frozen=True, eq=False)
class ProfiledIteration:
    """The provider's iteration of one profiled attempt, and what ended it.

    The fields of :class:`NewtonAttempt`, except that ``native`` is the native reason
    when the loop of the iteration ended, ``None`` when ``stop`` ended it first:
    ``inner_limit`` or ``indication``.
    """

    start: FloatArray
    state: AcceptedState
    native: NativeReason | None
    stop: ProfiledStop | None
    message: str
    iterations: int
    line_searches: int
    trials: int
    refused_trials: int
    evaluations: EvaluationCount
    warnings: tuple[str, ...]
    timing: Timing


def run_profiled_attempt(
    provider: NewtonProvider,
    point: PointLoss,
    problem: NewtonProblem,
    coefficients: FloatArray,
    options: SolverOptions,
    *,
    evaluation: ProfiledEvaluation,
    max_iter: int,
    budget: Admissions,
) -> ProfiledIteration:
    """One profiled attempt from ``coefficients``, at most ``max_iter`` steps.

    The solver is built as :func:`run_attempt` builds it, with
    :class:`ProfiledLinearLoss` of the provider's linear loss of ``point`` (zero
    offset), of ``evaluation`` and of the carried ``sandwich_dot``, in place of that
    linear loss, handed over by one cast since it provides the carried interface without
    deriving from it; ``evaluation`` is also the hooks of the iteration.  ``budget`` is
    the solve's one budget, which ``point`` charges and the profiled loss admits each
    evaluation to before its work.  Besides the evaluation limit and a value that is
    not representable, :class:`InnerLimitError` and :class:`IndicationError` end the
    attempt; no other
    exception is caught.
    """
    carried = provider.linear_loss(base_loss=point, fit_intercept=False)
    profiled = ProfiledLinearLoss(
        carried,
        point,
        evaluation,
        sandwich=vendor_linear_loss.sandwich_dot,
        admissions=budget,
    )
    solver = provider.solver(
        coef=np.array(coefficients, dtype=np.float64),
        linear_loss=cast("LinearLoss", profiled),
        l2_reg_strength=0.0,
        tol=options.gradient_tol,
        max_iter=max_iter,
        n_threads=1,
        verbose=0,
    )
    tally = Tally()
    before = dict(budget.calls)
    wall, cpu = time.perf_counter(), time.process_time()
    native: NativeReason | None = None
    stop: ProfiledStop | None = None
    recorder = warnings.catch_warnings(record=True)
    with recorder as caught, np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("always")
        try:
            native, message = _guarded(
                solver, problem, options, tally, hooks=evaluation
            )
        except InnerLimitError as error:
            stop, message = "inner_limit", str(error)
        except IndicationError as error:
            stop, message = "indication", str(error)
    shown = tuple(f"{type(item.message).__name__}: {item.message}" for item in caught)
    return ProfiledIteration(
        start=_immutable(coefficients),
        state=_state(solver, tally),
        native=native,
        stop=stop,
        message=message,
        iterations=tally.steps,
        line_searches=tally.line_searches,
        trials=tally.trials,
        refused_trials=tally.refused,
        evaluations=EvaluationCount.spent(before, budget.calls),
        warnings=shown,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
    )


# --------------------------------------------- the fitted-index organization


class IndexEvaluation(Hooks, Protocol):
    """The fitted-index state of one attempt: its hooks and its seams.

    :func:`run_index_attempt` hands it the provider's carried ``LinearModelLoss`` and
    the carried ``sandwich_dot`` (:meth:`provider_loss`), and :func:`_iterate` calls
    its seams: after the setup (:meth:`opened`), after the provider's inner solve,
    which a solve without slopes skips (:meth:`completed`), in place of the search on
    the slopes (:meth:`search`) and of the provider's convergence method
    (:meth:`converged`).  Its hooks are those of the profiled route.
    """

    @property
    def factorized(self) -> bool:
        """Whether the provider factorizes a slope system: some slope column."""
        ...

    def provider_loss(self, carried: LinearLoss, sandwich: Sandwich, /) -> object:
        """The linear loss that the provider drives, on its carried linear loss."""
        ...

    def opened(self, solver: Solver, /) -> None:
        """After the setup: the provider's raw prediction becomes the full index."""
        ...

    def completed(self, solver: Solver, /) -> NativeReason | None:
        """The full direction of the step; the reason that refuses it, if any."""
        ...

    def search(
        self,
        solver: Solver,
        options: SolverOptions,
        tally: Tally,
        /,
        *,
        absolute_sum: AbsoluteSum,
    ) -> bool:
        """The line search along the full direction, committed to the provider."""
        ...

    def converged(self, solver: Solver, /) -> bool:
        """The native test on the full score and the origin's unhalved decrement."""
        ...

    def full_gradient(self) -> FloatArray:
        """The full score of the accepted point."""
        ...

    def accepted_state(self, solver: Solver, tally: Tally, /) -> AcceptedState:
        """The accepted whole point, each part once it has been evaluated."""
        ...


def run_index_attempt(
    provider: NewtonProvider,
    point: PointLoss,
    problem: NewtonProblem,
    coefficients: FloatArray,
    options: SolverOptions,
    *,
    evaluation: IndexEvaluation,
    max_iter: int,
    budget: Admissions,
) -> ProfiledIteration:
    """One fitted-index attempt from the slopes ``coefficients``, at most ``max_iter``.

    The solver is built as :func:`run_attempt` builds it, with the linear loss that
    ``evaluation`` forms on the provider's carried linear loss of ``point`` (zero
    offset) and the carried ``sandwich_dot``, handed over by one cast as the profiled
    loss is; ``evaluation`` is also the hooks and the seams of the iteration.
    ``budget`` is the solve's one budget, which ``point`` charges and the evaluation
    admits each evaluation to before its work.  Besides the evaluation limit
    and a value that is not representable, :class:`InnerLimitError` and
    :class:`IndicationError` end the attempt; no other
    exception is caught.  The attempt's state is the evaluation's accepted whole point
    (the full index ``q`` as its raw prediction and the full score as its gradient),
    not the provider's Schur view.  The record is that of :func:`run_profiled_attempt`.
    """
    carried = provider.linear_loss(base_loss=point, fit_intercept=False)
    linear = evaluation.provider_loss(carried, vendor_linear_loss.sandwich_dot)
    solver = provider.solver(
        coef=np.array(coefficients, dtype=np.float64),
        linear_loss=cast("LinearLoss", linear),
        l2_reg_strength=0.0,
        tol=options.gradient_tol,
        max_iter=max_iter,
        n_threads=1,
        verbose=0,
    )
    tally = Tally()
    before = dict(budget.calls)
    wall, cpu = time.perf_counter(), time.process_time()
    native: NativeReason | None = None
    stop: ProfiledStop | None = None
    recorder = warnings.catch_warnings(record=True)
    with recorder as caught, np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("always")
        try:
            native, message = _guarded(
                solver, problem, options, tally, hooks=evaluation, index=evaluation
            )
        except InnerLimitError as error:
            stop, message = "inner_limit", str(error)
        except IndicationError as error:
            stop, message = "indication", str(error)
    shown = tuple(f"{type(item.message).__name__}: {item.message}" for item in caught)
    return ProfiledIteration(
        start=_immutable(coefficients),
        state=evaluation.accepted_state(solver, tally),
        native=native,
        stop=stop,
        message=message,
        iterations=tally.steps,
        line_searches=tally.line_searches,
        trials=tally.trials,
        refused_trials=tally.refused,
        evaluations=EvaluationCount.spent(before, budget.calls),
        warnings=shown,
        timing=Timing(time.perf_counter() - wall, time.process_time() - cpu),
    )
