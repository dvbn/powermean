"""The fitted-index state of a Newton attempt: full rows, provider view, trials, seams.

Internal module: the explicitly selected organization ``absorbed='index'``,
for the power family with ``tau`` in ``[1, 2]`` and NB2.  This module imports the
adapter, never the carried files: the adapter hands it the provider's carried
``LinearModelLoss`` and the carried ``sandwich_dot``
(:meth:`IndexEvaluator.provider_loss`).

The accepted point.  The authoritative point of an attempt is ``(gamma, q)``, the slopes
and the fitted index of every row, with its averaged criterion and its full score.  A
trial moves both along ONE direction with the unchanged search arithmetic,
``gamma + t d_gamma`` and ``q + t d_q``, and evaluates the unprofiled criterion and the
full score at that exact ``q``: nothing is profiled after the start.  A trial is held,
never accepted by itself; the accepted trial's whole bundle commits, and a refused or
interrupted search keeps the accepted one (:meth:`IndexEvaluator.end`).  The derivatives
are evaluated at the accepted point only.  The pointwise loss has a zero offset: ``q``
already holds the transformed offset, entered once in the base of the start.

The warm slope state.  Each direction's slope absorption prepares its weights afresh
from its own ``C`` and, with strictly positive current curvature on every row after
the attempt's first direction, starts from the slope
coefficients of the last accepted origin's successful direction
(:func:`~powermean.systems.profiled._slope_state`: owned coefficients bound to the
grouping and the ordered slope columns, nothing else); the absorber checks that binding,
rebuilds ``X - lift(a)`` and checks it afresh at the current tolerance.  The first
direction of an attempt, a retry's included, is cold, and every request at one origin
starts from that origin's fixed seed.  A request that succeeds whole (slope absorption,
Schur matrix and full-RHS solve) leaves its state pending, tagged with its origin; only
an accepted search promotes it (:meth:`IndexEvaluator.end`), and a failed request or a
refused or interrupted search promotes nothing.  Any zero uses the original cold
calculation and creates no slope state; accepting it clears the old seed.  The full-RHS
solve, the recovery and the final work stay cold; without slopes there is no slope
state.

The full score.  ``v = (-a r) / A`` is formed row by row in the
carried loss's order, and

    G = (X'v, group_sums(v) of every original block, every level in code order)

with ``X`` the retained scaled slopes: no omitted reference, level average, unscaled
column or intercept.  The line search's roundoff safeguard and the native test read
``G``; the provider never does.  :func:`trial_rows` is the carried
``LinearModelLoss.loss_gradient`` without an intercept, step for step, which also keeps
``v``; :func:`final_rows` forms the same quantities for the final work, outside the
budget.

The provider view.  :class:`IndexLinearLoss` gives the actual provider the slopes'
Schur view in K and K-by-K storage: ``g_S = Xt'v`` and ``H_S = sandwich(Xt, C/A)``, the
slopes absorbed with the curvature weights ``C = a c`` (``a w`` with the expected
curvature), after the full direction's inner work.  Its first loss, which the
provider's setup asks at the slope-only prediction ``X gamma``, is evaluated at the
initialized full index: the start of the attempt, admitted before its initialization.
No FE-square matrix and no indicator is formed.

Evaluations, schedule, indications and work are those of the profiled route
(:mod:`powermean.solve.profiled_loss`), read on the full index: the solve's one budget
admits ``loss`` before the initialization, ``gradient_hessian`` before the rows and
each direction's inner work and ``loss_gradient`` before every trial, an admitted
failure staying charged; the inner tolerance follows
:func:`~powermean.solve.profiled_loss.next_tolerance` with ``s = max |G|`` of each
accepted point, constant within a direction and its search.

The forcing factor of the full-RHS solve.  The attempt
keeps one history, the last issued pair ``(s, eta)`` and the accepted origin it was
issued at.  Each direction, before its inner work, issues at a NEW accepted origin:
``s = max |G|`` of the origin's full score already formed, ``eta = 1e-3`` for the
attempt's first direction and :func:`~powermean.solve.absorbed_rhs.next_forcing` of
the last pair otherwise, equal scores included; a repeated request at the same origin
reads it again.  Trials, held or refused, failed searches, inner iterations and the
final work issue nothing; the final work reads the most recently issued factor
(:attr:`IndexEvaluator.forcing`).  A retry's fresh evaluator starts again at ``1e-3``
and ``absorb_tol_start``, while the budget and the logged work go on.  The absolute
schedule above is separate and unchanged.  ZC and, at ``tau = 2``, GC
are evaluated at the completed start, at each newly accepted point after the native
test, and at the final point, never at a held or refused trial, and coalesce into one
stop.  Every initializer (profile or local start), slope absorption, full-RHS solve and
completed direction is logged (:class:`~powermean.solve.absorbed_direction.IndexLog`),
refused calls included.

The guard of the full-RHS exception.  The rows records carry the
raw-weight total ``A = np.sum(a)`` that their own native arithmetic forms for ``v =
(-a r) / A`` (:class:`OriginRows`, :class:`FinalRows`), and the direction hands that
same value to the full-RHS call: no second sum, never ``sum(C)``, no new row evaluation
or admission.  The automatic start is the context's
(:meth:`~powermean.solve.absorbed_direction.IndexContext.initialized`): a profile, or in
its local domain the local start; either returns the opening point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Unpack, cast

import numpy as np

from powermean.fe.absorb import AbsorptionResult, AbsorptionState
from powermean.fe.absorb_acceleration import BUDGET
from powermean.fe.absorb_numeric import PreparedWeights, group_sums
from powermean.fe.profile_arithmetic import (
    ProfileProblem,
    _cached_criterion,
    _cached_rows,
)
from powermean.fe.reductions import immutable_array
from powermean.model.rows import evaluate_criterion, evaluate_rows
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.absorbed_direction import (
    CallSite,
    Direction,
    DirectionCall,
    IndexContext,
    IndexPoint,
    IndexStart,
    complete_direction,
    reconstruct,
)
from powermean.solve.absorbed_rhs import FORCING_INITIAL, next_forcing
from powermean.solve.absorbed_start import LocalInitializer
from powermean.solve.line_search import AbsoluteSum, line_search
from powermean.solve.newton_convex import InnerLimitError, LinearLoss, Solver
from powermean.solve.newton_trace import AcceptedState, NativeReason, Tally
from powermean.solve.options import Curvature, SolverOptions
from powermean.solve.point_loss import PointLoss, _weighted
from powermean.solve.profiled_indications import (
    Alarm,
    ChordAlarm,
    Where,
    chord_alarm,
    zero_row_alarm,
)
from powermean.solve.profiled_linear_loss import (
    CarriedLoss,
    HessianCall,
    Intercept,
    LossCall,
    Sandwich,
    _point_initial,
)
from powermean.solve.profiled_loss import _stop, next_tolerance
from powermean.systems.profiled import (
    PointProfile,
    ProfiledProblem,
    _positive_slope_curvature,
    _slope_state,
    base_index,
)

ROWS = "the full rows are not representable in float64"
_ACCEPTED_CRITERION = evaluate_criterion
_ACCEPTED_ROWS = evaluate_rows


# ---------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class Bundle:
    """One whole accepted or held point: slopes, index, criterion and full score.

    ``gradient`` is ``None`` at the start of an attempt, whose first score the
    derivatives evaluate.
    """

    theta: FloatArray
    index: FloatArray
    loss: float
    gradient: FloatArray | None

    def __post_init__(self) -> None:
        """Own the arrays."""
        object.__setattr__(self, "theta", immutable_array(self.theta))
        object.__setattr__(self, "index", immutable_array(self.index))
        if self.gradient is not None:
            object.__setattr__(self, "gradient", immutable_array(self.gradient))


@dataclass(frozen=True, eq=False)
class OriginRows:
    """The rows of a direction origin, in the carried order.

    ``v = (-a r) / A``, ``scaled = C / A``, ``curvature = C``, ``right = b = a r``,
    ``gradient`` the full score ``G`` and ``total`` the raw-weight total ``A`` of those
    divisions, as formed there.
    """

    v: FloatArray
    scaled: FloatArray
    curvature: FloatArray
    right: FloatArray
    gradient: FloatArray
    total: float


# --------------------------------------------------------------------- full rows


def full_score(
    problem: ProfiledProblem, slopes: FloatArray, v: FloatArray
) -> FloatArray:
    """``G``: the slopes' score, then the level sums of ``v`` of every block, checked.

    ``FloatingPointError`` when a level sum or any entry is not representable.
    """
    parts = [slopes]
    for block in problem.grouping.blocks:
        parts.append(group_sums(v, block.codes, block.n_levels))
    found: FloatArray = np.concatenate(parts)
    if not np.isfinite(found).all():
        raise FloatingPointError(ROWS)
    return found


@dataclass(frozen=True, eq=False)
class TrialRows:
    """The averaged criterion and the full score at one index."""

    loss: np.float64
    gradient: FloatArray


def trial_rows(
    context: IndexContext, coef: FloatArray, raw: FloatArray, l2: float
) -> TrialRows:
    """The carried ``loss_gradient`` at the index ``raw``, step for step, with ``G``.

    The pointwise loss's charge consumes the pending admission.  ``FloatingPointError``
    when a row, the criterion or the score is not representable: a refused trial.
    """
    problem = context.problem
    point = context.point
    initial = (
        _point_initial(point, problem, problem.z)
        if type(point) is PointLoss
        else None
    )
    if initial is not None:
        point.budget.charge("loss_gradient")
        q = point.index(raw)
        values = _cached_criterion(initial, q)
        rows = _cached_rows(initial, q)
        loss_rows = _weighted(values, problem.weights)
        grad_pointwise = _weighted(-rows.score, problem.weights)
    else:
        loss_rows, grad_pointwise = point.loss_gradient(
            y_true=problem.z,
            raw_prediction=raw,
            sample_weight=problem.weights,
            n_threads=1,
        )
    sw_sum = np.sum(problem.weights)
    loss: np.float64 = loss_rows.sum() / sw_sum
    loss += cast("np.float64", 0.5 * l2 * (coef @ coef))
    grad_pointwise /= sw_sum
    slopes = problem.design.T @ grad_pointwise + l2 * coef
    if not math.isfinite(loss):
        raise FloatingPointError(ROWS)
    return TrialRows(loss, full_score(problem, slopes, grad_pointwise))


@dataclass(frozen=True, eq=False)
class FinalRows:
    """The criterion, full score, curvature weights and right-hand side at one index.

    ``total`` is the raw-weight total ``A`` that divides the criterion and the score.
    """

    loss: float
    gradient: FloatArray
    curvature: FloatArray
    right: FloatArray
    total: float


def final_rows(
    problem: ProfiledProblem, q: FloatArray, curvature: Curvature
) -> FinalRows:
    """The rows of the final work at ``q``, outside the iteration's budget.

    ``L/A``, ``G``, ``C`` (``a c``, or ``a w`` with the expected curvature),
    ``b = a r`` and ``A`` itself, in the order of the carried loss;
    ``FloatingPointError`` where a value is not representable (nothing is clipped).
    """
    initial: ProfileProblem | None = (
        getattr(problem, "_initial", None)
        if type(problem) is ProfiledProblem
        else None
    )
    if (
        initial is not None
        and evaluate_criterion is _ACCEPTED_CRITERION
        and evaluate_rows is _ACCEPTED_ROWS
    ):
        values = _cached_criterion(initial, q)
        rows = _cached_rows(initial, q)
    else:
        values = evaluate_criterion(problem.z, q, variance=problem.variance)
        rows = evaluate_rows(problem.z, q, variance=problem.variance)
    a = problem.weights
    kind = rows.observed_curvature if curvature == "observed" else rows.fisher_curvature
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.sum(a)
        loss = (a * values).sum() / total
        v = (a * -rows.score) / total
        slopes = problem.design.T @ v
        weights = a * kind
        right = a * rows.score
    found = (weights, right)
    if not math.isfinite(loss) or not all(np.isfinite(x).all() for x in found):
        raise FloatingPointError(ROWS)
    gradient = full_score(problem, slopes, v)
    return FinalRows(
        float(loss),
        immutable_array(gradient),
        immutable_array(weights),
        immutable_array(right),
        float(total),
    )


# ------------------------------------------------------------ the search view


class SearchView:
    """The line search's state on the index route (``line_search.SearchState``).

    ``coef`` is ``gamma`` and ``coef_newton`` its step; ``raw_prediction`` is the full
    index ``q``, which a trial moves along the explicit ``d_q``; ``gradient`` is the
    full score ``G``; ``gradient_times_newton`` the full derivative ``v' d_q``.  The
    step, the score and the derivative are set by each direction before its search.
    """

    def __init__(self, coef: FloatArray, q: FloatArray, loss: np.float64) -> None:
        self.coef: FloatArray = coef
        self.coef_old: FloatArray = coef
        self.coef_newton: FloatArray = np.zeros(coef.shape)
        self.raw_prediction: FloatArray = q
        self.loss_value: np.float64 = loss
        self.loss_value_old: np.float64 = loss
        self.gradient: FloatArray = np.zeros(0)
        self.gradient_old: FloatArray = self.gradient
        self.gradient_times_newton: np.float64 = np.float64(0.0)
        self.use_fallback_lbfgs_solve: bool = False


# ------------------------------------------------------------- provider view


def _finite(values: FloatArray | np.float64) -> None:
    """``FloatingPointError`` unless every value is finite."""
    if not np.isfinite(values).all():
        raise FloatingPointError(ROWS)


class RowEvaluation(Protocol):
    """The actual row consumer shared by index and retained-nonlinear attempts."""

    context: IndexContext

    def start(self, coef: FloatArray, /) -> FloatArray:
        """The initialized full index."""
        ...

    def current(self, coef: FloatArray, /) -> FloatArray:
        """The accepted full index."""
        ...

    def row_direction(self, rows: OriginRows, /) -> FloatArray | None:
        """Consume fresh full rows before the provider view is filled."""
        ...


class IndexLinearLoss:
    """The provider's view of the index route: the slopes' Schur system, admitted.

    The interface of the carried ``LinearModelLoss`` that the providers' setup and
    derivatives call by keyword; the provider's own line search is never called.

    Parameters
    ----------
    carried : CarriedLoss
        The provider's carried ``LinearModelLoss`` of the context's pointwise loss.
    evaluation : IndexEvaluator
        The whole-point state of the attempt.
    sandwich : Sandwich
        The carried ``sandwich_dot``, which the adapter hands over.
    """

    fit_intercept = False

    def __init__(
        self, carried: CarriedLoss, evaluation: RowEvaluation, *, sandwich: Sandwich
    ) -> None:
        self.base_loss = evaluation.context.point
        self.carried = carried
        self.evaluation = evaluation
        self.sandwich = sandwich
        self.admissions = evaluation.context.admissions

    def weight_intercept_raw(
        self, coef: FloatArray, design: FloatArray
    ) -> tuple[FloatArray, Intercept, FloatArray]:
        """The carried ``weight_intercept_raw``: ``X gamma``, without the offset."""
        return self.carried.weight_intercept_raw(coef, design)

    def loss(self, **call: Unpack[LossCall]) -> np.float64:
        """The averaged criterion at the initialized full index of the start, admitted.

        The ``raw_prediction`` of the call, the slope-only ``X gamma`` of the setup, is
        not the start's index and is not used.
        """
        coef = call["coef"]
        self.admissions.admit("loss")
        index = self.evaluation.start(coef)
        value = self.carried.loss(
            coef=coef,
            X=call["X"],
            y=call["y"],
            sample_weight=call["sample_weight"],
            l2_reg_strength=call["l2_reg_strength"],
            n_threads=call["n_threads"],
            raw_prediction=index,
        )
        _finite(value)
        return value

    def gradient_hessian(
        self, **call: Unpack[HessianCall]
    ) -> tuple[FloatArray, FloatArray, bool | np.bool_]:
        """The Schur gradient ``Xt'v`` and Hessian ``sandwich(Xt, C/A)``, admitted.

        The carried arithmetic of one output without intercept, step for step, at the
        accepted full index, and the full direction's inner work before the Schur view
        is written; the provider's K and K-by-K storage only.
        """
        coef, design, weights = call["coef"], call["X"], call["sample_weight"]
        grad, hess = call["gradient_out"], call["hessian_out"]
        n_features = design.shape[1]
        self.admissions.admit("gradient_hessian")
        current_index = self.evaluation.current(coef)
        initial = (
            _point_initial(
                self.base_loss,
                self.evaluation.context.problem,
                call["y"],
            )
            if type(self.base_loss) is PointLoss
            else None
        )
        if initial is not None:
            self.base_loss.budget.charge("gradient_hessian")
            q = self.base_loss.index(current_index)
            evaluated = _cached_rows(initial, q)
            hessian = (
                evaluated.observed_curvature
                if self.base_loss.curvature == "observed"
                else evaluated.fisher_curvature
            )
            grad_pointwise = _weighted(-evaluated.score, weights)
            hess_pointwise = _weighted(hessian, weights)
        else:
            grad_pointwise, hess_pointwise = self.base_loss.gradient_hessian(
                y_true=call["y"],
                raw_prediction=current_index,
                sample_weight=weights,
                n_threads=call["n_threads"],
            )
        curvature = np.array(hess_pointwise, copy=True)
        right = -grad_pointwise
        sw_sum = np.sum(weights)
        grad_pointwise /= sw_sum
        hess_pointwise /= sw_sum
        hessian_warning = np.average(hess_pointwise < 0, weights=weights) > 0.25
        hess_pointwise = np.abs(hess_pointwise)
        slopes = design.T @ grad_pointwise + call["l2_reg_strength"] * coef
        score = full_score(self.evaluation.context.problem, slopes, grad_pointwise)
        rows = OriginRows(
            grad_pointwise, hess_pointwise, curvature, right, score, float(sw_sum)
        )
        absorbed = self.evaluation.row_direction(rows)
        if absorbed is not None:
            grad[:n_features] = absorbed.T @ grad_pointwise
            _finite(grad)
            hess[:n_features, :n_features] = self.sandwich(absorbed, hess_pointwise)
        return grad, hess, hessian_warning


# ------------------------------------------------------------- the attempt state


@dataclass(frozen=True, eq=False)
class _Parts:
    """The inner results of the current direction: its rows, ``Xt`` and ``h``."""

    rows: OriginRows
    absorbed: FloatArray | None
    h: FloatArray


class IndexEvaluator:
    """The whole-point state of one fitted-index attempt (module docstring).

    The :class:`~powermean.solve.newton_convex.IndexEvaluation` of
    :func:`~powermean.solve.newton_convex.run_index_attempt`: the start, the trials,
    the accepted point, the inner work of each direction, the schedule and the
    indications.

    Parameters
    ----------
    context : IndexContext
        The fixed parts of the solve.
    attempt : int
        The index of the attempt in the solve.
    origin : IndexStart, IndexPoint or None
        The caller's full start, the retained whole point of a retry (its slopes and
        index taken as they are, without a sweep), or ``None`` for the automatic start.
    """

    def __init__(
        self,
        context: IndexContext,
        *,
        attempt: int,
        origin: IndexStart | IndexPoint | None,
    ) -> None:
        self.context = context
        self.attempt = attempt
        self.origin = origin
        variance = context.problem.variance
        self.gpml = type(variance) is PowerVariance and variance.tau == 2
        self.tolerance = context.controls.absorb_tol_start
        self.score: float | None = None
        self.issued: tuple[float, float] | None = None
        self.issued_origin: Bundle | None = None
        self.steps = 0
        self.initializer: PointProfile | LocalInitializer | None = None
        self.opening: IndexPoint | None = None
        self.view: SearchView | None = None
        self.accepted: Bundle | None = None
        self.previous: Bundle | None = None
        self.held: list[Bundle] = []
        self.parts: _Parts | None = None
        self.direction_found: Direction | None = None
        self.alarm: Alarm | None = None
        self.slope_seed: AbsorptionState | None = None
        self.slope_pending: tuple[Bundle, AbsorptionState] | None = None
        self.slope_cold_origin: Bundle | None = None

    @property
    def factorized(self) -> bool:
        """Whether the provider factorizes a slope system: some slope column."""
        return bool(self.context.problem.n_slopes)

    @property
    def forcing(self) -> float:
        """The most recently issued forcing factor, or ``1e-3`` before any issue."""
        return FORCING_INITIAL if self.issued is None else self.issued[1]

    def provider_loss(self, carried: LinearLoss, sandwich: Sandwich) -> IndexLinearLoss:
        """The provider's linear loss of the attempt, on its carried linear loss."""
        return IndexLinearLoss(carried, self, sandwich=sandwich)

    # ------------------------------------------------------------ the start

    def start(self, coef: FloatArray) -> FloatArray:
        """The full index of the start of the attempt: its initialization."""
        origin, problem = self.origin, self.context.problem
        if isinstance(origin, IndexPoint):
            _same(coef, origin.theta)
            opening = IndexPoint(origin.theta, origin.base, origin.index)
        elif isinstance(origin, IndexStart):
            _same(coef, origin.slopes)
            base = base_index(problem, coef)
            index = reconstruct(problem, base, origin.effects.coefficients)
            opening = IndexPoint(coef, base, index)
        else:
            point = self.context.initialized(self.attempt, coef)
            self.initializer = point
            opening = IndexPoint(coef, point.base, point.index)
        self.opening = opening
        return opening.index

    def opened(self, solver: Solver) -> None:
        """After the setup: the provider's raw prediction becomes the start's index."""
        opening = self._opened()
        solver.raw_prediction = opening.index
        self.view = SearchView(solver.coef, opening.index, solver.loss_value)

    def current(self, coef: FloatArray) -> FloatArray:
        """The full index of the accepted point, which ``coef`` must be."""
        accepted = self._accepted()
        if not np.array_equal(accepted.theta, coef):
            raise ValueError("the derivatives are evaluated at the accepted point only")
        return accepted.index

    # -------------------------------------------------------- the direction

    def row_direction(self, rows: OriginRows) -> FloatArray | None:
        """The existing index direction, including its replaceable method seam."""
        return self.direction(rows)

    def direction(self, rows: OriginRows) -> FloatArray | None:
        """The direction's inner work at the accepted point: ``Xt`` and ``h``.

        The origin's forcing factor first (:meth:`_issue`, before any inner work), then
        the slopes absorbed with ``C`` (when there are slopes) and the full-RHS solve
        with that factor and the origin's raw-weight total, both at the current
        tolerance and logged.  ``C`` is checked and prepared once for this request: by
        the slope absorption, whose fresh record the full-RHS solve then reads, or by
        the full-RHS solve alone without slopes; the record is not kept.  The slopes
        start from this origin's warm slope state (:attr:`slope_seed`) only if every
        fresh current weight is strictly positive; any zero uses the original cold
        calculation with all rows retained.  The first direction is cold.  A request
        that succeeds whole leaves its own state pending for :meth:`end`, tagged
        with this origin, and any request first discards the pending state of an
        earlier one.
        ``InnerLimitError`` when either spends its sweeps; ``FloatingPointError`` for a
        refusal, a range failure, a breakdown, an unrepresentable forcing factor or a
        Schur matrix that is not representable.  Returns ``Xt``, or ``None`` without
        slopes.
        """
        self.slope_pending = None
        self.slope_cold_origin = None
        view = self._view()
        view.gradient = rows.gradient
        forcing = self._issue(rows)
        absorbed = None
        prepared: PreparedWeights | None = None
        produced: AbsorptionResult | None = None
        if self.factorized:
            seed, site = self.slope_seed, self._site()
            bound = (
                {} if self.context.problem.__dict__.get("_reduced_execution") is None
                else {"index": self._accepted().index}
            )
            if seed is None:
                found, prepared = self.context._absorb(site, rows.curvature, **bound)
            else:
                found, prepared = self.context._absorb(
                    site, rows.curvature, state=seed, **bound
                )
            _required(found.result.status, found.result.converged, found.result.reason)
            if found.failure:
                raise FloatingPointError(found.failure)
            absorbed, produced = found.result.residual, found.result
        site, system = self._site(), (rows.curvature, rows.right)
        rhs = self.context._nuisance(
            site, system, forcing, rows.total, prepared=prepared
        )
        _required(rhs.status, rhs.converged, f"full right-hand side: {rhs.reason}")
        self.parts = _Parts(rows, absorbed, rhs.h)
        if produced is not None and prepared is not None:
            if _positive_slope_curvature(prepared.weights):
                self.slope_pending = (self._accepted(), _slope_state(produced))
            else:
                self.slope_cold_origin = self._accepted()
        return absorbed

    def _issue(self, rows: OriginRows) -> float:
        """The forcing factor of the accepted origin, issued once there.

        At an origin other than the last issued one: ``s = max |G|`` of ``rows``, and
        ``1e-3`` for the attempt's first direction or :func:`next_forcing` of the last
        issued pair, committed with that origin.  At the same origin the issued factor
        is read again.  ``FloatingPointError`` when the selected factor is not
        representable; nothing is committed then.
        """
        origin = self._accepted()
        if self.issued_origin is not origin:
            score = float(np.max(np.abs(rows.gradient)))
            eta = FORCING_INITIAL
            if self.issued is not None:
                eta = next_forcing(self.issued, score)
            self.issued, self.issued_origin = (score, eta), origin
        return self.forcing

    def completed(self, solver: Solver) -> NativeReason | None:
        """``d_q`` and its two scalars; ``non_descent_direction`` for an ascent."""
        parts, view = self._parts(), self._view()
        d_gamma = solver.coef_newton if self.factorized else np.zeros(0)
        rows = parts.rows
        found = complete_direction(
            parts.absorbed,
            d_gamma,
            parts.h,
            (rows.v, rows.scaled),
            tolerance=self.tolerance,
        )
        self.direction_found = found
        score = float(np.max(np.abs(rows.gradient)))
        self.context.log.directions.append(
            DirectionCall(
                self.attempt,
                self.steps,
                self.tolerance,
                found.derivative,
                found.decrement,
                score,
                self.factorized,
            )
        )
        if found.derivative > 0:
            return "non_descent_direction"
        view.coef_newton = found.d_gamma
        view.gradient_times_newton = np.float64(found.derivative)
        return None

    # ------------------------------------------------------------ the search

    def _trial(
        self, coef: FloatArray, raw: FloatArray
    ) -> tuple[np.float64, FloatArray]:
        """A trial of the search: admitted, its unprofiled rows evaluated, held."""
        self.context.admissions.admit("loss_gradient")
        rows = trial_rows(self.context, coef, raw, 0.0)
        self.held.append(Bundle(coef, raw, float(rows.loss), rows.gradient))
        return rows.loss, rows.gradient

    def begin(self) -> None:
        """A line search starts: no trial is held."""
        self.held = []

    def end(self, coef: FloatArray | None) -> None:
        """The search accepted the trial ``coef`` (its bundle commits), or none.

        Without an accepted trial the accepted point stays as it is, the held trials
        are discarded: the rollback of a refused or interrupted search, which also
        discards the pending slope state and keeps the slope seed.  When the trial's
        bundle commits, the pending slope state of the search's origin becomes the
        slope seed of the next origin.  A successful ineligible direction records only
        its origin; accepting its trial clears the old seed, without copying a state.
        """
        held, self.held = self.held, []
        slopes, self.slope_pending = self.slope_pending, None
        cold_origin, self.slope_cold_origin = self.slope_cold_origin, None
        accepted = self._accepted()
        if coef is None:
            return
        raw = self._view().raw_prediction
        found = [
            bundle
            for bundle in held
            if np.array_equal(bundle.theta, coef) and np.array_equal(bundle.index, raw)
        ]
        if not found:
            raise ValueError("the accepted trial was not evaluated in this search")
        self.previous, self.accepted = accepted, found[-1]
        if slopes is not None and slopes[0] is accepted:
            self.slope_seed = slopes[1]
        elif cold_origin is accepted:
            self.slope_seed = None

    def search(
        self,
        solver: Solver,
        options: SolverOptions,
        tally: Tally,
        *,
        absolute_sum: AbsoluteSum,
    ) -> bool:
        """The existing line search along ``d_q``, its accepted bundle committed.

        The accepted slopes, index and criterion become the provider's view; a refused
        or interrupted search leaves both views at the accepted point.
        """
        view = self._view()
        d_q = self._direction().d_q
        self.begin()
        accepted = False
        try:
            accepted = line_search(
                view, d_q, self._trial, options, tally, absolute_sum=absolute_sum
            )
        finally:
            self.end(view.coef if accepted else None)
        if accepted:
            solver.coef = view.coef
            solver.raw_prediction = view.raw_prediction
            solver.loss_value = view.loss_value
        return accepted

    def converged(self, solver: Solver) -> bool:
        """``max |G| <= tol`` at the accepted trial, then ``0.5 decrement2 <= tol``.

        The two formulas of scikit-learn's test at its place, on the full score and on
        the unhalved direction's decrement at its origin.
        """
        if not np.max(np.abs(self._view().gradient)) <= solver.tol:
            return False
        return 0.5 * self._direction().decrement <= solver.tol

    def full_gradient(self) -> FloatArray:
        """The full score of the accepted point."""
        return self._view().gradient

    def accepted_state(self, solver: Solver, tally: Tally) -> AcceptedState:
        """The accepted whole point: ``gamma``, ``q``, ``L/A`` and ``G``, once known."""
        view = self.view
        if view is None:
            return AcceptedState(immutable_array(solver.coef), None, None, None)
        gradient = view.gradient if tally.gradient_evaluated else None
        return AcceptedState(
            immutable_array(view.coef),
            immutable_array(view.raw_prediction),
            float(view.loss_value),
            None if gradient is None else immutable_array(gradient),
        )

    # ------------------------------------------------------------- the hooks

    def started(self, loss: float) -> None:
        """The start is evaluated with ``loss``: it is accepted; ZC at its index."""
        opening = self._opened()
        self.accepted = Bundle(self._view().coef, opening.index, loss, None)
        _stop(self._alarm("initial", opening.index, chord=None, converged=False))

    def stepped(self, converged: bool, gradient: FloatArray) -> None:
        """A step was accepted and tested natively: indications, then the schedule."""
        self.steps += 1
        accepted = self._accepted()
        chord = None
        if self.gpml and self.previous is not None:
            chord = chord_alarm(
                self.context.problem.z,
                self.context.problem.weights,
                (self.previous.loss, self.previous.index),
                (accepted.loss, accepted.index),
            )
        _stop(self._alarm("accepted", accepted.index, chord=chord, converged=converged))
        score = float(np.max(np.abs(gradient)))
        controls = self.context.controls
        self.tolerance = next_tolerance(
            self.tolerance,
            score,
            start=controls.absorb_tol_start,
            final=controls.absorb_tol_final,
        )
        self.score = score

    def final_alarm(
        self, index: FloatArray, losses: tuple[float, float]
    ) -> Alarm | None:
        """ZC at the final index and, at ``tau = 2``, GC from the accepted criterion.

        ``losses`` holds the accepted and the final ``L/A``, both at the same ``q``,
        so no chord of a recovery is fabricated.
        """
        chord = None
        if self.gpml:
            problem = self.context.problem
            before, after = (losses[0], index), (losses[1], index)
            chord = chord_alarm(problem.z, problem.weights, before, after)
        return self._alarm("final", index, chord=chord, converged=True)

    def _alarm(
        self,
        where: Where,
        index: FloatArray,
        *,
        chord: ChordAlarm | None,
        converged: bool,
    ) -> Alarm | None:
        """ZC at ``index`` and ``chord`` coalesced into one kept alarm, or ``None``."""
        zero = zero_row_alarm(self.context.problem.z, index, self.context.positions)
        if zero is None and chord is None:
            return None
        self.alarm = Alarm(where, self.attempt, self.steps, converged, zero, chord)
        return self.alarm

    # ------------------------------------------------------------ the parts

    def _site(self) -> CallSite:
        """Where a direction's inner call runs: attempt, step, tolerance, score."""
        return (self.attempt, self.steps, "direction", self.tolerance, self.score)

    def _opened(self) -> IndexPoint:
        """The start's whole point; ``ValueError`` before it is initialized."""
        if self.opening is None:
            raise ValueError("the start of the attempt is not initialized")
        return self.opening

    def _view(self) -> SearchView:
        """The search view; ``ValueError`` before the setup."""
        if self.view is None:
            raise ValueError("the start of the attempt is not evaluated")
        return self.view

    def _accepted(self) -> Bundle:
        """The accepted bundle; ``ValueError`` before the start is evaluated."""
        if self.accepted is None:
            raise ValueError("the start of the attempt is not evaluated")
        return self.accepted

    def _parts(self) -> _Parts:
        """The current direction's inner results; ``ValueError`` before them."""
        if self.parts is None:
            raise ValueError("the direction's inner work has not run")
        return self.parts

    def _direction(self) -> Direction:
        """The current full direction; ``ValueError`` before it is completed."""
        if self.direction_found is None:
            raise ValueError("the full direction is not completed")
        return self.direction_found


def _same(coef: FloatArray, theta: FloatArray) -> None:
    """``ValueError`` unless the provider starts from the origin's slopes."""
    if not np.array_equal(coef, theta):
        raise ValueError("the attempt starts from the slopes of its origin")


def _required(status: str, converged: bool, reason: str) -> None:
    """Refuse a direction's inner call that did not converge.

    ``InnerLimitError`` for a spent budget, ``FloatingPointError`` for any other
    failure; nothing when it converged.
    """
    if status == BUDGET:
        raise InnerLimitError(reason)
    if not converged:
        raise FloatingPointError(reason)
