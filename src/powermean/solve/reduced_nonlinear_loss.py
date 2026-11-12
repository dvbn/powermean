"""The retained-nonlinear backend of the existing shared Newton loop.

The first cold opening and the trials retain the other canonical FE coefficients
and solve the selected conditional block, not all blocks. Warm retry openings
and fresh final work keep their all-FE profiles. The
shared Newton iteration, evaluation budget, line search, retry and alarms are
not copied. The provider retains only its existing K-sized administrative view;
its inner solve is skipped because the full-current factor supplies the step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack, override

import numpy as np

from powermean.fe.absorb_numeric import checked_product
from powermean.fe.certificate_basis import embed
from powermean.fe.profile import ProfileState
from powermean.fe.profile_arithmetic import _cached_rows
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray
from powermean.solve import absorbed_rhs as rhs
from powermean.solve.absorbed_direction import (
    Direction,
    IndexContext,
    IndexLog,
    complete_direction,
)
from powermean.solve.absorbed_loss import (
    IndexLinearLoss,
    OriginRows,
    SearchView,
    trial_rows,
)
from powermean.solve.line_search import AbsoluteSum, line_search
from powermean.solve.newton_convex import LinearLoss, Sandwich, Solver
from powermean.solve.newton_trace import AcceptedState, NativeReason, Tally
from powermean.solve.options import SolverOptions
from powermean.solve.point_loss import PointLoss
from powermean.solve.profiled_linear_loss import HessianCall
from powermean.solve.profiled_loss import (
    REFUSED,
    Bundle,
    ProfileCall,
    ProfiledEvaluator,
    ProfileRole,
)
from powermean.solve.reduced_nonlinear import (
    ConditionalPoint,
    ConditionalSpace,
    _OPENING_SPACE_COORDINATES,
)

if TYPE_CHECKING:
    from powermean.solve.newton_profiled import _Context


class ReducedLinearLoss(IndexLinearLoss):
    """Reuse the full-row loss; no unused provider K curvature is uninitialized."""

    @override
    def gradient_hessian(
        self, **call: Unpack[HessianCall]
    ) -> tuple[FloatArray, FloatArray, bool | np.bool_]:
        call["gradient_out"].fill(0.0)
        call["hessian_out"].fill(0.0)
        return super().gradient_hessian(**call)


class ReducedEvaluator(ProfiledEvaluator):
    """One attempt, retained nonlinear state, and the existing transaction hooks."""

    def __init__(
        self, context: _Context, point: PointLoss, space: ConditionalSpace, *,
        attempt: int, warm: ProfileState | None,
    ) -> None:
        super().__init__(
            context.problem, context.absorbed, context.options.curvature,
            context.log, attempt=attempt, warm=warm,
            positions=context.sample.positions,
        )
        self.space = space
        self.context = IndexContext(
            context.problem, context.absorbed, context.options.curvature,
            context.sample.positions, IndexLog(), point, context.budget,
            context.options.gradient_tol,
        )
        self.view: SearchView | None = None
        self.outer_step: FloatArray | None = None
        self.full_step: Direction | None = None
        self.trial_derivative: float | None = None

    @property
    def factorized(self) -> bool:
        """The provider performs no separate K solve; one full factor supplies it."""
        return False

    def provider_loss(
        self, carried: LinearLoss, sandwich: Sandwich,
    ) -> ReducedLinearLoss:
        """Reuse the existing carried row arithmetic, not a second loss engine."""
        return ReducedLinearLoss(carried, self, sandwich=sandwich)

    @override
    def start(self, coef: FloatArray) -> FloatArray:
        """One conditional cold opening; keep the all-FE opening of a warm retry.

        The provider has already admitted this evaluation. The conditional root
        uses the original profile controls and is logged before reconstruction;
        it is not reported as a completed all-FE profile.
        """
        if (self.attempt != 0 or self.warm is not None
                or vars(type(self.space)).get("coordinates")
                is not _OPENING_SPACE_COORDINATES
                or vars(self.space).get("coordinates", _OPENING_SPACE_COORDINATES)
                is not _OPENING_SPACE_COORDINATES):
            return super().start(coef)
        point = self._conditional(self.space.opening(coef), "initial")
        self.opening = point
        self.warm = point.state
        return point.index

    def opened(self, solver: Solver) -> None:
        """Set the actual opening index before its existing indication check."""
        opening = self._opened()
        solver.raw_prediction = opening.index

    @override
    def started(self, loss: float) -> None:
        """Preserve initial ZC precedence before packing new working coordinates."""
        super().started(loss)
        opening = self._opened()
        values = (
            opening.coordinates if isinstance(opening, ConditionalPoint)
            else self.space.coordinates(opening.theta, opening.state)
        )
        self.view = SearchView(values, values, np.float64(loss))

    def row_direction(self, rows: OriginRows) -> None:
        """One full-current RHS; fresh original-level linear residual and tangent."""
        space = self.space
        backend = self.problem.__dict__["_reduced_execution"]
        solved, change, prepared = backend.joint_step(
            rows.curvature, rows.right
        )
        k = self.problem.n_slopes
        with np.errstate(all="ignore"):
            right = rows.right - rows.curvature * (self.problem.design @ solved[:k])
        if not np.isfinite(right).all():
            raise FloatingPointError("the reduced direction RHS is not representable")
        checked = rhs._evaluate(
            rhs._right(right, self.problem.n_rows), prepared,
            embed(space.basis, solved[k:, None]),
        )
        if checked.check is None or checked.check.maximum > self.tolerance:
            raise FloatingPointError(
                "the reduced full direction fails its fresh original-level bound"
            )
        # Same-q row reuse, not a cached numerical verdict. Expected curvature
        # may choose the outer metric; implicit roots still use the true Jacobian.
        evaluated = _cached_rows(
            self.problem.__dict__["_initial"], self._accepted().index
        )
        observed = checked_product(
            self.problem.weights, evaluated.observed_curvature,
            "conditional observed curvature",
        )
        tangent = space.tangent(solved, observed)
        with np.errstate(all="ignore"):
            derivative = rows.v @ tangent
        if not np.isfinite(derivative):
            raise FloatingPointError("the conditional derivative is not representable")
        self.trial_derivative = float(derivative)
        self.outer_step = immutable_array(solved[space.remaining])
        self.full_step = complete_direction(
            None, solved[:k], change, (rows.v, rows.scaled),
            tolerance=self.tolerance,
        )
        view = self._view()
        view.gradient = rows.gradient

    def completed(self, solver: Solver) -> NativeReason | None:
        """Hand the shared line search the retained step and full derivative."""
        if (self.full_step is None or self.outer_step is None
                or self.trial_derivative is None):
            raise ValueError("no current reduced direction was evaluated")
        solver.coef_newton = np.array(self.full_step.d_gamma, copy=True)
        solver.gradient_times_newton = np.float64(self.trial_derivative)
        if self.trial_derivative > 0:
            return "non_descent_direction"
        view = self._view()
        view.coef_newton = self.outer_step
        view.gradient_times_newton = np.float64(self.trial_derivative)
        return None

    def _conditional(
        self, values: FloatArray, role: ProfileRole
    ) -> ConditionalPoint:
        """Log the actual one-block solve before its checked reconstruction."""
        limits = (self.controls.profile_tol, self.controls.profile_max_sweeps)
        site = (self.attempt, self.steps, role, *limits)
        try:
            solved = self.space.solve(
                values, tolerance=limits[0], max_sweeps=limits[1], warm=self.warm
            )
        except FloatingPointError as error:
            self.log.profiles.append(ProfileCall(
                *site, REFUSED, str(error), None, None, None,
                scope="conditional", block=self.space.block,
            ))
            raise
        found = solved.result
        self.log.profiles.append(ProfileCall(
            *site, found.status, found.reason, found.work, solved.scalar,
            found.maximum, scope="conditional", block=self.space.block,
        ))
        return self.space.reconstruct(solved)

    def _trial(
        self, values: FloatArray, _raw: FloatArray
    ) -> tuple[np.float64, FloatArray]:
        """Admit once, solve only the conditional block, evaluate full fresh rows."""
        self.context.admissions.admit("loss_gradient")
        point = self._conditional(values, "trial")
        rows = trial_rows(self.context, point.theta, point.index, 0.0)
        self.held = [Bundle(point, float(rows.loss), rows.gradient)]
        self.warm = point.state
        return rows.loss, rows.gradient

    @override
    def end(self, coef: FloatArray | None) -> None:
        """Commit the actual conditional trial, or restore the accepted warm state."""
        held, self.held = self.held, []
        accepted = self._accepted()
        if coef is None:
            self.warm = accepted.state
            return
        found = [
            bundle for bundle in held
            if isinstance(bundle.point, ConditionalPoint)
            and np.array_equal(bundle.point.coordinates, coef)
        ]
        if not found:
            raise ValueError("the accepted reduced trial was not evaluated")
        self.previous, self.accepted = accepted, found[-1]
        self.warm = self.accepted.state

    def search(
        self, solver: Solver, options: SolverOptions, tally: Tally, *,
        absolute_sum: AbsoluteSum,
    ) -> bool:
        """The same Armijo/roundoff search, with retained nonlinear trial values."""
        view = self._view()
        if self.outer_step is None:
            raise ValueError("no reduced direction was evaluated")
        self.begin()
        accepted = False
        try:
            accepted = line_search(
                view, self.outer_step, self._trial, options, tally,
                absolute_sum=absolute_sum,
            )
        finally:
            self.end(view.coef if accepted else None)
        if accepted:
            point = self._accepted()
            solver.coef = np.array(point.theta, copy=True)
            solver.raw_prediction = point.index
            solver.loss_value = np.float64(point.loss)
        return accepted

    def converged(self, solver: Solver) -> bool:
        """Existing native score/decrement formulas on the actual full score."""
        if self.full_step is None:
            raise ValueError("no reduced direction was evaluated")
        return bool(
            np.max(np.abs(self._view().gradient)) <= solver.tol
            and 0.5 * self.full_step.decrement <= solver.tol
        )

    def full_gradient(self) -> FloatArray:
        """The actual full original score, not merely retained/slope entries."""
        return self._view().gradient

    def accepted_state(self, solver: Solver, tally: Tally) -> AcceptedState:
        """Keep ordinary slope/q coordinates and full score in the native trace."""
        accepted = self.accepted
        if accepted is None:
            return AcceptedState(immutable_array(solver.coef), None, None, None)
        gradient = self.full_gradient() if tally.gradient_evaluated else None
        return AcceptedState(
            immutable_array(accepted.theta), immutable_array(accepted.index),
            float(accepted.loss), None if gradient is None
            else immutable_array(gradient),
        )

    def _view(self) -> SearchView:
        """The retained-coordinate search view, available after setup only."""
        if self.view is None:
            raise ValueError("the reduced opening has not been evaluated")
        return self.view
