"""The profiled linear loss: the carried linear loss at the profiled index, admitted.

The Newton iteration runs on the criterion with the effects profiled, and the Newton
system is formed by weighted absorption.  Internal module.
This module imports neither :mod:`powermean.solve.newton_convex`
nor :mod:`powermean._vendor`: the adapter, the one module that imports the carried
files, builds :class:`ProfiledLinearLoss` with the provider's carried
``LinearModelLoss`` and the carried ``sandwich_dot``, handed over explicitly
(:class:`CarriedLoss`, :class:`Sandwich`).

:class:`ProfiledLinearLoss` provides the interface of the carried ``LinearModelLoss``
that both providers' classes and the adapter's line search call by keyword.
``loss`` evaluates the start of an attempt, ``loss_gradient`` a trial and
``gradient_hessian`` the accepted point: each takes the profiled index from the
evaluation of the attempt (:class:`PointEvaluation`) and hands it, as the raw
prediction, to the carried linear loss of the package's pointwise loss, whose offset is
zero since the index holds the transformed offset.  The value and gradient are thus the
carried arithmetic at the profiled index, with the slopes ``X``; a value or a gradient
that is not representable raises ``FloatingPointError``, which refuses a trial.
``gradient_hessian`` is the carried arithmetic of one output without
intercept, step for step, with one change: the sandwich is formed with the slopes
absorbed on the effects with the per-row hessian of the pointwise loss (``a c``, or
``a w`` with the expected curvature), the Schur complement of the effects.  The convex
scope has no negative row curvature, so the carried warning, computed by its rule, is
never raised.

Evaluations.  Each call is one evaluation of the solve's one budget
(:class:`Admissions`), charged BEFORE any of its work: the profile of the start or of
the trial, the rows and the direction's absorption.  A spent budget raises
``EvaluationLimitError`` before a profile starts; an admitted call whose profile, rows
or absorption then fails stays charged once.  The pointwise loss charges the same
budget, which counts the admitted call once: no second count, refund or new allowance.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, Unpack, override

import numpy as np

from powermean.fe.profile_arithmetic import (
    ProfileProblem,
    _cached_criterion,
    _cached_rows,
)
from powermean.model.rows import evaluate_criterion, evaluate_rows
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve.options import Curvature
from powermean.solve.point_loss import Budget, Call, PointLoss, _weighted
from powermean.systems.profiled import ProfiledProblem

Intercept = float | FloatArray


class LossCall(TypedDict):
    """The keywords of the providers' calls of a linear loss's value."""

    coef: FloatArray
    X: FloatArray
    y: FloatArray
    sample_weight: FloatArray
    l2_reg_strength: float
    n_threads: int
    raw_prediction: FloatArray


class HessianCall(LossCall):
    """The keywords of the providers' calls of a linear loss's derivatives."""

    gradient_out: FloatArray
    hessian_out: FloatArray


class CarriedLoss(Protocol):
    """The carried ``LinearModelLoss`` of a provider, as the profiled loss calls it."""

    def weight_intercept_raw(
        self, coef: FloatArray, design: FloatArray, /
    ) -> tuple[FloatArray, Intercept, FloatArray]:
        """The weights, intercept and raw prediction ``X coef``."""
        ...

    def loss(self, **call: Unpack[LossCall]) -> np.float64:
        """The averaged loss at the raw prediction of ``call``."""
        ...

    def loss_gradient(self, **call: Unpack[LossCall]) -> tuple[np.float64, FloatArray]:
        """The averaged loss and its gradient at the raw prediction of ``call``."""
        ...


class Sandwich(Protocol):
    """The carried sandwich product: ``design' diag(weights) design``."""

    def __call__(self, design: FloatArray, weights: FloatArray, /) -> FloatArray:
        """The product of the carried ``sandwich_dot``."""
        ...


class PointEvaluation(Protocol):
    """The profiled points of an attempt, as the profiled loss calls them."""

    def start(self, coef: FloatArray, /) -> FloatArray:
        """The profiled index of the start of the attempt."""
        ...

    def trial(self, coef: FloatArray, /) -> FloatArray:
        """The profiled index of a trial of the line search, profiled afresh."""
        ...

    def evaluated(self, coef: FloatArray, loss: float, gradient: FloatArray, /) -> None:
        """The trial ``coef`` was evaluated with this loss and gradient."""
        ...

    def current(self, coef: FloatArray, /) -> FloatArray:
        """The profiled index of the accepted point, which ``coef`` must be."""
        ...

    def direction(self, weights: FloatArray, /) -> FloatArray:
        """The slopes absorbed on the effects with the curvature ``weights``."""
        ...


class Admissions(Budget):
    """The one budget of a profiled solve, each evaluation charged before its work.

    The profiled loss admits a call (:meth:`admit`: the test and the count of
    :meth:`Budget.charge`) before its profile, rows and absorption; the pointwise loss's
    own charge then consumes that admission (:meth:`charge`) without counting it again.
    One count per call, made before any of its work, whether that work completes or
    fails.  ``limit`` and ``calls`` are those of :class:`Budget`, read as before.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(limit)
        self.admitted: Call | None = None

    def admit(self, call: Call) -> None:
        """Charge one ``call`` before its work, or raise ``EvaluationLimitError``.

        An earlier admitted call whose work ended before its pointwise loss (a profile
        that failed, say) stays charged: nothing is refunded.
        """
        self.admitted = None
        super().charge(call)
        self.admitted = call

    @override
    def charge(self, call: Call) -> None:
        """The pointwise loss's charge of the admitted ``call``: consumed, not counted.

        ``ValueError`` when no admission of ``call`` is pending: a profiled evaluation
        is charged when it is admitted, before its work.
        """
        if self.admitted != call:
            raise ValueError("a profiled evaluation is charged when it is admitted")
        self.admitted = None


def _is_zero_offset(offset: FloatArray, n_rows: int) -> bool:
    """Whether ``offset`` is a float64 vector of ``n_rows`` positive zeros."""
    return (
        type(offset) is np.ndarray
        and offset.dtype == np.float64
        and offset.shape == (n_rows,)
        and not offset.any()
        and not np.signbit(offset).any()
    )


def _point_initial(
    point: PointLoss, problem: object, y_true: FloatArray
) -> ProfileProblem | None:
    """Return ``problem._initial`` when ``point`` is the problem's zero-offset loss."""
    if (
        type(point) in (PointLoss, ProfiledPointLoss)
        and type(problem) is ProfiledProblem
        and y_true is problem.z
        and point.variance == problem.variance
        and _is_zero_offset(point.offset, problem.n_rows)
    ):
        return getattr(problem, "_initial", None)
    return None


class ProfiledPointLoss(PointLoss):
    """Zero-offset :class:`PointLoss` sharing a :class:`ProfiledProblem` row cache."""

    def __init__(
        self,
        problem: ProfiledProblem,
        *,
        offset: FloatArray,
        variance: PowerVariance | NB2Variance,
        curvature: Curvature,
        budget: Budget,
    ) -> None:
        super().__init__(
            offset=offset,
            variance=variance,
            curvature=curvature,
            budget=budget,
        )
        self._problem = problem

    @override
    def index(self, raw_prediction: FloatArray) -> FloatArray:
        """``q = raw_prediction + o``, avoiding a copy on immutable ``+0.0`` offsets."""
        if (
            type(raw_prediction) is np.ndarray
            and raw_prediction.dtype == np.float64
            and not raw_prediction.flags.writeable
            and raw_prediction.shape == self.offset.shape
            and not self.offset.flags.writeable
            and _is_zero_offset(self.offset, self._problem.n_rows)
        ):
            if not np.isfinite(raw_prediction).all():
                raise FloatingPointError("index is not finite")
            zero_mask = raw_prediction == 0.0
            if not zero_mask.any() or not np.signbit(raw_prediction[zero_mask]).any():
                return raw_prediction
        return super().index(raw_prediction)

    @override
    def loss(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """``a l(q; z)`` of each row, sharing ``problem._initial``'s 1-slot cache."""
        self.budget.charge("loss")
        q = self.index(raw_prediction)
        initial = _point_initial(self, self._problem, y_true)
        if initial is not None:
            values = _cached_criterion(initial, q)
        else:
            values = evaluate_criterion(y_true, q, variance=self.variance)
        return _weighted(values, sample_weight)

    @override
    def loss_gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """``a l(q; z)`` and ``-a r`` of each row, sharing ``problem._initial``."""
        self.budget.charge("loss_gradient")
        q = self.index(raw_prediction)
        initial = _point_initial(self, self._problem, y_true)
        if initial is not None:
            values = _cached_criterion(initial, q)
            rows = _cached_rows(initial, q)
        else:
            values = evaluate_criterion(y_true, q, variance=self.variance)
            rows = evaluate_rows(y_true, q, variance=self.variance)
        return _weighted(values, sample_weight), _weighted(-rows.score, sample_weight)

    @override
    def gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """``-a r`` of each row, sharing ``problem._initial``."""
        self.budget.charge("gradient")
        q = self.index(raw_prediction)
        initial = _point_initial(self, self._problem, y_true)
        if initial is not None:
            rows = _cached_rows(initial, q)
        else:
            rows = evaluate_rows(y_true, q, variance=self.variance)
        return _weighted(-rows.score, sample_weight)

    @override
    def gradient_hessian(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """``-a r`` and ``a c`` (or ``a w``) of each row, sharing ``_initial``."""
        self.budget.charge("gradient_hessian")
        q = self.index(raw_prediction)
        initial = _point_initial(self, self._problem, y_true)
        if initial is not None:
            rows = _cached_rows(initial, q)
        else:
            rows = evaluate_rows(y_true, q, variance=self.variance)
        if self.curvature == "observed":
            hessian = rows.observed_curvature
        else:
            hessian = rows.fisher_curvature
        return _weighted(-rows.score, sample_weight), _weighted(hessian, sample_weight)


def _finite(value: np.float64, gradient: FloatArray | None = None) -> None:
    """``FloatingPointError`` unless the value and the gradient are finite."""
    finite = bool(np.isfinite(value))
    if gradient is not None:
        finite = finite and bool(np.isfinite(gradient).all())
    if not finite:
        raise FloatingPointError("a profiled loss is not representable in float64")


class ProfiledLinearLoss:
    """The carried linear loss at the profiled index: one object for both providers.

    Parameters
    ----------
    carried : CarriedLoss
        The provider's carried ``LinearModelLoss`` of ``point``.
    point : PointLoss
        The package's pointwise loss of zero offset, charging ``admissions``.
    evaluation : PointEvaluation
        The profiled points of the attempt.
    sandwich : Sandwich
        The carried ``sandwich_dot``, which the adapter hands over.
    admissions : Admissions
        The one budget of the solve, which every call charges before its work.

    ``ValueError`` when ``point`` charges another budget than ``admissions``.  See the
    module docstring for the arithmetic of each call.
    """

    fit_intercept = False

    def __init__(
        self,
        carried: CarriedLoss,
        point: PointLoss,
        evaluation: PointEvaluation,
        *,
        sandwich: Sandwich,
        admissions: Admissions,
    ) -> None:
        if point.budget is not admissions:
            raise ValueError("the pointwise loss must charge the solve's admissions")
        self.base_loss = point
        self.carried = carried
        self.evaluation = evaluation
        self.sandwich = sandwich
        self.admissions = admissions

    def weight_intercept_raw(
        self, coef: FloatArray, design: FloatArray
    ) -> tuple[FloatArray, Intercept, FloatArray]:
        """The carried ``weight_intercept_raw``: ``X coef``, without the offset."""
        return self.carried.weight_intercept_raw(coef, design)

    def loss(self, **call: Unpack[LossCall]) -> np.float64:
        """The averaged criterion at the profiled start of the attempt, admitted."""
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

    def loss_gradient(self, **call: Unpack[LossCall]) -> tuple[np.float64, FloatArray]:
        """The averaged criterion and gradient at a trial, admitted, profiled afresh."""
        coef = call["coef"]
        self.admissions.admit("loss_gradient")
        index = self.evaluation.trial(coef)
        value, gradient = self.carried.loss_gradient(
            coef=coef,
            X=call["X"],
            y=call["y"],
            sample_weight=call["sample_weight"],
            l2_reg_strength=call["l2_reg_strength"],
            n_threads=call["n_threads"],
            raw_prediction=index,
        )
        _finite(value, gradient)
        self.evaluation.evaluated(coef, float(value), gradient)
        return value, gradient

    def gradient_hessian(
        self, **call: Unpack[HessianCall]
    ) -> tuple[FloatArray, FloatArray, bool | np.bool_]:
        """The gradient with ``X`` and the Schur hessian at the accepted point."""
        coef, design, weights = call["coef"], call["X"], call["sample_weight"]
        grad, hess = call["gradient_out"], call["hessian_out"]
        n_features = design.shape[1]
        self.admissions.admit("gradient_hessian")
        current_index = self.evaluation.current(coef)
        initial = (
            _point_initial(
                self.base_loss,
                getattr(self.evaluation, "problem", None),
                call["y"],
            )
            if type(self.base_loss) is PointLoss
            else None
        )
        if initial is not None:
            self.base_loss.budget.charge("gradient_hessian")
            q = self.base_loss.index(current_index)
            rows = _cached_rows(initial, q)
            hessian = (
                rows.observed_curvature
                if self.base_loss.curvature == "observed"
                else rows.fisher_curvature
            )
            grad_pointwise = _weighted(-rows.score, weights)
            hess_pointwise = _weighted(hessian, weights)
        else:
            grad_pointwise, hess_pointwise = self.base_loss.gradient_hessian(
                y_true=call["y"],
                raw_prediction=current_index,
                sample_weight=weights,
                n_threads=call["n_threads"],
            )
        curvature = np.array(hess_pointwise, copy=True)
        sw_sum = np.sum(weights)
        grad_pointwise /= sw_sum
        hess_pointwise /= sw_sum
        hessian_warning = np.average(hess_pointwise < 0, weights=weights) > 0.25
        hess_pointwise = np.abs(hess_pointwise)
        grad[:n_features] = design.T @ grad_pointwise + call["l2_reg_strength"] * coef
        _finite(np.float64(0.0), grad)
        absorbed = self.evaluation.direction(curvature)
        hess[:n_features, :n_features] = self.sandwich(absorbed, hess_pointwise)
        return grad, hess, hessian_warning
