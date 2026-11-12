"""The pointwise loss of the convex Newton adapter: the package rows in the iteration.

The carried
``LinearModelLoss`` averages a pointwise loss of its raw prediction over
the rows.  :class:`PointLoss` supplies that loss from the row kernels of
:mod:`powermean.model.rows`, on every row and for the convex working variances (power,
``tau`` in ``[1, 2]``, and NB2).

At the raw prediction ``eta = X gamma`` that ``LinearModelLoss`` forms, the index is
``q = eta + o``: the transformed offset ``o`` enters once, with derivative one, and the
outcome is the ``y_true`` that ``LinearModelLoss`` passes, the transformed outcome
``z``.  For each row, with ``r``, ``c``, ``w`` and ``l`` of :mod:`powermean.model.rows`:

    loss                 l(q; z)                       evaluate_criterion
    gradient in eta      -r                            evaluate_rows
    hessian in eta       c (observed) or w (expected)  evaluate_rows

each multiplied by the weight of its row when one is passed.  ``LinearModelLoss`` then
divides by ``A``, the sum of the weights (the number of rows without them), before its
matrix products: its loss, gradient and hessian are ``L/A``, ``-S/A`` and ``C/A``, or
``H/A`` with the expected curvature, where :class:`powermean.systems.joint.JointSystem`
returns the sums ``L``, ``S``, ``C`` and ``H``.  The expected curvature changes the
hessian, hence the Newton direction, and nothing else: the loss and the gradient are
those of the criterion whatever the curvature.

The raw prediction holds no offset.  The line search moves it along ``X d``, the
prediction of a displacement ``d`` of the coefficients, which must not receive the
offset; the offset is added here, at every evaluation, and nowhere else.

Paths.  ``loss`` evaluates the criterion of the rows and nothing else;
``gradient_hessian`` and ``gradient`` evaluate the rows and never the criterion; only
``loss_gradient`` needs both.  The row kernel evaluates ``r``, ``c`` and ``w`` together,
so a gradient also forms the curvatures, as it does in the system.  An index or a value
that is not representable in double precision raises ``FloatingPointError``, as in the
system: nothing is clipped, floored or replaced.

Work.  Every call of ``loss``, ``loss_gradient`` and ``gradient_hessian`` is charged to
the :class:`Budget` of the solve before it is made: once
``limit`` calls are made, the next one raises
:class:`EvaluationLimitError` without being evaluated.  ``gradient``, which the
iteration never calls, is charged too.  One budget serves every attempt of a solve, so
a retry receives what is left, not a new allowance.
"""

from __future__ import annotations

from typing import Literal, Protocol, override

import numpy as np

from powermean.model.rows import evaluate_criterion, evaluate_rows
from powermean.model.variance import (
    FloatArray,
    NB2Variance,
    PowerVariance,
    _finite_result,
)
from powermean.solve.options import Curvature

# The calls of a pointwise loss, the names of their counts and the one call that
# the iteration does not make.
Call = Literal["loss", "loss_gradient", "gradient_hessian", "gradient"]
CALLS: tuple[Call, ...] = ("loss", "loss_gradient", "gradient_hessian", "gradient")


class EvaluationLimitError(RuntimeError):
    """The evaluations allowed to a solve are spent.

    Its name ends in ``Error`` as the package names its exceptions; internal, never a
    public status.
    """


class Budget:
    """The evaluations of one solve, counted by call and charged before each call.

    Parameters
    ----------
    limit : int
        The number of calls allowed, the option ``max_evaluations``.

    Attributes
    ----------
    calls : dict
        The calls made so far, by name.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls: dict[Call, int] = dict.fromkeys(CALLS, 0)

    @property
    def used(self) -> int:
        """The calls made so far, of every kind."""
        return sum(self.calls.values())

    def charge(self, call: Call) -> None:
        """Count one call of ``call``, or raise :class:`EvaluationLimitError`.

        The test precedes the count and the call: with ``limit`` calls made,
        the next one is refused and not counted.
        """
        if self.used >= self.limit:
            raise EvaluationLimitError("Pointwise loss/derivative evaluation limit")
        self.calls[call] += 1


def _weighted(values: FloatArray, sample_weight: FloatArray | None) -> FloatArray:
    """``values`` times the weights, or a copy without them: a new writable array.

    ``LinearModelLoss`` divides what it receives in place; the rows are read-only.
    """
    if sample_weight is None:
        return np.array(values, dtype=np.float64)
    product: FloatArray = sample_weight * values
    return product


class PointwiseLoss(Protocol):
    """What the carried ``LinearModelLoss`` needs from a pointwise loss.

    The interface of the losses of scikit-learn, which ``LinearModelLoss`` calls by
    keyword and whose returned arrays it divides in place; the protocol ``BaseLoss`` of
    the carried files states the same interface, which this module may not import.
    """

    n_classes: int | None
    is_multiclass: bool

    def loss(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """The loss of each row."""
        ...

    def loss_gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The loss of each row and its derivative in the raw prediction."""
        ...

    def gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """The derivative of the loss of each row in the raw prediction."""
        ...

    def gradient_hessian(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The first and second derivatives of the loss of each row."""
        ...


class PointLoss(PointwiseLoss):
    """The pointwise loss of the package rows at ``q = raw_prediction + offset``.

    Parameters
    ----------
    offset : array
        The transformed offset ``o`` of the rows that the iteration sees, aligned with
        the raw prediction.
    variance : PowerVariance or NB2Variance
        The working variance, of exact type, as :mod:`powermean.model.rows` requires.
    curvature : {'observed', 'expected'}
        The hessian of a row: ``c``, or ``w`` for Fisher scoring.
    budget : Budget
        The evaluations of the solve, charged before each call.

    Notes
    -----
    ``n_threads`` is accepted for the interface and not used: NumPy evaluates the rows.
    """

    n_classes: int | None = None
    is_multiclass: bool = False

    def __init__(
        self,
        *,
        offset: FloatArray,
        variance: PowerVariance | NB2Variance,
        curvature: Curvature,
        budget: Budget,
    ) -> None:
        self.offset = offset
        self.variance = variance
        self.curvature = curvature
        self.budget = budget

    def index(self, raw_prediction: FloatArray) -> FloatArray:
        """``q = raw_prediction + o``; an index that is not a double raises.

        Raises ``ValueError`` when the raw prediction is not aligned with the offset,
        and ``FloatingPointError``, as the system does, when an entry of ``q`` is not
        representable (a trial step whose prediction overflowed, for instance).
        """
        if raw_prediction.shape != self.offset.shape:
            raise ValueError("raw_prediction must be aligned with the offset")
        with np.errstate(over="ignore", invalid="ignore"):
            q: FloatArray = raw_prediction + self.offset
        return _finite_result(q, "index")

    @override
    def loss(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """``a l(q; z)`` of each row, from the criterion alone."""
        self.budget.charge("loss")
        q = self.index(raw_prediction)
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
        """``a l(q; z)`` and ``-a r`` of each row."""
        self.budget.charge("loss_gradient")
        q = self.index(raw_prediction)
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
        """``-a r`` of each row, from the rows alone."""
        self.budget.charge("gradient")
        rows = evaluate_rows(y_true, self.index(raw_prediction), variance=self.variance)
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
        """``-a r`` and ``a c``, or ``a w`` with the expected curvature, of each row."""
        self.budget.charge("gradient_hessian")
        rows = evaluate_rows(y_true, self.index(raw_prediction), variance=self.variance)
        if self.curvature == "observed":
            hessian = rows.observed_curvature
        else:
            hessian = rows.fisher_curvature
        return _weighted(-rows.score, sample_weight), _weighted(hessian, sample_weight)
