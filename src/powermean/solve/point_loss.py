"""The pointwise loss of the convex Newton adapter: the package rows in the iteration.

``design/02_computation.md``, Section 5.1, and decision D40.  The carried
``LinearModelLoss`` of task M0.9 averages a pointwise loss of its raw prediction over
the rows.  :class:`PointLoss` supplies that loss from the row kernels of
:mod:`powermean.model.rows`, on every row and for the convex working variances (power,
``tau`` in ``[1, 2]``, and NB2), where the solver campaign used the compiled losses of
scikit-learn on positive rows and overwrote its zero rows (``PointLoss`` of S2
``experiment.py``, ``StableCountedPointLoss`` of S1 ``zero_safe_kernel.py``).

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

Reuse.  The iteration asks for ``gradient_hessian`` at the raw prediction of the trial
that the line search has just accepted, whose rows ``loss_gradient`` evaluated.  The
rows of the last ``loss_gradient`` are held with the outcome and the index they were
evaluated at, and ``gradient_hessian`` returns them when its outcome and index have
the same bits and the working variance is the same object: the row kernel is a
function of these three alone, so the rows are those it would evaluate.  Anything
else, or a row kernel replaced at run time, is evaluated afresh.

Work.  Every call is charged to the :class:`Budget` of the solve before it is made, as
``CountedPointLoss`` of S1 ``engine.py`` charges ``loss``, ``loss_gradient`` and
``gradient_hessian``: once ``limit`` calls are made, the next one raises
:class:`EvaluationLimitError` without being evaluated.  ``gradient``, which the
iteration never calls, is charged too.  One budget serves every attempt of a solve, so
a retry receives what is left, not a new allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, override

import numpy as np

from powermean.model.rows import (
    RowEvaluation,
    _immutable,
    evaluate_criterion,
    evaluate_rows,
    poisson_exposure_rows,
)
from powermean.model.variance import (
    FloatArray,
    NB2Variance,
    PowerVariance,
    _finite_result,
)
from powermean.solve.options import Curvature

# The calls of a pointwise loss, the names of the counts of S1 and the one call that
# the iteration does not make.
Call = Literal["loss", "loss_gradient", "gradient_hessian", "gradient"]
CALLS: tuple[Call, ...] = ("loss", "loss_gradient", "gradient_hessian", "gradient")

# The row kernel of the module, so that a kernel replaced at run time is never reused.
_EVALUATE_ROWS = evaluate_rows


class EvaluationLimitError(RuntimeError):
    """The evaluations allowed to a solve are spent (S1 ``EvaluationLimit``).

    S1's exception, of the same base class and message, under a name that ends in
    ``Error`` as the package names its exceptions; internal, never a public status.
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
        The calls made so far, by name, as the counts of S1 ``PointLoss``.
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

        The test precedes the count and the call, as in S1: with ``limit`` calls made,
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


def _same_bits(found: object, held: FloatArray) -> bool:
    """Whether ``found`` is a float64 array of the shape and the bits of ``held``."""
    return bool(
        type(found) is np.ndarray
        and found.dtype == np.float64
        and found.shape == held.shape
        and np.array_equal(found.view(np.uint64), held.view(np.uint64))
    )


@dataclass(frozen=True, eq=False)
class _HeldRows:
    """The rows of one evaluation, with what they were evaluated at.

    ``z`` is a read-only copy of the outcome; the index is ``rows.log_mean``, the
    read-only copy of ``q`` that the kernel keeps; ``variance`` is the object passed.
    """

    z: FloatArray
    variance: PowerVariance | NB2Variance
    rows: RowEvaluation

    def matches(
        self, z: FloatArray, q: FloatArray, variance: PowerVariance | NB2Variance
    ) -> bool:
        """Whether ``evaluate_rows(z, q, variance=variance)`` evaluated these rows."""
        return (
            variance is self.variance
            and _same_bits(q, self.rows.log_mean)
            and _same_bits(z, self.z)
        )


def _held_rows(
    z: FloatArray,
    q: FloatArray,
    variance: PowerVariance | NB2Variance,
    rows: RowEvaluation,
    held: _HeldRows | None,
) -> _HeldRows | None:
    """The record of ``rows = evaluate_rows(z, q, variance=variance)``, to hold.

    ``None`` unless ``z`` is a float64 array and the kept index ``rows.log_mean`` has
    the bits of ``q``; the copy of ``z`` already held is kept when ``z`` has its bits.
    """
    if (
        type(z) is not np.ndarray
        or z.dtype != np.float64
        or not _same_bits(q, rows.log_mean)
    ):
        return None
    kept = held.z if held is not None and _same_bits(z, held.z) else _immutable(z)
    return _HeldRows(kept, variance, rows)


class PointwiseLoss(Protocol):
    """What the carried ``LinearModelLoss`` needs from a pointwise loss.

    The interface of the losses of scikit-learn, which ``LinearModelLoss`` calls by
    keyword and whose returned arrays it divides in place; the protocol ``BaseLoss`` of
    the carried files states the same interface, which this module may not import
    (rule 4 of ``design/11_engineering.md``, Section 3.1).
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
        exposure: FloatArray | None = None,
    ) -> None:
        self.offset = offset
        self.variance = variance
        self.curvature = curvature
        self.budget = budget
        self.exposure = exposure
        self._held: _HeldRows | None = None

    def _rows(
        self, y_true: FloatArray, q: FloatArray, raw: FloatArray, *, hold: bool
    ) -> RowEvaluation:
        """``evaluate_rows(y_true, q, variance=self.variance)``, or the held rows.

        The held rows are returned only for the same bits of ``y_true`` and ``q`` and
        the same variance object (module docstring); with ``hold``, fresh rows are
        held in their place.
        """
        held = self._held
        current = evaluate_rows is _EVALUATE_ROWS
        identity = q if self.exposure is None else raw
        if (
            current
            and held is not None
            and held.matches(y_true, identity, self.variance)
        ):
            return held.rows
        rows = (
            evaluate_rows(y_true, q, variance=self.variance)
            if self.exposure is None
            else poisson_exposure_rows(y_true, q, raw, self.exposure)
        )
        if hold and current:
            self._held = _held_rows(y_true, identity, self.variance, rows, held)
        return rows

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
        rows = self._rows(y_true, q, raw_prediction, hold=True)
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
        q = self.index(raw_prediction)
        rows = (
            evaluate_rows(y_true, q, variance=self.variance)
            if self.exposure is None
            else poisson_exposure_rows(y_true, q, raw_prediction, self.exposure)
        )
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
        rows = self._rows(
            y_true, self.index(raw_prediction), raw_prediction, hold=False
        )
        if self.curvature == "observed":
            hessian = rows.observed_curvature
        else:
            hessian = rows.fisher_curvature
        return _weighted(-rows.score, sample_weight), _weighted(hessian, sample_weight)
