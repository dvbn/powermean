"""Implicit unit-multiplier tau=2 criterion for the shared Newton attempt.

Ordinary centering gives
S=B't and H=B'diag(t)B at tau=2, t=z exp(-q). The criterion used here is
sum(t)/n, with gradient -B_id't/n and Hessian B_id'diag(t)B_id/n. The original
unnormalized cancelled equations remain in systems.cancelled for fresh caller
assessment; this optimization loss makes no certificate or existence decision.

The accepted row_components kernel supplies t, including its log-scale checks.
No cancellation row/group equations or ownership machinery are duplicated.
Curvature is formed one implicit column at a time in the provider's d-by-d buffer.

Admission covers the complete evaluation, including its earliest raw product.
Setup's raw helper reserves a loss; search's raw direction reserves its first
loss_gradient. The corresponding loss consumes that reservation. Later trials
and derivative updates charge normally. A failed product stays charged even
when no row kernel ran. A new loss instance starts each attempt, with one shared
Budget across retries; reservations never cross attempts.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, Unpack

import numpy as np

from powermean.fe._cancel_storage import finite_product
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.cancelled_operator import CancelledOperator, CancelledProblem
from powermean.solve.point_loss import Budget, Call
from powermean.systems._cancel_rows import row_components


class CancelledCall(TypedDict):
    """Exact single-output provider keywords on this implicit carrier."""

    coef: FloatArray
    X: CancelledOperator
    y: FloatArray
    sample_weight: FloatArray | None
    l2_reg_strength: float
    n_threads: int
    raw_prediction: NotRequired[FloatArray | None]


class CancelledHessianCall(CancelledCall):
    """Actual provider-owned derivative buffers, written in place."""

    gradient_out: FloatArray
    hessian_out: FloatArray


class SingleOutput:
    """The provider's loss metadata; no pointwise provider method is invoked."""

    is_multiclass: Literal[False] = False
    n_classes: None = None


def _buffer(values: FloatArray, shape: tuple[int, ...]) -> None:
    """Require the actual writable float64 provider buffer with its exact shape."""
    if values.dtype != np.dtype(np.float64) or values.shape != shape:
        raise ValueError("provider derivative buffer has the wrong dtype or shape")
    if not values.flags.writeable:
        raise ValueError("provider derivative buffers must be writable")


class CancelledLinearLoss:
    """The audited single-output linear-loss ABI on a genuine implicit operator.

    Parameters
    ----------
    problem : CancelledProblem
        Owned unit-weight tau=2 problem with explicit coordinates and resource plan.
    budget : Budget
        The solve-wide budget, shared without reset across actual retry attempts.

    Notes
    -----
    The provider uses base_loss only for is_multiclass/n_classes and this object's
    fit_intercept. It calls the four methods below; no carried dense loss method,
    provider solve/fallback, multinomial branch or implicit conversion is used.
    """

    fit_intercept: Literal[False] = False

    def __init__(self, problem: CancelledProblem, budget: Budget) -> None:
        self.problem = problem
        self.budget = budget
        self.base_loss = SingleOutput()
        self._raw_kind: Literal["loss", "loss_gradient"] = "loss"
        self._prepaid: Call | None = None

    def _admit(self, kind: Call) -> None:
        """Consume one successful raw reservation, or charge before any new work."""
        if self._prepaid is None:
            self.budget.charge(kind)
        elif self._prepaid != kind:
            raise ValueError("a raw product reservation must precede its matching loss")
        self._prepaid = None

    def weight_intercept_raw(
        self, coef: FloatArray, design: CancelledOperator
    ) -> tuple[FloatArray, float, FloatArray]:
        """Reserve BEFORE B_id coef; both start and direction exclude the offset."""
        if design is not self.problem.operator or self._prepaid is not None:
            raise ValueError(
                "raw prediction needs the bound operator and no pending loss"
            )
        kind = self._raw_kind
        self.budget.charge(kind)
        # Set the token only after success: failed work stays charged, never reused.
        raw = design.matvec(finite_product(coef))
        self._prepaid = kind
        self._raw_kind = "loss_gradient"
        return coef, 0.0, raw

    def _rows(self, call: CancelledCall) -> FloatArray:
        """Evaluate t through the accepted cancellation kernel, offset exactly once."""
        operator = self.problem.operator
        if call["X"] is not operator or call["sample_weight"] is not None:
            raise ValueError(
                "provider call must use the bound operator and unit weights"
            )
        if call["l2_reg_strength"] != 0 or call["n_threads"] != 1:
            raise ValueError(
                "cancelled attempt requires no penalty and one provider thread"
            )
        if not np.array_equal(call["y"], self.problem.z):
            raise ValueError(
                "provider outcomes do not match the owned transformed rows"
            )
        if call["coef"].shape != (operator.shape[1],):
            raise ValueError("coefficient length must match the bound operator")
        finite_product(call["coef"])
        raw = call.get("raw_prediction")
        if raw is None:
            raw = operator.matvec(call["coef"])
        if raw.shape != (operator.shape[0],):
            raise ValueError("raw prediction must have one value per input row")
        with np.errstate(over="raise", invalid="raise"):
            q = finite_product(raw + self.problem.centered_offset)
        rows = row_components(self.problem.z, q, np.zeros_like(q), PowerVariance(2))
        return rows.t

    def loss(self, **call: Unpack[CancelledCall]) -> np.float64:
        """sum(t)/n, with the setup raw reservation consumed exactly once."""
        self._admit("loss")
        rows = self._rows(call)
        with np.errstate(over="raise", invalid="raise"):
            value = np.float64(np.sum(rows) / rows.size)
        if not np.isfinite(value):
            raise FloatingPointError("cancelled criterion is not representable")
        return value

    def loss_gradient(
        self, **call: Unpack[CancelledCall]
    ) -> tuple[np.float64, FloatArray]:
        """sum(t)/n and -B_id't/n; subsequent contracted trials charge afresh."""
        self._admit("loss_gradient")
        rows = self._rows(call)
        with np.errstate(over="raise", invalid="raise"):
            value = np.float64(np.sum(rows) / rows.size)
            gradient = finite_product(-self.problem.operator.rmatvec(rows) / rows.size)
        if not np.isfinite(value):
            raise FloatingPointError("cancelled criterion is not representable")
        return value, gradient

    def gradient_hessian(
        self, **call: Unpack[CancelledHessianCall]
    ) -> tuple[FloatArray, FloatArray, bool]:
        """Write -B_id't/n and B_id'diag(t)B_id/n into the actual provider buffers."""
        self._admit("gradient_hessian")
        operator = self.problem.operator
        d = operator.shape[1]
        grad, hess = call["gradient_out"], call["hessian_out"]
        _buffer(grad, (d,))
        _buffer(hess, (d, d))
        inputs = (call["coef"], call["y"], call.get("raw_prediction"))
        if np.shares_memory(grad, hess) or any(
            values is not None
            and (np.shares_memory(grad, values) or np.shares_memory(hess, values))
            for values in inputs
        ):
            raise ValueError("provider buffers must not alias inputs or each other")
        rows = self._rows(call)
        with np.errstate(over="raise", invalid="raise"):
            gradient = finite_product(-operator.rmatvec(rows) / rows.size)
            basis = np.zeros(d, dtype=np.float64)
            for j in range(d):
                basis[j] = 1.0
                column = operator.matvec(basis)
                column *= rows
                hess[:, j] = finite_product(operator.rmatvec(column) / rows.size)
                basis[j] = 0.0
        # Leave the last accepted gradient intact if any Hessian product failed.
        grad[:] = gradient
        return grad, hess, False
