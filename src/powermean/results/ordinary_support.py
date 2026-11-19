"""Certify support on the ordinary weighted sample before a point is proposed."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import numpy as np

from powermean.data.collinearity_numeric import EPSILON, numerical_rank
from powermean.data.retained import OrderRows, OrderSample, after_separation
from powermean.data.sample import SampleBase
from powermean.data.sample_fe_state import SEPARATED, RowLedger, ledger_of, replicas_of
from powermean.data.weights import ObservationWeights
from powermean.model.order import BoolArray
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.solve.support import (
    BASELINE,
    SupportResult,
    analyze_support,
    require_current,
)
from powermean.solve.support_gamma import GammaProblem, GammaResult, ordinary_recession
from powermean.solve.support_problem import SupportProblem

Family = PowerVariance | NB2Variance


@dataclass(frozen=True, eq=False)
class OrdinarySeparation:
    """The working sample and its witnessed row changes, including empty support."""

    sample: OrderSample
    steps: tuple[SupportResult, ...] = ()
    exhausted: RowLedger | None = None
    full_rank: bool = False

    @property
    def unresolved(self) -> bool:
        """Whether the last bounded diagnosis left undecided zero rows."""
        return bool(self.steps and self.steps[-1].status != "complete")


def _positive_rank(sample: OrderSample, positive: BoolArray) -> bool:
    """Establish full positive-row rank and the ordinary full-design rank gate.

    The scaled columns have maximum magnitude less than two. The largest singular
    value of the full matrix is bounded by its Frobenius norm, at most
    ``2 sqrt(n k)``. Adding rows cannot decrease the smallest singular value.
    Also rule out a direction allowed by the numerical witness contract. Its
    positive-row norm is bounded by ``sqrt(n_positive) * tolerance``; a strict zero
    row needs displacement above ``threshold`` in the same coordinates. Multiplying
    the coefficient-norm bound by the largest possible row norm ``2 sqrt(k)``
    bounds every zero-row displacement. Uncertain gaps use the bounded diagnosis.
    """
    n, k = sample.design.matrix.shape
    n_positive = int(np.count_nonzero(positive))
    if n_positive < k:
        return False
    try:
        rank = numerical_rank(
            sample.design.matrix[positive],
            [],
            kind="direct_svd",
            purpose="positive rows",
        )
    except np.linalg.LinAlgError:
        return False
    full_cutoff = EPSILON * max(n, k) * 2 * sqrt(n * k)
    witness_ratio = max(
        BASELINE.lp.tolerance / BASELINE.lp.threshold,
        BASELINE.ir.tolerance / BASELINE.ir.threshold,
    )
    witness_cutoff = 2 * sqrt(n_positive * k) * witness_ratio
    return bool(rank.values[-1] > rank.cutoff + max(full_cutoff, witness_cutoff))


def _support_problem(sample: OrderSample, variance: Family) -> SupportProblem:
    data, transformed = sample.data, sample.problem
    return SupportProblem(
        transformed.response,
        data.design,
        data.column_names,
        None,
        sample.positions,
        data.weight_values,
        transformed.offset,
        variance,
        sample.omega,
    )


def _ledger(sample: OrderSample) -> RowLedger:
    if sample.separation is not None:
        return sample.separation
    rows = OrderRows(
        sample.omega,
        sample.positive_only,
        sample.conditioned,
        sample.positions,
        sample.retained_index,
        sample.codes,
    )
    return ledger_of(rows, sample.missing, replicas_of(sample.data, rows))


def _remove(
    sample: OrderSample, base: SampleBase, step: SupportResult
) -> tuple[OrderSample, RowLedger | None]:
    positions = step.positions
    if not positions.size:
        return sample, None
    ledger = _ledger(sample).exclude(positions, SEPARATED)
    if not ledger.count:
        return sample, ledger
    return after_separation(sample, base, ledger), None


def separation(
    sample: OrderSample, base: SampleBase, variance: Family
) -> OrdinarySeparation:
    """Complete the characterized separation step using only checked deletions."""
    if not np.any(sample.problem.response == 0):
        return OrdinarySeparation(sample)
    if _positive_rank(sample, sample.problem.response > 0):
        return OrdinarySeparation(sample, full_rank=True)
    steps: list[SupportResult] = []
    while True:
        result = analyze_support(_support_problem(sample, variance), BASELINE)
        steps.append(result)
        sample, exhausted = _remove(sample, base, result)
        if exhausted is not None or result.status != "partial":
            return OrdinarySeparation(sample, tuple(steps), exhausted)


def replay_separation(
    initial: OrderSample,
    base: SampleBase,
    variance: Family,
    steps: tuple[SupportResult, ...],
) -> OrdinarySeparation:
    """Rebind each claimed deletion to the original equation and checked evidence."""
    sample, exhausted = initial, None
    for index, step in enumerate(steps):
        require_current(step, _support_problem(sample, variance))
        if step.controls != BASELINE:
            raise ValueError("the support controls must match the ordinary fit")
        checked = replace(step)
        sample, exhausted = _remove(sample, base, checked)
        if (exhausted is not None or checked.status != "partial") != (
            index + 1 == len(steps)
        ):
            raise ValueError("the support evidence does not end on the admitted sample")
    return OrdinarySeparation(sample, steps, exhausted)


def gamma_problem(sample: OrderSample, variance: PowerVariance) -> GammaProblem:
    """Use the original coordinates, typed weights and transformed offsets."""
    data, transformed = sample.data, sample.problem
    weights = data.weights
    if weights is None:
        weights = ObservationWeights(np.ones(sample.count))
    return GammaProblem(
        transformed.response,
        data.design,
        data.column_names,
        None,
        sample.positions,
        weights,
        transformed.offset,
        variance,
        sample.omega,
    )


def recession(sample: OrderSample, variance: PowerVariance) -> GammaResult | None:
    """Return an unresolved or directional refusal before the ordinary Gamma solve."""
    # The actual ordinary sample already validated every outcome and positive weight.
    if sample.omega == 1.0 and np.all(sample.problem.response > 0):
        return None
    return ordinary_recession(gamma_problem(sample, variance), BASELINE.lp)
