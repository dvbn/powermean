"""The small no-FE separation linear program, with fresh primal and dual witnesses.

This is the linear-programming step on the regressors when there are no absorbed
effects.  Internal module.

The program.  With ``A = X / s`` (the power-of-two column scales of
:func:`powermean.solve.support_projection.scales`), the zero rows ``R`` not yet in the
witnessed set ``S``, and ``c = 1/|R|`` on ``R`` and 0 on ``S``, each solve is

    minimize  -c' A_O v   subject to  -A_O v <= 0,  A_P v = 0,  -1 <= v <= 1,

every original constraint retained, by HiGHS dual simplex (``method="highs-ds"``,
presolve on) with the explicit feasibility tolerances and iteration budget of
:class:`LPControls`.  A status other than optimal, a missing or nonfinite output, or any
warning raised during the call (an unknown or replaced option included) is a failure,
never a support result; ``success`` and ``fun`` are recorded, not trusted.

Witnesses.  The returned ``v`` must lie in its box within the feasibility tolerance (the
optimizer's own constraint), then ``beta = v / s`` (exact for powers of two)
is checked afresh by :func:`powermean.solve.support_witness.primal_witness`.  A valid
witness with a strict row outside ``S`` adds its strict set and the next solve maximizes
over the rows still uncovered; all checked witnesses are kept, since a later vertex need
not cover earlier rows.  At most ``|O|`` additions precede a terminal solve.  Otherwise
SciPy's marginals (``m_I``, ``m_E``, ``m_L``, ``m_U``) are mapped to the original-row
candidate ``l_O = c - m_I``, ``l_P = m_E``, for which exact stationarity gives ``A' l =
-(m_L + m_U)``; the box contributions ``m_L + m_U`` are checked afresh per column
against ``e_LP sum_i |A_ij l_i|`` (the dual check's range-checked quotient, so that a
lost box ratio fails the terminal check), and the candidate, zeroed on ``S``, by
:func:`powermean.solve.support_witness.dual_witness` on the original columns.  When both
pass the set is numerically complete; otherwise the stop is ``lp_no_completion``, never
an absence of separation.  Without slopes the cone is ``{0}`` and ``l = 1_O`` is
checked.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np

from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_product,
    checked_quotient,
    checked_sum,
    group_sums,
)
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.support_problem import NONE, SupportProblem, indices
from powermean.solve.support_projection import scales
from powermean.solve.support_witness import (
    DualWitness,
    PrimalWitness,
    dual_witness,
    primal_witness,
    relative,
    union,
)

METHOD = "highs-ds"


class Marginals(Protocol):
    """The ``ineqlin``, ``eqlin``, ``lower`` or ``upper`` field of a result."""

    marginals: FloatArray | None


class LinprogResult(Protocol):
    """The fields of ``scipy.optimize.linprog``'s result that this module reads."""

    x: FloatArray | None
    fun: float | None
    status: int
    success: bool
    message: str
    nit: int
    ineqlin: Marginals
    eqlin: Marginals
    lower: Marginals
    upper: Marginals


class Linprog(Protocol):
    """``scipy.optimize.linprog``, the constraint arrays by position."""

    def __call__(
        self,
        c: FloatArray,
        a_ub: FloatArray,
        b_ub: FloatArray,
        a_eq: FloatArray | None,
        b_eq: FloatArray | None,
        /,
        *,
        bounds: tuple[float, float],
        method: str,
        options: Mapping[str, float | int | bool],
    ) -> LinprogResult:
        """The solution of the linear program."""
        ...


class Optimize(Protocol):
    """The object of ``scipy.optimize`` that this module uses."""

    linprog: Linprog


optimize = cast("Optimize", importlib.import_module("scipy.optimize"))


def _positive(value: object, name: str) -> float:
    """A positive finite float, or ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, float) or not value > 0:
        raise ValueError(f"{name} must be a positive finite float")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a positive finite float")
    return value


@dataclass(frozen=True)
class LPControls:
    """The explicit controls of the separation LP.

    ``tolerance`` and ``threshold`` are ``e_LP`` and ``d_LP``, ``0 < e < d < 1``;
    ``feasibility`` is HiGHS's primal and dual feasibility tolerance; ``max_iterations``
    its iteration budget per solve.
    """

    tolerance: float
    threshold: float
    feasibility: float
    max_iterations: int

    def __post_init__(self) -> None:
        """Check every control; refuse ``e >= d`` or ``d >= 1``."""
        e = _positive(self.tolerance, "tolerance")
        d = _positive(self.threshold, "threshold")
        _positive(self.feasibility, "feasibility")
        if not e < d < 1:
            raise ValueError("the tolerance must be below the threshold, below 1")
        budget = self.max_iterations
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("max_iterations must be a positive integer")

    @property
    def options(self) -> Mapping[str, float | int | bool]:
        """The requested HiGHS options, passed verbatim and recorded."""
        return MappingProxyType(
            {
                "presolve": True,
                "primal_feasibility_tolerance": self.feasibility,
                "dual_feasibility_tolerance": self.feasibility,
                "maxiter": self.max_iterations,
            }
        )


@dataclass(frozen=True, eq=False)
class LPSolve:
    """One charged solve: what the optimizer returned and its fresh primal check.

    ``remaining`` are the objective rows; ``status``, ``success``, ``message``,
    ``iterations``, ``reported`` (``fun``) and ``warnings`` the optimizer's report;
    ``coefficients`` ``v`` and the marginals (zero rows, positive rows, lower, upper)
    when present; ``objective`` the recomputed ``-c' A_O v``; ``failure`` empty, or
    ``lp_failure``, ``lp_box`` or ``range``.
    """

    remaining: IndexArray
    status: int
    success: bool
    message: str
    iterations: int
    reported: float | None
    warnings: tuple[str, ...]
    failure: str
    coefficients: FloatArray | None = None
    marginals: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None
    objective: float | None = None
    primal: PrimalWitness | None = None


@dataclass(frozen=True, eq=False)
class LPDual:
    """The terminal mapping: the original-row candidate and the box contributions.

    ``mapped`` is ``l`` (``c - m_I`` on ``O``, ``m_E`` on ``P``), ``box`` is ``m_L +
    m_U``, ``stationarity`` is ``A' l + box`` (a diagnostic of the optimizer's report)
    and ``box_ratios`` the fresh per-column ratios; ``witness`` the zeroed candidate's
    fresh dual check.
    """

    mapped: FloatArray
    box: FloatArray
    stationarity: FloatArray
    box_ratios: FloatArray
    witness: DualWitness | None
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether the box contributions and the dual witness both pass."""
        box = bool((self.box_ratios <= self.tolerance).all())
        return box and self.witness is not None and self.witness.passed


@dataclass(frozen=True, eq=False)
class LPOutcome:
    """The separation LP of one invocation: its solves, witnesses and verdict.

    ``witnesses`` are the valid primal witnesses whose strict sets cover ``S``;
    ``completion`` is ``covered`` (every zero row witnessed), ``dual`` (a passing dual
    and box check) or empty; ``reason`` names the stop.
    """

    controls: LPControls
    method: str
    options: Mapping[str, float | int | bool]
    scales: FloatArray
    solves: tuple[LPSolve, ...]
    witnesses: tuple[PrimalWitness, ...]
    dual: LPDual | None
    completion: str
    reason: str


def _objective_rows(problem: SupportProblem, remaining: IndexArray) -> FloatArray:
    """``c`` on the zero rows: ``1/|R|`` on ``R``, 0 on the witnessed rows."""
    rows = np.zeros(problem.zero.size)
    rows[np.isin(problem.zero, remaining)] = 1.0 / remaining.size
    return rows


def _call(
    problem: SupportProblem, a: FloatArray, rows: FloatArray, controls: LPControls
) -> tuple[LinprogResult, tuple[str, ...]]:
    """One HiGHS dual-simplex call with every warning of the call recorded."""
    zero, positive = a[problem.zero], a[problem.positive]
    # |A_ij| < 2 (power-of-two scales) and the weights c sum to 1: |cost_j| < 2.
    cost = -(zero.T @ rows)
    a_eq = positive if positive.shape[0] else None
    b_eq = np.zeros(positive.shape[0]) if positive.shape[0] else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimize.linprog(
            cost,
            -zero,
            np.zeros(zero.shape[0]),
            a_eq,
            b_eq,
            bounds=(-1.0, 1.0),
            method=METHOD,
            options=controls.options,
        )
    return result, tuple(str(item.message) for item in caught)


def _marginals(
    problem: SupportProblem, result: LinprogResult
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
    """The four marginal arrays when every one is present, shaped and finite."""
    found = (result.ineqlin, result.eqlin, result.lower, result.upper)
    shapes = (problem.zero.size, problem.positive.size, problem.n_slopes)
    expected = (shapes[0], shapes[1], shapes[2], shapes[2])
    arrays = []
    for field, size in zip(found, expected, strict=True):
        values = field.marginals
        if values is None or np.shape(values) != (size,):
            return None
        arrays.append(immutable_array(np.asarray(values, dtype=np.float64)))
    if not all(np.isfinite(x).all() for x in arrays):
        return None
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _solve(
    problem: SupportProblem,
    a: FloatArray,
    s: FloatArray,
    remaining: IndexArray,
    controls: LPControls,
) -> LPSolve:
    """Run one charged solve, validate its report and check its direction afresh."""
    rows = _objective_rows(problem, remaining)
    result, caught = _call(problem, a, rows, controls)
    reported = None if result.fun is None else float(result.fun)
    found = LPSolve(
        *(remaining, int(result.status), bool(result.success), str(result.message)),
        *(int(result.nit), reported, caught, "lp_failure"),
    )
    marginals = _marginals(problem, result)
    usable = not caught and result.status == 0 and result.success
    if not usable or result.x is None or marginals is None:
        return found
    v: FloatArray = np.array(result.x, dtype=np.float64, copy=True)
    if v.shape != (problem.n_slopes,) or not np.isfinite(v).all():
        return found
    found = replace(found, coefficients=immutable_array(v), marginals=marginals)
    if not float(np.max(np.abs(v))) <= 1.0 + controls.feasibility:
        return replace(found, failure="lp_box")
    try:
        with np.errstate(over="raise", invalid="raise"):
            objective = -float(rows @ (a[problem.zero] @ v))
        beta = checked_quotient(v, s, "original-coordinate slopes")
    except FloatingPointError:
        return replace(found, failure="range")
    witness = primal_witness(
        problem,
        beta,
        (),
        method="lp",
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    return replace(found, failure="", objective=objective, primal=witness)


Marginal = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def _terminal(
    problem: SupportProblem,
    a: FloatArray,
    rows: FloatArray,
    marginals: Marginal,
    *,
    separated: IndexArray,
    controls: LPControls,
) -> LPDual:
    """Map the terminal marginals to ``l``, check the box and the zeroed candidate.

    ``rows`` is the terminal ``c`` on the zero rows.
    """
    m_i, m_e, m_l, m_u = marginals
    mapped = np.zeros(problem.n_rows)
    tolerance = controls.tolerance
    single = indices(np.zeros(problem.n_rows))
    try:
        mapped[problem.zero] = checked_difference(rows, m_i, "mapped zero rows")
        mapped[problem.positive] = m_e
        box = checked_sum(m_l, m_u, "box contributions")
        products = checked_product(a, mapped[:, None], "stationarity products")
        stationarity = checked_sum(
            group_sums(products, single, 1)[0], box, "stationarity"
        )
        den = group_sums(np.abs(products), single, 1)[0]
        ratios = relative(np.abs(box), den)
    except FloatingPointError:
        failed = immutable_array(np.full(problem.n_slopes, np.inf))
        return LPDual(immutable_array(mapped), failed, failed, failed, None, tolerance)
    witness = dual_witness(
        problem,
        mapped,
        separated,
        tolerance=tolerance,
        threshold=controls.threshold,
    )
    return LPDual(
        immutable_array(mapped),
        immutable_array(box),
        immutable_array(stationarity),
        immutable_array(ratios),
        witness,
        tolerance,
    )


def _no_columns(problem: SupportProblem, controls: LPControls) -> LPDual:
    """The direct branch without slopes: the cone is ``{0}``; ``l = 1_O`` is checked."""
    mapped = np.zeros(problem.n_rows)
    mapped[problem.zero] = 1.0
    witness = dual_witness(
        problem,
        mapped,
        NONE,
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    empty = immutable_array(np.zeros(0))
    found = (immutable_array(mapped), empty, empty, empty)
    return LPDual(*found, witness, controls.tolerance)


def separation_lp(problem: SupportProblem, controls: LPControls) -> LPOutcome:
    """The separation LP of one fresh invocation on the current sample.

    Parameters
    ----------
    problem : SupportProblem
        The current sample, without blocks and with at least one zero row.
    controls : LPControls
        The explicit tolerance, threshold, feasibility tolerance and iteration budget.

    Returns
    -------
    LPOutcome
        Every charged solve, the valid witnesses covering ``S`` and the stop:
        ``covered``, ``dual``, ``lp_no_completion``, ``lp_failure``, ``lp_box`` or
        ``range``.  ``ValueError`` with blocks or without a zero row.
    """
    if problem.grouping is not None or not problem.zero.size:
        raise ValueError("the LP needs a problem without blocks and with a zero row")
    s = scales(problem.design)
    record = (controls, METHOD, controls.options, s)
    if not problem.n_slopes:
        dual = _no_columns(problem, controls)
        done = "dual" if dual.passed else ""
        return LPOutcome(*record, (), (), dual, done, done or "lp_no_completion")
    try:
        a = checked_quotient(problem.design, s[None, :], "scaled design")
    except FloatingPointError:
        return LPOutcome(*record, (), (), None, "", "range")
    witnesses: list[PrimalWitness] = []
    solves: list[LPSolve] = []
    while True:
        separated = union(witnesses)
        remaining = indices(np.setdiff1d(problem.zero, separated))
        if not remaining.size:
            covered = (tuple(solves), tuple(witnesses), None)
            return LPOutcome(*record, *covered, "covered", "covered")
        solve = _solve(problem, a, s, remaining, controls)
        solves.append(solve)
        marginals = solve.marginals
        if solve.failure or marginals is None:
            stopped = (tuple(solves), tuple(witnesses), None)
            return LPOutcome(*record, *stopped, "", solve.failure or "lp_failure")
        primal = solve.primal
        if primal is not None and _progress(primal, separated):
            witnesses.append(primal)
            continue
        rows = _objective_rows(problem, remaining)
        dual = _terminal(
            problem, a, rows, marginals, separated=separated, controls=controls
        )
        done = "dual" if dual.passed else ""
        found = (tuple(solves), tuple(witnesses), dual)
        return LPOutcome(*record, *found, done, done or "lp_no_completion")


def _progress(primal: PrimalWitness, separated: IndexArray) -> bool:
    """Whether a valid witness has a strict row outside the witnessed set."""
    return primal.valid and bool(np.setdiff1d(primal.strict, separated).size)
