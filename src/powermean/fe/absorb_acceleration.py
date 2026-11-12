"""Lifted per-column symmetric conjugate gradient, with its work and breakdown state.

The accelerator pair is the established symmetric-Kaczmarz and conjugate-gradient
combination that ``reghdfe`` uses by default (``reghdfe.ado`` lines 226 to 227;
``reghdfe.mata`` ``accelerate_cg`` and ``transform_sym_kaczmarz``), taken as a
behavioral reference only: nothing of that source is copied, and its Hestenes stopping
rule, ``safe_divide`` floors, free "iteration 0" and one-block shortcut are not
inherited.  ``reghdfe`` itself warns against pairing CG with the forward
(nonsymmetric) Kaczmarz walk.

Operator.  ``T`` applies ``Q_1, ..., Q_G, Q_{G-1}, ..., Q_1`` (the literal symmetric
order of :func:`powermean.fe.absorb_numeric.symmetric_order`) and ``A z = z - T z``.
On the positive-weight support ``A`` is self-adjoint and positive semidefinite in
the weighted inner product, with the common block-mean-zero space as its kernel;
this is the reason for the pairing, not a floating-point guarantee.

Recurrence, per active column ``j`` with its own scalars, never pooled:
``r = A z``, ``p = r``, ``rho = <r, r>_w``; then ``v = A p``, ``den = <p, v>_w``,
``alpha = rho / den``, ``a <- a + alpha p`` (the fitted coefficients),
``r <- r - alpha v``, the fresh check of the new candidate ``Y - lift(a)``, the
passing columns retired, ``rho_new = <r, r>_w`` on the remaining columns,
``beta = rho_new / rho``, ``p <- r + beta p``.  ``r``, ``p`` and ``v`` are carried as
block coefficient tuples whose lifts are their row vectors, so every vector stays in
the span of the indicators and the returned residual and the returned coefficients
describe the same point.

Breakdown (status ``breakdown``, never convergence, restart, fallback or a rank
diagnosis): for a column that still fails the fresh check, a ``rho`` or ``den`` that
is not finite and positive, or a ``rho_new`` that is exactly zero.  A nonzero ratio
rounded to zero or an unrepresentable value is a range failure of the checked
arithmetic (status ``range``).  Every unsuccessful outcome keeps the last candidate
whose residual was formed, its fresh check when that check completed, and the work.

A caller's requirement (for a single column: the same-q recovery of
:mod:`powermean.solve.absorbed_direction`, the one caller).  It is consulted only at a
candidate whose fresh check passed.  When it is met the call ends ``converged`` there,
as without one; when it is not, the SAME recurrence continues with that column active,
nothing restarted, no tolerance or budget changed.  A requirement only ever extends the
absorber's own stopping: a unit that does not fit the budget, or a recurrence scalar
that is not positive, at a candidate whose fresh check passed ends the call
``converged`` there (the requirement unmet is the caller's to report); a candidate that
fails its check keeps the rules above.  Without a requirement every path is the one
above.

Work.
A forward traversal of all blocks is one sweep; the symmetric operator costs two
sweeps for ``G > 1`` and visits ``2G - 1`` blocks when it completes (one sweep and one
visit for ``G = 1``, which the orchestrator handles as the direct projection and never
routes here).  The sweeps of an operator are reserved and charged against the budget
before it starts, whether or not it completes; the initial ``A z`` is charged like
every ``A p``; ``operators`` counts the applications attempted.  Beside that reserved
charge the record reports the work attempted: ``visits`` counts the block visits
begun (``2G - 1`` per completed operator, fewer when a visit refuses by range, the
refused one included) and ``diagnostic_passes`` counts the block-mean evaluations
begun by the fresh checks (``G`` per completed check), which are not sweeps.  A budget
of 1 is valid and, for ``G > 1``, yields ``budget`` at the checked initial point unless
that point already passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from powermean.fe.absorb_numeric import (
    Attempted,
    Coefficients,
    FreshCheck,
    PreparedWeights,
    added,
    candidate,
    checked_quotient,
    fresh_check,
    lift,
    replaced_columns,
    scaled,
    select_columns,
    subtracted,
    symmetric_order,
    walk,
    weighted_inner,
)
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

CONVERGED = "converged"
BUDGET = "budget"
RANGE = "range"
BREAKDOWN = "breakdown"
STATUSES = (CONVERGED, BUDGET, RANGE, BREAKDOWN)
REQUIREMENT = (
    "requirement must be None or a callable for a single column, returning a bool"
)

# A caller's requirement of a single-column candidate within the tolerance, given owned
# read-only copies of its block coefficients: ``True`` when the caller needs nothing
# more of it.
Requirement = Callable[[Coefficients], bool]


# ----------------------------------------------------------------- work, points


@dataclass(frozen=True)
class Work:
    """The reserved budget units and the attempted block work of one absorption.

    Attributes
    ----------
    sweeps : int
        Forward-traversal units reserved and charged against the budget: one per
        forward sweep, two per symmetric operator with more than one block, the whole
        unit when the sweep or operator is attempted, whether or not it completes.
    visits : int
        Block visits attempted by the sweeps and operators: ``G`` per completed
        forward sweep, ``2G - 1`` per completed operator, fewer when a visit refuses by
        range (the refused visit counted, the blocks never reached not).
    operators : int
        Symmetric operator applications attempted (the initial one included).
    diagnostic_passes : int
        Block-mean evaluations attempted by the fresh checks: ``G`` per completed
        check, fewer when an evaluation refuses (the refused one counted).
    """

    sweeps: int = 0
    visits: int = 0
    operators: int = 0
    diagnostic_passes: int = 0

    def reserved(self, sweeps: int, operators: int = 0) -> Work:
        """The work after ``sweeps`` units are reserved for a sweep or an operator."""
        return replace(
            self, sweeps=self.sweeps + sweeps, operators=self.operators + operators
        )

    def visited(self, visits: int) -> Work:
        """The work after ``visits`` block visits were attempted."""
        return replace(self, visits=self.visits + visits)

    def checked(self, passes: int) -> Work:
        """The work after a fresh check attempted ``passes`` block evaluations."""
        return replace(self, diagnostic_passes=self.diagnostic_passes + passes)


def operator_cost(n_blocks: int) -> tuple[int, int]:
    """The reserved sweeps and the visits of one completed symmetric operator.

    ``(2, 2G - 1)`` with several blocks, ``(1, 1)`` with one.
    """
    return (2, 2 * n_blocks - 1) if n_blocks > 1 else (1, 1)


@dataclass(frozen=True, eq=False)
class Point:
    """A candidate: coefficients, their residual, and its fresh check if it completed.

    ``failure`` names the range failure of the check when ``check`` is ``None``;
    ``passes`` is the number of block-mean evaluations the check attempted (every
    block when it completed, the refused one included when it did not).
    """

    coefficients: Coefficients
    residual: FloatArray
    check: FreshCheck | None
    failure: str | None
    passes: int


def evaluate_point(
    values: FloatArray, coefficients: Coefficients, prepared: PreparedWeights,
    *, _inputs=None,
) -> Point:
    """The residual ``Y - lift(a)`` and the fresh check of all blocks at it.

    Raises ``FloatingPointError`` when the residual itself is not representable: no
    candidate exists then.  A range failure inside the check leaves a point without
    a check, its reason and the evaluations attempted recorded.
    """
    residual = candidate(values, coefficients, prepared.grouping)
    attempted = Attempted()
    try:
        check = (fresh_check(residual, prepared, attempted, _inputs=_inputs)
                 if _inputs is not None and fresh_check is _INPUT_FRESH
                 else fresh_check(residual, prepared, attempted))
    except FloatingPointError as error:
        return Point(coefficients, residual, None, str(error), attempted.count)
    return Point(coefficients, residual, check, None, attempted.count)


def failing(check: FreshCheck, tolerance: float) -> IndexArray:
    """The indices of the columns whose fresh error exceeds ``tolerance``."""
    found: IndexArray = np.flatnonzero(~check.passing(tolerance))
    return found


@dataclass(frozen=True, eq=False)
class Outcome:
    """What a realization returns: the last point, a status, its reason, the work."""

    point: Point
    status: str
    reason: str
    work: Work


def budget_outcome(point: Point, work: Work, max_sweeps: int, cost: int) -> Outcome:
    """The ``budget`` outcome at a checked point whose next unit does not fit."""
    reason = (
        f"the next unit of {cost} sweep(s) exceeds the budget of {max_sweeps} after "
        f"{work.sweeps} sweep(s)"
    )
    return Outcome(point, BUDGET, reason, work)


def require_requirement(value: object, n_columns: int) -> Requirement | None:
    """``None``, or a callable for a single column; ``ValueError`` otherwise."""
    if value is None:
        return None
    if not callable(value) or n_columns != 1:
        raise ValueError(REQUIREMENT)
    return value


def satisfied(point: Point, requirement: Requirement | None) -> bool:
    """Whether a candidate within the tolerance needs nothing more of the absorber.

    ``True`` without a requirement; otherwise the caller's boolean answer, asked with
    owned read-only copies of the candidate's coefficients.
    """
    if requirement is None:
        return True
    found = requirement(tuple(immutable_array(a) for a in point.coefficients))
    if not isinstance(found, bool | np.bool_):
        raise ValueError(REQUIREMENT)
    return bool(found)


def closing(point: Point, work: Work, tolerance: float, otherwise: Outcome) -> Outcome:
    """``converged`` at ``point`` if its fresh check passed, else ``otherwise``.

    Only a requirement's extension stops at a point whose check passed in every column
    (the budget, or an exhausted recurrence): that check decides the status.
    """
    check = point.check
    if check is not None and check.passing(tolerance).all():
        return Outcome(point, CONVERGED, "", work)
    return otherwise


# ------------------------------------------------------------------ recurrence


@dataclass(frozen=True, eq=False)
class Recurrence:
    """The CG vectors of the active columns: ``r``, ``p`` as coefficients, ``rho``."""

    active: IndexArray
    r: Coefficients
    p: Coefficients
    rho: FloatArray


def _positive_finite(scalars: FloatArray) -> bool:
    """Whether every scalar is finite and strictly positive."""
    return bool(np.isfinite(scalars).all() and (scalars > 0).all())


def _operator(
    rows: FloatArray, prepared: PreparedWeights, attempted: Attempted,
    *, _inputs=None,
) -> Coefficients:
    """``A`` applied to ``rows``, as block coefficients: the literal symmetric walk.

    ``attempted`` counts the visits as they begin, the refused one included.
    """
    order = symmetric_order(prepared.n_blocks)
    _, coefficients, _ = (
        walk(rows, prepared, order, attempted, _inputs=_inputs)
        if _inputs is not None and walk is _INPUT_WALK else
        walk(rows, prepared, order, attempted)
    )
    return coefficients


def _initialize(
    point: Point, active: IndexArray, prepared: PreparedWeights, attempted: Attempted,
    *, _inputs=None,
) -> tuple[Recurrence, str | None]:
    """``r = A z``, ``p = r``, ``rho = <r, r>_w`` on the active columns.

    The second value names the breakdown when ``rho`` is not finite and positive for
    a column that fails the fresh check.
    """
    grouping = prepared.grouping
    r = (_operator(point.residual[:, active], prepared, attempted, _inputs=_inputs)
         if _inputs is not None and _operator is _INPUT_OPERATOR else
         _operator(point.residual[:, active], prepared, attempted))
    rows = lift(r, grouping)
    rho = weighted_inner(prepared, rows, rows)
    reason = None
    if not _positive_finite(rho):
        reason = (
            "zero or nonfinite recurrence residual norm for a column whose weighted "
            "group means exceed the tolerance"
        )
    return Recurrence(active, r, r, rho), reason


def _advance(
    values: FloatArray,
    prepared: PreparedWeights,
    point: Point,
    recurrence: Recurrence,
    attempted: Attempted,
    *, _inputs=None,
) -> tuple[Point, Coefficients, str | None]:
    """One CG update of the coefficients: ``v``, ``den``, ``alpha``, ``a``, ``r``.

    Returns the new point (its check attempted), the updated ``r`` coefficients and
    the breakdown reason when ``den`` is not finite and positive, in which case the
    point and ``r`` are those given.  ``attempted`` counts the operator's visits.
    """
    grouping = prepared.grouping
    p_rows = lift(recurrence.p, grouping)
    v = (_operator(p_rows, prepared, attempted, _inputs=_inputs)
         if _inputs is not None and _operator is _INPUT_OPERATOR else
         _operator(p_rows, prepared, attempted))
    den = weighted_inner(prepared, p_rows, lift(v, grouping))
    if not _positive_finite(den):
        reason = (
            "nonpositive or nonfinite curvature <p, A p>_w for a column whose "
            "weighted group means exceed the tolerance"
        )
        return point, recurrence.r, reason
    alpha = checked_quotient(recurrence.rho, den, "alpha")
    current = select_columns(point.coefficients, recurrence.active)
    coefficients = replaced_columns(
        point.coefficients,
        recurrence.active,
        added(current, scaled(recurrence.p, alpha)),
    )
    r = subtracted(recurrence.r, scaled(v, alpha))
    point = (evaluate_point(values, coefficients, prepared, _inputs=_inputs)
             if _inputs is not None and evaluate_point is _INPUT_EVALUATE else
             evaluate_point(values, coefficients, prepared))
    return point, r, None


def _continue(
    prepared: PreparedWeights,
    recurrence: Recurrence,
    r: Coefficients,
    still: IndexArray,
) -> tuple[Recurrence, str | None]:
    """``rho_new``, ``beta`` and ``p`` on the columns ``still`` active (positions).

    The second value names the breakdown when ``rho_new`` is exactly zero or not
    finite for a column that still fails the fresh check.
    """
    r_kept = select_columns(r, still)
    rows = lift(r_kept, prepared.grouping)
    rho_new = weighted_inner(prepared, rows, rows)
    if not _positive_finite(rho_new):
        reason = (
            "zero or nonfinite recurrence residual norm for a column whose weighted "
            "group means exceed the tolerance"
        )
        return recurrence, reason
    beta = checked_quotient(rho_new, recurrence.rho[still], "beta")
    p = added(r_kept, scaled(select_columns(recurrence.p, still), beta))
    return Recurrence(recurrence.active[still], r_kept, p, rho_new), None


def _still_active(
    recurrence: Recurrence, check: FreshCheck, tolerance: float
) -> IndexArray:
    """The positions, within the active columns, of those still above tolerance."""
    passing = check.passing(tolerance)
    found: IndexArray = np.flatnonzero(~passing[recurrence.active])
    return found


# ------------------------------------------------------------------- the loop


def conjugate_gradient(
    values: FloatArray,
    prepared: PreparedWeights,
    coefficients: Coefficients,
    *,
    tolerance: float,
    max_sweeps: int,
    requirement: Requirement | None = None,
    _inputs=None,
) -> Outcome:
    """Absorb ``values`` from the coefficients by lifted per-column symmetric CG.

    Parameters
    ----------
    values : array of float
        ``Y``, ``n`` rows and ``k`` columns, finite.
    prepared : PreparedWeights
        The current weights and totals; more than one block.
    coefficients : tuple of arrays
        The starting block coefficients (cold zeros or a warm state's).
    tolerance : float
        The requested maximum absolute weighted group mean; equality passes.
    max_sweeps : int
        The budget in sweeps, at least 1.
    requirement : callable, optional
        The caller's requirement of a single-column candidate within the tolerance
        (module docstring); none by default.

    Returns
    -------
    Outcome
        The last point, one of ``converged``, ``budget``, ``range`` or ``breakdown``,
        the reason and the work.  A residual that is not representable at the start
        raises ``FloatingPointError`` before any work.
    """
    required = require_requirement(requirement, values.shape[1])
    if _inputs is not None and (_start is not _INPUT_START or _iterate is not _INPUT_ITERATE):
        _inputs = None
    progress = (
        _start(values, prepared, coefficients, tolerance=tolerance,
               max_sweeps=max_sweeps, requirement=required, _inputs=_inputs)
        if _inputs is not None else
        _start(values, prepared, coefficients, tolerance=tolerance,
               max_sweeps=max_sweeps, requirement=required)
    )
    while not isinstance(progress, Outcome):
        progress = (
            _iterate(values, prepared, progress, tolerance=tolerance,
                     max_sweeps=max_sweeps, requirement=required, _inputs=_inputs)
            if _inputs is not None else
            _iterate(values, prepared, progress, tolerance=tolerance,
                     max_sweeps=max_sweeps, requirement=required)
        )
    return progress


@dataclass(frozen=True, eq=False)
class Progress:
    """The loop between two CG updates: the current point, recurrence and work."""

    point: Point
    recurrence: Recurrence
    work: Work


def _start(
    values: FloatArray,
    prepared: PreparedWeights,
    coefficients: Coefficients,
    *,
    tolerance: float,
    max_sweeps: int,
    requirement: Requirement | None,
    _inputs=None,
) -> Outcome | Progress:
    """The initial check and, if a column fails it and the budget admits, ``A z``.

    A starting candidate within the tolerance whose requirement is unmet starts
    ``A z`` on every column instead.
    """
    sweeps, _ = operator_cost(prepared.n_blocks)
    point = (evaluate_point(values, coefficients, prepared, _inputs=_inputs)
             if _inputs is not None and evaluate_point is _INPUT_EVALUATE else
             evaluate_point(values, coefficients, prepared))
    work = Work().checked(point.passes)
    if point.check is None:
        return Outcome(point, RANGE, point.failure or RANGE, work)
    active = failing(point.check, tolerance)
    if not active.size:
        if satisfied(point, requirement):
            return Outcome(point, CONVERGED, "", work)
        active = np.arange(values.shape[1], dtype=np.intp)
    if work.sweeps + sweeps > max_sweeps:
        exceeded = budget_outcome(point, work, max_sweeps, sweeps)
        return closing(point, work, tolerance, exceeded)
    work = work.reserved(sweeps, 1)
    attempted = Attempted()
    try:
        recurrence, reason = (
            _initialize(point, active, prepared, attempted, _inputs=_inputs)
            if _inputs is not None and _initialize is _INPUT_INITIALIZE else
            _initialize(point, active, prepared, attempted)
        )
    except FloatingPointError as error:
        return Outcome(point, RANGE, str(error), work.visited(attempted.count))
    work = work.visited(attempted.count)
    if reason is not None:
        return closing(point, work, tolerance, Outcome(point, BREAKDOWN, reason, work))
    return Progress(point, recurrence, work)


def _iterate(
    values: FloatArray,
    prepared: PreparedWeights,
    progress: Progress,
    *,
    tolerance: float,
    max_sweeps: int,
    requirement: Requirement | None,
    _inputs=None,
) -> Outcome | Progress:
    """One CG update: the reserved operator, the candidate, its check, ``beta``.

    A candidate within the tolerance whose requirement is unmet keeps every column of
    the update active.
    """
    sweeps, _ = operator_cost(prepared.n_blocks)
    point, recurrence, work = progress.point, progress.recurrence, progress.work
    if work.sweeps + sweeps > max_sweeps:
        exceeded = budget_outcome(point, work, max_sweeps, sweeps)
        return closing(point, work, tolerance, exceeded)
    work = work.reserved(sweeps, 1)
    attempted = Attempted()
    try:
        new_point, r, reason = (
            _advance(values, prepared, point, recurrence, attempted, _inputs=_inputs)
            if _inputs is not None and _advance is _INPUT_ADVANCE else
            _advance(values, prepared, point, recurrence, attempted)
        )
    except FloatingPointError as error:
        return Outcome(point, RANGE, str(error), work.visited(attempted.count))
    work = work.visited(attempted.count)
    if reason is not None:
        return closing(point, work, tolerance, Outcome(point, BREAKDOWN, reason, work))
    point, work = new_point, work.checked(new_point.passes)
    if point.check is None:
        return Outcome(point, RANGE, point.failure or RANGE, work)
    if point.check.passing(tolerance).all():
        if satisfied(point, requirement):
            return Outcome(point, CONVERGED, "", work)
        still = np.arange(recurrence.active.size, dtype=np.intp)
    else:
        still = _still_active(recurrence, point.check, tolerance)
    try:
        recurrence, reason = _continue(prepared, recurrence, r, still)
    except FloatingPointError as error:
        return Outcome(point, RANGE, str(error), work)
    if reason is not None:
        return closing(point, work, tolerance, Outcome(point, BREAKDOWN, reason, work))
    return Progress(point, recurrence, work)


_INPUT_FRESH, _INPUT_WALK = fresh_check, walk
_INPUT_OPERATOR, _INPUT_EVALUATE = _operator, evaluate_point
_INPUT_INITIALIZE, _INPUT_ADVANCE = _initialize, _advance
_INPUT_START, _INPUT_ITERATE = _start, _iterate
