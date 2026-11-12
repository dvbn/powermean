"""Weighted alternating projection of all columns onto the fixed-effect complement.

Internal module: the projection only.  No dense or sparse indicator matrix, no
profiling, no slope solve, no sample, support, collinearity or rank policy, no outer
schedule, no covariance, no certificate and no public path: other modules own those,
and ``powermean`` exports nothing of this module.  The controls are those of
:class:`powermean.solve.absorbed.AbsorbedOptions`, unchanged (``absorb_tol_start``
1e-4, ``absorb_tol_final`` 2e-11, ``absorb_max_sweeps`` 20000, ``absorb_acceleration``
``True``); this module takes them as explicit arguments and chooses no default.

Three realizations, one authoritative state.  The fitted component is carried as
one coefficient array per block (``a_b``, ``L_b``-by-``k``), the candidate is always
``z = Y - lift(a)`` and the returned residual is that candidate, so the residual and
the warm state describe the same point.  With one block, every call uses
the direct projection of that block (one sweep, one visit), followed by the fresh
check and repeated within the budget only if rounding leaves a mean above tolerance.
With several blocks and ``acceleration=False``, complete forward sweeps
``Q_1, ..., Q_G``.  With several blocks and ``acceleration=True``, the lifted
per-column symmetric conjugate gradient of :mod:`powermean.fe.absorb_acceleration`.
Columns that pass are held fixed while the others continue; every returned column is
checked at the returned candidate.

Stopping.  Success requires a completed range-valid evaluation of all
block means at the same candidate with the current weights, and a maximum absolute
weighted group mean over every block, level and column at most the requested
tolerance (equality passes).  No input-scale multiplier, displacement criterion,
recurrence residual, stale block or cached flag.  Finalization is a new call at the
final tolerance from the warm state: :func:`absorb` with ``tolerance`` equal to
``absorb_tol_final`` and ``state`` the previous result's state; its fresh check may
establish that nothing more is needed, a cached success never does.

A caller's requirement (optional, for a single column; the same-q recovery's
reconstruction checks are its one use).  The weighted group means do not bound the
pointwise error of the fitted component when there are several blocks, so a caller that
needs more of a single-column candidate than this predicate may pass a requirement.  It
is consulted only at a candidate whose fresh check passed: met, the call ends
``converged`` there exactly as without one; unmet, the SAME realization continues from
that candidate (the column active, the recurrence kept, nothing restarted), within the
same budget and at the same tolerance.  At a candidate whose fresh check passed, the
budget or an exhausted recurrence ends the call ``converged``, since that check decides
the status; the unmet requirement is the caller's to report.  The predicate, the
controls and the statuses are unchanged, and without a requirement every call is the
one described above.

Work.  A forward traversal is one sweep; a symmetric operator two sweeps;
every unit is reserved and charged against the budget before it starts, whether or not
it completes.  Beside the reserved sweeps the record reports the work attempted: the
block visits begun by the sweeps and operators (``G`` per completed forward sweep,
``2G - 1`` per completed operator, fewer when a visit refuses by range, the refused one
counted) and, apart, the block-mean evaluations begun by the fresh checks (diagnostic
passes, ``G`` per completed check).  ``max_sweeps=1`` is valid: an initially accurate
point passes with no sweep, the one-block and forward paths use their one sweep, and
the accelerated path with several blocks reports ``budget`` at the checked initial
point.

Outcomes.  Malformed values, weights, controls or an unbound state raise built-in
``ValueError``.
A residual that is not representable at the start raises ``FloatingPointError``.
Otherwise the result carries one of four statuses, ``converged``, ``budget``,
``range`` (a refused operation of the checked arithmetic) or ``breakdown`` (a
recurrence scalar that is not finite and positive, or an exactly zero recurrence
residual with a column still above tolerance); the three unsuccessful ones keep the
last candidate whose residual was formed, its fresh check when that check completed,
and the actual work.  No projection failure is reclassified as a statistical rank or
sample failure, and nothing is retried.

Warm state.  :class:`AbsorptionState` holds owned read-only block
coefficients and the exact binding: the grouping (positions, block names,
codes, level counts) and the caller's column identity in order.  A call with another
column tuple or another grouping refuses; no permutation is guessed.  The state holds
no recurrence vector, tolerance, success flag or weight: changed weights are a new
:class:`~powermean.fe.absorb_numeric.PreparedWeights`, changed values a new call, and
only the coefficients seed a solve.  With zero weights the fitted values on
zero-weight rows need not be unique; the rows are kept and nothing promises that a
warm and a cold start agree there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb_acceleration import (
    CONVERGED,
    RANGE,
    STATUSES,
    Outcome,
    Point,
    Requirement,
    Work,
    budget_outcome,
    closing,
    conjugate_gradient,
    evaluate_point,
    failing,
    require_requirement,
    satisfied,
)
from powermean.fe.absorb_numeric import (
    Attempted,
    Coefficients,
    PreparedWeights,
    added,
    forward_order,
    replaced_columns,
    select_columns,
    walk,
    zero_coefficients,
    _input_reductions,
)
from powermean.fe.groups import Block, Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray

VALUES = (
    "values must be a real matrix with one finite row per observation and one "
    "column per name"
)
COLUMNS = "columns must be a nonempty tuple of distinct names"
TOLERANCE = "tolerance must be a positive finite real number"
BUDGET_MESSAGE = "max_sweeps must be a positive integer"
ACCELERATION = "acceleration must be True or False"
BOUND = "the state is bound to other columns or another grouping"


# ---------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class AbsorptionState:
    """The fitted block coefficients and their exact binding: a warm start.

    Attributes
    ----------
    grouping : Grouping
        The accepted grouping the coefficients belong to: its positions, block
        names, codes and level counts are the row and level identity.
    columns : tuple of str
        The caller's identity of the columns, in order; never user row labels.
    coefficients : tuple of arrays of float
        One ``L_b``-by-``k`` array per block, finite, read-only and owned.

    Notes
    -----
    The lift of the coefficients is the fitted component in the span of the
    indicators; the state admits nothing else (no residual vector, no recurrence
    vector, no flag).  A direct construction is checked and owns its arrays.
    """

    grouping: Grouping
    columns: tuple[str, ...]
    coefficients: Coefficients

    def __post_init__(self) -> None:
        """Check the binding and the coefficients; store owned read-only arrays."""
        if not isinstance(self.grouping, Grouping):
            raise ValueError("grouping must be a Grouping record")
        columns = _columns(self.columns)
        given = tuple(self.coefficients)
        if len(given) != len(self.grouping.blocks):
            raise ValueError("provide one coefficient array per block")
        owned: list[FloatArray] = []
        for block, a in zip(self.grouping.blocks, given, strict=True):
            if np.iscomplexobj(a):
                raise ValueError("coefficients must be real")
            array: FloatArray = np.array(a, dtype=np.float64, copy=True)
            if array.shape != (block.n_levels, len(columns)):
                raise ValueError(
                    "coefficients must be one row per level, one column per name"
                )
            if not np.isfinite(array).all():
                raise ValueError("coefficients must be finite")
            owned.append(immutable_array(array))
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "coefficients", tuple(owned))

    @property
    def n_columns(self) -> int:
        """The number of columns."""
        return len(self.columns)


def cold_state(grouping: Grouping, columns: tuple[str, ...]) -> AbsorptionState:
    """The cold start: every coefficient zero, bound to ``grouping`` and ``columns``."""
    names = _columns(columns)
    return AbsorptionState(grouping, names, zero_coefficients(grouping, len(names)))


@dataclass(frozen=True, eq=False)
class AbsorptionResult:
    """The outcome of one absorption: residual, state, status, accuracy and work.

    Attributes
    ----------
    residual : array of float
        ``Y - lift(a)`` at the returned candidate, read-only and owned.
    state : AbsorptionState
        The coefficients of that candidate, with their binding.
    status : str
        ``converged``, ``budget``, ``range`` or ``breakdown``.
    reason : str
        Empty on convergence; otherwise what stopped the absorption.
    tolerance : float
        The requested maximum absolute weighted group mean.
    column_errors : array of float or None
        Per column, the fresh maximum absolute weighted group mean at the returned
        candidate; ``None`` when that evaluation itself failed by range.
    maximum : float or None
        The maximum of ``column_errors``.
    sweeps, visits, operators, diagnostic_passes : int
        The work, as :class:`powermean.fe.absorb_acceleration.Work` counts it: the
        sweeps reserved against the budget, the block visits attempted, the operator
        applications attempted and the block-mean evaluations attempted by the
        fresh checks.
    acceleration : bool
        The realization requested.
    """

    residual: FloatArray
    state: AbsorptionState
    status: str
    reason: str
    tolerance: float
    column_errors: FloatArray | None
    maximum: float | None
    sweeps: int
    visits: int
    operators: int
    diagnostic_passes: int
    acceleration: bool

    def __post_init__(self) -> None:
        """Check the status and own the arrays."""
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        object.__setattr__(self, "residual", immutable_array(self.residual))
        if self.column_errors is not None:
            object.__setattr__(
                self, "column_errors", immutable_array(self.column_errors)
            )

    @property
    def converged(self) -> bool:
        """Whether the fresh check passed at the returned candidate."""
        return self.status == CONVERGED


# ------------------------------------------------------------------- validation


def _columns(columns: object) -> tuple[str, ...]:
    """A nonempty tuple of distinct strings, or ``ValueError``."""
    if isinstance(columns, str) or not isinstance(columns, tuple) or not columns:
        raise ValueError(COLUMNS)
    if any(not isinstance(name, str) for name in columns) or len(set(columns)) != len(
        columns
    ):
        raise ValueError(COLUMNS)
    return columns


def _tolerance(value: object) -> float:
    """A positive finite real number as a float, or ``ValueError``."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.number
    ):
        raise ValueError(TOLERANCE)
    if np.iscomplexobj(value):
        raise ValueError(TOLERANCE)
    found = float(value)
    if not np.isfinite(found) or found <= 0:
        raise ValueError(TOLERANCE)
    return found


def _budget(value: object) -> int:
    """A positive integer (1 included), or ``ValueError``."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(BUDGET_MESSAGE)
    if value < 1:
        raise ValueError(BUDGET_MESSAGE)
    return int(value)


def _flag(value: object) -> bool:
    """``True`` or ``False`` by type, or ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(ACCELERATION)
    return value


def _values(values: ArrayLike, n_rows: int, n_columns: int) -> FloatArray:
    """An owned finite ``n_rows``-by-``n_columns`` matrix, or ``ValueError``."""
    if np.iscomplexobj(values):
        raise ValueError(VALUES)
    found: FloatArray = np.array(values, dtype=np.float64, copy=True)
    if found.shape != (n_rows, n_columns) or not np.isfinite(found).all():
        raise ValueError(VALUES)
    return found


def _bound(
    state: AbsorptionState, prepared: PreparedWeights, columns: tuple[str, ...]
) -> Coefficients:
    """The coefficients of ``state`` if it is bound exactly to the call's identity."""
    mine, theirs = state.grouping, prepared.grouping
    if (
        state.columns == columns
        and mine is theirs
        and type(mine) is Grouping
        and not mine.positions.flags.writeable
        and all(
            type(b) is Block and not b.codes.flags.writeable for b in mine.blocks
        )
    ):
        return state.coefficients
    same = (
        state.columns == columns
        and mine.names == theirs.names
        and np.array_equal(mine.positions, theirs.positions)
        and all(
            a.n_levels == b.n_levels and np.array_equal(a.codes, b.codes)
            for a, b in zip(mine.blocks, theirs.blocks, strict=True)
        )
    )
    if not same:
        raise ValueError(BOUND)
    return state.coefficients


# ------------------------------------------------------- forward and one block


def projection_sweeps(
    values: FloatArray,
    prepared: PreparedWeights,
    coefficients: Coefficients,
    *,
    tolerance: float,
    max_sweeps: int,
    requirement: Requirement | None = None,
    _inputs=None,
) -> Outcome:
    """Forward sweeps ``Q_1, ..., Q_G`` from the coefficients, checked after each.

    With one block this is the direct projection of that block: the exact one-block
    specialization, one sweep and one visit, still followed by the fresh check.
    Columns that pass are held fixed.  The returned outcome keeps the last point
    whose residual was formed.  Each sweep is reserved before it starts; its visits
    and the passes of the following check are recorded as attempted.  A single-column
    candidate within the tolerance whose requirement is unmet sweeps again until met,
    the budget ends, or a sweep makes no change.
    """
    required = require_requirement(requirement, values.shape[1])
    order = forward_order(prepared.n_blocks)
    point = (evaluate_point(values, coefficients, prepared, _inputs=_inputs)
             if _inputs is not None and evaluate_point is _INPUT_EVALUATE else
             evaluate_point(values, coefficients, prepared))
    work = Work().checked(point.passes)
    while True:
        if point.check is None:
            return Outcome(point, RANGE, point.failure or RANGE, work)
        active = failing(point.check, tolerance)
        if not active.size:
            if satisfied(point, required):
                return Outcome(point, CONVERGED, "", work)
            active = np.arange(values.shape[1], dtype=np.intp)
        if work.sweeps + 1 > max_sweeps:
            exceeded = budget_outcome(point, work, max_sweeps, 1)
            return closing(point, work, tolerance, exceeded)
        work = work.reserved(1)
        attempted = Attempted()
        try:
            _, delta, _ = (
                walk(point.residual[:, active], prepared, order, attempted, _inputs=_inputs)
                if _inputs is not None and walk is _INPUT_WALK else
                walk(point.residual[:, active], prepared, order, attempted)
            )
            if not any(np.any(d) for d in delta):
                return Outcome(point, CONVERGED, "", work.visited(attempted.count))
            current = select_columns(point.coefficients, active)
            updated = replaced_columns(
                point.coefficients, active, added(current, delta)
            )
            point = (evaluate_point(values, updated, prepared, _inputs=_inputs)
                     if _inputs is not None and evaluate_point is _INPUT_EVALUATE else
                     evaluate_point(values, updated, prepared))
        except FloatingPointError as error:
            return Outcome(point, RANGE, str(error), work.visited(attempted.count))
        work = work.visited(attempted.count).checked(point.passes)


# ------------------------------------------------------------------- the entry


def _result(
    outcome: Outcome,
    prepared: PreparedWeights,
    columns: tuple[str, ...],
    tolerance: float,
    acceleration: bool,
) -> AbsorptionResult:
    """The result record of an outcome."""
    point: Point = outcome.point
    check = point.check
    return AbsorptionResult(
        residual=point.residual,
        state=AbsorptionState(prepared.grouping, columns, point.coefficients),
        status=outcome.status,
        reason=outcome.reason,
        tolerance=tolerance,
        column_errors=None if check is None else check.column_errors,
        maximum=None if check is None else check.maximum,
        sweeps=outcome.work.sweeps,
        visits=outcome.work.visits,
        operators=outcome.work.operators,
        diagnostic_passes=outcome.work.diagnostic_passes,
        acceleration=acceleration,
    )


def absorb(
    values: ArrayLike,
    prepared: PreparedWeights,
    columns: tuple[str, ...],
    *,
    tolerance: float,
    max_sweeps: int,
    acceleration: bool,
    state: AbsorptionState | None = None,
    requirement: Requirement | None = None,
) -> AbsorptionResult:
    """Absorb the blocks of ``prepared`` from every column of ``values``.

    Parameters
    ----------
    values : array_like
        ``Y``: one finite row per observation of the grouping, one column per name
        of ``columns``.
    prepared : PreparedWeights
        The current weights and level totals (:func:`~powermean.fe.absorb_numeric
        .prepare_weights`); changed weights are a new record.
    columns : tuple of str
        The caller's identity of the columns, distinct, in order.
    tolerance : float
        The requested maximum absolute weighted group mean over every block, level
        and column; positive and finite; equality passes.
    max_sweeps : int
        The budget of sweeps, a positive integer, 1 included.
    acceleration : bool
        Lifted per-column symmetric CG when ``True`` and there are several blocks;
        forward sweeps when ``False``; the direct projection with one block either way.
    state : AbsorptionState, optional
        A warm start bound to the same grouping and columns; the cold start if omitted.
    requirement : callable, optional
        The caller's requirement of a single-column candidate within the tolerance,
        given owned read-only copies of its block coefficients (module docstring); none
        by default.

    Returns
    -------
    AbsorptionResult
        The residual and state of the returned candidate, its status, reason, fresh
        accuracy and work; unsuccessful statuses keep the last finite candidate.

    Raises
    ------
    ValueError
        Malformed values or controls, or a state bound to other columns or another
        grouping.
    FloatingPointError
        A starting residual that is not representable: no candidate exists.
    """
    names = _columns(columns)
    tol = _tolerance(tolerance)
    budget = _budget(max_sweeps)
    accelerated = _flag(acceleration)
    required = require_requirement(requirement, len(names))
    y = _values(values, prepared.n_rows, len(names))
    coefficients = (
        zero_coefficients(prepared.grouping, len(names))
        if state is None
        else _bound(state, prepared, names)
    )
    inputs = (_input_reductions(prepared)
              if required is None and _input_reductions is _INPUT_REDUCTIONS else None)
    # Only original loops receive the private local input lifetime. Changed hooks
    # keep their original argument lists and all numerical/requirement behavior.
    if accelerated and prepared.n_blocks > 1:
        outcome = (
            conjugate_gradient(y, prepared, coefficients, tolerance=tol,
                               max_sweeps=budget, requirement=required, _inputs=inputs)
            if inputs is not None and conjugate_gradient is _INPUT_CG else
            conjugate_gradient(y, prepared, coefficients, tolerance=tol,
                               max_sweeps=budget, requirement=required)
        )
    else:
        outcome = (
            projection_sweeps(y, prepared, coefficients, tolerance=tol,
                              max_sweeps=budget, requirement=required, _inputs=inputs)
            if inputs is not None and projection_sweeps is _INPUT_SWEEPS else
            projection_sweeps(y, prepared, coefficients, tolerance=tol,
                              max_sweeps=budget, requirement=required)
        )
    return _result(outcome, prepared, names, tol, accelerated)


_INPUT_REDUCTIONS = _input_reductions
_INPUT_EVALUATE, _INPUT_WALK = evaluate_point, walk
_INPUT_CG, _INPUT_SWEEPS = conjugate_gradient, projection_sweeps
