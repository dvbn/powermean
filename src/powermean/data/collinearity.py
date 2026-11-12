"""The within-design collinearity analysis: absorbed and collinear columns, by reason.

Internal module, with no public name, option, default, message or route.  The sample,
its fixed point and the repeated calls after exclusions are not decided here, and
neither are instruments.  This module orchestrates; the records and the admission of the
absorber's state are :mod:`powermean.data.collinearity_records` (whose names this module
still provides), the checked arithmetic and LAPACK calls
:mod:`powermean.data.collinearity_numeric`, the within rank, spans and relations
:mod:`powermean.data.collinearity_geometry`.

Input (:class:`~powermean.data.collinearity_records.CollinearityProblem`).  Finite
original columns on the retained rows, in formula order with their names; the exact
original row positions (duplicate labels stay distinct rows); the grouping of
every applicable estimated or cancelled block, or none.  Statistical weights, the
outcome, ``omega``, ``tau`` and the curvature are not inputs: unit weights are used, and
the sample has already excluded zero weights.

Procedure.  Exact zero and constant columns are decided by their entries: with blocks
both are "absorbed by fixed effects"; without, a zero column is "omitted (collinear)"
and nonzero columns, constants included, enter the rank analysis, so a lone constant
stays and repeated constants lose in reverse formula order.  Nothing is centered and no
variance is formed.  With blocks the other columns ``U = X / d`` (power-of-two scales,
exactly reversible) are absorbed once with unit weights by
:func:`powermean.fe.absorb.absorb`, at ``t_diag = min(t_requested, h)``, ``h = eps
max(n, p)``: a practical diagnostic resolution chosen once and recorded beside the
request, not the estimation default or an accuracy bound; the caller's budget and
acceleration are unchanged and nothing is retried.  The result must have converged, its
residual must be the values minus the ordered lift of its own coefficients bit for bit,
and a fresh unit-weight check of every block must find no mean above ``t_diag``; a
completed check is kept as evidence before that predicate decides, so that a ``means``
refusal shows the means it refused.  Each such column has its uncentered norm screen
``||R_j|| <= sqrt(s) ||U_j||``, ``s = min(1e-6, t_diag / 10)`` (``reghdfe.ado`` line 492
and ``reghdfe.mata`` lines 2212 to 2269 for its provenance; a candidate, never an
omission rule) and the span test of ``U_j`` against its separately gathered block
components: screen and span give "absorbed by fixed effects"; a span without the screen
is a contradiction; no span with a nonzero residual makes the column active whatever the
screen; a zero residual without the span is unresolved.  The active residual columns go
to the rank analysis; every omission it proposes must pass the original-column span test
against the FINAL kept columns and every block, else nothing is concluded.

What the outcome means.  A kept column has full numerical rank of the ACTUAL freshly
checked within matrix under this completed procedure: an operational floating-rank
convention, not an exact certificate of independence from the full fixed-effect space,
and not the negation of the compressed span test.  Small group means do not bound the
projection error, and nothing is promised on arbitrarily weak fixed-effect geometry.
An omission is positive numerical evidence, not exact dependence or a proof of the
projector's accuracy.  Near-collinearity is not collinearity: ``rcond_min``, pivots
and the screen omit nothing.

Outcomes (:class:`~powermean.data.collinearity_records.CollinearityAnalysis`).
``complete``: the reason of every column and, in ``"drop"`` mode, the kept and omitted
columns (none kept is valid).  ``refused``: ``"error"`` mode with at least one proposed
omission; the same reasons, nothing applied, an internal policy refusal and not the
``identification`` error, whose text says that dropping columns cannot resolve the
deficiency.  ``inconclusive``: a specified failure, with its cause and the evidence
gathered; never a rank or an absence of dependence.  Inputs are not modified and no
output shares their memory, except that every outcome keeps the owned problem record it
analyzed and the controls it used (the binding of
:mod:`powermean.data.collinearity_records`), so that a caller can compare them with its
own inputs without running the analysis again.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from powermean.data.collinearity_geometry import (
    Components,
    InconclusiveError,
    SpanTest,
    WithinRank,
    relation,
    span_test,
    within_rank,
)
from powermean.data.collinearity_numeric import Ledger, norm, reversible, screened
from powermean.data.collinearity_records import (
    ABSORBED,
    ACTIVE,
    CAUSES,
    COLLINEAR,
    COMPLETE,
    CONSTANT,
    GENERAL,
    INCONCLUSIVE,
    MODES,
    REFUSED,
    SCREEN_CAP,
    ZERO,
    CollinearityAnalysis,
    CollinearityProblem,
    ColumnDiagnostics,
    DiagnosticControls,
    FailedMeansError,
    Targets,
    _accepted,
    column_kinds,
    diagnostic_targets,
)
from powermean.fe.absorb import AbsorptionResult, absorb
from powermean.fe.absorb_numeric import FreshCheck, prepare_weights
from powermean.model.variance import FloatArray

# The names this internal module provides, the records defined in
# data/collinearity_records.py included: their import routes from here are kept.
__all__ = [
    "ABSORBED",
    "ACTIVE",
    "CAUSES",
    "COLLINEAR",
    "COMPLETE",
    "CONSTANT",
    "GENERAL",
    "INCONCLUSIVE",
    "MODES",
    "REFUSED",
    "SCREEN_CAP",
    "ZERO",
    "CollinearityAnalysis",
    "CollinearityProblem",
    "ColumnDiagnostics",
    "DiagnosticControls",
    "Targets",
    "analyze_collinearity",
    "column_kinds",
    "decision",
    "diagnostic_targets",
]


def decision(screen: bool, test: SpanTest, *, zero: bool) -> str:
    """The decision table: ``ABSORBED``, ``active`` or the cause of a failure."""
    if not test.nested:
        return "nonnested"
    if test.spanned:
        return ABSORBED if screen else "contradiction"
    return "zero_within" if zero else ACTIVE


class _Work:
    """The evidence of one analysis as it is gathered; local, never stored."""

    __slots__ = ("absorption", "columns", "fresh", "ledger", "targets", "within")

    def __init__(self, columns: list[ColumnDiagnostics]) -> None:
        """Start from the column records."""
        self.columns = columns
        self.ledger: Ledger = []
        self.targets: Targets | None = None
        self.absorption: AbsorptionResult | None = None
        self.fresh: FreshCheck | None = None
        self.within: WithinRank | None = None


def _fixed_effects(
    problem: CollinearityProblem,
    controls: DiagnosticControls,
    u: FloatArray,
    work: _Work,
    reasons: list[str | None],
) -> tuple[tuple[int, ...], FloatArray, Components | None]:
    """Absorb the general columns, screen and test each; the active ones."""
    grouping = problem.grouping
    general = tuple(c.index for c in work.columns if c.kind == GENERAL)
    if grouping is None or not general:
        return (), np.zeros((problem.n_rows, 0)), None
    work.targets = targets = diagnostic_targets(
        problem.n_rows, problem.n_columns, controls.tolerance
    )
    names, values = tuple(problem.names[j] for j in general), u[:, list(general)]
    prepared = prepare_weights(grouping, np.ones(problem.n_rows))
    work.absorption = result = absorb(
        values,
        prepared,
        names,
        tolerance=targets.effective,
        max_sweeps=controls.max_sweeps,
        acceleration=controls.acceleration,
    )
    try:
        work.fresh = _accepted(result, values, prepared, names, targets.effective)
    except FailedMeansError as refusal:
        work.fresh = refusal.check
        raise
    codes = tuple(block.codes for block in grouping.blocks)
    blocks = Components(result.state.coefficients, general, codes)
    found = []
    for i, j in enumerate(general):
        r = result.residual[:, i]
        original, within = norm(u[:, j], f"column {j}"), norm(r, f"within {j}")
        screen = screened(within, original, targets.screen)
        test = span_test(
            np.zeros((problem.n_rows, 0)),
            blocks.gathered(j),
            u[:, j],
            work.ledger,
            purpose=f"fixed-effect span of column {j}",
        )
        work.columns[j] = replace(
            work.columns[j],
            original_norm=original,
            within_norm=within,
            screen=screen,
            span=test,
        )
        found.append(decision(screen, test, zero=not bool(np.any(r != 0))))
    for j, outcome in zip(general, found, strict=True):
        if outcome not in (ABSORBED, ACTIVE):
            raise InconclusiveError(
                outcome, f"column {j} ({problem.names[j]}): {outcome}"
            )
        reasons[j] = ABSORBED if outcome == ABSORBED else None
    active = tuple(
        j for j, outcome in zip(general, found, strict=True) if outcome == ACTIVE
    )
    return active, result.residual[:, [general.index(j) for j in active]], blocks


def _analysis(
    problem: CollinearityProblem, controls: DiagnosticControls | None, work: _Work
) -> tuple[str | None, ...]:
    """The reasons of every column, gathering the evidence in ``work``."""
    u, scales = reversible(problem.design, "original columns")
    for j in range(problem.n_columns):
        work.columns[j] = replace(work.columns[j], scale=float(scales[j]))
    reasons: list[str | None] = [None] * problem.n_columns
    trivial = ABSORBED if problem.grouping is not None else COLLINEAR
    for c in work.columns:
        if c.kind == ZERO or (c.kind == CONSTANT and problem.grouping is not None):
            reasons[c.index] = trivial
    if problem.grouping is None or controls is None:
        active = tuple(c.index for c in work.columns if c.kind != ZERO)
        within, blocks = u[:, list(active)], None
    else:
        active, within, blocks = _fixed_effects(problem, controls, u, work, reasons)
    work.within, z = within_rank(within, active, work.ledger)
    for j in work.within.omitted:
        found = relation(
            j, work.within, z, original=u, blocks=blocks, ledger=work.ledger
        )
        work.columns[j] = replace(work.columns[j], relation=found)
        reasons[j] = COLLINEAR
    return tuple(reasons)


def analyze_collinearity(
    problem: CollinearityProblem,
    controls: DiagnosticControls | None,
    *,
    mode: str,
) -> CollinearityAnalysis:
    """The collinearity analysis of ``problem`` in ``mode``: one fresh invocation.

    Parameters
    ----------
    problem : CollinearityProblem
        The columns, names, rows and blocks.
    controls : DiagnosticControls or None
        The absorber controls; given exactly when the problem has blocks.
    mode : {"drop", "error"}
        ``"drop"`` applies the proposed omissions; ``"error"`` refuses them.

    Returns
    -------
    CollinearityAnalysis
        ``complete``, ``refused`` or ``inconclusive``, with the evidence and work.

    Raises
    ------
    ValueError
        A malformed problem, controls or mode, before any analysis.
    """
    if mode not in MODES or not isinstance(problem, CollinearityProblem):
        raise ValueError("mode must be 'drop' or 'error', with a CollinearityProblem")
    if (problem.grouping is None) != (controls is None) or not (
        controls is None or isinstance(controls, DiagnosticControls)
    ):
        raise ValueError("DiagnosticControls are given exactly when there are blocks")
    kinds = column_kinds(problem.design)
    named = zip(problem.names, kinds, strict=True)
    work = _Work([ColumnDiagnostics(j, n, k) for j, (n, k) in enumerate(named)])
    try:
        reasons = _analysis(problem, controls, work)
    except InconclusiveError as stop:
        failure = (INCONCLUSIVE, stop.cause, str(stop))
        return _result(problem, work, mode, failure, None, controls=controls)
    except FloatingPointError as error:
        failure = (INCONCLUSIVE, "range", str(error))
        return _result(problem, work, mode, failure, None, controls=controls)
    except np.linalg.LinAlgError as error:
        failure = (INCONCLUSIVE, "factorization", str(error))
        return _result(problem, work, mode, failure, None, controls=controls)
    proposed = sum(reason is not None for reason in reasons)
    if mode == "error" and proposed:
        refusal = (
            REFUSED,
            "error_mode",
            f"{proposed} proposed omission(s), none applied",
        )
        return _result(problem, work, mode, refusal, reasons, controls=controls)
    return _result(problem, work, mode, (COMPLETE, "", ""), reasons, controls=controls)


def _result(
    problem: CollinearityProblem,
    work: _Work,
    mode: str,
    outcome: tuple[str, str, str],
    reasons: tuple[str | None, ...] | None,
    *,
    controls: DiagnosticControls | None,
) -> CollinearityAnalysis:
    """The record of an outcome, with every piece of evidence gathered.

    The record keeps ``problem`` and ``controls``, the inputs of this invocation.
    """
    status = outcome[0]
    kept = omitted = None
    if status == COMPLETE and reasons is not None:
        kept = np.asarray(
            [j for j, reason in enumerate(reasons) if reason is None], dtype=np.intp
        )
        omitted = np.asarray(
            [j for j, reason in enumerate(reasons) if reason is not None], dtype=np.intp
        )
    return CollinearityAnalysis(
        mode,
        *outcome,
        problem.names,
        problem.positions,
        None if status == INCONCLUSIVE else reasons,
        kept,
        omitted,
        tuple(work.columns),
        work.targets,
        work.absorption,
        work.fresh,
        work.within,
        tuple(work.ledger),
        problem=problem,
        controls=controls,
    )
