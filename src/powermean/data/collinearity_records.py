"""The records of the collinearity analysis and the admission of the absorber's state.

Internal module: what :func:`powermean.data.collinearity.analyze_collinearity` takes and
returns, and the check of the state that the absorber returns to it.

Records: the problem, the absorber controls in the absorber's own domains, the
targets ``t_requested``, ``h = eps max(n, p)``, ``t_diag = min(t_requested, h)`` and
``s = min(1e-6, t_diag / 10)``, the evidence of each column and the outcome.  Each
checks its fields at construction, direct or through ``dataclasses.replace``, and
owns what it keeps: indices and ranks are exact integers checked before any
conversion (a fractional index is refused, not narrowed; a matrix of indices is
refused, not flattened), tuples and arrays are owned copies and the targets are
finite.  A malformed record raises ``ValueError``.

Binding.  An analysis keeps the problem record it ran on and its actual controls, which
a caller compares with its own inputs.  A bound record is checked to be structurally
theirs, without a factorization, solve or absorption: names, positions, controls exactly
with blocks, the requested targets, the absorber's realization and budget (a ``budget``
stop leaves less than one unit of one or two sweeps) and each column's kind
(:func:`column_kinds`).  An unbound record stays inspectable and authorizes nothing; no
record proves how it was computed.

Admission of the absorber's state (:func:`_accepted`): the absorption converged; its
state is bound to the same positions, block names, codes, level counts and column names;
its residual is the values minus the ordered lift of its own coefficients bit for bit;
and a fresh unit-weight check of every block finds no mean above ``t_diag``, equality
passing.  A completed check that fails this last predicate travels with its refusal
(:class:`FailedMeansError`), so that the analysis keeps the observed means beside the
``means`` cause; a check that never completed leaves no evidence.  This establishes the
returned numerical state, not its distance from the exact projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.collinearity_geometry import (
    InconclusiveError,
    Relation,
    SpanTest,
    WithinRank,
)
from powermean.data.collinearity_numeric import (
    EPSILON,
    Factorization,
    exact_integer,
    integers,
    real,
)
from powermean.data.inputs import _column_names
from powermean.fe.absorb import AbsorptionResult, _budget, _flag, _tolerance
from powermean.fe.absorb_acceleration import BUDGET
from powermean.fe.absorb_numeric import (
    FreshCheck,
    PreparedWeights,
    candidate,
    checked_quotient,
    fresh_check,
)
from powermean.fe.groups import Grouping, _positions
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array

ABSORBED = "absorbed by fixed effects"
COLLINEAR = "omitted (collinear)"
MODES = ("drop", "error")
COMPLETE, REFUSED, INCONCLUSIVE = "complete", "refused", "inconclusive"
CAUSES = (
    *("range", "absorption", "state", "means", "contradiction", "zero_within"),
    *("nonnested", "disagreement", "final_rank", "solve", "relation", "factorization"),
)
ZERO, CONSTANT, GENERAL, ACTIVE = "zero", "constant", "general", "active"
# The largest screen of the reghdfe convention, min(1e-6, t_diag / 10).
SCREEN_CAP = 1e-6
BOUND = "the analysis is not bound to its problem and controls"


def column_kinds(design: FloatArray) -> tuple[str, ...]:
    """Per column ``zero`` (every entry zero), ``constant`` (equal entries) or not."""
    zero = np.all(design == 0, axis=0)
    constant = np.all(design == design[0], axis=0)
    return tuple(
        ZERO if z else CONSTANT if c else GENERAL
        for z, c in zip(zero, constant, strict=True)
    )


def _names(value: object, count: int | None = None) -> tuple[str, ...]:
    """Distinct nonempty names of a tuple or list, ``count`` of them if given."""
    if not isinstance(value, tuple | list):
        raise ValueError("names must be a tuple or list of column names")
    return _column_names(value, len(value) if count is None else count)


@dataclass(frozen=True, eq=False)
class CollinearityProblem:
    """The columns, names and rows of one analysis, owned and checked.

    ``design``: ``n``-by-``p`` finite original columns of the retained rows, formula
    order; ``names``: distinct nonempty column names; ``positions``: the exact distinct
    nonnegative original row positions; ``grouping``: every applicable block, whose
    positions must be ``positions`` exactly, or ``None`` without blocks.
    """

    design: FloatArray
    names: tuple[str, ...]
    positions: IndexArray
    grouping: Grouping | None = None

    def __post_init__(self) -> None:
        """``ValueError`` for a malformed or misaligned input; arrays are owned."""
        design = _finite_array(self.design, "design")
        if design.ndim != 2 or not design.shape[0] or not design.shape[1]:
            raise ValueError("design must be a nonempty matrix")
        names = _names(self.names, int(design.shape[1]))
        positions = _positions(self.positions, int(design.shape[0]))
        if self.grouping is not None and not (
            isinstance(self.grouping, Grouping)
            and np.array_equal(self.grouping.positions, positions)
        ):
            raise ValueError("grouping must be a Grouping on exactly these positions")
        object.__setattr__(self, "design", _immutable(design))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "positions", positions)

    @property
    def n_rows(self) -> int:
        """The number of retained rows."""
        return int(self.design.shape[0])

    @property
    def n_columns(self) -> int:
        """The number of original columns."""
        return int(self.design.shape[1])


@dataclass(frozen=True)
class DiagnosticControls:
    """The caller's explicit absorber controls, in the absorber's domains.

    ``tolerance`` is the requested fixed-effect tolerance ``t_requested``;
    ``max_sweeps`` (1 included) and ``acceleration`` are used unchanged.
    """

    tolerance: float
    max_sweeps: int
    acceleration: bool

    def __post_init__(self) -> None:
        """``ValueError`` outside the absorber's domains."""
        object.__setattr__(self, "tolerance", _tolerance(self.tolerance))
        object.__setattr__(self, "max_sweeps", _budget(self.max_sweeps))
        _flag(self.acceleration)


def _screen_of(effective: float) -> float:
    """``s = min(1e-6, t_diag / 10)``, the quotient range-checked."""
    tenth = float(checked_quotient(np.asarray(effective), np.asarray(10.0), "screen"))
    return min(SCREEN_CAP, tenth)


@dataclass(frozen=True)
class Targets:
    """``t_requested``, ``h = eps max(n, p)``, ``t_diag = min(t_requested, h)``, s.

    The request is a finite positive real, the absorber's tolerance domain.
    """

    requested: float
    resolution: float
    effective: float
    screen: float

    def __post_init__(self) -> None:
        """``ValueError`` unless the four values follow from the first two."""
        requested = _tolerance(self.requested)
        resolution = real(self.resolution, "h")
        effective = min(requested, resolution)
        if not 0 < resolution < 1 or self.effective != effective:
            raise ValueError("targets must follow t_diag = min(t_requested, h)")
        if self.screen != _screen_of(effective):
            raise ValueError("the screen must be min(1e-6, t_diag / 10)")
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "effective", effective)
        object.__setattr__(self, "screen", _screen_of(effective))


def diagnostic_targets(n_rows: int, n_columns: int, requested: float) -> Targets:
    """The targets of an analysis; an ``h`` that is not below one is ``range``."""
    resolution = EPSILON * max(n_rows, n_columns)
    if not 0 < resolution < 1:
        raise InconclusiveError("range", "h = eps max(n, p) is not below one")
    effective = min(requested, resolution)
    return Targets(requested, resolution, effective, _screen_of(effective))


@dataclass(frozen=True, eq=False)
class ColumnDiagnostics:
    """What the analysis established for one original column, as far as it went.

    ``kind``: ``zero``, ``constant`` or ``general``; ``scale``: ``d_j``; with blocks,
    a general column has its ``original_norm`` ``||U_j||``, ``within_norm``
    ``||R_j||``, ``screen`` and fixed-effect ``span`` test; an omitted collinear
    column has its ``relation``.  Missing evidence is ``None``.
    """

    index: int
    name: str
    kind: str
    scale: float | None = None
    original_norm: float | None = None
    within_norm: float | None = None
    screen: bool | None = None
    span: SpanTest | None = None
    relation: Relation | None = None

    def __post_init__(self) -> None:
        """``ValueError`` for an unknown kind, a malformed index, value or record."""
        if self.kind not in (ZERO, CONSTANT, GENERAL):
            raise ValueError("unknown column kind")
        index = exact_integer(self.index, "a column index")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a column name is a nonempty string")
        if self.screen is not None and not isinstance(self.screen, bool | np.bool_):
            raise ValueError("the screen is a bool")
        if not (self.span is None or isinstance(self.span, SpanTest)) or not (
            self.relation is None or isinstance(self.relation, Relation)
        ):
            raise ValueError("the span and relation are their records")
        values = [
            None if value is None else real(value, what, positive=what == "scale")
            for value, what in (
                (self.scale, "scale"),
                (self.original_norm, "a norm"),
                (self.within_norm, "a norm"),
            )
        ]
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "scale", values[0])
        object.__setattr__(self, "original_norm", values[1])
        object.__setattr__(self, "within_norm", values[2])
        if self.screen is not None:
            object.__setattr__(self, "screen", bool(self.screen))


def _indices(values: object, what: str) -> IndexArray:
    """Owned read-only column indices, each an exact integer checked first."""
    return frozen_index(np.asarray(integers(values, what), dtype=np.intp))


@dataclass(frozen=True, eq=False)
class CollinearityAnalysis:
    """The outcome of one analysis: reasons, kept and omitted columns, evidence.

    ``reasons``: one per original column, the proposed reason or ``None``, absent
    when ``inconclusive``; ``kept``/``omitted``: ascending original indices, present
    when ``complete`` only; ``cause``: empty, ``error_mode`` or the failure;
    ``targets``, ``absorption`` (the absorber's own record), ``fresh`` (the fresh
    unit-weight check, kept when it completed, failing included), ``within`` and
    ``factorizations`` (every attempt, in order): the evidence and work, ``None`` or
    empty where none was formed; ``problem`` and ``controls``: the problem record and
    the actual controls of the invocation (module docstring, binding), ``None`` on an
    unbound private record.
    """

    mode: str
    status: str
    cause: str
    message: str
    names: tuple[str, ...]
    positions: IndexArray
    reasons: tuple[str | None, ...] | None
    kept: IndexArray | None
    omitted: IndexArray | None
    columns: tuple[ColumnDiagnostics, ...]
    targets: Targets | None = None
    absorption: AbsorptionResult | None = None
    fresh: FreshCheck | None = None
    within: WithinRank | None = None
    factorizations: tuple[Factorization, ...] = ()
    problem: CollinearityProblem | None = None
    controls: DiagnosticControls | None = None

    def __post_init__(self) -> None:
        """``ValueError`` for any inconsistency; arrays and tuples are owned."""
        names = _names(self.names)
        object.__setattr__(self, "names", names)
        positions = np.asarray(self.positions)
        object.__setattr__(self, "positions", _positions(positions, positions.size))
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "factorizations", tuple(self.factorizations))
        expected = {COMPLETE: ("",), REFUSED: ("error_mode",), INCONCLUSIVE: CAUSES}
        if self.mode not in MODES or self.cause not in expected.get(self.status, ()):
            raise ValueError("unknown mode, status or cause")
        if not isinstance(self.message, str):
            raise ValueError("the message is a string")
        self._records(names)
        self._outcome(len(names))
        problem = self.problem
        if (problem is not None or self.controls is not None) and not (
            isinstance(problem, CollinearityProblem) and self._bound(problem)
        ):
            raise ValueError(BOUND)

    def _bound(self, problem: CollinearityProblem) -> bool:
        """Names, rows, controls exactly with blocks, targets, work and column kinds."""
        controls, targets, absorption = self.controls, self.targets, self.absorption
        same = self.names == problem.names and np.array_equal(
            self.positions, problem.positions
        )
        same &= tuple(c.kind for c in self.columns) == column_kinds(problem.design)
        if not isinstance(controls, DiagnosticControls):
            free = targets is None and absorption is None and controls is None
            return same and free and problem.grouping is None
        same &= problem.grouping is not None
        n, p = problem.n_rows, problem.n_columns
        if targets is not None:
            same &= targets == diagnostic_targets(n, p, controls.tolerance)
        if absorption is None:
            return same
        work, budget = absorption.sweeps, controls.max_sweeps
        same &= work <= budget and absorption.acceleration == controls.acceleration
        # A budget stop: the next unit, of one or two sweeps, did not fit the budget.
        return same and (absorption.status != BUDGET or budget <= work + 1)

    def _records(self, names: tuple[str, ...]) -> None:
        """One column record per name, in order; every evidence field its record."""
        listed = [
            (c.index, c.name) if isinstance(c, ColumnDiagnostics) else None
            for c in self.columns
        ]
        if listed != list(enumerate(names)) or not all(
            isinstance(f, Factorization) for f in self.factorizations
        ):
            raise ValueError(
                "one diagnostic per column, in order; factorization records"
            )
        evidence = (
            (self.targets, Targets),
            (self.absorption, AbsorptionResult),
            (self.fresh, FreshCheck),
            (self.within, WithinRank),
        )
        if not all(
            value is None or isinstance(value, kind) for value, kind in evidence
        ):
            raise ValueError("targets, absorption, fresh and within are their records")

    def _outcome(self, p: int) -> None:
        """The reasons, kept and omitted columns agree with the status and mode."""
        if self.status == INCONCLUSIVE:
            if (self.reasons, self.kept, self.omitted) != (None, None, None):
                raise ValueError("an inconclusive analysis proposes nothing")
            return
        reasons = tuple(self.reasons or ())
        object.__setattr__(self, "reasons", reasons)
        if len(reasons) != p or not set(reasons) <= {None, ABSORBED, COLLINEAR}:
            raise ValueError("one known reason or None per column")
        proposed = [j for j, reason in enumerate(reasons) if reason is not None]
        if self.status == REFUSED:
            if self.mode != "error" or not proposed or self.kept is not None:
                raise ValueError("a refusal is error mode with proposals, nothing kept")
            if self.omitted is not None:
                raise ValueError("a refusal omits nothing")
            return
        if self.kept is None or self.omitted is None:
            raise ValueError("a complete analysis has kept and omitted columns")
        kept = _indices(self.kept, "kept columns")
        omitted = _indices(self.omitted, "omitted columns")
        rest = [j for j in range(p) if j not in proposed]
        if list(omitted) != proposed or list(kept) != rest:
            raise ValueError("kept and omitted columns must follow the reasons")
        if self.mode == "error" and proposed:
            raise ValueError("error mode applies no omission")
        object.__setattr__(self, "kept", kept)
        object.__setattr__(self, "omitted", omitted)


# ------------------------------------------------- admission of the absorber's state


class FailedMeansError(InconclusiveError):
    """The ``means`` refusal of a completed fresh check, which it carries."""

    def __init__(self, check: FreshCheck, effective: float) -> None:
        """Keep the completed check and state its maximum beside ``t_diag``."""
        super().__init__(
            "means",
            f"a fresh group mean {check.maximum!r} exceeds t_diag {effective!r}",
        )
        self.check = check


def _accepted(
    result: AbsorptionResult,
    values: FloatArray,
    prepared: PreparedWeights,
    names: tuple[str, ...],
    effective: float,
) -> FreshCheck:
    """Success, the exact binding of the residual and the fresh all-block means."""
    if not result.converged:
        raise InconclusiveError("absorption", f"{result.status}: {result.reason}")
    state, grouping = result.state, prepared.grouping
    bound = (
        state.columns == names
        and result.residual.shape == values.shape
        and state.grouping.names == grouping.names
        and np.array_equal(state.grouping.positions, grouping.positions)
        and all(
            a.n_levels == b.n_levels and np.array_equal(a.codes, b.codes)
            for a, b in zip(state.grouping.blocks, grouping.blocks, strict=True)
        )
    )
    if not bound:
        raise InconclusiveError(
            "state", "the absorption is bound to other rows or columns"
        )
    if not np.array_equal(
        candidate(values, state.coefficients, grouping), result.residual
    ):
        raise InconclusiveError(
            "state", "the residual is not the values minus the lift"
        )
    check = fresh_check(result.residual, prepared)
    if not check.maximum <= effective:
        raise FailedMeansError(check, effective)
    return check
