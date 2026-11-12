"""Private simple-projection inference, algorithm ``projection_map_v2``.

At the unchanged selected point,
use fresh C, one cold slope absorption, ALL-level canonical witnesses, reconstruction
and weighted-mean checks, one K-dimensional transpose factor/solve and full raw-H
and scaled-A backward checks. Deliver W=[Z;-TZ] to the unchanged full-W provider.
No correction, nuisance solve, fallback or forward-accuracy certificate is implied.

S_limit=M and E_limit=n_blocks*(c+1)+2K retain the original derived limits. The
absorber debits its actual sweeps/checks; reconstruction checks reserve (0,blocks),
the initial G check (0,K), H and A checks each (K,K). Fixed projection operations
are counted separately, before each attempt. All actions have round=0; no correction
rounds or PCG slots exist. Admission recomputes scalar predicates, action sequence,
counts and bindings without matrix work. Digests bind bytes, never validate numerics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.fe.absorb import AbsorptionResult, absorb
from powermean.fe.absorb_numeric import (
    Attempted,
    PreparedWeights,
    checked_difference,
    checked_product,
    checked_quotient,
    fresh_check,
    prepare_weights,
)
from powermean.fe.certificate_basis import SupportCheck, embed, lift
from powermean.model.variance import FloatArray
from powermean.results.ppml_residualized_map import (
    BudgetExceededError,
    Coordinates,
    SparseBlock,
    cross,
    normwise,
    stored_block,
    transpose_action,
    witness_coordinates,
)
from powermean.results.ppml_standard_inference import (
    NO_SLOPES,
    NOT_CONVERGED,
    NOT_FINITE_FACTOR,
    NOT_FINITE_MAP,
    _geometry,
    _provided,
)
from powermean.results.ppml_standard_records import (
    ALGORITHM,
    FULL_NOT_RUN,
    NUISANCE_SOLVER,
    BudgetRefusal,
    InferenceMethod,
    LedgerRecord,
    MapCheck,
    ProjectionRecord,
    Reservation,
    ResidualizedEvidence,
    StandardCheck,
    StandardControls,
    StandardFailure,
    StandardInference,
    StandardPointAssessment,
    bounded,
)
from powermean.solve.absorbed_certificate_factor import (
    OPTIONS,
    THRESHOLDS,
    Calls,
    FactorRecord,
    SuperLU,
    factor_side,
    factorized,
    failed_factor,
    index_width,
)
from powermean.solve.absorbed_certificate_geometry import (
    FEGeometry,
    _Entries,
    _slope_entries,
    canonical,
    equilibration,
    one_norm,
    scaled_ordinary,
)
from powermean.solve.absorbed_certificate_point import BOUND, StepStatus, WorkStep
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_certificate_sparse import SUPPORT, slope_map
from powermean.solve.newton_absorbed import array_digest

METHOD: InferenceMethod = "residualized"
UNADMITTED = "the residualized evidence does not admit against the reading"
MISSED = "misses its normwise backward bound"
WIDE_ACTIONS = ("H check", "A check")
FIXED_WORK = (
    "witness conversion",
    "reconstruction lift",
    "slope construction",
    "slope factor",
    "slope map",
    "initial T cross",
)
OUTCOMES = frozenset({"completed", "refused"})
CALL_KEYS = ("n", "n_columns", "t", "t_columns", "triangular", "triangular_columns")
_IDENTITIES = frozenset({"sample", "grouping", "family"})


class _RefusedError(Exception):
    """A refusal of the route at one stage, with its reason."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


# --------------------------------------------------------------------- ledgers


def limits(
    controls: StandardControls, slopes: int, blocks: int
) -> tuple[int, int, int]:
    """``S_limit = M``, ``E_limit = n_blocks (c + 1) + 2K`` and ``c`` (docstring)."""
    budget = controls.absorb_max_sweeps
    if blocks == 1 or not controls.absorb_acceleration:
        c = budget + 1
    else:
        c = max(1, budget // 2)
    return budget, blocks * (c + 1) + 2 * slopes, c


class _Ticket:
    """One reserved action: its units, the work begun and its outcome (local)."""

    def __init__(self, head: tuple[str, str, int, int], s: int, e: int) -> None:
        self.head = head
        self.units = (s, e)
        self.products = 0
        self.dots = 0
        self.checks = 0
        self.outcome = "begun"

    def begin(self, products: int = 0, dots: int = 0, checks: int = 0) -> None:
        """Count right-hand sides, dot evaluations and check columns as begun."""
        self.products += products
        self.dots += dots
        self.checks += checks

    def close(self, outcome: str) -> None:
        """The outcome of the action."""
        self.outcome = outcome

    def frozen(self) -> Reservation:
        """The typed reservation; an action left begun was refused by range."""
        outcome = "refused" if self.outcome == "begun" else self.outcome
        attempted = (self.products, self.products + self.dots + self.checks)
        counts = (self.products, self.dots, self.checks)
        return Reservation(*self.head, *self.units, *attempted, *counts, outcome)


class _Ledgers:
    """Both shared ledgers of one call (module docstring); local, frozen at the end."""

    def __init__(self, bounds: tuple[int, int, int]) -> None:
        self.s_limit, self.e_limit, self.c = bounds
        self.s_used = 0
        self.e_used = 0
        self.entries: list[Reservation | _Ticket] = []
        self.refusal: BudgetRefusal | None = None
        self.last_checked = "none: no candidate map was checked"

    def debit(self, head: tuple[str, str, int, int], s: int, e: int) -> _Ticket:
        """Reserve ``s`` S and ``e`` E units before an action, or refuse the budget."""
        if self.refusal is not None:
            raise BudgetExceededError(self.refusal)
        if self.s_used + s > self.s_limit or self.e_used + e > self.e_limit:
            self.refusal = BudgetRefusal(
                *head,
                s,
                e,
                self.s_used,
                self.e_used,
                self.s_limit - self.s_used,
                self.e_limit - self.e_used,
                self.last_checked,
                f"{head[1]} and every later action of the route; no provider call",
            )
            raise BudgetExceededError(self.refusal)
        self.s_used += s
        self.e_used += e
        ticket = _Ticket(head, s, e)
        self.entries.append(ticket)
        return ticket

    def absorbed(self, result: AbsorptionResult) -> None:
        """Debit the absorber's own reserved sweeps and diagnostic passes, checked."""
        s, e = result.sweeps, result.diagnostic_passes
        if self.s_used + s > self.s_limit or self.e_used + e > self.e_limit:
            raise _RefusedError("budget", "the absorption exceeds the shared ledgers")
        self.s_used += s
        self.e_used += e
        status = "completed" if result.converged else "refused"
        head = ("projection", "absorption", 0, -1)
        self.entries.append(Reservation(*head, s, e, s, e, 0, 0, e, status))

    def frozen(self) -> LedgerRecord:
        """The typed ledgers record, every reservation in order."""
        found = tuple(
            entry if isinstance(entry, Reservation) else entry.frozen()
            for entry in self.entries
        )
        return LedgerRecord(self.s_limit, self.e_limit, self.c, found, self.refusal)


@dataclass
class _Run:
    """The evidence of one call as it forms (local), frozen once at the end."""

    stage: str = "geometry"
    projection: ProjectionRecord | None = None
    factor: FactorRecord | None = None
    calls: Calls | None = None
    initial: tuple[MapCheck, ...] = ()
    backward: tuple[MapCheck, ...] = ()
    fixed: dict[str, int] = field(default_factory=dict)
    failures: list[StandardFailure] = field(default_factory=list)


def geometry_digest(geometry: FEGeometry) -> str:
    """The SHA-256 of the geometry's binding by value, its identities excepted.

    Every array of :meth:`~powermean.solve.absorbed_certificate_geometry.FEGeometry
    .binding` by dtype, shape and bytes, and its scalar items, in order; the sample,
    grouping and family identities, which a rebinding renames, are left out.
    """
    found = hashlib.sha256()
    for key, value in geometry.binding():
        if key in _IDENTITIES:
            continue
        found.update(key.encode())
        if isinstance(value, tuple) and len(value) == 3 and type(value[2]) is bytes:
            found.update(repr(value[:2]).encode())
            found.update(value[2])
        else:
            found.update(repr(value).encode())
    return found.hexdigest()


def _evidence(
    reading: StandardPointAssessment,
    geometry: FEGeometry | None,
    ledgers: _Ledgers,
    run: _Run,
    delivered: FloatArray | None,
) -> ResidualizedEvidence:
    """The scalar evidence of the call (module docstring)."""
    controls, point = reading.controls, reading.point
    grouping = reading.basis.grouping
    actions = (Calls() if run.calls is None else run.calls).actions()
    return ResidualizedEvidence(
        METHOD,
        METHOD,
        ALGORITHM,
        controls.absorb_tol_final,
        controls.tol,
        controls.absorb_max_sweeps,
        controls.absorb_acceleration,
        len(grouping.blocks),
        point.n_slopes,
        reading.basis.n_kept,
        FULL_NOT_RUN,
        0,
        0,
        NUISANCE_SOLVER,
        run.factor,
        actions,
        run.projection,
        run.initial,
        run.backward,
        tuple(run.fixed.items()),
        ledgers.frozen(),
        tuple(run.failures),
        None if delivered is None else array_digest(delivered),
        None if geometry is None or delivered is None else geometry_digest(geometry),
    )


def _steps(run: _Run, ledger: LedgerRecord) -> list[WorkStep]:
    """The actual work of the projection, factor and backward checks."""
    found: list[WorkStep] = []
    projection = run.projection
    if projection is not None:
        status: StepStatus = (
            "completed" if projection.status == "converged" else "failed"
        )
        counters: tuple[tuple[str, int], ...] = (
            ("sweeps", projection.sweeps),
            ("visits", projection.visits),
            ("operators", projection.operators),
            ("diagnostic_passes", projection.diagnostic_passes),
        )
        found.append(WorkStep("residualized projection", status, "", counters))
    factor = run.factor
    if factor is not None:
        status = "completed" if factor.status == "factored" else "failed"
        nonzeros = None
        if factor.l_nnz is not None and factor.u_nnz is not None:
            nonzeros = (("L nonzeros", factor.l_nnz), ("U nonzeros", factor.u_nnz))
        found.append(
            WorkStep("K-dimensional slope factor", status, factor.reason, nonzeros)
        )
    counters = (
        ("S used", ledger.s_used),
        ("S limit", ledger.s_limit),
        ("E used", ledger.e_used),
        ("E limit", ledger.e_limit),
        *run.fixed.items(),
    )
    status = "refused" if run.failures else "completed"
    reason = run.failures[0].reason if run.failures else ""
    found.append(WorkStep("simple projection checks", status, reason, counters))
    return found


# --------------------------------------------------------------- projection


def _admissible(geometry: FEGeometry, support: SupportCheck) -> None:
    """The refusals before any alternative work (module docstring)."""
    if not support.full:
        raise _RefusedError("support", SUPPORT)
    ordinary = geometry.ordinary
    if ordinary is None or geometry.scale is None:
        raise _RefusedError("equilibration", geometry.refusal)
    width = index_width(geometry.dimension, int(ordinary.data.size))
    if width:
        raise _RefusedError("index width", width)
    if not (math.isfinite(geometry.norm) and np.isfinite(geometry.hessian.data).all()):
        raise _RefusedError("geometry", "the admitted geometry is not finite")


def _absorbed(
    reading: StandardPointAssessment, geometry: FEGeometry, ledgers: _Ledgers, run: _Run
) -> tuple[AbsorptionResult, PreparedWeights]:
    """The one cold absorption of the slopes with ``C``, debited and checked."""
    run.stage = "projection"
    controls, basis = reading.controls, reading.basis
    names = reading.point.sample.active_names
    af = controls.absorb_tol_final
    try:
        prepared = prepare_weights(basis.grouping, geometry.rows.curvature)
        result = absorb(
            geometry.design,
            prepared,
            names,
            tolerance=af,
            max_sweeps=controls.absorb_max_sweeps,
            acceleration=controls.absorb_acceleration,
        )
    except ValueError as error:
        raise _RefusedError("projection", str(error)) from error
    ledgers.absorbed(result)
    errors = () if result.column_errors is None else tuple(result.column_errors)
    returned = bounded("absorption group means", result.maximum, (af,))
    run.projection = ProjectionRecord(
        result.status,
        result.reason,
        af,
        controls.absorb_max_sweeps,
        controls.absorb_acceleration,
        result.sweeps,
        result.visits,
        result.operators,
        result.diagnostic_passes,
        errors,
        0,
        0,
        (returned,),
    )
    if not result.converged:
        raise _RefusedError("projection", f"{result.status}: {result.reason}")
    state = result.state
    if state.columns != names or state.grouping is not basis.grouping:
        raise _RefusedError("projection", BOUND)
    if (
        not returned.passed
        or len(errors) != len(names)
        or not all(math.isfinite(error) and 0.0 <= error <= af for error in errors)
    ):
        raise _RefusedError("projection", returned.reason or "returned group means")
    return result, prepared


def _reconstructed(
    reading: StandardPointAssessment,
    geometry: FEGeometry,
    found: tuple[AbsorptionResult, PreparedWeights],
    ledgers: _Ledgers,
    run: _Run,
) -> Coordinates:
    """``T`` by the witnesses and the checks of ``R_c`` (module docstring)."""
    result, prepared = found
    controls, basis = reading.controls, reading.basis
    af, tol = controls.absorb_tol_final, controls.tol
    record = run.projection
    if record is None:
        raise ValueError(BOUND)
    run.fixed["witness conversion"] = 1
    coordinates = witness_coordinates(basis, result.state.coefficients)
    record = replace(
        record, witness_terms=coordinates.terms, inexact_ratios=coordinates.inexact
    )
    run.projection = record
    run.fixed["reconstruction lift"] = 1
    lifted = lift(basis, embed(basis, coordinates.values))
    rebuilt = checked_difference(geometry.design, lifted, "reconstructed residual")
    gap = float(np.max(np.abs(checked_difference(rebuilt, result.residual, "gap"))))
    checks: tuple[StandardCheck, ...] = (
        *record.checks,
        bounded("reconstruction gap", gap, (af, tol)),
    )
    run.projection = replace(record, checks=checks)
    if not checks[-1].passed:
        raise _RefusedError("projection", checks[-1].reason)
    ticket = ledgers.debit(
        ("projection", "reconstruction check", 0, -1), 0, len(basis.grouping.blocks)
    )
    attempted = Attempted()
    try:
        maximum = fresh_check(rebuilt, prepared, attempted).maximum
    finally:
        ticket.begin(checks=attempted.count)
    ticket.close("completed")
    checks = (*checks, bounded("reconstructed group means", maximum, (af,)))
    run.projection = replace(record, checks=checks)
    if not checks[-1].passed:
        raise _RefusedError("projection", checks[-1].reason)
    return coordinates


# ------------------------------------------------------------ the slope bread


@dataclass(frozen=True, eq=False)
class _Bread:
    """The one ordinary factor of the K-dimensional slope bread and its scaled block."""

    record: FactorRecord
    factor: SuperLU
    scale: FloatArray
    block: SparseBlock
    norm: float
    calls: Calls


def _bread(geometry: FEGeometry, residual: FloatArray, run: _Run) -> _Bread:
    """``G = R' D R``, its scaling, 1-norm and ordinary factor (module docstring)."""
    run.stage = "slope factor"
    k = residual.shape[1]
    run.fixed["slope construction"] = 1
    entries = _Entries()
    _slope_entries(residual, geometry.rows.curvature, entries)
    rows = np.concatenate([np.zeros(0, dtype=np.intp), *entries.rows])
    cols = np.concatenate([np.zeros(0, dtype=np.intp), *entries.cols])
    values = np.concatenate([np.zeros(0), *entries.values])
    try:
        matrix = canonical(rows, cols, values, k)
        scale = equilibration(matrix)
    except ValueError as error:
        raise _RefusedError("slope factor", f"the slope bread G: {error}") from error
    scaled = scaled_ordinary(matrix, scale)
    norm = one_norm(scaled)
    run.fixed["slope factor"] = 1
    record, factor = factorized(scaled, "ordinary")
    run.factor = record
    if factor is None:
        raise _RefusedError("slope factor", record.reason)
    lower, upper = factor_side(factor, "L"), factor_side(factor, "U")
    if not (np.isfinite(lower.data).all() and np.isfinite(upper.data).all()):
        run.factor = failed_factor(record, NOT_FINITE_FACTOR)
        raise _RefusedError("slope factor", NOT_FINITE_FACTOR)
    calls = Calls()
    run.calls = calls
    return _Bread(record, factor, scale, stored_block(scaled, 0, (0, k)), norm, calls)


# ---------------------------------------------------------- full map checks


@dataclass(frozen=True, eq=False)
class _Blocks:
    """The stored blocks of ``H`` and ``A``, ``E_s`` and ``||H'||_inf = ||H||_1``."""

    hessian: SparseBlock
    ordinary: SparseBlock
    unit: FloatArray
    norm: float


def _blocks(geometry: FEGeometry) -> _Blocks:
    """The original full H and A, without extracting a nuisance solver."""
    hessian, ordinary, k = geometry.hessian, geometry.ordinary, geometry.n_slopes
    if ordinary is None:
        raise ValueError(BOUND)
    size = hessian.shape[1]
    unit = np.zeros((size, k))
    unit[np.arange(k), np.arange(k)] = 1.0
    return _Blocks(
        stored_block(hessian, 0, (0, size)),
        stored_block(ordinary, 0, (0, size)),
        unit,
        one_norm(hessian),
    )


@dataclass(frozen=True, eq=False)
class _Context:
    """The local state of one simple projection and its full backward checks."""

    geometry: FEGeometry
    controls: StandardControls
    blocks: _Blocks
    coordinates: FloatArray
    bread: _Bread
    ledgers: _Ledgers
    run: _Run

    @property
    def slopes(self) -> int:
        """``K``."""
        return int(self.bread.scale.size)

    def debit(self, action: str, stage: int) -> _Ticket:
        """Reserve one K-column backward action on both ledgers."""
        k = self.slopes
        return self.ledgers.debit(("backward", action, stage, -1), k, k)


def _initial_map(context: _Context) -> FloatArray:
    """``Z`` solving ``G' Z = I`` by the slope map, checked, and ``W0 = [Z; -T Z]``."""
    run, bread, k = context.run, context.bread, context.slopes
    run.stage = "initial map"
    run.fixed["slope map"] = 1
    z = slope_map(bread.factor, bread.scale, k, bread.calls)
    if not np.isfinite(z).all():
        raise _RefusedError("initial map", NOT_FINITE_MAP)
    ticket = context.ledgers.debit(("initial", "slope initial check", 0, -1), 0, k)
    ticket.begin(checks=k)
    x = checked_product(bread.scale[:, None], z, "scaled slope solution")
    rhs = np.diag(checked_quotient(np.ones(k), bread.scale, "scaled unit"))
    residual = checked_difference(
        transpose_action(bread.block, x), rhs, "slope residual"
    )
    af = context.controls.absorb_tol_final
    run.initial = normwise(
        "slope initial", 0, residual, x, rhs, norm=bread.norm, tolerance=af
    )
    ticket.close("completed")
    if not all(check.passed for check in run.initial):
        raise _RefusedError("initial map", f"G' Z = I {MISSED}")
    run.fixed["initial T cross"] = k
    initial = np.vstack([z, -cross(context.coordinates, z)])
    if not np.isfinite(initial).all():
        raise _RefusedError("initial map", NOT_FINITE_MAP)
    return initial


def _backward(context: _Context, new: FloatArray, stage: int) -> tuple[MapCheck, ...]:
    """The raw ``H`` and scaled ``A`` backward predicates of each column of the map."""
    geometry, blocks = context.geometry, context.blocks
    unit, af, k = blocks.unit, context.controls.absorb_tol_final, context.slopes
    scale = geometry.scale
    if scale is None:
        raise ValueError(BOUND)
    ticket = context.debit("H check", stage)
    ticket.begin(products=k)
    product = transpose_action(blocks.hessian, new)
    residual = checked_difference(product, unit, "raw backward residual")
    raw = normwise(
        "H backward", stage, residual, new, unit, norm=blocks.norm, tolerance=af
    )
    ticket.close("completed")
    context.run.backward = raw
    ticket = context.debit("A check", stage)
    ticket.begin(products=k)
    x = checked_product(scale[:, None], new, "scaled map")
    rhs = checked_quotient(unit, scale[:, None], "scaled unit")
    residual = checked_difference(transpose_action(blocks.ordinary, x), rhs, "scaled")
    norm = geometry.norm
    found = normwise("A backward", stage, residual, x, rhs, norm=norm, tolerance=af)
    ticket.close("completed")
    return (*raw, *found)


def _route(
    reading: StandardPointAssessment, geometry: FEGeometry, ledgers: _Ledgers, run: _Run
) -> FloatArray:
    """Projection, one slope factor/solve and full-H/A backward checks."""
    support = reading.support
    if support is None:
        raise ValueError(NOT_CONVERGED)
    _admissible(geometry, support)
    blocks = _blocks(geometry)
    absorbed = _absorbed(reading, geometry, ledgers, run)
    coordinates = _reconstructed(reading, geometry, absorbed, ledgers, run).values
    bread = _bread(geometry, absorbed[0].residual, run)
    context = _Context(
        geometry, reading.controls, blocks, coordinates, bread, ledgers, run
    )
    initial = _initial_map(context)
    run.stage = "full map checks"
    run.backward = _backward(context, initial, 0)
    ledgers.last_checked = "projection map: full H and A backward checks recorded"
    if not all(check.passed for check in run.backward):
        raise _RefusedError("full map checks", f"H or A {MISSED}")
    return initial


def residualized_inference(
    reading: StandardPointAssessment, request: CovarianceRequest
) -> StandardInference:
    """The residualized inference at a converged reading's point (module docstring).

    ``request`` is the owned replica's covariance request of that reading.  A refusal
    is an inference ``refused`` with its evidence; it never starts another method.
    """
    point, rows, support = reading.point, reading.rows, reading.support
    if not point.n_slopes:
        return StandardInference(
            "not_applicable", NO_SLOPES, requested=METHOD, executed=METHOD
        )
    if rows is None or support is None or not reading.accepted:
        raise ValueError(NOT_CONVERGED)
    blocks = len(reading.basis.grouping.blocks)
    ledgers = _Ledgers(limits(reading.controls, point.n_slopes, blocks))
    run = _Run()
    steps: list[WorkStep] = []
    found: FEGeometry | str = _geometry(point, rows, reading.basis, support, steps)
    geometry = None if isinstance(found, str) else found
    delivered: FloatArray | None = None
    try:
        if geometry is None:
            raise _RefusedError("geometry", str(found))
        delivered = _route(reading, geometry, ledgers, run)
    except _RefusedError as error:
        run.failures.append(StandardFailure(error.stage, error.reason))
    except BudgetExceededError as error:
        run.failures.append(StandardFailure("budget", str(error)))
    except FloatingPointError as error:
        run.failures.append(StandardFailure(run.stage, f"range: {error}"))
    ledger = ledgers.frozen()
    steps.extend(_steps(run, ledger))
    evidence = _evidence(reading, geometry, ledgers, run, delivered)
    if delivered is None or geometry is None:
        failure = run.failures[0].reason
        return StandardInference(
            "refused",
            failure,
            geometry,
            work=tuple(steps),
            requested=METHOD,
            executed=METHOD,
            residualized=evidence,
        )
    provided = _provided(
        geometry,
        None,
        delivered,
        request=request,
        actions=None,
        steps=steps,
        method=METHOD,
    )
    return replace(provided, residualized=evidence)


# ------------------------------------------------------------------ admission


def _unadmitted(condition: bool | int) -> None:
    """``ValueError`` (the evidence is not admitted) when ``condition`` holds."""
    if condition:
        raise ValueError(UNADMITTED)


def _same_method(evidence: ResidualizedEvidence, inference: StandardInference) -> None:
    """The method, its labels, no full-system factor and the slope factor's counts."""
    names = (evidence.requested, evidence.executed, evidence.algorithm)
    _unadmitted(
        names != (METHOD, METHOD, ALGORITHM)
        or evidence.full_system_factor != FULL_NOT_RUN
        or (evidence.full_factor_calls, evidence.full_solve_calls) != (0, 0)
        or evidence.nuisance_solver != NUISANCE_SOLVER
        or inference.ordinary is not None
        or inference.actions is not None
    )
    factor = evidence.slope_factor
    factored = factor is not None and factor.status == "factored"
    _unadmitted(
        factor is not None
        and (
            factor.kind != "ordinary"
            or factor.threshold != THRESHOLDS["ordinary"]
            or dict(factor.options) != dict(OPTIONS["ordinary"])
        )
    )
    actions = dict(evidence.slope_actions)
    calls = actions.get("t", -1)
    _unadmitted(
        tuple(sorted(actions)) != tuple(sorted(CALL_KEYS))
        or any(
            actions[key]
            for key in ("n", "n_columns", "triangular", "triangular_columns")
        )
        or calls not in (0, 1)
        or (calls and not factored)
        or actions["t_columns"] != calls * evidence.slopes
        or (evidence.passed and calls != 1)
    )


def _same_controls(
    evidence: ResidualizedEvidence, reading: StandardPointAssessment
) -> None:
    """The evidence's controls and sizes are the reading's, bit for bit."""
    controls, basis = reading.controls, reading.basis
    af, tol = controls.absorb_tol_final, controls.tol
    floats = ((evidence.tolerance, af), (evidence.tol, tol))
    _unadmitted(
        not all(same_bits(np.array([a]), np.array([b])) for a, b in floats)
        or (evidence.max_sweeps, evidence.acceleration)
        != (controls.absorb_max_sweeps, controls.absorb_acceleration)
        or evidence.slopes != reading.point.n_slopes
        or evidence.kept != basis.n_kept
        or evidence.n_blocks != len(basis.grouping.blocks)
    )
    checks = (*evidence.initial, *evidence.backward)
    _unadmitted(any(check.tolerance != af for check in checks))
    projection = evidence.projection
    if projection is not None:
        expected = [(af,), (af, tol), (af,)][: len(projection.checks)]
        _unadmitted(
            (projection.tolerance, projection.max_sweeps, projection.acceleration)
            != (af, controls.absorb_max_sweeps, controls.absorb_acceleration)
            or [check.bounds for check in projection.checks] != expected
            or (
                projection.status == "converged"
                and (
                    len(projection.column_errors) != evidence.slopes
                    or not all(
                        math.isfinite(v) and 0.0 <= v <= af
                        for v in projection.column_errors
                    )
                )
            )
        )


def _expected_units(
    entry: Reservation, evidence: ResidualizedEvidence
) -> tuple[int, int]:
    """The S and E units that an action reserves (module docstring)."""
    k = evidence.slopes
    if entry.action in WIDE_ACTIONS:
        return k, k
    if entry.action == "reconstruction check":
        return 0, evidence.n_blocks
    if entry.action == "slope initial check":
        return 0, k
    raise ValueError(UNADMITTED)


def _same_reservation(entry: Reservation, evidence: ResidualizedEvidence) -> None:
    """One reservation's units, attempted work and outcome are coherent."""
    counts = (entry.products, entry.dots, entry.checks)
    _unadmitted(
        min(entry.s, entry.e, entry.attempted_s, entry.attempted_e, *counts) < 0
        or entry.attempted_s > entry.s
        or entry.attempted_e > entry.e
        or entry.outcome not in OUTCOMES
        or entry.round != 0
        or entry.rhs != -1
        or entry.dots != 0
        or (entry.outcome == "completed" and entry.attempted_e != entry.e)
    )
    phase = {
        "absorption": "projection",
        "reconstruction check": "projection",
        "slope initial check": "initial",
        "H check": "backward",
        "A check": "backward",
    }
    _unadmitted(entry.phase != phase.get(entry.action))
    projection = evidence.projection
    if entry.action == "absorption":
        sizes = (entry.s, entry.e, entry.attempted_s, entry.attempted_e)
        _unadmitted(
            projection is None
            or sizes != (projection.sweeps, projection.diagnostic_passes) * 2
            or entry.checks != entry.e
            or entry.products
            or entry.dots
        )
        return
    _unadmitted(
        (entry.s, entry.e) != _expected_units(entry, evidence)
        or entry.attempted_s != entry.products
        or entry.attempted_e != sum(counts)
        or (entry.outcome == "completed" and entry.products != entry.s)
    )


def _same_ledgers(
    evidence: ResidualizedEvidence, reading: StandardPointAssessment
) -> None:
    """Both ledgers' identities, recomputed from the typed reservations."""
    ledger = evidence.ledger
    entries = ledger.reservations
    bounds = limits(reading.controls, evidence.slopes, evidence.n_blocks)
    _unadmitted(
        (ledger.s_limit, ledger.e_limit, ledger.c) != bounds
        or ledger.s_used > ledger.s_limit
        or ledger.e_used > ledger.e_limit
        or (bool(entries) and entries[0].action != "absorption")
        or any(entry.action == "absorption" for entry in entries[1:])
    )
    for entry in entries:
        _same_reservation(entry, evidence)
    refusal = ledger.refusal
    if refusal is not None:
        left = (ledger.s_limit - ledger.s_used, ledger.e_limit - ledger.e_used)
        _unadmitted(
            (refusal.used_s, refusal.used_e) != (ledger.s_used, ledger.e_used)
            or (refusal.remaining_s, refusal.remaining_e) != left
            or (refusal.required_s <= left[0] and refusal.required_e <= left[1])
            or not any(failure.stage == "budget" for failure in evidence.failures)
        )
    expected = (
        "absorption",
        "reconstruction check",
        "slope initial check",
        "H check",
        "A check",
    )
    _unadmitted(
        tuple(entry.action for entry in entries) != expected[: len(entries)]
        or (evidence.passed and len(entries) != len(expected))
    )


def _same_records(evidence: ResidualizedEvidence) -> None:
    """Every recorded bound and verdict is that of its raw values, recomputed."""
    projection = evidence.projection
    if projection is not None:
        _unadmitted(
            any(
                check.passed
                is not (
                    check.value is not None
                    and all(check.value <= bound for bound in check.bounds)
                )
                for check in projection.checks
            )
        )
    checks = (*evidence.initial, *evidence.backward)
    _unadmitted(not all(check.verified for check in checks))
    k = evidence.slopes
    initial = [("slope initial", 0, j) for j in range(k)]
    backward = [(name, 0, j) for name in ("H backward", "A backward") for j in range(k)]
    for found, expected in ((evidence.initial, initial), (evidence.backward, backward)):
        places = [(c.name, c.round, c.column) for c in found]
        _unadmitted(places != expected[: len(places)])
    fixed = evidence.fixed_work
    _unadmitted(
        tuple(name for name, _ in fixed) != FIXED_WORK[: len(fixed)]
        or any(
            value != (k if name == "initial T cross" else 1) for name, value in fixed
        )
        or (evidence.passed and len(fixed) != len(FIXED_WORK))
    )


def _same_outcome(evidence: ResidualizedEvidence, inference: StandardInference) -> None:
    """A delivered map is the evidence's, every predicate holding; else a refusal."""
    found, geometry = inference.slope_map, inference.geometry
    if found is None:
        _unadmitted(
            evidence.passed
            or not evidence.failures
            or inference.status != "refused"
            or inference.scale is not None
            or evidence.map_digest is not None
        )
        return
    _unadmitted(
        not evidence.passed
        or geometry is None
        or evidence.map_digest != array_digest(found)
        or evidence.geometry_digest != geometry_digest(geometry)
    )


def admit_residualized(
    inference: StandardInference, reading: StandardPointAssessment
) -> None:
    """The scalar admission of the residualized evidence (module docstring).

    ``ValueError`` unless the inference names the residualized method and its evidence
    admits against ``reading``'s controls and the inference's map and geometry; no
    matrix action, factor, projection or solve runs.
    """
    _unadmitted((inference.requested, inference.executed) != (METHOD, METHOD))
    evidence = inference.residualized
    formed = inference.status in ("available", "degenerate", "refused")
    if evidence is None:
        _unadmitted(formed)
        return
    _unadmitted(not formed or type(evidence) is not ResidualizedEvidence)
    _same_method(evidence, inference)
    _same_controls(evidence, reading)
    _same_ledgers(evidence, reading)
    _same_records(evidence)
    _same_outcome(evidence, inference)
