"""Joint separation support: controls, method selection, the FE step and the result.

``design/02_computation.md``, Sections 2 and 4 (existence for joint estimation with
``tau < 2`` and NB2: the fixed-effect step, the linear-programming step when there are
no absorbed effects, the iterative rectifier otherwise); ``design/06_reuse_map.md``,
``solve/support``; ``plan/evidence/M1.7_contract.txt`` and its adopted clarifications;
decision D51.  Task M1.7: a private diagnostic, not a fit, a sample constructor or a
public option.  ``powermean`` exports nothing of it.

One invocation.  :func:`analyze_support` is one fresh analysis of the supplied current
sample (:class:`~powermean.solve.support_problem.SupportProblem`) under explicit
controls: nothing is cached, warm-started or reused between invocations, and the
sample is never modified.  The method is selected before anything runs
(:func:`select_method`): without zero rows nothing runs and the empty set is complete;
without blocks and with ``n (p + 1)`` at most the recorded workspace bound (2,000,000,
S6's loading-element guard used as a private routing convention, not a measured
crossover) the LP of :mod:`powermean.solve.support_lp`; otherwise the iterative
rectifier of :mod:`powermean.solve.support_rectifier`, preceded with blocks by the FE
step. The sample analyzer may certify an empty set with a bounded auxiliary LP after
both inconclusive IR attempts without witnesses. The selected method remains IR.

The FE step.  A level of an estimated block with no positive ``Z`` is separated in
full: the indicator direction is zero on every positive row and positive on its rows.
The step proposes the sum of those indicators over every block, checks it afresh as any
witness, and records whether its strict set is exactly the rows of those levels (the
exact combinatorial support).  Cascades, deletions and empty samples belong to task
M1.10.  On two-block PPML with slopes and no level-only proposal, a directed graph
can propose a joint FE direction.  Its checked strict rows are returned for removal
and fresh analysis of the remaining sample.

The controls (contract A), the private resolved values recorded with every result:

    quantity                                   BASELINE     TIGHTER
    LP residual tolerance e_LP                  1e-12        1e-13
    LP strict threshold d_LP                    1e-8         1e-9
    HiGHS primal and dual feasibility           1e-8         1e-9
    IR residual tolerance e_IR                  1e-8         1e-9
    IR strict threshold d_IR                    1e-4         1e-5
    support WLS absolute equation/absorption    2e-11        2e-12
    LP iterations per solve                     1000         1000
    IR WLS attempts per invocation              100          100
    absorption sweeps per call                  20000        20000
    absorption acceleration                     True         True

:data:`TIGHTER` is the explicitly tighter invocation that an iteration trigger requests
(tasks M1.11 and M1.12): a new call on the current sample, not a retry.

The result.  :class:`SupportResult` keeps every witness covering the separated set
``S``, the last dual check, the work of the method and the stop.  Its status follows
from that evidence: ``complete`` when every zero row is witnessed or a fresh dual
establishes numerical completeness; ``partial`` for a witnessed nonempty set without
completion; ``inconclusive`` otherwise.  No optimizer flag, small objective, unchanged
update or exhausted budget is an absence of separation.  The record binds its evidence
when it is built: every retained witness and the retained dual must belong to its
sample, to the union ``S`` of the retained witnesses and to the active controls of its
method, and each is checked again there
(:func:`~powermean.solve.support_witness.require_primal`,
:func:`~powermean.solve.support_witness.require_dual`); the method's outcome must be
recorded with those controls.  A foreign sample, an unwitnessed set, relabelled
controls or a stored flag cannot support a claim.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.support_lp import LPControls, LPOutcome, separation_lp
from powermean.solve.support_overlap_lp import OverlapLP, certify_overlap
from powermean.solve.support_problem import (
    NONE,
    SupportProblem,
    indices,
    require_same,
)
from powermean.solve.support_projection import ProjectionControls
from powermean.solve.support_rectifier import (
    RectifierControls,
    RectifierOutcome,
)
from powermean.solve.support_rectifier import (
    iterative_rectifier as _iterative_rectifier,
)
from powermean.solve.support_witness import (
    CONTROLS,
    DualWitness,
    PrimalWitness,
    primal_witness,
    require_dual,
    require_primal,
    union,
)

if TYPE_CHECKING:
    from powermean.fe.groups import Block
    from powermean.solve.support_gamma_lp import Sparse, SparseMatrix

Method = Literal["lp", "ir"]
iterative_rectifier = _iterative_rectifier

Status = Literal["complete", "partial", "inconclusive"]
COMPLETIONS = ("", "covered", "dual", "overlap_lp")
# S6's 2,000,000 loading-element guard, the private LP workspace routing bound.
WORKSPACE = 2_000_000


@dataclass(frozen=True)
class SupportControls:
    """The explicit, recorded controls of one invocation.

    ``lp`` and ``ir`` hold the method controls (the rectifier's include the support
    WLS); ``workspace`` is the loading-element bound for the LP route and auxiliary LP.
    """

    lp: LPControls
    ir: RectifierControls
    workspace: int

    def __post_init__(self) -> None:
        """Check the records and the positive integer workspace bound."""
        if not isinstance(self.lp, LPControls) or not isinstance(
            self.ir, RectifierControls
        ):
            raise ValueError("lp and ir must be LPControls and RectifierControls")
        bound = self.workspace
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValueError("workspace must be a nonnegative integer")

    @property
    def level(self) -> str:
        """``baseline`` or ``tighter`` for a row of the table, ``custom`` otherwise."""
        if self == BASELINE:
            return "baseline"
        return "tighter" if self == TIGHTER else "custom"


BASELINE = SupportControls(
    lp=LPControls(
        tolerance=1e-12, threshold=1e-8, feasibility=1e-8, max_iterations=1000
    ),
    ir=RectifierControls(
        tolerance=1e-8,
        threshold=1e-4,
        max_attempts=100,
        projection=ProjectionControls(
            tolerance=2e-11, max_sweeps=20000, acceleration=True
        ),
    ),
    workspace=WORKSPACE,
)
TIGHTER = SupportControls(
    lp=LPControls(
        tolerance=1e-13, threshold=1e-9, feasibility=1e-9, max_iterations=1000
    ),
    ir=RectifierControls(
        tolerance=1e-9,
        threshold=1e-5,
        max_attempts=100,
        projection=ProjectionControls(
            tolerance=2e-12, max_sweeps=20000, acceleration=True
        ),
    ),
    workspace=WORKSPACE,
)


def select_method(problem: SupportProblem, controls: SupportControls) -> Method:
    """``lp`` without blocks when ``n (p + 1) <= workspace``, else ``ir``.

    The product is an exact integer, formed before anything is allocated.
    """
    size = problem.n_rows * (problem.n_slopes + 1)
    return "lp" if problem.n_blocks == 0 and size <= controls.workspace else "ir"


# --------------------------------------------------------------------- the FE step


@dataclass(frozen=True, eq=False)
class FEStep:
    """The levels without a positive ``Z``, their rows and their indicator witness.

    ``levels`` holds per block the codes of those levels; ``rows`` the current rows of
    any of them; ``witness`` the fresh check of the summed indicators (``None`` when
    there is no such level); ``exact`` whether the witness's strict set is exactly
    ``rows``, all of them zero rows.
    """

    levels: tuple[IndexArray, ...]
    rows: IndexArray
    witness: PrimalWitness | None
    exact: bool


def fe_step(problem: SupportProblem, *, tolerance: float, threshold: float) -> FEStep:
    """The FE indicator proposal of every block, checked afresh with ``e`` and ``d``.

    ``ValueError`` without blocks.
    """
    if problem.grouping is None:
        raise ValueError("the FE step needs at least one block")
    positive = problem.z > 0
    levels, effects, rows = [], [], np.zeros(problem.n_rows, dtype=bool)
    for block in problem.grouping.blocks:
        counts = np.bincount(block.codes[positive], minlength=block.n_levels)
        empty = counts == 0
        levels.append(indices(np.flatnonzero(empty)))
        effects.append(empty.astype(np.float64))
        rows |= empty[block.codes]
    found = indices(np.flatnonzero(rows))
    if not found.size:
        return FEStep(tuple(levels), NONE, None, exact=True)
    witness = primal_witness(
        problem,
        np.zeros(problem.n_slopes),
        effects,
        method="fe",
        tolerance=tolerance,
        threshold=threshold,
    )
    exact = bool(np.array_equal(witness.strict, found) and (~positive[found]).all())
    return FEStep(tuple(levels), found, witness, exact)


class _Components(Protocol):
    """SciPy's component labels for a directed level graph."""

    def connected_components(
        self, graph: SparseMatrix, /, *, directed: bool, connection: str
    ) -> tuple[int, IndexArray]: ...


def _graph_direction(
    first: Block, second: Block, *, positive: IndexArray
) -> tuple[FloatArray, FloatArray] | None:
    """Construct FE coefficients whose strict rows cross directed components."""
    left, right = first.codes, second.codes + first.n_levels
    source = np.concatenate((left, right[positive]))
    target = np.concatenate((right, left[positive]))
    size = first.n_levels + second.n_levels
    sparse = cast("Sparse", importlib.import_module("scipy.sparse"))
    components = cast("_Components", importlib.import_module("scipy.sparse.csgraph"))
    graph = sparse.csr_array(
        (np.ones(source.size), (source, target)), shape=(size, size)
    )
    count, labels = components.connected_components(
        graph, directed=True, connection="strong"
    )
    left_component, right_component = labels[left], labels[right]
    crossing = left_component != right_component
    if not crossing.any():
        return None

    # Positive rows have both arcs and hence stay within a component. On the
    # component DAG, every arc goes from a smaller to a larger longest-path label.
    parents, children = left_component[crossing], right_component[crossing]
    successors: list[list[int]] = [[] for _ in range(count)]
    for parent, child in zip(parents.tolist(), children.tolist(), strict=True):
        successors[parent].append(child)
    pending = np.bincount(children, minlength=count).tolist()
    ready = [node for node, degree in enumerate(pending) if degree == 0]
    potential = np.zeros(count)
    while ready:
        parent = ready.pop()
        for child in successors[parent]:
            potential[child] = max(potential[child], potential[parent] + 1)
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return -potential[labels[: first.n_levels]], potential[labels[first.n_levels :]]


def _two_block_fe_witness(
    problem: SupportProblem, controls: RectifierControls
) -> PrimalWitness | None:
    """Propose and check a zero-slope direction from the two-block PPML graph."""
    grouping, variance = problem.grouping, problem.variance
    if (
        grouping is None
        or problem.n_blocks != 2
        or not problem.n_slopes
        or not isinstance(variance, PowerVariance)
        or variance.tau != 1.0
        or problem.omega != 1.0
    ):
        return None
    try:
        effects = _graph_direction(*grouping.blocks, positive=problem.positive)
    except (MemoryError, OverflowError):
        # The existing rectifier can run without the extra graph workspace.
        return None
    if effects is None:
        return None
    witness = primal_witness(
        problem,
        np.zeros(problem.n_slopes),
        effects,
        method="fe",
        tolerance=controls.tolerance,
        threshold=controls.threshold,
    )
    return witness if witness.valid and witness.strict.size else None


# ---------------------------------------------------------------------- the result


@dataclass(frozen=True, eq=False)
class SupportResult:
    """One invocation's evidence, work and stop on its bound current sample.

    Attributes
    ----------
    problem : SupportProblem
        The current sample the result is bound to.
    controls : SupportControls
        The resolved controls, recorded.
    method : {'lp', 'ir'}
        The method selected before running.
    fe : FEStep or None
        The FE step, when there are blocks and zero rows.
    lp, ir : LPOutcome or RectifierOutcome or None
        The selected method's outcome, with its charged work.
    witnesses : tuple of PrimalWitness
        The valid witnesses whose strict sets are the separated set ``S``.
    dual : DualWitness or None
        The last dual check, passing or not.
    completion : {'', 'covered', 'dual', 'overlap_lp'}
        What established completeness, if anything.
    reason : str
        Why the invocation stopped.
    """

    problem: SupportProblem
    controls: SupportControls
    method: Method
    fe: FEStep | None
    lp: LPOutcome | None
    ir: RectifierOutcome | None
    witnesses: tuple[PrimalWitness, ...]
    dual: DualWitness | None
    completion: str
    reason: str

    def __post_init__(self) -> None:
        """Check, afresh, that the bound evidence supports what the record states."""
        if self.completion not in COMPLETIONS:
            raise ValueError(f"completion must be one of {COMPLETIONS}")
        if self.completion in ("dual", "overlap_lp") and (
            self.dual is None or not self.dual.passed
        ):
            raise ValueError("a dual completion needs a passing dual witness")
        active, methods = self._active()
        e, d = active.tolerance, active.threshold
        for witness in self.witnesses:
            require_primal(
                witness, self.problem, methods=methods, tolerance=e, threshold=d
            )
        if self.dual is not None:
            require_dual(
                self.dual, self.problem, self.separated, tolerance=e, threshold=d
            )
        uncovered = np.setdiff1d(self.problem.zero, self.separated)
        if self.completion == "covered" and uncovered.size:
            raise ValueError("a covered completion needs every zero row witnessed")
        ir = self.ir
        overlap = None if ir is None else ir.overlap_lp
        if (self.completion == "overlap_lp") != (overlap is not None):
            raise ValueError("an overlap LP completion needs its auxiliary work")
        if overlap is not None and (
            self.method != "ir"
            or self.reason != "overlap_lp"
            or ir is None
            or ir.completion
            or ir.witnesses
            or self.fe is None
            or (self.fe.witness is not None and self.fe.witness.valid)
            or self.problem.grouping is None
            or self.witnesses
            or self.dual is None
            or not uncovered.size
            or (self.dual.raw[uncovered] < 1.0).any()
        ):
            raise ValueError("an overlap LP completion needs an unwitnessed FE sample")

    def _active(self) -> tuple[LPControls | RectifierControls, tuple[str, ...]]:
        """The active controls of the method and the steps that propose its witnesses.

        ``ValueError`` when the other method's outcome is present, or when the method's
        outcome or the FE step's witness was obtained under other controls.
        """
        controls, fe = self.controls, self.fe
        found: tuple[LPControls | RectifierControls, tuple[str, ...]]
        if self.method == "lp":
            found = (controls.lp, ("lp",))
            other = self.ir is not None or fe is not None
            outcome = self.lp is not None and self.lp.controls != controls.lp
        else:
            found = (controls.ir, ("fe", "ir"))
            other = self.lp is not None
            outcome = self.ir is not None and self.ir.controls != controls.ir
        if other or outcome:
            raise ValueError("the outcome is another method's or under other controls")
        step = None if fe is None else fe.witness
        if step is not None:
            require_same(step.problem, self.problem)
            if (step.method, step.tolerance, step.threshold) != (
                "fe",
                controls.ir.tolerance,
                controls.ir.threshold,
            ):
                raise ValueError(CONTROLS)
        return found

    @property
    def separated(self) -> IndexArray:
        """``S``: the current rows witnessed as separated, increasing."""
        return union(self.witnesses)

    @property
    def positions(self) -> IndexArray:
        """The original positions of ``S``."""
        return indices(self.problem.positions[self.separated])

    @property
    def status(self) -> Status:
        """``complete``, ``partial`` or ``inconclusive``, from the evidence."""
        if self.completion:
            return "complete"
        return "partial" if self.separated.size else "inconclusive"


def require_current(result: SupportResult, problem: SupportProblem) -> None:
    """``ValueError`` unless ``result`` was computed on ``problem``: no stale reuse."""
    require_same(result.problem, problem)


def analyze_support(
    problem: SupportProblem, controls: SupportControls, *, allow_overlap: bool = False
) -> SupportResult:
    """One fresh support analysis of the current sample under explicit controls.

    Where the iterative rectifier ends without a completion and no row is witnessed,
    by it or by the fixed-effect step, it runs once more with the slopes that the
    effects and the earlier slopes absorb set aside from the rank decision of its
    projections (owner's ruling 108; :mod:`powermean.solve.support_projection`,
    "Slopes").  The second attempt stands only if it concludes or witnesses a row;
    otherwise the first one is reported.  An analysis that the first attempt
    concludes, or in which a row is witnessed, is that attempt alone: the slopes are
    never set aside in it.

    Parameters
    ----------
    problem : SupportProblem
        The current sample; read, never modified.
    controls : SupportControls
        :data:`BASELINE`, :data:`TIGHTER` or an explicit record.
    allow_overlap : bool, default False
        Enable the sample composer's auxiliary certificate after both attempts fail.

    Returns
    -------
    SupportResult
        The selected method, its witnesses, dual check, work and stop.
    """
    method = select_method(problem, controls)
    found = (problem, controls, method)
    if not problem.zero.size:
        return SupportResult(
            *found, None, None, None, (), None, "covered", "no_zero_outcomes"
        )
    if method == "lp":
        lp = separation_lp(problem, controls.lp)
        dual = None if lp.dual is None else lp.dual.witness
        return SupportResult(
            *found, None, lp, None, lp.witnesses, dual, lp.completion, lp.reason
        )
    fe = None
    if problem.n_blocks:
        fe = fe_step(
            problem, tolerance=controls.ir.tolerance, threshold=controls.ir.threshold
        )
    initial: tuple[PrimalWitness, ...] = ()
    if fe is not None and fe.witness is not None and fe.witness.valid:
        initial = (fe.witness,)
    if fe is not None and fe.witness is None:
        graph = _two_block_fe_witness(problem, controls.ir)
        if graph is not None:
            return SupportResult(*found, fe, None, None, (graph,), None, "", "fe_graph")
    ir: RectifierOutcome = iterative_rectifier(problem, controls.ir, initial)
    if not (ir.completion or initial or ir.witnesses):
        # Inconclusive.  A slope that the effects, or the effects and earlier slopes,
        # absorb can be the cause: its absorption noise counts as rank.  One second
        # attempt with such slopes set aside; it stands only if it concludes.
        again = iterative_rectifier(problem, controls.ir, initial, set_aside=True)
        if again.completion or again.witnesses:
            ir = again
        elif (
            allow_overlap
            and fe is not None
            # A short IR investigation keeps its original stopping behavior.
            and controls.ir.max_attempts >= BASELINE.ir.max_attempts
            and controls.ir.projection.max_sweeps >= BASELINE.ir.projection.max_sweeps
        ):
            try:
                certified = certify_overlap(
                    problem,
                    controls.lp,
                    workspace=controls.workspace,
                    tolerance=controls.ir.tolerance,
                    threshold=controls.ir.threshold,
                )
                if certified is not None:
                    dual, iterations = certified
                    work = OverlapLP(
                        iterations,
                        again.reason,
                        again.charged,
                        sum(attempt.sweeps for attempt in again.attempts),
                    )
                    return SupportResult(
                        *found,
                        fe,
                        None,
                        replace(ir, overlap_lp=work),
                        (),
                        dual,
                        "overlap_lp",
                        "overlap_lp",
                    )
            except (MemoryError, ValueError):
                # Fresh certification and rebinding are part of the optional check.
                pass
    witnesses = (*initial, *ir.witnesses)
    return SupportResult(
        *found, fe, None, ir, witnesses, ir.dual, ir.completion, ir.reason
    )


def analyze_sample_support(
    problem: SupportProblem, controls: SupportControls
) -> SupportResult:
    """Analyze support during sample composition with the authorized certificate."""
    return analyze_support(problem, controls, allow_overlap=True)
