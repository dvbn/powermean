"""Joint separation support: controls, method selection, the FE step and the result.

Existence for joint estimation with ``tau < 2`` and NB2: the fixed-effect step, the
linear-programming step when there are no absorbed effects, the iterative rectifier
otherwise.  An internal diagnostic, not a fit, a sample constructor or a public option.
``powermean`` exports nothing of it.

One invocation.  :func:`analyze_support` is one fresh analysis of the supplied current
sample (:class:`~powermean.solve.support_problem.SupportProblem`) under explicit
controls: nothing is cached, warm-started or reused between invocations, and the
sample is never modified.  The method is selected before anything runs
(:func:`select_method`): without zero rows nothing runs and the empty set is complete;
without blocks and with ``n (p + 1)`` at most the recorded workspace bound (2,000,000,
a loading-element guard used as an internal routing convention, not a measured
crossover) the LP of :mod:`powermean.solve.support_lp`; otherwise the iterative
rectifier of :mod:`powermean.solve.support_rectifier`, preceded with blocks by the FE
step.  A failed or exhausted method never hands over to another.

The FE step.  A level of an estimated block with no positive ``Z`` is separated in
full: the indicator direction is zero on every positive row and positive on its rows.
The step proposes the sum of those indicators over every block, checks it afresh as any
witness, and records whether its strict set is exactly the rows of those levels (the
exact combinatorial support).  Cascades, deletions and empty samples belong to the
sample construction.

The controls, the internal resolved values recorded with every result:

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

:data:`TIGHTER` is the explicitly tighter invocation that an iteration trigger requests:
a new call on the current sample, not a retry.

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

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.model.order import IndexArray
from powermean.solve.support_lp import LPControls, LPOutcome, separation_lp
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
    iterative_rectifier,
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

Method = Literal["lp", "ir"]
Status = Literal["complete", "partial", "inconclusive"]
COMPLETIONS = ("", "covered", "dual")
# The 2,000,000 loading-element guard, the internal LP workspace routing bound.
WORKSPACE = 2_000_000


@dataclass(frozen=True)
class SupportControls:
    """The explicit, recorded controls of one invocation.

    ``lp`` and ``ir`` hold the method controls (the rectifier's include the support
    WLS); ``workspace`` is the routing bound on ``n (p + 1)``.
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
    completion : {'', 'covered', 'dual'}
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
        if self.completion == "dual" and (self.dual is None or not self.dual.passed):
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
    problem: SupportProblem, controls: SupportControls
) -> SupportResult:
    """One fresh support analysis of the current sample under explicit controls.

    Parameters
    ----------
    problem : SupportProblem
        The current sample; read, never modified.
    controls : SupportControls
        :data:`BASELINE`, :data:`TIGHTER` or an explicit record.

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
    ir: RectifierOutcome = iterative_rectifier(problem, controls.ir, initial)
    witnesses = (*initial, *ir.witnesses)
    return SupportResult(
        *found, fe, None, ir, witnesses, ir.dual, ir.completion, ir.reason
    )
