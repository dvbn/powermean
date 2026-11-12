"""The internal joint transformed ``tau = 2`` recession diagnostic: run, result, error.

The ``tau = 2`` criterion is ``q + Z / m``.  Internal module: ``powermean`` exports
nothing of it.

The question.  On the current sample let ``P = {Z > 0}``, ``B = [X, F]`` and ``a`` the
positive observation weights.  The transformed criterion is, up to constants,
``sum a_i (Z_i exp(-q_i) + q_i)``; finite offsets change neither condition below.  A
direction ``h = Bv`` with ``h_P >= 0`` is a ``negative_sum_recession`` when ``sum a_i
h_i < 0`` (the criterion is unbounded below) and a ``zero_sum_recession`` when ``sum
a_i h_i = 0`` and ``h_i > 0`` for some ``i`` in ``P`` (the exponential part strictly
decreases at no linear cost: the unattained boundary).  A flat, null or gauge direction
(``h_P = 0``, zero total) is neither.  ``no_recession`` excludes both; it establishes
no rank, uniqueness, verified fit or standard error.  Nothing is deleted: the
all-zero level rule belongs to the sample construction, and no further
GPML row is removed.  Cancelled, mixed, difference, IV, NB2 and ``tau != 2`` systems
are refused.

The sample is :class:`~powermean.solve.support_gamma_problem.GammaProblem`, defined in
its own module and still provided here under its name.

One invocation (:func:`analyze_recession`) under explicit
:class:`~powermean.solve.support_lp.LPControls` (:data:`BASELINE` or :data:`TIGHTER`, or
another record): no parameter, no program; the all-positive structural certificate
below, no program; the nonzero guard and the range-checked data
(:mod:`powermean.solve.support_gamma_lp`); LP1, stopping on a fresh valid negative
witness; LP2 only after LP1's passing dual and with positive rows (without them the
strict boundary is impossible); at most two charged calls, each one attempt, and no
other method, retry, cache or warm state.  A call is charged before the optimizer runs,
and a call that fails (an unusable report, or the optimizer's ``MemoryError``) is kept
with the work before it: it stops the invocation as ``lp_failure``, never as a kind or
``no_recession``.  The tighter invocation an iteration trigger requests
is a new call on the current sample.

The result (:class:`GammaResult`) binds every report to its controls, admits a report
whose ``failure`` is empty only when it is usable by the rule of the returned reports
(:func:`~powermean.solve.support_gamma_lp.usable`), and binds every
retained witness and dual to its sample, controls, program and report and checks it
again (:mod:`powermean.solve.support_gamma_witness`); its reason must be the one its
evidence determines, and its status follows.  :func:`raise_for_recession` raises the
existing ``ExistenceError`` (``existence.no_finite_root``, status ``no_finite_root``,
the result as certificate, keeping its kind) only after a fresh valid replay of the
strict witness on the current sample; an inconclusive result or a passing dual raises
nothing here.

The all-positive structural certificate.  At ``omega = 1`` with ``PowerVariance(2)``,
when every ``Z_i`` and every
weight ``a_i`` is finite and strictly positive, ``P`` is every row: a candidate ``h``
with ``h_P >= 0`` is nonnegative everywhere, so ``sum a_i h_i >= 0`` excludes the
negative sum, and a zero total forces every ``h_i = 0``, excluding the strict zero-sum
boundary, for any number of blocks, redundant or disconnected included.  It proves
neither rank, uniqueness nor a fitted result.  :func:`all_positive` reads that domain
freshly from the problem's own fields and admits it only when those fields, as they are
now, build a new :class:`GammaProblem` through its own constructor (every rule of the
record: real finite arrays, one distinct original position per row and the grouping's,
positive weights of a declared kind with exact frequency counts, the slope names, the
family and the order) that holds the same sample bit for bit, the
grouping's blocks and the grouping built again through their own constructors first,
so that no retained nested record stands in for its current values, and
whose own owned numeric values, the ``Z`` and the weights the constructor
stored as plain arrays of doubles, are finite and strictly positive (the
reading of the current fields and the byte comparison alone establish nothing, since
a standard masked array hides a masked zero from ``(z > 0).all()`` and fills it in
``tobytes``; a record that reads as the domain but whose fresh values hold a zero is
refused, neither certified nor passed on to the programs on its wrapper); the
invocation then stops with the reason :data:`ALL_POSITIVE` (status ``no_recession``),
no program and no charged call, after the empty-parameter refusal and before the
nonzero guard.  The result's constructor requires that reason, and refuses any program,
on exactly that domain, built by the analysis, directly or by ``dataclasses.replace``;
a sample with a zero, another order, mutated or malformed fields never carries it, and
keeps the two programs, their controls, exact multiplier signs and raw checks unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.data.weights import WEIGHT_KINDS, ObservationWeights
from powermean.errors import ExistenceError
from powermean.fe.groups import Block, Grouping
from powermean.model.variance import PowerVariance
from powermean.solve.support_gamma_lp import (
    LPData,
    LPReport,
    Stage,
    call,
    constraint_nonzeros,
    prepare,
    usable,
    within_limit,
)
from powermean.solve.support_gamma_problem import GammaProblem, _same_bits
from powermean.solve.support_gamma_witness import (
    BINDING,
    GammaDual,
    GammaWitness,
    dual_check,
    primal_witness,
)
from powermean.solve.support_lp import LPControls

# The names this private module provides, the sample record of its former home
# included (the import route ``powermean.solve.support_gamma.GammaProblem`` is kept).
__all__ = [
    "ADMISSION",
    "ALL_POSITIVE",
    "BASELINE",
    "OWNED",
    "STAGES",
    "STATUS",
    "TIGHTER",
    "GammaProblem",
    "GammaResult",
    "Solve",
    "Status",
    "all_positive",
    "analyze_recession",
    "raise_for_recession",
]

Status = Literal[
    "negative_sum_recession", "zero_sum_recession", "no_recession", "inconclusive"
]
STAGES: tuple[Stage, Stage] = ("negative", "zero_sum")
# The reason of the all-positive structural certificate: no program was run.
ALL_POSITIVE = "all_positive"
# The refusal of a record that reads as that domain but does not admit freshly.
ADMISSION = "the all-positive sample must admit freshly as the record it is now"
# Its detail when the freshly admitted record's own values are not the domain.
OWNED = "the fresh record's own values are not finite and strictly positive"
STATUS: dict[str, Status] = {
    "negative_sum_recession": "negative_sum_recession",
    "zero_sum_recession": "zero_sum_recession",
    "no_recession": "no_recession",
    "no_positive_rows": "no_recession",
    "empty_parameters": "no_recession",
    ALL_POSITIVE: "no_recession",
}
BASELINE = LPControls(
    tolerance=1e-12, threshold=1e-8, feasibility=1e-8, max_iterations=1000
)
TIGHTER = LPControls(
    tolerance=1e-13, threshold=1e-9, feasibility=1e-9, max_iterations=1000
)


# -------------------------------------------------------------------- the result


@dataclass(frozen=True, eq=False)
class Solve:
    """One charged program: its report and, when usable, both fresh checks."""

    report: LPReport
    witness: GammaWitness | None = None
    dual: GammaDual | None = None

    @property
    def continues(self) -> bool:
        """Whether LP1's evidence is a passing dual without a strict witness."""
        return (
            self.witness is not None
            and not self.witness.valid
            and self.dual is not None
            and self.dual.passed
        )


def _reason(problem: GammaProblem, solves: tuple[Solve, ...]) -> str:
    """The reason the evidence of the last program determines."""
    last = solves[-1]
    if last.witness is None or last.dual is None:
        return "lp_failure"
    kind = last.witness.kind
    if kind is not None:
        return kind
    if last.dual.passed:
        if last.report.stage == "zero_sum":
            return "no_recession"
        return "resource" if problem.positive.size else "no_positive_rows"
    check = last.witness.check
    if check is not None and check.failure:
        return check.failure
    failures = (*last.witness.failure, *last.dual.failure)
    return "range" if "range" in failures else "unresolved"


@dataclass(frozen=True, eq=False)
class GammaResult:
    """One invocation's charged programs, evidence and stop on its bound sample.

    ``solves`` are the programs run, in order (LP1, then LP2), each one attempt, a
    failed call included with the work before it; ``reason`` is why the invocation
    stopped: a kind, ``no_recession``, ``no_positive_rows``, ``empty_parameters``,
    ``all_positive``, ``range``, ``resource``, ``lp_failure``, ``lp_box``,
    ``lp_constraints``, ``lp_objective`` or ``unresolved``.  The record requires every
    report's controls
    to be its own and every report whose ``failure`` is empty to be usable, checks
    every retained witness and dual again on its sample and requires the reason its
    evidence determines.
    """

    problem: GammaProblem
    controls: LPControls
    solves: tuple[Solve, ...]
    reason: str

    def __post_init__(self) -> None:
        """Check the types, the order of the programs, the evidence and the reason.

        The reason is recomputed from the problem as it is now: the all-positive
        certificate (:data:`ALL_POSITIVE`) is admitted only on the domain
        :func:`all_positive` reads freshly, on the freshly admitted record's own
        values, and only without a program.
        """
        if not isinstance(self.problem, GammaProblem) or not isinstance(
            self.controls, LPControls
        ):
            raise ValueError("a result needs a GammaProblem and LPControls")
        solves = tuple(self.solves)
        object.__setattr__(self, "solves", solves)
        if len(solves) > len(STAGES):
            raise ValueError("an invocation runs at most two programs")
        for stage, solve in zip(STAGES, solves, strict=False):
            self._bind(stage, solve)
        if len(solves) == 2 and not (
            solves[0].continues and self.problem.positive.size
        ):
            raise ValueError("LP2 runs only after LP1's passing dual on positive rows")
        if self.reason != self._expected():
            raise ValueError("the reason is not the one the evidence determines")

    def _bind(self, stage: Stage, solve: Solve) -> None:
        """Check one program's report, witness and dual on the bound sample.

        A report whose ``failure`` is empty is evidence only when it is usable
        (:func:`~powermean.solve.support_gamma_lp.usable`), checked before anything is
        replayed; a failed report, whose fields may be unavailable, carries no check.
        """
        report, witness, dual = solve.report, solve.witness, solve.dual
        if report.stage != stage:
            raise ValueError("the programs must be LP1, then LP2")
        if report.controls != self.controls:
            raise ValueError(BINDING)
        failed = bool(report.failure) or not usable(report)
        if failed or witness is None or dual is None:
            if not report.failure or witness is not None or dual is not None:
                raise ValueError("a usable report needs both checks, a failed one none")
            return
        point = report.point
        for record in (witness, dual):
            record.replay(self.problem, controls=self.controls, stage=stage)
            if point is None or not _same_bits(record.boxed, point):
                raise ValueError("the evidence is not the report's point")
            if record.reported != report.reported:
                raise ValueError("the evidence is not the report's objective")
        found, given = dual.marginals, report.marginals or ()
        if len(given) != 4 or not all(map(_same_bits, found, given)):
            raise ValueError("the dual is not the report's marginals")

    def _expected(self) -> str | None:
        """The reason the evidence, or a refusal before any program, determines.

        ``None`` when no reason fits: programs after a refusal, or none without one;
        ``ValueError`` on a malformed record reading as the all-positive domain.
        """
        problem, solves = self.problem, self.solves
        if not problem.n_parameters:
            return None if solves else "empty_parameters"
        if _certified(problem):
            return None if solves else ALL_POSITIVE
        nonzeros = constraint_nonzeros(problem)
        if not within_limit(nonzeros):
            return None if solves else "resource"
        data = _data(problem)
        if data is None:
            return None if solves else "range"
        if not solves:
            return None
        reason = _reason(problem, solves)
        stacked = nonzeros + int(np.count_nonzero(data.total))
        return None if reason == "resource" and within_limit(stacked) else reason

    @property
    def status(self) -> Status:
        """The kind, ``no_recession`` or ``inconclusive``, from the checked reason."""
        return STATUS.get(self.reason, "inconclusive")

    @property
    def witness(self) -> GammaWitness | None:
        """The fresh valid strict witness, when the invocation found one."""
        last = self.solves[-1].witness if self.solves else None
        return last if last is not None and last.valid else None

    @property
    def attempts(self) -> int:
        """The charged calls: one per program run."""
        return len(self.solves)

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        """Rows, rows of ``P``, parameters ``k`` and the nonzeros counted for ``A_P``.

        The counts the resource guard compares before any matrix exists; the matrices
        actually passed are each report's ``rows`` and ``nonzeros``.
        """
        problem = self.problem
        found = (problem.n_rows, problem.positive.size, problem.n_parameters)
        return (*found, constraint_nonzeros(problem))


def _data(problem: GammaProblem) -> LPData | None:
    """The data of the programs, ``None`` when a quantity is not representable."""
    try:
        return prepare(problem)
    except FloatingPointError:
        return None


def _screened(problem: GammaProblem) -> bool:
    """Whether the record's own fields now read as the all-positive domain.

    The cheap reading, raising nothing: ``PowerVariance(2)`` and ``omega = 1`` exactly;
    a weight record of a declared kind; arrays of one shape, one finite, strictly
    positive ``Z`` and weight per row, a finite design and offset on those rows, one
    position per row and a grouping on them when there is one.
    """
    variance, omega, weights = problem.variance, problem.omega, problem.weights
    if type(variance) is not PowerVariance or variance.tau != 2.0:
        return False
    if type(omega) is not float or omega != 1.0:
        return False
    if not isinstance(weights, ObservationWeights) or weights.kind not in WEIGHT_KINDS:
        return False
    z, a, design = problem.z, weights.values, problem.design
    offset, positions = problem.offset, problem.positions
    if any(not isinstance(v, np.ndarray) for v in (z, a, design, offset, positions)):
        return False
    n = z.shape[0] if z.ndim == 1 else 0
    if not n or a.shape != (n,) or offset.shape != (n,) or positions.shape != (n,):
        return False
    if design.ndim != 2 or design.shape[0] != n:
        return False
    grouping = problem.grouping
    if grouping is not None and grouping.n_rows != n:
        return False
    finite = np.isfinite(z).all() and np.isfinite(a).all()
    finite = finite and np.isfinite(design).all() and np.isfinite(offset).all()
    return bool(finite and (z > 0).all() and (a > 0).all())


def _fresh_grouping(grouping: Grouping | None) -> Grouping | None:
    """The current grouping built again, its blocks first, through their constructors.

    A nested record is admitted by the same rule as the outer one: each
    :class:`~powermean.fe.groups.Block` is constructed from its current name, codes,
    counts and levels (codes numbering distinct levels by first occurrence, counts the
    rows of each level), then the :class:`~powermean.fe.groups.Grouping` from those
    fresh blocks and its current positions (unique names, aligned rows, distinct
    nonnegative positions).  The constructors raise their own ``ValueError``; a value
    that is not a record of its type is passed on for the outer constructor to refuse.
    Nothing of the retained grouping, its blocks or their arrays is reused: the fresh
    grouping aliases none of them.
    """
    if grouping is None or not isinstance(grouping, Grouping):
        return grouping
    blocks = tuple(
        Block(block.name, block.codes, block.counts, block._levels)
        if isinstance(block, Block)
        else block
        for block in grouping.blocks
    )
    return Grouping(blocks, grouping.positions)


def _admitted(problem: GammaProblem) -> GammaProblem:
    """The current field values built into the same record again; ``ValueError``.

    The record's own constructors are the admission: they check every rule again on
    the values as they are now (real finite arrays, one distinct nonnegative original
    position per row and the grouping's, positive weights of a declared kind with exact
    frequency counts, the slope names, the family and the order; and, nested, every
    block's codes, counts and levels and the grouping's names, alignment and positions,
    :func:`_fresh_grouping`) and own copies of them, which must hold this sample bit
    for bit, grouping included.  No prior construction, identity, hash, trusted marker
    or read-only flag stands in for it, at any depth of the record.  The fresh record
    is returned: the domain is then read on its own values (:func:`_positive`).
    """
    try:
        fresh = GammaProblem(
            problem.z,
            problem.design,
            problem.columns,
            _fresh_grouping(problem.grouping),
            problem.positions,
            problem.weights,
            problem.offset,
            problem.variance,
            problem.omega,
        )
    except (ValueError, TypeError, FloatingPointError) as refusal:
        raise ValueError(f"{ADMISSION}: {refusal}") from refusal
    if not fresh.same(problem):
        raise ValueError(f"{ADMISSION}: the fresh record holds another sample")
    return fresh


def _positive(fresh: GammaProblem) -> bool:
    """Whether the freshly admitted record's own values are the all-positive domain.

    The premise of the certificate, read on the arrays the constructor stored and
    nothing else: ``Z`` and the weights as plain owned arrays of doubles, every value
    finite and strictly positive.  A wrapper of the current record is no evidence: a
    standard masked array answers ``(z > 0).all()`` without its masked zero and fills
    that zero in ``tobytes``, so the preliminary reading of :func:`_screened` and the
    byte comparison of :func:`_admitted` hold while the values do not.  The
    two arrays alone carry the sign argument; the design, the offset, the
    positions and the grouping are the constructors' own checks on their plain values.
    """
    z, a = fresh.z, fresh.weights.values
    if type(z) is not np.ndarray or type(a) is not np.ndarray:
        return False
    if z.dtype != np.float64 or a.dtype != np.float64:
        return False
    finite = np.isfinite(z).all() and np.isfinite(a).all()
    return bool(finite and (z > 0).all() and (a > 0).all())


def _certified(problem: GammaProblem) -> bool:
    """Whether the structural certificate holds on ``problem`` now; the admission.

    ``False`` outside the domain of :func:`_screened`, where the two programs apply;
    ``True`` on it once :func:`_admitted` passes and the fresh record's own values are
    the domain (:func:`_positive`); ``ValueError`` on a record that reads as the domain
    but is malformed, or whose fresh values are not finite and strictly positive
    (:data:`OWNED`): it gets neither the certificate nor a program on its wrapper.
    """
    if not _screened(problem):
        return False
    fresh = _admitted(problem)
    if not _positive(fresh):
        raise ValueError(f"{ADMISSION}: {OWNED}")
    return True


def all_positive(problem: GammaProblem) -> bool:
    """Whether the all-positive structural certificate holds on ``problem`` now.

    Read freshly from the record's own fields, never from a prior result, an identity,
    a hash, a trusted marker or a read-only flag: the domain of :func:`_screened` (the
    family, the order, the declared kind, finite strictly positive ``Z`` and weights on
    aligned rows) on a record whose current values build a new :class:`GammaProblem`
    through its own constructor, holding the same sample bit for bit,
    its blocks and grouping built again first, and whose
    own owned values, as that fresh record stores them, are finite and strictly
    positive.  Any zero, other order, nonfinite, misaligned or
    malformed value (a complex or nonfinite array, a repeated or negative original
    position, a position that is not the grouping's, a frequency weight that is not an
    exact count, an undeclared kind, a wrong slope name, a block whose codes, counts or
    levels break its own rules, a grouping whose blocks or positions break its own, a
    wrapper whose reductions hide a zero of its values) is outside the domain:
    ``False``, where the analysis and the result's constructor run the programs or
    refuse the malformed record.
    """
    try:
        return _certified(problem)
    except ValueError:
        return False


# ------------------------------------------------------------------ the analysis


def _solve(
    problem: GammaProblem, data: LPData, stage: Stage, controls: LPControls
) -> Solve:
    """One charged call of ``stage`` and, when its report is usable, both checks."""
    report = call(problem, data, stage, controls)
    point, marginals = (report.point, report.reported), report.marginals
    if report.failure or point[0] is None or point[1] is None or marginals is None:
        return Solve(report)
    found = (point[0], point[1])
    witness = primal_witness(problem, controls, stage, *found)
    return Solve(
        report, witness, dual_check(problem, controls, stage, found, marginals)
    )


def analyze_recession(problem: GammaProblem, controls: LPControls) -> GammaResult:
    """One fresh recession analysis of the current sample under explicit controls.

    Parameters
    ----------
    problem : GammaProblem
        The current sample; read, never modified.
    controls : LPControls
        :data:`BASELINE`, :data:`TIGHTER` or another explicit record.

    Returns
    -------
    GammaResult
        The charged programs, their evidence and the stop; on the all-positive domain
        of :func:`all_positive`, the structural certificate with no program.

    Raises
    ------
    ValueError
        Another record than a ``GammaProblem`` with ``LPControls``, or a record that
        reads as the all-positive domain but does not admit freshly (:data:`ADMISSION`):
        no program is charged on it.
    """
    if not isinstance(problem, GammaProblem) or not isinstance(controls, LPControls):
        raise ValueError("the analysis needs a GammaProblem and LPControls")
    found = (problem, controls)
    if not problem.n_parameters:
        return GammaResult(*found, (), "empty_parameters")
    if _certified(problem):
        return GammaResult(*found, (), ALL_POSITIVE)
    nonzeros = constraint_nonzeros(problem)
    if not within_limit(nonzeros):
        return GammaResult(*found, (), "resource")
    data = _data(problem)
    if data is None:
        return GammaResult(*found, (), "range")
    first = _solve(problem, data, "negative", controls)
    if not first.continues or not problem.positive.size:
        return GammaResult(*found, (first,), _reason(problem, (first,)))
    if not within_limit(nonzeros + int(np.count_nonzero(data.total))):
        return GammaResult(*found, (first,), "resource")
    solves = (first, _solve(problem, data, "zero_sum", controls))
    return GammaResult(*found, solves, _reason(problem, solves))


def raise_for_recession(result: GammaResult, problem: GammaProblem) -> None:
    """``ExistenceError`` when ``result`` has a strict witness replaying on ``problem``.

    ``ValueError`` when the result is bound to another sample or its witness fails its
    fresh replay; nothing is raised for ``no_recession`` or ``inconclusive``.
    """
    result.problem.require_same(problem)
    witness = result.witness
    if witness is None:
        return
    witness.require(problem, controls=result.controls, stage=witness.stage)
    raise ExistenceError(
        "existence.no_finite_root", status="no_finite_root", certificate=result
    )
