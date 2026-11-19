"""The ordered sample composition of steps 1 to 9, coordinated, private (task M1.10).

``design/02_computation.md``, Sections 2 to 4 and 8.4; ``DESIGN.md``, Section 6.1;
decisions D14, D41, D51 and D68 to D72; the reviewed outline
``plan/evidence/M1.10_contract.txt``, its reviews, the terminal-interface readiness, the
adoption record and correction A (``plan/evidence/M1.10_revision_A.txt``).  The
accepted parts are called unchanged; no public name, option, default, message,
tolerance or route is added (the public preparation still refuses fixed effects,
decision D41).  The request, invocation and round records and the problems of the
current rows are :mod:`powermean.interface.sample_records`, whose names this module
still provides.

Order (outline Section 2).  Steps 3 and 4 (:func:`powermean.data.retained.
select_order`) with their refusals, and the exact weights of their rows validated once,
so that no later exclusion hides an outcome or weight error; the blocks encoded on those
rows.  Then rounds: steps 5 and 6 to their fixed point
(:func:`powermean.data.sample_fe.close`); the accepted transform of the current rows
only; for ``omega != 0`` with the power family on ``[1, 2)`` or NB2, one M1.7
invocation with the explicit requested controls (``BASELINE``, or a trigger's
``TIGHTER``).  An admitted witnessed union ``S`` removes exactly its rows, partial or
complete, and a new round starts; a complete invocation with ``S`` empty ends the
rounds; any other ends the composition unresolved, nothing removed.  At ``tau = 2`` one
M1.8 invocation follows the closure: a recession is the specified existence failure,
never a deletion; ``omega = 0`` has no cone.  For zero-containing GPML samples, the
identified-coordinate repair runs the same M1.9 invocation first, constructs the
active sample and its exact FE basis once, and supplies those coordinates to M1.8.
Every original column remains in M1.8's fresh primal and dual checks. A failed
existence check never publishes that provisional sample. Other samples keep M1.9
after M1.8. Every removal lowers the physical rows, which bounds the rounds; no
diagnostic is retried.

One walk, two evidences (correction A).  :func:`_walk` is that order, deterministic: it
forms each diagnostic's problem from the current rows and takes the invocation from an
evidence, either fresh (:func:`prepare_sample`: each accepted analysis called once,
its result admitted by the invocation record) or an outcome's own record, served in
order and required to be on the walk's own problem with the request's controls.
:class:`SampleOutcome`'s ordinary constructor admits it only if the walk over its
own invocations, from its prepared frame, request and origin, reproduces its status,
final ledger, closures, invocations and sample; nothing is analyzed again.  So no
foreign frame, stale history, relabelled failure, edited ledger or other sample is
published or reused.  The private owned producer publishes the exact fields of its
fresh admitted walk.  Correction B compares ledgers by every group of step 2's
account, and samples by their source blocks too, with original level labels, so that
another attribution of a missing row or another level map is not this outcome's.
Correction C keeps every support invocation's evidence in owned copies
(:mod:`powermean.interface.sample_evidence`), and derives the blocks of every round
from the walk's source, the owned labels of the prepared effects on the rows of steps
3 and 4, so that labels whose equality is identity compose through that relation while
a source of other labels or another frame is still refused.  The parent composer's
private routes (``_start``, ``_current``, ``_support``,
``_recession``, ``_collinearity``, ``_rounds``, ``_compose`` and their records and
aliases) are provided with their signatures, each the current implementation or a
delegation to it.

Outcomes.  ``complete`` publishes the sample; ``empty`` (no row before steps 5 to 7),
``not_identified`` (steps 5 to 7 removed every row: each remaining observation was
separated or alone in a level, so that no slope is identified, the outcome that
``ppmlhdfe`` reports as insufficient observations, error 2001),
``separation_unresolved``,
``no_finite_root``, ``recession_unresolved``, ``collinearity_refused`` (error mode:
the complete proposed reasons, none applied, a private policy refusal, not
``identification.rank_deficient``) and ``collinearity_unresolved`` keep the honest
intermediate ledger and every invocation's own problem, controls, result and work, and
publish no sample.  A trigger (:func:`reprepare_sample`) starts from a completed sample
of the same prepared frame and request, with new explicit controls only, and returns a
new outcome; the previous one is never changed.

Recording (task M1.11; the M1.11 sample bridge and partial trace).
:func:`reprepare_recorded` is that trigger with a
:class:`~powermean.interface.sample_records.SampleRecorder`: one route
(:func:`_triggered`, :func:`_composed`, :func:`_walk`, :func:`_steps`, :class:`_Fresh`)
takes an optional recorder, which the call binds when it enters and hands each stage
entered, each closure, each analysis as returned and as admitted, and the admitted
outcome; :func:`reprepare_sample` and :func:`_compose` keep their signatures and pass
none, and an outcome's admission replays its walk without one.
:func:`admitted_outcome` (correction A) is this coordinator's admission of a recorded
call's outcome for a solver that holds the record through the lower views of
:mod:`powermean.solve.profiled_trigger`: the call's origin, its derived request and the
outcome's recorded walk, admitted as an outcome's construction admits them, nothing
analyzed or recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean.data.collinearity import (
    COMPLETE,
    REFUSED,
    CollinearityProblem,
    DiagnosticControls,
    analyze_collinearity,
)
from powermean.data.retained import OrderRows, OrderSample
from powermean.data.sample_fe import (
    SEPARATED,
    Closure,
    FESample,
    RowLedger,
    _fixed_original,
    close,
    final_sample,
)
from powermean.data.sample_fe_source import _owned_source_original
from powermean.data.sample_fe_state import (
    _final_owned_sample,
    same_bits,
    same_data,
    same_missing,
)
from powermean.fe.certificate_basis import CertificateBasis, _prepare_fit_basis
from powermean.interface.sample_records import (
    BINDING,
    UNBOUND,
    VARIANCE,
    CollinearityInvocation,
    Current,
    Evidence,
    Family,
    RecessionInvocation,
    Recorded,
    RecordedTrace,
    Round,
    Rows,
    SampleRecorder,
    SampleRequest,
    Start,
    SupportInvocation,
    _cone,
    _numeric_start,
    _rows_of,
    collinearity_problem,
    current,
    gamma_problem,
    same_grouping,
    source_of,
    start,
    support_problem,
    witnessed,
)
from powermean.solve.profiled_trigger import TraceView
from powermean.solve.support import SupportControls
from powermean.solve.support import analyze_sample_support as analyze_support
from powermean.solve.support_gamma import GammaProblem, analyze_recession
from powermean.solve.support_lp import LPControls
from powermean.solve.support_problem import SupportProblem

# The names of this module and of the records that correction A moved from it, with the
# parent's private routes (correction B).
__all__ = [
    "BINDING",
    "STATUSES",
    "UNBOUND",
    "VARIANCE",
    "CollinearityInvocation",
    "Family",
    "RecessionInvocation",
    "Round",
    "Rows",
    "SampleOutcome",
    "SampleRequest",
    "SupportInvocation",
    "_Current",
    "_Start",
    "_collinearity",
    "_compose",
    "_cone",
    "_current",
    "_recession",
    "_rounds",
    "_rows_of",
    "_start",
    "_support",
    "prepare_sample",
    "reprepare_sample",
]
STATUSES = (
    "complete",
    "empty",
    "not_identified",
    "separation_unresolved",
    "no_finite_root",
    "recession_unresolved",
    "collinearity_refused",
    "collinearity_unresolved",
)


# The reasons of steps 5 to 7: a sample that they leave without a row is
# ``not_identified``; ``empty`` is a sample without a row before any of them.
_FE_REMOVALS = ("singleton", "separated_by_fixed_effect", "separated")


class _Trace(NamedTuple):
    """What a walk established: status, final ledger, rounds, invocations, sample."""

    status: str
    ledger: RowLedger
    rounds: tuple[Round, ...]
    recession: RecessionInvocation | None
    collinearity: CollinearityInvocation | None
    sample: FESample | None


def _entered(recorder: SampleRecorder | None, stage: str) -> None:
    """Record that the walk enters ``stage``, when a recorder is given (M1.11)."""
    if recorder is not None:
        recorder.entry(stage)


def _steps(
    prepared: PreparedFrame,
    request: SampleRequest,
    first: Start,
    ledger: RowLedger,
    evidence: Evidence,
    *,
    recorder: SampleRecorder | None = None,
    owned_order: OrderSample | None = None,
) -> tuple[str, RowLedger, list[Round], Current | None]:
    """The rounds of steps 5 to 7 to their fixed point, or the status that ends them.

    The blocks of every round derive from the walk's source (correction C).
    """
    rounds: list[Round] = []
    controls = request.support
    source = source_of(first)
    while True:
        _entered(recorder, "closure")
        closure = close(
            first.fixed, ledger, first.outcome, keep_singletons=request.keep_singletons
        )
        if recorder is not None:
            recorder.closure(closure)
        ledger = closure.ledger
        if not ledger.count:
            rounds.append(Round(closure, None))
            counts = ledger.counts
            removed = sum(counts[name] for name in _FE_REMOVALS)
            return ("not_identified" if removed else "empty"), ledger, rounds, None
        _entered(recorder, "current")
        if owned_order is not None and np.array_equal(
            ledger.positions, owned_order.positions
        ):
            now = Current(
                ledger,
                owned_order.data,
                owned_order.problem,
                None if first.fixed is None else first.fixed.grouping,
            )
        else:
            now = current(prepared, request.omega, ledger, source)
        support = None
        if controls is not None:
            _entered(recorder, "support")
            support = evidence.support(support_problem(now, request), controls)
        rounds.append(Round(closure, support))
        if support is None or (
            support.result.status == "complete" and not support.removed.size
        ):
            return "", ledger, rounds, now
        if not support.removed.size:
            return "separation_unresolved", ledger, rounds, now
        ledger = ledger.exclude(support.removed, SEPARATED)


def _finished_sample(
    prepared: PreparedFrame,
    first: Start,
    now: Current,
    collinearity: CollinearityInvocation,
    owned_order: OrderSample | None,
) -> FESample:
    """The existing step 9, once; a pre-LP sample is not published on refusal."""
    analysis, ledger = collinearity.analysis, now.ledger
    assert analysis.reasons is not None
    if owned_order is None:
        return final_sample(
            first.rows,
            ledger,
            base=prepared.base,
            data=now.data,
            problem=now.problem,
            source=None if first.fixed is None else first.fixed.grouping,
            grouping=now.grouping,
            reasons=analysis.reasons,
        )
    return _final_owned_sample(
        first.rows,
        ledger,
        base=prepared.base,
        data=now.data,
        problem=now.problem,
        groups=(
            None if first.fixed is None else first.fixed.grouping,
            now.grouping,
        ),
        reasons=analysis.reasons,
        prepared_design=(owned_order.design if now.data is owned_order.data else None),
    )


_EXISTENCE_BASIS = "_existence_basis"
_EXISTENCE_BASIS_PRODUCER = _prepare_fit_basis


def _basis_for_recession(
    sample: FESample,
    native: ModuleType | None,
) -> CertificateBasis | None:
    """The same current-sample FE construction later consumed by fitting."""
    if sample.grouping is None:
        return None
    basis = (
        _prepare_fit_basis(sample, _native=native, _retain_connectivity=True)
        if _prepare_fit_basis is _EXISTENCE_BASIS_PRODUCER
        else _prepare_fit_basis(sample)
    )
    object.__setattr__(sample, _EXISTENCE_BASIS, (basis, native))
    return basis


def _walk(
    prepared: PreparedFrame,
    request: SampleRequest,
    ledger: RowLedger | None,
    evidence: Evidence,
    *,
    recorder: SampleRecorder | None = None,
    start_state: Start | None = None,
    owned_order: OrderSample | None = None,
    basis_native: ModuleType | None = None,
) -> _Trace:
    """Current sample; zero-containing GPML reuses identification before its LPs."""
    first = start(prepared, request) if start_state is None else start_state
    initial = first.ledger if ledger is None else ledger
    status, ledger, found, now = _steps(
        prepared,
        request,
        first,
        initial,
        evidence,
        recorder=recorder,
        owned_order=owned_order,
    )
    known = (ledger, tuple(found))
    if now is None or status:
        return _Trace(status, *known, None, None, None)
    controls = request.recession
    recession = collinearity = None
    sample = None
    basis = None
    # Routing only. D79 still freshly admits and checks its all-positive domain.
    identified = controls is not None and bool(np.any(now.problem.response == 0))
    if identified:
        _entered(recorder, "collinearity")
        collinearity = evidence.collinearity(
            collinearity_problem(now),
            request.collinearity,
            request.mode,
        )
        analysis = collinearity.analysis
        if analysis.status != COMPLETE:
            status = (
                "collinearity_refused"
                if analysis.status == REFUSED
                else "collinearity_unresolved"
            )
            return _Trace(status, *known, None, collinearity, None)
        _entered(recorder, "sample")
        sample = _finished_sample(prepared, first, now, collinearity, owned_order)
        # Step 9 has admitted the canonical grouping used by this sample's basis.
        now = replace(now, grouping=sample.grouping)
        basis = (
            _basis_for_recession(sample, basis_native)
            if type(evidence) is _Fresh
            else None
        )
    if controls is not None:
        _entered(recorder, "recession")
        if identified and type(evidence) is _Fresh:
            assert collinearity is not None
            problem = gamma_problem(
                now,
                request,
                active=collinearity.analysis.kept,
                basis=basis,
            )
        elif type(evidence) is Recorded and evidence.recession_coordinates is not None:
            problem = gamma_problem(
                now,
                request,
                coordinates=evidence.recession_coordinates,
            )
        else:
            problem = gamma_problem(now, request)
        recession = evidence.recession(problem, controls)
    kind = "no_recession" if recession is None else recession.result.status
    if kind != "no_recession":
        status = "recession_unresolved" if kind == "inconclusive" else "no_finite_root"
        return _Trace(status, *known, recession, collinearity, None)
    if collinearity is None:
        _entered(recorder, "collinearity")
        collinearity = evidence.collinearity(
            collinearity_problem(now),
            request.collinearity,
            request.mode,
        )
    analysis = collinearity.analysis
    status = (
        "collinearity_refused"
        if analysis.status == REFUSED
        else "collinearity_unresolved"
    )
    if analysis.status == COMPLETE and analysis.reasons is not None:
        status = "complete"
        if sample is None:
            _entered(recorder, "sample")
            sample = _finished_sample(prepared, first, now, collinearity, owned_order)
    return _Trace(status, *known, recession, collinearity, sample)


class _Fresh:
    """The accepted analyses, each called once, their results admitted on record.

    With a recorder (task M1.11), each result is recorded as returned, before its
    admission, and the invocation once admitted.  With ``warm`` (the fixed-effect
    log-OLS route, ruling 79), the collinearity analysis starts its absorption from
    the coefficients of ``within``.
    """

    def __init__(
        self, recorder: SampleRecorder | None = None, *, warm: bool = False
    ) -> None:
        self.recorder = recorder
        self.warm = warm

    def support(
        self, problem: SupportProblem, controls: SupportControls
    ) -> SupportInvocation:
        """One fresh M1.7 analysis of ``problem``, admitted before any use."""
        result = analyze_support(problem, controls)
        recorder = self.recorder
        if recorder is not None:
            recorder.raw("support", problem, controls, result)
        found = SupportInvocation(problem, controls, result, witnessed(problem, result))
        if recorder is not None:
            recorder.admitted("support", found)
        return found

    def recession(
        self, problem: GammaProblem, controls: LPControls
    ) -> RecessionInvocation:
        """One fresh M1.8 analysis of ``problem``, admitted before any use."""
        result = analyze_recession(problem, controls)
        recorder = self.recorder
        if recorder is not None:
            recorder.raw("recession", problem, controls, result)
        found = RecessionInvocation(problem, controls, result)
        if recorder is not None:
            recorder.admitted("recession", found)
        return found

    def collinearity(
        self,
        problem: CollinearityProblem,
        controls: DiagnosticControls | None,
        mode: str,
    ) -> CollinearityInvocation:
        """One fresh M1.9 analysis of ``problem``, bound to its own inputs."""
        analysis = analyze_collinearity(
            problem, controls, mode=mode, near=True, warm=self.warm
        )
        recorder = self.recorder
        if recorder is not None:
            recorder.raw("collinearity", problem, controls, analysis)
        found = CollinearityInvocation(problem, controls, mode, analysis)
        if recorder is not None:
            recorder.admitted("collinearity", found)
        return found


def same_ledger(a: RowLedger, b: RowLedger) -> bool:
    """Whether two ledgers hold the same rows, causes, inverse, diagnostics, counts.

    Step 2's diagnostics are compared group by group (correction B), not by union.
    """
    arrays = ("codes", "blocks", "positions", "retained_index")
    same = all(same_bits(getattr(a, name), getattr(b, name)) for name in arrays)
    same &= same_missing(a.missing, b.missing)
    if a.replicas is None or b.replicas is None:
        return same and a.replicas is None and b.replicas is None
    return same and same_bits(a.replicas, b.replicas)


def _same_closure(a: Closure, b: Closure) -> bool:
    """Whether two closures made the same visits, separations and ledger."""

    def steps(closure: Closure) -> list[object]:
        found: list[object] = []
        for one in closure.passes:
            found += [(visit.block, visit.removed.tolist()) for visit in one.visits]
            separation = one.separation
            if separation is not None:
                levels = [level.tolist() for level in separation.levels]
                found += [separation.removed.tolist(), separation.blocks.tolist()]
                found.append(levels)
            found.append(separation is None)
        return found

    same = len(a.passes) == len(b.passes) and steps(a) == steps(b)
    return same and same_ledger(a.ledger, b.ledger)


def _same_sample(a: FESample | None, b: FESample | None) -> bool:
    """Whether two completed samples hold the same rows, base, record, blocks, maps.

    The blocks of the final rows and their source, with original level labels.
    """
    if a is None or b is None:
        return a is None and b is None
    found = (a.omega, a.positive_only, a.conditioned, a.reasons)
    same = found == (b.omega, b.positive_only, b.conditioned, b.reasons)
    same = same and a.base is b.base and same_ledger(a.ledger, b.ledger)
    same = same and same_grouping(a.source, b.source)
    return same and same_data(a.data, b.data) and same_grouping(a.grouping, b.grouping)


@dataclass(frozen=True, eq=False)
class SampleOutcome:
    """One composition: its status, ledger, rounds, invocations and sample.

    ``sample`` is present exactly when ``status`` is ``complete``; ``origin`` is the
    completed outcome of the same prepared frame and request that a trigger started
    from.  ``ValueError`` unless the walk over the outcome's own invocations reproduces
    it (module docstring).
    """

    request: SampleRequest
    prepared: PreparedFrame
    status: str
    ledger: RowLedger
    rounds: tuple[Round, ...]
    recession: RecessionInvocation | None = None
    collinearity: CollinearityInvocation | None = None
    sample: FESample | None = None
    origin: SampleOutcome | None = None

    def __post_init__(self) -> None:
        """``ValueError`` for an unknown status, a sample of another, or no binding."""
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if (self.sample is not None) != (self.status == "complete"):
            raise ValueError("only a complete outcome has a sample")
        object.__setattr__(self, "rounds", tuple(self.rounds))
        initial = self._initial()
        trace = _walk(
            self.prepared,
            self.request,
            initial,
            Recorded(self.rounds, self.recession, self.collinearity),
        )
        if not self._reproduced(trace):
            raise ValueError(UNBOUND)

    def _initial(self) -> RowLedger | None:
        """The ledger the walk starts from: a trigger's earlier one, else none."""
        parts = (
            type(self.prepared) is PreparedFrame,
            type(self.request) is SampleRequest,
            isinstance(self.ledger, RowLedger),
            all(isinstance(found, Round) for found in self.rounds),
            isinstance(self.recession, RecessionInvocation | None),
            isinstance(self.collinearity, CollinearityInvocation | None),
            isinstance(self.sample, FESample | None),
        )
        if not all(parts):
            raise ValueError(UNBOUND)
        origin = self.origin
        if origin is None:
            return None
        before = origin.sample if isinstance(origin, SampleOutcome) else None
        if before is None or origin.prepared is not self.prepared:
            raise ValueError(UNBOUND)
        request = self.request
        triggered = replace(
            origin.request, support=request.support, recession=request.recession
        )
        if triggered != request:
            raise ValueError(UNBOUND)
        return before.ledger

    def _reproduced(self, trace: _Trace) -> bool:
        """Whether the walk's status, ledger, rounds, invocations, sample are these."""
        rounds = len(trace.rounds) == len(self.rounds) and all(
            a.support is b.support and _same_closure(a.closure, b.closure)
            for a, b in zip(trace.rounds, self.rounds, strict=False)
        )
        same = rounds and trace.status == self.status
        same = same and same_ledger(trace.ledger, self.ledger)
        same = same and trace.recession is self.recession
        same = same and trace.collinearity is self.collinearity
        return same and _same_sample(trace.sample, self.sample)

    @property
    def support(self) -> tuple[SupportInvocation, ...]:
        """Every M1.7 invocation, in order."""
        return tuple(r.support for r in self.rounds if r.support is not None)

    def labels(self) -> pd.Index[Any]:
        """The user's labels of the current rows, by position, a fresh owned copy."""
        return self.prepared.labels.take(self.ledger.positions)


def _composed(
    prepared: PreparedFrame,
    request: SampleRequest,
    ledger: RowLedger | None,
    origin: SampleOutcome | None,
    recorder: SampleRecorder | None,
) -> SampleOutcome:
    """A fresh walk from the prepared frame, or from ``ledger``, a completed sample's.

    The outcome binds ``ledger`` to ``origin`` (the steps 3 and 4 without one).  With
    a recorder, the walk's events and the outcome's admission are recorded; its
    admission replays the walk on the outcome's own record without it.
    """
    found = _walk(prepared, request, ledger, _Fresh(recorder), recorder=recorder)
    _entered(recorder, "outcome")
    outcome = SampleOutcome(request, prepared, *found, origin=origin)
    if recorder is not None:
        recorder.final(outcome)
    return outcome


def _compose(
    prepared: PreparedFrame,
    request: SampleRequest,
    ledger: RowLedger | None,
    origin: SampleOutcome | None,
) -> SampleOutcome:
    """A fresh walk from the prepared frame, or from ``ledger``, a completed sample's.

    The outcome binds ``ledger`` to ``origin`` (the steps 3 and 4 without one).
    """
    return _composed(prepared, request, ledger, origin, None)


# ------------------------------------- the parent composer's private routes (B)

_Start = Start
_Current = Current
_start = start


def _current(
    prepared: PreparedFrame, request: SampleRequest, ledger: RowLedger
) -> Current:
    """The record, transform and blocks of the current rows (:func:`current`)."""
    return current(prepared, request.omega, ledger)


def _support(
    current: Current, request: SampleRequest, controls: SupportControls
) -> SupportInvocation:
    """One fresh admitted M1.7 analysis of the current rows."""
    return _Fresh().support(support_problem(current, request), controls)


def _recession(
    current: Current, request: SampleRequest, controls: LPControls
) -> RecessionInvocation:
    """One fresh admitted M1.8 analysis of the current rows."""
    return _Fresh().recession(gamma_problem(current, request), controls)


def _collinearity(current: Current, request: SampleRequest) -> CollinearityInvocation:
    """One fresh M1.9 analysis of every original column and block of these rows."""
    problem = collinearity_problem(current)
    return _Fresh().collinearity(problem, request.collinearity, request.mode)


def _rounds(
    prepared: PreparedFrame, request: SampleRequest, start: Start, ledger: RowLedger
) -> tuple[str, RowLedger, list[Round], Current | None]:
    """The rounds of steps 5 to 7 with fresh analyses (:func:`_steps`)."""
    return _steps(prepared, request, start, ledger, _Fresh())


def prepare_sample(prepared: PreparedFrame, request: SampleRequest) -> SampleOutcome:
    """The ordered sample of a prepared frame under explicit controls.

    Parameters
    ----------
    prepared : PreparedFrame
        The owned frame, its base and its fixed-effect labels; never changed.
    request : SampleRequest
        The order, domain, family and every diagnostic's explicit controls.

    Returns
    -------
    SampleOutcome
        The status, the ledger, every invocation and, when complete, the sample.

    Raises
    ------
    SampleError, ValueError, TypeError
        The accepted refusals of steps 3 and 4 and of the exact weights; a
        ``ValueError`` for stale or edited diagnostic evidence.
    FloatingPointError
        A transform of a current row that is not representable.
    SpecificationError
        A formula with instruments (milestone M4).
    """
    if type(prepared) is not PreparedFrame or type(request) is not SampleRequest:
        raise TypeError("prepare_sample takes a PreparedFrame and a SampleRequest")
    return _compose(prepared, request, None, None)


def _prepare_owned_sample(
    prepared: PreparedFrame,
    request: SampleRequest,
    *,
    order_parts: tuple[OrderSample, OrderRows] | None = None,
    basis_native: ModuleType | None = None,
) -> SampleOutcome:
    """One fresh walk on a NORMAL operation's already owned canonical boundary.

    Every sample/scientific analysis and invocation admission remains in _walk.
    Its exact returned fields need no second reconstruction to authenticate our
    own construction. Public prepare_sample/SampleOutcome and foreign/triggered
    entries keep their existing full replay and mutation checks.
    """
    first = _numeric_start(prepared, request, order_parts=order_parts)
    # The collinearity analysis starts cold on every route: from within's start its
    # decisions moved at their thresholds on near-collinear and nested designs.
    found = _walk(
        prepared,
        request,
        None,
        _Fresh(),
        start_state=first,
        owned_order=None if order_parts is None else order_parts[0],
        basis_native=basis_native,
    )
    outcome = object.__new__(SampleOutcome)
    for name, value in (
        ("request", request),
        ("prepared", prepared),
        ("status", found.status),
        ("ledger", found.ledger),
        ("rounds", found.rounds),
        ("recession", found.recession),
        ("collinearity", found.collinearity),
        ("sample", found.sample),
        ("origin", None),
    ):
        object.__setattr__(outcome, name, value)
    return outcome


_ORIGINAL_COMPOSE = _compose
_ORIGINAL_COMPOSED = _composed
_ORIGINAL_WALK = _walk
_ORIGINAL_START = start
_ORIGINAL_CLOSE = close
_ORIGINAL_CURRENT = current
_ORIGINAL_SUPPORT = analyze_support
_ORIGINAL_COLLINEARITY = analyze_collinearity
_ORIGINAL_FRAME_VIEW = PreparedFrame.frame
_ORIGINAL_OUTCOME = SampleOutcome
_ORIGINAL_OUTCOME_INIT = SampleOutcome.__init__
_ORIGINAL_OUTCOME_CHECK = SampleOutcome.__post_init__


def _owned_sample_walk_original() -> bool:
    """Keep ordinary replay at substituted coordinator and diagnostic boundaries."""
    return (
        _compose is _ORIGINAL_COMPOSE
        and _composed is _ORIGINAL_COMPOSED
        and _walk is _ORIGINAL_WALK
        and start is _ORIGINAL_START
        and close is _ORIGINAL_CLOSE
        and current is _ORIGINAL_CURRENT
        and analyze_support is _ORIGINAL_SUPPORT
        and analyze_collinearity is _ORIGINAL_COLLINEARITY
        and PreparedFrame.frame is _ORIGINAL_FRAME_VIEW
        and SampleOutcome is _ORIGINAL_OUTCOME
        and SampleOutcome.__init__ is _ORIGINAL_OUTCOME_INIT
        and SampleOutcome.__post_init__ is _ORIGINAL_OUTCOME_CHECK
        and _fixed_original()
        and _owned_source_original()
    )


def reprepare_sample(
    previous: SampleOutcome,
    *,
    support: SupportControls | None = None,
    recession: LPControls | None = None,
) -> SampleOutcome:
    """A new composition from a completed sample's rows with requested controls.

    The fresh analysis that an iteration trigger requests (``TIGHTER``, tasks M1.11
    and M1.12): steps 5 to 9 again from the final rows of ``previous``, every other
    control unchanged.  ``previous`` is an outcome, bound to its own frame, request and
    history when it was constructed, and is never changed.  ``ValueError`` unless
    ``previous`` is complete and the controls match its family.
    """
    return _triggered(previous, support, recession, None)


def reprepare_recorded(
    previous: SampleOutcome,
    *,
    support: SupportControls | None = None,
    recession: LPControls | None = None,
    recorder: SampleRecorder,
) -> SampleOutcome:
    """:func:`reprepare_sample`, recorded by ``recorder`` (task M1.11).

    The same call, request and walk: ``recorder`` is bound when the call enters, to
    ``previous`` (its origin, which must be the recorder's), the derived request and
    the sample of ``previous``, and records each event of the walk; a failure
    propagates as it is, after the events recorded before it.
    """
    return _triggered(previous, support, recession, recorder)


def _triggered(
    previous: SampleOutcome,
    support: SupportControls | None,
    recession: LPControls | None,
    recorder: SampleRecorder | None,
) -> SampleOutcome:
    """The triggered composition of :func:`reprepare_sample`, recorded or not."""
    if type(previous) is not SampleOutcome or previous.sample is None:
        raise ValueError("a triggered composition starts from a completed sample")
    request = replace(previous.request, support=support, recession=recession)
    if recorder is not None:
        recorder.bind(previous, request)
    ledger = previous.sample.ledger
    return _composed(previous.prepared, request, ledger, previous, recorder)


def _recorded_walk(trace: RecordedTrace, outcome: SampleOutcome) -> bool:
    """Whether the outcome's invocations are the ones the call recorded as admitted.

    The results of the admitted events, in order, are the outcome's own invocation
    results (the very records): every round's M1.7 result, then M1.8/M1.9 in the
    actual order. The identified zero-containing route runs M1.9 first.
    """
    kept = [event.result for event in trace.events if event.kind == "admitted"]
    walk: list[object] = [invocation.result for invocation in outcome.support]
    recession, collinearity = outcome.recession, outcome.collinearity
    identified = recession is not None and recession.problem.coordinates is not None
    if identified and collinearity is not None:
        walk.append(collinearity.analysis)
    if recession is not None:
        walk.append(recession.result)
    if not identified and collinearity is not None:
        walk.append(collinearity.analysis)
    same = len(kept) == len(walk)
    return same and all(a is b for a, b in zip(kept, walk, strict=False))


def admitted_outcome(trace: TraceView, /) -> SampleOutcome:
    """The outcome of one recorded triggered call, admitted again (task M1.11).

    This coordinator's own admission, for a solver that reads the record through the
    lower views of :mod:`powermean.solve.profiled_trigger` (correction A): nothing is
    analyzed or recorded, and nothing is admitted on a status or a structural stand-in.

    Parameters
    ----------
    trace : TraceView
        The :class:`~powermean.interface.sample_records.RecordedTrace` of one call of
        :func:`reprepare_recorded`.

    Returns
    -------
    SampleOutcome
        The trace's own outcome, once admitted again: the call started from a
        completed origin's sample, under the origin's request with the call's support
        and recession controls only, and completed; the outcome's origin has that
        sample and its request is the call's; its invocation results are the ones the
        call recorded as admitted, in order; and its construction admits it again, the
        walk replayed on its own record.

    Raises
    ------
    ValueError
        Any of these does not hold, or the outcome's own admission refuses it.
    """
    if type(trace) is not RecordedTrace:
        raise ValueError("an outcome is admitted from the record of a triggered call")
    origin, outcome = trace.origin, trace.outcome
    if type(origin) is not SampleOutcome or origin.sample is None:
        raise ValueError("the call did not start from a completed outcome")
    if trace.start is not origin.sample:
        raise ValueError("the call did not start from its origin's sample")
    request = trace.request
    support, recession = request.support, request.recession
    if request != replace(origin.request, support=support, recession=recession):
        raise ValueError("the call's request is not its origin's with new controls")
    events = trace.events
    if trace.failure is not None or not events or events[-1].kind != "final":
        raise ValueError("the call did not complete")
    if type(outcome) is not SampleOutcome:
        raise ValueError("the call holds no admitted outcome")
    before = outcome.origin
    if before is None or before.sample is not trace.start or outcome.request != request:
        raise ValueError("the outcome is another call's: another origin or request")
    if not _recorded_walk(trace, outcome):
        raise ValueError("the outcome's invocations are not the ones the call recorded")
    # The outcome's own admission: its walk replayed on its own record (Recorded).
    replace(outcome)
    return outcome
