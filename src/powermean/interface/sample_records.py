"""The request, invocation and round records of the ordered sample.

Internal: no public name, option, default, message, tolerance or route.
:mod:`powermean.interface.sample` walks with these records, binds its outcome to them
and still provides their names.  The rows of a composition (:func:`start`,
:func:`current`) and each diagnostic's problem are formed here once, for a composition
and the binding of its outcome alike.

Invocations are admitted when they are constructed, by the composer, directly or by
``dataclasses.replace``; a status alone authorizes nothing and no admission analyzes
anything again.  Support: ``require_current``, the controls and method that ran,
the result rebuilt on this problem and these controls from owned copies of its whole
evidence (:func:`powermean.interface.sample_evidence.owned_result`),
which replays every witness from its ORIGINAL proposal and the dual with the witnessed
union and is the record kept, and a removal exactly that union's original positions
(:func:`witnessed`).  Recession: ``require_same``, the controls, the rebuilt
record's binding of every report, witness and dual, the strict witness's replay;
``raise_for_recession`` is silent for ``no_recession`` and ``inconclusive`` alike, so an
outcome follows the checked status, never that silence.  Collinearity: the
analysis's OWN problem, actual controls and mode are the invocation's; an unbound
analysis authorizes nothing.  The recession and collinearity records own their arrays
and tuples through their constructors and are kept as given.  Blocks are
compared with their original level labels wherever the composition compares them
(:func:`same_grouping`): the partition checks of the support and recession analyses are
supplemented, never changed.  The blocks of the current rows derive from the walk's
source, the blocks of the rows of steps 3 and 4 (:func:`current`), so
that labels whose equality is identity compare through that relation.

The recorder.
:class:`SampleRecorder` is the optional passive record of one triggered composition:
bound to its very origin when the call enters, it receives the walk's events, entries
of :data:`STAGES`, closures, each analysis as returned (``raw``) and as admitted, and
the admitted outcome (``final``), and freezes them into a :class:`RecordedTrace`, with
the propagated exception after a failure.  It runs nothing, and its absence changes
nothing.  It owns what it keeps: a returned support result is kept as its
:class:`RawSupportResult`, an owned copy of its whole finite graph
made by the copying steps of :mod:`powermean.interface.sample_evidence`,
before the invocation's admission runs, so that a later write to an array of the
returned object changes nothing recorded.  The admitting ``SupportResult`` constructor
never runs on it, so evidence that no longer passes its fresh checks is kept as it
returned, and nothing admits it.  The recession and collinearity records, the
problems, the closures and the admitted invocations, which their constructors own, are
kept as given.
A returned object that is not its analysis record, or a part of whose graph cannot be
owned, is recorded without it and with the reason: never filtered, promoted or analyzed
again, while the invocation's admission then runs as without a recorder.  The origin of
the call is the one bound when it entered; the recorder's own reference is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, cast

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean.data.accounting import RETAINED, ZERO_WEIGHT, frozen_index
from powermean.data.collinearity import (
    MODES,
    CollinearityAnalysis,
    CollinearityProblem,
    DiagnosticControls,
)
from powermean.data.inputs import RegressionData
from powermean.data.retained import (
    OrderRows,
    OrderSample,
    _exact_index,
    _regression_data,
    select_order,
)
from powermean.data.sample_fe import (
    Closure,
    FESample,
    FixedEffects,
    RowLedger,
    _owned_fixed,
    ledger_of,
    replicas_of,
)
from powermean.data.sample_fe_state import (
    LARGEST,
    Derived,
    Source,
    same_bits,
    same_blocks,
)
from powermean.data.weights import ObservationWeights
from powermean.errors import SpecificationError
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.interface.sample_encoding import _integer_key_codes
from powermean.interface.sample_evidence import (
    _dual,
    _each,
    _fe,
    _ir,
    _kept,
    _lp,
    _maybe,
    _primal,
    _shared,
    owned_result,
)
from powermean.model.order import IndexArray, TransformedProblem, transform_problem
from powermean.model.order_scale import _mean_order
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve.support import (
    FEStep,
    SupportControls,
    SupportResult,
    require_current,
    select_method,
)
from powermean.solve.support_gamma import GammaProblem, GammaResult, raise_for_recession
from powermean.solve.support_lp import LPControls, LPOutcome
from powermean.solve.support_problem import STALE, SupportProblem, indices, same_sample
from powermean.solve.support_rectifier import RectifierOutcome
from powermean.solve.support_witness import DualWitness, PrimalWitness

Family = PowerVariance | NB2Variance
VARIANCE = (
    "the ordered composition covers PowerVariance and NB2Variance at omega != 0, "
    "and no working variance at omega = 0"
)
RANK_ONLY = (
    "outside 1 <= tau <= 2 the finite root is not characterized: rank checks only, "
    "no SupportControls or LPControls"
)
BINDING = "the result is bound to other controls, method, rows, columns or mode"
UNBOUND = "the outcome is not bound to its prepared frame, request and evidence"


def _cone(omega: float, variance: object) -> str:
    """The existence analysis of the order and family: support, recession, rank, none.

    ``"rank"`` for a power variance with ``tau`` outside ``[1, 2]`` at ``omega != 0``
    (the existence of a finite root is not characterized there): no existence
    analysis and no deletion for it; the closure
    of steps 5 and 6 and the collinearity analysis (rank checks) only, the solver and
    the certificate deciding.
    """
    if omega == 0:
        if variance is not None:
            raise ValueError(VARIANCE)
        return "none"
    if type(variance) is NB2Variance:
        return "support"
    if type(variance) is PowerVariance and 1.0 <= variance.tau < 2.0:
        return "support"
    if type(variance) is PowerVariance and variance.tau == 2.0:
        return "recession"
    if type(variance) is PowerVariance:
        return "rank"
    raise ValueError(VARIANCE)


@dataclass(frozen=True)
class SampleRequest:
    """The explicit internal controls of one composition; nothing is defaulted.

    ``omega`` and ``positive_only`` (steps 3 and 4); ``variance`` at ``omega != 0``
    (``None`` at ``omega = 0``); ``keep_singletons`` skips step 5 only; ``support``,
    the support controls, exactly for ``1 <= tau < 2`` or NB2; ``recession``, the
    recession controls, exactly for ``tau = 2``; ``collinearity``, the absorber
    controls of the collinearity analysis,
    exactly with fixed effects; ``mode``, ``"drop"`` or ``"error"``.  A family whose
    finite root is not characterized (``tau < 1`` or ``tau > 2``) takes neither
    existence control: the rank-only route of :func:`_cone`.  ``ValueError`` for
    mismatched controls.
    """

    omega: float
    positive_only: bool
    variance: Family | None
    keep_singletons: bool
    support: SupportControls | None
    recession: LPControls | None
    collinearity: DiagnosticControls | None
    mode: str

    def __post_init__(self) -> None:
        """Check every control against the order and family; store the order."""
        object.__setattr__(self, "omega", _mean_order(self.omega))
        if not isinstance(self.positive_only, bool) or not isinstance(
            self.keep_singletons, bool
        ):
            raise ValueError("positive_only and keep_singletons must be bools")
        if self.mode not in MODES:
            raise ValueError("mode must be 'drop' or 'error'")
        cone = _cone(self.omega, self.variance)
        support, recession = self.support, self.recession
        if cone == "rank" and (support is not None or recession is not None):
            raise ValueError(RANK_ONLY)
        if (support is not None) != (cone == "support") or not isinstance(
            support, SupportControls | None
        ):
            raise ValueError("SupportControls are given exactly for tau < 2 or NB2")
        if (recession is not None) != (cone == "recession") or not isinstance(
            recession, LPControls | None
        ):
            raise ValueError("LPControls are given exactly for tau = 2")
        if not isinstance(self.collinearity, DiagnosticControls | None):
            raise ValueError("collinearity must be DiagnosticControls or None")

    @property
    def family(self) -> Family:
        """The working variance; ``ValueError`` at ``omega = 0``, which has none."""
        if self.variance is None:
            raise ValueError(VARIANCE)
        return self.variance


# ---------------------------------------------------- the rows and their problems


@dataclass(frozen=True, eq=False)
class Start:
    """Steps 1 to 4 done once: their rows, the raw outcome by input row, the blocks."""

    rows: OrderRows
    outcome: FloatArray
    fixed: FixedEffects | None
    ledger: RowLedger


def start(prepared: PreparedFrame, request: SampleRequest) -> Start:
    """Steps 3 and 4, their exact weights and the blocks; refusals as accepted."""
    if prepared.frame.parts.instruments is not None:
        fields = {"feature": "instruments"}
        raise SpecificationError("specification.unimplemented", fields)
    return _start_rows(prepared, request)


def _numeric_start(
    prepared: PreparedFrame, request: SampleRequest, *,
    order_parts: tuple[OrderSample, OrderRows] | None = None,
) -> Start:
    """The same start of a private owned producer, without exporting its rich frame."""
    if prepared._frame.parts.instruments is not None:
        raise SpecificationError(
            "specification.unimplemented", {"feature": "instruments"}
        )
    if order_parts is None:
        return _start_rows(prepared, request)
    order, rows = order_parts
    effects, base = prepared.effects, prepared.base
    if (effects is None) != (request.collinearity is None):
        raise ValueError("DiagnosticControls are given exactly with fixed effects")
    outcome = np.zeros(rows.nobs)
    outcome[rows.positions] = order.data.y
    fixed = None if effects is None else _owned_fixed(
        effects, rows.positions, _encoder=_integer_key_codes
    )
    ledger = ledger_of(rows, base.missing, replicas_of(order.data, rows))
    return Start(rows, immutable_array(outcome), fixed, ledger)


def _start_rows(prepared: PreparedFrame, request: SampleRequest) -> Start:
    """The existing order/weight/block start, shared without changing any operation."""
    effects, base = prepared.effects, prepared.base
    if (effects is None) != (request.collinearity is None):
        raise ValueError("DiagnosticControls are given exactly with fixed effects")
    rows = select_order(base, request.omega, positive_only=request.positive_only)
    data = _regression_data(base.inputs, rows.positions)
    outcome = np.zeros(rows.nobs)
    outcome[rows.positions] = data.y
    fixed = None if effects is None else effects.fixed(rows.positions)
    ledger = ledger_of(rows, base.missing, replicas_of(data, rows))
    return Start(rows, immutable_array(outcome), fixed, ledger)


@dataclass(frozen=True, eq=False)
class Current:
    """The current rows: their ledger, numeric record, transformed problem, blocks."""

    ledger: RowLedger
    data: RegressionData
    problem: TransformedProblem
    grouping: Grouping | None


def source_of(first: Start) -> Source | None:
    """The source of a walk's blocks, those of the rows of steps 3 and 4, if any."""
    grouping = None if first.fixed is None else first.fixed.grouping
    return grouping.source if isinstance(grouping, Derived) else None


def current(
    prepared: PreparedFrame,
    omega: float,
    ledger: RowLedger,
    source: Source | None = None,
) -> Current:
    """The record, transform and blocks of the current rows only.

    The blocks are :class:`~powermean.data.sample_fe_state.Derived` from ``source``,
    the walk's blocks of the rows of steps 3 and 4, or without one from
    the prepared effects on the ledger's rows of steps 3 and 4 (codes 0 and 5 to 7):
    the encoding of the current rows, with the source level of each level.
    """
    data = _regression_data(prepared.base.inputs, ledger.positions)
    problem = transform_problem(data.y, omega, offset=data.offset)
    effects = prepared.effects
    grouping = None
    if effects is not None:
        if source is None:
            eligible = (ledger.codes == RETAINED) | (ledger.codes > ZERO_WEIGHT)
            rows = frozen_index(np.flatnonzero(eligible))
            source = Source(effects.terms, effects.columns, effects.rows, rows)
        grouping = Derived(source, ledger.positions)
    return Current(ledger, data, problem, grouping)


Rows = tuple[FloatArray, FloatArray, tuple[str, ...], Grouping | None, IndexArray]


def _rows_of(current: Current) -> Rows:
    """The transformed problem's rows: ``Z``, slopes, names, blocks and positions."""
    data, problem = current.data, current.problem
    found = (problem.response, data.design, data.column_names, current.grouping)
    return (*found, current.ledger.positions)


def support_problem(now: Current, request: SampleRequest) -> SupportProblem:
    """The support problem of the current rows, with their weights and offset."""
    weights, offset = now.data.weight_values, now.problem.offset
    found = (weights, offset, request.family, request.omega)
    return SupportProblem(*_rows_of(now), *found)


def gamma_problem(now: Current, request: SampleRequest) -> GammaProblem:
    """The recession problem of the current rows with their typed weights and offset."""
    weights = now.data.weights
    if weights is None:
        weights = ObservationWeights(np.ones(now.data.nobs))
    # The request routes only PowerVariance(2) here; GammaProblem refuses any other.
    variance = cast("PowerVariance", request.family)
    offset = now.problem.offset
    return GammaProblem(*_rows_of(now), weights, offset, variance, request.omega)


def collinearity_problem(now: Current) -> CollinearityProblem:
    """The collinearity problem: every original column and block of the current rows."""
    data, positions = now.data, now.ledger.positions
    return CollinearityProblem(data.design, data.column_names, positions, now.grouping)


def same_grouping(a: Grouping | None, b: Grouping | None) -> bool:
    """The same blocks of the same rows with their original level labels.

    The partition that the support and recession analyses compare, and every
    block's ordered level labels
    (:func:`powermean.data.sample_fe_state.same_blocks`).
    """
    return same_blocks(a, b)


def same_problem(a: CollinearityProblem, b: CollinearityProblem) -> bool:
    """Whether two collinearity problems hold the same columns, rows and blocks."""
    same = same_bits(a.design, b.design) and a.names == b.names
    same = same and same_bits(a.positions, b.positions)
    return same and same_grouping(a.grouping, b.grouping)


# ------------------------------------------------------------ the invocations


def witnessed(problem: SupportProblem, result: SupportResult) -> IndexArray:
    """The original positions of the witnessed union of ``result``, on ``problem``."""
    require_current(result, problem)
    return indices(problem.positions[result.separated])


@dataclass(frozen=True, eq=False)
class SupportInvocation:
    """One fresh support analysis: its problem, controls, result and rows removed.

    ``removed`` holds the original positions of the admitted witnessed union.
    """

    problem: SupportProblem
    controls: SupportControls
    result: SupportResult
    removed: IndexArray

    def __post_init__(self) -> None:
        """Admit the result on this problem and controls; own it and the removal."""
        problem, controls, result = self.problem, self.controls, self.result
        if not isinstance(result, SupportResult) or not isinstance(
            controls, SupportControls
        ):
            raise ValueError(BINDING)
        require_current(result, problem)
        if not same_grouping(result.problem.grouping, problem.grouping):
            raise ValueError(STALE)
        if result.controls != controls or result.method != select_method(
            problem, controls
        ):
            raise ValueError(BINDING)
        # The record rebuilt on this problem and these controls from owned copies of
        # its whole evidence runs the checks: every witness replayed from its
        # original proposal, the dual with the union.  It is the record kept, never a
        # caller's array, list or mapping.
        checked = owned_result(result, problem, controls)
        removed = _exact_index(self.removed, "removed rows", upper=LARGEST)
        if not same_bits(removed, witnessed(problem, checked)):
            raise ValueError("the removal must be the invocation's witnessed union")
        object.__setattr__(self, "result", checked)
        object.__setattr__(self, "removed", removed)


@dataclass(frozen=True, eq=False)
class RecessionInvocation:
    """One fresh recession analysis: its problem, controls and result."""

    problem: GammaProblem
    controls: LPControls
    result: GammaResult

    def __post_init__(self) -> None:
        """Admit the result on this problem and these controls, every proof replayed."""
        problem, controls, result = self.problem, self.controls, self.result
        if not isinstance(result, GammaResult):
            raise ValueError(BINDING)
        result.problem.require_same(problem)
        if not same_grouping(result.problem.grouping, problem.grouping):
            raise ValueError(STALE)
        if result.controls != controls:
            raise ValueError(BINDING)
        # The rebuilt record binds every program's report, witness and dual, replayed.
        replace(result, problem=problem, controls=controls)
        witness = result.witness
        if witness is not None:
            witness.require(problem, controls=controls, stage=witness.stage)

    def raise_for_recession(self) -> None:
        """The specified ``ExistenceError`` of an admitted recession."""
        raise_for_recession(self.result, self.problem)


@dataclass(frozen=True, eq=False)
class CollinearityInvocation:
    """One fresh collinearity analysis: problem, actual controls, mode and analysis."""

    problem: CollinearityProblem
    controls: DiagnosticControls | None
    mode: str
    analysis: CollinearityAnalysis

    def __post_init__(self) -> None:
        """The analysis's own problem, controls and mode are this invocation's."""
        analysis = self.analysis
        if not isinstance(analysis, CollinearityAnalysis) or analysis.problem is None:
            raise ValueError("an analysis without its problem authorizes nothing")
        problem = self.problem
        if not isinstance(problem, CollinearityProblem) or not same_problem(
            problem, analysis.problem
        ):
            raise ValueError(BINDING)
        if self.controls != analysis.controls or self.mode != analysis.mode:
            raise ValueError(BINDING)


@dataclass(frozen=True, eq=False)
class Round:
    """Steps 5 and 6 to the fixed point, then the round's support invocation, if any."""

    closure: Closure
    support: SupportInvocation | None

    def __post_init__(self) -> None:
        """``ValueError`` unless the invocation analyzed the closure's current rows."""
        closure, support = self.closure, self.support
        if not isinstance(closure, Closure):
            raise ValueError("a round holds its closure")
        if support is not None and not (
            isinstance(support, SupportInvocation)
            and same_bits(support.problem.positions, closure.ledger.positions)
        ):
            raise ValueError("a round's invocation is on the rows of its closure")


class Evidence(Protocol):
    """Where a walk takes each invocation: fresh analyses, or an outcome's record."""

    def support(
        self, problem: SupportProblem, controls: SupportControls
    ) -> SupportInvocation:
        """The support invocation on ``problem`` under ``controls``."""
        ...

    def recession(
        self, problem: GammaProblem, controls: LPControls
    ) -> RecessionInvocation:
        """The recession invocation on ``problem`` under ``controls``."""
        ...

    def collinearity(
        self,
        problem: CollinearityProblem,
        controls: DiagnosticControls | None,
        mode: str,
    ) -> CollinearityInvocation:
        """The collinearity invocation on ``problem`` under ``controls`` in ``mode``."""
        ...


class Recorded:
    """Recorded invocations, served in order, each on the walk's problem and controls.

    ``ValueError`` (:data:`UNBOUND`) for a missing invocation or one of another problem,
    controls or mode; nothing is analyzed.
    """

    def __init__(
        self,
        rounds: tuple[Round, ...],
        recession: RecessionInvocation | None,
        collinearity: CollinearityInvocation | None,
    ) -> None:
        """Take the invocations of an outcome's rounds and stages."""
        self._support = [r.support for r in rounds if r.support is not None]
        self._recession, self._collinearity = recession, collinearity

    def support(
        self, problem: SupportProblem, controls: SupportControls
    ) -> SupportInvocation:
        """The next recorded support invocation, on ``problem`` under ``controls``."""
        found = self._support.pop(0) if self._support else None
        if found is None or not same_sample(found.problem, problem):
            raise ValueError(UNBOUND)
        if not same_grouping(found.problem.grouping, problem.grouping):
            raise ValueError(UNBOUND)
        if found.controls != controls:
            raise ValueError(UNBOUND)
        return found

    def recession(
        self, problem: GammaProblem, controls: LPControls
    ) -> RecessionInvocation:
        """The recorded recession invocation, on ``problem`` under ``controls``."""
        found = self._recession
        if found is None or not found.problem.same(problem):
            raise ValueError(UNBOUND)
        if not same_grouping(found.problem.grouping, problem.grouping):
            raise ValueError(UNBOUND)
        if found.controls != controls:
            raise ValueError(UNBOUND)
        return found

    def collinearity(
        self,
        problem: CollinearityProblem,
        controls: DiagnosticControls | None,
        mode: str,
    ) -> CollinearityInvocation:
        """The recorded collinearity invocation: ``problem``, ``controls``, ``mode``."""
        found = self._collinearity
        if found is None or not same_problem(found.problem, problem):
            raise ValueError(UNBOUND)
        if found.controls != controls or found.mode != mode:
            raise ValueError(UNBOUND)
        return found


# ------------------------------------------------------------------ the recorder

STAGES = (
    "closure",
    "current",
    "support",
    "recession",
    "collinearity",
    "sample",
    "outcome",
)
Diagnostic = SupportProblem | GammaProblem | CollinearityProblem
Controls = SupportControls | LPControls | DiagnosticControls
Analysis = SupportResult | GammaResult | CollinearityAnalysis


@dataclass(frozen=True, eq=False)
class RawSupportResult:
    """A support result as its analysis returned it: owned, never admitted.

    The recorder's raw view: the fields of
    :class:`~powermean.solve.support.SupportResult`, in its order, each an owned part of
    the returned finite graph (:func:`raw_support`).  It is not a ``SupportResult`` and
    runs none of its checks: its witnesses and dual are kept as they returned, whether
    or not they still pass their fresh checks, and it states no status, separated set or
    removal.  No invocation, solver or outcome admits it.
    """

    problem: SupportProblem
    controls: SupportControls
    method: str
    fe: FEStep | None
    lp: LPOutcome | None
    ir: RectifierOutcome | None
    witnesses: tuple[PrimalWitness, ...]
    dual: DualWitness | None
    completion: str
    reason: str


Raw = RawSupportResult | GammaResult | CollinearityAnalysis


def raw_support(result: SupportResult) -> RawSupportResult:
    """The owned raw view of a returned support result.

    Every part of the finite graph is owned by the copying steps of
    :mod:`powermean.interface.sample_evidence`: each array in bytes of its own with its
    dtype, shape and every bit, read-only; each sequence a tuple; each record rebuilt
    from owned parts; the records that their constructors own (the problem, the
    controls) kept as the very objects.  Nothing is checked, replayed, filtered,
    converted or analyzed: the admitting ``SupportResult`` constructor is never called.
    ``ValueError`` (:data:`~powermean.interface.sample_evidence.UNOWNED`) for a part
    that cannot be owned so.
    """
    return RawSupportResult(
        _shared(result.problem, SupportProblem),
        _shared(result.controls, SupportControls),
        _kept(result.method),
        _maybe(result.fe, _fe),
        _maybe(result.lp, _lp),
        _maybe(result.ir, _ir),
        _each(result.witnesses, _primal),
        _maybe(result.dual, _dual),
        _kept(result.completion),
        _kept(result.reason),
    )


class Composed(Protocol):
    """An outcome of the composition (``powermean.interface.sample.SampleOutcome``)."""

    @property
    def status(self) -> str:
        """The status of the composition."""
        ...

    @property
    def sample(self) -> FESample | None:
        """The completed sample, when the status is complete."""
        ...


@dataclass(frozen=True, eq=False)
class RecordedEvent:
    """One event of a recorded call, as it happened.

    ``entry``: the walk entered ``stage`` (one of :data:`STAGES`); ``closure``: steps 5
    and 6 returned ``closure``; ``raw``: an analysis returned ``result`` on ``problem``
    under ``controls``, before any admission, kept as an owned record (a support result
    as its :class:`RawSupportResult`), or ``None`` with the reason in ``unowned`` when
    what returned is not its analysis record or a part of it cannot be owned;
    ``admitted``: the invocation record admitted it, with its problem, controls, kept
    result and, for support, ``removed``, for collinearity ``mode``; ``final``: the
    outcome was admitted.
    """

    kind: str
    stage: str
    problem: Diagnostic | None = None
    controls: Controls | None = None
    result: Analysis | RawSupportResult | None = None
    closure: Closure | None = None
    removed: IndexArray | None = None
    mode: str | None = None
    unowned: str = ""


@dataclass(frozen=True, eq=False)
class RecordedTrace:
    """The frozen record of one recorded call.

    ``origin`` is the completed outcome the call started from, ``start`` its sample and
    ``request`` the request derived for the call; ``events`` every event, in order;
    ``outcome`` the admitted outcome of a completed call, else ``None``; ``failure``
    the exception that propagated out of a failed call, as it was raised, and
    ``unfinished`` the stage of its last entry or returned analysis then.
    """

    origin: Composed
    start: FESample
    request: SampleRequest
    events: tuple[RecordedEvent, ...]
    outcome: Composed | None
    unfinished: str | None
    failure: Exception | None


Binding = tuple[Composed, FESample, SampleRequest]


def _owned_raw(result: object) -> tuple[Raw | None, str]:
    """A returned analysis as the recorder keeps it, or ``None`` and why (module doc).

    A support result is kept as its owned raw view (:func:`raw_support`), never
    admitted, whose only refusal, a part that cannot be owned (``ValueError``), leaves
    it unowned; the recession and collinearity records are their constructors' owned
    records, kept as given.
    """
    if isinstance(result, SupportResult):
        try:
            return raw_support(result), ""
        except ValueError as error:
            return None, f"the returned evidence cannot be owned: {error}"
    if isinstance(result, GammaResult | CollinearityAnalysis):
        return result, ""
    name = type(result).__name__
    return None, f"the analysis returned a value of type {name}, not its record"


class SampleRecorder:
    """The passive record of one triggered composition from ``origin``.

    The coordinator binds it when the call enters, to the very origin of the call, the
    request it derived and the origin's sample, then hands it each event of the walk;
    the recorder analyzes nothing, changes nothing it receives and is never consulted
    by the walk, and an outcome's admission, which replays the walk on its record,
    records nothing.  It keeps owned records (module docstring) and its origin is
    read-only: every record of the call holds the origin bound at its entry.
    :meth:`snapshot` freezes a completed call, :meth:`failed` a call out of which an
    exception propagated.  ``ValueError`` for a call from another origin, a second
    call, an event outside a bound call that has not ended, or a record of a call that
    has not ended so.
    """

    def __init__(self, origin: Composed) -> None:
        self._origin = origin
        self._binding: Binding | None = None
        self._events: list[RecordedEvent] = []
        self._outcome: Composed | None = None
        self._failure: Exception | None = None

    @property
    def origin(self) -> Composed:
        """The completed outcome this recorder records a call from (read-only)."""
        return self._origin

    def bind(self, origin: Composed, request: SampleRequest) -> None:
        """Bind the recorder to the call entering from ``origin`` under ``request``."""
        if origin is not self._origin:
            raise ValueError("the recorder is bound to another origin")
        if self._binding is not None:
            raise ValueError("a recorder records one call")
        if origin.sample is None:
            raise ValueError("a triggered composition starts from a completed sample")
        self._binding = (origin, origin.sample, request)

    def _bound(self) -> Binding:
        """The binding of the call being recorded, which has not ended."""
        ended = self._outcome is not None or self._failure is not None
        if self._binding is None or ended:
            raise ValueError("no call is being recorded")
        return self._binding

    def _record(self, event: RecordedEvent) -> None:
        """Append ``event`` to the call being recorded."""
        self._bound()
        self._events.append(event)

    def entry(self, stage: str) -> None:
        """The walk enters ``stage``."""
        self._record(RecordedEvent("entry", stage))

    def closure(self, closure: Closure) -> None:
        """Steps 5 and 6 returned ``closure``."""
        self._record(RecordedEvent("closure", "closure", closure=closure))

    def raw(
        self,
        stage: str,
        problem: Diagnostic,
        controls: Controls | None,
        result: Analysis,
    ) -> None:
        """An analysis of ``stage`` returned ``result``, not yet admitted.

        Kept as an owned record, or without it and with the reason (module docstring);
        nothing is raised for what was returned.
        """
        kept, unowned = _owned_raw(result)
        event = RecordedEvent("raw", stage, problem, controls, kept, unowned=unowned)
        self._record(event)

    def admitted(
        self,
        stage: str,
        invocation: SupportInvocation | RecessionInvocation | CollinearityInvocation,
    ) -> None:
        """The invocation record of ``stage`` admitted its analysis."""
        found = (stage, invocation.problem, invocation.controls)
        if isinstance(invocation, CollinearityInvocation):
            analysis, mode = invocation.analysis, invocation.mode
            event = RecordedEvent("admitted", *found, analysis, mode=mode)
        elif isinstance(invocation, SupportInvocation):
            removed = invocation.removed
            event = RecordedEvent(
                "admitted", *found, invocation.result, removed=removed
            )
        else:
            event = RecordedEvent("admitted", *found, invocation.result)
        self._record(event)

    def final(self, outcome: Composed) -> None:
        """The outcome of the call was admitted: the call is complete."""
        self._record(RecordedEvent("final", "outcome"))
        self._outcome = outcome

    def _trace(self, binding: Binding, unfinished: str | None) -> RecordedTrace:
        """The frozen record of the call, with the origin bound at its entry."""
        origin, sample, request = binding
        found = (origin, sample, request, tuple(self._events), self._outcome)
        return RecordedTrace(*found, unfinished, self._failure)

    def snapshot(self) -> RecordedTrace:
        """The frozen record of the completed call."""
        if self._binding is None or self._outcome is None:
            raise ValueError("the call has not completed")
        return self._trace(self._binding, None)

    def failed(self, error: Exception) -> RecordedTrace:
        """The frozen record of the call out of which ``error`` propagated."""
        if not isinstance(error, Exception):
            raise TypeError("error must be the exception that propagated")
        binding = self._bound()
        self._failure = error
        stages = (e.stage for e in reversed(self._events) if e.kind in ("entry", "raw"))
        return self._trace(binding, next(stages, None))
