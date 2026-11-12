"""The standard reading of a joint PPML point and the standard assessor of one fit.

Internal: only :mod:`powermean.interface.ppml_audit` and the other ``ppml_standard_*``
modules import it, and only on the explicitly requested ``audit_mode="standard"``.

Reading (:func:`standard_point`), in this order, on the records it is given: the point
admission (``admit_point``: the supplied ``q`` is authoritative and never re-profiled)
and the sample's basis ``J`` (``owned_basis``), whose refusals propagate as in
``certify_point``; the fresh rows ``b = a r`` and ``C = a c`` (observed curvature); the
criterion ``L/A`` and the native-coordinate gradient in the native order, ``v =
_weighted(-r, a)`` divided row by row by ``A = np.sum(a)``, then ``X'v``, followed on
the fitted-index route by the sequential level sums of
:func:`~powermean.solve.absorbed_loss.full_score` (profiled: the slopes only, an empty
vector and ``s = 0`` without slopes); the raw-weight level means of every block; the
exact rank of ``J`` on the positive-curvature rows (a loss is a failure, never a smaller
model); the unchanged ``inner_evidence(point, effects, C, tol)``, with its cold
auxiliary absorption.  A range refusal of a stage is a failure with its reason; nothing
is clipped, floored or replaced.  Checks: ``s <= gradient_tol``, the largest level mean
at most ``tol``, and every inner check against each of its bounds, equality included.

Opening-retry reading (:func:`opening_point`; the policy
``ppml_standard_index_opening_v2`` of :mod:`powermean.results.ppml_standard_records`),
of a fitted-index point only: the same admission, basis, fresh rows, criterion and
native gradient, and with rows the positive-curvature support, in the same order with
the same refusals and failures; the level means and ``inner_evidence`` (recovery, gaps,
normalization, auxiliary absorption) are not run.  ``s`` is the largest absolute
component of that fresh native gradient, NaN after its range refusal; a refusal of the
rows or a lost support rank is a failure that leaves a formed ``s`` as it is.  The
reading is never accepted and its error is NaN.

Assessor (:class:`StandardAssessor`).  The mode-specific assessor that the unchanged
retry sequence of the solvers calls; it is not ``FEAssessor``.  The fit's roots (sample,
family, absorbed options, request, controls and the native gradient control
``gradient_tol``) were captured and materialized at the entry
(:mod:`powermean.results.absorbed_assessment_capture`); every call captures them again,
``gradient_tol`` read from the solver's options, with its point and materializes the
point on that very replica (``prefix``): a change of any value the fit read leaves no
replica, and the call is refused (a reading that is not accepted, with NaN score and
error, its reason kept).  The first call binds the solver's sample, family and controls
objects; every later call must present them.  A reading reads its replica only, its
``gradient_tol`` included.  It returns ``Assessment(accepted, s, error)``.  Its
:meth:`~StandardAssessor.opening` is the opening callback of the index solver's internal
pair, which the internal entry registers on the index route only: the same ownership,
capture and refusals, then the opening-retry reading of its replica, kept with the role
``opening``, and ``Assessment(False, s, NaN)``.  Every other role is the call's place in
the call order, a label that decides nothing.

Selection (:meth:`StandardAssessor.select`).  The candidate's history is read afresh by
the existing readers.  A candidate whose options hold another ``gradient_tol`` than the
fit's captured control is not a candidate of the fit's controls and is refused
(``ValueError``), as the index history's reader refuses a final work guarded by another
one.  The selected record is captured with the audit's controls and the candidate's
``gradient_tol`` on the fit's replica.  A kept complete reading of that very record
whose tokens equal the fresh capture's is the selected reading (consumed); otherwise a
fresh complete reading runs on the new replica: an opening-retry reading is never
consumed.
Standard convergence needs both the accepted point predicate and a complete operative
native stop (:func:`native_failures`); a last point after a limit, a search, a
factorization, an ascent, a breakdown, an inner limit, a failed final work or an alarm
never converges, and no historical diagnostic is recomputed.
"""

from __future__ import annotations

import math
import operator
import time
from dataclasses import dataclass
from typing import cast

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.fe.certificate_basis import owned_basis, positive_support
from powermean.fe.reductions import immutable_array
from powermean.model.rows import evaluate_criterion, evaluate_rows
from powermean.model.variance import FloatArray
from powermean.results.absorbed_assessment_binding import Anchors
from powermean.results.absorbed_assessment_capture import (
    Capture,
    Replica,
    Token,
    capture,
    replicate,
)
from powermean.results.absorbed_assessment_handoff import FIT, _parts
from powermean.results.ppml_standard_records import (
    COMPLETE_KIND,
    EFFECTS,
    GRADIENT,
    INDEX_RETRY_POLICY,
    OPENING_RETRY_KIND,
    ROLES,
    RetainedStandardReading,
    Role,
    StandardCheck,
    StandardControls,
    StandardFailure,
    StandardPointAssessment,
    StandardStatus,
    bounded,
    not_run,
    opening_omitted_checks,
    summary,
)
from powermean.solve.absorbed_certificate_geometry import support_step
from powermean.solve.absorbed_certificate_history import (
    CandidateHistory,
    index_history,
    profiled_history,
)
from powermean.solve.absorbed_certificate_inner import (
    InnerCheck,
    InnerEvidence,
    inner_evidence,
)
from powermean.solve.absorbed_certificate_point import (
    BOUND,
    AdmittedPoint,
    EffectEquations,
    FreshRows,
    WorkStep,
    admit_point,
    effect_equations,
    fresh_rows,
    problem_of,
)
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_certificate_sparse import SUPPORT
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.absorbed_loss import ROWS, full_score
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate, IndexFinal
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.options import SolverOptions
from powermean.solve.point_loss import _weighted
from powermean.solve.profiled_loss import FinalWork
from powermean.solve.profiled_trigger import WholePoint
from powermean.systems.profiled import Family, PointProfile

CHANGED = "the inputs of the reading are not those of the fit"
CONTROL = "the candidate's gradient_tol is not the fit's native gradient control"
NOT_KEPT = "no kept standard reading of the selected record with its inputs"
NO_POINT = "the candidate holds no selected point"
# The roots of a standard fit: the handoff's five, then the native gradient control.
STANDARD_FIT = FIT + 1
# The parts of a complete final work, in its order, before the slope absorption.
PROFILED_PARTS = ("point", "equations")
INDEX_PARTS = ("recovery", "rows", "equations", "rhs")


# ---------------------------------------------------------------- the reading


def native_gradient(point: AdmittedPoint) -> tuple[float, FloatArray]:
    """The criterion ``L/A`` and the native-coordinate gradient at ``q`` (docstring).

    ``FloatingPointError`` when a row, the criterion or a component is not
    representable.
    """
    variance = point.variance
    if variance is None:
        raise ValueError(BOUND)
    problem = problem_of(point.sample, variance)
    a, z = problem.weights, problem.z
    values = evaluate_criterion(z, point.q, variance=variance)
    rows = evaluate_rows(z, point.q, variance=variance)
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.sum(a)
        criterion = float(_weighted(values, a).sum() / total)
        v = _weighted(-rows.score, a)
        v /= total
        slopes: FloatArray = problem.design.T @ v
    finite = np.isfinite(v).all() and np.isfinite(slopes).all()
    if not (math.isfinite(criterion) and finite):
        raise FloatingPointError(ROWS)
    if point.kind == "profiled":
        return criterion, immutable_array(slopes)
    return criterion, immutable_array(full_score(problem, slopes, v))


def _refused(name: str, error: Exception, failures: list[StandardFailure]) -> WorkStep:
    """The failure and the work step of a stage refused by range."""
    failures.append(StandardFailure(name, str(error)))
    return WorkStep(name, "refused", str(error), None)


def _rows(
    point: AdmittedPoint, failures: list[StandardFailure], steps: list[WorkStep]
) -> FreshRows | None:
    """The fresh observed rows, or ``None`` after a range refusal."""
    try:
        rows = fresh_rows(point)
    except FloatingPointError as error:
        steps.append(_refused("rows", error, failures))
        return None
    steps.append(WorkStep("rows", "completed", "", None))
    return rows


def _native(
    point: AdmittedPoint, failures: list[StandardFailure], steps: list[WorkStep]
) -> tuple[float | None, FloatArray | None]:
    """The criterion and the native gradient, or ``None`` after a range refusal."""
    try:
        criterion, gradient = native_gradient(point)
    except FloatingPointError as error:
        steps.append(_refused(GRADIENT, error, failures))
        return None, None
    counters = (("components", int(gradient.size)),)
    steps.append(WorkStep(GRADIENT, "completed", point.kind, counters))
    return criterion, gradient


def _effects(
    point: AdmittedPoint, failures: list[StandardFailure], steps: list[WorkStep]
) -> EffectEquations | None:
    """The fresh raw-weight level means, or ``None`` after a range refusal."""
    try:
        effects = effect_equations(point)
    except FloatingPointError as error:
        steps.append(_refused(EFFECTS, error, failures))
        return None
    steps.append(WorkStep("effect equations", "completed", "", None))
    return effects


def _inner_check(check: InnerCheck) -> StandardCheck:
    """An unchanged inner check, with the ratio of its value to each of its bounds."""
    value = check.value
    ratios = tuple(math.nan if value is None else value / b for b in check.bounds)
    return StandardCheck(
        check.name, value, check.bounds, ratios, check.passed, check.reason
    )


def _score(gradient: FloatArray | None) -> float | None:
    """``s``: the largest absolute component, ``0`` without one; ``None`` unformed."""
    if gradient is None:
        return None
    return float(np.max(np.abs(gradient))) if gradient.size else 0.0


def _checks(
    controls: StandardControls,
    gradient: FloatArray | None,
    effects: EffectEquations | None,
    inner: InnerEvidence | None,
) -> tuple[StandardCheck, ...]:
    """The numerical checks of a reading, in order: gradient, effects, inner."""
    score = _score(gradient)
    maximum = None if effects is None else effects.maximum
    found = [
        bounded(GRADIENT, score, (controls.gradient_tol,)),
        bounded(EFFECTS, maximum, (controls.tol,)),
    ]
    if inner is not None:
        found.extend(_inner_check(check) for check in inner.checks)
    return tuple(found)


def standard_point(
    whole: object,
    controls: CertificateControls,
    *,
    absorbed: AbsorbedOptions,
    gradient_tol: float,
) -> StandardPointAssessment:
    """The standard reading of one whole point (module docstring).

    ``TypeError`` or ``ValueError`` for a point that is not admitted, as
    ``certify_point`` refuses it.
    """
    point = admit_point(whole, absorbed)
    if point.sample.grouping is None:
        raise ValueError("an absorbed point needs fixed-effect blocks")
    basis = owned_basis(point.sample)
    steps = [WorkStep("basis", "completed", basis.route, (("rows", basis.rows),))]
    failures: list[StandardFailure] = []
    rows = _rows(point, failures, steps)
    criterion, gradient = _native(point, failures, steps)
    effects = _effects(point, failures, steps)
    support = inner = None
    if rows is not None:
        support = positive_support(basis, rows.curvature > 0)
        steps.append(support_step(support))
        if not support.full:
            failures.append(StandardFailure("support", SUPPORT))
        inner = inner_evidence(point, effects, rows.curvature, controls.tol)
        steps.extend(inner.steps)
    found = point.inner
    standard = StandardControls(
        gradient_tol,
        controls.tol,
        found.tolerance,
        found.max_sweeps,
        found.acceleration,
        found.profile_tol,
    )
    checks = _checks(standard, gradient, effects, inner)
    parts = (standard, point, basis, rows, effects, criterion, gradient, support, inner)
    return StandardPointAssessment(*parts, checks, tuple(failures), tuple(steps))


def opening_point(
    whole: object,
    controls: CertificateControls,
    *,
    absorbed: AbsorbedOptions,
    gradient_tol: float,
) -> StandardPointAssessment:
    """The opening-retry reading of one whole fitted-index point (module docstring).

    ``TypeError`` or ``ValueError`` for a point that is not admitted, as
    :func:`standard_point` refuses it; ``ValueError`` for a profiled point.
    """
    point = admit_point(whole, absorbed)
    if point.kind != "index" or point.sample.grouping is None:
        raise ValueError("an opening-retry reading needs a fitted-index point")
    basis = owned_basis(point.sample)
    steps = [WorkStep("basis", "completed", basis.route, (("rows", basis.rows),))]
    failures: list[StandardFailure] = []
    rows = _rows(point, failures, steps)
    criterion, gradient = _native(point, failures, steps)
    support = None
    if rows is not None:
        support = positive_support(basis, rows.curvature > 0)
        steps.append(support_step(support))
        if not support.full:
            failures.append(StandardFailure("support", SUPPORT))
    found = point.inner
    standard = StandardControls(
        gradient_tol,
        controls.tol,
        found.tolerance,
        found.max_sweeps,
        found.acceleration,
        found.profile_tol,
    )
    checks = (
        bounded(GRADIENT, _score(gradient), (standard.gradient_tol,)),
        not_run(EFFECTS, (standard.tol,)),
    )
    parts = (standard, point, basis, rows, None, criterion, gradient, support, None)
    return StandardPointAssessment(
        *parts,
        checks,
        tuple(failures),
        tuple(steps),
        policy=INDEX_RETRY_POLICY,
        kind=OPENING_RETRY_KIND,
        omitted=opening_omitted_checks(point.n_slopes),
    )


# ------------------------------------------------------------- native stop


def native_failures(history: CandidateHistory) -> tuple[StandardFailure, ...]:
    """Why the selected attempt is not a complete operative native stop; ``()`` if so.

    The operative and native reasons of the admitted history, its final work complete,
    no alarm, and every part of the final work present (the slope absorption with
    slopes); the profiled branch without slopes is ``profile_only`` with its complete
    final profile.
    """
    slopes = int(history.sample.active.size)
    profiled = history.route == "profiled"
    only = profiled and not slopes
    failures: list[StandardFailure] = []
    reason = "profile_only" if only else "native_stop"
    if history.reason != reason or history.native != (None if only else reason):
        text = f"{history.reason} (native {history.native}): {history.message}"
        failures.append(StandardFailure("native", text))
    final = history.final
    if not isinstance(final, FinalWork | IndexFinal):
        failures.append(StandardFailure("final", "no final work was run"))
    elif not final.complete:
        failures.append(StandardFailure("final", f"incomplete: {final.reason}"))
    if history.alarm is not None:
        failures.append(StandardFailure("alarm", "an indication stopped the attempt"))
    parts = (PROFILED_PARTS if profiled else INDEX_PARTS) + (
        ("absorbed",) if slopes else ()
    )
    missing = [part for part in parts if part not in history.partials]
    if missing:
        failures.append(StandardFailure("final", f"unreached final parts {missing}"))
    return tuple(failures)


def _point_failure(reading: StandardPointAssessment) -> tuple[StandardFailure, ...]:
    """The failure of a rejected point predicate, naming every failed part."""
    if reading.accepted:
        return ()
    texts = [f"{f.stage}: {f.reason}" for f in reading.failures]
    texts += [c.reason for c in reading.checks if not c.passed]
    return (StandardFailure("point", "; ".join(texts) or reading.verdict),)


# --------------------------------------------------------------- the assessor


@dataclass(frozen=True, eq=False)
class _Kept:
    """One call: its role, record, tokens and replica, reading and requested kind."""

    role: Role
    record: object
    tokens: tuple[Token, ...] | None
    replica: Replica | None
    reading: StandardPointAssessment | None
    wall: float
    kind: str = COMPLETE_KIND


@dataclass(frozen=True, eq=False)
class StandardSelection:
    """The selected point's standard reading and the standard convergence verdict.

    ``replica`` holds the owned inputs the reading read and ``capture`` the fresh
    capture of the admitted history's records (``None`` when refused); ``consumed``
    says whether the reading is a kept complete one and ``reason`` why not;
    ``readings`` are the detached summaries of the retry's readings in call order,
    ``selected`` that of the selected reading (a consumed one keeps its role, with no
    new seconds).
    """

    history: CandidateHistory
    controls: CertificateControls
    reading: StandardPointAssessment | None
    replica: Replica | None
    capture: Capture | None
    consumed: bool
    reason: str
    failures: tuple[StandardFailure, ...]
    readings: tuple[RetainedStandardReading, ...]
    selected: RetainedStandardReading | None
    status: StandardStatus


def standard_anchors(replica: Replica) -> Anchors:
    """The anchors of a standard replica's roots, in the order of the capture.

    The roots are the fit's sample, family, absorbed options, request, controls and
    ``gradient_tol``, then the point.
    """
    sample, variance, _, request, controls, _, point = replica.roots
    return Anchors(
        cast("FESample", sample),
        cast("Family", variance),
        cast("PointProfile | IndexPoint", point),
        cast("CovarianceRequest", request),
        cast("CertificateControls", controls),
    )


def _same_control(gradient_tol: object, fit: Replica) -> bool:
    """Whether ``gradient_tol`` captures to the very tokens of the fit's control."""
    found = capture((gradient_tol,))
    return found is not None and found.tokens == fit.capture.rest(FIT)


def _read(replica: Replica) -> StandardPointAssessment:
    """The standard reading of a replica's point, on the replica's records only."""
    mine = standard_anchors(replica)
    whole = WholePoint(mine.sample, mine.variance, mine.record)
    absorbed = cast("AbsorbedOptions", replica.roots[2])
    gradient_tol = cast("float", replica.roots[FIT])
    return standard_point(
        whole, mine.controls, absorbed=absorbed, gradient_tol=gradient_tol
    )


def _read_opening(replica: Replica) -> StandardPointAssessment:
    """The opening-retry reading of a replica's point, on the replica's records only."""
    mine = standard_anchors(replica)
    whole = WholePoint(mine.sample, mine.variance, mine.record)
    absorbed = cast("AbsorbedOptions", replica.roots[2])
    gradient_tol = cast("float", replica.roots[FIT])
    return opening_point(
        whole, mine.controls, absorbed=absorbed, gradient_tol=gradient_tol
    )


def _history(candidate: object) -> CandidateHistory:
    """The candidate's history, read afresh by its organization's existing reader."""
    if type(candidate) is ProfiledCandidate:
        return profiled_history(candidate)
    if type(candidate) is IndexCandidate:
        return index_history(candidate)
    raise TypeError("candidate must be a ProfiledCandidate or an IndexCandidate")


class StandardAssessor:
    """The standard assessor of one fit (module docstring); single selection.

    Parameters
    ----------
    fit : Replica
        The entry's replica of the fit's sample, family, absorbed options, request,
        certificate controls and ``gradient_tol``, in that order.
    request : CovarianceRequest
        The caller's request, captured with every point.
    absorbed : AbsorbedOptions
        The caller's absorbed options, those of the solver.
    options : SolverOptions
        The caller's solver options, those of the solver: each call reads their
        ``gradient_tol`` once and captures it with its point.
    """

    def __init__(
        self,
        fit: Replica,
        request: CovarianceRequest,
        absorbed: AbsorbedOptions,
        options: SolverOptions,
    ) -> None:
        self._fit = fit
        self._request = request
        self._absorbed = absorbed
        self._options = options
        self._anchors: tuple[object, object, object] | None = None
        self._kept: list[_Kept] = []
        self._selected = False
        self.capture = 0.0
        self.readings = 0

    def __call__(
        self,
        whole: WholePoint[PointProfile] | WholePoint[IndexPoint],
        controls: CertificateControls,
        /,
    ) -> Assessment:
        """The standard assessment of ``whole``: accepted, ``s`` and error."""
        role = ROLES[min(len(self._kept), len(ROLES) - 1)]
        reading = self._reading(whole, controls, role, COMPLETE_KIND)
        if reading is None:
            return Assessment(accepted=False, score=math.nan, error=math.nan)
        return Assessment(
            accepted=reading.accepted, score=reading.score, error=reading.error
        )

    def opening(
        self, whole: WholePoint[IndexPoint], controls: CertificateControls, /
    ) -> Assessment:
        """The opening-retry assessment of the retry rule's start (module docstring).

        ``Assessment(False, s, NaN)``: never accepted, ``s`` NaN when refused or when
        the native gradient was refused by range.
        """
        reading = self._reading(whole, controls, "opening", OPENING_RETRY_KIND)
        score = math.nan if reading is None else reading.score
        return Assessment(accepted=False, score=score, error=math.nan)

    def _reading(
        self,
        whole: WholePoint[PointProfile] | WholePoint[IndexPoint],
        controls: CertificateControls,
        role: Role,
        kind: str,
    ) -> StandardPointAssessment | None:
        """One call's reading of ``kind``, kept with its role; ``None`` when refused."""
        started = time.perf_counter()
        owned = None if self._selected else self._owned(whole, controls)
        self.capture += time.perf_counter() - started
        if owned is None:
            self._kept.append(_Kept(role, None, None, None, None, 0.0, kind))
            return None
        replica, record = owned
        read = _read_opening if kind == OPENING_RETRY_KIND else _read
        started = time.perf_counter()
        reading = read(replica)
        wall = time.perf_counter() - started
        self.readings += 1
        tokens = replica.capture.rest(STANDARD_FIT)
        self._kept.append(_Kept(role, record, tokens, replica, reading, wall, kind))
        return reading

    def _owned(
        self,
        whole: WholePoint[PointProfile] | WholePoint[IndexPoint],
        controls: CertificateControls,
    ) -> tuple[Replica, object] | None:
        """The replica of a call of this fit and its record, else ``None``."""
        parts = _parts(whole, controls, self._absorbed)
        if parts is None:
            return None
        sample, variance, point = parts
        anchors = (sample, variance, controls)
        if self._anchors is not None and not all(
            map(operator.is_, anchors, self._anchors)
        ):
            return None
        fit = (sample, variance, self._absorbed, self._request, controls)
        captured = capture((*fit, self._options.gradient_tol, point))
        replica = None if captured is None else replicate(captured, prefix=self._fit)
        if replica is None:
            return None
        if self._anchors is None:
            self._anchors = anchors
        return replica, point

    def select(
        self, candidate: ProfiledCandidate | IndexCandidate
    ) -> StandardSelection:
        """The selected point's reading and standard verdict (module docstring).

        ``ValueError`` for a candidate whose ``gradient_tol`` is not the fit's.
        """
        kept, self._kept, self._selected = self._kept, [], True
        history = _history(candidate)
        controls = certificate_controls(candidate.options)
        gradient = candidate.options.gradient_tol
        if not _same_control(gradient, self._fit):
            raise ValueError(CONTROL)
        readings = tuple(
            summary(k.reading, k.role, k.wall, CHANGED, k.kind) for k in kept
        )
        point = history.point
        if not isinstance(point, PointProfile | IndexPoint):
            return _unselected(history, controls, readings, NO_POINT, "unavailable")
        started = time.perf_counter()
        roots = (history.sample, candidate.variance, candidate.absorbed)
        captured = capture((*roots, self._request, controls, gradient, point))
        replica = None if captured is None else replicate(captured, prefix=self._fit)
        self.capture += time.perf_counter() - started
        if captured is None or replica is None:
            return _unselected(history, controls, readings, CHANGED, "refused")
        tokens = replica.capture.rest(STANDARD_FIT)
        found = next(
            (k for k in reversed(kept) if k.record is point and k.tokens == tokens),
            None,
        )
        role: Role = "selected_final"
        kept_reading = None if found is None else found.reading
        kept_replica = None if found is None else found.replica
        if (
            found is not None
            and kept_reading is not None
            and kept_replica is not None
            and kept_reading.complete
        ):
            reading, used, role, wall = kept_reading, kept_replica, found.role, 0.0
            reason = ""
        else:
            started = time.perf_counter()
            reading, used = _read(replica), replica
            wall = time.perf_counter() - started
            reason = NOT_KEPT
            self.readings += 1
        failures = (*native_failures(history), *_point_failure(reading))
        status: StandardStatus = "refused" if failures else "converged"
        return StandardSelection(
            history=history,
            controls=controls,
            reading=reading,
            replica=used,
            capture=replica.capture,
            consumed=not reason,
            reason=reason,
            failures=failures,
            readings=readings,
            selected=summary(reading, role, wall),
            status=status,
        )


def _unselected(
    history: CandidateHistory,
    controls: CertificateControls,
    readings: tuple[RetainedStandardReading, ...],
    reason: str,
    status: StandardStatus,
) -> StandardSelection:
    """The selection without a reading: no selected point, or inputs changed."""
    stage = "selection" if reason == NO_POINT else "ownership"
    return StandardSelection(
        history=history,
        controls=controls,
        reading=None,
        replica=None,
        capture=None,
        consumed=False,
        reason=reason,
        failures=(StandardFailure(stage, reason),),
        readings=readings,
        selected=None,
        status=status,
    )
