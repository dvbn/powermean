"""The achieved inner accuracy of an admitted absorbed point, and its work.

Internal module: the checks read the
records of :mod:`powermean.solve.absorbed_certificate_point`, which imports nothing
from here; no history is read.

``af`` is the inner final tolerance.  A profiled point's fresh equations must be at
most ``profile_tol``.  An index point needs its same-``q`` recovery: the accepted
``recover`` at ``af`` with the inner budget and acceleration when it holds none, else
its owned witness verified afresh (``eta``, effects, both reconstruction gaps at most
``af``); the recovery absorption's fresh raw-weight check at most ``af`` and ``tol``.
A WLS point needs a fresh normalization gap at most ``af``. With slopes, the actual
owned same-final observed projection is rebuilt and freshly checked at every
original ``C``-weighted level against ``af`` AND ``tol``. No second development
solve is run on that matched route. Unmatched or
changed-hook points retain the cold auxiliary and its original budget semantics.
No cached success verdict replaces the current check. Each operation returns its
:class:`~powermean.solve.absorbed_certificate_point.WorkStep` with the counters it
returned; a refusal before an accepted routine counted anything has none.

Root points (inner accuracy is the achieved tolerance of the LAST absorption and
profile).  An exact point of the
full joint system has no such approximate stage: its nuisance coordinates are
unknowns of the sparse full system, and no C-weighted slope absorption runs (the
observed ``C`` of a nonconvex equation can be negative, outside the domain of
:func:`~powermean.fe.absorb_numeric.prepare_weights`).  Its evidence holds one
explicit NOT APPLICABLE check per component (the nuisance representation and, with
slopes, the auxiliary absorption): no value, no bound, no work step, a reason, and
``applicable`` false.  Such evidence never passes: :attr:`InnerEvidence.passed`
requires at least one applicable check and every applicable check passed, so an
empty or all-inapplicable record is no success; the certificate states that the gate
does not apply (:mod:`powermean.solve.absorbed_certificate`).  Approximate checks
that actually ran (the convex initializer's) stay in their own certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe.absorb import AbsorptionResult, absorb
from powermean.fe.absorb import AbsorptionState
from powermean.fe.absorb_numeric import (
    Attempted,
    PreparedWeights,
    candidate,
    checked_difference,
    fresh_check,
    prepare_weights,
)
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_point import (
    BOUND,
    AdmittedPoint,
    Counters,
    EffectEquations,
    FreshRows,
    InnerControls,
    StepStatus,
    WorkStep,
    _same,
    problem_of,
)
from powermean.solve.absorbed_direction import (
    IndexPoint,
    Recovery,
    normalize,
    reconstruct,
    recover,
)
from powermean.solve.linear_absorbed_point import LinearPoint, PointNormalization
from powermean.systems.profiled import ProfiledProblem
from powermean.data.sample_fe_state import FESample
from powermean.solve import reduced_execution as _reduced_execution
from powermean.solve.reduced_execution import _completed_projection

AUXILIARY = "auxiliary absorption"
NUISANCE = "nuisance representation"
EXACT_NUISANCE = (
    "not applicable: an exact full-system root; the nuisance coordinates of E_J are "
    "unknowns of the sparse joint system, not formed by an absorption, profile or "
    "recovery, so no achieved inner tolerance exists"
)
EXACT_SLOPES = (
    "not applicable: an exact full-system root; the full sparse joint Jacobian is "
    "factorized, no C-weighted slope absorption runs, and the observed curvature of "
    "a nonconvex equation is not an absorption weight"
)


def absorption_counters(result: AbsorptionResult) -> Counters:
    """The counters an accepted absorption returned."""
    names = ("sweeps", "visits", "operators", "diagnostic_passes")
    return tuple(
        zip(
            names,
            (result.sweeps, result.visits, result.operators, result.diagnostic_passes),
            strict=True,
        )
    )


@dataclass(frozen=True)
class InnerCheck:
    """One inner check: its measured value, bounds, verdict and reason.

    ``applicable`` is false for a component that does not exist at the point (an
    exact root point's): its value is unavailable, it has no bound, it does not pass,
    and its reason says why it does not apply.
    """

    name: str
    value: float | None
    bounds: tuple[float, ...]
    passed: bool
    reason: str
    applicable: bool = True


@dataclass(frozen=True, eq=False)
class InnerEvidence:
    """The inner checks, the records they read (``None`` when not formed), the work."""

    checks: tuple[InnerCheck, ...]
    recovery: Recovery | None
    normalization: PointNormalization | None
    auxiliary: AbsorptionResult | None
    steps: tuple[WorkStep, ...]

    @property
    def applicable(self) -> bool:
        """Whether an inner check applies at the point."""
        return any(check.applicable for check in self.checks)

    @property
    def passed(self) -> bool:
        """Whether an inner check applies and every applicable check passed.

        Never vacuous: evidence without an applicable check does not pass.
        """
        found = [check.passed for check in self.checks if check.applicable]
        return bool(found) and all(found)

    @property
    def reason(self) -> str:
        """The reason of the first failed applicable check, empty when all passed."""
        failed = (c for c in self.checks if c.applicable and not c.passed)
        return next((c.reason for c in failed), "")

    @property
    def not_applicable(self) -> tuple[str, ...]:
        """The reasons of the components that do not apply at the point."""
        return tuple(c.reason for c in self.checks if not c.applicable)


def _bounded(name: str, value: float | None, bounds: tuple[float, ...]) -> InnerCheck:
    """A check that ``value`` is at most every bound; a missing value fails."""
    passed = value is not None and all(value <= bound for bound in bounds)
    reason = "" if passed else f"{name}: {value} exceeds {bounds} or is unavailable"
    return InnerCheck(name, value, bounds, passed, reason)


def _failed(name: str, bounds: tuple[float, ...], reason: str) -> InnerCheck:
    """A failed check without a value."""
    return InnerCheck(name, None, bounds, False, f"{name}: {reason}")


def _inapplicable(name: str, reason: str) -> InnerCheck:
    """A component that does not exist at the point: no value, bound or pass."""
    return InnerCheck(name, None, (), False, f"{name}: {reason}", applicable=False)


def _checked(
    values: FloatArray, prepared: PreparedWeights, result: AbsorptionResult
) -> float:
    """The fresh maximum of an absorption's residual, rebuilt bit for bit."""
    residual = candidate(values, result.state.coefficients, prepared.grouping)
    if result.state.grouping is not prepared.grouping:
        raise ValueError(BOUND)
    _same(residual, result.residual)
    return fresh_check(residual, prepared).maximum


def _gap(
    problem: ProfiledProblem, point: AdmittedPoint, effects: tuple[FloatArray, ...]
) -> float:
    """``max |q_rec - q|`` of ``effects`` rebuilt in the accepted order."""
    rebuilt = reconstruct(problem, point.base, effects)
    return float(np.max(np.abs(checked_difference(rebuilt, point.q, "recovery gap"))))


def _recovered(
    point: AdmittedPoint, tol: float
) -> tuple[Recovery, WorkStep, list[InnerCheck]]:
    """The recovery (fresh when the point holds none), its work step, its checks."""
    record = point.record
    if type(record) is not IndexPoint or point.variance is None:
        raise ValueError(BOUND)
    problem = point._numeric_problem
    if problem is None:
        problem = problem_of(point.sample, point.variance)
    inner = point.inner
    recovery = record.recovered
    step = WorkStep("recovery witness", "completed", "", None)
    if recovery is None:
        recovery = recover(
            problem,
            record,
            tolerance=inner.tolerance,
            max_sweeps=inner.max_sweeps,
            acceleration=inner.acceleration,
        )
        found = recovery.absorption
        counters = None if found is None else absorption_counters(found)
        status: StepStatus = "completed" if recovery.complete else "failed"
        step = WorkStep("recovery", status, recovery.reason, counters)
    return recovery, step, _recovery_checks(problem, point, recovery, tol)


def _recovery_checks(
    problem: ProfiledProblem, point: AdmittedPoint, recovery: Recovery, tol: float
) -> list[InnerCheck]:
    """Both reconstruction gaps and the fresh recovery absorption check, rebuilt."""
    af, absorption = point.inner.tolerance, recovery.absorption
    if not recovery.complete or absorption is None or recovery.tolerance != af:
        return [_failed("recovery", (af,), recovery.reason or "incomplete")]
    try:
        eta = checked_difference(point.q, point.base, "nuisance index")
        _same(eta, recovery.eta)
        effects = tuple(a[:, 0] for a in absorption.state.coefficients)
        owned = recovery.effects or ()
        if len(owned) != len(effects):
            raise ValueError(BOUND)
        for mine, theirs in zip(owned, effects, strict=True):
            _same(theirs, mine)
        normalized = normalize(problem.grouping, effects).effects
        gap = max(_gap(problem, point, effects), _gap(problem, point, normalized))
        prepared = prepare_weights(problem.grouping, problem.weights)
        maximum = _checked(eta[:, None], prepared, absorption)
    except (FloatingPointError, ValueError) as error:
        reason = f"the witness does not reconstruct the point ({error})"
        return [_failed("recovery", (af,), reason)]
    return [
        _bounded("recovery gap", gap, (af,)),
        _bounded("recovery absorption", maximum, (af, tol)),
    ]


def _auxiliary(
    point: AdmittedPoint, curvature: FloatArray, tol: float, *, _projection=None,
) -> tuple[AbsorptionResult | None, WorkStep, InnerCheck]:
    """Freshly check actual final projection data, else the original cold route."""
    sample, inner = point.sample, point.inner
    bounds = (inner.tolerance, tol)
    if sample.design is None or sample.grouping is None:
        raise ValueError(BOUND)
    names, values = sample.active_names, sample.design.matrix
    try:
        prepared = prepare_weights(sample.grouping, curvature)
    except (FloatingPointError, ValueError) as error:
        refused = WorkStep(AUXILIARY, "refused", str(error), None)
        return None, refused, _failed(AUXILIARY, bounds, str(error))
    offered = None
    if _projection is not None:
        try:
            if _projection_original(point) and type(prepared) is PreparedWeights:
                execution, rows, basis = _projection
                if type(rows) is FreshRows:
                    offered = _completed_projection(
                        execution, point, rows, basis, values, names, prepared.weights
                    )
        except MemoryError:
            # Optional matching only; no selected witness arithmetic ran.
            offered = None
    if offered is not None:
        # A selected failed reconstruction/range/bound remains failed. Never
        # retry a cold solve after the actual fresh numerical witness begins.
        return _verified_auxiliary(point, prepared, values, names, offered, tol)
    try:
        result = absorb(
            values,
            prepared,
            names,
            tolerance=inner.tolerance,
            max_sweeps=inner.max_sweeps,
            acceleration=inner.acceleration,
        )
    except (FloatingPointError, ValueError) as error:
        refused = WorkStep(AUXILIARY, "refused", str(error), None)
        return None, refused, _failed(AUXILIARY, bounds, str(error))
    status: StepStatus = "completed" if result.converged else "failed"
    step = WorkStep(AUXILIARY, status, result.reason, absorption_counters(result))
    try:
        if result.state.columns != names:
            raise ValueError(BOUND)
        maximum = _checked(values, prepared, result)
    except (FloatingPointError, ValueError) as error:
        return result, step, _failed(AUXILIARY, bounds, str(error))
    if not result.converged:
        return (
            result,
            step,
            _failed(AUXILIARY, bounds, f"{result.status}: {result.reason}"),
        )
    return result, step, _bounded(AUXILIARY, maximum, bounds)


def _verified_auxiliary(point, prepared, values, names, offered, tol):
    """Independent current-C check; retain data, not the live factor/fit owner."""
    bounds = (point.inner.tolerance, tol)
    projection_work = tuple(
        ("projection_" + name, value) for name, value in absorption_counters(offered)
    )
    attempted = Attempted()
    detail = "actual same-final observed projection; fresh verification; no cold solve"
    try:
        state = AbsorptionState(prepared.grouping, names, offered.state.coefficients)
        residual = candidate(values, state.coefficients, prepared.grouping)
        _same(residual, offered.residual)
        checked = fresh_check(residual, prepared, attempted)
    except (FloatingPointError, ValueError) as error:
        counters = projection_work + (("verification_diagnostic_passes", attempted.count),)
        step = WorkStep(AUXILIARY, "failed", f"{detail}: {error}", counters)
        return None, step, _failed(AUXILIARY, bounds, str(error))
    passed = checked.maximum <= point.inner.tolerance
    reason = "" if passed else "the final projection fails the fresh auxiliary bound"
    result = AbsorptionResult(
        residual, state, "converged" if passed else "breakdown", reason,
        point.inner.tolerance, checked.column_errors, checked.maximum,
        offered.sweeps, offered.visits, offered.operators, attempted.count,
        point.inner.acceleration,
    )
    counters = projection_work + (("verification_diagnostic_passes", attempted.count),)
    step = WorkStep(AUXILIARY, "completed" if passed else "failed", detail, counters)
    return result, step, _bounded(AUXILIARY, checked.maximum, bounds)


_PROJECTION_TYPES = (
    AdmittedPoint, FreshRows, InnerControls, FESample, Attempted, PreparedWeights,
)
_PROJECTION_MEMBERS = tuple(
    (kind, name, vars(kind).get(name)) for kind, names in (
        (AdmittedPoint, ("__getattribute__", "__getattr__", "sample", "inner",
                         "_numeric_problem", "q", "variance")),
        (FreshRows, ("__getattribute__", "__getattr__", "curvature", "weights")),
        (InnerControls, ("__getattribute__", "__getattr__", "tolerance", "acceleration")),
        (FESample, ("__getattribute__", "__getattr__", "design", "active_names", "grouping")),
        (Attempted, ("__getattribute__", "__getattr__", "__init__", "begin", "count")),
        (PreparedWeights, ("__getattribute__", "__getattr__", "weights")),
    ) for name in names
)
_PROJECTION_PRODUCERS = (
    absorb, prepare_weights, candidate, fresh_check, _same, _checked,
    _verified_auxiliary, _completed_projection, _reduced_execution._completed_projection,
    AbsorptionState, AbsorptionResult, Attempted,
)


def _projection_original(point):
    """No new generic/foreign callbacks before the original auxiliary work."""
    return (
        all(a is b for a, b in zip(
            (AdmittedPoint, FreshRows, InnerControls, FESample, Attempted, PreparedWeights),
            _PROJECTION_TYPES, strict=True
        ))
        and type(point) is AdmittedPoint
        and all(vars(kind).get(name) is original
                for kind, name, original in _PROJECTION_MEMBERS)
        and type(point.sample) is FESample
        and type(point.inner) is InnerControls
        and all(a is b for a, b in zip((
            absorb, prepare_weights, candidate, fresh_check, _same, _checked,
            _verified_auxiliary, _completed_projection, _reduced_execution._completed_projection,
            AbsorptionState, AbsorptionResult, Attempted,
        ), _PROJECTION_PRODUCERS, strict=True))
    )


def _normalization(
    point: AdmittedPoint,
) -> tuple[PointNormalization | None, InnerCheck]:
    """The fresh WLS normalization and its gap check."""
    bounds = (point.inner.tolerance,)
    if type(point.record) is not LinearPoint:
        raise ValueError(BOUND)
    try:
        normalization = PointNormalization(point.record)
    except FloatingPointError as error:
        return None, _failed("normalization", bounds, str(error))
    return normalization, _bounded("normalization", normalization.gap, bounds)


def inner_evidence(
    point: AdmittedPoint,
    effects: EffectEquations | None,
    curvature: FloatArray,
    tol: float,
    *, _projection=None,
) -> InnerEvidence:
    """Every inner check of the point (module docstring), in a fixed order.

    ``curvature`` is the fresh ``C``; ``effects`` is ``None`` after its range refusal.
    A root point has no approximate component: its evidence is the explicit
    inapplicable checks of the module docstring, with no work and no record read.
    """
    checks: list[InnerCheck] = []
    steps: list[WorkStep] = []
    recovery: Recovery | None = None
    normalization: PointNormalization | None = None
    if point.selected_root:
        checks.append(_inapplicable(NUISANCE, EXACT_NUISANCE))
        if point.n_slopes:
            checks.append(_inapplicable(AUXILIARY, EXACT_SLOPES))
        return InnerEvidence(tuple(checks), None, None, None, ())
    if point.kind in ("profiled", "reduced"):
        value = None if effects is None else effects.maximum
        checks.append(_bounded("profile", value, (point.inner.profile_tol or 0.0,)))
    elif point.kind == "index":
        recovery, step, found = _recovered(point, tol)
        steps.append(step)
        checks += found
    else:
        normalization, check = _normalization(point)
        checks.append(check)
    auxiliary: AbsorptionResult | None = None
    if point.n_slopes:
        auxiliary, step, check = (
            _auxiliary(point, curvature, tol, _projection=_projection)
            if _projection is not None and _auxiliary is _PROJECTION_AUXILIARY
            else _auxiliary(point, curvature, tol)
        )
        steps.append(step)
        checks.append(check)
    return InnerEvidence(
        tuple(checks), recovery, normalization, auxiliary, tuple(steps)
    )


_PROJECTION_AUXILIARY = _auxiliary
