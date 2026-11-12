"""The internal full-joint certificate of absorbed points and candidates.

:func:`certify_point` certifies one point with explicit ``CertificateControls``, its
matching inner controls and an :class:`FEScaleProvider`.  :func:`certify_candidate`
audits an actual ``ProfiledCandidate`` or ``IndexCandidate`` (certificate controls from
``candidate.options``, inner controls ``candidate.absorbed``) and
:func:`certify_linear_candidate` an ``AbsorbedLinearCandidate`` with the caller's
controls (inner controls ``candidate.controls``; nothing inferred from ``af``).  An
audit owns the admitted history before certifying the selected point and keeps both,
with no combined verdict: no historical veto, no erasure.  :class:`FEAssessor` is the
unchanged point-only retry assessor.

Sequence: admission, the fixed ``J`` (constructed once per sample and reused only for
its unchanged grouping, :func:`~powermean.fe.certificate_basis.owned_basis`), fresh rows
and effect equations (on the admission's owned preparation where already formed),
inner checks, observed geometry, sparse measures; then, with
slopes and a formed correction, an
:class:`~powermean.solve.absorbed_certificate_scale.FEScaleInput` binding the geometry,
the requested covariance, the direct report map and ``W``; the provider's result is
admitted as an owned snapshot (:func:`~powermean.solve.absorbed_certificate_scale.
admitted_scale`), which the certificate keeps private and hands out as a fresh copy at
each access; the unchanged corrections and the coefficient gate follow.
``K = 0`` calls no provider: an empty coefficient restriction.
The work of each stage is recorded when it
completes, before a later refusal: the support check, the assembly, the equilibration.

Gates (a NaN passes none): ``evaluation``, ``effects`` (largest raw-weight level mean at
most ``tol``), ``inner``, ``curvature``, ``conditioning``, ``score`` and ``index`` (at
most the level's tolerance) and ``coefficient`` (vacuous without slopes); the
secondary level relaxes only score and index.  :func:`assessment` is the retry
reading.  The certificate's work is recorded apart from any solver's.

Selected roots.  An
exact root point of the full joint system outside ``tau in [1, 2]``
(:class:`~powermean.solve.absorbed_root_records.RootPoint`, kind ``"root"``) is
certified as a regular root, not as a minimum, by the same evidence: its original
rows and every original effect equation, the observed geometry, the full ordinary
factor and its conditioning, the full index correction with its ``B'(b - C dq)``
diagnostic, the provider's direct reported standard errors and the coefficient gate.
Two gates differ, each recorded in :attr:`FEPointCertificate.not_applicable`, never
counted as passed: ``inner`` does not apply (no absorption or profile forms the point,
whose inner evidence holds explicit inapplicable components), and ``curvature`` is
waived where the scaled score has a metric: ``Hplus`` where it is positive definite,
else the fixed merit metric of the solve (``merit``,
:func:`~powermean.solve.absorbed_certificate_sparse.sparse_measures`).  Without a
metric the curvature gate stands failed.  :func:`audit_root_candidate` audits an
:class:`~powermean.solve.absorbed_root_records.AbsorbedRootCandidate`: its admitted
history and the fresh certificate of its selected point on the target equation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, get_args

import numpy as np

from powermean.fe.certificate_basis import (
    CertificateBasis,
    owned_basis,
    positive_support,
)
from powermean.fe.profile_arithmetic import ProfileProblem
from powermean.model.rows import RowEvaluation
from powermean.solve.absorbed_certificate_geometry import (
    FEGeometry,
    assembled,
    assembly_step,
    fresh_geometry,
    support_step,
)
from powermean.solve.absorbed_certificate_history import (
    CandidateHistory,
    index_history,
    linear_history,
    profiled_history,
    root_history,
)
from powermean.solve.absorbed_certificate_inner import InnerEvidence, inner_evidence
from powermean.solve.absorbed_certificate_point import (
    AdmittedPoint,
    EffectEquations,
    FreshRows,
    WorkStep,
    _admit_numeric_point,
    _prepared_effect_equations_from_rows,
    _share_row_evaluation,
    admit_prepared_point,
    effect_equations,
    fresh_rows,
    prepared_effect_equations,
)
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleInput,
    FEScaleProvider,
    FEScaleResult,
    admitted_scale,
    bound_request,
    owned_result,
)
from powermean.solve.absorbed_certificate_sparse import SparseMeasures, sparse_measures
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.absorbed_root_records import (
    AbsorbedRootCandidate,
    MeritFactor,
    RootPoint,
    ScoreMetric,
)
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.linear_absorbed_point import LinearPoint
from powermean.solve.linear_absorbed_state import AbsorbedLinearCandidate
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.reduced_execution import (
    FinalFactor,
    ReducedExecution,
    _OwnedReducedExecution,
)
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.solve.reported import CoefficientCheck, coefficient_check
from powermean.systems.profiled import PointProfile, ProfiledProblem

FEGate = Literal[
    "evaluation",
    "effects",
    "inner",
    "curvature",
    "conditioning",
    "score",
    "index",
    "coefficient",
]
FE_GATES: tuple[FEGate, ...] = get_args(FEGate)
Level = Literal["strict", "secondary"]


# ---------------------------------------------------------------- certificate


@dataclass(frozen=True, eq=False)
class FEPointCertificate:
    """The fresh certificate of one absorbed point: its evidence, gates and work.

    ``rows``, ``effects``, ``inner``, ``geometry``, ``measures``, the admitted scale
    and ``coefficients`` are ``None`` when not formed (nothing follows a refusal of the
    rows); ``errors`` names each failure with a reason; ``strict`` and ``secondary``
    list the gates failed at ``tol`` and ``tol_secondary``.  The admitted scale is kept
    private: :attr:`scale` hands out a fresh owned copy of it.  ``merit`` is the fixed
    merit metric of a selected root's solve, where one was given.
    """

    controls: CertificateControls
    point: AdmittedPoint
    basis: CertificateBasis
    rows: FreshRows | None
    effects: EffectEquations | None
    inner: InnerEvidence | None
    geometry: FEGeometry | None
    measures: SparseMeasures | None
    _scale: FEScaleResult | None
    coefficients: CoefficientCheck | None
    errors: Mapping[FEGate, str]
    work: tuple[WorkStep, ...]
    strict: tuple[FEGate, ...] = ()
    secondary: tuple[FEGate, ...] = ()
    merit: MeritFactor | None = None

    @property
    def scale(self) -> FEScaleResult | None:
        """A fresh owned copy of the admitted scale result, ``None`` when not formed."""
        return None if self._scale is None else owned_result(self._scale)

    @property
    def score_metric(self) -> ScoreMetric:
        """The metric of the scaled score: ``Hplus``, the fixed merit, or none."""
        return "none" if self.measures is None else self.measures.score_metric

    @property
    def require_positive_definite(self) -> bool:
        """Whether the curvature gate applies: not at a selected root (docstring)."""
        return not self.point.selected_root

    @property
    def not_applicable(self) -> tuple[FEGate, ...]:
        """The gates that do not apply at the point, never counted as passed.

        At a selected root: ``inner``, and ``curvature`` where the scaled score has a
        metric (module docstring); nothing elsewhere.
        """
        if not self.point.selected_root:
            return ()
        found: set[FEGate] = {"inner"}
        if self.score_metric != "none":
            found.add("curvature")
        return tuple(gate for gate in FE_GATES if gate in found)

    @property
    def empty(self) -> bool:
        """Whether the point has no slope: an empty coefficient restriction."""
        return self.point.n_slopes == 0

    @property
    def level(self) -> Level | None:
        """``"strict"``, else ``"secondary"`` if that level passes, else ``None``."""
        if not self.strict:
            return "strict"
        return "secondary" if not self.secondary else None

    @property
    def accepted(self) -> bool:
        """Whether the point passes every gate at one of the two levels."""
        return self.level is not None

    @property
    def degenerate(self) -> bool:
        """Whether a reported standard error is zero: degenerate inference."""
        return self.coefficients is not None and self.coefficients.degenerate

    @property
    def scaled_score(self) -> float:
        """``||Hplus^-1/2 S||``, NaN where not formed."""
        return math.nan if self.measures is None else self.measures.scaled_score

    @property
    def index_correction(self) -> float:
        """The root mean square of ``dq``, NaN where not formed."""
        return math.nan if self.measures is None else self.measures.index_correction


def failed_gates(c: FEPointCertificate, tolerance: float) -> tuple[FEGate, ...]:
    """The gates that ``c`` does not pass with ``tolerance`` on score and index.

    A gate that does not apply at the point (:attr:`FEPointCertificate.
    not_applicable`) is neither failed nor passed.
    """
    measures, effects, check = c.measures, c.effects, c.coefficients
    passed: dict[FEGate, bool] = {
        "evaluation": c.geometry is not None,
        "effects": effects is not None and effects.maximum <= c.controls.tol,
        "inner": c.inner is not None and c.inner.passed,
        "curvature": measures is not None and measures.positive_definite,
        "conditioning": measures is not None and measures.conditioned,
        "score": c.scaled_score <= tolerance,
        "index": c.index_correction <= tolerance,
        "coefficient": c.empty or (check is not None and check.accepted),
    }
    skipped = c.not_applicable
    return tuple(gate for gate in FE_GATES if gate not in skipped and not passed[gate])


def _refused(name: str, error: Exception) -> WorkStep:
    """The step of an operation refused by range, with no counters."""
    return WorkStep(name, "refused", str(error), None)


_FRESH_ROW_HOOKS = (
    fresh_rows, prepared_effect_equations, _share_row_evaluation,
    _prepared_effect_equations_from_rows,
)
_INNER_EVIDENCE = inner_evidence


def _fresh(
    point: AdmittedPoint,
    errors: dict[FEGate, str],
    steps: list[WorkStep],
    prepared: ProfileProblem | None,
) -> tuple[FreshRows | None, EffectEquations | None]:
    """The fresh rows, then the effect equations; a range refusal stops the work."""
    evaluated: list[RowEvaluation | None] | None = None
    try:
        current = (fresh_rows, prepared_effect_equations, _share_row_evaluation,
                   _prepared_effect_equations_from_rows)
        if (prepared is not None
                and all(a is b for a, b in zip(current, _FRESH_ROW_HOOKS, strict=True))
                and _share_row_evaluation(point, prepared)):
            evaluated = [None]
    except MemoryError:
        # Only optional admission/collector allocation, never row or equation math.
        evaluated = None
    try:
        rows = (fresh_rows(point) if evaluated is None
                else fresh_rows(point, _evaluated=evaluated))
    except FloatingPointError as error:
        errors["evaluation"] = str(error)
        steps.append(_refused("rows", error))
        return None, None
    steps.append(WorkStep("rows", "completed", "", None))
    evaluation = None if evaluated is None else evaluated[0]
    try:
        effects = (
            effect_equations(point)
            if prepared is None
            else prepared_effect_equations(prepared, point.q) if evaluation is None
            else _prepared_effect_equations_from_rows(
                prepared, point.q, evaluation
            )
        )
    except FloatingPointError as error:
        errors["effects"] = str(error)
        steps.append(_refused("effect equations", error))
        return rows, None
    steps.append(WorkStep("effect equations", "completed", "", None))
    return rows, effects


def _geometry(
    point: AdmittedPoint,
    rows: FreshRows,
    basis: CertificateBasis,
    steps: list[WorkStep],
    reuse: FinalFactor | None = None,
) -> FEGeometry | str:
    """The observed geometry on the fixed ``J``, or the text of its range refusal.

    The work of each completed stage is recorded before a later refusal.
    """
    support = positive_support(basis, rows.curvature > 0)
    steps.append(support_step(support))
    try:
        assembly = (
            assembled(point, rows, basis) if reuse is None else reuse.assembly(rows)
        )
    except FloatingPointError as error:
        steps.append(_refused("assembly", error))
        return str(error)
    steps.append(assembly_step(assembly))
    try:
        geometry = fresh_geometry(point, rows, basis, support, assembly)
    except FloatingPointError as error:
        steps.append(_refused("equilibration", error))
        return str(error)
    steps.extend(geometry.steps())
    return geometry


def _measured(
    point: AdmittedPoint,
    rows: FreshRows,
    basis: CertificateBasis,
    controls: CertificateControls,
    scale: FEScaleProvider,
    *,
    factors: tuple[MeritFactor | None, FinalFactor | None],
    errors: dict[FEGate, str],
    steps: list[WorkStep],
) -> tuple[
    FEGeometry | None,
    SparseMeasures | None,
    FEScaleResult | None,
    CoefficientCheck | None,
]:
    """The geometry, the sparse measures and the coefficient gate of the fresh rows."""
    merit, reuse = factors
    geometry = _geometry(point, rows, basis, steps, reuse)
    if isinstance(geometry, str):
        errors["evaluation"] = geometry
        return None, None, None, None
    ordinary = None if reuse is None else (reuse.record, reuse.factor)
    measures = sparse_measures(geometry, controls.rcond_min, merit, ordinary=ordinary)
    steps.extend(measures.steps())
    for gate in ("curvature", "conditioning"):
        if gate in measures.errors:
            errors[gate] = measures.errors[gate]
    scaled, check = _coefficients(geometry, measures, scale, controls.coef_tol)
    return geometry, measures, scaled, check


def _coefficients(
    geometry: FEGeometry,
    measures: SparseMeasures,
    provider: FEScaleProvider,
    coef_tol: float,
) -> tuple[FEScaleResult | None, CoefficientCheck | None]:
    """The provider's admitted result and the coefficient gate.

    None without a correction.
    """
    k, correction, slope_map = (
        geometry.n_slopes,
        measures.correction,
        measures.slope_map,
    )
    order = geometry.point.sample.order_scale
    if not k or correction is None or slope_map is None or order is None:
        return None, None
    request = bound_request(provider.request, geometry)
    scale_input = FEScaleInput(geometry, request, order, slope_map)
    result = admitted_scale(provider(scale_input), scale_input)
    reported = result.evidence.reported
    corrections = reported.corrections(correction[:k])
    return result, coefficient_check(corrections, reported.standard_errors, coef_tol)


def _merit_of(point: AdmittedPoint, merit: object) -> MeritFactor | None:
    """The fixed merit of a selected root, checked; ``None`` without one.

    Refused (``ValueError``) at any other point, whose gates never read it, and for a
    record that is not a formed factor of the point's joint dimension.
    """
    if merit is None:
        return None
    if type(merit) is not MeritFactor:
        raise TypeError("merit must be a MeritFactor or None")
    if not point.selected_root:
        raise ValueError("a merit metric scales the score of a selected root only")
    basis_size = point.n_slopes + owned_basis(point.sample).n_kept
    usable = (
        merit.record.kind == "merit"
        and merit.record.status == "factored"
        and merit.dimension == basis_size
        and merit.diagonal.shape == (basis_size,)
        and merit.perm_r.shape == (basis_size,)
        and bool(np.isfinite(merit.l_data).all())
        and bool(np.isfinite(merit.scale).all())
        and bool((merit.diagonal > 0).all())
    )
    if not usable:
        raise ValueError("the merit is not a formed factor of the joint dimension")
    return merit


def certify_point(
    whole: object,
    controls: CertificateControls,
    *,
    absorbed: object,
    scale: FEScaleProvider,
    merit: MeritFactor | None = None,
) -> FEPointCertificate:
    """The fresh certificate of one absorbed point (module docstring).

    ``whole`` is a ``WholePoint[PointProfile]``, ``WholePoint[IndexPoint]``,
    ``WholePoint[RootPoint]`` or ``LinearPoint`` and ``absorbed`` its matching
    ``AbsorbedOptions`` or ``LinearControls``; ``merit`` the fixed merit of a selected
    root's solve.  ``TypeError`` or ``ValueError`` for an argument not of its type, a
    point not bound to its sample, a request not bound to its rows, a provider result
    not bound to its input or a merit given for another point.
    """
    call = _CertificateCall(whole, controls, absorbed, scale, merit)
    return _certify(call, None)


@dataclass(frozen=True)
class _CertificateCall:
    """Unchanged certification arguments, shared by checked and owned entry points."""

    whole: object
    controls: CertificateControls
    absorbed: object
    scale: FEScaleProvider
    merit: MeritFactor | None


def _certify_numeric_point(
    whole: object, controls: CertificateControls, *, absorbed: AbsorbedOptions,
    scale: FEScaleProvider, problem: ProfiledProblem,
    basis: CertificateBasis | None = None,
) -> FEPointCertificate:
    """Fresh original equations and gates using fixed operation-owned row inputs."""
    fresh = problem._fresh_certificate_problem()
    reduced = problem.__dict__.get("_reduced_execution")
    execution = (
        reduced if (kind := type(reduced)) is ReducedExecution
        or kind is _OwnedReducedExecution else None
    )
    return _certify(
        _CertificateCall(whole, controls, absorbed, scale, None),
        fresh, basis, execution,
    )


def _certify(
    call: _CertificateCall, numeric: ProfiledProblem | None,
    prepared_basis: CertificateBasis | None = None,
    execution: ReducedExecution | None = None,
) -> FEPointCertificate:
    """The single existing certificate implementation, not a second gate."""
    whole, controls, absorbed, scale, merit = (
        call.whole, call.controls, call.absorbed, call.scale, call.merit
    )
    if type(controls) is not CertificateControls:
        raise TypeError("certify with CertificateControls")
    request: object = getattr(scale, "request", None)
    if not callable(scale) or type(request) is not CovarianceRequest:
        raise TypeError("certify with an FEScaleProvider holding a CovarianceRequest")
    point, prepared = (
        admit_prepared_point(whole, absorbed) if numeric is None
        else _admit_numeric_point(whole, absorbed, numeric)
    )
    if point.sample.grouping is None:
        raise ValueError("an absorbed point needs fixed-effect blocks")
    fixed = _merit_of(point, merit)
    basis = owned_basis(point.sample) if prepared_basis is None else prepared_basis
    if prepared_basis is not None and basis.grouping is not point.sample.grouping:
        raise ValueError("the prepared basis belongs to another sample grouping")
    steps = [WorkStep("basis", "completed", basis.route, (("rows", basis.rows),))]
    backend = vars(absorbed).get("basis_backend") if type(absorbed) is AbsorbedOptions else None
    if type(backend) is str and backend == "cpp":
        execution_info = vars(basis).get("_basis_backend") if type(basis) is CertificateBasis else None
        actual, native_work, python_work, decline, attempted = (
            ("python", None, None, "ineligible", False) if execution_info is None else execution_info
        )
        counters = [("rows", basis.rows), ("native_attempted", int(attempted))]
        if native_work is not None:
            counters.append(("native_attempt_work", native_work))
        if python_work is not None:
            counters.append(("accepted_python_work", python_work))
        detail = f"{basis.route}; requested=cpp; actual={actual}; decline={decline}"
        steps[0] = WorkStep("basis", "completed", detail, tuple(counters))
    errors: dict[FEGate, str] = {}
    rows, effects = _fresh(point, errors, steps, prepared)
    del prepared
    inner = geometry = measures = scaled = check = None
    if rows is not None:
        projection = None
        try:
            if execution is not None and inner_evidence is _INNER_EVIDENCE:
                projection = (execution, rows, basis)
        except MemoryError:
            pass  # Optional handoff only; original inner checks remain required.
        inner = (
            inner_evidence(point, effects, rows.curvature, controls.tol,
                           _projection=projection)
            if projection is not None else
            inner_evidence(point, effects, rows.curvature, controls.tol)
        )
        del projection
        steps.extend(inner.steps)
        if inner.applicable and not inner.passed:
            errors["inner"] = inner.reason
        reuse = None if execution is None else execution.offer(point, rows, basis)
        geometry, measures, scaled, check = _measured(
            point, rows, basis, controls, scale, factors=(fixed, reuse),
            errors=errors, steps=steps,
        )
    parts = (controls, point, basis, rows, effects, inner, geometry, measures)
    found = FEPointCertificate(
        *parts, scaled, check, MappingProxyType(errors), tuple(steps), merit=fixed
    )
    return replace(
        found,
        strict=failed_gates(found, controls.tol),
        secondary=failed_gates(found, controls.tol_secondary),
    )


def assessment(certificate: FEPointCertificate) -> Assessment:
    """The retry reading of a certificate.

    ``accepted`` at either level, ``score`` the scaled score, ``error`` the larger of
    the scaled score and the index correction, NaN when either is NaN.
    """
    score, index = certificate.scaled_score, certificate.index_correction
    error = math.nan if math.isnan(score) or math.isnan(index) else max(score, index)
    return Assessment(accepted=certificate.accepted, score=score, error=error)


@dataclass(frozen=True, eq=False)
class FEAssessor:
    """The point-only retry assessor of a nonlinear solve, by :func:`certify_point`."""

    absorbed: AbsorbedOptions
    scale: FEScaleProvider

    def __call__(
        self,
        whole: WholePoint[PointProfile] | WholePoint[IndexPoint],
        controls: CertificateControls,
        /,
    ) -> Assessment:
        """The assessment of ``whole`` under ``controls``."""
        found = certify_point(whole, controls, absorbed=self.absorbed, scale=self.scale)
        return assessment(found)


# ---------------------------------------------------------------- candidates


@dataclass(frozen=True, eq=False)
class CandidateAudit:
    """The admitted history, the controls and the selected point's certificate.

    ``certificate`` is ``None`` when the candidate holds no point; no combined verdict.
    """

    history: CandidateHistory
    controls: CertificateControls
    certificate: FEPointCertificate | None


def certify_candidate(
    candidate: ProfiledCandidate | IndexCandidate, *, scale: FEScaleProvider
) -> CandidateAudit:
    """Audit a profiled or fitted-index candidate (module docstring)."""
    if type(candidate) is ProfiledCandidate:
        history = profiled_history(candidate)
    elif type(candidate) is IndexCandidate:
        history = index_history(candidate)
    else:
        raise TypeError("candidate must be a ProfiledCandidate or an IndexCandidate")
    controls = certificate_controls(candidate.options)
    point, certificate = history.point, None
    if isinstance(point, PointProfile | ConditionalPoint | IndexPoint):
        whole = WholePoint(history.sample, candidate.variance, point)
        absorbed = candidate.absorbed
        certificate = certify_point(whole, controls, absorbed=absorbed, scale=scale)
    return CandidateAudit(history, controls, certificate)


def certify_linear_candidate(
    candidate: AbsorbedLinearCandidate,
    controls: CertificateControls,
    *,
    scale: FEScaleProvider,
) -> CandidateAudit:
    """Audit an absorbed WLS candidate with the caller's explicit controls."""
    if type(controls) is not CertificateControls:
        raise TypeError("controls must be CertificateControls")
    history = linear_history(candidate)
    admitted, point, certificate = history.candidate, history.point, None
    if isinstance(point, LinearPoint) and isinstance(admitted, AbsorbedLinearCandidate):
        inner = admitted.controls
        certificate = certify_point(point, controls, absorbed=inner, scale=scale)
    return CandidateAudit(history, controls, certificate)


def audit_root_candidate(candidate: AbsorbedRootCandidate) -> CandidateAudit:
    """Audit an absorbed joint root candidate (module docstring).

    The admitted history (:func:`~powermean.solve.absorbed_certificate_history.
    root_history`) and the certificate that the solve formed afresh for its selected
    point on the target equation, with its fixed merit (as the joint root solve
    without effects concludes): it must be that very point's, under the controls of
    the candidate's options; ``None`` where the candidate holds no point.  No
    combined verdict; ``ValueError`` for a certificate of another point, equation,
    merit or controls.
    """
    if type(candidate) is not AbsorbedRootCandidate:
        raise TypeError("candidate must be an AbsorbedRootCandidate")
    history = root_history(candidate)
    controls = certificate_controls(candidate.options)
    certificate = candidate.certificate
    if certificate is not None:
        record: object = certificate.point.record
        bound = (
            type(certificate) is FEPointCertificate
            and type(record) is RootPoint
            and record is history.point
            and certificate.point.variance == candidate.variance
            and certificate.controls == controls
            and certificate.merit is candidate.merit
        )
        if not bound:
            raise ValueError(
                "the certificate is not the selected point's on the target equation"
            )
    return CandidateAudit(history, controls, certificate)
