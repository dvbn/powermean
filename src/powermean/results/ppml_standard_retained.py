"""The detached retained state of a standard PPML fit and its narrow admission.

Internal: only the standard route of :mod:`powermean.interface.ppml_audit` calls it.  A
standard record is never an ``FEPointCertificate`` and never passes through
``absorbed_binding._check_scale``.

Binding (:func:`bound_standard`).  Only a complete reading binds: an opening-retry
reading (:mod:`powermean.results.ppml_standard_records`) is never retained, whatever its
records.  The selected reading and its inference were formed on an owned replica whose
capture equals the admitted history's records.  Exactly the records that hold an input
are re-formed with each replica anchor replaced by the admitted object at its place, by
the rebinding of the within-fit handoff
(:mod:`powermean.results.absorbed_assessment_binding`): the admitted point takes the
sample, family and record; the basis the blocks; the inner evidence's absorptions the
blocks; the geometry the point and basis; the scale input is formed afresh from the
re-formed geometry and the caller's request, and the provider's evidence is admitted
under it only when its original binding, renamed, is that binding item for item.
Nothing computed is recomputed.

Admission (:func:`admit_standard`), narrow and typed, against the admitted records:
first a complete reading, an opening-retry reading refused before any of its evidence is
read; the admitted point's ``theta``, base and ``q`` bit for bit; the reading's
controls, ``gradient_tol`` and ``tol`` those of the candidate's solver options and the
inner controls the admitted point's, bit for bit, and its native-gradient and effects
checks bounded by exactly those two, their verdicts included; the fresh rows, criterion,
native gradient and raw-weight level means recomputed on the admitted records, bit for
bit; the sample's canonical basis, field for field; the scale input's geometry, ``W``,
direct report map, request identity and full binding; the inference status against its
standard errors and the slopes; the inference's method: a full-joint inference holds no
residualized evidence, and a residualized one admits its scalar evidence against the
reading and its own map and geometry
(:func:`~powermean.results.ppml_residualized_inference.admit_residualized`, no numerical
work).  ``ValueError`` refuses a record that is not bound.

Detachment (:func:`compose_standard_envelope`).  Finite typed snapshots only, by the
snapshot utilities: the shared sample, outcome, history and point snapshots of the
interface admission; the standard evidence (checks, failures, work, criterion and every
gradient component, support, basis, level means, inner evidence, and with
``keep_influence`` the rows and geometry); the inference record (ordinary factor record,
counted solves, provider calls, the omitted no-call checks, the requested and executed
method and the residualized route's owned scalar evidence); the initial covariance and
direct-SE cache; with ``keep_influence`` the influence rows and ``W``.  No solver,
provider, SuperLU, caller array or live sample is held.  With ``keep_influence=False``
the cache stays and the influence, ``W``, rows and geometry arrays are not retained, so
no diagnostic of this record reconstructs them; a new covariance request, the influence,
``W``, the scores and the expected information refuse with
``specification.influence_discarded``.  A new request on retained influence uses
``fe_covariance`` and the direct map, as the full route's service does.  The envelope
and its selection carry the policy of the fit's route (``fit_policy``: the opening-retry
policy on the index route), the evidence its complete reading's tag.

The internal equation-aware stop (:mod:`powermean.solve.newton_absorbed`).  Its probes
are bound to the solve that made them (:func:`detach_probes`): the receipt's bounds are
the fit's captured ``tol`` and ``gradient_tol`` bit for bit, and every probe names an
attempt of the candidate and one of its accepted steps, in strictly increasing order; it
records finite values of the two native formulas passing, its own bounds, an outcome
that its finite nonnegative maximum (or its refusal and reason) gives, its step's
combined flag (``True`` at its attempt's last step only) and finite nonnegative own
seconds; at its attempt's last step, the SHA-256 of its ``theta``, base and ``q`` are
those of that attempt's point.  ``ValueError`` refuses a record that is not bound.
Bound, the probes are detached as owned records of scalars and strings, refused fits'
included, and the envelope and its selection then carry the equation-stop policy
(``fit_policy`` with ``stop``).  The envelope holds probes exactly under that policy, on
the index route: neither is relabelled as the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np

from powermean.data.sample_fe_state import FESample, same_bits
from powermean.errors import SpecificationError
from powermean.fe.certificate_basis import CertificateBasis, owned_basis
from powermean.fe.certificate_basis import _same_structure as same_structure
from powermean.inference.absorbed import fe_covariance
from powermean.inference.provider import direct_standard_errors
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.absorbed_assessment_binding import (
    Anchors,
    _inner,
    _scale,
    _UnboundError,
)
from powermean.results.absorbed_binding import _same_basis_indices, _same_basis_rank
from powermean.results.absorbed_geometry import (
    ExpectedInformationInputs,
    snapshot_geometry,
    snapshot_rows,
)
from powermean.results.absorbed_geometry_records import snapshot_basis
from powermean.results.absorbed_groupings import GroupingIdentityMap
from powermean.results.absorbed_handoff import (
    _sampling_units,
    check_cluster_request_rows,
    matches_initial_covariance_request,
)
from powermean.results.absorbed_history import RetainedHistorySnapshot
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_owned import (
    RetainedEffectEquations,
    _snapshot_inner_evidence,
)
from powermean.results.absorbed_point_records import RetainedPointSnapshot
from powermean.results.absorbed_points import PointIdentityMap
from powermean.results.absorbed_sample import RetainedSampleSnapshot, SampleIdentityMap
from powermean.results.absorbed_sample_binding import UNBOUND
from powermean.results.absorbed_scale_cache import (
    RetainedInitialScaleCache,
    snapshot_covariance_request,
    snapshot_initial_scale,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.ppml_residualized_inference import admit_residualized
from powermean.results.ppml_standard_assessment import (
    StandardSelection,
    native_gradient,
)
from powermean.results.ppml_standard_inference import standard_error_status
from powermean.results.ppml_standard_records import (
    EFFECTS,
    GRADIENT,
    INDEX_STOP_POLICY,
    POLICY,
    InferenceStatus,
    RetainedStandardEvidence,
    RetainedStandardInference,
    RetainedStandardSelection,
    RetainedStopProbes,
    StandardInference,
    StandardPointAssessment,
    StandardStatus,
    fit_policy,
)
from powermean.results.retained_input_values import owned_order_scale
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_history import CandidateHistory
from powermean.solve.absorbed_certificate_point import (
    AdmittedPoint,
    admit_point,
    effect_equations,
    fresh_rows,
)
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    admitted_scale,
    bound_request,
)
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import (
    EquationProbe,
    IndexAttempt,
    IndexCandidate,
    StopProbes,
    array_digest,
)
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.reported import ReportedScale
from powermean.systems.profiled import PointProfile

NOT_FORMED = "the requested inference was not formed"
METHOD = "the full-joint inference holds the evidence of another method"
PROBES = "the equation-stop probes do not bind to the solve's attempts"
RELABELLED = "the equation-stop policy and its probes go together, on the index route"
Maps = tuple[PointIdentityMap, SampleIdentityMap, GroupingIdentityMap]
Snapshots = tuple[
    RetainedSampleSnapshot, RetainedOutcomeSnapshot | None, RetainedHistorySnapshot
]


# ----------------------------------------------------------------- binding


@dataclass(frozen=True, eq=False)
class BoundStandard:
    """A standard reading and its inference, held by the admitted records."""

    reading: StandardPointAssessment
    inference: StandardInference


def _formed_on(
    reading: StandardPointAssessment, inference: StandardInference, mine: Anchors
) -> bool:
    """Whether the reading and inference hold the replica's anchors in their places."""
    point, basis, geometry = reading.point, reading.basis, inference.geometry
    links = geometry is None or (
        geometry.point is point
        and geometry.basis is basis
        and geometry.rows is reading.rows
    )
    return (
        links
        and point.sample is mine.sample
        and point.variance is mine.variance
        and point.record is mine.record
        and basis.grouping is mine.grouping
    )


def bound_standard(
    reading: StandardPointAssessment,
    inference: StandardInference,
    mine: Anchors,
    live: Anchors,
) -> BoundStandard | None:
    """``reading`` and ``inference`` formed on ``mine``, bound to ``live``, or None.

    ``None`` for a reading that is not complete (module docstring).
    """
    if not reading.complete:
        return None
    if live.grouping is None or not _formed_on(reading, inference, mine):
        return None
    point = replace(
        reading.point, sample=live.sample, variance=live.variance, record=live.record
    )
    basis = replace(reading.basis, grouping=live.grouping)
    geometry = inference.geometry
    if geometry is not None:
        geometry = replace(geometry, point=point, basis=basis)
    try:
        inner = _inner(reading.inner, mine, live)
        scale = _scale(inference.scale, geometry, mine, live)
    except _UnboundError:
        return None
    return BoundStandard(
        replace(reading, point=point, basis=basis, inner=inner),
        replace(inference, geometry=geometry, scale=scale),
    )


# --------------------------------------------------------------- admission


def _same_point(point: AdmittedPoint, admitted: AdmittedPoint) -> None:
    """``ValueError`` unless ``point`` is the admitted point, bit for bit."""
    meta = (point.kind, point.variance, point.inner)
    if meta != (admitted.kind, admitted.variance, admitted.inner) or not (
        same_bits(point.theta, admitted.theta)
        and same_bits(point.base, admitted.base)
        and same_bits(point.q, admitted.q)
    ):
        raise ValueError(UNBOUND)


def _same_float(found: float | None, given: float | None) -> bool:
    """Whether two optional floats are both absent or the same bits."""
    if found is None or given is None:
        return found is given
    return same_bits(np.array([found]), np.array([given]))


def _same_controls(
    reading: StandardPointAssessment,
    history: CandidateHistory,
    admitted: AdmittedPoint,
) -> None:
    """The reading's controls are the candidate's and their checks' bounds (docstring).

    ``ValueError`` unless the reading's ``gradient_tol`` and ``tol`` are those of the
    candidate's solver options and its inner controls the admitted point's, and its
    native-gradient and effects checks hold exactly these two bounds and verdicts.
    """
    candidate = history.candidate
    if not isinstance(candidate, ProfiledCandidate | IndexCandidate):
        raise ValueError(UNBOUND)
    options, inner, controls = candidate.options, admitted.inner, reading.controls
    found = (controls.gradient_tol, controls.tol, controls.absorb_tol_final)
    given = (options.gradient_tol, options.tol, inner.tolerance)
    settings = (controls.absorb_max_sweeps, controls.absorb_acceleration)
    if (
        not all(map(_same_float, found, given))
        or not _same_float(controls.profile_tol, inner.profile_tol)
        or settings != (inner.max_sweeps, inner.acceleration)
    ):
        raise ValueError(UNBOUND)
    checks = reading.checks[:2]
    bounds = ((GRADIENT, controls.gradient_tol), (EFFECTS, controls.tol))
    if len(checks) != len(bounds) or not all(
        check.name == name
        and len(check.bounds) == 1
        and _same_float(check.bounds[0], bound)
        and check.passed is bool(check.value is not None and check.value <= bound)
        for check, (name, bound) in zip(checks, bounds, strict=True)
    ):
        raise ValueError(UNBOUND)


def _same_rows(reading: StandardPointAssessment, admitted: AdmittedPoint) -> None:
    """The fresh rows and native gradient of the admitted point are the reading's."""
    rows = reading.rows
    try:
        found = fresh_rows(admitted)
    except FloatingPointError:
        if rows is not None:
            raise ValueError(UNBOUND) from None
    else:
        if rows is None or not (
            same_bits(rows.b, found.b)
            and same_bits(rows.curvature, found.curvature)
            and same_bits(rows.weights, found.weights)
            and rows.frequency == found.frequency
        ):
            raise ValueError(UNBOUND)
    gradient = reading.gradient
    try:
        criterion, fresh = native_gradient(admitted)
    except FloatingPointError:
        if gradient is not None or reading.criterion is not None:
            raise ValueError(UNBOUND) from None
        return
    same = gradient is not None and same_bits(fresh, gradient)
    if not (same and _same_float(criterion, reading.criterion)):
        raise ValueError(UNBOUND)


def _same_effects(reading: StandardPointAssessment, admitted: AdmittedPoint) -> None:
    """The fresh raw-weight level means of the admitted point are the reading's."""
    effects = reading.effects
    try:
        found = effect_equations(admitted)
    except FloatingPointError:
        if effects is not None:
            raise ValueError(UNBOUND) from None
        return
    if (
        effects is None
        or len(effects.means) != len(found.means)
        or not all(
            same_bits(a, b) for a, b in zip(effects.means, found.means, strict=True)
        )
        or not _same_float(effects.maximum, found.maximum)
    ):
        raise ValueError(UNBOUND)


def _same_basis(basis: CertificateBasis, sample: FESample) -> None:
    """The reading's basis is the sample's canonical one, field for field."""
    expected = owned_basis(sample)
    if (
        type(basis) is not CertificateBasis
        or basis.grouping is not sample.grouping
        or not same_structure(
            (basis.route, basis.rows, basis.echelon, basis.witnesses),
            (expected.route, expected.rows, expected.echelon, expected.witnesses),
        )
        or not _same_basis_rank(basis.rank, expected.rank)
        or not (
            _same_basis_indices((basis.selected,), (expected.selected,))
            and _same_basis_indices(basis.references, expected.references)
            and _same_basis_indices(basis.columns, expected.columns)
            and _same_basis_indices(basis.positions, expected.positions)
        )
    ):
        raise ValueError(UNBOUND)


def _same_scale(
    bound: BoundStandard, sample: FESample, request: CovarianceRequest
) -> None:
    """The scale input and evidence bind the geometry, ``W``, map and request."""
    reading, inference = bound.reading, bound.inference
    scale, geometry, slopes = inference.scale, inference.geometry, inference.slope_map
    if scale is None:
        if inference.status in ("available", "degenerate"):
            raise ValueError("an available inference holds its admitted scale")
        return
    order = sample.order_scale
    if geometry is None or slopes is None or order is None:
        raise ValueError(UNBOUND)
    found = scale.input
    admitted_scale(scale, found)
    links = (
        found.geometry is geometry
        and geometry.point is reading.point
        and geometry.rows is reading.rows
        and geometry.basis is reading.basis
    )
    maps = found.order.omega == order.omega and same_bits(
        found.order.scale, order.scale
    )
    if not (links and maps and same_bits(found.slope_map, slopes)):
        raise ValueError(UNBOUND)
    identity = bound_request(found.request, geometry).identity()
    if identity != request.identity():
        raise ValueError("the scale answers another covariance request")
    expected = (
        *geometry.binding(),
        ("W", (slopes.shape, slopes.tobytes())),
        ("order", (order.omega, order.scale.tobytes())),
        ("request", identity),
    )
    if found.binding.items != expected or scale.evidence.binding.items != expected:
        raise ValueError(UNBOUND)
    errors = scale.evidence.reported.standard_errors
    if standard_error_status(errors)[0] != inference.status:
        raise ValueError("the inference status does not match its standard errors")


def _same_method(bound: BoundStandard) -> None:
    """The inference's method and its evidence (module docstring); ``ValueError``.

    A full-joint inference holds no residualized evidence; any other method name is
    that of the residualized route, whose scalar evidence must admit against the
    reading (:func:`~powermean.results.ppml_residualized_inference
    .admit_residualized`).
    """
    inference = bound.inference
    if (inference.requested, inference.executed) == ("full_joint", "full_joint"):
        if inference.residualized is not None:
            raise ValueError(METHOD)
        return
    admit_residualized(inference, bound.reading)


def admit_standard(
    bound: BoundStandard,
    history: CandidateHistory,
    absorbed: AbsorbedOptions,
    request: CovarianceRequest,
) -> None:
    """The narrow standard admission (module docstring); ``ValueError`` if unbound."""
    reading, inference = bound.reading, bound.inference
    if not reading.complete:
        raise ValueError(UNBOUND)
    point, variance, record = reading.point, history.variance, history.point
    if (
        variance is None
        or not isinstance(record, PointProfile | IndexPoint)
        or point.sample is not history.sample
        or point.record is not record
        or point.variance is not variance
    ):
        raise ValueError(UNBOUND)
    admitted = admit_point(WholePoint(history.sample, variance, record), absorbed)
    _same_point(point, admitted)
    _same_controls(reading, history, admitted)
    _same_rows(reading, admitted)
    _same_effects(reading, admitted)
    _same_basis(reading.basis, history.sample)
    _same_scale(bound, history.sample, request)
    _same_method(bound)
    if (inference.status == "not_applicable") != (point.n_slopes == 0):
        raise ValueError("the inference status does not match the slopes")
    formed = inference.status in ("available", "degenerate", "refused")
    if formed and not reading.accepted:
        raise ValueError("inference was formed at a point that is not accepted")


# ------------------------------------------------------------- stop probes


def _coherent(probe: EquationProbe) -> bool:
    """Whether a probe's outcome is the one its maximum and bound, or refusal, give."""
    maximum = probe.maximum
    if maximum is None:
        return probe.outcome == "refused" and bool(probe.reason)
    if not math.isfinite(maximum) or maximum < 0.0:
        return False
    expected = "pass" if maximum <= probe.tol else "continue"
    return probe.outcome == expected and not probe.reason and probe.base is not None


def _at_point(probe: EquationProbe, attempt: IndexAttempt) -> bool:
    """Whether a probe of its attempt's last step binds that attempt's point bytes."""
    point = attempt.point
    if probe.step != attempt.iterations or point is None:
        return True
    found = (array_digest(point.theta), array_digest(point.base))
    return (probe.theta, probe.base, probe.q) == (*found, array_digest(point.index))


def _bound_probe(
    probe: object, receipt: StopProbes, attempts: tuple[IndexAttempt, ...]
) -> bool:
    """Whether one probe binds to the solve of its receipt (module docstring)."""
    if type(probe) is not EquationProbe or not 0 <= probe.attempt < len(attempts):
        return False
    attempt, bound = attempts[probe.attempt], probe.gradient_tol
    last = probe.step == attempt.iterations
    return (
        1 <= probe.step <= attempt.iterations
        and type(probe.stopped) is bool
        and (last or not probe.stopped)
        and all(
            math.isfinite(value)
            for value in (probe.score, probe.decrement, probe.wall, probe.cpu)
        )
        and probe.score <= bound
        and probe.decrement <= bound
        and _same_float(probe.tol, receipt.tol)
        and _same_float(bound, receipt.gradient_tol)
        and probe.wall >= 0.0
        and probe.cpu >= 0.0
        and _coherent(probe)
        and _at_point(probe, attempt)
    )


def detach_probes(
    receipt: StopProbes, candidate: object, captured: tuple[float, float]
) -> RetainedStopProbes:
    """The detached probes of the solve of ``candidate`` (module docstring).

    ``captured`` holds the fit's captured ``tol`` and ``gradient_tol``.  ``ValueError``
    unless the receipt's bounds are those bit for bit and every probe binds to its
    attempt, accepted step and point, in strictly increasing order.
    """
    if type(receipt) is not StopProbes or type(candidate) is not IndexCandidate:
        raise ValueError(PROBES)
    bounds = (receipt.tol, receipt.gradient_tol)
    attempts, probes = candidate.attempts, receipt.probes
    if not all(map(_same_float, bounds, captured)) or not all(
        _bound_probe(probe, receipt, attempts) for probe in probes
    ):
        raise ValueError(PROBES)
    places = [(probe.attempt, probe.step) for probe in probes]
    if places != sorted(set(places)):
        raise ValueError(PROBES)
    owned = tuple(replace(probe) for probe in probes)
    return RetainedStopProbes(INDEX_STOP_POLICY, *bounds, owned)


# -------------------------------------------------------------- the envelope


@dataclass(frozen=True)
class StandardRetainedEnvelope:
    """The detached retained state of one standard fit (module docstring).

    ``probes`` holds the bound probes of the private equation-aware stop, exactly when
    ``policy`` is the equation-stop policy, on the index route only.
    """

    route: Literal["profiled", "index"]
    keep_influence: bool
    standard_status: StandardStatus
    inference_status: InferenceStatus
    sample: RetainedSampleSnapshot
    outcome: RetainedOutcomeSnapshot | None
    history: RetainedHistorySnapshot
    selected_point: RetainedPointSnapshot | None
    selection: RetainedStandardSelection
    evidence: RetainedStandardEvidence | None
    inference: RetainedStandardInference
    _order: OrderScale | None
    initial_scale: RetainedInitialScaleCache | None
    _observed_influence: FloatArray | None = None
    _slope_map: FloatArray | None = None
    policy: str = POLICY
    probes: RetainedStopProbes | None = None

    def __post_init__(self) -> None:
        """Own the arrays; nothing of the influence is kept without keep_influence.

        ``ValueError`` (a relabelled policy) unless the probes and the equation-stop
        policy go together, on the index route, the probes carrying that policy.
        """
        probes, stopped = self.probes, self.policy == INDEX_STOP_POLICY
        if stopped != (probes is not None) or (
            probes is not None
            and (self.route != "index" or probes.policy != INDEX_STOP_POLICY)
        ):
            raise ValueError(RELABELLED)
        object.__setattr__(self, "_order", owned_order_scale(self._order))
        keep = bool(self.keep_influence)
        object.__setattr__(self, "keep_influence", keep)
        cache = self.initial_scale
        if not keep and cache is not None:
            object.__setattr__(self, "initial_scale", cache.without_influence())
        for name in ("_observed_influence", "_slope_map"):
            value = getattr(self, name)
            object.__setattr__(self, name, _optional(value) if keep else None)

    @property
    def order(self) -> OrderScale | None:
        """A fresh owned direct report map, or ``None``."""
        return owned_order_scale(self._order)

    @property
    def empty(self) -> bool:
        """Whether the model has no slope parameters (``K = 0``)."""
        return self.sample.n_slopes == 0

    def require_influence(self, request: str) -> None:
        """``specification.influence_discarded`` when the influence was discarded."""
        if not self.keep_influence:
            raise SpecificationError(
                "specification.influence_discarded", {"request": request}
            )

    def coef(self) -> FloatArray:
        """A fresh copy of the reported slope coefficients of the selected point."""
        if self.selected_point is None:
            raise ValueError("no selected point coefficients available")
        if self.empty:
            return owned_array(np.empty(0, dtype=np.float64))
        if self.initial_scale is not None:
            return self.initial_scale.coefficients
        return self.selected_point.beta

    def scale_evaluation(
        self, request: CovarianceRequest | None, *, label: str
    ) -> RetainedInitialScaleCache:
        """The initial cache, or a new covariance of the retained influence rows."""
        if matches_initial_covariance_request(self.initial_scale, self.sample, request):
            assert self.initial_scale is not None
            return self.initial_scale
        if request is None:
            reason = f"{NOT_FORMED}: {self.inference_status}: {self.inference.reason}"
            raise ValueError(reason)
        self.require_influence(label)
        influence, order = self._observed_influence, self.order
        point = self.selected_point
        if self.empty or influence is None or order is None or point is None:
            raise ValueError(f"{NOT_FORMED}: no observed influence rows were retained")
        check_cluster_request_rows(request, self.sample)
        weights = self.sample.weight_values
        units = _sampling_units(weights, self.sample.count)
        active = units.identifiers
        covariance = fe_covariance(
            influence[active],
            weights[active],
            frequency=self.sample.frequency,
            request=request,
            units=units,
        )
        errors = direct_standard_errors(covariance, order)
        reported = ReportedScale(gamma=point.theta, standard_errors=errors, order=order)
        matrix: FloatArray | None = None
        failure: str | None = None
        try:
            matrix = order.covariance(covariance.covariance)
        except FloatingPointError as error:
            failure = str(error)
        head = (snapshot_covariance_request(request), order, bool((errors == 0).any()))
        tail = (covariance, reported, matrix, failure, owned_array(influence))
        return RetainedInitialScaleCache(
            head[0],
            head[1],
            head[2],
            tail[0],
            tail[1],
            tail[2],
            tail[3],
            tail[4],
        )

    def se(self) -> FloatArray:
        """The initial direct standard errors (the empty restriction without slopes)."""
        if self.empty and self.selected_point is not None:
            return owned_array(np.empty(0, dtype=np.float64))
        return self.scale_evaluation(None, label="se").standard_errors

    def vcov(self) -> FloatArray:
        """The initial reported covariance (empty without slopes)."""
        if self.empty and self.selected_point is not None:
            return owned_array(np.empty((0, 0), dtype=np.float64))
        return self.scale_evaluation(None, label="vcov").require_reported_covariance()

    def influence(self) -> FloatArray | None:
        """A fresh copy of the ``(n, K)`` observed influence rows ``u_i``."""
        self.require_influence("influence")
        return _optional(self._observed_influence)

    @property
    def slope_map(self) -> FloatArray | None:
        """A fresh copy of ``W``: ``K`` columns of ``H^-T``, in scaled units."""
        self.require_influence("slope_map")
        return _optional(self._slope_map)

    def scores(self) -> FloatArray | None:
        """A fresh copy of the raw weighted scores ``b`` at the selected point."""
        self.require_influence("scores")
        rows = None if self.evidence is None else self.evidence.rows
        return None if rows is None else rows.b

    def expected_inputs(self) -> ExpectedInformationInputs | None:
        """Lossless same-point inputs of the expected information, or ``None``."""
        self.require_influence("expected_information")
        evidence, point = self.evidence, self.selected_point
        if evidence is None or evidence.rows is None or point is None:
            return None
        design = self.sample.design
        count = self.sample.count
        matrix = np.zeros((count, 0)) if design is None else design.matrix
        return ExpectedInformationInputs(
            self.sample,
            point,
            evidence.basis,
            evidence.rows,
            order=self.order,
            design=matrix,
        )

    def expected_information(self) -> SparseMatrix | None:
        """The expected joint Hessian ``B' diag(C_expected) B``, assembled on demand."""
        inputs = self.expected_inputs()
        return None if inputs is None else inputs.expected_hessian()


# ------------------------------------------------------------- composition


def _evidence(
    bound: BoundStandard,
    sample: RetainedSampleSnapshot,
    point: RetainedPointSnapshot,
    maps: Maps,
    *,
    keep_influence: bool,
) -> RetainedStandardEvidence:
    """The detached standard evidence of the bound reading (module docstring)."""
    reading, geometry = bound.reading, bound.inference.geometry
    point_map, sample_map, grouping_map = maps
    grouping = sample._final_grouping
    basis = snapshot_basis(
        reading.basis, partition_key=sample.partition_key, grouping_snapshot=grouping
    )
    rows = None
    if reading.rows is not None and keep_influence:
        rows = snapshot_rows(
            reading.rows,
            response=sample._owned_response,
            q=point.q,
            variance=reading.point.variance,
        )
    inner = _snapshot_inner_evidence(
        reading.inner,
        pt_snap=point,
        sample_snapshot=sample,
        grp_snap=grouping,
        point_map=point_map,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    retained = None
    if geometry is not None and rows is not None:
        retained = snapshot_geometry(
            geometry, point_snapshot=point, rows_snapshot=rows, basis_snapshot=basis
        )
    effects = reading.effects
    return RetainedStandardEvidence(
        reading.policy,
        reading.controls,
        reading.criterion,
        reading.gradient,
        reading.support,
        basis,
        None
        if effects is None
        else RetainedEffectEquations(effects.maximum, effects.means),
        inner,
        rows,
        retained,
    )


def compose_standard_envelope(
    selection: StandardSelection,
    bound: BoundStandard | None,
    inference: StandardInference,
    *,
    snapshots: Snapshots,
    maps: Maps,
    keep_influence: bool,
    probes: RetainedStopProbes | None = None,
) -> StandardRetainedEnvelope:
    """The detached envelope of an admitted standard fit (module docstring).

    ``probes`` are the bound probes of the fit's equation-aware stop
    (:func:`detach_probes`), ``None`` when the fit's solve ran without that stop; the
    policy of the envelope and its selection follows them.
    """
    sample, outcome, history = snapshots
    point = history.selected_point
    evidence = None
    if bound is not None and point is not None:
        evidence = _evidence(bound, sample, point, maps, keep_influence=keep_influence)
    formed = inference if bound is None else bound.inference
    scale = formed.scale
    selected = selection.selected
    if selected is not None:
        selected = replace(selected, inference=formed.status)
    policy = fit_policy(history.route, stop=probes is not None)
    retained = RetainedStandardSelection(
        policy,
        selection.status,
        selection.readings,
        selected,
        selection.consumed,
        selection.reason,
        selection.failures,
    )
    record = RetainedStandardInference(
        formed.status,
        formed.reason,
        formed.ordinary,
        formed.actions,
        formed.work,
        formed.provider_calls,
        requested=formed.requested,
        executed=formed.executed,
        residualized=formed.residualized,
    )
    cache = None
    if scale is not None:
        cache = snapshot_initial_scale(scale, keep_influence=keep_influence)
    return StandardRetainedEnvelope(
        route=cast("Literal['profiled', 'index']", history.route),
        keep_influence=keep_influence,
        standard_status=selection.status,
        inference_status=formed.status,
        sample=sample,
        outcome=outcome,
        history=history,
        selected_point=point,
        selection=retained,
        evidence=evidence,
        inference=record,
        _order=sample.order_scale,
        initial_scale=cache,
        _observed_influence=None if scale is None else scale.evidence.influence,
        _slope_map=formed.slope_map,
        policy=policy,
        probes=probes,
    )
