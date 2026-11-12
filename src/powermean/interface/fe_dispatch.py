"""Normal private FE composition using unchanged admitted solvers and retention.

Explicit organization and sample/resource records are required. The initial
requested covariance is used by every current-point assessment, never patched
after a raw acceptance gate. The broader covariance gate remains an obligation.
A joint power variance outside ``[1, 2]`` at ``omega != 0`` takes the selected-root
route of :mod:`powermean.interface.fe_root` on the same composed sample (its
``CoverageWarning`` once, before the solve); every other joint fit keeps its
unchanged convex or linear route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import MethodType, ModuleType

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import ClusterRequest, HCRequest
from powermean.data.numeric_fit import NumericFit
from powermean.data.raw import _object_array
from powermean.data.request_values import Omitted, given
from powermean.data.retained import OrderSample
from powermean.data.sample_fe import _owned_fixed
from powermean.data.sample_fe_state import FESample
from powermean.data.specs import FixedEffectFormulation
from powermean.errors import SpecificationError
from powermean.fe import certificate_basis as _count_basis
from powermean.fe.connectivity import IndicatorRank
from powermean.inference import absorbed_rank as _count_rank
from powermean.inference.absorbed_rank import AbsorbedCount, IntegerArray
from powermean.interface import numeric_execution as _count_execution
from powermean.fe.cancel import cancel_structure, center_sorted, to_original, to_sorted
from powermean.fe.certificate_basis import (
    CertificateBasis, _witness_proof, _operational_data, _rebind_operational_basis,
)
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.groups import Grouping
from powermean.fe.profile import cold_state
from powermean.formula.frame import ModelFrame
from powermean.inference.absorbed import AbsorbedScale, preflight_absorbed_scale
from powermean.inference.absorbed_preflight import _preflight_numeric
from powermean.inference.convention import ClusterCodes
from powermean.inference.correction.budget import BiasBounds
from powermean.inference.ssc import STATA, resolve_convention
from powermean.inference.wz_binding import WZSource, layout
from powermean.inference.wz_binding import unavailable as wz_unavailable
from powermean.inference.wz_coefficient import workspace as coefficient_workspace
from powermean.inference.wz_covariance import WZControls, ppml
from powermean.inference.wz_resources import joint_bounds
from powermean.interface.cancelled_fit import fit_cancelled_private
from powermean.interface.cancelled_selection import (
    CancelledBlockInputs,
    CancelledFitPolicy,
    NamedFEBlock,
    select_cancelled_block,
)
from powermean.interface.fe_bias import (
    BiasFitInputs,
    apply_cancelled_bias,
    apply_joint_wz_bias,
    bias_sampling,
    coefficient_ppml,
    unsupported_bias,
    wz_pair_name,
)
from powermean.interface.fe_bias import (
    unavailable as coefficient_unavailable,
)
from powermean.interface.fe_cancelled_result import CancelledResultBackend
from powermean.interface.fe_joint_result import AbsorbedResultBackend
from powermean.interface.fe_options import ResolvedFESolverOptions, wz_coefficient_vcov
from powermean.interface.fe_result_backends import (
    AbsorbedState,
    CancelledState,
    FEView,
)
from powermean.interface.fe_root import coverage_warning, root_audit
from powermean.interface.numeric_execution import (
    NumericExecution,
    _bind_retained_basis,
    bind_numeric_execution,
    bind_numeric_sample,
)
from powermean.interface.options import Specification
from powermean.interface.ppml_audit import _Coordinator
from powermean.interface.preparation import PreparedProblem
from powermean.interface.retained_binding import _bind_owned_metadata_parts
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_outcomes import _admit_owned_sample_outcome
from powermean.interface.retained_postfit import retain_absorbed_fit
from powermean.interface.retained_preparation import RetainedPreparation
from powermean.interface.retention import bind_sample_metadata
from powermean.interface.sample import (
    SampleOutcome,
    _prepare_owned_sample,
    admitted_outcome,
    prepare_sample,
)
from powermean.interface.sample_records import SampleRequest
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.absorbed_assessment_handoff import AssessmentHandoff
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.cancelled_bias import CancelledBias
from powermean.results.cancelled_retained import CancelledReporting
from powermean.results.numeric_assessment import NumericAssessmentHandoff
from powermean.results.result import Result
from powermean.results.retained_hc import RetainedHC, initial_hc_from_envelope
from powermean.results.retained_inference import (
    InferenceRequest,
    RetainedInference,
    initial_inference_from_envelope,
)
from powermean.results.wz_bias import WZBias
from powermean.solve import linear_absorbed, newton_absorbed, newton_profiled
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.absorbed_certificate import (
    CandidateAudit,
    certify_linear_candidate,
)
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_direction import IndexStart
from powermean.solve.absorbed_root_records import OPERATION
from powermean.solve.assessment import certificate_controls
from powermean.solve.certificate import outside_convex
from powermean.solve.reduced_execution import ReducedExecution
from powermean.solve.reduced_nonlinear import ConditionalSpace
from powermean.solve.starts import Start
from powermean.systems.admitted import FitRows
from powermean.systems.profiled import ProfiledProblem


_RETAINED_BASIS_BINDER = _bind_retained_basis
_OPERATIONAL_BASIS_ADAPTERS = (_operational_data, _rebind_operational_basis)


def unavailable(feature: str) -> SpecificationError:
    """Refuse an unavailable gate before any solve, leaving its obligation open."""
    return SpecificationError(
        "specification.unsupported_inference",
        {"method": feature, "case": "normal private FE fit-time gate"},
    )


def _wz_domain(spec: Specification) -> bool:
    """Admit the published WZ domain independently of coefficient correction."""
    wz = isinstance(spec.fe_vcov, ClusterRequest) and spec.fe_vcov.adjustment == "wz"
    if wz:
        ppml(spec.estimator.mean.omega, spec.estimator.variance)
        if spec.bread != "observed":
            raise wz_unavailable("observed PPML bread required")
        if resolve_convention(spec.ssc).origin != "preset":
            raise wz_unavailable("named SSC presets are required")
    return wz


def _coefficient_domain(spec: Specification) -> None:
    """Keep the published PPML request distinct from general BBP admission."""
    if spec.bias_correction == "wz":
        coefficient_ppml(spec.estimator.mean.omega, spec.estimator.variance)
        if spec.bread != "observed":
            raise coefficient_unavailable("observed PPML bread required")
    elif spec.bias_correction is not None:
        if spec.cancel is None:
            raise unsupported_bias("joint-cell integration remains open; select cancel")
        if spec.estimator.mean.omega == 0:
            raise unsupported_bias("bias correction is not needed at omega=0")


def _initial_domain(spec: Specification) -> None:
    """Admit the actual current-point covariance provider's domain."""
    _coefficient_domain(spec)
    wz = _wz_domain(spec)
    if spec.cancel is None:
        if spec.fe_vcov == "cluster_cancelled":
            raise unavailable("cluster_cancelled")
        if isinstance(spec.fe_vcov, HCRequest) and spec.fe_vcov.method != "HC1":
            resolved = resolve_convention(spec.ssc)
            if resolved.source == "explicit" and resolved.values.robust != "none":
                raise SpecificationError(
                    "specification.conflicting",
                    {
                        "first": resolved.label,
                        "second": f"vcov={spec.fe_vcov.method!r}",
                    },
                )
        return
    if spec.bread != "observed" or isinstance(spec.fe_vcov, HCRequest):
        raise unavailable(str(spec.fe_vcov))
    if not wz and resolve_convention(spec.ssc).values != STATA:
        raise unavailable(str(spec.ssc))
    if isinstance(spec.fe_vcov, ClusterRequest) and (
        len(spec.fe_vcov.dimensions) != 1 or given(spec.fe_vcov.multiway)
    ):
        raise unavailable("multiway")


def _joint_covariance_request(
    spec: Specification,
    sample: FESample,
    dependence: BoundDependence,
) -> CovarianceRequest:
    """Build the exact current-point :class:`CovarianceRequest` for a joint FE fit."""
    fe_vcov = spec.fe_vcov
    ssc = spec.ssc
    bread = spec.bread
    conv: str | ConventionValues = "stata" if isinstance(ssc, Omitted) else ssc
    if isinstance(fe_vcov, HCRequest):
        return CovarianceRequest(
            kind="hc",
            convention=conv,
            method=fe_vcov.method,
            bread=bread,
            ssc=ssc,
        )
    if isinstance(fe_vcov, ClusterRequest):
        clusters = tuple(
            (name, dependence.cluster_codes(name).codes) for name in fe_vcov.dimensions
        )
        if (
            len(clusters) == 1
            and bread == "observed"
            and isinstance(fe_vcov.multiway, Omitted)
            and isinstance(fe_vcov.psd, Omitted)
            and isinstance(ssc, Omitted)
            and isinstance(fe_vcov.adjustment, Omitted)
        ):
            cname0, ccodes0 = clusters[0]
            return CovarianceRequest("cluster", cname0, ccodes0, sample.positions)
        return CovarianceRequest(
            kind="cluster",
            name=clusters[0][0] if len(clusters) == 1 else None,
            codes=clusters[0][1] if len(clusters) == 1 else None,
            positions=sample.positions,
            convention=conv,
            clusters=clusters,
            bread=bread,
            multiway=fe_vcov.multiway,
            psd=fe_vcov.psd,
            adjustment=fe_vcov.adjustment,
            ssc=ssc,
        )
    if bread == "observed" and isinstance(ssc, Omitted):
        return CovarianceRequest("robust")
    return CovarianceRequest(
        kind="robust",
        convention=conv,
        bread=bread,
        ssc=ssc,
    )


def _mapped_start(
    spec: Specification, names: tuple[str, ...], scales: FloatArray
) -> FloatArray | None:
    """Map a reported active-slope start with the actual order scale."""
    if spec.start is None:
        return None
    if spec.start.shape != (len(names),):
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "start",
                "value": spec.start.tolist(),
                "valid": f"one value per active coefficient ({len(names)})",
            },
        )
    return OrderScale(spec.estimator.mean.omega, scales).solver_coefficients(spec.start)


def _selected_root(spec: Specification) -> PowerVariance | None:
    """The target variance of the joint selected-root route, else ``None``."""
    variance = spec.estimator.variance
    if spec.estimator.mean.omega == 0 or not outside_convex(variance):
        return None
    assert isinstance(variance, PowerVariance)
    return variance


def _joint_audit(
    spec: Specification,
    sample: FESample,
    outcome: SampleOutcome,
    scale: AbsorbedScale,
    mapped: FloatArray | None,
    *, numeric: NumericFit | None = None,
    prepared_basis: CertificateBasis | None = None,
    execution: NumericExecution | None = None,
) -> CandidateAudit:
    """Keep the existing solver/assessment sequence under the resolved request.

    Outside ``[1, 2]`` at ``omega != 0``: the selected-root route of
    :func:`~powermean.interface.fe_root.root_audit`, its root options the resolved
    record of the fit's ``solver_options``.
    """
    controls = spec.fe_solver_options
    assert controls is not None
    inner = controls.absorbed_options
    assert inner is not None
    omega = spec.estimator.mean.omega
    target = _selected_root(spec)
    audit: CandidateAudit
    if target is not None:
        root = controls.root_options
        assert root is not None
        audit = root_audit(
            sample,
            target,
            spec.options,
            inner,
            root,
            scale=scale,
            outcome=outcome,
            start=mapped,
        )
    elif omega == 0:
        candidate_linear = linear_absorbed.solve(
            sample,
            absorb_tol_final=inner.absorb_tol_final,
            absorb_max_sweeps=inner.absorb_max_sweeps,
            absorb_acceleration=inner.absorb_acceleration,
        )
        audit = certify_linear_candidate(
            candidate_linear, certificate_controls(spec.options), scale=scale
        )
    else:
        owned_problem = (
            None if numeric is None else
            ProfiledProblem._from_numeric(numeric, spec.estimator.variance)
        )
        if (owned_problem is not None and prepared_basis is not None
                and owned_problem.n_slopes):
            object.__setattr__(
                owned_problem, "_reduced_execution",
                ReducedExecution._from_owned(owned_problem, prepared_basis),
            )
            if inner.absorbed == "profiled":
                if (all(a is b for a, b in zip(
                        (_operational_data, _rebind_operational_basis),
                        _OPERATIONAL_BASIS_ADAPTERS, strict=True))
                        and _operational_data(prepared_basis) is not None):
                    conditional_basis = _rebind_operational_basis(
                        prepared_basis, owned_problem.grouping
                    )
                else:
                    conditional_basis = (
                        replace(
                            prepared_basis, grouping=owned_problem.grouping,
                            witnesses=proof,
                            echelon=object.__getattribute__(prepared_basis, "echelon"),
                        ) if (proof := _witness_proof(prepared_basis)) is not None else
                        replace(prepared_basis, grouping=owned_problem.grouping)
                    )
                if _bind_retained_basis is _RETAINED_BASIS_BINDER:
                    _bind_retained_basis(execution, prepared_basis, conditional_basis, owned_problem)
                reduced = ConditionalSpace.for_problem(owned_problem, conditional_basis)
                if reduced is not None:
                    object.__setattr__(owned_problem, "_reduced_nonlinear", reduced)
        assessor = (
            AssessmentHandoff(inner, scale) if owned_problem is None else
            NumericAssessmentHandoff(sample, owned_problem, inner, scale, basis=prepared_basis)
        )
        trigger = _Coordinator(outcome)
        candidate: newton_absorbed.IndexCandidate | newton_profiled.ProfiledCandidate
        if inner.absorbed == "index":
            assert sample.grouping is not None
            start = (
                None
                if mapped is None
                else IndexStart(mapped, cold_state(sample.grouping))
            )
            if owned_problem is None:
                candidate = newton_absorbed.solve(
                    sample, spec.estimator.variance, spec.options, inner,
                    trigger=trigger, admit=admitted_outcome,
                    assess=assessor, start=start,
                )
            else:
                call = newton_absorbed._IndexCall(
                    sample, spec.estimator.variance, spec.options, inner,
                    trigger, admitted_outcome, assessor, start,
                )
                candidate = newton_absorbed._solve_numeric(call, owned_problem)
        else:
            profiled = None if mapped is None else Start(mapped, "user")
            if owned_problem is None:
                candidate = newton_profiled.solve(
                    sample, spec.estimator.variance, spec.options, inner,
                    trigger=trigger, admit=admitted_outcome, assess=assessor,
                    start=profiled,
                )
            else:
                call = newton_profiled._ProfiledCall(
                    sample, spec.estimator.variance, spec.options, inner,
                    trigger, admitted_outcome, assessor, profiled,
                )
                candidate = newton_profiled._solve_numeric(call, owned_problem)
        audit, _ = assessor.audit(candidate, scale=scale)
    return audit


def _joint_sample(
    problem: PreparedProblem,
    prep: RetainedPreparation,
    frame: PreparedFrame,
    request: SampleRequest,
    omega: float,
) -> tuple[ModelFrame | None, OrderSample, SampleOutcome]:
    """Produce the original ordered sample and fresh outcome in their exact order."""
    model = frame._frame if prep.context._operation is not None else None
    if prep.context._operation is not None and prep.context._normal_origin is not None:
        order_sample, order_rows = frame._order_parts(
            omega, positive_only=problem.positive_only
        )
        outcome = _prepare_owned_sample(
            frame, request, order_parts=(order_sample, order_rows)
        )
    else:
        order_sample = frame.for_order(omega, positive_only=problem.positive_only)
        outcome = prepare_sample(frame, request)
    return model, order_sample, outcome


_OWNED_COORDINATOR = _prepare_owned_sample
_OWNED_JOINT_SAMPLE = _joint_sample


def _numeric_source_original() -> bool:
    """Changed coordinator callbacks retain their existing ordered rich boundary."""
    return (_prepare_owned_sample is _OWNED_COORDINATOR
            and _joint_sample is _OWNED_JOINT_SAMPLE)


def _numeric_joint_sample(
    problem: PreparedProblem, prep: RetainedPreparation, frame: PreparedFrame,
    request: SampleRequest, omega: float,
) -> tuple[ModelFrame | None, OrderSample, SampleOutcome,
           RetainedOutcomeSnapshot, NumericExecution | None]:
    """Produce authoritative numeric rows before their compatible retained views."""
    model, order_sample, outcome = _joint_sample(problem, prep, frame, request, omega)
    execution = (
        bind_numeric_sample(prep, outcome, order_sample, source=problem._numeric_source)
        if _numeric_source_original() else None
    )
    try:
        snapshot = _admit_owned_sample_outcome(
            outcome, context=prep.context, order_sample=order_sample,
            execution=execution,
        )
        if execution is None:
            execution = bind_numeric_execution(
                prep, outcome, snapshot, source=problem._numeric_source
            )
        return model, order_sample, outcome, snapshot, execution
    except BaseException:
        # A failed boundary view must not publish a surviving numeric lifetime.
        # Propagate the original refusal/allocation error; never retry a solver.
        if execution is not None:
            execution.release()
        raise


def _joint(
    problem: PreparedProblem, spec: Specification, stacklevel: int, *,
    _native_basis: ModuleType | None = None,
) -> Result:
    """Fresh sample, actual current-point provider, authentic solver and handoff.

    ``stacklevel`` is the frame of the selected-root route's coverage warning.
    """
    prep, controls = problem.retained_preparation, spec.fe_solver_options
    assert prep is not None
    assert controls is not None
    assert controls.sample_request is not None
    inner, request = controls.absorbed_options, controls.sample_request
    assert inner is not None
    omega = spec.estimator.mean.omega
    variance = None if omega == 0 else spec.estimator.variance
    if (
        request.omega != omega
        or request.variance != variance
        or request.positive_only != problem.positive_only
    ):
        raise SpecificationError(
            "specification.conflicting",
            {"first": "sample_request", "second": "current estimator/domain"},
        )
    frame = prep.context.new_fit_frame()
    model, order_sample, outcome, snapshot, execution = _numeric_joint_sample(
        problem, prep, frame, request, omega
    )
    if type(execution) is NumericExecution and type(_native_basis) is ModuleType:
        object.__setattr__(execution, "_basis_native", _native_basis)
    if execution is not None:
        prep = execution.preparation
        model = frame._frame
    try:
        retained = None
        initial: RetainedInference | RetainedHC | None = None
        status = outcome.status
        correction = WZBias(None, None) if spec.bias_correction == "wz" else None
        if outcome.sample is not None and outcome.status == "complete":
            sample = outcome.sample
            dependence = bind_sample_metadata(prep, sample, weights=sample.data.weights)
            spec, coefficient_controls, coefficient_cluster = _joint_bias_setup(
                spec, sample, (model if model is not None else frame.frame).parts.fixed_effects
            )
            cov = _joint_covariance_request(spec, sample, dependence)
            scale, prepared_basis = _joint_scale(
                spec, sample, cov,
                (model if model is not None else frame.frame).parts.fixed_effects,
                execution=execution,
            )
            mapped = _mapped_start(
                spec,
                sample.active_names,
                np.empty(0) if sample.design is None else sample.design.scale,
            )
            target = _selected_root(spec)
            if target is not None:
                coverage_warning(sample, target, stacklevel=stacklevel)
            audit = _joint_audit(
                spec, sample, outcome, scale, mapped,
                numeric=None if execution is None else execution.numeric,
                prepared_basis=prepared_basis,
                execution=execution,
            )
            if coefficient_controls is not None and coefficient_cluster is not None:
                correction = apply_joint_wz_bias(
                    audit.certificate, coefficient_controls, coefficient_cluster
                )
            retained = retain_absorbed_fit(
                audit,
                preparation=prep,
                outcome=outcome,
                dependence=dependence,
                estimator=spec.estimator,
                request=cov,
                keep_influence=spec.keep_influence,
            )
            status = (
                ("verified" if retained.level == "strict" else "verified_secondary")
                if retained.accepted
                else audit.history.reason
            )
            if retained.accepted and not retained.empty:
                if cov.kind == "hc":
                    initial = initial_hc_from_envelope(retained.envelope)
                else:
                    initial = initial_inference_from_envelope(retained.envelope)
        estimator = replace(
            spec.estimator,
            fixed_effects=FixedEffectFormulation(
                tuple("^".join(term) for term in (model if model is not None else frame.frame).parts.fixed_effects)
            ),
        )
        view = FEView(
            estimator,
            order_sample,
            frame.labels,
            frame.columns,
            frame.subset_missing,
            (model if model is not None else frame.frame).parts,
            (model if model is not None else frame.frame).design,
            problem.offset_design,
            spec.options,
            (spec.fe_vcov, spec.ssc, spec.bread),
            spec.start,
        )
        state = AbsorbedState(
            retained,
            initial,
            snapshot,
            None,
            status,
            inner,
            "linear" if omega == 0 else inner.absorbed,
            controls.max_dense_bytes,
            correction,
            None if _selected_root(spec) is None else OPERATION,
            None if _selected_root(spec) is None else controls.root_options,
        )
        return Result(AbsorbedResultBackend(view, state))
    finally:
        if execution is not None:
            execution.release()

def _joint_bias_setup(
    spec: Specification, sample: FESample, terms: tuple[tuple[str, ...], ...]
) -> tuple[Specification, WZControls | None, ClusterCodes | None]:
    """Resolve WZ's default before covariance assessment; reserve the added whole call.

    The saved optional byte choice and existing finite tier mapping supply
    LIMITS only. A fresh shared-plus-coefficient plan is compared against them;
    covariance's prior reservations/successes supply no coefficient allowance.
    This retains the internal numeric-buffer/array-work scope, not an OS/RSS or
    cumulative-fit-work guarantee.
    """
    if spec.bias_correction != "wz":
        return spec, None, None
    if sample.grouping is None or sample.design is None:
        raise coefficient_unavailable(
            "current original WZ grouping and slopes required"
        )
    controls = spec.fe_solver_options
    assert controls is not None
    assert controls.absorbed_options is not None
    source = WZSource(terms, sample.grouping, sample.data.row_labels)
    bounds = joint_bounds(controls.max_dense_bytes)
    selected = (
        spec.fe_vcov.dimensions[0]
        if spec.explicit_vcov
        and isinstance(spec.fe_vcov, ClusterRequest)
        and len(spec.fe_vcov.dimensions) == 1
        else None
    )
    pair_name = wz_pair_name(
        source, sample.design.matrix.shape, bounds, selected=selected
    )
    spec = replace(
        spec,
        fe_vcov=wz_coefficient_vcov(
            spec.fe_vcov, explicit=spec.explicit_vcov, pair_name=pair_name
        ),
        explicit_vcov=True,
    )
    _initial_domain(spec)
    coefficient = WZControls(source, controls.absorbed_options, spec.options, bounds)
    cluster = ClusterCodes(
        pair_name, next(b.codes for b in sample.grouping.blocks if b.name == pair_name)
    )
    layout(
        source,
        pair_name,
        sample.design.matrix.shape,
        controls.absorbed_options,
        bounds,
        workspace=coefficient_workspace,
    )
    return spec, coefficient, cluster


def _cancelled_bias_setup(
    spec: Specification,
    source: WZSource | None,
    selected_name: str,
    shape: tuple[int, int],
    *,
    conditioned: bool,
) -> tuple[Specification, WZControls | None]:
    """Use cancellation's saved finite inference limits for a fresh coefficient plan."""
    if spec.bias_correction != "wz":
        return spec, None
    if conditioned:
        raise coefficient_unavailable("unconditioned PPML required")
    assert source is not None
    controls = spec.fe_solver_options
    assert controls is not None
    assert controls.cancelled_controls is not None
    assert controls.absorbed_options is not None
    bounds = controls.cancelled_controls.inference
    pair_name = wz_pair_name(source, shape, bounds, selected=selected_name)
    if selected_name != pair_name:
        raise coefficient_unavailable(
            "WZ coefficients require genuine cancellation of the pair role"
        )
    spec = replace(
        spec,
        fe_vcov=wz_coefficient_vcov(
            spec.fe_vcov, explicit=spec.explicit_vcov, pair_name=pair_name
        ),
        explicit_vcov=True,
    )
    _initial_domain(spec)
    coefficient = WZControls(source, controls.absorbed_options, spec.options, bounds)
    layout(
        source,
        pair_name,
        shape,
        controls.absorbed_options,
        bounds,
        workspace=coefficient_workspace,
    )
    return spec, coefficient


def _joint_scale(
    spec: Specification,
    sample: FESample,
    request: CovarianceRequest,
    terms: tuple[tuple[str, ...], ...],
    *, execution: NumericExecution | None = None,
) -> tuple[AbsorbedScale, CertificateBasis | None]:
    """Bind the same covariance and one actual operation's structural preparation."""
    if not isinstance(spec.fe_vcov, ClusterRequest) or spec.fe_vcov.adjustment != "wz":
        if (execution is not None and spec.estimator.mean.omega != 0
                and _selected_root(spec) is None):
            producer = execution.prepare_basis
            prepare_count: Callable[
                [CertificateBasis, list[IntegerArray]], AbsorbedCount | None
            ] | None = None
            try:
                def count_from_basis(
                    basis: CertificateBasis, clusters: list[IntegerArray],
                ) -> AbsorbedCount | None:
                    return _owned_convention_count(sample, basis, clusters, producer)
                prepare_count = count_from_basis
            except MemoryError:
                pass  # Optional composition, before preflight work.

            preflight, basis = _preflight_numeric(
                sample, request, producer, prepare_count
            )
        else:
            preflight, basis = preflight_absorbed_scale(sample, request), None
        return AbsorbedScale(request, preflight), basis
    ppml(sample.omega, spec.estimator.variance, weights=sample.data.weights is not None)
    grouping = sample.grouping
    if grouping is None:
        raise wz_unavailable("the current original FE grouping is unavailable")
    controls = spec.fe_solver_options
    assert controls is not None
    assert controls.absorbed_options is not None
    wz = WZControls(
        WZSource(terms, grouping, sample.data.row_labels),
        controls.absorbed_options,
        spec.options,
        joint_bounds(controls.max_dense_bytes),
    )
    if sample.design is None or request.name is None:
        raise wz_unavailable("current original slopes and exact pair source required")
    layout(wz.source, request.name, sample.design.matrix.shape, wz.absorbed, wz.bounds)
    basis = None if execution is None else execution.prepare_basis(sample)
    return AbsorbedScale(request, wz=wz), basis


def _cancelled_blocks(grouping: Grouping) -> tuple[NamedFEBlock, ...]:
    """Preserve each original FE level, including an interaction tuple, as a scalar."""
    return tuple(
        NamedFEBlock(b.name, _object_array(b.levels)[b.codes]) for b in grouping.blocks
    )


def _cancelled_wz_controls(
    request: InferenceRequest | None,
    selected_name: str,
    source: WZSource | None,
    spec: Specification,
) -> WZControls | None:
    """Bind an explicit WZ source without doing WZ numerical work before fit timing."""
    if request is None or request.adjustment != "wz":
        return None
    assert source is not None
    if request.clusters[0].name != selected_name:
        raise wz_unavailable("cancel must name the exact requested pair source")
    controls = spec.fe_solver_options
    assert controls is not None
    assert controls.cancelled_controls is not None
    assert controls.absorbed_options is not None
    return WZControls(
        source,
        controls.absorbed_options,
        spec.options,
        controls.cancelled_controls.inference,
    )


def _cancelled(problem: PreparedProblem, spec: Specification) -> Result:
    """Normal selected-block route through the accepted complete-fit lifecycle."""
    prep, controls = problem.retained_preparation, spec.fe_solver_options
    assert prep is not None
    assert controls is not None
    assert controls.cancelled_controls is not None
    assert spec.cancel is not None
    frame = prep.context.new_fit_frame()
    model = frame._frame if prep.context._operation is not None else None
    sample = frame.for_order(
        spec.estimator.mean.omega, positive_only=problem.positive_only
    )
    assert frame.effects is not None
    grouping = _owned_fixed(frame.effects, sample.positions).grouping
    blocks = _cancelled_blocks(grouping)

    slopes = GeometryInputs(
        sample.design.matrix, sample.positions, sample.data.column_names
    )
    selected = select_cancelled_block(slopes, blocks, cancel=spec.cancel)
    structure = cancel_structure(
        selected.inputs.cancelled, selected.inputs.retained, positions=sample.positions
    )
    offset = to_original(
        structure, center_sorted(structure, to_sorted(structure, sample.problem.offset))
    )
    dependence = (
        _bind_owned_metadata_parts(
            frame, sample, weights=sample.data.weights,
            input_key=prep.context.input_key,
        )
        if (model is not None and frame.dependence is not None
            and any("_retained_sources" in vars(column)
                    for column in frame.dependence._columns.values()))
        else bind_sample_metadata(prep, sample, weights=sample.data.weights)
    )
    source = (
        WZSource(
            (model if model is not None else frame.frame).parts.fixed_effects,
            grouping,
            tuple(frame.labels.take(sample.positions).tolist()),
        )
        if spec.bias_correction == "wz" or _wz_domain(spec)
        else None
    )
    spec, coefficient_controls = _cancelled_bias_setup(
        spec,
        source,
        selected.cancelled_name,
        sample.design.matrix.shape,
        conditioned=sample.positive_only or sample.conditioned,
    )
    requested = None
    if spec.explicit_vcov:
        dimension = spec.fe_vcov
        clusters: tuple[ClusterCodes, ...] = ()
        if isinstance(dimension, ClusterRequest):
            clusters = tuple(
                dependence.cluster_codes(name) for name in dimension.dimensions
            )
        elif dimension == "cluster_cancelled":
            clusters = (ClusterCodes(selected.cancelled_name, structure.cancel_codes),)
        requested = (
            InferenceRequest.from_cluster_request(
                dimension,
                clusters,
                ssc=spec.ssc,
                bread=spec.bread,
            )
            if isinstance(dimension, ClusterRequest)
            else InferenceRequest(
                "cluster" if clusters else "robust", clusters, spec.ssc, spec.bread
            )
        )
    sampling = None
    if spec.bias_correction is not None and spec.bias_correction != "wz":
        inference_bounds = controls.cancelled_controls.inference
        bias_bounds = BiasBounds(inference_bounds.max_bytes, inference_bounds.max_work)
        sampling = bias_sampling(
            spec.bias_correction,
            structure.cancel_codes,
            selected.cancelled_name,
            requested,
            bounds=bias_bounds,
        )
    mapped = _mapped_start(spec, sample.data.column_names, sample.design.scale)
    if mapped is not None:
        bounds = controls.cancelled_controls
        geometry = selected.build_geometry(
            gram_rtol=bounds.gram_rtol,
            gap_min=bounds.gap_min,
            max_bytes=bounds.geometry_bytes,
        )
        full = np.zeros(geometry.n_columns_full)
        full[: geometry.n_slopes_full] = mapped
        mapped = geometry.reduce(full)
    wz_controls = (
        _cancelled_wz_controls(
            requested,
            selected.cancelled_name,
            source,
            spec,
        )
        if requested is not None and requested.adjustment == "wz"
        else None
    )
    policy = CancelledFitPolicy(
        requested,
        mapped,
        spec.keep_influence,
        "return",
        variance=spec.estimator.variance,
        root_options=controls.root_options,
        absorbed_options=controls.absorbed_options,
        wz=wz_controls,
        wz_coefficient=coefficient_controls,
    )
    assert spec.cancel is not None
    envelope = fit_cancelled_private(
        CancelledBlockInputs(slopes, blocks, spec.cancel),
        FitRows(sample.problem.response, offset, None),
        CancelledReporting(
            sample.omega,
            sample.design.scale,
            frame.labels.take(sample.positions).to_numpy(copy=True),
        ),
        controls.cancelled_controls,
        spec.options,
        policy=policy,
    )
    correction: CancelledBias | WZBias | None = envelope._wz_bias
    if sampling is not None:
        correction = apply_cancelled_bias(
            envelope,
            sampling,
            BiasFitInputs(
                slopes,
                sample.problem.response,
                offset,
                spec.estimator.variance,
                bias_bounds,
                spec.options.rcond_min,
            ),
        )
    estimator = replace(
        spec.estimator,
        fixed_effects=FixedEffectFormulation(
            envelope.block_order, envelope.cancelled_name
        ),
    )
    view = FEView(
        estimator,
        sample,
        frame.labels,
        frame.columns,
        frame.subset_missing,
        (model if model is not None else frame.frame).parts,
        (model if model is not None else frame.frame).design,
        problem.offset_design,
        spec.options,
        (spec.fe_vcov, spec.ssc, spec.bread),
        spec.start,
    )
    assert controls.absorbed_options is not None
    state = CancelledState(
        envelope,
        dependence,
        grouping,
        controls.cancelled_controls,
        controls.absorbed_options,
        correction,
    )
    return Result(CancelledResultBackend(view, state))


def fit_fe_prepared(
    problem: PreparedProblem, spec: Specification, *, stacklevel: int = 1
) -> Result:
    """Dispatch a genuine normal private FE fit, never the no-FE solver.

    ``stacklevel`` is that of :func:`~powermean.interface.dispatch.fit_prepared`: the
    frames above it at which a warning names the caller's line.
    """
    _initial_domain(spec)
    if problem.frame.columns.weights is not None:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "weights",
                "value": problem.frame.columns.weights,
                "valid": "no weights column with fixed effects in this version",
            },
        )
    prep = problem.retained_preparation
    assert prep is not None
    call_prep = prep._for_fit()
    if call_prep is not prep:
        source = problem._numeric_source
        problem = replace(problem, retained_preparation=call_prep)
        object.__setattr__(problem, "_numeric_source", source)
    try:
        native_basis = None
        controls = spec.fe_solver_options
        inner = (vars(controls).get("absorbed_options")
                 if type(controls) is ResolvedFESolverOptions else None)
        backend = vars(inner).get("basis_backend") if type(inner) is AbsorbedOptions else None
        if type(backend) is str and backend == "cpp":
            native_basis = _count_basis._resolve_native_echelon()
        if spec.cancel is not None:
            return _cancelled(problem, spec)
        # The warning frames are unchanged: no new forwarding fit function.
        return (_joint(problem, spec, stacklevel + 4) if native_basis is None else
                _joint(problem, spec, stacklevel + 4, _native_basis=native_basis))
    finally:
        call_prep.context.release_fit_state()


_COUNT_EXECUTION = NumericExecution
_COUNT_BASIS = CertificateBasis
_COUNT_RANK = IndicatorRank
_COUNT_BLOCK = _count_rank.BlockCount
_COUNT_RECORD = _count_rank.AbsorbedCount
_COUNT_GROUPING = Grouping
_COUNT_DERIVED = _count_execution.Derived
_COUNT_GROUP_BLOCK = _count_execution.Block
_COUNT_ABSENT = object()
_COUNT_MEMBERS = tuple((kind, name, vars(kind).get(name, _COUNT_ABSENT)) for kind, names in (
    (NumericExecution, ("prepare_basis", "__getattribute__", "__getattr__")),
    (CertificateBasis, ("__getattribute__", "__getattr__", "rank", "grouping")),
    (IndicatorRank, ("__getattribute__", "__getattr__", "__init__",
                     "levels", "n_components", "pairwise_components")),
    (Grouping, ("__getattribute__", "__getattr__", "blocks", "positions")),
    (_COUNT_DERIVED, ("__getattribute__", "__getattr__", "blocks", "positions")),
    (_COUNT_GROUP_BLOCK, ("name", "codes", "counts")),
) for name in names)
_COUNT_PRODUCERS = tuple((module, name, vars(module)[name]) for module, names in (
    (_count_execution, ("NumericExecution", "_prepare_fit_basis",
                        "_same_group_storage", "_retained_header",
                        "Grouping", "Derived", "Block", "NumericFit")),
    (_count_basis, ("_prepare_fit_basis", "_fit_basis_prepared",
                    "_operational_basis_prepared", "_indicator_rank_prepared",
                    "_indicator_rank_counted", "components",
                    "_fit_pair_counts_generated", "CertificateBasis", "IndicatorRank")),
    (_count_rank, ("_prepared_count", "_prepared_count_original")),
) for name in names)


def _owned_convention_count(
    sample: FESample, basis: CertificateBasis, clusters: list[IntegerArray],
    producer: object,
) -> AbsorbedCount | None:
    """Use fresh structural scalars only at the live original owned callback site."""
    if (type(producer) is not MethodType
            or type(producer.__self__) is not _COUNT_EXECUTION
            or producer.__func__ is not _COUNT_MEMBERS[0][2]
            or NumericExecution is not _COUNT_EXECUTION
            or CertificateBasis is not _COUNT_BASIS or type(basis) is not _COUNT_BASIS
            or IndicatorRank is not _COUNT_RANK
            or _count_rank.BlockCount is not _COUNT_BLOCK
            or _count_rank.AbsorbedCount is not _COUNT_RECORD
            or Grouping is not _COUNT_GROUPING
            or _count_execution.Derived is not _COUNT_DERIVED
            or _count_execution.Block is not _COUNT_GROUP_BLOCK):
        return None
    try:
        if (any(vars(kind).get(name, _COUNT_ABSENT) is not original
                for kind, name, original in _COUNT_MEMBERS)
                or any(vars(module).get(name) is not original
                       for module, name, original in _COUNT_PRODUCERS)
                or not _count_basis._fit_pair_counts_generated()
                or not _count_rank._prepared_count_original()):
            return None
        rank = vars(basis).get("rank")
        if type(rank) is not _COUNT_RANK:
            return None
        grouping = vars(basis).get("grouping")
        numeric = vars(producer.__self__).get("numeric")
        if (type(numeric) is not NumericFit
                or any(type(c) is not np.ndarray for c in clusters)
                or not _count_execution._same_group_storage(
                    grouping, vars(numeric).get("_grouping"))):
            return None
    except MemoryError:
        return None  # Optional selection only, before any count/nesting action.
    return _count_rank._prepared_count(
        tuple((block.name, block.codes) for block in grouping.blocks),
        rank.levels, rank.n_components, rank.pairwise_components, clusters,
    )
