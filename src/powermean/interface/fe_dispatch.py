"""Normal private FE composition using unchanged admitted solvers and retention.

Explicit organization and sample/resource records are required. The initial
requested covariance is used by every current-point assessment, never patched
after a raw acceptance gate. The broader covariance gate remains an obligation.
A joint power variance outside ``[1, 2]`` at ``omega != 0`` takes the selected-root
route of :mod:`powermean.interface.fe_root` on the same composed sample (its
``CoverageWarning`` once, before the solve); every other joint fit at
``omega != 0`` keeps its unchanged convex route.  Log-OLS (``omega = 0``) is solved
by plain absorption without certificate (:mod:`powermean.interface.fe_linear`,
owner's ruling 52).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from types import MethodType, ModuleType
from typing import Literal, cast

import numpy as np

from powermean._prepare.compose import PreparedFrame, _isolated_sample_base
from powermean.data import sample_fe as _sample_fe
from powermean.data.accounting import SUBSET, frozen_codes
from powermean.data.collinearity import (
    ABSORBED,
    COLLINEAR,
    COMPLETE,
    REFUSED,
    CollinearityAnalysis,
    CollinearityProblem,
    DiagnosticControls,
    _within_levels,
    analyze_collinearity,
    near_omitted,
)
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import ClusterRequest, HCRequest
from powermean.data.numeric_fit import NumericFit
from powermean.data.raw import _object_array
from powermean.data.request_values import OMITTED, Omitted
from powermean.data.retained import OrderSample
from powermean.data.sample_fe import _owned_fixed, close
from powermean.data.sample_fe_source import Derived, Source, _owned_source_original
from powermean.data.sample_fe_state import FESample, same_bits
from powermean.data.specs import FixedEffectFormulation
from powermean.errors import CoverageWarning, NumericalWarning, SpecificationError
from powermean.fe import certificate_basis as _count_basis
from powermean.fe import retained_live as _count_live
from powermean.fe.cancel import (
    CancelStructure,
    cancel_structure,
    center_sorted,
    to_original,
    to_sorted,
)
from powermean.fe.certificate_basis import (
    CertificateBasis,
    Witness,
    _operational_data,
    _rebind_operational_basis,
    _witness_proof,
)
from powermean.fe.connectivity import IndicatorRank
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_errors import GeometryAdmissionError, GeometryRankError
from powermean.fe.groups import (
    Grouping,
    _cancelled_label_partition,
    _CancelledLabelPartition,
)
from powermean.fe.profile import cold_state
from powermean.formula.frame import Design, ModelFrame
from powermean.inference import absorbed_rank as _count_rank
from powermean.inference.absorbed import AbsorbedScale, preflight_absorbed_scale
from powermean.inference.absorbed_preflight import _preflight_numeric
from powermean.inference.absorbed_rank import AbsorbedCount, IntegerArray
from powermean.inference.convention import ClusterCodes
from powermean.inference.correction.budget import BiasBounds
from powermean.inference.ssc import STATA, resolve_convention
from powermean.inference.wz_absorbed import _WZInformation
from powermean.inference.wz_binding import WZSource, identify, layout
from powermean.inference.wz_binding import unavailable as wz_unavailable
from powermean.inference.wz_coefficient import workspace as coefficient_workspace
from powermean.inference.wz_covariance import WZControls, ppml
from powermean.inference.wz_resources import joint_bounds
from powermean.interface import numeric_execution as _count_execution
from powermean.interface._ordinary_pair_default import _ordinary_pair
from powermean.interface.cancelled_fit import (
    ShortGroupHandoff,
    _PreparedCancelledLayout,
    fit_cancelled_private,
)
from powermean.interface.cancelled_selection import (
    CancelledBlockInputs,
    CancelledFitPolicy,
    CancelledSelection,
    NamedFEBlock,
    default_diagnostic,
    select_cancelled_block,
)
from powermean.interface.fe_bias import (
    BiasFitInputs,
    apply_cancelled_bias,
    apply_joint_wz_bias,
    bias_sampling,
    coefficient_ppml,
    unsupported_bias,
    wz_formula_intercept,
    wz_pair_name,
)
from powermean.interface.fe_bias import (
    unavailable as coefficient_unavailable,
)
from powermean.interface.fe_cancelled_result import CancelledResultBackend
from powermean.interface.fe_joint_result import (
    AbsorbedResultBackend,
    exact_count_domain,
)
from powermean.interface.fe_light import light_fit
from powermean.interface.fe_linear import linear_absorbed_fit, linear_request
from powermean.interface.fe_options import (
    FEVcovRequest,
    ResolvedFESolverOptions,
    wz_coefficient_vcov,
)
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
    _same_sample,
    admitted_outcome,
    prepare_sample,
)
from powermean.interface.sample_records import (
    CollinearityInvocation,
    SampleRequest,
    same_problem,
    start,
)
from powermean.model.order import IndexArray, _flags
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.absorbed_assessment_handoff import AssessmentHandoff
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.cancelled_bias import CancelledBias
from powermean.results.cancelled_retained import (
    NO_RETAINED_SLOPE,
    CancelledReporting,
)
from powermean.results.fe_sample import ClosedRows
from powermean.results.numeric_assessment import NumericAssessmentHandoff
from powermean.results.result import Result
from powermean.results.retained_hc import RetainedHC, initial_hc_from_envelope
from powermean.results.retained_hc_records import HCInferenceRequest
from powermean.results.retained_inference import (
    InferenceRequest,
    RetainedInference,
    initial_inference_from_envelope,
)
from powermean.results.wz_bias import WZBias
from powermean.solve import newton_absorbed, newton_profiled
from powermean.solve.absorbed_certificate import CandidateAudit
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.absorbed_direction import IndexStart
from powermean.solve.absorbed_root_records import OPERATION
from powermean.solve.cancelled_records import RootOptions
from powermean.solve.certificate import outside_convex
from powermean.solve.m1_options import AbsorbedOptions
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
        {"method": feature, "case": "this fit with fixed effects"},
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
    if spec.cancel is None:
        if spec.fe_vcov == "cluster_cancelled":
            raise unavailable("cluster_cancelled")
        return
    if spec.bread != "observed":
        raise unavailable(str(spec.fe_vcov))
    if (
        spec.bias_correction == "wz"
        and not wz
        and resolve_convention(spec.ssc).values != STATA
    ):
        raise unavailable(str(spec.ssc))


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


def _joint_controls_after_cancel(spec: Specification) -> Specification:
    """Supply unresolved joint root defaults when a cancellation is refused."""
    controls = spec.fe_solver_options
    assert controls is not None
    if controls.root_options is None and _selected_root(spec) is not None:
        return replace(
            spec, fe_solver_options=replace(controls, root_options=RootOptions())
        )
    return spec


@contextmanager
def _joint_workspace(
    owned_problem: ProfiledProblem | None,
    prepared_basis: CertificateBasis | None,
    execution: NumericExecution | None,
    *,
    profiled: bool,
) -> Iterator[None]:
    """Keep reduced workspaces through the last assessment of this fit."""
    try:
        if (
            owned_problem is not None
            and prepared_basis is not None
            and owned_problem.n_slopes
        ):
            object.__setattr__(
                owned_problem,
                "_reduced_execution",
                ReducedExecution._from_owned(owned_problem, prepared_basis),
            )
            if profiled:
                if (
                    all(
                        a is b
                        for a, b in zip(
                            (_operational_data, _rebind_operational_basis),
                            _OPERATIONAL_BASIS_ADAPTERS,
                            strict=True,
                        )
                    )
                    and _operational_data(prepared_basis) is not None
                ):
                    conditional_basis = _rebind_operational_basis(
                        prepared_basis, owned_problem.grouping
                    )
                else:
                    conditional_basis = (
                        replace(
                            prepared_basis,
                            grouping=owned_problem.grouping,
                            # CertificateBasis exposes materialized witnesses over
                            # its packed storage through the original accessor.
                            witnesses=cast("tuple[tuple[Witness, ...], ...]", proof),
                            echelon=object.__getattribute__(prepared_basis, "echelon"),
                        )
                        if (proof := _witness_proof(prepared_basis)) is not None
                        else replace(prepared_basis, grouping=owned_problem.grouping)
                    )
                if _bind_retained_basis is _RETAINED_BASIS_BINDER:
                    _bind_retained_basis(
                        execution, prepared_basis, conditional_basis, owned_problem
                    )
                reduced = ConditionalSpace.for_problem(owned_problem, conditional_basis)
                if reduced is not None:
                    object.__setattr__(owned_problem, "_reduced_nonlinear", reduced)
        yield
    finally:
        if owned_problem is not None:
            # These workspaces refer back to their owner. Their last
            # consumers are the solver and the final assessment above.
            vars(owned_problem).pop("_reduced_nonlinear", None)
            vars(owned_problem).pop("_reduced_execution", None)


def _same_polish_sample(original: FESample, diagnosed: FESample) -> bool:
    """Compare the admitted samples, including the exact scaled slope coordinates."""
    a, b = original.design, diagnosed.design
    return (
        a is not None
        and b is not None
        and a.column_names == b.column_names
        and same_bits(a.scale, b.scale)
        and same_bits(a.matrix, b.matrix)
        and _same_sample(original, diagnosed)
    )


def _joint_audit(
    spec: Specification,
    sample: FESample,
    outcome: SampleOutcome,
    scale: AbsorbedScale,
    mapped: FloatArray | None,
    *,
    numeric: NumericFit | None = None,
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
    else:
        # omega != 0: log-OLS is solved by absorption in
        # :mod:`powermean.interface.fe_linear` and never reaches this audit.
        assert omega != 0
        owned_problem = (
            None
            if numeric is None
            else ProfiledProblem._from_numeric(numeric, spec.estimator.variance)
        )
        with _joint_workspace(
            owned_problem,
            prepared_basis,
            execution,
            profiled=inner.absorbed == "profiled",
        ):
            assessor = (
                AssessmentHandoff(inner, scale)
                if owned_problem is None
                else NumericAssessmentHandoff(
                    sample, owned_problem, inner, scale, basis=prepared_basis
                )
            )
            trigger = _Coordinator(outcome)
            candidate: (
                newton_absorbed.IndexCandidate | newton_profiled.ProfiledCandidate
            )
            if inner.absorbed == "index":
                assert sample.grouping is not None
                start = (
                    None
                    if mapped is None
                    else IndexStart(mapped, cold_state(sample.grouping))
                )
                if owned_problem is None:
                    candidate = newton_absorbed.solve(
                        sample,
                        spec.estimator.variance,
                        spec.options,
                        inner,
                        trigger=trigger,
                        admit=admitted_outcome,
                        assess=assessor,
                        start=start,
                    )
                else:
                    call = newton_absorbed._IndexCall(
                        sample,
                        spec.estimator.variance,
                        spec.options,
                        inner,
                        trigger,
                        admitted_outcome,
                        assessor,
                        start,
                    )
                    candidate = newton_absorbed._solve_numeric(call, owned_problem)
            else:
                # The ordinary assessor sees only complete profiles. Conditional
                # profiles use the numeric handoff on the private owned entry.
                profiled_assessor = cast("newton_profiled.FullAssessor", assessor)
                profiled = None if mapped is None else Start(mapped, "user")
                if owned_problem is None:
                    candidate = newton_profiled.solve(
                        sample,
                        spec.estimator.variance,
                        spec.options,
                        inner,
                        trigger=trigger,
                        admit=admitted_outcome,
                        assess=profiled_assessor,
                        start=profiled,
                        zero_row_polish=newton_profiled._ZeroRowPolish(
                            _same_polish_sample, assessor.strictly_assess
                        ),
                    )
                else:
                    numeric_assessor = cast("NumericAssessmentHandoff", assessor)
                    profiled_call = newton_profiled._ProfiledCall(
                        sample,
                        spec.estimator.variance,
                        spec.options,
                        inner,
                        trigger,
                        admitted_outcome,
                        profiled_assessor,
                        profiled,
                        newton_profiled._ZeroRowPolish(
                            _same_polish_sample, numeric_assessor.strictly_assess
                        ),
                    )
                    candidate = newton_profiled._solve_numeric(
                        profiled_call, owned_problem
                    )
            audit, _ = assessor.audit(candidate, scale=scale)
    return audit


def _joint_sample(
    problem: PreparedProblem,
    prep: RetainedPreparation,
    frame: PreparedFrame,
    request: SampleRequest,
    omega: float,
    *,
    basis_native: ModuleType | None = None,
) -> tuple[ModelFrame | None, OrderSample, SampleOutcome]:
    """Produce the original ordered sample and fresh outcome in their exact order."""
    model = frame._frame if prep.context._operation is not None else None
    if prep.context._operation is not None and prep.context._normal_origin is not None:
        order_sample, order_rows = frame._order_parts(
            omega, positive_only=problem.positive_only
        )
        outcome = (
            _prepare_owned_sample(
                frame,
                request,
                order_parts=(order_sample, order_rows),
                basis_native=basis_native,
            )
            if _prepare_owned_sample is _OWNED_COORDINATOR
            else _prepare_owned_sample(
                frame, request, order_parts=(order_sample, order_rows)
            )
        )
    else:
        order_sample = frame.for_order(omega, positive_only=problem.positive_only)
        outcome = prepare_sample(frame, request)
    return model, order_sample, outcome


_OWNED_COORDINATOR = _prepare_owned_sample
_OWNED_JOINT_SAMPLE = _joint_sample


def _numeric_source_original() -> bool:
    """Changed coordinator callbacks retain their existing ordered rich boundary."""
    return (
        _prepare_owned_sample is _OWNED_COORDINATOR
        and _joint_sample is _OWNED_JOINT_SAMPLE
    )


def _numeric_joint_sample(
    problem: PreparedProblem,
    prep: RetainedPreparation,
    frame: PreparedFrame,
    request: SampleRequest,
    omega: float,
    *,
    basis_native: ModuleType | None = None,
) -> tuple[
    ModelFrame | None,
    OrderSample,
    SampleOutcome,
    RetainedOutcomeSnapshot,
    NumericExecution | None,
]:
    """Produce authoritative numeric rows before their compatible retained views."""
    model, order_sample, outcome = (
        _joint_sample(problem, prep, frame, request, omega, basis_native=basis_native)
        if _joint_sample is _OWNED_JOINT_SAMPLE
        else _joint_sample(problem, prep, frame, request, omega)
    )
    execution = (
        bind_numeric_sample(prep, outcome, order_sample, source=problem._numeric_source)
        if _numeric_source_original()
        else None
    )
    try:
        snapshot = _admit_owned_sample_outcome(
            outcome,
            context=prep.context,
            order_sample=order_sample,
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


_NUMERIC_JOINT_SAMPLE = _numeric_joint_sample


def _announce_defaults(
    problem: PreparedProblem, spec: Specification
) -> tuple[str, ...]:
    """Identify omitted FE options for the summary of the fitted specification.

    None for fixed-effect log-OLS (``omega = 0``): it cancels no block and has no
    bias correction (both refused), so the fit chooses neither.
    """
    if spec.estimator.mean.omega == 0:
        return ()
    explicit_cancel = (
        problem.explicit_cancel or spec.explicit_cancel or spec.cancel is not None
    )
    explicit_bias = (
        problem.explicit_bias_correction
        or spec.explicit_bias_correction
        or spec.bias_correction is not None
    )
    return (() if explicit_cancel else ("cancel",)) + (
        () if explicit_bias else ("bias_correction",)
    )


def _with_zero_row_warning(
    audit: CandidateAudit | None, result: Result, *, stacklevel: int
) -> Result:
    """Report the original fitted-mean warning on the strictly verified return."""
    candidate = None if audit is None else audit.history.candidate
    if type(candidate) is not newton_profiled.ProfiledCandidate:
        return result
    alarm = candidate.polished_alarm
    if alarm is None:
        return result
    assert alarm.zero_rows is not None
    provenance = result.provenance
    if not provenance.certified or provenance.status != "verified":
        raise ValueError("a fit polished after a zero-row alarm must be verified")
    warnings.warn(
        NumericalWarning(
            "numerical.zero_row_false_alarm",
            {"positions": alarm.zero_rows.positions.tolist()},
            status=provenance.status,
            certificate=result.certificate,
        ),
        stacklevel=stacklevel,
    )
    return result


def _joint(
    problem: PreparedProblem,
    spec: Specification,
    stacklevel: int,
    *,
    _native_basis: ModuleType | None = None,
    _refused_cancel: str | None = None,
) -> Result:
    """Fresh sample, actual current-point provider, authentic solver and handoff.

    ``stacklevel`` is the frame of the selected-root route's coverage warning.
    ``_refused_cancel`` is the cancelled block of a cancelled fit whose sample step
    (:func:`_cancelled_sample`) did not complete: the same walk is reported here, with
    its status, as the refusal of that fit.
    """
    prep, controls = problem.retained_preparation, spec.fe_solver_options
    assert prep is not None
    assert controls is not None
    inner, request = controls.absorbed_options, controls.sample_request
    assert inner is not None
    omega = spec.estimator.mean.omega
    variance = None if omega == 0 else spec.estimator.variance
    if request is None:  # omitted by the call: the package's default request
        request = SampleRequest.default(
            omega, variance, problem.positive_only, default_diagnostic()
        )
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
    model, order_sample, outcome, snapshot, execution = (
        _numeric_joint_sample(
            problem,
            prep,
            frame,
            request,
            omega,
            basis_native=_native_basis,
        )
        if _numeric_joint_sample is _NUMERIC_JOINT_SAMPLE
        else _numeric_joint_sample(problem, prep, frame, request, omega)
    )
    if type(execution) is NumericExecution and type(_native_basis) is ModuleType:
        object.__setattr__(execution, "_basis_native", _native_basis)
    if execution is not None:
        prep = execution.preparation
        model = frame._frame
    try:
        retained = audit = None
        initial: RetainedInference | RetainedHC | None = None
        linear_saturated = False
        status = outcome.status
        correction = WZBias(None, None) if spec.bias_correction == "wz" else None
        if outcome.sample is not None and outcome.status == "complete":
            sample = outcome.sample
            _note_omitted(
                sample.data.column_names,
                sample.reasons,
                (model if model is not None else frame.frame).design,
                stacklevel=stacklevel,
            )
            dependence = bind_sample_metadata(prep, sample, weights=sample.data.weights)
            formula_intercept = (
                wz_formula_intercept(
                    (model if model is not None else frame.frame).design
                )
                if spec.bias_correction == "wz"
                or (
                    isinstance(spec.fe_vcov, ClusterRequest)
                    and spec.fe_vcov.adjustment == "wz"
                )
                else None
            )
            spec, coefficient_controls, coefficient_cluster = _joint_bias_setup(
                spec,
                sample,
                (model if model is not None else frame.frame).parts.fixed_effects,
                formula_intercept=formula_intercept,
            )
            cov = None
            # WZ setup has already resolved its own request. Explicit requests
            # retain their provenance and never enter ordinary default selection.
            if not spec.explicit_vcov and spec.fe_vcov == "robust":
                pair = _ordinary_pair(
                    sample,
                    (model if model is not None else frame.frame).parts.fixed_effects,
                )
                if pair is not None:
                    spec = replace(spec, fe_vcov=ClusterRequest((pair.name,)))
                    cov = CovarianceRequest(
                        kind="cluster",
                        name=pair.name,
                        codes=pair.codes,
                        positions=sample.positions,
                        convention=(
                            "stata" if isinstance(spec.ssc, Omitted) else spec.ssc
                        ),
                        bread=spec.bread,
                        ssc=spec.ssc,
                    )
            if cov is None:
                cov = _joint_covariance_request(spec, sample, dependence)
            if omega == 0:
                # Log-OLS: plain absorption, no exact basis and no certificate
                # (owner's ruling 52); its covariance is formed with the fit.
                prepared_hc = linear_request(sample, cov)
                _mapped_start(
                    spec,
                    sample.active_names,
                    np.empty(0) if sample.design is None else sample.design.scale,
                )
                retained, initial, status, linear_saturated = linear_absorbed_fit(
                    sample,
                    outcome,
                    inner,
                    spec.options,
                    prepared_hc,
                    preparation=prep,
                    dependence=dependence,
                    estimator=spec.estimator,
                    request=cov,
                    keep_influence=spec.keep_influence,
                )
            else:
                scale, prepared_basis = _joint_scale(
                    spec,
                    sample,
                    cov,
                    (model if model is not None else frame.frame).parts.fixed_effects,
                    formula_intercept=formula_intercept,
                    execution=execution,
                    identified=(
                        None
                        if coefficient_controls is None
                        else coefficient_controls.identified
                    ),
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
                    spec,
                    sample,
                    outcome,
                    scale,
                    mapped,
                    numeric=None if execution is None else execution.numeric,
                    prepared_basis=prepared_basis,
                    execution=execution,
                )
                if coefficient_controls is not None and coefficient_cluster is not None:
                    correction = apply_joint_wz_bias(
                        audit.certificate,
                        coefficient_controls,
                        coefficient_cluster,
                        shared=(
                            None
                            if scale._wz_information is None
                            else scale._wz_information.take()
                        ),
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
                tuple(
                    "^".join(term)
                    for term in (
                        model if model is not None else frame.frame
                    ).parts.fixed_effects
                ),
                *(() if _refused_cancel is None else (_refused_cancel,)),
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
            announce_defaults=_announce_defaults(problem, spec),
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
            saturated=linear_saturated,
        )
        return _with_zero_row_warning(
            audit, Result(AbsorbedResultBackend(view, state)), stacklevel=stacklevel
        )
    finally:
        if execution is not None:
            execution.release()


def _joint_bias_setup(
    spec: Specification,
    sample: FESample,
    terms: tuple[tuple[str, ...], ...],
    *,
    formula_intercept: int | None = None,
) -> tuple[Specification, WZControls | None, ClusterCodes | None]:
    """Resolve D25 before covariance assessment and reserve the added whole call.

    The saved optional byte choice and existing finite D13 tier mapping supply
    LIMITS only. A fresh shared-plus-coefficient plan is compared against them;
    covariance's prior reservations/successes supply no coefficient allowance.
    This retains the private numeric-buffer/array-work scope, not an OS/RSS,
    cumulative-fit-work guarantee or a public numeric-default adoption.
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
    source = WZSource(
        terms,
        sample.grouping,
        sample.data.row_labels,
        formula_intercept=formula_intercept,
    )
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
    if not isinstance(spec.fe_vcov, HCRequest):
        # A named HC request stays the fit's covariance request: the joint fit's
        # named HC covariance at the uncorrected root (rulings 55 and 63); the
        # correction keeps the pair as its unit.
        spec = replace(
            spec,
            fe_vcov=wz_coefficient_vcov(
                spec.fe_vcov, explicit=spec.explicit_vcov, pair_name=pair_name
            ),
            explicit_vcov=True,
        )
    _initial_domain(spec)
    cluster = ClusterCodes(
        pair_name, next(b.codes for b in sample.grouping.blocks if b.name == pair_name)
    )
    panel = layout(
        source,
        pair_name,
        sample.design.matrix.shape,
        bounds,
        workspace=coefficient_workspace,
    )
    # The exact coordinates depend on the fixed codes only: form them once
    # here, carried by value and digest-checked at every later evaluation.
    coefficient = WZControls(
        source,
        controls.absorbed_options,
        spec.options,
        bounds,
        identify(panel, bounds).held,
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
        bounds,
        workspace=coefficient_workspace,
    )
    return spec, coefficient


def _joint_scale(
    spec: Specification,
    sample: FESample,
    request: CovarianceRequest,
    terms: tuple[tuple[str, ...], ...],
    *,
    execution: NumericExecution | None = None,
    identified: tuple[str, bytes] | None = None,
    formula_intercept: int | None = None,
) -> tuple[AbsorbedScale, CertificateBasis | None]:
    """Bind the same covariance and one actual operation's structural preparation.

    ``identified`` is the same fit's exact WZ coordinates from the coefficient
    setup, reused only for the identical structure digest.
    """
    if not isinstance(spec.fe_vcov, ClusterRequest) or spec.fe_vcov.adjustment != "wz":
        if (
            execution is not None
            and spec.estimator.mean.omega != 0
            and _selected_root(spec) is None
        ):
            producer = execution.prepare_basis
            prepare_count: (
                Callable[[CertificateBasis, list[IntegerArray]], AbsorbedCount | None]
                | None
            ) = None
            try:

                def count_from_basis(
                    basis: CertificateBasis,
                    clusters: list[IntegerArray],
                ) -> AbsorbedCount | None:
                    return _owned_convention_count(basis, clusters, producer)

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
        WZSource(
            terms,
            grouping,
            sample.data.row_labels,
            formula_intercept=formula_intercept,
        ),
        controls.absorbed_options,
        spec.options,
        joint_bounds(controls.max_dense_bytes),
    )
    if sample.design is None or request.name is None:
        raise wz_unavailable("current original slopes and exact pair source required")
    panel = layout(wz.source, request.name, sample.design.matrix.shape, wz.bounds)
    wz = replace(wz, identified=identify(panel, wz.bounds, identified).held)
    basis = None if execution is None else execution.prepare_basis(sample)
    return AbsorbedScale(
        request,
        wz=wz,
        _wz_information=(_WZInformation() if spec.bias_correction == "wz" else None),
    ), basis


def _cancelled_blocks(
    grouping: Grouping, positions: IndexArray
) -> tuple[NamedFEBlock, ...]:
    """Preserve each original FE level, including an interaction tuple, as a scalar."""
    # The owned builder uses the source's blocks and a view of its row vector.
    if (
        type(grouping) is Derived
        and _owned_fixed is _sample_fe._owned_fixed
        and _sample_fe._fixed_original()
        and _owned_source_original()
        and type(grouping.source) is Source
        and grouping.blocks is grouping.source.blocks
        and type(positions) is np.ndarray
        and positions.dtype == np.intp
        and type(grouping.positions) is np.ndarray
        and grouping.positions.dtype == np.intp
        and grouping.positions.base is grouping.source.rows
        and np.array_equal(grouping.positions, positions)
    ):
        native = []
        for block in grouping.blocks:
            labels = _cancelled_label_partition(block, positions.size)
            if labels is None:
                break
            native.append(NamedFEBlock(block.name, labels))
        else:
            return tuple(native)
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


def _cancelled_dimension(spec: Specification) -> FEVcovRequest:
    """Return the requested covariance; an explicit ``ssc`` alone keeps the groups."""
    if spec.explicit_vcov:
        return spec.fe_vcov
    return "cluster_cancelled"


def _cancelled_sample(
    problem: PreparedProblem,
    spec: Specification,
    frame: PreparedFrame,
    full: OrderSample,
) -> (
    tuple[
        OrderSample,
        RetainedOutcomeSnapshot | ClosedRows | None,
        CollinearityInvocation | None,
    ]
    | str
):
    """The joint route's sample step for a cancelled fit with zero outcomes.

    Owner's ruling 70: with zeros in the outcome every route drops the separated
    observations and the singletons that joint absorbed estimation drops
    (``ppmlhdfe``), then fits what remains.  The walk is the joint route's own
    (:func:`_numeric_joint_sample`, every block estimated, the request of the joint
    default), so that the rows and the refusals are the same.  Returns the status of
    a walk that did not complete; else the order sample to fit, ``full`` itself when
    the walk removed nothing, with the walk's retained outcome when it removed rows.
    At ``tau = 2`` the walk removes rows by steps 5 and 6 only and then tests the
    joint estimating equation for a recession direction; that verdict is a property
    of the joint equation, not of the cancelled one (the two estimators differ away
    from ``tau = 1``).  The cancelled fit therefore runs steps 5 and 6 alone there,
    keeps their rows (their ledger is the accounting) and is not refused by a
    recession of the joint equation: its own certificate decides.  The last element
    is the walk's collinearity analysis of the rows to fit, which the cancelled fit
    uses when it is that of its own problem; none at ``tau = 2``.
    """
    prep, controls = problem.retained_preparation, spec.fe_solver_options
    assert prep is not None
    assert controls is not None
    omega = spec.estimator.mean.omega
    request = controls.sample_request
    if request is None:
        request = SampleRequest.default(
            omega,
            spec.estimator.variance,
            problem.positive_only,
            default_diagnostic(),
        )
    snapshot: RetainedOutcomeSnapshot | None = None
    invocation: CollinearityInvocation | None = None
    if request.recession is not None:
        # tau = 2: only steps 5 and 6 remove rows, and the joint equation's recession
        # analysis is not run for a cancelled fit (docstring).
        first = start(frame, request)
        ledger = close(
            first.fixed,
            first.ledger,
            first.outcome,
            keep_singletons=request.keep_singletons,
        ).ledger
        if not ledger.count:
            return "not_identified"
        kept = ledger.positions
    else:
        _, _, outcome, snapshot, execution = _numeric_joint_sample(
            problem, prep, frame, request, omega
        )
        if execution is not None:
            execution.release()
        if outcome.sample is None or outcome.status != "complete":
            return outcome.status
        kept = outcome.sample.positions
        invocation = outcome.collinearity
    if np.array_equal(kept, full.positions):
        return full, None, invocation
    base = _isolated_sample_base(frame.base)
    mask = np.zeros(base.nobs, dtype=np.bool_)
    mask[kept] = True
    codes = np.array(base.codes, dtype=np.int8, copy=True)
    codes[base.complete & ~mask] = SUBSET
    base = replace(
        base, complete=_flags(base.complete & mask), codes=frozen_codes(codes)
    )
    fitted = base.for_order(omega, positive_only=problem.positive_only)
    if request.recession is not None:
        return fitted, ClosedRows(ledger, fitted), None
    return fitted, snapshot, invocation


def _note_omitted(
    names: tuple[str, ...],
    reasons: Iterable[str | None],
    design: Design,
    *,
    stacklevel: int,
) -> None:
    """One ``coverage.collinear_omitted`` note per omitted regressor, ``stacklevel`` up.

    Owner's ruling 65: ``fixest`` and ``ppmlhdfe`` name every regressor they omit
    because of collinearity; the note is issued once per fit, before the solve, on
    both routes, exactly and nearly collinear columns alike.  The formula's intercept,
    which the fixed effects absorb, is not named: ``fixest`` and ``pyfixest`` remove
    it without a message.
    """
    intercept = wz_formula_intercept(design)
    for j, (name, reason) in enumerate(zip(names, reasons, strict=True)):
        if reason is not None and j != intercept:
            warnings.warn(
                CoverageWarning("coverage.collinear_omitted", {"name": name}),
                stacklevel=stacklevel,
            )


# The causes of a refused cancelled geometry that are properties of the data: kept
# regressors too ill conditioned for the route's Gram-based coordinates at their
# resolution (original-space reconstruction, null space and aliases; the fresh
# spectral rank, the rank and reasons of the slopes; near constancy within the
# cancelled block; the left inverse of the maps; unresolved spectral gaps inside the
# admission).  They end the fit ``singular_system``.  Every other cause is raised:
# the bindings of the fit's own rows, labels and columns, the exact incidence
# (``FE_quotient_rank``), missing evidence, a budget, and a built-in error inside the
# admission (``ValueError``, ``TypeError``, ``FloatingPointError``, ``LinAlgError``).
_CONDITIONING_CAUSES = (
    "original_reconstruction",
    "original_null",
    "slope_alias",
    "fresh_rank",
    "slope_rank",
    "slope_reasons",
    "raw_constancy",
    "left_inverse",
    "GeometryRankError",
)


def _cancelled_refused(
    problem: PreparedProblem,
    stated: Specification,
    frame: PreparedFrame,
    model: ModelFrame | None,
    *,
    sample: OrderSample,
    selected: CancelledSelection,
    walked: tuple[OrderSample, object] | None,
    status: str,
) -> Result:
    """A cancelled fit that ends with ``status`` before its solve, without estimates.

    ``collinearity_unresolved`` or ``collinearity_refused``: the shared analysis did
    not complete, the statuses of the joint route's sample step.  ``singular_system``:
    the analysis kept regressors that the route's coordinates, built on Gram matrices,
    cannot represent at their resolution (an ill-conditioned identified Gram, whose
    maps fail the original-space checks).  The result carries no coefficient;
    ``on_failure="raise"`` raises it as any status that is not certified.  ``stated``
    is the specification as called (a named HC request is set aside only for a
    solve).
    """
    spec = stated
    controls = spec.fe_solver_options
    assert controls is not None
    inner = controls.absorbed_options
    assert inner is not None
    parts = model if model is not None else frame.frame
    unsolved = (
        walked[1]
        if walked is not None and isinstance(walked[1], RetainedOutcomeSnapshot)
        else None
    )
    estimator = replace(
        spec.estimator,
        fixed_effects=FixedEffectFormulation(
            selected.block_order, selected.cancelled_name
        ),
    )
    view = FEView(
        estimator,
        sample,
        frame.labels,
        frame.columns,
        frame.subset_missing,
        parts.parts,
        parts.design,
        problem.offset_design,
        spec.options,
        (stated.fe_vcov, stated.ssc, stated.bread),
        spec.start,
        announce_defaults=_announce_defaults(problem, stated),
    )
    state = AbsorbedState(
        None,
        None,
        unsolved,
        None,
        status,
        inner,
        inner.absorbed,
        controls.max_dense_bytes,
    )
    return Result(AbsorbedResultBackend(view, state))


def _without_near_collinear(
    matrix: FloatArray,
    scale: FloatArray,
    names: tuple[str, ...],
    positions: IntegerArray,
    grouping: Grouping,
    controls: DiagnosticControls,
    *,
    mode: str,
    invocation: CollinearityInvocation | None = None,
) -> tuple[FloatArray, CollinearityAnalysis]:
    """The slopes of a cancelled fit, with relative-policy omissions represented.

    ``matrix`` holds the sample's scaled columns and ``scale`` their powers of two:
    the shared analysis binds the original columns, as on the joint route, then
    evaluates ruling 92 in normalized coordinates.

    Owner's ruling 92: a relative-policy omission is dropped by name on the
    cancelled route as on the joint route.  The within-design
    analysis of the joint route decides, once per fit: the admission of the slopes
    checks the geometry against this analysis (returned with the columns) and does
    not run it again.  A column that it omits by the pivot rule alone (near, not
    exact, dependence) is replaced here by a copy of the first kept column when that
    column comes before it, else by zeros, so that the cancelled geometry, which keeps
    the earlier of two equal columns, omits it by its own exact rules and reports it
    by name.  A relative FE omission is represented by zeros, with no slope alias
    or exact FE-span assertion.  The values of an omitted column enter no estimate.
    An analysis that
    does not complete changes nothing.  ``invocation``, the analysis of the sample
    step's walk with zero outcomes, is used when it analyzed this problem with these
    controls and mode; else the analysis runs here.
    """
    problem = CollinearityProblem(
        matrix * scale, names, cast("IndexArray", positions), grouping
    )
    if (
        invocation is not None
        and invocation.controls == controls
        and invocation.mode == mode
        and same_problem(invocation.problem, problem)
    ):
        analysis = invocation.analysis
    else:
        analysis = analyze_collinearity(problem, controls, mode=mode, near=True)
    if analysis.reasons is None or analysis.kept is None:
        return matrix, analysis
    near = near_omitted(analysis)
    if not near:
        return matrix, analysis
    replaced = np.array(matrix, dtype=np.float64, copy=True)
    first = int(analysis.kept[0]) if analysis.kept.size else len(names)
    for j in near:
        replaced[:, j] = (
            matrix[:, first] if analysis.reasons[j] == COLLINEAR and first < j else 0.0
        )
    return replaced, analysis


def _without_cancelled_constant(
    selected: CancelledSelection, analysis: CollinearityAnalysis, codes: IndexArray
) -> CancelledSelection:
    """The selection, an absorbed column constant within the cancelled block zeros.

    A column that the shared analysis absorbs and that is exactly constant within the
    levels of the cancelled block has a centered column of zero in exact arithmetic;
    its rounding residue would fail the geometry's original-space reconstruction.  As
    zeros, the geometry omits it as constant, as the analysis decided.
    """
    slopes = selected.inputs.slopes
    matrix = np.asarray(slopes.raw_slopes)
    found = [
        j
        for j, reason in enumerate(analysis.reasons or ())
        if reason == ABSORBED
        and np.any(matrix[:, j] != 0)
        and _within_levels(matrix[:, j], codes)
    ]
    if not found:
        return selected
    replaced = np.array(matrix, dtype=np.float64, copy=True)
    replaced[:, found] = 0.0
    inputs = selected.inputs._replace(slopes=slopes._replace(raw_slopes=replaced))
    return replace(selected, inputs=inputs)


def _without_named_hc(
    spec: Specification,
) -> tuple[Specification, HCInferenceRequest | None]:
    """The specification the solve reads, and the named HC request set aside.

    A named HC request does not enter the solve or a coefficient correction of a
    cancelled fit: the fit is the one without ``vcov`` (its certificates read the
    one-way scale on the cancelled block, the BBP or WZ correction its default
    sampling unit), and the request is evaluated once at the accepted point
    (rulings 55 and 63).
    """
    if not isinstance(spec.fe_vcov, HCRequest):
        return spec, None
    request = HCInferenceRequest.from_hc_request(
        spec.fe_vcov, ssc=spec.ssc, bread=spec.bread
    )
    return replace(spec, fe_vcov="robust", explicit_vcov=False, ssc=OMITTED), request


_CANCEL_STRUCTURE = cancel_structure
_FIT_CANCELLED_PRIVATE = fit_cancelled_private


def _cancelled_layout_inputs(
    selected: CancelledSelection, positions: IndexArray, offset: FloatArray
) -> tuple[GeometryInputs, CancelStructure, FloatArray]:
    """Build the selected layout and center the offset in its group order."""
    inputs = selected.inputs
    structure = cancel_structure(inputs.cancelled, inputs.retained, positions=positions)
    centered = center_sorted(structure, to_sorted(structure, offset))
    return inputs.slopes, structure, to_original(structure, centered)


def _cancelled(
    problem: PreparedProblem, spec: Specification, stacklevel: int
) -> Result:
    """Normal selected-block route through the accepted complete-fit lifecycle.

    ``stacklevel`` is :func:`_joint`'s, for a refusal of the sample step.
    """
    prep, controls = problem.retained_preparation, spec.fe_solver_options
    assert prep is not None
    assert controls is not None
    assert controls.cancelled_controls is not None
    assert spec.cancel is not None
    stated = spec
    spec, hc_request = _without_named_hc(spec)
    frame = prep.context.new_fit_frame()
    model = frame._frame if prep.context._operation is not None else None
    sample = frame.for_order(
        spec.estimator.mean.omega, positive_only=problem.positive_only
    )
    walked = None
    invocation: CollinearityInvocation | None = None
    if spec.estimator.mean.omega != 0 and bool((sample.data.y == 0).any()):
        stepped = _cancelled_sample(problem, spec, frame, sample)
        if isinstance(stepped, str):
            return _joint(
                problem,
                _joint_controls_after_cancel(stated),
                stacklevel + 1,
                _refused_cancel=str(spec.cancel),
            )
        invocation = stepped[2]
        if stepped[1] is not None:
            walked = (sample, stepped[1])
            sample = stepped[0]
    assert frame.effects is not None
    grouping = _owned_fixed(frame.effects, sample.positions).grouping
    blocks = _cancelled_blocks(grouping, sample.positions)

    decided, analysis = _without_near_collinear(
        sample.design.matrix,
        sample.design.scale,
        sample.data.column_names,
        sample.positions,
        grouping,
        controls.cancelled_controls.diagnostic,
        mode=controls.cancelled_controls.collinear,
        invocation=invocation,
    )
    slopes = GeometryInputs(
        decided, sample.positions, sample.data.column_names, analysis
    )
    selected = select_cancelled_block(
        slopes, blocks, cancel=cast("str | Literal[True]", spec.cancel)
    )
    if analysis.status != COMPLETE:
        return _cancelled_refused(
            problem,
            stated,
            frame,
            model,
            sample=sample,
            selected=selected,
            walked=walked,
            status="collinearity_refused"
            if analysis.status == REFUSED
            else "collinearity_unresolved",
        )
    omitted = (
        sample.data.column_names,
        analysis.reasons or (),
        (model if model is not None else frame.frame).design,
    )
    selected = _without_cancelled_constant(
        selected, analysis, grouping.blocks[selected.cancelled_index].codes
    )
    slopes, structure, offset = _cancelled_layout_inputs(
        selected, sample.positions, sample.problem.offset
    )
    dependence = (
        _bind_owned_metadata_parts(
            frame,
            sample,
            weights=sample.data.weights,
            input_key=prep.context.input_key,
        )
        if (
            model is not None
            and frame.dependence is not None
            and any(
                "_retained_sources" in vars(column)
                for column in frame.dependence._columns.values()
            )
        )
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
    if spec.explicit_vcov or not isinstance(spec.ssc, Omitted):
        dimension = _cancelled_dimension(spec)
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
    mapped = None
    if spec.start is not None:
        # A start is one value per reported coefficient (docs/reference/options.md):
        # the slopes the geometry keeps; an omitted slope takes zero, the geometry's
        # normalization of omitted coefficients.
        bounds = controls.cancelled_controls
        geometry = selected.build_geometry(
            gram_rtol=bounds.gram_rtol,
            gap_min=bounds.gap_min,
            max_bytes=bounds.geometry_bytes,
        )
        kept = geometry.kept_slope_indices
        names = tuple(sample.data.column_names[j] for j in kept)
        reported = _mapped_start(spec, names, sample.design.scale[kept])
        assert reported is not None
        full = np.zeros(geometry.n_columns_full)
        full[kept] = reported
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
    assert spec.cancel is not None
    short_group = (
        ShortGroupHandoff()
        if sampling is not None
        and sampling.method == "short_group"
        and sample.omega == 1.0
        and type(spec.estimator.variance) is PowerVariance
        and spec.estimator.variance.tau == 1.0
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
        hc=hc_request,
        merit_metric_given=controls.merit_metric_given,
        report_secondary=False,
        short_group=short_group,
    )
    fit = (
        partial(
            fit_cancelled_private,
            _prepared_layout=_PreparedCancelledLayout(selected.inputs, structure),
        )
        if fit_cancelled_private is _FIT_CANCELLED_PRIVATE
        and cancel_structure is _CANCEL_STRUCTURE
        and all(type(block.labels) is _CancelledLabelPartition for block in blocks)
        else fit_cancelled_private
    )
    try:
        envelope = fit(
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
    except SpecificationError as error:
        if error.fields.get("method") != NO_RETAINED_SLOPE:
            raise
        # Every regressor is collinear with the fixed effects on the fitted sample:
        # no slope is left for the cancelled equation.  The fit is the joint route's
        # (owner's ruling 70: the cancelled route reaches the joint route's outcome),
        # which drops the regressors by name and estimates the effects alone.
        return _joint(
            problem,
            _joint_controls_after_cancel(stated),
            stacklevel + 1,
            _refused_cancel=str(spec.cancel),
        )
    except (GeometryAdmissionError, GeometryRankError) as error:
        if (
            isinstance(error, GeometryAdmissionError)
            and error.cause not in _CONDITIONING_CAUSES
        ):
            raise
        _note_omitted(*omitted, stacklevel=stacklevel)
        return _cancelled_refused(
            problem,
            stated,
            frame,
            model,
            sample=sample,
            selected=selected,
            walked=walked,
            status="singular_system",
        )
    _note_omitted(*omitted, stacklevel=stacklevel)
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
            fit_point=None if short_group is None else short_group.take(),
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
        (stated.fe_vcov, stated.ssc, stated.bread),
        spec.start,
        announce_defaults=_announce_defaults(problem, stated),
    )
    assert controls.absorbed_options is not None
    state = CancelledState(
        envelope,
        dependence,
        grouping,
        controls.cancelled_controls,
        controls.absorbed_options,
        correction,
        walked,
    )
    return Result(CancelledResultBackend(view, state))


def fit_fe_prepared(
    problem: PreparedProblem,
    spec: Specification,
    *,
    stacklevel: int = 1,
    _light: bool = True,
) -> Result:
    """Dispatch a genuine normal private FE fit, never the no-FE solver.

    ``stacklevel`` is that of :func:`~powermean.interface.dispatch.fit_prepared`: the
    frames above it at which a warning names the caller's line.  Fixed-effect log-OLS
    is tried first on the light pipeline (:mod:`powermean.interface.fe_light`, owner's
    ruling 87), which declines to the exact route below; ``_light=False`` is that
    exact route, which the light result calls for what it does not hold.
    """
    _initial_domain(spec)
    exact_count_domain(spec.ssc, spec.estimator.mean.omega)
    if problem.frame.columns.weights is not None:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "weights",
                "value": problem.frame.columns.weights,
                "valid": "no weights column with fixed effects in this version",
            },
        )
    if _light and spec.estimator.mean.omega == 0 and spec.cancel is None:
        light = light_fit(
            problem,
            spec,
            lambda: fit_fe_prepared(problem, spec, stacklevel=stacklevel, _light=False),
            stacklevel=stacklevel + 4,
        )
        if light is not None:
            return light
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
        inner = (
            vars(controls).get("absorbed_options")
            if type(controls) is ResolvedFESolverOptions
            else None
        )
        backend = (
            vars(inner).get("basis_backend") if type(inner) is AbsorbedOptions else None
        )
        if type(backend) is str and backend == "cpp":
            native_basis = _count_basis._resolve_native_echelon()
        if spec.cancel is not None:
            return _cancelled(problem, spec, stacklevel + 4)
        # The warning frames are unchanged: no new forwarding fit function.
        return (
            _joint(problem, spec, stacklevel + 4)
            if native_basis is None
            else _joint(problem, spec, stacklevel + 4, _native_basis=native_basis)
        )
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
_COUNT_MEMBERS = tuple(
    (kind, name, vars(kind).get(name, _COUNT_ABSENT))
    for kind, names in (
        (NumericExecution, ("prepare_basis", "__getattribute__", "__getattr__")),
        (CertificateBasis, ("__getattribute__", "__getattr__", "rank", "grouping")),
        (
            IndicatorRank,
            (
                "__getattribute__",
                "__getattr__",
                "__init__",
                "levels",
                "n_components",
                "pairwise_components",
            ),
        ),
        (Grouping, ("__getattribute__", "__getattr__", "blocks", "positions")),
        (_COUNT_DERIVED, ("__getattribute__", "__getattr__", "blocks", "positions")),
        (_COUNT_GROUP_BLOCK, ("name", "codes", "counts")),
    )
    for name in names
)
_COUNT_PRODUCERS = tuple(
    (module, name, vars(module)[name])
    for module, names in (
        (
            _count_execution,
            (
                "NumericExecution",
                "_prepare_fit_basis",
                "_same_group_storage",
                "_retained_header",
                "Grouping",
                "Derived",
                "Block",
                "NumericFit",
            ),
        ),
        (
            _count_basis,
            (
                "_prepare_fit_basis",
                "_fit_basis_prepared",
                "_operational_basis_prepared",
                "_indicator_rank_prepared",
                "_indicator_rank_counted",
                "components",
                "_fit_pair_counts_generated",
                "CertificateBasis",
                "IndicatorRank",
            ),
        ),
        (_count_rank, ("_prepared_count", "_prepared_count_original")),
        # The storage comparator above is canonical lower code; pin the names
        # it actually resolves, so a changed helper keeps the original route.
        (
            _count_live,
            ("_same_group_storage", "_retained_header", "Grouping", "Derived", "Block"),
        ),
    )
    for name in names
)


def _owned_convention_count(
    basis: CertificateBasis,
    clusters: list[IntegerArray],
    producer: object,
) -> AbsorbedCount | None:
    """Use fresh structural scalars only at the live original owned callback site."""
    if (
        type(producer) is not MethodType
        or type(producer.__self__) is not _COUNT_EXECUTION
        or producer.__func__ is not _COUNT_MEMBERS[0][2]
        or NumericExecution is not _COUNT_EXECUTION
        or CertificateBasis is not _COUNT_BASIS
        or type(basis) is not _COUNT_BASIS
        or IndicatorRank is not _COUNT_RANK
        or _count_rank.BlockCount is not _COUNT_BLOCK
        or _count_rank.AbsorbedCount is not _COUNT_RECORD
        or Grouping is not _COUNT_GROUPING
        or _count_execution.Derived is not _COUNT_DERIVED
        or _count_execution.Block is not _COUNT_GROUP_BLOCK
    ):
        return None
    try:
        if (
            any(
                vars(kind).get(name, _COUNT_ABSENT) is not original
                for kind, name, original in _COUNT_MEMBERS
            )
            or any(
                vars(module).get(name) is not original
                for module, name, original in _COUNT_PRODUCERS
            )
            or not _count_basis._fit_pair_counts_generated()
            or not _count_rank._prepared_count_original()
        ):
            return None
        rank = vars(basis).get("rank")
        if type(rank) is not _COUNT_RANK:
            return None
        grouping = vars(basis).get("grouping")
        numeric = vars(producer.__self__).get("numeric")
        if (
            type(numeric) is not NumericFit
            or any(type(c) is not np.ndarray for c in clusters)
            or not _count_execution._same_group_storage(
                grouping, vars(numeric).get("_grouping")
            )
        ):
            return None
    except MemoryError:
        return None  # Optional selection only, before any count/nesting action.
    blocks = cast("Grouping | _count_execution.Derived", grouping).blocks
    return _count_rank._prepared_count(
        tuple((block.name, block.codes) for block in blocks),
        rank.levels,
        rank.n_components,
        rank.pairwise_components,
        clusters,
    )
