"""The internal handoff from a fit to its result, validated against the retained inputs.

The handoff binds the records of a fit to the same coefficient point, equation, sample
and coordinates; this does not waive the covariance-dependent coefficient gate when a
candidate is initially certified.  The public ``fit`` composes the stages
and hands their records to :func:`bind`, which forms the :class:`~powermean.results.
retained.RetainedFit` of a result.  Nothing is solved, certified afresh, reselected or
trained here, and no record of a stage is modified or replaced.

What a :class:`Handoff` carries, explicitly, because no record of the earlier stages
carries it alone: the estimator and its equation; the prepared frame of the sample
(for its owned fitted right-hand side, its labels, the columns it read and the base of
the construction); the sample of the mean order; the requested inference; the resolved
settings; and the authentic certificate of the point, as the :class:`~powermean.solve.
status.Conclusion` of a solve or as a bare :class:`~powermean.solve.certificate.
Certificate` of a supplied point, or on the linear route the candidate of the
least-squares solve, with the fresh :class:`~powermean.solve.linear_status.
LinearConclusion` of its point, or on the power route the
:class:`~powermean.solve.no_point.StartFailure` of an automatic start that had no
finite value.  A certificate carries a candidate and an evaluation, not the
system, the estimator or the inference that supplied its standard errors; the
fingerprint of a sample omits the order and the variance; and the labels of a prepared
frame are taken by position for any sample with as many input rows.  Equal coefficient
bytes, equal counts or an equal fingerprint therefore authenticate nothing, and
:func:`bind` checks the binding against the retained inputs themselves:

- the sample is the frame's and the preparation's (:mod:`powermean.results.binding`):
  the retained rows of the frame's evaluated design and outcome are the sample's
  numeric record, column for column; the accounting, the raw offset or exposure, the
  weights with their kind and values, the row identity and the classifications of the
  sample are those of the owned inputs of the preparation at the retained positions;
  and the design of the solver is the retained design divided by the column scale,
  bit for bit as the preparation formed it;
- the estimator is the sample's: the same mean order and the same route;
- the certificate is this equation's, at this point, on these rows: the joint system
  of the sample and the estimator, evaluated afresh at the certificate's candidate,
  reproduces the certificate's evaluation exactly (residuals, curvatures, weights,
  sums, contributions and units), so that a certificate of the same ``gamma`` under
  another working variance, another sample or another offset is refused, and a
  certificate without an evaluation is bound only where the candidate is not finite
  or the retained equation is not representable at it, as the certificate reports;
  the frequency flag is the stated kind of the weights; the controls are those of the
  resolved settings; and a reported scale, where the provider formed one, was formed
  on this coordinate basis: its dense map is the order scale's transform, or its
  direct map is the order scale itself (the same order and column scale).  Two
  samples whose transformed problems coincide, as the same regressors at two column
  scales or at two orders of opposite sign do, evaluate alike: their certificates are
  told apart by that reported scale, which every accepted certificate carries;
- the certificate's uncertainty is the requested inference's: the standard errors
  that its coefficient gate compared the Newton correction with are those that the
  inference requested at the fit forms at this evaluation, by the provider
  arithmetic on the retained influence record (the robust covariance, or the one of
  the requested classification, the ``"stata"`` factors and the map of the order
  scale, dense or direct), or, for a named HC method, by
  :func:`~powermean.inference.hc.joint_hc` of the retained system at the
  certificate's evaluation or :func:`~powermean.inference.hc.linear_hc` of the
  captured solve, under the convention in force at the fit, or,
  for a typed cluster request, by the convention engine at the influence record of
  that evaluation (:mod:`powermean.results.cluster_covariance`: the final reported
  matrix's standard errors of an adjusted signed request), bit
  for bit; a certificate whose gate was passed over the
  standard errors of another inference, a robust certificate under a clustered
  request say, is refused rather than reported as certified under uncertainty it
  never measured.  Nothing is certified anew and the certificate is not replaced; a
  covariance requested later (``vcov``) is a recomputation that leaves the
  certification alone;
- a conclusion is its certificate's: the candidate's retained coefficients are the
  certificate's bytes, the native reason is that of the retained attempt, the status
  is the one that the status rules compose, and the solver ran with the resolved
  settings; a direct certificate of a supplied point is never paired with the trace
  of a solve that moved elsewhere; a start requested on the reported scale, where one
  is handed over, is the solver's start through the forward map;
- on the linear route, the candidate solved the retained problem: its dimension,
  units and weights are the sample's, and its residuals are ``(l - o) - X beta`` of
  the retained response, offset and scaled design, bit for bit; a linear conclusion
  is its candidate's and its certificate's: the certificate's point is the
  candidate's coefficients, the original linear equation of the sample evaluated
  afresh at that point reproduces the certificate's evaluation exactly (residuals,
  weights, sums, contributions and units), its reported scale is this basis and its
  standard errors the requested inference's at the fresh evaluation, and the status
  is the one that :func:`~powermean.solve.linear_status.linear_status` composes; a
  candidate without a point carries no certificate and the status of its own reason;
  a captured linear solve, where one is handed over, holds the retained
  equation as the solve received it and the very candidate handed over, bit for bit,
  and is retained with the influence for the named HC covariances of the linear path;
- a start failure is this sample's: the mean whose logarithm had no finite value is
  the plain mean of the transformed outcome over the rows of positive weight, its
  logarithm is not finite, and the intercept is a column of ones of the design;
- a joint root conclusion outside ``[1, 2]`` (:mod:`powermean.solve.joint_root`) is
  that route's: the estimator's power variance lies outside ``[1, 2]``; the settings
  record every root-Newton option, and the solve ran with them; the native reason is
  the target stage's and the status the one of
  :func:`~powermean.solve.joint_root.joint_root_status`; a point without a root
  stage carries no reported scale and no acceptance (an exact root may take zero
  steps in an actual stage); the initializer is the
  caller's start where one was requested, else the nearest convex fit of the rule,
  whose certificate reproduces the fresh evaluation of the convex equation of THIS
  sample (the same rows, design, offset and weights) at its point with the fit's
  controls, and whose candidate, status and reason are that certificate's; the root
  solve opened at the initializer's point; and its merit factor is the one that the
  requested metric gives afresh at that point.  The certificate's curvature gate is
  the equation's own: required for a convex variance, waived outside ``[1, 2]``.

The status of a solve is the conclusion's.  For a supplied point without a solve the
status rules give a status from the gates alone where they do (``verified``,
``verified_secondary``, ``nonfinite``, ``singular_system``); a finite, regular point
that fails the score, the index or the coefficient gate has no status without the native
reason of its solve, and the handoff refuses it rather than invent one.  Native solver
success is not certification: only the certificate's level makes a point certified.
Failed candidates keep their last state and their diagnostics; their inference is
withheld by the result.  A bare linear candidate handed over without its conclusion
keeps no certificate and no status, and the result reports its point without inference.

What is retained is owned (:mod:`powermean.results.owned`): the sample, the certificate,
the candidate of the solve and the linear candidate are re-formed, before the checks,
with arrays of their own, so that a later edit of a record the caller kept, of the shape
of an array say, reaches nothing retained; the values, types and diagnostics are those
of the records handed over.  The influence record and the default covariance are formed
once, after the checks, from the certificate's evaluation (the joint one, or the fresh
linear one; or the bare linear candidate with its design), by the primitives
(:mod:`powermean.results.point`); with ``keep_influence=False`` the influence record is
discarded after the default covariance was formed, and a later request that needs it is
refused by the result, which never rebuilds it from the certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean.data.computation import Computation
from powermean.data.retained import OrderSample
from powermean.data.specs import Estimator, Inference
from powermean.fe.admission_records import WorkLog
from powermean.formula.frame import Design
from powermean.inference.convention import reported_covariance
from powermean.inference.hc import LinearSolve, linear_request, solve_request
from powermean.inference.provider import direct_standard_errors
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.binding import check_sample, refuse
from powermean.results.owned import (
    owned_array,
    owned_candidate,
    owned_certificate,
    owned_design,
    owned_expression_design,
    owned_linear,
    owned_linear_certificate,
    owned_sample,
)
from powermean.results.point import (
    Point,
    frequency_of,
    joint_provider,
    linear_certified_point,
    linear_point,
    power_point,
)
from powermean.results.retained import AnyCandidate, AnyCertificate, RetainedFit
from powermean.solve.assessment import certificate_controls
from powermean.solve.cancelled_records import (
    ROOT_DOMAINS,
    RootOptions,
    iter_continuation_variances,
)
from powermean.solve.certificate import Certificate, certify, outside_convex
from powermean.solve.iols.adapter import (
    apply_transformed_normalization,
    build_transformed_normalization_state,
)
from powermean.solve.iols.candidate import IolsCandidate
from powermean.solve.iols.kernel import DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS
from powermean.solve.iols.options import (
    IolsOptions,
    effective_backend_request,
    options_to_kernel_control,
)
from powermean.solve.joint_root import (
    joint_root_status,
    merit_factor,
    nearest_convex,
    same_system,
    summary_of,
)
from powermean.solve.joint_root_records import (
    JointRootCandidate,
    JointRootStage,
    StageSummary,
)
from powermean.solve.linear import LinearCandidate
from powermean.solve.linear_certificate import LinearCertificate
from powermean.solve.linear_evaluation import LinearEvaluation, LinearProblem
from powermean.solve.linear_status import LinearConclusion, linear_status
from powermean.solve.newton_trace import NativeReason, NewtonCandidate
from powermean.solve.no_point import StartFailure
from powermean.solve.options import SolverOptions, resolve_options
from powermean.solve.reported import ReportedScale
from powermean.solve.starts import automatic_start
from powermean.solve.status import Conclusion, Status, compose_status
from powermean.systems.joint import JointEvaluation, JointSystem

# The arrays of an evaluation that the fresh evaluation must reproduce exactly.
EVALUATION_ARRAYS: tuple[str, ...] = (
    "gamma",
    "log_mean",
    "residual",
    "curvature",
    "expected_curvature",
    "weights",
    "moment",
    "bread",
    "information",
    "contributions",
)
# The arrays of a linear evaluation that the fresh one must reproduce exactly.
LINEAR_EVALUATION_ARRAYS: tuple[str, ...] = (
    "gamma",
    "residual",
    "weights",
    "moment",
    "bread",
    "contributions",
)


@dataclass(frozen=True, eq=False)
class Handoff:
    """The records of one fit, handed to :func:`bind` (module docstring).

    Parameters
    ----------
    estimator : Estimator
        The estimator record.
    prepared : PreparedFrame
        The prepared frame that the sample was derived from; read for its owned
        fitted design, its labels, its columns, its subset count and the base of
        the construction, not retained.
    sample : OrderSample
        The sample of the mean order of the estimator.
    inference : Inference
        The covariance requested at the fit, whose standard errors the certificate's
        coefficient gate compared the Newton correction with.
    computation : Computation
        The resolved settings of the solve, as a storage-only snapshot.
    conclusion : Conclusion, optional
        The conclusion of the solve, on the power route.
    certificate : Certificate, optional
        A direct certificate of a supplied point, on the power route, without a
        solve; not together with ``conclusion``.
    linear : LinearCandidate, optional
        The candidate of the linear path, on the linear route.
    keep_influence : bool, default True
        Whether the influence record is kept after the default covariance is formed.
    linear_conclusion : LinearConclusion, optional
        The fresh conclusion of the linear candidate, on the linear route; without
        it the candidate is bare and uncertified.
    failure : StartFailure, optional
        The failure of the automatic start, on the power route, in place of a
        conclusion or a certificate: no candidate exists.
    requested_start : array, optional
        The start that the caller gave on the reported scale, where one was given.
    offset_design : Design, optional
        The fitted design of an offset or exposure expression, for its replay.
    linear_solve : LinearSolve, optional
        On the linear route, the owned equation captured before the
        solve and the candidate that this one solve returned
        (:func:`linear_solve_of`); ``linear`` is then its candidate.  Retained, with
        the influence, for the named HC covariances of the linear path; never formed
        after the fit.
    """

    estimator: Estimator
    prepared: PreparedFrame
    sample: OrderSample
    inference: Inference
    computation: Computation
    conclusion: Conclusion | None = None
    certificate: Certificate | None = None
    linear: LinearCandidate | None = None
    keep_influence: bool = True
    linear_conclusion: LinearConclusion | None = None
    failure: StartFailure | None = None
    requested_start: FloatArray | None = None
    offset_design: Design | None = None
    linear_solve: LinearSolve | None = None


# ----------------------------------------------------------------- the refusals


def _refuse(detail: str) -> ValueError:
    """The refusal of a handoff whose records are not bound to one another."""
    return refuse(detail)


def _check_types(handoff: Handoff) -> None:
    """Each record is of its exact type, and the route names its candidate."""
    checks: tuple[tuple[object, type, str], ...] = (
        (handoff.estimator, Estimator, "estimator"),
        (handoff.prepared, PreparedFrame, "prepared"),
        (handoff.sample, OrderSample, "sample"),
        (handoff.inference, Inference, "inference"),
        (handoff.computation, Computation, "computation"),
    )
    for value, expected, name in checks:
        if type(value) is not expected:
            raise TypeError(f"{name} must be {expected.__name__}")
    if not isinstance(handoff.keep_influence, bool):
        raise TypeError("keep_influence must be a bool")
    optional: tuple[tuple[object, type, str], ...] = (
        (handoff.conclusion, Conclusion, "conclusion"),
        (handoff.certificate, Certificate, "certificate"),
        (handoff.linear, LinearCandidate, "linear"),
        (handoff.linear_conclusion, LinearConclusion, "linear_conclusion"),
        (handoff.failure, StartFailure, "failure"),
        (handoff.offset_design, Design, "offset_design"),
        (handoff.linear_solve, LinearSolve, "linear_solve"),
    )
    for value, expected, name in optional:
        if value is not None and type(value) is not expected:
            raise TypeError(f"{name} must be a {expected.__name__}")
    _check_routes(handoff)


def _check_routes(handoff: Handoff) -> None:
    """The route of the estimator names exactly the records of its route."""
    power = (
        (handoff.conclusion is not None)
        + (handoff.certificate is not None)
        + (handoff.failure is not None)
    )
    linear = handoff.linear is not None or handoff.linear_conclusion is not None
    if handoff.estimator.route == "power" and (
        power != 1 or linear or handoff.linear_solve is not None
    ):
        raise _refuse("the power route takes one conclusion or one certificate")
    if handoff.estimator.route == "linear" and (power or handoff.linear is None):
        raise _refuse("the linear route takes the candidate of the linear path")


def system_of(estimator: Estimator, sample: OrderSample) -> JointSystem:
    """The joint system of the retained problem: the equation that was solved.

    The transformed response and offset, the scaled design, the working variance of
    the estimator and the raw weights, as the dispatch of the solve forms them.
    """
    return JointSystem(
        sample.problem.response,
        sample.design.matrix,
        variance=estimator.variance,
        offset=sample.problem.offset,
        weights=sample.data.weight_values,
    )


def linear_problem_of(sample: OrderSample) -> LinearProblem:
    """The original linear equation of the retained problem (``omega = 0``).

    The log response, the scaled design, the raw offset and the raw weights, as the
    dispatch of the linear solve forms them.
    """
    return LinearProblem(
        sample.problem.response,
        sample.design.matrix,
        offset=sample.problem.offset,
        weights=sample.data.weight_values,
    )


def linear_solve_of(sample: OrderSample) -> LinearSolve:
    """The least-squares solve of the retained problem, captured with it.

    The log response, the scaled design, the raw offset and the raw weights that the
    dispatch of the linear solve hands to :func:`powermean.solve.linear.solve`, with the
    stated kind of the weights, are captured as the owned
    :class:`~powermean.inference.hc.LinearRequest` BEFORE the one solve, which
    :func:`~powermean.inference.hc.solve_request` runs on them.  The dispatch calls this
    once, in place of the bare solve, and hands over the pair and its candidate; nothing
    calls it after a fit.
    """
    return solve_request(
        linear_request(
            sample.problem.response,
            sample.design.matrix,
            offset=sample.problem.offset,
            weights=sample.data.weight_values,
            frequency=frequency_of(sample),
        )
    )


def _same_array(found: object, given: object) -> bool:
    """Whether two arrays (or two ``None``) have one shape, dtype and the same bytes."""
    if found is None or given is None:
        return found is None and given is None
    if not isinstance(found, np.ndarray) or not isinstance(given, np.ndarray):
        return False
    return bool(
        found.shape == given.shape
        and found.dtype == given.dtype
        and found.tobytes() == given.tobytes()
    )


def _check_linear_solve(
    sample: OrderSample, linear: LinearCandidate, solved: LinearSolve
) -> None:
    """The captured solve is the retained equation's, and its candidate the one given.

    The captured equation holds the retained log response, scaled design, raw offset
    and raw weights (as the linear equation of the sample normalizes them) and the
    kind of the weights; its candidate has the reason, dimension, units, weights,
    factorization, coefficients and residuals of the candidate handed over, bit for
    bit.  Nothing is solved here.
    """
    problem = linear_problem_of(sample)
    request = solved.request
    if request.frequency != frequency_of(sample) or not all(
        _same_array(getattr(request, name), getattr(problem, name))
        for name in ("response", "design", "offset", "weights")
    ):
        raise _refuse("the captured linear solve is not that of the retained equation")
    candidate = solved.candidate
    found, given = candidate.factorization, linear.factorization
    same = (
        candidate.reason == linear.reason
        and candidate.dimension == linear.dimension
        and _same_array(candidate.units.identifiers, linear.units.identifiers)
        and _same_array(candidate.units.inactive, linear.units.inactive)
        and candidate.units.nobs == linear.units.nobs
        and _same_array(candidate.weights, linear.weights)
        and _same_array(candidate.coefficients, linear.coefficients)
        and _same_array(candidate.residuals, linear.residuals)
        and (found is None) == (given is None)
        and (
            found is None
            or given is None
            or (
                (found.driver, found.cutoff, found.rank)
                == (given.driver, given.cutoff, given.rank)
                and _same_array(found.singular_values, given.singular_values)
            )
        )
    )
    if not same:
        raise _refuse("the linear candidate is not that of the captured solve")


def _same_units(
    fresh: JointEvaluation | LinearEvaluation, given: JointEvaluation | LinearEvaluation
) -> bool:
    """Whether two evaluations hold the same sampling units, exactly."""
    units = given.units
    return bool(
        np.array_equal(fresh.units.identifiers, units.identifiers)
        and np.array_equal(fresh.units.inactive, units.inactive)
        and fresh.units.nobs == units.nobs
    )


def _same_evaluation(fresh: JointEvaluation, given: JointEvaluation) -> bool:
    """Whether two evaluations hold the same arrays and units, exactly."""
    for name in EVALUATION_ARRAYS:
        if not np.array_equal(getattr(fresh, name), getattr(given, name)):
            return False
    return _same_units(fresh, given)


def _same_linear_evaluation(fresh: LinearEvaluation, given: LinearEvaluation) -> bool:
    """Whether two linear evaluations hold the same arrays and units, exactly."""
    for name in LINEAR_EVALUATION_ARRAYS:
        if not np.array_equal(getattr(fresh, name), getattr(given, name)):
            return False
    return _same_units(fresh, given)


def _same_basis(reported: ReportedScale, scale: OrderScale) -> bool:
    """Whether a reported scale was formed on this coordinate basis.

    A direct map is the order scale itself: the same order and the same column
    scale.  A dense map is the order scale's transform, bit for bit.
    """
    order = reported.order
    if order is not None:
        return order.omega == scale.omega and bool(
            np.array_equal(order.scale, scale.scale)
        )
    return bool(np.array_equal(reported.transform, scale.transform))


_CERT_OPTION_KEYS: frozenset[str] = frozenset(
    {"tol", "tol_secondary", "coef_tol", "rcond_min"}
)


def _resolve_iols_computation(
    computation: Computation,
) -> tuple[IolsOptions, SolverOptions]:
    """Resolve stored values with backend-presence metadata kept separate."""
    raw = computation.options
    cert_dict = {key: value for key, value in raw.items() if key in _CERT_OPTION_KEYS}
    iols_dict = {
        key: value for key, value in raw.items() if key not in _CERT_OPTION_KEYS
    }
    if computation.iols_backend_explicit is False:
        if iols_dict.get("iols_backend") != "cpp":
            raise _refuse("an implicit IOLS backend must retain the cpp preference")
        iols_dict.pop("iols_backend")
    return IolsOptions.from_dict(iols_dict), resolve_options(cert_dict)


def _root_computation(computation: Computation) -> tuple[SolverOptions, RootOptions]:
    """The resolved options and root-Newton options of a joint root fit's settings.

    Every root-Newton option must be recorded: none is reconstructed from a default.
    """
    raw = computation.options
    root = {key: value for key, value in raw.items() if key in ROOT_DOMAINS}
    other = {key: value for key, value in raw.items() if key not in ROOT_DOMAINS}
    if set(root) != set(ROOT_DOMAINS):
        raise _refuse("the settings of a joint root fit record every root option")
    return resolve_options(other), RootOptions.from_dict(root)


def _is_root(handoff: Handoff) -> bool:
    """Whether the handoff concludes a joint root solve outside ``[1, 2]``."""
    conclusion = handoff.conclusion
    return conclusion is not None and isinstance(
        conclusion.candidate, JointRootCandidate
    )


def _check_controls(handoff: Handoff, certificate: AnyCertificate) -> None:
    """The certificate's frequency flag and controls are the fit's."""
    if certificate.frequency != frequency_of(handoff.sample):
        raise _refuse("the certificate's frequency flag is not the kind of the weights")
    if handoff.computation.solver == "iols":
        _, options = _resolve_iols_computation(handoff.computation)
    elif _is_root(handoff):
        options, _ = _root_computation(handoff.computation)
    else:
        options = resolve_options(handoff.computation.options)
    if certificate.controls != certificate_controls(options):
        raise _refuse("the certificate's controls are not those of the settings")


def _check_certificate(
    handoff: Handoff, sample: OrderSample, certificate: Certificate
) -> None:
    """The certificate is this equation's, at this point, on these rows and basis."""
    if certificate.gamma.shape != (sample.data.nparams,):
        raise _refuse("the certificate's candidate has another number of coefficients")
    _check_controls(handoff, certificate)
    if certificate.require_positive_definite == outside_convex(
        handoff.estimator.variance
    ):
        raise _refuse("the certificate's curvature gate is not its equation's")
    system = system_of(handoff.estimator, sample)
    if outside_convex(system.variance):
        if not _is_root(handoff):
            raise _refuse("a selected root requires its actual root conclusion")
        assert handoff.conclusion is not None
        candidate = handoff.conclusion.candidate
        assert isinstance(candidate, JointRootCandidate)
        fresh_certificate = _fresh_root_certificate(
            handoff,
            system,
            certificate,
            candidate.merit_cholesky,
            reported=bool(candidate.stages),
        )
        _check_fresh_certificate(
            certificate, fresh_certificate, "another equation, sample or point"
        )
        return
    given = certificate.evaluation
    if not np.isfinite(certificate.gamma).all():
        if given is not None:
            raise _refuse("the certificate evaluates a candidate that is not finite")
        return
    try:
        fresh = system.evaluate(certificate.gamma)
    except FloatingPointError:
        # The certificate reports no evaluation where the retained equation is not
        # representable at the point: the authentic record of that failure.
        if given is not None:
            raise _refuse(
                "the retained equation is not representable at the point"
            ) from None
        return
    if given is None or not _same_evaluation(fresh, given):
        raise _refuse("the certificate was formed on another equation, sample or point")
    reported = certificate.reported
    if reported is not None and not _same_basis(reported, sample.order_scale):
        raise _refuse("the reported scale of the certificate is not this basis")


def _check_reported_scale(certificate: AnyCertificate, point: Point) -> None:
    """The certificate's standard errors are the requested inference's at this point.

    The provider of the certificate formed its record from the evaluation at the
    candidate, by :func:`powermean.inference.convention.reported_covariance` of the
    convention covariance on its dense map, or by
    :func:`powermean.inference.provider.direct_standard_errors` on its direct map;
    the same arithmetic on the retained influence record of that evaluation, for the
    inference requested at the fit, must give the same bytes; for a typed cluster
    request, the standard errors of its record formed at that evaluation
    (``Point.errors``).  A certificate without a reported scale has no coefficient
    gate to bind (its conditioning failed, or no provider was given) and is left as
    it is.
    """
    reported = certificate.reported
    if reported is None:
        return
    if point.errors is not None:
        expected = point.errors
    elif point.convention is None:
        return
    elif reported.order is not None:
        expected = direct_standard_errors(point.convention, reported.order)
    else:
        expected = reported_covariance(
            point.convention, reported.transform
        ).standard_errors
    if not np.array_equal(expected, reported.standard_errors, equal_nan=True):
        raise _refuse(
            "the certificate's standard errors are not those of the inference "
            "requested at the fit, at this point"
        )


def _check_iols_candidate(handoff: Handoff, candidate: IolsCandidate) -> None:
    """Bind actual IOLS inputs, controls, backend and mapped point to this fit."""
    sample = handoff.sample
    system = system_of(handoff.estimator, sample)
    if not isinstance(system.variance, PowerVariance):
        raise _refuse("IOLS requires Power working variance")
    if not 0 <= candidate.intercept < system.dimension:
        raise _refuse("IOLS has an invalid intercept position")
    control = options_to_kernel_control(candidate.iols_options)
    direct, adapter = candidate.direct_result, candidate.adapter_result
    same_inputs = (
        candidate.omega == sample.omega
        and candidate.tau == float(system.variance.tau)
        and np.array_equal(candidate.outcome, system.z)
        and np.array_equal(candidate.solver_design, system.design)
        and np.array_equal(candidate.unscaled_design, sample.data.design)
        and np.array_equal(candidate.design_scale, sample.design.scale)
        and np.all(candidate.unscaled_design[:, candidate.intercept] == 1.0)
    )
    if not same_inputs:
        raise _refuse("the IOLS solve used another outcome, design, order or scale")
    slope_columns = [
        col for col in range(system.dimension) if col != candidate.intercept
    ]
    slopes = candidate.unscaled_design[:, slope_columns]
    normalization = build_transformed_normalization_state(slopes, mode="scale")
    if (
        normalization.scale is None
        or adapter.normalization.scale is None
        or not np.array_equal(normalization.scale, adapter.normalization.scale)
        or not np.array_equal(direct._input_endog, candidate.outcome)
        or not np.array_equal(
            direct._input_slopes, apply_transformed_normalization(slopes, normalization)
        )
    ):
        raise _refuse("the IOLS backend inputs are not the retained scaled problem")
    expected_backend = effective_backend_request(candidate.iols_options)
    if (
        candidate.requested_backend != expected_backend
        or candidate.actual_backend != direct.backend
        or candidate.actual_backend != adapter.actual_backend
        or candidate.fallback_used != direct.fallback_used
        or candidate.fallback_used != adapter.fallback_used
        or candidate.availability_reason != direct.availability_reason
        or candidate.availability_reason != adapter.availability_reason
        or direct.requested_backend != expected_backend
        or (
            candidate.fallback_used
            and (expected_backend != "auto" or candidate.actual_backend != "numpy")
        )
    ):
        raise _refuse("the IOLS backend record differs from the resolved request")
    if (
        any(
            direct.control[name] != getattr(control, name)
            for name in DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS
        )
        or direct.backend_details.get("direct_restart_start")
        != control.direct_restart_start
        or direct.backend_details.get("pilot_deltas") != control._pilot_deltas
    ):
        raise _refuse("the IOLS kernel ran with different controls")
    if adapter.theta is None:
        if not np.all(np.isnan(candidate.coefficients)):
            raise _refuse("IOLS retained coefficients without an adapter point")
    else:
        mapped = np.asarray(adapter.theta * candidate.design_scale, dtype=np.float64)
        mapped[candidate.intercept] = adapter.theta[candidate.intercept]
        if not np.array_equal(mapped, candidate.coefficients, equal_nan=True):
            raise _refuse("the IOLS point is not the mapped adapter result")


def _same_number(found: float, given: float) -> bool:
    """Exact finite values, or two undefined values; no numerical tolerance."""
    return found == given or (math.isnan(found) and math.isnan(given))


def _same_summary(found: StageSummary, given: StageSummary) -> bool:
    """The actual certificate quantities/gates, not merely a claimed level."""
    return (
        found.level == given.level
        and found.strict == given.strict
        and found.secondary == given.secondary
        and found.positive_definite == given.positive_definite
        and found.conditioned == given.conditioned
        and found.score_metric == given.score_metric
        and dict(found.errors) == dict(given.errors)
        and all(
            _same_number(a, b)
            for a, b in (
                (found.scaled_score, given.scaled_score),
                (found.index_correction, given.index_correction),
                (found.rcond, given.rcond),
            )
        )
    )


def _fresh_root_certificate(
    handoff: Handoff,
    system: JointSystem,
    certificate: Certificate,
    merit: FloatArray | None,
    *,
    reported: bool = True,
    point: FloatArray | None = None,
) -> Certificate:
    """Recheck this equation/point with this fit's actual inference and units.

    The provider of the inference requested at the fit, formed afresh: the robust or
    one-way one, or the named HC one on the rows that every equation of the fit
    shares, with the request's preflight formed again (:func:`~powermean.results.
    point.joint_provider`).
    """
    provider = joint_provider(handoff.sample, handoff.inference, system)
    return certify(
        system,
        certificate.gamma if point is None else point,
        certificate.controls,
        reported=provider if reported else None,
        frequency=frequency_of(handoff.sample),
        merit_cholesky=merit,
    )


def _check_fresh_certificate(
    given: Certificate, fresh: Certificate, context: str
) -> None:
    """Bind evaluation, numerical gates and the full reported coefficient check."""
    evaluation = given.evaluation
    if (evaluation is None) != (fresh.evaluation is None) or (
        evaluation is not None
        and fresh.evaluation is not None
        and not _same_evaluation(fresh.evaluation, evaluation)
    ):
        raise _refuse(f"the certificate was formed on {context}")
    same = (
        given.require_positive_definite == fresh.require_positive_definite
        and _same_summary(summary_of(fresh), summary_of(given))
        and _same_array(fresh.correction, given.correction)
        and (fresh.reported is None) == (given.reported is None)
        and (fresh.coefficients is None) == (given.coefficients is None)
    )
    if fresh.reported is not None and given.reported is not None:
        same = same and _same_array(
            fresh.reported.standard_errors, given.reported.standard_errors
        )
        same = same and _same_array(fresh.reported.gamma, given.reported.gamma)
        assert fresh.reported.order is not None
        same = same and _same_basis(given.reported, fresh.reported.order)
    if fresh.coefficients is not None and given.coefficients is not None:
        a, b = fresh.coefficients, given.coefficients
        same = (
            same
            and a.coef_tol == b.coef_tol
            and all(
                _same_array(x, y)
                for x, y in (
                    (a.corrections, b.corrections),
                    (a.standard_errors, b.standard_errors),
                    (a.passed, b.passed),
                    (a.ratios, b.ratios),
                )
            )
        )
    if not same:
        raise _refuse("the root certificate quantities/gates are not fresh")


def _check_convex_certificate(
    handoff: Handoff, convex: JointSystem, certificate: Certificate
) -> None:
    """The initializer's certificate is the convex equation's of THIS sample.

    The fresh evaluation of the same rows, design, offset and weights under the convex
    variance reproduces the certificate's evaluation exactly, with the fit's
    controls, frequency flag and coordinate basis, and its curvature gate required.
    """
    _check_controls(handoff, certificate)
    if not certificate.require_positive_definite:
        raise _refuse("the initializer's certificate waives the convex curvature gate")
    fresh = _fresh_root_certificate(handoff, convex, certificate, None)
    _check_fresh_certificate(certificate, fresh, "another equation or sample")


def _check_convex_initializer(
    handoff: Handoff, convex: JointSystem, candidate: JointRootCandidate
) -> None:
    """The nearest convex initializer: its candidate, certificate, status and point."""
    initializer = candidate.initializer
    newton, certificate = initializer.newton, initializer.certificate
    if newton is None or certificate is None:
        raise _refuse("the nearest convex initializer keeps its candidate")
    reason = newton.attempts[newton.retained].reason
    bound = (
        _same_array(newton.coefficients, certificate.gamma)
        and _same_array(certificate.gamma, initializer.point)
        and newton.options == candidate.options
        and newton.start.source == "automatic"
        and candidate.start.source == "automatic"
        and _same_array(candidate.start.coefficients, newton.start.coefficients)
        and initializer.reason == reason
        and initializer.status == compose_status(certificate, reason)
        and initializer.certified == certificate.accepted
    )
    if not bound:
        raise _refuse("the initializer's record is not that of its convex fit")
    names = handoff.sample.data.column_names
    intercept = names.index("Intercept") if "Intercept" in names else None
    if intercept is not None and not np.all(convex.design[:, intercept] == 1.0):
        intercept = None
    opening = automatic_start(convex, intercept=intercept)
    if (
        not _same_array(opening.coefficients, newton.start.coefficients)
        or opening.intercept != newton.start.intercept
        or opening.mean != newton.start.mean
    ):
        raise _refuse("the initializer did not use this sample's automatic start")
    _check_convex_certificate(handoff, convex, certificate)
    if candidate.stages and not certificate.accepted:
        raise _refuse("an uncertified automatic initializer cannot open root stages")


def _check_user_initializer(candidate: JointRootCandidate) -> None:
    """The caller's start: accepted as given, never certified, the solve's start."""
    initializer = candidate.initializer
    if (
        initializer.kind != "user_start"
        or initializer.variance is not None
        or initializer.certified
        or initializer.newton is not None
        or initializer.certificate is not None
        or initializer.status is not None
        or initializer.reason is not None
        or candidate.start.source != "user"
        or not _same_array(initializer.point, candidate.start.coefficients)
    ):
        raise _refuse("the initializer is not the caller's start")


def _check_merit(system: JointSystem, candidate: JointRootCandidate) -> None:
    """The merit factor is the requested metric's at the initializer's point."""
    factor = candidate.merit_cholesky
    if factor is None or not candidate.stages:
        if (factor is None) != (not candidate.stages):
            raise _refuse("a root stage opens with its merit factor, and only then")
        initializer = candidate.initializer
        if initializer.kind != "user_start" and not initializer.certified:
            if candidate.reason != initializer.reason:
                raise _refuse("the failed initializer's native reason was not retained")
            return
        try:
            merit_factor(system, initializer.point, candidate.merit_metric, WorkLog())
        except np.linalg.LinAlgError:
            reason = "factorization_failure"
        except (FloatingPointError, ValueError):
            reason = "numerical_breakdown"
        else:
            raise _refuse("a formable merit metric must open an actual root stage")
        if candidate.reason != reason:
            raise _refuse("the absent merit factor does not have its actual reason")
        return
    try:
        fresh = merit_factor(
            system, candidate.initializer.point, candidate.merit_metric, WorkLog()
        )
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        raise _refuse("the merit metric does not factorize at the start") from None
    if not _same_array(fresh, factor):
        raise _refuse("the merit factor is not the metric's at the initializer's point")


def _check_initializer(handoff: Handoff, candidate: JointRootCandidate) -> None:
    """The caller's start where one was requested, else the nearest convex fit."""
    system = system_of(handoff.estimator, handoff.sample)
    initializer = candidate.initializer
    if handoff.requested_start is not None:
        _check_user_initializer(candidate)
    else:
        kind, variance = nearest_convex(handoff.estimator.variance)
        if initializer.kind != kind or initializer.variance != variance:
            raise _refuse("the initializer is not the nearest convex fit of the rule")
        _check_convex_initializer(handoff, same_system(system, variance), candidate)
    stages = candidate.stages
    if stages and not _same_array(stages[0].attempts[0].opening, initializer.point):
        raise _refuse("the root solve did not open at its initializer's point")
    _check_merit(system, candidate)


def _check_root(
    handoff: Handoff, conclusion: Conclusion, candidate: JointRootCandidate
) -> None:
    """A joint root conclusion is its route's (module docstring)."""
    certificate = conclusion.certificate
    if handoff.computation.solver != "newton" or not outside_convex(
        handoff.estimator.variance
    ):
        raise _refuse("a root conclusion belongs to the joint route outside [1, 2]")
    options, root = _root_computation(handoff.computation)
    if (
        candidate.options != options
        or candidate.root_options != root
        or candidate.merit_metric != root.merit_metric
    ):
        raise _refuse("the solver ran with other settings than the resolved ones")
    if conclusion.reason != candidate.reason:
        raise _refuse("the native reason is not that of the retained attempt")
    _check_root_stages(handoff, candidate, certificate)
    if conclusion.status != joint_root_status(candidate, certificate):
        raise _refuse("the status is not the one composed by the accepted rules")
    if not candidate.stages and (
        certificate.reported is not None or certificate.accepted
    ):
        raise _refuse("a point without a root stage carries no selected root")
    if candidate.start.coefficients.shape != certificate.gamma.shape:
        raise _refuse("the start of the solve has another number of coefficients")
    if not _same_array(candidate.coefficients, certificate.gamma):
        raise _refuse(
            "the selected root candidate has another coordinate shape or basis"
        )
    _check_initializer(handoff, candidate)
    _check_continuation(handoff, candidate, certificate)


def _check_continuation(
    handoff: Handoff, candidate: JointRootCandidate, certificate: Certificate
) -> None:
    """The recorded continuation follows the actual certified-point schedule."""
    stages = candidate.stages
    if len(stages) <= 1:
        return
    if (
        not candidate.initializer.certified
        or candidate.initializer.kind == "user_start"
        or candidate.root_options.continuation != "fallback"
    ):
        raise _refuse("continuation requires its certified automatic initializer")
    variance = handoff.estimator.variance
    assert isinstance(variance, PowerVariance)
    _, origin = nearest_convex(variance)
    expected = tuple(
        iter_continuation_variances(
            variance,
            candidate.root_options.continuation_step,
            len(stages) - 1,
            origin=origin.tau,
        )
    )
    if tuple(stage.variance for stage in stages[1:]) != expected:
        raise _refuse("the continuation variances are not the initializer's schedule")
    previous = candidate.initializer.point
    system = system_of(handoff.estimator, handoff.sample)
    for index, stage in enumerate(stages[1:], 1):
        if not _same_array(stage.attempts[0].opening, previous):
            raise _refuse("continuation did not open at the last certified point")
        retained = stage.attempts[stage.retained]
        fresh = _fresh_root_certificate(
            handoff,
            same_system(system, stage.variance),
            certificate,
            candidate.merit_cholesky,
            point=retained.point,
        )
        if not _same_summary(summary_of(fresh), stage.summary) or (
            compose_status(fresh, retained.native) != stage.status
        ):
            raise _refuse("a continuation stage's certificate/status is not fresh")
        if not fresh.accepted and index + 1 < len(stages):
            raise _refuse("continuation proceeds only from a certified point")
        previous = retained.point


def _check_root_stages(
    handoff: Handoff, candidate: JointRootCandidate, certificate: Certificate
) -> None:
    """Bind the actual target stage, its point/reason, and truthful stage work."""
    stages, target = candidate.stages, candidate.target_stage
    if not stages:
        if target is not None or candidate.continuation_used:
            raise _refuse("a root without stages has no target stage or continuation")
        return
    if type(target) is not int or not 0 <= target < len(stages):
        raise _refuse("the root target stage is not an actual stage")
    variance = handoff.estimator.variance
    if stages[0].variance != variance or stages[target].variance != variance:
        raise _refuse("the root target stage is not the requested equation")
    if target != max(i for i, stage in enumerate(stages) if stage.variance == variance):
        raise _refuse("the root target stage is not the last actual target stage")
    if candidate.continuation_used != (len(stages) > 1):
        raise _refuse("the root continuation flag is not its actual stage sequence")
    for stage in stages:
        _check_stage_work(stage)
    retained = stages[target].attempts[stages[target].retained]
    if not _same_array(retained.point, certificate.gamma):
        raise _refuse("the root certificate is not the target stage's retained point")
    if candidate.reason != retained.native:
        raise _refuse("the root reason is not the target stage's retained reason")
    if not _same_summary(stages[target].summary, summary_of(certificate)):
        raise _refuse("the target stage summary is not the conclusion's certificate")


def _check_stage_work(stage: JointRootStage) -> None:
    """A stage retains one actual attempt with its actual nonnegative work."""
    attempts = stage.attempts
    if not 1 <= len(attempts) <= 2 or type(stage.retained) is not int:
        raise _refuse("a root stage retains one of its actual attempts")
    if not 0 <= stage.retained < len(attempts):
        raise _refuse("a root stage retains one of its actual attempts")
    if (
        stage.iterations != sum(a.iterations for a in attempts)
        or stage.evaluations != sum(a.evaluations for a in attempts)
        or any(min(a.iterations, a.evaluations, a.backtracks) < 0 for a in attempts)
    ):
        raise _refuse("the root stage work is not that of its attempts")
    if len(attempts) == 2 and not _same_array(attempts[1].opening, attempts[0].point):
        raise _refuse("the root retry did not open at the first attempt's point")


def _check_conclusion(handoff: Handoff, conclusion: Conclusion) -> None:
    """The conclusion is its certificate's and its candidate's, with the settings."""
    candidate, certificate = conclusion.candidate, conclusion.certificate
    if candidate.coefficients.tobytes() != certificate.gamma.tobytes():
        raise _refuse("the certificate is not that of the retained candidate")
    if isinstance(candidate, JointRootCandidate):
        _check_root(handoff, conclusion, candidate)
        return
    reason = (
        candidate.reason
        if isinstance(candidate, IolsCandidate)
        else candidate.attempts[candidate.retained].reason
    )
    if conclusion.reason != reason:
        raise _refuse("the native reason is not that of the retained attempt")
    if conclusion.status != compose_status(certificate, conclusion.reason):
        raise _refuse("the status is not the one composed by the accepted rules")
    if handoff.computation.solver == "iols":
        iols_opts, cert_opts = _resolve_iols_computation(handoff.computation)
        if (
            not isinstance(candidate, IolsCandidate)
            or candidate.iols_options != iols_opts
            or candidate.options != cert_opts
            or candidate.iols_options.backend_explicit != iols_opts.backend_explicit
        ):
            raise _refuse("the solver ran with other settings than the resolved ones")
    elif not isinstance(
        candidate, NewtonCandidate
    ) or candidate.options != resolve_options(handoff.computation.options):
        raise _refuse("the solver ran with other settings than the resolved ones")
    if isinstance(candidate, IolsCandidate):
        _check_iols_candidate(handoff, candidate)
    elif candidate.start.coefficients.shape != certificate.gamma.shape:
        raise _refuse("the start of the solve has another number of coefficients")


def _requested_start(handoff: Handoff, sample: OrderSample) -> FloatArray | None:
    """The start on the reported scale, owned and bound to the solver's start.

    Where a conclusion was handed over, the solver started from the user's start
    mapped by the accepted forward map of the order scale, bit for bit.  On the
    linear route, whose solve reads no start, the map must exist on this basis
    (``theta_j = d_j beta_j``): the record of the provenance reports it.
    """
    given = handoff.requested_start
    if given is None:
        return None
    if np.iscomplexobj(given):
        raise _refuse("the requested start is not real-valued")
    start = np.asarray(given, dtype=np.float64)
    if start.shape != (sample.data.nparams,) or not np.isfinite(start).all():
        raise _refuse("the requested start is not a finite vector of the coefficients")
    try:
        mapped = sample.order_scale.solver_coefficients(start)
    except FloatingPointError:
        raise _refuse(
            "the requested start is not representable in the coordinates of the solver"
        ) from None
    conclusion = handoff.conclusion
    if conclusion is not None:
        if isinstance(conclusion.candidate, IolsCandidate):
            raise _refuse("IOLS does not accept a caller-supplied start")
        solver = conclusion.candidate.start
        if solver.source != "user" or mapped.tobytes() != solver.coefficients.tobytes():
            raise _refuse("the requested start is not the start of the solve")
    return owned_array(start)


def _check_failure(sample: OrderSample, failure: StartFailure) -> None:
    """The start failure is this sample's: its mean, its logarithm, its intercept."""
    weights = sample.data.weight_values
    active = weights > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean = float(np.mean(sample.problem.response[active]))
        log_mean = float(np.log(mean))
    if mean != failure.mean or math.isfinite(log_mean):
        raise _refuse("the start failure is not that of the retained problem")
    column = failure.intercept
    if not 0 <= column < sample.data.nparams or not np.all(
        sample.design.matrix[:, column] == 1.0
    ):
        raise _refuse("the intercept of the start failure is not a column of ones")


def _check_offset_design(handoff: Handoff, sample: OrderSample) -> Design | None:
    """The expression design is the one the preparation read, owned whole.

    Its one column is the offset or exposure column that the preparation read, and
    its evaluated values at the retained positions are the sample's raw offset or
    exposure, exactly.  The design handed over is the object the preparation
    evaluated, which no prepared frame copied: it is owned whole, its mutable fitted
    spec, matrix, axes and labels included (:func:`~powermean.results.owned.
    owned_expression_design`), so that an edit of the caller's object reaches no
    replay.
    """
    design = handoff.offset_design
    if design is None:
        return None
    columns = handoff.prepared.columns
    name = columns.offset if columns.offset is not None else columns.exposure
    if name is None or design.columns != (name,):
        raise _refuse("the offset design is not the expression the preparation read")
    values = design.matrix.to_numpy(dtype=np.float64, na_value=np.nan)[:, 0]
    if values.shape != (sample.nobs,):
        raise _refuse("the offset design has another number of input rows")
    retained = values[sample.positions]
    data = sample.data
    expected = data.offset if columns.offset is not None else data.exposure
    if expected is None or not np.array_equal(retained, expected):
        raise _refuse("the offset design does not evaluate to the sample's offset")
    return owned_expression_design(design)


def supplied_status(certificate: Certificate) -> Status | None:
    """The status of a supplied point from its gates alone, or ``None``.

    The rules of :func:`powermean.solve.status.compose_status` that need no native
    reason: the two verified levels, then the gates that describe the candidate.  A
    finite and regular candidate that fails the score, the index or the coefficient
    gate takes its status from the native reason of a solve, which a supplied point
    has not.
    """
    if certificate.level == "strict":
        return "verified"
    if certificate.level == "secondary":
        return "verified_secondary"
    if "evaluation" in certificate.strict:
        return "nonfinite"
    if "curvature" in certificate.strict or "conditioning" in certificate.strict:
        return "singular_system"
    return None


def _check_linear(sample: OrderSample, linear: LinearCandidate) -> None:
    """The candidate solved the retained problem, and its residuals are its own."""
    if linear.dimension != sample.data.nparams:
        raise _refuse("the linear candidate has another number of coefficients")
    if linear.units.nobs != sample.count:
        raise _refuse("the linear candidate solved another number of rows")
    weights = sample.data.weight_values
    active = np.flatnonzero(weights > 0)
    if not np.array_equal(linear.units.identifiers, active) or not np.array_equal(
        linear.weights, weights[active]
    ):
        raise _refuse(
            "the units or weights of the linear candidate are not the sample's"
        )
    if linear.coefficients is None:
        if linear.residuals is not None:
            raise _refuse("residuals without coefficients on the linear candidate")
        return
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        target: FloatArray = (
            sample.problem.response[active] - sample.problem.offset[active]
        )
        residuals: FloatArray = (
            target - sample.design.matrix[active] @ linear.coefficients
        )
    if linear.residuals is None or not np.array_equal(residuals, linear.residuals):
        raise _refuse(
            "the residuals of the linear candidate are not those of the point"
        )


def _check_linear_certificate(
    handoff: Handoff,
    sample: OrderSample,
    linear: LinearCandidate,
    certificate: LinearCertificate,
) -> None:
    """The linear certificate is this equation's, at the candidate's point and basis."""
    if (
        linear.coefficients is None
        or certificate.gamma.tobytes() != linear.coefficients.tobytes()
    ):
        raise _refuse("the linear certificate is not that of the candidate's point")
    _check_controls(handoff, certificate)
    problem = linear_problem_of(sample)
    given = certificate.evaluation
    try:
        fresh = problem.evaluate(certificate.gamma)
    except FloatingPointError:
        if given is not None:
            raise _refuse(
                "the retained linear equation is not representable at the point"
            ) from None
        return
    if given is None or not _same_linear_evaluation(fresh, given):
        raise _refuse(
            "the linear certificate was formed on another equation, sample or point"
        )
    reported = certificate.reported
    if reported is not None and not _same_basis(reported, sample.order_scale):
        raise _refuse("the reported scale of the linear certificate is not this basis")


def _check_linear_conclusion(
    handoff: Handoff,
    sample: OrderSample,
    linear: LinearCandidate,
    conclusion: LinearConclusion,
) -> LinearCertificate | None:
    """The linear conclusion is its candidate's and its certificate's, owned."""
    candidate = conclusion.candidate
    same = (
        candidate.reason == linear.reason
        and candidate.dimension == linear.dimension
        and np.array_equal(candidate.units.identifiers, linear.units.identifiers)
        and candidate.units.nobs == linear.units.nobs
        and np.array_equal(candidate.weights, linear.weights)
        and (candidate.coefficients is None) == (linear.coefficients is None)
        and (
            candidate.coefficients is None
            or linear.coefficients is None
            or candidate.coefficients.tobytes() == linear.coefficients.tobytes()
        )
    )
    if not same:
        raise _refuse("the linear conclusion is not that of the linear candidate")
    certificate = conclusion.certificate
    if (certificate is None) != (linear.coefficients is None):
        raise _refuse("a linear point carries its certificate, and no point none")
    owned: LinearCertificate | None = None
    if certificate is not None:
        owned = owned_linear_certificate(certificate)
        _check_linear_certificate(handoff, sample, linear, owned)
    if conclusion.status != linear_status(linear.reason, owned):
        raise _refuse("the status is not the one composed by the accepted rules")
    return owned


@dataclass(frozen=True, eq=False)
class _Power:
    """The owned power-route records of a handoff, with the status of the point."""

    certificate: Certificate | None
    candidate: AnyCandidate | None
    reason: NativeReason | None
    status: Status
    failure: StartFailure | None


def _power_records(handoff: Handoff, sample: OrderSample) -> _Power | None:
    """The records of the power route, owned, with their status; ``None`` without."""
    if handoff.conclusion is not None:
        conclusion = handoff.conclusion
        _check_conclusion(handoff, conclusion)
        return _Power(
            owned_certificate(conclusion.certificate),
            owned_candidate(conclusion.candidate),
            conclusion.reason,
            conclusion.status,
            None,
        )
    if handoff.failure is not None:
        _check_failure(sample, handoff.failure)
        return _Power(None, None, None, handoff.failure.status, handoff.failure)
    if handoff.certificate is None:
        return None
    certificate = owned_certificate(handoff.certificate)
    status = supplied_status(certificate)
    if status is None:
        raise _refuse(
            "a supplied point that fails the score, the index or the coefficient "
            "gate has no status without the native reason of its solve"
        )
    return _Power(certificate, None, None, status, None)


@dataclass(frozen=True, eq=False)
class _Bound:
    """The owned route records and the point of a handoff, after the checks."""

    certificate: AnyCertificate | None
    candidate: AnyCandidate | None
    linear: LinearCandidate | None
    reason: NativeReason | None
    status: Status | None
    failure: StartFailure | None
    point: Point


def _bind_power(handoff: Handoff, sample: OrderSample, power: _Power) -> _Bound:
    """The point of the power route, with its records checked."""
    certificate = power.certificate
    if certificate is None:
        point = Point(None, None, None, None)
    else:
        _check_certificate(handoff, sample, certificate)
        system = system_of(handoff.estimator, sample)
        point = power_point(sample, handoff.inference, certificate, system)
        _check_reported_scale(certificate, point)
    return _Bound(
        certificate,
        power.candidate,
        None,
        power.reason,
        power.status,
        power.failure,
        point,
    )


def _bind_linear(handoff: Handoff, sample: OrderSample) -> _Bound:
    """The point of the linear route, bare or with its fresh conclusion."""
    assert handoff.linear is not None  # the route check above requires it
    linear = owned_linear(handoff.linear)
    _check_linear(sample, linear)
    if handoff.linear_solve is not None:
        _check_linear_solve(sample, linear, handoff.linear_solve)
    conclusion = handoff.linear_conclusion
    if conclusion is None:
        point = linear_point(sample, handoff.inference, linear)
        return _Bound(None, None, linear, None, None, None, point)
    certificate = _check_linear_conclusion(handoff, sample, linear, conclusion)
    if certificate is None:
        point = Point(None, None, None, None)
    else:
        point = linear_certified_point(
            sample, handoff.inference, certificate, handoff.linear_solve
        )
        _check_reported_scale(certificate, point)
    return _Bound(certificate, None, linear, None, conclusion.status, None, point)


def bind(handoff: Handoff) -> RetainedFit:
    """The retained state of a result, after the checks of the module docstring.

    Parameters
    ----------
    handoff : Handoff
        The records of the fit.

    Returns
    -------
    RetainedFit
        The owned state bound to the point, with the influence record and the default
        covariance formed once.

    Raises
    ------
    TypeError
        A record of another type; a classification whose values cannot be compared.
    ValueError
        Records that are not bound to one another; a supplied point whose certificate
        fails at a finite and regular candidate without the native reason of a solve.
    SpecificationError
        A requested cluster that was not retained, or that has a missing value on
        the estimation sample.
    """
    _check_types(handoff)
    prepared = handoff.prepared
    sample = owned_sample(handoff.sample)
    frame = prepared.frame
    check_sample(handoff.estimator, prepared, sample, frame)
    power = _power_records(handoff, sample)
    if power is not None:
        bound = _bind_power(handoff, sample, power)
    else:
        bound = _bind_linear(handoff, sample)
    point = bound.point
    return RetainedFit(
        estimator=handoff.estimator,
        inference=handoff.inference,
        computation=handoff.computation,
        sample=sample,
        labels=prepared.labels,
        columns=prepared.columns,
        subset_missing=prepared.subset_missing,
        parts=frame.parts,
        design=owned_design(frame.design),
        certificate=bound.certificate,
        candidate=bound.candidate,
        linear=bound.linear,
        reason=bound.reason,
        status=bound.status,
        theta=point.theta,
        influence=point.influence if handoff.keep_influence else None,
        default=point.default,
        keep_influence=handoff.keep_influence,
        failure=bound.failure,
        requested_start=_requested_start(handoff, sample),
        offset_design=_check_offset_design(handoff, sample),
        linear_solve=handoff.linear_solve if handoff.keep_influence else None,
    )
