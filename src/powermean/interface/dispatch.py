"""The dispatch of one fit: the accepted stages composed, the point certified afresh.

``DESIGN.md``, Sections 4.1, 4.4, 4.6, 5.1 and 5.3; ``design/02_computation.md``,
Sections 1 (rule 3: a solver proposes, the certificate decides), 5.1, 5.5, 6, 7 and 9;
``design/01_statistical_core.md``, Sections 3.1 to 3.3; decisions D41, D45, D49, D51
and D58; task M0.22.  :func:`fit_prepared` takes a prepared problem and the typed
specification of one fit, derives the sample of the mean order on the frozen domain,
and composes the accepted stages in their order, with no fallback of solver, start,
tolerance or sample and no second acceptance policy:

- without fixed effects, a regressor that is an exact linear combination of the
  regressors before it in the formula is omitted, in formula order, by the
  collinearity step of the routes with fixed effects called without blocks
  (:func:`powermean.data.collinearity.analyze_collinearity`), before anything is
  solved (owner's rulings 109 and 110).  One ``coverage.collinear_omitted`` note
  names each omitted regressor, as on those routes; the sample of the fit holds the
  other columns (:func:`powermean.data.retained.without_columns`) and the fit is that
  of the formula without the omitted regressors.  A design that the step leaves
  whole reaches the stages below as before, a design of deficient numerical rank
  included;
- nonzero order: the transformed problem of the sample (``Z = Y^omega``, the offset
  ``omega o``, formed once by the preparation) becomes the accepted
  :class:`~powermean.systems.joint.JointSystem`; the start is the accepted automatic
  one (the plain mean of ``Z`` in the intercept, zero slopes) or the user's vector on
  the reported scale mapped to the solver's coordinates by the accepted forward map
  of the order scale; the convex Newton solver runs with the resolved options and
  the :class:`~powermean.solve.certificate.CertificateAssessor` of its retry, whose
  provider reports corrections and standard errors by the direct map of the order
  scale; and :func:`~powermean.solve.status.conclude` certifies the retained
  candidate afresh.  Native success of the solver certifies nothing;
- zero order: the log response, the raw offset and the accepted SVD solve of
  :func:`powermean.solve.linear.solve`; the point is certified afresh against the
  original linear equation (:func:`~powermean.solve.linear_status.conclude_linear`),
  with the standard errors of the fresh evaluation supplied by the linear callback
  of the inference stage; no working variance enters the equation, which the
  estimator keeps as provenance only.  A start given by the caller is accepted and
  recorded as at any order (``design/02``, Section 6): the same shape check, the
  same forward map (``theta_j = d_j beta_j`` at the order zero), owned in both
  coordinates by the result; the least-squares solve computes its solution
  independently of it, and no iterative trace is invented for it;
- nonzero order with a power variance outside ``[1, 2]``: the same transformed
  problem and :class:`~powermean.systems.joint.JointSystem`, solved and concluded by
  :func:`powermean.solve.joint_root.solve_joint_root`: root Newton from the caller's
  mapped start, or from the certified nearest convex fit of the same system (PPML
  below 1, Gamma above 2) run by the unchanged convex solver and certificate; the
  resolved root-Newton options are recorded with the other settings.  Whether the
  equation has a finite root is not characterized for ``tau < 1``, or above 2 with
  zero outcomes (``design/02``, Section 4): no row is dropped for it, the solver and
  the certificate decide, and a ``CoverageWarning`` says so once, here;
- the records are handed to :func:`powermean.results.handoff.bind`, which checks the
  binding of every record to the retained inputs and forms the result.

The covariance requested at the fit supplies the standard errors of the certificate's
coefficient gate on every route (``plan/evidence/M1_contract.txt``, rule B12(a)): the
robust or one-way provider of the first milestone, for the exact string of a named
HC method the provider of :mod:`powermean.inference.hc_provider`, or, for a typed
cluster request (several dimensions, a multiway form, a ``"psd"`` key or an explicit
``ssc``) and for the robust covariance with an explicit ``ssc``, the provider of
:mod:`powermean.results.cluster_covariance`
(:func:`~powermean.results.point.joint_provider`,
:func:`~powermean.results.point.linear_provider`).  The refusals of a named method
that need no point (HC1 at ``N = K``, an explicit second correction), and those of a
request of that engine (a dimension that was not retained or has a missing value,
fewer than two clusters among the active rows, a convention
without positive degrees of freedom), run before anything is solved
(:func:`~powermean.results.point.fit_preflight`), so that a refusal costs no solve;
no other covariance replaces a refused request.

Failures.  A conclusion whose status is ``verified_secondary`` is reported with the
existing ``NumericalWarning`` once, here, at the dispatch; a retrieval of the result
emits nothing again.  Any other status raises ``ConvergenceError`` with the last
state and the certificate (``on_failure="raise"``, the default) or returns the
result flagged as not certified, with its inference withheld
(``on_failure="return"``).  Where the accepted automatic start raises its documented
``FloatingPointError`` before any candidate exists, the same boundary applies with
the :class:`~powermean.solve.no_point.StartFailure` as the last state and no
certificate: no point, trace or existence claim is fabricated, and the design with
no intercept, whose zero start is finite, proceeds. The IOLS scale-floor refusal
likewise retains its measured scales before any backend evaluation.  Where the
least-squares solve
forms no point, its own reason decides the status and no certificate exists.
Malformed inputs, an unavailable provider and the refusals of the preparation are
not convergence failures and are never swallowed.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

from powermean.data.collinearity import COMPLETE, analyze_collinearity
from powermean.data.collinearity_numeric import numerical_rank
from powermean.data.collinearity_records import CollinearityProblem
from powermean.data.computation import Computation, _snapshot
from powermean.data.retained import OrderSample, without_columns
from powermean.errors import (
    CoverageWarning,
    ExistenceError,
    IdentificationError,
    NumericalWarning,
    SampleError,
    SpecificationError,
)
from powermean.inference.hc import HCPreflight
from powermean.interface.options import Specification
from powermean.interface.preparation import PreparedProblem
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.handoff import (
    Handoff,
    bind,
    linear_problem_of,
    linear_solve_of,
    system_of,
)
from powermean.results.indefinite import NOTE as INDEFINITE
from powermean.results.ordinary_support import recession, separation
from powermean.results.point import (
    fit_preflight,
    frequency_of,
    joint_provider,
    linear_provider,
)
from powermean.results.result import Result, not_certified
from powermean.solve.cancelled_records import RootOptions
from powermean.solve.certificate import CertificateAssessor, outside_convex
from powermean.solve.iols.candidate import refuse_iols, solve_iols
from powermean.solve.iols.normalization import IolsScaleFailure, IolsScaleFloorError
from powermean.solve.iols.options import IolsOptions
from powermean.solve.joint_root import solve_joint_root
from powermean.solve.linear_status import conclude_linear
from powermean.solve.newton_convex import solve as newton_solve
from powermean.solve.no_point import StartFailure
from powermean.solve.reported import ReportedScaleProvider
from powermean.solve.starts import Start, automatic_start, user_start
from powermean.solve.status import Conclusion, conclude
from powermean.systems.joint import JointSystem

if TYPE_CHECKING:
    from powermean.interface.fe_dispatch import fit_fe_prepared

CERTIFIED: tuple[str, ...] = ("verified", "verified_secondary")
INTERCEPT = "Intercept"
# Around sqrt(machine epsilon), forming y**omega can lose half the significant digits.
_SMALL_MEAN_ORDER = 1e-8


def intercept_column(sample: OrderSample) -> int | None:
    """The index of the intercept of the design: a column of ones named by Formulaic.

    ``None`` where the design has no such column, and the automatic start is then
    the zero vector; a column of another name is never presumed to be the intercept.
    """
    names = sample.data.column_names
    if INTERCEPT not in names:
        return None
    column = names.index(INTERCEPT)
    if not np.all(sample.design.matrix[:, column] == 1.0):
        return None
    return column


def mapped_start(sample: OrderSample, requested: FloatArray) -> FloatArray:
    """The caller's start on the reported scale, in the coordinates of the solver.

    One value per coefficient of the design, mapped by the accepted range-safe
    forward map of the order scale (``theta_j = d_j omega beta_j``, ``d_j beta_j``
    at the order zero); the same checks at every order.

    Raises
    ------
    SpecificationError
        ``specification.invalid_value`` for a start of another length, or one that
        is not representable in the coordinates of the solver.
    """
    if requested.shape != (sample.data.nparams,):
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "start",
                "value": requested.tolist(),
                "valid": f"one value per coefficient ({sample.data.nparams})",
            },
        )
    try:
        return sample.order_scale.solver_coefficients(requested)
    except FloatingPointError as error:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "start",
                "value": requested.tolist(),
                "valid": "a start representable in the coordinates of the solver",
            },
        ) from error


def _start(
    system: JointSystem, sample: OrderSample, specification: Specification
) -> Start | StartFailure:
    """The start of the solve: the user's, mapped, or the accepted automatic one."""
    requested = specification.start
    if requested is not None:
        return user_start(system, mapped_start(sample, requested))
    intercept = intercept_column(sample)
    try:
        return automatic_start(system, intercept=intercept)
    except FloatingPointError as error:
        assert intercept is not None  # the zero start of no intercept is finite
        with np.errstate(over="ignore"):
            mean = float(np.mean(system.active_z))
        return StartFailure(message=str(error), mean=mean, intercept=intercept)


def root_options_of(specification: Specification) -> RootOptions:
    """The resolved root-Newton options of a fit outside ``[1, 2]``.

    The record that :func:`~powermean.interface.options.specification_of` resolved,
    or the defaults of ``design/02``, Section 5.7, for a specification formed without
    one; the same record is recorded with the settings and run by the solve.
    """
    found = specification.root_options
    return RootOptions() if found is None else found


def _root_conclusion(
    system: JointSystem,
    sample: OrderSample,
    specification: Specification,
    provider: ReportedScaleProvider | None,
    frequency: bool,
) -> Conclusion | StartFailure:
    """The joint root solve outside ``[1, 2]``, from the caller's start or the rule."""
    requested = specification.start
    start: Start | None = None
    if requested is not None:
        start = user_start(system, mapped_start(sample, requested))
    return solve_joint_root(
        system,
        specification.options,
        root_options_of(specification),
        start=start,
        intercept=intercept_column(sample),
        reported=provider,
        frequency=frequency,
    )


def _power_handoff(
    problem: PreparedProblem,
    sample: OrderSample,
    specification: Specification,
    computation: Computation,
    *,
    preflight: HCPreflight | None = None,
) -> Handoff:
    """The handoff of a fit of nonzero order: the solve and its fresh conclusion.

    The provider of the requested inference is formed before the solve; a named HC
    request takes the preflight that the fit formed, or forms it here.
    """
    system = system_of(specification.estimator, sample)
    frequency = frequency_of(sample)
    provider = joint_provider(
        sample, specification.inference, system, preflight=preflight
    )
    conclusion: Conclusion | None = None
    failure: StartFailure | IolsScaleFailure | None = None
    if specification.solver == "iols":
        try:
            iols_candidate = solve_iols(
                system,
                intercept=intercept_column(sample),
                iols_options=specification.iols_options,
                cert_options=specification.options,
                unscaled_design=sample.data.design,
                design_scale=sample.design.scale,
                omega=specification.estimator.mean.omega,
            )
        except IolsScaleFloorError as error:
            failure = error.failure.with_slope_name(sample.data.column_names)
        else:
            conclusion = conclude(
                system, iols_candidate, reported=provider, frequency=frequency
            )
    elif outside_convex(system.variance):
        found = _root_conclusion(system, sample, specification, provider, frequency)
        if isinstance(found, StartFailure):
            failure = found
        else:
            conclusion = found
    else:
        start = _start(system, sample, specification)
        if isinstance(start, StartFailure):
            failure = start
        else:
            candidate = newton_solve(
                system,
                start,
                specification.options,
                assess=CertificateAssessor(reported=provider, frequency=frequency),
            )
            conclusion = conclude(
                system, candidate, reported=provider, frequency=frequency
            )
    return Handoff(
        specification.estimator,
        problem.frame,
        sample,
        specification.inference,
        computation,
        conclusion=conclusion,
        keep_influence=specification.keep_influence,
        failure=failure,
        requested_start=specification.start,
        offset_design=problem.offset_design,
    )


def _linear_handoff(
    problem: PreparedProblem,
    sample: OrderSample,
    specification: Specification,
    computation: Computation,
) -> Handoff:
    """The handoff of a log OLS fit: the SVD solve and its fresh linear conclusion.

    A requested start is validated and mapped as on the power route, before the
    solve, and handed over for the record; the solve does not read it.  The equation
    is captured before its one solve (:func:`~powermean.results.handoff.
    linear_solve_of`, task M1.22) and the pair is handed over with its candidate.
    """
    if specification.start is not None:
        mapped_start(sample, specification.start)
    linear = linear_problem_of(sample)
    solved = linear_solve_of(sample)
    candidate = solved.candidate
    frequency = frequency_of(sample)
    provider = linear_provider(sample, specification.inference, solved)
    conclusion = conclude_linear(
        linear,
        candidate,
        specification.options,
        reported=provider,
        frequency=frequency,
    )
    return Handoff(
        specification.estimator,
        problem.frame,
        sample,
        specification.inference,
        computation,
        linear=candidate,
        keep_influence=specification.keep_influence,
        linear_conclusion=conclusion,
        requested_start=specification.start,
        offset_design=problem.offset_design,
        linear_solve=solved,
    )


def not_identified_fields(result: Result) -> dict[str, int]:
    """The counts of ``identification.not_identified`` from the result's sample."""
    counts = result.sample.counts
    by_effect, singleton = (
        counts.get("separated_by_fixed_effect", 0),
        counts.get("singleton", 0),
    )
    separated = counts.get("separated", 0) + by_effect
    return {
        "count": separated + singleton,
        "separated": separated,
        "separated_by_fixed_effect": by_effect,
        "singleton": singleton,
    }


def _report(result: Result, specification: Specification, stacklevel: int) -> Result:
    """The result, its notes once, or the refusal of a failed fit.

    The notes of a certified fit: the secondary warning, then the note of a root of
    indefinite curvature (owner's ruling 78, :mod:`powermean.results.indefinite`).
    """
    status = result.provenance.status
    if status == "verified_secondary":
        options = specification.options
        warnings.warn(
            NumericalWarning(
                "numerical.secondary_convergence",
                {"tol_secondary": options.tol_secondary, "tol": options.tol},
                status=status,
                certificate=result.certificate,
            ),
            stacklevel=stacklevel + 2,
        )
    if status in CERTIFIED:
        omega = specification.estimator.mean.omega
        if 0 < abs(omega) <= _SMALL_MEAN_ORDER:
            warnings.warn(
                NumericalWarning(
                    "numerical.small_mean_order",
                    {"omega": omega},
                    status=status,
                    certificate=result.certificate,
                ),
                stacklevel=stacklevel + 2,
            )
        if result._indefinite_root():
            warnings.warn(
                NumericalWarning(
                    INDEFINITE, status=status, certificate=result.certificate
                ),
                stacklevel=stacklevel + 2,
            )
        return result
    if specification.on_failure == "raise":
        if status == "empty":
            raise SampleError("sample.empty", {"count": result.sample.nobs})
        if status == "not_identified":
            raise IdentificationError(
                "identification.not_identified", not_identified_fields(result)
            )
        withheld = not_certified(result)
        if status == "no_finite_root":
            raise ExistenceError(
                "existence.no_finite_root",
                status=status,
                last_state=withheld.last_state,
                certificate=withheld.certificate,
            )
        raise withheld
    return result


def fit_prepared(
    problem: PreparedProblem, specification: Specification, *, stacklevel: int = 1
) -> Result:
    """Fit a prepared problem under one specification (module docstring).

    Parameters
    ----------
    problem : PreparedProblem
        The prepared problem, whose domain and base are reused.
    specification : Specification
        The typed records of the fit.
    stacklevel : int, default 1
        The frames above this function at which the secondary warning, and the
        coverage warning of a joint fit outside ``[1, 2]`` (with or without absorbed
        effects), name the caller's line: 1 for a direct caller.

    Returns
    -------
    Result
        The result of the fit, certified or, with ``on_failure="return"``, visibly
        not certified.

    Raises
    ------
    ConvergenceError
        No certified solution, with ``on_failure="raise"``: the last state and the
        certificate, where one was formed, are attached.
    SampleError
        An order at or below zero on an outcome with zeros without
        ``positive_only``; no row left.
    SpecificationError
        A cluster that the preparation did not retain, or with a missing value on
        the estimation sample; a start of another length than the design or not
        representable in the coordinates of the solver, on either route;
        ``newton_provider="sklearn"`` without a verified installation; for a named
        HC method, before any solve, HC1 at ``N = K``
        (``specification.no_residual_dof``) or an explicit second correction
        (``specification.conflicting``).
    ValueError
        For a named HC method, before any solve, a design that is not of full column
        rank on its rows of positive weight, whose leverage is undefined.
    """
    if (
        specification.positive_only is not None
        and specification.positive_only != problem.positive_only
    ):
        raise SpecificationError(
            "specification.conflicting",
            {
                "first": f"positive_only={specification.positive_only!r}",
                "second": f"the prepared positive_only={problem.positive_only!r}",
            },
        )
    estimator = specification.estimator
    if problem.has_fixed_effects:
        if specification.solver == "iols":
            refuse_iols("fixed effects")
        fit_fe = (
            fit_fe_prepared
            if TYPE_CHECKING
            else import_module("powermean.interface.fe_dispatch").fit_fe_prepared
        )
        fitted = fit_fe(problem, specification, stacklevel=stacklevel)
        return _report(fitted, specification, stacklevel)
    if specification.solver == "iols":
        if problem.frame.columns.weights is not None:
            refuse_iols("weights are given")
        if (
            problem.frame.columns.offset is not None
            or problem.frame.columns.exposure is not None
        ):
            refuse_iols("an offset or exposure is given")
    sample = problem.sample(estimator.mean.omega)
    if estimator.route == "linear":
        sample = _identified(sample, stacklevel)
        computation = _computation(sample, specification)
        fit_preflight(sample, specification.inference)
        handoff = _linear_handoff(problem, sample, specification, computation)
    else:
        handoff = _ordinary_power_handoff(
            problem, sample, specification, stacklevel + 1
        )
    return _report(Result(bind(handoff)), specification, stacklevel)


def _ordinary_power_handoff(
    problem: PreparedProblem,
    sample: OrderSample,
    specification: Specification,
    stacklevel: int,
) -> Handoff:
    """Admit the characterized sample and invoke the same ordinary solver once."""
    estimator = specification.estimator
    variance = estimator.variance
    support = None
    gamma = None
    if isinstance(variance, NB2Variance) or (
        isinstance(variance, PowerVariance) and 1 <= variance.tau < 2
    ):
        support = separation(sample, problem.frame.base, variance)
        sample = support.sample
    if isinstance(variance, PowerVariance) and variance.tau == 2:
        gamma = recession(sample, variance)
    refused = gamma is not None or (
        support is not None and (support.exhausted is not None or support.unresolved)
    )
    if not refused and (support is None or not support.full_rank):
        sample = _identified(sample, stacklevel)
    computation = _computation(sample, specification)
    steps = () if support is None else support.steps
    if refused:
        if specification.start is not None:
            mapped_start(sample, specification.start)
        return Handoff(
            estimator,
            problem.frame,
            sample,
            specification.inference,
            computation,
            keep_influence=specification.keep_influence,
            requested_start=specification.start,
            offset_design=problem.offset_design,
            support=steps,
            recession=gamma,
        )
    preflight = fit_preflight(sample, specification.inference)
    _coverage(sample, specification, stacklevel)
    handoff = _power_handoff(
        problem, sample, specification, computation, preflight=preflight
    )
    return replace(handoff, support=steps) if steps else handoff


def _identified(sample: OrderSample, stacklevel: int) -> OrderSample:
    """The sample without its collinear regressors, each named once to the caller.

    The collinearity step of the routes with fixed effects, called without blocks on
    the retained rows: its exact rank and relation analysis, in formula order (owner's
    rulings 109 and 110). Full direct rank on the already scaled design needs no
    Gram matrix or diagnostic records. Only the full analysis may select omissions.
    A sample for which the step proposes nothing, or does not
    complete, is returned as it came, the same object.  Otherwise one
    ``coverage.collinear_omitted`` note per omitted regressor is issued at the
    caller's line, before the solve, and the sample keeps the other columns
    (:func:`~powermean.data.retained.without_columns`).
    """
    if _full_rank(sample.design.matrix):
        return sample
    data = sample.data
    analysis = analyze_collinearity(
        CollinearityProblem(data.design, data.column_names, sample.positions, None),
        None,
        mode="drop",
    )
    reasons = analysis.reasons if analysis.status == COMPLETE else None
    kept = [j for j, reason in enumerate(reasons or ()) if reason is None]
    if reasons is None or not kept or len(kept) == len(reasons):
        return sample
    for name, reason in zip(data.column_names, reasons, strict=True):
        if reason is not None:
            warnings.warn(
                CoverageWarning("coverage.collinear_omitted", {"name": name}),
                stacklevel=stacklevel + 2,
            )
    return without_columns(sample, kept)


def _full_rank(scaled: FloatArray) -> bool:
    """Use the collinearity step's direct rank on the already scaled design.

    Its unabsorbed analysis excludes zero columns and then scales by column maxima
    rounded down to powers of two. With no zero column, the second scaling in
    `within_rank` is the identity because the maxima are already in [1, 2).
    At full direct rank the Gram rank cannot cause an omission. Uncertain designs
    still go through the existing analysis and its relation checks.
    """
    n, k = scaled.shape
    if n < k or not np.any(scaled, axis=0).all():
        return False
    try:
        rank = numerical_rank(scaled, [], kind="direct_svd", purpose="within columns")
    except np.linalg.LinAlgError:
        return False
    return rank.rank == int(k)


def _coverage(
    sample: OrderSample, specification: Specification, stacklevel: int
) -> None:
    """The ``CoverageWarning`` of an equation whose finite root BBP leaves open.

    ``design/02_computation.md``, Section 4: the existence of a finite root of the
    joint equation is not characterized for ``tau < 1``, nor for ``tau > 2`` with zero
    outcomes; rank checks only, the solver and the certificate decide, and no row is
    dropped for it.  Emitted once, before the solve, at the caller's line.
    """
    variance = specification.estimator.variance
    if not isinstance(variance, PowerVariance) or not outside_convex(variance):
        return
    active = sample.data.weight_values > 0
    zeros = bool(np.any(sample.problem.response[active] == 0))
    if variance.tau > 2.0 and not zeros:
        return
    rows = "with zero outcomes" if zeros else "without zero outcomes"
    case = (
        f"The joint fit at tau = {variance.tau!r} {rows}, whose existence of a "
        "finite root is not characterized (rank checks only; the certificate decides),"
    )
    warnings.warn(
        CoverageWarning("coverage.outside_bbp", {"case": case}),
        stacklevel=stacklevel + 2,
    )


def _computation(sample: OrderSample, specification: Specification) -> Computation:
    """Retain IOLS presence and canonical values without an unproved mapping cast."""
    if specification.solver != "iols":
        estimator = specification.estimator
        if estimator.route == "power" and outside_convex(estimator.variance):
            resolved = {
                **specification.options.to_dict(),
                **root_options_of(specification).to_dict(),
            }
            return Computation(options=_snapshot(resolved), solver="newton")
        return Computation(
            options=specification.options.to_dict(), solver=specification.solver
        )
    if intercept_column(sample) is None:
        refuse_iols("the design has no intercept")
    if sample.data.nparams < 2:
        refuse_iols("the design is intercept-only")
    if sample.count <= sample.data.nparams:
        refuse_iols(f"n <= K (n = {sample.count}, K = {sample.data.nparams})")
    iols_opts = specification.iols_options or IolsOptions()
    values = dict(_snapshot(iols_opts.to_dict(), solver="iols"))
    values.update(
        {
            "tol": specification.options.tol,
            "tol_secondary": specification.options.tol_secondary,
            "coef_tol": specification.options.coef_tol,
            "rcond_min": specification.options.rcond_min,
        }
    )
    return Computation(
        options=values, solver="iols", iols_backend_explicit=iols_opts.backend_explicit
    )
