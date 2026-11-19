"""Fresh original cancelled equations and covariance-dependent D49 certificate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, finite_product
from powermean.fe.admission_records import WorkLog
from powermean.fe.geometry_design import NumericalGeometry
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    PowerVariance,
    cancelled_score_unit,
)
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.cancelled_attempt import run_cancelled_attempts
from powermean.solve.cancelled_operator import (
    AttemptBounds,
    cancelled_problem,
    coordinate_map,
)
from powermean.solve.cancelled_records import (
    CancelledCertificate,
    CancelledMeasures,
    MeritMetric,
    RootExecutionBounds,
    RootInitializerRecord,
    RootOptions,
    RootSolveResult,
    ScoreMetricKind,
    has_convex_criterion,
    reserve_certificate,
)
from powermean.solve.certificate import GateName, failed_gates
from powermean.solve.factorization import (
    Factor,
    SingularJacobianError,
    factorize,
    forward_solve,
    symmetric_cholesky,
)
from powermean.solve.options import SolverOptions
from powermean.solve.quantities import (
    Curvature,
    euclidean_norm,
    relative_score,
    root_mean_square,
)
from powermean.solve.reported import ReportedScale, coefficient_check, saturated
from powermean.solve.status import Status
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import IdentifiedState

_DEFAULT_VARIANCE = PowerVariance(2.0)


class CancelledScaleProvider(Protocol):
    """Same-point reporting supplied by inference using this actual H factor."""

    def __call__(self, state: IdentifiedState, factor: Factor, /) -> ReportedScale:
        """Return the direct scale of kept original slopes at this exact point."""
        ...


@dataclass(frozen=True)
class CertifiedCancelledPoint:
    """Local fresh state plus its immutable diagnostic certificate."""

    certificate: CancelledCertificate
    state: IdentifiedState | None


@dataclass
class _PointWork:
    """One certification's partial quantities, including failed source evidence."""

    state: IdentifiedState | None
    measures: CancelledMeasures
    score: FloatStorage | None
    hessian: FloatStorage | None
    correction: FloatStorage | None
    errors: dict[GateName, str]
    lower: FloatArray | None = None


def certified_again(
    made: Sequence[CertifiedCancelledPoint], point: FloatArray
) -> CertifiedCancelledPoint | None:
    """The last certification in ``made``, where ``point`` is its point bit for bit.

    Within one route the capture, controls, bounds, variance and scale provider
    are fixed, so a certification is a function of its point, and it leaves the
    provider holding that point's inference. A request at the point of the last
    certification, with none in between, would evaluate the same record again; it
    is returned instead, and nothing is evaluated or charged a second time.
    """
    if not made:
        return None
    last = made[-1]
    values, gamma = np.asarray(point), last.certificate.gamma
    if values.dtype != np.float64 or values.shape != gamma.shape:
        return None
    return last if values.tobytes() == gamma.tobytes() else None


def materialize_hessian(
    state: IdentifiedState, work: WorkLog, label: str = "hessian_materialization"
) -> FloatArray:
    """Materialize the ``d x d`` negative Jacobian ``H(theta)`` of ``state``.

    The state's row assembly of the same matrix is used where it applies and is
    charged as the ``d`` ``jac_vec`` columns it stands for; otherwise each column
    is one ``jac_vec`` product, which decides and raises as before.
    """
    d = state.theta.size
    work.begin(label)
    assembled = state.jacobian()
    if assembled is None:
        hessian = np.empty((d, d), dtype=np.float64)
        basis = np.zeros(d, dtype=np.float64)
        for column in range(d):
            basis[column] = 1.0
            hessian[:, column] = work.run("jac_vec", lambda: state.jac_vec(basis))
            basis[column] = 0.0
    else:
        hessian = assembled
        work.begin("jac_vec", d)
        work.complete("jac_vec", d)
    result = finite_product(hessian)
    work.complete(label)
    return result


def build_merit_cholesky(
    capture: CapturedFit,
    theta_start: FloatArray,
    metric: MeritMetric,
    work: WorkLog,
    certificate: CancelledCertificate | None = None,
    *,
    lower: FloatArray | None = None,
) -> FloatArray:
    """Build the fixed SPD Cholesky factor ``L_M`` of ``M`` at ``theta_start``.

    With the Gamma metric, the tau = 2 negative Jacobian at ``theta_start`` is the
    one that ``certificate`` holds where it certified this point, bit for bit, on
    ``PowerVariance(2.0)``: the same fresh equations and materialization formed
    it there, so it is not formed again; and ``lower``, the factor that its
    curvature gate formed of that matrix by the same ``symmetric_cholesky``, is
    the factor, which is then not formed again either.
    """
    if metric == "gamma":
        hessian_gamma = _gamma_hessian(certificate, theta_start)
        if (
            hessian_gamma is not None
            and lower is not None
            and isinstance(lower, np.ndarray)
            and lower.dtype == np.float64
            and lower.shape == hessian_gamma.shape
        ):
            return lower.copy()
        if hessian_gamma is None:
            gamma_state = work.run(
                "merit_gamma_equations",
                lambda: capture.equations(theta_start, variance=PowerVariance(2.0)),
            )
            hessian_gamma = materialize_hessian(
                gamma_state, work, "merit_hessian_materialization"
            )
        return work.run("merit_cholesky", lambda: symmetric_cholesky(hessian_gamma))
    gram_id = capture.admission.geometry.identified_gram
    return work.run("merit_cholesky", lambda: symmetric_cholesky(gram_id))


def _gamma_hessian(
    certificate: CancelledCertificate | None, point: FloatArray
) -> FloatArray | None:
    """The Gamma negative Jacobian that ``certificate`` formed at ``point``, or None."""
    if certificate is None or certificate.variance != PowerVariance(2.0):
        return None
    hessian, values = certificate.hessian, np.asarray(point)
    gamma = certificate.gamma
    if (
        hessian is None
        or values.dtype != np.float64
        or values.shape != gamma.shape
        or values.tobytes() != gamma.tobytes()
    ):
        return None
    return hessian


def whitened_score(
    score: FloatArray, merit_cholesky: FloatArray, work: WorkLog
) -> tuple[FloatArray, float, float]:
    """Compute ``u = L_M^-1 S``, ``||u||_2``, and ``phi = 1/2 ||u||_2^2``."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        u = finite_product(
            work.run("merit_solve", lambda: forward_solve(merit_cholesky, score))
        )
        norm_u = euclidean_norm(u)
        phi = 0.5 * norm_u * norm_u
    if not np.isfinite(phi):
        raise FloatingPointError("merit value overflowed double precision")
    return u, norm_u, phi


def _evaluate(
    capture: CapturedFit,
    point: FloatArray,
    variance: CancelledFamily,
    work: WorkLog,
    *,
    require_pd: bool,
) -> _PointWork:
    measures = CancelledMeasures(require_positive_definite=require_pd)
    result = _PointWork(None, measures, None, None, None, {})
    try:
        state = work.run(
            "fresh_equations", lambda: capture.equations(point, variance=variance)
        )
        result.state = state
        result.score = work.run(
            "sum_score_snapshot", lambda: FloatStorage.store(state.score)
        )
        hessian = materialize_hessian(state, work, "hessian_materialization")
        result.hessian = FloatStorage.store(hessian)
        result.measures = replace(result.measures, evaluated=True)
    except (FloatingPointError, ValueError) as error:
        result.errors["evaluation"] = f"{type(error).__name__}: {error}"
    return result


def _curvature(
    moment: FloatArray, bread: FloatArray
) -> tuple[Curvature, FloatArray | None]:
    """:func:`powermean.solve.quantities.curvature`, with the Cholesky factor it forms.

    The same calls in the same order: ``symmetric_cholesky`` of the bread, then
    the norm of the forward solve; the factor is kept for the Gamma merit metric
    at the same point (:func:`build_merit_cholesky`).
    """
    try:
        lower = symmetric_cholesky(bread)
    except np.linalg.LinAlgError as error:
        return (
            Curvature(positive_definite=False, scaled_score=math.nan, error=str(error)),
            None,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = forward_solve(lower, moment)
    found = Curvature(
        positive_definite=True, scaled_score=euclidean_norm(whitened), error=""
    )
    return found, lower


def _factor(
    result: _PointWork,
    controls: CertificateControls,
    work: WorkLog,
    *,
    merit_cholesky: FloatArray | None,
    merit_metric: MeritMetric,
    require_pd: bool,
    unit: float,
) -> Factor | None:
    assert result.score is not None
    assert result.hessian is not None
    score, hessian = result.score.view(), result.hessian.view()
    checked, result.lower = work.run(
        "cholesky_gate", lambda: _curvature(score, hessian)
    )
    scaled = checked.scaled_score
    metric_kind: ScoreMetricKind = (
        "current_symmetric_hessian" if checked.positive_definite else "none"
    )
    if not checked.positive_definite and merit_cholesky is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            whitened = work.run(
                "merit_scaled_score", lambda: forward_solve(merit_cholesky, score)
            )
        scaled, metric_kind = euclidean_norm(whitened), "fixed_merit"
    # The Gamma metric is the tau = 2 Jacobian, linear in the outcome as S is; the
    # Gram metric of the design does not scale with it.
    scaled = relative_score(
        scaled,
        unit,
        unit_free_metric=metric_kind == "fixed_merit" and merit_metric == "gram",
    )
    result.measures = replace(
        result.measures,
        positive_definite=checked.positive_definite,
        scaled_score=scaled,
        require_positive_definite=require_pd,
        score_metric=metric_kind,
    )
    if checked.error and (require_pd or merit_cholesky is None):
        result.errors["curvature"] = checked.error
    try:
        factor = work.run("actual_lu", lambda: factorize(hessian, controls.rcond_min))
    except SingularJacobianError as error:
        result.measures = replace(result.measures, rcond=error.rcond)
        result.errors["conditioning"] = str(error)
        return None
    result.measures = replace(result.measures, conditioned=True, rcond=factor.rcond)
    return factor


def _correct(
    result: _PointWork,
    factor: Factor,
    controls: CertificateControls,
    reported: CancelledScaleProvider,
    work: WorkLog,
) -> None:
    assert result.state is not None
    assert result.score is not None
    state, score = result.state, result.score.view()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            correction = finite_product(
                work.run("correction_solve", lambda: factor.solve(score))
            )
        result.correction = FloatStorage.store(correction)
        change = work.run("index_action", lambda: state.design.matvec(correction))
        rms = root_mean_square(change, np.ones(change.size))
        result.measures = replace(result.measures, index_correction=rms)
    except (FloatingPointError, ValueError) as error:
        result.errors["index"] = f"{type(error).__name__}: {error}"
        return
    if result.measures.saturated:
        return  # no coefficient gate and no standard error (solve.reported.saturated)
    try:
        scale = work.run("same_point_scale", lambda: reported(state, factor))
        k = state.design.n_slopes_identified
        if type(scale) is not ReportedScale or not np.array_equal(
            scale.gamma, state.theta[:k]
        ):
            raise ValueError("cancelled reporting scale names another slope point")
        check = coefficient_check(
            scale.corrections(correction[:k]), scale.standard_errors, controls.coef_tol
        )
        result.measures = replace(result.measures, coefficients=check)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        result.errors["coefficient"] = f"{type(error).__name__}: {error}"


def saturated_geometry(geometry: NumericalGeometry) -> bool:
    """Whether a cancelled fit is saturated: rows = identified coordinates + groups.

    :func:`powermean.solve.reported.saturated` on the cancelled equation, which lives
    in the within-group centered space of dimension ``n_rows - n_groups``.
    """
    structure = geometry.structure
    return saturated(structure.n_rows, geometry.n_identified + structure.n_groups)


def _route_failed_gates(
    measures: CancelledMeasures, tolerance: float
) -> tuple[GateName, ...]:
    gates = failed_gates(measures, tolerance)
    if not measures.require_positive_definite and measures.score_metric != "none":
        return tuple(gate for gate in gates if gate != "curvature")
    return gates


def certify_cancelled(
    capture: CapturedFit,
    point: FloatArray,
    controls: CertificateControls,
    *,
    bounds: AttemptBounds,
    reported: CancelledScaleProvider,
    work: WorkLog,
    variance: CancelledFamily = _DEFAULT_VARIANCE,
    merit_cholesky: FloatArray | None = None,
    merit_metric: MeritMetric = "gamma",
) -> CertifiedCancelledPoint:
    """Fresh original sum moments, actual H, correction, RMS and D49 at one point."""
    certified, _ = _certify(
        capture,
        point,
        controls,
        bounds=bounds,
        reported=reported,
        work=work,
        variance=variance,
        merit_cholesky=merit_cholesky,
        merit_metric=merit_metric,
    )
    return certified


def _certify(
    capture: CapturedFit,
    point: FloatArray,
    controls: CertificateControls,
    *,
    bounds: AttemptBounds,
    reported: CancelledScaleProvider,
    work: WorkLog,
    variance: CancelledFamily = _DEFAULT_VARIANCE,
    merit_cholesky: FloatArray | None = None,
    merit_metric: MeritMetric = "gamma",
) -> tuple[CertifiedCancelledPoint, FloatArray | None]:
    """:func:`certify_cancelled`, with the curvature gate's Cholesky factor, if any."""
    reserve_certificate(capture.admission.geometry, bounds)
    if np.iscomplexobj(point):
        raise ValueError("cancelled certificate point must be real")
    values = np.asarray(point, dtype=np.float64)
    if values.shape != (capture.admission.geometry.n_identified,):
        raise ValueError("cancelled certificate point must match identified dimension")
    reset = getattr(reported, "reset", None)
    if callable(reset):
        reset()
    require_pd = has_convex_criterion(variance)
    candidate = FloatStorage.store(values)
    result = _evaluate(capture, candidate.view(), variance, work, require_pd=require_pd)
    if saturated_geometry(capture.admission.geometry):
        result.measures = replace(result.measures, saturated=True)
    if result.measures.evaluated:
        factor = _factor(
            result,
            controls,
            work,
            merit_cholesky=merit_cholesky,
            merit_metric=merit_metric,
            require_pd=require_pd,
            unit=cancelled_score_unit(capture.z, capture.obs_weights),
        )
        if factor is not None:
            _correct(result, factor, controls, reported, work)
    strict = _route_failed_gates(result.measures, controls.tol)
    secondary = _route_failed_gates(result.measures, controls.tol_secondary)
    certificate = CancelledCertificate(
        candidate,
        controls,
        result.measures,
        result.score,
        result.hessian,
        result.correction,
        result.errors,
        strict,
        secondary,
        variance=variance,
    )
    return CertifiedCancelledPoint(certificate, result.state), result.lower


def prepare_root_initializer(
    capture: CapturedFit,
    options: SolverOptions,
    exec_bounds: RootExecutionBounds,
    *,
    reported: CancelledScaleProvider,
    work: WorkLog,
    history: list[CancelledCertificate],
    start: FloatArray | None,
) -> RootInitializerRecord:
    """Resolve either the prescribed Gamma initializer or a checked user start."""
    record, _ = root_initializer(
        capture,
        options,
        exec_bounds,
        reported=reported,
        work=work,
        history=history,
        start=start,
    )
    return record


def root_initializer(
    capture: CapturedFit,
    options: SolverOptions,
    exec_bounds: RootExecutionBounds,
    *,
    reported: CancelledScaleProvider,
    work: WorkLog,
    history: list[CancelledCertificate],
    start: FloatArray | None,
) -> tuple[RootInitializerRecord, FloatArray | None]:
    """:func:`prepare_root_initializer`, with its certificate's curvature factor.

    The factor is the Cholesky factor of the symmetrized Gamma Hessian at the
    initializer, which the certificate's curvature gate formed; None for a user
    start or where the gate formed none.
    """
    geometry = capture.admission.geometry
    if start is not None:
        pt = FloatStorage.store(geometry.checked_coefficients(start))
        record = RootInitializerRecord("user_start", False, None, pt, None, None, 0, 0)
        return record, None
    controls, bounds = certificate_controls(options), exec_bounds.certificate
    att_bounds = bounds if exec_bounds.attempt is None else exec_bounds.attempt
    zero_start = np.zeros(geometry.n_identified, dtype=np.float64)
    operator = coordinate_map(
        geometry.design,
        geometry.basis_matrix,
        geometry.reduction_matrix,
        identity_atol=exec_bounds.identity_atol,
        bounds=att_bounds,
    )
    problem = work.run(
        "cancelled_problem",
        lambda: cancelled_problem(
            operator,
            capture.z,
            capture.centered_offset,
            variance=PowerVariance(2.0),
            obs_weights=capture.obs_weights,
        ),
    )

    made: list[CertifiedCancelledPoint] = []
    lowers: list[FloatArray | None] = []

    def assess_tau2(point: FloatArray) -> Assessment:
        cert, lower = _certify(
            capture, point, controls, bounds=bounds, reported=reported, work=work
        )
        history.append(cert.certificate)
        made.append(cert)
        lowers.append(lower)
        return cert.certificate.assessment()

    attempts = work.run(
        "init_tau2_attempts",
        lambda: run_cancelled_attempts(
            problem, zero_start, options, assess=assess_tau2
        ),
    )
    retained = attempts.attempts[attempts.retained]
    point = zero_start if retained.state is None else retained.state.coefficients
    certified = certified_again(made, point)
    if certified is None:
        certified, lower = _certify(
            capture, point, controls, bounds=bounds, reported=reported, work=work
        )
    else:
        lower = lowers[-1]
    history.append(certified.certificate)
    init_iters = sum(item.iterations for item in attempts.attempts)
    init_evals = sum(item.evaluations.total for item in attempts.attempts)
    status = certified.certificate.status(retained.reason)
    pt, cert = FloatStorage.store(point), certified.certificate
    record = RootInitializerRecord(
        "gamma_tau2", cert.accepted, status, pt, cert, attempts, init_iters, init_evals
    )
    return record, lower


def failed_root_result(
    capture: CapturedFit,
    variance: CancelledFamily,
    initializer: RootInitializerRecord,
    *,
    controls: CertificateControls,
    bounds: AttemptBounds,
    reported: CancelledScaleProvider,
    work: WorkLog,
    root_options: RootOptions,
) -> tuple[CertifiedCancelledPoint, RootSolveResult]:
    """Certify ``initializer.point`` on ``variance`` and build an early failure."""
    failed_cert = certify_cancelled(
        capture,
        initializer.point,
        controls,
        bounds=bounds,
        reported=reported,
        work=work,
        variance=variance,
    )
    status: Status = failed_cert.certificate.status("factorization_failure")
    iters, evals = initializer.iterations, initializer.evaluations
    res = RootSolveResult(
        failed_cert,
        (),
        status,
        root_options.merit_metric,
        replace(root_options),
        None,
        initializer,
        (),
        False,
        iters,
        evals,
    )
    return failed_cert, res
