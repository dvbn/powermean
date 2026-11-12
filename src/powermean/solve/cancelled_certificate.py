"""Fresh original cancelled equations and the covariance-dependent certificate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, finite_product
from powermean.fe.admission_records import WorkLog
from powermean.model.variance import CancelledFamily, FloatArray, PowerVariance
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
from powermean.solve.quantities import curvature, euclidean_norm, root_mean_square
from powermean.solve.reported import ReportedScale, coefficient_check
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


def materialize_hessian(
    state: IdentifiedState, work: WorkLog, label: str = "hessian_materialization"
) -> FloatArray:
    """Materialize the ``d x d`` negative Jacobian ``H(theta)`` from ``jac_vec``."""
    d = state.theta.size
    work.begin(label)
    hessian, basis = np.empty((d, d), dtype=np.float64), np.zeros(d, dtype=np.float64)
    for column in range(d):
        basis[column] = 1.0
        hessian[:, column] = work.run("jac_vec", lambda: state.jac_vec(basis))
        basis[column] = 0.0
    result = finite_product(hessian)
    work.complete(label)
    return result


def build_merit_cholesky(
    capture: CapturedFit,
    theta_start: FloatArray,
    metric: MeritMetric,
    work: WorkLog,
) -> FloatArray:
    """Build the fixed SPD Cholesky factor ``L_M`` of ``M`` at ``theta_start``."""
    if metric == "gamma":
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


def _factor(
    result: _PointWork,
    controls: CertificateControls,
    work: WorkLog,
    *,
    merit_cholesky: FloatArray | None,
    require_pd: bool,
) -> Factor | None:
    assert result.score is not None
    assert result.hessian is not None
    score, hessian = result.score.view(), result.hessian.view()
    checked = work.run("cholesky_gate", lambda: curvature(score, hessian))
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
) -> CertifiedCancelledPoint:
    """Fresh original sum moments, actual H, correction, RMS and coefficient gate.

    All at one point.
    """
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
    if result.measures.evaluated:
        factor = _factor(
            result,
            controls,
            work,
            merit_cholesky=merit_cholesky,
            require_pd=require_pd,
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
    return CertifiedCancelledPoint(certificate, result.state)


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
    geometry = capture.admission.geometry
    if start is not None:
        pt = FloatStorage.store(geometry.checked_coefficients(start))
        return RootInitializerRecord("user_start", False, None, pt, None, None, 0, 0)
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

    def assess_tau2(point: FloatArray) -> Assessment:
        cert = certify_cancelled(
            capture, point, controls, bounds=bounds, reported=reported, work=work
        )
        history.append(cert.certificate)
        return cert.certificate.assessment()

    attempts = work.run(
        "init_tau2_attempts",
        lambda: run_cancelled_attempts(
            problem, zero_start, options, assess=assess_tau2
        ),
    )
    retained = attempts.attempts[attempts.retained]
    point = zero_start if retained.state is None else retained.state.coefficients
    certified = certify_cancelled(
        capture, point, controls, bounds=bounds, reported=reported, work=work
    )
    history.append(certified.certificate)
    init_iters = sum(item.iterations for item in attempts.attempts)
    init_evals = sum(item.evaluations.total for item in attempts.attempts)
    status = certified.certificate.status(retained.reason)
    pt, cert = FloatStorage.store(point), certified.certificate
    return RootInitializerRecord(
        "gamma_tau2", cert.accepted, status, pt, cert, attempts, init_iters, init_evals
    )


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
    status: Status = (
        "singular_system"
        if initializer.certified or initializer.status is None
        else initializer.status
    )
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
