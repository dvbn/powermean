"""The joint system with absorbed effects outside ``tau in [1, 2]``: sparse root Newton.

Root Newton: the actual Jacobian by equilibrated LU with its reciprocal condition
recorded, the merit ``S' M^-1 S / 2`` with a fixed positive-definite ``M``, Armijo on
differences, native stops, then the certificate.  A joint fit outside ``[1, 2]`` starts
from the certified solution at the nearer of ``tau = 1`` and ``tau = 2``, with the
continuation as fallback, and its result is a selected local root, certified as a
regular root, not as unique.  The nearest convex start is PPML below 1 and Gamma above
2, on the same sample and formulation; the cancelled and mixed Gamma start is
unchanged, and caller starts are kept.  The analogue with absorbed
effects of :mod:`powermean.solve.joint_root`; the convex solvers, their starts,
options and certificate are unchanged and never see a variance outside ``[1, 2]``.

The system.  ``B = [X, E_J]``: ``X`` the sample's scaled slopes, ``E_J`` the kept
incidence of the accepted fixed basis of the sample
(:func:`~powermean.fe.certificate_basis.owned_basis`); neither ``B`` nor any dummy
matrix is formed.  A point ``gamma = [theta, alpha]`` is a
:class:`~powermean.solve.absorbed_root_records.RootPoint`: ``base = fl(X theta + o)``
(:func:`~powermean.solve.absorbed_certificate_point.root_base`), the effects
``embed(alpha)`` and ``q`` their ordered additions to the base (the arithmetic of
:func:`~powermean.fe.profile_arithmetic.reconstruct_index`, which the point's
admission repeats and compares bit for bit).  At a stage variance its fresh rows are
``b = a r`` and ``C = a c`` with the OBSERVED curvature of the accepted row kernel,
signed, and its moment is the raw ``S = B'b``
(:func:`~powermean.solve.absorbed_certificate_geometry.moment`).

The steps of :func:`solve_absorbed_root`, in order:

1. The initializer.  A caller's start is accepted as given and recorded: its slopes,
   with zero nuisance coordinates (the cold coefficient witness of the fitted-index
   caller start); it is not certified, and no continuation leaves from it.
   Otherwise the nearest convex member of the family is fitted on the SAME sample
   object, by the organization the fit requested, with its unchanged solver, start,
   options, trigger and audit (the caller's ``convex`` callable); nothing selects,
   drops or adds a row for it.  Its point enters the full coordinates with the slopes
   it holds and ``alpha = sum_d w_d e_d`` over every original level ``d``, the exact
   rational witnesses ``D_d = E_J w_d`` of the basis applied to its effects (the
   normalized recovery of a fitted-index point, the profile state of a profiled
   one); the start's index gap to the convex index is recorded, not gated.
2. The merit metric, formed once at the start: ``"gamma"`` (the default), the
   observed bread of the Gamma equation of the same rows at the start,
   ``B' diag(a c_Gamma) B``; or ``"gram"``, ``B'B`` on the rows of positive weight;
   both assembled sparsely by the certificate's accepted
   :func:`~powermean.solve.absorbed_certificate_geometry.hessian`, equilibrated and
   symmetrized as the certificate's curvature metric, and factorized once by the
   certificate's sparse factor (kind ``"merit"``).  It is positive definite only
   when that factor holds ``perm_r == perm_c``, finite ``L`` and ``U`` and every
   pivot positive; otherwise the requested metric is refused (``LinAlgError``:
   ``factorization_failure``; a value not representable: ``numerical_breakdown``) and
   no other metric is substituted.  ``u = D^-1/2 L^-1 Pr (S/s)`` then gives
   ``||u||^2 = S' M^-1 S`` exactly in the arithmetic of the factor
   (:func:`~powermean.solve.absorbed_certificate_sparse.merit_whitened`).
3. The root solve.  The shared kernel of :mod:`powermean.solve.root_kernel`, its
   attempt, retry, Armijo on differences and continuation unchanged: the equations
   above under each stage variance; the direction ``H delta = S`` with the actual
   observed ``H = B' diag(C) B`` assembled sparsely, equilibrated, factorized by the
   certificate's ordinary SuperLU and refused (``SingularJacobianError``) where the
   factor fails or ``rcond <= rcond_min`` by the certificate's own conditioning
   estimate; the whitening of step 2; every stage point certified afresh by
   :func:`~powermean.solve.absorbed_certificate.certify_point` on that stage's
   equation with the fixed merit.  The continuation leaves from the initializer's
   own ``tau``.  Every direction's factor outcome, nonzeros, conditioning and counted
   solves are recorded, and every operation is counted in the work log.
4. The conclusion.  The point of the target stage, whose certificate is the fresh
   certificate on the target equation (the stage of the target variance is always
   the one retained); where no root stage opened (the automatic initializer was not
   certified, its point had no representable effects, or the metric could not be
   formed) no point is selected and the operative reason is the initializer's or the
   metric's: no inference is formed at an initializer, and the convex fit's own
   audit stays in its record.

No absorption, profile, dense indicator, ridge, pseudoinverse, weight floor, metric
swap or second optimizer is used by the root stages.  Budgets: the convex
initializer's accepted steps and pointwise evaluations are charged to ``max_iter``
and ``max_evaluations`` before the root solve receives what remains.  The selected
root is not claimed unique, global or a minimum: the certificate establishes a
regular root.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.absorb_numeric import checked_difference, checked_product, checked_sum
from powermean.fe.admission_records import WorkLog
from powermean.fe.certificate_basis import CertificateBasis, embed, owned_basis
from powermean.fe.certificate_coordinates import coordinates
from powermean.fe.reductions import immutable_array
from powermean.model.rows import evaluate_rows
from powermean.model.variance import CancelledFamily, FloatArray, PowerVariance
from powermean.solve.absorbed_certificate import (
    CandidateAudit,
    FEPointCertificate,
    certify_point,
)
from powermean.solve.absorbed_certificate import assessment as fe_assessment
from powermean.solve.absorbed_certificate_factor import (
    Calls,
    SuperLU,
    factor_side,
    factorized,
)
from powermean.solve.absorbed_certificate_geometry import (
    equilibration,
    hessian,
    moment,
    one_norm,
    scaled_curvature,
    scaled_ordinary,
)
from powermean.solve.absorbed_certificate_point import root_base
from powermean.solve.absorbed_certificate_scale import FEScaleProvider
from powermean.solve.absorbed_certificate_sparse import conditioning, merit_whitened
from powermean.solve.absorbed_direction import IndexPoint, normalize
from powermean.solve.absorbed_root_records import (
    OPERATION,
    AbsorbedRootCandidate,
    AbsorbedRootInitializer,
    AbsorbedRootStage,
    AbsorbedRootWork,
    MeritFactor,
    RootDirectionRecord,
    RootPoint,
    RootStageSummary,
)
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.cancelled_records import MeritMetric, RootOptions
from powermean.solve.certificate import outside_convex
from powermean.solve.factorization import SingularJacobianError
from powermean.solve.joint_root import GAMMA, NOT_ROOT, nearest_convex
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.newton_trace import NativeReason, Timing
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.quantities import euclidean_norm
from powermean.solve.root_kernel import (
    RootBudget,
    RootKernel,
    RootRun,
    StageOutcome,
    Whitened,
    run_continuation,
    run_root_stage,
)
from powermean.solve.status import NATIVE_STATUS, Status
from powermean.systems.profiled import PointProfile

ConvexFit = tuple[IndexCandidate | ProfiledCandidate, CandidateAudit]
# The caller's same-sample convex fit of the requested organization at a variance.
ConvexRunner = Callable[[PowerVariance], ConvexFit]
NOT_REPRESENTABLE = "the root point is not representable in float64"


# ------------------------------------------------------------------- the points


def ordered_index(
    base: FloatArray, effects: tuple[FloatArray, ...], basis: CertificateBasis
) -> FloatArray:
    """``((base + e_1[c_1]) + e_2[c_2]) + ...``, checked, in block order.

    The ordered additions of the accepted
    :func:`~powermean.fe.profile_arithmetic.reconstruct_index`; a root point's
    admission rebuilds its index with that function and requires these bits.
    """
    q = base
    for block, values in zip(basis.grouping.blocks, effects, strict=True):
        q = checked_sum(q, values[block.codes], "reconstructed index")
    return q


def point_at(
    sample: FESample,
    basis: CertificateBasis,
    gamma: FloatArray,
    *,
    normalized: bool = False,
) -> RootPoint:
    """The root point of ``gamma = [theta, alpha]`` (module docstring).

    With ``normalized`` the accepted component normalization of its effects is
    formed and kept (a selected point's).  ``FloatingPointError`` where the point,
    its base or its index is not representable.
    """
    values = np.asarray(gamma, dtype=np.float64)
    k = int(sample.active.size)
    if values.shape != (k + basis.n_kept,):
        raise ValueError("gamma must hold the slopes and the kept coordinates")
    if not np.isfinite(values).all():
        raise FloatingPointError(NOT_REPRESENTABLE)
    # Owned copies, as the admission reads the record's: the same product operands.
    theta, alpha = immutable_array(values[:k]), immutable_array(values[k:])
    base = root_base(sample, theta)
    effects = embed(basis, alpha)
    q = ordered_index(base, effects, basis)
    found = None if not normalized else normalize(basis.grouping, effects)
    return RootPoint(theta, alpha, base, effects, q, found)


# ------------------------------------------------------------------- the fit


@dataclass(frozen=True, eq=False)
class _Fit:
    """One solve's fixed inputs, its work log and its running records."""

    sample: FESample
    variance: PowerVariance
    options: SolverOptions
    root_options: RootOptions
    absorbed: AbsorbedOptions
    scale: FEScaleProvider
    basis: CertificateBasis
    design: FloatArray
    controls: CertificateControls
    work: WorkLog
    clock: tuple[float, float]
    directions: list[RootDirectionRecord] = field(default_factory=list)
    merit_calls: Calls = field(default_factory=Calls)

    def timing(self) -> Timing:
        """The time since the solve began."""
        wall, cpu = self.clock
        return Timing(time.perf_counter() - wall, time.process_time() - cpu)


@dataclass(frozen=True, eq=False)
class _State:
    """One fresh state of the full joint equations at a point under a variance."""

    point: RootPoint
    variance: PowerVariance
    score: FloatArray
    curvature: FloatArray

    @property
    def theta(self) -> FloatArray:
        """The point in the coordinates of the root solve, ``[theta, alpha]``."""
        return self.point.gamma


def _stage_variance(variance: CancelledFamily) -> PowerVariance:
    """A stage variance of a joint root solve: a power variance outside ``[1, 2]``."""
    if not isinstance(variance, PowerVariance) or not outside_convex(variance):
        raise TypeError(NOT_ROOT)
    return variance


def _equations(fit: _Fit) -> Callable[[FloatArray, CancelledFamily], _State]:
    """Fresh rows and ``S = B'b`` of the sample's rows under a stage variance."""
    sample = fit.sample
    response, weights = sample.problem.response, sample.data.weight_values

    def equations(gamma: FloatArray, variance: CancelledFamily) -> _State:
        stage = _stage_variance(variance)
        point = point_at(sample, fit.basis, gamma)
        rows = evaluate_rows(response, point.index, variance=stage)
        b = checked_product(weights, rows.score, "weighted score")
        c = checked_product(weights, rows.observed_curvature, "weighted curvature")
        if not (np.isfinite(b).all() and np.isfinite(c).all()):
            raise FloatingPointError("the fresh rows are not representable")
        score = moment(fit.design, b, fit.basis)
        return _State(point, stage, score, immutable_array(c))

    return equations


DirectionStatus = Literal["solved", "refused", "failed"]
Nonzeros = tuple[int | None, int | None]
UNMEASURED = (math.nan, math.nan)


class _RefusedError(SingularJacobianError):
    """A refused direction, with the nonzeros and conditioning formed before it."""

    def __init__(
        self,
        message: str,
        nonzeros: Nonzeros = (None, None),
        measured: tuple[float, float] = UNMEASURED,
    ) -> None:
        super().__init__(message, measured[1])
        self.nonzeros = nonzeros
        self.measured = measured


def _record(
    fit: _Fit,
    state: _State,
    outcome: tuple[DirectionStatus, str],
    nonzeros: Nonzeros,
    measured: tuple[float, float],
    *,
    calls: Calls,
) -> None:
    """Keep one direction's record (``outcome`` is its status and reason)."""
    status, reason = outcome
    l_nnz, u_nnz = nonzeros
    estimate, rcond = measured
    fit.directions.append(
        RootDirectionRecord(
            state.variance.tau,
            status,
            reason,
            l_nnz,
            u_nnz,
            estimate,
            rcond,
            calls.actions(),
        )
    )


def _solved(
    fit: _Fit, state: _State, calls: Calls
) -> tuple[FloatArray, Nonzeros, tuple[float, float]]:
    """The equilibrated ordinary factor of the actual ``H``, its test and ``delta``.

    One sparse assembly of ``H = B' diag(C) B``; its equilibration (a zero row or
    column refuses: an exactly singular system), the certificate's ordinary factor
    and its conditioning estimate, then one counted ``N`` solve.
    ``SingularJacobianError`` (:class:`_RefusedError`) where the factor fails or
    ``rcond <= rcond_min``; ``FloatingPointError`` where a value is not
    representable.
    """
    work = fit.work
    matrix = work.run(
        "root_hessian", lambda: hessian(fit.design, state.curvature, fit.basis)
    )
    try:
        scale = equilibration(matrix)
    except ValueError as error:
        raise _RefusedError(str(error)) from error
    ordinary = scaled_ordinary(matrix, scale)
    norm = one_norm(ordinary)
    record, factor = work.run("root_lu", lambda: factorized(ordinary, "ordinary"))
    nonzeros = (record.l_nnz, record.u_nnz)
    if factor is None:
        raise _RefusedError(record.reason or "the ordinary factor failed", nonzeros)
    found: SuperLU = factor
    size = int(scale.size)
    estimate, rcond, reason = work.run(
        "root_conditioning",
        lambda: conditioning(found, norm, size, fit.options.rcond_min, calls),
    )
    if reason:
        raise _RefusedError(reason, nonzeros, (estimate, rcond))
    with np.errstate(all="ignore"):
        rhs = state.score / scale
        solved = work.run("root_direction_solve", lambda: calls.solve(found, rhs, "N"))
        delta = solved / scale
    if not np.isfinite(delta).all():
        raise FloatingPointError("the root direction is not representable")
    return immutable_array(delta), nonzeros, (estimate, rcond)


def _direction(fit: _Fit) -> Callable[[_State], FloatArray]:
    """The root-Newton step ``H delta = S`` by the sparse factor of the actual ``H``.

    Every outcome is recorded before it reaches the attempt: a refusal stops it with
    ``factorization_failure``, a value out of range with ``numerical_breakdown``.
    """

    def direction(state: _State) -> FloatArray:
        calls = Calls()
        try:
            delta, nonzeros, measured = _solved(fit, state, calls)
        except _RefusedError as error:
            outcome: tuple[DirectionStatus, str] = ("refused", str(error))
            _record(fit, state, outcome, error.nonzeros, error.measured, calls=calls)
            raise
        except (FloatingPointError, ValueError) as error:
            failed: tuple[DirectionStatus, str] = ("failed", str(error))
            _record(fit, state, failed, (None, None), UNMEASURED, calls=calls)
            raise
        _record(fit, state, ("solved", ""), nonzeros, measured, calls=calls)
        return delta

    return direction


def _whitener(fit: _Fit, merit: MeritFactor) -> Callable[[FloatArray], Whitened]:
    """``u = D^-1/2 L^-1 Pr (S/s)``, ``||u||`` and ``phi = ||u||^2 / 2``."""

    def whiten(score: FloatArray) -> Whitened:
        u = fit.work.run(
            "merit_solve", lambda: merit_whitened(merit, score, fit.merit_calls)
        )
        norm = euclidean_norm(u)
        with np.errstate(over="ignore"):
            phi = 0.5 * norm * norm
        if not math.isfinite(phi):
            raise FloatingPointError("merit value overflowed double precision")
        return u, norm, phi

    return whiten


# ------------------------------------------------------------------- the metric


def _metric_curvature(
    sample: FESample, start: RootPoint, metric: MeritMetric, work: WorkLog
) -> FloatArray:
    """The row weights of the metric: ``a c_Gamma`` at the start, or ``1[a > 0]``."""
    weights = sample.data.weight_values
    if metric == "gram":
        return immutable_array((weights > 0).astype(np.float64))
    response = sample.problem.response
    rows = work.run(
        "merit_gamma_equations",
        lambda: evaluate_rows(response, start.index, variance=GAMMA),
    )
    found = checked_product(weights, rows.observed_curvature, "Gamma curvature")
    if not np.isfinite(found).all():
        raise FloatingPointError("the Gamma curvature is not representable")
    return immutable_array(found)


def merit_factor(
    sample: FESample, start: RootPoint, metric: MeritMetric, *, work: WorkLog
) -> MeritFactor:
    """The fixed merit metric at ``start`` (module docstring, step 2).

    ``LinAlgError`` where the metric is not positive definite (a zero row or column,
    a failed factor, an off-diagonal pivot, a factor value that is not finite or a
    nonpositive pivot); ``FloatingPointError`` where it cannot be formed in range.
    """
    curvature = _metric_curvature(sample, start, metric, work)
    basis = owned_basis(sample)
    design = sample.design
    slopes = np.zeros((sample.count, 0)) if design is None else design.matrix
    matrix = work.run("merit_assembly", lambda: hessian(slopes, curvature, basis))
    try:
        scale = equilibration(matrix)
    except ValueError as error:
        raise np.linalg.LinAlgError(str(error)) from error
    symmetric = scaled_curvature(matrix, scale)
    record, factor = work.run("merit_factor", lambda: factorized(symmetric, "merit"))
    if factor is None:
        raise np.linalg.LinAlgError(record.reason or "the merit factor failed")
    lower, upper = factor_side(factor, "L"), factor_side(factor, "U")
    perm_r, perm_c, diagonal = record.perm_r, record.perm_c, record.diagonal
    if perm_r is None or perm_c is None or diagonal is None:
        raise np.linalg.LinAlgError("the merit factor holds no permutation or pivots")
    if not np.array_equal(perm_r, perm_c):
        raise np.linalg.LinAlgError(
            "a threshold-zero off-diagonal pivot of the merit (perm_r != perm_c)"
        )
    finite = np.isfinite(lower.data).all() and np.isfinite(upper.data).all()
    if not (finite and np.isfinite(diagonal).all()):
        raise np.linalg.LinAlgError("the merit factor holds a value that is not finite")
    if not (diagonal > 0).all():
        raise np.linalg.LinAlgError(
            "the merit metric is not positive definite: a nonpositive pivot"
        )
    return MeritFactor(
        metric,
        matrix,
        scale,
        symmetric,
        record,
        np.asarray(lower.indptr),
        np.asarray(lower.indices),
        np.asarray(lower.data, dtype=np.float64),
        diagonal,
        perm_r,
    )


# ------------------------------------------------------------------- certificates


@dataclass(frozen=True, eq=False)
class _Unformed:
    """A stage point whose certificate cannot be formed: it is not representable."""

    gamma: FloatArray
    reason: str


Certified = FEPointCertificate | _Unformed


def stage_status(certified: Certified, native: NativeReason) -> Status:
    """The status of a stage point.

    The certificate's level, then the gates that describe the point (``nonfinite``
    for an evaluation that is not formed, ``singular_system`` for a failed curvature
    or conditioning gate), then the native reason (:data:`~powermean.solve.status.
    NATIVE_STATUS`); an unformed point is ``nonfinite``.
    """
    if not isinstance(certified, FEPointCertificate):
        return "nonfinite"
    if certified.level == "strict":
        return "verified"
    if certified.level == "secondary":
        return "verified_secondary"
    if "evaluation" in certified.strict:
        return "nonfinite"
    if "curvature" in certified.strict or "conditioning" in certified.strict:
        return "singular_system"
    return NATIVE_STATUS[native]


@dataclass(frozen=True, eq=False)
class _StageCertifier:
    """The fresh certificate of a stage point on that stage's equation (step 3)."""

    fit: _Fit
    merit: MeritFactor

    def certify(self, variance: CancelledFamily, point: FloatArray) -> Certified:
        """Certify ``point`` on the equation of ``variance`` with the fixed merit."""
        stage = _stage_variance(variance)
        fit = self.fit
        try:
            root = point_at(fit.sample, fit.basis, point, normalized=True)
        except FloatingPointError as error:
            return _Unformed(immutable_array(np.asarray(point)), str(error))
        whole = WholePoint(fit.sample, stage, root)
        return fit.work.run(
            "root_certificate",
            lambda: certify_point(
                whole,
                fit.controls,
                absorbed=fit.absorbed,
                scale=fit.scale,
                merit=self.merit,
            ),
        )

    def accepted(self, certified: Certified) -> bool:
        """Whether the certificate passes at either level."""
        return isinstance(certified, FEPointCertificate) and certified.accepted

    def assessment(self, certified: Certified) -> Assessment:
        """The retry reading of the certificate; none of an unformed point."""
        if isinstance(certified, FEPointCertificate):
            return fe_assessment(certified)
        return Assessment(accepted=False, score=math.nan, error=math.nan)

    def status(self, certified: Certified, native: NativeReason) -> Status:
        """The status of the stage point (:func:`stage_status`)."""
        return stage_status(certified, native)

    def point(self, certified: Certified) -> FloatArray:
        """The point that the certificate assessed."""
        if isinstance(certified, FEPointCertificate):
            record = certified.point.record
            assert isinstance(record, RootPoint)
            return record.gamma
        return certified.gamma


def summary_of(certified: Certified) -> RootStageSummary:
    """The gates and quantities of a stage certificate, without its evaluation."""
    if isinstance(certified, _Unformed):
        nan = math.nan
        errors = MappingProxyType({"evaluation": certified.reason})
        return RootStageSummary(
            None,
            ("evaluation",),
            ("evaluation",),
            (),
            nan,
            nan,
            nan,
            False,
            False,
            "none",
            nan,
            errors,
        )
    measures, effects = certified.measures, certified.effects
    return RootStageSummary(
        certified.level,
        certified.strict,
        certified.secondary,
        certified.not_applicable,
        certified.scaled_score,
        certified.index_correction,
        math.nan if measures is None else measures.rcond,
        measures is not None and measures.positive_definite,
        measures is not None and measures.conditioned,
        certified.score_metric,
        math.nan if effects is None else effects.maximum,
        MappingProxyType({str(k): v for k, v in certified.errors.items()}),
    )


# ------------------------------------------------------------------- the start


def _convex_effects(audit: CandidateAudit) -> tuple[FloatArray, ...] | None:
    """The effects of a convex point as its certificate holds them, if formed.

    A fitted-index point's normalized recovery (the certificate's own, checked); a
    profiled point's profile state.  ``None`` where none was formed.
    """
    certificate = audit.certificate
    if certificate is None:
        return None
    record = certificate.point.record
    if isinstance(record, PointProfile):
        return tuple(record.state.coefficients)
    if isinstance(record, IndexPoint):
        inner = certificate.inner
        recovery = None if inner is None else inner.recovery
        if recovery is None or not recovery.complete:
            return None
        normalization = recovery.normalization
        return None if normalization is None else normalization.effects
    return None


def convex_start(
    sample: FESample, basis: CertificateBasis, audit: CandidateAudit
) -> tuple[RootPoint | None, float | None]:
    """The convex point in the full coordinates, and its index gap, if formed."""
    certificate = audit.certificate
    effects = _convex_effects(audit)
    if certificate is None or effects is None:
        return None, None
    try:
        alpha = coordinates(basis, effects)
        gamma = np.concatenate([certificate.point.theta, alpha])
        start = point_at(sample, basis, gamma)
        gap = checked_difference(start.index, certificate.point.q, "start gap")
    except FloatingPointError:
        return None, None
    return start, float(np.max(np.abs(gap))) if gap.size else 0.0


def _convex_status(audit: CandidateAudit) -> tuple[bool, str]:
    """Whether the convex audit accepted its point, and its FE status."""
    certificate, reason = audit.certificate, audit.history.reason
    if certificate is None or not certificate.accepted:
        return False, reason
    return True, "verified" if certificate.level == "strict" else "verified_secondary"


def _initializer(
    fit: _Fit, start: FloatArray | None, convex: ConvexRunner | None
) -> AbsorbedRootInitializer:
    """The caller's start, or the same-sample nearest convex fit (step 1)."""
    if start is not None:
        alpha = np.zeros(fit.basis.n_kept)
        point: RootPoint | None = None
        try:
            point = point_at(fit.sample, fit.basis, np.concatenate([start, alpha]))
        except FloatingPointError:
            point = None
        return AbsorbedRootInitializer(
            "user_start", None, False, None, None, None, None, point, None
        )
    if convex is None:
        raise ValueError("an automatic start needs the same-sample convex fit")
    kind, variance = nearest_convex(fit.variance)
    candidate, audit = convex(variance)
    if candidate.sample is not fit.sample or audit.history.sample is not fit.sample:
        raise ValueError("the convex initializer must solve the same sample")
    certified, status = _convex_status(audit)
    point, gap = convex_start(fit.sample, fit.basis, audit)
    return AbsorbedRootInitializer(
        kind,
        variance,
        certified,
        status,
        audit.history.reason,
        candidate,
        audit,
        point,
        gap,
    )


def merit_at_start(
    sample: FESample,
    root_options: RootOptions,
    initializer: AbsorbedRootInitializer,
    *,
    work: WorkLog,
) -> tuple[MeritFactor | None, str, str]:
    """The fixed merit at the start, or the reason and text of its absence.

    An uncertified automatic initializer, or one whose point has no representable
    effects, runs no root solve: its own operative reason.
    """
    point = initializer.point
    if initializer.kind != "user_start" and not initializer.certified:
        reason = initializer.reason or "native_stop"
        text = f"the nearest convex initializer was not certified ({reason})"
        return None, reason, text
    if point is None:
        what = (
            "the caller's start"
            if initializer.kind == "user_start"
            else "the certified convex point"
        )
        text = f"{what} has no representable full joint coordinates"
        return None, "numerical_breakdown", text
    metric = root_options.merit_metric
    try:
        factor = merit_factor(sample, point, metric, work=work)
    except np.linalg.LinAlgError as error:
        return None, "factorization_failure", f"merit {metric!r}: {error}"
    except (FloatingPointError, ValueError) as error:
        return None, "numerical_breakdown", f"merit {metric!r}: {error}"
    return factor, "", ""


# ------------------------------------------------------------------- the solve


def _check(
    sample: object,
    variance: object,
    options: object,
    root_options: object,
    absorbed: object,
    *,
    start: FloatArray | None,
) -> None:
    """The preconditions of :func:`solve_absorbed_root`, with built-in errors."""
    if type(sample) is not FESample or sample.grouping is None:
        raise TypeError("sample must be an FESample with fixed-effect blocks")
    if sample.omega == 0:
        raise ValueError("omega = 0 is the linear path, not a Newton solve")
    if type(variance) is not PowerVariance or not outside_convex(variance):
        raise ValueError(NOT_ROOT)
    if type(options) is not SolverOptions or type(root_options) is not RootOptions:
        raise TypeError("options must be SolverOptions and RootOptions")
    if type(absorbed) is not AbsorbedOptions or absorbed.absorbed not in (
        "index",
        "profiled",
    ):
        raise TypeError("absorbed must be the explicit options of an organization")
    if start is not None:
        values = np.asarray(start)
        if values.shape != (sample.active.size,) or not np.isfinite(values).all():
            raise ValueError("start must hold one finite value per slope")


def _stage(outcome: StageOutcome[Certified]) -> AbsorbedRootStage:
    """The record of one stage outcome, with the summary of its retained point."""
    variance = _stage_variance(outcome.variance)
    return AbsorbedRootStage(
        variance,
        outcome.attempts,
        outcome.retained,
        summary_of(outcome.certified),
        str(outcome.status),
        outcome.iterations,
        outcome.evaluations,
    )


def _root(
    fit: _Fit, initializer: AbsorbedRootInitializer, merit: MeritFactor
) -> tuple[tuple[StageOutcome[Certified], ...], int, bool]:
    """The stages, the target stage's index and whether continuation ran."""
    start = initializer.point
    assert start is not None
    options = fit.options
    certifier = _StageCertifier(fit, merit)
    kernel: RootKernel[_State] = RootKernel(
        _equations(fit),
        _direction(fit),
        _whitener(fit, merit),
        options,
        fit.root_options,
        RootBudget(max(0, options.max_evaluations - initializer.evaluations)),
        fit.work,
    )
    iterations = RootBudget(max(0, options.max_iter - initializer.iterations))
    run = RootRun(kernel, iterations, certifier, initializer.certified)
    direct = run_root_stage(run, fit.variance, start.gamma)
    _, origin = nearest_convex(fit.variance)
    stages, final, used = run_continuation(
        run, fit.variance, start.gamma, direct, origin=origin.tau
    )
    index = next(position for position, stage in enumerate(stages) if stage is final)
    return stages, index, used


def _work(fit: _Fit) -> AbsorbedRootWork:
    """The counted work of the solve so far."""
    return AbsorbedRootWork(
        fit.work.snapshot(), tuple(fit.directions), fit.merit_calls.actions()
    )


def _without_root(
    fit: _Fit,
    initializer: AbsorbedRootInitializer,
    start: FloatArray | None,
    absence: tuple[str, str],
) -> AbsorbedRootCandidate:
    """The candidate where no root stage opened (module docstring, step 4)."""
    reason, message = absence
    return AbsorbedRootCandidate(
        sample=fit.sample,
        variance=fit.variance,
        start=None if start is None else FloatStorage.store(np.asarray(start)),
        options=fit.options,
        root_options=fit.root_options,
        absorbed=fit.absorbed,
        initializer=initializer,
        merit=None,
        merit_reason=message,
        stages=(),
        target_stage=None,
        continuation_used=False,
        point=None,
        certificate=None,
        reason=reason,
        message=message,
        native=None,
        work=_work(fit),
        timing=fit.timing(),
    )


def solve_absorbed_root(
    sample: FESample,
    variance: PowerVariance,
    options: SolverOptions,
    root_options: RootOptions,
    absorbed: AbsorbedOptions,
    *,
    scale: FEScaleProvider,
    convex: ConvexRunner | None,
    start: FloatArray | None = None,
) -> AbsorbedRootCandidate:
    """Solve the absorbed joint equation outside ``[1, 2]`` (module docstring).

    Parameters
    ----------
    sample : FESample
        The authoritative target sample, with fixed-effect blocks.
    variance : PowerVariance
        The target working variance, ``tau`` outside ``[1, 2]``.
    options : SolverOptions
        The resolved options: the convex initializer's, the root solve's native
        tolerance, budgets, contraction, retry and ``rcond_min``, and the
        certificate's controls.
    root_options : RootOptions
        ``root_armijo``, ``root_max_backtracks``, ``root_max_iter``,
        ``merit_metric``, ``continuation`` and ``continuation_step``.
    absorbed : AbsorbedOptions
        The explicit options of the organization requested at the fit; its convex
        initializer runs that organization.
    scale : FEScaleProvider
        The fit's covariance provider of the coefficient gate.
    convex : callable or None
        The caller's same-sample convex fit of the requested organization at a
        convex variance, returning its candidate and its audit; required without
        ``start``.
    start : array of float, optional
        The caller's slopes in the solver's coordinates.

    Returns
    -------
    AbsorbedRootCandidate
        The candidate and its trace; no claim of uniqueness or existence.
    """
    _check(sample, variance, options, root_options, absorbed, start=start)
    clock = (time.perf_counter(), time.process_time())
    basis = owned_basis(sample)
    design = sample.design
    matrix = np.zeros((sample.count, 0)) if design is None else design.matrix
    fit = _Fit(
        sample,
        variance,
        options,
        root_options,
        absorbed,
        scale,
        basis,
        immutable_array(matrix),
        certificate_controls(options),
        WorkLog(),
        clock,
    )
    caller = None if start is None else immutable_array(np.asarray(start))
    initializer = _initializer(fit, caller, convex)
    merit, reason, text = merit_at_start(
        sample, root_options, initializer, work=fit.work
    )
    if merit is None:
        return _without_root(fit, initializer, caller, (reason, text))
    stages, index, used = _root(fit, initializer, merit)
    final = stages[index]
    native = final.attempts[final.retained].native
    certified = final.certified
    point: RootPoint | None = None
    certificate: FEPointCertificate | None = None
    message = (
        f"{OPERATION} from the {initializer.kind} start: stage {index} of "
        f"{len(stages)} on tau = {variance.tau!r}, native {native}"
    )
    if isinstance(certified, FEPointCertificate):
        record = certified.point.record
        assert isinstance(record, RootPoint)
        point, certificate = record, certified
    else:
        message = f"{message}; the point is not representable: {certified.reason}"
    return AbsorbedRootCandidate(
        sample=sample,
        variance=variance,
        start=None if caller is None else FloatStorage.store(caller),
        options=options,
        root_options=root_options,
        absorbed=absorbed,
        initializer=initializer,
        merit=merit,
        merit_reason="",
        stages=tuple(_stage(stage) for stage in stages),
        target_stage=index,
        continuation_used=used,
        point=point,
        certificate=certificate,
        reason=native,
        message=message,
        native=native,
        work=_work(fit),
        timing=fit.timing(),
    )
