"""The absorbed linear path: weighted least squares of ``log Y`` with absorbed effects.

At ``omega = 0``, log OLS is least squares of ``log Y`` on the regressors with the fixed
effects absorbed and no working variance.  There is no fallback, a solver proposes,
and near-collinearity is not collinearity; the linear path is exact up to the
absorption tolerance.  Internal module: it forms no route, certificate, covariance or
inference.

The problem.  For the admitted sample at ``omega = 0`` (route
``linear``, at least one fixed-effect block), with ``l = log Y``, the RAW offset ``o``,
the raw weights ``a`` of the sample's stated kind (one without weights; every kind
enters the same objective), the original active columns ``Xraw`` and all blocks ``F``,

    minimize  1/2 sum_i a_i (l_i - o_i - x_i' beta - f_i' alpha)^2,

whose equations are ``Xraw'(a e) = 0`` and ``F'(a e) = 0`` at ``e = l - q``.  Nothing is
exponentiated, no working variance enters and nothing is divided by an order; a zero
log response is an ordinary value.  The signed response arithmetic stays internal.

The solve, in the stages of :data:`~powermean.solve.linear_absorbed_state.STAGES`:

1. ``preparation``: :func:`~powermean.fe.absorb_numeric.prepare_weights` of the
   sample's final grouping and raw weights (its constructor revalidates every total).
2. ``joint``: one cold absorption of ``[t, Xs]``, ``t = l - o``, with the explicit
   ``absorb_tol_final``, ``absorb_max_sweeps`` and ``absorb_acceleration`` and the
   positional columns ``("joint 0", ..., "joint K")``.  Its start is the values
   themselves, so the absorber's start refusal cannot occur here.
3. ``weighting``, ``rank``, ``factorization`` (with slopes only): ``W = sqrt(a)
   Xtilde`` and ``r = sqrt(a) ttilde`` (checked); the singular values of ``W``
   (NumPy, ``compute_uv=False``) and its numerical rank, the values strictly above
   ``eps max(n, K) s_max`` with the physical rows ``n``, never a frequency total, the
   rule of :mod:`powermean.solve.linear` and of the collinearity analysis; a rank
   below ``K`` refuses, with no basic subset, minimum-norm solution,
   ridge or omission; then the pivoted economic QR ``W[:, P] = Q R`` (SciPy), ``Q'r``
   and the back substitution ``R z = Q'r`` (SciPy), ``theta[P] = z``.  No Gram matrix
   is formed.  Each attempt is recorded and its outputs checked before use, as
   :mod:`powermean.solve.linear_absorbed_numeric` states.
4. ``seed``: ``alpha_b = a_t,b - a_X,b theta`` from the joint state, in a NEW
   single-column :class:`~powermean.fe.absorb.AbsorptionState` bound to the grouping
   and the column ``"final 0"``.
5. ``residual``, ``final``: the declared second, fixed-slope absorption of ``t - Xs
   theta`` from that seed with the same three controls and its own call budget.  Its
   failure is the solve's failure: no within residual stands in for it, no third
   absorption, retry or slope iteration follows.  Without slopes both absorptions run
   on ``t`` alone, with no decomposition, intercept or order scale.
6. ``point``, ``normalization``, ``equations``: the full point (``beta`` by the
   sample's order scale, once; the authoritative ``q``), the accepted normalization
   with its gap at most ``absorb_tol_final`` (equality passes), and the fresh original
   equations, all in :mod:`powermean.solve.linear_absorbed_point`.

Failures.  A value that is not representable ends the solve with status ``range`` at
its stage; an absorption that does not converge, ``absorption`` or
``final_absorption``; a failed or unusable decomposition, ``factorization``; a rank
below ``K``, ``rank_deficient``; a gap above the tolerance, ``normalization``.  Every
record formed before is kept, and the point once formed; none of these outcomes is
evidence that no solution exists, and none is retried.  Only ``FloatingPointError``
and ``numpy.linalg.LinAlgError`` are caught, where named above.  The candidate
(:mod:`powermean.solve.linear_absorbed_state`) admits the records as they formed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.fe.absorb import AbsorptionResult, AbsorptionState, absorb
from powermean.fe.absorb_numeric import (
    PreparedWeights,
    checked_product,
    prepare_weights,
)
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray
from powermean.solve.linear_absorbed_numeric import (
    FINAL_COLUMNS,
    Decomposition,
    back_substitution,
    direct_svd,
    fixed_slope_residual,
    full_rank,
    joint_columns,
    joint_values,
    pivoted_qr,
    raw_weights,
    seeds,
    signed_response,
    slopes,
)
from powermean.solve.linear_absorbed_point import (
    FreshEquations,
    LinearControls,
    LinearPoint,
    PointNormalization,
    blocks_of,
    linear_sample,
)
from powermean.solve.linear_absorbed_state import (
    AbsorbedLinearCandidate,
    Stage,
    Status,
)

PROJECTED = "the projected response Q'r is not representable in float64"


# --------------------------------------------------------------------- the solve


class _Stop(NamedTuple):
    """Where and how the solve ended, and why."""

    stage: Stage
    status: Status
    reason: str


class _Joint(NamedTuple):
    """The weight preparation and the converged joint absorption."""

    prepared: PreparedWeights
    result: AbsorptionResult


class _Run:
    """The records of one solve as they form, then its candidate."""

    def __init__(self, sample: FESample, controls: LinearControls) -> None:
        self.sample = sample
        self.controls = controls
        self.t = signed_response(sample)
        self.n_slopes = int(sample.active.size)
        self.prepared: PreparedWeights | None = None
        self.joint: AbsorptionResult | None = None
        self.decompositions: list[Decomposition] = []
        self.seed: AbsorptionState | None = None
        self.final: AbsorptionResult | None = None
        self.point: LinearPoint | None = None
        self.normalized: PointNormalization | None = None
        self.equations: FreshEquations | None = None

    def absorbed(
        self,
        values: FloatArray,
        prepared: PreparedWeights,
        columns: tuple[str, ...],
        state: AbsorptionState | None = None,
    ) -> AbsorptionResult:
        """One absorption with the three explicit controls, cold or from ``state``."""
        controls = self.controls
        return absorb(
            values,
            prepared,
            columns,
            tolerance=controls.absorb_tol_final,
            max_sweeps=controls.absorb_max_sweeps,
            acceleration=controls.absorb_acceleration,
            state=state,
        )

    def candidate(self, stop: _Stop) -> AbsorbedLinearCandidate:
        """The candidate of the records formed, ended by ``stop``."""
        return AbsorbedLinearCandidate(
            self.sample,
            self.controls,
            stop.stage,
            stop.status,
            stop.reason,
            prepared=self.prepared,
            joint=self.joint,
            decompositions=tuple(self.decompositions),
            seed=self.seed,
            final=self.final,
            point=self.point,
            normalized=self.normalized,
            equations=self.equations,
        )


def _joint(run: _Run) -> _Stop | _Joint:
    """The weight preparation and the cold joint absorption of ``[t, Xs]``."""
    try:
        prepared = prepare_weights(blocks_of(run.sample), raw_weights(run.sample))
    except FloatingPointError as error:
        return _Stop("preparation", "range", f"weight preparation: {error}")
    run.prepared = prepared
    values = joint_values(run.sample, run.t)
    result = run.absorbed(values, prepared, joint_columns(run.n_slopes))
    run.joint = result
    if not result.converged:
        reason = f"joint absorption: {result.status}: {result.reason}"
        return _Stop("joint", "absorption", reason)
    return _Joint(prepared, result)


def _slopes(run: _Run, joint: _Joint) -> _Stop | FloatArray:
    """``theta``: the weighted within design, its rank, then its QR solve."""
    if not run.n_slopes:
        return immutable_array(np.zeros(0))
    within = joint.result.residual
    root = np.sqrt(joint.prepared.weights)
    try:
        design = checked_product(root[:, None], within[:, 1:], "weighted design")
        response = checked_product(root, within[:, 0], "weighted response")
    except FloatingPointError as error:
        return _Stop("weighting", "range", str(error))
    record, message = direct_svd(design)
    run.decompositions.append(record)
    if record.rank is None:
        return _Stop("rank", "factorization", f"direct_svd: {message}")
    if not full_rank(record, run.n_slopes):
        reason = (
            f"numerical rank {record.rank} of {run.n_slopes} at the cutoff "
            f"{record.cutoff:.3e}"
        )
        return _Stop("rank", "rank_deficient", reason)
    return _solved(run, design, response)


def _solved(run: _Run, design: FloatArray, response: FloatArray) -> _Stop | FloatArray:
    """The pivoted QR, ``Q'r`` and the back substitution of a full-rank ``W``."""
    qr, factors, message = pivoted_qr(design)
    run.decompositions.append(qr)
    if factors is None:
        return _Stop("factorization", "factorization", f"pivoted_qr: {message}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        projected: FloatArray = factors.q.T @ response
    if not np.isfinite(projected).all():
        return _Stop("factorization", "range", PROJECTED)
    triangular, message = back_substitution(factors, projected)
    run.decompositions.append(triangular)
    if triangular.outcome != "ok":
        return _Stop("factorization", "factorization", f"triangular: {message}")
    return slopes(qr, triangular)


def _final(run: _Run, joint: _Joint, theta: FloatArray) -> _Stop | AbsorptionResult:
    """The seed state, the fixed-slope residual and the final absorption from it."""
    grouping = blocks_of(run.sample)
    try:
        alpha = seeds(joint.result.state.coefficients, theta)
    except FloatingPointError as error:
        return _Stop("seed", "range", str(error))
    seed = AbsorptionState(grouping, FINAL_COLUMNS, alpha)
    run.seed = seed
    try:
        u = fixed_slope_residual(run.sample, run.t, theta)
    except FloatingPointError as error:
        return _Stop("residual", "range", str(error))
    try:
        final = run.absorbed(u[:, None], joint.prepared, FINAL_COLUMNS, seed)
    except FloatingPointError as error:
        return _Stop("final", "range", f"final absorption: {error}")
    run.final = final
    if not final.converged:
        reason = f"final absorption: {final.status}: {final.reason}"
        return _Stop("final", "final_absorption", reason)
    return final


def _point(run: _Run, theta: FloatArray, final: AbsorptionResult) -> _Stop | None:
    """The full point, its normalization within the tolerance, its fresh equations."""
    effects = tuple(a[:, 0] for a in final.state.coefficients)
    try:
        point = LinearPoint(run.sample, theta, effects)
    except FloatingPointError as error:
        return _Stop("point", "range", str(error))
    run.point = point
    try:
        normalized = PointNormalization(point)
    except FloatingPointError as error:
        return _Stop("normalization", "range", str(error))
    run.normalized = normalized
    tolerance = run.controls.absorb_tol_final
    if not normalized.within(tolerance):
        reason = (
            f"the normalized effects rebuild the index within {normalized.gap:.3e}, "
            f"above the final tolerance {tolerance:.3e}"
        )
        return _Stop("normalization", "normalization", reason)
    try:
        run.equations = FreshEquations(point)
    except FloatingPointError as error:
        return _Stop("equations", "range", str(error))
    return None


def solve(
    sample: FESample,
    *,
    absorb_tol_final: float,
    absorb_max_sweeps: int,
    absorb_acceleration: bool,
) -> AbsorbedLinearCandidate:
    """Weighted least squares of ``log Y`` with every block absorbed: a candidate.

    Parameters
    ----------
    sample : FESample
        The admitted final sample at ``omega = 0``, with at least one
        fixed-effect block; read, never modified or prepared again.
    absorb_tol_final : float
        The absolute weighted-mean tolerance of both absorptions and the bound of the
        normalization gap; positive and finite, equality passes.
    absorb_max_sweeps : int
        The sweep budget of each absorption call, a positive integer, 1 included.
    absorb_acceleration : bool
        The accepted acceleration of both absorptions.

    Returns
    -------
    AbsorbedLinearCandidate
        Where and how the solve ended, every record formed and, when it formed, the
        full point with its normalization and fresh equations; never a certificate.

    Raises
    ------
    TypeError
        ``sample`` is not an ``FESample``.
    ValueError
        Another mean order or route, no fixed-effect block, or a control outside its
        domain.
    """
    found = linear_sample(sample)
    controls = LinearControls(absorb_tol_final, absorb_max_sweeps, absorb_acceleration)
    run = _Run(found, controls)
    joint = _joint(run)
    if isinstance(joint, _Stop):
        return run.candidate(joint)
    theta = _slopes(run, joint)
    if isinstance(theta, _Stop):
        return run.candidate(theta)
    final = _final(run, joint, theta)
    if isinstance(final, _Stop):
        return run.candidate(final)
    stop = _point(run, theta, final)
    return run.candidate(stop or _Stop("complete", "complete", ""))
