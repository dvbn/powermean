"""Conditioning, scaled score and full correction of the absorbed certificate.

Internal module: the measures composed from the factors and counted actions of
:mod:`powermean.solve.absorbed_certificate_factor`; no dense inverse, truncation,
ridge, pseudoinverse, cutoff, incomplete factor or other solver.

Conditioning: ``k`` is the estimate of ``||A^-1||_1`` from the ordinary factor; it must
be finite and positive, and ``rcond = (1/k)/||A||_1`` strictly above ``rcond_min``.
Only then is the correction formed: ``delta = F(S/s)/s``, the change
``dq = X delta_X + E_J delta_J``, its root mean square over the observations (frequency
counts as replicas), the diagnostic ``B'(b - C dq)`` and, with slopes, ``W =
F^T(E_s/s)/s`` from ``K`` right-hand sides.  Curvature: ``J`` must keep its rank on the
positive-curvature rows (else refused before the call); ``Pr Aplus Pc = L U`` with
``Pr[perm_r[j], j] = 1`` succeeds only with ``perm_r == perm_c`` (an off-diagonal
threshold-zero pivot is no success), finite ``L`` and ``U`` (every stored entry, not
only the pivots) and every ``d = diag(U) > 0``; the score is
``||L^-1 Pr(S/s) / sqrt(d)||``, ``Pr`` the scatter ``out[perm_r] = v``: an LDL-style
numerical test, not Cholesky parity or an inertia proof.

An eligible categorical block may be eliminated analytically before either
factorization (:mod:`powermean.solve.absorbed_certificate_block`). Its implicit
FULL factors retain the finite/pivot test and complete whitened score. Original
full-matrix norms, inverse actions and inference maps are unchanged; the
block ordering can change numerical boundary outcomes. No factor is retained
across points, and ordinary and curvature systems remain separate.

Selected roots (the metric is the symmetrized observed negative Jacobian when positive
definite, else the fixed merit metric).  Given the
fixed :class:`~powermean.solve.absorbed_root_records.MeritFactor` of a root solve and
a curvature that is not positive definite (or refused), the score is
``||D_M^-1/2 L_M^-1 Pr_M (S/s_M)||``, whose square is ``S' M^-1 S`` for the
factorized symmetric ``M`` (:func:`merit_whitened`): the same merit as the solve's
Armijo, not ``||M^-1 S||``; its triangular solve is counted.  No curvature error is
then recorded, the curvature factor's own failure stays in its record and
``positive_definite`` stays false; ``score_metric`` names the metric.  Without a
merit nothing changes.  Conditioning, the correction and its diagnostic are the
same at every point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from powermean.fe.absorb_numeric import checked_difference, checked_product
from powermean.fe.certificate_basis import embed, lift
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_block import (
    Block,
    BlockFactor,
    block_factorized,
    elimination_block,
)
from powermean.solve.absorbed_certificate_factor import (
    Calls,
    FactorRecord,
    SuperLU,
    factor_record,
    factor_side,
    factor_step,
    failed_factor,
    inverse_norm,
    stored_lower,
)
from powermean.solve.absorbed_certificate_geometry import (
    FEGeometry,
    SparseMatrix,
    moment,
)
from powermean.solve.absorbed_certificate_point import WorkStep
from powermean.solve.compact_matrix import CompactMatrix
from powermean.solve.absorbed_root_records import MeritFactor, ScoreMetric
from powermean.solve.quantities import euclidean_norm, root_mean_square

SUPPORT = "the positive-curvature rows lose the rank of the nuisance coordinates"
NOT_FINITE = "the curvature factor holds a value that is not finite"
# delta, dq (None when not representable), its root mean square, B'(b - C dq)
Correction = tuple[FloatArray | None, FloatArray | None, float, FloatArray | None]


@dataclass(frozen=True, eq=False)
class SparseMeasures:
    """The factors, the conditioning, the scaled score and the correction.

    ``estimate`` is ``k``; ``correction`` ``delta``, ``change`` ``dq``, ``residual``
    ``B'(b - C dq)`` and ``slope_map`` ``W`` are ``None`` when not formed, NaN the
    scalars; ``actions`` counts the N and T solves, the triangular solves and their
    right-hand sides; ``errors`` names the conditioning and curvature failures;
    ``score_metric`` the metric of the scaled score (module docstring).
    """

    ordinary: FactorRecord
    curvature: FactorRecord
    estimate: float
    rcond: float
    conditioned: bool
    positive_definite: bool
    scaled_score: float
    correction: FloatArray | None
    change: FloatArray | None
    index_correction: float
    residual: FloatArray | None
    slope_map: FloatArray | None
    actions: Mapping[str, int]
    errors: Mapping[str, str]
    score_metric: ScoreMetric = "current_symmetric_hessian"

    def steps(self) -> tuple[WorkStep, ...]:
        """The work: each factor attempt with its nonzeros, then the counted solves."""
        solves = WorkStep("solves", "completed", "", tuple(self.actions.items()))
        return (factor_step(self.ordinary), factor_step(self.curvature), solves)


def row_permutation(perm_r: IndexArray, values: FloatArray) -> FloatArray:
    """``Pr v`` with ``Pr[perm_r[j], j] = 1``: the scatter ``out[perm_r] = v``."""
    found = np.empty_like(values)
    found[perm_r] = values
    return found


# --------------------------------------------------------------- the measures


def conditioning(
    factor: SuperLU, norm: float, size: int, rcond_min: float, calls: Calls
) -> tuple[float, float, str]:
    """``k``, ``rcond = (1/k)/||A||_1`` and the failure reason (empty when passed)."""
    estimate = inverse_norm(factor, calls, size)
    if not (math.isfinite(estimate) and estimate > 0):
        return (
            estimate,
            math.nan,
            "the inverse-norm estimate is not finite and positive",
        )
    rcond = (1.0 / estimate) / norm
    if not rcond > rcond_min:
        return estimate, rcond, f"rcond {rcond:.3e} <= rcond_min"
    return estimate, rcond, ""


def whitened_score(
    matrix: SparseMatrix | CompactMatrix, scaled: FloatArray, calls: Calls, block: Block | None = None
) -> tuple[FactorRecord, float]:
    """The curvature factor of ``matrix`` and ``||L^-1 Pr scaled / sqrt(d)||``.

    The score is NaN unless the factor holds ``perm_r == perm_c``, finite ``L`` and
    ``U`` data and every ``d > 0``.
    """
    record, factor = block_factorized(matrix, "curvature", block)
    if isinstance(factor, BlockFactor):
        reason = factor.curvature_reason()
        if reason:
            return failed_factor(record, reason), math.nan
        with np.errstate(all="ignore"):
            return record, euclidean_norm(factor.whitened(scaled, calls))
    perm_r, perm_c, d = record.perm_r, record.perm_c, record.diagonal
    if factor is None or perm_r is None or perm_c is None or d is None:
        return record, math.nan
    lower, upper = factor_side(factor, "L"), factor_side(factor, "U")
    finite = np.isfinite(lower.data).all() and np.isfinite(upper.data).all()
    if not np.array_equal(perm_r, perm_c):
        reason = "a threshold-zero off-diagonal pivot (perm_r != perm_c)"
    elif not (finite and np.isfinite(d).all()):
        reason = NOT_FINITE
    elif not (d > 0).all():
        reason = "the curvature is not positive definite: a nonpositive pivot"
    else:
        with np.errstate(all="ignore"):
            permuted = row_permutation(perm_r, scaled)
            whitened = calls.triangular(lower, permuted) / np.sqrt(d)
        return record, euclidean_norm(whitened)
    return failed_factor(record, reason), math.nan


def merit_whitened(merit: MeritFactor, values: FloatArray, calls: Calls) -> FloatArray:
    """``u = D^-1/2 L^-1 Pr (values/s)`` by the fixed merit factor, counted.

    ``||u||^2 = values' M^-1 values`` for the factorized symmetric ``M``;
    ``FloatingPointError`` where ``u`` is not representable, ``ValueError`` for a
    vector of another size.
    """
    size = merit.dimension
    if values.shape != (size,):
        raise ValueError("the merit metric belongs to another joint dimension")
    lower = stored_lower(merit.l_indptr, merit.l_indices, merit.l_data, size)
    with np.errstate(all="ignore"):
        permuted = row_permutation(merit.perm_r, values / merit.scale)
        found: FloatArray = calls.triangular(lower, permuted) / np.sqrt(merit.diagonal)
    if not np.isfinite(found).all():
        raise FloatingPointError("the whitened merit residual is not representable")
    return found


def merit_score(moment_: FloatArray, merit: MeritFactor, calls: Calls) -> float:
    """``||D_M^-1/2 L_M^-1 Pr_M (S/s_M)||``, NaN where not representable."""
    try:
        return euclidean_norm(merit_whitened(merit, moment_, calls))
    except FloatingPointError:
        return math.nan


def curvature_score(
    geometry: FEGeometry, calls: Calls, block: Block | None = None
) -> tuple[FactorRecord, float]:
    """The curvature factor and the scaled score of ``S/s`` (NaN unless formed).

    A support rank loss refuses before a scaling refusal: it names the cause.
    """
    matrix, scale = geometry.curvature, geometry.scale
    if not geometry.support.full:
        return factor_record("curvature", "refused", SUPPORT), math.nan
    if matrix is None or scale is None:
        return factor_record("curvature", "refused", geometry.refusal), math.nan
    with np.errstate(all="ignore"):
        scaled = geometry.moment / scale
    return whitened_score(matrix, scaled, calls, block)


# ----------------------------------------------------------------- correction


def _change(geometry: FEGeometry, delta: FloatArray) -> FloatArray | None:
    """``dq = X delta_X + E_J delta_J``; ``None`` when not representable."""
    k, basis = geometry.n_slopes, geometry.basis
    try:
        nuisance = lift(basis, embed(basis, delta[k:]))
    except FloatingPointError:
        return None
    with np.errstate(all="ignore"):
        found: FloatArray = geometry.design @ delta[:k] + nuisance
    return immutable_array(found)


def _residual(geometry: FEGeometry, change: FloatArray) -> FloatArray | None:
    """The diagnostic ``B'(b - C dq)``, ``None`` when not representable."""
    rows = geometry.rows
    try:
        fitted = checked_product(rows.curvature, change, "fitted change")
        values = checked_difference(rows.b, fitted, "Newton residual")
        return moment(geometry.design, values, geometry.basis)
    except FloatingPointError:
        return None


def slope_map(factor: SuperLU, scale: FloatArray, k: int, calls: Calls) -> FloatArray:
    """``W = F^T(E_s/s)/s``, the first ``k`` columns of ``H^-T``, by one ``T`` solve."""
    rhs = np.zeros((scale.size, k))
    rhs[np.arange(k), np.arange(k)] = 1.0 / scale[:k]
    with np.errstate(all="ignore"):
        found: FloatArray = calls.solve(factor, rhs, "T") / scale[:, None]
    return immutable_array(found)


def _corrected(
    geometry: FEGeometry, factor: SuperLU, scale: FloatArray, calls: Calls
) -> Correction:
    """``delta``, ``dq``, its root mean square over the observations, the residual."""
    with np.errstate(all="ignore"):
        delta = calls.solve(factor, geometry.moment / scale, "N") / scale
    change = _change(geometry, delta)
    if change is None:
        return immutable_array(delta), None, math.nan, None
    rows = geometry.rows
    multiplicity = rows.weights if rows.frequency else np.ones(change.shape)
    index = root_mean_square(change, multiplicity)
    return immutable_array(delta), change, index, _residual(geometry, change)


def _ordinary(
    geometry: FEGeometry, rcond_min: float, calls: Calls, block: Block | None = None,
    prepared: tuple[FactorRecord, SuperLU] | None = None,
) -> tuple[FactorRecord, tuple[float, float, str], Correction, FloatArray | None]:
    """The ordinary factor, its conditioning and, only when conditioned, ``delta``."""
    nothing: Correction = (None, None, math.nan, None)
    if geometry.ordinary is None or geometry.scale is None:
        record = factor_record("ordinary", "refused", geometry.refusal)
        return record, (math.nan, math.nan, geometry.refusal), nothing, None
    record, factor = (
        block_factorized(geometry.ordinary, "ordinary", block)
        if prepared is None else prepared
    )
    if factor is None:
        return record, (math.nan, math.nan, record.reason), nothing, None
    size, k, scale = geometry.dimension, geometry.n_slopes, geometry.scale
    measured = conditioning(factor, geometry.norm, size, rcond_min, calls)
    if measured[2]:
        return record, measured, nothing, None
    found = _corrected(geometry, factor, scale, calls)
    return record, measured, found, slope_map(factor, scale, k, calls) if k else None


def sparse_measures(
    geometry: FEGeometry, rcond_min: float, merit: MeritFactor | None = None,
    *, ordinary: tuple[FactorRecord, SuperLU] | None = None,
) -> SparseMeasures:
    """Both factors, the conditioning, the scaled score and the correction (docstring).

    A scaling refusal of the geometry refuses both factors before any call.  With
    ``merit`` (a selected root's fixed metric) a curvature that is not positive
    definite gives the fixed-merit score instead of a curvature error.
    """
    calls, errors = Calls(), {}
    block = elimination_block(geometry)
    curvature, score = curvature_score(geometry, calls, block)
    definite = curvature.status == "factored" and math.isfinite(score)
    metric: ScoreMetric = "current_symmetric_hessian"
    if not definite:
        if merit is not None:
            score, metric = merit_score(geometry.moment, merit, calls), "fixed_merit"
        else:
            metric = "none"
            errors["curvature"] = (
                curvature.reason or "the curvature is not positive definite"
            )
    ordinary_record, measured, found, mapped = _ordinary(
        geometry, rcond_min, calls, block, ordinary
    )
    estimate, rcond, reason = measured
    if reason:
        errors["conditioning"] = reason
    return SparseMeasures(
        ordinary_record,
        curvature,
        estimate,
        rcond,
        not reason,
        definite,
        score,
        *found,
        mapped,
        calls.actions(),
        MappingProxyType(errors),
        metric,
    )
