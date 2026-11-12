"""General identified geometry preparation.

B = M_C[X,F_R], G = B'B, T = diag(S_X,Q_R), P = (T'GT)^(-1)T'G.
At a valid rank, BT has full column rank, PT = I and BTP = B; T complements
ker(B), generally oblique in coefficient space. Slopes are normalized by
setting omitted original slope coefficients to zero; see slope_alias_matrix.

Internal numerical preparation only. Gram truncation does not certify structural
identification. Use geometry_checks.check_geometry for fresh raw binding and
original-space checks; general structural admission remains outside this layer.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_errors import GeometryRankError
from powermean.fe.geometry_gram import (
    _checked_controls,
    geometry_gram,
    spectral_passes,
)
from powermean.fe.geometry_rank import (
    _block_diagnostics,
    _build_retained_basis,
    _check_scalar_gap,
    _identify_slopes,
    _normalize_column_sign,
    _spectral_rank,
)
from powermean.fe.geometry_storage import BoolStorage, _build_record
from powermean.fe.retained import CenteredDesign


def numerical_geometry(
    design: CenteredDesign,
    *,
    gram_rtol: float,
    gap_min: float,
    max_bytes: int,
    nuisance_basis: str = "gram_eigh",
    custom_nuisance_basis: object = None,
) -> NumericalGeometry:
    """Construct a slope-preserving complement to the numerical nullspace.

    Parameters
    ----------
    design : CenteredDesign
        Owned B = M_C[X,F_R]; M_C is ordinary unweighted group centering.
    gram_rtol, gap_min : float
        Explicit relative Gram cutoff and separation/conditioning controls.
    max_bytes : int
        Task-local construction reservation; no package default is chosen.
    nuisance_basis : str
        Explicit Gram eigenbasis, projector QR, or caller-supplied custom basis.
    custom_nuisance_basis : ndarray or None
        Real finite L-by-rank(M_C F_R) array, used only for a custom basis.

    Returns
    -------
    NumericalGeometry
        Owned maps and diagnostics. Slopes use omitted coefficients = 0.
        slope_alias_matrix maps original slopes to those normalized slopes;
        slope_identified_mask flags zero numerical-nullspace rows under rtol.
        Rank/annihilation diagnostics are numerical at the supplied controls,
        not proofs of exact structural dependence for arbitrarily scaled data.
    """
    with np.errstate(over="raise", invalid="raise"):
        return _numerical_geometry(
            design,
            gram_rtol=gram_rtol,
            gap_min=gap_min,
            max_bytes=max_bytes,
            nuisance_basis=nuisance_basis,
            custom_nuisance_basis=custom_nuisance_basis,
        )


def _numerical_geometry(
    design: CenteredDesign,
    *,
    gram_rtol: float,
    gap_min: float,
    max_bytes: int,
    nuisance_basis: str,
    custom_nuisance_basis: object,
) -> NumericalGeometry:
    rtol, gmin, limit = _checked_controls(gram_rtol, gap_min, max_bytes)
    gram, budget, passes = geometry_gram(design, max_bytes=limit)
    k, p = design.n_slopes, design.n_columns
    evals, evecs, zero_full, thresh_full, gap_full = _spectral_rank(
        gram,
        rtol,
        gmin,
        "full_gram",
        exact_zero_columns=passes.exact_zero_columns,
    )
    full_top = max(float(evals.max(initial=0.0)), 0.0)
    rank_full = int(np.sum(~zero_full))
    null_basis = evecs[:, zero_full].copy()
    for col in range(null_basis.shape[1]):
        null_basis[:, col] = _normalize_column_sign(null_basis[:, col])

    g_rr = gram[k:, k:]
    evals_r, evecs_r, zero_r, _, gap_r = _spectral_rank(
        g_rr,
        rtol,
        gmin,
        "retained_gram",
        exact_zero_columns=tuple(j - k for j in passes.exact_zero_columns if j >= k),
    )
    kept_r_idx = np.flatnonzero(~zero_r)[::-1]
    v_kept = evecs_r[:, kept_r_idx]
    evals_r_kept = evals_r[kept_r_idx]
    r_top = max(float(evals_r.max(initial=0.0)), 0.0)
    q_r = _build_retained_basis(
        g_rr,
        v_kept,
        r_top=r_top,
        gram_rtol=rtol,
        gap_min=gmin,
        nuisance_basis=nuisance_basis,
        custom_nuisance_basis=custom_nuisance_basis,
    )

    kept_slopes, omitted_slopes, reasons = _identify_slopes(
        gram[:k, :k],
        gram[:k, k:],
        v_kept,
        evals_r_kept,
        full_top=full_top,
        gram_rtol=rtol,
        gap_min=gmin,
    )
    d_x, d_r = int(kept_slopes.size), int(q_r.shape[1])
    d = d_x + d_r
    if d != rank_full:
        raise GeometryRankError(
            "schur_rank_consistency",
            0.0,
            gmin,
            gram_rtol=rtol,
            threshold=thresh_full,
            smallest_retained=float(min(d, rank_full)),
            largest_discarded=float(max(d, rank_full)),
        )

    basis = np.zeros((p, d), dtype=np.float64)
    for out_col, slope_col in enumerate(kept_slopes):
        basis[int(slope_col), out_col] = 1.0
    if d_r > 0:
        basis[k:, d_x:] = q_r

    id_gram = basis.T @ gram @ basis
    id_gram = 0.5 * (id_gram + id_gram.T)
    if d > 0:
        evals_id, _, zero_id, _, _ = _spectral_rank(
            id_gram, rtol, gmin, "identified_gram", floor_scale=full_top
        )
        if np.any(zero_id):
            raise GeometryRankError(
                "identified_gram_rank",
                0.0,
                gmin,
                gram_rtol=rtol,
                threshold=thresh_full,
                smallest_retained=0.0,
                largest_discarded=full_top,
            )
        _check_scalar_gap(
            float(evals_id.min()),
            max(float(evals_id.max()), full_top),
            rtol,
            gmin,
            "identified_gram_conditioning",
        )
        reduction = np.linalg.solve(id_gram, basis.T @ gram)
    else:
        reduction = np.zeros((0, p), dtype=np.float64)

    block_ranks, block_defs, ann_blocks, ann_levels = _block_diagnostics(
        design, g_rr, r_top, rtol, gmin
    )
    cross_redundancy = sum(block_ranks) - d_r
    if cross_redundancy < 0:
        raise GeometryRankError(
            "block_rank_consistency",
            0.0,
            gmin,
            gram_rtol=rtol,
            threshold=rtol * r_top,
            smallest_retained=float(sum(block_ranks)),
            largest_discarded=float(d_r),
        )
    col_map = np.concatenate([kept_slopes, np.full(d_r, -1, dtype=np.intp)]).astype(
        np.intp
    )
    slope_alias = reduction[:d_x, :k].copy()
    slope_identified = np.sum(null_basis[:k] ** 2, axis=1) <= rtol
    arrays = (
        gram,
        evals,
        evals_r,
        null_basis,
        basis,
        reduction,
        id_gram,
        kept_slopes,
        omitted_slopes,
        col_map,
        slope_alias,
        slope_identified,
        ann_blocks,
        *ann_levels,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("geometry map or diagnostics are not finite")
    retained_bytes = int(sum(arr.nbytes for arr in arrays))
    passes = spectral_passes(passes, k)

    return cast(
        NumericalGeometry,
        _build_record(
            NumericalGeometry,
            design=design,
            _gram=FloatStorage.store(gram),
            _eigenvalues=FloatStorage.store(evals),
            _retained_eigenvalues=FloatStorage.store(evals_r),
            _null_basis=FloatStorage.store(null_basis),
            _basis_matrix=FloatStorage.store(basis),
            _reduction_matrix=FloatStorage.store(reduction),
            _identified_gram=FloatStorage.store(id_gram),
            _kept_slope_indices=IndexStorage.store(kept_slopes),
            _omitted_slope_indices=IndexStorage.store(omitted_slopes),
            slope_omission_reasons=reasons,
            _original_column_map=IndexStorage.store(col_map),
            _slope_alias_matrix=FloatStorage.store(slope_alias),
            _slope_identified_mask=BoolStorage.store(slope_identified),
            block_ranks=block_ranks,
            block_deficiencies=block_defs,
            _annihilated_blocks=BoolStorage.store(ann_blocks),
            _annihilated_levels=tuple(
                BoolStorage.store(values) for values in ann_levels
            ),
            cross_block_redundancy=cross_redundancy,
            gram_rtol=rtol,
            gap_min=gmin,
            threshold=float(thresh_full),
            gap_ratio=float(gap_full),
            retained_gap_ratio=float(gap_r),
            nuisance_basis_kind=nuisance_basis,
            budget=budget,
            passes=passes,
            max_bytes=limit,
            retained_bytes=retained_bytes,
        ),
    )
