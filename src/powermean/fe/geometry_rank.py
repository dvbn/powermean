"""Spectral rank, slope normalization and block diagnostics.

Rank: lambda <= gram_rtol * spectral scale is discarded. The retained/discarded
spectral gap must reach gap_min. The existing scalar Schur/conditioning guard
also requires a nonzero residual / cutoff >= gap_min; it is stricter than rank
alone, and may refuse a full-rank but poorly conditioned coordinate system.
All controls are explicit caller inputs. No ridge or statistical policy is added.
"""

from __future__ import annotations

import numpy as np

from powermean.fe._cancel_storage import IndexArray
from powermean.fe.geometry_eigh import witnessed_eigh
from powermean.fe.geometry_errors import GeometryRankError
from powermean.fe.geometry_storage import BoolArray
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray

_VALID_BASIS_KINDS = ("gram_eigh", "gram_qr", "custom")
_FLOAT64_MAX = float(np.finfo(np.float64).max)


def _normalize_column_sign(vector: FloatArray) -> FloatArray:
    peak = float(np.abs(vector).max(initial=0.0))
    if peak == 0.0:
        return vector.copy()
    first = int(np.argmax(np.abs(vector)))
    return (-vector).copy() if vector[first] < 0.0 else vector.copy()


def _gap_ratio(smallest_retained: float, largest_discarded: float) -> float:
    """Compute ``smallest_retained / largest_discarded`` without scalar overflow.

    Returns ``inf`` when ``largest_discarded == 0`` or when the exact positive
    ratio exceeds the finite float64 range; never substitutes ``tiny`` for zero.
    """
    small = float(smallest_retained)
    large = float(largest_discarded)
    if not (np.isfinite(small) and np.isfinite(large)) or small <= 0.0 or large < 0.0:
        raise FloatingPointError(
            "gap ratio requires positive finite retained and "
            "nonnegative finite discarded values"
        )
    if large == 0.0:
        return float("inf")
    numerator, numerator_den = small.as_integer_ratio()
    denominator, denominator_den = large.as_integer_ratio()
    maximum, maximum_den = _FLOAT64_MAX.as_integer_ratio()
    if (
        numerator * denominator_den * maximum_den
        > maximum * denominator * numerator_den
    ):
        return float("inf")
    ratio = small / large
    if ratio == 0.0:
        raise FloatingPointError("positive gap ratio underflowed to zero")
    return ratio


def _spectral_rank(
    matrix: FloatArray,
    gram_rtol: float,
    gap_min: float,
    stage: str,
    floor_scale: float = 0.0,
    *,
    exact_zero_columns: tuple[int, ...] = (),
) -> tuple[FloatArray, FloatArray, BoolArray, float, float]:
    if matrix.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float64)
        return (
            empty,
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=bool),
            0.0,
            float("inf"),
        )
    evals, evecs = witnessed_eigh(matrix, exact_zero_columns)
    if not (
        np.all(np.isfinite(evals))
        and np.all(np.isfinite(evecs))
        and np.isfinite(floor_scale)
    ):
        raise GeometryRankError(
            stage + "_non_psd",
            0.0,
            gap_min,
            gram_rtol=gram_rtol,
            threshold=float("nan"),
            smallest_retained=float(evals.min()),
            largest_discarded=float(floor_scale),
        )
    top = max(float(evals.max()), float(floor_scale), 0.0)
    threshold = float(gram_rtol * top)
    if not np.isfinite(threshold) or float(evals.min()) < -threshold:
        raise GeometryRankError(
            stage + "_non_psd",
            0.0,
            gap_min,
            gram_rtol=gram_rtol,
            threshold=threshold,
            smallest_retained=float(evals.min()),
            largest_discarded=top,
        )
    zero = evals <= threshold
    kept = evals[~zero]
    discarded = np.abs(evals[zero])
    if kept.size and discarded.size:
        small = float(kept.min())
        large = float(discarded.max())
        gap = _gap_ratio(small, large)
        if gap < gap_min:
            raise GeometryRankError(
                stage,
                gap,
                gap_min,
                gram_rtol=gram_rtol,
                threshold=threshold,
                smallest_retained=small,
                largest_discarded=large,
            )
    else:
        gap = float("inf")
    return evals, evecs, zero, threshold, gap


def _check_scalar_gap(
    residual: float, scale: float, gram_rtol: float, gap_min: float, stage: str
) -> bool:
    """Classify a scalar residual and refuse unresolved/ill-conditioned geometry.

    The cutoff is gram_rtol*scale. A retained residual also needs residual /
    cutoff >= gap_min. This existing guard is stricter than spectral rank alone.
    """
    cutoff = (
        float(gram_rtol * max(float(scale), 0.0))
        if np.isfinite(scale)
        else float("nan")
    )
    if not (np.isfinite(residual) and np.isfinite(cutoff)) or residual < -cutoff:
        raise GeometryRankError(
            stage + "_invalid_residual",
            0.0,
            gap_min,
            gram_rtol=gram_rtol,
            threshold=cutoff,
            smallest_retained=residual,
            largest_discarded=scale,
        )
    if residual <= cutoff:
        return True
    ratio = _gap_ratio(float(residual), cutoff)
    if ratio < gap_min:
        raise GeometryRankError(
            stage,
            ratio,
            gap_min,
            gram_rtol=gram_rtol,
            threshold=cutoff,
            smallest_retained=residual,
            largest_discarded=cutoff,
        )
    return False


def _validate_custom_basis(
    g_rr: FloatArray,
    shape: tuple[int, int],
    *,
    r_top: float,
    gram_rtol: float,
    gap_min: float,
    custom_nuisance_basis: object,
) -> FloatArray:
    levels, d_r = shape
    if custom_nuisance_basis is None:
        raise ValueError(
            "custom_nuisance_basis is required when nuisance_basis='custom'"
        )
    if not isinstance(custom_nuisance_basis, np.ndarray):
        raise TypeError("custom_nuisance_basis must be a real numeric ndarray")
    if custom_nuisance_basis.shape != (levels, d_r):
        raise ValueError(f"custom_nuisance_basis: expected shape ({levels}, {d_r})")
    if custom_nuisance_basis.dtype.kind not in "biuf":
        raise TypeError("custom_nuisance_basis must be a real numeric ndarray")
    custom = np.asarray(custom_nuisance_basis, dtype=np.float64)
    if custom.shape != (levels, d_r) or not np.all(np.isfinite(custom)):
        raise ValueError(
            f"custom_nuisance_basis: expected finite array of shape ({levels}, {d_r})"
        )
    if d_r == 0:
        return np.zeros((levels, 0), dtype=np.float64)
    sub_gram = custom.T @ g_rr @ custom
    sub_gram = 0.5 * (sub_gram + sub_gram.T)
    evals_c, _, zero_c, _, _ = _spectral_rank(
        sub_gram, gram_rtol, gap_min, "custom_nuisance_basis", floor_scale=r_top
    )
    if np.any(zero_c):
        raise ValueError(
            "custom_nuisance_basis is rank-deficient on the centered retained subspace"
        )
    _check_scalar_gap(
        float(evals_c.min()),
        max(float(evals_c.max()), r_top),
        gram_rtol,
        gap_min,
        "custom_nuisance_basis_conditioning",
    )
    return custom.copy()


def _build_retained_basis(
    g_rr: FloatArray,
    v_kept: FloatArray,
    *,
    r_top: float,
    gram_rtol: float,
    gap_min: float,
    nuisance_basis: str,
    custom_nuisance_basis: object,
) -> FloatArray:
    levels, d_r = v_kept.shape
    if nuisance_basis not in _VALID_BASIS_KINDS:
        raise ValueError(f"nuisance_basis must be one of {_VALID_BASIS_KINDS}")
    if nuisance_basis == "custom":
        return _validate_custom_basis(
            g_rr,
            (levels, d_r),
            r_top=r_top,
            gram_rtol=gram_rtol,
            gap_min=gap_min,
            custom_nuisance_basis=custom_nuisance_basis,
        )
    if custom_nuisance_basis is not None:
        raise ValueError(
            "custom_nuisance_basis is only accepted when nuisance_basis='custom'"
        )
    if d_r == 0:
        return np.zeros((levels, 0), dtype=np.float64)
    if nuisance_basis == "gram_eigh":
        return np.column_stack(
            [_normalize_column_sign(v_kept[:, j]) for j in range(d_r)]
        )
    return _projector_qr(v_kept, gram_rtol, gap_min)


def _projector_qr(v_kept: FloatArray, gram_rtol: float, gap_min: float) -> FloatArray:
    """Select a deterministic basis from the retained-space projector."""
    levels, d_r = v_kept.shape
    projector = v_kept @ v_kept.T
    cols: list[FloatArray] = []
    for col in range(levels):
        vec = projector[:, col].copy()
        for _ in range(2):
            for prev in cols:
                vec = vec - float(prev @ vec) * prev
        norm_sq = float(vec @ vec)
        if _check_scalar_gap(
            norm_sq, 1.0, gram_rtol, gap_min, f"retained_qr_col_{col}"
        ):
            continue
        unit = vec / np.sqrt(norm_sq)
        cols.append(-unit if unit[col] < 0.0 else unit)
        if len(cols) == d_r:
            break
    if len(cols) != d_r:
        raise GeometryRankError(
            "retained_qr",
            0.0,
            gap_min,
            gram_rtol=gram_rtol,
            threshold=gram_rtol,
            smallest_retained=0.0,
            largest_discarded=1.0,
        )
    q_qr = np.column_stack(cols)
    return v_kept @ (v_kept.T @ q_qr)


def _identify_slopes(
    g_xx: FloatArray,
    g_xr: FloatArray,
    v_kept: FloatArray,
    evals_kept: FloatArray,
    *,
    full_top: float,
    gram_rtol: float,
    gap_min: float,
) -> tuple[IndexArray, IndexArray, tuple[str | None, ...]]:
    k = g_xx.shape[0]
    if k == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, ()
    if evals_kept.size > 0:
        proj = (v_kept.T @ g_xr.T) / np.sqrt(evals_kept)[:, None]
        schur = g_xx - proj.T @ proj
        schur = 0.5 * (schur + schur.T)
    else:
        schur = g_xx.copy()
    kept: list[int] = []
    omitted: list[int] = []
    reasons: list[str | None] = [None] * k
    for col in range(k):
        diag_val = float(g_xx[col, col])
        if _check_scalar_gap(
            diag_val, max(full_top, 0.0), gram_rtol, gap_min, f"slope_{col}_centering"
        ):
            omitted.append(col)
            reasons[col] = "centered_constant"
            continue
        res_ret = float(schur[col, col])
        ref_scale = max(diag_val, full_top)
        if _check_scalar_gap(
            res_ret,
            ref_scale,
            gram_rtol,
            gap_min,
            f"slope_{col}_retained_collinearity",
        ):
            omitted.append(col)
            reasons[col] = "collinear_with_retained"
            continue
        if kept:
            s_idx = np.array(kept, dtype=np.intp)
            sub_ss = schur[np.ix_(s_idx, s_idx)]
            sub_sk = schur[s_idx, col]
            res_slp = float(res_ret - sub_sk @ np.linalg.solve(sub_ss, sub_sk))
            if _check_scalar_gap(
                res_slp,
                ref_scale,
                gram_rtol,
                gap_min,
                f"slope_{col}_slope_collinearity",
            ):
                omitted.append(col)
                reasons[col] = "collinear_with_slopes"
                continue
        kept.append(col)
    return (
        np.array(kept, dtype=np.intp),
        np.array(omitted, dtype=np.intp),
        tuple(reasons),
    )


def _block_diagnostics(
    design: CenteredDesign,
    g_rr: FloatArray,
    retained_top: float,
    gram_rtol: float,
    gap_min: float,
) -> tuple[tuple[int, ...], tuple[int, ...], BoolArray, tuple[BoolArray, ...]]:
    k = design.n_slopes
    floor = max(retained_top, 0.0)
    ranks: list[int] = []
    deficiencies: list[int] = []
    ann_blocks: list[bool] = []
    ann_levels: list[BoolArray] = []
    blocks = zip(design.block_starts, design.structure.block_sizes, strict=True)
    for block_idx, (start, size) in enumerate(blocks):
        offset = start - k
        sub = g_rr[offset : offset + size, offset : offset + size]
        diag = np.diag(sub)
        level_zero = np.array(
            [
                _check_scalar_gap(
                    float(diag[level]),
                    floor,
                    gram_rtol,
                    gap_min,
                    f"block_{block_idx}_level_{level}",
                )
                for level in range(size)
            ],
            dtype=bool,
        )
        _, _, zero_b, _, _ = _spectral_rank(
            sub, gram_rtol, gap_min, f"retained_block_{block_idx}", floor_scale=floor
        )
        rank_b = int(np.sum(~zero_b))
        ranks.append(rank_b)
        deficiencies.append(int(size - rank_b))
        ann_blocks.append(rank_b == 0)
        ann_levels.append(level_zero)
    return (
        tuple(ranks),
        tuple(deficiencies),
        np.array(ann_blocks, dtype=bool),
        tuple(ann_levels),
    )
