"""Spectral rank, slope normalization and block diagnostics (design/03, 3.2).

Rank: lambda <= gram_rtol * spectral scale is discarded. The retained/discarded
spectral gap must reach gap_min. The existing scalar Schur/conditioning guard
also requires a nonzero residual / cutoff >= gap_min; it is stricter than rank
alone, and may refuse a full-rank but poorly conditioned coordinate system.
All controls are explicit caller inputs. No ridge or statistical policy is added.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, IndexArray, IndexStorage
from powermean.fe.admission_incidence import QuotientRank
from powermean.fe.certificate_basis import Witness
from powermean.fe.geometry_design import NullProjection, SelectedProof
from powermean.fe.geometry_eigh import witnessed_eigh
from powermean.fe.geometry_errors import GeometryRankError
from powermean.fe.geometry_gram import symmetric_gram
from powermean.fe.geometry_storage import BoolArray, _build_record
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray

_VALID_BASIS_KINDS = ("gram_eigh",)
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


def _build_retained_basis(v_kept: FloatArray, *, nuisance_basis: str) -> FloatArray:
    levels, d_r = v_kept.shape
    if nuisance_basis not in _VALID_BASIS_KINDS:
        raise ValueError(f"nuisance_basis must be one of {_VALID_BASIS_KINDS}")
    if d_r == 0:
        return np.zeros((levels, 0), dtype=np.float64)
    return np.column_stack([_normalize_column_sign(v_kept[:, j]) for j in range(d_r)])


def _solve_gram(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    """Apply the same exact binary scaling to a subnormal Gram and its RHS."""
    # A positive semidefinite Gram is largest on its diagonal. Tiny pivots
    # can overflow a LAPACK reciprocal even when solving G against itself.
    tiny = np.finfo(float).tiny
    if 0.0 < matrix[0, 0] < tiny and np.max(matrix.diagonal()) < tiny:
        _, rhs_exponent = math.frexp(float(np.abs(rhs).max(initial=0.0)))
        shift = min(1022, max(0, 1023 - rhs_exponent))
        matrix = np.ldexp(matrix, shift)
        rhs = np.ldexp(rhs, shift)
    return np.linalg.solve(matrix, rhs)


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
        schur = symmetric_gram(schur)
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
            res_slp = float(res_ret - sub_sk @ _solve_gram(sub_ss, sub_sk))
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


# The factor by which a decision of the selected-columns construction must clear its
# boundary; a closer value is left to the spectral construction, which decides.
CLEAR_MARGIN = 2.0
CHOLESKY_LIMIT = 1e290
_DEKKER_SPLIT = 134217729.0
_DEKKER_SAFE = 1e140
_UNIT = float(np.finfo(np.float64).eps) / 2.0


class _SolveTriangular(Protocol):
    def __call__(
        self,
        a: FloatArray,
        b: FloatArray,
        *,
        lower: bool = False,
    ) -> FloatArray: ...


_solve_triangular = cast(
    _SolveTriangular, importlib.import_module("scipy.linalg").solve_triangular
)


@dataclass(slots=True)
class SelectionTracker:
    """Mutable fit-local counter for attempted proof and compensated passes."""

    n_cholesky_calls: int = 0
    n_triangular_solve_calls: int = 0
    n_compensated_matmul_cols: int = 0
    n_compensated_twosum_calls: int = 0
    n_operator_level_bincounts: int = 0
    n_slope_gram_builds: int = 0
    n_proof_attempts: int = 0
    n_residual_bound_passes: int = 0
    n_split_products: int = 0


def _pairwise_twosum(
    values: FloatArray,
    comp: FloatArray,
    tracker: SelectionTracker | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Sum rows of ``values + comp`` by tree ``TwoSum`` with rigorous error bound."""
    if tracker is not None:
        tracker.n_compensated_twosum_calls += 1
    n_rows, width = values.shape
    if width == 0:
        zeros = np.zeros(n_rows, dtype=np.float64)
        return zeros, zeros
    abs_sum = np.abs(values).sum(axis=1) * (1.0 + float(width) * _UNIT)
    s, e = values, comp
    levels = 0
    while s.shape[1] > 1:
        levels += 1
        half = s.shape[1] // 2
        left, right = s[:, :half], s[:, half : 2 * half]
        total = left + right
        virtual = total - left
        err = (left - (total - virtual)) + (right - virtual)
        e_next = err + (e[:, :half] + e[:, half : 2 * half])
        if s.shape[1] & 1:
            s = np.concatenate((total, s[:, -1:]), axis=1)
            e = np.concatenate((e_next, e[:, -1:]), axis=1)
        else:
            s, e = total, e_next
    out = s[:, 0] + e[:, 0]
    underflow = float(width) * (2.0**-1020)
    bound = (
        _UNIT * np.abs(out)
        + 4.0 * float(levels + 1) * (_UNIT * _UNIT) * abs_sum
        + underflow
    )
    return out, bound


def _accurate_matmul(
    left: FloatArray,
    right: FloatArray,
    tracker: SelectionTracker | None = None,
) -> tuple[FloatArray, FloatArray] | None:
    """Evaluate ``left @ right`` via Dekker ``TwoProd`` and tree ``TwoSum``."""
    with np.errstate(all="ignore"):
        l_max = float(np.abs(left).max(initial=0.0))
        r_max = float(np.abs(right).max(initial=0.0))
    if not (np.isfinite(l_max) and np.isfinite(r_max)):
        return None
    if l_max > _DEKKER_SAFE or r_max > _DEKKER_SAFE:
        return None
    n_rows, n_cols = left.shape[0], right.shape[1]
    if tracker is not None:
        tracker.n_compensated_matmul_cols += n_cols
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    err = np.zeros((n_rows, n_cols), dtype=np.float64)
    cx = _DEKKER_SPLIT * left
    hx = cx - (cx - left)
    tx = left - hx
    for col in range(n_cols):
        vec = right[:, col]
        prod = left * vec
        cy = _DEKKER_SPLIT * vec
        hy = cy - (cy - vec)
        ty = vec - hy
        comp = ((hx * hy - prod) + hx * ty + tx * hy) + tx * ty
        out[:, col], err[:, col] = _pairwise_twosum(prod, comp, tracker)
    return out, err


def _split_residual(
    target: FloatArray,
    left: FloatArray,
    witnesses: FloatArray,
    tracker: SelectionTracker | None = None,
) -> tuple[FloatArray, FloatArray] | None:
    """``target - left @ witnesses`` with an entrywise error bound, integer witnesses.

    ``left = high + low`` exactly, with ``high`` on a power-of-two grid ``step``
    chosen so that every partial sum of ``high @ witnesses`` is an integer
    multiple of ``step`` below ``2**53 * step``: that product carries no rounding
    in any order of summation, with or without fused multiply-add. ``low`` is at
    most ``step / 2`` in magnitude, so the standard bound on ``low @ witnesses``
    is of order ``unit * step``. Two subtractions remain, each with relative
    error ``unit``. Returns ``None`` where the witnesses are not integers or the
    grid does not fit; the caller then uses the compensated products.
    """
    _, width = left.shape
    with np.errstate(all="ignore"):
        w_abs = np.abs(witnesses)
        w_max = float(w_abs.max(initial=0.0))
        l_max = float(np.abs(left).max(initial=0.0))
        if not (np.isfinite(w_max) and np.isfinite(l_max)):
            return None
        if not np.array_equal(witnesses, np.rint(witnesses)):
            return None
        if w_max == 0.0 or l_max == 0.0:
            return target.copy(), np.zeros_like(target)
        if (width + 8) * _UNIT >= 0.25 or w_max * float(width) > 2.0**50:
            return None
        bulk = (
            w_max * float(np.abs(left).sum(axis=1).max()) * (1.0 + 4.0 * width * _UNIT)
        )
        if not (0.0 < bulk <= CHOLESKY_LIMIT):
            return None
        exponent = math.frexp(bulk)[1] - 51
        if exponent < -1000:
            return None
        step = math.ldexp(1.0, exponent)
        high = np.rint(left / step) * step
        low = left - high
        # Exact partial sums need w_max * sum_j |high_ij| < 2**53 * step.
        high_bulk = w_max * float(np.abs(high).sum(axis=1).max())
        if not high_bulk <= 2.0**52 * step:
            return None
        if tracker is not None:
            tracker.n_split_products += 1
        counts = np.count_nonzero(witnesses, axis=0).astype(np.float64)
        gamma_w = (counts + 2.0) * _UNIT / (1.0 - (counts + 2.0) * _UNIT)
        low_err = (
            gamma_w * (0.5 * step) * w_abs.sum(axis=0) + (counts + 2.0) * 2.0**-1060
        )
        first = target - high @ witnesses
        residual = first - low @ witnesses
        err = (_UNIT * (np.abs(first) + np.abs(residual)) + low_err[None, :]) * (
            1.0 + 8.0 * _UNIT
        )
    if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(err))):
        return None
    return residual, err


def _dependent_witnesses(
    quotient: QuotientRank,
) -> tuple[IndexArray, FloatArray] | None:
    """Non-annihilated dependent retained indices and their ``r``-by-``m`` witnesses."""
    held = set(quotient.kept)
    others = [
        c for c, witness in enumerate(quotient.witnesses) if c not in held and witness
    ]
    found = np.zeros((len(quotient.kept), len(others)), dtype=np.float64)
    try:
        for out, c in enumerate(others):
            for position, numerator, denominator in quotient.witnesses[c]:
                found[position, out] = numerator / denominator
    except OverflowError:
        return None
    if not np.all(np.isfinite(found)):
        return None
    return np.array(others, dtype=np.intp), found


def _block_majorant(norm_off: float, norm_diag: float) -> float:
    """Spectral norm upper bound for symmetric ``[0, R; R', E]``."""
    if norm_off == 0.0 and norm_diag == 0.0:
        return 0.0
    bound = 0.5 * (norm_diag + float(np.hypot(norm_diag, 2.0 * norm_off)))
    return max(bound, 2.0**-1022) * (1.0 + 8.0 * _UNIT)


def _frobenius_upper(values: FloatArray) -> float:
    """Scaled Frobenius upper bound, including dimension-dependent reduction error."""
    guard = (2 * values.size + 8) * _UNIT
    scale = float(np.abs(values).max(initial=0.0))
    if not np.isfinite(scale) or guard >= 0.25:
        return float("inf")
    if scale == 0.0:
        return 0.0
    # Scaling keeps at least one squared entry equal to one, including when
    # every input is subnormal. A serial reduction bound also covers a tree.
    upper = scale * (float(np.linalg.norm(values / scale)) / (1.0 - guard))
    return float(np.nextafter(upper, np.inf))


def _eval_null_residual(
    blocks: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    witnesses: FloatArray,
    k: int,
    *,
    mode: str,
    tracker: SelectionTracker | None = None,
) -> tuple[float, float] | None:
    """Bound ``||G - P' G[K, K] P||_2`` on full and retained Gram blocks.

    On symmetric ``G``, ``R_K = G_KD - G_{K, k+J} W`` and ``R_ret = R_K[k:]``;
    the actual bottom-right block of ``G - P' G[K, K] P`` is
    ``E_D = G_DD - W' G_{k+J, k+J} W = (G_DD - G_{D, k+J} W) + R_ret' W``.
    ``mode`` is ``"fast"`` (ordinary products with their a priori bound),
    ``"split"`` (exact split products, integer witnesses only) or ``"accurate"``
    (compensated products); the three bound the same quantity.
    """
    g_kj, g_kd, g_dj, g_dd = blocks
    if tracker is not None:
        tracker.n_residual_bound_passes += 1
    r_size = witnesses.shape[0]
    if (r_size + 12) * _UNIT >= 0.25:
        return None
    abs_w = np.abs(witnesses)
    if mode == "split":
        top = _split_residual(g_kd, g_kj, witnesses, tracker)
        bottom = _split_residual(g_dd, g_dj, witnesses, tracker)
        if top is None or bottom is None:
            return None
        r_k, err_r = top
        diff_dd, err_dd = bottom
    else:
        if mode == "accurate":
            kj_res = _accurate_matmul(g_kj, witnesses, tracker)
            dj_res = _accurate_matmul(g_dj, witnesses, tracker)
            if kj_res is None or dj_res is None:
                return None
            kj_w, kj_err = kj_res
            dj_w, dj_err = dj_res
        else:
            counts = np.count_nonzero(witnesses, axis=0)
            if np.any((counts + 2) * _UNIT >= 0.25):
                return None
            gamma_w = (counts + 1) * _UNIT / (1.0 - (counts + 1) * _UNIT)
            kj_w = g_kj @ witnesses
            kj_err = (np.abs(g_kj) @ abs_w) * (gamma_w / (1.0 - gamma_w))
            dj_w = g_dj @ witnesses
            dj_err = (np.abs(g_dj) @ abs_w) * (gamma_w / (1.0 - gamma_w))
        r_k = g_kd - kj_w
        err_r = (_UNIT * np.abs(r_k) + kj_err) / (1.0 - 4.0 * _UNIT)
        diff_dd = g_dd - dj_w
        err_dd = _UNIT * np.abs(diff_dd) + dj_err
    r_ret, err_r_ret = r_k[k:, :], err_r[k:, :]
    gamma_r = (r_size + 1) * _UNIT / (1.0 - (r_size + 1) * _UNIT)
    pad_r = 1.0 / (1.0 - (r_size + 12) * _UNIT)
    e_d = diff_dd + r_ret.T @ witnesses
    err_e = pad_r * (
        _UNIT * np.abs(e_d)
        + err_dd
        + gamma_r * (np.abs(r_ret).T @ abs_w)
        + err_r_ret.T @ abs_w
    )
    norm_rk = _frobenius_upper(np.abs(r_k) + err_r)
    norm_ret = _frobenius_upper(np.abs(r_ret) + err_r_ret)
    norm_ed = _frobenius_upper(np.abs(e_d) + err_e)
    delta_full = _block_majorant(norm_rk, norm_ed)
    delta_ret = min(_block_majorant(norm_ret, norm_ed), delta_full)
    if not (np.isfinite(delta_full) and np.isfinite(delta_ret)):
        return None
    return delta_full, delta_ret


def _gram_null_defects(
    gram: FloatArray,
    k: int,
    quotient: QuotientRank,
    dep: tuple[IndexArray, FloatArray],
    *,
    rtol: float,
    tracker: SelectionTracker,
) -> tuple[float, float] | None:
    """Upper bounds on full and retained float64 Gram null-subspace perturbations."""
    held = set(quotient.kept)
    ann_cols = np.array(
        [
            k + c
            for c, witness in enumerate(quotient.witnesses)
            if c not in held and not witness
        ],
        dtype=np.intp,
    )
    if ann_cols.size and (
        np.any(gram[ann_cols, :] != 0.0) or np.any(gram[:, ann_cols] != 0.0)
    ):
        return None
    dep_r, witnesses = dep
    if dep_r.size == 0:
        return 0.0, 0.0
    kept_r = k + np.asarray(quotient.kept, dtype=np.intp)
    columns = np.concatenate([np.arange(k, dtype=np.intp), kept_r])
    dep_cols = k + dep_r
    blocks = (
        gram[np.ix_(columns, kept_r)],
        gram[np.ix_(columns, dep_cols)],
        gram[np.ix_(dep_cols, kept_r)],
        gram[np.ix_(dep_cols, dep_cols)],
    )
    with np.errstate(all="ignore"):
        d_max = float(np.diag(gram).max(initial=0.0))
        r_max = float(np.diag(gram[k:, k:]).max(initial=0.0))
        # Integer witnesses: the exact split products cost what the ordinary ones
        # cost and bound the same quantity more tightly, so they go first.
        split = _eval_null_residual(blocks, witnesses, k, mode="split", tracker=tracker)
        if split is not None:
            return split
        fast = _eval_null_residual(blocks, witnesses, k, mode="fast", tracker=tracker)
        if (
            fast is not None
            and fast[0] <= 0.25 * rtol * d_max
            and fast[1] <= 0.25 * rtol * r_max
        ):
            return fast
        return _eval_null_residual(
            blocks, witnesses, k, mode="accurate", tracker=tracker
        )


def _level_norm_bounds(
    design: CenteredDesign, gram: FloatArray, tracker: SelectionTracker
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Per-level ``sqrt(n_j)``, exact-norm upper bounds, and operational lower norms."""
    structure = design.structure
    n_rows = structure.n_rows
    c_max = int(structure.counts.max(initial=1))
    if (2 * n_rows + c_max + 16) * _UNIT >= 0.05:
        return None
    parts = [
        np.bincount(codes, minlength=size)
        for codes, size in zip(
            structure.retained_sorted_codes, structure.block_sizes, strict=True
        )
    ]
    tracker.n_operator_level_bincounts += len(parts)
    n_f = np.concatenate(parts).astype(np.float64)
    k = design.n_slopes
    with np.errstate(all="ignore"):
        diag_r = np.maximum(np.diag(gram[k:, k:]), 0.0)
        pad = 8.0 * (float(c_max) * np.maximum(n_f, 1.0) + n_f + 4.0) * _UNIT
        if not np.all(np.isfinite(pad)) or np.any(pad >= 0.25):
            return None
        diag_err = (pad / (1.0 - pad)) * n_f
        if not np.all(np.isfinite(diag_err)):
            return None
        true_upper = np.sqrt(diag_r + diag_err) * (1.0 + 4.0 * _UNIT)
        true_lower = np.sqrt(np.maximum(diag_r - diag_err, 0.0)) * (1.0 - 4.0 * _UNIT)
        sqrt_n = np.sqrt(n_f)
        gamma_cen = (c_max + 3) * _UNIT / (1.0 - (c_max + 3) * _UNIT)
        gamma_norm = 2.0 * n_rows * _UNIT / (1.0 - 2.0 * n_rows * _UNIT)
        tiny = float(max(n_rows, 1)) * (2.0**-1070)
        single_err = gamma_cen * sqrt_n * (1.0 + 4.0 * _UNIT)
        op_lower = np.maximum(
            (1.0 - gamma_norm) * np.maximum(true_lower - single_err, 0.0) - tiny, 0.0
        )
    return sqrt_n, true_upper, op_lower


def _witness_column_defects(
    c: int,
    witness: Witness,
    kept: tuple[int, ...],
    norms: tuple[FloatArray, FloatArray, FloatArray],
    params: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Bound operational ``(null, reconstruction)`` defects for column ``c``."""
    sqrt_n, true_upper, op_lower = norms
    gamma_cen, eta_op, gamma_norm, tiny = params
    own_low = float(op_lower[c])
    dot_guard = float(len(witness) + 4) * _UNIT
    if own_low <= 0.0 or dot_guard >= 0.25:
        return None
    sw = scale_low = conv_err = 0.0
    try:
        for position, numerator, denominator in witness:
            kept_j = kept[position]
            val = numerator / denominator
            if not np.isfinite(val):
                return None
            abs_val = abs(val)
            sw += abs_val * float(sqrt_n[kept_j])
            scale_low += abs_val * float(op_lower[kept_j])
            if val.as_integer_ratio() != (numerator, denominator):
                conv_err += (_UNIT * abs_val + 2.0**-1074) * float(true_upper[kept_j])
    except OverflowError:
        return None
    own_sqrt = float(sqrt_n[c])
    # The witness can have many terms. Protect the accumulated numerator sums
    # as well as the denominator; dyadic conversion alone does not remove this
    # accumulation error or the operator's centering/gather error.
    sw /= 1.0 - dot_guard
    conv_err /= 1.0 - dot_guard
    scale_low *= 1.0 - dot_guard
    rec_raw = gamma_cen * own_sqrt + eta_op * sw + conv_err
    null_raw = eta_op * (own_sqrt + sw) + conv_err
    rec_num = ((1.0 + gamma_norm) * (1.0 + _UNIT) * rec_raw + tiny) * (
        1.0 + 16.0 * _UNIT
    )
    null_num = ((1.0 + gamma_norm) * null_raw + tiny) * (1.0 + 16.0 * _UNIT)
    rec_def = (rec_num / own_low) * (1.0 + 4.0 * _UNIT)
    null_def = (null_num / (own_low + scale_low)) / (1.0 - dot_guard)
    if not (np.isfinite(rec_def) and np.isfinite(null_def)):
        return None
    return null_def, rec_def


def _operator_defect_bounds(
    design: CenteredDesign,
    gram: FloatArray,
    quotient: QuotientRank,
    rtol: float,
    tracker: SelectionTracker,
) -> tuple[float, float] | None:
    """Certify operational reconstruction/null bounds at ``rtol``, including dyadics."""
    structure = design.structure
    n_blocks = len(structure.block_sizes)
    if n_blocks == 0 or sum(structure.block_sizes) == 0:
        return 0.0, 0.0
    norms = _level_norm_bounds(design, gram, tracker)
    if norms is None:
        return None
    n_rows = structure.n_rows
    c_max = int(structure.counts.max(initial=1))
    b_steps = max(n_blocks - 1, 0)
    gamma_cen = (c_max + 3) * _UNIT / (1.0 - (c_max + 3) * _UNIT)
    gamma_blk = b_steps * _UNIT / (1.0 - b_steps * _UNIT)
    eta_op = (gamma_cen * (1.0 + gamma_blk) + gamma_blk) * (1.0 + 8.0 * _UNIT)
    gamma_norm = 2.0 * n_rows * _UNIT / (1.0 - 2.0 * n_rows * _UNIT)
    tiny = float(max(n_rows, 1)) * (2.0**-1070)
    params = (gamma_cen, eta_op, gamma_norm, tiny)
    held = set(quotient.kept)
    max_null = max_rec = 0.0
    for c, witness in enumerate(quotient.witnesses):
        if c in held or not witness:
            continue
        pair = _witness_column_defects(c, witness, quotient.kept, norms, params)
        if pair is None:
            return None
        max_null = max(max_null, pair[0])
        max_rec = max(max_rec, pair[1])
    if max_null > rtol or max_rec > rtol:
        return None
    return max_null, max_rec


def _slopes_diagonal_clear(
    g_xx: FloatArray,
    base: FloatArray,
    scale: float,
    controls: tuple[float, float],
    margin: float,
) -> bool:
    """Check centering and retained-projection diagonal cutoffs for every slope."""
    gram_rtol, gap_min = controls
    for col in range(g_xx.shape[0]):
        diag_val = float(g_xx[col, col])
        res_ret = float(base[col, col])
        ref = max(diag_val, scale)
        if not (
            np.isfinite(diag_val)
            and np.isfinite(res_ret)
            and diag_val >= margin * gap_min * gram_rtol * scale
            and res_ret >= margin * gap_min * gram_rtol * ref
        ):
            return False
    return True


def _slopes_cholesky_clear(
    g_xx: FloatArray,
    base: FloatArray,
    scale: float,
    controls: tuple[float, float],
    *,
    tracker: SelectionTracker | None,
    margin: float,
) -> bool:
    """Certify all ``k >= 2`` leading slope Schur prefixes in ``O(k^3)``."""
    gram_rtol, gap_min = controls
    k = g_xx.shape[0]
    gamma_k = k * _UNIT / (1.0 - k * _UNIT)
    gamma_k1 = (k + 1) * _UNIT / (1.0 - (k + 1) * _UNIT)
    trace_b = max(float(np.trace(base)), 0.0)
    max_diag = float(np.diag(base).max(initial=0.0))
    chol_shift = 4.0 * gamma_k1 * trace_b + 8.0 * _UNIT * max_diag
    if not np.isfinite(chol_shift) or chol_shift >= float(np.diag(base).min()):
        return False
    if tracker is not None:
        tracker.n_cholesky_calls += 1
    try:
        chol = np.linalg.cholesky(base - chol_shift * np.eye(k, dtype=np.float64))
    except np.linalg.LinAlgError:
        return False
    l_sq = float(np.einsum("ij,ij->", chol, chol))
    back_err = (gamma_k1 * l_sq + 2.0 * _UNIT * (max_diag + chol_shift)) * (
        1.0 + 8.0 * _UNIT
    )
    if not (np.isfinite(back_err) and chol_shift >= back_err):
        return False
    if tracker is not None:
        tracker.n_triangular_solve_calls += 1
    try:
        l_inv = np.asarray(
            _solve_triangular(chol, np.eye(k, dtype=np.float64), lower=True),
            dtype=np.float64,
        )
    except np.linalg.LinAlgError:
        return False
    inv_frob = _frobenius_upper(l_inv)
    err_inv = gamma_k * np.sqrt(l_sq) * inv_frob
    if not (np.isfinite(err_inv) and err_inv < 0.5 and inv_frob > 0.0):
        return False
    mu_s = ((1.0 - err_inv) / inv_frob) ** 2
    lu_err = (
        8.0 * float(k) * np.sqrt(float(k)) * gamma_k * max(float(np.trace(g_xx)), 0.0)
    )
    if not (np.isfinite(lu_err) and lu_err < 0.25 * mu_s):
        return False
    rel_solve = 2.0 * lu_err / mu_s + 2.0 * gamma_k1
    for col in range(1, k):
        diag_val = float(g_xx[col, col])
        ref = max(diag_val, scale)
        res_low = float(chol[col, col]) ** 2 - rel_solve * diag_val
        if not (np.isfinite(res_low) and res_low >= margin * gap_min * gram_rtol * ref):
            return False
    return True


def _slopes_clear(
    g_xx: FloatArray,
    schur: FloatArray,
    scale: float,
    controls: tuple[float, float],
    *,
    diag_err: FloatArray | None = None,
    tracker: SelectionTracker | None = None,
    margin: float = CLEAR_MARGIN,
) -> bool:
    """Whether every slope clears :func:`_identify_slopes` in ``O(k^3)``."""
    k = g_xx.shape[0]
    if not (np.isfinite(scale) and scale >= 0.0):
        return False
    if diag_err is not None and not np.all(np.isfinite(diag_err)):
        return False
    if k == 0:
        return True
    if (k + 4) * _UNIT >= 0.25:
        return False
    with np.errstate(all="ignore"):
        sym = 0.5 * (schur + schur.T)
        base = sym if diag_err is None else sym - np.diag(np.maximum(diag_err, 0.0))
        if not _slopes_diagonal_clear(g_xx, base, scale, controls, margin):
            return False
        if k == 1:
            return True
        return _slopes_cholesky_clear(
            g_xx, base, scale, controls, tracker=tracker, margin=margin
        )


def _original_constants(
    design: CenteredDesign, raw: FloatArray | None, indices: IndexArray
) -> bool:
    """Verify exact within-group equality on the original input-order columns."""
    if (
        raw is None
        or raw.dtype.kind not in "biuf"
        or raw.shape != (design.structure.n_rows, design.n_slopes)
    ):
        return False
    layout = design.structure
    heads = layout.order[layout.starts][layout.cancel_codes]
    return all(
        np.all(np.isfinite(raw[:, j])) and np.array_equal(raw[:, j], raw[heads, j])
        for j in indices
    )


def _slopes_precheck(
    design: CenteredDesign,
    controls: tuple[float, float],
    tracker: SelectionTracker,
    original_slopes: FloatArray | None,
) -> IndexArray | None:
    """Cheap slope-only screen before exact quotient or pair Gram assembly."""
    k = design.n_slopes
    active = np.arange(k, dtype=np.intp)
    if k == 0:
        return active
    n_rows = design.structure.n_rows
    if (2 * n_rows + k + 16) * _UNIT >= 0.05:
        return None
    x = design.centered_slopes
    zero = ~np.any(x, axis=0)
    if np.any(zero):
        # The public collinearity policy can zero a merely near-absorbed column.
        # Only original columns exactly constant on the true groups qualify.
        if np.all(zero) or not _original_constants(
            design, original_slopes, np.flatnonzero(zero)
        ):
            return None
        active = active[~zero]
        x = x[:, active]
        k = int(active.size)
    with np.errstate(all="ignore"):
        tracker.n_slope_gram_builds += 1
        raw_xx = x.T @ x
        g_xx = 0.5 * (raw_xx + raw_xx.T)
        diag_xx = np.diag(g_xx)
        if not np.all(np.isfinite(g_xx)) or np.any(diag_xx <= 0.0):
            return None
        gamma_n = n_rows * _UNIT / (1.0 - n_rows * _UNIT)
        x_low = max(float(diag_xx.max(initial=0.0)) * (1.0 - 4.0 * gamma_n), 0.0)
        trace_xx = max(float(np.trace(g_xx)), 0.0)
        upper_xx = g_xx + (4.0 * gamma_n * trace_xx) * np.eye(k, dtype=np.float64)
        padded_xx = g_xx * (1.0 + 4.0 * gamma_n)
    clear = _slopes_clear(
        padded_xx,
        upper_xx,
        x_low,
        controls,
        tracker=tracker,
        margin=1.5,
    )
    return active if clear else None


def _retained_min_bound(
    l_rr: FloatArray,
    c_req: float,
    w_norm_sq: float,
    delta_ord: float,
    tracker: SelectionTracker,
) -> float | None:
    """Certified positive lower bound on ``lambda_min(G[k + J, k + J])``."""
    fast_mu = c_req / (1.0 + w_norm_sq)
    if fast_mu > 8.0 * delta_ord:
        return fast_mu
    r = l_rr.shape[0]
    tracker.n_triangular_solve_calls += 1
    try:
        l_inv = np.asarray(
            _solve_triangular(l_rr, np.eye(r, dtype=np.float64), lower=True),
            dtype=np.float64,
        )
    except np.linalg.LinAlgError:
        return None
    l_frob_sq = float(np.einsum("ij,ij->", l_rr, l_rr))
    inv_frob = _frobenius_upper(l_inv)
    gamma_r = r * _UNIT / (1.0 - r * _UNIT)
    gamma_r1 = (r + 1) * _UNIT / (1.0 - (r + 1) * _UNIT)
    err_inv = gamma_r * np.sqrt(l_frob_sq) * inv_frob
    if not (np.isfinite(err_inv) and err_inv < 0.5 and inv_frob > 0.0):
        return None
    mu_chol = ((1.0 - err_inv) / inv_frob) ** 2 - gamma_r1 * l_frob_sq
    mu_r = max(fast_mu, mu_chol)
    return mu_r if np.isfinite(mu_r) and mu_r > 8.0 * delta_ord else None


def _slopes_proven(
    gram: FloatArray,
    quotient: QuotientRank,
    witnesses: FloatArray,
    *,
    active: IndexArray,
    n_slopes: int,
    bounds: tuple[float, float],
    controls: tuple[float, float],
    tracker: SelectionTracker,
) -> bool:
    """Prove all ``k`` slopes clear :func:`_identify_slopes` on stored ``gram``."""
    top_high, delta_full = bounds
    rtol, gmin = controls
    k = int(active.size)
    if k == 0:
        return True
    g_xx = gram[:k, :k] if k == n_slopes else gram[np.ix_(active, active)]
    diag_xx = np.diag(g_xx)
    if np.any(diag_xx <= 0.0):
        return False
    kept_r = n_slopes + np.asarray(quotient.kept, dtype=np.intp)
    r = int(kept_r.size)
    if r == 0:
        return _slopes_clear(g_xx, g_xx, top_high, controls, tracker=tracker)
    if (r + k + 2) * _UNIT >= 0.25:
        return False
    order = np.concatenate([kept_r, active])
    tracker.n_cholesky_calls += 1
    try:
        factor = np.linalg.cholesky(gram[np.ix_(order, order)])
    except np.linalg.LinAlgError:
        return False
    with np.errstate(all="ignore"):
        l_rr, l_xx = factor[:r, :r], factor[r:, r:]
        schur = l_xx @ l_xx.T
        gamma_rk = (r + k + 1) * _UNIT / (1.0 - (r + k + 1) * _UNIT)
        delta_ord = gamma_rk * float(np.einsum("ij,ij->", factor, factor))
        w_norm_sq = float(np.einsum("ij,ij->", witnesses, witnesses))
        c_req = max(gmin, CLEAR_MARGIN) * rtol * top_high
        mu_r = _retained_min_bound(l_rr, c_req, w_norm_sq, delta_ord, tracker)
        if mu_r is None:
            return False
        mu_spec = max(mu_r - delta_full, c_req)
        rho_ord = delta_ord / (0.75 * mu_r)
        rho_spec = delta_full / (0.75 * mu_spec)
        levels = gram.shape[0] - n_slopes
        gamma_l = levels * _UNIT / (1.0 - levels * _UNIT)
        r_inf = float(
            np.abs(gram[n_slopes:, n_slopes:]).sum(axis=1).max(initial=0.0)
        ) / (1.0 - gamma_l)
        g_xr = gram[:k, k:] if k == n_slopes else gram[active, n_slopes:]
        scaled_xr = _frobenius_upper(g_xr / np.sqrt(diag_xx)[:, None])
        eps_proj = 2.0 * gamma_l * np.sqrt(float(r)) * scaled_xr / np.sqrt(
            0.75 * mu_spec
        ) + 3.0 * _UNIT * np.sqrt(float(k))
        eigh_err = 16.0 * float(levels) * _UNIT * r_inf / (0.75 * mu_spec)
        gamma_r = r * _UNIT / (1.0 - r * _UNIT)
        gamma_k = k * _UNIT / (1.0 - k * _UNIT)
        lxx_err = gamma_k * float(np.einsum("ij,ij->", l_xx, l_xx))
        eta_rel = (
            rho_ord
            + rho_spec
            + (2.0 * np.sqrt(float(k)) + 1.0) * eps_proj
            + eps_proj * eps_proj
            + float(k) * (_UNIT + 2.0 * gamma_r + eigh_err)
        )
        eta_abs = lxx_err + 2.5 * delta_ord + (2.0 + rho_spec + eigh_err) * delta_full
        diag_err = eta_rel * diag_xx + eta_abs
    return _slopes_clear(
        g_xx, schur, top_high, controls, diag_err=diag_err, tracker=tracker
    )


def _lower_bound_proven(
    matrix: FloatArray, floor: float, tracker: SelectionTracker
) -> float | None:
    """Certified excess ``lambda_min(matrix) - floor >= 0`` via one shifted Cholesky."""
    size = matrix.shape[0]
    if size == 0:
        return 0.0 if np.isfinite(floor) and floor >= 0.0 else None
    if (size + 1) * _UNIT >= 0.25:
        return None
    with np.errstate(all="ignore"):
        bound = float(np.abs(matrix).sum(axis=1).max(initial=0.0))
        diagonal = float(np.abs(np.diag(matrix)).max(initial=0.0))
        trace = float(np.trace(matrix))
    if not (0.0 < bound <= CHOLESKY_LIMIT and 0.0 < floor <= CHOLESKY_LIMIT):
        return None
    growth = (size + 1) * _UNIT / (1.0 - (size + 1) * _UNIT)
    shift = floor * (1.0 + 2.0**-20) + 4.0 * growth * max(trace, 0.0)
    shift += 8.0 * _UNIT * (diagonal + floor)
    if not np.isfinite(shift) or shift > diagonal:
        return None
    shifted = matrix - shift * np.eye(size, dtype=np.float64)
    tracker.n_cholesky_calls += 1
    try:
        factor = np.linalg.cholesky(shifted)
    except np.linalg.LinAlgError:
        return None
    frobenius = 2.0 * float(np.einsum("ij,ij->", factor, factor))
    lower = shift - growth * frobenius - 4.0 * _UNIT * (diagonal + shift)
    if not (np.isfinite(lower) and lower >= floor * (1.0 + 2.0**-30)):
        return None
    return max(float(lower - floor) * (1.0 - 4.0 * _UNIT), 0.0)


def _build_null_projection(
    kept_r: IndexArray,
    dep_cols: IndexArray,
    witnesses: FloatArray,
    tracker: SelectionTracker,
) -> NullProjection | None:
    """Factor only the smaller reporting system, once, after selection is proved."""
    r, m = witnesses.shape
    if r == 0 or m == 0:
        return None
    if 2.0 * (m + r + 4) * _UNIT >= 0.25:
        return None
    r_side = r <= m
    with np.errstate(all="ignore"):
        product = witnesses @ witnesses.T if r_side else witnesses.T @ witnesses
        gram_side = np.eye(min(r, m), dtype=np.float64) + product
        gram_side = 0.5 * (gram_side + gram_side.T)
    if not np.all(np.isfinite(gram_side)):
        return None
    tracker.n_cholesky_calls += 1
    try:
        chol = np.linalg.cholesky(gram_side)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(chol)):
        return None
    return cast(
        NullProjection,
        _build_record(
            NullProjection,
            _kept_indices=IndexStorage.store(kept_r),
            _dep_indices=IndexStorage.store(dep_cols),
            _witnesses=FloatStorage.store(witnesses),
            _chol=FloatStorage.store(chol),
            r_side=r_side,
        ),
    )


def _woodbury_gram(
    witnesses: FloatArray, tracker: SelectionTracker
) -> tuple[FloatArray, float] | None:
    """``(N, eps)`` with ``W (I + W'W)^-1 W' >= N - eps I`` in the Loewner order.

    ``S = I + W'W`` is factored after an upward shift that exceeds the rounding
    of its formation and of its Cholesky factorization, so the computed factor
    satisfies ``L L' >= S`` and therefore ``(L L')^-1 <= S^-1``. ``Y`` solves
    ``L Y = W'``; then ``Y'Y = W (L L')^-1 W'``, and ``eps`` bounds the rounding
    of the triangular solves (``||L^-1||_2 <= 1`` because ``L L' >= I``) and of
    the product ``Y'Y``. No inverse of an ill-conditioned matrix is formed.
    """
    r, m = witnesses.shape
    if (r + m + 8) * _UNIT >= 0.01:
        return None
    gamma_r = (r + 2) * _UNIT / (1.0 - (r + 2) * _UNIT)
    gamma_m = (m + 2) * _UNIT / (1.0 - (m + 2) * _UNIT)
    with np.errstate(all="ignore"):
        w_sq = _frobenius_upper(witnesses) ** 2 * (1.0 + 8.0 * _UNIT)
        raw = witnesses.T @ witnesses
        gram_w = 0.5 * (raw + raw.T)
        diag_max = float(np.diag(gram_w).max(initial=0.0))
        form_err = gamma_r * w_sq + 4.0 * _UNIT * (1.0 + diag_max)
        shift = 2.0 * (form_err + 4.0 * gamma_m * (float(m) + w_sq)) + 2.0**-40
        if not (np.isfinite(shift) and shift < 0.5):
            return None
        gram_w[np.diag_indices_from(gram_w)] += 1.0 + shift
        if not np.all(np.isfinite(gram_w)):
            return None
        tracker.n_cholesky_calls += 1
        try:
            chol = np.linalg.cholesky(gram_w)
        except np.linalg.LinAlgError:
            return None
        l_frob = _frobenius_upper(chol)
        chol_err = 2.0 * gamma_m * l_frob * l_frob
        if not (np.isfinite(chol_err) and shift >= (form_err + chol_err) * 1.01):
            return None
        tracker.n_triangular_solve_calls += 1
        try:
            solved = np.asarray(
                _solve_triangular(chol, witnesses.T, lower=True), dtype=np.float64
            )
        except np.linalg.LinAlgError:
            return None
        y_frob = _frobenius_upper(solved)
        solve_err = 2.0 * gamma_m * l_frob * y_frob
        raw_n = solved.T @ solved
        found = 0.5 * (raw_n + raw_n.T)
        eps = solve_err * (2.0 * y_frob + solve_err) + 2.0 * gamma_m * y_frob * y_frob
        eps *= 1.0 + 16.0 * _UNIT
    if not (np.all(np.isfinite(found)) and np.isfinite(eps)):
        return None
    return found, eps


def _pencil_proven(
    id_gram: FloatArray,
    witnesses: FloatArray,
    k: int,
    floor: float,
    tracker: SelectionTracker,
) -> float | None:
    """Certified excess of the smallest nonzero eigenvalue of ``P' M P`` over ``floor``.

    ``M = G[K, K]`` and ``P = diag(I_k, [I W])`` in the selected ordering, so
    ``P P' = diag(I_k, A)`` with ``A = I + W W'``. The nonzero eigenvalues of
    ``P' M P`` are those of ``M^(1/2) P P' M^(1/2)``; they are all at least ``c``
    exactly when ``Q(c) = M - c diag(I_k, A^-1)`` is positive semidefinite, and
    ``A^-1 = I - W (I + W'W)^-1 W'``. With ``(N, eps)`` of :func:`_woodbury_gram`,
    ``Q(c) >= M - c I + c diag(0, N) - c eps I``: one shifted Cholesky of the
    computed right-hand matrix certifies it above ``c eps`` plus the rounding
    of its formation. ``diag(I_k, A^-1) <= I``, so an excess of ``Q(c)`` is an
    excess of the pencil eigenvalue.
    """
    d = id_gram.shape[0]
    r, m = witnesses.shape
    if r == 0 or m == 0:
        return _lower_bound_proven(id_gram, floor, tracker)
    if d != k + r or (d + 8) * _UNIT >= 0.01:
        return None
    woodbury = _woodbury_gram(witnesses, tracker)
    if woodbury is None:
        return None
    found, eps = woodbury
    with np.errstate(all="ignore"):
        matrix = id_gram.copy()
        matrix[k:, k:] += floor * found
        matrix[np.diag_indices_from(matrix)] -= floor
        form_err = (
            8.0
            * _UNIT
            * (
                _frobenius_upper(id_gram)
                + floor * (_frobenius_upper(found) + float(np.sqrt(d)))
            )
        )
        slack = (floor * eps + form_err) * (1.0 + 2.0**-20)
    if not (np.all(np.isfinite(matrix)) and np.isfinite(slack) and slack > 0.0):
        return None
    excess = _lower_bound_proven(matrix, slack, tracker)
    return None if excess is None else excess


def _spectrum_proven(
    id_gram: FloatArray,
    witnesses: FloatArray,
    *,
    k: int,
    bounds: tuple[float, float, float],
    controls: tuple[float, float],
    tracker: SelectionTracker,
) -> tuple[float, float, float] | None:
    """Certify the positive spectra of the original Gram on the selected block.

    The spectral construction tests the nonzero spectrum of ``G``: its smallest
    retained eigenvalue against ``gap_min * gram_rtol * top`` and against the
    discarded ones. With ``M = G[K, K]`` and ``||G - P' M P||_2 <= delta_full``,
    that spectrum is the spectrum of the pencil ``(M, (P P')^-1)`` up to
    ``delta_full`` (Weyl). :func:`_pencil_proven` certifies the pencil at
    ``required = gap floor + delta_full``; its retained principal block gives
    the same for the retained Gram.

    The selected block itself is certified at the rank cutoff only:
    ``lambda_min(M) >= CLEAR_MARGIN * gram_rtol * top_high``. The pencil implies
    ``lambda_min(M) >= required / (1 + ||W||_F^2)`` because
    ``(P P')^-1 >= I / (1 + ||W||_2^2)``; where that value does not reach the
    rank cutoff, a direct shifted Cholesky must. The certified lower bound is
    returned for the record. ``lambda_min(M) >= gap_min * gram_rtol * top_high``
    is NOT established here.
    """
    top_high, delta_full, delta_ret = bounds
    rtol, gmin = controls
    eff_gmin = max(gmin, CLEAR_MARGIN)
    min_floor = eff_gmin * rtol * top_high
    required = (min_floor + delta_full) * (1.0 + 64.0 * _UNIT)
    rank_floor = CLEAR_MARGIN * rtol * top_high * (1.0 + 64.0 * _UNIT)
    if not (np.isfinite(required) and required > 0.0 and rank_floor > 0.0):
        return None
    excess = _pencil_proven(id_gram, witnesses, k, required, tracker)
    if excess is None:
        return None
    full_min_low = max(
        (required + excess - delta_full) * (1.0 - 16.0 * _UNIT), min_floor
    )
    ret_min_low = max((required + excess - delta_ret) * (1.0 - 16.0 * _UNIT), min_floor)
    with np.errstate(all="ignore"):
        w_sq = _frobenius_upper(witnesses) ** 2 * (1.0 + 8.0 * _UNIT)
        implied = (required / (1.0 + w_sq)) * (1.0 - 16.0 * _UNIT)
    if np.isfinite(implied) and implied >= rank_floor:
        selected_min_low = implied
    else:
        own_excess = _lower_bound_proven(id_gram, rank_floor, tracker)
        if own_excess is None:
            return None
        selected_min_low = (rank_floor + own_excess) * (1.0 - 16.0 * _UNIT)
    values = (full_min_low, ret_min_low, selected_min_low)
    if not all(np.isfinite(value) for value in values):
        return None
    return full_min_low, ret_min_low, selected_min_low


def _component_proven(
    sub: FloatArray,
    comp: IndexArray,
    bounds: tuple[float, float, float, float],
    tracker: SelectionTracker,
) -> bool:
    """Prove one connected single-block Laplacian component clears rank and gap."""
    rtol, gmin, r_low, b_high = bounds
    s = int(comp.size)
    if s < 2 or (s + 2) * _UNIT >= 0.25:
        return False
    sub_c = sub[np.ix_(comp, comp)]
    gamma_s = s * _UNIT / (1.0 - s * _UNIT)
    row_abs = np.abs(sub_c).sum(axis=1)
    u = sub_c.sum(axis=1)
    u_err = gamma_s * row_abs
    rk_norm = _frobenius_upper(np.abs(u[:-1]) + u_err[:-1])
    ed_abs = abs(float(u[-1] - u[:-1].sum())) + float(
        u_err.sum() + gamma_s * np.abs(u[:-1]).sum()
    )
    delta_m = _block_majorant(rk_norm, ed_abs)
    if delta_m > 0.25 * rtol * r_low:
        u, u_err = _pairwise_twosum(sub_c, np.zeros_like(sub_c), tracker)
        rk_norm = _frobenius_upper(np.abs(u[:-1]) + u_err[:-1])
        piv_sum, piv_err = _pairwise_twosum(
            u[:-1].reshape(1, -1), np.zeros((1, s - 1), dtype=np.float64), tracker
        )
        ed_abs = abs(float(u[-1] - piv_sum[0])) + float(u_err.sum() + piv_err[0])
        delta_m = _block_majorant(rk_norm, ed_abs)
    if not (np.isfinite(delta_m) and delta_m <= 0.5 * rtol * r_low):
        return False
    eff_gmin = max(gmin, CLEAR_MARGIN)
    c_m = (eff_gmin * rtol * b_high + delta_m) * (1.0 + 64.0 * _UNIT)
    piv = comp[:-1]
    sub_piv = sub[np.ix_(piv, piv)]
    mat = sub_piv + (c_m / float(s))
    mat[np.diag_indices_from(mat)] -= c_m
    scale_m = float(np.abs(sub_piv).sum(axis=1).max(initial=0.0)) + 2.0 * c_m
    floor_m = max(4.0 * _UNIT * scale_m, 2.0**-1000)
    return bool(np.isfinite(floor_m)) and (
        _lower_bound_proven(mat, floor_m, tracker) is not None
    )


def _block_components_proven(
    sub: FloatArray,
    ann_mask: BoolArray,
    rank_b: int,
    bounds: tuple[float, float, float, float],
    tracker: SelectionTracker,
) -> bool:
    """Prove all connected components of one retained block match ``rank_b``."""
    size = sub.shape[0]
    if np.any(sub[ann_mask, :] != 0.0) or np.any(sub[:, ann_mask] != 0.0):
        return False
    visited = ann_mask.copy()
    proven_rank = 0
    for start_v in range(size):
        if visited[start_v]:
            continue
        comp_mask = np.zeros(size, dtype=bool)
        comp_mask[start_v] = True
        frontier = sub[start_v] != 0.0
        frontier[start_v] = False
        while np.any(frontier):
            comp_mask |= frontier
            frontier = np.any(sub[frontier] != 0.0, axis=0) & ~comp_mask
        if np.any(comp_mask & ann_mask):
            return False
        visited |= comp_mask
        comp = np.flatnonzero(comp_mask)
        if not _component_proven(sub, comp, bounds, tracker):
            return False
        proven_rank += int(comp.size) - 1
    return proven_rank == rank_b


def _block_slice_proven(
    sub: FloatArray,
    ann_mask: BoolArray,
    rank_b: int,
    *,
    single_block: bool,
    bounds: tuple[float, float, float, float],
    tracker: SelectionTracker,
) -> bool:
    """Prove per-level cutoffs and component spectrum for one retained block."""
    rtol, gmin, _, b_high = bounds
    diag = np.diag(sub)
    if np.any(diag[ann_mask] != 0.0):
        return False
    if np.any(diag[~ann_mask] < CLEAR_MARGIN * gmin * rtol * b_high):
        return False
    if rank_b == 0:
        return bool(np.all(ann_mask) and np.all(sub == 0.0))
    if single_block:
        return True
    return _block_components_proven(sub, ann_mask, rank_b, bounds, tracker)


def _blocks_proven(
    design: CenteredDesign,
    g_rr: FloatArray,
    quotient: QuotientRank,
    controls: tuple[float, float],
    tracker: SelectionTracker,
    *,
    retained_bounds: tuple[float, float, float] | None,
) -> bool:
    """Prove every per-level and per-block check of :func:`_block_diagnostics`."""
    rtol, gmin = controls
    if quotient.rank == 0:
        return bool(np.all(g_rr == 0.0)) and all(
            rank == 0 for rank in quotient.block_ranks
        )
    if retained_bounds is None:
        return False
    gamma_l, r_low, r_high = retained_bounds
    if not (np.isfinite(r_high) and r_low > 0.0):
        return False
    k = design.n_slopes
    single_block = len(design.structure.block_sizes) == 1
    for block_idx, (start, size) in enumerate(
        zip(design.block_starts, design.structure.block_sizes, strict=True)
    ):
        offset = start - k
        sub = g_rr[offset : offset + size, offset : offset + size]
        ann_mask = np.asarray(quotient.annihilated_levels[block_idx], dtype=bool)
        sub_inf = (
            r_high
            if single_block
            else float(np.abs(sub).sum(axis=1).max(initial=0.0)) / (1.0 - gamma_l)
        )
        bounds = (rtol, gmin, r_low, max(r_high, sub_inf))
        if not _block_slice_proven(
            sub,
            ann_mask,
            quotient.block_ranks[block_idx],
            single_block=single_block,
            bounds=bounds,
            tracker=tracker,
        ):
            return False
    return True


def _retained_bounds(g_rr: FloatArray) -> tuple[float, float, float] | None:
    """Roundoff allowance and eigenvalue bounds used by both retained checks."""
    levels = g_rr.shape[0]
    if levels == 0 or (levels + 1) * _UNIT >= 0.25:
        return None
    gamma_l = levels * _UNIT / (1.0 - levels * _UNIT)
    with np.errstate(all="ignore"):
        r_inf = float(np.abs(g_rr).sum(axis=1).max(initial=0.0))
        r_diag = float(np.diag(g_rr).max(initial=0.0))
    r_low = r_diag - gamma_l * r_inf
    return gamma_l, r_low, r_inf / (1.0 - gamma_l)


def _candidate_spectrum_proven(
    design: CenteredDesign,
    gram: FloatArray,
    id_gram: FloatArray,
    quotient: QuotientRank,
    controls: tuple[float, float],
    *,
    active: IndexArray,
    tracker: SelectionTracker | None = None,
) -> tuple[SelectedProof, NullProjection | None] | None:
    """Certify spectral, block, slope, and operator bounds for selection."""
    prep = SelectionTracker() if tracker is None else tracker
    prep.n_proof_attempts += 1
    rtol, gmin = controls
    p, k = gram.shape[0], design.n_slopes
    if p == 0 or (p + 1) * _UNIT >= 0.25 or not np.array_equal(gram, gram.T):
        return None
    gamma_p = p * _UNIT / (1.0 - p * _UNIT)
    with np.errstate(all="ignore"):
        top = float(np.abs(gram).sum(axis=1).max(initial=0.0))
        d_max = float(np.diag(gram).max(initial=0.0))
    top_low = d_max - gamma_p * top
    top_high = float(np.nextafter(top / (1.0 - gamma_p), np.inf))
    if not (np.isfinite(top_high) and top_low > 0.0):
        return None
    op_bounds = _operator_defect_bounds(design, gram, quotient, rtol, prep)
    dep = _dependent_witnesses(quotient) if op_bounds is not None else None
    if op_bounds is None or dep is None:
        return None
    defects = _gram_null_defects(gram, k, quotient, dep, rtol=rtol, tracker=prep)
    if defects is None or defects[0] > 0.5 * rtol * top_low:
        return None
    delta_full, delta_ret = defects
    kept_r = k + np.asarray(quotient.kept, dtype=np.intp)
    g_rr = gram[k:, k:]
    retained_bounds = None
    if quotient.kept:
        retained_bounds = _retained_bounds(g_rr)
        if retained_bounds is None or not (
            retained_bounds[1] > 0.0 and delta_ret <= 0.5 * rtol * retained_bounds[1]
        ):
            return None
    spec_bounds = _spectrum_proven(
        id_gram,
        dep[1],
        k=int(active.size),
        bounds=(top_high, delta_full, delta_ret),
        controls=(rtol, gmin),
        tracker=prep,
    )
    if spec_bounds is None or not (
        _blocks_proven(
            design, g_rr, quotient, (rtol, gmin), prep, retained_bounds=retained_bounds
        )
        and _slopes_proven(
            gram,
            quotient,
            dep[1],
            active=active,
            n_slopes=k,
            bounds=(top_high, delta_full),
            controls=(rtol, gmin),
            tracker=prep,
        )
    ):
        return None
    null_proj = _build_null_projection(kept_r, k + dep[0], dep[1], prep)
    if dep[1].size and null_proj is None:
        return None
    proof = cast(
        SelectedProof,
        _build_record(
            SelectedProof,
            top_high=top_high,
            delta_full=delta_full,
            delta_ret=delta_ret,
            full_min_low=spec_bounds[0],
            ret_min_low=spec_bounds[1],
            selected_min_low=spec_bounds[2],
            max_null_defect=op_bounds[0],
            max_reconstruction_defect=op_bounds[1],
        ),
    )
    return proof, null_proj
