"""General identified geometry preparation (design/03, Sections 3.1-3.2).

B = M_C[X,F_R], G = B'B, T = diag(S_X,Q_R), P = (T'GT)^(-1)T'G.
At a valid rank, BT has full column rank, PT = I and BTP = B; T complements
ker(B), generally oblique in coefficient space. Slopes are normalized by
setting omitted original slope coefficients to zero; see slope_alias_matrix.

With ``nuisance_basis="exact_selection"``, Q_R = E_J selects the pivot columns J
of the exact quotient of the retained incidence (fe.admission_incidence): every
other retained column c is B e_c = B E_J w_c for its exact rational witness w_c,
so T'GT = G[K, K] for the kept columns K, P e_c = (0, w_c) exactly and the null
space is spanned by the e_c - E_J w_c. Full, fresh, retained, identified,
per-block and slope-identification invariants are proven on the stored float64
Gram with explicit null-residual bounds; any case not proven with margin or
refused by the quotient hands off to the spectral construction below.

Internal numerical preparation only. Gram truncation does not certify structural
identification. Use geometry_checks.check_geometry for fresh raw binding and
original-space checks; general structural admission remains outside this layer.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from powermean.fe._cancel_storage import FloatStorage, IndexArray, IndexStorage
from powermean.fe.admission_incidence import QuotientRank, quotient_rank
from powermean.fe.admission_records import AdmissionWork, WorkLog
from powermean.fe.geometry_design import (
    NullProjection,
    NumericalGeometry,
    SelectedProof,
)
from powermean.fe.geometry_errors import GeometryRankError
from powermean.fe.geometry_gram import (
    AssemblyWork,
    GeometryBudget,
    GeometryPasses,
    SelectionOutcome,
    SelectionPreparation,
    _checked_controls,
    assembled_gram,
    geometry_gram,
    reserve_geometry,
    spectral_passes,
    symmetric_gram,
    with_selection_preparation,
)
from powermean.fe.geometry_rank import (
    SelectionTracker,
    _block_diagnostics,
    _build_retained_basis,
    _candidate_spectrum_proven,
    _check_scalar_gap,
    _gap_ratio,
    _identify_slopes,
    _normalize_column_sign,
    _slopes_precheck,
    _solve_gram,
    _spectral_rank,
)
from powermean.fe.geometry_storage import BoolStorage, _build_record
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray


def numerical_geometry(
    design: CenteredDesign,
    *,
    gram_rtol: float,
    gap_min: float,
    max_bytes: int,
    nuisance_basis: str = "gram_eigh",
    original_slopes: FloatArray | None = None,
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
        Explicit Gram eigenbasis ("gram_eigh"), or the pivot columns of the exact
        quotient ("exact_selection", which hands every case it does not prove to the
        Gram eigenbasis).
    original_slopes : ndarray, optional
        Original input-order slopes before policy omissions. Selection may deflate
        an exactly zero centered axis only after verifying raw within-group constancy.

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
        controls = _checked_controls(gram_rtol, gap_min, max_bytes)
        selected, ledger, exact, basis_kind = _try_selected(
            design, nuisance_basis, controls, original_slopes
        )
        if selected is not None:
            return selected
        resolved_ledger = (
            geometry_gram(design, max_bytes=controls[2]) if ledger is None else ledger
        )
        held_quotient = None if exact is None else (exact[0], exact[1].snapshot())
        try:
            return _numerical_geometry(
                design,
                controls=controls,
                nuisance_basis=basis_kind,
                ledger=resolved_ledger,
                quotient=held_quotient,
            )
        except (GeometryRankError, FloatingPointError, np.linalg.LinAlgError) as error:
            if resolved_ledger[2].selection.attempted:
                _attach_selection_preparation(error, resolved_ledger[2].selection)
            raise


class _SelectionRefusal(Protocol):
    """An exception carrying the selection evidence attached before it is reraised."""

    selection_preparation: SelectionPreparation


def _attach_selection_preparation(
    error: Exception, preparation: SelectionPreparation
) -> None:
    """Add evidence to the same exception instance without changing its type."""
    cast("_SelectionRefusal", error).selection_preparation = preparation


Ledger = tuple[FloatArray, GeometryBudget, GeometryPasses]


def _numerical_geometry(
    design: CenteredDesign,
    *,
    controls: tuple[float, float, int],
    nuisance_basis: str,
    ledger: Ledger,
    quotient: tuple[QuotientRank, AdmissionWork] | None = None,
) -> NumericalGeometry:
    rtol, gmin, limit = controls
    gram, budget, passes = ledger
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
    q_r = _build_retained_basis(v_kept, nuisance_basis=nuisance_basis)

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

    # T'G, formed once: the identified Gram and the solve for P below both use it.
    mapped = basis.T @ gram
    id_gram = mapped @ basis
    id_gram = symmetric_gram(id_gram)
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
        reduction = _solve_gram(id_gram, mapped)
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
            spectrum_provenance="spectral_eigh",
            threshold_kind="spectral_scale",
            gap_kind="spectral_ratio",
            budget=budget,
            passes=passes,
            max_bytes=limit,
            retained_bytes=retained_bytes,
            _quotient=quotient,
            _selected_columns=None,
            _null_projection=None,
            _selected_proof=None,
        ),
    )


def _make_selection_prep(
    outcome: SelectionOutcome,
    asm_info: tuple[int, int, int, int],
    *,
    canonical_builds: int,
    quotient_attempts: int,
    quotient_work: AdmissionWork,
    tracker: SelectionTracker,
) -> SelectionPreparation:
    """Build immutable fit-local :class:`SelectionPreparation` record."""
    return cast(
        SelectionPreparation,
        _build_record(
            SelectionPreparation,
            attempted=True,
            outcome=outcome,
            n_assembled_gram_attempts=asm_info[0],
            n_assembled_pairs=asm_info[1],
            n_assembled_bincount_calls=asm_info[2],
            n_assembled_unique_calls=asm_info[3],
            n_canonical_gram_builds=canonical_builds,
            n_quotient_attempts=quotient_attempts,
            quotient_work=quotient_work,
            n_cholesky_calls=tracker.n_cholesky_calls,
            n_triangular_solve_calls=tracker.n_triangular_solve_calls,
            n_compensated_matmul_cols=tracker.n_compensated_matmul_cols,
            n_compensated_twosum_calls=tracker.n_compensated_twosum_calls,
            n_operator_level_bincounts=tracker.n_operator_level_bincounts,
            n_slope_gram_builds=tracker.n_slope_gram_builds,
            n_proof_attempts=tracker.n_proof_attempts,
            n_residual_bound_passes=tracker.n_residual_bound_passes,
            n_split_products=tracker.n_split_products,
        ),
    )


def _fallback_ledger(
    design: CenteredDesign,
    limit: int,
    existing_canonical: Ledger | None,
    prep: SelectionPreparation,
) -> Ledger:
    """Attach ``prep`` to canonical ``geometry_gram`` on selection decline."""
    try:
        canon = (
            geometry_gram(design, max_bytes=limit)
            if existing_canonical is None
            else existing_canonical
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        _attach_selection_preparation(error, prep)
        raise
    return canon[0], canon[1], with_selection_preparation(canon[2], prep)


def _attempt_assembled(
    design: CenteredDesign,
    limit: int,
    q_work: AdmissionWork,
    tracker: SelectionTracker,
) -> tuple[Ledger | None, tuple[int, int, int, int]]:
    """Retain attempt evidence on an assembly refusal without changing its error."""
    work = AssemblyWork()
    try:
        assembled = assembled_gram(design, max_bytes=limit, work=work)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        prep = _make_selection_prep(
            "assembly_refused",
            work.snapshot(),
            canonical_builds=0,
            quotient_attempts=1,
            quotient_work=q_work,
            tracker=tracker,
        )
        _attach_selection_preparation(error, prep)
        raise
    return assembled, work.snapshot()


def _try_selected(
    design: CenteredDesign,
    kind: str,
    controls: tuple[float, float, int],
    original_slopes: FloatArray | None,
) -> tuple[
    NumericalGeometry | None, Ledger | None, tuple[QuotientRank, WorkLog] | None, str
]:
    """The selected-columns geometry, or what the spectral construction reuses.

    Any other ``kind`` is returned unchanged with nothing built. Where the selected
    construction declines, ``gram_eigh`` decides on canonical ``geometry_gram``,
    with all attempted precheck/quotient/assembled/proof counts recorded in
    ``passes.selection``.
    """
    if kind != "exact_selection":
        return None, None, None, kind
    rtol, gmin, limit = controls
    reserve_geometry(design, limit)
    tracker = SelectionTracker()
    no_asm = (0, 0, 0, 0)
    active = _slopes_precheck(design, (rtol, gmin), tracker, original_slopes)
    if active is None:
        prep = _make_selection_prep(
            "slopes_precheck_declined",
            no_asm,
            canonical_builds=1,
            quotient_attempts=0,
            quotient_work=AdmissionWork(()),
            tracker=tracker,
        )
        return None, _fallback_ledger(design, limit, None, prep), None, "gram_eigh"
    exact, q_work = _exact_quotient(design)
    if exact is None:
        prep = _make_selection_prep(
            "quotient_refused",
            no_asm,
            canonical_builds=1,
            quotient_attempts=1,
            quotient_work=q_work,
            tracker=tracker,
        )
        return None, _fallback_ledger(design, limit, None, prep), None, "gram_eigh"
    if design.n_slopes + exact[0].rank == 0:
        prep = _make_selection_prep(
            "zero_columns_mismatch",
            no_asm,
            canonical_builds=1,
            quotient_attempts=1,
            quotient_work=q_work,
            tracker=tracker,
        )
        return None, _fallback_ledger(design, limit, None, prep), exact, "gram_eigh"
    assembled, asm_info = _attempt_assembled(design, limit, q_work, tracker)
    if assembled is not None:
        found, canon = assembled, None
        canon_builds = 0
    else:
        prep = _make_selection_prep(
            "canonical_gram_refused",
            asm_info,
            canonical_builds=1,
            quotient_attempts=1,
            quotient_work=q_work,
            tracker=tracker,
        )
        found = _fallback_ledger(design, limit, None, prep)
        canon = found
        canon_builds = 1
    selected, outcome = _selected_geometry(
        design,
        found,
        exact,
        controls,
        (asm_info, canon_builds, q_work, tracker),
        active=active,
    )
    if selected is not None:
        return selected, None, exact, kind
    prep = _make_selection_prep(
        outcome,
        asm_info,
        canonical_builds=1,
        quotient_attempts=1,
        quotient_work=q_work,
        tracker=tracker,
    )
    return None, _fallback_ledger(design, limit, canon, prep), exact, "gram_eigh"


def _exact_quotient(
    design: CenteredDesign,
) -> tuple[tuple[QuotientRank, WorkLog] | None, AdmissionWork]:
    """The exact quotient of the retained incidence, preserving partial work."""
    work = WorkLog()
    try:
        found = quotient_rank(design.structure, work)
    except ValueError:
        return None, work.snapshot()
    return (found, work), work.snapshot()


def _witness_maps(
    k: int, quotient: QuotientRank, columns: IndexArray, n_columns: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """T, P and the null basis of the selection, from the exact witnesses.

    Retained column c is B e_c = B E_J w_c, so P e_c places w_c on the retained
    coordinates and e_c - E_J w_c spans the null space; kept columns map to their
    own coordinate. Each rational is its correctly rounded quotient.
    """
    d = columns.size
    basis = np.zeros((n_columns, d), dtype=np.float64)
    basis[columns, np.arange(d)] = 1.0
    reduction = np.zeros((d, n_columns), dtype=np.float64)
    active = columns[columns < k]
    n_active = active.size
    omitted = np.setdiff1d(np.arange(k), active, assume_unique=True)
    reduction[np.arange(n_active), active] = 1.0
    kept = quotient.kept
    held = set(kept)
    others = [c for c in range(len(quotient.witnesses)) if c not in held]
    null_basis = np.zeros((n_columns, len(others) + omitted.size), dtype=np.float64)
    null_basis[omitted, np.arange(omitted.size)] = 1.0
    for c, witness in enumerate(quotient.witnesses):
        for position, numerator, denominator in witness:
            reduction[n_active + position, k + c] = numerator / denominator
    for out, c in enumerate(others, start=int(omitted.size)):
        null_basis[k + c, out] = 1.0
        for position, numerator, denominator in quotient.witnesses[c]:
            null_basis[k + kept[position], out] = -(numerator / denominator)
        null_basis[:, out] = _normalize_column_sign(null_basis[:, out])
    return basis, reduction, null_basis


def _selected_geometry(
    design: CenteredDesign,
    ledger: Ledger,
    exact: tuple[QuotientRank, WorkLog],
    controls: tuple[float, float, int],
    prep_ctx: tuple[tuple[int, int, int, int], int, AdmissionWork, SelectionTracker],
    *,
    active: IndexArray,
) -> tuple[NumericalGeometry | None, SelectionOutcome]:
    """The geometry on the exact quotient's pivot columns, or decline reason."""
    quotient, work = exact
    rtol, gmin, limit = controls
    asm_info, canon_builds, q_work, tracker = prep_ctx
    gram = ledger[0]
    k, p = design.n_slopes, design.n_columns
    kept_r = k + np.asarray(quotient.kept, dtype=np.intp)
    columns = np.concatenate([active, kept_r])
    zero_columns = set(ledger[2].exact_zero_columns)
    annihilated = tuple(c for levels in quotient.annihilated_levels for c in levels)
    expected_zeros = {k + c for c, is_ann in enumerate(annihilated) if is_ann}
    omitted = np.setdiff1d(np.arange(k), active, assume_unique=True)
    expected_zeros.update(omitted.tolist())
    # Direct sum with a zero block: the complement has exactly the same
    # positive spectrum and the discarded coordinate axes have zero residual.
    if omitted.size and (np.any(gram[omitted, :]) or np.any(gram[:, omitted])):
        return None, "zero_columns_mismatch"
    if columns.size == 0 or zero_columns != expected_zeros:
        return None, "zero_columns_mismatch"
    id_gram = gram[np.ix_(columns, columns)]
    try:
        proven = _candidate_spectrum_proven(
            design,
            gram,
            id_gram,
            quotient,
            (rtol, gmin),
            active=active,
            tracker=tracker,
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        prep = _make_selection_prep(
            "proof_refused",
            asm_info,
            canonical_builds=canon_builds,
            quotient_attempts=1,
            quotient_work=q_work,
            tracker=tracker,
        )
        _attach_selection_preparation(error, prep)
        raise
    if proven is None:
        return None, "proof_declined"
    try:
        basis, reduction, null_basis = _witness_maps(k, quotient, columns, p)
    except OverflowError:
        return None, "proof_declined"
    prep = _make_selection_prep(
        "selected",
        asm_info,
        canonical_builds=canon_builds,
        quotient_attempts=1,
        quotient_work=q_work,
        tracker=tracker,
    )
    passes = with_selection_preparation(ledger[2], prep)
    record = _selected_record(
        design,
        (gram, id_gram, basis, reduction, null_basis),
        (quotient, work, columns),
        (ledger[1], passes),
        (rtol, gmin, limit, proven[0], proven[1]),
    )
    return record, "selected"


def _selected_record(
    design: CenteredDesign,
    arrays: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray],
    exact: tuple[QuotientRank, WorkLog, IndexArray],
    ledger: tuple[GeometryBudget, GeometryPasses],
    controls: tuple[float, float, int, SelectedProof, NullProjection | None],
) -> NumericalGeometry:
    """Assemble the record of the selected-columns geometry.

    No eigendecomposition runs (``spectrum_provenance="certified_bounds"``):
    eigenvalue arrays are empty, ``threshold`` is ``gram_rtol * top_high``
    (``threshold_kind="gershgorin_upper_bound"``), and ``gap_ratio`` /
    ``retained_gap_ratio`` report the certified finite separation lower bounds
    (``gap_kind="certified_lower_bound"``, infinite only when the corresponding
    perturbation bound is zero or the kept/discarded spectrum is empty).
    """
    gram, id_gram, basis, reduction, null_basis = arrays
    quotient, work, columns = exact
    rtol, gmin, limit, proof, null_proj = controls
    k, r = design.n_slopes, len(quotient.kept)
    p = design.n_columns
    levels = p - k
    d = int(columns.size)
    sizes = design.structure.block_sizes
    kept_slopes = columns[columns < k]
    omitted = np.setdiff1d(np.arange(k), kept_slopes, assume_unique=True)
    col_map = np.concatenate([kept_slopes, np.full(r, -1, dtype=np.intp)])
    slope_alias = reduction[: kept_slopes.size, :k].copy()
    slope_identified = np.sum(null_basis[:k] ** 2, axis=1) <= rtol
    ann_blocks = np.array([rank == 0 for rank in quotient.block_ranks], dtype=bool)
    ann_levels = tuple(np.array(a, dtype=bool) for a in quotient.annihilated_levels)
    empty = np.zeros(0, dtype=np.float64)
    stored: tuple[np.ndarray, ...] = (
        gram,
        empty,
        empty,
        null_basis,
        basis,
        reduction,
        id_gram,
    )
    stored += (kept_slopes, omitted, col_map, slope_alias, slope_identified, columns)
    stored += (ann_blocks, *ann_levels)
    if null_proj is not None:
        stored += (
            null_proj.kept_indices,
            null_proj.dep_indices,
            null_proj.witnesses,
            null_proj.chol,
        )
    if not all(np.all(np.isfinite(array)) for array in stored):
        raise FloatingPointError("geometry map or diagnostics are not finite")
    gap_full = (
        _gap_lower_bound(proof.full_min_low, proof.delta_full)
        if 0 < d < p
        else float("inf")
    )
    gap_ret = (
        _gap_lower_bound(proof.ret_min_low, proof.delta_ret)
        if 0 < r < levels
        else float("inf")
    )
    return cast(
        NumericalGeometry,
        _build_record(
            NumericalGeometry,
            design=design,
            _gram=FloatStorage.store(gram),
            _eigenvalues=FloatStorage.store(empty),
            _retained_eigenvalues=FloatStorage.store(empty),
            _null_basis=FloatStorage.store(null_basis),
            _basis_matrix=FloatStorage.store(basis),
            _reduction_matrix=FloatStorage.store(reduction),
            _identified_gram=FloatStorage.store(id_gram),
            _kept_slope_indices=IndexStorage.store(kept_slopes),
            _omitted_slope_indices=IndexStorage.store(omitted),
            slope_omission_reasons=tuple(
                None if v else "centered_constant" for v in slope_identified
            ),
            _original_column_map=IndexStorage.store(col_map),
            _slope_alias_matrix=FloatStorage.store(slope_alias),
            _slope_identified_mask=BoolStorage.store(slope_identified),
            block_ranks=quotient.block_ranks,
            block_deficiencies=tuple(
                size - rank
                for size, rank in zip(sizes, quotient.block_ranks, strict=True)
            ),
            _annihilated_blocks=BoolStorage.store(ann_blocks),
            _annihilated_levels=tuple(BoolStorage.store(a) for a in ann_levels),
            cross_block_redundancy=sum(quotient.block_ranks) - r,
            gram_rtol=rtol,
            gap_min=gmin,
            threshold=float(np.nextafter(rtol * proof.top_high, np.inf)),
            gap_ratio=float(gap_full),
            retained_gap_ratio=float(gap_ret),
            nuisance_basis_kind="exact_selection",
            spectrum_provenance="certified_bounds",
            threshold_kind="gershgorin_upper_bound",
            gap_kind="certified_lower_bound",
            budget=ledger[0],
            passes=ledger[1],
            max_bytes=limit,
            retained_bytes=int(sum(array.nbytes for array in stored)),
            _quotient=(quotient, work.snapshot()),
            _selected_columns=IndexStorage.store(columns),
            _null_projection=null_proj,
            _selected_proof=proof,
        ),
    )


def _gap_lower_bound(smallest: float, discarded: float) -> float:
    """Round a positive separation bound downward; finite bounds stay finite."""
    if discarded == 0.0:
        return float("inf")
    ratio = min(_gap_ratio(smallest, discarded), float(np.finfo(np.float64).max))
    return float(np.nextafter(ratio, 0.0))
