"""Implicit B'B and prospective workspace ledger (design/03, Section 3.1).

The input design is already owned. Construction makes p forward and p transpose
operator calls, with only row vectors, never a retained indicator matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from powermean.fe._cancel_storage import BuilderRecord, FloatStorage, IndexArray
from powermean.fe.admission_records import AdmissionWork
from powermean.fe.cancel import CancelStructure, pair_positions
from powermean.fe.geometry_errors import GeometryBudgetError
from powermean.fe.geometry_storage import _build_record
from powermean.fe.retained import CenteredDesign
from powermean.model.matrix import symmetric_average
from powermean.model.variance import FloatArray, _finite_scalar

SelectionOutcome = Literal[
    "not_requested",
    "slopes_precheck_declined",
    "quotient_refused",
    "zero_columns_mismatch",
    "assembly_refused",
    "canonical_gram_refused",
    "proof_refused",
    "proof_declined",
    "selected",
]


@dataclass(slots=True)
class AssemblyWork:
    """Attempted assembly passes, retained even when the resulting Gram declines."""

    attempts: int = 0
    pairs: int = 0
    bincount_calls: int = 0
    unique_calls: int = 0

    def snapshot(self) -> tuple[int, int, int, int]:
        """Return attempts, generated pairs, bincounts and level scans."""
        return self.attempts, self.pairs, self.bincount_calls, self.unique_calls


@dataclass(frozen=True, eq=False, init=False)
class SelectionPreparation(BuilderRecord):
    """Fit-local accounting of attempted selected-columns preparation work.

    Records speculative pair-assembled Gram passes, exact-quotient attempts
    (including partial work on refusal), dense proof Cholesky/triangular-solve
    calls, Dekker/TwoSum compensated passes, exact split products, and per-block
    level bincounts
    whether selection succeeds or declines to canonical ``geometry_gram``.
    Construction counts denote invocations, including a refused construction.
    Expected terminal assembly/canonical/spectral refusals carry this same record
    as ``error.selection_preparation`` without replacing their type or message.
    """

    __slots__ = (
        "attempted",
        "n_assembled_bincount_calls",
        "n_assembled_gram_attempts",
        "n_assembled_pairs",
        "n_assembled_unique_calls",
        "n_canonical_gram_builds",
        "n_cholesky_calls",
        "n_compensated_matmul_cols",
        "n_compensated_twosum_calls",
        "n_operator_level_bincounts",
        "n_proof_attempts",
        "n_quotient_attempts",
        "n_residual_bound_passes",
        "n_slope_gram_builds",
        "n_split_products",
        "n_triangular_solve_calls",
        "outcome",
        "quotient_work",
    )

    attempted: bool
    outcome: SelectionOutcome
    n_assembled_gram_attempts: int
    n_assembled_pairs: int
    n_assembled_bincount_calls: int
    n_assembled_unique_calls: int
    n_canonical_gram_builds: int
    n_quotient_attempts: int
    quotient_work: AdmissionWork
    n_cholesky_calls: int
    n_triangular_solve_calls: int
    n_compensated_matmul_cols: int
    n_compensated_twosum_calls: int
    n_operator_level_bincounts: int
    n_slope_gram_builds: int
    n_proof_attempts: int
    n_residual_bound_passes: int
    n_split_products: int


@dataclass(frozen=True, eq=False, init=False)
class GeometryBudget(BuilderRecord):
    """Prospective byte budget for ``p``-dimensional Gram and basis construction.

    Computed before allocating the ``p`` by ``p`` Gram matrix, eigensolver workspaces,
    or ``p`` by ``d`` coordinate maps. These byte counts describe declared array and
    workspace allocations, not peak process memory.
    """

    __slots__ = (
        "basis_map_max_bytes",
        "eigensolver_bytes",
        "gram_bytes",
        "metadata_bytes",
        "n_columns_full",
        "n_groups",
        "n_retained_levels",
        "n_rows",
        "n_slopes",
        "original_check_bytes",
        "prospective_bytes",
        "row_work_bytes",
    )

    n_rows: int
    n_groups: int
    n_slopes: int
    n_retained_levels: int
    n_columns_full: int
    gram_bytes: int
    eigensolver_bytes: int
    basis_map_max_bytes: int
    row_work_bytes: int
    original_check_bytes: int
    metadata_bytes: int
    prospective_bytes: int


@dataclass(frozen=True, eq=False, init=False)
class GeometryPasses(BuilderRecord):
    """Structural pass and work-array accounting for implicit Gram assembly.

    Top-level fields count operator evaluations, reductions, and spectral
    deflation for the final stored ``G = B' B``; ``selection`` records fit-local
    attempted selected-columns preparation (including discarded assembled Gram
    and refused quotient work). This is structural accounting, not a speed claim.
    """

    __slots__ = (
        "eigh_complement_dimensions",
        "exact_zero_columns",
        "max_matrix_work_shape",
        "max_row_work_shape",
        "n_bincount_calls",
        "n_canonical_zero_axes",
        "n_center_sorted_calls",
        "n_columns_evaluated",
        "n_deflated_eigh_calls",
        "n_eigenvector_lift_values",
        "n_matvec_sorted_calls",
        "n_rmatvec_sorted_calls",
        "selection",
    )

    n_columns_evaluated: int
    n_matvec_sorted_calls: int
    n_rmatvec_sorted_calls: int
    n_center_sorted_calls: int
    n_bincount_calls: int
    max_row_work_shape: tuple[int, ...]
    max_matrix_work_shape: tuple[int, ...]
    exact_zero_columns: tuple[int, ...]
    eigh_complement_dimensions: tuple[int, ...]
    n_deflated_eigh_calls: int
    n_canonical_zero_axes: int
    n_eigenvector_lift_values: int
    selection: SelectionPreparation


def default_selection_preparation() -> SelectionPreparation:
    """Return the inactive fit-local selection preparation ledger."""
    return cast(
        SelectionPreparation,
        _build_record(
            SelectionPreparation,
            attempted=False,
            outcome="not_requested",
            n_assembled_gram_attempts=0,
            n_assembled_pairs=0,
            n_assembled_bincount_calls=0,
            n_assembled_unique_calls=0,
            n_canonical_gram_builds=0,
            n_quotient_attempts=0,
            quotient_work=AdmissionWork(()),
            n_cholesky_calls=0,
            n_triangular_solve_calls=0,
            n_compensated_matmul_cols=0,
            n_compensated_twosum_calls=0,
            n_operator_level_bincounts=0,
            n_slope_gram_builds=0,
            n_proof_attempts=0,
            n_residual_bound_passes=0,
            n_split_products=0,
        ),
    )


def with_selection_preparation(
    passes: GeometryPasses, selection: SelectionPreparation
) -> GeometryPasses:
    """Attach fit-local ``selection`` preparation counts to ``passes``."""
    return cast(
        GeometryPasses,
        _build_record(
            GeometryPasses,
            n_columns_evaluated=passes.n_columns_evaluated,
            n_matvec_sorted_calls=passes.n_matvec_sorted_calls,
            n_rmatvec_sorted_calls=passes.n_rmatvec_sorted_calls,
            n_center_sorted_calls=passes.n_center_sorted_calls,
            n_bincount_calls=passes.n_bincount_calls,
            max_row_work_shape=passes.max_row_work_shape,
            max_matrix_work_shape=passes.max_matrix_work_shape,
            exact_zero_columns=passes.exact_zero_columns,
            eigh_complement_dimensions=passes.eigh_complement_dimensions,
            n_deflated_eigh_calls=passes.n_deflated_eigh_calls,
            n_canonical_zero_axes=passes.n_canonical_zero_axes,
            n_eigenvector_lift_values=passes.n_eigenvector_lift_values,
            selection=selection,
        ),
    )


def _checked_max_bytes(max_bytes: object) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, np.integer)):
        raise TypeError("max_bytes must be an integer byte limit")
    limit = int(max_bytes)
    if limit < 0:
        raise ValueError("max_bytes must be nonnegative")
    return limit


def _checked_controls(
    gram_rtol: float, gap_min: float, max_bytes: int
) -> tuple[float, float, int]:
    rtol = _finite_scalar(gram_rtol, "gram_rtol")
    gmin = _finite_scalar(gap_min, "gap_min")
    if not (np.isfinite(rtol) and 0.0 < rtol < 1.0):
        raise ValueError("gram_rtol must be finite and strictly between 0 and 1")
    if not (np.isfinite(gmin) and gmin >= 1.0):
        raise ValueError("gap_min must be finite and >= 1.0")
    return rtol, gmin, _checked_max_bytes(max_bytes)


def geometry_budget(design: CenteredDesign) -> GeometryBudget:
    """Declare construction workspace before any Gram/basis allocation.

    Parameters
    ----------
    design : CenteredDesign
        Already owned input; its storage is excluded from this incremental ledger.

    Notes
    -----
    With p = K + sum(levels), reserve one p-square Gram, 24 p-square
    float64 slots for simultaneously live maps, full/retained eigenvectors,
    Schur/projector/product temporaries and record copies, and eight p-square
    slots plus 16p+16 scalars for a sequential eigensolve/solve/QR workspace.
    The latter includes a declared LAPACK work/copy allowance (DSYEVD uses
    2p^2+6p+1 doubles and 5p+3 integers); vendor-internal overhead is excluded.
    Eight row vectors, four group vectors and eight p-vectors cover operator
    temporaries. 32p+16 scalar slots cover spectra, masks, indices and their
    finalization copies. Each block has >=1 level, so block count <= p.

    Witnessed-zero deflation stays inside these unchanged allowances. Its reduced
    principal matrix, reduced eigenvectors, LAPACK matrix copy and 2p-square
    DSYEVD work use at most five of the eight eigensolver matrix slots. The
    directly lifted output uses the existing full/retained eigenvector slot;
    no p-square sorted permutation copy is formed. Active/canonical/output
    indices, mask and spectra fit in the existing metadata allowance. The fresh
    witness tuple and structural counters are Python metadata, excluded as below.

    The Gram plus 24 matrix slots cover this managed-array sum: Gram (1), full
    eigenvectors (1), nullspace (1), retained eigenvectors/selection/basis (3),
    slope Schur/projection/solve (4), maps including slope aliases (4),
    products/projector/sign or QR intermediates (5), and immutable
    finalization (6). On ``exact_selection``, small-side null-projection factors
    (``r``-by-``m`` witnesses and ``min(r, m)``-square Cholesky) and proof
    pencils reuse unused eigenvector/projector ``p^2`` slots, and selected/null
    index vectors fit in ``metadata_bytes``. Each is at most p^2;
    mutually exclusive branches are deliberately summed. No input design,
    caller-owned custom basis, Python object/header or peak process memory
    is claimed here. Custom conversion is bounded by checking ndarray shape
    before conversion. Action/state allocations have a separate lifetime.
    The additional original_check_bytes reserve covers eight n-by-K buffers
    (conversion, raw snapshot, sorted rows, group means, gathers and comparisons),
    eight row vectors, eight
    p-square verification/finalization slots and eight p-vectors. Checks never
    form an n-by-retained-level or n-by-identified design. The original owned
    design and Python objects remain excluded.
    """
    if type(design) is not CenteredDesign:
        raise TypeError("design must come from centered_design")
    structure = design.structure
    n = structure.n_rows
    g = structure.n_groups
    k = design.n_slopes
    levels = sum(structure.block_sizes)
    p = design.n_columns
    gram_bytes = 8 * p * p
    eigensolver_bytes = 8 * (8 * p * p + 16 * p + 16)
    basis_map_max_bytes = 8 * 24 * p * p
    row_work_bytes = 8 * (8 * n + 4 * g + 8 * p)
    # Raw columns, sorted/centered copies, exact binding comparisons and
    # immutable finalization coexist at the raw-input boundary (n by K only).
    # Original-space checks stream one column/direction at a time.
    original_check_bytes = 8 * (8 * n * k + 8 * n + 8 * p * p + 8 * p)
    metadata_bytes = 8 * (32 * p + 16)
    prospective = (
        gram_bytes
        + eigensolver_bytes
        + basis_map_max_bytes
        + row_work_bytes
        + original_check_bytes
        + metadata_bytes
    )
    return cast(
        GeometryBudget,
        _build_record(
            GeometryBudget,
            n_rows=n,
            n_groups=g,
            n_slopes=k,
            n_retained_levels=levels,
            n_columns_full=p,
            gram_bytes=gram_bytes,
            eigensolver_bytes=eigensolver_bytes,
            basis_map_max_bytes=basis_map_max_bytes,
            row_work_bytes=row_work_bytes,
            original_check_bytes=original_check_bytes,
            metadata_bytes=metadata_bytes,
            prospective_bytes=prospective,
        ),
    )


def symmetric_gram(matrix: FloatArray) -> FloatArray:
    """Symmetrize a Gram in one pass unless paired entries overflow their sum."""
    try:
        with np.errstate(over="raise", under="ignore"):
            return 0.5 * (matrix + matrix.T)
    except FloatingPointError:
        return symmetric_average(matrix)


def geometry_gram(
    design: CenteredDesign, *, max_bytes: int
) -> tuple[FloatArray, GeometryBudget, GeometryPasses]:
    """Assemble ``G = B' B`` implicitly after checking the byte reservation.

    Parameters
    ----------
    design : CenteredDesign
        Owned full-column operator, with p = K + sum(levels).
    max_bytes : int
        Explicit task-local construction reservation limit.

    Returns
    -------
    gram, budget, passes : tuple
        Immutable p-square Gram, declared workspace ledger and operator counts.
    """
    budget = reserve_geometry(design, max_bytes)
    p = budget.n_columns_full
    n_blocks = len(design.structure.block_sizes)
    gram = np.zeros((p, p), dtype=np.float64)
    unit = np.zeros(p, dtype=np.float64)
    zero_columns: list[int] = []
    for col in range(p):
        unit[col] = 1.0
        u_sorted = design.matvec_sorted(unit)
        if not np.any(u_sorted):
            zero_columns.append(col)
        gram[:, col] = design.rmatvec_sorted(u_sorted)
        unit[col] = 0.0
    gram = symmetric_gram(gram)
    if zero_columns:
        gram[zero_columns, :] = 0.0
        gram[:, zero_columns] = 0.0
    if not np.all(np.isfinite(gram)):
        raise FloatingPointError("Gram matrix entries are not finite")
    passes = cast(
        GeometryPasses,
        _build_record(
            GeometryPasses,
            n_columns_evaluated=p,
            n_matvec_sorted_calls=p,
            n_rmatvec_sorted_calls=p,
            n_center_sorted_calls=(2 * p if n_blocks > 0 else 0),
            n_bincount_calls=p * n_blocks,
            max_row_work_shape=(budget.n_rows,),
            max_matrix_work_shape=(p, p),
            exact_zero_columns=tuple(zero_columns),
            eigh_complement_dimensions=(),
            n_deflated_eigh_calls=0,
            n_canonical_zero_axes=0,
            n_eigenvector_lift_values=0,
            selection=default_selection_preparation(),
        ),
    )
    return FloatStorage.store(gram).view(), budget, passes


def _constant_levels(
    design: CenteredDesign, work: AssemblyWork | None = None
) -> IndexArray:
    """The full columns of retained levels constant within every group they meet.

    A level is so when, in every group with one of its rows, all the group's rows
    have it; its centered column is then zero exactly, as is a slope column all of
    whose centered values are zero. Decided from the integer codes, by the count of
    every (group, level) cell that occurs, in memory linear in the rows.
    """
    structure = design.structure
    groups, counts = structure.sorted_codes, structure.counts
    found = [int(j) for j in np.flatnonzero(~np.any(design.centered_slopes, axis=0))]
    for codes, start, size in zip(
        structure.retained_sorted_codes,
        design.block_starts,
        structure.block_sizes,
        strict=True,
    ):
        if work is not None:
            work.unique_calls += 1
        cells, seen = np.unique(
            groups.astype(np.int64) * size + codes, return_counts=True
        )
        partial = np.zeros(size, dtype=bool)
        partial[(cells % size)[seen != counts[cells // size]]] = True
        found.extend(start + int(level) for level in np.flatnonzero(~partial))
    return np.array(found, dtype=np.intp)


def _fill_retained_gram(
    structure: CancelStructure,
    blocks: tuple[tuple[IndexArray, int, int], ...],
    gram: FloatArray,
    work: AssemblyWork | None = None,
) -> None:
    """Accumulate symmetric ``F_R' M_C F_R`` blocks from ordered within-group pairs."""
    left, right = pair_positions(structure)
    if work is not None:
        work.pairs += int(left.size)
    counts = structure.counts
    weight = (
        np.where(left == right, 1.0, 0.0) - 1.0 / counts[structure.sorted_codes[left]]
    )
    for b1, (rows, row, n_rows) in enumerate(blocks):
        scaled = rows[left]
        for b2 in range(b1, len(blocks)):
            cols, col, n_cols = blocks[b2]
            if work is not None:
                work.bincount_calls += 1
            block_mat = np.bincount(
                scaled * n_cols + cols[right],
                weights=weight,
                minlength=n_rows * n_cols,
            ).reshape(n_rows, n_cols)
            gram[row : row + n_rows, col : col + n_cols] = block_mat
            if b1 != b2:
                gram[col : col + n_cols, row : row + n_rows] = block_mat.T


def assembled_gram(
    design: CenteredDesign, *, max_bytes: int, work: AssemblyWork | None = None
) -> tuple[FloatArray, GeometryBudget, GeometryPasses] | None:
    """``G = B' B`` summed from the rows of each group, after the same reservation.

    ``X'X`` from the centered slopes, ``F_R' M_C X`` by level sums of them, and
    ``F_R' M_C F_R`` with the entries ``[i = j] - 1/c`` of ``M_C`` over every
    ordered pair of rows of a group: the entries of :func:`geometry_gram` with the
    sums in another order, so equal to rounding. The columns of levels constant
    within their groups are exact zeros (:func:`_constant_levels`), as there.
    None where the pairs of rows number more than ``p**2`` (keeping pair buffers
    inside the already reserved ``p**2`` slots of :func:`geometry_budget`).
    """
    budget = reserve_geometry(design, max_bytes)
    if work is not None:
        work.attempts += 1
    structure, x = design.structure, design.centered_slopes
    p, k = design.n_columns, design.n_slopes
    counts = structure.counts
    if float(np.dot(counts, counts.astype(np.float64))) > float(p) * p:
        return None
    n_pairs = int(np.dot(counts.astype(np.int64), counts.astype(np.int64)))
    gram = np.zeros((p, p), dtype=np.float64)
    gram[:k, :k] = x.T @ x
    blocks = tuple(
        zip(
            structure.retained_sorted_codes,
            design.block_starts,
            structure.block_sizes,
            strict=True,
        )
    )
    for codes, start, size in blocks:
        for column in range(k):
            if work is not None:
                work.bincount_calls += 1
            gram[start : start + size, column] = np.bincount(
                codes, weights=x[:, column], minlength=size
            )
    gram[:k, k:] = gram[k:, :k].T
    if blocks:
        _fill_retained_gram(structure, blocks, gram, work)
    gram = symmetric_gram(gram)
    zero_columns = _constant_levels(design, work)
    gram[zero_columns, :] = 0.0
    gram[:, zero_columns] = 0.0
    if not np.all(np.isfinite(gram)):
        return None
    n_blocks = len(blocks)
    passes = cast(
        GeometryPasses,
        _build_record(
            GeometryPasses,
            n_columns_evaluated=p,
            n_matvec_sorted_calls=0,
            n_rmatvec_sorted_calls=0,
            n_center_sorted_calls=0,
            n_bincount_calls=k * n_blocks + n_blocks * (n_blocks + 1) // 2,
            max_row_work_shape=(n_pairs,),
            max_matrix_work_shape=(p, p),
            exact_zero_columns=tuple(zero_columns.tolist()),
            eigh_complement_dimensions=(),
            n_deflated_eigh_calls=0,
            n_canonical_zero_axes=0,
            n_eigenvector_lift_values=0,
            selection=default_selection_preparation(),
        ),
    )
    return FloatStorage.store(gram).view(), budget, passes


def spectral_passes(passes: GeometryPasses, n_slopes: int) -> GeometryPasses:
    """Record completed full/retained deflation work, not estimated vendor work.

    Called only after both spectral constructions returned. An empty complement
    has canonical lifts but no numerical eigh call. Existing operator work stays
    exact; only the same-construction stream supplied the witness indices.
    """
    p = passes.n_columns_evaluated
    zeros = passes.exact_zero_columns
    sizes = ((p, len(zeros)), (p - n_slopes, sum(j >= n_slopes for j in zeros)))
    deflated = tuple((size, count) for size, count in sizes if count)
    dimensions = tuple(size - count for size, count in deflated)
    return cast(
        GeometryPasses,
        _build_record(
            GeometryPasses,
            n_columns_evaluated=p,
            n_matvec_sorted_calls=passes.n_matvec_sorted_calls,
            n_rmatvec_sorted_calls=passes.n_rmatvec_sorted_calls,
            n_center_sorted_calls=passes.n_center_sorted_calls,
            n_bincount_calls=passes.n_bincount_calls,
            max_row_work_shape=passes.max_row_work_shape,
            max_matrix_work_shape=passes.max_matrix_work_shape,
            exact_zero_columns=zeros,
            eigh_complement_dimensions=dimensions,
            n_deflated_eigh_calls=sum(dimension > 0 for dimension in dimensions),
            n_canonical_zero_axes=sum(count for _, count in deflated),
            n_eigenvector_lift_values=sum(
                (size - count) ** 2 + count for size, count in deflated
            ),
            selection=passes.selection,
        ),
    )


def reserve_geometry(design: CenteredDesign, max_bytes: int) -> GeometryBudget:
    """Check the whole p-based reservation before arrays or operator work."""
    limit = _checked_max_bytes(max_bytes)
    budget = geometry_budget(design)
    if budget.prospective_bytes > limit:
        raise GeometryBudgetError(
            budget.prospective_bytes,
            limit,
            n_rows=budget.n_rows,
            n_slopes=budget.n_slopes,
            n_retained_levels=budget.n_retained_levels,
        )
    return budget
