"""The fresh observed geometry of an admitted absorbed point: raw S, H and scaling.

Internal module.  Mathematically ``B = [X, E_J]`` (``X`` the sample's scaled slopes,
``E_J`` the kept incidence of :mod:`powermean.fe.certificate_basis`), ``S = B'b`` and
``H = B' diag(C) B`` from the fresh rows; ``B`` is never formed.

Assembly.  Every ordered entry is the row-order sequential binary64 sum of the
accepted checked reduction (:func:`~powermean.fe.absorb_numeric.group_sums`) of the
ordered products ``B_ij (C_i B_ik)``, each product checked (its range refusals
kept): slope-slope products, slope-level and level-slope sums, the level totals of
``C`` and, for each pair of blocks, the sums over the nonempty level pairs, whose keys
are code tuples sorted lexicographically (no overflowing scalar key or ``L_g L_h``
allocation).  Both orientations of a level pair sum the products ``C_i`` of the
same rows in the same order, so each unordered pair of blocks is aggregated once
and every such sum is stored at both orientations: that one row-order sum, not an
average or a symmetry repair.  ``S`` holds the slope sums of ``X_ij b_i`` and the
level sums of ``b``.  Both stay RAW sums, never divided by the rows or the weight
total.  ``H`` has owned lossless numeric partitions when a categorical principal block
exists, with canonical rectangular CSC couplings; otherwise it is canonical CSC.
Full CSC access remains lossless and detached. Each representation has no
duplicate, exact zeros omitted.

Scaling.  ``s_j = sqrt(max(rowmax_j |H|, colmax_j |H|))``; a zero or nonfinite scale
refuses (``ValueError``).  The ordinary ``A_ij = (H_ij / s_i) / s_j`` keeps that order
of operations for the observed-``H`` actions, and ``||A||_1`` is the largest column sum
of ``|A|`` in CSC order.  The curvature metric is separate: for each unordered pair
``j <= k`` the range-safe mean ``hplus`` of ``(H_jk, H_kj)`` (missing entries zero:
``(x + y)/2`` where the sum is finite, else ``x/2 + y/2``) is scaled ONCE as
``(hplus / s_j) / s_k`` and the identical bits copied to both positions of ``Aplus``.
The canonical ``H``, whose pattern is symmetric (checked afresh), pairs each stored
entry with its stored transpose and gives ``Aplus`` its structure; any other record
is paired by sorting its unordered pairs.  No symmetry tolerance, repair, floor or
rescaling of the raw weights.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, overload

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.sample_fe_source import Derived
from powermean.fe._cancel_storage import IndexStorage
from powermean.fe import absorb_numeric as _numeric
from powermean.fe import reductions as _reductions
from powermean.fe.absorb_numeric import checked_product, group_sums
from powermean.fe.certificate_basis import CertificateBasis, SupportCheck
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_point import (
    AdmittedPoint,
    FreshRows,
    WorkStep,
)
from powermean.solve.absorbed_pair_layout import native_pair_inputs, native_pair_layout
from powermean.solve.compact_assembly import (
    CompactEntries,
    JointLayout,
    _JointStructure,
    _PreparingEntries,
    _ValueEntries,
    kept_block,
    original_entries,
)
from powermean.solve.compact_matrix import CompactMatrix, matrix_nnz
from powermean.solve.compact_scaling import equilibration_parts, one_norm_parts
from powermean.solve.sparse_matrix import SparseMatrix

ZERO_SCALE = "zero row/column in the Hessian"


@dataclass(frozen=True, eq=False)
class FEGeometry:
    """The fresh geometry of one admitted point, bound to its sample and coordinates.

    ``point`` (sample, family, owned theta/base/q, inner controls), ``rows`` (fresh
    ``b``, ``C``, raw weights), ``basis`` (``J``, witnesses), ``support`` (``J`` on the
    positive-curvature rows), ``design`` the owned scaled slopes, ``moment`` ``S``,
    ``hessian`` ``H``; ``scale`` ``s``, ``ordinary`` ``A``, ``norm`` ``||A||_1`` and
    ``curvature`` ``Aplus`` are ``None`` (``norm`` NaN) after a scaling refusal, whose
    reason is ``refusal``.
    """

    point: AdmittedPoint
    rows: FreshRows
    basis: CertificateBasis
    support: SupportCheck
    design: FloatArray
    moment: FloatArray
    hessian: SparseMatrix | CompactMatrix
    scale: FloatArray | None
    ordinary: SparseMatrix | CompactMatrix | None
    norm: float
    curvature: SparseMatrix | CompactMatrix | None
    refusal: str

    @property
    def n_slopes(self) -> int:
        """The number of slopes ``K``."""
        return int(self.design.shape[1])

    @property
    def dimension(self) -> int:
        """``K + |J|``."""
        return self.n_slopes + self.basis.n_kept

    def binding(self) -> tuple[tuple[str, object], ...]:
        """The binding by value: sample and grouping identities, every array's bytes."""
        point, rows, h = self.point, self.rows, self.hessian
        sample = point.sample
        arrays: dict[str, FloatArray | IndexArray | None] = {
            "positions": sample.positions,
            "response": sample.problem.response,
            "offset": sample.problem.offset,
            "weights": rows.weights,
            "theta": point.theta,
            "base": point.base,
            "q": point.q,
            "design": self.design,
            "J": self.basis.selected,
            "b": rows.b,
            "C": rows.curvature,
            "S": self.moment,
            "s": self.scale,
        }
        found: list[tuple[str, object]] = [(k, _bytes(v)) for k, v in arrays.items()]
        if isinstance(h, CompactMatrix):
            found.append(("H partition", h.binding()))
        else:
            found.extend((("H", _bytes(h.data)), ("H rows", _bytes(h.indices)),
                          ("H columns", _bytes(h.indptr))))
        weights = sample.data.weights
        found += [
            ("sample", sample),
            ("grouping", sample.grouping),
            ("family", point.variance),
            ("omega", sample.omega),
            ("names", sample.active_names),
            ("kind", None if weights is None else weights.kind),
            ("frequency", rows.frequency),
            ("references", tuple(_bytes(r) for r in self.basis.references)),
        ]
        return tuple(found)

    def steps(self) -> tuple[WorkStep, ...]:
        """The work of the scaling: completed with its counts, or its refusal."""
        ordinary, curvature = self.ordinary, self.curvature
        if self.refusal or ordinary is None or curvature is None:
            return (WorkStep("equilibration", "refused", self.refusal, None),)
        counters = (
            ("scales", self.dimension),
            ("A nonzeros", matrix_nnz(ordinary)),
            ("Aplus nonzeros", matrix_nnz(curvature)),
        )
        return (WorkStep("equilibration", "completed", "", counters),)


def _bytes(
    values: FloatArray | IndexArray | None,
) -> tuple[str, tuple[int, ...], bytes] | None:
    """An array by value (dtype, shape and bytes), or ``None``."""
    if values is None:
        return None
    return (values.dtype.str, values.shape, values.tobytes())


# ------------------------------------------------------------------- assembly


def _total(values: FloatArray) -> FloatArray:
    """The row-order sequential sum of every column of ``values`` (checked)."""
    single = np.zeros(values.shape[0], dtype=np.intp)
    found: FloatArray = group_sums(values, single, 1)[0]
    return found


def moment(
    design: FloatArray, values: FloatArray, basis: CertificateBasis
) -> FloatArray:
    """``B'v``: the slope sums of ``X_ij v_i``, then each kept level's sum of ``v``."""
    k = design.shape[1]
    found = np.zeros(k + basis.n_kept)
    if k:
        found[:k] = _total(checked_product(design, values[:, None], "moment term"))
    for block, position in zip(basis.grouping.blocks, basis.positions, strict=True):
        sums = group_sums(values, block.codes, block.n_levels)
        kept = position >= 0
        found[k + position[kept]] = sums[kept]
    return immutable_array(found)


class _EntrySink(Protocol):
    """Accept each distinct already-summed Hessian entry batch."""

    def add(self, rows: IndexArray, cols: IndexArray, values: FloatArray) -> None:
        """Retain a checked batch in its chosen numeric representation."""
        ...


class _Entries:
    """The ordered entries of ``H`` as they are summed: rows, columns, values."""

    def __init__(self) -> None:
        self.rows: list[IndexArray] = []
        self.cols: list[IndexArray] = []
        self.values: list[FloatArray] = []

    def add(self, rows: IndexArray, cols: IndexArray, values: FloatArray) -> None:
        """Keep a batch of entries."""
        self.rows.append(np.asarray(rows, dtype=np.intp))
        self.cols.append(np.asarray(cols, dtype=np.intp))
        self.values.append(np.asarray(values, dtype=np.float64))


def _slope_values(
    design: FloatArray, curvature: FloatArray, k: int,
) -> Iterator[FloatArray]:
    """The original checked slope products/reductions, one column at a time."""
    for column in range(k):
        inner = checked_product(curvature, design[:, column], "curvature product")
        sums = _total(checked_product(design, inner[:, None], "Hessian term"))
        yield sums


def _slope_entries(
    design: FloatArray, curvature: FloatArray, entries: _EntrySink
) -> None:
    """``H_jk = sum_i X_ij (C_i X_ik)`` for every ordered pair of slopes."""
    k = design.shape[1]
    for column, sums in enumerate(_slope_values(design, curvature, k)):
        entries.add(np.arange(k), np.full(k, column), sums)


def _mixed_values(
    design: FloatArray,
    curvature: FloatArray,
    basis: CertificateBasis,
    k: int,
    selected: tuple[tuple[IndexStorage, IndexStorage], ...] | None = None,
    *, shared: bool = False,
) -> Iterator[tuple[IndexArray, int, int, FloatArray]]:
    """Original orientations, sharing only inside an original owned consumer.

    Binary64 multiplication commutes, including the same range refusal and signed
    zero. The owned consumer uses one checked product and identical row-order sums;
    generic/injected calls retain both products and both reducer invocations.
    """
    slope_level = checked_product(design, curvature[:, None], "Hessian term")
    level_slope = (slope_level if shared else
                   checked_product(curvature[:, None], design, "Hessian term"))
    for number, (block, position) in enumerate(zip(
        basis.grouping.blocks, basis.positions, strict=True,
    )):
        if selected is None:
            kept = np.flatnonzero(position >= 0)
            index = k + position[kept]
        else:
            kept, index = (part.view() for part in selected[number])
        if k:
            sums = group_sums(slope_level, block.codes, block.n_levels)[kept]
            yield index, kept.size, 0, sums
            if not shared:
                sums = group_sums(level_slope, block.codes, block.n_levels)[kept]
            yield index, kept.size, 1, sums
        totals = group_sums(curvature, block.codes, block.n_levels)
        yield index, kept.size, 2, totals[kept]


def _mixed_entries(
    design: FloatArray, curvature: FloatArray, basis: CertificateBasis,
    entries: _EntrySink, *, shared: bool = False,
) -> None:
    """Original dynamic coordinates and callback order for every fresh batch."""
    k = design.shape[1]
    batches = (_mixed_values(design, curvature, basis, k, shared=True) if shared
               else _mixed_values(design, curvature, basis, k))
    for index, count, direction, values in batches:
        if direction == 0:
            entries.add(
                np.tile(np.arange(k), count), np.repeat(index, k), values.ravel(),
            )
        elif direction == 1:
            entries.add(
                np.repeat(index, k), np.tile(np.arange(k), count), values.ravel(),
            )
        else:
            entries.add(index, index, values)


def _pair_entries(
    curvature: FloatArray, basis: CertificateBasis, k: int, entries: _EntrySink
) -> None:
    """``sum C_i`` over the rows of each nonempty level pair of two distinct blocks.

    Each unordered pair of blocks ``g < h`` is aggregated once.  A level pair holds
    the same rows, in the same order, in both orientations, so its one row-order sum
    is the entry at ``(g, h)`` and, transposed, at ``(h, g)``.
    """
    blocks, positions = basis.grouping.blocks, basis.positions
    for g, first in enumerate(blocks):
        for h in range(g + 1, len(blocks)):
            native = native_pair_inputs(blocks, first, h)
            keys = np.column_stack([first.codes, blocks[h].codes])
            layout = (
                native_pair_layout(keys, first.codes, blocks[h].codes)
                if native
                else None
            )
            pairs, inverse = (
                np.unique(keys, axis=0, return_inverse=True)
                if layout is None
                else layout
            )
            sums = group_sums(curvature, inverse.reshape(-1), int(pairs.shape[0]))
            rows, cols = positions[g][pairs[:, 0]], positions[h][pairs[:, 1]]
            kept = (rows >= 0) & (cols >= 0)
            rows, cols, values = k + rows[kept], k + cols[kept], sums[kept]
            entries.add(rows, cols, values)
            entries.add(cols, rows, values)


def canonical(
    rows: IndexArray, cols: IndexArray, values: FloatArray, size: int
) -> SparseMatrix:
    """The owned canonical CSC of distinct entries, exact zeros omitted.

    ``ValueError`` for a repeated entry: each ordered entry is summed once.
    """
    keep = values != 0
    rows, cols, values = rows[keep], cols[keep], values[keep]
    order = np.lexsort((rows, cols))
    rows, cols, values = rows[order], cols[order], values[order]
    if np.any((np.diff(rows) == 0) & (np.diff(cols) == 0)):
        raise ValueError("an ordered entry of the Hessian was summed twice")
    indptr = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=size))))
    return _owned(size, indptr, rows, values)


def _owned(
    size: int, indptr: IndexArray, rows: IndexArray, values: FloatArray
) -> SparseMatrix:
    """The owned read-only CSC record of sorted, distinct, nonzero entries."""
    return SparseMatrix(
        (size, size),
        frozen_index(np.asarray(indptr, dtype=np.intp)),
        frozen_index(np.asarray(rows, dtype=np.intp)),
        immutable_array(values),
    )


def hessian(
    design: FloatArray, curvature: FloatArray, basis: CertificateBasis
) -> SparseMatrix:
    """``H = B' diag(C) B`` in owned canonical CSC, every ordered entry summed once."""
    k = design.shape[1]
    entries = _Entries()
    _slope_entries(design, curvature, entries)
    _mixed_entries(design, curvature, basis, entries)
    _pair_entries(curvature, basis, k, entries)
    rows = np.concatenate([np.zeros(0, dtype=np.intp), *entries.rows])
    cols = np.concatenate([np.zeros(0, dtype=np.intp), *entries.cols])
    values = np.concatenate([np.zeros(0), *entries.values])
    return canonical(rows, cols, values, k + basis.n_kept)


# -------------------------------------------------------------------- scaling


def equilibration(matrix: SparseMatrix | CompactMatrix) -> FloatArray:
    """``s_j = sqrt(max(rowmax_j, colmax_j))`` of ``|H|``; a zero scale refuses."""
    if isinstance(matrix, CompactMatrix):
        return equilibration_parts(matrix.parts)
    size = matrix.shape[0]
    absolute = np.abs(matrix.data)
    rowmax, colmax = np.zeros(size), np.zeros(size)
    np.maximum.at(rowmax, matrix.indices, absolute)
    np.maximum.at(colmax, matrix.columns, absolute)
    scale = np.sqrt(np.maximum(rowmax, colmax))
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError(ZERO_SCALE)
    return immutable_array(scale)


@overload
def scaled_ordinary(matrix: SparseMatrix, scale: FloatArray) -> SparseMatrix: ...


@overload
def scaled_ordinary(matrix: CompactMatrix, scale: FloatArray) -> CompactMatrix: ...


def scaled_ordinary(
    matrix: SparseMatrix | CompactMatrix, scale: FloatArray
) -> SparseMatrix | CompactMatrix:
    """``A_ij = (H_ij / s_i) / s_j`` entry by entry, the structure of ``H``."""
    if isinstance(matrix, CompactMatrix):
        found = matrix.scaled(scale, "ordinary")
        _ = found.parts
        return found
    with np.errstate(all="ignore"):
        data = (matrix.data / scale[matrix.indices]) / scale[matrix.columns]
    if not np.isfinite(data).all():
        raise FloatingPointError("the scaled Hessian is not representable")
    return SparseMatrix(
        matrix.shape, matrix.indptr, matrix.indices, immutable_array(data)
    )


def one_norm(matrix: SparseMatrix | CompactMatrix) -> float:
    """``||A||_1``: the largest column sum of ``|A|``, each summed in CSC order."""
    if isinstance(matrix, CompactMatrix):
        return one_norm_parts(matrix.parts)
    sums = np.zeros(matrix.shape[1])
    with np.errstate(all="ignore"):
        np.add.at(sums, matrix.columns, np.abs(matrix.data))
    norm = float(np.max(sums)) if sums.size else 0.0
    if not np.isfinite(norm):
        raise FloatingPointError("the one-norm of the scaled Hessian is not finite")
    return norm


def symmetric_mean(x: FloatArray, y: FloatArray) -> FloatArray:
    """``(x + y)/2`` where the sum is finite, else ``x/2 + y/2`` (range safe)."""
    with np.errstate(over="ignore", under="ignore"):
        total = x + y
        found: FloatArray = np.where(np.isfinite(total), total / 2, x / 2 + y / 2)
    return found


def _transposes(matrix: SparseMatrix, scale: FloatArray) -> IndexArray | None:
    """The stored position of every entry's transpose, or ``None``.

    Checked afresh from the values at every call, never from the record's origin:
    an exact :class:`SparseMatrix` of plain ``intp`` and ``float64`` vectors in
    canonical CSC (``indptr`` from 0 to the stored count, nondecreasing; rows
    strictly increasing in each column) whose pattern is symmetric, and a plain
    ``float64`` scale of its size.  The strictly lower entries in row order are then
    the strictly upper ones in column order, transposed; every row is then also a
    stored column, so in range.
    """
    if type(matrix) is not SparseMatrix or type(scale) is not np.ndarray:
        return None
    shape, indptr, rows, data = matrix.shape, matrix.indptr, matrix.indices, matrix.data
    if not (
        type(shape) is tuple
        and len(shape) == 2
        and type(shape[0]) is int
        and type(shape[1]) is int
        and shape[0] == shape[1]
        and all(type(a) is np.ndarray and a.ndim == 1 for a in (indptr, rows, data))
        and indptr.dtype == rows.dtype == np.intp
        and data.dtype == scale.dtype == np.float64
        and indptr.size == shape[0] + 1
        and scale.shape == (shape[0],)
        and 0 < rows.size == data.size
        and indptr[0] == 0
        and indptr[-1] == rows.size
    ):
        return None
    if np.any(np.diff(indptr) < 0):
        return None
    cols = matrix.columns
    if np.any((np.diff(rows) <= 0) & (cols[1:] == cols[:-1])):
        return None
    above, below = np.flatnonzero(rows < cols), np.flatnonzero(rows > cols)
    below = below[np.argsort(rows[below], kind="stable")]
    if not (
        above.size == below.size
        and np.array_equal(rows[above], cols[below])
        and np.array_equal(cols[above], rows[below])
    ):
        return None
    transpose = np.arange(rows.size, dtype=np.intp)
    transpose[above], transpose[below] = below, above
    return transpose


@overload
def scaled_curvature(matrix: SparseMatrix, scale: FloatArray) -> SparseMatrix: ...


@overload
def scaled_curvature(matrix: CompactMatrix, scale: FloatArray) -> CompactMatrix: ...


def scaled_curvature(
    matrix: SparseMatrix | CompactMatrix, scale: FloatArray
) -> SparseMatrix | CompactMatrix:
    """``Aplus``: each unordered pair's ``(hplus / s_lo) / s_hi`` at both positions.

    A canonical CSC of symmetric pattern (:func:`_transposes`), as the observed ``H``
    is, pairs each stored ``(j, k)``, ``j <= k``, with its stored transpose, and
    ``Aplus`` keeps its structure, exact zeros omitted.  Any other record is paired
    by :func:`_grouped_curvature`.
    """
    if isinstance(matrix, CompactMatrix):
        found = matrix.scaled(scale, "curvature")
        _ = found.parts
        return found
    transpose = _transposes(matrix, scale)
    if transpose is None:
        return _grouped_curvature(matrix, scale)
    rows, cols, data = matrix.indices, matrix.columns, matrix.data
    upper = np.flatnonzero(rows <= cols)
    lower = transpose[upper]
    with np.errstate(all="ignore"):
        hplus = symmetric_mean(data[upper], data[lower])
        values = (hplus / scale[rows[upper]]) / scale[cols[upper]]
    if not np.isfinite(values).all():
        raise FloatingPointError("the scaled curvature is not representable")
    mirrored = np.empty(data.size)
    mirrored[upper], mirrored[lower] = values, values
    keep = mirrored != 0
    indptr = np.concatenate(([0], np.cumsum(keep)))[matrix.indptr]
    return _owned(matrix.shape[0], indptr, rows[keep], mirrored[keep])


def _grouped_curvature(matrix: SparseMatrix, scale: FloatArray) -> SparseMatrix:
    """``Aplus`` of any record: its entries sorted by unordered pair, then canonical."""
    rows, cols, data = matrix.indices, matrix.columns, matrix.data
    lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
    order = np.lexsort((hi, lo))
    lo_s, hi_s = lo[order], hi[order]
    start = np.ones(lo_s.size, dtype=bool)
    start[1:] = (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])
    group = np.cumsum(start) - 1
    x, y = np.zeros(int(start.sum())), np.zeros(int(start.sum()))
    upper, lower = (rows <= cols)[order], (rows >= cols)[order]
    x[group[upper]] = data[order][upper]
    y[group[lower]] = data[order][lower]
    first, second = lo_s[start], hi_s[start]
    with np.errstate(all="ignore"):
        values = (symmetric_mean(x, y) / scale[first]) / scale[second]
    if not np.isfinite(values).all():
        raise FloatingPointError("the scaled curvature is not representable")
    off = first != second
    all_rows = np.concatenate([first, second[off]])
    all_cols = np.concatenate([second, first[off]])
    return canonical(
        all_rows, all_cols, np.concatenate([values, values[off]]), matrix.shape[0]
    )


# ------------------------------------------------------------------- geometry


@dataclass(frozen=True, eq=False)
class Assembly:
    """The raw sums of one admitted point: the owned slopes ``X``, ``S`` and ``H``."""

    design: FloatArray
    moment: FloatArray
    hessian: SparseMatrix | CompactMatrix


def assembled(
    point: AdmittedPoint, rows: FreshRows, basis: CertificateBasis
) -> Assembly:
    """``S`` and ``H`` of the fresh rows on the fixed ``J`` (``FloatingPointError``)."""
    if basis.grouping is not point.sample.grouping:
        raise ValueError("the basis belongs to another grouping")
    design = point.sample.design
    numeric = point._numeric_problem
    matrix = (
        (np.zeros((point.sample.count, 0)) if design is None else design.matrix)
        if numeric is None else numeric.design
    )
    owned = immutable_array(matrix) if numeric is None else matrix.view()
    s = moment(owned, rows.b, basis)
    block = kept_block(basis, owned.shape[1])
    if block is None:
        h: SparseMatrix | CompactMatrix = hessian(owned, rows.curvature, basis)
    else:
        entries = CompactEntries(owned.shape[1] + basis.n_kept, block)
        _slope_entries(owned, rows.curvature, entries)
        _mixed_entries(owned, rows.curvature, basis, entries)
        _pair_entries(rows.curvature, basis, owned.shape[1], entries)
        h = entries.finish()
    return Assembly(owned, s, h)


def _preparing_hessian(
    design: FloatArray, curvature: FloatArray, layout: JointLayout,
) -> tuple[CompactMatrix, _JointStructure | None]:
    """The first original private assembly also records its fixed potential slots."""
    assert layout.block is not None
    shared = _shared_assembly(design, curvature, layout)
    try:
        entries = _PreparingEntries(design.shape[1] + layout.basis.n_kept, layout.block)
    except MemoryError:
        entries = None
    if entries is None:
        original = CompactEntries(design.shape[1] + layout.basis.n_kept, layout.block)
        _slope_entries(design, curvature, original)
        _mixed_entries(design, curvature, layout.basis, original)
        layout.add_pairs(curvature, original)
        return original.finish(), None
    _slope_entries(design, curvature, entries)
    if shared:
        _mixed_entries(design, curvature, layout.basis, entries, shared=True)
        layout.add_pairs(curvature, entries, shared=True)
    else:
        _mixed_entries(design, curvature, layout.basis, entries)
        layout.add_pairs(curvature, entries)
    return entries.prepare(layout)


def _prepared_hessian(
    design: FloatArray, curvature: FloatArray, layout: JointLayout,
    structure: _JointStructure,
) -> CompactMatrix:
    """Same fresh checked values, with no repeated coordinates, masks or sorting."""
    entries = _ValueEntries(structure)
    k = design.shape[1]
    for values in _slope_values(design, curvature, k):
        entries.add(values)
    shared = _shared_assembly(design, curvature, layout)
    mixed = (_mixed_values(design, curvature, layout.basis, k, structure.mixed,
                           shared=True) if shared else
             _mixed_values(design, curvature, layout.basis, k, structure.mixed))
    for _, _, direction, values in mixed:
        entries.add(values.ravel() if direction != 2 else values)
    pairs = (layout._pair_values(curvature, shared=True) if shared else
             layout._pair_values(curvature))
    for selected, (_, _, sums) in zip(structure.pairs, pairs, strict=True):
        values = sums[selected.view()]
        entries.add(values)
        entries.add(values)
    return entries.finish(structure)


def support_step(support: SupportCheck) -> WorkStep:
    """The completed check of ``J`` on the positive rows, with its exact counts."""
    counters = (
        ("rows", support.rows),
        ("distinct", support.distinct),
        ("rank", support.rank),
    )
    detail = "eliminated" if support.eliminated else "structural"
    return WorkStep("support", "completed", detail, counters)


def assembly_step(assembly: Assembly) -> WorkStep:
    """The completed assembly of ``S`` and ``H``, with the stored entries of ``H``."""
    counters = (("H nonzeros", matrix_nnz(assembly.hessian)),)
    return WorkStep("assembly", "completed", "", counters)


def fresh_geometry(
    point: AdmittedPoint,
    rows: FreshRows,
    basis: CertificateBasis,
    support: SupportCheck,
    assembly: Assembly,
) -> FEGeometry:
    """The scaled geometry of the assembled sums; ``FloatingPointError`` on range.

    A zero scale is kept in the record (``refusal``), not raised.
    """
    owned, s, h = assembly.design, assembly.moment, assembly.hessian
    try:
        scale = equilibration(h)
    except ValueError as error:
        found = (None, None, float("nan"), None, str(error))
        return FEGeometry(point, rows, basis, support, owned, s, h, *found)
    ordinary = scaled_ordinary(h, scale)
    curvature = scaled_curvature(h, scale)
    norm = one_norm(ordinary)
    return FEGeometry(
        point, rows, basis, support, owned, s, h, scale, ordinary, norm, curvature, ""
    )


def _shared_assembly(
    design: FloatArray, curvature: FloatArray, layout: JointLayout,
) -> bool:
    """Current original private consumers, not a stored numerical eligibility flag."""
    return (original_values() and original_entries(layout)
            and type(design) is np.ndarray and type(curvature) is np.ndarray
            and design.dtype == curvature.dtype == np.dtype(np.float64)
            and design.dtype.metadata is None and curvature.dtype.metadata is None
            and design.ndim == 2 and curvature.shape == (design.shape[0],)
            and type(layout.basis) is CertificateBasis
            and (type(layout.basis.grouping) is _numeric.Grouping
                 or type(layout.basis.grouping) is Derived)
            and type(layout.basis.grouping.blocks) is tuple
            and all(type(block) is _numeric.Block
                    for block in layout.basis.grouping.blocks))


def original_values() -> bool:
    """Only original defining emitters/reducers omit the dynamic sink callbacks."""
    return (all(globals()[name] is value for name, value in _VALUE_PRODUCERS)
            and _numeric._input_original is _ARITHMETIC_ORIGINAL
            and _ARITHMETIC_ORIGINAL()
            and all(getattr(_reductions, name) is call
                    for name, call in _REDUCTION_CALLS))


_ARITHMETIC_ORIGINAL = _numeric._input_original
_REDUCTION_CALLS = tuple((call.__name__, call)
                         for call in _reductions._PREPARED_ORIGINALS)
_SLOPE_ENTRIES = _slope_entries
_MIXED_ENTRIES = _mixed_entries
_VALUE_PRODUCERS = tuple((name, globals()[name]) for name in (
    "_slope_entries", "_mixed_entries", "_slope_values", "_mixed_values", "_total",
    "checked_product", "group_sums", "_preparing_hessian", "_prepared_hessian",
    "CompactEntries", "JointLayout", "_PreparingEntries", "_ValueEntries",
    "_shared_assembly", "original_entries", "_numeric", "_reductions", "Derived",
))
