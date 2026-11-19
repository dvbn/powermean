"""Existence of a finite root of the cancelled ``tau = 2`` equation (owner's ruling 71).

Ruling 71: existence is decided per estimator, on its own estimating equation; a fit
is certified only where the requested estimator has a finite root, and where it has
none the fit ends with a named refusal (``no_finite_root``), not with a solver status.
The joint route decides it for the joint equation in its sample step
(:mod:`powermean.solve.support_gamma`); the cancelled equation is another equation
away from ``tau = 1``, and this module decides it for the cancelled equation at
``tau = 2``, the one power where an exact test exists.

The equation.  With the cancelled effects profiled out, the ``tau = 2`` slope equation
is ``S(theta) = B' t``, ``t_i = z_i exp(-q_i)``, ``q = B theta + o``, where ``B`` is
the design centered (unweighted) within the cancelled groups
(:class:`~powermean.fe.retained.CenteredDesign`, whose range the identified
coordinates span: ``B T P = B``, :mod:`powermean.fe.geometry`) and ``o`` the centered
offset.  ``S = -grad F`` for the convex ``F(theta) = sum_i z_i exp(-q_i)``, to which
only the rows ``P = {z > 0}`` contribute.

The dichotomy.  A recession direction is a ``w`` with ``h = B w`` such that
``h_P >= 0`` and ``h_P != 0``.  Where one exists, the derivative of ``F`` along
``w``, ``-sum_P z_i exp(-q_i) h_i``, is negative at every ``theta``: the gradient
never vanishes and no finite root exists.  Where none exists, ``F`` grows without
bound along every direction that moves a positive row and is constant along the
others, so it attains its minimum: a finite root exists (not necessarily unique).

The program. First apply the one-dimensional sign alternative where all retained
columns vanish under centering, and test signed slope columns for a recession
witness. These tests introduce no tolerance or tuning parameter. Otherwise solve
the sparse feasibility program below with ``method="highs-ipm"``. In fully positive
cancelled groups, nonnegative centered rows sum to zero, so every such row must be
zero. Keep those rows as equalities and normalize only the remaining positive rows.
This reduction preserves the constraints supplied by the fully positive groups.
When all retained columns vanish under centering, omit their group-mean auxiliary
variables and divide each nonzero positive slope row by its largest magnitude.
Positive row scaling preserves the alternative and prevents tiny opposing rows
from disappearing into an absolute LP feasibility tolerance.
For interior point, one retained level in each connected component of its
within-group co-occurrence graph is fixed at zero. Each removed shift is constant
inside every cancelled group and is therefore annihilated by centering.
A rejected interior-point witness or another undecided outcome triggers one solve
of the same reduced program with ``method="highs"`` and free reference levels.
Both calls use the joint route's
feasibility tolerance and the original centered witness rule. Neither has a package
iteration budget or time limit. The LP still runs only after a failed fit, and its
worst-case running time is not bounded by these choices.

So that the program is as sparse as the data, the centering of the retained levels is
written with one free variable ``c_g`` per cancelled group: ``h = M_C X beta +
D alpha - c_g(i)`` with ``n_g c_g = sum_(i in g) (D alpha)_i`` over every row of the
group, zero outcomes included (``M_C X``, the stored centered slopes, sums to zero
within each group); then ``h_P >= 0`` and ``sum_P h = 1``, every variable free.  Its
nonzeros are about ``|P| (K + blocks + 1)`` plus the distinct pairs of a group and a
retained level; no dense or centered indicator matrix is formed.

The decision.  An optimal point is checked afresh, by the witness rule of the joint
route's ``tau = 2`` recession (:mod:`powermean.solve.support_gamma_witness`, its
zero-sum case): ``h`` is recomputed from ``(beta, alpha)`` through the centered design
on every row, ``m = max |h|``, and the point is a recession direction when
``min_P h >= -e m`` and ``max_P h > d m``, with ``e`` and ``d`` the tolerance and the
threshold of :data:`~powermean.solve.support_gamma.BASELINE`: ``"no_finite_root"``.
A numerical witness is then checked with exact integer arithmetic on the
exactly centered original slopes and retained indicators. The failed-fit caller
supplies its owned original slopes; a design-only caller uses the re-centered
stored slopes instead. If necessary, its
candidate binding equalities are solved over the rationals and every positive-row
sign is checked again. Only an exactly nonnegative, nonzero direction is accepted.
The floating-point feasibility tolerance selects a candidate face, never a sign.
This removes the ambiguity of accepting a small negative directional change.
A program that HiGHS reports infeasible has no recession direction:
``"finite_root"`` (HiGHS's verdict within its tolerance; no Farkas certificate is
formed or checked).  Anything else (a limit, numerical difficulties, a warning during
the call, an allocation failure, or a point that fails the fresh check):
``"undecided"``, and the fit keeps its ending.

When: only after a cancelled solve at ``tau = 2`` that does not certify
(:mod:`powermean.interface.cancelled_fit`); never on a certified fit or at another
power, since on a large sample the program can cost more than the fit itself.  The
fit records the decision (``provenance.failure["existence"]``) and, on
``"no_finite_root"``, ends with that status: ``on_failure="raise"`` raises
``ExistenceError`` (``existence.no_finite_root``) and the result reports no point.
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Mapping
from typing import Literal, Protocol, cast

import numpy as np

from powermean.fe._cancel_storage import IndexArray
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray
from powermean.solve import cancelled_existence_exact as exact
from powermean.solve.support_gamma import BASELINE
from powermean.solve.support_gamma_lp import Optimize, SparseMatrix, sparse
from powermean.solve.support_lp import LinprogResult

Existence = Literal["finite_root", "no_finite_root", "undecided"]
# SciPy's documented linprog statuses: 0 optimal, 2 infeasible (1 limit, 3 unbounded,
# 4 numerical difficulties: undecided).
_OPTIMAL, _INFEASIBLE = 0, 2


class _ConstraintMatrix(SparseMatrix, Protocol):
    """The extra CSR operations used by the equality reduction."""

    @property
    def shape(self) -> tuple[int, int]:
        """Number of constraint rows and variables."""
        ...

    def __getitem__(self, index: object) -> _ConstraintMatrix:
        """Select constraint rows."""
        ...

    def sum(self, *, axis: int) -> FloatArray:
        """Sum sparse entries over an axis."""
        ...


class _SparseStack(Protocol):
    """The sparse stacking operation, isolated from the joint-route protocol."""

    def vstack(self, blocks: list[SparseMatrix], sparse_format: str, /) -> SparseMatrix:
        """Stack constraint blocks vertically."""
        ...


class _Components(Protocol):
    """Connected components of an undirected retained-level graph."""

    def connected_components(
        self, graph: SparseMatrix, /, *, directed: bool
    ) -> tuple[int, IndexArray]:
        """Return the component count and the label of every level."""
        ...


class _BoundedProgram(Protocol):
    """The linprog call with one bound pair per variable."""

    def __call__(
        self,
        objective: FloatArray,
        upper: SparseMatrix,
        upper_right: FloatArray,
        equal: SparseMatrix,
        equal_right: FloatArray,
        /,
        *,
        bounds: list[tuple[float, float]],
        method: str,
        options: Mapping[str, float | int | bool],
    ) -> LinprogResult:
        """Solve the sparse feasibility program."""
        ...


def _program(
    design: CenteredDesign, positive: IndexArray
) -> tuple[SparseMatrix, SparseMatrix, FloatArray, int]:
    """The inequality and equality matrices, the equality right side, and ``K + L``.

    Variables: the slopes, the levels of each retained block in block order, then one
    ``c_g`` per cancelled group; rows in the structure's sorted order.
    """
    structure = design.structure
    n_groups = structure.n_groups
    groups = structure.sorted_codes
    slopes = design.centered_slopes
    k = design.n_slopes
    sizes = structure.block_sizes
    starts = np.concatenate(([k], k + np.cumsum(sizes, dtype=np.int64)))
    n_full = int(starts[-1])
    n_vars = n_full + n_groups
    n_pos = positive.size
    rows: list[IndexArray] = []
    cols: list[IndexArray] = []
    data: list[FloatArray] = []
    # h_P >= 0 as -h_P <= 0: slopes, retained levels, minus the group's c.
    ones = np.arange(n_pos)
    if k:
        rows.append(np.repeat(ones, k))
        cols.append(np.tile(np.arange(k), n_pos))
        data.append(-slopes[positive].ravel())
    eq_rows: list[IndexArray] = []
    eq_cols: list[IndexArray] = []
    eq_data: list[FloatArray] = []
    total = np.zeros(n_vars)
    if k:
        total[:k] = slopes[positive].sum(axis=0)
    for codes, start, size in zip(
        structure.retained_sorted_codes, starts[:-1], sizes, strict=True
    ):
        rows.append(ones)
        cols.append(start + codes[positive])
        data.append(np.full(n_pos, -1.0))
        # n_g c_g = sum over the group of alpha at each row's level, all rows.
        pairs, counts = np.unique(
            groups.astype(np.int64) * size + codes, return_counts=True
        )
        eq_rows.append(pairs // size)
        eq_cols.append(start + pairs % size)
        eq_data.append(counts.astype(np.float64))
        total[start : start + size] = np.bincount(codes[positive], minlength=size)
    rows.append(ones)
    cols.append(n_full + groups[positive])
    data.append(np.ones(n_pos))
    eq_rows.append(np.arange(n_groups))
    eq_cols.append(n_full + np.arange(n_groups))
    eq_data.append(-structure.counts.astype(np.float64))
    total[n_full:] = -np.bincount(groups[positive], minlength=n_groups)
    upper = sparse.csr_array(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_pos, n_vars),
    )
    nonzero = np.flatnonzero(total)
    equal = sparse.csr_array(
        (
            np.concatenate([*eq_data, total[nonzero]]),
            (
                np.concatenate([*eq_rows, np.full(nonzero.size, n_groups)]),
                np.concatenate([*eq_cols, nonzero]),
            ),
        ),
        shape=(n_groups + 1, n_vars),
    )
    return upper, equal, np.concatenate((np.zeros(n_groups), [1.0])), n_full


def _retained_vanish(design: CenteredDesign) -> bool:
    """Whether every retained indicator is constant inside cancelled groups."""
    structure = design.structure
    return all(
        np.array_equal(
            np.minimum.reduceat(codes, structure.starts),
            np.maximum.reduceat(codes, structure.starts),
        )
        for codes in structure.retained_sorted_codes
    )


def _slope_rows(values: FloatArray, n_columns: int) -> SparseMatrix:
    """Store slope entries, leaving annihilated retained columns empty."""
    n_rows, k = values.shape
    return sparse.csr_array(
        (
            values.ravel(),
            (np.repeat(np.arange(n_rows), k), np.tile(np.arange(k), n_rows)),
        ),
        shape=(n_rows, n_columns),
    )


def _slope_program(
    design: CenteredDesign, positive: IndexArray
) -> tuple[SparseMatrix, SparseMatrix, FloatArray, int]:
    """Normalize rows when centering annihilates every retained indicator."""
    structure = design.structure
    rows = np.empty((positive.size, design.n_slopes))
    for column in range(design.n_slopes):
        rows[:, column] = _column(design, column)[positive]
    scales = np.max(np.abs(rows), axis=1, initial=0.0)
    scales[scales == 0] = 1.0
    with np.errstate(divide="raise", invalid="raise", over="raise", under="raise"):
        rows = rows / scales[:, None]
    counts = np.bincount(structure.sorted_codes[positive], minlength=structure.n_groups)
    forced = (counts == structure.counts)[structure.sorted_codes[positive]]
    # Positive row scaling preserves signs. Normalizing the sum of these scaled
    # rows is equivalent to normalizing the original nonnegative, nonzero vector.
    equal = np.vstack((rows[forced], rows[~forced].sum(axis=0, keepdims=True)))
    right = np.concatenate((np.zeros(np.count_nonzero(forced)), [1.0]))
    return (
        _slope_rows(-rows[~forced], design.n_columns),
        _slope_rows(equal, design.n_columns),
        right,
        design.n_columns,
    )


def _reduced_program(
    design: CenteredDesign, positive: IndexArray
) -> tuple[SparseMatrix, SparseMatrix, FloatArray, int]:
    """Keep fully positive groups as equalities, not disposable observations."""
    if _retained_vanish(design):
        return _slope_program(design, positive)
    upper, equal, _, n_full = _program(design, positive)
    structure = design.structure
    counts = np.bincount(structure.sorted_codes[positive], minlength=structure.n_groups)
    pure = counts == structure.counts
    forced = pure[structure.sorted_codes[positive]]
    constraints = cast(_ConstraintMatrix, upper)
    mixed = constraints[~forced]
    # A nonnegative centered vector sums to zero in a fully positive group,
    # so every such row is zero. Only mixed groups can supply normalization.
    normalization = sparse.csr_array(
        -np.asarray(mixed.sum(axis=0)).reshape(1, -1),
        shape=(1, n_full + structure.n_groups),
    )
    equal = cast(_SparseStack, sparse).vstack(
        [cast(_ConstraintMatrix, equal)[:-1], constraints[forced], normalization],
        "csr",
    )
    n_equalities = structure.n_groups + int(np.count_nonzero(forced))
    right = np.concatenate((np.zeros(n_equalities), [1.0]))
    return mixed, equal, right, n_full


def _bounds(
    design: CenteredDesign, n_variables: int, *, references: bool
) -> list[tuple[float, float]]:
    """Remove only shifts that centering annihilates identically."""
    bounds = [(-math.inf, math.inf)] * n_variables
    if not references:
        return bounds
    structure = design.structure
    components = cast(_Components, importlib.import_module("scipy.sparse.csgraph"))
    for codes, start, size in zip(
        structure.retained_sorted_codes,
        design.block_starts,
        structure.block_sizes,
        strict=True,
    ):
        anchors = codes[structure.starts][structure.sorted_codes]
        graph = sparse.csr_array(
            (np.ones(structure.n_rows), (codes, anchors)), shape=(size, size)
        )
        count, labels = components.connected_components(graph, directed=False)
        representatives = np.full(count, size, dtype=np.int64)
        np.minimum.at(representatives, labels, np.arange(size))
        for representative in representatives:
            bounds[start + int(representative)] = (0.0, 0.0)
    return bounds


def _solve(
    design: CenteredDesign, positive: IndexArray, *, method: str = "highs-ipm"
) -> tuple[int, FloatArray | None, int] | None:
    """HiGHS's status and point, and ``K + L``; None on a warning or MemoryError."""
    try:
        upper, equal, right, n_full = _reduced_program(design, positive)
        bounds = _bounds(
            design,
            cast(_ConstraintMatrix, upper).shape[1],
            references=method == "highs-ipm",
        )
        optimize = cast(Optimize, importlib.import_module("scipy.optimize"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = cast(_BoundedProgram, optimize.linprog)(
                np.zeros(cast(_ConstraintMatrix, upper).shape[1]),
                upper,
                np.zeros(cast(_ConstraintMatrix, upper).shape[0]),
                equal,
                right,
                bounds=bounds,
                method=method,
                options={
                    "primal_feasibility_tolerance": BASELINE.feasibility,
                    "dual_feasibility_tolerance": BASELINE.feasibility,
                },
            )
    except (MemoryError, FloatingPointError):
        return None
    if caught:
        return None
    return int(found.status), found.x, n_full


def _witness(
    design: CenteredDesign,
    point: FloatArray,
    positive: IndexArray,
    *,
    slopes: FloatArray | None = None,
) -> bool:
    """The fresh check of a direction ``(beta, alpha)`` (module docstring)."""
    direction = np.asarray(point, dtype=np.float64)
    if not np.isfinite(direction).all():
        return False
    try:
        with np.errstate(under="raise"):
            h = design.matvec_sorted(direction)
    except FloatingPointError:
        return False
    m = float(np.max(np.abs(h)))
    on_positive = h[positive]
    accepted = (
        m > 0
        and float(on_positive.min()) >= -BASELINE.tolerance * m
        and float(on_positive.max()) > BASELINE.threshold * m
    )
    if not accepted:
        return False
    try:
        return exact.witness(design, direction, positive, candidate_h=h, slopes=slopes)
    except MemoryError:
        return False


def _column(design: CenteredDesign, column: int) -> FloatArray:
    """A centered slope, with structurally constant groups exactly zero."""
    values = design.centered_slopes[:, column].copy()
    structure = design.structure
    lower = np.minimum.reduceat(values, structure.starts)
    upper = np.maximum.reduceat(values, structure.starts)
    # Subtracting a floating-point mean from a constant group can leave the
    # same rounding residue on every row. Its centered column is exactly zero.
    values[(lower == upper)[structure.sorted_codes]] = 0.0
    return values


def _one_slope(
    design: CenteredDesign, positive: IndexArray, *, slopes: FloatArray | None = None
) -> Existence | None:
    """The sign alternative when retained columns are annihilated by centering."""
    if design.n_slopes > 1:
        return None
    if not _retained_vanish(design):
        return None
    if design.n_slopes == 0:
        return "finite_root"
    direction = np.zeros(design.n_columns)
    direction[0] = 1.0
    has_positive, has_negative = exact.signs(design, direction, positive, slopes=slopes)
    return "no_finite_root" if has_positive != has_negative else "finite_root"


def _slope_witness(
    design: CenteredDesign, positive: IndexArray, *, slopes: FloatArray | None = None
) -> bool:
    """Apply the exact sign alternative to individual centered slope columns."""
    structure = design.structure
    counts = np.bincount(structure.sorted_codes[positive], minlength=structure.n_groups)
    for column in range(design.n_slopes):
        full = _column(design, column)
        varies = np.minimum.reduceat(full, structure.starts) != np.maximum.reduceat(
            full, structure.starts
        )
        if np.any((counts == structure.counts) & varies):
            continue
        values = full[positive]
        lower, upper = float(values.min()), float(values.max())
        if (lower >= 0 and upper > 0) or (upper <= 0 and lower < 0):
            direction = np.zeros(design.n_columns)
            direction[column] = 1.0 if upper > 0 else -1.0
            if exact.witness(design, direction, positive, slopes=slopes):
                return True
    return False


def _decision(
    design: CenteredDesign,
    positive: IndexArray,
    found: tuple[int, FloatArray | None, int] | None,
    *,
    slopes: FloatArray | None = None,
) -> Existence:
    """Check the solver's candidate through the original centered operator."""
    if found is None:
        return "undecided"
    status, point, n_full = found
    if status == _INFEASIBLE:
        return "finite_root"
    if (
        status == _OPTIMAL
        and point is not None
        and _witness(design, point[:n_full], positive, slopes=slopes)
    ):
        return "no_finite_root"
    return "undecided"


def recession(
    design: CenteredDesign, z: FloatArray, *, raw_slopes: FloatArray | None = None
) -> Existence:
    """Whether the cancelled ``tau = 2`` equation has a finite root (module docstring).

    Parameters
    ----------
    design : CenteredDesign
        The fit's centered design, the one its solve and certificate read.
    z : ndarray
        The transformed outcomes, nonnegative, in input row order.
    raw_slopes : ndarray, optional
        Owned original slopes in input row order, supplied by the failed-fit route
        for exact certification before floating-point centering. If omitted, exact
        checks use the re-centered stored slopes of ``design``.

    Returns
    -------
    {"finite_root", "no_finite_root", "undecided"}
        ``"no_finite_root"`` with a recession direction checked afresh,
        ``"finite_root"`` where HiGHS reports that none exists (or no outcome is
        positive: ``F`` is zero and every point a root), else ``"undecided"``.
    """
    slopes = (
        None
        if raw_slopes is None
        else np.asarray(raw_slopes, dtype=np.float64)[design.structure.order]
    )
    z_sorted = np.asarray(z, dtype=np.float64)[design.structure.order]
    positive = np.flatnonzero(z_sorted > 0)
    if positive.size == 0:
        return "finite_root"
    structure = design.structure
    counts = np.bincount(structure.sorted_codes[positive], minlength=structure.n_groups)
    if not np.any((counts > 0) & (counts < structure.counts)):
        return "finite_root"
    try:
        analytic = _one_slope(design, positive, slopes=slopes)
        if analytic is not None:
            return analytic
        if _slope_witness(design, positive, slopes=slopes):
            return "no_finite_root"
    except MemoryError:
        return "undecided"
    decision = _decision(design, positive, _solve(design, positive), slopes=slopes)
    if decision != "undecided":
        return decision
    return _decision(
        design, positive, _solve(design, positive, method="highs"), slopes=slopes
    )
