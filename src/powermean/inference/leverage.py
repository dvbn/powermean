"""Full-design per-replica leverage and the exact structural unit-leverage test.

Expected-information leverage for the nonlinear HC methods, full nuisance geometry
for absorbed fits.  Internal module: the leverage that the named HC methods of
:mod:`powermean.inference.hc` adjust their scores with.  No public name, option or
default; nothing here reads a request or applies a factor.

Geometry.  With the full design ``B`` (the slopes ``X`` alone, or ``[X, E_J]`` with the
kept incidence of :mod:`powermean.fe.certificate_basis`), the raw weights ``a`` and the
expected row curvature ``w_i = m_i^2 / V(m_i)`` of the transformed problem (one for the
linear equation), the Gram weight of a row is ``g_i = a_i w_i`` and the hat of its
replicas is

    h_i = (g_i / f_i) B_i' (B' diag(g) B)^-1 B_i,    f_i = a_i for frequency counts,
                                                     f_i = 1 otherwise,

so that ``sum_i f_i h_i`` is the identified full-design rank ``K`` (the trace of a
projector) and ``p = round(sum_i f_i h_i)`` must equal it: a mismatch is an error
witness, raised, never repaired.  The compressed sum ``sum_i h_i`` is NOT the trace.  A
row of zero weight has ``h_i = 0`` and is no sampling unit.  The hat is invariant to the
column scales and to the order (a change of basis of ``B``), and the expected geometry
is used for every working variance: at ``tau = 1`` it equals the observed one, at
``tau = 2`` it is the weighted least-squares hat of ``X`` while the observed curvature
``z/m`` is not.

Dense slopes (:func:`dense_leverage`).  The full column rank of ``X`` on the active
rows is decided first, exactly, by the rational echelon of the structural test below
(:func:`exact_full_rank`): a dependent column refuses, since the thin QR that follows
cannot tell.  Then ``sqrt(g_i) x_i`` is factored once by a Householder QR (LAPACK
through ``numpy.linalg.qr``); the hat is the squared norm of the ``i``-th row of the
thin ``Q``, ``||q_i||^2 = g_i x_i' (X'GX)^-1 x_i``, divided by ``f_i`` for frequency
counts.  The projector's diagonal is formed without the Gram matrix or its inverse,
whose conditioning is the square of the design's: the stable choice of the references'
``hatvalues``.  Limits, documented: a row whose exact hat is one
computes a value within rounding of one, not one; nothing is clamped, floored or
inferred from it (structural unit leverage is decided exactly, below).  The thin ``Q``
has ``K`` orthonormal columns whatever the rank of ``X``, so its trace is ``K`` by
construction and is NOT a rank witness (an exactly rank-one ``3 x 2`` design would
report ``p = 2``): the integer trace check of a dense design is an arithmetic witness
of the computed hats alone, and the rank is the exact decision made before it.  The
exact decision is made again by a caller that refuses before any solve
(:func:`powermean.inference.hc.hc_preflight`); the two do not share the one
decision.

Absorbed full design (:func:`absorbed_leverage`).  The expected Hessian
``H = B' diag(g) B`` is assembled by the certificate assembly (no dense
indicator), ``W = H^-T E_s`` by the sparse actions of
:func:`powermean.inference.bread.slope_map_of`, and the hat is the decomposition of
the reference's ``hatvalues`` for fixed effects, nuisance diagonal plus
residualized-slope diagonal, both on the SAME full basis:

    h_i = (g_i / f_i) [ d_i' R^-1 d_i  +  v_i' S v_i ],
    R = E_J' diag(g) E_J,    v_i = B_i W,    S = A^-1,    A = W_X = (X~' G X~)^-1,

where ``d_i`` is the row of ``E_J``, ``X~ = X - E_J R^-1 E_J' G X`` the slopes
residualized on the effects with weights ``g`` (``X~_i = S v_i``), so that
``v_i' S v_i = x~_i' S^-1 x~_i`` is the residualized-slope hat and ``d_i' R^-1 d_i``
the nuisance hat.  The nuisance quadratic forms are solved on the distinct code tuples
of the rows, in batches of right-hand sides against the transient ordinary factor of
the equilibrated ``R`` (the same ``equilibration``, ``scaled_ordinary`` and
``factorized`` of the certificate): no dense ``N x G`` matrix, no ``N x N`` hat, no
dense inverse of ``R``.  Each right-hand side is solved on its own by the factor, so
the batch size changes no bit of a hat; it bounds the memory of one solve
(``rows_per_solve``, a named parameter, not a tolerance).  Two more integer witnesses
hold exactly in real arithmetic and are checked as integers: the nuisance trace
``sum f_i g_i d_i' R^-1 d_i / f_i`` rounds to ``|J|`` and the slope trace to ``K``.
Slope-only residualized hats never stand for the full hat.

Structural unit leverage and exact rank (:func:`exact_full_rank`,
:func:`structural_unit_leverage`).  ``1 - h_i = 0`` for a replica exactly when
removing that ONE replica (``f_i`` to ``f_i - 1``) lowers the full-design rank, which
needs ``f_i = 1`` since ``f_i h_i <= 1``; deleting a whole compressed row is not the
test (``X = [[1, 0], [1, 1]]``, ``f = (3, 3)`` has ``h = 1/3`` and no refusal,
though either row's deletion lowers the rank).  The property depends on the design
alone, not on the point or the weights, so it is decided before solving and exactly:
in the rational arithmetic of the exact values of the doubles, the reduced row echelon
form of ``B'`` (one row per coordinate, one column per active row) is formed, every
coordinate must yield a pivot (else the design is not of full column rank: refused,
``ValueError``), and the rows of unit leverage are the pivot columns whose reduced row
is that unit vector, since ``e_i`` lies in the column space of ``B`` exactly when it is
a basis row of its canonical form.  :class:`FullRank` records the decision: the exact
rank (every coordinate a pivot), the active rows and the structural rows.  No rounded
hat, threshold or floor enters.  Limit, documented: the cost is that of an exact
Gauss-Jordan elimination on a ``(K + |J|) x n`` matrix with sparse integer indicator
rows, of the order of ``(K + |J|)^2 n`` rational operations at worst; it is the exact
test, not an approximation.

Ownership (:func:`detached`).  A :class:`Leverage` handed to another record is copied
whole: every array header and every nested record anew, so that a caller's later edit
of its own record (its shape, its nested arrays) reaches nothing retained.  Read-only
bytes alone do not prevent a change of shape or of metadata on a shared header.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from powermean.data.accounting import frozen_index
from powermean.fe.absorb_numeric import checked_product
from powermean.fe.certificate_basis import CertificateBasis
from powermean.inference.absorbed_code_tuples import CachedCodeTuples, _code_tuples
from powermean.inference.bread import SlopeMap, bread_influence, slope_map_of
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array, _finite_result
from powermean.solve.absorbed_certificate_factor import (
    Calls,
    FactorRecord,
    factorized,
)
from powermean.solve.absorbed_certificate_geometry import (
    SparseMatrix,
    equilibration,
    hessian,
    scaled_ordinary,
)

BoolArray = NDArray[np.bool_]
Geometry = Literal["expected"]
# The nuisance quadratic forms are solved on this many distinct code tuples at a time:
# a bound on the memory of one solve (|J| by rows_per_solve doubles), not a tolerance;
# each right-hand side is solved on its own, so the hats do not depend on it.
ROWS_PER_SOLVE = 256


# -------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class NuisanceSolve:
    """The provenance of the nuisance quadratic forms ``d_i' R^-1 d_i``.

    Attributes
    ----------
    scale : array
        The equilibration of ``R``.
    factor : FactorRecord
        The record of the transient ordinary factor of the scaled ``R``.
    distinct : int
        The distinct code tuples solved.
    actions : tuple of (str, int)
        The counted solves and their right-hand sides.
    """

    scale: FloatArray
    factor: FactorRecord
    distinct: int
    actions: tuple[tuple[str, int], ...]


@dataclass(frozen=True, eq=False)
class Leverage:
    """The per-replica hats of the rows of one design, with their provenance.

    Attributes
    ----------
    hat : array
        ``h_i`` per row given, zero on a row of zero weight; owned.
    gram_weights : array
        ``g_i = a_i w_i``, the weights of the Gram matrix; owned.
    multiplicity : array
        ``f_i``: the frequency counts, else one on a row of positive weight and zero
        on a row of zero weight; owned.
    frequency : bool
        Whether the weights are frequency counts.
    trace : float
        ``sum_i f_i h_i``.
    rank : int
        ``K``, the identified full-design rank the caller states: slopes plus kept
        nuisance coordinates.
    p : int
        ``round(trace)``, equal to ``rank`` (checked).
    geometry : str
        ``"expected"``: the expected-information hat, the least-squares hat
        where the expected curvature is one.
    slope_hat, nuisance_hat : array or None
        The two parts of an absorbed hat, per replica; ``None`` for dense slopes.
    slope_map : SlopeMap or None
        ``W = H^-T E_s`` of the expected Hessian of an absorbed design.
    nuisance : NuisanceSolve or None
        The provenance of the nuisance solves of an absorbed design.
    diagonal : array or None
        ``|diag R|`` of the QR factor of a dense design, recorded; the rank is the
        exact decision of :func:`exact_full_rank`, never read from it.
    """

    hat: FloatArray
    gram_weights: FloatArray
    multiplicity: FloatArray
    frequency: bool
    trace: float
    rank: int
    p: int
    geometry: Geometry
    slope_hat: FloatArray | None
    nuisance_hat: FloatArray | None
    slope_map: SlopeMap | None
    nuisance: NuisanceSolve | None
    diagonal: FloatArray | None

    @property
    def active(self) -> IndexArray:
        """The positions of the rows of positive weight."""
        return frozen_index(np.flatnonzero(self.multiplicity > 0))

    @property
    def n_rows(self) -> int:
        """The rows of the design."""
        return int(self.hat.size)


@dataclass(frozen=True, eq=False)
class FullRank:
    """The exact full-column-rank decision of one design on its active rows.

    Attributes
    ----------
    rank : int
        ``K + |J|``: every coordinate of ``B = [X, E_J]`` yielded a pivot of the exact
        echelon on the active rows.
    n_slopes : int
        ``K``, the slope columns.
    n_kept : int
        ``|J|``, the kept nuisance coordinates; zero without a basis.
    active : array of int
        The positions of the rows of positive weight (a positive count), increasing.
    unit : array of int
        The positions of the rows of structural unit leverage, increasing; empty where
        there is none.
    """

    rank: int
    n_slopes: int
    n_kept: int
    active: IndexArray
    unit: IndexArray


# --------------------------------------------------------------------- inputs


def _frequency(frequency: object) -> bool:
    """``frequency`` is a ``bool``; the kind of the weights is never inferred."""
    if not isinstance(frequency, bool):
        raise TypeError("frequency must be a bool")
    return frequency


def _rows(
    design: FloatArray, weights: FloatArray, fisher: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The design, the raw weights and the expected curvature, aligned and finite."""
    matrix = _finite_array(design, "design")
    a = _finite_array(weights, "weights")
    w = _finite_array(fisher, "fisher")
    if matrix.ndim != 2 or not matrix.shape[0]:
        raise ValueError("design must be a nonempty matrix")
    n = matrix.shape[0]
    if a.shape != (n,) or w.shape != (n,):
        raise ValueError("weights and fisher must have one value per row of design")
    if np.any(a < 0) or np.any(w <= 0):
        raise ValueError(
            "weights must be nonnegative and the expected curvature positive"
        )
    return matrix, a, w


def _multiplicity(weights: FloatArray, *, frequency: bool) -> FloatArray:
    """``f_i``: the counts, else one on a positive weight and zero on a zero weight."""
    if frequency:
        counts = weights.tolist()
        if any(not (c.is_integer() and c >= 0) for c in counts):
            raise ValueError("frequency weights must be integer counts")
        return np.array(weights, dtype=np.float64)
    return (weights > 0).astype(np.float64)


def _unit(
    gram_weights: FloatArray,
    weights: FloatArray,
    fisher: FloatArray,
    *,
    frequency: bool,
) -> FloatArray:
    """``g_i / f_i``: the expected curvature of one replica, ``g_i`` otherwise.

    Zero on a row of zero weight in either case: it is no replica.
    """
    if frequency:
        return np.where(weights > 0, fisher, 0.0)
    return np.array(gram_weights, dtype=np.float64)


def _traced(
    hat: FloatArray, multiplicity: FloatArray, rank: int, what: str
) -> tuple[float, int]:
    """``sum f_i h_i`` and its rounding, which must be ``rank`` (an integer witness)."""
    with np.errstate(over="ignore", invalid="ignore"):
        trace = float(np.sum(multiplicity * hat))
    if not math.isfinite(trace):
        raise FloatingPointError(f"the {what} trace is not representable in float64")
    p = round(trace)
    if p != rank:
        raise ValueError(
            f"the {what} leverage trace {trace!r} rounds to {p}, not to the identified "
            f"rank {rank}"
        )
    return trace, p


# ---------------------------------------------------------------- dense slopes


def dense_leverage(
    design: FloatArray, weights: FloatArray, fisher: FloatArray, *, frequency: bool
) -> Leverage:
    """The per-replica hats of a dense design without effects (module docstring).

    Parameters
    ----------
    design : array
        ``X`` on the rows of the system, one row per row of ``weights``, ``K >= 1``
        columns, of full column rank ``K`` on the rows of positive weight: decided
        here exactly before the factorization, never by the thin QR.
    weights : array
        The raw weights ``a`` of those rows, nonnegative.
    fisher : array
        The expected row curvature ``w_i``, positive: ones for the linear equation.
    frequency : bool
        Whether the weights are frequency counts (integers).

    Returns
    -------
    Leverage
        The hats, ``g``, ``f``, the trace, ``K``, ``p`` and ``|diag R|``.

    Raises
    ------
    ValueError
        Misaligned or invalid inputs, a design that is not of full column rank on the
        rows of positive weight (the exact rank witness), or a trace that does not
        round to ``K``.
    FloatingPointError
        A value that is not representable in float64.
    TypeError
        ``frequency`` is not a ``bool``.
    """
    kind = _frequency(frequency)
    matrix, a, w = _rows(design, weights, fisher)
    k = int(matrix.shape[1])
    if not k:
        raise ValueError("design must have at least one column")
    g = checked_product(a, w, "Gram weight")
    f = _multiplicity(a, frequency=kind)
    _decided(matrix, f, None)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled: FloatArray = np.sqrt(g)[:, None] * matrix
    scaled = _finite_result(scaled, "weighted design")
    q, r = np.linalg.qr(scaled, mode="reduced")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        squared: FloatArray = np.einsum("ij,ij->i", q, q)
        if kind:
            # One division per replica count; a zero count is no replica.
            hat: FloatArray = np.divide(
                squared, f, out=np.zeros_like(squared), where=f > 0
            )
        else:
            hat = np.where(f > 0, squared, 0.0)
    hat = _finite_result(hat, "leverage")
    trace, p = _traced(hat, f, k, "full-design")
    return Leverage(
        hat=_immutable(hat),
        gram_weights=_immutable(g),
        multiplicity=_immutable(f),
        frequency=kind,
        trace=trace,
        rank=k,
        p=p,
        geometry="expected",
        slope_hat=None,
        nuisance_hat=None,
        slope_map=None,
        nuisance=None,
        diagonal=_immutable(np.abs(np.diag(r))),
    )


# ------------------------------------------------------------ absorbed design


def _nuisance_block(matrix: SparseMatrix, k: int) -> SparseMatrix:
    """``R``: the trailing block of the canonical CSC ``H`` beyond the ``k`` slopes."""
    size = matrix.shape[0] - k
    columns = matrix.columns
    keep = (matrix.indices >= k) & (columns >= k)
    rows = matrix.indices[keep] - k
    cols = columns[keep] - k
    data = matrix.data[keep]
    indptr = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=size))))
    return SparseMatrix(
        (size, size),
        frozen_index(np.asarray(indptr, dtype=np.intp)),
        frozen_index(np.asarray(rows, dtype=np.intp)),
        _immutable(np.asarray(data, dtype=np.float64)),
    )


def _nuisance_forms(
    block: SparseMatrix, distinct: IndexArray, rows_per_solve: int
) -> tuple[FloatArray, NuisanceSolve]:
    """``e_c' R^-1 e_c`` for each distinct tuple ``c``, by batched solves."""
    scale = equilibration(block)
    ordinary = scaled_ordinary(block, scale)
    record, factor = factorized(ordinary, "ordinary")
    if factor is None:
        raise ValueError(f"the nuisance Gram matrix has no factor: {record.reason}")
    calls = Calls()
    forms = np.zeros(distinct.shape[0])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        inverse_scale = 1.0 / scale
    for start in range(0, distinct.shape[0], rows_per_solve):
        batch = distinct[start : start + rows_per_solve]
        rhs = np.zeros((scale.size, batch.shape[0]))
        for column in range(batch.shape[1]):
            kept = batch[:, column] >= 0
            rhs[batch[kept, column], np.flatnonzero(kept)] = inverse_scale[
                batch[kept, column]
            ]
        solved = calls.solve(factor, rhs, "N")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            forms[start : start + batch.shape[0]] = np.einsum("ji,ji->i", rhs, solved)
    found = _finite_result(forms, "nuisance leverage")
    provenance = NuisanceSolve(
        scale, record, int(distinct.shape[0]), tuple(calls.actions().items())
    )
    return found, provenance


def _slope_forms(
    design: FloatArray, basis: CertificateBasis, columns: FloatArray
) -> FloatArray:
    """``v_i' S v_i`` with ``v_i = B_i W`` and ``S = W_X^-1``: the residualized hat."""
    k = design.shape[1]
    ones = np.ones(design.shape[0])
    v = bread_influence(design, basis, ones, columns)
    residualized = np.linalg.solve(columns[:k], v.T).T
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        forms: FloatArray = np.einsum("ij,ij->i", v, residualized)
    return _finite_result(forms, "slope leverage")


def absorbed_leverage(
    design: FloatArray,
    basis: CertificateBasis,
    weights: FloatArray,
    fisher: FloatArray,
    *,
    frequency: bool,
    rows_per_solve: int = ROWS_PER_SOLVE,
    code_tuples: CachedCodeTuples | None = None,
) -> Leverage:
    """The per-replica hats of the full design ``[X, E_J]`` (module docstring).

    Parameters
    ----------
    design : array
        ``X``, the scaled slopes on every row of the grouping (zero columns allowed).
    basis : CertificateBasis
        The kept nuisance coordinates ``J`` of the same rows.
    weights : array
        The raw weights ``a`` of the rows, nonnegative; a zero weight makes no replica.
    fisher : array
        The expected row curvature ``w_i``, positive: ones for the linear equation.
    frequency : bool
        Whether the weights are frequency counts.
    rows_per_solve : int, optional
        The distinct code tuples solved together; a memory bound, never a result.

    Returns
    -------
    Leverage
        The hats with both parts, ``W``, the nuisance provenance, the trace and ``p``.

    Raises
    ------
    ValueError
        Misaligned inputs, a design that is not identified (no scale or no factor),
        or a trace that does not round to its rank.
    FloatingPointError
        A value that is not representable in float64.
    """
    kind = _frequency(frequency)
    matrix, a, w = _rows(design, weights, fisher)
    if type(basis) is not CertificateBasis:
        raise TypeError("basis must be a CertificateBasis")
    if basis.grouping.n_rows != matrix.shape[0]:
        raise ValueError("the basis belongs to other rows")
    if not isinstance(rows_per_solve, int) or rows_per_solve < 1:
        raise ValueError("rows_per_solve must be a positive integer")
    k, kept = int(matrix.shape[1]), basis.n_kept
    g = checked_product(a, w, "Gram weight")
    f = _multiplicity(a, frequency=kind)
    unit = _unit(g, a, w, frequency=kind)
    full = hessian(matrix, g, basis)
    distinct, inverse = (
        _code_tuples(basis) if code_tuples is None else code_tuples.for_basis(basis)
    )
    block = _nuisance_block(full, k)
    forms, provenance = _nuisance_forms(block, distinct, rows_per_solve)
    nuisance = _finite_result(unit * forms[inverse], "nuisance leverage")
    slope_map: SlopeMap | None = None
    slope = np.zeros(matrix.shape[0])
    if k:
        slope_map = slope_map_of(full, k, bread="expected")
        forms_x = _slope_forms(matrix, basis, slope_map.columns)
        slope = _finite_result(unit * forms_x, "slope leverage")
        _traced(slope, f, k, "residualized-slope")
    _traced(nuisance, f, kept, "nuisance")
    hat = _finite_result(nuisance + slope, "leverage")
    trace, p = _traced(hat, f, k + kept, "full-design")
    return Leverage(
        hat=_immutable(hat),
        gram_weights=_immutable(g),
        multiplicity=_immutable(f),
        frequency=kind,
        trace=trace,
        rank=k + kept,
        p=p,
        geometry="expected",
        slope_hat=_immutable(slope),
        nuisance_hat=_immutable(nuisance),
        slope_map=slope_map,
        nuisance=provenance,
        diagonal=None,
    )


# ------------------------------------------------- structural unit leverage

Row = dict[int, Fraction]  # a sparse rational row of B': active row -> value


def _coordinate_rows(
    design: FloatArray, basis: CertificateBasis | None, active: IndexArray
) -> list[Row]:
    """The rows of ``B'`` on the active rows: each slope, then each kept coordinate."""
    found: list[Row] = []
    for column in design.T:
        values = column[active].tolist()
        found.append({i: Fraction(v) for i, v in enumerate(values) if v != 0.0})
    if basis is None:
        return found
    rows: list[Row] = [{} for _ in range(basis.n_kept)]
    for block, position in zip(basis.grouping.blocks, basis.positions, strict=True):
        coordinates = position[block.codes[active]].tolist()
        for i, j in enumerate(coordinates):
            if j >= 0:
                rows[j][i] = Fraction(1)
    return found + rows


def _reduced(row: Row, pivots: dict[int, Row]) -> Row:
    """``row`` with every pivot column eliminated (each pivot row is reduced)."""
    found = dict(row)
    for p in [c for c in found if c in pivots]:
        factor = found.pop(p)
        for c, value in pivots[p].items():
            if c == p:
                continue
            if new := found.get(c, 0) - factor * value:
                found[c] = new
            else:
                found.pop(c, None)
    return found


def _echelon(rows: list[Row]) -> dict[int, Row]:
    """The reduced row echelon form of the rows: pivot column -> its reduced row.

    Every row must yield a pivot (full column rank of ``B``); ``ValueError`` otherwise.
    """
    pivots: dict[int, Row] = {}
    for row in rows:
        found = _reduced(row, pivots)
        if not found:
            raise ValueError(
                "the design is not of full column rank: a coordinate is dependent"
            )
        p = min(found)
        lead = found[p]
        found = {c: value / lead for c, value in found.items()}
        for other in pivots.values():
            factor = other.pop(p, None)
            if factor is None:
                continue
            for c, value in found.items():
                if c == p:
                    continue
                if new := other.get(c, 0) - factor * value:
                    other[c] = new
                else:
                    other.pop(c, None)
        pivots[p] = found
    return pivots


def _decided(
    matrix: FloatArray, multiplicity: FloatArray, basis: CertificateBasis | None
) -> FullRank:
    """The exact decision on validated rows: full column rank, else ``ValueError``."""
    active = np.flatnonzero(multiplicity > 0)
    k, kept = int(matrix.shape[1]), 0 if basis is None else basis.n_kept
    if not active.size or not k + kept:
        raise ValueError("the design needs a row of positive weight and a column")
    pivots = _echelon(_coordinate_rows(matrix, basis, active))
    single = multiplicity[active] == 1.0
    unit = sorted(
        int(active[p]) for p, row in pivots.items() if len(row) == 1 and single[p]
    )
    return FullRank(
        len(pivots),
        k,
        kept,
        frozen_index(active),
        frozen_index(np.asarray(unit, dtype=np.intp)),
    )


def exact_full_rank(
    design: FloatArray,
    weights: FloatArray,
    *,
    frequency: bool,
    basis: CertificateBasis | None = None,
) -> FullRank:
    """The exact full-column-rank decision of a design, with its structural rows.

    Parameters
    ----------
    design : array
        ``X`` on the rows (every row of the grouping for an absorbed design).
    weights : array
        The raw weights of the rows; a row of zero weight is no replica and never
        has unit leverage.
    frequency : bool
        Whether the weights are frequency counts: a count above one leaves a copy.
    basis : CertificateBasis, optional
        The kept nuisance coordinates of the same rows, for an absorbed design.

    Returns
    -------
    FullRank
        The exact rank ``K + |J|``, the active rows and the rows of structural unit
        leverage.  It needs no point and no solve.

    Raises
    ------
    ValueError
        Misaligned inputs, or a design that is not of full column rank on the rows
        of positive weight (the exact rank witness).
    TypeError
        ``frequency`` is not a ``bool``.
    """
    kind = _frequency(frequency)
    matrix = _finite_array(design, "design")
    a = _finite_array(weights, "weights")
    if matrix.ndim != 2 or a.shape != (matrix.shape[0],) or np.any(a < 0):
        raise ValueError("design and weights must be aligned, the weights nonnegative")
    if basis is not None and (
        type(basis) is not CertificateBasis or basis.grouping.n_rows != matrix.shape[0]
    ):
        raise ValueError("the basis belongs to other rows")
    return _decided(matrix, _multiplicity(a, frequency=kind), basis)


def structural_unit_leverage(
    design: FloatArray,
    weights: FloatArray,
    *,
    frequency: bool,
    basis: CertificateBasis | None = None,
) -> IndexArray:
    """The rows whose one replica's removal lowers the full-design rank, exactly.

    The ``unit`` rows of :func:`exact_full_rank`, with its arguments and refusals.

    Returns
    -------
    array
        The positions (in the rows of ``design``) of the rows of structural unit
        leverage, increasing; empty where there is none.
    """
    return exact_full_rank(design, weights, frequency=frequency, basis=basis).unit


# ------------------------------------------------------------------- ownership


def _owned(value: FloatArray | None) -> FloatArray | None:
    """An owned read-only copy, or ``None``."""
    return None if value is None else _immutable(value)


def _owned_index(value: IndexArray | None) -> IndexArray | None:
    """An owned read-only index copy, or ``None``."""
    return None if value is None else frozen_index(value)


def _owned_factor(record: FactorRecord) -> FactorRecord:
    """The factor record with its arrays copied; its mappings are read-only views."""
    return replace(
        record,
        perm_r=_owned_index(record.perm_r),
        perm_c=_owned_index(record.perm_c),
        diagonal=_owned(record.diagonal),
    )


def detached(leverage: Leverage) -> Leverage:
    """An owned copy of a leverage record: every array header and record anew.

    Parameters
    ----------
    leverage : Leverage
        The record to copy.

    Returns
    -------
    Leverage
        A record that shares no array header, no bytes and no nested record with
        ``leverage``; its values are those of ``leverage`` bit for bit.

    Raises
    ------
    TypeError
        ``leverage`` is not a ``Leverage`` record.
    """
    if type(leverage) is not Leverage:
        raise TypeError("leverage must be a Leverage record")
    slope_map: SlopeMap | None = None
    if leverage.slope_map is not None:
        found = leverage.slope_map
        slope_map = SlopeMap(
            found.bread,
            _immutable(found.scale),
            _owned_factor(found.factor),
            _immutable(found.columns),
        )
    nuisance: NuisanceSolve | None = None
    if leverage.nuisance is not None:
        solve = leverage.nuisance
        nuisance = NuisanceSolve(
            _immutable(solve.scale),
            _owned_factor(solve.factor),
            int(solve.distinct),
            tuple((str(name), int(count)) for name, count in solve.actions),
        )
    return Leverage(
        hat=_immutable(leverage.hat),
        gram_weights=_immutable(leverage.gram_weights),
        multiplicity=_immutable(leverage.multiplicity),
        frequency=bool(leverage.frequency),
        trace=float(leverage.trace),
        rank=int(leverage.rank),
        p=int(leverage.p),
        geometry=leverage.geometry,
        slope_hat=_owned(leverage.slope_hat),
        nuisance_hat=_owned(leverage.nuisance_hat),
        slope_map=slope_map,
        nuisance=nuisance,
        diagonal=_owned(leverage.diagonal),
    )


__all__ = [
    "ROWS_PER_SOLVE",
    "FullRank",
    "Leverage",
    "NuisanceSolve",
    "absorbed_leverage",
    "dense_leverage",
    "detached",
    "exact_full_rank",
    "structural_unit_leverage",
]
