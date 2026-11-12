"""The checked arithmetic and the LAPACK attempts of the absorbed linear path.

Internal module: the formulas that the solve
(:mod:`powermean.solve.linear_absorbed`) and the admission of its records
(:mod:`powermean.solve.linear_absorbed_point`,
:mod:`powermean.solve.linear_absorbed_state`) share.  Nothing here binds a record to a
sample, certifies a point or forms inference.

Notation.  At ``omega = 0``: ``l`` the log response, ``o`` the RAW offset (entered
once), ``a`` the raw weights (one without weights), ``Xraw`` the original active
columns, ``Xs = Xraw D^-1``, ``theta = D beta`` and ``t = l - o``.  The joint
absorption of ``[t, Xs]`` gives block ``b`` the coefficients ``[a_t,b, a_X,b]``; its
seed is ``alpha_b = a_t,b - a_X,b theta``; the final absorption of ``t - Xs theta``
from the seeds gives the effects.  A point's authoritative index is

    q = ((o + Xraw beta) + alpha_1[c_1]) + alpha_2[c_2] + ...

in that order, and its raw slope equations are ``Xraw'(a e)`` at ``e = l - q``.  Each
formula is one function, shared by the solve and the admission.  Arithmetic is that of
:mod:`powermean.fe.absorb_numeric`: overflow, invalid values and inexact underflow raise
``FloatingPointError``, exact zeros and subnormals pass; a matrix-vector product is the
checked products of its columns added in column order.

The attempts.  With ``W = sqrt(a) Xtilde``: the singular values of ``W`` (NumPy,
``compute_uv=False``) and its numerical rank, the values strictly above ``eps max(n,
K) s_max`` with the physical rows ``n``, never a frequency total; the pivoted economic
QR ``W[:, P] = Q R`` (SciPy); and the back substitution ``R z = Q'r`` (SciPy) with
``theta[P] = z``.  No Gram matrix is formed.  Each attempt is one
:class:`Decomposition`, its outputs checked before use: real
floating arrays of the declared shapes, finite, singular values nonnegative and
nonincreasing, pivots a permutation; a ``LinAlgError`` of the call is ``failed``, an
output that fails a check ``unusable``.  SciPy has no type information:
``scipy.linalg`` is reached by :func:`importlib.import_module` with the typed interface
of the calls made, as in :mod:`powermean.solve.support_projection`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Protocol, cast, get_args

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.collinearity_numeric import (
    EPSILON,
    _doubles,
    _pivoted,
    _singular,
    count_above,
    integers,
    owned,
)
from powermean.data.sample_fe_state import FESample
from powermean.fe.absorb_numeric import (
    Coefficients,
    checked_difference,
    checked_product,
    checked_sum,
    group_sums,
)
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

Kind = Literal["direct_svd", "pivoted_qr", "triangular"]
KINDS: tuple[Kind, ...] = get_args(Kind)
Outcome = Literal["ok", "failed", "unusable"]
OUTCOMES: tuple[Outcome, ...] = get_args(Outcome)
# The one column of the final absorption: a positional string, never a user label.
FINAL_COLUMNS = ("final 0",)
UNUSABLE = "the output is unusable"
EMPTY = np.zeros(0)


def joint_columns(n_slopes: int) -> tuple[str, ...]:
    """``("joint 0", ..., "joint K")``: ``t`` first, then each slope, by position."""
    return tuple(f"joint {j}" for j in range(n_slopes + 1))


# --------------------------------------------------------------- the arithmetic


def signed_response(sample: FESample) -> FloatArray:
    """``t = l - o``: the log response net of the raw offset, once, checked.

    ``|log Y|`` is below 746 for every positive double, so the difference is finite
    for every finite offset, and a tiny difference of two doubles is exact.
    """
    problem = sample.problem
    found = checked_difference(
        problem.response, problem.offset, "response net of the offset"
    )
    return immutable_array(found)


def raw_weights(sample: FESample) -> FloatArray:
    """``a``: the sample's raw weights (ones without weights), never rescaled."""
    return sample.data.weight_values


def joint_values(sample: FESample, t: FloatArray) -> FloatArray:
    """``[t, Xs]``: the columns of the joint absorption, ``t`` first."""
    design = sample.design
    if design is None:
        return immutable_array(t[:, None])
    return immutable_array(np.column_stack([t, design.matrix]))


def combination(columns: FloatArray, coefficients: FloatArray, what: str) -> FloatArray:
    """``columns @ coefficients``: the checked products added in column order.

    At least one column; every product and partial sum is checked, ``what`` names them.
    """
    total = checked_product(columns[:, 0], coefficients[:1], what)
    for j in range(1, coefficients.size):
        term = checked_product(columns[:, j], coefficients[j : j + 1], what)
        total = checked_sum(total, term, what)
    return total


def seeds(coefficients: Coefficients, theta: FloatArray) -> Coefficients:
    """``alpha_b = a_t,b - a_X,b theta`` per block, one ``L_b``-by-1 array each.

    ``coefficients`` are the joint state's ``[a_t,b, a_X,b]``; without slopes the seed
    is ``a_t,b`` itself and no product is formed.  Checked.
    """
    found: list[FloatArray] = []
    for block in coefficients:
        alpha: FloatArray = block[:, 0]
        if theta.size:
            product = combination(block[:, 1:], theta, "seed product")
            alpha = checked_difference(alpha, product, "seed")
        found.append(immutable_array(alpha[:, None]))
    return tuple(found)


def fixed_slope_residual(
    sample: FESample, t: FloatArray, theta: FloatArray
) -> FloatArray:
    """``t - Xs theta``: the column of the final absorption; ``t`` without slopes."""
    design = sample.design
    if design is None:
        return t
    product = combination(design.matrix, theta, "slope index")
    return checked_difference(t, product, "fixed-slope residual")


def base_index(sample: FESample, beta: FloatArray) -> FloatArray:
    """``o + Xraw beta`` on the original active columns; ``o`` without slopes."""
    offset = sample.problem.offset
    if not sample.active.size:
        return offset
    raw = sample.data.design[:, sample.active]
    return checked_sum(offset, combination(raw, beta, "original index"), "base index")


def reconstruct(
    base: FloatArray, effects: tuple[FloatArray, ...], grouping: Grouping
) -> FloatArray:
    """``((base + alpha_1[c_1]) + alpha_2[c_2]) + ...``: gathers in block order."""
    q = base
    for block, alpha in zip(grouping.blocks, effects, strict=True):
        q = checked_sum(q, alpha[block.codes], "reconstructed index")
    return q


def raw_slope_sums(sample: FESample, weighted: FloatArray) -> FloatArray:
    """``Xraw'(a e)``: each raw active column's checked products, summed by row."""
    if not sample.active.size:
        return immutable_array(np.zeros(0))
    raw = sample.data.design[:, sample.active]
    terms = checked_product(raw, weighted[:, None], "slope equation term")
    single = np.zeros(sample.count, dtype=np.intp)
    return immutable_array(group_sums(terms, single, 1)[0])


def _own(record: object, **values: object) -> None:
    """Store the derived values of a frozen record."""
    for name, value in values.items():
        object.__setattr__(record, name, value)


# ---------------------------------------------------------------- the attempts


@dataclass(frozen=True, eq=False)
class Decomposition:
    """One LAPACK attempt of the slope solve, as its checked output stood.

    ``kind``: ``direct_svd`` (singular values of ``W = sqrt(a) Xtilde``), ``pivoted_qr``
    (``|diag R|`` and pivots of ``W[:, P] = Q R``) or ``triangular`` (``z`` of ``R z =
    Q'r``, pivoted order); ``shape``: that of ``W``, physical rows; ``outcome``: ``ok``,
    ``failed`` (``LinAlgError``) or ``unusable`` (an output failed its checks).  Only an
    ``ok`` attempt keeps values (and pivots).  An ``ok`` SVD derives ``cutoff = eps
    max(n, K) s_max`` and ``rank``, the count strictly above it; else both are ``None``.
    """

    kind: Kind
    shape: tuple[int, int]
    outcome: Outcome
    values: FloatArray
    pivots: IndexArray | None = None
    cutoff: float | None = field(init=False)
    rank: int | None = field(init=False)

    def __post_init__(self) -> None:
        """Check the attempt and derive the cutoff and rank of a direct SVD."""
        if self.kind not in KINDS or self.outcome not in OUTCOMES:
            raise ValueError("unknown decomposition kind or outcome")
        shape = integers(self.shape, "shape")
        values = owned(self.values)
        if len(shape) != 2 or not all(shape):
            raise ValueError("shape must be two positive sizes")
        _own(self, shape=shape, values=values, cutoff=None, rank=None)
        if self.outcome != "ok":
            if values.size or self.pivots is not None:
                raise ValueError("a failed or unusable attempt keeps no output")
        elif self.kind == "direct_svd":
            self._singular(values)
        elif self.kind == "pivoted_qr":
            self._pivoted(values)
        elif values.size != shape[1] or self.pivots is not None:
            raise ValueError("a back substitution solves for one value per column")

    def _singular(self, values: FloatArray) -> None:
        """``min(shape)`` nonnegative nonincreasing values; their cutoff and rank."""
        ordered = bool(np.all(values[1:] <= values[:-1]) and np.all(values >= 0))
        if values.size != min(self.shape) or not ordered or self.pivots is not None:
            raise ValueError("an SVD keeps min(shape) ordered singular values")
        cutoff = EPSILON * max(self.shape) * float(values[0])
        _own(self, cutoff=cutoff, rank=count_above(values, cutoff))

    def _pivoted(self, values: FloatArray) -> None:
        """One nonnegative ``|R_jj|`` per column and the pivots, a permutation."""
        columns = self.shape[1]
        order = () if self.pivots is None else integers(self.pivots, "pivots")
        permutation = sorted(order) == list(range(columns))
        if values.size != columns or np.any(values < 0) or not permutation:
            raise ValueError("a pivoted QR keeps |diag R| and a permutation")
        _own(self, pivots=frozen_index(np.asarray(order, dtype=np.intp)))


def full_rank(svd: Decomposition, n_slopes: int) -> bool:
    """Whether a direct SVD attempt found the full column rank ``K``."""
    return svd.rank == n_slopes


def slopes(qr: Decomposition, triangular: Decomposition) -> FloatArray:
    """``theta`` with ``theta[P] = z``: the back substitution in the formula order."""
    if qr.pivots is None:
        raise ValueError("the QR attempt holds no pivots")
    theta = np.empty(qr.shape[1])
    theta[qr.pivots] = triangular.values
    return immutable_array(theta)


# ------------------------------------------------------------ the LAPACK calls


class Qr(Protocol):
    """``scipy.linalg.qr`` with column pivoting and the economic factors."""

    def __call__(
        self, a: FloatArray, /, *, mode: str, pivoting: bool, check_finite: bool
    ) -> tuple[FloatArray, FloatArray, IndexArray]:
        """``Q``, ``R`` and the pivots ``P`` of ``a[:, P] = Q R``."""
        ...


class SolveTriangular(Protocol):
    """``scipy.linalg.solve_triangular`` on an upper triangular matrix."""

    def __call__(
        self, a: FloatArray, b: FloatArray, /, *, lower: bool, check_finite: bool
    ) -> FloatArray:
        """The solution of ``a x = b`` by back substitution."""
        ...


class Linalg(Protocol):
    """The objects of ``scipy.linalg`` that this module uses."""

    qr: Qr
    solve_triangular: SolveTriangular


linalg = cast("Linalg", importlib.import_module("scipy.linalg"))


def direct_svd(matrix: FloatArray) -> tuple[Decomposition, str]:
    """The singular values of ``W`` and its numerical rank, checked and recorded.

    The rank and cutoff are those the record derives; the text is the library's error
    of a failed call, ``UNUSABLE`` for an output that fails its checks, else empty.
    """
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    try:
        output = np.linalg.svd(matrix, compute_uv=False, hermitian=False)
    except np.linalg.LinAlgError as error:
        return Decomposition("direct_svd", shape, "failed", EMPTY), str(error)
    values = _singular(output, min(shape))
    if values is None:
        return Decomposition("direct_svd", shape, "unusable", EMPTY), UNUSABLE
    return Decomposition("direct_svd", shape, "ok", values), ""


class Factors(NamedTuple):
    """The checked factors ``Q`` and ``R`` of a pivoted economic QR."""

    q: FloatArray
    r: FloatArray


def pivoted_qr(matrix: FloatArray) -> tuple[Decomposition, Factors | None, str]:
    """The pivoted economic QR of a full-column-rank ``W``, checked and recorded.

    ``R`` must be ``K``-by-``K`` and finite with a permutation of pivots,
    ``Q`` a finite real ``n``-by-``K`` array; ``None`` factors otherwise.
    """
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    try:
        q, r, pivots = linalg.qr(
            matrix, mode="economic", pivoting=True, check_finite=False
        )
    except np.linalg.LinAlgError as error:
        return Decomposition("pivoted_qr", shape, "failed", EMPTY), None, str(error)
    checked = _pivoted(r, pivots, shape)
    basis = np.asarray(q)
    if (
        checked is None
        or basis.dtype.kind != "f"
        or basis.shape != shape
        or not np.isfinite(basis).all()
    ):
        return Decomposition("pivoted_qr", shape, "unusable", EMPTY), None, UNUSABLE
    diagonal, order = checked
    record = Decomposition("pivoted_qr", shape, "ok", diagonal, pivots=order)
    factors = Factors(basis.astype(np.float64), np.asarray(r, dtype=np.float64))
    return record, factors, ""


def back_substitution(
    factors: Factors, projected: FloatArray
) -> tuple[Decomposition, str]:
    """``z`` of ``R z = Q'r``, checked (finite real, one per column) and recorded."""
    shape = (int(factors.q.shape[0]), int(factors.q.shape[1]))
    try:
        output = linalg.solve_triangular(
            factors.r, projected, lower=False, check_finite=False
        )
    except np.linalg.LinAlgError as error:
        return Decomposition("triangular", shape, "failed", EMPTY), str(error)
    found = np.asarray(output)
    usable = found.dtype.kind == "f" and found.shape == (shape[1],)
    values = _doubles(found) if usable else None
    if values is None:
        return Decomposition("triangular", shape, "unusable", EMPTY), UNUSABLE
    return Decomposition("triangular", shape, "ok", values), ""
