"""The two bounded linear programs of the GPML recession diagnostic.

At joint ``tau = 2``: a linear program, and ``ExistenceError`` with the direction when
the condition fails.  Internal module.  The programs use largest-magnitude slope
scales, bounded homogeneous programs and a separate boundary program.

Data (:func:`prepare`).  ``s_j = max_i |X_ij|``, 1 for a zero column, not a power of
two; ``X / s``; ``rho = (a / max a) / sum(a / max a)`` and ``rhoP``
by the same procedure on the rows of ``P = {Z > 0}``.  Every quotient runs with each
floating-point condition raised (:mod:`powermean.fe.absorb_numeric`): no positive
weight can vanish, and a lost quotient raises ``FloatingPointError``.  With
``A = [X / s, F]``, ``F`` every level of every block in block order (redundant levels
included, never formed densely), ``t = A' rho`` and ``p = A_P' rhoP`` are
range-checked row-order column sums and level sums.

Programs (:func:`call`).  LP1 (``negative``): minimize ``t'v`` subject to
``-A_P v <= 0`` and ``-1 <= v <= 1``; LP2 (``zero_sum``): minimize ``-p'v`` under the
same and ``t'v = 0``.  ``-A_P`` is sparse (CSR), one row per row of ``P``, every
column; its nonzeros, and LP2's stacked ``[A_P; t']``, are counted before allocation
and refused above :data:`NONZERO_LIMIT` (an internal resource limit, not a
statement about the memory of HiGHS).  A call is HiGHS dual simplex with the explicit
options of :class:`~powermean.solve.support_lp.LPControls`.  It is charged before the
optimizer runs: its :class:`LPReport` is established with the program, the controls and
the sizes of the matrices built for it, every other field unavailable, and the call
then fills it in.  Every warning raised during the call is recorded.  A ``MemoryError``
of the call (:data:`RAISED`) ends it as ``lp_failure`` with that exception's text and
the warnings before it; any other exception propagates.  Every field of a returned
report is validated before it is converted or used (:func:`_report`): a field that is
missing or not of its genuine type, shape, domain or finiteness is unavailable
(``None``), never coerced and never a fabricated zero.  A report is usable evidence
only when no warning was raised, the status is 0, ``success`` is ``True`` and every
other field is available; otherwise it is ``lp_failure``.  That rule has one
definition, :func:`usable`, which a result applies again to every report it admits
whose ``failure`` is empty: an empty ``failure`` alone admits nothing.

Raw check (:func:`raw_check`), on a usable point ``v`` before any mapping:
``max |v| <= 1 + f`` (``lp_box``); ``min_P (A v) >= -f`` and, in LP2, ``|t'v| <= f``
(``lp_constraints``), ``A v`` reconstructed from ``X / s`` and the lift (:func:`rows`);
``|c'v - fun| <= e sum |c_j|`` (``lp_objective``) through
:func:`~powermean.solve.support_witness.relative` (an exact zero scale needs an exact
zero difference).  The fresh witnesses are :mod:`powermean.solve.support_gamma_witness`.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast

import numpy as np

from powermean.fe.absorb_numeric import (
    checked_product,
    checked_quotient,
    checked_sum,
    group_sums,
    lift,
)
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.support_gamma_problem import GammaProblem
from powermean.solve.support_lp import METHOD, LinprogResult, LPControls
from powermean.solve.support_problem import indices
from powermean.solve.support_witness import relative

Stage = Literal["negative", "zero_sum"]
Marginal = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
Effects = tuple[FloatArray, ...]
# The 10,000,000-nonzero guard.
NONZERO_LIMIT = 10_000_000
# The exception a valid call can raise at run time, read in SciPy's linprog, its
# HiGHS adapter and wrapper (SciPy 1.17.0 and 1.18.1): an allocation failure
# (NumPy's, SciPy's sparse conversions', or HiGHS's
# std::bad_alloc through pybind11).  HiGHS returns its errors as a status; the input
# checks raise TypeError or ValueError only for malformed inputs, which this module
# never builds, so those and every other exception propagate.
RAISED = MemoryError
# SciPy's documented linprog statuses: 0 optimal, 1 limit, 2 infeasible, 3 unbounded,
# 4 numerical difficulties.
STATUSES = range(5)


class SparseMatrix(Protocol):
    """A sparse matrix of SciPy, as these programs pass it."""

    @property
    def nnz(self) -> int:
        """The stored nonzeros."""
        ...


class CsrArray(Protocol):
    """``scipy.sparse.csr_array`` from coordinates or from a dense matrix."""

    def __call__(
        self,
        arg1: tuple[FloatArray, tuple[IndexArray, IndexArray]] | FloatArray,
        /,
        *,
        shape: tuple[int, int],
    ) -> SparseMatrix:
        """The CSR array."""
        ...


class Sparse(Protocol):
    """The object of ``scipy.sparse`` that this module uses."""

    csr_array: CsrArray


class Linprog(Protocol):
    """``scipy.optimize.linprog`` with sparse constraint arrays, by position."""

    def __call__(
        self,
        c: FloatArray,
        a_ub: SparseMatrix | None,
        b_ub: FloatArray | None,
        a_eq: SparseMatrix | None,
        b_eq: FloatArray | None,
        /,
        *,
        bounds: tuple[float, float],
        method: str,
        options: Mapping[str, float | int | bool],
    ) -> LinprogResult:
        """The solution of the linear program."""
        ...


class Optimize(Protocol):
    """The object of ``scipy.optimize`` that this module uses."""

    linprog: Linprog


sparse = cast("Sparse", importlib.import_module("scipy.sparse"))
optimize = cast("Optimize", importlib.import_module("scipy.optimize"))


# ---------------------------------------------------------------------------- data


def sums(values: FloatArray, codes: IndexArray, count: int) -> FloatArray:
    """Row-order sums by code, range-checked; zeros when there is nothing to sum.

    The accepted :func:`~powermean.fe.absorb_numeric.group_sums` refuses an empty
    input; the empty sum is zero.
    """
    if not values.size:
        return np.zeros((count, *values.shape[1:]))
    return group_sums(values, codes, count)


def single(n: int) -> IndexArray:
    """The code 0 for each of ``n`` rows: one group, every row in it."""
    return indices(np.zeros(n))


def total(values: FloatArray) -> FloatArray:
    """The row-order sum of a vector, range-checked, as a one-element array."""
    return sums(values, single(values.size), 1)


def dot(a: FloatArray, b: FloatArray) -> float:
    """``a'b``: range-checked products summed in order."""
    return float(total(checked_product(a, b, "inner product"))[0])


def normalized(a: FloatArray) -> FloatArray:
    """``(a / max a) / sum(a / max a)``, each quotient range-checked; empty for none.

    ``FloatingPointError`` when a quotient is not representable, a positive weight
    rounding to zero or to an inexact subnormal included: no positive weight vanishes.
    """
    if not a.size:
        return immutable_array(np.zeros(0))
    divided = checked_quotient(a, np.array([np.max(a)]), "normalized weights")
    return immutable_array(checked_quotient(divided, total(divided), "normalized"))


def scales(design: FloatArray) -> FloatArray:
    """``s_j = max_i |X_ij|`` per column, 1 for a zero column."""
    largest = np.max(np.abs(design), axis=0) if design.size else np.zeros(0)
    return immutable_array(np.where(largest == 0, 1.0, largest))


@dataclass(frozen=True, eq=False)
class LPData:
    """The scaled design, the normalized weights and the two objectives.

    ``scales`` are ``s`` and ``scaled`` is ``X / s``; ``weights`` is ``rho`` and
    ``positive_weights`` ``rhoP`` (empty without positive rows); ``total`` is ``t`` and
    ``positive_total`` is ``p``, the slopes then every level of every block.
    """

    scales: FloatArray
    scaled: FloatArray
    weights: FloatArray
    positive_weights: FloatArray
    total: FloatArray
    positive_total: FloatArray


def objective(
    problem: GammaProblem, scaled: FloatArray, rows_r: IndexArray, w: FloatArray
) -> FloatArray:
    """``A_R' w``: scaled slope column sums, then the level sums of every block."""
    products = checked_product(scaled[rows_r], w[:, None], "objective products")
    parts = [sums(products, single(rows_r.size), 1)[0]]
    blocks = () if problem.grouping is None else problem.grouping.blocks
    parts += [sums(w, b.codes[rows_r], b.n_levels) for b in blocks]
    return immutable_array(np.concatenate(parts))


def prepare(problem: GammaProblem) -> LPData:
    """The data of both programs on the current sample (``FloatingPointError``)."""
    s = scales(problem.design)
    scaled = checked_quotient(problem.design, s[None, :], "scaled design")
    rho = normalized(problem.weights.values)
    positive = problem.positive
    rho_p = normalized(problem.weights.values[positive])
    every = indices(np.arange(problem.n_rows))
    return LPData(
        s,
        immutable_array(scaled),
        rho,
        rho_p,
        objective(problem, scaled, every, rho),
        objective(problem, scaled, positive, rho_p),
    )


def stage_objective(data: LPData, stage: Stage) -> FloatArray:
    """``c``: ``t`` for LP1, ``-p`` for LP2."""
    return data.total if stage == "negative" else immutable_array(-data.positive_total)


# ---------------------------------------------------------------- resource guard


def constraint_nonzeros(problem: GammaProblem) -> int:
    """The nonzeros of ``A_P``, counted from ``X`` and the blocks, nothing allocated.

    A nonzero ``X_ij`` stays nonzero in ``X / s`` (an underflowing quotient refuses).
    """
    per_row = np.count_nonzero(problem.design, axis=1)
    found = int(per_row[problem.positive].sum())
    return found + problem.positive.size * problem.n_blocks


def within_limit(nonzeros: int) -> bool:
    """Whether a matrix of ``nonzeros`` stored entries may be constructed."""
    return nonzeros <= NONZERO_LIMIT


def _inequality(problem: GammaProblem, data: LPData) -> SparseMatrix:
    """``-A_P`` in CSR: the nonzero scaled slopes, then ``-1`` at each block's level."""
    positive = problem.positive
    block = data.scaled[positive]
    r, c = np.nonzero(block)
    row_parts, column_parts, value_parts = [r], [c], [-block[r, c]]
    offset = problem.n_slopes
    for b in () if problem.grouping is None else problem.grouping.blocks:
        row_parts.append(np.arange(positive.size))
        column_parts.append(offset + b.codes[positive])
        value_parts.append(-np.ones(positive.size))
        offset += b.n_levels
    where = (indices(np.concatenate(row_parts)), indices(np.concatenate(column_parts)))
    values: FloatArray = np.concatenate(value_parts)
    return sparse.csr_array((values, where), shape=(positive.size, offset))


# ------------------------------------------------------------------------- calls


def own(record: object, arrays: tuple[str, ...], tuples: tuple[str, ...] = ()) -> None:
    """Store owned read-only copies of a frozen record's arrays and tuples of arrays.

    Each copy holds bytes of its own (:func:`~powermean.fe.reductions.immutable_array`),
    made when the record is constructed, ``dataclasses.replace`` included: a caller's
    array, a same-byte one included, is never retained, a later write to it changes
    nothing, and the write flag of a stored copy cannot be set again.  ``None`` stays.
    """
    for name in arrays:
        value = getattr(record, name)
        if value is not None:
            object.__setattr__(record, name, immutable_array(value))
    for name in tuples:
        value = getattr(record, name)
        if value is not None:
            object.__setattr__(record, name, tuple(immutable_array(a) for a in value))


@dataclass(frozen=True, eq=False)
class LPReport:
    """One charged call, established before the optimizer runs, then its report.

    ``stage``, ``controls``, ``rows`` and ``nonzeros`` are the program, its controls and
    the sizes of the constraint matrices built for it (``A_P``, and for LP2 ``t'``).
    The optimizer's report follows, each field ``None`` unless returned as a genuine
    value of its type (:func:`_report`), never a fabricated zero: ``status`` (0 to 4),
    ``success``, ``message``, ``iterations`` (``nit``), ``reported`` (``fun``),
    ``point`` (``v``) and ``marginals`` (``m_I``, ``m_E``, ``m_L``, ``m_U``, finite
    doubles of the program's shapes, owned and read-only).  ``warnings`` are those
    raised during the call and ``raised`` the text of the :data:`RAISED` exception
    that ended it.  ``failure`` is empty only for usable evidence (:func:`usable`), else
    ``lp_failure``; the record does not check that relation itself, a result does
    before it admits the report.
    """

    stage: Stage
    controls: LPControls
    rows: int
    nonzeros: int
    status: int | None = None
    success: bool | None = None
    message: str | None = None
    iterations: int | None = None
    reported: float | None = None
    warnings: tuple[str, ...] = ()
    raised: str | None = None
    failure: str = "lp_failure"
    point: FloatArray | None = None
    marginals: Marginal | None = None

    def __post_init__(self) -> None:
        """Own the point, the marginals and the warnings."""
        own(self, ("point",), ("marginals",))
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _integer(value: object) -> int | None:
    """A Python or NumPy integer, never a boolean, as ``int``; ``None`` otherwise."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        return None
    return int(value)


def _flag(value: object) -> bool | None:
    """A Python or NumPy boolean as ``bool``; else ``None``, never a truth value."""
    return bool(value) if isinstance(value, bool | np.bool_) else None


def _number(value: object) -> float | None:
    """A finite real scalar of at most double precision as ``float``; else ``None``."""
    if not isinstance(value, float | np.float16 | np.float32):
        return None
    found = float(value)
    return found if np.isfinite(found) else None


def _vector(value: object, size: int) -> FloatArray | None:
    """An owned copy of a finite real vector of ``size`` doubles; ``None`` otherwise.

    Only an array of a real floating type of at most 64 bits qualifies: the copy to
    doubles is exact, and nothing nonnumeric or wider is converted.
    """
    if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
        return None
    if value.dtype.itemsize > 8 or value.shape != (size,):
        return None
    found = immutable_array(value)
    return found if np.isfinite(found).all() else None


def _marginals(result: object, shapes: tuple[int, int, int, int]) -> Marginal | None:
    """The four marginal arrays when every block returns a valid one, else ``None``."""
    arrays = []
    for name, size in zip(("ineqlin", "eqlin", "lower", "upper"), shapes, strict=True):
        found = _vector(getattr(getattr(result, name, None), "marginals", None), size)
        if found is None:
            return None
        arrays.append(found)
    return arrays[0], arrays[1], arrays[2], arrays[3]


def usable(report: LPReport) -> bool:
    """Whether ``report`` is usable evidence of its program: the one rule.

    No warning and no exception (``raised``), the status the integer 0 and ``success``
    the boolean ``True``; the message a string, the iteration count a nonnegative
    integer, the objective a finite float, the point a finite real vector of ``k``
    doubles and the four marginal blocks finite real vectors of ``rows - e``, ``e``,
    ``k`` and ``k`` (``e`` 1 for LP2, else 0; ``rows`` an integer, ``k`` the point's
    length).  Each field is checked as :func:`_report` checks a returned one (Python
    and NumPy scalars of their genuine types), nothing converted.  Binding ``k`` and
    the shapes to the current sample is the replay's.
    """
    point, rows = report.point, _integer(report.rows)
    if rows is None or not isinstance(point, np.ndarray) or point.ndim != 1:
        return False
    k, equalities = point.shape[0], 1 if report.stage == "zero_sum" else 0
    shapes = (rows - equalities, equalities, k, k)
    found, nit = report.marginals or (), _integer(report.iterations)
    blocks = len(found) == len(shapes) and all(
        _vector(m, size) is not None for m, size in zip(found, shapes, strict=True)
    )
    return (
        not report.warnings
        and report.raised is None
        and _integer(report.status) == 0
        and _flag(report.success) is True
        and isinstance(report.message, str)
        and nit is not None
        and nit >= 0
        and _number(report.reported) is not None
        and _vector(point, k) is not None
        and blocks
    )


def _report(
    attempt: LPReport, result: object, caught: tuple[str, ...], k: int
) -> LPReport:
    """The established call filled in from a returned report, each field validated.

    A missing attribute is an unavailable field; nothing is converted before its type,
    shape, domain and finiteness are checked.  Usable only by :func:`usable`: no
    warning was raised, the status is 0, ``success`` is ``True`` and every other field
    is available.
    """
    equalities = 1 if attempt.stage == "zero_sum" else 0
    status = _integer(getattr(result, "status", None))
    nit = _integer(getattr(result, "nit", None))
    message = getattr(result, "message", None)
    found = replace(
        attempt,
        status=status if status in STATUSES else None,
        success=_flag(getattr(result, "success", None)),
        message=str(message) if isinstance(message, str) else None,
        iterations=nit if nit is not None and nit >= 0 else None,
        reported=_number(getattr(result, "fun", None)),
        warnings=caught,
        point=_vector(getattr(result, "x", None), k),
        marginals=_marginals(result, (attempt.rows - equalities, equalities, k, k)),
    )
    return replace(found, failure="") if usable(found) else found


def call(
    problem: GammaProblem, data: LPData, stage: Stage, controls: LPControls
) -> LPReport:
    """One charged HiGHS dual-simplex call of ``stage``, every warning recorded.

    The report is established, with the sizes of the matrices built, before the
    optimizer runs.  LP2 needs a nonempty ``P``; the caller checks the nonzeros
    against :data:`NONZERO_LIMIT` before this call allocates its matrices.
    """
    k = problem.n_parameters
    a_ub = _inequality(problem, data) if problem.positive.size else None
    b_ub = None if a_ub is None else np.zeros(problem.positive.size)
    a_eq = b_eq = None
    if stage == "zero_sum":
        a_eq = sparse.csr_array(np.array(data.total[None, :]), shape=(1, k))
        b_eq = np.zeros(1)
    nonzeros = sum(0 if a is None else a.nnz for a in (a_ub, a_eq))
    rows_m = problem.positive.size + (0 if a_eq is None else 1)
    attempt = LPReport(stage, controls, rows_m, nonzeros)
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        try:
            result = optimize.linprog(
                np.array(stage_objective(data, stage)),
                a_ub,
                b_ub,
                a_eq,
                b_eq,
                bounds=(-1.0, 1.0),
                method=METHOD,
                options=controls.options,
            )
        except RAISED as error:
            caught = tuple(str(item.message) for item in found)
            raised = f"{type(error).__name__}: {error}"
            return replace(attempt, warnings=caught, raised=raised)
    caught = tuple(str(item.message) for item in found)
    return _report(attempt, result, caught, k)


# ------------------------------------------------------------------- raw check


def split(problem: GammaProblem, v: FloatArray) -> tuple[FloatArray, Effects]:
    """``v`` as slope coordinates and one array per block, copies, in block order."""
    found, start = [], problem.n_slopes
    for block in () if problem.grouping is None else problem.grouping.blocks:
        found.append(np.array(v[start : start + block.n_levels]))
        start += block.n_levels
    return np.array(v[: problem.n_slopes]), tuple(found)


def rows(
    design: FloatArray, grouping: Grouping | None, slopes: FloatArray, effects: Effects
) -> FloatArray:
    """``design slopes + lift(effects)``: columns, then blocks, range-checked."""
    z: FloatArray = np.zeros(design.shape[0])
    for j in range(design.shape[1]):
        term = checked_product(design[:, j], slopes[j], "witness product")
        z = checked_sum(z, term, "witness sum")
    if grouping is not None:
        lifted = lift(tuple(a[:, None] for a in effects), grouping)
        z = checked_sum(z, lifted[:, 0], "witness lift")
    return z


def ratio(numerator: float, scale: float) -> float:
    """``relative(|numerator|, scale)`` for one scalar (exact-zero branches)."""
    return float(relative(np.array([abs(numerator)]), np.array([scale]))[0])


@dataclass(frozen=True)
class RawValues:
    """The values of a raw check.

    ``activity`` is ``min_P A v`` (``None`` without positive rows), ``balance`` ``t'v``
    (LP2), ``objective`` ``c'v``, ``scale`` ``sum |c_j|`` and ``agreement`` the ratio of
    ``|c'v - fun|`` to that scale.
    """

    activity: float | None
    balance: float | None
    objective: float
    scale: float
    agreement: float


@dataclass(frozen=True)
class RawCheck:
    """The raw check of a point.

    ``failure`` is empty when it passes; ``box`` is ``max |v|``; ``values`` are set once
    computed.
    """

    failure: str
    box: float
    values: RawValues | None = None


def raw_check(
    problem: GammaProblem,
    data: LPData,
    stage: Stage,
    point: tuple[FloatArray, float],
    controls: LPControls,
) -> RawCheck:
    """Box, constraints and objective of the boxed ``v`` against the reported ``fun``.

    ``point`` is ``(v, fun)``.
    """
    v, fun = point
    f, box = controls.feasibility, float(np.max(np.abs(v)))
    if not box <= 1.0 + f:
        return RawCheck("lp_box", box)
    c = stage_objective(data, stage)
    try:
        slopes, effects = split(problem, v)
        activity = rows(data.scaled, problem.grouping, slopes, effects)[
            problem.positive
        ]
        balance = dot(data.total, v) if stage == "zero_sum" else None
        found, scale = dot(c, v), float(total(np.abs(c))[0])
        agreement = ratio(found - fun, scale)
    except FloatingPointError:
        return RawCheck("range", box)
    least = float(np.min(activity)) if activity.size else None
    values = RawValues(least, balance, found, scale, agreement)
    infeasible = least is not None and not least >= -f
    if infeasible or (balance is not None and not abs(balance) <= f):
        return RawCheck("lp_constraints", box, values)
    if not agreement <= controls.tolerance:
        return RawCheck("lp_objective", box, values)
    return RawCheck("", box, values)
