"""The SciPy factors, their provenance and the counted actions of the certificate.

Internal module: every call
that the certificate makes to SciPy's sparse routines, inspected at the declared
minimum 1.15.0 and executed on the installed release; no dense inverse, truncation,
ridge, pseudoinverse, cutoff, incomplete factor or other solver.

Factors: ``splu`` of a private writable CSC copy, ``permc_spec='MMD_AT_PLUS_A'``,
``relax=None``, ``panel_size=None`` and the options ``Fact='DOFACT'``, ``Equil=False``,
``IterRefine='NOREFINE'``, ``RowPerm='NOROWPERM'``, ``ReplaceTinyPivot=False``; the
ordinary ``A`` with ``diag_pivot_thresh=1``, ``SymmetricMode=False``, the curvature
``Aplus`` with threshold ``0``, ``SymmetricMode=True``; the fixed merit metric of a
selected root's solve (:mod:`powermean.solve.absorbed_root`), kind ``"merit"``, with
the curvature's threshold and options, so that the same pivot test decides its
positive definiteness.  An index width beyond ``intc``
refuses before the call, an exactly singular factor fails, and only copies of the
permutations, nonzeros and the curvature and merit diagonals are kept.  Each record
keeps its
provenance: the version of the SciPy that runs, and the complete keywords of the call
that was made (``None`` when it was refused before any call).  No SuperLU version is
inferred: SciPy does not report one.

The admitted ordinary diagonal-Schur route may instead use the internal packed
partial-pivoting dense LU backend, including final projection and certificate
actions. Its record names dgetrf and does not invent full L/U nonzero counts.
An eligible curvature Schur remainder within the existing dense workspace bound
may instead use the dpotrf Cholesky factor and dtrtrs whitening. Its
record names the actual call, keeps implicit LDL pivots diag(R)**2 and no L/U
counts; failed leading minors are not retried. Ineligible curvature and every
merit factor retain the sparse policy above. Full-system actions, ordinary
conditioning estimates and inference are unchanged.

Actions: ``F = A^-1`` is a float64 ``LinearOperator`` whose actions are
``SuperLU.solve`` with ``trans='N'`` (matvec, matmat) and ``'T'`` (rmatvec, rmatmat),
vector and one-column shapes, no dtype probe; ``k = onenormest(F, t=1, itmax=5)``,
whose random branches need ``t > 1``; the unit lower triangular solve of a private
``L`` by ``spsolve_triangular`` (lower, unit diagonal, no overwrite).  Every solve is
counted, with its right-hand sides, before it is made.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from powermean.data.accounting import frozen_index
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_point import StepStatus, WorkStep

FactorKind = Literal["ordinary", "curvature", "merit"]
FactorStatus = Literal["factored", "failed", "refused"]
LIBRARY: Mapping[str, object] = MappingProxyType(
    {"permc_spec": "MMD_AT_PLUS_A", "relax": None, "panel_size": None}
)
_FIXED = {
    "Fact": "DOFACT",
    "Equil": False,
    "IterRefine": "NOREFINE",
    "RowPerm": "NOROWPERM",
    "ReplaceTinyPivot": False,
}
OPTIONS: Mapping[FactorKind, Mapping[str, object]] = MappingProxyType(
    {
        "ordinary": MappingProxyType({**_FIXED, "SymmetricMode": False}),
        "curvature": MappingProxyType({**_FIXED, "SymmetricMode": True}),
        "merit": MappingProxyType({**_FIXED, "SymmetricMode": True}),
    }
)
THRESHOLDS: Mapping[FactorKind, float] = MappingProxyType(
    {"ordinary": 1.0, "curvature": 0.0, "merit": 0.0}
)
TRIANGULAR: Mapping[str, bool] = MappingProxyType(
    {"lower": True, "unit_diagonal": True, "overwrite_A": False, "overwrite_b": False}
)
INDEX_LIMIT = int(np.iinfo(np.intc).max)


# ----------------------------------------------------------- SciPy interfaces


class CscView(Protocol):
    """The parts of a SciPy CSC array read here."""

    @property
    def data(self) -> FloatArray:
        """The stored values."""
        ...

    @property
    def indptr(self) -> NDArray[np.integer]:
        """The column pointers."""
        ...

    @property
    def indices(self) -> NDArray[np.integer]:
        """The row index of every stored value."""
        ...

    @property
    def nnz(self) -> int:
        """The stored entries."""
        ...

    def diagonal(self) -> FloatArray:
        """The diagonal."""
        ...


class SuperLU(Protocol):
    """A SciPy ``SuperLU`` factor, as used here (``L`` and ``U`` read by name)."""

    @property
    def perm_r(self) -> NDArray[np.intc]:
        """The row permutation."""
        ...

    @property
    def perm_c(self) -> NDArray[np.intc]:
        """The column permutation."""
        ...

    def solve(self, rhs: FloatArray, trans: str = "N") -> FloatArray:
        """``A x = rhs`` (``trans='N'``) or ``A' x = rhs`` (``trans='T'``)."""
        ...


class Whitening(Protocol):
    """A symmetric remainder factor with the whitening action R^-1 v."""

    def whiten(self, rhs: FloatArray) -> FloatArray:
        """R^-1 rhs for one vector, where A = R R'."""
        ...


class Linalg(Protocol):
    """The objects of ``scipy.sparse.linalg`` used here."""

    def splu(self, a: object, /, **options: object) -> SuperLU:
        """The sparse LU factor."""
        ...

    def spsolve_triangular(
        self, a: object, b: FloatArray, /, **options: bool
    ) -> FloatArray:
        """A sparse triangular solve."""
        ...

    def onenormest(self, a: object, /, *, t: int, itmax: int) -> float:
        """The 1-norm estimate of an operator."""
        ...

    LinearOperator: Callable[..., object]


class Sparse(Protocol):
    """The object of ``scipy.sparse`` used here."""

    def csc_array(self, arg: object, /, *, shape: tuple[int, int]) -> CscView:
        """A CSC array of ``(data, indices, indptr)``."""
        ...


class Library(Protocol):
    """The ``scipy`` package, for the version that runs."""

    __version__: str


linalg = cast("Linalg", importlib.import_module("scipy.sparse.linalg"))
sparse = cast("Sparse", importlib.import_module("scipy.sparse"))
VERSION = str(cast("Library", importlib.import_module("scipy")).__version__)


# -------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class FactorRecord:
    """One factor attempt: kind, threshold, options, outcome, diagnostics, provenance.

    ``perm_r``, ``perm_c``, the nonzeros of ``L`` and ``U`` and (curvature) the
    diagonal ``d`` of ``U`` (or implicit LDL pivots for dense Cholesky) are
    ``None`` when no factor was returned: unavailable.
    ``version`` is the SciPy that runs. For the original full factor, ``call``
    holds the keywords ``splu`` received, or ``None`` before a call. Analytical
    block factors record their full/reduced dimensions, implicit full-factor
    nonzeros and the actual reduced ``splu`` keywords under ``call["splu"]``.
    Their nonzero counts describe the represented full factors, not materialized
    full L/U arrays; the recorded reduced counts identify actual factor storage.
    """

    kind: FactorKind
    threshold: float
    options: Mapping[str, object]
    status: FactorStatus
    reason: str
    perm_r: IndexArray | None
    perm_c: IndexArray | None
    l_nnz: int | None
    u_nnz: int | None
    diagonal: FloatArray | None
    version: str
    call: Mapping[str, object] | None


def factor_step(record: FactorRecord) -> WorkStep:
    """The work step of a factor attempt: its nonzeros, unavailable without a factor."""
    status: StepStatus = "completed" if record.status == "factored" else "failed"
    if record.status == "refused":
        status = "refused"
    counters = None
    if record.l_nnz is not None and record.u_nnz is not None:
        counters = (("L nonzeros", record.l_nnz), ("U nonzeros", record.u_nnz))
    return WorkStep(f"{record.kind} factor", status, record.reason, counters)


class Calls:
    """The solves of one certificate, counted with their right-hand sides."""

    def __init__(self) -> None:
        self.counts = dict.fromkeys(
            ("n", "n_columns", "t", "t_columns", "triangular", "triangular_columns"), 0
        )

    def solve(self, factor: SuperLU, rhs: object, trans: str) -> FloatArray:
        """``factor.solve`` of a float64 copy, counted before the call."""
        values = np.array(rhs, dtype=np.float64)
        key = "n" if trans == "N" else "t"
        self.counts[key] += 1
        self.counts[f"{key}_columns"] += 1 if values.ndim == 1 else values.shape[1]
        with np.errstate(all="ignore"):
            found = factor.solve(values, trans=trans)
        return np.asarray(found, dtype=np.float64)

    def triangular(self, lower: CscView, rhs: FloatArray) -> FloatArray:
        """The unit lower triangular solve of a private ``L``, counted."""
        self.counts["triangular"] += 1
        self.counts["triangular_columns"] += 1
        with np.errstate(all="ignore"):
            found = linalg.spsolve_triangular(lower, rhs, **TRIANGULAR)
        return np.asarray(found, dtype=np.float64)

    def whitening(self, factor: Whitening, rhs: FloatArray) -> FloatArray:
        """One dense lower triangular solve of one column, counted before call."""
        self.counts["triangular"] += 1
        self.counts["triangular_columns"] += 1
        with np.errstate(all="ignore"):
            found = factor.whiten(rhs)
        return np.asarray(found, dtype=np.float64)

    def actions(self) -> Mapping[str, int]:
        """The counts so far, read-only."""
        return MappingProxyType(dict(self.counts))


# -------------------------------------------------------------------- factors


def index_width(size: int, nnz: int) -> str:
    """The refusal of an index width beyond ``intc``, else empty."""
    if size > INDEX_LIMIT or nnz > INDEX_LIMIT:
        return f"the sparse index width exceeds {INDEX_LIMIT}"
    return ""


def factor_record(
    kind: FactorKind,
    status: FactorStatus,
    reason: str,
    call: Mapping[str, object] | None = None,
) -> FactorRecord:
    """A factor record without a returned factor: its diagnostics unavailable."""
    return FactorRecord(
        kind=kind,
        threshold=THRESHOLDS[kind],
        options=OPTIONS[kind],
        status=status,
        reason=reason,
        perm_r=None,
        perm_c=None,
        l_nnz=None,
        u_nnz=None,
        diagonal=None,
        version=VERSION,
        call=call,
    )


def failed_factor(record: FactorRecord, reason: str) -> FactorRecord:
    """A returned factor that is no success, its diagnostics kept."""
    return replace(record, status="failed", reason=reason)


def factor_side(factor: SuperLU, name: str) -> CscView:
    """The factor ``L`` or ``U`` of ``factor``, a fresh CSC array."""
    found: CscView = getattr(factor, name)
    return found


def stored_lower(
    indptr: IndexArray, indices: IndexArray, data: FloatArray, size: int
) -> CscView:
    """A private CSC array of a stored unit lower factor, for its triangular solve.

    The stored arrays of a factor's ``L`` as it was returned, copied: the solve reads
    the same structure and values, never the caller's buffers.
    """
    found: CscView = sparse.csc_array(
        (data.copy(), indices.copy(), indptr.copy()), shape=(size, size)
    )
    return found


def _call(kind: FactorKind) -> tuple[dict[str, object], Mapping[str, object]]:
    """The keywords of the ``splu`` call of ``kind`` and their read-only record."""
    options = dict(OPTIONS[kind])
    keywords: dict[str, object] = {
        "diag_pivot_thresh": THRESHOLDS[kind],
        "options": options,
        **LIBRARY,
    }
    record = {**keywords, "options": MappingProxyType(dict(options))}
    return keywords, MappingProxyType(record)


def factorized(
    matrix: SparseMatrix, kind: FactorKind
) -> tuple[FactorRecord, SuperLU | None]:
    """The factor of a private CSC copy with the options of ``kind``, or its refusal."""
    refusal = index_width(matrix.shape[0], int(matrix.data.size))
    if refusal:
        return factor_record(kind, "refused", refusal), None
    copy = sparse.csc_array(
        (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy()),
        shape=matrix.shape,
    )
    keywords, call = _call(kind)
    try:
        factor = linalg.splu(copy, **keywords)
    except RuntimeError as error:
        return factor_record(kind, "failed", str(error), call), None
    lower, upper = factor_side(factor, "L"), factor_side(factor, "U")
    symmetric = kind in ("curvature", "merit")
    record = replace(
        factor_record(kind, "factored", "", call),
        perm_r=frozen_index(np.asarray(factor.perm_r, dtype=np.intp)),
        perm_c=frozen_index(np.asarray(factor.perm_c, dtype=np.intp)),
        l_nnz=int(lower.nnz),
        u_nnz=int(upper.nnz),
        diagonal=immutable_array(upper.diagonal()) if symmetric else None,
    )
    return record, factor


# -------------------------------------------------------------------- actions


def inverse_operator(factor: SuperLU, calls: Calls, size: int) -> object:
    """``F = A^-1`` with its adjoint, as a float64 ``LinearOperator``."""

    def forward(values: FloatArray) -> FloatArray:
        return calls.solve(factor, values, "N")

    def adjoint(values: FloatArray) -> FloatArray:
        return calls.solve(factor, values, "T")

    return linalg.LinearOperator(
        (size, size),
        matvec=forward,
        rmatvec=adjoint,
        matmat=forward,
        rmatmat=adjoint,
        dtype=np.float64,
    )


def inverse_norm(factor: SuperLU, calls: Calls, size: int) -> float:
    """``k = onenormest(F, t=1, itmax=5)``: its actions counted by ``calls``."""
    operator = inverse_operator(factor, calls, size)
    return float(linalg.onenormest(operator, t=1, itmax=5))
