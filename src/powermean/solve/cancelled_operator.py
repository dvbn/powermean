"""Owned supplied coordinates for the internal tau=2 attempt.

This is a bridge to a coordinate map, not a geometry builder. The caller supplies
T and P, with PT=I; no rank, alias, normalization or reference level is selected.
B_id v=B(Tv), B_id' u=T'B'u use the accepted full-column implicit design. P is
the reduction of coefficients, generally different from T'. Neither row-by-level
nor row-by-identified-column storage is constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from powermean.fe._cancel_storage import (
    BuilderRecord,
    FloatStorage,
    finite_product,
    finite_vector,
)
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray, PowerVariance, Variance


@dataclass(frozen=True)
class AttemptBounds:
    """Explicit limits on bridge numeric buffers and array work, with no defaults.

    ``max_bytes`` bounds the conservative live numeric-buffer reservation, not
    process memory. ``max_work`` bounds bridge/core array visits per evaluation
    and coordinate validation. Provider/native factorization workspace, Python
    objects, caller inputs and caller assessments are outside these two limits.
    """

    max_bytes: int
    max_work: int

    def __post_init__(self) -> None:
        """Refuse nonpositive or noninteger resource limits."""
        for value in (self.max_bytes, self.max_work):
            if type(value) is not int or value <= 0:
                raise ValueError("attempt bounds must be positive Python integers")


@dataclass(frozen=True)
class WorkspacePlan:
    """Conservative reservation for every bridge/full/identified numeric buffer.

    Float slots reserve: 4pd for both owned maps and validation/staging; 3d² for
    provider Hessian, solve copy and checking; 48d for coefficients, directions,
    gradients, trials, two attempt snapshots and coordinate checks; 128n for
    owned outcomes/offset, raw/trial rows, accepted row-component temporaries and
    core centering; 16p and 16G for full products, reductions and group scratch.
    Additional bool/index reservations cover validation masks and indexing.
    Shared accepted-design numeric storage is reported separately, not copied.
    The plan includes no dense n-by-d or n-by-retained-level buffer.
    """

    owned_bytes: int
    shared_numeric_bytes: int
    work: int

    def require(self, bounds: AttemptBounds) -> None:
        """Refuse before map copies or provider dense curvature allocation."""
        if self.owned_bytes > bounds.max_bytes or self.work > bounds.max_work:
            raise ValueError("cancelled attempt exceeds supplied byte/work bounds")


def workspace_plan(design: CenteredDesign, dimension: int) -> WorkspacePlan:
    """Count explicit buffer reservations and bridge/core array visits.

    Work counts a multiply/add or elementwise kernel visit as one unit, not time.
    At most 2d+2 B_id/B_id' calls occur in a derivative evaluation, including a
    raw product if needed. The row-kernel reservation is 256n visits. Native
    factorization cost is deliberately not represented as bridge array work.
    """
    n, p, k = design.structure.n_rows, design.n_columns, design.n_slopes
    groups = design.structure.n_groups
    blocks = len(design.structure.block_sizes)
    d = dimension
    floats = 4 * p * d + 3 * d * d + 48 * d + 128 * n + 16 * p + 16 * groups
    masks = 4 * p * d + 8 * n + d * d
    indices = 4 * n + 4 * groups + 2 * p + 2 * d
    index_bytes = np.dtype(np.intp).itemsize
    shared = 8 * n * k + index_bytes * ((5 + blocks) * n + 2 * groups + 1)
    product = 2 * p * d + 8 * n * (k + blocks + 1) + 8 * (p + groups)
    validation = 2 * p * d * d + 16 * p * d
    work = max(validation, (2 * d + 2) * product + 256 * n + d * d)
    return WorkspacePlan(8 * floats + masks + index_bytes * indices, shared, work)


def _matrix(values: FloatArray, shape: tuple[int, int], name: str) -> None:
    """Check supplied float64 matrix metadata before any copying."""
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be a float64 ndarray")
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with shape {shape}")


@dataclass(frozen=True, eq=False, init=False)
class CancelledOperator(BuilderRecord):
    """Owned coordinate adapter bound to one immutable accepted full design.

    Built only by ``coordinate_map``. Supplied map validity does not establish
    that its span exhausts the design, or that B_id has full rank. Those are
    fresh downstream geometry gates, never a trusted-success field here.
    """

    __slots__ = ("_expand", "_reduce", "bounds", "design", "plan")

    _expand: FloatStorage
    _reduce: FloatStorage
    design: CenteredDesign
    bounds: AttemptBounds
    plan: WorkspacePlan

    @property
    def shape(self) -> tuple[int, int]:
        """Rows by supplied identified coordinates; genuine operator metadata."""
        return self.design.structure.n_rows, self._expand.shape[1]

    @property
    def expansion(self) -> FloatArray:
        """Detached immutable view of T, with independent shape metadata."""
        return self._expand.view()

    @property
    def reduction(self) -> FloatArray:
        """Detached immutable view of P; P is not used as a score pullback."""
        return self._reduce.view()

    def expand(self, values: FloatArray) -> FloatArray:
        """Map coefficients by T, with a fresh shape/finite check."""
        eta = finite_vector(values, self.shape[1], "identified coefficients")
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(self.expansion @ eta)

    def reduce(self, values: FloatArray) -> FloatArray:
        """Map full coefficients by P, without asserting index reconstruction."""
        theta = finite_vector(values, self.design.n_columns, "full coefficients")
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(self.reduction @ theta)

    def matvec(self, values: FloatArray) -> FloatArray:
        """B_id v through the accepted implicit full design, in input row order."""
        return self.design.matvec(self.expand(values))

    def rmatvec(self, values: FloatArray) -> FloatArray:
        """B_id' u=T'B'u, distinct from coefficient reduction P."""
        full = self.design.rmatvec(values)
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(self.expansion.T @ full)


def coordinate_map(
    design: CenteredDesign,
    expand: FloatArray,
    reduce: FloatArray,
    *,
    identity_atol: float,
    bounds: AttemptBounds,
) -> CancelledOperator:
    """Bind explicit T and P to a design and detach both maps.

    Parameters
    ----------
    design : CenteredDesign
        Accepted immutable full-column implicit design.
    expand, reduce : ndarray
        Float64 p-by-d T and d-by-p P, supplied by the caller's geometry adapter.
    identity_atol : float
        Explicit absolute tolerance for PT=I, not a rank or omission cutoff.
    bounds : AttemptBounds
        Required numeric workspace and bridge array-work limits.

    Returns
    -------
    CancelledOperator
        Owned maps and shared design. No default geometry is provided.
    """
    if type(design) is not CenteredDesign or type(bounds) is not AttemptBounds:
        raise TypeError("use an accepted CenteredDesign and explicit AttemptBounds")
    if not isinstance(expand, np.ndarray) or expand.ndim != 2:
        raise ValueError("T must be a matrix")
    p, d = expand.shape
    if p != design.n_columns or not 0 < d <= p:
        raise ValueError("T must have full-column rows and 1 <= d <= p")
    if not math.isfinite(identity_atol) or not 0 <= identity_atol < 1:
        raise ValueError("PT identity tolerance must be finite in [0, 1)")
    plan = workspace_plan(design, d)
    plan.require(bounds)
    _matrix(expand, (p, d), "T")
    _matrix(reduce, (d, p), "P")
    expansion, reduction = FloatStorage.store(expand), FloatStorage.store(reduce)
    with np.errstate(over="raise", invalid="raise"):
        for j in range(d):
            error = reduction.view() @ expansion.view()[:, j]
            error[j] -= 1.0
            if not np.isfinite(error).all() or np.any(np.abs(error) > identity_atol):
                raise ValueError("supplied maps must satisfy PT=I")
    record = object.__new__(CancelledOperator)
    for name, value in (
        ("_expand", expansion),
        ("_reduce", reduction),
        ("design", design),
        ("bounds", bounds),
        ("plan", plan),
    ):
        object.__setattr__(record, name, value)
    return record


@dataclass(frozen=True, eq=False, init=False)
class CancelledProblem(BuilderRecord):
    """Owned transformed rows, with no filtering or offset recentering."""

    __slots__ = ("_offset", "_z", "operator")

    operator: CancelledOperator
    _z: FloatStorage
    _offset: FloatStorage

    @property
    def z(self) -> FloatArray:
        """All transformed nonnegative outcomes, in input row order."""
        return self._z.view()

    @property
    def centered_offset(self) -> FloatArray:
        """Already transformed, ordinary-centered offset, added once by the loss."""
        return self._offset.view()


class ImplicitNewtonProblem(NamedTuple):
    """Actual shared-loop carrier; X is an operator, never a dummy ndarray."""

    design: CancelledOperator
    z: FloatArray
    weights: None


def cancelled_problem(
    operator: CancelledOperator,
    z: FloatArray,
    centered_offset: FloatArray,
    *,
    variance: Variance,
    obs_weights: FloatArray | None,
) -> CancelledProblem:
    """Own a unit-multiplier, exact PowerVariance(2) private attempt problem.

    Parameters
    ----------
    operator : CancelledOperator
        Explicit checked coordinate adapter, bound to the immutable design.
    z, centered_offset : ndarray
        Transformed float64 rows. The caller establishes offset centering.
    variance : Variance
        Only exact PowerVariance(2) is supported by this attempt criterion.
    obs_weights : ndarray or None
        Only unit multipliers are admitted; no weighted/frequency policy follows.

    Returns
    -------
    CancelledProblem
        Detached rows. Singletons, zeros and all-zero groups are retained.
    """
    if type(operator) is not CancelledOperator:
        raise TypeError("operator must come from coordinate_map")
    if type(variance) is not PowerVariance or not isinstance(variance, PowerVariance):
        raise ValueError("cancelled convex attempts require exact PowerVariance(2)")
    if variance.tau != 2.0:
        raise ValueError("cancelled convex attempts require exact PowerVariance(2)")
    operator.plan.require(operator.bounds)
    n = operator.shape[0]
    for values in (z, centered_offset):
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64):
            raise TypeError("cancelled rows must be float64 ndarrays")
        finite_vector(values, n, "cancelled rows")
    if np.any(z < 0):
        raise ValueError("transformed z must be nonnegative")
    if obs_weights is not None and np.any(
        finite_vector(obs_weights, n, "weights") != 1
    ):
        raise ValueError("cancelled convex attempts support only unit multipliers")
    record = object.__new__(CancelledProblem)
    object.__setattr__(record, "operator", operator)
    object.__setattr__(record, "_z", FloatStorage.store(z))
    object.__setattr__(record, "_offset", FloatStorage.store(centered_offset))
    return record
