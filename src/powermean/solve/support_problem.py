"""The current sample of a support diagnostic: validation, binding and index storage.

Internal module: ``powermean`` exports nothing of it.

The sample.  :class:`SupportProblem` is the joint transformed problem on the accepted
current rows: ``Z = Y^omega``, the slopes ``X``, the accepted grouping of the estimated
blocks (``F``, never formed), the original positions, the positive statistical weights
and the finite offset.  Weights and offset never enter the cone of ``B = [X, F]``; they
are validated and kept so that every witness and result is bound to its exact sample
(:func:`same_sample`).  Only ``PowerVariance`` with ``1 <= tau < 2`` and
``NB2Variance`` have this characterized cone: another family, ``tau = 2``
or ``omega = 0`` is refused, so nothing dispatches here silently.  Nothing is deleted.

Positions.  The original positions are distinct nonnegative integers, the grouping's
when there is one.  Each is checked to be an exact nonnegative ``np.intp`` before the
array is narrowed to that type, so that no position wraps (an unsigned ``2^63`` would
otherwise be stored as ``-2^63``); a refused vector stores nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.accounting import frozen_index
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance

Family = PowerVariance | NB2Variance

VARIANCE = (
    "the joint separation cone is characterized for PowerVariance with 1 <= tau < 2 "
    "and NB2Variance only"
)
STALE = "the record is bound to another sample than the current one"
# The largest original position that np.intp holds exactly.
LARGEST_POSITION = int(np.iinfo(np.intp).max)


def _real(value: ArrayLike, name: str) -> FloatArray:
    """An owned finite real array of doubles, or ``ValueError``."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    found: FloatArray = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(found).all():
        raise ValueError(f"{name} must be finite")
    return found


def _family(variance: object) -> Family:
    """The variance if its joint cone is characterized; ``ValueError`` otherwise."""
    if type(variance) is PowerVariance and 1.0 <= variance.tau < 2.0:
        return variance
    if type(variance) is NB2Variance:
        return variance
    raise ValueError(VARIANCE)


def _order(omega: object) -> float:
    """A finite nonzero real order as a float; ``omega = 0`` is log OLS, refused."""
    if isinstance(omega, bool | np.bool_) or not isinstance(omega, int | float):
        raise ValueError("omega must be a finite nonzero real number")
    if not np.isfinite(omega) or omega == 0:
        raise ValueError("omega must be a finite nonzero real number")
    return float(omega)


def indices(rows: ArrayLike) -> IndexArray:
    """An owned read-only index array of ``rows``."""
    return frozen_index(np.asarray(rows, dtype=np.intp))


NONE = indices(np.zeros(0))


def _positions(value: ArrayLike, n: int, grouping: Grouping | None) -> IndexArray:
    """Distinct nonnegative integer positions, the grouping's when there is one.

    The largest value is compared, as an exact integer, with the largest ``np.intp``
    before the vector is narrowed to that type: nothing wraps.
    """
    found = np.asarray(value)
    if found.dtype.kind not in "iu" or found.shape != (n,):
        raise ValueError("positions must be an integer vector, one per row")
    if (found < 0).any() or np.unique(found).size != n:
        raise ValueError("positions must be distinct nonnegative integers")
    if int(found.max()) > LARGEST_POSITION:
        raise ValueError("positions must be representable as np.intp")
    positions = indices(found)
    if grouping is not None and not np.array_equal(positions, grouping.positions):
        raise ValueError("positions must be the grouping's original positions")
    return positions


@dataclass(frozen=True, eq=False)
class SupportProblem:
    """The joint transformed problem on the accepted current positive-weight rows.

    Attributes
    ----------
    z, design : array of float
        ``Z`` (nonnegative, finite, nonempty) and ``X`` (``n``-by-``p``, finite; ``p``
        may be zero).
    columns : tuple of str
        The distinct names of the slope columns, in order.
    grouping : Grouping or None
        The accepted grouping of the estimated blocks; ``None`` without effects.
    positions : array of int
        The original position of each row: distinct, nonnegative, the grouping's.
    weights, offset : array of float
        Positive statistical weights and a finite offset, bound, never in the cone.
    variance : PowerVariance or NB2Variance
        ``1 <= tau < 2``, or NB2 with a finite ``kappa >= 0``.
    omega : float
        The nonzero order that formed ``Z``.
    """

    z: FloatArray
    design: FloatArray
    columns: tuple[str, ...]
    grouping: Grouping | None
    positions: IndexArray
    weights: FloatArray
    offset: FloatArray
    variance: Family
    omega: float

    def __post_init__(self) -> None:
        """Validate every field and store owned read-only copies."""
        z, design = _real(self.z, "z"), _real(self.design, "design")
        if z.ndim != 1 or not z.size or (z < 0).any():
            raise ValueError("z must be a nonempty vector of nonnegative values")
        n = int(z.size)
        if design.ndim != 2 or design.shape[0] != n:
            raise ValueError("design must be a matrix with one row per value of z")
        columns = tuple(self.columns)
        if len(columns) != design.shape[1] or len(set(columns)) != len(columns):
            raise ValueError("provide one distinct column name per slope")
        if any(not isinstance(name, str) for name in columns):
            raise ValueError("column names must be strings")
        grouping = self.grouping
        if grouping is not None and (
            not isinstance(grouping, Grouping) or grouping.n_rows != n
        ):
            raise ValueError("grouping must be a Grouping on the same rows")
        weights, offset = _real(self.weights, "weights"), _real(self.offset, "offset")
        if weights.shape != (n,) or (weights <= 0).any():
            raise ValueError("weights must be positive, one per current row")
        if offset.shape != (n,):
            raise ValueError("offset must have one value per current row")
        for name, value in (
            ("z", immutable_array(z)),
            ("design", immutable_array(design)),
            ("columns", columns),
            ("positions", _positions(self.positions, n, grouping)),
            ("weights", immutable_array(weights)),
            ("offset", immutable_array(offset)),
            ("variance", _family(self.variance)),
            ("omega", _order(self.omega)),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_rows(self) -> int:
        """The number of current rows."""
        return int(self.z.size)

    @property
    def n_slopes(self) -> int:
        """The number of slope columns ``p``."""
        return int(self.design.shape[1])

    @property
    def n_blocks(self) -> int:
        """The number of estimated blocks ``G``."""
        return 0 if self.grouping is None else len(self.grouping.blocks)

    @property
    def zero(self) -> IndexArray:
        """``O``: the current rows with ``Z = 0``, increasing."""
        return indices(np.flatnonzero(self.z == 0))

    @property
    def positive(self) -> IndexArray:
        """``P``: the current rows with ``Z > 0``, increasing."""
        return indices(np.flatnonzero(self.z > 0))


def same_sample(first: SupportProblem, second: SupportProblem) -> bool:
    """Whether two records hold the same sample, bit for bit, grouping included."""
    if first is second:
        return True
    same = all(
        getattr(first, name).shape == getattr(second, name).shape
        and getattr(first, name).tobytes() == getattr(second, name).tobytes()
        for name in ("z", "design", "positions", "weights", "offset")
    )
    same = same and first.columns == second.columns and first.omega == second.omega
    same = same and first.variance.to_dict() == second.variance.to_dict()
    a, b = first.grouping, second.grouping
    if a is None or b is None:
        return same and a is None and b is None
    return (
        same
        and a.names == b.names
        and all(
            np.array_equal(x.codes, y.codes) and np.array_equal(x.counts, y.counts)
            for x, y in zip(a.blocks, b.blocks, strict=True)
        )
    )


def require_same(first: SupportProblem, second: SupportProblem) -> None:
    """``ValueError`` unless the two records hold the same sample: no stale reuse."""
    if not same_sample(first, second):
        raise ValueError(STALE)
