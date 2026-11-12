"""Current sample of the internal joint transformed ``tau = 2`` recession diagnostic.

Internal module: ``powermean`` exports nothing of it.
:mod:`powermean.solve.support_gamma` also provides this record under the name
``GammaProblem``.

:class:`GammaProblem` keeps ``Z``, ``X`` and its column order, the grouping, the exact
original positions (checked as exact nonnegative ``np.intp`` values before they are
narrowed), the raw positive weights with their declared kind (frequency counts
validated as exact counts by :class:`~powermean.data.weights.ObservationWeights`), the
finite offset, ``PowerVariance(2)`` and the order ``omega``, each as an owned
read-only copy.  It is not the separation record
:class:`~powermean.solve.support_problem.SupportProblem`, which refuses ``tau = 2``.
:meth:`GammaProblem.same` compares two samples bit for bit, weight kind and grouping
included, and :meth:`GammaProblem.require_same` refuses another one: no stale reuse.
Nothing is deleted: the all-zero level rule belongs to the sample construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.weights import ObservationWeights
from powermean.fe.groups import Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.support_problem import LARGEST_POSITION, indices

VARIANCE = "the recession diagnostic is characterized for PowerVariance(2) only"
STALE = "the record is bound to another sample than the current one"


def _real(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    """An owned read-only finite real array of ``ndim`` dimensions; ``ValueError``."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    found = immutable_array(value)
    if found.ndim != ndim or not np.isfinite(found).all():
        raise ValueError(f"{name} must be a finite array of {ndim} dimensions")
    return found


def _positions(value: ArrayLike, n: int, grouping: Grouping | None) -> IndexArray:
    """Distinct nonnegative integer positions, checked before they are narrowed."""
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


def _weights(weights: object, n: int) -> ObservationWeights:
    """An owned copy of positive weights of their declared kind, one per row."""
    if not isinstance(weights, ObservationWeights):
        raise ValueError("weights must be an ObservationWeights record")
    if weights.values.shape != (n,) or not (weights.values > 0).all():
        raise ValueError("weights must be positive, one per current row")
    return ObservationWeights(weights.values, weights.kind)


def _names(columns: tuple[str, ...], p: int) -> tuple[str, ...]:
    """``p`` distinct string names of the slopes, in order."""
    found = tuple(columns)
    if len(found) != p or any(not isinstance(c, str) for c in found):
        raise ValueError("provide one string name per slope")
    if len(set(found)) != p:
        raise ValueError("slope names must be distinct")
    return found


def _order(omega: object) -> float:
    """A finite nonzero real order as a float."""
    if isinstance(omega, bool | np.bool_) or not isinstance(omega, int | float):
        raise ValueError("omega must be a finite nonzero real number")
    if not np.isfinite(omega) or omega == 0:
        raise ValueError("omega must be a finite nonzero real number")
    return float(omega)


@dataclass(frozen=True, eq=False)
class GammaProblem:
    """The joint transformed ``tau = 2`` problem on the current positive-weight rows.

    Attributes
    ----------
    z, design : array of float
        ``Z`` (nonnegative, finite, nonempty) and ``X`` (``n`` by ``p``, ``p >= 0``).
    columns : tuple of str
        The distinct names of the slopes, in order.
    grouping : Grouping or None
        The accepted grouping of the estimated blocks; ``None`` without effects.
    positions : array of int
        The original position of each row: distinct, nonnegative, the grouping's.
    weights : ObservationWeights
        Positive weights and their declared kind (exact counts for ``frequency``).
    offset : array of float
        A finite offset, bound to the sample, never in the cone.
    variance, omega : PowerVariance, float
        ``PowerVariance(2)`` and the nonzero order that formed ``Z``.
    """

    z: FloatArray
    design: FloatArray
    columns: tuple[str, ...]
    grouping: Grouping | None
    positions: IndexArray
    weights: ObservationWeights
    offset: FloatArray
    variance: PowerVariance
    omega: float

    def __post_init__(self) -> None:
        """Validate every field and store owned read-only copies."""
        z = _real(self.z, "z", 1)
        if not z.size or (z < 0).any():
            raise ValueError("z must be a nonempty vector of nonnegative values")
        n, design = z.size, _real(self.design, "design", 2)
        if design.shape[0] != n:
            raise ValueError("design must have one row per value of z")
        columns = _names(self.columns, design.shape[1])
        grouping = self.grouping
        if grouping is not None and (
            not isinstance(grouping, Grouping) or grouping.n_rows != n
        ):
            raise ValueError("grouping must be a Grouping on the same rows")
        offset = _real(self.offset, "offset", 1)
        if offset.shape != (n,):
            raise ValueError("offset must have one value per current row")
        if type(self.variance) is not PowerVariance or self.variance.tau != 2.0:
            raise ValueError(VARIANCE)
        for name, value in (
            ("z", z),
            ("design", design),
            ("columns", columns),
            ("positions", _positions(self.positions, n, grouping)),
            ("weights", _weights(self.weights, n)),
            ("offset", offset),
            ("omega", _order(self.omega)),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_rows(self) -> int:
        """The number of current rows."""
        return int(self.z.size)

    @property
    def n_slopes(self) -> int:
        """The number of slopes ``p``."""
        return int(self.design.shape[1])

    @property
    def n_blocks(self) -> int:
        """The number of estimated blocks."""
        return 0 if self.grouping is None else len(self.grouping.blocks)

    @property
    def n_parameters(self) -> int:
        """``k``: the slopes and every level of every block."""
        blocks = () if self.grouping is None else self.grouping.blocks
        return self.n_slopes + sum(b.n_levels for b in blocks)

    @property
    def positive(self) -> IndexArray:
        """``P``: the current rows with ``Z > 0``, increasing."""
        return indices(np.flatnonzero(self.z > 0))

    def same(self, other: GammaProblem) -> bool:
        """Whether ``other`` holds this sample, bit for bit, kind and grouping too."""
        if self is other:
            return True
        names = ("z", "design", "positions", "offset")
        pairs = [(getattr(self, n), getattr(other, n)) for n in names]
        pairs.append((self.weights.values, other.weights.values))
        same = all(_same_bits(a, b) for a, b in pairs)
        same = same and self.weights.kind == other.weights.kind
        same = same and (self.columns, self.omega) == (other.columns, other.omega)
        return same and _same_grouping(self.grouping, other.grouping)

    def require_same(self, other: object) -> None:
        """``ValueError`` unless ``other`` holds this sample: no stale reuse."""
        if not isinstance(other, GammaProblem) or not self.same(other):
            raise ValueError(STALE)


def _same_bits(a: FloatArray | IndexArray, b: FloatArray | IndexArray) -> bool:
    """Whether two arrays have one shape and the same bytes."""
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _same_grouping(a: Grouping | None, b: Grouping | None) -> bool:
    """Whether two groupings have the same names, codes and counts."""
    if a is None or b is None:
        return a is None and b is None
    return a.names == b.names and all(
        _same_bits(x.codes, y.codes) and _same_bits(x.counts, y.counts)
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )
