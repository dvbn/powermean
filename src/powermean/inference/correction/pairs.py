"""Explicit ordered cell-pair data with prospective construction/validation bounds.

No sampling scheme is inferred. A CellPairs instance is data, never a trust
cache: every correction revalidates its current array, diagonal and symmetry.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import IndexStorage
from powermean.inference.correction.budget import (
    BiasBounds,
    checked_count,
    enforce,
    reserve_pairs,
)


@dataclass(frozen=True, slots=True)
class CellPairs:
    """Owned ordered positions; multiplicity is one per distinct ordered pair."""

    _pairs: IndexStorage
    n_obs: int
    n_pairs: int
    is_symmetric: bool

    @property
    def pairs(self) -> NDArray[np.intp]:
        """Fresh shape metadata over immutable one-dimensional index storage."""
        return self._pairs.view().reshape((self.n_pairs, 2))

    @property
    def left_indices(self) -> NDArray[np.intp]:
        """Left position i in each ordered (i,j) pair."""
        return self.pairs[:, 0]

    @property
    def right_indices(self) -> NDArray[np.intp]:
        """Right position j in each ordered (i,j) pair."""
        return self.pairs[:, 1]


def pair_count(pairs: ArrayLike | CellPairs | None, n_obs: int) -> int:
    """Check declared dimensions before any pair copy/conversion."""
    checked_count(n_obs, "n_obs", 1)
    if pairs is None:
        raise ValueError(
            "explicit ordered cell pairs required; sampling policy cannot be inferred"
        )
    raw = pairs.pairs if type(pairs) is CellPairs else pairs
    if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape[1] != 2:
        raise TypeError("pairs must be an ndarray of shape (n_pairs, 2)")
    if raw.dtype.kind not in "iu":
        raise TypeError("pairs must contain integers")
    return int(raw.shape[0])


def validate_cell_pairs(
    pairs: ArrayLike | CellPairs | None,
    n_obs: int,
    *,
    bounds: BiasBounds,
    require_symmetric: bool = True,
) -> CellPairs:
    """Revalidate all ordered entries, including data in a previously built record.

    Lexicographic sorting avoids overflow in i*n+j keys. Missing diagonals are
    checked by their count after uniqueness, without out-of-bounds searches.
    require_symmetric=False is an algebra-only seam for directed oracle controls;
    it supplies no dependence interpretation or covariance validity assertion.
    """
    count = pair_count(pairs, n_obs)
    reserve_pairs(n_obs, count, bounds=bounds)
    if type(require_symmetric) is not bool:
        raise TypeError("require_symmetric must be bool")
    raw = pairs.pairs if type(pairs) is CellPairs else pairs
    if not isinstance(raw, np.ndarray):
        raise TypeError("pairs must be an ndarray")
    if np.any(raw < 0) or np.any(raw >= n_obs):
        raise ValueError("pair positions must lie in [0,n_obs)")
    arr = np.array(raw, dtype=np.intp, copy=True)
    ordered = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
    if np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
        raise ValueError("duplicate ordered cell pairs")
    if np.count_nonzero(ordered[:, 0] == ordered[:, 1]) != n_obs:
        raise ValueError("all diagonal pairs (i,i) must be included")
    reverse = arr[:, ::-1]
    rev_sorted = reverse[np.lexsort((reverse[:, 1], reverse[:, 0]))]
    symmetric = bool(np.array_equal(ordered, rev_sorted))
    if require_symmetric and not symmetric:
        raise ValueError("pairs must include both (i,j) and (j,i)")
    return CellPairs(IndexStorage.store(arr.reshape(-1)), n_obs, count, symmetric)


def diagonal_cell_pairs(n_obs: int, *, bounds: BiasBounds) -> CellPairs:
    """Construct independent-cell diagonal pairs after explicit reservation."""
    checked_count(n_obs, "n_obs", 1)
    reserve_pairs(n_obs, n_obs, bounds=bounds)
    arr: NDArray[np.intp] = np.empty((n_obs, 2), dtype=np.intp)
    arr[:, 0] = np.arange(n_obs)
    arr[:, 1] = arr[:, 0]
    return validate_cell_pairs(arr, n_obs, bounds=bounds)


def cluster_cell_pairs(cluster_codes: ArrayLike, *, bounds: BiasBounds) -> CellPairs:
    """Enumerate ordered within-cluster pairs only after counting in Python ints.

    One final pair array is filled in cluster slices; no list of quadratic arrays
    or Python set is built. Bounds include stable sort, index and validation
    transients. This helper does not choose whether clusters are appropriate.
    """
    if not isinstance(cluster_codes, np.ndarray) or cluster_codes.ndim != 1:
        raise TypeError("cluster_codes must be a 1D integer ndarray")
    if cluster_codes.dtype.kind not in "iu" or not cluster_codes.size:
        raise ValueError("cluster_codes must contain nonempty integer data")
    n = int(cluster_codes.size)
    enforce(256 * n, 128 * n * (n.bit_length() + 1), bounds)
    order = np.argsort(cluster_codes, kind="stable")
    sorted_codes = cluster_codes[order]
    cuts: NDArray[np.intp] = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    pointers = np.concatenate(([0], cuts, [n]))
    count = sum((int(b) - int(a)) ** 2 for a, b in pairwise(pointers))
    reserve_pairs(n, count, bounds=bounds)
    arr: NDArray[np.intp] = np.empty((count, 2), dtype=np.intp)
    pos = 0
    for start, stop in pairwise(pointers):
        members = order[start:stop]
        size = int(members.size)
        for member in members:
            arr[pos : pos + size, 0] = member
            arr[pos : pos + size, 1] = members
            pos += size
    return validate_cell_pairs(arr, n, bounds=bounds)
