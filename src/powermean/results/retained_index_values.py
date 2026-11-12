"""Fresh numeric row-index snapshots for retained graph authentication.

Current inputs are authenticated before results are detached. This replaces object
boxing, not a live validation. Every comparison captures new numeric values before any
name-equality callback. Object, tuple, extension and subclass indexes keep the ordered
generic route.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from powermean.data.raw import AnyArray


def _numeric_dtype(dtype: object) -> bool:
    """Whether boxing this dtype yields exact Boolean, integer or float values."""
    return isinstance(dtype, np.dtype) and (
        dtype.kind in "biu" or (dtype.kind == "f" and dtype.itemsize <= 8)
    )


def _numeric_index(index: pd.Index[Any]) -> bool:
    """Require ordinary storage consistent with the index's advertised dtype."""
    if type(index) is not pd.Index and type(index) is not pd.RangeIndex:
        return False
    stored = getattr(index, "_data", None)
    if type(stored) is not np.ndarray:
        return False
    dtype = index.dtype
    return _numeric_dtype(dtype) and stored.dtype == dtype


def numeric_index_snapshots(
    left: pd.Index[Any], right: pd.Index[Any]
) -> tuple[AnyArray, AnyArray] | None:
    """Capture both numeric indexes, or leave both on the generic object route.

    Snapshots occur left then right, before dtype/name/equality predicates in the
    caller, just like the replaced object conversions. A custom index name may
    mutate either live index afterward; it cannot change these comparison values.
    No RangeIndex bounds, uniqueness or unchanged cache contents are assumed.
    A replaced RangeIndex cache may contradict its fixed advertised int64 dtype;
    incompatible storage or a cached ndarray subclass keeps the object route.
    Native numeric copies preserve exact integers beyond 2**53. Wider-than-double
    floats stay generic because conversion to Python float affects the existing
    missing-value predicate.
    """
    if not (_numeric_index(left) and _numeric_index(right)):
        return None
    l_arr, r_arr = np.asarray(left).copy(), np.asarray(right).copy()
    if any(
        a.ndim != 1 or not _numeric_dtype(a.dtype) or a.dtype != index.dtype
        for a, index in ((l_arr, left), (r_arr, right))
    ):
        return None
    return l_arr, r_arr


def same_numeric_index_values(left: AnyArray, right: AnyArray) -> bool:
    """Compare captured numeric values with the existing object-route semantics.

    Equal dtype and shape remain required. Signed zeros compare equal. For the
    admitted float widths, every nonfinite value is a missing scalar under the
    original predicate, so NaN and either infinity are mutually missing-equivalent.
    Integer and Boolean comparisons never pass through a floating representation.
    """
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind != "f":
        return bool(np.array_equal(left, right))
    with np.errstate(invalid="ignore"):
        equal = left == right
        missing = ~np.isfinite(left) & ~np.isfinite(right)
    return bool(np.all(equal | missing))
