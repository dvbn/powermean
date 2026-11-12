"""Exact integral coding at the interface-owned sample boundary.

The delivered A4 hash encoder belongs here, not in the NumPy-only FE layer.
Only admitted integer vectors use it; first-occurrence codes and representatives
are the existing ones. Wide products keep the original exact lexicographic path.
No state or result is retained between requests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from powermean.fe.primitive_codes import first_rows, key_codes
from powermean.model.order import IndexArray


def _integer_key_codes(
    keys: Sequence[NDArray[np.integer[Any]]],
) -> tuple[IndexArray, IndexArray]:
    """Number one or two integral keys in input order without a row sort.

    For two keys, the exact Python-integer span product must fit int64 before
    any shift/product is allocated. That bijection preserves tuple equality,
    including signed/unsigned extremes; no float or string conversion occurs.
    Wider or other inputs retain the original encoder, not an approximate key.
    """
    found = tuple(keys)
    if len(found) not in (1, 2) or any(
        type(key) is not np.ndarray or key.ndim != 1 or not key.size
        or key.dtype.kind not in "iu" for key in found
    ):
        return key_codes(found)
    if len(found) == 1:
        key = found[0]
    else:
        a, b = found
        if a.size != b.size:
            return key_codes(found)
        low_a, low_b = a.min(), b.min()
        span_a = int(a.max()) - int(low_a) + 1
        span_b = int(b.max()) - int(low_b) + 1
        if span_a * span_b > np.iinfo(np.int64).max:
            return key_codes(found)
        key = (np.subtract(
            a, low_a, dtype=np.int64 if a.dtype.kind == "i" else np.uint64,
        ).astype(np.int64) * span_b + np.subtract(
            b, low_b, dtype=np.int64 if b.dtype.kind == "i" else np.uint64,
        ).astype(np.int64))
    codes = pd.factorize(key, sort=False, use_na_sentinel=False)[0]
    codes = codes.astype(np.intp, copy=False)
    first = first_rows(codes)
    return key_codes(found) if first is None else (codes, first)
