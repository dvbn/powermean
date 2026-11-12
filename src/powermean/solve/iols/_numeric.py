"""Float64 overflow-guarded exponentiation for the IOLS kernel."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = (
    np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    | np.ndarray[tuple[int, ...], np.dtype[np.int64]]
)

FLOAT64_MAX: float = float(np.finfo(np.float64).max)
FLOAT64_LOG_MAX: float = float(np.log(FLOAT64_MAX))


def exp_with_float64_overflow_guard(values: npt.ArrayLike) -> FloatArray:
    """Exponentiate in float64, saturating only above the representable boundary."""
    array: FloatArray = np.asarray(values, dtype=np.float64)
    result: FloatArray = np.empty_like(array, dtype=np.float64)
    overflow_mask = np.isposinf(array) | (array > FLOAT64_LOG_MAX)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        np.exp(array, out=result, where=~overflow_mask)

    if np.any(overflow_mask):
        result[overflow_mask] = FLOAT64_MAX
    return result
