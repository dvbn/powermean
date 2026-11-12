"""Owned, immutable arrays of the data records.

A real array-like becomes an owned array of finite doubles in C order, in bytes of its
own, so that no caller can set its write flag again (``ndarray.setflags(write=True)``
raises ``ValueError``), and a later change of the input does not reach the record.  The
checks are made in this order: real-valued, the number of dimensions, finite.  The
storage is :func:`powermean.model.rows._immutable`, which forms the same bytes as
``np.frombuffer(array.tobytes(order="C"), ...)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray


def frozen_array(
    value: ArrayLike, *, ndim: int | None = None, name: str = "array"
) -> FloatArray:
    """An owned read-only copy of a real array of finite doubles.

    Parameters
    ----------
    value : array_like
        The values, real.
    ndim : int, optional
        The number of dimensions that the values must have; any number if omitted.
    name : str, default "array"
        The name of the values in the messages.

    Returns
    -------
    array
        The values as doubles, in C order, in bytes of their own, read-only.

    Raises
    ------
    ValueError
        Complex values, another number of dimensions, or a value that is not finite.
    """
    if np.iscomplexobj(np.asarray(value)):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    # Immutable bytes prevent callers from re-enabling write access on the array.
    return _immutable(array)
