"""Small numerical operations for inference: finite arrays and symmetric parts.

:func:`finite_array` and :func:`immutable_array`, which
:mod:`powermean.inference.influence` and :mod:`powermean.inference.vcov` import, and
:func:`symmetric_part`, which the covariance of :mod:`powermean.inference.vcov` uses.

- :func:`immutable_array` stores its copy in bytes of its own, as
  :func:`powermean.model.rows._immutable` does, so that the write flag of the result
  cannot be set again.
- :func:`symmetric_part` names its matrix ``matrix``; the package reserves ``a`` for
  observation weights.

The symmetric part.  Each entry of ``(A + A')/2`` is rounded once to the nearest
double.  Where both entries of a pair are at most half the largest double in absolute
value, their sum is formed first and halved: below ``2^-1021`` the sum is exact and
only the halving rounds, so that the mean of two equal subnormal entries is that entry,
and above it only the sum rounds and the halving is exact.  Elsewhere each entry is
halved first, which is then exact for the large entry, and the halves are added, so
that no sum overflows; the halving of a tiny partner there loses at most ``2^-1075``,
far below the rounding of a result of at least ``2^1021``.  Where halving each entry of
a pair is exact, as it is for every entry of at least ``2^-1021`` in absolute value
(the ordinary range), both forms equal ``A/2 + A'/2`` bit for bit.  Below that bound
halving rounds an odd multiple of ``2^-1074``, a normal entry included:
``x = 0x1.0000000000001p-1022`` is the mean of the pair ``(x, x)``, and ``x/2 + x/2``
is ``2^-1022``.  ``A/2 + A'/2`` likewise rounds a subnormal half before the sum, and
turns a representable covariance of ``2^-1074`` into zero.  A diagonal entry is kept
exactly.  A mean of at most half the smallest subnormal, as that of ``2^-1074`` and
zero, still rounds to zero (to even), which is the genuine final underflow of a result
that is not representable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray


def finite_array(value: ArrayLike, name: str) -> FloatArray:
    """A real array of finite doubles, or ``ValueError``.

    Parameters
    ----------
    value : array_like
        The values.
    name : str
        The name of the values in the messages.

    Returns
    -------
    array
        The values as doubles; not a copy where the input already is one.
    """
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def immutable_array(value: ArrayLike) -> FloatArray:
    """An owned read-only copy of a real array of doubles.

    The copy holds bytes of its own, whose write flag cannot be set again.

    Parameters
    ----------
    value : array_like
        The values, real.

    Returns
    -------
    array
        The copy, in the shape of the values.
    """
    if np.iscomplexobj(value):
        raise ValueError("value must be real-valued")
    return _immutable(np.asarray(value, dtype=np.float64))


def symmetric_part(value: ArrayLike) -> FloatArray:
    """Average a finite square matrix and its transpose without range loss.

    Adding first preserves subnormal entries; halving first avoids overflow for large
    entries.  Neither a positive direction nor the diagonal is discarded merely to
    symmetrize rounding differences.

    Parameters
    ----------
    value : array_like
        The square matrix ``A``, finite and real.

    Returns
    -------
    array
        ``(A + A')/2``, each entry rounded once.
    """
    matrix = finite_array(value, "matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    small = np.abs(matrix) <= np.finfo(float).max / 2
    add_first = small & small.T
    result = np.empty_like(matrix)
    # Underflow of a genuinely unrepresentable average is ordinary rounding,
    # unlike halving two equal subnormals before forming their sum.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result[add_first] = (matrix[add_first] + matrix.T[add_first]) / 2
        result[~add_first] = matrix[~add_first] / 2 + matrix.T[~add_first] / 2
    return result
