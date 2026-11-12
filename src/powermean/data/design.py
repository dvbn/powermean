"""The design of the estimation sample and its internal column scaling.

Internal column scaling, reversed in all reported quantities and not reported: the
design matrix is scaled and handed over as it is.  The solving stages receive the scaled
design ``X D^-1`` with ``D = diag(d)``, ``d > 0`` finite, and work in the coordinates
``theta = D gamma``; :class:`powermean.model.order_scale.OrderScale` maps their results
back with the same ``d``, once.  Nothing else changes: no column is centered, so an
intercept keeps its meaning and no other column shifts it; no column is dropped,
combined or reordered, so an exactly dependent, constant or all-zero column reaches the
solver as it is and a later certificate, not the preparation, refuses a model that it
cannot solve (the collinearity resolution is not part of this step).

The convention (a routine numerical choice).  ``d_j`` is the power of two with ``d_j <=
max_i |x_ij| < 2 d_j``, that is ``2^floor(log2 max_i |x_ij|)``, so that the largest
entry of a scaled column lies in ``[1, 2)``; an all-zero column has ``d_j = 1``.  This
is a column-only scaling by powers of the radix, a choice of this package inspired by
the general principle that LAPACK's equilibration routine ``DGEEQUB`` states for its
factors, that scaling by powers of the radix introduces no rounding error.  It is not
that routine and claims no parity with it: ``DGEEQUB`` computes row factors first, takes
the column maxima on the row-scaled entries, forms its factors as
``RADIX**INT(LOG(...)/LOGRDX)`` and clips them to safe bounds, and refuses an all-zero
column, which the preparation keeps with a harmless unit scale.

What the choice guarantees, and no more.  Dividing a double by a power of two is exact
wherever the quotient is a normal double, since only the exponent changes; it rounds
once, correctly, where the quotient is subnormal; it overflows nowhere here, since every
quotient is below two in magnitude.  So each scaled entry is the exact quotient on
normal doubles, ``(X D^-1) D`` restores ``X`` bit for bit wherever the quotients are
normal (a rounded subnormal quotient times ``d_j`` is that rounded value in the exponent
of ``x_ij``, not ``x_ij``), each product ``(x_ij / d_j) theta_j`` is the double ``x_ij
(theta_j / d_j)`` wherever the two quotients ``x_ij / d_j`` and ``theta_j / d_j`` are
exact, as each is where it is a normal double or an exact zero, the quotient of a zero
numerator (a quotient that rounds to zero is zero and not exact: with ``x_ij = d_j =
2^1023`` and ``theta_j = 2^-1074``, ``theta_j / d_j`` rounds to zero, ``x_ij (theta_j /
d_j)`` is zero and ``(x_ij / d_j) theta_j`` is ``theta_j``), since the two products then
round the one real value ``x_ij theta_j / d_j`` once; and the maps of ``OrderScale``,
which divide by ``d`` with exact exponents, are exact in ``d``.  That the two products
are normal is not enough for their identity: with ``x_ij = d_j = 2^1023`` and
``theta_j`` the double after one, ``theta_j / d_j`` rounds in the subnormal range,
``x_ij (theta_j / d_j)`` is one and ``(x_ij / d_j) theta_j`` is ``theta_j``, two normal
products that differ in their last bit.  Nothing is claimed of a sum of such products:
two matrix-vector products that accumulate the same terms in another order, as another
BLAS kernel may, agree to the rounding of the summation, not bit for bit.  Nor is
anything claimed of the conditioning of the scaled problem or of the running time of a
solve: this step changes no option, tolerance or default of the solving stages, and what
it does to the conditioning of a given design is the certificate's to report.  A scale
by the root mean square or the standard deviation, as a standardizing regression would
use, needs a centered or a rounded divisor and would change the bits of every entry;
this one changes no bit of a normal entry.

Range.  ``d_j`` lies between ``2^-1074`` and ``2^1023`` for any finite column (the
column whose largest entry is the smallest subnormal, and the one whose largest entry is
the largest double), every scaled entry lies below two in magnitude, and a scaled entry
in the subnormal range is rounded once by IEEE arithmetic and kept.  A nonzero entry
whose quotient rounds to zero, which happens where the entry is at most half the
smallest subnormal times ``d_j``, is refused (``FloatingPointError``), as the order
layer refuses a power that changes the zero mask: no zero is made of a nonzero value.  A
column with a missing or infinite value, which the sample accounting excludes before
this step, is refused here, never deleted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.inputs import _column_names
from powermean.model.order_scale import OrderScale
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array


def _matrix(design: ArrayLike) -> FloatArray:
    """A nonempty finite real matrix: the design of the retained rows."""
    matrix = _finite_array(design, "design")
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("design must be a nonempty matrix")
    return matrix


def column_scale(design: ArrayLike) -> FloatArray:
    """The diagonal ``d``: the power of two with ``d_j <= max_i |x_ij| < 2 d_j``.

    Parameters
    ----------
    design : array_like
        The design of the retained rows, finite, one row per observation and at least
        one column.

    Returns
    -------
    array
        One positive finite power of two per column, read-only; one for an all-zero
        column.

    Raises
    ------
    ValueError
        A design that is not a nonempty finite real matrix.
    """
    matrix = _matrix(design)
    largest = np.max(np.abs(matrix), axis=0)
    # largest = m 2^e with m in [0.5, 1) (frexp, exact), so 2^(e - 1) <= largest < 2^e.
    _, exponent = np.frexp(largest)
    scale = np.asarray(np.ldexp(1.0, exponent - 1), dtype=np.float64)
    scale[largest == 0] = 1.0
    return _immutable(scale)


@dataclass(frozen=True, eq=False)
class ScaledDesign:
    """The design of the retained rows, scaled column by column, with its scale.

    Attributes
    ----------
    matrix : array
        ``X D^-1``, one row per retained observation, read-only.
    scale : array
        The diagonal ``d`` of ``D``, one positive finite power of two per column.
    column_names : tuple of str
        The names of the columns, in the order of the design, unchanged.

    Notes
    -----
    Every column of the design is present, in its order: nothing is dropped for its
    rank, and no column is centered.
    """

    matrix: FloatArray
    scale: FloatArray
    column_names: tuple[str, ...]

    @property
    def nobs(self) -> int:
        """The number of retained rows."""
        return int(self.matrix.shape[0])

    @property
    def nparams(self) -> int:
        """The number of columns, every column of the design."""
        return int(self.matrix.shape[1])

    def unscaled(self) -> FloatArray:
        """``(X D^-1) D``: the design, bit for bit where the quotients are normal."""
        return _immutable(self.matrix * self.scale)

    def order_scale(self, omega: float) -> OrderScale:
        """The accepted :class:`OrderScale` of this scale at the mean order ``omega``.

        Parameters
        ----------
        omega : float
            The mean order, finite and real; zero for log OLS.

        Returns
        -------
        OrderScale
            The map from the coordinates of the solver on this scaled design to the
            reported scale, applied once by its methods.
        """
        return OrderScale(omega, self.scale)


def scale_columns(
    design: ArrayLike, column_names: Sequence[str] | None = None
) -> ScaledDesign:
    """Scale the columns of the design of the retained rows by :func:`column_scale`.

    Parameters
    ----------
    design : array_like
        The design of the retained rows, finite, one row per observation and at least
        one column.
    column_names : sequence of str, optional
        The names of the columns, unique and nonempty; ``x0``, ``x1``, ... if omitted.

    Returns
    -------
    ScaledDesign
        ``X D^-1`` with ``d``, every column kept in its order.

    Raises
    ------
    ValueError
        A design that is not a nonempty finite real matrix, or malformed names.
    FloatingPointError
        A nonzero entry whose scaled value rounds to zero.
    """
    matrix = _matrix(design)
    scale = column_scale(matrix)
    # A quotient by a power of two is exact where it is normal; in the subnormal range
    # it rounds once, and the refusal below keeps every nonzero entry nonzero.
    with np.errstate(under="ignore"):
        scaled: FloatArray = matrix / scale
    if not np.array_equal(scaled == 0, matrix == 0):
        raise FloatingPointError(
            "column scaling changed the zero pattern of the design"
        )
    names = _column_names(column_names, int(matrix.shape[1]))
    return ScaledDesign(matrix=_immutable(scaled), scale=scale, column_names=names)
