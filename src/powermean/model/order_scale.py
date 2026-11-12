"""The reported scale of a mean order: what the transformed problem maps back to.

The systems, solvers, the certificate and the covariances of the package work in the
coordinates of the solver: the coefficients ``theta`` of the transformed problem of
:mod:`powermean.model.order`, on a design whose columns the data layer scales, ``X
D^-1`` with ``D = diag(d)`` and ``d > 0``.  So ``gamma = D^-1 theta``, and the reported
coefficients are ``beta = gamma / omega`` (``theta / scale / omega``):

    M = D^-1 / omega     at an order omega != 0
    M = D^-1             at omega = 0, log OLS: the identity order scale

Every coefficient of the system is reported: the selector ``L`` is the identity, and a
selection of reported coefficients takes rows of ``M``.  :class:`OrderScale` applies the
column unscaling and the order once, together, to quantities in the coordinates of the
solver:

    in the coordinates of theta              reported                    method
    coefficients, corrections, biases t      t_j / d_j / omega           coefficients
    coefficient influences u_i = A s_i       u_ij / d_j / omega          influences
    covariance V                             V_jk / (d_j d_k) / omega^2  covariance
    standard errors s                        s_j / d_j / |omega|         standard_errors
    the map of the reported coefficients     M = diag(1 / (omega d))     transform

    on the reported scale               in the coordinates of theta  method
    a start beta of the coefficients    d_j omega beta_j             solver_coefficients

At ``omega = 0`` no division by ``omega`` is formed, and the forward map of a start is
``d_j beta_j``.  The forward map is the inverse of ``coefficients``, written with the
same exact exponents: ``omega = d = 2^-600`` and ``beta = 2^300`` give ``theta =
2^-900`` although the product ``omega d`` underflows, and ``omega = d = 2^600`` and
``beta = 2^-900`` give ``theta = 2^300`` although ``omega d`` overflows; the sign of
``beta`` and of ``omega`` is kept, a nonzero result that is not representable raises
``FloatingPointError``, and the start is otherwise the exact value requested where it is
a double.  A negative order reverses the signed maps, not the covariance or the standard
errors: ``se_gamma / omega`` holds for a positive order only and is not used.  The
influences are coefficient influences, the contributions of an influence record mapped
by its influence map (``powermean.inference.influence.influence_values``): neither the
raw scores nor the bread are rescaled, and the order is applied once.  A missing
standard error (NaN) stays missing and a zero one stays zero.  ``M`` is the map that the
certificate and ``powermean.inference.provider.CovarianceScale`` receive; their
candidate stays in the coordinates of the solver.  Absorbed effects, which carry no
column scale, map with a scale of ones.  A working variance never enters.

Arithmetic.  The divisors are applied with exact exponents.  A value is split into its
significand and its power of two (``frexp``, exact); an intermediate quotient or product
rounds its significand once, as a double would; and the last quotient is one correctly
rounded division of two exact doubles, normal or subnormal (``ldexp``).  So no
intermediate ``omega^2``, ``omega d`` or ``theta / d`` overflows or underflows on the
way to a representable result, and no finite order is restricted for it.  ``omega^2`` is
``omega**2`` wherever that is a normal double, and the square of the significand of
``omega`` elsewhere.  Where the direct expressions (``theta / scale / omega``, ``V /
omega**2``, ``se / omega``) form normal intermediates, the result is theirs bit for bit;
elsewhere it keeps the bits that they lose.  A result beyond the largest double, or a
nonzero result that rounds to zero, raises ``FloatingPointError``: nothing is floored,
clipped or replaced, and no zero is made of a nonzero value.  Arguments are not
modified, and the results are owned read-only arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array, _finite_scalar

IntegerArray = NDArray[np.integer[Any]]
DOUBLE = np.finfo(np.float64)
# The exponent of frexp at the smallest normal double, 2^-1022 = 0.5 * 2^-1021: a
# significand of [0.5, 1) times two to any exponent at or above it is a normal double.
LOWEST_EXPONENT = int(DOUBLE.minexp) + 1
# The exponents e of frexp(omega) at which omega**2 is a normal double: omega^2 lies in
# [2^(2e - 2), 2^(2e)), which is normal from 2e - 2 = -1022 and finite up to 2e = 1024.
SQUARE_EXPONENTS = range((int(DOUBLE.minexp) + 2) // 2, int(DOUBLE.maxexp) // 2 + 1)


# ---------------------------------------------------------- the exact arithmetic


class Split(NamedTuple):
    """A value as ``mantissa * 2^exponent``, exactly: ``frexp``'s significand and power.

    ``|mantissa|`` lies in ``[0.5, 1)``, or the mantissa is a signed zero whose exponent
    does not matter; the exponents stay far inside the range of their integer type.
    """

    mantissa: FloatArray
    exponent: IntegerArray


def _split(values: ArrayLike) -> Split:
    """The exact significands and powers of two of finite doubles (``frexp``)."""
    mantissa, exponent = np.frexp(np.asarray(values, dtype=np.float64))
    return Split(mantissa, exponent)


def _product(left: Split, right: Split) -> Split:
    """``left * right``: the product of the significands rounded once, powers added.

    The significands lie in ``[0.5, 1)``, so their product lies in ``[0.25, 1)``: it
    neither overflows nor underflows, and it rounds as the product of the doubles does
    wherever that is normal.
    """
    mantissa, carry = np.frexp(left.mantissa * right.mantissa)
    return Split(mantissa, left.exponent + right.exponent + carry)


def _quotient(left: Split, right: Split) -> Split:
    """``left / right``, a nonzero divisor: the quotient of significands, rounded once.

    The quotient of the significands lies in ``(0.5, 2)``, or is zero, and the powers
    are subtracted exactly: the significand of the quotient of the doubles, without its
    range.
    """
    mantissa, carry = np.frexp(left.mantissa / right.mantissa)
    return Split(mantissa, left.exponent - right.exponent + carry)


def _final_quotient(left: Split, right: Split, name: str) -> FloatArray:
    """``left / right`` as one correctly rounded division of two exact doubles.

    The numerator is the significand of ``left`` times ``2^(shift + lift)`` and the
    denominator that of ``right`` times ``2^lift``, ``shift`` being the difference of
    the powers and ``lift`` zero unless the numerator would fall below the normal range:
    both are then exact doubles, and their IEEE quotient is rounded once, into the
    normal or the subnormal range, with the sign of the quotient for a zero numerator.
    A nonzero result beyond the largest double, or that rounds to zero, raises
    ``FloatingPointError``, which ``name`` describes.
    """
    shift = left.exponent - right.exponent
    lift = np.maximum(LOWEST_EXPONENT - shift, 0)
    with np.errstate(over="ignore", under="ignore"):
        numerator = np.ldexp(left.mantissa, shift + lift)
        value: FloatArray = numerator / np.ldexp(right.mantissa, lift)
    lost = (left.mantissa != 0) & ((value == 0) | ~np.isfinite(value))
    if np.any(lost):
        raise FloatingPointError(f"{name} is not representable in float64")
    return value


def _square(omega: float) -> Split:
    """``omega^2``: ``omega**2`` where it is a normal double, else carried exactly.

    Outside :data:`SQUARE_EXPONENTS` the double ``omega**2`` overflows or loses bits
    below the normal range; its significand is then the square of that of ``omega``,
    rounded once, with twice its exponent.
    """
    if math.frexp(omega)[1] in SQUARE_EXPONENTS:
        return _split(omega**2)
    root = _split(omega)
    return _product(root, root)


# ------------------------------------------------------------------ the inputs


def _mean_order(value: object) -> float:
    """A finite real order as ``float``, the order zero without a sign.

    Typed as ``object``: the run-time check of what a caller passes, with the refusals
    of :func:`powermean.model.variance._finite_scalar` (booleans, non-reals, NaN and the
    infinities).  Adding zero turns ``-0.0`` into ``0.0`` and leaves every other double
    unchanged, so that the order zero has one form in the records and the messages.
    """
    return _finite_scalar(value, "omega") + 0.0


def _errors(errors: ArrayLike, size: int) -> FloatArray:
    """Standard errors: one real value per coefficient, finite and nonnegative, or NaN.

    NaN is a missing standard error, as the covariance of the inference layer reports a
    negative variance; ``-0.0`` is a zero.  The sign is read on the values that are not
    NaN only, so that no ordered comparison meets a NaN.
    """
    if np.iscomplexobj(errors):
        raise ValueError("standard errors must be real-valued")
    values = np.asarray(errors, dtype=np.float64)
    if values.shape != (size,):
        raise ValueError("standard errors must have one value per coefficient")
    present = values[~np.isnan(values)]
    if np.any(np.isinf(present)) or np.any(present < 0):
        raise ValueError(
            "standard errors must be finite and nonnegative, or missing (NaN)"
        )
    return values


# ------------------------------------------------------------ the order scale


@dataclass(frozen=True, eq=False)
class OrderScale:
    """The map from the coordinates of the solver to the reported scale of an order.

    Parameters
    ----------
    omega : float
        The mean order, finite and real; zero for log OLS, whose order scale is the
        identity.  No working variance enters.
    scale : array
        The column scale ``d`` of the design that the solver received: one positive
        finite value per coefficient of ``theta``; ones for an unscaled design.

    Notes
    -----
    ``omega`` is stored as a ``float``, ``-0.0`` as ``0.0``, and ``scale`` as an owned
    read-only copy; another order or scale raises ``ValueError``.  The methods take
    arrays in the coordinates of ``theta`` and return them on the reported scale, as the
    module docstring states.
    """

    omega: float
    scale: FloatArray

    def __post_init__(self) -> None:
        """Check the order and the column scale, and store them."""
        omega = _mean_order(self.omega)
        scale = _finite_array(self.scale, "scale")
        if scale.ndim != 1 or not scale.size:
            raise ValueError("scale must be a nonempty vector")
        if not np.all(scale > 0):
            raise ValueError("scale must be positive")
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "scale", _immutable(scale))

    @property
    def dimension(self) -> int:
        """The number of coefficients of ``theta``."""
        return int(self.scale.size)

    @property
    def transform(self) -> FloatArray:
        """``M = diag(1 / (omega d))``, or ``diag(1 / d)`` at ``omega = 0``.

        One row per reported coefficient and one column per coefficient of ``theta``,
        the map of a reported-scale provider.  Raises ``FloatingPointError`` where an
        entry is not representable.
        """
        divisor = _split(self.scale)
        if self.omega != 0:
            divisor = _product(divisor, _split(self.omega))
        ones = _split(np.ones(self.dimension))
        diagonal = _final_quotient(ones, divisor, "an entry of the transform")
        return _immutable(np.diag(diagonal))

    def _signed(self, values: FloatArray, name: str) -> FloatArray:
        """``values / d / omega`` on the last axis; ``values / d`` at ``omega = 0``."""
        split, scale = _split(values), _split(self.scale)
        if self.omega == 0:
            return _final_quotient(split, scale, name)
        return _final_quotient(_quotient(split, scale), _split(self.omega), name)

    def coefficients(self, values: ArrayLike) -> FloatArray:
        """``values / d / omega``: coefficients, corrections or biases of ``theta``.

        Parameters
        ----------
        values : array_like
            One finite real value per coefficient of ``theta``.

        Returns
        -------
        array
            The values on the reported scale; ``values / d`` at ``omega = 0``.

        Raises
        ------
        ValueError
            The values are not finite and real, or not one per coefficient.
        FloatingPointError
            A reported value is not representable.
        """
        vector = _finite_array(values, "coefficients")
        if vector.shape != (self.dimension,):
            raise ValueError("coefficients must have one value per coefficient")
        return _immutable(self._signed(vector, "a reported coefficient"))

    def influences(self, rows: ArrayLike) -> FloatArray:
        """``rows / d / omega``, column by column: coefficient influences of ``theta``.

        Parameters
        ----------
        rows : array_like
            The rows ``u_i = A s_i`` of an influence record, one per sampling unit and
            one column per coefficient of ``theta``: the contributions already mapped
            by the influence map, never the raw scores.

        Returns
        -------
        array
            The rows on the reported scale.

        Raises
        ------
        ValueError
            The rows are not a nonempty finite real matrix of one column per
            coefficient.
        FloatingPointError
            A reported value is not representable.
        """
        matrix = _finite_array(rows, "influences")
        if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[1] != self.dimension:
            raise ValueError(
                "influences must be a nonempty matrix with one column per coefficient"
            )
        return _immutable(self._signed(matrix, "a reported influence"))

    def covariance(self, matrix: ArrayLike) -> FloatArray:
        """``V_jk / (d_j d_k) / omega^2``: a covariance of ``theta``, reported.

        Parameters
        ----------
        matrix : array_like
            The covariance ``V`` of ``theta``, finite and real, square, one row per
            coefficient; its sign pattern is not checked.

        Returns
        -------
        array
            The covariance on the reported scale, symmetric where ``V`` is.

        Raises
        ------
        ValueError
            The matrix is not finite and real, or not of its shape.
        FloatingPointError
            A reported entry is not representable.
        """
        values = _finite_array(matrix, "covariance")
        size = self.dimension
        if values.shape != (size, size):
            raise ValueError(
                "covariance must be a square matrix of one row per coefficient"
            )
        scale = _split(self.scale)
        rows = Split(scale.mantissa[:, None], scale.exponent[:, None])
        columns = Split(scale.mantissa[None, :], scale.exponent[None, :])
        outer = _product(rows, columns)
        name = "a reported covariance"
        if self.omega == 0:
            return _immutable(_final_quotient(_split(values), outer, name))
        unscaled = _quotient(_split(values), outer)
        return _immutable(_final_quotient(unscaled, _square(self.omega), name))

    def solver_coefficients(self, values: ArrayLike) -> FloatArray:
        """``values * d * omega``: a start on the reported scale, in ``theta``.

        Parameters
        ----------
        values : array_like
            One finite real value per reported coefficient: a start ``beta``.

        Returns
        -------
        array
            ``theta_j = d_j omega beta_j``, or ``d_j beta_j`` at ``omega = 0``, each
            the one correctly rounded product of exact factors: no intermediate
            ``omega d`` is formed, so neither its overflow nor its underflow reaches
            a representable result.

        Raises
        ------
        ValueError
            The values are not finite and real, or not one per coefficient.
        FloatingPointError
            A nonzero product beyond the doubles, or one that rounds to zero.
        """
        vector = _finite_array(values, "start")
        if vector.shape != (self.dimension,):
            raise ValueError("start must have one value per coefficient")
        product = _product(_split(vector), _split(self.scale))
        if self.omega != 0:
            product = _product(product, _split(self.omega))
        ones = _split(np.ones(self.dimension))
        return _immutable(_final_quotient(product, ones, "a start of the solver"))

    def standard_errors(self, errors: ArrayLike) -> FloatArray:
        """``s / d / |omega|``: standard errors of ``theta`` on the reported scale.

        Parameters
        ----------
        errors : array_like
            One standard error per coefficient of ``theta``: finite and nonnegative, or
            NaN where it is missing.

        Returns
        -------
        array
            The standard errors on the reported scale, NaN where they are missing.

        Raises
        ------
        ValueError
            The errors are not real, not one per coefficient, negative or infinite.
        FloatingPointError
            A reported standard error is not representable.
        """
        values = _errors(errors, self.dimension)
        missing = np.isnan(values)
        present = np.where(missing, 0.0, values)
        name = "a reported standard error"
        scale = _split(self.scale)
        if self.omega == 0:
            mapped = _final_quotient(_split(present), scale, name)
        else:
            unscaled = _quotient(_split(present), scale)
            mapped = _final_quotient(unscaled, _split(abs(self.omega)), name)
        mapped[missing] = np.nan
        return _immutable(mapped)
