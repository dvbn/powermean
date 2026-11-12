"""The order layer: the transformed problem of a mean order.

For a mean order ``omega`` the model is ``E(Y^omega | X) = exp(omega X'beta)``, and at
``omega = 0`` it is ``E(log Y | X) = X'beta`` on positive outcomes.  The systems,
solvers, the certificate and the covariances of the package work on the transformed
problem and know nothing of ``omega``: :func:`transform_problem` forms that problem, and
:mod:`powermean.model.order_scale` maps its results back, once.

    omega != 0   route "power"    Z = Y^omega   offset omega o   gamma = omega beta
    omega = 0    route "linear"   log Y         offset o         least squares

The offset ``o`` has unit coefficient on the log power mean, so the index of the
transformed problem receives ``omega o``; an exposure ``e`` is the offset ``log e``,
which the caller forms.  The route names the stage that solves the problem, none of
which this layer calls.  ``"power"`` goes to an estimating system with a working
variance of the caller's choice: ``omega`` is not a parameter of the working variance.
``"linear"`` goes to the linear path, ``powermean.solve.linear.solve(response, design,
offset=offset, weights=...)``, whose influence record
(``powermean.inference.contributions.linear_influence``) needs the design of every row
given to that solve, rows of zero weight included; its order scale is the identity,
``OrderScale(0, d)``.  At ``omega = 0`` nothing is raised to a power or exponentiated,
nothing is divided by ``omega`` and no working variance enters.  A linear candidate is
not a ``JointEvaluation``: the certificate of the linear path, which a public log OLS
fit needs, is not part of this layer.

The domain, in this order:

1. A negative outcome raises ``SampleError`` (``sample.negative_outcome``), at every
   order and whatever ``positive_only``.
2. A zero outcome is admitted at ``omega > 0``, where ``Z`` is ``+0.0`` exactly at the
   zero outcomes and only there.  At ``omega <= 0`` it raises ``SampleError``
   (``sample.zero_outcome_at_order``) unless ``positive_only`` is true.
3. With ``positive_only`` the rows of zero outcome are not in the problem, at every
   order: the literal meaning of the flag, under which ``power_effects`` fits all its
   orders on the positive outcomes.
   A problem left without a row raises ``SampleError`` (``sample.empty``).

No row leaves silently.  :class:`Selection` holds, aligned with the input rows, the mask
of the rows kept, their positions, the rows removed and whether the problem conditions
on positive outcomes, for the sample accounting.  A position ``k`` among the rows of the
problem, the identifier of a sampling unit of a later system or solve for instance, is
the input row ``positions[k]``; the caller selects the design, the weights and the
cluster codes of the problem by ``positions``.  Missing values, ``subset`` and weights
belong to the data layer: the inputs are finite arrays, and a malformed one raises
``ValueError`` or ``TypeError``, as in the other numerical layers.

The power.  ``Z = Y^omega`` is formed as ``Y ** omega`` on the rows of the problem, and
checked in this order: a positive outcome whose power rounds to zero is refused rather
than made a new zero ("power transformation changed the zero mask"), and so is a power
beyond the doubles ("non-finite transformed outcome"), each a ``FloatingPointError``; a
subnormal power is kept.  The transformed offset is the one rounded product ``omega o``,
refused where it is beyond the doubles or where a nonzero offset rounds to zero.
Arguments are not modified, and the records hold owned read-only arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.errors import SampleError
from powermean.model.order_scale import _mean_order
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array

BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.intp]
Route = Literal["power", "linear"]
ROUTES: tuple[Route, ...] = get_args(Route)


def _flags(mask: BoolArray) -> BoolArray:
    """An owned read-only copy of a boolean array, in bytes of its own, in its shape."""
    stored = np.frombuffer(mask.tobytes(), dtype=np.bool_)
    copy: BoolArray = stored.reshape(mask.shape)
    return copy


def _positions(mask: BoolArray) -> IndexArray:
    """The positions at which ``mask`` holds, increasing, read-only, owned."""
    found = np.flatnonzero(mask).tobytes()
    positions: IndexArray = np.frombuffer(found, dtype=np.intp)
    return positions


# ---------------------------------------------------------------------- the records


@dataclass(frozen=True, eq=False)
class Selection:
    """The input rows of a transformed problem, and those removed by ``positive_only``.

    Attributes
    ----------
    mask : array of bool
        One flag per input row, true where the row is in the problem.
    positions : array of int
        The positions of those rows among the input rows, increasing: the ``k``-th row
        of the problem is the input row ``positions[k]``.
    removed : array of int
        The positions of the rows of zero outcome that ``positive_only`` removed;
        empty without it.
    nobs : int
        The number of input rows.
    positive_only : bool
        Whether the caller asked for the positive outcomes only.

    Notes
    -----
    Read-only arrays in bytes of their own.  Every input row is in ``positions`` or in
    ``removed``, and in one of them only.
    """

    mask: BoolArray
    positions: IndexArray
    removed: IndexArray
    nobs: int
    positive_only: bool

    @property
    def count(self) -> int:
        """The number of rows of the problem."""
        return int(self.positions.size)

    @property
    def conditioned(self) -> bool:
        """Whether the problem conditions on positive outcomes: a zero was removed.

        A change of population, which the output states.
        """
        return bool(self.removed.size)


@dataclass(frozen=True, eq=False)
class TransformedProblem:
    """The problem that a mean order hands to the solving stages, on its rows.

    Attributes
    ----------
    omega : float
        The mean order.
    route : {'power', 'linear'}
        ``'power'`` at ``omega != 0``, for an estimating system on ``(Z, gamma)``;
        ``'linear'`` at ``omega = 0``, for the linear path on ``log Y``.
    response : array
        ``Z = Y^omega`` or ``log Y``, one value per row of the problem.
    offset : array
        The offset of the index of the problem, ``omega o``, or ``o`` itself at
        ``omega = 0``, aligned with ``response``.
    selection : Selection
        The input rows of the problem, and the rows removed.

    Notes
    -----
    The arrays are stored as owned read-only copies.  No working variance, design,
    weight or solver is held.
    """

    omega: float
    route: Route
    response: FloatArray
    offset: FloatArray
    selection: Selection

    def __post_init__(self) -> None:
        """Store the response and the offset as owned read-only copies."""
        object.__setattr__(self, "response", _immutable(self.response))
        object.__setattr__(self, "offset", _immutable(self.offset))


# ----------------------------------------------------------------------- the inputs


def _outcome(y: ArrayLike) -> FloatArray:
    """The outcome as supplied: a nonempty vector of finite real values."""
    outcome = _finite_array(y, "y")
    if outcome.ndim != 1 or not outcome.size:
        raise ValueError("y must be a nonempty vector")
    return outcome


def _offset(offset: ArrayLike | None, rows: int) -> FloatArray:
    """The offset ``o``: zero if omitted, a scalar for every row, else one per row."""
    if offset is None:
        return np.zeros(rows)
    shift = _finite_array(offset, "offset")
    if shift.ndim == 0:
        shift = np.full(rows, float(shift))
    if shift.shape != (rows,):
        raise ValueError("offset must be a scalar or a vector aligned with y")
    return shift


def _flag(value: object) -> bool:
    """``positive_only``, which must be a ``bool``: the run-time check of its type."""
    if not isinstance(value, bool):
        raise TypeError("positive_only must be a bool")
    return value


# ---------------------------------------------------------------------- the domain


def _selection(outcome: FloatArray, omega: float, positive_only: bool) -> Selection:
    """The rows of the problem, by the domain of the order (the list of the module)."""
    negative = int(np.count_nonzero(outcome < 0))
    if negative:
        raise SampleError("sample.negative_outcome", {"count": negative})
    zero = outcome == 0
    zeros = int(np.count_nonzero(zero))
    if zeros and omega <= 0 and not positive_only:
        raise SampleError(
            "sample.zero_outcome_at_order", {"count": zeros, "omega": omega}
        )
    mask = ~zero if positive_only else np.ones(outcome.shape, dtype=np.bool_)
    if not mask.any():
        raise SampleError("sample.empty", {"count": int(outcome.size)})
    return Selection(
        mask=_flags(mask),
        positions=_positions(mask),
        removed=_positions(~mask),
        nobs=int(outcome.size),
        positive_only=positive_only,
    )


# ------------------------------------------------------------- the transformations


def _power(values: FloatArray, omega: float) -> FloatArray:
    """``Z = Y^omega`` on the rows of the problem, with its checks.

    ``values ** omega`` is NumPy's power, its fast paths included (the array of doubles
    that it returns is taken as it is); a zero outcome is then set to ``+0.0``, since
    the power of ``-0.0`` at an odd integer order is ``-0.0``.  Refused, in this order:
    a positive outcome whose power is zero, then a power that is not finite.  At ``omega
    < 0`` every value is positive.
    """
    zero = values == 0
    with np.errstate(over="ignore", under="ignore"):
        z = np.asarray(values**omega, dtype=np.float64)
    z[zero] = 0.0
    if not np.array_equal(z == 0, zero):
        raise FloatingPointError("power transformation changed the zero mask")
    if not np.isfinite(z).all():
        raise FloatingPointError("non-finite transformed outcome")
    return z


def _logarithm(values: FloatArray) -> FloatArray:
    """``log Y`` of positive outcomes: finite for every positive double."""
    response: FloatArray = np.log(values)
    return response


def _transformed_offset(shift: FloatArray, omega: float) -> FloatArray:
    """``omega o``, one rounded product, refused where it is not representable.

    A product beyond the doubles, or a nonzero offset whose product rounds to zero,
    raises ``FloatingPointError``; a zero offset stays zero.
    """
    with np.errstate(over="ignore", under="ignore"):
        product: FloatArray = omega * shift
    lost = (shift != 0) & ((product == 0) | ~np.isfinite(product))
    if np.any(lost):
        raise FloatingPointError("transformed offset is not representable in float64")
    return product


def transform_problem(
    y: ArrayLike,
    omega: float,
    *,
    offset: ArrayLike | None = None,
    positive_only: bool = False,
) -> TransformedProblem:
    """The transformed problem of the outcome ``y`` at the mean order ``omega``.

    Parameters
    ----------
    y : array_like
        The outcome as supplied, one finite real value per input row.
    omega : float
        The mean order, finite and real.
    offset : float or array_like, optional
        The offset ``o`` of the log power mean, with unit coefficient: zero if omitted,
        the same value for every row if a scalar, else one finite value per row.  An
        exposure enters as its logarithm, which the caller forms.
    positive_only : bool, default False
        Whether the problem keeps the rows of positive outcome only.

    Returns
    -------
    TransformedProblem
        The route, the response and the offset on the rows kept, and their selection.

    Raises
    ------
    ValueError
        ``y``, ``omega`` or ``offset`` is not finite and real, or not of its shape.
    TypeError
        ``positive_only`` is not a ``bool``.
    SampleError
        A negative outcome; a zero outcome at ``omega <= 0`` without ``positive_only``;
        no row left.
    FloatingPointError
        A power or a transformed offset that is not representable.
    """
    order = _mean_order(omega)
    outcome = _outcome(y)
    shift = _offset(offset, int(outcome.size))
    selection = _selection(outcome, order, _flag(positive_only))
    values, kept = outcome[selection.mask], shift[selection.mask]
    if order == 0:
        return TransformedProblem(
            omega=order,
            route="linear",
            response=_logarithm(values),
            offset=kept,
            selection=selection,
        )
    return TransformedProblem(
        omega=order,
        route="power",
        response=_power(values, order),
        offset=_transformed_offset(kept, order),
        selection=selection,
    )
