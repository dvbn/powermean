"""Starting values of the convex Newton solver: one vector, automatic or the user's.

The damped
Newton solver takes one starting value and iterates from it: no pilot, no path, no
restart ladder and no grid.  For the convex criteria of the joint system (the power
family with ``tau`` in ``[1, 2]``, and NB2) the root does not depend on the start, which
affects the time only.

Automatic start, with no pilot or truth-based start: the logarithm of the mean of
``z`` in the intercept column and zero in every
other column, or the zero vector when the design has no intercept.  The intercept is
named by the caller, as the index of its column, and is never presumed: a design of
the package may have none or hold it elsewhere.  The
named column must be a column of ones.  The mean is the plain mean of ``z`` over the
rows of positive weight, those that the system solves, as ``log(z.mean())``: neither
weighted nor adjusted for the offset, which would be a pilot.

No early return is made when the mean is not positive: without an intercept the zero
start is finite and the solve proceeds (an
outcome of zeros on the rows ``-1`` and ``+1`` of a single column has the Poisson root
zero).  With an intercept and a zero mean, ``log(0)`` has no finite value, and
:func:`automatic_start` raises ``FloatingPointError``, as the system does for a value
that is not representable; nothing is clipped or floored.  Whether the equation has a
finite root at all is the question of the existence analysis, not answered here.

User start.  A finite vector with one value per column of the design, copied and
recorded; always accepted.  :class:`Start` records the vector and its
origin, for the trace of the solver and the provenance of a result.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array
from powermean.systems.joint import JointSystem

StartSource = Literal["automatic", "user"]
SOURCES: tuple[StartSource, ...] = ("automatic", "user")


@dataclass(frozen=True, eq=False)
class Start:
    """One starting value of the solver and its origin.

    Parameters
    ----------
    coefficients : array
        The start, a nonempty finite vector, stored as an owned read-only copy.
    source : {'automatic', 'user'}
        :func:`automatic_start` or :func:`user_start`.
    intercept : int, optional
        The column that the automatic start identified as the intercept, if any.
    mean : float, optional
        The mean of ``z`` whose logarithm the automatic start placed there.

    Notes
    -----
    A vector that is not finite, not one-dimensional or empty raises ``ValueError``,
    as does an unknown source.
    """

    coefficients: FloatArray
    source: StartSource
    intercept: int | None = None
    mean: float | None = None

    def __post_init__(self) -> None:
        """Check the vector and the source; store the vector as a read-only copy."""
        values = _finite_array(self.coefficients, "start")
        if values.ndim != 1 or not values.size:
            raise ValueError("start must be a nonempty vector")
        object.__setattr__(self, "coefficients", _immutable(values))
        object.__setattr__(self, "source", _source(self.source))


def _source(value: object) -> StartSource:
    """``value`` if it is one of :data:`SOURCES`, the run-time check of the field."""
    for source in SOURCES:
        if value == source:
            return source
    raise ValueError("start source must be 'automatic' or 'user'")


def _intercept_column(system: JointSystem, intercept: object) -> int:
    """The index of the intercept column, checked: an integer column of ones.

    Typed as ``object`` because it is the run-time check of the caller's metadata: a
    boolean, a float or another object is refused, as is an index outside the design
    or a column that is not one in every row.
    """
    if isinstance(intercept, (bool, np.bool_)) or not isinstance(intercept, Integral):
        raise ValueError("intercept must be the integer index of a column, or None")
    column = operator.index(intercept)
    if not 0 <= column < system.dimension:
        raise ValueError("intercept must be the index of a column of the design")
    if not np.all(system.design[:, column] == 1.0):
        raise ValueError("the intercept column must hold one in every row")
    return column


def automatic_start(system: JointSystem, *, intercept: int | None) -> Start:
    """The automatic start: ``log(mean z)`` in the intercept, zero slopes.

    Parameters
    ----------
    system : JointSystem
        The system to solve; its rows of positive weight give the mean.
    intercept : int or None
        The index of the intercept column, a column of ones; ``None`` when the design
        has no intercept, and the start is then the zero vector.

    Returns
    -------
    Start
        The automatic start, with the intercept column and the mean it used.

    Raises
    ------
    ValueError
        The intercept is not the index of a column of ones of the design.
    FloatingPointError
        The logarithm of the mean is not a finite double: a zero mean, as with an
        outcome of zeros on every row of positive weight, or a sum beyond the doubles.
    """
    coefficients = np.zeros(system.dimension)
    if intercept is None:
        return Start(coefficients=coefficients, source="automatic")
    column = _intercept_column(system, intercept)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean = float(np.mean(system.active_z))
        log_mean = float(np.log(mean))
    if not math.isfinite(log_mean):
        raise FloatingPointError(
            f"the automatic start has no finite intercept: log of the mean {mean!r}"
        )
    coefficients[column] = log_mean
    return Start(
        coefficients=coefficients, source="automatic", intercept=column, mean=mean
    )


def user_start(system: JointSystem, values: ArrayLike) -> Start:
    """A start given by the user, accepted if it is finite and of the right length.

    Parameters
    ----------
    system : JointSystem
        The system to solve.
    values : array_like
        One finite value per column of the design.

    Returns
    -------
    Start
        The vector, copied, with the source ``'user'``.

    Raises
    ------
    ValueError
        A value is not finite or not real, or the length is not that of the design.
    """
    coefficients = _finite_array(values, "start")
    if coefficients.shape != (system.dimension,):
        raise ValueError("start must have one value per column of the design")
    return Start(coefficients=coefficients, source="user")
