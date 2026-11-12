"""A solve that ended before a candidate existed: the failure of the automatic start.

The
automatic start of the convex Newton solver (:func:`powermean.solve.starts.
automatic_start`) places the logarithm of the mean of the transformed outcome in the
intercept; where that logarithm is not a finite double, as on an outcome of zeros on
every row of positive weight with an intercept, or where the mean's sum is beyond
the doubles, it raises ``FloatingPointError`` and no start, no iteration and no
candidate exist.  The accepted algorithm is unchanged: nothing is clipped, no other
start is substituted, and whether the equation has a finite root at all is the
question of an existence analysis, which is not answered here.

:class:`StartFailure` is the truthful record of that boundary: the text of the error,
the mean whose logarithm had no finite value, and the intercept column.  Its status
is ``nonfinite`` (:data:`START_FAILURE_STATUS`): the value that the solve would have
started from is not finite, as the gate ``evaluation`` of a certificate reads a
candidate that is not finite.  No certificate, trace or point is attached to it, and
the sample accounting of the preparation stays available beside it.  A design without
an intercept has the finite zero start and proceeds; a malformed user start and an
unavailable provider remain the errors of their own stages, never this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from powermean.solve.status import Status

StartSource = Literal["automatic"]
START_FAILURE_STATUS: Status = "nonfinite"


@dataclass(frozen=True)
class StartFailure:
    """The automatic start had no finite value: no candidate exists.

    Attributes
    ----------
    message : str
        The text of the ``FloatingPointError`` of the automatic start.
    mean : float
        The mean of the transformed outcome on the rows of positive weight, whose
        logarithm is not a finite double.
    intercept : int
        The intercept column that the start would have filled.
    source : {"automatic"}
        The origin of the start that failed; a user start is checked before.
    """

    message: str
    mean: float
    intercept: int
    source: StartSource = "automatic"

    @property
    def status(self) -> Status:
        """``nonfinite``: the status of this boundary."""
        return START_FAILURE_STATUS
