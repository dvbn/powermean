"""The existence indications of the profiled Newton solver: ZC, GC and their alarm.

During iteration the solver stops when fitted means at zero rows collapse or the
criterion decreases without bound, and nothing is dropped on the basis of an iterate.
Internal module, separate from the whole-point state of
:mod:`powermean.solve.profiled_loss`.

Indications: :func:`zero_row_alarm` (ZC) and
:func:`chord_alarm` (GC, ``tau = 2`` only), evaluated by the whole-point state at the
completed initial profile, at each newly accepted point after the native convergence
test (before the iteration could honor it), and at the fresh final point; never at a
held or refused trial.  ZC is an UNSCALED adaptation, in the original transformed units,
of the log-mean heuristic of ``ppmlhdfe`` 2.3.4 (``ppmlhdfe.mata``, lines 137 to 139,
591 to 592 and 683 to 690, read as a convention; no code copied): ``ppmlhdfe``
standardizes the outcome by default, so no parity of its alarms or unit invariance is
claimed.  GC compares consecutive
accepted points of one attempt with exact comparisons and a checked normalization;
its rounded
chord is an indication, not a reconstructed recession direction, and a finite decrease
proves no unboundedness.  Indications at one point coalesce into one :class:`Alarm`.
Neither the presence nor the absence of an alarm is an existence certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.fe.absorb_numeric import checked_difference
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.support_gamma_lp import dot, normalized
from powermean.solve.support_problem import indices

# ppmlhdfe 2.3.4, ppmlhdfe.mata, line 138: mu_tol = 1e-6, used through log(mu_tol)
# (line 592).
LOG_MEAN_TOLERANCE = math.log(1e-6)
# ppmlhdfe 2.3.4, ppmlhdfe.mata, line 687: the margin 5 added to the least index of a
# positive row.
INDEX_MARGIN = 5.0

Where = Literal["initial", "accepted", "final"]


@dataclass(frozen=True, eq=False)
class ZeroRowAlarm:
    """ZC: zero rows whose index is at most ``h``.

    ``threshold`` is ``h = log(1e-6) + min(lowest + 5, 0)``, ``lowest`` the least index
    of a positive row; ``rows`` the triggering rows of the sample, ``positions`` their
    original positions and ``indexes`` their indexes.
    """

    threshold: float
    lowest: float
    rows: IndexArray
    positions: IndexArray
    indexes: FloatArray


def zero_row_alarm(
    z: FloatArray, index: FloatArray, positions: IndexArray
) -> ZeroRowAlarm | None:
    """ZC at one point, or ``None``: some ``q[i] <= h`` on a zero row.

    Evaluated on the log scale, with no exponential, zeroing or clipping.  Without a
    zero row there is no ZC; without a positive row ``ValueError``: the sample's
    support is unresolved, and no minimum is fabricated.
    """
    positive = z > 0
    if not positive.any():
        raise ValueError("no row has a positive outcome: the support is unresolved")
    lowest = float(np.min(index[positive]))
    threshold = LOG_MEAN_TOLERANCE + min(lowest + INDEX_MARGIN, 0.0)
    rows = np.flatnonzero(~positive & (index <= threshold))
    if not rows.size:
        return None
    return ZeroRowAlarm(
        threshold,
        lowest,
        indices(rows),
        indices(positions[rows]),
        immutable_array(index[rows]),
    )


@dataclass(frozen=True, eq=False)
class ChordAlarm:
    """GC: an accepted GPML decrease along a recession-shaped chord.

    ``kind`` is ``negative_sum`` (``sigma < 0``) or ``zero_sum`` (``sigma = 0`` with
    some positive-row movement); ``losses`` the two accepted ``L/A``; ``sigma`` the
    normalized weighted sum of the chord ``d``; ``lowest`` and ``highest`` its extrema
    over every row and ``positive_lowest`` over the positive rows; ``before`` and
    ``after`` the two indexes.  An indication only: the chord is rounded.
    """

    kind: str
    losses: tuple[float, float]
    sigma: float
    lowest: float
    highest: float
    positive_lowest: float
    before: FloatArray
    after: FloatArray


def chord_alarm(
    z: FloatArray,
    weights: FloatArray,
    before: tuple[float, FloatArray],
    after: tuple[float, FloatArray],
) -> ChordAlarm | None:
    """GC between two consecutive accepted points ``(L/A, q)``, or ``None``.

    Requires ``L_new < L_old``; ``d = q_new - q_old`` by the checked difference; when
    every ``d[P] >= 0``, ``sigma = sum(rho d)`` with the checked normalized weights
    and product sum, and the alarm when ``sigma < 0`` or ``sigma == 0`` with some
    ``d[P] > 0``.  Exact comparisons, no margin.  ``ValueError`` for points of other
    rows; ``FloatingPointError`` when a required quantity is not representable, which
    is a numerical failure and never an absence of alarm.
    """
    (old, q_old), (new, q_new) = before, after
    if q_old.shape != z.shape or q_new.shape != z.shape:
        raise ValueError("the two points must index the rows of the sample")
    if not new < old:
        return None
    d = checked_difference(q_new, q_old, "index chord")
    moved = d[z > 0]
    if not (moved >= 0).all():
        return None
    sigma = dot(normalized(weights), d)
    if sigma < 0:
        kind = "negative_sum"
    elif sigma == 0 and (moved > 0).any():
        kind = "zero_sum"
    else:
        return None
    return ChordAlarm(
        kind,
        (old, new),
        sigma,
        float(d.min()),
        float(d.max()),
        float(moved.min()),
        q_old,
        q_new,
    )


@dataclass(frozen=True, eq=False)
class Alarm:
    """The indications at one point, coalesced: one stop, one diagnostic request.

    ``where`` is ``initial``, ``accepted`` or ``final``; ``attempt`` and ``iteration``
    locate it (accepted steps before it); ``converged`` is the native convergence flag
    there, the native history kept apart from the operative stop.
    """

    where: Where
    attempt: int
    iteration: int
    converged: bool
    zero_rows: ZeroRowAlarm | None
    chord: ChordAlarm | None
