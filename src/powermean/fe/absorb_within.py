"""A warm start of the absorber from ``within`` (``within-py``, ``pyfixest``'s engine).

Owner's ruling 79: the absorption of fixed-effect log-OLS uses ``within``.  It solves
the weighted normal equations of the indicator design, ``D'A D alpha = D'A y`` for
every column ``y``, by preconditioned LSMR.  Its coefficients are handed to
:func:`~powermean.fe.absorb.absorb` as a warm :class:`~powermean.fe.absorb.
AbsorptionState`; the absorber then forms the residual ``y - lift(alpha)`` itself and
applies its own fresh check (the largest weighted group mean of the residual over
every block, level and column at most the tolerance).  Coefficients that pass are
returned without a sweep; coefficients that fail are continued from by the absorber's
own iteration to its tolerance.  The acceptance of the absorption is thus the
package's, whatever ``within`` returns.

No start (``None``, the cold start of the absorber) when ``within`` raises, reports a
column that did not converge, returns a coefficient that is not finite, or numbers the
levels of a block otherwise than the grouping's codes.  ``within`` is run with its
defaults except the tolerance: its relative normal-equation residual at most
:data:`TOLERANCE`.  It is multithreaded: across thread counts its coefficients differ
in the last bits (its documentation), so the residuals that pass the fresh check do too.
"""

from __future__ import annotations

import numpy as np
import within

from powermean.fe.absorb import AbsorptionState
from powermean.fe.absorb_numeric import PreparedWeights
from powermean.model.variance import FloatArray

TOLERANCE = 1e-12
"""``within``'s relative normal-equation residual, ``||D'A(y - D a)|| / ||D'A y||``."""


def within_start(
    values: FloatArray, prepared: PreparedWeights, columns: tuple[str, ...]
) -> AbsorptionState | None:
    """The coefficients of ``within`` for every column of ``values``, or ``None``.

    ``values`` has one row per row of ``prepared.grouping`` and one column per name of
    ``columns``; the weights are ``prepared.weights``.  ``None`` is the cold start
    (module docstring).
    """
    grouping = prepared.grouping
    blocks = grouping.blocks
    codes = np.empty((grouping.n_rows, len(blocks)), dtype=np.uint32, order="F")
    for b, block in enumerate(blocks):
        codes[:, b] = block.codes
    y = np.asfortranarray(values, dtype=np.float64)
    try:
        found = within.solve_batch(
            codes, y, prepared.weights, within.LsmrOptions(tol=TOLERANCE)
        )
    except (RuntimeError, ValueError, TypeError):
        return None
    if not all(found.converged):
        return None
    x = np.asarray(found.x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    layout = found.layout
    coefficients = []
    for b, block in enumerate(blocks):
        n_levels = block.n_levels
        if layout.n_levels(b) != n_levels:
            return None
        start = layout.index(b, 0, 0)
        if layout.index(b, n_levels - 1, 0) != start + n_levels - 1:
            return None
        coefficients.append(x[start : start + n_levels])
    if not all(np.isfinite(a).all() for a in coefficients):
        return None
    return AbsorptionState(grouping, columns, tuple(coefficients))


__all__ = ["TOLERANCE", "within_start"]
