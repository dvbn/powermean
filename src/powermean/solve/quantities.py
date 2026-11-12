"""The quantities of the certificate, from the sums and the observed bread.

At a fresh evaluation of the
joint system (:mod:`powermean.systems.joint`), with the moment ``S``, the observed
bread ``C = -dS/dgamma'`` and the active design ``X``:

    curvature        C_sym = (C + C')/2 positive definite: its Cholesky factor L,
                     L L' = C_sym, exists (:mod:`powermean.solve.factorization`)
    scaled score     ||C_sym^(-1/2) S|| = ||L^-1 S||, where C_sym is positive definite
    correction       p = C^-1 S by the equilibrated LU, when its estimate of the
                     reciprocal condition number exceeds rcond_min: gamma + p is the
                     next Newton iterate, an additive correction
    index correction sqrt(sum_i n_i (x_i' p)^2 / sum_i n_i), the root mean square
                     over the observations of the change of the fitted index that one
                     more Newton step makes, x_i' the row i of the system's active
                     design and n_i its frequency weight, or one

The sums are those of the system, not the averages of the iteration, whose common
factor ``1/A`` leaves ``p`` unchanged but divides the scaled score by ``sqrt(A)``, and
the curvature is the observed bread, not the expected information.  Frequency weights
count replicated observations; analytic and probability weights, or none, count each
row once: the caller states
which, and the kind is never inferred from the values.  Rows of zero weight are not
in the active design and enter nothing.  Each norm is formed on the values divided by
the largest of them, and the multiplicities by theirs, so that no square or sum
overflows.  The root mean square takes the mean of the squared ratios, at most one,
before its root and the product by the largest value, so that the root mean square of
finite values is finite where the root of their sum of squares is beyond the doubles.
A NaN or infinite value is kept as such and passes no gate.  A factorization that
fails returns its error and no quantity in its place: nothing is ridged, floored or
replaced by a pseudoinverse or the expected curvature.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray
from powermean.solve.factorization import (
    SingularJacobianError,
    factorize,
    forward_solve,
    symmetric_cholesky,
)
from powermean.systems.joint import JointSystem


def _scaled_root(values: FloatArray, share: FloatArray, total: float) -> float:
    """``sqrt(sum_i s_i v_i^2 / total)`` as ``max|v| sqrt(m)``; NaN or inf propagate.

    ``m = sum_i s_i (v_i / max|v|)^2 / total`` is formed before its root and the
    product by ``max|v|``, so that no square of a value is formed.
    """
    if np.isnan(values).any():
        return math.nan
    largest = float(np.max(np.abs(values)))
    if largest == 0.0 or math.isinf(largest):
        return largest
    ratio = values / largest
    return largest * math.sqrt(float(np.sum(share * ratio * ratio)) / total)


def euclidean_norm(values: FloatArray) -> float:
    """``||v||``, without overflow of a square: the scaled score of ``L^-1 S``.

    Infinite only where the norm itself is beyond the doubles, up to its rounding.
    """
    return _scaled_root(values, np.ones(values.shape), 1.0)


def root_mean_square(values: FloatArray, multiplicity: FloatArray) -> float:
    """``sqrt(sum_i n_i v_i^2 / sum_i n_i)`` for positive multiplicities ``n_i``.

    The shares ``s_i = n_i / max n`` cannot overflow their sum, and the mean of the
    squared ratios, at most one, is formed before the root and the product by
    ``max|v|``: the root mean square of finite values is finite, at most ``max|v|``,
    where the root of their sum of squares can be beyond the doubles.
    """
    share = multiplicity / np.max(multiplicity)
    return _scaled_root(values, share, float(np.sum(share)))


def index_correction(
    system: JointSystem, correction: FloatArray, *, frequency: bool
) -> float:
    """The root mean square of ``X p`` over the observations of the active rows.

    ``X`` is the system's ``active_design``; each row counts its frequency weight of
    observations when ``frequency`` holds, and one otherwise.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        change = system.active_design @ correction
    multiplicity = system.active_weights if frequency else np.ones(change.shape)
    return root_mean_square(change, multiplicity)


class Curvature(NamedTuple):
    """The curvature gate and the scaled score, or the error of the factorization."""

    positive_definite: bool
    scaled_score: float
    error: str


def curvature(moment: FloatArray, bread: FloatArray) -> Curvature:
    """``C_sym`` positive definite, and then ``||L^-1 S||``; otherwise NaN."""
    try:
        lower = symmetric_cholesky(bread)
    except np.linalg.LinAlgError as error:
        return Curvature(
            positive_definite=False, scaled_score=math.nan, error=str(error)
        )
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = forward_solve(lower, moment)
    return Curvature(
        positive_definite=True, scaled_score=euclidean_norm(whitened), error=""
    )


class Step(NamedTuple):
    """The Newton correction and the estimate, or the error of the factorization."""

    correction: FloatArray | None
    rcond: float
    error: str


def newton_step(moment: FloatArray, bread: FloatArray, rcond_min: float) -> Step:
    """``p = C^-1 S`` from the equilibrated LU of ``C``, when it passes its test."""
    try:
        factor = factorize(bread, rcond_min)
    except SingularJacobianError as error:
        return Step(correction=None, rcond=error.rcond, error=str(error))
    with np.errstate(over="ignore", invalid="ignore"):
        correction = factor.solve(moment)
    return Step(correction=_immutable(correction), rcond=factor.rcond, error="")
