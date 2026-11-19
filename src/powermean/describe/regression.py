"""The linear regression behind the balance table and the balance regression.

The regression of a variable on arm indicators, or of an arm indicator on the
variables, with an intercept or with one classification of strata absorbed, and with
the covariance that the call names.  Arrays in, one record out; the formulas are those
of :mod:`powermean.describe.numeric`.

==================  ================================================================
``vcov``            covariance and reference
==================  ================================================================
``"robust"``        the sandwich on rows, or on cluster sums with ``clusters``, times
                    the factor of the ``ssc`` preset, with the preset's reference
``"classical"``     ``s^2 (X'WX)^{-1}`` with ``t`` on ``n - K`` degrees of freedom,
                    the covariance of ``lm`` in R (balance regression only)
==================  ================================================================

A design that has not full column rank gives no inference: the record says so, and
the caller reports empty cells (balance table) or refuses (balance regression).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from powermean.describe.numeric import (
    Counts,
    FloatArray,
    IntArray,
    Preset,
    Reference,
    absorb,
    classical,
    convention,
    least_squares,
    parameter_counts,
    sandwich,
)


class Estimates(NamedTuple):
    """The slopes of a regression with their covariance and its reference.

    ``beta`` and ``covariance`` concern the slopes only, in the order of the columns
    given; ``intercept`` and ``intercept_variance`` are NaN with strata.
    ``clusters`` and ``strata`` are zero where the call names none.
    """

    beta: FloatArray
    covariance: FloatArray
    intercept: float
    intercept_variance: float
    reference: Reference
    deficient: bool
    n: int
    clusters: int
    strata: int


def consecutive(codes: IntArray) -> tuple[IntArray, int]:
    """The codes renumbered ``0..count-1`` on the rows given, with their number."""
    levels, inverse = np.unique(codes, return_inverse=True)
    return np.asarray(inverse, dtype=np.intp), int(levels.size)


def _nested(strata: IntArray, clusters: IntArray, count: int) -> bool:
    """Whether every stratum lies inside one cluster."""
    pairs = np.unique(np.column_stack([strata, clusters]), axis=0)
    return int(pairs.shape[0]) == count


def regress(
    y: FloatArray,
    columns: FloatArray,
    w: FloatArray | None,
    clusters: IntArray | None,
    strata: IntArray | None,
    *,
    vcov: Literal["robust", "classical"],
    ssc: Preset,
) -> Estimates:
    """Regress ``y`` on ``columns`` with an intercept, or within strata.

    Parameters
    ----------
    y : array
        The dependent variable, ``n`` rows.
    columns : array
        The ``n`` by ``k`` regressors, without a constant.
    w : array, optional
        Positive weights.
    clusters, strata : array, optional
        Integer codes of the clusters of the covariance and of the absorbed
        classification, on the same rows.
    vcov : {"robust", "classical"}
        The covariance (module docstring).
    ssc : {"stata", "none", "regression"}
        The small-sample convention of a robust covariance.
    """
    n, slopes = columns.shape
    cluster_codes, cluster_count = (
        (None, 0) if clusters is None else consecutive(clusters)
    )
    strata_count = 0
    nested = False
    if strata is None:
        design = np.column_stack([columns, np.ones(n)])
        target = y
    else:
        strata_codes, strata_count = consecutive(strata)
        design = absorb(columns, strata_codes, w)
        target = absorb(y[:, None], strata_codes, w)[:, 0]
        nested = cluster_codes is not None and _nested(
            strata_codes, cluster_codes, strata_count
        )
    fit = least_squares(target, design, w)
    k_factor, k_reference = parameter_counts(slopes, strata_count, nested=nested)
    if vcov == "classical":
        covariance = classical(fit, w, n, k_reference)
        reference = Reference("t", float(n - k_reference))
    else:
        counts = Counts(n, cluster_count, k_factor, k_reference)
        factor, reference = convention(ssc, counts)
        covariance = factor * sandwich(design, fit, w, cluster_codes)
    deficient = fit.rank < design.shape[1]
    if deficient:
        covariance = np.full_like(covariance, np.nan)
    with_intercept = strata is None
    return Estimates(
        beta=fit.beta[:slopes],
        covariance=covariance[:slopes, :slopes],
        intercept=float(fit.beta[slopes]) if with_intercept else float("nan"),
        intercept_variance=(
            float(covariance[slopes, slopes]) if with_intercept else float("nan")
        ),
        reference=reference,
        deficient=deficient,
        n=n,
        clusters=cluster_count,
        strata=strata_count,
    )
