"""HC, signed CGM and explicit CGM2 cluster covariance of influence rows.

If beta_hat-beta_0 is approximately n^{-1} sum_i phi_i, these functions estimate its
covariance, not the covariance multiplied by n.  No independence or moment-validity
assumption is inferred from the supplied labels.  The module provides the robust
estimator, one-way clustering, and the two multiway forms: the signed
inclusion-exclusion of Cameron, Gelbach and Miller (``"cgm"``) and the sum of the
one-way components (``"cgm2"``, the ``"ddg"`` of the package).

These are primitives with their own conventions: ``HC0`` and ``HC1``
with an explicit residual count, and the cluster corrections ``"none"``,
``"component"`` and ``"min"``.  They are not the convention of the package: the
robust and one-way estimators with the factors of the ``"stata"`` preset are
:mod:`powermean.inference.convention`, and multiway clustering, the other presets and
the adjustment of an indefinite signed estimate are in other modules;
nothing here clips, floors or repairs an estimate, and an indefinite or singular one is
returned with its diagnostics and a :class:`CovarianceWarning`, which is not the
package's ``NumericalWarning`` of secondary convergence.

Two points of the implementation:

1. Cluster labels arrive as integer codes prepared by the interface
   layer; this module does not import pandas.  ``clusters`` is a NumPy array of integer
   codes, one-dimensional for one dimension or two-dimensional with one column per
   dimension, or a mapping from names to such one-dimensional arrays; each is aligned
   with the influence rows by position.  :func:`_codes` checks them: an array of
   another dtype, a negative code (the marker of a missing label in factorized codes)
   or a misaligned array is refused, nothing is parsed, coerced or reordered, and the
   codes of each dimension are numbered in order of first appearance, as
   ``pandas.factorize(sort=False)`` numbers labels.
2. :func:`_result` forms the symmetric part of the covariance with
   :func:`powermean.inference.numeric.symmetric_part`, each entry rounded once,
   instead of ``cov/2 + cov.T/2``, which rounds a representable subnormal covariance
   to zero (rows ``[[2**-536], [0]]``: an HC0 variance of ``2**-1074``, reported as
   zero with zero-covariance diagnostics).  Where halving each entry is exact, as for
   every entry of at least ``2**-1021`` in absolute value, the two are identical bit
   for bit; below that bound they can differ on a normal entry too, whose half rounds
   (``0x1.0000000000001p-1022``).

The records are frozen without generated equality (``eq=False``): a generated
comparison could not compare their arrays.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.inference.influence import (
    gmm_influence_map,
    influence_values,
    moment_influence_map,
)
from powermean.inference.numeric import finite_array, immutable_array, symmetric_part
from powermean.model.variance import FloatArray

CodeArray = NDArray[np.intp]
IntegerArray = NDArray[np.integer[Any]]
Clusters = IntegerArray | Mapping[str, IntegerArray]


class CovarianceWarning(UserWarning):
    """A covariance is singular or indefinite; no numerical repair was made."""


@dataclass(frozen=True, eq=False)
class ClusterComponent:
    """One unsigned covariance component in inclusion-exclusion.

    The final covariance sums ``sign * covariance``. ``factor`` is the
    actual small-sample multiplier, and ``n_clusters`` counts intersections.
    """

    dimensions: tuple[str, ...]
    sign: int
    n_clusters: int
    factor: float
    covariance: FloatArray


@dataclass(frozen=True, eq=False)
class CovarianceEstimate:
    """Coefficient covariance and transparent normalization diagnostics.

    ``df_resid`` is supplied by the caller, never guessed from the number
    of moment equations. No t-reference degrees of freedom is selected here.
    Indefinite multiway estimates are returned with a warning, not clipped.
    """

    covariance: FloatArray
    method: str
    nobs: int
    df_resid: int | None
    correction: str
    factor: float | None
    components: tuple[ClusterComponent, ...]
    eigenvalues: FloatArray
    rank: int
    diagnostics: tuple[str, ...]

    @property
    def standard_errors(self) -> FloatArray:
        """The square roots of the variances; a negative variance raises."""
        diagonal = np.diag(self.covariance)
        if np.any(diagonal < 0):
            raise ValueError(
                "negative variance estimates cannot give real standard errors"
            )
        return immutable_array(np.sqrt(diagonal))

    @property
    def is_positive_semidefinite(self) -> bool:
        """Whether no eigenvalue is below minus the tolerance of the rank."""
        return "indefinite_covariance" not in self.diagnostics


def _rows(influence: ArrayLike) -> FloatArray:
    """Influence rows: real, finite, at least two rows and one column."""
    if np.iscomplexobj(influence):
        raise ValueError("influence must be real-valued")
    values = finite_array(influence, "influence")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("influence must have at least two rows and one column")
    return values


def _df(value: int | None, nobs: int, *, required: bool) -> int | None:
    """The residual count of a correction: a positive integer at most ``nobs``."""
    if value is None:
        if required:
            raise ValueError("df_resid must be supplied for this correction")
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("df_resid must be a positive integer")
    if not 0 < value <= nobs:
        raise ValueError("df_resid must lie between 1 and the observation count")
    return int(value)


def _result(
    covariance: FloatArray,
    *,
    method: str,
    nobs: int,
    df_resid: int | None,
    correction: str,
    factor: float | None,
    components: tuple[ClusterComponent, ...] = (),
) -> CovarianceEstimate:
    """The record of a covariance: symmetric part, eigenvalues, rank, diagnostics."""
    cov = finite_array(covariance, "covariance")
    # Roundoff cleanup, not an eigenvalue repair; each entry rounded once.
    cov = symmetric_part(cov)
    eigenvalues = np.linalg.eigvalsh(cov)
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = np.finfo(float).eps * cov.shape[0] * scale
    rank = int(np.count_nonzero(np.abs(eigenvalues) > tolerance))
    diagnostics: list[str] = []
    if np.min(eigenvalues) < -tolerance:
        diagnostics.append("indefinite_covariance")
    if rank < cov.shape[0]:
        diagnostics.append("singular_covariance")
    if not np.any(cov):
        diagnostics.append("zero_covariance")
    if np.any(np.diag(cov) < 0):
        diagnostics.append("negative_diagonal")
    if diagnostics:
        warnings.warn(
            "Covariance diagnostics: "
            + ", ".join(diagnostics)
            + ". No pseudoinverse, zero fallback, or PSD correction was applied.",
            CovarianceWarning,
            stacklevel=3,
        )
    return CovarianceEstimate(
        immutable_array(cov),
        method,
        nobs,
        df_resid,
        correction,
        factor,
        components,
        immutable_array(eigenvalues),
        rank,
        tuple(diagnostics),
    )


def hc_covariance(
    influence: ArrayLike, *, correction: str = "HC0", df_resid: int | None = None
) -> CovarianceEstimate:
    """Estimate sum_i phi_i phi_i'/n^2, optionally multiplied by n/df_resid.

    ``HC0`` makes no finite-sample adjustment. ``HC1`` requires an explicit
    residual degrees-of-freedom count. Influence rows are not automatically
    centered, matching the usual empirical score sandwich.

    Parameters
    ----------
    influence : array_like
        The rows ``phi_i``, at least two, average-normalized.
    correction : {"HC0", "HC1"}
        The factor one, or ``n/df_resid``.
    df_resid : int, optional
        The residual count of ``HC1``, between 1 and ``n``.

    Returns
    -------
    CovarianceEstimate
        The covariance and its diagnostics.
    """
    values = _rows(influence)
    if correction not in {"HC0", "HC1"}:
        raise ValueError("correction must be 'HC0' or 'HC1'")
    nobs = len(values)
    df = _df(df_resid, nobs, required=correction == "HC1")
    factor = 1.0 if correction == "HC0" else nobs / cast("int", df)
    # Scale before cross-products to avoid unnecessary overflow of n^2 terms.
    scaled = values / nobs
    cov = factor * (scaled.T @ scaled)
    return _result(
        cov,
        method=correction,
        nobs=nobs,
        df_resid=df,
        correction=correction,
        factor=float(factor),
    )


def _codes(name: str, labels: object, nobs: int) -> CodeArray:
    """The integer codes of one dimension, numbered in order of first appearance.

    ``labels`` must be a one-dimensional NumPy array of integers (signed or unsigned,
    not boolean) with one nonnegative code per row; a negative code is the marker of a
    missing label.  Codes 0, 1, ... are assigned to the distinct values in the order in
    which they first appear, as ``pandas.factorize(sort=False)`` assigns them to labels;
    the partition is the one supplied, and nothing is coerced or reordered.
    """
    if not isinstance(labels, np.ndarray) or labels.dtype.kind not in ("i", "u"):
        raise ValueError(
            f"cluster dimension {name!r} must be an array of integer codes"
        )
    if labels.ndim != 1 or len(labels) != nobs:
        raise ValueError(f"cluster dimension {name!r} must match the row count")
    if np.any(labels < 0):
        raise ValueError(f"cluster dimension {name!r} contains missing labels")
    unique, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    if len(unique) < 2:
        raise ValueError(
            f"cluster dimension {name!r} must contain at least two clusters"
        )
    order = np.empty(len(unique), dtype=np.intp)
    order[np.argsort(first)] = np.arange(len(unique), dtype=np.intp)
    codes: CodeArray = order[inverse.reshape(-1)]
    return codes


def _cluster_columns(
    clusters: object, *, nobs: int
) -> tuple[tuple[str, ...], CodeArray]:
    """The names of the dimensions and their codes, one column per dimension."""
    columns: list[tuple[str, object]]
    if isinstance(clusters, Mapping):
        columns = [(str(name), labels) for name, labels in clusters.items()]
    elif isinstance(clusters, np.ndarray):
        if clusters.ndim == 1:
            columns = [("cluster_0", clusters)]
        elif clusters.ndim == 2:
            columns = [
                (f"cluster_{j}", clusters[:, j]) for j in range(clusters.shape[1])
            ]
        else:
            raise ValueError(
                "clusters must be row-aligned labels or a mapping of dimensions"
            )
    else:
        raise ValueError(
            "clusters must be an array of integer codes or a mapping of dimensions"
        )
    names = tuple(name for name, _ in columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("at least one uniquely named cluster dimension is required")
    coded = [_codes(name, labels, nobs) for name, labels in columns]
    return names, np.column_stack(coded)


def cluster_covariance(
    influence: ArrayLike,
    clusters: Clusters,
    *,
    correction: str = "none",
    multiway_method: str = "cgm",
    df_resid: int | None = None,
) -> CovarianceEstimate:
    """One-/multiway covariance with an explicit multiway convention.

    ``multiway_method='cgm'`` (the unchanged default) uses exact signed
    inclusion-exclusion. ``'cgm2'`` sums only the one-way components: no
    intersections are subtracted. This sum is positive semidefinite up to
    numerical rounding, but is a different covariance convention, not an
    eigenvalue repair or a universal finite-sample coverage guarantee.
    Result methods are respectively ``'cluster'`` and ``'cluster_cgm2'``.

    ``clusters`` is an n-row integer-code vector or matrix, or a mapping from
    names to aligned integer-code vectors. Codes are aligned with
    the influence rows by position; nothing is reordered.

    For each nonempty subset A of dimensions, aggregate influence rows by
    intersection clusters, form S_A=sum_g phi_g phi_g', and add
    (-1)^(|A|+1) factor_A S_A/n^2. Corrections:

    * ``none``: factor_A=1;
    * ``component``: G_A/(G_A-1)*(n-1)/df_resid;
    * ``min``: min_d G_d/(min_d G_d-1)*(n-1)/df_resid for every A.

    The latter two require explicit df_resid. These optional corrections
    are conventions, not exact finite-sample unbiasedness or t-distribution
    claims. Cluster-robust validity requires adequate independent clusters
    in every relevant dimension. The work grows as 2**D-1 for D dimensions.

    Parameters
    ----------
    influence : array_like
        The rows ``phi_i``, at least two, average-normalized.
    clusters : array or mapping
        The integer codes of the dimensions, aligned with the rows.
    correction : {"none", "component", "min"}
        The convention of the factors.
    multiway_method : {"cgm", "cgm2"}
        Signed inclusion-exclusion, or the sum of the one-way components.
    df_resid : int, optional
        The residual count of the corrections ``"component"`` and ``"min"``.

    Returns
    -------
    CovarianceEstimate
        The covariance, its components and its diagnostics.
    """
    values = _rows(influence)
    if not isinstance(multiway_method, str) or multiway_method not in {"cgm", "cgm2"}:
        raise ValueError("multiway_method must be 'cgm' or 'cgm2'")
    if correction not in {"none", "component", "min"}:
        raise ValueError("cluster correction must be 'none', 'component', or 'min'")
    nobs = len(values)
    df = _df(df_resid, nobs, required=correction != "none")
    names, codes = _cluster_columns(clusters, nobs=nobs)
    group_counts = tuple(len(np.unique(codes[:, j])) for j in range(codes.shape[1]))
    min_groups = min(group_counts)
    # df is an integer whenever a correction is requested: _df refused None.
    common_factor = (
        min_groups / (min_groups - 1) * (nobs - 1) / cast("int", df)
        if correction == "min"
        else None
    )
    components: list[ClusterComponent] = []
    covariance = np.zeros((values.shape[1], values.shape[1]))
    scaled = values / nobs
    maximum_size = len(names) if multiway_method == "cgm" else 1
    for size in range(1, maximum_size + 1):
        sign = 1 if size % 2 else -1
        for subset in combinations(range(len(names)), size):
            if len(subset) == 1:
                # _cluster_columns already numbers each owned column densely in
                # first-appearance order. Its sorted unique values are 0..G-1,
                # so the original inverse is this column itself. Reuse the G
                # already obtained above; no row sort or additional scan.
                inverse = codes[:, subset[0]]
                groups = group_counts[subset[0]]
            else:
                unique, inverse = np.unique(
                    codes[:, subset], axis=0, return_inverse=True
                )
                groups = len(unique)
            sums = np.zeros((groups, values.shape[1]))
            np.add.at(sums, inverse, scaled)
            if correction == "component":
                factor = groups / (groups - 1) * (nobs - 1) / cast("int", df)
            elif correction == "min":
                factor = cast("float", common_factor)
            else:
                factor = 1.0
            component = float(factor) * (sums.T @ sums)
            covariance += sign * component
            components.append(
                ClusterComponent(
                    tuple(names[j] for j in subset),
                    sign,
                    groups,
                    float(factor),
                    immutable_array(component),
                )
            )
    return _result(
        covariance,
        method="cluster" if multiway_method == "cgm" else "cluster_cgm2",
        nobs=nobs,
        df_resid=df,
        correction=correction,
        factor=1.0 if correction == "none" else common_factor,
        components=tuple(components),
    )


def sandwich_covariance(
    contributions: ArrayLike,
    jacobian: ArrayLike,
    *,
    weight: ArrayLike | None = None,
    covariance_type: str = "HC0",
    clusters: Clusters | None = None,
    df_resid: int | None = None,
    cluster_correction: str = "none",
) -> CovarianceEstimate:
    """Build a just-identified or frozen-weight GMM sandwich from score rows.

    jacobian is the actual derivative of the AVERAGE moment. Contributions
    are individual, unscaled moment rows. Supplying weight selects the GMM
    map, including in exact identification. For a projected-i2SLS equation,
    explicitly compose projected_influence_map with influence_values and one
    of the covariance functions instead.

    Parameters
    ----------
    contributions : array_like
        The individual moment rows, unscaled.
    jacobian : array_like
        The derivative of the average moment.
    weight : array_like, optional
        The frozen GMM weight; the moment map without it.
    covariance_type : {"HC0", "HC1", "cluster"}
        The estimator.
    clusters : array or mapping, optional
        The integer codes of the cluster dimensions, aligned with the rows.
    df_resid : int, optional
        The residual count of ``HC1`` and of the cluster corrections.
    cluster_correction : {"none", "component", "min"}
        The cluster correction.

    Returns
    -------
    CovarianceEstimate
        The covariance and its diagnostics.
    """
    if covariance_type not in {"HC0", "HC1", "cluster"}:
        raise ValueError("covariance_type must be 'HC0', 'HC1', or 'cluster'")
    if covariance_type != "cluster" and (
        clusters is not None or cluster_correction != "none"
    ):
        raise ValueError("cluster labels/corrections require covariance_type='cluster'")
    linear_map = (
        moment_influence_map(jacobian)
        if weight is None
        else gmm_influence_map(jacobian, weight)
    )
    values = influence_values(contributions, linear_map)
    if covariance_type == "cluster":
        if clusters is None:
            raise ValueError("cluster covariance requires explicit cluster labels")
        return cluster_covariance(
            values, clusters, correction=cluster_correction, df_resid=df_resid
        )
    return hc_covariance(values, correction=covariance_type, df_resid=df_resid)
