"""Robust and one-way cluster covariances with the factors of the ``"stata"`` preset.

This module provides the robust estimator and one cluster dimension, with the factors
of the ``"stata"`` convention and the normal reference.  Multiway clustering, the
other presets, the expected bread and the eigenvalue adjustment of an indefinite
estimate are not in this module; the primitives of :mod:`powermean.inference.vcov`
keep their own conventions, which are not these.

From an :class:`~powermean.inference.contributions.Influence` record, with the map
``A = C^-1`` and the contributions ``s_i`` of its ``m`` rows of positive weight:

    robust     V = N/(N - 1) sum_i u_i u_i'        u_i = A s_i, or A s_i / sqrt(f_i)
    one-way    V = G/(G - 1) sum_g U_g U_g'        U_g = sum_{i in g} A s_i
    reference  normal

Weights.  Analytic and probability weights multiply the
contributions, ``s_i = a_i r_i x_i``, so the robust meat is
``sum_i a_i^2 r_i^2 x_i x_i'``.  Frequency weights are replicated rows: each replicate
is a sampling unit of the robust meat, ``sum_i f_i r_i^2 x_i x_i'``, so the
contributions are divided by ``sqrt(f_i)`` for it alone; a cluster total keeps
``sum_{i in g} f_i r_i x_i``, the total of the replicates, before its crossproduct.

Normalization.  The primitives expect influences of an average,
``beta_hat - beta_0 = n^-1 sum_i phi_i``, and divide by ``n^2``.  The record holds the
sum normalization, ``gamma_hat - gamma_0 = sum_i u_i``, so the primitives receive the
source-average influences ``phi_i = m u_i`` of the ``m`` stored rows, whose divisor
``m^2`` cancels: ``(phi_i/m)(phi_i/m)'`` is ``u_i u_i'`` up to the rounding of the
product and the quotient.  The primitive needs two rows, and one compressed frequency
row with a count above one is a valid robust sample; its influence ``u_1`` then enters
as the two rows ``2 u_1`` and zero, whose divisor two cancels exactly, doubling and
halving being exact in binary, and whose zero row adds nothing: the estimate is
``u_1 u_1'`` bit for bit.  ``N`` is never a divisor of the stored rows.  An influence
``m u_i`` beyond the doubles raises ``FloatingPointError``; nothing is rescaled.

Factors.  Robust: ``N/(N - 1)``, ``N`` the count of the record
(the total frequency count, else the number of rows of positive weight), never
``N/(N - K)``; ``N < 2`` is refused.  One-way: ``G/(G - 1)``, ``G`` the clusters of
the rows of positive weight.  Each estimate is the primitive's ``HC0`` or uncorrected
one-way covariance, its symmetric part formed, then multiplied once by
the factor, so that the covariance stays exactly symmetric.  The primitive's estimate
is kept (``Covariance.estimate``): its unscaled covariance, components, eigenvalues and
diagnostics, which a positive factor does not change, and its
:class:`~powermean.inference.vcov.CovarianceWarning`, which is not a
``NumericalWarning``.  A singular or zero covariance, as at an exact saturated root, is
returned as it is with its diagnostics; nothing is floored or replaced.  A product
beyond the doubles raises ``FloatingPointError``.

Cluster codes.  :class:`ClusterCodes` holds one named dimension with one integer code
per input row of the system (``units.nobs`` codes, rows of zero weight included, in
their order).  Only the codes of the rows of positive weight are read, at their
positions ``units.identifiers``: a row of zero weight enters no alignment, cluster or
count.  They are checked before any arithmetic, as the primitive checks them: a
NumPy array of integers, one per input row; a negative code on a row of positive
weight is a missing label and is refused, as are fewer than two clusters.  Nothing is
coerced or reordered.

Reported scale.  :func:`reported_covariance` maps the covariance ``V`` of ``gamma`` to
``M V M'`` for a matrix ``M`` that the caller supplies, the identity by default (the
order layer supplies its own), keeps its symmetric part, rounded once
per entry, and forms the standard errors ``sqrt(diag(M V M'))``.  A zero variance gives
a zero standard error, which the certificate flags as degenerate inference; a
negative variance gives a missing one (NaN), never zero, which passes no gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.inference.contributions import Influence
from powermean.inference.influence import influence_values
from powermean.inference.numeric import finite_array, symmetric_part
from powermean.inference.vcov import (
    CovarianceEstimate,
    cluster_covariance,
    hc_covariance,
)
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result
from powermean.systems.joint import SamplingUnits

IntegerArray = NDArray[np.integer[Any]]
Kind = Literal["robust", "cluster", "hc"]


@dataclass(frozen=True, eq=False)
class ClusterCodes:
    """One cluster dimension: its name and an integer code per input row.

    Parameters
    ----------
    name : str
        The name of the dimension, nonempty.
    codes : array
        A one-dimensional NumPy array of integers, signed or unsigned, one per input
        row of the system, rows of zero weight included, in their order.

    Notes
    -----
    Stored as an owned read-only copy of the same dtype.  Another type, dtype or
    shape raises ``ValueError``; the values are checked on the rows of positive weight
    when a covariance reads them.
    """

    name: str
    codes: IntegerArray

    def __post_init__(self) -> None:
        """Check the name and the array of codes, and store an owned copy.

        The values are read as ``object``: this is the run-time check of what the
        fields declare.
        """
        name: object = self.name
        if not isinstance(name, str) or not name:
            raise ValueError("a cluster dimension needs a nonempty name")
        codes: object = self.codes
        if (
            not isinstance(codes, np.ndarray)
            or codes.dtype.kind not in ("i", "u")
            or codes.ndim != 1
        ):
            raise ValueError("cluster codes must be a one-dimensional integer array")
        checked: IntegerArray = codes
        stored: IntegerArray = np.frombuffer(checked.tobytes(), dtype=checked.dtype)
        object.__setattr__(self, "codes", stored)


@dataclass(frozen=True, eq=False)
class Covariance:
    """A covariance of ``gamma`` under the ``"stata"`` convention.

    Attributes
    ----------
    covariance : array
        ``V``, the factor times the primitive's covariance, exactly symmetric.
    kind : {"robust", "cluster"}
        The robust estimator or one cluster dimension.
    factor : float
        ``N/(N - 1)`` or ``G/(G - 1)``.
    count : int
        ``N``, the observations of the record.
    clusters : int or None
        ``G`` for one cluster dimension, ``None`` for the robust estimator.
    estimate : CovarianceEstimate
        The primitive's estimate, without the factor: its covariance, components,
        eigenvalues, rank and diagnostics.
    reference : {"normal"}
        The reference distribution of the convention.
    """

    covariance: FloatArray
    kind: Kind
    factor: float
    count: int
    clusters: int | None
    estimate: CovarianceEstimate
    reference: Literal["normal", "t"] = "normal"


def _check_influence(influence: object) -> None:
    """``influence`` is an :class:`~powermean.inference.contributions.Influence`."""
    if type(influence) is not Influence:
        raise TypeError("influence must be an Influence record")


def _source_average(values: FloatArray) -> FloatArray:
    """``phi_i = m u_i`` of the ``m`` rows: the normalization of the primitives."""
    rows = values.shape[0]
    with np.errstate(over="ignore", invalid="ignore"):
        scaled: FloatArray = rows * values
    return _finite_result(scaled, "source-average influence")


def _scaled(
    estimate: CovarianceEstimate, factor: float, influence: Influence, kind: Kind
) -> Covariance:
    """The primitive's estimate times the factor of the convention, once."""
    with np.errstate(over="ignore", invalid="ignore"):
        covariance: FloatArray = factor * estimate.covariance
    clusters = estimate.components[0].n_clusters if kind == "cluster" else None
    return Covariance(
        covariance=_immutable(_finite_result(covariance, "covariance")),
        kind=kind,
        factor=factor,
        count=influence.count,
        clusters=clusters,
        estimate=estimate,
    )


def robust_factor(count: int) -> float:
    """``N/(N - 1)``, the robust factor of the ``"stata"`` preset; ``N >= 2``."""
    if count < 2:
        raise ValueError("the robust factor N/(N - 1) needs at least two observations")
    return count / (count - 1)


def _replicated_rows(contributions: FloatArray, weights: FloatArray) -> FloatArray:
    """``s_i / sqrt(f_i)``: the rows whose crossproducts are ``f_i r_i^2 x_i x_i'``."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rows: FloatArray = contributions / np.sqrt(weights)[:, None]
    return _finite_result(rows, "contribution matrix")


def robust_covariance(influence: Influence) -> Covariance:
    """The robust covariance of a candidate, with the factor ``N/(N - 1)``.

    Parameters
    ----------
    influence : Influence
        The record of the candidate.

    Returns
    -------
    Covariance
        ``N/(N - 1) sum_i u_i u_i'`` and the primitive's ``HC0`` estimate.

    Raises
    ------
    TypeError
        ``influence`` is not an ``Influence`` record.
    ValueError
        ``N < 2``.
    FloatingPointError
        A value is not representable in float64.
    """
    _check_influence(influence)
    factor = robust_factor(influence.count)
    rows = influence.contributions
    if influence.frequency:
        rows = _replicated_rows(rows, influence.weights)
    values = influence_values(rows, influence.influence_map)
    if values.shape[0] == 1:
        # One stored row: the rows 2 u_1 and zero, whose divisor two cancels exactly.
        values = np.vstack([values, np.zeros_like(values)])
    estimate = hc_covariance(_source_average(values))
    return _scaled(estimate, factor, influence, "robust")


def _active_codes(cluster: ClusterCodes, units: SamplingUnits) -> IntegerArray:
    """The codes of the rows of positive weight, checked before any arithmetic."""
    name = cluster.name
    if cluster.codes.shape != (units.nobs,):
        raise ValueError(f"cluster dimension {name!r} must have a code per input row")
    codes: IntegerArray = cluster.codes[units.identifiers]
    if np.any(codes < 0):
        raise ValueError(f"cluster dimension {name!r} contains missing labels")
    if np.unique(codes).size < 2:
        raise ValueError(
            f"cluster dimension {name!r} must contain at least two clusters"
        )
    return codes


def one_way_covariance(influence: Influence, cluster: ClusterCodes) -> Covariance:
    """The one-way cluster covariance of a candidate, with the factor ``G/(G - 1)``.

    Parameters
    ----------
    influence : Influence
        The record of the candidate.
    cluster : ClusterCodes
        The dimension, one code per input row of the system.

    Returns
    -------
    Covariance
        ``G/(G - 1) sum_g U_g U_g'`` and the primitive's uncorrected estimate.

    Raises
    ------
    TypeError
        ``influence`` is not an ``Influence`` record or ``cluster`` not
        ``ClusterCodes``.
    ValueError
        Codes not aligned with the input rows, a missing label or fewer than two
        clusters among the rows of positive weight.
    FloatingPointError
        A value is not representable in float64.
    """
    _check_influence(influence)
    if type(cluster) is not ClusterCodes:
        raise TypeError("cluster must be ClusterCodes")
    codes = _active_codes(cluster, influence.units)
    values = influence_values(influence.contributions, influence.influence_map)
    estimate = cluster_covariance(_source_average(values), {cluster.name: codes})
    groups = estimate.components[0].n_clusters
    return _scaled(estimate, groups / (groups - 1), influence, "cluster")


@dataclass(frozen=True, eq=False)
class ReportedCovariance:
    """A covariance on the reported scale: ``M``, ``M V M'`` and the standard errors.

    ``standard_errors`` holds ``sqrt(diag(M V M'))``, NaN where a variance is negative;
    a zero variance gives a zero standard error.  Owned read-only arrays.
    """

    transform: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray


def _transform(transform: ArrayLike, size: int) -> FloatArray:
    """``M``: a finite real matrix with a row and one column per coefficient."""
    matrix = finite_array(transform, "transform")
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[1] != size:
        raise ValueError(
            "transform must be a matrix with a row and one column per coefficient"
        )
    return matrix


def reported_covariance(
    covariance: Covariance, transform: ArrayLike | None = None
) -> ReportedCovariance:
    """``M V M'`` and its standard errors, for a map ``M`` of the coefficients.

    Parameters
    ----------
    covariance : Covariance
        The covariance ``V`` of ``gamma``.
    transform : array_like, optional
        ``M``, one row per reported coefficient and one column per coefficient of
        ``gamma``; the identity if omitted.

    Returns
    -------
    ReportedCovariance
        ``M``, the symmetric part of ``M V M'`` and the standard errors.

    Raises
    ------
    TypeError
        ``covariance`` is not a ``Covariance``.
    ValueError
        ``M`` is not a finite real matrix of that shape.
    FloatingPointError
        ``M V M'`` is not representable in float64.
    """
    if type(covariance) is not Covariance:
        raise TypeError("covariance must be a Covariance")
    size = covariance.covariance.shape[0]
    matrix = np.eye(size) if transform is None else _transform(transform, size)
    with np.errstate(over="ignore", invalid="ignore"):
        product: FloatArray = matrix @ covariance.covariance @ matrix.T
    reported = symmetric_part(_finite_result(product, "reported covariance"))
    diagonal = np.diag(reported)
    errors = np.full(diagonal.shape, np.nan)
    valid = diagonal >= 0
    errors[valid] = np.sqrt(diagonal[valid])
    return ReportedCovariance(
        transform=_immutable(matrix),
        covariance=_immutable(reported),
        standard_errors=_immutable(errors),
    )
