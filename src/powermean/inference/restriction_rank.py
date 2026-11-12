"""Numerical rank, consistency, range and generalized inverse of restrictions.

Joint tests under a rank-aware policy. Redundant
restrictions count once; an untestable discrepancy component has no p-value;
empty or inconsistent requests refuse. Nothing here fits, repairs a covariance
or changes the exact-zero, negative-variance or withholding rules.

Rank rule. The numerical rank of a matrix with singular values ``s`` counts the
values above ``max(s) * max(shape) * eps``, the documented default of
``numpy.linalg.matrix_rank`` with the float64 machine epsilon. A symmetric
restriction covariance ``S = R V R'`` has singular values ``|lambda_i|``, so the
same cutoff selects its retained eigen-directions. Those directions serve both the
generalized inverse ``Q = sum_i (u_i'd)^2 / lambda_i`` and the effective numerator
degrees of freedom ``r``; no separate ``pinv`` default, floor or clipping is used.
A retained negative eigenvalue is a genuinely indefinite covariance, left as it is.
A discarded direction of either sign is numerically zero under this rule.
The caller separately rejects supplied negative marginal variances at every scale;
discarding a numerical direction cannot turn a negative variance into a zero SE.

Consistency uses the left null space of R, with the same numerical rank rule.
The normalized right-hand side must have no left-null component above
``max(R.shape) * eps``. Unlike comparing independently scaled augmented ranks,
a large q cannot hide inconsistent dependent rows by raising R's rank cutoff.

Range uses a machine-precision bound ``m * eps * (max|d| + scale)``, where
``scale = max_j (sum_k |r_jk beta_k| + |q_j|)`` accounts for forming d.
There is no condition-number multiplier that could admit a material untestable
component. R's stable null basis is eligible only for GENUINE algebraic row
dependencies: its exact rational rank must equal the retained rank of S, its
numerical null dimension must agree, and S must have no resolved leakage from
that structural space. Equal numerical ranks alone are not a certificate: a
small but nonzero coefficient of R can be amplified by V into a retained S
direction. Exact rank is used only for this structural certificate, never for
the inverse or degrees of freedom. Without it, use the discarded S vectors.
Neither guard changes d, S, the inverse directions, nor its effective df.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np

from powermean.model.variance import FloatArray

EPS: Final[float] = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Retained eigen-directions of a symmetric restriction covariance.

    ``values`` and ``vectors`` hold the retained eigenvalues and unit eigenvectors
    (columns); ``discarded`` holds the unit eigenvectors below the cutoff. The
    cutoff is ``max |lambda| * m * eps``. ``indefinite`` records a retained
    negative eigenvalue, which is reported as missing inference, never repaired.
    """

    values: FloatArray
    vectors: FloatArray
    discarded: FloatArray
    cutoff: float
    indefinite: bool

    @property
    def rank(self) -> int:
        """The number of retained directions, the effective restriction count."""
        return int(self.values.shape[0])


def rank_cutoff(singular_values: FloatArray, shape: tuple[int, ...]) -> float:
    """``max(s) * max(shape) * eps``, the documented ``matrix_rank`` default.

    Absolute values are taken, so the eigenvalues of a symmetric matrix, whose
    magnitudes are its singular values, may be passed directly.
    """
    largest = float(np.max(np.abs(singular_values), initial=0.0))
    return largest * (max(shape) * EPS)


def numerical_rank(matrix: FloatArray) -> int:
    """The number of singular values above the rank cutoff of ``matrix``."""
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return 0
    singular = np.linalg.svd(matrix / scale, compute_uv=False)
    return int(np.count_nonzero(singular > rank_cutoff(singular, matrix.shape)))


def consistent(rows: FloatArray, rhs: FloatArray) -> bool:
    """A normalized q must lie in R's numerical column space, not mask its rank."""
    scale = float(np.max(np.abs(rhs), initial=0.0))
    if scale == 0.0:
        return True
    components = restriction_null(rows).T @ (rhs / scale)
    return bool(np.max(np.abs(components), initial=0.0) <= max(rows.shape) * EPS)


def restriction_null(rows: FloatArray) -> FloatArray:
    """Left null vectors of R at its documented, uniformly scale-invariant cutoff."""
    scale = float(np.max(np.abs(rows), initial=0.0))
    if scale == 0.0:
        return np.eye(rows.shape[0], dtype=np.float64)
    vectors, singular, _ = np.linalg.svd(rows / scale, full_matrices=True)
    rank = int(np.count_nonzero(singular > rank_cutoff(singular, rows.shape)))
    return vectors[:, rank:]


def spectrum(covariance: FloatArray) -> Spectrum:
    """Eigen-directions of a finite nonzero symmetric ``S`` split at the cutoff."""
    values, vectors = np.linalg.eigh(covariance)
    cutoff = rank_cutoff(values, covariance.shape)
    retained = np.abs(values) > cutoff
    return Spectrum(
        values[retained],
        vectors[:, retained],
        vectors[:, ~retained],
        cutoff,
        bool(np.any(values[retained] < 0.0)),
    )


def range_tolerance(difference: FloatArray, scale: float) -> float:
    """Machine-precision projection/formation error, never a kappa-sized waiver."""
    norm = float(np.max(np.abs(difference), initial=0.0))
    multiplier = difference.shape[0] * EPS
    return float(multiplier * norm + multiplier * scale)


def outside_range(
    spectrum: Spectrum,
    difference: FloatArray,
    scale: float,
    covariance: FloatArray,
    row_null: FloatArray,
    *,
    rows: FloatArray,
) -> bool:
    """Whether ``d`` has a numerically nonzero component in a discarded direction."""
    if spectrum.discarded.shape[1] == 0:
        return False
    null = spectrum.discarded
    if row_null.shape[1] == null.shape[1] and exact_rank_is(rows, spectrum.rank):
        largest = float(np.max(np.abs(covariance)))
        leak = row_null.T @ (covariance / largest)
        if np.max(np.abs(leak), initial=0.0) <= difference.size * EPS:
            null = row_null
    components = null.T @ difference
    tolerance = range_tolerance(difference, scale)
    return bool(np.max(np.abs(components)) > tolerance)


def exact_rank_is(rows: FloatArray, expected: int) -> bool:
    """Certify algebraic row rank of finite supplied doubles, without a cutoff.

    Fraction elimination establishes genuine dependencies of R, not a different
    numerical rank policy. This is used only when choosing a structural range
    basis; S's unchanged spectrum alone selects its inverse and effective df.
    Stop as soon as the proposed rank is disproved. No exact rank, factorization
    or success flag is retained across reporting calls.
    """
    basis: dict[int, list[Fraction]] = {}
    for values in rows:
        row = [Fraction(float(value)) for value in values]
        for pivot, previous in basis.items():
            multiplier = row[pivot]
            if multiplier:
                row = [a - multiplier * b for a, b in zip(row, previous, strict=True)]
        for pivot, value in enumerate(row):
            if value:
                basis[pivot] = [entry / value for entry in row]
                if len(basis) > expected:
                    return False
                break
    return len(basis) == expected


def quadratic_form(spectrum: Spectrum, difference: FloatArray) -> float:
    """``sum_i (u_i'd)^2 / lambda_i`` over the retained directions only."""
    components = spectrum.vectors.T @ difference
    scaled = components / np.sqrt(spectrum.values)
    return float(np.sum(scaled * scaled))


def formation_scale(rows: FloatArray, beta: FloatArray, rhs: FloatArray) -> float:
    """``max_j (sum_k |r_jk beta_k| + |q_j|)``, the rounding scale of ``d``."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.max(np.abs(rows) @ np.abs(beta) + np.abs(rhs)))


def negative_variance(covariance: FloatArray) -> bool:
    """A supplied negative marginal variance stays missing, even below rank cutoff."""
    return bool(np.any(np.diag(covariance) < 0.0))
