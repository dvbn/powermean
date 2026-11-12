"""Linear influence maps for averaged moments, with an explicit sign convention.

``jacobian`` always means ``dm_n/dbeta'``, not its negative.  No numerical
preconditioner is used as a substitute for this derivative.  A map is minus the
inverse of the actual Jacobian, transposes kept where it is nonsymmetric.

    moment map      -J^-1             a just-identified root of m_n(beta) = 0
    GMM map         -(J'WJ)^-1 J'W    a frozen-weight GMM system; a primitive of
                                      the instrumental systems, not an estimator
    projected map   -(CJ)^-1 C        a projected equation C m_n(beta) = 0
    influence       psi_i A'          score rows mapped to coefficient rows

Normalization.  For a moment that is an average, ``m_n = P_n psi_i``, the
expansion ``beta_hat - beta_0 = P_n[-J^-1 psi_i]`` gives the average-normalized
influences that :mod:`powermean.inference.vcov` expects.  The functions themselves
only invert and multiply: given the Jacobian ``-C`` of the sum ``S = sum_i s_i`` of the
joint system (:mod:`powermean.systems.joint`), :func:`moment_influence_map` returns the
sum map ``C^-1``, and ``u_i = C^-1 s_i`` sums to the first-order change of ``gamma``.
:mod:`powermean.inference.convention` states how ``u_i`` reaches the average
normalization; nothing here divides by a number of rows.

Refusals: a matrix that is complex, not finite, empty or not
two-dimensional, of the wrong shape or not of full column rank raises ``ValueError``
or ``numpy.linalg.LinAlgError``, without a pseudoinverse; a map that is not
representable in float64 raises ``ValueError``.  Results are owned read-only arrays.

The helpers come from :mod:`powermean.inference.numeric`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from powermean.inference.numeric import finite_array, immutable_array
from powermean.model.variance import FloatArray


def _matrix(value: ArrayLike, name: str) -> FloatArray:
    """A nonempty two-dimensional real matrix of finite doubles."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    array = finite_array(value, name)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    return array


def _full_column_rank(matrix: FloatArray, name: str) -> None:
    """Refuse a matrix whose numerical rank is below its number of columns."""
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise np.linalg.LinAlgError(f"{name} must have full column rank")


def moment_influence_map(jacobian: ArrayLike) -> FloatArray:
    """Return -J^{-1} for a just-identified moment root.

    For m_n=P_n psi_i and J=dm_n/dbeta', the first-order expansion is
    beta_hat-beta_0 = P_n[-J^{-1} psi_i] plus a negligible remainder.
    J may be nonsymmetric. Singular matrices raise, without a pseudoinverse.

    Parameters
    ----------
    jacobian : array_like
        The square Jacobian ``J`` of the moment, finite and real.

    Returns
    -------
    array
        ``-J^-1``, owned and read-only.
    """
    jac = _matrix(jacobian, "jacobian")
    if jac.shape[0] != jac.shape[1]:
        raise ValueError("just-identified inference requires a square jacobian")
    _full_column_rank(jac, "jacobian")
    result = -np.linalg.solve(jac, np.eye(jac.shape[0]))
    return immutable_array(finite_array(result, "moment influence map"))


def gmm_influence_map(jacobian: ArrayLike, weight: ArrayLike) -> FloatArray:
    """Return -(J'WJ)^{-1}J'W for an identified frozen-weight GMM system.

    J has one row per moment and one column per coefficient. W must be
    symmetric positive semidefinite with J'WJ positive definite. This allows
    the empirical null space of a Tikhonov operator without adding a ridge.
    This first-order map presumes valid moments,
    a consistent regular endpoint, and a consistently estimated frozen W.
    It is not the inverse of the exact finite-sample criterion Hessian.

    Parameters
    ----------
    jacobian : array_like
        ``J``, one row per moment and one column per coefficient.
    weight : array_like
        ``W``, square with one row per moment.

    Returns
    -------
    array
        ``-(J'WJ)^-1 J'W``, owned and read-only.
    """
    jac, w = _matrix(jacobian, "jacobian"), _matrix(weight, "weight")
    if jac.shape[0] < jac.shape[1]:
        raise ValueError("GMM requires at least as many moments as coefficients")
    if w.shape != (jac.shape[0], jac.shape[0]):
        raise ValueError("weight must be square with one row per moment")
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        raise np.linalg.LinAlgError("A zero weight cannot identify coefficients")
    w = w / scale  # a common positive weight scale leaves the influence map unchanged
    tolerance = 100 * np.finfo(float).eps * len(w)
    if not np.allclose(w, w.T, rtol=0, atol=tolerance):
        raise ValueError("weight must be symmetric")
    # Remove only roundoff-level antisymmetry after explicitly validating it.
    w = w / 2 + w.T / 2
    if np.linalg.eigvalsh(w)[0] < -tolerance:
        raise np.linalg.LinAlgError("weight must be positive semidefinite")
    column_scale = np.max(np.abs(jac), axis=0)
    if np.any(column_scale == 0):
        raise np.linalg.LinAlgError("jacobian must have full column rank")
    scaled_jac = jac / column_scale
    _full_column_rank(scaled_jac, "jacobian")
    normal = scaled_jac.T @ w @ scaled_jac
    _full_column_rank(normal, "GMM information matrix")
    np.linalg.cholesky((normal + normal.T) / 2)
    result = -np.linalg.solve(normal, scaled_jac.T @ w) / column_scale[:, None]
    return immutable_array(finite_array(result, "GMM influence map"))


def projected_influence_map(jacobian: ArrayLike, projection: ArrayLike) -> FloatArray:
    """Return -(CJ)^{-1}C for a projected equation C m_n(beta)=0.

    This is distinct from overidentified nonlinear GMM. For fixed-design
    i2SLS, C can be (X'Z/n)(Z'Z/n)^{-1}. Inference requires the actual moment
    derivative J; the projected regressor Gram is not generally CJ under IV.
    An estimated C has no first-order term when the complete population
    moment is zero and the appropriate regularity conditions hold.

    Parameters
    ----------
    jacobian : array_like
        ``J``, one row per moment and one column per coefficient.
    projection : array_like
        ``C``, one row per coefficient and one column per moment.

    Returns
    -------
    array
        ``-(CJ)^-1 C``, owned and read-only.
    """
    jac = _matrix(jacobian, "jacobian")
    transform = _matrix(projection, "projection")
    if transform.shape != (jac.shape[1], jac.shape[0]):
        raise ValueError("projection must have coefficient-by-moment shape")
    bread = transform @ jac
    _full_column_rank(bread, "projected moment jacobian")
    result = -np.linalg.solve(bread, transform)
    return immutable_array(finite_array(result, "projected influence map"))


def influence_values(contributions: ArrayLike, influence_map: ArrayLike) -> FloatArray:
    """Map n-by-moment score rows into n-by-coefficient influence rows.

    Contributions are unscaled individual scores, not scores divided by n.
    Group scores may instead be supplied as rows when the derivative uses
    the same group-average normalization.

    Parameters
    ----------
    contributions : array_like
        The score rows, one row per unit and one column per moment.
    influence_map : array_like
        The map ``A``, one row per coefficient and one column per moment.

    Returns
    -------
    array
        The rows ``psi_i A'``, owned and read-only.
    """
    scores = _matrix(contributions, "contributions")
    linear_map = _matrix(influence_map, "influence_map")
    if scores.shape[1] != linear_map.shape[1]:
        raise ValueError("score columns must match influence-map columns")
    result = scores @ linear_map.T
    return immutable_array(finite_array(result, "influence values"))
