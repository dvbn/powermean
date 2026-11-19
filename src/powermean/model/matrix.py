"""Numerical operations and rank diagnostics for real model matrices."""

import numpy as np

from powermean.model.variance import FloatArray


class MomentJacobianRankError(np.linalg.LinAlgError):
    """The just-identified moment Jacobian failed the inference map's rank test."""


def symmetric_average(matrix: FloatArray) -> FloatArray:
    """Average paired entries of an already checked square array."""
    small = np.abs(matrix) <= np.finfo(float).max / 2
    add_first = small & small.T
    result = np.empty_like(matrix)
    # Underflow of a genuinely unrepresentable average is ordinary rounding,
    # unlike halving two equal subnormals before forming their sum.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result[add_first] = (matrix[add_first] + matrix.T[add_first]) / 2
        result[~add_first] = matrix[~add_first] / 2 + matrix.T[~add_first] / 2
    return result
