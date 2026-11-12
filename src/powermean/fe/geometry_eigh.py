"""Same-construction exact-zero Gram deflation, without a rank-policy change."""

from __future__ import annotations

import numpy as np

from powermean.model.variance import FloatArray


def witnessed_eigh(
    matrix: FloatArray, exact_zero_columns: tuple[int, ...]
) -> tuple[FloatArray, FloatArray]:
    """Lift the same complementary eigh and canonical witnessed-zero axes.

    Only the fresh operator stream supplies these indices. No Gram diagonal,
    tolerance, stored annihilation flag or previous result discovers an axis.
    Full sorted eigenvalues still enter the unchanged spectral classification.
    """
    if not exact_zero_columns:
        return np.linalg.eigh(matrix)
    p = matrix.shape[0]
    active_mask = np.ones(p, dtype=bool)
    zero_axes = np.asarray(exact_zero_columns, dtype=np.intp)
    active_mask[zero_axes] = False
    active = np.flatnonzero(active_mask)
    principal = matrix[np.ix_(active, active)]
    if active.size:
        values, vectors = np.linalg.eigh(principal)
    else:
        values = np.zeros(0, dtype=np.float64)
        vectors = np.zeros((0, 0), dtype=np.float64)
    del principal
    # Append canonical zeros after any exact zeros of the complementary eigh.
    # Direct placement gives the ascending union without a second p-square copy.
    insertion = int(np.searchsorted(values, 0.0, side="right"))
    columns = np.arange(active.size, dtype=np.intp)
    columns[insertion:] += zero_axes.size
    full_values = np.zeros(p, dtype=np.float64)
    full_vectors = np.zeros((p, p), dtype=np.float64)
    full_values[columns] = values
    full_vectors[np.ix_(active, columns)] = vectors
    canonical = np.arange(insertion, insertion + zero_axes.size, dtype=np.intp)
    full_vectors[zero_axes, canonical] = 1.0
    return full_values, full_vectors
