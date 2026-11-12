"""Current observed row influence from the certificate transpose slope map."""

from __future__ import annotations

import numpy as np

from powermean.fe.certificate_basis import embed, lift
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result
from powermean.solve.absorbed_certificate_scale import (
    FEScaleInput,
)


def influence_rows(scale_input: FEScaleInput) -> FloatArray:
    """``u_i = b_i (B_i W)`` for every final row; ``FloatingPointError`` on range."""
    geometry, slope_map = scale_input.geometry, scale_input.slope_map
    k, basis = geometry.n_slopes, geometry.basis
    nuisance = lift(basis, embed(basis, slope_map[k:]))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        rows: FloatArray = geometry.design @ slope_map[:k] + nuisance
        found: FloatArray = geometry.rows.b[:, None] * rows
    return _immutable(_finite_result(found, "influence row"))
