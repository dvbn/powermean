"""Same-point bread-to-slope influence on the full nuisance basis, observed or expected.

The bread is the observed Jacobian; ``bread="expected"`` substitutes the expected
information on joint systems, and the two coincide at ``tau = 1``.  Internal
module: a computation at a retained point, never another fit.

At one accepted coefficient point with the full joint design ``B = [X, E_J]`` (``X``
the scaled slopes, ``E_J`` the kept incidence of
:mod:`powermean.fe.certificate_basis`) and the raw weighted score multipliers ``b``, the
primitive score rows are ``s_i = b_i B_i``.  Under a bread ``H`` (the observed
``B' diag(C) B`` of the certificate, or the expected ``B' diag(C_e) B`` that
:class:`powermean.results.absorbed_geometry.ExpectedInformationInputs` assembles on the
SAME basis) the slope influence of row ``i`` is ``u_i = s_i H^-T E_s``: with
``W = H^-T E_s`` the first ``p`` columns of ``H^-T``,

    u_i = b_i (X_i W_X + (E_J)_i W_J) = b_i (X W_X + lift(embed(W_J)))_i

on EVERY row, slopes and nuisance alike: swapping a slope-only inverse while keeping
observed-residualized scores is not this.  ``W`` is formed by the sparse
actions of the certificate: the equilibration ``s`` of ``H``, the scaled ordinary
``A = S^-1 H S^-1``, its transient LU factor (``factorized``, the ordinary options) and
the ``K`` transposed solves of ``slope_map``, ``W = S^-1 A^-T S^-1 E_s``.  The factor is
local to the call, its record kept as provenance; no live factor, dense inverse,
optimizer, recertification, ridge, pseudoinverse or fallback: a zero scale or a failed
factor refuses with its reason.  The reporting map is not applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.fe.certificate_basis import CertificateBasis, embed, lift
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result
from powermean.solve.absorbed_certificate_factor import (
    Calls,
    FactorRecord,
    factorized,
)
from powermean.solve.absorbed_certificate_geometry import (
    SparseMatrix,
    equilibration,
    scaled_ordinary,
)
from powermean.solve.absorbed_certificate_sparse import slope_map

Bread = Literal["observed", "expected"]
BREADS: tuple[Bread, ...] = ("observed", "expected")


@dataclass(frozen=True, eq=False)
class SlopeMap:
    """``W = H^-T E_s`` under one bread, with the provenance of its transient factor.

    Attributes
    ----------
    bread : {"observed", "expected"}
        The bread ``H`` was.
    scale : array
        The equilibration ``s`` of ``H``.
    factor : FactorRecord
        The record of the ordinary factor of ``A`` (no live factor is kept).
    columns : array
        ``W``, ``(K + |J|) x K``, owned.
    """

    bread: Bread
    scale: FloatArray
    factor: FactorRecord
    columns: FloatArray


@dataclass(frozen=True, eq=False)
class BreadInfluence:
    """The slope influence rows of every row under one bread.

    Attributes
    ----------
    bread : {"observed", "expected"}
        The bread.
    slope_map : SlopeMap
        ``W`` and its provenance.
    influence : array
        ``u_i = b_i (B_i W)`` for every row, ``n x K``, owned.
    """

    bread: Bread
    slope_map: SlopeMap
    influence: FloatArray


def _bread(bread: object) -> Bread:
    """``"observed"`` or ``"expected"``."""
    if bread not in BREADS:
        raise ValueError("bread must be 'observed' or 'expected'")
    found: Bread = "observed" if bread == "observed" else "expected"
    return found


def slope_map_of(hessian: SparseMatrix, n_slopes: int, *, bread: Bread) -> SlopeMap:
    """``W = H^-T E_s`` for the full joint ``H`` of a bread, by the accepted actions.

    Parameters
    ----------
    hessian : SparseMatrix
        ``H``, square over the slopes and the kept nuisance coordinates, canonical CSC.
    n_slopes : int
        ``K``, the number of slopes, at least one.
    bread : {"observed", "expected"}
        Which bread ``H`` is; recorded.

    Returns
    -------
    SlopeMap
        ``W`` and the provenance of the transient factor.

    Raises
    ------
    ValueError
        ``K`` outside ``1 .. dimension``, a zero row or column of ``H`` (no scale), or
        a factor that was refused or failed, with its reason.
    FloatingPointError
        A scaled entry or a column of ``W`` that is not representable.
    """
    kind = _bread(bread)
    if type(hessian) is not SparseMatrix:
        raise TypeError("hessian must be a SparseMatrix")
    size = hessian.shape[0]
    if hessian.shape != (size, size) or not 1 <= n_slopes <= size:
        raise ValueError("the slopes must be a nonempty leading block of the Hessian")
    scale = equilibration(hessian)
    ordinary = scaled_ordinary(hessian, scale)
    record, factor = factorized(ordinary, "ordinary")
    if factor is None:
        raise ValueError(f"the {kind} bread has no factor: {record.reason}")
    columns = slope_map(factor, scale, n_slopes, Calls())
    return SlopeMap(
        kind, scale, record, _immutable(_finite_result(columns, "slope map"))
    )


def bread_influence(
    design: FloatArray, basis: CertificateBasis, b: FloatArray, columns: FloatArray
) -> FloatArray:
    """``u_i = b_i (X_i W_X + lift(embed(W_J))_i)`` on every row; owned.

    Parameters
    ----------
    design : array
        ``X``, one row per row of the grouping, ``K`` columns.
    basis : CertificateBasis
        The kept nuisance coordinates ``J`` of the same rows.
    b : array
        The raw weighted score multipliers, one per row.
    columns : array
        ``W``, ``(K + |J|) x K``.

    Raises
    ------
    ValueError
        Shapes that do not match the rows, the slopes and ``J``.
    FloatingPointError
        A lifted nuisance term or an influence entry that is not representable.
    """
    rows, k = design.shape
    if b.shape != (rows,) or columns.shape != (k + basis.n_kept, k) or not k:
        raise ValueError("the design, the scores and the slope map are not aligned")
    if basis.grouping.n_rows != rows:
        raise ValueError("the basis belongs to other rows")
    nuisance = lift(basis, embed(basis, columns[k:]))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        full: FloatArray = design @ columns[:k] + nuisance
        found: FloatArray = b[:, None] * full
    return _immutable(_finite_result(found, "influence row"))


def same_point_influence(
    design: FloatArray,
    basis: CertificateBasis,
    b: FloatArray,
    hessian: SparseMatrix,
    *,
    bread: Bread,
) -> BreadInfluence:
    """The slope influence of every row under the bread ``hessian`` (module docstring).

    Parameters
    ----------
    design, basis, b
        The same-point design, kept nuisance coordinates and score multipliers.
    hessian : SparseMatrix
        The bread on the same full basis: the certificate's observed ``H`` or the
        retained expected information.
    bread : {"observed", "expected"}
        Which it is; recorded.
    """
    found = slope_map_of(hessian, int(design.shape[1]), bread=bread)
    return BreadInfluence(
        found.bread, found, bread_influence(design, basis, b, found.columns)
    )


__all__ = [
    "BREADS",
    "Bread",
    "BreadInfluence",
    "SlopeMap",
    "bread_influence",
    "same_point_influence",
    "slope_map_of",
]
