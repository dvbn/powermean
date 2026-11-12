"""The private reporting constant of an absorbed fit: its functional and influence row.

Internal module: the consumer of the retained nuisance geometry
(:mod:`powermean.solve.absorbed_certificate_geometry`,
:mod:`powermean.fe.certificate_basis`) that the legacy sequence of
:mod:`powermean.inference.psd` needs for a fit whose reported coefficients hold no
intercept.  Nothing here fits, reselects a sample or reevaluates a formula; no
coefficient, standard error, name or option is added for the constant.

Both reference programs recover a reporting constant for an absorbed fit from weighted
means: the constant is the weighted mean, over the estimation sample, of the total
absorbed effect at the fitted point, the effects being normalized so that this mean is
carried by the constant.  With the kept incidence ``E_J`` of the certificate basis
(``D = E M``; the total absorbed effect of row ``i`` is ``(E_J theta_J)_i``, the
``lift`` of the ``embed`` of the nuisance coordinates) and the staged weights ``w_i``
of the program at the fitted point, the constant is the linear functional

    c = L' theta_J,    L_p = sum_i w_i (E_J)_ip / sum_i w_i

of the nuisance coordinates (:func:`constant_functional`).  The staged weights are the
true weights for the linear route and the true weights times the IRLS working weight
``m^2 / V(m)`` of the row at the fitted point for a working variance (the fitted mean
``mu`` at ``tau = 1``, which is ``ppmlhdfe``'s ``true_w`` times ``mu``; their
normalization by a constant changes no weighted mean), the same quantity as the
expected curvature that the retained ``ExpectedInformationInputs`` of
:mod:`powermean.results.absorbed_geometry` forms; the caller supplies them.  Location
shifts of the programs (an offset subtracted, ``log(sd_y)`` added) do not enter the
covariance and are not formed.

Influence (:func:`constant_influence`).  The joint influence of row ``i`` on
``(theta_X, theta_J)`` is ``H^-1 B_i' b_i`` with ``B = [X, E_J]``, so the constant's
influence row is the same functional of the nuisance block,

    c_i = b_i (B_i v),    v = H^-T [0_K; L],

one transposed solve against the retained full joint Hessian ``H`` (the observed one of
the certificate, or the expected information assembled on the same basis for
``bread="expected"``) by the sparse actions of the certificate (the
equilibration ``s``, the scaled ordinary ``A = S^-1 H S^-1``, its transient LU factor
and one ``T`` solve, ``v = S^-1 A^-T (S^-1 r)``, as ``slope_map`` forms ``W``).  The
slopes' own influence rows are the retained ones; this row sits beside them so that the
raw signed multiway port forms the constant's variance and its cross-covariances with
the slopes at the SAME fitted point, with the same cluster totals and the same factor
applied once by its owner.  No live factor, dense inverse, refit, substitute matrix or
unweighted mean: a failed factor or an undefined weight total refuses with its reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.fe.absorb_numeric import group_sums
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


@dataclass(frozen=True, eq=False)
class ConstantRow:
    """The private reporting constant's functional and influence row, owned.

    Attributes
    ----------
    functional : array
        ``L`` over the kept nuisance coordinates ``J``.
    total : float
        ``sum_i w_i``, the weight total of the mean.
    map : array
        ``v = H^-T [0; L]``, one column over the slopes and ``J``.
    influence : array
        ``c_i = b_i (B_i v)`` for every row of the grouping, in its order.
    factor : FactorRecord
        The record of the transient ordinary factor of ``H`` (no live factor is kept).
    """

    functional: FloatArray
    total: float
    map: FloatArray
    influence: FloatArray
    factor: FactorRecord


def constant_functional(basis: CertificateBasis, weights: FloatArray) -> FloatArray:
    """``L_p = sum_i w_i (E_J)_ip / sum_i w_i`` over the kept coordinates ``J``.

    Parameters
    ----------
    basis : CertificateBasis
        The kept incidence coordinates of the grouping.
    weights : array
        The staged weights of the program at the fitted point, one per row of the
        grouping, finite and nonnegative, with a positive total.

    Returns
    -------
    array
        ``L``, owned, one entry per kept coordinate; a reference level of a later
        block contributes to no coordinate, as the ``embed`` and ``lift`` of the basis
        give it no coefficient.

    Raises
    ------
    ValueError
        Weights that are not one finite nonnegative value per row, or a total that is
        not positive.
    """
    values = np.asarray(weights, dtype=np.float64)
    rows = basis.grouping.n_rows
    if values.shape != (rows,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("the staged weights are one finite nonnegative value per row")
    total = float(np.sum(values))
    if not total > 0:
        raise ValueError("the staged weights of the reporting constant sum to zero")
    functional = np.zeros(basis.n_kept)
    for block, position in zip(basis.grouping.blocks, basis.positions, strict=True):
        sums = group_sums(values, block.codes, block.n_levels)
        kept = position >= 0
        functional[position[kept]] += sums[kept]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        found: FloatArray = functional / total
    return _immutable(_finite_result(found, "the constant functional"))


def constant_influence(
    design: FloatArray,
    basis: CertificateBasis,
    b: FloatArray,
    hessian: SparseMatrix,
    weights: FloatArray,
) -> ConstantRow:
    """The private constant's influence row on every row (module docstring).

    Parameters
    ----------
    design : array
        ``X``, the scaled slopes of the certificate, one row per row of the grouping.
    basis : CertificateBasis
        The kept nuisance coordinates ``J`` of the same rows.
    b : array
        The raw weighted score multipliers, one per row.
    hessian : SparseMatrix
        The full joint bread ``H`` on the same basis, canonical CSC.
    weights : array
        The staged weights of the program at the fitted point, one per row.

    Raises
    ------
    TypeError
        ``hessian`` is not a ``SparseMatrix``.
    ValueError
        Shapes that do not match the rows, the slopes and ``J``; a zero row or column
        of ``H`` (no scale); a factor that was refused or failed, with its reason; an
        undefined weight total.
    FloatingPointError
        A map or influence entry that is not representable.
    """
    if type(hessian) is not SparseMatrix:
        raise TypeError("hessian must be a SparseMatrix")
    rows, k = design.shape
    size = k + basis.n_kept
    if b.shape != (rows,) or hessian.shape != (size, size) or not k:
        raise ValueError("the design, the scores and the Hessian are not aligned")
    if basis.grouping.n_rows != rows:
        raise ValueError("the basis belongs to other rows")
    functional = constant_functional(basis, weights)
    total = float(np.sum(np.asarray(weights, dtype=np.float64)))
    scale = equilibration(hessian)
    ordinary = scaled_ordinary(hessian, scale)
    record, factor = factorized(ordinary, "ordinary")
    if factor is None:
        raise ValueError(f"the bread has no factor: {record.reason}")
    rhs = np.concatenate([np.zeros(k), functional])
    with np.errstate(all="ignore"):
        mapped: FloatArray = Calls().solve(factor, rhs / scale, "T") / scale
    mapped = _finite_result(mapped, "the constant map")
    nuisance = lift(basis, embed(basis, mapped[k:]))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        full: FloatArray = design @ mapped[:k] + nuisance
        influence: FloatArray = b * full
    return ConstantRow(
        functional=functional,
        total=total,
        map=_immutable(mapped),
        influence=_immutable(_finite_result(influence, "the constant influence")),
        factor=record,
    )


__all__ = ["ConstantRow", "constant_functional", "constant_influence"]
