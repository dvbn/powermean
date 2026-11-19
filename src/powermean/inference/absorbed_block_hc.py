"""Named HC covariances of a joint fixed-effect fit on the block-elimination leverage.

Owner's rulings 55, 63 and 65 (13 November 2026); ``HC_DESIGN_NOTE.md``, Sections 1,
5 and 6.  Private.  The definitions are those of :mod:`powermean.inference.hc`
(multipliers, factor of HC1, bread, the composite with expected-information hats);
what differs from the earlier absorbed composition is how the request is prepared
and how the hats are computed.

Singleton rows (rule U1).  A row that is alone in a level of one block is fitted
exactly by that level: its score is identically zero and its leverage is exactly one.
Removing it can leave another row alone in a level, so the rule is applied until no
such row remains (:func:`singleton_rows`), as ``fixest`` and ``reghdfe`` remove
singletons.  These rows are no observations of the request: ``N`` and the full rank
exclude them (each spans one dimension of the full design), the factor of HC1 is
``N / (N - K)`` on what remains, and their number is reported in the note
``covariance.singletons_excluded``.  The covariance equals the one of the fit on the
sample without them.  With frequency counts a row is a singleton only when its count
is one: a lone row of count two or more is two replicas of leverage one half; it
stays an observation, and it still determines its level for the rows around it.

No rank or unit-leverage decision.  The request runs no exact elimination of the
design.  HC0 and HC1 read no leverage and compute none.  For HC2, HC3, HC4, HC4m and
HC5 the hats are those of :func:`powermean.inference.block_leverage.block_leverage`,
whose reduced matrix must have a Cholesky factor and whose trace must round to the
full rank.  A row that is not a singleton and whose leverage is numerically one (a
row that alone links two parts of the panel, an observation-specific regressor) is
handled with the note ``covariance.unstable_leverage``; see :func:`_unstable`.
The same near-one criterion used by ``sandwich`` for its warning and by ``fixest``
for refusal makes the covariance undefined here: the request returns a missing
covariance and missing standard errors after the note.  This includes an exact
unit hat rounded below one (:attr:`HCCovariance.undefined`).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.inference_requests import HCMethod
from powermean.errors import CovarianceWarning, SpecificationError
from powermean.fe.absorb_numeric import checked_product
from powermean.fe.certificate_basis import CertificateBasis
from powermean.inference.absorbed_rank import (
    AbsorbedCount,
    IntegerArray,
    absorbed_count,
    exact_absorbed,
)
from powermean.inference.block_leverage import block_leverage
from powermean.inference.bread import Bread, same_point_influence
from powermean.inference.contributions import statistical_count
from powermean.inference.hc import (
    LEVERAGE_METHODS,
    UNSTABLE_ABOVE,
    HCCovariance,
    Multipliers,
    _method,
    _unstable,
    adjusted_estimate,
    missing_covariance,
    multipliers,
)
from powermean.inference.leverage import Leverage, _frequency, _multiplicity
from powermean.inference.ssc import (
    AppliedConvention,
    ResolvedConvention,
    apply_convention,
    convention_counts,
    scaled_covariance,
)
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array
from powermean.solve.absorbed_certificate_geometry import hessian
from powermean.solve.absorbed_certificate_point import FreshRows


def singleton_rows(
    codes: Sequence[IntegerArray], multiplicity: FloatArray
) -> IndexArray:
    """The rows of leverage exactly one by the levels alone (module docstring).

    Parameters
    ----------
    codes : sequence of array of int
        The level of every row in each block.
    multiplicity : array
        ``f_i``: the replicas of each row, zero for a row that is no observation.

    Returns
    -------
    array of int
        The positions of the singleton rows, increasing.
    """
    alive = multiplicity > 0
    one_replica = multiplicity == 1
    alone = np.zeros(alive.shape, dtype=bool)
    while True:
        found = np.zeros(alive.shape, dtype=bool)
        for block in codes:
            rows = np.bincount(block[alive], minlength=int(block.max()) + 1)
            found |= alive & (rows[block] == 1)
        # A replicated row continues to support its other levels on later passes.
        found &= one_replica
        if not found.any():
            break
        alone |= found
        alive &= ~found
    return frozen_index(np.flatnonzero(alone))


@dataclass(frozen=True, eq=False)
class BlockPreflight:
    """What a named HC request on a joint fixed-effect fit decides before the solve.

    Attributes
    ----------
    method : HC method
        The method.
    applied : AppliedConvention
        The convention in force applied to the counts below.
    rows : array of int
        The observations of the request: the rows of positive weight that are not
        singletons, increasing.
    singletons : array of int
        The singleton rows, excluded.
    count : int
        ``N``, the statistical count of ``rows``.
    rank : int
        Slopes plus kept nuisance coordinates, less one per singleton row.
    frequency : bool
        Whether the weights are frequency counts.
    n_rows : int
        The rows of the sample, the rows of zero weight included.
    absorbed : AbsorbedCount
        The convention count of the effects on ``rows``.
    """

    method: HCMethod
    applied: AppliedConvention
    rows: IndexArray
    singletons: IndexArray
    count: int
    rank: int
    frequency: bool
    n_rows: int
    absorbed: AbsorbedCount


def block_preflight(
    n_slopes: int,
    basis: CertificateBasis,
    weights: FloatArray,
    method: HCMethod,
    convention: ResolvedConvention,
    *,
    frequency: bool,
) -> BlockPreflight:
    """The counts and refusals of the request that need no point (module docstring).

    Raises
    ------
    SpecificationError
        ``specification.no_residual_dof`` (HC1 without residual degrees of freedom,
        or no observation left) or ``specification.conflicting`` (an explicit second
        correction).
    """
    found = _method(method)
    a = _finite_array(weights, "weights")
    f = _multiplicity(a, frequency=_frequency(frequency))
    blocks = basis.grouping.blocks
    if a.shape != (basis.grouping.n_rows,):
        raise ValueError("the weights belong to other rows")
    singletons = singleton_rows([block.codes for block in blocks], f)
    kept = f > 0
    kept[singletons] = False
    rows = frozen_index(np.flatnonzero(kept))
    rank = int(n_slopes) + basis.n_kept - int(singletons.size)
    if not rows.size:
        raise SpecificationError(
            "specification.no_residual_dof",
            {"method": found, "count": 0, "rank": rank},
        )
    count = statistical_count(a[rows], frequency=frequency)
    on_rows = tuple((block.name, block.codes[rows]) for block in blocks)
    absorbed = exact_absorbed(
        absorbed_count(on_rows, ()), lambda: on_rows, rank - int(n_slopes)
    )
    applied = apply_convention(
        convention,
        convention_counts(count, int(n_slopes), rank, absorbed),
        kind="robust",
        method=found,
    )
    return BlockPreflight(
        found, applied, rows, singletons, count, rank, frequency, int(a.size), absorbed
    )


def block_composition(
    design: FloatArray,
    basis: CertificateBasis,
    rows: FreshRows,
    fisher: FloatArray,
    preflight: BlockPreflight,
    *,
    bread: Bread,
    observed_influence: FloatArray | None = None,
) -> HCCovariance:
    """The named HC covariance of given rows of a joint fixed-effect fit.

    ``design`` is ``X`` on every row of the grouping, ``basis`` the kept nuisance
    coordinates, ``rows`` the observed ``b = a r`` and ``C = a c`` with the raw
    weights and their kind, ``fisher`` the expected row curvature at the same point.
    ``preflight`` is the request's own (:func:`block_preflight` of the same basis,
    weights and kind: the method, the counts, the singleton rows), and
    ``observed_influence`` the influence rows of the observed bread where the caller
    holds them; they are formed here otherwise.  As the earlier composition, this is
    an internal helper: the caller binds the four inputs to one point.
    """
    if type(rows) is not FreshRows or type(basis) is not CertificateBasis:
        raise TypeError("rows must be FreshRows and basis a CertificateBasis")
    matrix = _finite_array(design, "design")
    a, frequency = rows.weights, rows.frequency
    n = basis.grouping.n_rows
    if matrix.ndim != 2 or matrix.shape[0] != n or not matrix.shape[1]:
        raise ValueError("design must have one row per row of the grouping, a column")
    if a.shape != (n,) or rows.b.shape != (n,) or rows.curvature.shape != (n,):
        raise ValueError("the rows belong to another grouping")
    pre = preflight
    if type(pre) is not BlockPreflight:
        raise TypeError("preflight must be a BlockPreflight")
    if pre.n_rows != n or pre.frequency != frequency:
        raise ValueError("the preflight belongs to other rows")
    found = pre.method
    active = pre.rows
    if pre.singletons.size:
        warnings.warn(
            CovarianceWarning(
                "covariance.singletons_excluded",
                {
                    "method": found,
                    "count": int(pre.singletons.size),
                    "kept": pre.count,
                },
            ),
            stacklevel=2,
        )
    leverage: Leverage | None = None
    weighted: Multipliers | None = None
    unstable = undefined = 0
    values = np.ones(active.size)
    if found in LEVERAGE_METHODS:
        leverage = block_leverage(
            matrix, basis, a, fisher, frequency=frequency, dropped=pre.singletons
        )
        unstable = _unstable(found, leverage)
        undefined = unstable
        if not undefined:
            weighted = multipliers(found, leverage, pre.count)
            values = weighted.values
    if bread == "observed" and observed_influence is not None:
        influence = observed_influence
    else:
        curvature = (
            rows.curvature
            if bread == "observed"
            else checked_product(a, fisher, "Gram weight")
        )
        influence = same_point_influence(
            matrix, basis, rows.b, hessian(matrix, curvature, basis), bread=bread
        ).influence
    estimate = adjusted_estimate(
        influence[active], a[active], values, frequency=frequency
    )
    scaled = scaled_covariance(estimate, pre.applied)
    return HCCovariance(
        found,
        missing_covariance(scaled.covariance) if undefined else scaled.covariance,
        estimate,
        pre.applied,
        leverage,
        weighted,
        bread,
        "absorbed",
        _immutable(influence[active]),
        dropped=int(pre.singletons.size),
        unstable=unstable,
        undefined=undefined,
    )


__all__ = [
    "UNSTABLE_ABOVE",
    "BlockPreflight",
    "block_composition",
    "block_preflight",
    "singleton_rows",
]
