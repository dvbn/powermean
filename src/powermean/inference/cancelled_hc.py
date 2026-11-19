"""Named HC covariances of a cancelled fit (candidate A1 of ``HC_DESIGN_NOTE.md``).

Owner's rulings 55, 63 and 65 (13 November 2026); design note, Sections 2.2, 3 and
5.1.  Private.  The multipliers, the factor of HC1 and the meat primitive are those
of :mod:`powermean.inference.hc`; the singleton rule, sandwich's warning and the
missing covariance of an undefined request are those of
:mod:`powermean.inference.absorbed_block_hc`.

The system.  A cancelled fit solves ``sum_i bt_i rho_i = 0`` with
``bt_i = b_i - sum_{j in g} pi_j b_j`` (``b`` the row of ``[X, F_R]``, ``pi`` the
cancellation shares ``w_i / W_g``) and ``rho_i = t_i - pi_i T_g``
(:mod:`powermean.systems.cancelled`).  The rows of one cancelled group sum to zero,
so the cancelled scales are estimated in closed form: the system is the one of the
full design ``[B, cancelled-group indicators]`` with row residual
``t_i - A_g w_i`` and expected Jacobian weights ``gamma_i = (T_g / W_g) w_i``.

Leverage.  The hat of that full design under ``gamma``,

    h_i = pi_i + gamma_i bt_i' (sum_j gamma_j bt_j bt_j')^-1 bt_i,

computed by :func:`powermean.inference.block_leverage.eliminated_forms` with the
cancelled block as the eliminated block: no solve per row, no matrix with one row
per observation and one column per retained level.  On a fit whose retained
coordinates are a selection of the retained levels the engine reads those levels;
on the spectral basis it reads every level and the basis.  At the PPML variance
``gamma`` is the fitted mean and ``h`` is the joint PPML fit's leverage; at any
variance ``h_i`` is the derivative of the fitted mean ``(T_g / W_g) m_i`` with
respect to the row's own outcome at an exact fit.

Covariance.  ``V = c H^-1 [sum_i q_i rho_i^2 bt_i bt_i'] H^-T`` on the slopes: the
row terms ``rho_i bt_i' W`` of the fit's own ``W = H^-T L'`` (the observed bread,
the only one a cancelled fit admits), the multipliers ``q_i = (1 - h_i)^-e_i`` and
``c = N / (N - p)`` for HC1, one otherwise.

Counts.  The observations are the rows of a group with a positive total that are
no singletons (rule U1: a row alone in a cancelled group or in a retained level,
applied until none remains, has leverage one and a zero score; it is excluded and
counted in the note ``covariance.singletons_excluded``).  ``N`` is their number
and ``p`` the full count: the identified coordinates of the cancelled system plus
the cancelled groups with a positive total, less one per singleton row.  HC0 and
HC1 compute no leverage.
"""

from __future__ import annotations

import warnings

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.inference_requests import HCMethod
from powermean.errors import CovarianceWarning, SpecificationError
from powermean.fe.cancel import to_original
from powermean.inference.absorbed_block_hc import singleton_rows
from powermean.inference.absorbed_rank import absorbed_count
from powermean.inference.block_leverage import eliminated_forms
from powermean.inference.hc import (
    LEVERAGE_METHODS,
    HCCovariance,
    Multipliers,
    _method,
    _unstable,
    adjusted_estimate,
    missing_covariance,
    multipliers,
)
from powermean.inference.leverage import Leverage, _traced
from powermean.inference.ssc import (
    AppliedConvention,
    ResolvedConvention,
    apply_convention,
    convention_counts,
    scaled_covariance,
)
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray
from powermean.systems.identified import IdentifiedState


def expected_weights(state: IdentifiedState) -> FloatArray:
    """``gamma_i = (T_g / W_g) w_i`` in input row order; zero in an all-zero group."""
    full = state.full_state
    structure = full.design.structure
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scale = full.group_t / full.group_w
        gamma: FloatArray = scale[structure.sorted_codes] * full.w
    return to_original(structure, gamma)


def cancelled_leverage(
    state: IdentifiedState, gamma: FloatArray, singletons: IndexArray, rank: int
) -> Leverage:
    """The hats of the full design of the cancelled system (module docstring).

    Parameters
    ----------
    state : IdentifiedState
        The accepted point of the cancelled fit.
    gamma : array
        :func:`expected_weights` of the same point.
    singletons : array of int
        The singleton rows: their hat is one in the full design; the record leaves
        them out (zero hat, zero multiplicity).
    rank : int
        ``p``, the full count less the singleton rows.

    Raises
    ------
    ValueError
        A full design that is not identified on the rows of a positive total, or a
        trace that does not round to ``p``.
    """
    geometry, design = state.design, state.full_state.design
    structure = design.structure
    slopes = to_original(structure, design.centered_slopes)
    k_full = design.n_slopes
    levels = design.n_columns - k_full
    columns = geometry.selected_columns
    basis: FloatArray | None = None
    if columns is None:
        # Spectral basis: every retained level, and the basis of the identified space.
        lookup = np.arange(levels, dtype=np.intp)
        basis, order = geometry.basis_matrix, levels
    else:
        kept = columns[columns >= k_full] - k_full
        lookup = np.full(levels, -1, dtype=np.intp)
        lookup[kept] = np.arange(kept.size)
        slopes, order = slopes[:, columns[columns < k_full]], int(kept.size)
    coordinates = tuple(
        to_original(structure, lookup[(start - k_full) + codes])
        for start, codes in zip(
            design.block_starts, structure.retained_sorted_codes, strict=True
        )
    )
    forms = eliminated_forms(
        gamma, structure.cancel_codes, slopes, coordinates, order, basis=basis
    )
    f = (gamma > 0).astype(np.float64)
    f[singletons] = 0.0
    hat = np.where(f > 0, forms, 0.0)
    trace, p = _traced(hat, f, rank, "cancelled full-design")
    return Leverage(
        hat=_immutable(hat),
        gram_weights=_immutable(gamma),
        multiplicity=_immutable(f),
        frequency=False,
        trace=trace,
        rank=rank,
        p=p,
        geometry="expected",
        slope_hat=None,
        nuisance_hat=None,
        slope_map=None,
        nuisance=None,
        diagonal=None,
    )


def cancelled_counts(
    state: IdentifiedState, gamma: FloatArray
) -> tuple[IndexArray, IndexArray, int]:
    """The observations, the singleton rows and the full count ``p`` (docstring)."""
    structure = state.design.structure
    alive = (gamma > 0).astype(np.float64)
    codes = [structure.cancel_codes]
    inverse = structure.inverse
    codes.extend(block[inverse] for block in structure.retained_sorted_codes)
    singletons = singleton_rows(codes, alive)
    kept = alive > 0
    groups = int(np.unique(structure.cancel_codes[kept]).size)
    kept[singletons] = False
    rank = state.design.n_identified + groups - int(singletons.size)
    return frozen_index(np.flatnonzero(kept)), singletons, rank


def cancelled_convention(
    state: IdentifiedState,
    method: HCMethod,
    convention: ResolvedConvention,
    active: IndexArray,
    rank: int,
    *,
    cancelled_index: int,
) -> AppliedConvention:
    """The convention in force on the counts of the request; its refusals.

    The counts are those a joint fit of the same blocks forms: ``N``, the
    identified slopes, the full count ``p`` as the rank, and the convention count
    of the effects on the observations, the blocks in the order of the formula
    (``cancelled_index`` is the place of the cancelled block in it).  The factor of
    HC1 reads ``N`` and ``p``; the convention count serves the equality rule of an
    explicit ``ssc`` and is not a factor of a named method.

    Raises
    ------
    SpecificationError
        ``specification.no_residual_dof`` (no observation left, or HC1 without
        residual degrees of freedom) or ``specification.conflicting`` (an explicit
        second correction).
    """
    count = int(active.size)
    if count < 1:
        raise SpecificationError(
            "specification.no_residual_dof",
            {"method": method, "count": 0, "rank": rank},
        )
    structure = state.design.structure
    codes = [block[structure.inverse] for block in structure.retained_sorted_codes]
    codes.insert(cancelled_index, structure.cancel_codes)
    absorbed = absorbed_count(
        tuple((str(place), block[active]) for place, block in enumerate(codes)), ()
    )
    counts = convention_counts(count, state.design.n_slopes_identified, rank, absorbed)
    return apply_convention(convention, counts, kind="robust", method=method)


def cancelled_hc(
    state: IdentifiedState,
    influence: FloatArray,
    method: HCMethod,
    convention: ResolvedConvention,
    *,
    cancelled_index: int = 0,
) -> HCCovariance:
    """The named HC covariance of the slopes of a cancelled fit at ``state``.

    Parameters
    ----------
    state : IdentifiedState
        The accepted point.
    influence : array
        The row terms ``rho_i bt_i' W`` of the same point in input row order, one
        column per identified slope
        (:func:`powermean.inference.cancelled.contracted_influence` with
        ``rows=True``).
    method : HC method
        One of the seven names.
    convention : ResolvedConvention
        The convention in force; the method owns its factor.
    cancelled_index : int
        The place of the cancelled block among the blocks of the formula.

    Returns
    -------
    HCCovariance
        The covariance of the identified slopes (missing above sandwich's
        near-one warning criterion), the applied convention with ``N``
        and ``p``, the hats and the multipliers (``None`` for HC0 and HC1).
    """
    found = _method(method)
    n = state.design.structure.n_rows
    if influence.ndim != 2 or influence.shape[0] != n:
        raise ValueError("the row terms belong to other rows")
    gamma = expected_weights(state)
    active, singletons, rank = cancelled_counts(state, gamma)
    count = int(active.size)
    applied = cancelled_convention(
        state, found, convention, active, rank, cancelled_index=cancelled_index
    )
    if singletons.size:
        warnings.warn(
            CovarianceWarning(
                "covariance.singletons_excluded",
                {"method": found, "count": int(singletons.size), "kept": count},
            ),
            stacklevel=2,
        )
    leverage: Leverage | None = None
    weighted: Multipliers | None = None
    unstable = undefined = 0
    values = np.ones(count)
    if found in LEVERAGE_METHODS:
        leverage = cancelled_leverage(state, gamma, singletons, rank)
        unstable = _unstable(found, leverage)
        undefined = unstable
        if not undefined:
            weighted = multipliers(found, leverage, count)
            values = weighted.values
    rows = influence[active]
    estimate = adjusted_estimate(rows, np.ones(count), values, frequency=False)
    scaled = scaled_covariance(estimate, applied)
    return HCCovariance(
        found,
        missing_covariance(scaled.covariance) if undefined else scaled.covariance,
        estimate,
        applied,
        leverage,
        weighted,
        "observed",
        "absorbed",
        _immutable(rows),
        dropped=int(singletons.size),
        unstable=unstable,
        undefined=undefined,
    )


__all__ = [
    "cancelled_convention",
    "cancelled_counts",
    "cancelled_hc",
    "cancelled_leverage",
    "expected_weights",
]
