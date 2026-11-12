"""Detachment helpers and temporary identity map for absorbed point records.

This module defines :data:`RawPointRecord`, :data:`AbsorbedRoute`, :data:`PointMapper`,
:func:`snapshot_linear_normalization`, :func:`snapshot_fresh_equations`,
:func:`_compute_beta`, :func:`snapshot_point_record`, and :class:`PointIdentityMap`, and
re-exports record classes from :mod:`powermean.results.absorbed_point_records`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from powermean.model.variance import FloatArray
from powermean.results.absorbed_groupings import GroupingIdentityMap
from powermean.results.absorbed_point_records import (
    Family,
    ReferenceTuple,
    RetainedFreshEquations,
    RetainedLinearNormalization,
    RetainedPointSnapshot,
    _owned_references,
)
from powermean.results.absorbed_sample import RetainedSampleSnapshot, SampleIdentityMap
from powermean.results.absorbed_states import _owned_profile_result, _owned_recovery
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import PointKey, SampleKey, _tuple_arrays
from powermean.solve.absorbed_certificate_history import Route
from powermean.solve.absorbed_certificate_point import AdmittedPoint, InnerControls
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.absorbed_root_records import RootPoint
from powermean.solve.linear_absorbed_point import (
    FreshEquations,
    LinearPoint,
    PointNormalization,
)
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.systems.profiled import PointProfile

RawPointRecord = PointProfile | ConditionalPoint | IndexPoint | LinearPoint | RootPoint
AbsorbedRoute = Route
PointMapper = Callable[[RawPointRecord], RetainedPointSnapshot]

__all__ = [
    "AbsorbedRoute",
    "Family",
    "PointIdentityMap",
    "PointMapper",
    "RawPointRecord",
    "ReferenceTuple",
    "RetainedFreshEquations",
    "RetainedLinearNormalization",
    "RetainedPointSnapshot",
    "_compute_beta",
    "_owned_references",
    "_snapshot_linear_point",
    "snapshot_fresh_equations",
    "snapshot_linear_normalization",
    "snapshot_point_record",
]


def snapshot_linear_normalization(
    norm: PointNormalization | RetainedLinearNormalization | None,
    *,
    point_key: PointKey = 0,
    sample_key: SampleKey = 0,
    get_point: PointMapper | None = None,
    sample_map: SampleIdentityMap | None = None,
) -> RetainedLinearNormalization | None:
    """Snapshot ``PointNormalization`` using its own ``norm.point`` identity."""
    if norm is None:
        return None
    if isinstance(norm, RetainedLinearNormalization):
        pk = norm.point_key if point_key == 0 else point_key
        sk = norm.sample_key if sample_key == 0 else sample_key
        head = (pk, sk, norm.effects, norm.references, norm.shifts)
        return RetainedLinearNormalization(*head, norm.index, norm.gap, norm.rank)
    pk, sk = point_key, sample_key
    if get_point is not None:
        pt_snap = get_point(norm.point)
        pk, sk = pt_snap.key, pt_snap.sample_key
    elif sample_map is not None:
        sk = sample_map.capture(norm.point.sample).key
    eff, refs = _tuple_arrays(norm.effects), _owned_references(norm.references)
    sh, idx = _tuple_arrays(norm.shifts), owned_array(norm.index)
    return RetainedLinearNormalization(
        pk, sk, eff, refs, sh, idx, float(norm.gap), norm.rank
    )


def snapshot_fresh_equations(
    eqs: FreshEquations | RetainedFreshEquations | None,
    *,
    point_key: PointKey = 0,
    sample_key: SampleKey = 0,
    get_point: PointMapper | None = None,
    sample_map: SampleIdentityMap | None = None,
) -> RetainedFreshEquations | None:
    """Snapshot ``FreshEquations`` using its own ``eqs.point`` identity."""
    if eqs is None:
        return None
    if isinstance(eqs, RetainedFreshEquations):
        pk = eqs.point_key if point_key == 0 else point_key
        sk = eqs.sample_key if sample_key == 0 else sample_key
        head = (pk, sk, eqs.residual, eqs.weighted, eqs.slope_sums)
        tail = (eqs.level_sums, eqs.level_totals, eqs.level_means)
        return RetainedFreshEquations(*head, *tail, eqs.largest_level_mean)
    pk, sk = point_key, sample_key
    if get_point is not None:
        pt_snap = get_point(eqs.point)
        pk, sk = pt_snap.key, pt_snap.sample_key
    elif sample_map is not None:
        sk = sample_map.capture(eqs.point.sample).key
    res, wtd = owned_array(eqs.residual), owned_array(eqs.weighted)
    head = (pk, sk, res, wtd, owned_array(eqs.slope_sums))
    ls, lt = _tuple_arrays(eqs.level_sums), _tuple_arrays(eqs.level_totals)
    tail = (ls, lt, _tuple_arrays(eqs.level_means))
    return RetainedFreshEquations(*head, *tail, float(eqs.largest_level_mean))


def _compute_beta(sample: RetainedSampleSnapshot, theta: FloatArray) -> FloatArray:
    """Compute reported slope vector ``beta`` from solver-coordinate ``theta``.

    The map is read from the sample's owned storage; the result is owned anew.
    """
    order = sample._owned_order_scale if theta.size else None
    if order is None:
        return owned_array(np.zeros(0, dtype=np.float64))
    try:
        return owned_array(order.coefficients(theta))
    except FloatingPointError:
        return owned_array(np.full(theta.shape, np.nan, dtype=np.float64))


def _snapshot_profiled_point(
    rec: PointProfile,
    sample_snapshot: RetainedSampleSnapshot,
    *,
    var: Family | None,
    inn: InnerControls | None,
    pkey: PointKey,
    grouping_map: GroupingIdentityMap | None,
) -> RetainedPointSnapshot:
    """Detach a ``PointProfile`` record into a :class:`RetainedPointSnapshot`."""
    b = _compute_beta(sample_snapshot, rec.theta)
    pr = _owned_profile_result(
        rec.result,
        grouping_snapshot=sample_snapshot._final_grouping,
        grouping_map=grouping_map,
    )
    s, k, conv = sample_snapshot, int(rec.theta.size), bool(rec.converged)
    pk, eff = s.partition_key, _tuple_arrays(rec.state.coefficients)
    rest = (s, var, inn, k, conv, rec.theta, b, rec.base, rec.index, pr, rec.scalar)
    return RetainedPointSnapshot(pkey, s.key, pk, "profiled", *rest, _raw_effects=eff)


def _snapshot_index_point(
    rec: IndexPoint,
    sample_snapshot: RetainedSampleSnapshot,
    *,
    var: Family | None,
    inn: InnerControls | None,
    pkey: PointKey,
    grouping_map: GroupingIdentityMap | None,
) -> RetainedPointSnapshot:
    """Detach an ``IndexPoint`` record into a :class:`RetainedPointSnapshot`.

    The recovery's owned effects and normalization are read from its storage; the
    point snapshot owns its own copies of them at its ingress.
    """
    rs = _owned_recovery(
        rec.recovered,
        grouping_snapshot=sample_snapshot._final_grouping,
        grouping_map=grouping_map,
    )
    nm, conv = (None, None) if rs is None else (rs._normalization, rs.complete)
    b = _compute_beta(sample_snapshot, rec.theta)
    s, k, pk = sample_snapshot, int(rec.theta.size), sample_snapshot.partition_key
    raw_e = None if rs is None else rs._effects
    ne, rf, sh = (
        (None, None, None) if nm is None else (nm.effects, nm.references, nm.shifts)
    )
    r1 = (s, var, inn, k, conv, rec.theta, b, rec.base, rec.index)
    r2 = (None, None, rs, None, raw_e, ne, rf, sh)
    rest = (*r1, *r2)
    return RetainedPointSnapshot(pkey, s.key, pk, "index", *rest)


def _snapshot_root_point(
    rec: RootPoint,
    sample_snapshot: RetainedSampleSnapshot,
    *,
    var: Family | None,
    inn: InnerControls | None,
    pkey: PointKey,
) -> RetainedPointSnapshot:
    """Detach a ``RootPoint`` record into a :class:`RetainedPointSnapshot`.

    Its coordinates, its effects as raw effects and, where the point holds its
    accepted normalization, the normalized effects, references and shifts; no
    profile, recovery or inner convergence exists for it (``converged`` ``None``).
    """
    b = _compute_beta(sample_snapshot, rec.theta)
    s, k, pk = sample_snapshot, int(rec.theta.size), sample_snapshot.partition_key
    norm = rec.normalization
    ne, rf, sh = (
        (None, None, None)
        if norm is None
        else (norm.effects, norm.references, norm.shifts)
    )
    r1 = (s, var, inn, k, None, rec.theta, b, rec.base, rec.index)
    r2 = (None, None, None, None, _tuple_arrays(rec.effects), ne, rf, sh, None)
    return RetainedPointSnapshot(
        pkey, s.key, pk, "root", *r1, *r2, _coordinates=rec.alpha
    )


def _snapshot_linear_point(
    rec: LinearPoint,
    sample_snapshot: RetainedSampleSnapshot,
    *,
    inn: InnerControls | None,
    pkey: PointKey,
    linear_normalization: PointNormalization | None = None,
    equations: FreshEquations | None = None,
    sample_map: SampleIdentityMap | None = None,
) -> RetainedPointSnapshot:
    """Detach a ``LinearPoint`` record into a :class:`RetainedPointSnapshot`."""
    actual = (
        sample_map.capture(rec.sample) if sample_map is not None else sample_snapshot
    )
    skey, k = actual.key, int(rec.theta.size)
    sn = snapshot_linear_normalization(
        linear_normalization, point_key=pkey, sample_key=skey, sample_map=sample_map
    )
    feq = snapshot_fresh_equations(
        equations, point_key=pkey, sample_key=skey, sample_map=sample_map
    )
    ne, rf, sh = (
        (None, None, None) if sn is None else (sn.effects, sn.references, sn.shifts)
    )
    r1 = (actual, None, inn, k, True, rec.theta, rec.beta, rec.base, rec.index)
    r2 = (None, None, None, sn, _tuple_arrays(rec.effects), ne, rf, sh, feq)
    rest = (*r1, *r2)
    return RetainedPointSnapshot(pkey, skey, actual.partition_key, "linear", *rest)


def snapshot_point_record(
    point: AdmittedPoint | RawPointRecord,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    variance: Family | None = None,
    inner: InnerControls | None = None,
    linear_normalization: PointNormalization | None = None,
    point_index: int = 0,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedPointSnapshot:
    """Detach an admitted or raw point record into a :class:`RetainedPointSnapshot`."""
    rec, var, inn = (
        (point.record, point.variance, point.inner)
        if isinstance(point, AdmittedPoint)
        else (point, variance, inner)
    )
    pkey = PointKey(int(point_index))
    if isinstance(rec, PointProfile):
        return _snapshot_profiled_point(
            rec, sample_snapshot, var=var, inn=inn, pkey=pkey, grouping_map=grouping_map
        )
    if isinstance(rec, ConditionalPoint):
        # No completed all-FE profile is claimed for an intermediate trial.
        # Full effects and q remain detached and available to rich accessors.
        s, k = sample_snapshot, int(rec.theta.size)
        return RetainedPointSnapshot(
            pkey, s.key, s.partition_key, "reduced", s, var, inn, k, None,
            rec.theta, _compute_beta(s, rec.theta), rec.base, rec.index,
            scalar_work=rec.scalar, _raw_effects=rec.state.coefficients,
            _coordinates=rec.coordinates,
        )
    if isinstance(rec, IndexPoint):
        return _snapshot_index_point(
            rec, sample_snapshot, var=var, inn=inn, pkey=pkey, grouping_map=grouping_map
        )
    if isinstance(rec, RootPoint):
        return _snapshot_root_point(rec, sample_snapshot, var=var, inn=inn, pkey=pkey)
    if isinstance(rec, LinearPoint):
        return _snapshot_linear_point(
            rec,
            sample_snapshot,
            inn=inn,
            pkey=pkey,
            linear_normalization=linear_normalization,
            sample_map=sample_map,
        )
    raise TypeError(f"unsupported point record type {type(rec)!r}")


class PointIdentityMap:
    """Temporary identity map assigning one ``PointKey`` per original point object."""

    def __init__(
        self,
        *,
        sample_snapshot: RetainedSampleSnapshot | None = None,
        variance: Family | None = None,
        inner: InnerControls | None = None,
        selected_raw: RawPointRecord | None = None,
        selected_snapshot: RetainedPointSnapshot | None = None,
        sample_map: SampleIdentityMap | None = None,
        grouping_map: GroupingIdentityMap | None = None,
    ) -> None:
        """Initialize the identity map and optionally seed the selected point."""
        self._sample = sample_snapshot
        self._variance = variance
        self._inner = inner
        self._sample_map = sample_map
        self._grouping_map = grouping_map
        self._by_id: dict[int, RetainedPointSnapshot] = {}
        self._refs: list[object] = []
        self._next_key: PointKey = 0
        if selected_raw is not None and selected_snapshot is not None:
            self.register_selected(selected_raw, selected_snapshot)

    def bind_context(
        self,
        *,
        sample_snapshot: RetainedSampleSnapshot,
        variance: Family | None,
        inner: InnerControls | None,
        selected_raw: RawPointRecord | None = None,
        selected_snapshot: RetainedPointSnapshot | None = None,
    ) -> PointIdentityMap:
        """Update default sample/variance/inner context and seed selected point."""
        self._sample = sample_snapshot
        self._variance = variance
        self._inner = inner
        if selected_raw is not None and selected_snapshot is not None:
            self.register_selected(selected_raw, selected_snapshot)
        return self

    def allocate_key(self, raw: RawPointRecord | None = None) -> PointKey:
        """Return existing key for ``raw`` if already mapped, else allocate next key."""
        if raw is not None and id(raw) in self._by_id:
            return self._by_id[id(raw)].key
        idx = self._next_key
        self._next_key += 1
        return idx

    def register_selected(
        self, selected_raw: RawPointRecord, selected_snapshot: RetainedPointSnapshot
    ) -> RetainedPointSnapshot:
        """Seed or update ``selected_raw`` with ``selected_snapshot``."""
        self._by_id[id(selected_raw)] = selected_snapshot
        self._refs.append(selected_raw)
        self._next_key = max(self._next_key, int(selected_snapshot.key) + 1)
        return selected_snapshot

    def get(
        self,
        raw: RawPointRecord,
        *,
        sample_snapshot: RetainedSampleSnapshot | None = None,
        variance: Family | None = None,
        inner: InnerControls | None = None,
        linear_normalization: PointNormalization | None = None,
        equations: FreshEquations | None = None,
    ) -> RetainedPointSnapshot:
        """Return the single :class:`RetainedPointSnapshot` for ``raw``."""
        oid = id(raw)
        existing = self._by_id.get(oid)
        if existing is not None:
            return existing
        samp = sample_snapshot if sample_snapshot is not None else self._sample
        if samp is None:
            raise ValueError("sample_snapshot is required to detach a point record")
        var = variance if variance is not None else self._variance
        inn = inner if inner is not None else self._inner
        idx = self._next_key
        self._next_key += 1
        if isinstance(raw, LinearPoint):
            snap = _snapshot_linear_point(
                raw,
                samp,
                inn=inn,
                pkey=idx,
                linear_normalization=linear_normalization,
                equations=equations,
                sample_map=self._sample_map,
            )
        else:
            snap = snapshot_point_record(
                raw,
                sample_snapshot=samp,
                variance=var,
                inner=inn,
                linear_normalization=linear_normalization,
                point_index=idx,
                sample_map=self._sample_map,
                grouping_map=self._grouping_map,
            )
        self._by_id[oid] = snap
        self._refs.append(raw)
        return snap

    def clear_anchors(self) -> None:
        """Release strong references to original point objects after capture."""
        self._refs.clear()
