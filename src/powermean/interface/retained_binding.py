"""Concrete positional metadata binding from a prepared frame and estimation sample.

This module binds retained dependence metadata to
a concrete ``OrderSample``, ``FESample``, or ``RetainedSampleSnapshot`` from a concrete
``PreparedFrame`` and explicit request fields without ``_SampleLike`` duck-typing or
importing :mod:`powermean.interface.retention`.

The binding and the check read only the base, labels and dependence provenance.
Their authoritative logic accepts the original frame, preserving its observation
order, or fresh :class:`~powermean.results.primitive_snapshot.MetadataParts` from a
primitive capture without rebuilding a frame.  In particular, a frame's labels
are not copied until the preceding validation has succeeded.
"""

from __future__ import annotations

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.dependence import UsedDependence
from powermean.data.inference_requests import ConleyRequest, HacRequest
from powermean.data.raw import AnyArray
from powermean.data.retained import OrderSample
from powermean.data.sample import SampleBase
from powermean.data.sample_fe_state import FESample, same_bits
from powermean.data.weights import ObservationWeights, WeightKind
from powermean.interface.retained_metadata import (
    BoundDependence,
    RetainedColumnMetadata,
    _take_owned_column,
)
from powermean.interface.retained_space import RetainedSpaceView
from powermean.interface.retained_time import RetainedTimeView
from powermean.model.order import IndexArray
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_sample_binding import UNBOUND
from powermean.results.owned import owned_array
from powermean.results.primitive_snapshot import MetadataParts
from powermean.results.retained_input_values import (
    InputKey,
    SampleKey,
    _same_column_values,
)

SampleTarget = OrderSample | FESample | RetainedSampleSnapshot

__all__ = [
    "BoundDependence",
    "RetainedColumnMetadata",
    "SampleTarget",
    "_bind_metadata_parts",
    "_check_dependence_against_parts",
    "_check_dependence_against_prepared",
    "_collect_bound_columns",
    "_extract_frequency_counts",
    "_resolve_initial_views",
    "_same_initial_space_view",
    "_same_initial_time_view",
    "_verify_sample_column",
    "bind_sample_metadata",
]


def _validate_sample_target(
    sample: object,
) -> tuple[SampleTarget, int, SampleKey | None]:
    """Validate concrete sample union and extract ``(sample, nobs, sample_key)``."""
    if type(sample) is OrderSample:
        return sample, int(sample.nobs), None
    if type(sample) is FESample:
        return sample, int(sample.ledger.nobs), None
    if type(sample) is RetainedSampleSnapshot:
        return sample, int(sample.ledger.nobs), sample.key
    raise TypeError(
        "sample must be a concrete OrderSample, FESample, or RetainedSampleSnapshot"
    )


def _extract_frequency_counts(
    base: SampleBase,
    positions: IndexArray,
    weights: ObservationWeights | None = None,
) -> tuple[WeightKind | None, tuple[int, ...] | None]:
    """Extract exact integer frequency counts at ``positions`` when applicable."""
    raw_weights = base.inputs.weights
    if raw_weights is None:
        if weights is not None and weights.kind == "frequency":
            return "frequency", tuple(int(v) for v in weights.values.tolist())
        return (None if weights is None else weights.kind), None
    if raw_weights.kind != "frequency":
        return raw_weights.kind, None
    return "frequency", tuple(int(v) for v in raw_weights.values[positions].tolist())


def _verify_sample_column(sample: SampleTarget, name: str, expected: AnyArray) -> None:
    """Verify that ``sample.metadata[name]`` matches ``expected`` when present."""
    if name not in sample.metadata.columns:
        return
    sample_vals = sample.metadata[name]
    if sample_vals.shape != expected.shape or not _same_column_values(
        sample_vals, expected
    ):
        raise ValueError(f"sample.metadata[{name!r}] does not match prepared column")


def _collect_bound_columns(
    parts: MetadataParts | PreparedFrame, sample: SampleTarget, pos: IndexArray
) -> dict[str, RetainedColumnMetadata]:
    """Collect and verify sample-aligned metadata columns from ``parts``."""
    dep = parts.dependence
    cols: dict[str, RetainedColumnMetadata] = {}
    if dep is not None:
        for name, prov in dep.columns.items():
            taken = prov.take(pos)
            _verify_sample_column(sample, name, taken)
            cols[name] = RetainedColumnMetadata(
                name=name,
                roles=prov.roles,
                used_in_fit=prov.used_in_fit,
                source_dtype=prov.source_dtype,
                pandas_kind=prov.pandas_kind,
                full_missing_count=int(np.count_nonzero(prov.missing_mask)),
                values=taken,
            )
    for name in sample.metadata:
        if name not in cols:
            raw_col = parts.base.inputs.metadata.columns.get(name)
            if raw_col is None:
                raise ValueError(
                    f"sample.metadata[{name!r}] not found in prepared base"
                )
            taken = raw_col.take(pos).values
            _verify_sample_column(sample, name, taken)
            cols[name] = RetainedColumnMetadata(
                name=name,
                roles=("retained",),
                used_in_fit=False,
                source_dtype=str(taken.dtype),
                pandas_kind=taken.dtype.kind,
                full_missing_count=0,
                values=taken,
            )
    return cols


def _resolve_initial_views(
    bound: BoundDependence,
    used: UsedDependence,
    *,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
) -> tuple[RetainedTimeView | None, RetainedSpaceView | None]:
    """Build optional initial time and space views from explicit fit-time requests."""
    init_time = (
        bound.time_view(hac_request)
        if hac_request is not None
        else (
            bound.time_view(time=used.time, unit=used.unit)
            if used.time is not None
            else None
        )
    )
    coords = used.coordinates
    init_space = (
        bound.space_view(conley_request)
        if conley_request is not None
        else (
            bound.space_view(latitude=coords[0], longitude=coords[1])
            if coords is not None
            else None
        )
    )
    return init_time, init_space


def bind_sample_metadata(
    prepared: PreparedFrame,
    sample: SampleTarget,
    *,
    weights: ObservationWeights | None = None,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    input_key: InputKey | None = None,
    sample_key: SampleKey | None = None,
) -> BoundDependence:
    """Bind retained dependence metadata to ``sample`` from concrete ``prepared``."""
    if type(prepared) is not PreparedFrame:
        raise TypeError("prepared must be a concrete PreparedFrame")
    return _bind_metadata_parts(
        prepared,
        sample,
        weights=weights,
        hac_request=hac_request,
        conley_request=conley_request,
        input_key=input_key,
        sample_key=sample_key,
    )


def _bind_metadata_parts(
    parts: MetadataParts | PreparedFrame,
    sample: SampleTarget,
    *,
    weights: ObservationWeights | None = None,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    input_key: InputKey | None = None,
    sample_key: SampleKey | None = None,
) -> BoundDependence:
    """Bind retained dependence metadata to ``sample`` from the frame parts it reads.

    The authoritative binding behind :func:`bind_sample_metadata`: ``parts`` is
    the original frame, read in validation order, or fresh metadata parts from a
    primitive capture.  General frames must not eagerly materialize their labels.
    """
    checked_sample, nobs, inferred_skey = _validate_sample_target(sample)
    if nobs != parts.base.nobs:
        raise ValueError("sample.nobs does not match prepared.base.nobs")
    pos = owned_array(np.asarray(checked_sample.positions, dtype=np.intp))
    if pos.ndim != 1 or pos.size != checked_sample.count:
        raise ValueError("sample.positions does not match sample.count")
    if pos.size > 0 and (int(np.min(pos)) < 0 or int(np.max(pos)) >= nobs):
        raise ValueError("sample.positions out of bounds for prepared.base.nobs")
    if pos.size > 1 and not bool(np.all(pos[1:] > pos[:-1])):
        raise ValueError("sample.positions must be strictly increasing")

    dep = parts.dependence
    used = dep.used if dep is not None else UsedDependence()
    cols = _collect_bound_columns(parts, checked_sample, pos)
    kind, freqs = _extract_frequency_counts(parts.base, pos, weights)
    labels = parts.labels.take(pos)
    resolved_skey = sample_key if sample_key is not None else inferred_skey
    bound = BoundDependence(
        nobs=nobs,
        used_dependence=used,
        columns=cols,
        _positions=pos,
        _labels=labels,
        weight_kind=kind,
        frequency_counts=freqs,
        input_key=input_key,
        sample_key=resolved_skey,
    )
    t_view, s_view = _resolve_initial_views(
        bound, used, hac_request=hac_request, conley_request=conley_request
    )
    if (t_view, s_view) == (None, None):
        return bound
    return BoundDependence(
        nobs=nobs,
        used_dependence=used,
        columns=cols,
        _positions=pos,
        _labels=labels,
        weight_kind=kind,
        frequency_counts=freqs,
        _initial_time_view=t_view,
        _initial_space_view=s_view,
        input_key=input_key,
        sample_key=resolved_skey,
    )



def _bind_owned_metadata_parts(
    parts: PreparedFrame, sample: FESample | OrderSample, *,
    weights: ObservationWeights | None = None,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    input_key: InputKey | None = None, sample_key: SampleKey | None = None,
) -> BoundDependence:
    """Normal NumericExecution's already admitted private FINAL-row metadata.

    Never call FESample.metadata here: it reconstructs every column on each access.
    The context has already admitted the sample and proved its scalar/copy-free
    source columns. Select each authoritative source once. Foreign/callback entries
    continue through _bind_metadata_parts and all original comparison/copy hooks.
    """
    pos = owned_array(sample.positions)
    nobs = sample.nobs if type(sample) is OrderSample else sample.ledger.nobs
    dep = parts.dependence
    used = dep.used if dep is not None else UsedDependence()
    columns: dict[str, RetainedColumnMetadata] = {}
    if dep is not None:
        for name, prov in dep.columns.items():
            if "_retained_sources" in vars(prov):
                columns[name] = RetainedColumnMetadata._from_retained(prov, pos)
                continue
            column = _take_owned_column(prov._column_record, pos)
            columns[name] = RetainedColumnMetadata._from_owned_column(
                name=name, roles=prov.roles, used_in_fit=prov.used_in_fit,
                source_dtype=prov.source_dtype, pandas_kind=prov.pandas_kind,
                full_missing_count=int(np.count_nonzero(prov._missing_mask)),
                column=column,
            )
    for name, raw in parts.base.inputs.metadata.columns.items():
        if name not in columns:
            column = _take_owned_column(raw, pos)
            columns[name] = RetainedColumnMetadata._from_owned_column(
                name=name, roles=("retained",), used_in_fit=False,
                source_dtype=str(column.dtype), pandas_kind=column.dtype.kind,
                full_missing_count=0, column=column,
            )
    kind, frequencies = _extract_frequency_counts(parts.base, pos, weights)
    labels = parts._labels.take(pos)
    bound = BoundDependence._from_owned(
        nobs=nobs, used_dependence=used, columns=columns,
        positions=pos, labels=labels, weight_kind=kind, frequency_counts=frequencies,
        input_key=input_key, sample_key=sample_key,
    )
    time, space = _resolve_initial_views(
        bound, used, hac_request=hac_request, conley_request=conley_request
    )
    if time is None and space is None:
        return bound
    return BoundDependence._from_owned(
        nobs=nobs, used_dependence=used, columns=columns,
        positions=pos, labels=labels, weight_kind=kind, frequency_counts=frequencies,
        input_key=input_key, sample_key=sample_key, initial_time=time, initial_space=space,
    )


def _same_initial_time_view(
    got: RetainedTimeView | None, exp: RetainedTimeView | None
) -> bool:
    """Return True if ``got`` and ``exp`` agree on every recorded time-view field."""
    if got is None or exp is None:
        return got is None and exp is None
    if (
        type(got) is not RetainedTimeView
        or got.time_column != exp.time_column
        or got.unit_column != exp.unit_column
        or got.step != exp.step
        or got.lag != exp.lag
        or got.sorted_times != exp.sorted_times
        or got.within_unit_diffs != exp.within_unit_diffs
        or got.tie_count != exp.tie_count
        or got.gap_count != exp.gap_count
        or got.has_ties != exp.has_ties
        or got.has_gaps != exp.has_gaps
        or got.min_positive_step != exp.min_positive_step
        or got.max_step != exp.max_step
        or got.has_frequency_replicas != exp.has_frequency_replicas
        or got.max_frequency_count != exp.max_frequency_count
        or not _same_column_values(got.raw_time, exp.raw_time)
        or not same_bits(got.order, exp.order)
        or not same_bits(got.inverse, exp.inverse)
        or not same_bits(got.unit_starts, exp.unit_starts)
        or not same_bits(got.unit_stops, exp.unit_stops)
        or (got.raw_unit is None) != (exp.raw_unit is None)
        or (got.unit_codes is None) != (exp.unit_codes is None)
        or (got.unit_labels is None) != (exp.unit_labels is None)
    ):
        return False
    if (
        got.raw_unit is not None
        and exp.raw_unit is not None
        and not _same_column_values(got.raw_unit, exp.raw_unit)
    ):
        return False
    if (
        got.unit_codes is not None
        and exp.unit_codes is not None
        and not same_bits(got.unit_codes, exp.unit_codes)
    ):
        return False
    return not (
        got.unit_labels is not None
        and exp.unit_labels is not None
        and not _same_column_values(
            np.array(got.unit_labels, dtype=object),
            np.array(exp.unit_labels, dtype=object),
        )
    )


def _same_initial_space_view(
    got: RetainedSpaceView | None, exp: RetainedSpaceView | None
) -> bool:
    """Return True if ``got`` and ``exp`` agree on every recorded space-view field."""
    if got is None or exp is None:
        return got is None and exp is None
    return (
        type(got) is RetainedSpaceView
        and got.latitude_column == exp.latitude_column
        and got.longitude_column == exp.longitude_column
        and got.n_locations == exp.n_locations
        and got.row_multiplicities == exp.row_multiplicities
        and got.location_replica_counts == exp.location_replica_counts
        and got.repeated_location_count == exp.repeated_location_count
        and got.requires_pooling_confirmation == exp.requires_pooling_confirmation
        and got.pool_times_confirmed == exp.pool_times_confirmed
        and got.cutoff == exp.cutoff
        and got.units == exp.units
        and got.distance == exp.distance
        and same_bits(got.raw_latitude, exp.raw_latitude)
        and same_bits(got.raw_longitude, exp.raw_longitude)
        and same_bits(got.canonical_latitude, exp.canonical_latitude)
        and same_bits(got.canonical_longitude, exp.canonical_longitude)
        and same_bits(got.unique_canonical_latitude, exp.unique_canonical_latitude)
        and same_bits(got.unique_canonical_longitude, exp.unique_canonical_longitude)
        and same_bits(got.row_to_location, exp.row_to_location)
        and same_bits(got.location_row_counts, exp.location_row_counts)
        and len(got.location_to_rows) == len(exp.location_to_rows)
        and all(
            same_bits(a, b)
            for a, b in zip(got.location_to_rows, exp.location_to_rows, strict=True)
        )
    )


def _check_dependence_against_prepared(
    dependence: BoundDependence,
    sample: FESample,
    prepared: PreparedFrame,
    *,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    input_key: InputKey | None = None,
    sample_key: SampleKey | None = None,
) -> None:
    """Verify that ``dependence`` columns, labels, and views match ``prepared``."""
    _check_dependence_against_parts(
        dependence,
        sample,
        prepared,
        hac_request=hac_request,
        conley_request=conley_request,
        input_key=input_key,
        sample_key=sample_key,
    )


def _check_dependence_against_parts(
    dependence: BoundDependence,
    sample: FESample,
    parts: MetadataParts | PreparedFrame,
    *,
    hac_request: HacRequest | None = None,
    conley_request: ConleyRequest | None = None,
    input_key: InputKey | None = None,
    sample_key: SampleKey | None = None,
) -> None:
    """Verify that ``dependence`` columns, labels, and views match the frame ``parts``.

    The authoritative check behind :func:`_check_dependence_against_prepared`, on
    the original frame or fresh primitive metadata parts.  Original frame access
    remains lazy so earlier refusals precede any custom-label copy callbacks.
    """
    if (
        dependence.nobs != parts.base.nobs
        or dependence.count != sample.count
        or not same_bits(dependence.positions, sample.positions)
    ):
        raise ValueError(UNBOUND)
    if (
        input_key is not None
        and dependence.input_key is not None
        and dependence.input_key != input_key
    ):
        raise ValueError(UNBOUND)
    if (
        sample_key is not None
        and dependence.sample_key is not None
        and dependence.sample_key != sample_key
    ):
        raise ValueError(UNBOUND)
    exp_kind, exp_freqs = _extract_frequency_counts(
        parts.base, sample.positions, sample.data.weights
    )
    if dependence.weight_kind != exp_kind or dependence.frequency_counts != exp_freqs:
        raise ValueError(UNBOUND)
    exp_used = (
        parts.dependence.used if parts.dependence is not None else UsedDependence()
    )
    if dependence.used_dependence != exp_used:
        raise ValueError(UNBOUND)
    exp_labels = parts.labels.take(sample.positions)
    if (
        not dependence.labels.equals(exp_labels)
        or dependence.labels.dtype != exp_labels.dtype
        or dependence.labels.name != exp_labels.name
        or list(dependence.labels.names) != list(exp_labels.names)
    ):
        raise ValueError(UNBOUND)
    expected_cols = _collect_bound_columns(parts, sample, sample.positions)
    if set(dependence.columns) != set(expected_cols):
        raise ValueError(UNBOUND)
    for name, exp_meta in expected_cols.items():
        got = dependence.columns[name]
        if (
            got.name != exp_meta.name
            or (got.roles, got.used_in_fit, got.source_dtype, got.pandas_kind)
            != (
                exp_meta.roles,
                exp_meta.used_in_fit,
                exp_meta.source_dtype,
                exp_meta.pandas_kind,
            )
            or (got.full_missing_count, got.sample_missing_count)
            != (exp_meta.full_missing_count, exp_meta.sample_missing_count)
            or not _same_column_values(got.values, exp_meta.values)
        ):
            raise ValueError(UNBOUND)
    exp_time, exp_space = _resolve_initial_views(
        dependence,
        exp_used,
        hac_request=hac_request,
        conley_request=conley_request,
    )
    if not _same_initial_time_view(
        dependence.initial_time_view, exp_time
    ) or not _same_initial_space_view(dependence.initial_space_view, exp_space):
        raise ValueError(UNBOUND)

