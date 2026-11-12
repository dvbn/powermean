"""Detached FE sample snapshots and temporary per-handoff sample identity map.

This module detaches admitted :class:`FESample` instances into finite typed
:class:`RetainedSampleSnapshot` records with result-local keys and provides
:class:`SampleIdentityMap` for temporary per-handoff deduplication by original object
identity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from powermean._prepare.owned import _owned_index
from powermean.data.accounting import MissingAccount
from powermean.data.design import ScaledDesign
from powermean.data.fingerprint import sample_fingerprint
from powermean.data.inputs import RegressionData
from powermean.data.numeric_fit import NumericFit, _immutable_bytes
from powermean.data.raw import AlignedColumns, _OwnedMapping
from powermean.data.retained import OrderSample
from powermean.data.sample import SampleBase
from powermean.data.sample_fe_state import FESample, RowLedger
from powermean.data.weights import ObservationWeights
from powermean.inference.absorbed import sampling_units
from powermean.model.order import IndexArray, Selection, TransformedProblem
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.absorbed_partitions import (
    GroupingIdentityMap,
    GroupingKey,
    PartitionIdentityMap,
    PartitionLabelsView,
    RetainedBlockSnapshot,
    RetainedGroupingSnapshot,
    RetainedPartitionSnapshot,
    RetainedTopologySnapshot,
    _owned_components,
    detach_grouping,
    validate_partition_lineage,
)
from powermean.results.owned import (
    _owned_data,
    owned_array,
    owned_units,
)
from powermean.results.retained_input_values import (
    InputKey,
    PartitionKey,
    RetainedInputSnapshot,
    SampleKey,
    _owned_row_ledger,
    _owned_scaled_design,
    _owned_transformed_problem,
    owned_order_scale,
)
from powermean.results.retained_inputs import InputIdentityMap
from powermean.systems.joint import SamplingUnits

__all__ = [
    "GroupingIdentityMap",
    "GroupingKey",
    "PartitionIdentityMap",
    "PartitionLabelsView",
    "RetainedBlockSnapshot",
    "RetainedGroupingSnapshot",
    "RetainedPartitionSnapshot",
    "RetainedSampleSnapshot",
    "RetainedTopologySnapshot",
    "SampleIdentityMap",
    "_owned_components",
    "_owned_row_ledger",
    "_owned_scaled_design",
    "_owned_transformed_problem",
    "detach_grouping",
    "snapshot_fe_sample",
    "validate_partition_lineage",
]


def _resolve_sample_partition(
    sample: FESample,
    nobs: int,
    partition_snapshot: RetainedPartitionSnapshot | None,
) -> tuple[
    RetainedPartitionSnapshot | None,
    RetainedTopologySnapshot | None,
    PartitionKey | None,
]:
    """Detach or bind the partition and topology snapshots for ``sample``."""
    if sample.source is None or sample.grouping is None:
        return None, None, None
    part = partition_snapshot or RetainedPartitionSnapshot(
        sample.source, sample.grouping, nobs
    )
    # The topology owns its own copy of the ordinals at its ingress, from the
    # partition's owned storage, not from an outward copy copied once more.
    topo = RetainedTopologySnapshot(
        sample.grouping.blocks,
        sample.grouping.n_rows,
        source_ordinals=part._ordinals,
    )
    return part, topo, part.key


def _plain_fingerprint_metadata(data: RegressionData) -> bool:
    """Do not omit observable encode hooks on accepted string subclasses.

    RegressionData already requires exact int/str row labels and owns numeric
    arrays. Only the small column-name tuple and weight-kind string admit
    subclasses; keep their original repeated fresh encodings, without invoking
    them to decide eligibility.
    """
    return all(type(name) is str for name in data.column_names) and (
        data.weights is None or type(data.weights.kind) is str
    )


def _share_sample_values(sample: FESample, order: OrderSample | None) -> bool:
    """Eligibility for the SAME checked private producer, never a public record.

    Changed rows/active design, subclasses, callbacks and mutable/partial storage
    keep the existing constructors. This inspects only types and byte ownership,
    not array values or a stored validity token. The private admission caller has
    already established the actual operation lifetime and coordinator provenance.
    """
    if (type(order) is not OrderSample or sample.data is not order.data
            or sample.problem is not order.problem
            or sample.design is not order.design):
        return False
    ledger, data, problem, design = (
        sample.ledger, sample.data, sample.problem, sample.design
    )
    if (type(ledger) is not RowLedger or type(data) is not RegressionData
            or type(problem) is not TransformedProblem
            or type(design) is not ScaledDesign):
        return False
    if (type(ledger.missing) is not MissingAccount
            or type(ledger.missing.groups) is not _OwnedMapping
            or type(problem.selection) is not Selection
            or (data.weights is not None
                and type(data.weights) is not ObservationWeights)
            or not _plain_fingerprint_metadata(data)):
        return False
    selection = problem.selection
    arrays = [(ledger.codes, np.int8, 1), (data.design, np.float64, 2),
              (design.matrix, np.float64, 2)]
    arrays.extend((a, np.intp, 1) for a in (
        ledger.blocks, ledger.positions, ledger.retained_index,
        selection.positions, selection.removed,
    ))
    arrays.extend((a, np.bool_, 1) for a in (
        ledger.missing.excluded, *ledger.missing.groups.values(), selection.mask,
    ))
    arrays.extend((a, np.float64, 1) for a in (
        data.y, data.offset, problem.response, problem.offset, design.scale,
        data.exposure, None if data.weights is None else data.weights.values,
    ) if a is not None)
    if ledger.replicas is not None:
        arrays.append((ledger.replicas, np.intp, 1))
    return all(type(a) is np.ndarray and a.dtype == dtype and a.ndim == ndim
               and a.dtype.fields is None and a.dtype.metadata is None
               and _immutable_bytes(a) is not None for a, dtype, ndim in arrays)


def _ledger_headers(ledger: RowLedger) -> RowLedger:
    """Keep exact admitted ledger facts; detach every header and the mask mapping."""
    missing = MissingAccount._owned(
        {name: owned_array(mask) for name, mask in ledger.missing.groups.items()},
        owned_array(ledger.missing.excluded),
    )
    result = object.__new__(RowLedger)
    for name, value in (
        ("codes", owned_array(ledger.codes)), ("blocks", owned_array(ledger.blocks)),
        ("positions", owned_array(ledger.positions)),
        ("retained_index", owned_array(ledger.retained_index)), ("missing", missing),
        ("replicas", None if ledger.replicas is None else owned_array(ledger.replicas)),
    ):
        object.__setattr__(result, name, value)
    return result


def _data_headers(data: RegressionData) -> RegressionData:
    """Detach raw-data headers without replaying checked weight/exposure/finite work."""
    weights = None
    if data.weights is not None:
        weights = object.__new__(ObservationWeights)
        object.__setattr__(weights, "values", owned_array(data.weights.values))
        object.__setattr__(weights, "kind", data.weights.kind)
    result = object.__new__(RegressionData)
    for name, value in (
        ("y", owned_array(data.y)), ("design", owned_array(data.design)),
        ("offset", owned_array(data.offset)),
        ("exposure", None if data.exposure is None else owned_array(data.exposure)),
        ("weights", weights), ("row_labels", data.row_labels),
        ("column_names", data.column_names),
    ):
        object.__setattr__(result, name, value)
    return result


def _problem_headers(problem: TransformedProblem) -> TransformedProblem:
    """Retain the exact transform and selection, with no shared mutable header."""
    s = problem.selection
    selection = Selection(
        owned_array(s.mask), owned_array(s.positions), owned_array(s.removed),
        s.nobs, s.positive_only,
    )
    result = object.__new__(TransformedProblem)
    for name, value in (
        ("omega", problem.omega), ("route", problem.route),
        ("response", owned_array(problem.response)),
        ("offset", owned_array(problem.offset)), ("selection", selection),
    ):
        object.__setattr__(result, name, value)
    return result


def _design_headers(design: ScaledDesign | None) -> ScaledDesign | None:
    """The existing scaled values/names, with fresh independent array metadata."""
    if design is None:
        return None
    return ScaledDesign(
        owned_array(design.matrix), owned_array(design.scale), design.column_names
    )


@dataclass(frozen=True, init=False)
class RetainedSampleSnapshot:
    """Detached snapshot of an admitted :class:`FESample` with result-local keys."""

    key: SampleKey
    input_key: InputKey
    partition_key: PartitionKey | None
    omega: float
    positive_only: bool
    conditioned: bool
    nobs: int
    count: int
    effective_n: int
    fingerprint: str
    reasons: tuple[str | None, ...]
    active_names: tuple[str, ...]
    _active: IndexArray
    input_snapshot: RetainedInputSnapshot
    partition: RetainedPartitionSnapshot | None
    topology: RetainedTopologySnapshot | None
    _ledger: RowLedger
    _data: RegressionData
    _problem: TransformedProblem
    _design: ScaledDesign | None
    _order_scale: OrderScale | None
    _units: SamplingUnits
    _labels: pd.Index[Any]

    def __init__(
        self,
        sample: FESample,
        *,
        key: SampleKey = 0,
        labels: pd.Index[Any] | None = None,
        input_snapshot: RetainedInputSnapshot | None = None,
        partition_snapshot: RetainedPartitionSnapshot | None = None,
    ) -> None:
        """Admit and detach an ``FESample`` without retaining live sample references."""
        RetainedSampleSnapshot._initialize(
            self, sample, key=key, labels=labels, input_snapshot=input_snapshot,
            partition_snapshot=partition_snapshot, owned_order=None,
        )

    def __getstate__(self) -> Any:
        """Export declared rich state without carrying a private owner recipe.

        Generic/subclass copying retains object.__getstate__, including slots and
        existing reduction hooks. Only this exact private native view expands.
        """
        state = object.__getstate__(self)
        if type(self) is RetainedSampleSnapshot and "_numeric_grouping" in vars(self):
            state = dict(vars(self))
            for field in fields(self):
                state[field.name] = getattr(self, field.name)
            for name in ("_numeric_grouping",):
                state.pop(name, None)
            # The requested legacy fields now ARE the owned state. Future rich
            # copies do not reconstruct another full-row projection.
            vars(self).update(state)
            for name in tuple(vars(self)):
                if name not in state:
                    vars(self).pop(name)
        return state

    @classmethod
    def _from_owned_order(
        cls, sample: FESample, order: OrderSample, *, key: SampleKey,
        input_snapshot: RetainedInputSnapshot,
        partition_snapshot: RetainedPartitionSnapshot | None,
    ) -> RetainedSampleSnapshot:
        """Detach one normal operation's fresh order/sample producer pair.

        No caller-owned hash or numerical buffers are adopted. The authoritative
        fingerprint already computed for these exact raw data can be shared as
        an immutable string; changed final rows take the usual fresh encoder.
        """
        result = object.__new__(cls)
        RetainedSampleSnapshot._initialize(
            result, sample, key=key, labels=None, input_snapshot=input_snapshot,
            partition_snapshot=partition_snapshot, owned_order=order,
        )
        return result

    @classmethod
    def _from_numeric(
        cls, sample: FESample, order: OrderSample, numeric: NumericFit, *,
        key: SampleKey, input_snapshot: RetainedInputSnapshot,
        partition_snapshot: RetainedPartitionSnapshot,
    ) -> RetainedSampleSnapshot:
        """Compatible retained shell over the authoritative numeric operation."""
        result = object.__new__(cls)
        result._initialize(
            sample, key=key, labels=None, input_snapshot=input_snapshot,
            partition_snapshot=partition_snapshot, owned_order=order, numeric=numeric,
        )
        return result

    def __getattribute__(self, name: str) -> Any:
        """Resolve exact-native presentation fields; generic lookup stays unchanged."""
        if type(self) is RetainedSampleSnapshot and type(name) is str:
            state = object.__getattribute__(self, "__dict__")
            if "_numeric_grouping" in state and name not in state:
                if name == "_labels":
                    return self.input_snapshot.labels_at(self._ledger.positions)
                if name == "topology":
                    part = self.partition
                    if part is None:
                        return None
                    return RetainedTopologySnapshot(
                        self._numeric_grouping.blocks, self.count,
                        source_ordinals=part._ordinals,
                    )
        return object.__getattribute__(self, name)

    def _initialize(
        self, sample: FESample, *, key: SampleKey, labels: pd.Index[Any] | None,
        input_snapshot: RetainedInputSnapshot | None,
        partition_snapshot: RetainedPartitionSnapshot | None,
        owned_order: OrderSample | None,
        numeric: NumericFit | None = None,
    ) -> None:
        """Own all snapshot fields at the existing capture boundary, in order."""
        if type(sample) is not FESample:
            raise TypeError("sample must be an FESample")
        share = _share_sample_values(sample, owned_order)
        ledger = (_ledger_headers(sample.ledger) if share
                  else _owned_row_ledger(sample.ledger))
        data = _data_headers(sample.data) if share else _owned_data(sample.data)
        in_snap = input_snapshot or RetainedInputSnapshot(
            sample.base, labels=pd.Index(range(ledger.nobs))
        )
        lbls = None if numeric is not None else (
            _owned_index(labels)
            if labels is not None
            else (
                in_snap.labels_at(ledger.positions)
                if input_snapshot is not None
                else pd.Index(ledger.positions.tolist())
            )
        )
        if lbls is not None and len(lbls) != ledger.count:
            raise ValueError("labels length must match sample.count")
        if numeric is None:
            part, topo, p_key = _resolve_sample_partition(
                sample, ledger.nobs, partition_snapshot
            )
        else:
            part, topo = partition_snapshot, None
            assert part is not None
            p_key = part.key
            object.__setattr__(self, "_numeric_grouping", numeric.grouping)

        for attr, val in (
            ("key", int(key)),
            ("input_key", in_snap.key),
            ("partition_key", p_key),
            ("omega", sample.omega),
            ("positive_only", sample.positive_only),
            ("conditioned", sample.conditioned),
            ("nobs", ledger.nobs),
            ("count", ledger.count),
            ("effective_n", data.effective_n),
            ("fingerprint", (
                owned_order.fingerprint
                if (type(owned_order) is OrderSample and sample.data is owned_order.data
                    and _plain_fingerprint_metadata(data))
                else sample_fingerprint(data)
            )),
            ("reasons", tuple(sample.reasons)),
            ("active_names", sample.active_names),
            ("_active", owned_array(sample.active)),
            ("input_snapshot", in_snap),
            ("partition", part),
            ("topology", topo),
            ("_ledger", ledger),
            ("_data", data),
            ("_problem", (_problem_headers(sample.problem) if share
                          else _owned_transformed_problem(sample.problem))),
            ("_design", (_design_headers(sample.design) if share
                         else _owned_scaled_design(sample.design))),
            ("_order_scale", owned_order_scale(sample.order_scale)),
            ("_units", owned_units(sampling_units(sample))),
            ("_labels", lbls),
        ):
            if numeric is None or attr not in ("_labels", "topology"):
                object.__setattr__(self, attr, val)

    @property
    def active(self) -> IndexArray:
        """Return a fresh read-only copy of the active column indices."""
        return owned_array(self._active)

    @property
    def positions(self) -> IndexArray:
        """Return a fresh read-only copy of the final original row positions."""
        return owned_array(self._ledger.positions)

    @property
    def ledger(self) -> RowLedger:
        """Return a fresh deeply owned RowLedger."""
        return _owned_row_ledger(self._ledger)

    @property
    def base(self) -> SampleBase:
        """Return a fresh deeply owned SampleBase from the input snapshot."""
        return self.input_snapshot.base

    @property
    def data(self) -> RegressionData:
        """Return a fresh deeply owned RegressionData."""
        return _owned_data(self._data)

    @property
    def problem(self) -> TransformedProblem:
        """Return a fresh deeply owned TransformedProblem."""
        return _owned_transformed_problem(self._problem)

    @property
    def source_and_grouping(
        self,
    ) -> tuple[
        RetainedGroupingSnapshot | None,
        RetainedGroupingSnapshot | None,
        tuple[IndexArray, ...] | None,
    ]:
        """Return fresh detached (source, grouping, ordinals) sharing one view memo."""
        if self.partition is None:
            return None, None, None
        return self.partition.compound_view()

    @property
    def source(self) -> RetainedGroupingSnapshot | None:
        """Return a fresh detached source grouping snapshot."""
        return self.source_and_grouping[0]

    @property
    def grouping(self) -> RetainedGroupingSnapshot | None:
        """Return a fresh detached final grouping snapshot."""
        return self.source_and_grouping[1]

    @property
    def source_ordinals(self) -> tuple[IndexArray, ...] | None:
        """Return fresh owned source-to-final level ordinals per block."""
        return None if self.partition is None else self.partition.ordinals

    # -- internal consumption ---------------------------------------------------
    # What the retained assembly (points, history, certificate, admission context)
    # reads of this sample: the owned storage itself, never an outward copy.  Each
    # record that keeps one of these stores it privately and copies it anew on its
    # own outward exposure; nothing here is handed outward.

    @property
    def _final_grouping(self) -> RetainedGroupingSnapshot | None:
        """Reuse copy-free labels; keep the ordinary custom compound copy path."""
        if self.partition is None:
            return None
        if ("_numeric_partition" not in vars(self.partition)
                and not self.partition.batch._copy_free):
            return self.grouping
        return self.partition._final_view

    @property
    def _source_grouping(self) -> RetainedGroupingSnapshot | None:
        """Reuse copy-free labels; keep the ordinary custom compound copy path."""
        if self.partition is None:
            return None
        if ("_numeric_partition" not in vars(self.partition)
                and not self.partition.batch._copy_free):
            return self.source
        return self.partition._source_view

    @property
    def _owned_response(self) -> FloatArray:
        """The owned transformed response of the retained rows."""
        return self._problem.response

    @property
    def _owned_order_scale(self) -> OrderScale | None:
        """The owned column-scale map of the active columns, or ``None``."""
        return self._order_scale

    @property
    def design(self) -> ScaledDesign | None:
        """Return a fresh deeply owned ScaledDesign."""
        return _owned_scaled_design(self._design)

    @property
    def n_slopes(self) -> int:
        """Return the number of active slope columns ``K``."""
        return int(self._active.size)

    @property
    def weight_values(self) -> FloatArray:
        """Return a fresh owned copy of the raw weight values."""
        return owned_array(self._data.weight_values)

    @property
    def frequency(self) -> bool:
        """Return True if the sample uses frequency weights."""
        return self._data.weights is not None and self._data.weights.frequency

    @property
    def order_scale(self) -> OrderScale | None:
        """Return a fresh owned OrderScale for the active columns, or None."""
        return owned_order_scale(self._order_scale)

    @property
    def metadata(self) -> AlignedColumns:
        """Return a fresh deeply owned AlignedColumns at final positions."""
        return self.input_snapshot.metadata_at(self._ledger.positions)

    @property
    def units(self) -> SamplingUnits:
        """Return a fresh deeply owned SamplingUnits."""
        return owned_units(self._units)

    @property
    def labels(self) -> pd.Index[Any]:
        """Return a fresh owned copy of the user row Index at final positions."""
        return _owned_index(self._labels)


def snapshot_fe_sample(
    sample: FESample,
    *,
    key: SampleKey = 0,
    labels: pd.Index[Any] | None = None,
    input_snapshot: RetainedInputSnapshot | None = None,
    partition_snapshot: RetainedPartitionSnapshot | None = None,
) -> RetainedSampleSnapshot:
    """Snapshot an admitted :class:`FESample` into a :class:`RetainedSampleSnapshot`."""
    return RetainedSampleSnapshot(
        sample,
        key=key,
        labels=labels,
        input_snapshot=input_snapshot,
        partition_snapshot=partition_snapshot,
    )


class SampleIdentityMap:
    """Temporary per-handoff identity map from original ``FESample`` to snapshot."""

    def __init__(
        self,
        *,
        input_map: InputIdentityMap | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> None:
        """Initialize an empty sample identity map."""
        self.input_map = InputIdentityMap() if input_map is None else input_map
        self.partition_map = (
            PartitionIdentityMap() if partition_map is None else partition_map
        )
        self._by_id: dict[int, RetainedSampleSnapshot] = {}
        self._refs: list[FESample] = []
        self._next_key: SampleKey = 0

    def register(
        self,
        sample: FESample,
        snapshot: RetainedSampleSnapshot | None = None,
        *,
        input_snapshot: RetainedInputSnapshot | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedSampleSnapshot:
        """Bind or capture a ``RetainedSampleSnapshot`` for ``sample``."""
        if snapshot is None:
            return self.capture(
                sample,
                input_snapshot=input_snapshot,
                partition_map=partition_map,
            )
        if snapshot.key >= self._next_key:
            self._next_key = snapshot.key + 1
        self._by_id[id(sample)] = snapshot
        self._refs.append(sample)
        self.input_map.register_snapshot(snapshot.input_snapshot, base=sample.base)
        return snapshot

    def capture(
        self,
        sample: FESample,
        *,
        labels: pd.Index[Any] | None = None,
        input_snapshot: RetainedInputSnapshot | None = None,
        partition_map: PartitionIdentityMap | None = None,
    ) -> RetainedSampleSnapshot:
        """Return the ``RetainedSampleSnapshot`` for ``sample`` by object identity."""
        return SampleIdentityMap._capture(
            self, sample, labels=labels, input_snapshot=input_snapshot,
            partition_map=partition_map, owned_order=None,
        )

    def _capture_numeric(
        self, sample: FESample, order: OrderSample, numeric: NumericFit, *,
        input_snapshot: RetainedInputSnapshot, partition_map: PartitionIdentityMap,
    ) -> RetainedSampleSnapshot:
        """Numeric ownership has already been established by the private caller."""
        sid = id(sample)
        if sid in self._by_id:
            return self._by_id[sid]
        assert sample.source is not None
        assert sample.grouping is not None
        part = partition_map._capture_numeric(
            sample.source, sample.grouping, sample.ledger.nobs
        )
        snapshot = RetainedSampleSnapshot._from_numeric(
            sample, order, numeric, key=self._next_key,
            input_snapshot=input_snapshot, partition_snapshot=part,
        )
        return self.register(sample, snapshot)

    def _capture_owned_order(
        self, sample: FESample, order: OrderSample, *,
        input_snapshot: RetainedInputSnapshot,
        partition_map: PartitionIdentityMap,
    ) -> RetainedSampleSnapshot:
        """Consume the private coordinator's fresh order without retaining it."""
        return SampleIdentityMap._capture(
            self, sample, labels=None, input_snapshot=input_snapshot,
            partition_map=partition_map, owned_order=order,
        )

    def _capture(
        self, sample: FESample, *, labels: pd.Index[Any] | None,
        input_snapshot: RetainedInputSnapshot | None,
        partition_map: PartitionIdentityMap | None,
        owned_order: OrderSample | None,
    ) -> RetainedSampleSnapshot:
        """Share the existing identity/lifetime map, not a second fingerprint cache."""
        sid = id(sample)
        if sid in self._by_id:
            return self._by_id[sid]
        key = self._next_key
        self._next_key += 1
        in_snap = input_snapshot or self.input_map.capture(sample.base)
        pmap = self.partition_map if partition_map is None else partition_map
        capture_partition = (pmap._capture_owned_components
                             if type(pmap) is PartitionIdentityMap
                             and _share_sample_values(sample, owned_order)
                             else pmap.capture)
        part_snap = (
            capture_partition(sample.source, sample.grouping, sample.ledger.nobs)
            if sample.source is not None and sample.grouping is not None
            else None
        )
        snap = (
            RetainedSampleSnapshot(
                sample, key=key, labels=labels, input_snapshot=in_snap,
                partition_snapshot=part_snap,
            )
            if owned_order is None
            else RetainedSampleSnapshot._from_owned_order(
                sample, owned_order, key=key, input_snapshot=in_snap,
                partition_snapshot=part_snap,
            )
        )
        self._by_id[sid] = snap
        self._refs.append(sample)
        return snap

    def _owned_snapshot(self, sample: FESample) -> RetainedSampleSnapshot | None:
        """Return only this live handoff's already captured exact sample identity."""
        if (not any(anchor is sample for anchor in self._refs)
                or not _plain_fingerprint_metadata(sample.data)):
            return None
        return self._by_id.get(id(sample))

    def clear_anchors(self) -> None:
        """Release strong references to original sample objects after capture."""
        self._refs.clear()
