"""Private normal-fit numeric lifetime after one original admission boundary.

The completed owned sample is the numerical owner, not another capture of the
preparation/outcome/audit graph. Its construction maps survive through requested
inference and are released at the end of this fit. Public/foreign entry points and
the explicit primitive-snapshot route retain their existing checked behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, FunctionType
from weakref import ref

import numpy as np
import pandas as pd

from powermean.data.numeric_fit import NumericFit, NumericParts
from powermean.data.raw import Column
from powermean.data.retained import OrderSample
from powermean.data.sample_fe_source import Derived
from powermean.data.sample_fe_state import FESample
from powermean.fe.certificate_basis import (
    CertificateBasis, _construction_code, _matching_retained_headers,
    _prepare_fit_basis, _produced_retained_action, _retained_header, _retained_headers,
    _operational_data,
)
from powermean.fe.groups import Block, Grouping
from powermean.interface import retained_binding as _binding
from powermean.interface import retained_metadata as _metadata
from powermean.interface import retained_outcomes as _outcomes
from powermean.interface.owned_fit_state import _passive_values
from powermean.interface.retained_metadata import RetainedColumnMetadata
from powermean.interface.retained_context import PerFitPreparationContext, _OwnedOperation
from powermean.interface.retained_preparation import RetainedPreparation
from powermean.interface.sample import SampleOutcome
from powermean.results import absorbed_outcome_records as _outcome_records
from powermean.results import absorbed_partitions as _partitions
from powermean.results import absorbed_sample as _snapshots
from powermean.results.absorbed_outcome_records import (
    RetainedCollinearityInvocation,
    RetainedOutcomeSnapshot,
    RetainedRecessionInvocation,
    RetainedRound,
)
from powermean.results.absorbed_partitions import (
    GroupingIdentityMap,
    PartitionIdentityMap,
    RetainedPartitionSnapshot,
    RetainedTopologySnapshot,
)
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    SampleIdentityMap,
    _share_sample_values,
)
from powermean.results.primitive_snapshot import PRIMITIVE
from powermean.results.retained_input_values import RetainedInputSnapshot
from powermean.results.retained_labels import _copy_free_labels

# ONE ingress guard for callbacks which the old eager export path would execute.
# Normal numerical kernels keep their existing separate fresh validity checks.
_VIEW_TYPES = (
    RetainedSampleSnapshot, RetainedOutcomeSnapshot, RetainedPartitionSnapshot,
    RetainedTopologySnapshot, RetainedRound, RetainedRecessionInvocation,
    RetainedCollinearityInvocation,
    GroupingIdentityMap, PartitionIdentityMap, SampleIdentityMap,
    RetainedColumnMetadata,
)
_VIEW_METHODS = tuple((kind, tuple((name, vars(kind).get(name)) for name in (
    "__init__", "__post_init__", "__getattribute__", "__getattr__", "__getstate__",
    "key_for", "register", "register_detached", "_capture_numeric",
))) for kind in _VIEW_TYPES)
_VIEW_FIELDS = tuple((
    kind, vars(kind).get("__dataclass_fields__"),
    tuple((name, vars(kind).get(name))
          for name in vars(kind).get("__dataclass_fields__", {})),
) for kind in _VIEW_TYPES)
_VIEW_GLOBALS = tuple((kind.__name__, kind) for kind in _VIEW_TYPES)
_VIEW_ALIASES = tuple((module, name, getattr(module, name)) for module, names in (
    (_metadata, ("RetainedColumnMetadata",)),
    (_binding, ("RetainedColumnMetadata",)),
    (_partitions, ("RetainedPartitionSnapshot", "RetainedTopologySnapshot",
                   "PartitionIdentityMap", "GroupingIdentityMap")),
    (_snapshots, ("RetainedSampleSnapshot", "RetainedTopologySnapshot",
                  "SampleIdentityMap")),
    (_outcome_records, ("RetainedOutcomeSnapshot", "RetainedRecessionInvocation",
                        "RetainedCollinearityInvocation", "RetainedRound")),
    (_outcomes, ("RetainedOutcomeSnapshot", "RetainedRecessionInvocation",
                 "RetainedCollinearityInvocation", "RetainedRound")),
) for name in names)
_VIEW_BATCH = _partitions.build_label_batch
_VIEW_LABELS = _partitions._view_over_labels


def _numeric_ownership_original() -> bool:
    """Unknown or changed rich callbacks keep their original eager ordered path."""
    return (all(globals().get(name) is original for name, original in _VIEW_GLOBALS)
            and all(vars(module).get(name) is original
                    for module, name, original in _VIEW_ALIASES)
            and all(vars(kind).get(name) is original
                    for kind, methods in _VIEW_METHODS for name, original in methods)
            and all(vars(kind).get("__dataclass_fields__") is declared
                    and all(vars(kind).get(name) is original
                            for name, original in attrs)
                    for kind, declared, attrs in _VIEW_FIELDS)
            and _partitions.build_label_batch is _VIEW_BATCH
            and _partitions._view_over_labels is _VIEW_LABELS)


def _plain_column(column: Column) -> bool:
    """Eligibility only: unknown callback-bearing metadata keeps the old route."""
    if type(column) is not Column:
        return False
    stored = column._stored
    if type(stored) is tuple:
        return _copy_free_labels(stored)
    return (type(stored) is np.ndarray and stored.dtype.kind != "O"
            and stored.dtype.fields is None and stored.dtype.metadata is None)


def _plain_labels(snapshot: RetainedSampleSnapshot) -> bool:
    """Use the existing callback-free label proof, not equality of row labels."""
    index = snapshot._labels
    if type(index) is pd.RangeIndex:
        return True
    if type(index) is not pd.Index:
        return False
    values = object.__getattribute__(index, "_data")
    if type(values) is not np.ndarray:
        return False
    return values.dtype.kind != "O" or _copy_free_labels(tuple(values))


def _numeric_grouping(snapshot: RetainedSampleSnapshot) -> Grouping:
    """A canonical internal code view on already admitted detached topology bytes."""
    final = snapshot._final_grouping
    assert final is not None
    blocks: list[Block] = []
    for stored in final.blocks:
        # bind_numeric_execution has just checked the actual snapshot's copy-free
        # labels. Keep its immutable tuple; do not export it through Column's
        # generic ownership constructor merely to prove/copy every level again.
        levels = object.__new__(Column)
        object.__setattr__(levels, "_stored", stored._levels_tuple)
        block = object.__new__(Block)
        for name, value in (("name", stored.name), ("codes", stored._codes.view()),
                            ("counts", stored._counts.view()),
                            ("_levels", levels)):
            object.__setattr__(block, name, value)
        blocks.append(block)
    grouping = object.__new__(Grouping)
    object.__setattr__(grouping, "blocks", tuple(blocks))
    object.__setattr__(grouping, "positions", snapshot._ledger.positions.view())
    return grouping


def _numeric_rows(snapshot: RetainedSampleSnapshot) -> NumericFit:
    """Transfer the authoritative detached row buffers without rebuilding a sample."""
    design = snapshot._design
    matrix = np.zeros((snapshot.count, 0)) if design is None else design.matrix
    scale = np.zeros(0) if design is None else design.scale
    parts = NumericParts(
        snapshot._problem.response, matrix, snapshot._problem.offset,
        snapshot._data.weight_values, snapshot._ledger.positions, snapshot._active,
        scale, _numeric_grouping(snapshot), snapshot.active_names,
        snapshot.omega, snapshot.frequency,
    )
    return NumericFit.from_owned_parts(parts)


@dataclass(frozen=True)
class NumericExecution:
    """A single operation's numeric owner and existing retained-construction maps."""

    numeric: NumericFit
    preparation: RetainedPreparation
    _sample: FESample | None = None
    _basis_native: ModuleType | None = None

    def prepare_basis(self, sample: FESample) -> CertificateBasis:
        """One structural construction for a live actually admitted sample owner.

        There is no caller-provided basis, validity flag or cross-fit cache.
        The private operation's actual sample or admitted map anchor establishes
        its sample identity; release removes those anchors, so an expired or
        foreign operation cannot consume it.
        """
        operation = self.preparation.context._operation
        if (operation is None or operation.numeric is not self.numeric
                or type(sample) is not FESample
                or not (self._sample is sample or any(
                    owned is sample for owned in operation.maps.sample_map._refs))):
            raise ValueError(
                "the basis needs this live numeric operation's admitted sample"
            )
        backend = vars(self).get("_basis_native") if type(self) is NumericExecution else None
        basis = (
            _prepare_fit_basis(sample, _native=backend)
            if (type(self) is NumericExecution
                and type(backend) is ModuleType
                and _prepare_fit_basis is _PREPARE_RETAINED_ISSUER)
            else _prepare_fit_basis(sample)
        )
        if (type(self) is NumericExecution
                and _prepare_fit_basis is _PREPARE_RETAINED_ISSUER
                and _produced_retained_action is _PRODUCED_RETAINED
                and _live_retained_original is _LIVE_RETAINED_GUARD):
            try:
                if not _live_retained_original() or vars(self).get("_sample") is not sample:
                    return basis
                numeric = vars(self)["numeric"]
                preparation = vars(self)["preparation"]
                context = vars(preparation)["_context"]
                payload = _produced_retained_action(basis, sample)
                if (payload is not None and _same_group_storage(
                        vars(sample).get("grouping"), vars(numeric).get("_grouping"))):
                    headers = _retained_headers(basis, vars(basis)["grouping"])
                    scope = (ref(self), ref(numeric), ref(sample), ref(operation),
                             ref(basis), ref(basis), ref(context), None, ref(preparation))

                    def construction():
                        return scope, payload, headers

                    object.__setattr__(basis, _LIVE_RETAINED, construction)
            except (MemoryError, TypeError):
                pass  # Optional live binding only, never a numerical action.
        return basis

    def rebind_retained_basis(self, source: CertificateBasis,
                             found: CertificateBasis, problem: object) -> None:
        """Explicitly transfer only the genuine live source reader across rebind."""
        try:
            carried = _live_retained(source, None)
            if carried is None or carried[0][0]() is not self:
                return
            scope0, payload, _ = carried
            numeric = vars(self)["numeric"]
            a, b = vars(source), vars(found)
            if (type(found) is not CertificateBasis
                    or any(a.get(name) is not b.get(name) for name in (
                        "route", "rank", "rows", "selected", "references", "columns",
                        "positions", "witnesses", "echelon", "_operational"))
                    or vars(problem).get("_numeric_owner") is not numeric
                    or vars(problem).get("grouping") is not b.get("grouping")
                    or not _same_group_storage(b.get("grouping"), vars(numeric).get("_grouping"))):
                return
            headers = _retained_headers(found, b["grouping"])
            if headers is None:
                return
            scope = (*scope0[:4], ref(found), scope0[5], scope0[6], ref(problem), scope0[8])

            def construction():
                return scope, payload, headers

            object.__setattr__(found, _LIVE_RETAINED, construction)
        except (MemoryError, TypeError):
            return

    def release(self) -> None:
        """Drop original-object anchors; retained outputs keep only owned records."""
        object.__setattr__(self, "_sample", None)
        if vars(self).get("_basis_native") is not None:
            object.__setattr__(self, "_basis_native", None)
        self.preparation.context.release_fit_state()


_LIVE_RETAINED = "_live_retained_rows"
_PREPARE_RETAINED_ISSUER = _prepare_fit_basis
_PRODUCED_RETAINED = _produced_retained_action


def _same_group_storage(left: object, right: object) -> bool:
    """Compare complete immutable owners and headers, not row values or flags."""
    if ((type(left) is not Grouping and type(left) is not Derived)
            or type(right) is not Grouping):
        return False
    a, b = vars(left), vars(right)
    aa, bb = a.get("blocks"), b.get("blocks")
    if type(aa) is not tuple or type(bb) is not tuple or len(aa) != len(bb):
        return False
    arrays = [(a.get("positions"), b.get("positions"))]
    for x, y in zip(aa, bb, strict=True):
        if type(x) is not Block or type(y) is not Block:
            return False
        arrays.extend((vars(x).get(k), vars(y).get(k)) for k in ("codes", "counts"))
    for x, y in arrays:
        first, second = _retained_header(x), _retained_header(y)
        if (first is None or second is None or first[1] is not second[1]
                or first[2:] != second[2:]):
            return False
    return True


def _live_retained(basis: object, problem: object):
    """Code-authenticated weak live handoff; copied records grant no authority."""
    if (type(basis) is not CertificateBasis
            or _live_retained_original is not _LIVE_RETAINED_GUARD
            or not _live_retained_original()):
        return None
    reader = vars(basis).get(_LIVE_RETAINED)
    if (type(reader) is not FunctionType
            or reader.__code__ is not _construction_code(NumericExecution.prepare_basis)
            and reader.__code__ is not _construction_code(NumericExecution.rebind_retained_basis)):
        return None
    scope, payload, headers = reader()
    execution, numeric, sample, operation = (weak() for weak in scope[:4])
    if (execution is None or numeric is None or sample is None or operation is None
            or type(execution) is not NumericExecution or type(numeric) is not NumericFit
            or scope[4]() is not basis or vars(execution).get("numeric") is not numeric
            or vars(execution).get("_sample") is not sample):
        return None
    preparation = vars(execution).get("preparation")
    context = scope[6]()
    if (type(preparation) is not RetainedPreparation or scope[8]() is not preparation
            or type(context) is not PerFitPreparationContext
            or type(operation) is not _OwnedOperation
            or vars(preparation).get("_context") is not context
            or vars(context).get("_operation") is not operation
            or vars(operation).get("numeric") is not numeric):
        return None
    source = scope[5]()
    actual = _produced_retained_action(source, sample)
    if (actual is None or actual[0] is not payload[0]
            or (_operational_data(source) is not None
                and _operational_data(basis) is not payload[0])
            or not _matching_retained_headers(basis, vars(basis).get("grouping"), headers)
            or not _same_group_storage(vars(basis).get("grouping"), vars(numeric).get("_grouping"))):
        return None
    if problem is not None and (scope[7] is None or scope[7]() is not problem
            or vars(problem).get("_numeric_owner") is not numeric
            or vars(problem).get("grouping") is not vars(basis).get("grouping")):
        return None
    return scope, payload, headers


def _direct_retained_action(basis: object, problem: object):
    carried = _live_retained(basis, problem)
    return None if carried is None else carried[1][0]


_DIRECT_RETAINED_ACTION = _direct_retained_action
_LIVE_RETAINED_CLASSES = (NumericExecution, NumericFit, CertificateBasis, FESample, Grouping,
                         Derived, Block, RetainedPreparation, PerFitPreparationContext, _OwnedOperation)
_LIVE_RETAINED_METHODS = tuple((kind, name, vars(kind).get(name)) for kind in _LIVE_RETAINED_CLASSES
                             for name in ("__getattribute__", "__getattr__", "__dict__"))
_LIVE_RETAINED_CALLS = (_prepare_fit_basis, _produced_retained_action, _retained_header,
    _retained_headers, _matching_retained_headers, _construction_code,
    _same_group_storage, _live_retained, NumericExecution.prepare_basis,
    NumericExecution.rebind_retained_basis, _operational_data)
_LIVE_RETAINED_FIELDS = tuple((kind, name, vars(kind).get(name)) for kind, name in (
    (NumericExecution, "numeric"), (NumericExecution, "preparation"),
    (NumericExecution, "_sample"), (RetainedPreparation, "context"),
))


def _live_retained_original() -> bool:
    kinds = (NumericExecution, NumericFit, CertificateBasis, FESample, Grouping,
             Derived, Block, RetainedPreparation, PerFitPreparationContext, _OwnedOperation)
    if any(a is not b for a, b in zip(kinds, _LIVE_RETAINED_CLASSES, strict=True)):
        return False
    current = (_prepare_fit_basis, _produced_retained_action, _retained_header,
        _retained_headers, _matching_retained_headers, _construction_code,
        _same_group_storage, _live_retained, NumericExecution.prepare_basis,
        NumericExecution.rebind_retained_basis, _operational_data)
    return (all(a is b for a, b in zip(current, _LIVE_RETAINED_CALLS, strict=True))
            and all(vars(kind).get(name) is expected for kind, name, expected in _LIVE_RETAINED_METHODS)
            and all(vars(kind).get(name) is expected for kind, name, expected in _LIVE_RETAINED_FIELDS))


_LIVE_RETAINED_GUARD = _live_retained_original


def _bind_retained_basis(execution: object, source: CertificateBasis,
                         found: CertificateBasis, problem: object) -> None:
    try:
        if (type(execution) is NumericExecution
                and _live_retained_original is _LIVE_RETAINED_GUARD
                and _live_retained_original()):
            NumericExecution.rebind_retained_basis(execution, source, found, problem)
    except MemoryError:
        return


def bind_numeric_sample(
    preparation: RetainedPreparation, outcome: SampleOutcome, order: OrderSample, *,
    source: RetainedInputSnapshot | None,
) -> NumericExecution | None:
    """The normal numeric producer, BEFORE any retained graph is constructed.

    Public/foreign inputs still use the checked lowering route below. Actual
    operation/frame/source ownership and current producer storage are required;
    no retained snapshot, caller-provided validity flag or sample-map registration
    establishes numerical ownership.
    """
    context, sample = preparation.context, outcome.sample
    operation = context._operation
    if (not _numeric_ownership_original()
            or operation is None or source is None
            or source is not context._normal_origin
            or outcome.status != "complete" or outcome.origin is not None
            or type(sample) is not FESample
            or outcome.prepared is not operation.frame
            or not _share_sample_values(sample, order)):
        return None
    maps = operation.maps
    if (type(maps.sample_map) is not SampleIdentityMap
            or type(maps.partition_map) is not PartitionIdentityMap
            or type(maps.grouping_map) is not GroupingIdentityMap):
        return None
    grouping = sample.grouping
    if type(grouping) is not Grouping and type(grouping) is not Derived:
        return None
    if not _passive_values((outcome.prepared._frame.design.spec,)):
        return None
    if not all(_plain_column(column) for column in
               source._base.inputs.metadata.columns.values()):
        return None
    # Normalized input ownership has already admitted original labels/effects.
    # NumericFit detaches current numeric headers on immutable complete bytes.
    context.authenticate_frame(outcome.prepared, sample=sample)
    design = sample.design
    assert design is not None
    numeric = NumericFit.from_owned_parts(NumericParts(
        sample.problem.response, design.matrix, sample.problem.offset,
        sample.data.weight_values, sample.positions, sample.active, design.scale,
        grouping, sample.active_names, sample.omega,
        sample.data.weights is not None and sample.data.weights.frequency,
    ))
    context = context._after_numeric_admission(
        outcome.prepared, operation.input_snapshot, numeric
    )
    if context is not preparation.context:
        return None
    return NumericExecution(numeric, preparation, sample)


def bind_numeric_execution(
    preparation: RetainedPreparation,
    outcome: SampleOutcome,
    snapshot: RetainedOutcomeSnapshot,
    *,
    source: RetainedInputSnapshot | None,
) -> NumericExecution | None:
    """Enter the normal owned route only AFTER actual original graph admission.

    A real private normalized-input owner is required. Unknown mutable/callback
    labels and metadata, direct foreign preparations and explicit primitive
    snapshots retain their old route; none becomes a scientific refusal.
    """
    sample, detached = outcome.sample, snapshot.sample
    if (source is None or sample is None or detached is None
            or preparation.context.snapshot_route == PRIMITIVE):
        return None
    if source is not preparation.context._normal_origin:
        return None
    if not _passive_values((outcome.prepared._frame.design.spec,)):
        return None
    final = detached._final_grouping
    if final is None or not final._copy_free or not _plain_labels(detached):
        return None
    columns = snapshot.input_snapshot._base.inputs.metadata.columns
    if not all(_plain_column(column) for column in columns.values()):
        return None
    numeric = _numeric_rows(detached)
    context = preparation.context._after_numeric_admission(
        outcome.prepared, snapshot.input_snapshot, numeric
    )
    operation = context._operation
    assert operation is not None
    if context is preparation.context:
        # Its existing maps already contain this exact admitted sample, input,
        # partition and outcome. Carry them through solve/result; don't discard
        # and reconstruct a second key space at numeric-owner binding.
        return NumericExecution(numeric, preparation)
    maps = operation.maps
    maps.input_map.register_snapshot(snapshot.input_snapshot, base=sample.base,
                                     prepared=outcome.prepared)
    maps.sample_map.register(sample, detached)
    if sample.grouping is not None:
        maps.grouping_map.register_detached(sample.grouping, final)
    maps.outcome_memo[id(outcome)] = snapshot
    maps._outcome_refs.append(outcome)
    retained = object.__new__(RetainedPreparation)
    for name in preparation.__dataclass_fields__:
        object.__setattr__(retained, name, getattr(preparation, name))
    object.__setattr__(retained, "_context", context)
    object.__setattr__(retained, "input_snapshot", snapshot.input_snapshot)
    return NumericExecution(numeric, retained)
