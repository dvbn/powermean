"""Private normal-fit numeric lifetime after one original admission boundary.

The completed owned sample is the numerical owner, not another capture of the
preparation/outcome/audit graph. Its construction maps survive through requested
inference and are released at the end of this fit. Public/foreign entry points and
the explicit primitive-snapshot route retain their existing checked behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import cast

import numpy as np
import pandas as pd

from powermean.data.numeric_fit import NumericFit, NumericParts
from powermean.data.raw import Column
from powermean.data.retained import OrderSample
from powermean.data.sample_fe_source import Derived as _Derived
from powermean.data.sample_fe_state import FESample
from powermean.fe.certificate_basis import (
    CertificateBasis,
    _construction_code,
    _matching_retained_headers,
    _operational_data,
    _prepare_fit_basis,
    _produced_retained_action,
    _retained_header,
    _retained_headers,
)
from powermean.fe.groups import Block as _Block
from powermean.fe.groups import Grouping
from powermean.fe.retained_live import (
    _LIVE_RETAINED as _LIVE_RETAINED,  # noqa: PLC0414 - explicit re-export
)
from powermean.fe.retained_live import (
    _direct_retained_action,
    _issue_live_retained,
    _live_retained,
    _rebind_live_retained,
)
from powermean.fe.retained_live import (
    _same_group_storage as _same_group_storage,  # noqa: PLC0414 - explicit re-export
)
from powermean.interface import retained_binding as _binding
from powermean.interface import retained_metadata as _metadata
from powermean.interface import retained_outcomes as _outcomes
from powermean.interface.owned_fit_state import _passive_values
from powermean.interface.retained_context import (
    PerFitPreparationContext,
    _OwnedOperation,
)
from powermean.interface.retained_metadata import RetainedColumnMetadata
from powermean.interface.retained_preparation import RetainedPreparation
from powermean.interface.sample import _EXISTENCE_BASIS, SampleOutcome
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
from powermean.results.retained_input_values import RetainedInputSnapshot
from powermean.results.retained_labels import _copy_free_labels

# Live dispatch reads these exact bindings when it checks shared group storage.
Derived = _Derived
Block = _Block

# ONE ingress guard for callbacks which the old eager export path would execute.
# Normal numerical kernels keep their existing separate fresh validity checks.
_VIEW_TYPES = (
    RetainedSampleSnapshot,
    RetainedOutcomeSnapshot,
    RetainedPartitionSnapshot,
    RetainedTopologySnapshot,
    RetainedRound,
    RetainedRecessionInvocation,
    RetainedCollinearityInvocation,
    GroupingIdentityMap,
    PartitionIdentityMap,
    SampleIdentityMap,
    RetainedColumnMetadata,
)
_VIEW_METHODS = tuple(
    (
        kind,
        tuple(
            (name, vars(kind).get(name))
            for name in (
                "__init__",
                "__post_init__",
                "__getattribute__",
                "__getattr__",
                "__getstate__",
                "key_for",
                "register",
                "register_detached",
                "_capture_numeric",
            )
        ),
    )
    for kind in _VIEW_TYPES
)
_VIEW_FIELDS = tuple(
    (
        kind,
        vars(kind).get("__dataclass_fields__"),
        tuple(
            (name, vars(kind).get(name))
            for name in vars(kind).get("__dataclass_fields__", {})
        ),
    )
    for kind in _VIEW_TYPES
)
_VIEW_GLOBALS = tuple((kind.__name__, kind) for kind in _VIEW_TYPES)
_VIEW_ALIASES = tuple(
    (module, name, getattr(module, name))
    for module, names in (
        (_metadata, ("RetainedColumnMetadata",)),
        (_binding, ("RetainedColumnMetadata",)),
        (
            _partitions,
            (
                "RetainedPartitionSnapshot",
                "RetainedTopologySnapshot",
                "PartitionIdentityMap",
                "GroupingIdentityMap",
            ),
        ),
        (
            _snapshots,
            ("RetainedSampleSnapshot", "RetainedTopologySnapshot", "SampleIdentityMap"),
        ),
        (
            _outcome_records,
            (
                "RetainedOutcomeSnapshot",
                "RetainedRecessionInvocation",
                "RetainedCollinearityInvocation",
                "RetainedRound",
            ),
        ),
        (
            _outcomes,
            (
                "RetainedOutcomeSnapshot",
                "RetainedRecessionInvocation",
                "RetainedCollinearityInvocation",
                "RetainedRound",
            ),
        ),
    )
    for name in names
)
_VIEW_BATCH = _partitions.build_label_batch
_VIEW_LABELS = _partitions._view_over_labels


def _numeric_ownership_original() -> bool:
    """Unknown or changed rich callbacks keep their original eager ordered path."""
    return (
        all(globals().get(name) is original for name, original in _VIEW_GLOBALS)
        and all(
            vars(module).get(name) is original
            for module, name, original in _VIEW_ALIASES
        )
        and all(
            vars(kind).get(name) is original
            for kind, methods in _VIEW_METHODS
            for name, original in methods
        )
        and all(
            vars(kind).get("__dataclass_fields__") is declared
            and all(vars(kind).get(name) is original for name, original in attrs)
            for kind, declared, attrs in _VIEW_FIELDS
        )
        and _partitions.build_label_batch is _VIEW_BATCH
        and _partitions._view_over_labels is _VIEW_LABELS
    )


def _plain_column(column: Column) -> bool:
    """Eligibility only: unknown callback-bearing metadata keeps the old route."""
    if type(column) is not Column:
        return False
    stored = column._stored
    if type(stored) is tuple:
        return _copy_free_labels(stored)
    return (
        type(stored) is np.ndarray
        and stored.dtype.kind != "O"
        and stored.dtype.fields is None
        and stored.dtype.metadata is None
    )


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
        for name, value in (
            ("name", stored.name),
            ("codes", stored._codes.view()),
            ("counts", stored._counts.view()),
            ("_levels", levels),
        ):
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
        snapshot._problem.response,
        matrix,
        snapshot._problem.offset,
        snapshot._data.weight_values,
        snapshot._ledger.positions,
        snapshot._active,
        scale,
        _numeric_grouping(snapshot),
        snapshot.active_names,
        snapshot.omega,
        snapshot.frequency,
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
        if (
            operation is None
            or operation.numeric is not self.numeric
            or type(sample) is not FESample
            or not (
                self._sample is sample
                or any(owned is sample for owned in operation.maps.sample_map._refs)
            )
        ):
            raise ValueError(
                "the basis needs this live numeric operation's admitted sample"
            )
        backend = (
            vars(self).get("_basis_native") if type(self) is NumericExecution else None
        )
        basis = None
        # Consume only this sample's freshly produced immutable construction.
        # A supplied attribute alone grants no authority; the existing reader
        # authenticates the producer, owner and complete current storage headers.
        offered = vars(sample).pop(_EXISTENCE_BASIS, None)
        if (
            type(self) is NumericExecution
            and self._sample is sample
            and _prepare_fit_basis is _PREPARE_RETAINED_ISSUER
            and _produced_retained_action is _PRODUCED_RETAINED
            and type(offered) is tuple
            and len(offered) == 2
            and offered[1] is backend
            and _produced_retained_action(offered[0], sample, basis_only=True)
            is not None
        ):
            basis = cast(CertificateBasis, offered[0])
        if basis is None:
            basis = (
                _prepare_fit_basis(sample, _native=backend)
                if (
                    type(self) is NumericExecution
                    and type(backend) is ModuleType
                    and _prepare_fit_basis is _PREPARE_RETAINED_ISSUER
                )
                else _prepare_fit_basis(sample)
            )
        if (
            type(self) is NumericExecution
            and _prepare_fit_basis is _PREPARE_RETAINED_ISSUER
            and _produced_retained_action is _PRODUCED_RETAINED
            and _live_retained_original is _LIVE_RETAINED_GUARD
        ):
            try:
                if (
                    not _live_retained_original()
                    or vars(self).get("_sample") is not sample
                ):
                    return basis
                preparation = vars(self)["preparation"]
                # The canonical lower issuer reads the producer proof, headers
                # and storage itself; this owner supplies only its own weak
                # anchors and import-time pins, re-read there at every use.
                _issue_live_retained(
                    basis,
                    sample,
                    vars(self)["numeric"],
                    (self, preparation, vars(preparation)["_context"], operation),
                    _RETAINED_OWNER,
                )
            except (MemoryError, TypeError):
                pass  # Optional live binding only, never a numerical action.
        return basis

    def rebind_retained_basis(
        self, source: CertificateBasis, found: CertificateBasis, problem: object
    ) -> None:
        """Explicitly transfer only the genuine live source reader across rebind.

        The canonical lower reader carries this owner's own live binding of the
        source; a copied, released or foreign owner, basis or problem gets none.
        """
        try:
            _rebind_live_retained(self, source, found, problem)
        except (MemoryError, TypeError):
            return

    def release(self) -> None:
        """Drop original-object anchors; retained outputs keep only owned records."""
        if type(self._sample) is FESample:
            vars(self._sample).pop(_EXISTENCE_BASIS, None)
        object.__setattr__(self, "_sample", None)
        if vars(self).get("_basis_native") is not None:
            object.__setattr__(self, "_basis_native", None)
        self.preparation.context.release_fit_state()


_PREPARE_RETAINED_ISSUER = _prepare_fit_basis
_PRODUCED_RETAINED = _produced_retained_action

# The live reader, its storage comparator and its consumer are canonical lower
# code (powermean.fe.retained_live); these names stay bound to the same objects
# for the existing historic coordinate fallback and the count-site pins.
_DIRECT_RETAINED_ACTION = _direct_retained_action
_LIVE_RETAINED_CLASSES = (
    NumericExecution,
    NumericFit,
    CertificateBasis,
    FESample,
    Grouping,
    Derived,
    Block,
    RetainedPreparation,
    PerFitPreparationContext,
    _OwnedOperation,
)
_LIVE_RETAINED_METHODS = tuple(
    (kind, name, vars(kind).get(name))
    for kind in _LIVE_RETAINED_CLASSES
    for name in ("__getattribute__", "__getattr__", "__dict__")
)
_LIVE_RETAINED_CALLS = (
    _prepare_fit_basis,
    _produced_retained_action,
    _retained_header,
    _retained_headers,
    _matching_retained_headers,
    _construction_code,
    _same_group_storage,
    _live_retained,
    NumericExecution.prepare_basis,
    NumericExecution.rebind_retained_basis,
    _operational_data,
    _issue_live_retained,
    _rebind_live_retained,
)
_LIVE_RETAINED_FIELDS = tuple(
    (kind, name, vars(kind).get(name))
    for kind, name in (
        (NumericExecution, "numeric"),
        (NumericExecution, "preparation"),
        (NumericExecution, "_sample"),
        (RetainedPreparation, "context"),
    )
)


def _live_retained_original() -> bool:
    """Issue-time owner guard over this module's aliases, classes and slots.

    It gates issuance and rebinding only. Each use is decided by the canonical
    lower reader, which re-reads its own helpers and the owner pins below;
    replacing only one of these upper aliases after issuance is not a use-time
    revocation (author-approved Layer18 contract change).
    """
    kinds = (
        NumericExecution,
        NumericFit,
        CertificateBasis,
        FESample,
        Grouping,
        Derived,
        Block,
        RetainedPreparation,
        PerFitPreparationContext,
        _OwnedOperation,
    )
    if any(a is not b for a, b in zip(kinds, _LIVE_RETAINED_CLASSES, strict=True)):
        return False
    current = (
        _prepare_fit_basis,
        _produced_retained_action,
        _retained_header,
        _retained_headers,
        _matching_retained_headers,
        _construction_code,
        _same_group_storage,
        _live_retained,
        NumericExecution.prepare_basis,
        NumericExecution.rebind_retained_basis,
        _operational_data,
        _issue_live_retained,
        _rebind_live_retained,
    )
    return (
        all(a is b for a, b in zip(current, _LIVE_RETAINED_CALLS, strict=True))
        and all(
            vars(kind).get(name) is expected
            for kind, name, expected in _LIVE_RETAINED_METHODS
        )
        and all(
            vars(kind).get(name) is expected
            for kind, name, expected in _LIVE_RETAINED_FIELDS
        )
    )


_LIVE_RETAINED_GUARD = _live_retained_original

# This owner's passive live description, issued with each binding and re-read by
# the canonical lower reader at every use: exact anchor classes; (anchor, field,
# target) links over (execution, preparation, context, operation, numeric,
# sample); and the import-time class slot, field and issuer-method originals.
_RETAINED_OWNER_KINDS = (
    NumericExecution,
    RetainedPreparation,
    PerFitPreparationContext,
    _OwnedOperation,
)
_RETAINED_OWNER = (
    _RETAINED_OWNER_KINDS,
    (
        (0, "numeric", 4),
        (0, "_sample", 5),
        (0, "preparation", 1),
        (1, "_context", 2),
        (2, "_operation", 3),
        (3, "numeric", 4),
    ),
    tuple(
        (kind, name, vars(kind).get(name))
        for kind, names in (
            (
                NumericExecution,
                (
                    "__getattribute__",
                    "__getattr__",
                    "__dict__",
                    "numeric",
                    "preparation",
                    "_sample",
                    "prepare_basis",
                    "rebind_retained_basis",
                ),
            ),
            (
                RetainedPreparation,
                ("__getattribute__", "__getattr__", "__dict__", "context"),
            ),
            (
                PerFitPreparationContext,
                ("__getattribute__", "__getattr__", "__dict__"),
            ),
            (_OwnedOperation, ("__getattribute__", "__getattr__", "__dict__")),
        )
        for name in names
    ),
)


def _bind_retained_basis(
    execution: object,
    source: CertificateBasis,
    found: CertificateBasis,
    problem: object,
) -> None:
    try:
        if (
            type(execution) is NumericExecution
            and _live_retained_original is _LIVE_RETAINED_GUARD
            and _live_retained_original()
        ):
            NumericExecution.rebind_retained_basis(execution, source, found, problem)
    except MemoryError:
        return


def bind_numeric_sample(
    preparation: RetainedPreparation,
    outcome: SampleOutcome,
    order: OrderSample,
    *,
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
    if (
        not _numeric_ownership_original()
        or operation is None
        or source is None
        or source is not context._normal_origin
        or outcome.status != "complete"
        or outcome.origin is not None
        or type(sample) is not FESample
        or outcome.prepared is not operation.frame
        or not _share_sample_values(sample, order)
    ):
        return None
    maps = operation.maps
    if (
        type(maps.sample_map) is not SampleIdentityMap
        or type(maps.partition_map) is not PartitionIdentityMap
        or type(maps.grouping_map) is not GroupingIdentityMap
    ):
        return None
    grouping = sample.grouping
    if type(grouping) is not Grouping and type(grouping) is not Derived:
        return None
    if not _passive_values((outcome.prepared._frame.design.spec,)):
        return None
    if not all(
        _plain_column(column)
        for column in source._base.inputs.metadata.columns.values()
    ):
        return None
    # Normalized input ownership has already admitted original labels/effects.
    # NumericFit detaches current numeric headers on immutable complete bytes.
    context.authenticate_frame(outcome.prepared, sample=sample)
    design = sample.design
    assert design is not None
    numeric = NumericFit.from_owned_parts(
        NumericParts(
            sample.problem.response,
            design.matrix,
            sample.problem.offset,
            sample.data.weight_values,
            sample.positions,
            sample.active,
            design.scale,
            grouping,
            sample.active_names,
            sample.omega,
            sample.data.weights is not None and sample.data.weights.frequency,
        )
    )
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
    labels and metadata and direct foreign preparations retain checked admission.
    """
    sample, detached = outcome.sample, snapshot.sample
    if source is None or sample is None or detached is None:
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
    maps.input_map.register_snapshot(
        snapshot.input_snapshot, base=sample.base, prepared=outcome.prepared
    )
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
