"""Per-fit preparation context and temporary interface admission identity maps.

``design/02_computation.md``, Sections 2 to 4 and 7;
``plan/evidence/M1.15_contract.txt``, items 2, 3, 7 and 9;
``brief/layout_C_ownership.txt`` (P16).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from powermean._prepare.compose import PreparedFrame, _materialize_retained
from powermean._prepare.dependence import DependenceProvenance, UsedDependence
from powermean.data.inference_requests import (
    ClusterRequest,
    ConleyRequest,
    HacRequest,
)
from powermean.data.numeric_fit import NumericFit
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_state import FESample, same_bits, same_missing
from powermean.formula.frame import Design, ModelFrame
from powermean.interface.owned_fit_state import (
    normalized_fit_frame,
    normalized_numeric_inputs,
    own_prepared_once,
    passive_preparation,
    single_fit_frame,
)
from powermean.results.absorbed_groupings import GroupingIdentityMap
from powermean.results.absorbed_handoff import CaptureMaps
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_partitions import PartitionIdentityMap
from powermean.results.absorbed_points import PointIdentityMap
from powermean.results.absorbed_sample import SampleIdentityMap
from powermean.results.absorbed_sample_binding import (
    UNBOUND,
    _check_prepared_frame_and_provenance,
    _same_index_values,
    check_prepared_matches_sample,
)
from powermean.results.retained_input_values import (
    InputKey,
    RetainedInputSnapshot,
    _label_values,
    _owned_prepared_frame,
    _same_column_values,
)
from powermean.results.retained_inputs import InputIdentityMap, ProblemIdentityMap
from powermean.results.retained_labels import _same_design_spec, _same_opt_bits

__all__ = [
    "PerFitPreparationContext",
    "_AdmissionMaps",
    "_IssuedFitFrame",
    "_check_frame_matches_baseline",
    "_frames_match",
    "authenticate_unscoped_frame",
    "authenticate_unscoped_sample",
]

type _Ref[T] = weakref.ref[T]
type _FrameEntry = tuple[_Ref[PreparedFrame], PreparedFrame, RetainedInputSnapshot]
type _BaseEntry = tuple[
    _Ref[SampleBase], _Ref[PreparedFrame], PreparedFrame, RetainedInputSnapshot
]

_ISSUED_BY_FRAME_ID: dict[int, _FrameEntry] = {}
_ISSUED_BY_BASE_ID: dict[int, _BaseEntry] = {}


@dataclass(frozen=True)
class _IssuedFitFrame:
    """An admitted ``PreparedFrame``, its detached baseline and input snapshot."""

    frame: PreparedFrame
    baseline: PreparedFrame
    input_snapshot: RetainedInputSnapshot


def _register_weak_evidence(
    f: PreparedFrame, b: PreparedFrame, s: RetainedInputSnapshot
) -> None:
    """Index the baseline and snapshot by weak refs with cleanup callbacks."""
    fid, bid = id(f), id(f.base)

    def _on_frame_dead(ref: _Ref[PreparedFrame]) -> None:
        frames = cast("dict[int, _FrameEntry] | None", _ISSUED_BY_FRAME_ID)
        bases = cast("dict[int, _BaseEntry] | None", _ISSUED_BY_BASE_ID)
        if frames is None or bases is None:
            return  # interpreter shutdown: the module globals are already cleared
        if (cf := frames.get(fid)) is not None and cf[0] is ref:
            del frames[fid]
        cb = bases.get(bid)
        if cb is not None and cb[1] is ref and cb[0]() is None:
            del bases[bid]

    def _on_base_dead(ref: _Ref[SampleBase]) -> None:
        bases = cast("dict[int, _BaseEntry] | None", _ISSUED_BY_BASE_ID)
        if bases is None:
            return  # interpreter shutdown
        if (cb := bases.get(bid)) is not None and cb[0] is ref:
            del bases[bid]

    f_ref = weakref.ref(f, _on_frame_dead)
    b_ref = weakref.ref(f.base, _on_base_dead)
    _ISSUED_BY_FRAME_ID[fid] = (f_ref, b, s)
    _ISSUED_BY_BASE_ID[bid] = (b_ref, f_ref, b, s)


def _same_model_frame(f_a: ModelFrame, f_b: ModelFrame) -> bool:
    """Return True if two ``ModelFrame`` instances match on all stored fields."""
    if type(f_a) is not ModelFrame or type(f_b) is not ModelFrame:
        return False
    d_a, d_b = f_a.design, f_b.design
    if (
        type(d_a) is not Design
        or (type(d_a), f_a.parts, f_a.outcome.name, f_a.outcome.dtype)
        != (type(d_b), f_b.parts, f_b.outcome.name, f_b.outcome.dtype)
        or (d_a.terms, d_a.variables, d_a.formulaic_version, d_a.columns)
        != (d_b.terms, d_b.variables, d_b.formulaic_version, d_b.columns)
        or (d_a.matrix.shape, tuple(d_a.matrix.dtypes))
        != (d_b.matrix.shape, tuple(d_b.matrix.dtypes))
        or not _same_design_spec(d_a.spec, d_b.spec)
        or not _same_index_values(f_a.outcome.index, f_b.outcome.index)
        or not _same_column_values(f_a.outcome.to_numpy(), f_b.outcome.to_numpy())
        or not same_bits(f_a.outcome_invalid, f_b.outcome_invalid)
        or not _same_index_values(d_a.labels, d_b.labels)
        or not same_bits(d_a.referenced_invalid, d_b.referenced_invalid)
        or not same_bits(d_a.evaluated_invalid, d_b.evaluated_invalid)
        or not _same_index_values(d_a.matrix.index, d_b.matrix.index)
        or not _same_index_values(d_a.matrix.columns, d_b.matrix.columns)
    ):
        return False
    return same_bits(
        d_a.matrix.to_numpy(dtype=np.float64, copy=False),
        d_b.matrix.to_numpy(dtype=np.float64, copy=False),
    )


def _same_sample_base(b_a: SampleBase, b_b: SampleBase) -> bool:
    """Return True if two ``SampleBase`` instances match on all arrays and metadata."""
    i_a, i_b = b_a.inputs, b_b.inputs
    w_a, w_b = i_a.weights, i_b.weights
    w_ok = (w_a is None and w_b is None) or (
        w_a is not None
        and w_b is not None
        and w_a.kind == w_b.kind
        and _same_column_values(w_a.values, w_b.values)
    )
    if (
        (b_a.nobs, i_a.column_names, i_a.metadata.rows, i_b.metadata.rows)
        != (b_b.nobs, i_b.column_names, b_a.nobs, b_b.nobs)
        or tuple(i_a.metadata) != tuple(i_b.metadata)
        or not _same_opt_bits(b_a.subset, b_b.subset)
        or not same_bits(b_a.codes, b_b.codes)
        or not same_bits(b_a.complete, b_b.complete)
        or not same_bits(b_a.negative_weight, b_b.negative_weight)
        or not same_bits(b_a.zero_weight, b_b.zero_weight)
        or not same_missing(b_a.missing, b_b.missing)
        or not same_bits(i_a.y, i_b.y)
        or not same_bits(i_a.design, i_b.design)
        or not w_ok
        or not _same_opt_bits(i_a.offset, i_b.offset)
        or not _same_opt_bits(i_a.exposure, i_b.exposure)
    ):
        return False
    return all(
        _same_column_values(i_a.metadata[k], i_b.metadata[k]) for k in i_a.metadata
    )


def _same_effects(e_a: EffectColumns | None, e_b: EffectColumns | None) -> bool:
    """Return True if two optional ``EffectColumns`` records match.

    Each column's labels are compared as ``Column.values`` hands them out, plain labels
    read from storage without copies (:func:`_label_values`).
    """
    if e_a is None or e_b is None:
        return e_a is None and e_b is None
    return (e_a.rows, e_a.terms, tuple(e_a.columns)) == (
        e_b.rows,
        e_b.terms,
        tuple(e_b.columns),
    ) and all(
        _same_column_values(
            _label_values(e_a.columns[k])[0], _label_values(e_b.columns[k])[0]
        )
        for k in e_a.columns
    )


def _same_dependence_provenance(
    p_a: DependenceProvenance | None, p_b: DependenceProvenance | None
) -> bool:
    """Return True if two optional ``DependenceProvenance`` records match."""
    if p_a is None or p_b is None:
        return p_a is None and p_b is None
    if (p_a.used, tuple(p_a.columns)) != (
        p_b.used,
        tuple(p_b.columns),
    ) or not same_bits(p_a.used_missing_union, p_b.used_missing_union):
        return False
    for k, c_a in p_a.columns.items():
        c_b = p_b.columns[k]
        if (
            c_a.name != k
            or (c_a.name, c_a.roles, c_a.used_in_fit, c_a.source_dtype, c_a.pandas_kind)
            != (c_b.name, c_b.roles, c_b.used_in_fit, c_b.source_dtype, c_b.pandas_kind)
            or not same_bits(c_a.missing_mask, c_b.missing_mask)
            or not _same_column_values(c_a.values, c_b.values)
        ):
            return False
    return True


def _frames_match(frame: PreparedFrame, baseline: PreparedFrame) -> bool:
    """Return True if two concrete ``PreparedFrame`` instances match on every part."""
    return (
        (frame.subset_missing, frame.columns)
        == (baseline.subset_missing, baseline.columns)
        and _same_index_values(frame._labels, baseline._labels)
        and _same_model_frame(frame._frame, baseline._frame)
        and _same_sample_base(frame.base, baseline.base)
        and _same_effects(frame.effects, baseline.effects)
        and _same_dependence_provenance(frame.dependence, baseline.dependence)
    )


def _check_frame_matches_baseline(
    frame: PreparedFrame, baseline: PreparedFrame
) -> None:
    """Validate that ``frame`` has not been mutated relative to ``baseline``.

    The ``PreparedFrame`` baseline is compared part by part (:func:`_frames_match`);
    then the live frame's own consistency is verified.
    """
    if type(frame) is not PreparedFrame:
        raise TypeError("frame and baseline must be concrete PreparedFrame instances")
    if type(baseline) is not PreparedFrame:
        raise TypeError("frame and baseline must be concrete PreparedFrame instances")
    if not _frames_match(frame, baseline):
        raise ValueError(
            f"PreparedFrame contents do not match issued baseline ({UNBOUND})"
        )
    _check_prepared_frame_and_provenance(frame)


def authenticate_unscoped_frame(
    frame: PreparedFrame, *, sample: FESample | None = None
) -> RetainedInputSnapshot | None:
    """Validate ``frame`` against its issued baseline and optional ``sample``."""
    if type(frame) is not PreparedFrame:
        raise TypeError("prepared must be a concrete PreparedFrame")
    _check_prepared_frame_and_provenance(frame)
    rec = _ISSUED_BY_FRAME_ID.get(id(frame))
    snap = None if rec is None or rec[0]() is not frame else rec[2]
    if rec is not None and rec[0]() is frame:
        _check_frame_matches_baseline(frame, rec[1])
    if sample is not None:
        check_prepared_matches_sample(frame, sample)
    return snap


def authenticate_unscoped_sample(sample: FESample) -> None:
    """Validate an unscoped ``sample`` against its issued frame baseline."""
    rec = _ISSUED_BY_BASE_ID.get(id(sample.base))
    if rec is None or rec[0]() is not sample.base:
        return
    baseline = rec[2]
    if (frame := rec[1]()) is not None:
        _check_frame_matches_baseline(frame, baseline)
        check_prepared_matches_sample(frame, sample)
    elif not _same_sample_base(sample.base, baseline.base):
        raise ValueError(f"SampleBase does not match issued baseline ({UNBOUND})")
    else:
        check_prepared_matches_sample(baseline, sample)


@dataclass(frozen=True)
class _OwnedOperation:
    """One unexposed operation's working data and snapshot construction maps."""

    frame: PreparedFrame
    input_snapshot: RetainedInputSnapshot
    maps: _AdmissionMaps
    numeric: NumericFit | None = None


@dataclass(frozen=True, init=False)
class PerFitPreparationContext:
    """Reusable preparation context isolating template storage from per-fit frames.

    The reusable context holds an owned template and tracks the frames it issues.
    A private numeric fit uses its own operation, frame and snapshot maps.
    """

    input_key: InputKey
    used_dependence: UsedDependence
    cluster_request: ClusterRequest | None
    hac_request: HacRequest | None
    conley_request: ConleyRequest | None
    input_snapshot: RetainedInputSnapshot
    _template: PreparedFrame
    _admitted_frames: list[PreparedFrame]
    _issued_frames: list[_IssuedFitFrame]
    _passive_template: PreparedFrame | None
    _operation: _OwnedOperation | None
    _normal_template: PreparedFrame | None
    _normal_origin: RetainedInputSnapshot | None

    def __init__(
        self,
        prepared: PreparedFrame,
        *,
        used_dependence: UsedDependence | None = None,
        cluster_request: ClusterRequest | None = None,
        hac_request: HacRequest | None = None,
        conley_request: ConleyRequest | None = None,
        clean_fit: bool = False,
    ) -> None:
        """Store an owned template ``PreparedFrame`` and track derived fit frames.

        ``clean_fit`` permits a single owned boundary capture for passive inputs.
        Each public fit then creates a separate operation with its own working
        records and construction maps; no public preparation is marked trusted.
        Foreign inputs keep the ordered general route.  The original is validated
        afresh first.
        """
        if type(prepared) is not PreparedFrame:
            raise TypeError("prepared must be a PreparedFrame")
        _check_prepared_frame_and_provenance(prepared)
        clean = clean_fit and passive_preparation(prepared)
        template = (
            own_prepared_once(prepared) if clean else _owned_prepared_frame(prepared)
        )
        in_snap = (
            RetainedInputSnapshot._from_owned_prepared(template)
            if clean
            else RetainedInputSnapshot.from_prepared(template)
        )
        dependence = template.dependence
        resolved_used = (
            dependence.used
            if used_dependence is None and dependence is not None
            else (used_dependence or UsedDependence())
        )
        object.__setattr__(self, "input_key", in_snap.key)
        object.__setattr__(self, "used_dependence", resolved_used)
        object.__setattr__(self, "cluster_request", cluster_request)
        object.__setattr__(self, "hac_request", hac_request)
        object.__setattr__(self, "conley_request", conley_request)
        object.__setattr__(self, "input_snapshot", in_snap)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_admitted_frames", [])
        object.__setattr__(self, "_issued_frames", [])
        object.__setattr__(self, "_passive_template", template if clean else None)
        object.__setattr__(self, "_operation", None)
        object.__setattr__(self, "_normal_template", None)
        object.__setattr__(self, "_normal_origin", None)
        if clean:
            # One detached boundary capture, not a copy for each registry entry.
            entry = _IssuedFitFrame(prepared, template, in_snap)
            self._admitted_frames.append(prepared)
            self._issued_frames.append(entry)
            _register_weak_evidence(prepared, template, in_snap)
        else:
            self._register_issued_frame(prepared, snapshot=in_snap)
            self._register_issued_frame(template, snapshot=in_snap)

    @classmethod
    def _from_normalized(
        cls,
        prepared: PreparedFrame,
        *,
        used_dependence: UsedDependence,
        cluster_request: ClusterRequest | None,
        single_fit: bool = False,
        deferred_retained: bool = False,
    ) -> PerFitPreparationContext | None:
        """Own one fresh canonical producer; do not replay a second private graph.

        prepare_frame already performed the external normalization boundary.
        Its original public frame gets an independent private capture here.
        Direct/foreign constructors retain their original comparisons and copies.
        A single_fit producer is never exposed as a reusable public preparation:
        eligible rows transfer into the existing operation without a second frame.
        deferred_retained is an unstored instruction from that producer only;
        actual deferred records are completed only if their consumer declines.
        """
        if not normalized_numeric_inputs(prepared):
            if single_fit and deferred_retained:
                _materialize_retained(prepared)
            return None
        transferred = single_fit_frame(prepared) if single_fit else None
        if transferred is None and single_fit and deferred_retained:
            # own_prepared_once eagerly copies provenance, but copies only the
            # metadata keys already present. Complete BOTH before it captures.
            _materialize_retained(prepared)
        template = own_prepared_once(prepared) if transferred is None else transferred
        snapshot = RetainedInputSnapshot._from_owned_prepared(template)
        operation = (
            None
            if transferred is None
            else _OwnedOperation(template, snapshot, _AdmissionMaps())
        )
        result = object.__new__(cls)
        entries = [
            _IssuedFitFrame(prepared, template, snapshot),
            _IssuedFitFrame(template, template, snapshot),
        ]
        for name, value in (
            ("input_key", snapshot.key),
            ("used_dependence", used_dependence),
            ("cluster_request", cluster_request),
            ("hac_request", None),
            ("conley_request", None),
            ("input_snapshot", snapshot),
            ("_template", template),
            ("_admitted_frames", [prepared, template]),
            ("_issued_frames", entries),
            ("_passive_template", None),
            ("_operation", operation),
            ("_normal_template", template if operation is None else None),
            ("_normal_origin", snapshot),
        ):
            object.__setattr__(result, name, value)
        if transferred is None:
            # A transferred template shares ``prepared.base``: registered as the
            # baseline it would keep that base, and so its own entry, alive for good.
            # The transferred frame is never handed out, so nothing can present it.
            _register_weak_evidence(prepared, template, snapshot)
        return result

    def _after_numeric_admission(
        self, frame: PreparedFrame, snapshot: RetainedInputSnapshot, numeric: NumericFit
    ) -> PerFitPreparationContext:
        """One private operation after its original full admission and ownership.

        Only normal composition calls this with its detached admitted snapshot.
        The reusable preparation and all public/foreign constructors remain
        unchanged. No passive eligibility result is changed or inferred here.
        """
        previous = self._operation
        if (
            type(previous) is _OwnedOperation
            and previous.numeric is None
            and previous.frame is frame
            and previous.input_snapshot is snapshot
        ):
            # This is the SAME unexposed operation that admitted the sample and
            # populated these maps, not a new owner of a public/foreign graph.
            object.__setattr__(
                self,
                "_operation",
                _OwnedOperation(frame, snapshot, previous.maps, numeric),
            )
            # Earlier private headers can share this base. Retain the actual
            # issued fit frame, as the original fresh-context composition did.
            object.__setattr__(self, "_admitted_frames", [frame])
            object.__setattr__(
                self, "_issued_frames", [_IssuedFitFrame(frame, frame, snapshot)]
            )
            return self
        operation = _OwnedOperation(frame, snapshot, _AdmissionMaps(), numeric)
        result = object.__new__(PerFitPreparationContext)
        for name in self.__dataclass_fields__:
            object.__setattr__(result, name, getattr(self, name))
        for name, value in (
            ("input_key", snapshot.key),
            ("input_snapshot", snapshot),
            ("_template", frame),
            ("_admitted_frames", [frame]),
            ("_issued_frames", [_IssuedFitFrame(frame, frame, snapshot)]),
            ("_passive_template", None),
            ("_operation", operation),
            ("_normal_template", None),
            ("_normal_origin", self._normal_origin),
        ):
            object.__setattr__(result, name, value)
        return result

    def for_fit(self) -> PerFitPreparationContext:
        """A separate operation from the already owned public preparation.

        No trust flag is toggled on this reusable preparation. Each caller gets
        separate working data, array headers and snapshot construction maps.
        Public callers cannot mutate this private template through their input
        dataframe or outward exports. Do not replay whole-frame comparisons to
        detect arbitrary private-record tampering.
        """
        normal = self._normal_template
        template = self._passive_template if normal is None else normal
        if template is None:
            return self
        frame = (
            own_prepared_once(template)
            if normal is None
            else normalized_fit_frame(template)
        )
        snapshot = RetainedInputSnapshot._from_owned_prepared(frame, key=self.input_key)
        operation = _OwnedOperation(frame, snapshot, _AdmissionMaps())
        ctx = object.__new__(PerFitPreparationContext)
        for name, value in (
            ("input_key", self.input_key),
            ("used_dependence", self.used_dependence),
            ("cluster_request", self.cluster_request),
            ("hac_request", self.hac_request),
            ("conley_request", self.conley_request),
            ("input_snapshot", snapshot),
            ("_template", frame),
            ("_admitted_frames", [frame]),
            ("_issued_frames", [_IssuedFitFrame(frame, frame, snapshot)]),
            ("_passive_template", None),
            ("_operation", operation),
            ("_normal_template", None),
            ("_normal_origin", self._normal_origin),
        ):
            object.__setattr__(ctx, name, value)
        return ctx

    def release_fit_state(self) -> None:
        """Release temporary construction anchors even when a fit raises."""
        if self._operation is not None:
            maps = self._operation.maps
            # Lazy outcomes keep their maps for diagnostic expansion. Keeping
            # them in the completed fit's memo would create a reference cycle.
            maps.outcome_memo.clear()
            maps.clear_anchors()

    def _register_issued_frame(
        self,
        frame: PreparedFrame,
        *,
        snapshot: RetainedInputSnapshot | None = None,
    ) -> _IssuedFitFrame:
        """Register the frame with a detached baseline and the shared snapshot."""
        baseline = _owned_prepared_frame(frame)
        snap = snapshot or RetainedInputSnapshot(
            baseline.base, key=self.input_key, prepared=baseline
        )
        entry = _IssuedFitFrame(frame=frame, baseline=baseline, input_snapshot=snap)
        self._admitted_frames.append(frame)
        self._issued_frames.append(entry)
        if frame is not baseline:
            # The module registries hold the evidence strongly until the frame dies;
            # the template as its own baseline would never die.  It is never handed
            # out, so no unscoped entry can present it: it lives with this context.
            _register_weak_evidence(frame, baseline, snap)
        return entry

    def _owned_template(self) -> PreparedFrame:
        """The private template of this context."""
        return self._template

    @property
    def template_frame(self) -> PreparedFrame:
        """Return a fresh deeply owned copy of the template frame."""
        return _owned_prepared_frame(self._owned_template())

    def new_fit_frame(self) -> PreparedFrame:
        """Create and register an isolated frame for one fit."""
        if self._operation is not None:
            return self._operation.frame
        frame = _owned_prepared_frame(self._owned_template())
        self._register_issued_frame(frame)
        return frame

    def owns_prepared(self, other: object) -> bool:
        """Return True if ``other`` is the template or a registered fit frame."""
        return type(other) is PreparedFrame and any(
            other is entry.frame for entry in self._issued_frames
        )

    def matching_frame(self, sample: FESample) -> PreparedFrame | None:
        """Return the admitted ``PreparedFrame`` whose ``base`` is ``sample.base``."""
        return next(
            (e.frame for e in self._issued_frames if e.frame.base is sample.base), None
        )

    def authenticate_frame(
        self, frame: object, *, sample: FESample | None = None
    ) -> RetainedInputSnapshot:
        """Verify ``frame`` identity and evidence before use.

        An unexposed owned operation consumes the records made by its own sample
        coordinator, with no repeated implementation-verification scans. It is
        never the state of a reusable public preparation. For reusable contexts the
        live frame is compared in full against its detached baseline, then the sample
        against the live frame.
        """
        for entry in self._issued_frames:
            if entry.frame is frame:
                if self._operation is not None:
                    if sample is not None and sample.base is not entry.frame.base:
                        raise ValueError(UNBOUND)
                    return entry.input_snapshot
                _check_frame_matches_baseline(entry.frame, entry.baseline)
                if sample is not None:
                    check_prepared_matches_sample(entry.frame, sample)
                return entry.input_snapshot
        raise ValueError(
            "outcome.prepared is not the exact PreparedFrame owned by preparation "
            f"({UNBOUND})"
        )

    def authenticate_sample(
        self, sample: FESample
    ) -> tuple[PreparedFrame, RetainedInputSnapshot]:
        """Verify the issued frame owning ``sample.base`` against its baseline."""
        frame = self.matching_frame(sample)
        snap = self.authenticate_frame(frame, sample=sample)
        assert frame is not None
        return frame, snap


@dataclass(frozen=True)
class _AdmissionMaps:
    """Grouped temporary per-handoff identity maps for interface admission."""

    capture_maps: CaptureMaps = field(default_factory=CaptureMaps)
    outcome_memo: dict[int, RetainedOutcomeSnapshot] = field(default_factory=dict)
    _outcome_refs: list[object] = field(default_factory=list)

    @property
    def input_map(self) -> InputIdentityMap:
        """Return the shared input identity map."""
        return self.capture_maps.input_map

    @property
    def grouping_map(self) -> GroupingIdentityMap:
        """Return the shared grouping identity map."""
        return self.capture_maps.grouping_map

    @property
    def partition_map(self) -> PartitionIdentityMap:
        """Return the shared partition identity map."""
        return self.capture_maps.partition_map

    @property
    def sample_map(self) -> SampleIdentityMap:
        """Return the shared sample identity map."""
        return self.capture_maps.sample_map

    @property
    def point_map(self) -> PointIdentityMap:
        """Return the shared point identity map."""
        return self.capture_maps.point_map

    @property
    def problem_map(self) -> ProblemIdentityMap:
        """Return the shared problem identity map."""
        return self.capture_maps.problem_map

    def clear_anchors(self) -> None:
        """Release all temporary original-object reference anchors after capture."""
        self.capture_maps.clear_anchors()
        self._outcome_refs.clear()
