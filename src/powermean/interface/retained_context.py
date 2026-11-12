"""Per-fit preparation context and temporary interface admission identity maps.

The opt-in primitive snapshot route of :class:`PerFitPreparationContext` captures its
state in the immutable :class:`~powermean.results.primitive_snapshot.PrimitiveCapture`,
read by its consumers through the once-materialized
:class:`~powermean.results.primitive_parts.PrimitiveParts`.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.dependence import DependenceProvenance, UsedDependence
from powermean.data.inference_requests import (
    ClusterRequest,
    ConleyRequest,
    HacRequest,
)
from powermean.data.sample import SampleBase
from powermean.data.numeric_fit import NumericFit
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_state import FESample, same_bits, same_missing
from powermean.formula.frame import Design, ModelFrame
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
from powermean.results.primitive_parts import PrimitiveParts
from powermean.results.primitive_snapshot import (
    GENERAL,
    PRIMITIVE,
    MetadataParts,
    PrimitiveCapture,
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
from powermean.interface.owned_fit_state import (
    own_prepared_once, passive_preparation, normalized_fit_frame, normalized_numeric_inputs,
    single_fit_frame,
)

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
# The evidence an issued frame is compared against: its detached ``PreparedFrame``
# baseline (general route) or the immutable capture of its context (primitive route).
type _Baseline = PreparedFrame | PrimitiveCapture
# The once-materialized parts of a primitive capture, which validate a live sample
# presented without the frame (``None`` on the general route).
type _Parts = PrimitiveParts | None
type _FrameEntry = tuple[_Ref[PreparedFrame], _Baseline, RetainedInputSnapshot, _Parts]
type _BaseEntry = tuple[
    _Ref[SampleBase], _Ref[PreparedFrame], _Baseline, RetainedInputSnapshot, _Parts
]

_ISSUED_BY_FRAME_ID: dict[int, _FrameEntry] = {}
_ISSUED_BY_BASE_ID: dict[int, _BaseEntry] = {}


@dataclass(frozen=True)
class _IssuedFitFrame:
    """Strong reference to an admitted ``PreparedFrame`` and its evidence.

    ``baseline`` is the frame's detached copy on the general route, or the immutable
    capture of its context on the primitive route.
    """

    frame: PreparedFrame
    baseline: _Baseline
    input_snapshot: RetainedInputSnapshot


def _register_weak_evidence(
    f: PreparedFrame, b: _Baseline, s: RetainedInputSnapshot, parts: _Parts
) -> None:
    """Index ``b``, ``s`` and ``parts`` by weak refs with cleanup callbacks."""
    fid, bid = id(f), id(f.base)

    def _on_frame_dead(ref: _Ref[PreparedFrame]) -> None:
        if (cf := _ISSUED_BY_FRAME_ID.get(fid)) is not None and cf[0] is ref:
            del _ISSUED_BY_FRAME_ID[fid]
        cb = _ISSUED_BY_BASE_ID.get(bid)
        if cb is not None and cb[1] is ref and cb[0]() is None:
            del _ISSUED_BY_BASE_ID[bid]

    def _on_base_dead(ref: _Ref[SampleBase]) -> None:
        if (cb := _ISSUED_BY_BASE_ID.get(bid)) is not None and cb[0] is ref:
            del _ISSUED_BY_BASE_ID[bid]

    f_ref = weakref.ref(f, _on_frame_dead)
    b_ref = weakref.ref(f.base, _on_base_dead)
    _ISSUED_BY_FRAME_ID[fid] = (f_ref, b, s, parts)
    _ISSUED_BY_BASE_ID[bid] = (b_ref, f_ref, b, s, parts)


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


def _check_frame_matches_baseline(frame: PreparedFrame, baseline: _Baseline) -> None:
    """Validate that ``frame`` has not been mutated relative to ``baseline``.

    A ``PreparedFrame`` baseline is compared part by part (:func:`_frames_match`); a
    :class:`PrimitiveCapture` baseline re-encodes the live ``frame`` and requires the
    identical capture (:meth:`PrimitiveCapture.check_frame`).  Both then verify the
    live frame's own consistency.
    """
    if type(frame) is not PreparedFrame:
        raise TypeError("frame and baseline must be concrete PreparedFrame instances")
    if type(baseline) is PrimitiveCapture:
        baseline.check_frame(frame)
    elif type(baseline) is not PreparedFrame:
        raise TypeError("frame and baseline must be concrete PreparedFrame instances")
    elif not _frames_match(frame, baseline):
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
    baseline, parts = rec[2], rec[4]
    if (frame := rec[1]()) is not None:
        _check_frame_matches_baseline(frame, baseline)
        check_prepared_matches_sample(frame, sample)
    elif type(baseline) is PrimitiveCapture:
        if not baseline.matches_base(sample.base):
            raise ValueError(f"SampleBase does not match issued baseline ({UNBOUND})")
        assert parts is not None
        parts.check_sample(sample)
    elif not _same_sample_base(sample.base, cast("PreparedFrame", baseline).base):
        raise ValueError(f"SampleBase does not match issued baseline ({UNBOUND})")
    else:
        check_prepared_matches_sample(cast("PreparedFrame", baseline), sample)


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

    ``snapshot_route`` records which route serves this context.  ``"general"`` (the
    default): the owned template ``_template`` is copied for every issued frame and
    for the input snapshot, every issued frame has a detached baseline, and every
    authentication compares the live frame in full against it.  ``"primitive"``
    (requested with ``primitive_snapshot=True`` and
    granted only to an eligible frame): the original is validated afresh and captured
    once into the immutable ``_capture``
    (:class:`~powermean.results.primitive_snapshot.PrimitiveCapture`, owned bytes,
    tuples and plain scalars; ``_template`` is ``None``); the parts of the capture
    that internal consumers read are materialized once into ``_parts``
    (:class:`~powermean.results.primitive_parts.PrimitiveParts`, read-only views of
    the capture's bytes and plain tuples, shared by this context, its input snapshot
    and the registry of its issued frames, never handed outward); every issued frame
    is a fresh detached materialization of the capture (the first one the
    materialization that proved the capture exact, ``_spare``); the capture is the
    evidence of every issued frame; every authentication validates the live sample it
    presents against ``_parts`` afresh, and the metadata consumer reads the base,
    labels and dependence of ``_parts`` (:meth:`_materialized_metadata`), never the
    live original graph and never a rebuilt frame.  ``snapshot_reason`` is the
    eligibility text when a primitive request was served by the general route.

    Ordinary passive preparation instead owns one private template. Each public
    reuse makes an isolated operation from that owned state, without comparing
    unexposed implementation records to prove our own construction again. Public
    input normalization and the foreign/generic route keep their original checks.
    """

    input_key: InputKey
    used_dependence: UsedDependence
    cluster_request: ClusterRequest | None
    hac_request: HacRequest | None
    conley_request: ConleyRequest | None
    input_snapshot: RetainedInputSnapshot
    snapshot_route: str
    snapshot_reason: str | None
    _template: PreparedFrame | None
    _capture: PrimitiveCapture | None
    _parts: PrimitiveParts | None
    _spare: list[PreparedFrame]
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
        primitive_snapshot: bool = False,
        clean_fit: bool = False,
    ) -> None:
        """Store an owned template ``PreparedFrame`` and track derived fit frames.

        ``clean_fit`` permits a single owned boundary capture for passive inputs.
        Each public fit then creates a separate operation with its own working
        records and construction maps; no public preparation is marked trusted.
        Foreign inputs keep the ordered general route. ``primitive_snapshot``
        requests the separate explicit primitive route (class docstring); it is
        never taken implicitly.  The original is validated afresh first; a request
        that the frame is not eligible for, or whose exact materialization does not
        match the original under the accepted comparison, is served by the general
        route with the reason recorded.
        """
        if type(prepared) is not PreparedFrame:
            raise TypeError("prepared must be a PreparedFrame")
        _check_prepared_frame_and_provenance(prepared)
        clean = clean_fit and not primitive_snapshot and passive_preparation(prepared)
        route, reason = GENERAL, None
        capture: PrimitiveCapture | None = None
        parts: PrimitiveParts | None = None
        spare: list[PreparedFrame] = []
        if primitive_snapshot:
            captured = PrimitiveCapture.capture(prepared)
            if isinstance(captured, str):
                reason = captured
            elif _frames_match(prepared, captured[1]):
                capture, detached = captured
                # The materialization that proved the capture exact is held nowhere
                # else: it is the first fit frame this context issues.  The parts the
                # consumers read are materialized once more, of their own, so that no
                # issued frame shares an object with them.
                spare.append(detached)
                parts = PrimitiveParts.of(capture)
                route = PRIMITIVE
            else:
                reason = "primitive materialization does not match the original"
        template: PreparedFrame | None = None
        dependence: DependenceProvenance | None
        if parts is not None:
            in_snap = RetainedInputSnapshot._from_parts(parts)
            dependence = parts.dependence
        else:
            template = own_prepared_once(prepared) if clean else _owned_prepared_frame(prepared)
            in_snap = (RetainedInputSnapshot._from_owned_prepared(template) if clean
                       else RetainedInputSnapshot.from_prepared(template))
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
        object.__setattr__(self, "snapshot_route", route)
        object.__setattr__(self, "snapshot_reason", reason)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_capture", capture)
        object.__setattr__(self, "_parts", parts)
        object.__setattr__(self, "_spare", spare)
        object.__setattr__(self, "_admitted_frames", [])
        object.__setattr__(self, "_issued_frames", [])
        object.__setattr__(self, "_passive_template", template if clean else None)
        object.__setattr__(self, "_operation", None)
        object.__setattr__(self, "_normal_template", None)
        object.__setattr__(self, "_normal_origin", None)
        if clean:
            # One detached boundary capture, not a copy for each registry entry.
            assert template is not None
            entry = _IssuedFitFrame(prepared, template, in_snap)
            self._admitted_frames.append(prepared)
            self._issued_frames.append(entry)
            _register_weak_evidence(prepared, template, in_snap, None)
        else:
            self._register_issued_frame(prepared, snapshot=in_snap)
            if template is not None:
                self._register_issued_frame(template, snapshot=in_snap)

    @classmethod
    def _from_normalized(
        cls, prepared: PreparedFrame, *, used_dependence: UsedDependence,
        cluster_request: ClusterRequest | None,
        single_fit: bool = False,
    ) -> PerFitPreparationContext | None:
        """Own one fresh canonical producer; do not replay a second private graph.

        prepare_frame already performed the external normalization boundary.
        Its original public frame gets an independent private capture here.
        Direct/foreign constructors retain their original comparisons and copies.
        A single_fit producer is never exposed as a reusable public preparation:
        eligible rows transfer into the existing operation without a second frame.
        """
        if not normalized_numeric_inputs(prepared):
            return None
        transferred = single_fit_frame(prepared) if single_fit else None
        template = own_prepared_once(prepared) if transferred is None else transferred
        snapshot = RetainedInputSnapshot._from_owned_prepared(template)
        operation = (
            None if transferred is None
            else _OwnedOperation(template, snapshot, _AdmissionMaps())
        )
        result = object.__new__(cls)
        entries = [_IssuedFitFrame(prepared, template, snapshot),
                   _IssuedFitFrame(template, template, snapshot)]
        for name, value in (
            ("input_key", snapshot.key), ("used_dependence", used_dependence),
            ("cluster_request", cluster_request), ("hac_request", None),
            ("conley_request", None), ("input_snapshot", snapshot),
            ("snapshot_route", GENERAL), ("snapshot_reason", None),
            ("_template", template), ("_capture", None), ("_parts", None),
            ("_spare", []), ("_admitted_frames", [prepared, template]),
            ("_issued_frames", entries), ("_passive_template", None),
            ("_operation", operation),
            ("_normal_template", template if operation is None else None),
            ("_normal_origin", snapshot),
        ):
            object.__setattr__(result, name, value)
        _register_weak_evidence(prepared, template, snapshot, None)
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
        if (type(previous) is _OwnedOperation and previous.numeric is None
                and previous.frame is frame and previous.input_snapshot is snapshot):
            # This is the SAME unexposed operation that admitted the sample and
            # populated these maps, not a new owner of a public/foreign graph.
            object.__setattr__(self, "_operation",
                               _OwnedOperation(frame, snapshot, previous.maps, numeric))
            # Earlier private headers can share this base. Retain the actual
            # issued fit frame, as the original fresh-context composition did.
            object.__setattr__(self, "_admitted_frames", [frame])
            object.__setattr__(self, "_issued_frames",
                               [_IssuedFitFrame(frame, frame, snapshot)])
            return self
        operation = _OwnedOperation(frame, snapshot, _AdmissionMaps(), numeric)
        result = object.__new__(PerFitPreparationContext)
        for name in self.__dataclass_fields__:
            object.__setattr__(result, name, getattr(self, name))
        for name, value in (
            ("input_key", snapshot.key), ("input_snapshot", snapshot),
            ("_template", frame), ("_capture", None), ("_parts", None),
            ("_spare", []), ("_admitted_frames", [frame]),
            ("_issued_frames", [_IssuedFitFrame(frame, frame, snapshot)]),
            ("_passive_template", None), ("_operation", operation),
            ("_normal_template", None), ("_normal_origin", self._normal_origin),
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
        frame = own_prepared_once(template) if normal is None else normalized_fit_frame(template)
        snapshot = RetainedInputSnapshot._from_owned_prepared(frame, key=self.input_key)
        operation = _OwnedOperation(frame, snapshot, _AdmissionMaps())
        ctx = object.__new__(PerFitPreparationContext)
        for name, value in (
            ("input_key", self.input_key), ("used_dependence", self.used_dependence),
            ("cluster_request", self.cluster_request), ("hac_request", self.hac_request),
            ("conley_request", self.conley_request), ("input_snapshot", snapshot),
            ("snapshot_route", GENERAL), ("snapshot_reason", None),
            ("_template", frame), ("_capture", None), ("_parts", None),
            ("_spare", []), ("_admitted_frames", [frame]),
            ("_issued_frames", [_IssuedFitFrame(frame, frame, snapshot)]),
            ("_passive_template", None), ("_operation", operation),
            ("_normal_template", None), ("_normal_origin", self._normal_origin),
        ):
            object.__setattr__(ctx, name, value)
        return ctx

    def release_fit_state(self) -> None:
        """Release temporary construction anchors even when a fit raises."""
        if self._operation is not None:
            self._operation.maps.clear_anchors()

    def _register_issued_frame(
        self,
        frame: PreparedFrame,
        *,
        snapshot: RetainedInputSnapshot | None = None,
    ) -> _IssuedFitFrame:
        """Record strong references and the comparison evidence for ``frame``.

        General route: a detached deep copy of ``frame`` is its baseline.  Primitive
        route: the immutable capture is the evidence of every issued frame (each is a
        materialization of the capture, or the validated original it was captured
        from), and the one input snapshot and the once-materialized parts are shared.
        """
        baseline: _Baseline
        if self._capture is not None:
            baseline, snap = self._capture, self.input_snapshot
        else:
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
            _register_weak_evidence(frame, baseline, snap, self._parts)
        return entry

    def _owned_template(self) -> PreparedFrame:
        """The owned template of the general route."""
        template = self._template
        assert template is not None
        return template

    def _materialized(self) -> PreparedFrame:
        """A fresh detached ``PreparedFrame`` materialized from the capture."""
        capture = self._capture
        assert capture is not None
        return capture.prepared_frame()

    def _materialized_metadata(self) -> MetadataParts:
        """The base, labels and dependence of the once-materialized parts.

        What the metadata consumer of the primitive route reads, its own (class
        docstring): the three parts that binding and checking dependence metadata
        read, and nothing else, materialized once with the context's parts, so no
        model frame, effect columns, ``PreparedFrame`` or second copy of those parts
        is rebuilt for it (:attr:`PrimitiveParts.metadata`).  The general route's
        consumer keeps its original frame and reads its parts lazily, in validation
        order, so it does not copy custom labels before earlier refusals.
        """
        parts = self._parts
        assert parts is not None
        return parts.metadata

    @property
    def template_frame(self) -> PreparedFrame:
        """Return a fresh deeply owned copy of the template ``PreparedFrame``.

        Primitive route: a fresh detached materialization of the capture.
        """
        if self._capture is not None:
            return self._materialized()
        return _owned_prepared_frame(self._owned_template())

    def new_fit_frame(self) -> PreparedFrame:
        """Create and register an isolated ``PreparedFrame`` for one fit.

        Primitive route: one detached materialization of the immutable capture (the
        first fit the verified materialization the capture was proved exact with,
        every later fit a fresh one), registered with the capture as its evidence; no
        baseline copy is made.
        """
        if self._operation is not None:
            return self._operation.frame
        if self._capture is None:
            frame = _owned_prepared_frame(self._owned_template())
        else:
            frame = self._spare.pop() if self._spare else self._materialized()
        self._register_issued_frame(frame)
        return frame

    @contextmanager
    def _primitive_entry(self) -> Iterator[None]:
        """Mark one external entry (``admit_interface_graph``); holds no state.

        Every authentication within the entry that presents a live sample validates
        that sample afresh against the capture (:meth:`authenticate_frame`): a prior
        success within the entry is not evidence that the sample is unchanged.
        """
        yield

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
        never the state of a reusable public preparation. General route: the live
        frame is compared in full against its detached
        baseline, then ``sample`` against the live frame.  Primitive route: the frame
        is identified as issued, and ``sample`` is validated afresh, on every call,
        against the once-materialized parts of the immutable capture that consumers
        read (:meth:`PrimitiveParts.check_sample`); the live frame's contents are not
        consumed and so are not re-observed.
        """
        for entry in self._issued_frames:
            if entry.frame is frame:
                if self._operation is not None:
                    if sample is not None and sample.base is not entry.frame.base:
                        raise ValueError(UNBOUND)
                    return entry.input_snapshot
                if self._parts is not None:
                    if sample is not None:
                        self._parts.check_sample(sample)
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
