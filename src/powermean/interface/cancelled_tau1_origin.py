"""Genuine ordered-coordinator origin for the exact private tau=1 inner sample.

Design 02.2-4 and the existing profiled_trigger contract. No changed sample is
adopted: a completed normal coordinator walk must reproduce the already constructed
inner problem exactly before its origin may be passed to the absorbed engines.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from powermean._prepare.compose import Columns, PreparedFrame
from powermean.data.collinearity_records import DiagnosticControls
from powermean.data.raw import Column, RawInputs
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_state import (
    FESample,
    _same_design,
    same_bits,
    same_transform,
)
from powermean.fe.admission_records import WorkLog
from powermean.formula.frame import model_frame
from powermean.interface.retention import admit_sample_outcome
from powermean.interface.sample import (
    SampleOutcome,
    SampleRequest,
    _owned_sample_walk_original,
    _prepare_owned_sample,
    _same_sample,
    admitted_outcome,
    prepare_sample,
    reprepare_recorded,
)
from powermean.interface.sample_records import SampleRecorder
from powermean.model.variance import CancelledFamily
from powermean.solve import cancelled_tau1_bridge
from powermean.solve.cancelled_records import Tau1OriginError
from powermean.solve.cancelled_tau1_bridge import (
    _BRIDGE_INPUTS,
    _bridge_inputs,
    _BridgeLayout,
)
from powermean.solve.profiled_trigger import (
    Admission,
    TraceView,
    Trigger,
    TriggerRequest,
)
from powermean.solve.support import BASELINE
from powermean.systems.admitted import CapturedFit


def _prepared(sample: FESample) -> PreparedFrame:
    """Bind normal formula metadata to the existing exact owned inner base."""
    source = sample.source
    if source is None:
        raise ValueError("tau=1 coordinator requires the exact inner FE source")
    raw = sample.base.inputs
    names = raw.column_names
    effect_names = tuple(block.name for block in source.blocks)
    numeric_formula = (
        model_frame is _MODEL_FRAME
        and type(raw) is RawInputs
        and all(type(name) is str for name in names)
        and (
            names == tuple(f"x{j}" for j in range(raw.nparams))
            or names == ("_absorbed_dummy",)
        )
        and all(type(name) is str and name.isidentifier() for name in effect_names)
        and not set(effect_names).intersection(("__z", "__offset", *names))
    )
    columns: dict[str, Column] = {}
    terms: list[tuple[str, ...]] = []
    for block in source.blocks:
        # Inner block names are package-created identifiers, not user formula text.
        # Retained admission compares the dtype to the encoded object levels.
        values = np.asarray(block.levels, dtype=object)[block.codes]
        columns[block.name] = Column(values, name=block.name)
        terms.append((block.name,))
    effects = EffectColumns(terms, columns, sample.base.nobs)
    return _prepared_frame(sample.base, effects, numeric_formula=numeric_formula)


def _prepared_frame(
    base: SampleBase, effects: EffectColumns, *, numeric_formula: bool
) -> PreparedFrame:
    """Fit the ordinary bridge formula on the already owned numeric inputs."""
    raw, effect_names = base.inputs, effects.names
    names = raw.column_names
    if numeric_formula:
        # The coordinator takes offset and FE values from base and effects.
        data = pd.DataFrame(
            {"__z": raw.y, **{name: raw.design[:, j] for j, name in enumerate(names)}},
            copy=False,
        )
    else:
        data = pd.DataFrame({"__z": raw.y, "__offset": raw.offset})
        for j, name in enumerate(names):
            data[name] = raw.design[:, j]
        for name, column in effects.columns.items():
            data[name] = column.values
    slopes = " + ".join(names)
    fixed = " + ".join(effect_names)
    frame = model_frame(f"__z ~ 0 + {slopes} | {fixed}", data)
    return PreparedFrame(
        frame=frame,
        labels=data.index,
        columns=Columns(offset="__offset"),
        base=base,
        subset_missing=0,
        effects=effects,
    )


_PREPARED = _prepared
_PREPARE = prepare_sample
_OWNED_PREPARE = _prepare_owned_sample
_REQUEST = SampleRequest
_MODEL_FRAME = model_frame
_PREPARED_FRAME = _prepared_frame
_SAME_SAMPLE = _same_sample
_WORK_LOG = WorkLog
_WORK_RUN = WorkLog.run


def _owned_origin_original(work: WorkLog) -> bool:
    """Use the direct binding only at the original coordinator boundaries."""
    return (
        type(work) is _WORK_LOG
        and getattr(work.run, "__func__", None) is _WORK_RUN
        and _prepared is _PREPARED
        and _prepared_frame is _PREPARED_FRAME
        and prepare_sample is _PREPARE
        and _prepare_owned_sample is _OWNED_PREPARE
        and SampleRequest is _REQUEST
        and model_frame is _MODEL_FRAME
        and _same_sample is _SAME_SAMPLE
        and _owned_sample_walk_original()
    )


def _same_inner_sample(found: FESample | None, sample: FESample) -> bool:
    """Check the bridge's exact original rows and all derived solver coordinates."""
    return (
        found is not None
        and _same_sample(found, sample)
        and type(sample.active) is np.ndarray
        and found.active.dtype == sample.active.dtype
        and same_bits(found.active, sample.active)
        and type(found.problem) is type(sample.problem)
        and type(found.problem.selection) is type(sample.problem.selection)
        and same_transform(found.problem, sample.problem)
        and type(found.design) is type(sample.design)
        and _same_design(found.design, sample.design)
    )


class _Coordinator:
    """The existing recorded recomposition/admission policy, on a genuine origin."""

    def __init__(self, origin: SampleOutcome) -> None:
        self._origin = origin

    def __call__(self, request: TriggerRequest, /) -> TraceView:
        """Return the normal frozen trace; a propagated failure remains a failure."""
        if request.sample is not self._origin.sample:
            raise ValueError("tau=1 alarm names another inner sample")
        recorder = SampleRecorder(self._origin)
        try:
            reprepare_recorded(
                self._origin,
                support=request.support,
                recession=request.recession,
                recorder=recorder,
            )
        except Exception as error:
            return recorder.failed(error)
        return recorder.snapshot()


def tau1_coordinator(
    sample: FESample,
    variance: CancelledFamily,
    *,
    diagnostic: DiagnosticControls,
    mode: str,
    work: WorkLog,
) -> tuple[Trigger, Admission]:
    """Require a real complete, exactly equivalent origin; never adopt its changes."""
    owned = type(sample) is FESample and _owned_origin_original(work)

    prepared = work.run("tau1_origin_frame", lambda: _prepared(sample))
    request = SampleRequest(
        omega=1.0,
        positive_only=False,
        variance=variance,
        keep_singletons=True,
        support=BASELINE,
        recession=None,
        collinearity=diagnostic,
        mode=mode,
    )
    prepare = _prepare_owned_sample if owned else prepare_sample
    origin = work.run("tau1_origin_prepare", lambda: prepare(prepared, request))
    if origin.status != "complete" or not _same_inner_sample(origin.sample, sample):
        found = origin.sample
        detail = {"status": origin.status}
        if found is not None:
            detail.update(
                rows_equal=str(np.array_equal(found.positions, sample.positions)),
                reasons_equal=str(found.reasons == sample.reasons),
                base_same=str(found.base is sample.base),
                codes_equal=str(
                    np.array_equal(found.ledger.codes, sample.ledger.codes)
                ),
            )
        # A refusal retains the ordinary replayed origin, with all of its evidence.
        checked = replace(origin) if owned else origin
        snapshot = work.run(
            "tau1_origin_failure_retention", lambda: admit_sample_outcome(checked)
        )
        raise Tau1OriginError(
            f"tau=1 coordinator origin changes the exact inner problem: {detail}",
            snapshot,
        )

    def bind() -> SampleOutcome:
        if not owned:
            return replace(origin, sample=sample)
        # The fresh, unpublished origin already holds the checks from its real
        # walk. The sample just checked against it must be the solver's exact
        # object so any later alarm is bound to the right problem.
        object.__setattr__(origin, "sample", sample)
        return origin

    bound = work.run("tau1_origin_bind", bind)
    return _Coordinator(bound), admitted_outcome


def reduced_tau1_origin(
    capture: CapturedFit,
    variance: CancelledFamily,
    *,
    diagnostic: DiagnosticControls,
    mode: str,
    work: WorkLog,
) -> tuple[_BridgeLayout, Trigger, Admission] | None:
    """Run the genuine coordinator once and bind its unchanged full-row sample."""
    if not (
        _owned_origin_original(work)
        and cancelled_tau1_bridge._bridge_inputs is _BRIDGE_INPUTS
        and type(capture) is CapturedFit
        and capture.obs_weights is None
    ):
        return None
    geometry = capture.admission.geometry
    structure = geometry.structure
    if (
        geometry.selected_columns is None
        or not geometry.n_slopes_identified
        or not structure.block_sizes
        or geometry.annihilated_block_indices
        or structure.n_groups < max(structure.block_sizes)
    ):
        return None
    z = capture.z
    if np.any(z == 0.0) and np.any(
        np.bincount(structure.cancel_codes, weights=z, minlength=structure.n_groups)
        == 0.0
    ):
        return None  # Keep the existing positive-group bridge and outer accounting.
    base, effects, retained = work.run(
        "tau1_bridge_inputs", lambda: _bridge_inputs(capture, slice(None), z.size)
    )
    if len(retained) != len(structure.block_sizes):
        return None
    prepared = work.run(
        "tau1_origin_frame",
        lambda: _prepared_frame(base, effects, numeric_formula=True),
    )
    request = SampleRequest(
        omega=1.0,
        positive_only=False,
        variance=variance,
        keep_singletons=True,
        support=BASELINE,
        recession=None,
        collinearity=diagnostic,
        mode=mode,
    )
    parts = prepared._order_parts(1.0)
    origin = work.run(
        "tau1_origin_prepare",
        lambda: _prepare_owned_sample(prepared, request, order_parts=parts),
    )
    sample, order = origin.sample, parts[0]
    if (
        origin.status != "complete"
        or sample is None
        or sample.count != z.size
        or sample.base is not base
        or sample.data is not order.data
        or sample.problem is not order.problem
        or sample.design is not order.design
        or any(reason is not None for reason in sample.reasons)
    ):
        return None
    return (
        _BridgeLayout(sample, order.positions, retained),
        _Coordinator(origin),
        admitted_outcome,
    )
