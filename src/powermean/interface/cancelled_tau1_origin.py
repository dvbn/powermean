"""Genuine ordered-coordinator origin for the exact private tau=1 inner sample.

This follows the existing profiled_trigger contract. No changed sample is
adopted: a completed normal coordinator walk must reproduce the already constructed
inner problem exactly before its origin may be passed to the absorbed engines.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from powermean._prepare.compose import Columns, PreparedFrame
from powermean.data.collinearity_records import DiagnosticControls
from powermean.data.raw import Column
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_state import FESample
from powermean.fe.admission_records import WorkLog
from powermean.formula.frame import model_frame
from powermean.interface.retention import admit_sample_outcome
from powermean.interface.sample import (
    SampleOutcome,
    SampleRequest,
    _same_sample,
    admitted_outcome,
    prepare_sample,
    reprepare_recorded,
)
from powermean.interface.sample_records import SampleRecorder
from powermean.model.variance import CancelledFamily
from powermean.solve.cancelled_records import Tau1OriginError
from powermean.solve.profiled_trigger import (
    Admission,
    TraceView,
    Trigger,
    TriggerRequest,
)
from powermean.solve.support import BASELINE


def _prepared(sample: FESample) -> PreparedFrame:
    """Bind normal formula metadata to the existing exact owned inner base."""
    source = sample.source
    if source is None:
        raise ValueError("tau=1 coordinator requires the exact inner FE source")
    raw = sample.base.inputs
    data = pd.DataFrame({"__z": raw.y, "__offset": raw.offset})
    for j, name in enumerate(raw.column_names):
        data[name] = raw.design[:, j]
    columns: dict[str, Column] = {}
    terms: list[tuple[str, ...]] = []
    for block in source.blocks:
        # Inner block names are package-created identifiers, not user formula text.
        values = np.asarray(block.levels, dtype=object)[block.codes]
        data[block.name] = values
        columns[block.name] = Column(values, name=block.name)
        terms.append((block.name,))
    slopes = " + ".join(raw.column_names)
    effects = " + ".join(block.name for block in source.blocks)
    frame = model_frame(f"__z ~ 0 + {slopes} | {effects}", data)
    return PreparedFrame(
        frame=frame,
        labels=data.index,
        columns=Columns(offset="__offset"),
        base=sample.base,
        subset_missing=0,
        effects=EffectColumns(terms, columns, sample.base.nobs),
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
    origin = work.run("tau1_origin_prepare", lambda: prepare_sample(prepared, request))
    if origin.status != "complete" or not _same_sample(origin.sample, sample):
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
        snapshot = work.run(
            "tau1_origin_failure_retention", lambda: admit_sample_outcome(origin)
        )
        raise Tau1OriginError(
            f"tau=1 coordinator origin changes the exact inner problem: {detail}",
            snapshot,
        )
    # The ordinary constructor replays the genuine walk and refuses a mismatch.
    bound = work.run("tau1_origin_bind", lambda: replace(origin, sample=sample))
    return _Coordinator(bound), admitted_outcome
