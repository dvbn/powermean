"""The recorded recomposition trigger of a joint fixed-effect fit.

:class:`_Coordinator` is the trigger that :mod:`powermean.interface.fe_dispatch`
hands to the profiled solve: one recorded recomposition of the fit's own sample
origin (``SampleRecorder`` and ``reprepare_recorded``), never a caller's callback.

The private joint-PPML standard and full audit entries that this module held
(contract V1 of 26 October 2026), with the six ``results.ppml_*`` modules they used,
left the 0.1 tree under owner's rulings 105 and 106: nothing in the package called
them.  They are unchanged at commit ``4a5121b``.
"""

from __future__ import annotations

from powermean.interface.sample import SampleOutcome, reprepare_recorded
from powermean.interface.sample_records import SampleRecorder
from powermean.solve.profiled_trigger import TraceView, TriggerRequest


class _Coordinator:
    """The entry's own trigger: one recorded recomposition of the origin (docstring)."""

    def __init__(self, origin: SampleOutcome) -> None:
        self._origin = origin

    def __call__(self, request: TriggerRequest, /) -> TraceView:
        """The frozen trace of the call, a propagated failure included."""
        recorder = SampleRecorder(self._origin)
        try:
            reprepare_recorded(
                self._origin,
                support=request.support,
                recession=request.recession,
                recorder=recorder,
            )
        except Exception as error:  # the trace keeps it: the Trigger protocol's record
            return recorder.failed(error)
        return recorder.snapshot()
