"""The joint FE fit outside ``tau in [1, 2]``: initializer, root solve, audit.

Outside ``[1, 2]`` the existence of a finite root is not characterized: rank checks
only, the solver and the certificate decide, with a ``CoverageWarning``.  This module
is the interface layer of
:func:`~powermean.solve.absorbed_root.solve_absorbed_root`, called by the normal joint
FE dispatch (:mod:`powermean.interface.fe_dispatch`) for a power variance outside
``[1, 2]`` at ``omega != 0``.

The sample.  The joint dispatch composed it, as for every joint fit, from its
explicit request: for such a variance the rank-only route
(:func:`~powermean.interface.sample_records._cone`): the closure of steps 5 and 6
and the collinearity analysis, no existence analysis and no deletion for it.  It is
the authoritative sample of the fit and of its initializer alike.

The initializer (:func:`convex_runner`).  The nearest convex member of the family on
that very sample object, by the organization the fit requested and its unchanged
solver, options, automatic start and internal within-fit assessment handoff, audited
as the convex joint route audits its candidate.  Its trigger
(:class:`SameSampleTrigger`) is the coordinator's recorded recomposition call of the
fit's own outcome with the family's TIGHTER controls: the rank-only request of the
target refuses existence controls before any analysis, and that refusal is the
call's recorded failure.  Nothing is recomposed, adopted or dropped; the solver
reads the record as it reads any trigger's, and an alarm leaves the initializer
uncertified, so that no root stage opens.

The coverage (:func:`coverage_warning`).  ``tau < 1``, or ``tau > 2`` with a zero
outcome on the composed sample, emits the existing ``coverage.outside_bbp`` warning
once, before the solve, at the caller's line.
"""

from __future__ import annotations

import warnings

import numpy as np

from powermean.data.sample_fe_state import FESample
from powermean.errors import CoverageWarning
from powermean.interface.sample import (
    SampleOutcome,
    admitted_outcome,
    reprepare_recorded,
)
from powermean.interface.sample_records import RecordedTrace, SampleRecorder
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.absorbed_assessment_handoff import AssessmentHandoff
from powermean.solve import newton_absorbed, newton_profiled
from powermean.solve.absorbed_certificate import CandidateAudit, audit_root_candidate
from powermean.solve.absorbed_certificate_scale import FEScaleProvider
from powermean.solve.absorbed_root import ConvexFit, ConvexRunner, solve_absorbed_root
from powermean.solve.cancelled_records import RootOptions
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_trigger import TriggerRequest


class SameSampleTrigger:
    """The convex initializer's trigger: one recorded recomposition call of the fit.

    The call of the coordinator (:func:`~powermean.interface.sample.
    reprepare_recorded`) from the fit's own outcome with the request's controls,
    recorded.  A refusal that propagates before the call entered the walk is
    recorded on the outcome's own request, as the call's failure; nothing is
    analyzed, recomposed or adopted (module docstring).
    """

    def __init__(self, origin: SampleOutcome) -> None:
        self._origin = origin

    def __call__(self, request: TriggerRequest, /) -> RecordedTrace:
        """The frozen record of the call, a propagated failure included."""
        origin = self._origin
        recorder = SampleRecorder(origin)
        try:
            reprepare_recorded(
                origin,
                support=request.support,
                recession=request.recession,
                recorder=recorder,
            )
        except Exception as error:  # the trace keeps it: the Trigger protocol's record
            try:
                return recorder.failed(error)
            except ValueError:
                recorder.bind(origin, origin.request)
                return recorder.failed(error)
        return recorder.snapshot()


def convex_runner(
    sample: FESample,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    scale: FEScaleProvider,
    outcome: SampleOutcome,
) -> ConvexRunner:
    """The same-sample convex fit of the requested organization (module docstring).

    The solver call and audit of the convex joint dispatch, unchanged, on ``sample``
    at the convex variance given, with :class:`SameSampleTrigger` of ``outcome``.
    """

    def run(variance: PowerVariance) -> ConvexFit:
        assessor = AssessmentHandoff(absorbed, scale)
        trigger = SameSampleTrigger(outcome)
        candidate: IndexCandidate | ProfiledCandidate
        if absorbed.absorbed == "index":
            candidate = newton_absorbed.solve(
                sample,
                variance,
                options,
                absorbed,
                trigger=trigger,
                admit=admitted_outcome,
                assess=assessor,
            )
        else:
            candidate = newton_profiled.solve(
                sample,
                variance,
                options,
                absorbed,
                trigger=trigger,
                admit=admitted_outcome,
                assess=assessor,
            )
        audit, _ = assessor.audit(candidate, scale=scale)
        return candidate, audit

    return run


def coverage_warning(
    sample: FESample, variance: PowerVariance, *, stacklevel: int
) -> None:
    """The ``CoverageWarning`` of an equation whose finite root BBP leaves open.

    ``tau < 1``, or ``tau > 2`` with a zero outcome on a row of positive weight of the
    composed sample; emitted once, before the solve, ``stacklevel`` frames up.
    """
    active = sample.data.weight_values > 0
    zeros = bool(np.any(sample.problem.response[active] == 0))
    if variance.tau > 2.0 and not zeros:
        return
    rows = "with zero outcomes" if zeros else "without zero outcomes"
    case = (
        f"The joint fit with absorbed fixed effects at tau = {variance.tau!r} {rows}, "
        "whose existence of a finite root is not characterized (rank checks only; "
        "the certificate decides),"
    )
    warnings.warn(
        CoverageWarning("coverage.outside_bbp", {"case": case}),
        stacklevel=stacklevel,
    )


def root_audit(
    sample: FESample,
    variance: PowerVariance,
    options: SolverOptions,
    absorbed: AbsorbedOptions,
    root_options: RootOptions,
    *,
    scale: FEScaleProvider,
    outcome: SampleOutcome,
    start: FloatArray | None,
) -> CandidateAudit:
    """Solve the joint FE equation outside ``[1, 2]`` and audit its candidate.

    The caller's mapped slopes, or the same-sample convex initializer; the root
    solve; :func:`~powermean.solve.absorbed_certificate.audit_root_candidate`.
    """
    convex = (
        None
        if start is not None
        else convex_runner(sample, options, absorbed, scale, outcome)
    )
    candidate = solve_absorbed_root(
        sample,
        variance,
        options,
        root_options,
        absorbed,
        scale=scale,
        convex=convex,
        start=start,
    )
    return audit_root_candidate(candidate)
