"""Internal assessment handoff for a normal operation's admitted numeric owner.

This is not a public/foreign graph-admission replacement. Normal composition has already
admitted and detached its input sample, request and metadata. The solver, its callbacks
and provider are the package's own. Reuse the selected point's fresh certificate/actions
without capture/replicate of that same internal graph. The public AssessmentHandoff and
its mutation/callback contract are unchanged.
"""

from __future__ import annotations

from powermean.data.sample_fe_state import FESample
from powermean.fe.certificate_basis import CertificateBasis
from powermean.inference.absorbed import AbsorbedScale
from powermean.results.absorbed_assessment_handoff import HandoffReceipt
from powermean.solve.absorbed_certificate import (
    CandidateAudit,
    FEPointCertificate,
    _certify_numeric_point,
    assessment,
)
from powermean.solve.absorbed_certificate_history import index_history, profiled_history
from powermean.solve.absorbed_direction import IndexPoint
from powermean.solve.assessment import (
    Assessment,
    CertificateControls,
    certificate_controls,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.systems.profiled import PointProfile, ProfiledProblem


class NumericAssessmentHandoff:
    """One fit, one fixed numeric owner and only its actual assessed points."""

    def __init__(self, sample: FESample, problem: ProfiledProblem,
                 absorbed: AbsorbedOptions, scale: AbsorbedScale, *,
                 basis: CertificateBasis | None = None) -> None:
        self.sample, self.problem = sample, problem
        self.absorbed, self.scale, self.basis = absorbed, scale, basis
        self._kept: list[
            tuple[PointProfile | ConditionalPoint | IndexPoint, FEPointCertificate]
        ] = []
        self._assessed = 0
        self._audited = False

    def __call__(
        self,
        whole: WholePoint[PointProfile | ConditionalPoint] | WholePoint[IndexPoint],
        controls: CertificateControls, /,
    ) -> Assessment:
        """Form the same fresh full-system certificate, on fixed private inputs."""
        if (self._audited or whole.sample is not self.sample
                or whole.variance is not self.problem.variance):
            raise ValueError("the numeric assessment belongs to another fit")
        found = _certify_numeric_point(whole, controls, absorbed=self.absorbed,
                                       scale=self.scale, problem=self.problem,
                                       basis=self.basis)
        self._assessed += 1
        self._kept.append((whole.point, found))
        return assessment(found)

    def audit(self, candidate: ProfiledCandidate | IndexCandidate, *,
              scale: AbsorbedScale) -> tuple[CandidateAudit, HandoffReceipt]:
        """Admit native history; retain only the exact selected point's actions."""
        if self._audited:
            raise ValueError("the handoff has already audited a candidate")
        self._audited = True
        if type(candidate) is ProfiledCandidate:
            history = profiled_history(candidate)
        elif type(candidate) is IndexCandidate:
            history = index_history(candidate)
        else:
            raise TypeError("candidate must be a ProfiledCandidate or an IndexCandidate")
        if (history.sample is not self.sample or candidate.variance is not self.problem.variance
                or candidate.absorbed != self.absorbed or scale is not self.scale):
            raise ValueError("the numeric audit belongs to another fit")
        controls = certificate_controls(candidate.options)
        point, kept = history.point, self._kept
        self._kept = []
        certificate = next((c for p, c in reversed(kept)
                            if p is point and c.controls == controls), None)
        consumed = certificate is not None
        if certificate is None and isinstance(
            point, PointProfile | ConditionalPoint | IndexPoint
        ):
            certificate = _certify_numeric_point(
                WholePoint(history.sample, candidate.variance, point), controls,
                absorbed=candidate.absorbed, scale=scale, problem=self.problem,
                basis=self.basis,
            )
        reason = "" if consumed else "no owned certificate was kept for the selected record"
        receipt = HandoffReceipt(consumed, reason, self._assessed, len(kept), 0.0)
        return CandidateAudit(history, controls, certificate), receipt
