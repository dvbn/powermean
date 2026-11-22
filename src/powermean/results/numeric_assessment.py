"""Private assessment handoff for a normal operation's admitted numeric owner.

This is not a public/foreign graph-admission replacement. Normal composition has
already admitted and detached its input sample, request and metadata. The solver,
its callbacks and provider are the package's own. Reuse the selected point's fresh
certificate/actions without capture/replicate of that same private graph.
The public AssessmentHandoff and its mutation/callback contract are unchanged.
"""

from __future__ import annotations

from powermean.data.sample_fe_state import FESample
from powermean.fe.certificate_basis import CertificateBasis
from powermean.inference.absorbed import AbsorbedScale
from powermean.inference.wz_operator import Information
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
from powermean.solve.m1_options import AbsorbedOptions
from powermean.solve.newton_absorbed import IndexCandidate
from powermean.solve.newton_profiled import ProfiledCandidate
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.systems.profiled import PointProfile, ProfiledProblem


class NumericAssessmentHandoff:
    """One fit, one fixed numeric owner and only its actual assessed points."""

    def __init__(
        self,
        sample: FESample,
        problem: ProfiledProblem,
        absorbed: AbsorbedOptions,
        scale: AbsorbedScale,
        *,
        basis: CertificateBasis | None = None,
    ) -> None:
        self.sample, self.problem = sample, problem
        self.absorbed, self.scale, self.basis = absorbed, scale, basis
        self._kept: list[
            tuple[
                PointProfile | ConditionalPoint | IndexPoint,
                FEPointCertificate,
                Information | None,
            ]
        ] = []
        self._assessed = 0
        self._audited = False

    def __call__(
        self,
        whole: WholePoint[PointProfile | ConditionalPoint] | WholePoint[IndexPoint],
        controls: CertificateControls,
        /,
    ) -> Assessment:
        """Form the same fresh full-system certificate, on fixed private inputs."""
        return assessment(self._assess(whole, controls))

    def strictly_assess(
        self,
        whole: WholePoint[PointProfile | ConditionalPoint],
        controls: CertificateControls,
        /,
    ) -> bool:
        """Keep the independent point certificate for the final audit's handoff."""
        return self._assess(whole, controls, keep_only_strict=True).level == "strict"

    def _assess(
        self,
        whole: WholePoint[PointProfile | ConditionalPoint] | WholePoint[IndexPoint],
        controls: CertificateControls,
        *,
        keep_only_strict: bool = False,
    ) -> FEPointCertificate:
        """Certify and keep this point's exact record for possible final selection."""
        if (
            self._audited
            or whole.sample is not self.sample
            or whole.variance is not self.problem.variance
        ):
            raise ValueError("the numeric assessment belongs to another fit")
        handoff = self.scale._wz_information
        if handoff is not None:
            handoff.take()
        found = _certify_numeric_point(
            whole,
            controls,
            absorbed=self.absorbed,
            scale=self.scale,
            problem=self.problem,
            basis=self.basis,
        )
        self._assessed += 1
        information = None if handoff is None else handoff.take()
        if not keep_only_strict or found.level == "strict":
            self._kept.append((whole.point, found, information))
        return found

    def audit(
        self, candidate: ProfiledCandidate | IndexCandidate, *, scale: AbsorbedScale
    ) -> tuple[CandidateAudit, HandoffReceipt]:
        """Admit native history; retain only the exact selected point's actions."""
        if self._audited:
            raise ValueError("the handoff has already audited a candidate")
        self._audited = True
        if (
            type(candidate) is not ProfiledCandidate
            and type(candidate) is not IndexCandidate
        ):
            raise TypeError(
                "candidate must be a ProfiledCandidate or an IndexCandidate"
            )
        if (
            candidate.sample is not self.sample
            or candidate.variance is not self.problem.variance
            or candidate.absorbed != self.absorbed
            or scale is not self.scale
        ):
            raise ValueError("the numeric audit belongs to another fit")
        # Reuse the fixed inputs and keep a fresh row cache for the history checks.
        problem = self.problem._fresh_certificate_problem()
        history = (
            profiled_history(candidate, _problem=problem)
            if type(candidate) is ProfiledCandidate
            else index_history(candidate, _problem=problem)
        )
        controls = certificate_controls(candidate.options)
        point, kept = history.point, self._kept
        self._kept = []
        selected = next(
            (
                (c, shared)
                for p, c, shared in reversed(kept)
                if p is point and c.controls == controls
            ),
            None,
        )
        certificate = None if selected is None else selected[0]
        if scale._wz_information is not None:
            scale._wz_information.latest = None if selected is None else selected[1]
        consumed = certificate is not None
        if certificate is None and isinstance(
            point, PointProfile | ConditionalPoint | IndexPoint
        ):
            certificate = _certify_numeric_point(
                WholePoint(history.sample, candidate.variance, point),
                controls,
                absorbed=candidate.absorbed,
                scale=scale,
                problem=self.problem,
                basis=self.basis,
            )
        if (
            type(candidate) is ProfiledCandidate
            and candidate.polished_alarm is not None
            and (certificate is None or certificate.level != "strict")
        ):
            raise ValueError("a fit polished after a zero-row alarm must be strict")
        reason = (
            "" if consumed else "no owned certificate was kept for the selected record"
        )
        receipt = HandoffReceipt(consumed, reason, self._assessed, len(kept), 0.0)
        return CandidateAudit(history, controls, certificate), receipt
