"""The internal within-fit handoff of an owned full point assessment to its audit.

Only on this explicitly activated route is the complete fresh certificate that the retry
assessor forms at the unchanged selected point, computed on owned inputs, passed to the
candidate's audit instead of being discarded and formed again.

The point-only retry assessor :class:`~powermean.solve.absorbed_certificate.FEAssessor`,
the audit :func:`~powermean.solve.absorbed_certificate.certify_candidate` and every
default are unchanged and separately executable. Normal joint composition activates this
handoff for the current covariance provider.

Activation.  The caller builds one :class:`AssessmentHandoff` per fit, with the absorbed
options and the provider that it would give ``FEAssessor``.  It passes the handoff to
the solver as the assessor, then has it audit the solver's candidate.

The owned route.  A call ``handoff(whole, controls)`` takes it exactly when all of these
hold:

- the handoff has not audited;
- the records are exact: a ``WholePoint`` of an ``FESample``, a built-in family
  (:class:`~powermean.model.variance.PowerVariance` or
  :class:`~powermean.model.variance.NB2Variance`), a ``PointProfile`` or an
  ``IndexPoint``, and ``CertificateControls``; the handoff holds exact
  ``AbsorbedOptions``, and its provider is the built-in
  :class:`~powermean.inference.absorbed.AbsorbedScale` of an exact
  :class:`~powermean.solve.absorbed_certificate_scale.CovarianceRequest`;
- the call belongs to this fit.  The first such call binds the fit to its sample,
  family and controls objects (a solver forms its controls once and passes the same
  three to every assessment); every later call must present those very objects again;
- the inputs are capturable, the fit part of the capture (sample, family, options,
  request, controls) equals the fit's, and the replica is exact
  (:mod:`powermean.results.absorbed_assessment_capture`).

Such a call is certified by the unchanged :func:`certify_point` on its replica only:

    certify_point(WholePoint(sample', family', point'), controls',
                  absorbed=options', scale=AbsorbedScale(request', wz=wz'))

The fit's replica of the sample, family, options, request and controls is made at the
fit's first call; WZ controls/source share that capture and its memo. Each call gets
a new replica of its point. Nothing the caller holds
is read by this computation, so no change of the caller's records, during it or after
it, reaches the certificate.  The handoff keeps the certificate with the capture of its
point and returns the certificate's assessment.  Every other call is ``FEAssessor``'s:
``certify_point`` of the caller's own records, kept for nothing.

The audit.  ``handoff.audit(candidate, scale=scale)`` follows ``certify_candidate``'s
sequence: the same type check, accepted history reader, controls of the candidate's
options and selected point, in that order and with the same refusals.  It is single
use: it takes every kept certificate and the fit's replica, and leaves none.  The
selected point's certificate is a kept one exactly when all of these hold:

- a kept call assessed that very record;
- ``scale`` is the built-in provider of a covariance request;
- the history's sample and the candidate's family are the fit's;
- a fresh capture of the audit's records equals, token for token, the capture that the
  call's replica was materialized from.  The audit's records are the history's sample,
  the candidate's family, the candidate's absorbed options, the request of ``scale``,
  the audit's controls and the selected record.

The certificate is then bound to the admitted history
(:mod:`powermean.results.absorbed_assessment_binding`).  Otherwise it is certified
afresh by ``certify_candidate``'s own call.  A :class:`HandoffReceipt` says which, and
why.  The history is admitted on its own and never selects, vetoes or accepts a
certificate; no verdict is combined.

Ownership.  Identities only scope a kept certificate to this fit.  They prove nothing
about values: the values are always compared, and the computation never reads the
caller's records.  No hash, marker, frozen status or write flag is evidence.

Limits.  A handoff is not thread safe.  A consumed certificate is always the certificate
of the very values that the audit's capture read.  A write made while a capture reads
the records can at most give a capture that no later one equals, and then the audit
certifies afresh.  Interpreter introspection (``gc``, closure cells, ``ctypes``) and
replacing the package's code are outside the route.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import cast

from powermean.data.sample_fe_state import FESample
from powermean.inference.absorbed import AbsorbedScale
from powermean.inference.wz_binding import layout
from powermean.inference.wz_binding import unavailable as wz_unavailable
from powermean.inference.wz_covariance import WZControls
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.results.absorbed_assessment_binding import Anchors, bound_certificate
from powermean.results.absorbed_assessment_capture import (
    Replica,
    Token,
    capture,
    replicate,
)
from powermean.solve.absorbed_certificate import (
    CandidateAudit,
    FEPointCertificate,
    assessment,
    certify_point,
)
from powermean.solve.absorbed_certificate_history import (
    CandidateHistory,
    index_history,
    profiled_history,
)
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleProvider,
)
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
from powermean.systems.profiled import PointProfile

AUDITED = "the handoff has already audited a candidate"
NO_POINT = "the candidate holds no selected point"
NOT_KEPT = "no owned certificate was kept for the selected record"
PROVIDER = "the audit's provider is not the built-in provider of a covariance request"
FOREIGN = "the history's sample or family is not the fit's"
UNCAPTURABLE = "the inputs of the audit are not capturable"
CHANGED = "the inputs of the audit are not those of the owned certificate"
UNBOUND = "the owned certificate does not bind to the admitted history"
FIT = 6  # sample, family, options, request, controls and WZ controls/source


@dataclass(frozen=True)
class HandoffReceipt:
    """What one audit did with the kept certificates (module docstring).

    ``consumed`` says whether the selected point's certificate is a kept one, and
    ``reason`` why not (empty when it is).  ``assessed`` counts the assessor's calls
    before the audit and ``kept`` the owned certificates kept for it.  ``capture`` is
    the elapsed seconds of every capture, replica, comparison and binding, the
    assessor's and the audit's.
    """

    consumed: bool
    reason: str
    assessed: int
    kept: int
    capture: float


@dataclass(frozen=True, eq=False)
class _Fit:
    """The fit of a handoff: the caller's three anchors and the fit's replica."""

    anchors: tuple[object, object, object]
    replica: Replica
    scale: AbsorbedScale


@dataclass(frozen=True, eq=False)
class _Kept:
    """An owned certificate, the record it was kept for and the capture of its point."""

    record: PointProfile | IndexPoint
    tokens: tuple[Token, ...]
    replica: Replica
    certificate: FEPointCertificate


def _built_in(scale: object) -> CovarianceRequest | None:
    """The request of the built-in provider of an exact request, else ``None``.

    The request is read only once the provider's exact class is known, so that no
    caller's code runs.
    """
    if type(scale) is not AbsorbedScale:
        return None
    request = scale.request
    return request if type(request) is CovarianceRequest else None


def _parts(
    whole: object, controls: object, absorbed: object
) -> tuple[FESample, PowerVariance | NB2Variance, PointProfile | IndexPoint] | None:
    """The sample, family and record of an owned call, each read once, else ``None``.

    Module docstring: the exact records of the owned route.
    """
    if type(whole) is not WholePoint:
        return None
    if (
        type(controls) is not CertificateControls
        or type(absorbed) is not AbsorbedOptions
    ):
        return None
    sample, variance, point = whole.sample, whole.variance, whole.point
    family = type(variance) is PowerVariance or type(variance) is NB2Variance
    organization = type(point) is PointProfile or type(point) is IndexPoint
    if type(sample) is not FESample or not (family and organization):
        return None
    return sample, variance, point


def _anchors(replica: Replica) -> Anchors:
    """The anchors of the roots of a call's replica, in the order of the capture."""
    sample, variance, _, request, controls, _, point = replica.roots
    return Anchors(
        cast("FESample", sample),
        cast("PowerVariance | NB2Variance", variance),
        cast("PointProfile | IndexPoint", point),
        cast("CovarianceRequest", request),
        cast("CertificateControls", controls),
    )


class AssessmentHandoff:
    """The private within-fit handoff of one fit (module docstring).

    Parameters
    ----------
    absorbed : AbsorbedOptions
        The absorbed options that ``FEAssessor`` would hold.
    scale : FEScaleProvider
        The provider that ``FEAssessor`` would hold.
    """

    def __init__(self, absorbed: AbsorbedOptions, scale: FEScaleProvider) -> None:
        self._absorbed = absorbed
        self._scale = scale
        self._fit: _Fit | None = None
        self._kept: list[_Kept] = []
        self._audited = False
        self._assessed = 0
        self._capture = 0.0

    def __call__(
        self,
        whole: WholePoint[PointProfile] | WholePoint[IndexPoint],
        controls: CertificateControls,
        /,
    ) -> Assessment:
        """``FEAssessor``'s assessment of ``whole`` under ``controls``."""
        self._assessed += 1
        started = time.perf_counter()
        owned = self._owned(whole, controls)
        self._capture += time.perf_counter() - started
        if owned is None:
            if _wz_provider(self._scale):
                raise wz_unavailable("current WZ inputs could not be owned exactly")
            found = certify_point(
                whole, controls, absorbed=self._absorbed, scale=self._scale
            )
            return assessment(found)
        replica, record = owned
        assert self._fit is not None
        mine = _anchors(replica)
        found = certify_point(
            WholePoint(mine.sample, mine.variance, mine.record),
            mine.controls,
            absorbed=replica.roots[2],
            scale=self._fit.scale,
        )
        tokens = replica.capture.rest(FIT)
        self._kept.append(_Kept(record, tokens, replica, found))
        return assessment(found)

    def _owned(
        self,
        whole: WholePoint[PointProfile] | WholePoint[IndexPoint],
        controls: CertificateControls,
    ) -> tuple[Replica, PointProfile | IndexPoint] | None:
        """The exact replica of an owned call and its record, else ``None``.

        Module docstring: the owned route.  Each input is read once, and the very
        objects read are checked, captured and kept as the record's anchor.
        """
        parts = None if self._audited else _parts(whole, controls, self._absorbed)
        request = _built_in(self._scale)
        if parts is None or request is None:
            return None
        sample, variance, point = parts
        anchors = (sample, variance, controls)
        fit = self._fit
        if fit is not None and not all(map(operator.is_, anchors, fit.anchors)):
            return None
        wz = self._scale.wz if type(self._scale) is AbsorbedScale else None
        _reserve_wz(wz, request, sample)
        captured = capture(
            (sample, variance, self._absorbed, request, controls, wz, point)
        )
        if captured is None:
            return None
        replica = replicate(captured, prefix=None if fit is None else fit.replica)
        if replica is None:
            return None
        if fit is None:
            pref = replica.prefix(FIT)
            rep_req = cast("CovarianceRequest", pref.roots[3])
            rep_scale = AbsorbedScale(
                rep_req,
                self._scale.preflight if type(self._scale) is AbsorbedScale else None,
                cast("WZControls | None", pref.roots[5]),
            )
            self._fit = _Fit(anchors, pref, rep_scale)
        return replica, point

    def audit(
        self, candidate: ProfiledCandidate | IndexCandidate, *, scale: FEScaleProvider
    ) -> tuple[CandidateAudit, HandoffReceipt]:
        """Audit ``candidate`` as ``certify_candidate`` does (module docstring)."""
        kept, audited, assessed, fit = (
            self._kept,
            self._audited,
            self._assessed,
            self._fit,
        )
        self._kept, self._audited, self._fit = [], True, None
        if type(candidate) is ProfiledCandidate:
            history = profiled_history(candidate)
        elif type(candidate) is IndexCandidate:
            history = index_history(candidate)
        else:
            raise TypeError(
                "candidate must be a ProfiledCandidate or an IndexCandidate"
            )
        controls = certificate_controls(candidate.options)
        point, certificate, reason = history.point, None, NO_POINT
        if isinstance(point, PointProfile | IndexPoint):
            certificate, reason = None, AUDITED
            if not audited:
                started = time.perf_counter()
                audit = (history, candidate, controls)
                certificate, reason = self._consumed(kept, fit, audit, scale)
                self._capture += time.perf_counter() - started
            if certificate is None:
                whole = WholePoint(history.sample, candidate.variance, point)
                absorbed = candidate.absorbed
                if _wz_provider(scale):
                    started = time.perf_counter()
                    certificate = _fresh_wz_certificate(
                        audit=(history, candidate, controls), scale=scale
                    )
                    self._capture += time.perf_counter() - started
                else:
                    certificate = certify_point(
                        whole, controls, absorbed=absorbed, scale=scale
                    )
        receipt = HandoffReceipt(not reason, reason, assessed, len(kept), self._capture)
        return CandidateAudit(history, controls, certificate), receipt

    def _consumed(
        self,
        kept: list[_Kept],
        fit: _Fit | None,
        audit: tuple[
            CandidateHistory, ProfiledCandidate | IndexCandidate, CertificateControls
        ],
        scale: object,
    ) -> tuple[FEPointCertificate | None, str]:
        """The selected point's owned certificate, bound, and ``""``; else why not.

        ``audit`` is the admitted history, the candidate and the audit's controls.
        """
        history, candidate, controls = audit
        point = history.point
        mine = [k for k in reversed(kept) if k.record is point]
        if not mine or fit is None:
            return None, NOT_KEPT
        request = _built_in(scale)
        if request is None:
            return None, PROVIDER
        sample, variance, _ = fit.anchors
        if history.sample is not sample or candidate.variance is not variance:
            return None, FOREIGN
        wz = scale.wz if type(scale) is AbsorbedScale else None
        _reserve_wz(wz, request, sample)
        roots = (sample, variance, candidate.absorbed, request, controls, wz, point)
        captured = capture(roots)
        if captured is None:
            return None, UNCAPTURABLE
        if captured.prefix(FIT).tokens != fit.replica.capture.tokens:
            return None, CHANGED
        given = captured.rest(FIT)
        found = next((k for k in mine if k.tokens == given), None)
        if found is None:
            return None, CHANGED
        record = cast("PointProfile | IndexPoint", point)
        live = Anchors(
            sample,
            variance,
            record,
            request,
            controls,
        )
        bound = bound_certificate(found.certificate, _anchors(found.replica), live)
        return (None, UNBOUND) if bound is None else (bound, "")


def _reserve_wz(
    wz: WZControls | None, request: CovarianceRequest, sample: FESample
) -> None:
    """Reserve WZ's current call before encoding any owned numerical replica."""
    if wz is not None and request.adjustment == "wz":
        if sample.design is None or request.name is None:
            raise ValueError("WZ requires current slopes and one pair source")
        layout(
            wz.source, request.name, sample.design.matrix.shape, wz.absorbed, wz.bounds
        )


def _wz_provider(scale: object) -> bool:
    """Whether the exact built-in provider requests WZ."""
    return type(scale) is AbsorbedScale and scale.request.adjustment == "wz"


def _fresh_wz_certificate(
    *,
    audit: tuple[
        CandidateHistory, ProfiledCandidate | IndexCandidate, CertificateControls
    ],
    scale: FEScaleProvider,
) -> FEPointCertificate:
    """Fresh WZ audits also compute on owned inputs and compare actual values."""
    history, candidate, controls = audit
    point = history.point
    if not isinstance(point, PointProfile | IndexPoint):
        raise wz_unavailable("the current joint point is unavailable")
    fresh = AssessmentHandoff(candidate.absorbed, scale)
    if isinstance(point, PointProfile):
        fresh(WholePoint(history.sample, candidate.variance, point), controls)
    else:
        fresh(WholePoint(history.sample, candidate.variance, point), controls)
    certificate, reason = fresh._consumed(fresh._kept, fresh._fit, audit, scale)
    if certificate is None:
        raise wz_unavailable(
            f"current WZ audit inputs changed during capture: {reason}"
        )
    return certificate
