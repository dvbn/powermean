"""The binding of an owned point certificate to the history admitted by its audit.

Part of the internal within-fit full-assessment handoff.  A complete fresh certificate,
formed on an owned replica of its inputs
(:mod:`powermean.results.absorbed_assessment_capture`), may be passed through the fresh
admission of the candidate's history for the identical selected problem, point and
request.  This module is that passage and nothing else.  Only
:mod:`powermean.results.absorbed_assessment_handoff` imports it.

Why it is needed.  A certificate is bound to its inputs by identity as well as by value.
``point.sample``, ``point.record`` and ``basis.grouping`` are the very records it was
formed from, and so are the sample, the grouping and the family in the binding of its
scale input and evidence.  The existing admission of a retained result
(:func:`~powermean.results.absorbed_binding.admit_absorbed_graph`) requires them to be
those of the admitted history.  An owned certificate holds the replica's records
instead.

What it does (:func:`bound_certificate`).  The handoff calls it only once a fresh closed
capture of the audit's records equals, token for token, the capture that the replica was
materialized from.  It then re-forms, by their own constructors, exactly the records of
the certificate that hold an input, with each replica anchor replaced by the admitted
object at its place:

- the admitted point takes the sample, the family and the record;
- the basis takes the blocks;
- the inner evidence's auxiliary absorption, and a recovery formed by the certificate,
  take the blocks; a recovery held by an index record is that record's own;
- the geometry takes the point and the basis;
- the controls are those of the audit.

The scale input is formed afresh by its constructor from the re-formed geometry and the
audit's request, which binds it to the admitted records by identity and by value.  The
provider's evidence is re-formed under that binding only when its ORIGINAL binding, with
each replica anchor renamed to its admitted object, is that binding item for item.  The
pair is then admitted by the unchanged
:func:`~powermean.solve.absorbed_certificate_scale.admitted_scale`.

Every computed part is carried as it was formed and nothing is recomputed: rows, effect
equations, inner checks, geometry arrays, factors, measures, covariance, standard
errors, the coefficient check, errors, work and gate lists.  The gates are read again
from the re-formed certificate and must be the same.  ``None`` when any of this does not
hold; the handoff then certifies afresh.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from powermean.data.sample_fe_state import FESample
from powermean.fe.absorb import AbsorptionResult, AbsorptionState
from powermean.fe.certificate_basis import rebound_basis
from powermean.fe.groups import Grouping
from powermean.model.variance import NB2Variance, PowerVariance
from powermean.solve.absorbed_certificate import FEPointCertificate, failed_gates
from powermean.solve.absorbed_certificate_geometry import FEGeometry
from powermean.solve.absorbed_certificate_inner import InnerEvidence
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleInput,
    FEScaleResult,
    admitted_scale,
    bound_request,
)
from powermean.solve.absorbed_direction import IndexPoint, Recovery
from powermean.solve.assessment import CertificateControls
from powermean.systems.profiled import PointProfile

Family = PowerVariance | NB2Variance


@dataclass(frozen=True, eq=False)
class Anchors:
    """The inputs that a certificate holds by identity: one side of a binding.

    ``sample``, ``variance`` and ``record`` are the whole point, ``request`` the
    provider's covariance request and ``controls`` the certificate controls.
    """

    sample: FESample
    variance: Family
    record: PointProfile | IndexPoint
    request: CovarianceRequest
    controls: CertificateControls

    @property
    def grouping(self) -> Grouping | None:
        """The blocks of the sample."""
        return self.sample.grouping

    @property
    def recovered(self) -> Recovery | None:
        """The recovery that an index record holds, ``None`` otherwise."""
        record = self.record
        return record.recovered if type(record) is IndexPoint else None


class _UnboundError(Exception):
    """The certificate does not bind to the admitted records."""


def _absorption(
    result: AbsorptionResult | None, mine: Anchors, live: Anchors
) -> AbsorptionResult | None:
    """An absorption on the replica's blocks, re-formed on the admitted blocks."""
    if result is None:
        return None
    state = result.state
    if state.grouping is not mine.grouping or live.grouping is None:
        raise _UnboundError
    rebound = AbsorptionState(live.grouping, state.columns, state.coefficients)
    return replace(result, state=rebound)


def _inner(
    inner: InnerEvidence | None, mine: Anchors, live: Anchors
) -> InnerEvidence | None:
    """The inner evidence with its recovery and auxiliary absorption rebound."""
    if inner is None:
        return None
    if inner.normalization is not None:
        raise _UnboundError  # the evidence of a WLS point, which is never kept
    recovery = inner.recovery
    if recovery is not None and recovery is mine.recovered:
        recovery = live.recovered
        if recovery is None:
            raise _UnboundError
    elif recovery is not None:
        absorption = _absorption(recovery.absorption, mine, live)
        recovery = replace(recovery, absorption=absorption)
    auxiliary = _absorption(inner.auxiliary, mine, live)
    return replace(inner, recovery=recovery, auxiliary=auxiliary)


def _renamed(value: object, names: tuple[tuple[object, object], ...]) -> object:
    """The admitted object of a replica anchor; any other value as it is."""
    return next((live for mine, live in names if value is mine), value)


def _scale(
    found: FEScaleResult | None,
    geometry: FEGeometry | None,
    mine: Anchors,
    live: Anchors,
) -> FEScaleResult | None:
    """The provider's evidence, admitted under the scale input of the admitted ones."""
    if found is None:
        return None
    order = live.sample.order_scale
    if geometry is None or order is None:
        raise _UnboundError
    request = bound_request(live.request, geometry)
    scale_input = FEScaleInput(geometry, request, order, found.input.slope_map)
    names = (
        (mine.sample, live.sample),
        (mine.grouping, live.grouping),
        (mine.variance, live.variance),
    )
    evidence = found.evidence
    items = evidence.binding.items
    original = tuple((key, _renamed(item, names)) for key, item in items)
    if original != scale_input.binding.items:
        raise _UnboundError
    rebound = replace(evidence, binding=scale_input.binding, request=request)
    return admitted_scale(FEScaleResult(scale_input, rebound), scale_input)


def _formed_on(certificate: FEPointCertificate, mine: Anchors) -> bool:
    """Whether the certificate holds the replica's anchors in their places."""
    point, basis, geometry = certificate.point, certificate.basis, certificate.geometry
    links = geometry is None or (
        geometry.point is point
        and geometry.basis is basis
        and geometry.rows is certificate.rows
    )
    return (
        links
        and point.sample is mine.sample
        and point.variance is mine.variance
        and point.record is mine.record
        and basis.grouping is mine.grouping
        and certificate.controls is mine.controls
    )


def bound_certificate(
    certificate: FEPointCertificate, mine: Anchors, live: Anchors
) -> FEPointCertificate | None:
    """``certificate``, formed on ``mine``, bound to ``live`` (module docstring).

    Parameters
    ----------
    certificate : FEPointCertificate
        The certificate that ``certify_point`` formed on the replica.
    mine : Anchors
        The replica's records that the certificate was formed from.
    live : Anchors
        The audit's admitted records, whose closed capture equals the replica's.

    Returns
    -------
    FEPointCertificate or None
        The same evidence held by the admitted records; ``None`` when it does not bind.
    """
    if live.grouping is None or not _formed_on(certificate, mine):
        return None
    point = replace(
        certificate.point,
        sample=live.sample,
        variance=live.variance,
        record=live.record,
    )
    basis = rebound_basis(
        certificate.basis,
        live.grouping,
        source_owner=mine.sample,
        target_owner=live.sample,
    )
    geometry = certificate.geometry
    if geometry is not None:
        geometry = replace(geometry, point=point, basis=basis)
    try:
        inner = _inner(certificate.inner, mine, live)
        scale = _scale(certificate._scale, geometry, mine, live)
    except _UnboundError:
        return None
    found = replace(
        certificate,
        controls=live.controls,
        point=point,
        basis=basis,
        inner=inner,
        geometry=geometry,
        _scale=scale,
    )
    tol, secondary = live.controls.tol, live.controls.tol_secondary
    gates = (failed_gates(found, tol), failed_gates(found, secondary))
    return found if gates == (found.strict, found.secondary) else None
