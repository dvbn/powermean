"""Same-point observed influence, initial covariance and direct scale of FE points.

Influence from the actual Jacobian; robust and one-way cluster covariance under the
``"stata"`` preset; frequency weights as replicated rows; the standard errors of the
coefficient gate.  Internal module: the provider of
:mod:`powermean.solve.absorbed_certificate`, whose contract
(:mod:`powermean.solve.absorbed_certificate_scale`) it imports; nothing of the solve
stage imports this module.  The evidence it forms owns its arrays from construction.

Influence.  With the fresh geometry's ``B = [X, E_J]``, ``b`` and the slope map
``W = H^-T E_s`` that the certificate formed from ``K`` transpose solves, the observed
same-point influence of row ``i`` is ``u_i = b_i (B_i W)``: the slope rows of the full
joint ``H^-1 B' diag(b)``, by gathers in block order, no dense inverse and no
fabricated ``Influence`` or ``JointEvaluation`` record.

Covariance (:func:`fe_covariance`), the existing convention once, by its
primitives: ``N`` the exact frequency total or the rows of positive weight
(``statistical_count``), never a divisor of the rows; robust: rows ``u`` (``u/sqrt(f)``
for frequency weights), one stored row padded with a zero row as the convention does,
``hc_covariance`` of the source-average rows ``m u`` and the factor ``N/(N - 1)``;
one cluster: the codes of the request bound to the sample's real ``SamplingUnits``
(``_active_codes``), ``cluster_covariance`` of ``m u`` and ``G/(G - 1)``.  The
primitive estimate, its symmetric part, diagnostics and warnings are kept in the
:class:`~powermean.inference.convention.Covariance` record.  Standard errors are the
direct map of the actual ``OrderScale`` (``direct_standard_errors``): representable
where the full reported covariance need not be; a zero stays zero, a negative
variance missing; nothing is floored.  An error of the convention is raised, not
caught.  No other covariance method, factor or bread is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.inference.absorbed_composition import generalized_scale
from powermean.inference.absorbed_preflight import (
    AbsorbedFrameRecord,
    AbsorbedPreflight,
    _preflight_matches,
    default_details,
    preflight_absorbed_scale,
    sampling_units,
)
from powermean.inference.absorbed_rows import influence_rows
from powermean.inference.contributions import statistical_count
from powermean.inference.convention import (
    ClusterCodes,
    Covariance,
    _active_codes,
    _replicated_rows,
    _source_average,
    robust_factor,
)
from powermean.inference.provider import direct_standard_errors
from powermean.inference.vcov import (
    CovarianceEstimate,
    cluster_covariance,
    hc_covariance,
)
from powermean.inference.wz_absorbed import joint_scale
from powermean.inference.wz_binding import unavailable as wz_unavailable
from powermean.inference.wz_covariance import WZControls
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result
from powermean.solve.absorbed_certificate_scale import (
    CovarianceRequest,
    FEScaleEvidence,
    FEScaleInput,
    FEScaleResult,
)
from powermean.solve.reported import ReportedScale
from powermean.systems.joint import SamplingUnits


def _cluster(
    influence: FloatArray, request: CovarianceRequest, units: SamplingUnits
) -> tuple[CovarianceEstimate, float, int]:
    """The uncorrected one-way estimate of the bound codes, ``G/(G - 1)`` and ``G``."""
    if request.name is None or request.codes is None:
        raise ValueError("a cluster request needs its name and codes")
    cluster = ClusterCodes(request.name, request.codes)
    codes = _active_codes(cluster, units)
    estimate = cluster_covariance(_source_average(influence), {cluster.name: codes})
    groups = estimate.components[0].n_clusters
    return estimate, groups / (groups - 1), groups


def fe_covariance(
    influence: FloatArray,
    weights: FloatArray,
    *,
    frequency: bool,
    request: CovarianceRequest,
    units: SamplingUnits,
) -> Covariance:
    """The requested covariance of influence rows, its factor applied once.

    ``influence`` and ``weights`` are those of the units' active rows, in their order.
    """
    count = statistical_count(weights, frequency=frequency)
    clusters: int | None = None
    if request.kind == "robust":
        rows = _replicated_rows(influence, weights) if frequency else influence
        if rows.shape[0] == 1:
            rows = np.vstack([rows, np.zeros_like(rows)])
        estimate, factor = hc_covariance(_source_average(rows)), robust_factor(count)
    else:
        estimate, factor, clusters = _cluster(influence, request, units)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance: FloatArray = factor * estimate.covariance
    return Covariance(
        covariance=_immutable(_finite_result(covariance, "covariance")),
        kind=request.kind,
        factor=factor,
        count=count,
        clusters=clusters,
        estimate=estimate,
    )


@dataclass(frozen=True, eq=False)
class AbsorbedScale:
    """The FE scale provider of one requested covariance.

    Called with an :class:`~powermean.solve.absorbed_certificate_scale.FEScaleInput`,
    it returns an ``FEScaleResult`` holding that very input and evidence bound to it.
    """

    request: CovarianceRequest
    preflight: AbsorbedPreflight | None = None
    wz: WZControls | None = None

    def __post_init__(self) -> None:
        """The request must be a :class:`CovarianceRequest`."""
        if type(self.request) is not CovarianceRequest:
            raise TypeError("request must be a CovarianceRequest")
        if self.preflight is not None and type(self.preflight) is not AbsorbedPreflight:
            raise TypeError("preflight must be an AbsorbedPreflight or None")

    def __call__(self, scale_input: FEScaleInput, /) -> FEScaleResult:
        """The influence, covariance and direct standard errors at ``scale_input``."""
        if type(scale_input) is not FEScaleInput:
            raise TypeError("scale_input must be an FEScaleInput")
        if scale_input.request is not self.request:
            raise ValueError("the input binds another covariance request")
        if self.request.adjustment == "wz":
            if self.wz is None:
                raise wz_unavailable(
                    "current WZ controls and original declared FE source required"
                )
            return joint_scale(scale_input, self.wz)
        geometry = scale_input.geometry
        if self.request.is_default_stata:
            rows, units = geometry.rows, sampling_units(geometry.point.sample)
            influence = influence_rows(scale_input)
            active = units.identifiers
            covariance = fe_covariance(
                influence[active],
                rows.weights[active],
                frequency=rows.frequency,
                request=self.request,
                units=units,
            )
            errors = direct_standard_errors(covariance, scale_input.order)
            reported = ReportedScale(
                gamma=geometry.point.theta,
                standard_errors=errors,
                order=scale_input.order,
            )
            evidence = FEScaleEvidence(
                scale_input.binding,
                scale_input.request,
                influence,
                covariance,
                reported,
                details=(
                    default_details(self.preflight, scale_input, covariance)
                    if self.preflight is not None
                    and _preflight_matches(self.preflight, scale_input)
                    else None
                ),
            )
            return FEScaleResult(scale_input, evidence)
        return self._generalized(scale_input)

    def _generalized(self, scale_input: FEScaleInput) -> FEScaleResult:
        """Compose the full actual request at this point."""
        return generalized_scale(scale_input, self.preflight)


__all__ = [
    "AbsorbedFrameRecord",
    "AbsorbedPreflight",
    "AbsorbedScale",
    "fe_covariance",
    "influence_rows",
    "preflight_absorbed_scale",
    "sampling_units",
]
