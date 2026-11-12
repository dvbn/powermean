"""Narrow owned scale-boundary records to actual current inference classes."""

from __future__ import annotations

from dataclasses import dataclass

from powermean.data.inference_conventions import ConventionValues, PsdTreatment
from powermean.data.request_values import Omitted
from powermean.inference.absorbed import AbsorbedFrameRecord
from powermean.inference.absorbed_rank import AbsorbedCount
from powermean.inference.hc import HCCovariance
from powermean.inference.multiway import MultiwayCovariance
from powermean.inference.psd import PsdAdjustment
from powermean.inference.ssc import AppliedConvention
from powermean.inference.wz_operator import PairCovariance
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_scale_details import (
    Bread,
    FactorOwner,
    FEScaleDetails,
    SSCSource,
)
from powermean.systems.joint import SamplingUnits


@dataclass(frozen=True)
class InferenceDetails:
    """Actual kernel classes, following exact runtime type checks at the boundary."""

    bread: Bread
    requested_ssc: str | ConventionValues | Omitted
    resolved_preset: str
    ssc_source: SSCSource
    factor_owner: FactorOwner
    units: SamplingUnits
    count: int
    rank: int
    applied: AppliedConvention
    absorbed: AbsorbedCount | None
    multiway: MultiwayCovariance | None
    adjustment: PsdAdjustment | None
    frame: AbsorbedFrameRecord | None
    hc: HCCovariance | None
    psd: PsdTreatment | None
    reported_covariance: FloatArray | None
    covariance_error: str | None
    wz: PairCovariance | None


def typed_details(details: FEScaleDetails) -> InferenceDetails:
    """Reject arbitrary metadata classes instead of casting them into kernel types."""
    applied, absorbed = details.applied, details.absorbed
    multiway, adjustment, frame, hc = (
        details.multiway,
        details.adjustment,
        details.frame,
        details.hc,
    )
    if type(applied) is not AppliedConvention:
        raise TypeError("scale details need an actual AppliedConvention")
    if absorbed is not None and type(absorbed) is not AbsorbedCount:
        raise TypeError("scale details need an actual AbsorbedCount")
    if multiway is not None and type(multiway) is not MultiwayCovariance:
        raise TypeError("scale details need an actual MultiwayCovariance")
    if adjustment is not None and type(adjustment) is not PsdAdjustment:
        raise TypeError("scale details need an actual PsdAdjustment")
    if frame is not None and type(frame) is not AbsorbedFrameRecord:
        raise TypeError("scale details need an actual AbsorbedFrameRecord")
    if hc is not None and type(hc) is not HCCovariance:
        raise TypeError("scale details need an actual HCCovariance")
    if details.wz is not None and type(details.wz) is not PairCovariance:
        raise TypeError("scale details need an actual PairCovariance")
    return InferenceDetails(
        details.bread,
        details.requested_ssc,
        details.resolved_preset,
        details.ssc_source,
        details.factor_owner,
        details.units,
        details.count,
        details.rank,
        applied,
        absorbed,
        multiway,
        adjustment,
        frame,
        hc,
        details.psd,
        details.reported_covariance,
        details.covariance_error,
        details.wz,
    )
