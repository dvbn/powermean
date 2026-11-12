"""Lazy cancelled-effect recovery at one whole stored reporting point.

BBP ``app:fe-postestimation``: for the original working index v excluding the cancelled
effect, log C = log mean_g(Z exp(-v)), log A = log C + mean_g(v), and the reported
effect is log C / omega. The arithmetic scale A uses the centered index M_C v. No
variance enters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from powermean.fe.cancel import CancelStructure, segment_sum, to_sorted
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.cancelled_bias import CancelledBias

if TYPE_CHECKING:
    from powermean.data.retained import OrderSample
    from powermean.results.cancelled_retained import RetainedCancelledEnvelope
    from powermean.results.wz_bias import WZBias

RecoveryPoint = Literal["original_uncorrected", "bbp_reporting_point"]


@dataclass(frozen=True)
class RecoveredScale:
    """Call-local arithmetic, with true zeros distinct from positive range loss."""

    effect: FloatArray
    scale: FloatArray
    log_scale: FloatArray
    status: tuple[str, ...]
    effect_status: tuple[str, ...]


def _log_group_mean(
    terms: FloatArray,
    structure: CancelStructure,
    positive: NDArray[np.bool_],
    present: NDArray[np.bool_],
) -> FloatArray:
    """Return grouped log means without exponentiating an unshifted row term."""
    peaks = np.maximum.reduceat(terms, structure.starts)
    relative = np.zeros(terms.size, dtype=np.float64)
    # Negligible positive summands may underflow here; the maximum stays one.
    with np.errstate(over="ignore", under="ignore"):
        relative[positive] = np.exp(
            terms[positive] - peaks[structure.sorted_codes[positive]]
        )
    sums = segment_sum(structure, relative)
    result = np.full(structure.counts.size, -np.inf, dtype=np.float64)
    result[present] = (
        peaks[present] + np.log(sums[present]) - np.log(structure.counts[present])
    )
    return result


def recover_scales(
    z: FloatArray, v: FloatArray, structure: CancelStructure, omega: float
) -> RecoveredScale:
    """Recover the cancelled-effect scales from already admitted, segment-sorted rows.

    Positive terms use group log-sum-exp, never exp(v) or exp(M_C v).
    Compute log A directly at M_C v to avoid cancellation of a common offset;
    retain the uncentered log C for the normalized log effect.
    Division before group summation avoids overflow in otherwise finite means.
    All-zero groups have scale zero and missing effects; positive out-of-range
    arithmetic scales are missing while their log coordinates are preserved.
    No tolerance, floor or variance-dependent weight is used.
    """
    counts = structure.counts
    codes = structure.sorted_codes
    mean_v = segment_sum(structure, v / counts[codes])
    positive = z > 0.0
    present = np.logical_or.reduceat(positive, structure.starts)
    log_z = np.full(z.size, -np.inf, dtype=np.float64)
    log_z[positive] = np.log(z[positive])
    log_c = _log_group_mean(log_z - v, structure, positive, present)
    with np.errstate(over="raise", invalid="raise"):
        centered = v - mean_v[codes]
    log_a = _log_group_mean(log_z - centered, structure, positive, present)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        scale = np.exp(log_a)
        effect = np.full(counts.size, np.nan, dtype=np.float64)
        # Unit FE column scales: exactly the nonzero-order OrderScale quotient.
        effect[present] = log_c[present] / omega
    overflow = present & np.isinf(scale)
    underflow = present & (scale == 0.0)
    scale[overflow | underflow] = np.nan
    status = np.full(counts.size, "positive", dtype=object)
    status[~present] = "zero"
    status[overflow] = "overflow"
    status[underflow] = "underflow"
    effect_status = np.full(counts.size, "finite", dtype=object)
    effect_status[~present] = "no_finite_effect"
    effect_status[present & np.isinf(effect)] = "overflow"
    lost_effect = present & (log_c != 0.0) & (effect == 0.0)
    effect_status[lost_effect] = "underflow"
    effect[np.isinf(effect) | lost_effect] = np.nan
    return RecoveredScale(
        effect, scale, log_a, tuple(status.tolist()), tuple(effect_status.tolist())
    )


def recovery_series(
    recovered: RecoveredScale,
    structure: CancelStructure,
    name: str,
    *,
    point: RecoveryPoint = "original_uncorrected",
) -> pd.Series[float]:
    """Present detached effects and centered-index scale metadata in code order."""
    out = pd.Series(
        recovered.effect.copy(),
        index=pd.Index(
            list(structure.cancel_levels),
            dtype=object,
            name=name,
            tupleize_cols=False,
        ),
        name=name,
    )
    out.attrs.update(
        cancelled_recovery={
            "scale": recovered.scale.copy(),
            "log_scale": recovered.log_scale.copy(),
            "status": recovered.status,
            "log_effect_status": recovered.effect_status,
            "point": point,
            "scale_definition": "groupmean(Z exp(-M_C v))",
            "normalization": "log(groupmean(Z exp(-v)))/omega",
        },
        evaluation_point=(
            "BBP reporting point postestimation"
            if point == "bbp_reporting_point"
            else "original uncorrected estimating root"
        ),
        certificate_point="uncorrected estimating root",
        incidental_effect_consistency="not asserted; fixed short groups remain noisy",
        plug_in_unbiasedness="not asserted for estimated scales or log effects",
        inference=(
            "no nuisance standard errors; standardized-functional inference is "
            "not provided"
        ),
    )
    return out


def _working_point(
    envelope: RetainedCancelledEnvelope, bias: CancelledBias | WZBias | None
) -> FloatArray:
    """Read the original T point, or invert BOTH stored BBP reporting vectors."""
    geometry = envelope.geometry
    if not isinstance(bias, CancelledBias):
        return geometry.expand(envelope._point.view())
    slopes = np.zeros(geometry.n_slopes_full, dtype=np.float64)
    slopes[geometry.kept_slope_indices] = envelope.order.solver_coefficients(
        bias.coefficients()
    )
    effects = bias.retained_effects()
    working_effects = (
        OrderScale(envelope.order.omega, np.ones(effects.size)).solver_coefficients(
            effects
        )
        if effects.size
        else effects
    )
    return np.concatenate((slopes, working_effects))


def cancelled_effect(
    sample: OrderSample,
    envelope: RetainedCancelledEnvelope,
    bias: CancelledBias | WZBias | None,
) -> pd.Series[float]:
    """Recover lazily from existing normal-API rows and one coherent point.

    Original and WZ views use the whole original point. BBP uses its stored
    corrected slopes and retained effects with omitted original slopes zero.
    Sorted implicit gathers include every retained block without dense dummies.
    The OrderSample's original transformed offset survives influence discard.
    """
    full = _working_point(envelope, bias)
    structure = envelope.geometry.structure
    at = envelope.geometry.n_slopes_full
    with np.errstate(over="raise", invalid="raise"):
        v = to_sorted(
            structure, sample.problem.offset + sample.design.matrix @ full[:at]
        )
        for codes, size in zip(
            structure.retained_sorted_codes, structure.block_sizes, strict=True
        ):
            v += full[at : at + size][codes]
            at += size
    recovered = recover_scales(
        to_sorted(structure, sample.problem.response), v, structure, sample.omega
    )
    out = recovery_series(
        recovered,
        structure,
        envelope.cancelled_name,
        point=(
            "bbp_reporting_point"
            if isinstance(bias, CancelledBias)
            else "original_uncorrected"
        ),
    )
    if bias is not None and not isinstance(bias, CancelledBias):
        out.attrs["nuisance_correction"] = "not_supplied"
    return out
