"""Warning ownership of raw versus adjusted absorbed multiway estimates."""

from __future__ import annotations

import warnings

from powermean.data.inference_requests import Multiway
from powermean.inference.multiway import multiway_estimate
from powermean.inference.psd import PsdAdjustment
from powermean.inference.vcov import CovarianceEstimate
from powermean.inference.vcov import CovarianceWarning as PortWarning
from powermean.model.variance import FloatArray
from powermean.solve.absorbed_certificate_scale import IntegerArray


def _raw_multiway(
    rows: FloatArray, codes: tuple[tuple[str, IntegerArray], ...], form: Multiway
) -> tuple[CovarianceEstimate, list[warnings.WarningMessage]]:
    """Form the raw multiway estimate and capture ``PortWarning`` diagnostics."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PortWarning)
        estimate = multiway_estimate(rows, codes, form=form)
    held: list[warnings.WarningMessage] = []
    for item in caught:
        if isinstance(item.message, PortWarning):
            held.append(item)
        else:
            warnings.warn(item.message, stacklevel=3)
    return (estimate, held)


def _reissue_port_warnings(
    held: list[warnings.WarningMessage], adjustment: PsdAdjustment | None
) -> None:
    """Reissue raw port warnings only when no eigenvalue adjustment is active."""
    if adjustment is not None and adjustment.active:
        return
    for item in held:
        warnings.warn(item.message, stacklevel=3)
