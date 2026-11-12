"""Current-candidate WZ inference composition, separate from coefficient bias.

The published left inverse and P/(P-1) own the reported
covariance. SSC retains its reference distribution and request provenance only.
No certificate, optimizer, symmetric CR2 transform or PSD repair is invoked here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.data.inference_conventions import ConventionValues
from powermean.data.request_values import Omitted
from powermean.fe.admission import AdmissionInputs, admit_geometry
from powermean.fe.admission_budget import admission_budget
from powermean.fe.geometry_checks import GeometryInputs
from powermean.inference.convention import ClusterCodes, Covariance
from powermean.inference.ssc import (
    AppliedConvention,
    ConventionCounts,
    ConventionCovariance,
    ResolvedConvention,
    ScalarFactor,
    direct_errors,
    reference_of,
    resolve_convention,
)
from powermean.inference.vcov import _result
from powermean.inference.wz_binding import WZSource, capture_rows, layout, unavailable
from powermean.inference.wz_operator import PairCovariance, evaluate
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.options import SolverOptions
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import IdentifiedState


def convention(
    requested: str | ConventionValues | Omitted,
    counts: ConventionCounts,
    *,
    inherited: ResolvedConvention | None = None,
) -> AppliedConvention:
    """Own P/(P-1) once, retaining the existing convention's reference rules."""
    resolved = resolve_convention(requested, inherited)
    if resolved.origin != "preset" or len(counts.clusters) != 1:
        raise unavailable("named SSC presets and one pair cluster are supported")
    pairs = counts.clusters[0]
    factor = ScalarFactor(pairs, pairs - 1, "WZ P/(P-1)", "method")
    return AppliedConvention(
        resolved,
        counts,
        "cluster",
        None,
        factor,
        (factor,),
        "min",
        reference_of(resolved.values, counts, clustered=True),
        "none",
    )


def bind_request(source: WZSource, cluster: ClusterCodes) -> None:
    """Require the named source's exact codes, not another equal partition."""
    blocks = [b for b in source.grouping.blocks if b.name == cluster.name]
    if len(blocks) != 1 or not np.array_equal(blocks[0].codes, cluster.codes):
        raise unavailable(
            "cluster must retain the exact declared pair source and codes"
        )


@dataclass(frozen=True)
class WZControls:
    """Existing explicit fit controls; no new bounds, tolerance or optimizer policy."""

    source: WZSource
    absorbed: AbsorbedOptions
    options: SolverOptions
    bounds: AttemptBounds

    def cancelled(
        self,
        state: IdentifiedState,
        capture: CapturedFit,
        cluster: ClusterCodes,
        *,
        ssc: str,
    ) -> PairCovariance:
        """Recover the joint fitted mean by the exact PPML pair-scale algebra.

        At tau=1, t=y and mu_it=pi_it sum_t y_it. These are the fresh current
        cancelled rows, evaluated by the existing capture under unit multipliers.
        No companion fit, effect recovery, or final accepted flag is needed.
        """
        geometry = capture.admission.geometry
        n, k = geometry.structure.n_rows, geometry.n_slopes_full
        p, d = geometry.n_columns_full, geometry.n_identified
        extra = (
            admission_budget(geometry.design).prospective_bytes,
            4096
            * (
                n * (p + 1) * (d + 1)
                + (p + 1) ** 3
                + (capture.admission.controls.max_sweeps + 1) * n * (k + 4)
            ),
        )
        panel = layout(
            self.source, cluster.name, (n, k), self.absorbed, self.bounds, extra=extra
        )
        bind_request(self.source, cluster)
        if state.design is not geometry or geometry.n_slopes_identified != k:
            raise unavailable(
                "all original slopes must remain identified in this current state"
            )
        if capture.obs_weights is not None and not np.all(capture.obs_weights == 1):
            raise unavailable(
                "only unit equation multipliers and no statistical weights"
            )
        pair = next(b for b in self.source.grouping.blocks if b.name == cluster.name)
        structure = geometry.structure
        if (
            not np.array_equal(pair.codes, structure.cancel_codes)
            or pair.levels != structure.cancel_levels
        ):
            raise unavailable(
                "current cancellation must own the exact pair source labels"
            )
        _cancelled_sources(self.source, capture, panel.plan.bytes)
        rows = state.rows()
        y, x = capture.z, capture.admission.raw_slopes
        with np.errstate(over="raise", invalid="raise", under="raise"):
            mu = rows.pi * state.full_state.group_t[geometry.structure.cancel_codes]
        current = capture_rows(
            panel,
            x,
            y,
            mu,
            capture.admission.check.source_columns,
            route="pair_cancelled_current",
        )
        return evaluate(current, self.options, self.bounds, ssc=ssc, plan=panel.plan)


def _cancelled_sources(source: WZSource, capture: CapturedFit, max_bytes: int) -> None:
    """Fresh original-label, row, rank and raw-slope admission under reservation."""
    geometry, structure = (
        capture.admission.geometry,
        capture.admission.geometry.structure,
    )
    if not np.array_equal(source.grouping.positions, structure.positions):
        raise unavailable("original cancellation row positions changed")
    blocks = source.grouping.blocks
    original = []
    for block in blocks:
        levels = block.levels
        original.append(tuple(levels[int(code)] for code in block.codes))
    cancelled = tuple(structure.cancel_levels[int(c)] for c in structure.cancel_codes)
    retained = tuple(
        tuple(levels[int(c)] for c in codes[structure.inverse])
        for levels, codes in zip(
            structure.retained_levels, structure.retained_sorted_codes, strict=True
        )
    )
    if (
        cancelled not in original
        or tuple(v for v in original if v != cancelled) != retained
    ):
        raise unavailable(
            "current cancellation no longer owns the original declared labels"
        )
    labels = tuple(_objects(v) for v in original)
    index = original.index(cancelled)
    inputs = AdmissionInputs(
        GeometryInputs(
            capture.admission.raw_slopes,
            structure.positions,
            capture.admission.check.source_columns,
        ),
        labels[index],
        tuple(v for i, v in enumerate(labels) if i != index),
    )
    admit_geometry(
        geometry,
        inputs,
        controls=capture.admission.controls,
        mode=capture.admission.mode,
        max_bytes=max_bytes,
    )


def _objects(
    labels: tuple[object, ...],
) -> np.ndarray[tuple[int], np.dtype[np.object_]]:
    values = np.empty(len(labels), dtype=object)
    for index, label in enumerate(labels):
        values[index] = label
    return values


def covariance_record(result: PairCovariance, applied: AppliedConvention) -> Covariance:
    """Keep the published signed matrix and the existing diagnostic convention."""
    estimate = _result(
        result.unscaled_adjusted,
        method="WZ",
        nobs=result.provenance.observations,
        df_resid=None,
        correction="WZ P/(P-1)",
        factor=1.0,
    )
    assert applied.factor is not None
    return Covariance(
        result.adjusted,
        "cluster",
        applied.factor.value,
        applied.counts.count,
        result.provenance.pairs,
        estimate,
        applied.reference.kind,
    )


def retained_covariance_record(
    result: PairCovariance, applied: AppliedConvention
) -> ConventionCovariance:
    """Retain the actual convention around the already-scaled WZ matrix.

    The lower joint-scale record and its signed, unscaled estimate are preserved.
    This boundary adds reference/count/factor provenance, not another factor.
    """
    covariance = covariance_record(result, applied)
    return ConventionCovariance(
        covariance.covariance, covariance.estimate, applied, "cluster"
    )


def reported(
    result: PairCovariance,
    order: OrderScale,
) -> tuple[FloatArray, FloatArray | None, str | None]:
    """Apply the actual report map once; keep the negative/zero/range semantics."""
    errors = direct_errors(result.adjusted, order)
    try:
        matrix = order.covariance(result.adjusted)
    except FloatingPointError as exc:
        return errors, None, str(exc)
    return errors, matrix, None


def ppml(omega: float, variance: object, *, weights: bool = False) -> None:
    """Check the exact built-in PPML and no-statistical-weight domain."""
    if (
        omega != 1
        or type(variance) is not PowerVariance
        or variance.tau != 1
        or weights
    ):
        raise unavailable(
            "builtin PowerVariance(1), omega=1 and no statistical weights required"
        )
