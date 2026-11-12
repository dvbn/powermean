"""WZ scale at an admitted joint PPML candidate, before certification.

Normal composition supplies the existing AttemptBounds from internal resource
admission. No public manual counts, final certificate or root-solving call is
used. Benchmark-derived production limits are a separate admission gate.
"""

from __future__ import annotations

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.inference.absorbed_preflight import sampling_units
from powermean.inference.absorbed_rank import absorbed_count
from powermean.inference.absorbed_rows import influence_rows
from powermean.inference.convention import ClusterCodes
from powermean.inference.ssc import convention_counts, resolve_convention
from powermean.inference.wz_binding import capture_rows, layout, unavailable
from powermean.inference.wz_covariance import (
    WZControls,
    bind_request,
    convention,
    covariance_record,
    ppml,
    reported,
)
from powermean.inference.wz_operator import evaluate
from powermean.solve.absorbed_certificate_scale import (
    FEScaleDetails,
    FEScaleEvidence,
    FEScaleInput,
    FEScaleResult,
)
from powermean.solve.reported import ReportedScale


def joint_scale(scale_input: FEScaleInput, controls: WZControls) -> FEScaleResult:
    """Evaluate the requested WZ matrix on this input's fresh geometry and point."""
    request, geometry = scale_input.request, scale_input.geometry
    point, sample = geometry.point, geometry.point.sample
    if request.adjustment != "wz" or len(request.clusters) != 1:
        raise unavailable("one pair cluster with adjustment='wz' required")
    ppml(sample.omega, point.variance, weights=sample.data.weights is not None)
    design = sample.design
    if design is None:
        raise unavailable("all original slopes must remain identified")
    name, codes = request.clusters[0]
    panel = layout(
        controls.source, name, design.matrix.shape, controls.absorbed, controls.bounds
    )
    # The admitted point already supplies the owned current sample. A closed
    # assessment replica intentionally omits preparatory base/source fields.
    # Reconstructing FESample would read those fields, not fresh WZ inputs.
    source = controls.source
    group = sample.grouping
    if group is None or any(v is not None for v in sample.reasons):
        raise unavailable("original three-way design without omitted slopes required")
    if sample.positive_only or sample.conditioned or request.bread != "observed":
        raise unavailable("unconditioned observed PPML required")
    if sample.data.row_labels != source.row_labels or not same_bits(
        sample.positions, source.grouping.positions
    ):
        raise unavailable("current original row ownership changed")
    for actual, original in zip(group.blocks, source.grouping.blocks, strict=True):
        if (actual.name, actual.levels) != (
            original.name,
            original.levels,
        ) or not same_bits(actual.codes, original.codes):
            raise unavailable("current declared FE source labels or codes changed")
    bind_request(source, ClusterCodes(name, codes))
    with np.errstate(over="raise", invalid="raise", under="raise"):
        mu = np.exp(point.q)
    rows = capture_rows(
        panel,
        design.matrix,
        sample.data.y,
        mu,
        sample.active_names,
        route="joint_current",
    )
    applied_convention = resolve_convention(request.ssc)
    result = evaluate(
        rows,
        controls.options,
        controls.bounds,
        ssc=applied_convention.preset or "custom",
        plan=panel.plan,
    )
    blocks = tuple((b.name, b.codes) for b in group.blocks)
    absorbed = absorbed_count(blocks, [codes])
    counts = convention_counts(
        sample.count, geometry.n_slopes, geometry.dimension, absorbed, [rows.pairs]
    )
    applied = convention(request.ssc, counts)
    covariance = covariance_record(result, applied)
    errors, matrix, failure = reported(result, scale_input.order)
    details = FEScaleDetails(
        "observed",
        request.ssc,
        applied.convention.preset or "custom",
        applied.convention.source,
        "method",
        sampling_units(sample),
        sample.count,
        geometry.dimension,
        applied,
        absorbed,
        reported_covariance=matrix,
        covariance_error=failure,
        wz=result,
    )
    evidence = FEScaleEvidence(
        scale_input.binding,
        request,
        influence_rows(scale_input),
        covariance,
        ReportedScale(
            gamma=point.theta, order=scale_input.order, standard_errors=errors
        ),
        details=details,
    )
    return FEScaleResult(scale_input, evidence)
