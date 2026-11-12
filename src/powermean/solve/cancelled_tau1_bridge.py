"""Exact tau=1 joint-absorbed inner layout and cancelled-coordinate mapping.

Positive-total cancelled groups form the inner computational problem;
all original outer rows remain in fresh cancelled certification and accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from powermean.data.collinearity_records import ABSORBED
from powermean.data.raw import Column, RawInputs
from powermean.data.retained import _regression_data, select_order
from powermean.data.sample import build_sample
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_state import (
    FESample,
    final_sample,
    ledger_of,
    replicas_of,
    restrict,
)
from powermean.fe.admission_records import WorkLog
from powermean.fe.geometry_design import NumericalGeometry
from powermean.model.order import IndexArray, transform_problem
from powermean.model.variance import (
    FloatArray,
    NB2Variance,
    PowerVariance,
)
from powermean.solve.absorbed_direction import IndexPoint, recover
from powermean.solve.cancelled_certificate import (
    CancelledScaleProvider,
)
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.cancelled_records import (
    CancelledCertificate,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_absorbed import (
    _problem,
)
from powermean.solve.options import SolverOptions
from powermean.solve.profiled_trigger import Admission, Trigger
from powermean.systems.admitted import CapturedFit

if TYPE_CHECKING:
    from powermean.solve.newton_absorbed import IndexCandidate
    from powermean.solve.newton_profiled import ProfiledCandidate


@dataclass(frozen=True)
class _BridgeLayout:
    """Mapping between the inner joint ``FESample`` and ``NumericalGeometry``."""

    sample: FESample
    active_rows: IndexArray
    retained_block_indices: tuple[int, ...]


@dataclass
class _Tau1Context:
    """Internal execution state for the tau=1 joint-absorbed solve."""

    capture: CapturedFit
    layout: _BridgeLayout
    family: PowerVariance | NB2Variance
    options: SolverOptions
    absorbed: AbsorbedOptions
    bounds: AttemptBounds
    reported: CancelledScaleProvider
    work: WorkLog
    fallback: FloatArray
    trigger: Trigger
    admit: Admission
    start: FloatArray | None = None
    history: list[CancelledCertificate] = field(default_factory=list)
    candidate: IndexCandidate | ProfiledCandidate | None = None


def _build_effect_columns(
    geometry: NumericalGeometry, active_rows: IndexArray
) -> tuple[EffectColumns, tuple[int, ...]]:
    """Construct ``EffectColumns`` for the cancelled and non-annihilated blocks."""
    structure = geometry.structure
    cancel_codes = structure.cancel_codes
    terms: list[tuple[str, ...]] = [("fe_cancel",)]
    columns: dict[str, Column] = {
        "fe_cancel": Column(cancel_codes[active_rows], name="fe_cancel")
    }
    retained_indices: list[int] = []
    ann_blocks = geometry.annihilated_blocks
    for b_idx, sorted_codes in enumerate(structure.retained_sorted_codes):
        if bool(ann_blocks[b_idx]):
            continue
        codes_b = sorted_codes[structure.inverse][active_rows]
        if np.unique(codes_b).size <= 1:
            continue
        col_name = f"fe_ret_{b_idx}"
        terms.append((col_name,))
        columns[col_name] = Column(codes_b, name=col_name)
        retained_indices.append(b_idx)
    return EffectColumns(terms, columns, int(active_rows.size)), tuple(retained_indices)


def _build_bridge_layout(capture: CapturedFit) -> _BridgeLayout:
    """Construct the inner ``FESample`` on positive-outcome cancelled groups."""
    geometry = capture.admission.geometry
    structure, z, offset = geometry.structure, capture.z, capture.centered_offset
    if capture.obs_weights is not None and np.any(capture.obs_weights != 1.0):
        raise ValueError("cancelled tau=1 bridge supports only unit multipliers")
    cancel_codes = structure.cancel_codes
    group_totals = np.bincount(cancel_codes, weights=z, minlength=structure.n_groups)
    active_rows = np.flatnonzero((group_totals > 0.0)[cancel_codes]).astype(np.intp)
    n_active = int(active_rows.size)
    if n_active == 0:
        raise ValueError("no positive-outcome cancelled group remains for tau=1")

    effects, retained_indices = _build_effect_columns(geometry, active_rows)
    d_x = geometry.n_slopes_identified
    if d_x > 0:
        raw_slopes = capture.admission.raw_slopes
        design_matrix = raw_slopes[active_rows][:, geometry.kept_slope_indices]
        col_names = tuple(f"x{j}" for j in range(d_x))
        reasons: tuple[str | None, ...] = (None,) * d_x
    else:
        design_matrix = np.ones((n_active, 1), dtype=np.float64)
        col_names, reasons = ("_absorbed_dummy",), (ABSORBED,)

    y_act, off_act = z[active_rows], offset[active_rows]
    raw_inputs = RawInputs(
        y=y_act, design=design_matrix, column_names=col_names, offset=off_act
    )
    base = build_sample(raw_inputs)
    order_rows = select_order(base, 1.0, positive_only=False)
    reg_initial = _regression_data(base.inputs, order_rows.positions)
    replicas = replicas_of(reg_initial, order_rows)
    ledger = ledger_of(order_rows, base.missing, replicas)
    source = effects.encode(order_rows.positions)
    grouping = restrict(source, ledger.positions, ledger.nobs)
    data = _regression_data(base.inputs, ledger.positions)
    problem = transform_problem(data.y, 1.0, offset=data.offset, positive_only=False)
    sample = final_sample(
        order_rows,
        ledger,
        base=base,
        data=data,
        problem=problem,
        source=source,
        grouping=grouping,
        reasons=reasons,
    )
    return _BridgeLayout(sample, active_rows, retained_indices)


def _effects_to_identified(
    geometry: NumericalGeometry,
    layout: _BridgeLayout,
    scaled_theta: FloatArray,
    block_effects: tuple[FloatArray, ...],
) -> FloatArray:
    """Map scaled inner slopes and block effects to identified ``theta``."""
    v_full = np.zeros(geometry.n_columns_full, dtype=np.float64)
    sample = layout.sample
    if geometry.n_slopes_identified > 0 and sample.design is not None:
        v_full[geometry.kept_slope_indices] = scaled_theta / sample.design.scale
    grouping = sample.grouping
    assert grouping is not None
    starts, sizes = geometry.design.block_starts, geometry.structure.block_sizes
    ann_levels = geometry.annihilated_levels
    for inner_pos, b_idx in enumerate(layout.retained_block_indices, start=1):
        levels = np.asarray(grouping.blocks[inner_pos].levels, dtype=np.intp)
        start, size = starts[b_idx], sizes[b_idx]
        full_alpha = np.zeros(size, dtype=np.float64)
        full_alpha[levels] = block_effects[inner_pos]
        full_alpha[ann_levels[b_idx]] = 0.0
        v_full[start : start + size] = full_alpha
    return geometry.reduce(v_full)


def _index_point_to_identified(ctx: _Tau1Context, point: IndexPoint) -> FloatArray:
    """Transform an ``IndexPoint`` to identified cancelled coordinates."""
    ab, rec = ctx.absorbed, point.recovered
    if rec is None:
        prob = _problem(ctx.layout.sample, ctx.family)
        rec = recover(
            prob,
            point,
            tolerance=ab.absorb_tol_final,
            max_sweeps=ab.absorb_max_sweeps,
            acceleration=ab.absorb_acceleration,
        )
    norm = rec.normalization
    effects = norm.effects if norm is not None else rec.effects
    if effects is None:
        raise FloatingPointError("failed to recover retained fixed effects")
    return _effects_to_identified(
        ctx.capture.admission.geometry, ctx.layout, point.theta, effects
    )


def _warm_effects_from_start(
    geometry: NumericalGeometry, layout: _BridgeLayout, theta_start: FloatArray
) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    """Map an identified start ``theta_start`` to inner scaled slopes and FE start."""
    sample = layout.sample
    grouping = sample.grouping
    assert grouping is not None
    v_full = geometry.expand(theta_start)
    d_x = geometry.n_slopes_identified
    if d_x > 0 and sample.design is not None:
        scaled_slopes = theta_start[:d_x] * sample.design.scale
        x_part = sample.design.matrix @ scaled_slopes
    else:
        scaled_slopes = np.zeros(0, dtype=np.float64)
        x_part = np.zeros(sample.count, dtype=np.float64)

    starts, sizes = geometry.design.block_starts, geometry.structure.block_sizes
    retained_effects: list[FloatArray] = []
    ret_linear = np.zeros(sample.count, dtype=np.float64)
    for inner_pos, b_idx in enumerate(layout.retained_block_indices, start=1):
        block = grouping.blocks[inner_pos]
        levels = np.asarray(block.levels, dtype=np.intp)
        start, size = starts[b_idx], sizes[b_idx]
        alpha_m = np.asarray(v_full[start : start + size][levels], dtype=np.float64)
        retained_effects.append(alpha_m)
        ret_linear += alpha_m[block.codes]

    c_block = grouping.blocks[0]
    resp = sample.problem.response
    z_grp = np.bincount(c_block.codes, weights=resp, minlength=c_block.n_levels)
    eta = sample.problem.offset + x_part + ret_linear
    shift = float(np.max(eta))
    w_exp = np.exp(eta - shift)
    denom = np.bincount(c_block.codes, weights=w_exp, minlength=c_block.n_levels)
    alpha_c = np.log(z_grp) - (shift + np.log(denom))
    return scaled_slopes, (alpha_c, *retained_effects)
