"""Prospective managed workspace for one structural capture and its actions."""

from __future__ import annotations

from dataclasses import dataclass

from powermean.fe.geometry_errors import GeometryBudgetError
from powermean.fe.geometry_gram import _checked_max_bytes, geometry_budget
from powermean.fe.retained import CenteredDesign


@dataclass(frozen=True)
class AdmissionBudget:
    """Conservative simultaneous managed numeric storage, never process memory.

    The accepted geometry/design is shared and reported separately. Added slots
    cover raw capture/check/collinearity input, scaled slopes, absorber recurrence,
    coefficients, per-block gathered ORIGINAL witnesses, fresh means, analysis
    records and immutable evidence finalization. Sequential span/QR/SVD/least
    squares allow 32 rectangular witness buffers plus 32 square buffers. The
    existing geometry reservation is included in full for its candidate veto.
    In-flight allowance covers 128 row, 32 group and 32 p buffers plus 16 p-square
    maps/actions, index/mask buffers and capture copies. Every item is an 8-byte
    slot, conservatively also for bools. No n-by-FE or n-by-identified design.

    Python containers, exact integers/Fractions and labels (including custom
    deepcopy behavior), headers, allocator and vendor-internal work are excluded.
    Exact sparse elimination can fill in up to L-square object entries, where L
    is retained levels, NEVER (G_C+L)-square. These exclusions are real costs;
    max_bytes is not a hard process cap. Absorber sweep budgets are unchanged.
    """

    prospective_bytes: int
    shared_design_numeric_bytes: int
    witness_width: int


def admission_budget(design: CenteredDesign) -> AdmissionBudget:
    """Derive a shape-only bound before snapshots, labels, rank or point work."""
    base = geometry_budget(design)
    n, k, p = base.n_rows, base.n_slopes, base.n_columns_full
    g, levels = base.n_groups, base.n_retained_levels
    blocks = len(design.structure.block_sizes) + 1
    width = k + blocks + 1
    slots = (
        64 * n * k
        + 32 * (g + levels) * (k + 1)
        + 32 * n * width
        + 32 * width * width
        + 16 * p * p
        + 128 * n
        + 32 * g
        + 32 * p
        + 16 * n * blocks
    )
    return AdmissionBudget(
        base.prospective_bytes + 8 * slots,
        8 * n * k + 8 * ((5 + blocks) * n + 2 * g + 1),
        width,
    )


def reserve_admission(design: CenteredDesign, max_bytes: int) -> AdmissionBudget:
    """Refuse the complete added reservation before managed allocation/work."""
    limit = _checked_max_bytes(max_bytes)
    plan = admission_budget(design)
    if plan.prospective_bytes > limit:
        raise GeometryBudgetError(
            plan.prospective_bytes,
            limit,
            n_rows=design.structure.n_rows,
            n_slopes=design.n_slopes,
            n_retained_levels=sum(design.structure.block_sizes),
        )
    return plan
