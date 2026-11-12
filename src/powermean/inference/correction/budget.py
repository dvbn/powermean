"""Explicit incremental array-byte and array-visit bounds.

These bounds include final immutable copies, LU/solve arrays, full-coordinate
vectors, selected outputs and fresh accepted geometry checks. They exclude the
already owned design and geometry, caller inputs, Python metadata, user callback
internals and vendor-private work. They are not OS memory or wall-time limits.
There is no FE-count cap and no newly chosen numerical/budget default.
"""

from dataclasses import dataclass

from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_gram import geometry_budget
from powermean.fe.retained import CenteredDesign


@dataclass(frozen=True, slots=True)
class BiasBounds:
    """Caller-specified positive incremental bytes and array visits."""

    max_bytes: int
    max_work_units: int

    def __post_init__(self) -> None:
        """Refuse ambiguous/nonpositive bounds."""
        checked_count(self.max_bytes, "max_bytes", 1)
        checked_count(self.max_work_units, "max_work_units", 1)


def checked_count(value: int, name: str, minimum: int = 0) -> int:
    """Validate a size without fixed-width integer arithmetic."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def enforce(total_bytes: int, work_units: int, bounds: BiasBounds) -> None:
    """Reserve before allocations; work units count managed scalar array visits."""
    if type(bounds) is not BiasBounds:
        raise TypeError("bounds must be a BiasBounds")
    if total_bytes > bounds.max_bytes:
        raise MemoryError(
            f"requires {total_bytes} array bytes > max_bytes={bounds.max_bytes}"
        )
    if work_units > bounds.max_work_units:
        raise MemoryError(
            f"requires {work_units} array visits > "
            f"max_work_units={bounds.max_work_units}"
        )


@dataclass(frozen=True, slots=True)
class OperatorWorkspacePlan:
    """Declared new workspace, including upstream fresh binding work."""

    n_obs: int
    n_groups: int
    n_identified: int
    n_full: int
    total_bytes: int
    work_units: int


@dataclass(frozen=True, slots=True)
class ShortGroupWorkspacePlan:
    """Operator plus LU, contraction, selector, and result storage allowance."""

    operator: OperatorWorkspacePlan
    n_selected: int
    total_bytes: int
    work_units: int


@dataclass(frozen=True, slots=True)
class CellWorkspacePlan:
    """Operator plus ordered-pair validation and cell contraction allowance."""

    operator: OperatorWorkspacePlan
    n_selected: int
    n_pairs: int
    total_bytes: int
    work_units: int


def reserve_bias_operator(
    design: CenteredDesign | NumericalGeometry, *, bounds: BiasBounds
) -> OperatorWorkspacePlan:
    """Reserve even the cell-capable path before copying inputs or callbacks.

    128 row, 32 group, 64 full-coordinate vector slots cover snapshots, core
    rows, log primitives, products and immutable copies. Accepted geometry's
    complete prospective reservation covers its n*K and p^2 fresh checks. No
    n*d or n*levels buffer is reserved or constructed by this component.
    """
    if type(design) is NumericalGeometry:
        base, d = design.design, design.n_identified
        upstream = geometry_budget(base).prospective_bytes
    elif type(design) is CenteredDesign:
        base, d, upstream = design, design.n_columns, 0
    else:
        raise TypeError("design must be CenteredDesign or NumericalGeometry")
    n, g, p, k = (
        base.structure.n_rows,
        base.structure.n_groups,
        base.n_columns,
        base.n_slopes,
    )
    blocks = len(base.structure.block_sizes)
    byte_count = upstream + 8 * (128 * n + 32 * g + 64 * p + 16 * d * d)
    visits = 128 * (p**3 + n * (p + 1) * (k + blocks + 1) + p * p * (d + 1))
    enforce(byte_count, visits, bounds)
    return OperatorWorkspacePlan(n, g, d, p, byte_count, visits)


def _correction_space(op: OperatorWorkspacePlan, selected: int) -> tuple[int, int]:
    """Conservative simultaneous slots; matrix multiplication adds no third tensor.

    64 d^2 slots include H, inverse, equilibrated LU, meat, C, solve products,
    symmetrization, immutable result copies and declared factorization copies.
    Selector/output storage is O(q*d+q^2), one group column O(G). No G*d score
    matrix is stored. Work includes q adjoints and streamed d-column products.
    """
    q = checked_count(selected, "n_selected", 1)
    n, g, d, p = op.n_obs, op.n_groups, op.n_identified, op.n_full
    byte_count = op.total_bytes + 8 * (
        64 * d * d + 32 * q * d + 32 * q * q + 16 * g + 128 * n + 64 * p + 64 * q
    )
    visits = op.work_units + 128 * (
        d**3 + (d + q + 1) * (op.work_units // 128 + g * q + q * d)
    )
    return byte_count, visits


def reserve_short_group_bias(
    geometry: NumericalGeometry, n_selected: int, *, bounds: BiasBounds
) -> ShortGroupWorkspacePlan:
    """Reserve the full short-group call before selector allocation."""
    op = reserve_bias_operator(geometry, bounds=bounds)
    byte_count, visits = _correction_space(op, n_selected)
    enforce(byte_count, visits, bounds)
    return ShortGroupWorkspacePlan(op, n_selected, byte_count, visits)


def reserve_pairs(n_obs: int, n_pairs: int, *, bounds: BiasBounds) -> None:
    """Reserve sorting/copying pair arrays, with no unbounded Python pair sets."""
    n = checked_count(n_obs, "n_obs", 1)
    count = checked_count(n_pairs, "n_pairs", n)
    # Integer pair copies, lexicographic permutations, reverse/sorted arrays and
    # immutable finalization; 32 slots/pair also cover construction temporaries.
    enforce(
        8 * (32 * count + 32 * n), 128 * (count * (count.bit_length() + 1) + n), bounds
    )


def reserve_cell_bias(
    geometry: NumericalGeometry, n_selected: int, n_pairs: int, *, bounds: BiasBounds
) -> CellWorkspacePlan:
    """Reserve the full cell call, including pair validation and its transients."""
    op = reserve_bias_operator(geometry, bounds=bounds)
    reserve_pairs(op.n_obs, n_pairs, bounds=bounds)
    byte_count, visits = _correction_space(op, n_selected)
    byte_count += 8 * (32 * n_pairs + 32 * op.n_obs)
    visits += 128 * (
        n_pairs * (n_pairs.bit_length() + 1) + n_pairs * (op.n_identified + 1)
    )
    enforce(byte_count, visits, bounds)
    return CellWorkspacePlan(op, n_selected, n_pairs, byte_count, visits)
