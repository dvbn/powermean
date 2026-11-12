"""Numerical geometry of the collinearity analysis: within rank, spans and relations.

Internal module: the within rank, the original-column span tests and the relations that
:mod:`powermean.data.collinearity` runs, and the records of their evidence.  The checked
coordinates, norms, comparators, Gram products, LAPACK calls and their ledger are
:mod:`powermean.data.collinearity_numeric`, whose names this module still provides.
Nothing here absorbs, decides a sample, fits or forms an indicator matrix.

Within rank (:func:`within_rank`).  ``G = Z'Z`` by range-checked products and
row-order sums; SciPy's pivoted economic QR of ``G`` (LAPACK ``dgeqp3``) proposes a
deficiency where ``|diag R| <= eps k ||G||_2``; the direct rank of ``Z`` at the frozen
``T = eps max(n, k) sigma_max(Z)`` vetoes or confirms it, and one traversal in reverse
formula order omits a column only if the rank without it is the current rank.  Pivots
never choose a column.  Subsets are ranked at the given cutoff, never rescaled.

Span test (:func:`span_test`).  ``B`` holds original columns and the gathered
fixed-effect components that are not exactly zero, ``H = [B, U_j]``, one shared
reversible scaling, ``B`` ranked at ``H``'s cutoff.  A component gathered from a
block's coefficients by its codes lies exactly in that block's indicator span, so an
exact rank equality would prove the relation; the numerical equality is evidence at
the machine-rank resolution, not exact dependence, and a failure says nothing about
the full indicator space.  :func:`relation` proposes coefficients by a full-rank-only
``dgelsd`` solve on ``Z_K`` at ``cond = T / sigma_max(Z_K)``, maps them to the
original coordinates by the exact power-of-two ratios ``e_j / e_K`` and tests the
ORIGINAL column against ``K`` and every block.

Records.  :class:`WithinRank`, :class:`SpanTest` and :class:`Relation` check their
fields at construction, direct or through ``dataclasses.replace``: indices, ranks and
counts are exact integers checked before any conversion; tuples, nested visits and
arrays are owned copies that a caller's later change cannot reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.data.collinearity_numeric import (
    EPSILON,
    Factorization,
    Ledger,
    Rank,
    count_above,
    exact_integer,
    gram,
    increasing,
    integers,
    least_squares,
    norm,
    numerical_rank,
    owned,
    pivoted_diagonal,
    real,
    reversible,
    screened,
)
from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_product,
    checked_quotient,
)
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

# The names this private module provides, with the numerical names of
# collinearity_numeric, whose import routes from here are kept (the patchable
# ``linalg`` boundary is that module's own, not provided here).
__all__ = [
    "EPSILON",
    "Components",
    "Factorization",
    "InconclusiveError",
    "Ledger",
    "Rank",
    "Relation",
    "SpanTest",
    "WithinRank",
    "count_above",
    "gram",
    "least_squares",
    "norm",
    "numerical_rank",
    "pivoted_diagonal",
    "relation",
    "reversible",
    "screened",
    "span_test",
    "within_rank",
]


class InconclusiveError(Exception):
    """A specified check failed or two specified checks disagree: no conclusion."""

    def __init__(self, cause: str, message: str) -> None:
        """Keep the cause, one of the analysis's causes, beside the message."""
        super().__init__(message)
        self.cause = cause


# ------------------------------------------------------------- the within rank


def _visits(
    values: object, active: tuple[int, ...]
) -> tuple[tuple[int, int, int], ...]:
    """Owned visits ``(column, rank with it, rank without it)`` of active columns."""
    if not isinstance(values, tuple | list):
        raise ValueError("visits must be a tuple or list of triples")
    found = [integers(visit, "a visit") for visit in values]
    if any(len(visit) != 3 or visit[0] not in active for visit in found):
        raise ValueError("a visit is an active column and two ranks")
    return tuple((visit[0], visit[1], visit[2]) for visit in found)


@dataclass(frozen=True, eq=False)
class WithinRank:
    """The rank analysis of the active within columns.

    ``active``: original indices in formula order; ``scales``: their power-of-two
    standardization ``e``.  The Gram candidate (``gram_cutoff``, ``gram_rank``,
    ``gram_diagonal``, ``pivots``) and the direct rank (``cutoff`` ``T``,
    ``direct_rank``, ``singular_values``) are ``None`` or empty without active
    columns.  ``visits``: ``(column, rank with it, rank without it)`` as visited;
    ``kept``: the final ``K``; ``kept_values``: ``Z_K``'s singular values.
    """

    active: tuple[int, ...]
    scales: FloatArray
    gram_cutoff: float | None
    gram_rank: int | None
    gram_diagonal: FloatArray
    pivots: IndexArray
    cutoff: float | None
    direct_rank: int | None
    singular_values: FloatArray
    visits: tuple[tuple[int, int, int], ...]
    kept: tuple[int, ...]
    kept_values: FloatArray

    def __post_init__(self) -> None:
        """Check and own every field; ``kept`` is ordered within ``active``."""
        active = increasing(self.active, "active columns")
        kept = increasing(self.kept, "kept columns")
        scales = owned(self.scales)
        nested = set(kept) <= set(active) and scales.size == len(active)
        if not (nested and np.all(scales > 0)):
            raise ValueError("active and kept columns: ordered, nested, one scale each")
        pivots = integers(self.pivots, "pivots")
        if pivots and sorted(pivots) != list(range(len(active))):
            raise ValueError("the pivots must permute the active columns")
        gram_rank, direct = (
            None if rank is None else exact_integer(rank, "a rank")
            for rank in (self.gram_rank, self.direct_rank)
        )
        if gram_rank is not None and gram_rank > len(active):
            raise ValueError("the Gram rank exceeds the number of active columns")
        if direct is not None and direct != len(kept):
            raise ValueError("the kept columns must have the direct rank")
        gram_cutoff, cutoff = (
            None if c is None else real(c, "a cutoff")
            for c in (self.gram_cutoff, self.cutoff)
        )
        stored = (
            active,
            scales,
            gram_cutoff,
            gram_rank,
            owned(self.gram_diagonal),
            frozen_index(np.asarray(pivots, dtype=np.intp)),
            cutoff,
            direct,
            owned(self.singular_values),
            _visits(self.visits, active),
            kept,
            owned(self.kept_values),
        )
        for field, value in zip(fields(self), stored, strict=True):
            object.__setattr__(self, field.name, value)

    @property
    def omitted(self) -> tuple[int, ...]:
        """The active columns not kept, in formula order."""
        return tuple(j for j in self.active if j not in self.kept)


def _traverse(
    z: FloatArray, active: tuple[int, ...], direct: Rank, ledger: Ledger
) -> tuple[list[int], list[tuple[int, int, int]], FloatArray]:
    """One visit per column in reverse formula order at the frozen ``T``.

    Returns the kept positions in ``z``, the visits and ``Z_K``'s singular values.
    """
    current, values = list(range(len(active))), direct.values
    visits: list[tuple[int, int, int]] = []
    for position in reversed(range(len(active))):
        without = [c for c in current if c != position]
        found = Rank(np.zeros(0), direct.cutoff, 0)
        if without:
            purpose = f"without column {active[position]}"
            found = numerical_rank(
                z[:, without],
                ledger,
                kind="subset_svd",
                purpose=purpose,
                cutoff=direct.cutoff,
            )
        if direct.rank - found.rank not in (0, 1):
            raise InconclusiveError(
                "nonnested", f"column {active[position]}: rank jump"
            )
        visits.append((active[position], direct.rank, found.rank))
        if found.rank == direct.rank:
            current, values = without, found.values
    if direct.rank != len(current):
        raise InconclusiveError("final_rank", "the kept columns lack full rank at T")
    return current, visits, values


def within_rank(
    within: FloatArray, active: tuple[int, ...], ledger: Ledger
) -> tuple[WithinRank, FloatArray]:
    """The Gram candidate, the direct rank and, both deficient, the traversal.

    Returns the record and ``Z``.  Direct full rank keeps every column whatever the
    Gram candidate; full Gram rank with deficient direct rank is ``disagreement``.
    Without active columns nothing is factorized.
    """
    e, none = np.zeros(0), np.zeros(0, dtype=np.intp)
    if not active:
        return WithinRank((), e, None, None, e, none, None, None, e, (), (), e), within
    z, scales = reversible(within, "within columns")
    g = gram(z)
    size = numerical_rank(g, ledger, kind="gram_norm", purpose="Gram matrix")
    diagonal, pivots, gram_rank = pivoted_diagonal(g, ledger, cutoff=size.cutoff)
    direct = numerical_rank(z, ledger, kind="direct_svd", purpose="within columns")
    kept, values = list(range(len(active))), direct.values
    visits: list[tuple[int, int, int]] = []
    if direct.rank < len(active):
        if gram_rank == len(active):
            raise InconclusiveError("disagreement", "full Gram, deficient direct rank")
        kept, visits, values = _traverse(z, active, direct, ledger)
    found = (direct.cutoff, direct.rank, direct.values, tuple(visits))
    record = WithinRank(
        active,
        scales,
        size.cutoff,
        gram_rank,
        diagonal,
        pivots,
        *found,
        tuple(active[c] for c in kept),
        values,
    )
    return record, z


# ------------------------------------------------------------------ span tests


@dataclass(frozen=True, eq=False)
class SpanTest:
    """The ranks of ``B`` and ``H = [B, U_j]`` at one cutoff ``T_H``.

    ``basis`` original columns and ``components`` nonzero gathered fixed-effect
    components form ``B``; ``scales`` are the shared scales of ``H``'s columns.
    """

    basis: int
    components: int
    scales: FloatArray
    cutoff: float
    singular_b: FloatArray
    singular_h: FloatArray
    rank_b: int
    rank_h: int

    def __post_init__(self) -> None:
        """Own the arrays; one scale per column of ``H``; ranks count the values."""
        basis = exact_integer(self.basis, "the basis count")
        components = exact_integer(self.components, "the component count")
        scales, cutoff = owned(self.scales), real(self.cutoff, "the cutoff")
        singular_b, singular_h = owned(self.singular_b), owned(self.singular_h)
        if scales.size != basis + components + 1 or np.any(scales <= 0):
            raise ValueError("one scale per column of H")
        ranks = (
            exact_integer(self.rank_b, "rank_b"),
            exact_integer(self.rank_h, "rank_h"),
        )
        if ranks != (count_above(singular_b, cutoff), count_above(singular_h, cutoff)):
            raise ValueError("ranks must count the singular values above the cutoff")
        stored = (basis, components, scales, cutoff, singular_b, singular_h, *ranks)
        for field, value in zip(fields(self), stored, strict=True):
            object.__setattr__(self, field.name, value)

    @property
    def nested(self) -> bool:
        """Whether adding ``U_j`` changed the rank by zero or one."""
        return self.rank_h - self.rank_b in (0, 1)

    @property
    def spanned(self) -> bool:
        """Whether ``U_j`` lies in the numerical span of ``B``: equal ranks."""
        return self.rank_b == self.rank_h


def span_test(
    basis: FloatArray,
    components: Sequence[FloatArray],
    column: FloatArray,
    ledger: Ledger,
    *,
    purpose: str,
) -> SpanTest:
    """``B = [basis, nonzero components]`` against ``H = [B, column]`` at ``T_H``.

    Only exactly zero components are dropped; ``B`` may be rank deficient.
    """
    kept = [v for v in components if bool(np.any(v != 0))]
    scaled, scales = reversible(np.column_stack([basis, *kept, column]), purpose)
    whole = numerical_rank(scaled, ledger, kind="span_svd", purpose=f"{purpose}: H")
    part, width = Rank(np.zeros(0), whole.cutoff, 0), int(scaled.shape[1]) - 1
    if width:
        part = numerical_rank(
            scaled[:, :width],
            ledger,
            kind="span_svd",
            purpose=f"{purpose}: B",
            cutoff=whole.cutoff,
        )
    counts = (int(basis.shape[1]), len(kept))
    return SpanTest(
        *counts, scales, whole.cutoff, part.values, whole.values, part.rank, whole.rank
    )


class Components(NamedTuple):
    """The absorber's block coefficients, their original columns and block codes.

    ``coefficients[b]`` is block ``b``'s ``L_b``-by-``m`` array, whose column ``i``
    belongs to original column ``columns[i]``; ``codes[b]`` are block ``b``'s codes.
    """

    coefficients: tuple[FloatArray, ...]
    columns: tuple[int, ...]
    codes: tuple[IndexArray, ...]

    def gathered(self, column: int) -> list[FloatArray]:
        """``a_b[:, j][codes_b]`` for every block: exact gathers, never summed."""
        i = self.columns.index(column)
        return [a[:, i][c] for a, c in zip(self.coefficients, self.codes, strict=True)]


@dataclass(frozen=True, eq=False)
class Relation:
    """The fresh relation of one omitted column against the final ``K``.

    ``solved``: the ``dgelsd`` proposal in ``Z`` coordinates; ``mapped``: the same in
    ``U`` coordinates, ``c e_j / e_K``; ``components``: the owned ``eta_b = a_b[:, j]
    - A_b[:, K] c``, one per block; ``span``: the original column against
    ``[U_K, gathered eta]``.
    """

    basis: tuple[int, ...]
    solved: FloatArray
    mapped: FloatArray
    components: tuple[FloatArray, ...]
    span: SpanTest

    def __post_init__(self) -> None:
        """Own the basis and arrays; one coefficient per column of ``K``."""
        basis = increasing(self.basis, "the basis")
        solved, mapped = owned(self.solved), owned(self.mapped)
        if not len(basis) == solved.size == mapped.size:
            raise ValueError("one coefficient per kept column")
        if not isinstance(self.components, tuple | list):
            raise ValueError("the components are a tuple of vectors")
        if not (isinstance(self.span, SpanTest) and self.span.basis == len(basis)):
            raise ValueError("the span test must hold the kept columns")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "solved", solved)
        object.__setattr__(self, "mapped", mapped)
        object.__setattr__(self, "components", tuple(map(owned, self.components)))


def _eta(
    blocks: Components, block: int, j: int, kept: tuple[int, ...], mapped: FloatArray
) -> FloatArray:
    """``a_b[:, j] - sum_k a_b[:, k] c_k``, every operation range-checked, in order."""
    a, where = blocks.coefficients[block], blocks.columns.index
    eta: FloatArray = a[:, where(j)]
    for k, c in zip(kept, mapped, strict=True):
        term = checked_product(a[:, where(k)], np.array([c]), "eta term")
        eta = checked_difference(eta, term, "eta component")
    return immutable_array(eta)


def relation(
    j: int,
    within: WithinRank,
    z: FloatArray,
    *,
    original: FloatArray,
    blocks: Components | None,
    ledger: Ledger,
) -> Relation:
    """The original-column relation of the omitted column ``j`` against FINAL ``K``.

    Empty ``K``: empty coefficients, no solve.  A deficient solve is ``solve``, a
    failed test ``relation``, a nonnested one ``nonnested``.
    """
    kept, where = within.kept, {c: i for i, c in enumerate(within.active)}
    purpose, solved, mapped = f"relation of column {j}", np.zeros(0), np.zeros(0)
    if kept:
        cond = float(checked_quotient(np.asarray(within.cutoff), within.kept_values[0]))
        columns = [where[k] for k in kept]
        solved, rank = least_squares(
            z[:, columns], z[:, where[j]], ledger, cond=cond, purpose=purpose
        )
        if rank != len(kept):
            raise InconclusiveError("solve", f"{purpose}: deficient solve on K")
        ratio = checked_quotient(within.scales[where[j]], within.scales[columns])
        mapped = checked_product(solved, ratio, "mapped coefficients")
    etas: tuple[FloatArray, ...] = ()
    gathers: list[FloatArray] = []
    if blocks is not None:
        etas = tuple(_eta(blocks, b, j, kept, mapped) for b in range(len(blocks.codes)))
        gathers = [eta[c] for eta, c in zip(etas, blocks.codes, strict=True)]
    test = span_test(
        original[:, list(kept)], gathers, original[:, j], ledger, purpose=purpose
    )
    if not test.nested:
        raise InconclusiveError("nonnested", f"{purpose}: the rank jumps")
    if not test.spanned:
        raise InconclusiveError("relation", f"{purpose}: outside the original span")
    return Relation(kept, solved, mapped, etas, test)
