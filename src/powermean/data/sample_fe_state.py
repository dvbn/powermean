"""The state of the ordered sample: the positional row ledger and the completed sample.

Internal module: no pandas, Formulaic or ``solve``.  :mod:`powermean.data.sample_fe`
(blocks and steps 5 and 6) still provides these names.

Rows.  :class:`RowLedger` keeps one code per input row (:data:`FE_REASONS`: the four
reasons of steps 1 to 4, unchanged, then ``singleton``, ``separated_by_fixed_effect``
and ``separated``; zero while current), the block of the first cause of a step 5 or 6
exclusion, the current positions and their inverse, step 2's diagnostics and, with
frequency weights, the exact count of each row of steps 3 and 4.  Physical counts and
exact replica totals are distinct.  A row leaves once: :meth:`RowLedger.exclude`
refuses a row that is not current, so nothing overwrites a first cause.

Ownership.  Records own read-only arrays, also when constructed or replaced directly;
every index is an exact integer checked before it is narrowed or used
(:func:`powermean.data.retained._exact_index`).  A ledger's current rows are those of
code zero, increasing, with their exact inverse; a block is given exactly for steps 5
and 6; replica counts are zero exactly on the rows of steps 1 to 4, else exact counts of
at most ``2**53``, also in total.  Step 2's account holds the seven groups of
:data:`~powermean.data.accounting.MISSING_GROUPS`, exact Boolean masks of every input
row that may overlap, their union the rows coded missing; the ledger keeps its own
read-only copy and is compared with a context group by group.

The sample (step 9).  :class:`FESample` is bound to the prepared base of every input
row: its account of step 2 is the base's, its numeric record the base's record at its
final positions (outcome, typed weights, the offset formed once, every original column
and name, positions as labels), its transformed problem that record's transform at its
order and its scaled design step 9 of that record's active columns, bit for bit, and it
keeps those canonical records, never a caller's; its metadata is the base's at its
positions, its conditioning its ledger's step 3 history.  Its blocks are those of
``source``, the blocks of the rows of steps 3 and 4, on its final rows
(:func:`restrict`), original level labels included; ``None`` for both declares no
blocks.  The outcome that holds it binds that source to its prepared effects, the
request and the diagnostics.

Sources of the blocks.  :mod:`powermean.data.sample_fe_source` is their topic:
:class:`Source`, :class:`Derived`, :func:`restrict`, :func:`same_blocks` and
:func:`same_bits`, defined there with their internal helpers; this module still provides
each of their names, the very same object.  The sample keeps its source on its final
rows, the canonical blocks, never a caller's grouping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.accounting import (
    EXCLUSION_REASONS,
    MISSING,
    MISSING_GROUPS,
    RETAINED,
    ZERO_OUTCOME,
    ZERO_WEIGHT,
    CodeArray,
    MissingAccount,
    bool_mask,
    frozen_codes,
    frozen_index,
)
from powermean.data.collinearity_records import ABSORBED, COLLINEAR
from powermean.data.design import ScaledDesign, scale_columns
from powermean.data.inputs import RegressionData
from powermean.data.raw import AlignedColumns
from powermean.data.retained import OrderRows, _exact_index, _inverse, _regression_data
from powermean.data.sample import SampleBase
from powermean.data.sample_fe_source import (
    Derived,
    Source,
    _encoded,
    _one_source,
    _ordinals,
    restrict,
    same_bits,
    same_blocks,
)
from powermean.data.weights import MAXIMUM_COUNT
from powermean.fe.groups import Grouping
from powermean.model.order import (
    IndexArray,
    TransformedProblem,
    _flags,
    transform_problem,
)
from powermean.model.order_scale import OrderScale, _mean_order

# The names of this module and of the sources of the blocks defined in
# data/sample_fe_source.py: their import routes from here are kept.
__all__ = [
    "BOUND",
    "FE_REASONS",
    "FIXED_EFFECT",
    "LARGEST",
    "LEDGER",
    "NO_BLOCK",
    "SEPARATED",
    "SINGLETON",
    "Canonical",
    "Derived",
    "FESample",
    "RowLedger",
    "Source",
    "_encoded",
    "_one_source",
    "_ordinals",
    "final_sample",
    "ledger_of",
    "replicas_of",
    "restrict",
    "same_bits",
    "same_blocks",
    "same_data",
    "same_missing",
    "same_transform",
    "scaled",
]

# The reasons of the ordered sample: those of steps 1 to 4, then steps 5, 6 and 7.
FE_REASONS: tuple[str, ...] = (
    *EXCLUSION_REASONS,
    "singleton",
    "separated_by_fixed_effect",
    "separated",
)
SINGLETON, FIXED_EFFECT, SEPARATED = 5, 6, 7
NO_BLOCK = -1  # the block of a row that step 5 or 6 did not exclude
LARGEST = int(np.iinfo(np.intp).max)  # no index beyond it is representable
LEDGER = "the ledger's codes, rows, inverse, blocks, missing rows and counts disagree"
BOUND = "the parts of the sample are not bound to its rows"


def same_missing(a: MissingAccount, b: MissingAccount) -> bool:
    """Whether two accounts of step 2 mark the same rows in each named group and in all.

    Groups overlap, so each is compared by its name: a union is no group's account.
    """
    if set(a.groups) != set(b.groups):
        return False
    pairs = [(a.groups[group], b.groups[group]) for group in a.groups]
    found = [(np.asarray(x), np.asarray(y)) for x, y in pairs]
    found.append((np.asarray(a.excluded), np.asarray(b.excluded)))
    return all(x.dtype == y.dtype and same_bits(x, y) for x, y in found)


def _account(missing: object, codes: IndexArray) -> MissingAccount:
    """An owned copy of step 2's account, refused unless it states the missing rows.

    Exactly the seven groups, each an exact Boolean mask of every input row
    (:func:`~powermean.data.accounting.bool_mask`), their union the excluded rows and
    those the rows coded missing; kept in the design's order, read-only, in owned bytes.
    """
    if not isinstance(missing, MissingAccount) or not isinstance(
        missing.groups, Mapping
    ):
        raise ValueError(LEDGER)
    if set(missing.groups) != set(MISSING_GROUPS):
        raise ValueError(LEDGER)
    rows = codes.size
    groups = {
        group: _flags(bool_mask(missing.groups[group], rows, f"missing[{group!r}]"))
        for group in MISSING_GROUPS
    }
    excluded = _flags(bool_mask(missing.excluded, rows, "missing rows"))
    if not np.array_equal(np.logical_or.reduce([*groups.values()]), excluded):
        raise ValueError(LEDGER)
    if not np.array_equal(excluded, codes == MISSING):
        raise ValueError(LEDGER)
    return MissingAccount._owned(groups, excluded)


@dataclass(frozen=True, eq=False)
class RowLedger:
    """Every input row: its first exclusion and block, or its current position.

    ``codes``: zero where current, else one plus the reason's index in
    :data:`FE_REASONS`; ``blocks``: the first-cause block of steps 5 and 6, else
    ``-1``; ``positions`` and ``retained_index``: the current rows, increasing, and
    each input row's place or ``-1``; ``missing``: step 2's diagnostics, overlapping
    groups kept apart, an owned copy; ``replicas``: with frequency weights the exact
    count of each row of steps 3 and 4 (zero elsewhere), else ``None``.  ``ValueError``
    unless the fields state one ledger (module docstring).
    """

    codes: CodeArray
    blocks: IndexArray
    positions: IndexArray
    retained_index: IndexArray
    missing: MissingAccount
    replicas: IndexArray | None

    def __post_init__(self) -> None:
        """Own exact arrays and step 2's account; ``ValueError`` unless one ledger."""
        codes = _exact_index(self.codes, "codes", upper=len(FE_REASONS) + 1)
        rows = codes.size
        blocks = _exact_index(self.blocks, "blocks", lower=NO_BLOCK, upper=LARGEST)
        positions = _exact_index(self.positions, "positions", upper=rows)
        index = _exact_index(self.retained_index, "inverse", lower=-1, upper=rows)
        missing = _account(self.missing, codes)
        caused = (codes == SINGLETON) | (codes == FIXED_EFFECT)
        agree = np.array_equal(positions, np.flatnonzero(codes == RETAINED))
        agree &= np.array_equal(index, _inverse(positions, rows))
        agree &= np.array_equal(blocks >= 0, caused)
        if self.replicas is not None:
            replicas = _exact_index(self.replicas, "replicas", upper=MAXIMUM_COUNT + 1)
            counted = (codes == RETAINED) | (codes > ZERO_WEIGHT)
            agree &= np.array_equal(replicas > 0, counted)
            agree &= sum(replicas.tolist()) <= MAXIMUM_COUNT
            object.__setattr__(self, "replicas", replicas)
        if not agree:
            raise ValueError(LEDGER)
        object.__setattr__(self, "codes", frozen_codes(codes.astype(np.int8)))
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "retained_index", index)
        object.__setattr__(self, "missing", missing)

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return int(self.codes.size)

    @property
    def count(self) -> int:
        """The number of current rows."""
        return int(self.positions.size)

    @property
    def counts(self) -> Mapping[str, int]:
        """The physical rows of each reason, in the order of the steps, and current."""
        found = {
            reason: int(np.count_nonzero(self.codes == code))
            for code, reason in enumerate(FE_REASONS, start=1)
        }
        found["retained"] = self.count
        return MappingProxyType(found)

    @property
    def totals(self) -> Mapping[str, int] | None:
        """With frequency weights, the exact replica totals of steps 5 to 7 and current.

        The rows of steps 1 to 4 hold no validated count and have none.
        """
        if self.replicas is None:
            return None
        found: dict[str, int] = {}
        for code in (SINGLETON, FIXED_EFFECT, SEPARATED, RETAINED):
            rows = self.replicas[self.codes == code].tolist()
            found[FE_REASONS[code - 1] if code else "retained"] = sum(rows)
        return MappingProxyType(found)

    def reason(self, position: int) -> str | None:
        """The first reason of the input row ``position``, or ``None`` if current."""
        code = int(self.codes[position])
        return None if code == RETAINED else FE_REASONS[code - 1]

    def table(self, names: Sequence[str]) -> tuple[tuple[str, int], ...]:
        """The physical counts in the order of the steps; steps 5, 6 by block.

        ``names`` are the block names, in formula order.
        """
        counts = self.counts
        groups = [(f"missing: {group}", n) for group, n in self.missing.counts.items()]
        found = [("subset", counts["subset"]), ("missing", counts["missing"]), *groups]
        found += [(reason, counts[reason]) for reason in FE_REASONS[2:4]]
        for code in (SINGLETON, FIXED_EFFECT):
            reason = FE_REASONS[code - 1]
            found.append((reason, counts[reason]))
            for g, name in enumerate(names):
                rows = (self.codes == code) & (self.blocks == g)
                found.append((f"{reason}: {name}", int(np.count_nonzero(rows))))
        return (*found, ("separated", counts["separated"]), ("retained", self.count))

    def exclude(
        self, rows: ArrayLike, code: int, blocks: ArrayLike | None = None
    ) -> RowLedger:
        """A new ledger with the current ``rows`` excluded by ``code`` (5, 6 or 7).

        ``blocks`` are the rows' first blocks, given exactly for steps 5 and 6.  Rows
        and blocks are exact integers, checked before they are narrowed or used.
        ``ValueError`` for another code, a row that is not current or is given twice,
        or blocks missing, extra or malformed.
        """
        exact = isinstance(code, int | np.integer) and not isinstance(code, bool)
        if not exact or code not in (SINGLETON, FIXED_EFFECT, SEPARATED):
            raise ValueError("only steps 5, 6 and 7 exclude rows here")
        found = _exact_index(rows, "rows to exclude", upper=self.nobs)
        once = np.unique(found).size == found.size
        if not once or (self.retained_index[found] < 0).any():
            raise ValueError("only current rows, each once, can be excluded")
        if (blocks is None) != (code == SEPARATED):
            raise ValueError("first-cause blocks are given exactly for steps 5 and 6")
        first, codes = self.blocks.copy(), self.codes.copy()
        if blocks is not None:
            causes = _exact_index(blocks, "first-cause blocks", upper=LARGEST)
            if causes.size != found.size:
                raise ValueError("one first-cause block per excluded row")
            first[found] = causes
        codes[found] = code
        positions = frozen_index(np.flatnonzero(codes == RETAINED))
        inverse = _inverse(positions, self.nobs)
        found_codes, found_first = frozen_codes(codes), frozen_index(first)
        missing, replicas = self.missing, self.replicas
        return RowLedger(
            found_codes, found_first, positions, inverse, missing, replicas
        )


def replicas_of(data: RegressionData, rows: OrderRows) -> IndexArray | None:
    """The exact count of each row of ``data`` (frequency weights only), by input row.

    The record validated them as exact integers of at most ``2**53``.
    """
    if data.weights is None or not data.weights.frequency:
        return None
    counts = np.zeros(rows.nobs, dtype=np.intp)
    counts[rows.positions] = [int(value) for value in data.weights.values.tolist()]
    return frozen_index(counts)


def ledger_of(
    rows: OrderRows, missing: MissingAccount, replicas: IndexArray | None
) -> RowLedger:
    """The ledger of the rows that steps 1 to 4 keep, before any fixed-effect step."""
    blocks = frozen_index(np.full(rows.nobs, NO_BLOCK, dtype=np.intp))
    return RowLedger(
        rows.codes, blocks, rows.positions, rows.retained_index, missing, replicas
    )


# ------------------------------------------------------------------------ step 9


def same_data(a: RegressionData, b: RegressionData) -> bool:
    """Whether two numeric records hold the same rows, columns, weights and labels."""
    pairs = [(a.y, b.y), (a.design, b.design), (a.offset, b.offset)]
    if a.exposure is not None and b.exposure is not None:
        pairs.append((a.exposure, b.exposure))
    if a.weights is None or b.weights is None:
        weights = a.weights is None and b.weights is None
    else:
        weights = a.weights.kind == b.weights.kind
        pairs.append((a.weights.values, b.weights.values))
    labels = (a.row_labels, a.column_names) == (b.row_labels, b.column_names)
    exposure = (a.exposure is None) == (b.exposure is None)
    return weights and labels and exposure and all(same_bits(x, y) for x, y in pairs)


def same_transform(a: TransformedProblem, b: TransformedProblem) -> bool:
    """Whether two transformed problems are the same, bit for bit."""
    s, t = a.selection, b.selection
    pairs = [(a.response, b.response), (a.offset, b.offset), (s.mask, t.mask)]
    pairs += [(s.positions, t.positions), (s.removed, t.removed)]
    flags = (a.omega, a.route, s.nobs, s.positive_only)
    same = flags == (b.omega, b.route, t.nobs, t.positive_only)
    return same and all(same_bits(x, y) for x, y in pairs)


def scaled(data: RegressionData, active: IndexArray) -> ScaledDesign | None:
    """Step 9 on the ``active`` original columns of ``data``; ``None`` without one."""
    if not active.size:
        return None
    found = active.tolist()
    names = tuple(data.column_names[j] for j in found)
    return scale_columns(data.design[:, found], names)


def _same_design(a: ScaledDesign | None, b: ScaledDesign | None) -> bool:
    """Whether two scaled designs are the same, bit for bit, or both absent."""
    if a is None or b is None:
        return a is None and b is None
    same = same_bits(a.matrix, b.matrix) and same_bits(a.scale, b.scale)
    return same and a.column_names == b.column_names


# The canonical transform, scaled design and blocks that a bound sample keeps.
Canonical = tuple[TransformedProblem, ScaledDesign | None, Grouping | None]


@dataclass(frozen=True, eq=False)
class FESample:
    """The completed ordered sample: rows, problem, columns and maps.

    ``omega``, ``positive_only``, ``conditioned`` as for an order; ``ledger`` every
    input row's first exclusion or final position; ``base`` the prepared base of every
    input row; ``data`` the final rows with every original candidate column, unscaled,
    labeled by their positions, weights of their kind; ``problem`` the transform of the
    final rows; ``source`` the blocks on the rows of steps 3 and 4 (codes and level
    labels) or ``None``; ``grouping`` the blocks on the final rows or ``None``;
    ``reasons`` per original column, ``None`` when active, else its collinearity reason;
    ``active`` the original indices of the active columns, increasing, possibly empty;
    ``design`` step 9 on the active columns, ``None`` without one.  ``ValueError``
    unless every part is bound (module docstring); the record keeps the canonical
    transform, design and blocks that it is bound to (the blocks: its source on its
    final rows, :func:`restrict`).
    """

    omega: float
    positive_only: bool
    conditioned: bool
    ledger: RowLedger
    base: SampleBase
    data: RegressionData
    problem: TransformedProblem
    source: Grouping | None
    grouping: Grouping | None
    reasons: tuple[str | None, ...]
    active: IndexArray
    design: ScaledDesign | None

    def __post_init__(self) -> None:
        """Own every part; refuse a part of other rows or data, keep canonical ones."""
        object.__setattr__(self, "omega", _mean_order(self.omega))
        reasons = tuple(self.reasons)
        object.__setattr__(self, "reasons", reasons)
        active = _exact_index(self.active, "active columns", upper=len(reasons))
        object.__setattr__(self, "active", active)
        canonical = self._bound() if self._typed() else None
        if canonical is None:
            raise ValueError(BOUND)
        problem, design, grouping = canonical
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "grouping", grouping)

    def _typed(self) -> bool:
        """Whether every part is of its record type and the reasons of their domain."""
        parts = (
            isinstance(self.ledger, RowLedger),
            isinstance(self.base, SampleBase),
            isinstance(self.data, RegressionData),
            isinstance(self.problem, TransformedProblem),
            isinstance(self.source, Grouping | None),
            isinstance(self.grouping, Grouping | None),
            isinstance(self.design, ScaledDesign | None),
            isinstance(self.positive_only, bool),
            isinstance(self.conditioned, bool),
        )
        known = (None, ABSORBED, COLLINEAR)
        return all(parts) and all(reason in known for reason in self.reasons)

    def _bound(self) -> Canonical | None:
        """The canonical transform, design and blocks of a bound record, or ``None``."""
        ledger, data, positions = self.ledger, self.data, self.ledger.positions
        kept = [j for j, reason in enumerate(self.reasons) if reason is None]
        if ledger.count == 0 or ledger.nobs != self.base.nobs:
            return None
        if len(self.reasons) != data.nparams or kept != self.active.tolist():
            return None
        history = bool((ledger.codes == ZERO_OUTCOME).any())
        if self.conditioned != history or (history and not self.positive_only):
            return None
        if not same_missing(ledger.missing, self.base.missing):
            return None
        if not self._blocks():
            return None
        if not same_data(data, _regression_data(self.base.inputs, positions)):
            return None
        problem = transform_problem(data.y, self.omega, offset=data.offset)
        design = scaled(data, self.active)
        same = same_transform(self.problem, problem)
        if not same or not _same_design(self.design, design):
            return None
        source = self.source
        blocks = None if source is None else restrict(source, positions, ledger.nobs)
        return problem, design, blocks

    def _blocks(self) -> bool:
        """None without a source (and no first-cause block), else the source restricted.

        The source holds exactly the rows of steps 3 and 4 (codes 0 and 5 to 7), so the
        final rows (code 0) are an exact subset of them.
        """
        ledger, source, grouping = self.ledger, self.source, self.grouping
        if source is None or grouping is None:
            free = not (ledger.blocks >= 0).any()
            return source is None and grouping is None and free
        eligible = (ledger.codes == RETAINED) | (ledger.codes > ZERO_WEIGHT)
        if not same_bits(source.positions, np.flatnonzero(eligible)):
            return False
        if int(ledger.blocks.max()) >= len(source.blocks):
            return False
        found = restrict(source, ledger.positions, ledger.nobs)
        return same_blocks(grouping, found)

    @property
    def positions(self) -> IndexArray:
        """The original positions of the final rows, increasing."""
        return self.ledger.positions

    @property
    def count(self) -> int:
        """The number of final rows."""
        return self.ledger.count

    @property
    def effective_n(self) -> int:
        """``N``: the exact total of frequency counts, else the rows of the record."""
        return self.data.effective_n

    @property
    def active_names(self) -> tuple[str, ...]:
        """The names of the active columns, in formula order."""
        return tuple(self.data.column_names[j] for j in self.active.tolist())

    @property
    def order_scale(self) -> OrderScale | None:
        """The accepted map of the active columns' scale; ``None`` without one."""
        return None if self.design is None else self.design.order_scale(self.omega)

    @property
    def metadata(self) -> AlignedColumns:
        """The base's aligned metadata at the final positions, a fresh owned record."""
        return self.base.inputs.metadata.take(self.positions)


def final_sample(
    rows: OrderRows,
    ledger: RowLedger,
    *,
    base: SampleBase,
    data: RegressionData,
    problem: TransformedProblem,
    source: Grouping | None,
    grouping: Grouping | None,
    reasons: Sequence[str | None],
) -> FESample:
    """Step 9 on the final active columns of the final rows, and the sample.

    ``source`` is the blocks encoded on ``rows`` (steps 3 and 4), ``None`` without.
    """
    found = [j for j, reason in enumerate(reasons) if reason is None]
    active = frozen_index(np.asarray(found, dtype=np.intp))
    return FESample(
        omega=rows.omega,
        positive_only=rows.positive_only,
        conditioned=rows.conditioned,
        ledger=ledger,
        base=base,
        data=data,
        problem=problem,
        source=source,
        grouping=grouping,
        reasons=tuple(reasons),
        active=active,
        design=scaled(data, active),
    )


def _final_owned_sample(
    rows: OrderRows, ledger: RowLedger, *, base: SampleBase, data: RegressionData,
    problem: TransformedProblem, groups: tuple[Grouping | None, Grouping | None],
    reasons: Sequence[str | None], prepared_design: ScaledDesign | None = None,
) -> FESample:
    """Compose a private coordinator's actual already checked canonical parts.

    Order, weights and numeric inputs were checked by the producer; the fresh
    coordinator admitted every diagnostic on these exact current rows. Public
    FESample/final_sample keep their original binding and ownership checks.
    """
    source, grouping = groups
    reasons = tuple(reasons)
    active = frozen_index(np.asarray(
        [j for j, reason in enumerate(reasons) if reason is None], dtype=np.intp
    ))
    design = (
        prepared_design if prepared_design is not None and active.size == data.nparams
        else scaled(data, active)
    )
    result = object.__new__(FESample)
    for name, value in (
        ("omega", rows.omega), ("positive_only", rows.positive_only),
        ("conditioned", rows.conditioned), ("ledger", ledger), ("base", base),
        ("data", data), ("problem", problem), ("source", source),
        ("grouping", grouping), ("reasons", reasons), ("active", active),
        ("design", design),
    ):
        object.__setattr__(result, name, value)
    return result
