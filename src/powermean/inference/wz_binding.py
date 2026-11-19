"""Declared three-way PPML provenance and bounded present-cell rows (03.5.4, 04.7).

Labels are read from the actual formula's owned blocks, never recoded to satisfy
an adapter. Equal partitions cannot substitute for the requested source name.
Exact int/str labels on present directed pair-date cells identify structural
roles before version admission. Version 0.1 requires one common country set
and every retained pair at every observed date; pairs may be absent altogether,
including any pattern of self-pairs. The general present-cell kernel realizes
rows by pair offsets, never padding to P*T, and carries the actual present P.

Identified nuisance coordinates (:class:`Coordinates`) are exact: the canonical
leading columns of the integer echelon of within-pair differences of the two
retained interaction indicators. The pair-cancelled route reuses its own fit's
admitted exact quotient (fe.admission_incidence); the joint route forms the same
canonical set once per fit from the exact complete-date incidence structure,
with certificate_basis's exact elimination for other layouts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Protocol, SupportsInt, cast

import numpy as np
from numpy.typing import NDArray

from powermean.data.collinearity_records import ABSORBED
from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.fe.cancel import CancelStructure
from powermean.fe.certificate_basis import Reduced, _new_tail, _over, _reduce_by
from powermean.inference.wz_operator import (
    IndexArray,
    Reservation,
    Rows,
    Sizes,
    _plan,
    component_labels,
)
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds

type Label = str | int
type Workspace = Callable[[Sizes], tuple[int, int]]
BUDGET = "WZ covariance exceeds explicit byte/array-work bounds"
# Integer keys in range(bound) use a direct boolean table while bound <= 4*n,
# keeping seen + position (9*bound <= 36*n B) well within reserve_validation's
# 128*n byte pre-plan allowance; wider key ranges fall back to sorting.
DIRECT_TABLE = 4
INTERCEPT = "Intercept"
_csr_matrix = import_module("scipy.sparse").csr_matrix
_connected_components = import_module("scipy.sparse.csgraph").connected_components


class SourceBlock(Protocol):
    """Lossless original FE block, live or retained, with detached public arrays."""

    @property
    def name(self) -> str:
        """Declared term name."""
        ...

    @property
    def codes(self) -> IndexArray:
        """Original first-occurrence codes."""
        ...

    @property
    def levels(self) -> tuple[object, ...]:
        """Original labels, including interaction components."""
        ...


class SourceGrouping(Protocol):
    """Original positions and ordered blocks, without requiring a live sample."""

    @property
    def blocks(self) -> tuple[SourceBlock, ...]:
        """Blocks in declared term order."""
        ...

    @property
    def positions(self) -> IndexArray:
        """Original row positions."""
        ...


def omitted_slopes(
    names: Sequence[str], reasons: Sequence[str | None], formula_intercept: int | None
) -> bool:
    """Refuse original omissions except the formula intercept absorbed by FEs.

    ``formula_intercept`` is the original column position from the fitted literal
    term, not a name-based guess. No column is rescanned or reclassified, and
    active/reporting maps are unchanged. Missing provenance grants no exception.
    """
    return any(
        reason is not None
        and not (
            index == formula_intercept and name == INTERCEPT and reason == ABSORBED
        )
        for index, (name, reason) in enumerate(zip(names, reasons, strict=True))
    )


def unavailable(case: str) -> SpecificationError:
    """Refuse an unsupported WZ request explicitly, without fallback."""
    return SpecificationError(
        "specification.unsupported_inference",
        {"method": "WZ pair covariance", "case": case},
    )


@dataclass(frozen=True)
class WZSource:
    """Declared FE terms and fitted formula-intercept original-column identity.

    Every call rereads current owned labels. Legacy three-argument construction
    has no intercept provenance and therefore grants no omission exception.
    """

    terms: tuple[tuple[str, ...], ...]
    grouping: SourceGrouping
    row_labels: tuple[object, ...]
    formula_intercept: int | None = None

    def _pair_index(self, name: str, blocks: tuple[SourceBlock, ...]) -> int:
        """Validate declared three-way blocks and locate the named pair block."""
        terms = self.terms
        if len(terms) != 3 or tuple(b.name for b in blocks) != tuple(
            "^".join(term) for term in terms
        ):
            raise unavailable("exact declared three-way FE source required")
        matches = [i for i, b in enumerate(blocks) if b.name == name]
        if len(matches) != 1:
            raise unavailable("cluster must name the actual declared pair FE source")
        return matches[0]

    def _roles(
        self, name: str, blocks: tuple[SourceBlock, ...]
    ) -> tuple[int, tuple[int, int], tuple[int, int], tuple[str, ...], str]:
        """Pair block, interaction blocks, their country sides, pair term, origin.

        ``blocks`` is the caller's single read of ``grouping.blocks``.
        """
        terms = self.terms
        pair = self._pair_index(name, blocks)
        indices = [i for i in range(3) if i != pair]
        first, second = terms[indices[0]], terms[indices[1]]
        common = set(first) & set(second)
        if len(first) != 2 or len(second) != 2 or len(common) != 1:
            raise unavailable("two declared interactions sharing one date required")
        date = next(iter(common))
        origin, destination = (
            next(v for v in term if v != date) for term in (first, second)
        )
        pair_term = terms[pair]
        if len(pair_term) != 1 and set(pair_term) != {origin, destination}:
            raise unavailable(
                "the third term must be the pair column or its interaction"
            )
        return (
            pair,
            (indices[0], indices[1]),
            (first.index(origin), second.index(destination)),
            pair_term,
            origin,
        )

    def components(self, name: str) -> tuple[tuple[Label, ...], ...]:
        """Recognize two interactions sharing a date from their declared terms."""
        blocks = self.grouping.blocks
        pair, indices, sides, pair_term, origin = self._roles(name, blocks)
        a, ta = _interaction(blocks[indices[0]], sides[0])
        b, tb = _interaction(blocks[indices[1]], sides[1])
        if ta != tb:
            raise unavailable("shared date source labels differ")
        _pair_source(blocks[pair], tuple(zip(a, b, strict=True)), pair_term, origin)
        return a, b, ta


def _is_plain_scalar(value: object) -> bool:
    """Passive identity-only check for exact built-in int or str."""
    kind = type(value)
    return kind is int or kind is str


def _label(value: object) -> Label:
    if type(value) not in (int, str):
        raise unavailable(
            "only exact original int/str component and row labels are supported"
        )
    if not isinstance(value, int | str):
        raise TypeError("unsupported label")
    return value


def _interaction(
    block: SourceBlock, side: int
) -> tuple[tuple[Label, ...], tuple[Label, ...]]:
    levels = block.levels
    components: list[tuple[Label, Label]] = []
    for label in levels:
        if type(label) is not tuple or len(label) != 2:
            raise unavailable("original interaction tuple labels required")
        components.append((_label(label[side]), _label(label[1 - side])))
    rows = tuple(components[int(code)] for code in block.codes)
    return tuple(v[0] for v in rows), tuple(v[1] for v in rows)


def _pair_source_read(
    levels: Sequence[object],
    codes: Sequence[SupportsInt] | IndexArray,
    pairs: tuple[tuple[Label, Label], ...],
    term: tuple[str, ...],
    origin: str,
) -> None:
    """Validate already-read pair levels and codes against row pairs."""
    forward: dict[object, tuple[Label, Label]] = {}
    reverse: dict[tuple[Label, Label], object] = {}
    for code, key in zip(codes, pairs, strict=True):
        label = levels[int(code)]
        if len(term) == 2:
            expected = key if term[0] == origin else key[::-1]
            if type(label) is not tuple or tuple(map(_label, label)) != expected:
                raise unavailable("pair interaction source labels differ")
        elif type(label) is tuple and len(label) == 2:
            tuple(map(_label, label))
        else:
            _label(label)
        if label in forward and forward[label] != key:
            raise unavailable("declared pair source combines distinct directed pairs")
        if key in reverse and reverse[key] != label:
            raise unavailable("declared pair source splits a directed pair")
        forward[label], reverse[key] = key, label


def _pair_source(
    block: SourceBlock,
    pairs: tuple[tuple[Label, Label], ...],
    term: tuple[str, ...],
    origin: str,
) -> None:
    levels = block.levels
    _pair_source_read(levels, block.codes, pairs, term, origin)


@dataclass(frozen=True)
class WZSupport:
    """One call's recognized present cells; never reused as admission."""

    origins: tuple[Label, ...]
    destinations: tuple[Label, ...]
    dates: tuple[Label, ...]
    pairs: int


@dataclass(frozen=True)
class WZCells:
    """One recognized role in integer codes, equal to coding the label tuples.

    ``origin``, ``destination`` and ``date`` are each row's codes in order of
    first appearance among the rows; ``found`` is the recognized present-cell support
    of the same rows.
    """

    origin: IndexArray
    destination: IndexArray
    date: IndexArray
    found: WZSupport


@dataclass(frozen=True)
class _RawComponents:
    """Validated single-read integer components of ``WZSource.components``."""

    a: IndexArray
    b: IndexArray
    ta: IndexArray
    dense: IndexArray
    pairs: int
    countries: tuple[Label, ...]
    periods: tuple[Label, ...]


def _code_index(code: object, limit: int) -> int:
    """Match ``_interaction``'s list indexing without rereading ``codes``."""
    idx = int(cast("SupportsInt", code))
    if idx < 0:
        idx += limit
    if idx < 0 or idx >= limit:
        raise IndexError("list index out of range")
    return idx


def _coerce_codes(raw: object, limit: int) -> IndexArray:
    """Validate or convert already-read block codes to 1D ``np.intp``."""
    if (
        isinstance(raw, np.ndarray)
        and raw.ndim == 1
        and raw.dtype.kind == "i"
        and (raw.size == 0 or (int(raw.min()) >= 0 and int(raw.max()) < limit))
    ):
        return raw.astype(np.intp, copy=False)
    items = [_code_index(code, limit) for code in cast("Sequence[object]", raw)]
    return np.asarray(items, dtype=np.intp)


def _read_interaction(
    block: SourceBlock,
    side: int,
    country: dict[Label, int],
    date: dict[Label, int],
) -> tuple[IndexArray, IndexArray]:
    """Read one interaction block once in exact ``_interaction`` order."""
    levels = block.levels
    c_ids: list[int] = []
    d_ids: list[int] = []
    other = 1 - side
    for label in levels:
        if type(label) is not tuple or len(label) != 2:
            raise unavailable("original interaction tuple labels required")
        c_val = _label(label[side])
        d_val = _label(label[other])
        c_ids.append(country.setdefault(c_val, len(country)))
        d_ids.append(date.setdefault(d_val, len(date)))
    codes = _coerce_codes(block.codes, len(levels))
    if len(c_ids) <= DIRECT_TABLE * max(1, codes.size):
        return (
            np.asarray(c_ids, dtype=np.intp)[codes],
            np.asarray(d_ids, dtype=np.intp)[codes],
        )
    return (
        np.fromiter((c_ids[int(c)] for c in codes), dtype=np.intp, count=codes.size),
        np.fromiter((d_ids[int(c)] for c in codes), dtype=np.intp, count=codes.size),
    )


def _plain_pair_levels(levels: Sequence[object], interaction: bool) -> bool:
    """Whether every pair level passes passive exact-type checks and is unique."""
    for label in levels:
        if type(label) is tuple:
            if (
                len(label) != 2
                or not _is_plain_scalar(label[0])
                or not _is_plain_scalar(label[1])
            ):
                return False
        elif interaction or not _is_plain_scalar(label):
            return False
    return len(set(levels)) == len(levels)


def _functional(keys: IndexArray, values: IndexArray, count: int) -> bool:
    """Whether every key in ``range(count)`` carries a single value."""
    held = np.empty(count, dtype=values.dtype)
    held[keys] = values
    return bool(np.array_equal(held[keys], values))


def _dense_pairs(
    a: IndexArray, b: IndexArray, n_countries: int
) -> tuple[IndexArray, int]:
    """Dense directed-pair codes and pair count from integer country codes."""
    if a.size == 0:
        return np.empty(0, dtype=np.intp), 0
    if n_countries <= math.isqrt(int(np.iinfo(np.intp).max)):
        dense = _dense(a * n_countries + b, n_countries * n_countries)[1]
    else:
        pairs_2d = np.column_stack((a, b))
        _, inv = np.unique(pairs_2d, axis=0, return_inverse=True)
        dense = inv.astype(np.intp, copy=False).reshape(a.shape)
    return dense, int(dense.max()) + 1


def _pair_ends_match(
    levels: Sequence[object],
    codes: IndexArray,
    ends: tuple[IndexArray, IndexArray],
    country: dict[Label, int],
) -> bool:
    """Whether an interaction pair block's components match ``ends``."""
    for side, want in enumerate(ends):
        found = np.fromiter(
            (
                country.get(cast("tuple[Label, Label]", label)[side], -1)
                for label in levels
            ),
            dtype=np.intp,
            count=len(levels),
        )
        matched = bool(np.array_equal(found[codes], want))
        del found
        if not matched:
            return False
    return True


def _fast_pair_codes(
    levels: Sequence[object], raw_codes: object, n: int, interaction: bool
) -> IndexArray | None:
    """Return 1D ``np.intp`` pair codes when eligible for array bijection checks."""
    if (
        not isinstance(raw_codes, np.ndarray)
        or raw_codes.ndim != 1
        or raw_codes.dtype.kind != "i"
        or raw_codes.shape != (n,)
        or len(levels) > DIRECT_TABLE * max(1, n)
        or (
            n > 0
            and (
                not levels
                or int(raw_codes.min()) < 0
                or int(raw_codes.max()) >= len(levels)
            )
        )
        or not _plain_pair_levels(levels, interaction)
    ):
        return None
    return raw_codes.astype(np.intp, copy=False)


def _read_pair(
    block: SourceBlock,
    country: dict[Label, int],
    a: IndexArray,
    b: IndexArray,
    *,
    pair_term: tuple[str, ...],
    origin_name: str,
) -> tuple[IndexArray, int]:
    """Read and validate the pair block once without speculative rereads."""
    levels = block.levels
    raw_codes = block.codes
    codes = _fast_pair_codes(levels, raw_codes, a.size, len(pair_term) == 2)
    if codes is not None:
        del raw_codes
        if a.size == 0:
            return np.empty(0, dtype=np.intp), 0
        if len(pair_term) == 2:
            ends = (a, b) if pair_term[0] == origin_name else (b, a)
            if not _pair_ends_match(levels, codes, ends, country):
                raise unavailable("pair interaction source labels differ")
        dense, pairs = _dense_pairs(a, b, len(country))
        if _functional(codes, dense, len(levels)) and _functional(dense, codes, pairs):
            return dense, pairs
        raw_for_fallback: Sequence[SupportsInt] | IndexArray = codes
    else:
        raw_for_fallback = cast("Sequence[SupportsInt] | IndexArray", raw_codes)
    countries = tuple(country)
    row_pairs = tuple(
        (countries[int(i)], countries[int(j)]) for i, j in zip(a, b, strict=True)
    )
    _pair_source_read(levels, raw_for_fallback, row_pairs, pair_term, origin_name)
    return _dense_pairs(a, b, len(country))


def _read_components(source: WZSource, name: str) -> _RawComponents:
    """Single-read integer equivalent of ``WZSource.components(name)``."""
    blocks = source.grouping.blocks
    pair, indices, sides, pair_term, origin_name = source._roles(name, blocks)
    country: dict[Label, int] = {}
    date: dict[Label, int] = {}
    a, ta = _read_interaction(blocks[indices[0]], sides[0], country, date)
    b, tb = _read_interaction(blocks[indices[1]], sides[1], country, date)
    if ta.size != tb.size or not np.array_equal(ta, tb):
        raise unavailable("shared date source labels differ")
    del tb
    dense, pairs = _read_pair(
        blocks[pair],
        country,
        a,
        b,
        pair_term=pair_term,
        origin_name=origin_name,
    )
    return _RawComponents(a, b, ta, dense, pairs, tuple(country), tuple(date))


def _first_appearance(ids: IndexArray, bound: int) -> tuple[IndexArray, IndexArray]:
    """Codes by first appearance, and the integer in ``range(bound)`` of each code."""
    if ids.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    if bound > 2 * ids.size:
        unique_vals, local_ids = np.unique(ids, return_inverse=True)
        coded, local_order = _first_appearance(
            local_ids.astype(np.intp, copy=False).reshape(ids.shape),
            int(unique_vals.size),
        )
        return coded, unique_vals[local_order].astype(np.intp, copy=False)
    first = np.full(bound, ids.size, dtype=np.intp)
    np.minimum.at(first, ids, np.arange(ids.size, dtype=np.intp))
    present = np.flatnonzero(first < ids.size)
    order = present[np.argsort(first[present], kind="stable")].astype(
        np.intp, copy=False
    )
    del first, present
    rank = np.empty(bound, dtype=np.intp)
    rank[order] = np.arange(order.size, dtype=np.intp)
    return rank[ids], order


def _repeated_cells(dense: IndexArray, t: IndexArray, pairs: int, n_dates: int) -> bool:
    """Whether any ``(pair, date)`` cell occurs more than once."""
    if pairs <= np.iinfo(np.intp).max // max(1, n_dates):
        cells = dense * n_dates + t
        seen = _seen(cells, pairs * n_dates)
        if seen is None:
            return bool(np.unique(cells).size != cells.size)
        return int(np.count_nonzero(seen)) != cells.size
    order = np.lexsort((t, dense))
    sd, st = dense[order], t[order]
    return bool(np.any((sd[1:] == sd[:-1]) & (st[1:] == st[:-1])))


def _support_cells(raw: _RawComponents) -> WZCells:
    """Recognize present directed pair-date cells on validated integer components.

    No N*N*T grid is built.  Repeated (origin, destination, date) cells refuse, as do
    fewer than 3 origins, 3 destinations or 2 dates, and a panel in which no pair is
    present at two dates (no within-pair variation exists).
    """
    i, origins = _first_appearance(raw.a, len(raw.countries))
    j, destinations = _first_appearance(raw.b, len(raw.countries))
    t, dates = _first_appearance(raw.ta, len(raw.periods))
    if origins.size < 3 or destinations.size < 3 or dates.size < 2:
        raise unavailable(
            "present directed panel with at least 3 origins, 3 destinations "
            "and 2 dates required"
        )
    if _repeated_cells(raw.dense, t, raw.pairs, int(dates.size)):
        raise unavailable("repeated pair-time cells")
    if raw.pairs == raw.a.size:
        raise unavailable("at least one pair present at two dates required")
    found = WZSupport(
        tuple(raw.countries[int(v)] for v in origins),
        tuple(raw.countries[int(v)] for v in destinations),
        tuple(raw.periods[int(v)] for v in dates),
        raw.pairs,
    )
    return WZCells(i, j, t, found)


def recognize(source: WZSource, name: str) -> WZCells:
    """The components of ``name``, read once, and their present-cell support."""
    return _support_cells(_read_components(source, name))


def _admit_cells(raw: _RawComponents) -> WZCells:
    """Admit a resolved role on complete retained pairs and one country set.

    General support already rules out repeated cells, so ``N = P*T`` is equivalent to
    every actual pair appearing at every observed date, without a grid or another row
    scan; country coverage at every date follows.  This version limit is applied only
    after structural role disambiguation.
    """
    cells = _support_cells(raw)
    found = cells.found
    if set(found.origins) != set(found.destinations):
        raise unavailable(
            "WZ 0.1 requires the same origin and destination country set; "
            "general unequal-set support is unavailable until 0.2"
        )
    if cells.origin.size != found.pairs * len(found.dates):
        raise unavailable(
            "WZ 0.1 requires each retained pair at every observed date or absent "
            "altogether; partial pair dates and absent country dates are "
            "unavailable until 0.2"
        )
    return cells


def _check_row_labels(labels: Sequence[object]) -> None:
    """Validate row labels in order using passive exact-type checks."""
    for item in labels:
        kind = type(item)
        if kind is not int and kind is not str:
            _label(item)


@dataclass(frozen=True)
class WZLayout:
    """Per-call present-cell indices and whole-call reservation; never admission.

    ``pairs`` is the actual P. ``order`` sorts rows by pair, then date, and
    ``offsets`` (P+1) delimit each pair's present rows. ``first``/``second`` are
    each row's dense present origin-date/destination-date level; ``levels``
    their counts. ``plan`` uses a structural upper bound on the identified
    dimension; ``extra`` is the caller's addition.
    """

    origins: IndexArray
    destinations: IndexArray
    dates: IndexArray
    first: IndexArray
    second: IndexArray
    order: IndexArray
    offsets: IndexArray
    levels: tuple[int, int]
    origin_count: int
    destination_count: int
    periods: int
    pairs: int
    sizes: Sizes
    extra: tuple[int, int]
    plan: Reservation


def reserve_validation(shape: tuple[int, int], bounds: AttemptBounds) -> None:
    """Reserve source validation's code arrays before reading/copying labels.

    Label tuples and dictionaries are Python objects, outside the numeric
    accounting; the codes, pair ids and sort temporaries are about 16n words.
    """
    n, k = shape
    Reservation(128 * n, 64 * n, n, k, 0, 0).require(bounds)


def _seen(values: IndexArray, bound: int) -> NDArray[np.bool_] | None:
    """Which of ``range(bound)`` occur, in one pass; None for too large a table."""
    if not values.size or bound > DIRECT_TABLE * values.size:
        return None
    seen = np.zeros(bound, dtype=np.bool_)
    seen[values] = True
    return seen


def _dense(
    values: IndexArray, bound: int | None = None
) -> tuple[IndexArray, IndexArray]:
    """Present values in increasing order and each row's dense code.

    When the values are known to lie in ``range(bound)`` and that table fits
    within ``DIRECT_TABLE * n``, they are counted instead of sorted; the two
    results are identical arrays.
    """
    seen = None if bound is None else _seen(values, bound)
    if seen is None:
        present, inverse = np.unique(values, return_inverse=True)
        return present.astype(np.intp), inverse.astype(np.intp).reshape(values.shape)
    position = np.cumsum(seen, dtype=np.intp)
    position -= 1
    inverse = position[values]
    del position
    return np.flatnonzero(seen).astype(np.intp, copy=False), inverse


def _bound(levels: tuple[IndexArray, IndexArray], counts: tuple[int, int, int]) -> int:
    """Structural upper bound of the identified nuisance dimension.

    Every origin's constant, every destination's constant and every date's
    (+1 origin, -1 destination) direction is constant inside each pair, so
    rank <= L1 + L2 - (No + Nd + T - components), the components of the graph
    joining each present level's country to its date. On complete-date panels
    this bound is (No+Nd-1)(T-1); disconnected pair support can lower the rank.
    """
    no, nd, tt = counts
    first, second = levels
    u = np.concatenate((first // tt, no + second // tt))
    v = np.concatenate((no + nd + first % tt, no + nd + second % tt))
    components = np.unique(component_labels(no + nd + tt, u, v)).size
    return first.size + second.size - (no + nd + tt - components)


def _sizes(
    shape: tuple[int, int],
    counts: IndexArray,
    levels: tuple[IndexArray, IndexArray],
    roles: tuple[int, int, int],
) -> Sizes:
    no, nd, tt = roles
    multi = counts[counts >= 2].astype(np.int64)
    per_country = np.concatenate(
        (
            np.bincount(levels[0] // tt, minlength=no),
            np.bincount(levels[1] // tt, minlength=nd),
        )
    ).astype(np.int64)
    return Sizes(
        shape[0],
        shape[1],
        int(counts.size),
        int(levels[0].size + levels[1].size),
        _bound(levels, roles),
        tt,
        int(counts.max()),
        int(np.sum(multi**2)),
        int(np.sum(multi**3)),
        int(np.sum(per_country**2)),
        int(np.sum(per_country**3)),
    )


def layout(
    source: WZSource,
    name: str,
    shape: tuple[int, int],
    bounds: AttemptBounds,
    *,
    extra: tuple[int, int] = (0, 0),
    workspace: Workspace | None = None,
) -> WZLayout:
    """Reserve validation first, then the whole present-cell call before snapshots."""
    n, k = shape
    reserve_validation(shape, bounds)
    raw = _read_components(source, name)
    if raw.a.size != n or len(source.row_labels) != n:
        raise unavailable("source labels must align with every current row")
    _check_row_labels(source.row_labels)
    if k < 1:
        raise unavailable("at least one current slope, K>=1, required")
    cells = _admit_cells(raw)
    del raw
    admitted = cells.found
    i, j, t = cells.origin, cells.destination, cells.date
    del cells
    roles = (len(admitted.origins), len(admitted.destinations), len(admitted.dates))
    pair = _dense(i * roles[1] + j, roles[0] * roles[1])[1]
    order = np.lexsort((t, pair)).astype(np.intp)
    counts = np.bincount(pair, minlength=admitted.pairs).astype(np.intp)
    del pair
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.intp)
    first_levels, first = _dense(i * roles[2] + t, roles[0] * roles[2])
    second_levels, second = _dense(j * roles[2] + t, roles[1] * roles[2])
    sizes = _sizes(shape, counts, (first_levels, second_levels), roles)
    added = (0, 0) if workspace is None else workspace(sizes)
    total = (extra[0] + added[0], extra[1] + added[1])
    plan = _plan(sizes, extra_bytes=total[0], extra_work=total[1])
    plan.require(bounds)
    return WZLayout(
        i,
        j,
        t,
        first,
        second,
        order,
        offsets,
        (int(first_levels.size), int(second_levels.size)),
        roles[0],
        roles[1],
        roles[2],
        admitted.pairs,
        sizes,
        total,
        plan,
    )


@dataclass(frozen=True)
class Coordinates:
    """Identified nuisance coordinates of one layout, with their exact origin.

    ``positions`` maps each level of the elimination's column space to its
    coordinate (-1 outside the identified set); ``first``/``second`` are each
    row's coordinates. ``plan`` re-plans the call with the actual dimension
    and charges this call's elimination work.
    """

    digest: str
    positions: IndexStorage
    first: IndexStorage
    second: IndexStorage
    dimension: int
    route: str
    work: int
    plan: Reservation

    @property
    def held(self) -> tuple[str, bytes]:
        """Plain capturable form for the same fit's later evaluations."""
        return self.digest, self.positions.data


def _structure_digest(panel: WZLayout) -> str:
    digest = hashlib.sha256(np.asarray(panel.levels, dtype=np.int64).tobytes())
    for array in (panel.order, panel.offsets, panel.first, panel.second):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _coordinates(
    panel: WZLayout,
    found: tuple[str, IndexArray, IndexArray, IndexArray],
    route: str,
    work: int,
) -> Coordinates:
    digest, positions, first, second = found
    dimension = int(np.count_nonzero(positions >= 0))
    plan = _plan(
        replace(panel.sizes, nuisance=dimension),
        extra_bytes=panel.extra[0],
        extra_work=panel.extra[1] + work,
    )
    return Coordinates(
        digest,
        IndexStorage.store(positions),
        IndexStorage.store(first),
        IndexStorage.store(second),
        dimension,
        route,
        work,
        plan,
    )


def _complete_ends(
    levels: IndexArray, order: IndexArray, dates: IndexArray, count: int
) -> IndexArray | None:
    """Prove each pair's ordered columns are one country's full date block."""
    by_pair = levels[order].reshape(-1, dates.size)
    starts = by_pair[:, 0]
    ends = starts // dates.size
    if (
        int(ends.min()) < 0
        or int(ends.max()) >= count
        or np.any(starts % dates.size)
        or not np.array_equal(by_pair, starts[:, None] + dates)
        or not np.bincount(ends, minlength=count).all()
    ):
        return None
    return ends


def _complete_pivots(panel: WZLayout, budget: int) -> tuple[IndexArray, int] | None:
    """Exact echelon pivots on complete pair dates, including disconnected support.

    Each pair's consecutive dates supply a basis of the date contrast space.
    The full row space is the tensor product of this space with the bipartite
    pair incidence row space. On each connected component, the incidence
    columns have one relation (sum of origins equals sum of destinations),
    and the date columns have one relation (their sum is zero). Consequently
    in country-then-date order the lexicographically first independent columns
    omit the last date of each country and the last destination of each
    component. The remaining incidence and contrast columns are independent
    bases, and every omitted column is spanned by earlier kept columns.
    """
    no, nd, tt = panel.origin_count, panel.destination_count, panel.periods
    width, height = panel.levels
    n, pairs = panel.order.size, panel.pairs
    count = no + nd
    if (
        tt < 2
        or pairs < 1
        or n != pairs * tt
        or panel.levels != (no * tt, nd * tt)
        or max(count, pairs) >= np.iinfo(np.int32).max
        or panel.offsets.size != pairs + 1
        or panel.offsets[0] != 0
        or not np.all(np.diff(panel.offsets) == tt)
    ):
        return None
    # The existing plan reserves linear layout scans and O(n + P + L) index
    # storage. Charge four incidence entries per difference as before. The
    # graph carries only P edges and does no rational entry updates.
    work = 4 * (n - pairs)
    if work > budget:
        raise ValueError(BUDGET)
    dates = np.arange(tt, dtype=np.intp)
    origins = _complete_ends(panel.first, panel.order, dates, no)
    destinations = _complete_ends(panel.second, panel.order, dates, nd)
    if origins is None or destinations is None:
        return None
    if pairs == no * nd and np.all(
        (origins[1:] > origins[:-1])
        | ((origins[1:] == origins[:-1]) & (destinations[1:] > destinations[:-1]))
    ):
        # Distinct sorted edges fill the Cartesian support, hence it is connected.
        last = np.array([count - 1], dtype=np.intp)
    else:
        graph = _csr_matrix(
            (np.ones(pairs, dtype=bool), (origins, no + destinations)),
            shape=(count, count),
        )
        del origins, destinations
        components, membership = _connected_components(graph, directed=False)
        last = np.full(components, -1, dtype=np.intp)
        np.maximum.at(last, membership[no:], np.arange(no, count, dtype=np.intp))
    keep = np.ones((count, tt), dtype=bool)
    keep[:, -1] = False
    keep[last] = False
    kept = np.flatnonzero(keep)
    positions = np.full(width + height, -1, dtype=np.intp)
    positions[kept] = np.arange(kept.size, dtype=np.intp)
    return positions, work


def _eliminate(panel: WZLayout, budget: int) -> tuple[IndexArray, int]:
    """Canonical leading columns of the within-pair differences, exactly.

    Prefer the complete-date proof when the columns have that structure.
    Otherwise, each consecutive pair of present dates (s, r) gives the integer
    row e_first(r) - e_first(s) + e_second(r) - e_second(s) over the present
    levels (first, then second). certificate_basis's exact fully reduced
    elimination (_reduce_by/_new_tail; integers and Fractions, no tolerance)
    keeps every pivot row zero at the other pivots, so the pivots are the
    canonical leading-column set of the row space: the set that the
    pair-cancelled admission's exact quotient selects, for any row order.
    Entry updates are counted; past the explicit work bound the call refuses.
    """
    complete = _complete_pivots(panel, budget)
    if complete is not None:
        return complete
    width = panel.levels[0]
    first = panel.first[panel.order]
    second = width + panel.second[panel.order]
    later = np.ones(panel.order.size, dtype=bool)
    later[panel.offsets[:-1]] = False
    rows = np.flatnonzero(later)
    tails: dict[int, Reduced] = {}
    holders: dict[int, set[int]] = {}
    work = 0
    for a_r, a_s, b_r, b_s in zip(
        first[rows].tolist(),
        first[rows - 1].tolist(),
        second[rows].tolist(),
        second[rows - 1].tolist(),
        strict=True,
    ):
        found: Reduced = {a_r: 1, a_s: -1, b_r: 1, b_s: -1}
        work += 4 + _reduce_by(found, tails)
        if found:
            pivot = min(found)
            lead = found.pop(pivot)
            tail: Reduced = {c: _over(value, lead) for c, value in found.items()}
            work += _new_tail(pivot, tail, tails, holders)
        if work > budget:
            raise ValueError(BUDGET)
    positions = np.full(width + panel.levels[1], -1, dtype=np.intp)
    kept = np.asarray(sorted(tails), dtype=np.intp)
    positions[kept] = np.arange(kept.size, dtype=np.intp)
    return positions, work


def identify(
    panel: WZLayout,
    bounds: AttemptBounds,
    held: tuple[str, bytes] | None = None,
) -> Coordinates:
    """Exact identified coordinates on a joint layout, formed once per fit.

    ``held`` is the same fit's earlier result (WZControls.identified); it is
    used only for the identical structure digest, else exact identification
    runs afresh within the work the plan leaves under the explicit bound.
    """
    digest = _structure_digest(panel)
    width = panel.levels[0]
    if held is not None and held[0] == digest:
        positions = np.frombuffer(held[1], dtype=np.intp)
        if positions.size != width + panel.levels[1]:
            raise ValueError("held WZ coordinates do not match the current levels")
        work = 0
    else:
        budget = bounds.max_work - panel.plan.work
        if budget <= 0:
            raise ValueError(BUDGET)
        positions, work = _eliminate(panel, budget)
    found = (digest, positions, positions[panel.first], positions[width + panel.second])
    return _coordinates(panel, found, "exact_pair_elimination", work)


def quotient_coordinates(
    panel: WZLayout, structure: CancelStructure, kept: tuple[int, ...]
) -> Coordinates:
    """The pair-cancelled fit's own exact quotient columns, without recomputation.

    ``kept`` are the pivot columns of fe.admission_incidence.quotient_rank on the
    same structure: retained columns ordered by block (first, then second
    interaction) and code. The caller has bound the structure's labels.
    """
    width, height = structure.block_sizes
    positions = np.full(width + height, -1, dtype=np.intp)
    chosen = np.asarray(kept, dtype=np.intp)
    positions[chosen] = np.arange(chosen.size, dtype=np.intp)
    codes, inverse = structure.retained_sorted_codes, structure.inverse
    first = positions[codes[0][inverse]]
    second = positions[width + codes[1][inverse]]
    digest = hashlib.sha256(positions.tobytes())
    digest.update(first.tobytes())
    digest.update(second.tobytes())
    found = (digest.hexdigest(), positions, first, second)
    return _coordinates(panel, found, "exact_pair_quotient", 0)


def capture_rows(
    panel: WZLayout,
    x: FloatArray,
    y: FloatArray,
    mu: FloatArray,
    names: tuple[str, ...],
    *,
    route: str,
    coordinates: Coordinates,
) -> Rows:
    """Own finite current numerical inputs after the caller's complete reservation.

    Pair totals use the layout's offsets over present rows; Rows.pairs is P.
    """
    n, k = x.shape
    first, second = coordinates.first.view(), coordinates.second.view()
    if (
        y.shape != (n,)
        or mu.shape != (n,)
        or len(names) != k
        or panel.order.shape != (n,)
        or first.shape != (n,)
    ):
        raise ValueError("WZ current rows/names are not aligned")
    for array in (x, y, mu):
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise FloatingPointError("WZ requires finite float64 current arrays")
    if np.any(mu <= 0) or np.any(y < 0):
        raise unavailable(
            "strictly positive representable fitted means and nonnegative y required"
        )
    if np.any(np.add.reduceat(y[panel.order], panel.offsets[:-1]) <= 0):
        raise unavailable("all pairs must have positive outcome totals; no trimming")
    digest = hashlib.sha256()
    for bound_array in (
        x,
        y,
        mu,
        panel.origins,
        panel.destinations,
        panel.dates,
        panel.order,
        first,
        second,
    ):
        digest.update(np.ascontiguousarray(bound_array).view(np.uint8))
    return Rows(
        FloatStorage.store(x),
        FloatStorage.store(y),
        FloatStorage.store(mu),
        IndexStorage.store(panel.origins),
        IndexStorage.store(panel.destinations),
        IndexStorage.store(panel.dates),
        IndexStorage.store(panel.order),
        IndexStorage.store(panel.offsets),
        coordinates.first,
        coordinates.second,
        panel.origin_count,
        panel.destination_count,
        panel.periods,
        panel.pairs,
        coordinates.dimension,
        names,
        digest.hexdigest(),
        route,
        coordinates.route,
        replace(panel.sizes, nuisance=coordinates.dimension),
    )
