"""Exact incidence coordinates of the fixed effects for the absorbed certificate.

Internal module: the coordinates of the full joint system, fixed from the codes alone,
before any score or curvature.

``D = [D_1 ... D_G]`` holds every level indicator of a grouping; it is never
formed.  The reference-reduced incidence ``E`` keeps every first-block level and omits,
in each later block, the smallest level code of each connected component of all
blocks: the references of :func:`powermean.solve.absorbed_direction.normalize`.  Its
columns are ordered by block, then code; its rows are the code tuples of the rows in
their order.  A reference of block ``k`` in component ``K`` is the sum of the
first-block columns of ``K`` minus the other block-``k`` columns of ``K``: the integer
map ``M`` with ``D = E M``.

Kept columns ``J``.  Where the exact indicator rank equals the columns of ``E``
(always for one and two blocks) ``J`` is every column (route ``connectivity``).
Otherwise (``elimination``) the distinct code rows are reduced by exact integer row
elimination, never a dummy matrix: each existing pivot present in a row, in increasing
column order, is eliminated by ``v <- u[p] v - v[p] u``, after which the entries are
divided by their positive gcd and the leading entry made positive; a nonzero remainder
becomes the pivot of its smallest column, an exact zero row is discarded.  Python
integers cannot overflow.  ``R_J`` is upper triangular with a nonzero diagonal, so
``E_J`` has full column rank, and exact rational back substitution ``R_J t = R_c``
gives every dependent column ``E_c = E_J t``.

Witnesses.  Every ORIGINAL level column ``d`` has a canonical rational ``w_d``
(positive reduced denominators) with ``D_d = E_J w_d``: a unit vector, a dependent
column's ``t``, or ``M``'s combination for a reference.  Each witness is checked
exactly on every distinct code row (``ValueError`` otherwise), so the span covers every
original column; the public rank bounds are kept beside the exact rank.

Positive support (:func:`positive_support`): the SAME ``J`` on the rows of positive
curvature, by the same exact elimination on the columns of ``J``; a rank below ``|J|``
refuses the nuisance curvature and never shrinks the model.  Where every row is
positive the rows are the structural rows, of rank ``|J|`` by construction.
:func:`embed` places the coefficients of ``J`` at their levels, exact zeros elsewhere;
:func:`lift` gathers them by row in block order with checked sums.  No effect is
overwritten and no unique three-plus-block effect is claimed.

Owned construction (:func:`owned_basis`).
The basis is a deterministic function of what it reads of the grouping: the blocks in
order, the codes of each and its number of levels.  An owner (the admitted sample)
keeps, as a private attribute, the values of the one construction made for its
grouping: that grouping's ``id``, its content (those inputs with the block names and
row positions, as bytes and integers) and every field of the basis, arrays as bytes,
the rank as its integers, in a tuple of immutable values that no attribute assignment
changes.  It is used again only when fresh validation against the current grouping
proves the held values are the exact canonical construction on every field, and then as
a new basis with arrays of its own; anything else is constructed by
:func:`certificate_basis`, as is every grouping without an owner.

The owner's one construction is first proposed (:func:`_proposal`): the same steps,
with the echelon found by elimination against fully reduced rows of the later blocks
instead of chains of pivots, and the dependent witnesses read from those rows.  A
proposal has no standing of its own: it is used only when the same fresh validation
admits its values as the exact canonical construction; otherwise, or past its work
budget, :func:`certificate_basis` constructs (or refuses) as before.

Within-fit admission (:func:`retention_basis`).  An owned construction can also
carry a private reader issued here only after the canonical validation above.
Its closure holds only the immutable construction and weak identity scopes, not
a live sample or provider graph.  The reader is authenticated by its package code,
not by a marker, dataclass, digest or write flag.  A fresh complete grouping-content
comparison is required at consumption, before the caller compares every supplied
basis field with the rebuilt expectation.  Only exact built-in grouping classes
are eligible; all other inputs keep :func:`owned_basis` and its original checks.
The assessment handoff explicitly transfers this reader across its already
value-bound replica/original boundary (:func:`rebound_basis`).  It is not stored
on the sample, does not survive detached exports, and never saves row, support,
curvature, scale or inference checks.  As for that handoff, interpreter attacks
on closure cells/code and concurrent writes during admission are outside scope.
"""

from __future__ import annotations

import heapq
import importlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from types import CodeType, FunctionType, ModuleType
from typing import Literal, NamedTuple, cast
from weakref import ReferenceType, ref

import numpy as np
from numpy.typing import NDArray

from powermean.data.accounting import frozen_index
from powermean.data.sample_fe_source import Derived
from powermean.data.sample_fe_state import FESample
from powermean.fe.absorb_numeric import checked_sum
from powermean.fe.certificate_witness import (
    FactoredRows, HeadRecipe, HeadRows, OperationalBasis, PackedHeads, RetainedRows, SparseValues,
    WitnessBlock, WitnessContext, WitnessProof,
)
from powermean.fe import certificate_witness as _witness_values
from powermean.fe import connectivity as _connectivity
from powermean.fe import groups as _groups
from powermean.fe.connectivity import (
    Components,
    IndicatorRank,
    _indicator_rank_prepared,
    _indicator_rank_counted,
    components,
)
from powermean.fe.connectivity import indicator_rank as accepted_rank
from powermean.fe.groups import Block, Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray

Route = Literal["connectivity", "elimination"]
BoolArray = NDArray[np.bool_]
Row = dict[int, int]  # a sparse integer row: column -> nonzero integer
Witness = tuple[tuple[int, int, int], ...]  # (position in J, numerator, denominator)
Rational = dict[int, int | Fraction]  # exact: an integral value may be an int
SPAN = "the span witnesses do not reproduce the incidence of the grouping"


NATIVE_ECHELON_MODULE_NAME = "powermean._native.basis._echelon_cpp"


def _resolve_native_echelon() -> ModuleType:
    """Resolve only an explicitly requested compiled backend.

    Absence, broken imports and missing entry points are errors. Only an actual
    representation/work decline or optional allocation failure runs Python.
    """
    module = importlib.import_module(NATIVE_ECHELON_MODULE_NAME)
    if type(module) is not ModuleType:
        raise ImportError("the exact basis backend must be a module")
    for name in ("attempt_from_mapped", "audit_from_mapped", "backend_info"):
        if not callable(vars(module).get(name)):
            raise ImportError(f"the exact basis backend lacks {name}")
    return module


@dataclass(frozen=True, eq=False)
class CertificateBasis:
    """The kept incidence coordinates of one grouping, with their exact evidence.

    ``references`` per later block, the reference code of each component; ``columns``
    per block, the ``E`` column of each level (``-1`` at a reference); ``selected``
    ``J``, increasing; ``positions`` per block, each level's coordinate in ``J``
    (``-1`` at a reference or dependent level); ``witnesses`` per block and level;
    ``echelon`` the integer rows by pivot (route ``elimination`` only); ``rows`` the
    distinct code rows checked; ``rank`` the accepted public bounds.
    """

    grouping: Grouping
    route: Route
    rank: IndicatorRank
    references: tuple[IndexArray, ...]
    columns: tuple[IndexArray, ...]
    selected: IndexArray
    positions: tuple[IndexArray, ...]
    witnesses: tuple[tuple[Witness, ...], ...]
    echelon: tuple[tuple[tuple[int, int], ...], ...]
    rows: int

    def __getattribute__(self, name: str) -> object:
        """Keep full tuple-valued public access over private implicit unit storage."""
        found: object = object.__getattribute__(self, name)
        if name in ("witnesses", "echelon") and type(found) is _OPERATIONAL_DATA_CLASS:
            witnesses, echelon = found.audit(_OPERATIONAL_AUDIT)
            return witnesses if name == "witnesses" else echelon
        if name == "witnesses" and type(found) is WitnessProof:
            return found.materialize()
        if name == "echelon" and type(found) is WitnessProof:
            return found.materialize_echelon()
        return found

    @property
    def n_incidence(self) -> int:
        """The number of columns of ``E``."""
        return _count(self.columns)

    @property
    def n_kept(self) -> int:
        """``|J|``, the nuisance coordinates of the certificate."""
        return int(self.selected.size)


@dataclass(frozen=True)
class SupportCheck:
    """The exact rank of ``E_J`` on the rows of positive curvature.

    ``rows`` are those rows, ``distinct`` their distinct code rows; ``eliminated``
    says whether an elimination ran (not when every row, or no row, is positive).
    """

    rows: int
    distinct: int
    rank: int
    kept: int
    eliminated: bool

    @property
    def full(self) -> bool:
        """Whether the kept coordinates keep their rank on the positive rows."""
        return self.rank == self.kept


# ------------------------------------------------------------------ the incidence


def _references(grouping: Grouping, parts: Components) -> tuple[IndexArray, ...]:
    """The smallest level code of each component in each later block (normalize's)."""
    found: list[IndexArray] = []
    for k in range(1, len(grouping.blocks)):
        count = grouping.blocks[k].n_levels
        refs = np.full(parts.n_components, count, dtype=np.intp)
        np.minimum.at(refs, parts.of_levels[k], np.arange(count, dtype=np.intp))
        found.append(frozen_index(refs))
    return tuple(found)


def _columns(
    grouping: Grouping, references: tuple[IndexArray, ...]
) -> tuple[IndexArray, ...]:
    """The ``E`` column of every level, by block then code; ``-1`` at a reference."""
    found: list[IndexArray] = []
    offset = 0
    for k, block in enumerate(grouping.blocks):
        kept = np.ones(block.n_levels, dtype=bool)
        if k:
            kept[references[k - 1]] = False
        column = np.full(block.n_levels, -1, dtype=np.intp)
        column[kept] = offset + np.arange(int(kept.sum()), dtype=np.intp)
        offset += int(kept.sum())
        found.append(frozen_index(column))
    return tuple(found)


def _count(columns: tuple[IndexArray, ...]) -> int:
    """The number of columns of ``E``."""
    return sum(int(np.count_nonzero(c >= 0)) for c in columns)


def _distinct(grouping: Grouping, mask: BoolArray | None = None) -> IndexArray:
    """The distinct code tuples of the rows (``mask`` selects rows), in row order."""
    codes = np.column_stack([block.codes for block in grouping.blocks])
    if mask is not None:
        codes = codes[mask]
    _, first = np.unique(codes, axis=0, return_index=True)
    found: IndexArray = codes[np.sort(first)]
    return found


def _first_rows(grouping: Grouping) -> IndexArray:
    """First source index of each exact code tuple, in original row order.

    Sort indices over the existing code arrays, not an N-by-G raw-code table.
    The minimum index in each equal-key run is explicit, independent of sort
    stability. Sorting and comparison temporaries die before mapped rows exist.
    """
    order = np.lexsort(tuple(block.codes for block in grouping.blocks))
    if not order.size:
        return order
    boundary = np.zeros(order.size, dtype=bool)
    boundary[0] = True
    for block in grouping.blocks:
        values = block.codes[order]
        boundary[1:] |= values[1:] != values[:-1]
        del values
    first = np.minimum.reduceat(order, np.flatnonzero(boundary))
    first.sort()
    return first


def _row_columns(row: IndexArray, columns: tuple[IndexArray, ...]) -> list[int]:
    """The columns of one code row: each block's column of its level, if any."""
    found = [int(column[code]) for column, code in zip(columns, row, strict=True)]
    return [c for c in found if c >= 0]


# ----------------------------------------------------------------- elimination


def _divide(row: Row, divisor: int) -> None:
    """Divide every entry of ``row``, in place, by the exact ``divisor``."""
    for c in row:
        row[c] //= divisor


def _reduced(row: Row, pivots: dict[int, Row], *, _primitives: tuple | None = None) -> Row:
    """``row`` with every existing pivot eliminated, in increasing column order.

    A copy of ``row`` is updated in place.  The heap holds every present pivot
    column (and columns since cancelled, skipped when popped), so each pop is the
    smallest present pivot.  ``v <- u[p] v - v[p] u`` changes only the columns of
    ``u`` (all of ``v`` when ``u[p] != 1``); after each elimination a nonzero row is
    divided by its gcd, signed to make the leading entry positive, unless that
    divisor is one.  An entry ``1`` or ``-1`` of the updated row proves the gcd one,
    which is then not computed.
    """
    row = dict(row)
    values = row.values()
    present = [c for c in row if c in pivots]
    (heapq.heapify if _primitives is None else _primitives[1])(present)
    while present:
        p = (heapq.heappop if _primitives is None else _primitives[2])(present)
        b = row.get(p)
        if b is None:
            continue
        u = pivots[p]
        a = u[p]
        if a != 1:
            for c in row:
                row[c] *= a
        for c, value in u.items():
            old = row.get(c)
            if old is None:
                row[c] = -b * value
                if c in pivots:
                    (heapq.heappush if _primitives is None else _primitives[3])(present, c)
            elif new := old - b * value:
                row[c] = new
            else:
                del row[c]
        if not row:
            return row
        divisor = (1 if 1 in values or -1 in values else
                   (math.gcd if _primitives is None else _primitives[0])(*values))
        if row[min(row)] < 0:
            divisor = -divisor
        if divisor != 1:
            (_divide if _primitives is None else _primitives[4])(row, divisor)
    return row


def eliminate(rows: Iterable[list[int]]) -> tuple[dict[int, Row], int]:
    """The integer echelon of 0/1 rows given by their columns, and the rows seen."""
    pivots: dict[int, Row] = {}
    seen = 0
    for columns in rows:
        seen += 1
        row = _reduced(dict.fromkeys(columns, 1), pivots)
        if row:
            pivots[min(row)] = row
    return pivots, seen


def _dependent(pivots: dict[int, Row], kept: list[int], column: int,
               *, _fraction=None) -> Rational:
    """``t`` with ``R_J t = R_column``, by exact back substitution over ``kept``."""
    t: Rational = {}
    for p in reversed(kept):
        row = pivots[p]
        rhs = (Fraction if _fraction is None else _fraction)(row.get(column, 0))
        for j, value in row.items():
            if j != p and j in t:
                rhs -= value * t[j]
        if rhs:
            t[p] = rhs / row[p]
    return t


def _kept(
    columns: tuple[IndexArray, ...], rank: IndicatorRank, distinct: IndexArray
) -> tuple[Route, dict[int, Row], list[int]]:
    """The route, the echelon (empty on the connectivity route) and ``J``."""
    count = _count(columns)
    if rank.exact == count:
        return "connectivity", {}, list(range(count))
    pivots, _ = eliminate(_row_columns(row, columns) for row in distinct)
    return "elimination", pivots, sorted(pivots)


# --------------------------------------------------------------- the witnesses


def _canonical(found: Rational) -> Witness:
    """A witness as sorted ``(position, numerator, denominator)`` of nonzero entries."""
    return tuple(
        (p, value.numerator, value.denominator)
        for p, value in sorted(found.items())
        if value
    )


def _column_witnesses(
    pivots: dict[int, Row], kept: list[int], count: int, *, _audit: tuple | None = None,
) -> dict[int, Rational]:
    """Each ``E`` column in ``J`` coordinates: a unit, or a dependent column's ``t``."""
    place = {c: i for i, c in enumerate(kept)}
    found: dict[int, Rational] = {}
    for c in range(count):
        if c in place:
            found[c] = {place[c]: (Fraction if _audit is None else _audit[1])(1)}
        else:
            t = (_dependent(pivots, kept, c) if _audit is None else
                 _audit[0](pivots, kept, c, _fraction=_audit[1]))
            found[c] = {place[j]: value for j, value in t.items()}
    return found


def _reference(terms: list[tuple[int, Rational]], *, _canonical_value=None) -> Witness:
    """``sum_i m_i w_i`` of integer multiples of rational witnesses, canonical."""
    total: Rational = {}
    for multiple, witness in terms:
        for p, value in witness.items():
            total[p] = total.get(p, 0) + multiple * value
    return (_canonical if _canonical_value is None else _canonical_value)(total)


def _witnesses(
    grouping: Grouping,
    parts: Components,
    columns: tuple[IndexArray, ...],
    of_column: dict[int, Rational],
    *, _audit: tuple | None = None,
) -> tuple[tuple[Witness, ...], ...]:
    """The witness of every original level: its column's, or ``M``'s for a reference."""
    first, found = (parts.of_levels[0] if _audit is None else _audit[0][0]), []
    for k, block in enumerate(grouping.blocks if _audit is None else range(len(columns))):
        mine: list[Witness] = []
        for level in range(block.n_levels if _audit is None else _audit[1][k]):
            column = int(columns[k][level])
            if column >= 0:
                mine.append((_canonical if _audit is None else _audit[3])(of_column[column]))
                continue
            component = (parts.of_levels[k][level] if _audit is None
                         else _audit[0][k][level])
            heads = (np.flatnonzero if _audit is None else _audit[2])(first == component).tolist()
            same = (np.flatnonzero if _audit is None else _audit[2])(
                (parts.of_levels[k] if _audit is None else _audit[0][k]) == component
            ).tolist()
            terms = [(1, of_column[int(columns[0][h])]) for h in heads]
            terms += [(-1, of_column[int(columns[k][s])]) for s in same if s != level]
            mine.append(_reference(terms) if _audit is None else
                        _audit[4](terms, _canonical_value=_audit[3]))
        found.append(tuple(mine))
    return tuple(found)


def _verified(
    positions: tuple[IndexArray, ...],
    witnesses: tuple[tuple[Witness, ...], ...],
    distinct: IndexArray,
) -> None:
    """``D_d = E_J w_d`` for every level ``d`` on every distinct code row, exactly."""
    inverse: dict[int, list[tuple[tuple[int, int], Fraction]]] = {}
    for k, levels in enumerate(witnesses):
        for level, witness in enumerate(levels):
            for p, num, den in witness:
                inverse.setdefault(p, []).append(((k, level), Fraction(num, den)))
    for row in distinct:
        sums: dict[tuple[int, int], Fraction] = {}
        for k, code in enumerate(row):
            for target, value in inverse.get(int(positions[k][code]), []):
                sums[target] = sums.get(target, Fraction(0)) + value
        own = {(k, int(code)) for k, code in enumerate(row)}
        if any(sums.get(target, 0) != 1 for target in own) or any(
            value != 0 for target, value in sums.items() if target not in own
        ):
            raise ValueError(SPAN)


# ------------------------------------------------------------------- the basis


def certificate_basis(grouping: object) -> CertificateBasis:
    """The kept incidence coordinates of ``grouping`` and their exact witnesses.

    Parameters
    ----------
    grouping : Grouping
        The accepted grouping of the admitted sample.

    Returns
    -------
    CertificateBasis
        References, ``E`` columns, ``J``, positions, witnesses and echelon.

    Raises
    ------
    TypeError
        ``grouping`` is not a :class:`~powermean.fe.groups.Grouping`.
    ValueError
        The exact rank leaves the public bounds, or a witness fails its exact check.
    """
    if not isinstance(grouping, Grouping):
        raise TypeError("grouping must be a Grouping")
    parts = components(grouping.blocks)
    rank = accepted_rank(grouping.blocks)
    references = _references(grouping, parts)
    columns = _columns(grouping, references)
    distinct = _distinct(grouping)
    route, pivots, kept = _kept(columns, rank, distinct)
    if not rank.lower <= len(kept) <= rank.upper:
        raise ValueError("the exact rank leaves the accepted indicator rank bounds")
    place = {c: i for i, c in enumerate(kept)}
    positions = tuple(
        frozen_index(np.array([place.get(int(c), -1) for c in col], dtype=np.intp))
        for col in columns
    )
    of_column = _column_witnesses(pivots, kept, _count(columns))
    witnesses = _witnesses(grouping, parts, columns, of_column)
    _verified(positions, witnesses, distinct)
    return CertificateBasis(
        grouping=grouping,
        route=route,
        rank=rank,
        references=references,
        columns=columns,
        selected=frozen_index(np.asarray(kept, dtype=np.intp)),
        positions=positions,
        witnesses=witnesses,
        echelon=tuple(tuple(sorted(pivots[p].items())) for p in sorted(pivots)),
        rows=int(distinct.shape[0]),
    )


# ------------------------------------------------------- the owned construction

HELD = "_certificate_basis"  # the owner's private attribute
CONSTRUCTION = "_fit_basis_construction"  # never a retained/exported field
Content = tuple[bytes | tuple[str, bytes, bytes], ...]


class _Held(NamedTuple):
    """One construction for one grouping, as immutable values (module docstring)."""

    grouping: int
    content: Content
    route: Route
    rank: tuple[tuple[int, ...], int, tuple[tuple[int, int, int], ...], int, int]
    references: tuple[bytes, ...]
    columns: tuple[bytes, ...]
    selected: bytes
    positions: tuple[bytes, ...]
    witnesses: tuple[tuple[Witness, ...], ...]
    echelon: tuple[tuple[tuple[int, int], ...], ...]
    rows: int


def _index_bytes(array: object) -> bytes | None:
    """The bytes of a one-dimensional index array, ``None`` for anything else."""
    if type(array) is not np.ndarray or array.dtype != np.intp or array.ndim != 1:
        return None
    return array.tobytes()


def _content(grouping: object) -> Content | None:
    """What the construction reads of ``grouping``, with names and positions, as values.

    The positions, then per block its name, codes and counts.  ``None`` unless a
    :class:`Grouping` (the sample's derived blocks included) whose blocks and positions
    are its own attributes, a tuple of :class:`Block` records with string names and
    one-dimensional index arrays: such a grouping is constructed, never reused.
    """
    own = getattr(grouping, "__dict__", None)
    if not isinstance(grouping, Grouping) or type(own) is not dict:
        return None
    blocks = own.get("blocks")
    if type(blocks) is not tuple or blocks is not grouping.blocks:
        return None
    if own.get("positions") is not grouping.positions:
        return None
    positions = _index_bytes(grouping.positions)
    found: list[bytes | tuple[str, bytes, bytes]] = []
    for block in blocks:
        if type(block) is not Block or type(block.name) is not str:
            return None
        codes, counts = _index_bytes(block.codes), _index_bytes(block.counts)
        if codes is None or counts is None:
            return None
        found.append((block.name, codes, counts))
    return None if positions is None else (positions, *found)


def _rank_values(
    rank: IndicatorRank,
) -> tuple[tuple[int, ...], int, tuple[tuple[int, int, int], ...], int, int]:
    """The accepted rank bounds and counts as a tuple of exact integers."""
    pairwise = tuple((int(i), int(j), int(c)) for i, j, c in rank.pairwise_components)
    levels = tuple(int(n) for n in rank.levels)
    return (levels, int(rank.n_components), pairwise, int(rank.lower), int(rank.upper))


def _held(basis: CertificateBasis, grouping: int, content: Content) -> _Held:
    """The values of a basis just constructed: arrays as bytes, the rank's integers."""
    return _Held(
        grouping,
        content,
        basis.route,
        _rank_values(basis.rank),
        tuple(a.tobytes() for a in basis.references),
        tuple(a.tobytes() for a in basis.columns),
        basis.selected.tobytes(),
        tuple(a.tobytes() for a in basis.positions),
        basis.witnesses,
        basis.echelon,
        int(basis.rows),
    )


def _same_structure(a: object, b: object) -> bool:
    """Whether ``a`` and ``b`` have the exact same types, tuple arities and values."""
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _same_structure(x, y) for x, y in zip(a, b, strict=True)
        )
    return bool(a == b)


def _parsed_echelon_row(row_tuple: object, count: int) -> tuple[int, Row] | None:
    """One primitive positive-leading integer row of ``echelon``, or ``None``."""
    if type(row_tuple) is not tuple or not row_tuple:
        return None
    row: Row = {}
    prev_c = -1
    for pair in row_tuple:
        if type(pair) is not tuple or len(pair) != 2:
            return None
        c, val = pair
        if (
            type(c) is not int
            or type(val) is not int
            or val == 0
            or not prev_c < c < count
        ):
            return None
        prev_c = c
        row[c] = val
    p = row_tuple[0][0]
    values = row.values()
    coprime = 1 in values or -1 in values or math.gcd(*values) == 1
    return (p, row) if row[p] > 0 and coprime else None


def _parsed_echelon(
    echelon: object, count: int
) -> tuple[dict[int, Row], list[int]] | None:
    """The primitive positive-leading integer rows of ``echelon``, or ``None``."""
    if type(echelon) is not tuple:
        return None
    pivots: dict[int, Row] = {}
    kept: list[int] = []
    prev_p = -1
    for row_tuple in echelon:
        parsed = _parsed_echelon_row(row_tuple, count)
        if parsed is None or not prev_p < parsed[0] < count:
            return None
        prev_p, row = parsed
        pivots[prev_p] = row
        kept.append(prev_p)
    return pivots, kept


def _parsed_dependent_witness(
    witness: object, n_kept: int
) -> tuple[Rational, int, list[tuple[int, int]]] | None:
    """The rational map, common denominator and scaled entries of ``witness``."""
    if type(witness) is not tuple:
        return None
    rat: Rational = {}
    dens: list[int] = []
    entries: list[tuple[int, int, int]] = []
    prev_idx = -1
    for item in witness:
        if type(item) is not tuple or len(item) != 3:
            return None
        idx, num, den = item
        ints = type(idx) is int and type(num) is int and type(den) is int
        if (
            not ints
            or not prev_idx < idx < n_kept
            or num == 0
            or den <= 0
            or math.gcd(num, den) != 1
        ):
            return None
        prev_idx = idx
        rat[idx] = num if den == 1 else Fraction(num, den)
        dens.append(den)
        entries.append((idx, num, den))
    lcm_d = math.lcm(*dens) if dens else 1
    scaled = [(idx, num * (lcm_d // den)) for idx, num, den in entries]
    return rat, lcm_d, scaled


def _echelon_columns(
    pivots: dict[int, Row], kept: list[int]
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, dict[int, int]]]:
    """Split ``pivots`` entries into kept columns ``R_J`` and dependent ``R_c``."""
    col_rows: dict[int, list[tuple[int, int]]] = {p: [] for p in kept}
    dep_rhs: dict[int, dict[int, int]] = {}
    for p, row in pivots.items():
        for c, val in row.items():
            if c in col_rows:
                col_rows[c].append((p, val))
            else:
                dep_rhs.setdefault(c, {})[p] = val
    return col_rows, dep_rhs


def _valid_reference_witnesses(
    witnesses: tuple[tuple[Witness, ...], ...],
    parts: Components,
    columns: tuple[IndexArray, ...],
    of_column: dict[int, Rational],
) -> bool:
    """Check that every reference level witness equals ``M``'s combination."""
    first = parts.of_levels[0]
    for k, w_blk in enumerate(witnesses):
        for level, w in enumerate(w_blk):
            if int(columns[k][level]) >= 0:
                continue
            comp = parts.of_levels[k][level]
            heads = np.flatnonzero(first == comp).tolist()
            same = np.flatnonzero(parts.of_levels[k] == comp).tolist()
            terms = [(1, of_column[int(columns[0][h])]) for h in heads]
            terms += [(-1, of_column[int(columns[k][s])]) for s in same if s != level]
            if not _same_structure(w, _reference(terms)):
                return False
    return True


def _check_dependent_column(
    w: object,
    c: int,
    kept: list[int],
    col_rows: dict[int, list[tuple[int, int]]],
    dep_rhs: dict[int, dict[int, int]],
) -> tuple[Rational, int, list[tuple[int, int]]] | None:
    """Validate one dependent column witness ``w`` against ``R_J t = R_c``."""
    parsed = _parsed_dependent_witness(w, len(kept))
    if parsed is None:
        return None
    rat, lcm_d, scaled = parsed
    lhs: dict[int, int] = {}
    for idx, s_val in scaled:
        for p, r_val in col_rows[kept[idx]]:
            lhs[p] = lhs.get(p, 0) + r_val * s_val
    if {p: v for p, v in lhs.items() if v} != {
        p: v * lcm_d for p, v in dep_rhs.get(c, {}).items()
    }:
        return None
    return rat, lcm_d, scaled


def _valid_witnesses(
    witnesses: object,
    grouping: Grouping,
    parts: Components,
    columns: tuple[IndexArray, ...],
    *,
    pivots: dict[int, Row],
    kept: list[int],
    place: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, int]] | None:
    """Check canonical unit, ``R_J t = R_c`` and reference witnesses."""
    if type(witnesses) is not tuple or len(witnesses) != len(grouping.blocks):
        return None
    col_rows, dep_rhs = _echelon_columns(pivots, kept)
    of_column: dict[int, Rational] = {}
    dep_inv: dict[int, list[tuple[int, int]]] = {}
    dep_scale: dict[int, int] = {}
    for k, (block, w_blk) in enumerate(zip(grouping.blocks, witnesses, strict=True)):
        if type(w_blk) is not tuple or len(w_blk) != block.n_levels:
            return None
        for level, w in enumerate(w_blk):
            c = int(columns[k][level])
            if c < 0:
                continue
            if c in place:
                if not _same_structure(w, ((place[c], 1, 1),)):
                    return None
                of_column[c] = {place[c]: 1}
                continue
            checked = _check_dependent_column(w, c, kept, col_rows, dep_rhs)
            if checked is None:
                return None
            rat, lcm_d, scaled = checked
            for idx, s_val in scaled:
                dep_inv.setdefault(kept[idx], []).append((c, s_val))
            dep_scale[c] = lcm_d
            of_column[c] = rat
    if not _valid_reference_witnesses(witnesses, parts, columns, of_column):
        return None
    return dep_inv, dep_scale


Part = list[tuple[int, int]]  # a column's (dependent column, scaled entry) pairs
Seen = dict[int, dict[int, int | Fraction]]  # a seen pivot -> its reduced kept row


def _dependent_parts(
    count: int, dep_inv: dict[int, Part], dep_scale: dict[int, int]
) -> list[Part | None]:
    """Each ``E`` column's part of a row's scaled sums at the dependent columns.

    A kept column ``j`` adds ``lcm_c t_c[j]`` at every dependent ``c`` (``dep_inv``),
    a dependent column ``c`` adds ``-lcm_c`` at itself, and a reference (index
    ``-1``, the last) nothing: ``E_c = E_J t_c`` on a row is a zero sum of its parts.
    """
    parts: list[Part | None] = [None] * (count + 1)
    for j, entries in dep_inv.items():
        parts[j] = entries
    for c, scale in dep_scale.items():
        parts[c] = [(c, -scale)]
    return parts


def _spanned(found: list[list[int]], parts: list[Part | None]) -> bool:
    """Whether the parts of every row sum to zero: ``E_c = E_J t_c``, row by row.

    The parts before a row's last nonempty part are added, exact zeros omitted;
    their sum must equal that last part negated (formed once per column).
    """
    negated: dict[int, dict[int, int]] = {}
    for row in zip(*found, strict=True):
        total: dict[int, int] | None = None
        last: Part | None = None
        at = -1
        for c in row:
            part = parts[c]
            if not part:
                continue
            if last is not None and total is None:
                total = dict(last)
            elif last is not None and total is not None:
                for d, value in last:
                    if new := total.get(d, 0) + value:
                        total[d] = new
                    else:
                        total.pop(d, None)
            last, at = part, c
        if last is not None:
            if at not in negated:
                negated[at] = {d: -value for d, value in last}
            if total != negated[at]:
                return False
    return True


def _current(
    q: int, reduced: Seen, stamps: dict[int, int]
) -> dict[int, int | Fraction]:
    """The seen pivot ``q``'s row reduced against every seen row, on unseen columns.

    A stored row is exact as of its stamp, the number of pivots then seen.  Within
    more seen pivots, a reduced row is the stored one less, at each pivot seen
    since, its entry there times that pivot's own current reduced row.  A reduced
    row holds columns beyond its pivot only, so the walk ends; an empty row is
    exact for good.
    """
    now = len(reduced)
    stack = [q]
    while stack:
        x = stack[-1]
        row = reduced[x]
        if not row or stamps[x] == now:
            stack.pop()
            continue
        hits = [h for h in row if h in reduced]
        stale = [h for h in hits if reduced[h] and stamps[h] != now]
        if stale:
            stack += stale
            continue
        for h in hits:
            factor = row.pop(h)
            for c, value in reduced[h].items():
                if new := row.get(c, 0) - factor * value:
                    row[c] = new
                else:
                    row.pop(c, None)
        stamps[x] = now
        stack.pop()
    return reduced[q]


def _sees(
    part: dict[int, int | Fraction],
    pivots: dict[int, Row],
    place: dict[int, int],
    reduced: Seen,
    stamps: dict[int, int],
) -> bool:
    """Sees the pivot of a row's reduced kept part, if any; ``False`` refuses.

    A zero part sees nothing.  Otherwise the part must be a multiple of the held row
    of its smallest column ``p`` on the kept columns (the same support, entries in
    the same ratio); ``p`` is then seen with that held row over its leading entry,
    ``p`` itself omitted: no seen column, since the part has none.
    """
    nonzero = {c: value for c, value in part.items() if value}
    if not nonzero:
        return True
    p = min(nonzero)
    row = pivots[p]
    lead, own = row[p], nonzero[p]
    held = 0
    for c, value in row.items():
        if c in place:
            held += 1
            if nonzero.get(c, 0) * lead != value * own:
                return False
    if held != len(nonzero):
        return False
    reduced[p] = {
        c: value if lead == 1 else Fraction(value, lead)
        for c, value in row.items()
        if c != p and c in place
    }
    stamps[p] = len(reduced)
    return True


def _ordered(
    pivots: dict[int, Row], place: dict[int, int], found: list[list[int]]
) -> bool:
    """Whether the rows' kept columns, in row order, see exactly every held pivot.

    With ``S`` the pivots seen and ``U`` the others, the seen rows reduced among
    themselves, ``G = R_SS^-1 R_SU`` (:func:`_current`), reduce a row's kept part to
    ``v_U - v_S G``, free of ``S``.  While no held row of ``U`` holds a column of
    ``S``, the row's coordinates ``v_J R_J^-1`` on ``U`` are that part times
    ``R_UU^-1``: at most one nonzero, at ``p``, exactly when the part is zero or a
    multiple of the held row of its smallest column ``p`` (:func:`_sees`), whose
    row then holds no column of ``S``.  Once a held row of ``U`` holds a seen
    column it is no such multiple and is never seen, so the echelon is refused, as
    it must be: that row holds a pivot seen before it.
    """
    reduced: Seen = {}
    stamps: dict[int, int] = {}
    for row in zip(*found, strict=True):
        part: dict[int, int | Fraction] = {}
        for c in row:
            if c < 0:
                continue
            tail = reduced.get(c)
            if tail is None:
                part[c] = part.get(c, 0) + 1
            elif tail:
                if stamps[c] != len(reduced):
                    tail = _current(c, reduced, stamps)
                for d, value in tail.items():
                    part[d] = part.get(d, 0) - value
        if part and not _sees(part, pivots, place, reduced, stamps):
            return False
    return len(reduced) == len(pivots)


def _valid_elimination(
    pivots: dict[int, Row],
    kept: list[int],
    place: dict[int, int],
    columns: tuple[IndexArray, ...],
    *,
    distinct: IndexArray,
    dep_inv: dict[int, Part],
    dep_scale: dict[int, int],
) -> bool:
    """Check that ``distinct`` in row order produces ``pivots`` and spans ``E``.

    For checked rows (primitive, leading entry positive, increasing pivots ``J``)
    and dependent witnesses (``R_J t_c = R_c``) this holds exactly when ``pivots``
    is :func:`eliminate`'s echelon of the rows.  Every row's dependent columns are
    its kept ones by ``t`` (:func:`_spanned`), so each row ``v`` is ``a R`` with
    ``a = v_J R_J^-1``.  In row order, ``a`` has at most one nonzero at the pivots
    not yet seen, and the pivot of a row with one is then seen, its held row free
    of the pivots seen before (:func:`_ordered`); at the end every pivot is seen.
    These are the rows on which :func:`eliminate` adds a pivot, with its rows:
    each is the primitive, positive-leading remainder of its code row.
    """
    count = _count(columns)
    found = [col[distinct[:, k]] for k, col in enumerate(columns)]
    parts = _dependent_parts(count, dep_inv, dep_scale)
    if not _spanned([codes.tolist() for codes in found], parts):
        return False
    is_kept: BoolArray = np.zeros(count + 1, dtype=bool)
    is_kept[kept] = True
    return _ordered(
        pivots, place, [np.where(is_kept[codes], codes, -1).tolist() for codes in found]
    )


def _valid_kept(
    held: _Held, rank: IndicatorRank, columns: tuple[IndexArray, ...], count: int
) -> tuple[dict[int, Row], list[int], dict[int, int]] | None:
    """Validate ``route``, ``echelon``, ``selected`` and ``positions``."""
    if rank.exact == count:
        if not _same_structure((held.route, held.echelon), ("connectivity", ())):
            return None
        pivots: dict[int, Row] = {}
        kept = list(range(count))
    else:
        if not _same_structure(held.route, "elimination"):
            return None
        parsed = _parsed_echelon(held.echelon, count)
        if parsed is None:
            return None
        pivots, kept = parsed
    if not rank.lower <= len(kept) <= rank.upper:
        return None
    expected_selected = np.asarray(kept, dtype=np.intp).tobytes()
    if type(held.selected) is not bytes or held.selected != expected_selected:
        return None
    place = {c: i for i, c in enumerate(kept)}
    expected_positions = tuple(
        np.array([place.get(int(c), -1) for c in col], dtype=np.intp).tobytes()
        for col in columns
    )
    if not _same_structure(held.positions, expected_positions):
        return None
    return pivots, kept, place


def _valid(held: _Held, grouping: Grouping, content: Content) -> bool:
    """Whether ``held`` is the exact canonical construction of ``grouping``."""
    if not _same_structure((held.grouping, held.content), (id(grouping), content)):
        return False
    parts = components(grouping.blocks)
    rank = accepted_rank(grouping.blocks)
    references = _references(grouping, parts)
    columns = _columns(grouping, references)
    distinct = _distinct(grouping)
    return _valid_prepared(
        held, grouping, parts, rank, references, columns, distinct
    )


def _valid_prepared(
    held: _Held, grouping: Grouping, parts: Components, rank: IndicatorRank,
    references: tuple[IndexArray, ...], columns: tuple[IndexArray, ...],
    distinct: IndexArray,
) -> bool:
    """The SAME canonical map/witness/elimination checks on prepared topology."""
    if (
        type(held.rows) is not int
        or held.rows != int(distinct.shape[0])
        or not _same_structure(held.rank, _rank_values(rank))
        or not _same_structure(held.references, tuple(a.tobytes() for a in references))
        or not _same_structure(held.columns, tuple(a.tobytes() for a in columns))
    ):
        return False
    kept_state = _valid_kept(held, rank, columns, _count(columns))
    if kept_state is None:
        return False
    pivots, kept, place = kept_state
    checked = _valid_witnesses(
        held.witnesses,
        grouping,
        parts,
        columns,
        pivots=pivots,
        kept=kept,
        place=place,
    )
    if checked is None:
        return False
    if held.route == "connectivity":
        return True
    return _valid_elimination(
        pivots,
        kept,
        place,
        columns,
        distinct=distinct,
        dep_inv=checked[0],
        dep_scale=checked[1],
    )


def _owned(stored: bytes) -> IndexArray:
    """An owned read-only index array of stored bytes, in bytes of its own."""
    return frozen_index(np.frombuffer(stored, dtype=np.intp))


def _rebuilt(held: _Held, grouping: Grouping) -> CertificateBasis:
    """A new basis of the held values: new arrays and rank, the same tuples."""
    levels, joint, pairwise, lower, upper = held.rank
    return CertificateBasis(
        grouping=grouping,
        route=held.route,
        rank=IndicatorRank(levels, joint, pairwise, lower, upper),
        references=tuple(map(_owned, held.references)),
        columns=tuple(map(_owned, held.columns)),
        selected=_owned(held.selected),
        positions=tuple(map(_owned, held.positions)),
        witnesses=held.witnesses,
        echelon=held.echelon,
        rows=held.rows,
    )


# ------------------------------------------------- the proposed first construction

Entry = int | Fraction
Reduced = dict[int, Entry]  # a sparse rational row: column -> nonzero value
WORK = 32  # the proposal's entry updates allowed per code-row entry and E column


def _add(row: Reduced, column: int, value: Entry) -> None:
    """``row[column] += value`` in place, dropping an exact zero."""
    if new := row.get(column, 0) + value:
        row[column] = new
    else:
        row.pop(column, None)


def _primitive(found: Reduced) -> Row:
    """The primitive integer row, leading entry positive, proportional to ``found``."""
    scale = math.lcm(*(value.denominator for value in found.values()))
    row = {c: int(value * scale) for c, value in found.items()}
    divisor = math.gcd(*row.values())
    if row[min(row)] < 0:
        divisor = -divisor
    return {c: value // divisor for c, value in row.items()}



def _primitive_head(
    found: Reduced, *, compact: bool = False,
) -> tuple[tuple[int, int], ...] | SparseValues:
    """Freeze an exact integer head without two intermediate integer dictionaries.

    Integer denominators are one, so the generic scale is exactly one. Preserve
    its gcd, leading sign, zero entries and arbitrary-width values. Unknown
    scalar/key/container types retain the original primitive construction.
    """
    if type(found) is not dict or not found or any(
        type(column) is not int or type(value) is not int
        for column, value in found.items()
    ):
        if compact and type(found) is dict and found and all(
            type(column) is int and (type(value) is int or type(value) is Fraction)
            for column, value in found.items()
        ):
            primitive = _primitive(found)
            ordered = sorted(primitive)
            return SparseValues(
                tuple(ordered), tuple(primitive[column] for column in ordered)
            )
        pairs = tuple(sorted(_primitive(found).items()))
        return (SparseValues(tuple(c for c, _ in pairs), tuple(v for _, v in pairs))
                if compact else pairs)
    row = cast("Row", found)
    columns = sorted(row)
    divisor = math.gcd(*row.values())
    if row[columns[0]] < 0:
        divisor = -divisor
    if compact:
        return SparseValues(tuple(columns), tuple(row[c] // divisor for c in columns))
    if divisor == 1:
        return tuple((column, row[column]) for column in columns)
    return tuple((column, row[column] // divisor) for column in columns)


_INTEGER_GCD = math.gcd
_INTEGER_LCM = math.lcm

def _over(value: Entry, lead: Entry) -> Entry:
    """``value / lead`` exactly: an integer when it divides, else a fraction."""
    if type(value) is int and type(lead) is int and not value % lead:
        return value // lead
    return Fraction(value) / lead


def _reduce_by(found: Reduced, tails: dict[int, Reduced]) -> int:
    """Eliminate, in place, every later pivot present in ``found``; the work done.

    A reduced row is zero at every other later pivot, so no pivot comes back.
    """
    work = 0
    for q in [c for c in found if c in tails]:
        factor = found.pop(q)
        work += len(tails[q])
        for c, value in tails[q].items():
            _add(found, c, -factor * value)
    return work


def _new_tail(
    p: int, row: Reduced, tails: dict[int, Reduced], holders: dict[int, set[int]]
) -> int:
    """Add the reduced row of the later pivot ``p``; the work done.

    ``p`` is eliminated from every other reduced row (``holders`` indexes them by
    column), which keeps them zero at every other later pivot.
    """
    work = 0
    for q in holders.pop(p, ()):
        other = tails[q]
        factor = other.pop(p)
        work += len(row)
        for c, value in row.items():
            had = c in other
            _add(other, c, -factor * value)
            if c in other and not had:
                holders.setdefault(c, set()).add(q)
            elif had and c not in other:
                holders[c].discard(q)
    tails[p] = row
    for c in row:
        holders.setdefault(c, set()).add(p)
    return work


class _EchelonRows(NamedTuple):
    """Canonical head rows once; only later pivots need mutable exact dictionaries."""

    heads: dict[int, tuple[tuple[int, int], ...] | SparseValues | HeadRecipe] | PackedHeads
    later: dict[int, Row | SparseValues]
    tails: dict[int, Reduced]


def _head_work(
    head: HeadRecipe, later: list[int], tails: dict[int, Reduced],
) -> int:
    """Charge the ORIGINAL primitive-head reduction, without expanding that head.

    Its positive primitive scale cancels in the test for a nonzero pivot
    coefficient. Only current tail pivots can contribute to _reduce_by's charge.
    """
    candidates = {column for column in later if column in tails}
    candidates.update(column for column in head.anchor if column in tails)
    for _, row in head.birth:
        candidates.update(column for column in row.positions if column in tails)
    return sum(len(tails[column]) for column in candidates
               if int(column in later) != head.at(column))


def _unpacked_echelon(
    mapped: Iterable[IndexArray], budget: int,
) -> _EchelonRows | None:
    """The same primitive remainders, retaining small anchors and birth versions.

    R_current R_birth A = R_current A because later rows remain fully reduced.
    Thus reducing current minus original anchor changes the old remainder only
    by its positive primitive head scale. Its canonical primitive is identical.
    The old entry-update charge is computed separately, never relaxed.
    """
    heads: dict[int, HeadRecipe] = {}
    pivots: dict[int, Row | SparseValues] = {}
    tails: dict[int, Reduced] = {}
    holders: dict[int, set[int]] = {}
    versions: dict[int, SparseValues] = {}
    work = 0
    for values in zip(*mapped, strict=True):
        head = int(values[0])
        later = [int(column) for column in values[1:] if column >= 0]
        previous = heads.get(head)
        if previous is None:
            birth: list[tuple[int, SparseValues]] = []
            for column in later:
                if column in tails:
                    row = versions.get(column)
                    if row is None:
                        row = _sparse_witness(tails[column])
                        versions[column] = row
                    birth.append((column, row))
            heads[head] = HeadRecipe(head, tuple(later), tuple(birth))
            work += sum(len(tails[column]) for column in later if column in tails)
        else:
            found: Reduced = dict.fromkeys(later, 1)
            for column in previous.anchor:
                _add(found, column, -1)
            work += _head_work(previous, later, tails)
            _reduce_by(found, tails)
            if found:
                pivot = min(found)
                pivots[pivot] = _primitive_head(found, compact=True)
                lead = found.pop(pivot)
                tail = {column: _over(value, lead) for column, value in found.items()}
                # Birth recipes retain the old immutable value. Only versions
                # of rows actually changed by this pivot become unavailable.
                for changed in holders.get(pivot, ()):
                    versions.pop(changed, None)
                work += _new_tail(pivot, tail, tails, holders)
        if work > budget:
            return None
    return _EchelonRows(heads, pivots, tails)


def _packed_head_work(
    anchor: IndexArray, births: IndexArray, versions: list[SparseValues],
    later: list[int], tails: dict[int, Reduced],
) -> int:
    """The original primitive-head charge, without a per-head value object."""
    candidates = {column for column in later if column in tails}
    candidates.update(int(column) for column in anchor if int(column) in tails)
    for version in births:
        if version >= 0:
            candidates.update(column for column in versions[int(version)].positions
                              if column in tails)
    work = 0
    for column in candidates:
        value: int | Fraction = int(column in anchor)
        for pivot, version in zip(anchor, births, strict=True):
            if version >= 0:
                if column == pivot:
                    value -= 1
                value -= versions[int(version)].at(column)
        if int(column in later) != value:
            work += len(tails[column])
    return work


_UNPACKED_ECHELON = _unpacked_echelon


def _packed_echelon(
    mapped: tuple[IndexArray, ...] | list[IndexArray], budget: int, count: int,
) -> _EchelonRows | None | Literal[False]:
    """Packed dense heads; False abandons optional storage before old fallback.

    Required primitive/reduction/WORK operations are never caught. Optional
    allocation failure returns out of this frame before the old producer runs.
    """
    try:
        offsets = np.empty(count + 1, dtype=np.intp)
        anchors = np.empty(count * (len(mapped) - 1), dtype=np.intp)
        births = np.empty(anchors.size, dtype=np.intp)
        versions: list[SparseValues] = []
        current: dict[int, int] = {}
    except (MemoryError, OverflowError, ValueError):
        return False
    offsets[0] = 0
    seen, used, work = 0, 0, 0
    pivots: dict[int, Row | SparseValues] = {}
    tails: dict[int, Reduced] = {}
    holders: dict[int, set[int]] = {}
    for values in zip(*mapped, strict=True):
        head = int(values[0])
        later = [int(column) for column in values[1:] if column >= 0]
        if head < 0 or head > seen or head >= count:
            return False
        if head == seen:
            for column in later:
                version = current.get(column, -1)
                if column in tails and version < 0:
                    row = _sparse_witness(tails[column])
                    try:
                        version = len(versions)
                        versions.append(row)
                        current[column] = version
                    except MemoryError:
                        return False
                anchors[used], births[used] = column, version
                used += 1
            seen += 1
            offsets[seen] = used
            work += sum(len(tails[column]) for column in later if column in tails)
        else:
            low, high = int(offsets[head]), int(offsets[head + 1])
            anchor, born = anchors[low:high], births[low:high]
            found: Reduced = dict.fromkeys(later, 1)
            for column in anchor:
                _add(found, int(column), -1)
            work += _packed_head_work(anchor, born, versions, later, tails)
            _reduce_by(found, tails)
            if found:
                pivot = min(found)
                pivots[pivot] = _primitive_head(found, compact=True)
                lead = found.pop(pivot)
                tail = {column: _over(value, lead) for column, value in found.items()}
                for changed in holders.get(pivot, ()):
                    current.pop(changed, None)
                work += _new_tail(pivot, tail, tails, holders)
        if work > budget:
            return None
    if seen != count:
        return False
    try:
        heads = PackedHeads(count, offsets.tobytes(), anchors[:used].tobytes(),
                            births[:used].tobytes(), tuple(versions),
                            tuple(pivots[p] for p in sorted(pivots)))
    except MemoryError:
        return False
    return _EchelonRows(heads, pivots, tails)


def _factored_echelon(
    mapped: Iterable[IndexArray], budget: int, *, head_count: int | None = None,
) -> _EchelonRows | None:
    """One exact elimination, selecting only optional packed representation."""
    try:
        eligible = (type(head_count) is int and head_count > 0
                    and (type(mapped) is tuple or type(mapped) is list)
                    and _packed_original is _PACKED_GUARD and _packed_original()
                    and mapped and all(type(a) is np.ndarray and a.dtype == np.intp
                                       and a.ndim == 1 for a in mapped)
                    and mapped[0].size)
    except MemoryError:
        eligible = False
    if eligible:
        packed = _packed_echelon(mapped, budget, head_count)
        if packed is not False:
            return packed
    return _UNPACKED_ECHELON(mapped, budget)


def _echelon_rows(
    columns: tuple[IndexArray, ...], distinct: IndexArray, budget: int,
    *, mapped: tuple[IndexArray, ...] | None = None, compact: bool = False,
) -> _EchelonRows | None:
    """The existing exact elimination, keeping its first-block proof rows directly.

    A first-block pivot never changes after construction. Freeze its primitive
    row once, instead of retaining a dictionary and rebuilding the same tuple
    later. Later pivots and fully reduced tails retain the existing exact
    arithmetic and entry-update budget. The fresh private producer may supply
    those same mapped rows directly; generic callers retain the original gather.
    """
    heads: dict[int, tuple[tuple[int, int], ...] | SparseValues] = {}
    pivots: dict[int, Row | SparseValues] = {}
    tails: dict[int, Reduced] = {}
    holders: dict[int, set[int]] = {}
    work = 0
    freeze_heads = (
        _fit_generated()
        and math.gcd is _INTEGER_GCD and math.lcm is _INTEGER_LCM
        and type(columns) is tuple
        and all(type(column) is np.ndarray and column.dtype == np.intp
                and column.ndim == 1 for column in columns)
        and type(distinct) is np.ndarray and distinct.dtype == np.intp
        and distinct.ndim == (2 if mapped is None else 1)
    )
    compact = compact and freeze_heads
    col_arrays = (
        [col[distinct[:, k]] for k, col in enumerate(columns)]
        if mapped is None else mapped
    )
    if compact:
        return _factored_echelon(col_arrays, budget,
                                 head_count=columns[0].size if columns else None)
    for values in zip(*col_arrays, strict=True):
        head = int(values[0])
        later = [int(c) for c in values[1:] if c >= 0]
        row = heads.get(head)
        found: Reduced
        if row is None:
            found = dict.fromkeys(later, 1)
        else:
            items = row.items() if type(row) is SparseValues else row
            found = {c: -value for c, value in items if c != head}
            lead = row.numerators[0] if type(row) is SparseValues else row[0][1]
            for c in later:
                _add(found, c, lead)
        work += _reduce_by(found, tails)
        if row is None:
            found[head] = 1
            heads[head] = (
                _primitive_head(found, compact=True) if compact else
                _primitive_head(found) if freeze_heads
                else tuple(sorted(_primitive(found).items()))
            )
        elif found:
            p = min(found)
            primitive = _primitive(found)
            if compact:
                ordered = sorted(primitive)
                pivots[p] = SparseValues(tuple(ordered), tuple(primitive[c] for c in ordered))
            else:
                pivots[p] = primitive
            lead = found.pop(p)
            tail = {c: _over(value, lead) for c, value in found.items()}
            work += _new_tail(p, tail, tails, holders)
        if work > budget:
            return None
    return _EchelonRows(heads, pivots, tails)


def _indexed_echelon(
    grouping: Grouping, columns: tuple[IndexArray, ...],
    first: IndexArray, budget: int,
) -> _EchelonRows | None:
    """Gather canonical coordinates once from already selected source rows."""
    mapped = tuple(
        column[block.codes[first]]
        for column, block in zip(columns, grouping.blocks, strict=True)
    )
    return _echelon_rows(columns, first, budget, mapped=mapped, compact=True)


def _materialized_pivots(rows: _EchelonRows) -> dict[int, Row]:
    """Consume fresh rows into the generic dictionary shape, releasing head tuples."""
    pivots = cast("dict[int, Row]", rows.later)
    if type(rows.later) is dict and any(
        type(row) is SparseValues for row in rows.later.values()
    ):
        pivots = {
            p: dict(row.items()) if type(row) is SparseValues else row
            for p, row in rows.later.items()
        }
    while rows.heads:
        p, row = rows.heads.popitem()
        if type(row) is HeadRecipe:
            row = row.expanded()
        pivots[p] = dict(row.items() if type(row) is SparseValues else row)
    return pivots


def _proposed_echelon(
    columns: tuple[IndexArray, ...], distinct: IndexArray, budget: int
) -> tuple[dict[int, Row], dict[int, Reduced]] | None:
    """The same bounded exact proposal, in its original generic dictionary shape."""
    rows = _echelon_rows(columns, distinct, budget)
    return None if rows is None else (_materialized_pivots(rows), rows.tails)


def _witness_row(
    p: int, pivots: dict[int, Row], tails: dict[int, Reduced]
) -> Reduced:
    """The existing exact proposed row, shared by generic and private producers."""
    row = tails.get(p)
    if row is None:
        u = pivots[p]
        reduced: Reduced = {}
        for c, coefficient in u.items():
            if c == p:
                continue
            if c in tails:
                for d, entry in tails[c].items():
                    _add(reduced, d, -coefficient * entry)
            else:
                _add(reduced, c, coefficient)
        row = {c: _over(value, u[p]) for c, value in reduced.items()}
    return row


def _proposed_witnesses(
    pivots: dict[int, Row], tails: dict[int, Reduced], kept: list[int], count: int
) -> dict[int, Rational]:
    """A proposal of :func:`_column_witnesses` from the fully reduced rows.

    With ``F = R_JJ^-1 R`` (one at its own pivot, zero at every other), a dependent
    column's ``t`` is ``F``'s column: a later pivot's reduced row, or a head's row
    with its later pivots eliminated, divided by its pivot entry.
    """
    place = {c: i for i, c in enumerate(kept)}
    found: dict[int, Rational] = {c: {place[c]: Fraction(1)} for c in kept}
    for c in range(count):
        found.setdefault(c, {})
    for p in kept:
        row = _witness_row(p, pivots, tails)
        for c, value in row.items():
            found[c][place[p]] = Fraction(value)
    return found


def _proposal(grouping: Grouping) -> CertificateBasis | None:
    """A proposed basis of ``grouping``, or ``None`` past the work budget.

    The steps of :func:`certificate_basis`, with the echelon and the column witnesses
    proposed: :func:`_valid` decides.  Its exact span check is replaced by what makes
    it hold for admitted values: every row's levels in one component, so ``D = E M``
    (else ``None``: the construction checks, and refuses, as before).
    """
    parts = components(grouping.blocks)
    blocks = zip(parts.of_levels, grouping.blocks, strict=True)
    of_rows = [part[block.codes] for part, block in blocks]
    if not all(np.array_equal(of_rows[0], found) for found in of_rows):
        return None
    rank = accepted_rank(grouping.blocks)
    references = _references(grouping, parts)
    columns = _columns(grouping, references)
    distinct = _distinct(grouping)
    return _proposal_prepared(grouping, parts, rank, references, columns, distinct)


def _proposal_prepared(
    grouping: Grouping, parts: Components, rank: IndicatorRank,
    references: tuple[IndexArray, ...], columns: tuple[IndexArray, ...],
    distinct: IndexArray,
) -> CertificateBasis | None:
    """The SAME bounded proposal construction on fixed freshly prepared facts."""
    count = _count(columns)
    route: Route
    pivots: dict[int, Row] = {}
    if rank.exact == count:
        route, kept = "connectivity", list(range(count))
        of_column = _column_witnesses(pivots, kept, count)
    else:
        budget = WORK * (distinct.size + count)
        proposed = _proposed_echelon(columns, distinct, budget)
        if proposed is None:
            return None
        pivots, tails = proposed
        route, kept = "elimination", sorted(pivots)
        of_column = _proposed_witnesses(pivots, tails, kept, count)
    if not rank.lower <= len(kept) <= rank.upper:
        return None
    place = {c: i for i, c in enumerate(kept)}
    positions = tuple(
        frozen_index(np.array([place.get(int(c), -1) for c in col], dtype=np.intp))
        for col in columns
    )
    return CertificateBasis(
        grouping=grouping,
        route=route,
        rank=rank,
        references=references,
        columns=columns,
        selected=frozen_index(np.asarray(kept, dtype=np.intp)),
        positions=positions,
        witnesses=_witnesses(grouping, parts, columns, of_column),
        echelon=tuple(tuple(sorted(pivots[p].items())) for p in sorted(pivots)),
        rows=int(distinct.shape[0]),
    )



def _fit_index(value: object) -> bool:
    """The original content eligibility metadata, without serializing owned rows."""
    return (
        type(value) is np.ndarray
        and value.dtype == np.intp
        and value.ndim == 1
    )


def _fit_grouping(grouping: object) -> bool:
    """Use only exact built-in owned records; unknown inputs keep owned_basis."""
    if type(grouping) not in (Grouping, Derived):
        return False
    own = vars(grouping)
    blocks = own.get("blocks")
    if (
        type(blocks) is not tuple
        or blocks is not grouping.blocks
        or own.get("positions") is not grouping.positions
        or not _fit_index(grouping.positions)
    ):
        return False
    return all(
        type(block) is Block
        and type(block.name) is str
        and _fit_index(block.codes)
        and _fit_index(block.counts)
        for block in blocks
    )


def _fit_echelon(
    pivots: dict[int, Row], count: int
) -> tuple[list[int], tuple[tuple[tuple[int, int], ...], ...]] | None:
    """Prove primitive positive-leading rows and freeze their canonical order once."""
    if type(pivots) is not dict or any(
        type(p) is not int or not 0 <= p < count for p in pivots
    ):
        return None
    kept = sorted(pivots)
    echelon = []
    for p in kept:
        row = pivots[p]
        if type(row) is not dict or not row:
            return None
        if any(
            type(c) is not int or type(v) is not int or v == 0 or not 0 <= c < count
            for c, v in row.items()
        ):
            return None
        if min(row) != p or row[p] <= 0 or math.gcd(*row.values()) != 1:
            return None
        echelon.append(tuple(sorted(row.items())))
    return kept, tuple(echelon)


def _fit_dependent(
    mapping: Rational, column: int, kept: list[int],
    col_rows: dict[int, list[tuple[int, int]]], dep_rhs: dict[int, dict[int, int]],
) -> tuple[int, list[tuple[int, int]]] | None:
    """Check the fresh rational map directly against R_J t = R_column."""
    denominator = math.lcm(*(value.denominator for value in mapping.values()))
    scaled = [
        (idx, value.numerator * (denominator // value.denominator))
        for idx, value in sorted(mapping.items()) if value
    ]
    lhs: dict[int, int] = {}
    for idx, value in scaled:
        for p, entry in col_rows[kept[idx]]:
            lhs[p] = lhs.get(p, 0) + entry * value
    if {p: v for p, v in lhs.items() if v} != {
        p: v * denominator for p, v in dep_rhs.get(column, {}).items()
    }:
        return None
    return denominator, scaled


def _fit_maps(
    pivots: dict[int, Row], tails: dict[int, Reduced], place: dict[int, int], count: int
) -> dict[int, Rational] | None:
    """Only genuine dependent maps; kept identities are the existing place map."""
    found: dict[int, Rational] = {c: {} for c in range(count) if c not in place}
    for p, position in place.items():
        row = _witness_row(p, pivots, tails)
        if type(row) is not dict:
            return None
        for c, value in row.items():
            if (type(c) is not int or not 0 <= c < count
                    or type(value) not in (int, Fraction)):
                return None
            if c in place:
                # The generic map's only admitted kept entry is its own unit.
                # Do not hide a malformed cross-pivot or nonunit contribution.
                if c != p or value != 1:
                    return None
            else:
                found[c][position] = Fraction(value)
    return found


def _fit_row_maps(
    rows: _EchelonRows, place: dict[int, int], count: int,
) -> dict[int, Rational]:
    """Exact dependent columns from the same head/tail rows and one scratch row.

    A head's nonpivot entries are reduced as in _witness_row. Only genuine
    dependent entries are emitted, after exact cancellation and division.
    """
    dependent: dict[int, Rational] = {c: {} for c in range(count) if c not in place}
    scratch: Reduced = {}
    for p, position in place.items():
        row = rows.tails.get(p)
        lead = 1
        if row is None:
            scratch.clear()
            head = rows.heads[p]
            if type(head) is HeadRecipe:
                head = head.expanded()
            lead = head.numerators[0] if type(head) is SparseValues else head[0][1]
            items = head.items() if type(head) is SparseValues else head
            for c, coefficient in items:
                if c == p:
                    continue
                if c in rows.tails:
                    for d, entry in rows.tails[c].items():
                        _add(scratch, d, -coefficient * entry)
                else:
                    _add(scratch, c, coefficient)
            row = scratch
        for c, value in row.items():
            # _over already keeps an exact integer when division is integral.
            # A Fraction wrapper here makes every later reference addition pay
            # rational-object costs even when all its arithmetic is integral.
            dependent[c][position] = _over(value, lead)
    return dependent


def _fit_column_maps(
    pivots: dict[int, Row], kept: list[int], place: dict[int, int],
    count: int, of_column: dict[int, Rational],
) -> tuple[dict[int, Part], dict[int, int]] | None:
    """Prove the explicit dependent maps; kept identities follow from place."""
    if (type(of_column) is not dict or len(of_column) != count - len(kept) or any(
        type(c) is not int or not 0 <= c < count or c in place for c in of_column
    )):
        return None
    col_rows, dep_rhs = _echelon_columns(pivots, kept) if pivots else ({}, {})
    dep_inv: dict[int, Part] = {}
    dep_scale: dict[int, int] = {}
    for c, mapping in of_column.items():
        if type(mapping) is not dict or any(
            type(idx) is not int or not 0 <= idx < len(kept)
            or type(value) not in (int, Fraction)
            for idx, value in mapping.items()
        ):
            return None
        checked = _fit_dependent(mapping, c, kept, col_rows, dep_rhs)
        if checked is None:
            return None
        denominator, scaled = checked
        for idx, value in scaled:
            dep_inv.setdefault(kept[idx], []).append((c, value))
        dep_scale[c] = denominator
    return dep_inv, dep_scale


def _fit_add(
    total: Rational, column: int, multiple: int,
    place: dict[int, int], dependent: dict[int, Rational],
) -> None:
    """Accumulate exact sparse reference terms, dropping zeros as they cancel."""
    position = place.get(column)
    if position is not None:
        if new := total.get(position, 0) + multiple:
            total[position] = new
        else:
            total.pop(position, None)
    else:
        for p, value in dependent[column].items():
            if new := total.get(p, 0) + multiple * value:
                total[p] = new
            else:
                total.pop(p, None)


def _fit_witnesses_checked(
    grouping: Grouping, parts: Components, columns: tuple[IndexArray, ...],
    place: dict[int, int], dependent: dict[int, Rational],
) -> tuple[tuple[Witness, ...], ...]:
    """Emit the same complete canonical witnesses, with implicit kept identities."""
    first, found = parts.of_levels[0], []
    for k, block in enumerate(grouping.blocks):
        mine: list[Witness] = []
        for level in range(block.n_levels):
            column = int(columns[k][level])
            position = place.get(column)
            if position is not None:
                mine.append(((position, 1, 1),))
            elif column >= 0:
                mine.append(_canonical(dependent[column]))
            else:
                component = parts.of_levels[k][level]
                total: Rational = {}
                for h in np.flatnonzero(first == component).tolist():
                    _fit_add(total, int(columns[0][h]), 1, place, dependent)
                for s in np.flatnonzero(parts.of_levels[k] == component).tolist():
                    if s != level:
                        _fit_add(total, int(columns[k][s]), -1, place, dependent)
                mine.append(_canonical(total))
        found.append(tuple(mine))
    return tuple(found)


def _fit_component_sums(
    columns: IndexArray, components: IndexArray,
    place: dict[int, int], dependent: dict[int, Rational],
) -> dict[int, Rational]:
    """Sum first-block witnesses once per component, using exact integer units."""
    totals: dict[int, Rational] = {}
    for component, column in zip(components.tolist(), columns.tolist(), strict=True):
        total = totals.get(component)
        if total is None:
            total = {}
            totals[component] = total
        _fit_add(total, column, 1, place, dependent)
    return totals


def _fit_reference_block(
    layout: tuple[IndexArray, IndexArray], heads: dict[int, Rational],
    place: dict[int, int], dependent: dict[int, Rational],
) -> dict[int, Witness]:
    """Each reference is its component's head sum less its nonreference levels."""
    columns, components = layout
    references = np.flatnonzero(columns < 0).tolist()
    totals = {
        int(components[level]): dict(heads[int(components[level])])
        for level in references
    }
    for component, column in zip(components.tolist(), columns.tolist(), strict=True):
        if column >= 0:
            _fit_add(totals[component], column, -1, place, dependent)
    return {
        level: _canonical(totals[int(components[level])]) for level in references
    }


def _fit_witnesses(
    grouping: Grouping, parts: Components, columns: tuple[IndexArray, ...],
    place: dict[int, int], dependent: dict[int, Rational],
) -> tuple[tuple[Witness, ...], ...]:
    """Keep the original eager checked route for changed generators/callbacks."""
    return _fit_witnesses_checked(grouping, parts, columns, place, dependent)


def _compact_witnesses(
    grouping: Grouping, parts: Components,
    layout: tuple[tuple[IndexArray, ...], tuple[IndexArray, ...]],
    place: dict[int, int], dependent: dict[int, Rational],
    *, echelon: tuple[tuple[tuple[int, int], ...] | SparseValues | HeadRecipe, ...] | PackedHeads,
    rows: _EchelonRows | None = None,
) -> WitnessProof:
    """Freeze dependent values once; leave units and reference combinations implicit.

    _columns orders E by block/level. Generated heads are in its first block;
    sorted heads then sorted later pivots therefore give increasing J. Its
    consecutive place values within each block are exactly start + level minus
    preceding nonkept levels, including every disconnected/reference/dependent gap.
    Changed generators keep the checked eager route, never this encoding.
    """
    columns, positions = layout
    packed = rows is not None and type(rows.heads) is PackedHeads
    factored = packed or rows is not None and all(
        type(row) is HeadRecipe for row in rows.heads.values()
    )
    minimum = rows.heads.count if packed else 0
    count = (minimum + sum(int(np.count_nonzero(c >= 0)) for c in columns[1:])
             if packed else _count(columns))
    dependent_columns = tuple(
        column for column in range(minimum, count) if column not in place
    ) if factored else tuple(sorted(dependent))
    values = () if factored else tuple(
        _sparse_witness(dependent[c]) for c in dependent_columns
    )
    frozen = {} if factored else dict(zip(dependent_columns, values, strict=True))
    factors = None
    if factored:
        assert rows is not None
        tail_columns = tuple(sorted(rows.tails))
        factors = FactoredRows(
            rows.heads if packed else
            tuple(cast("HeadRecipe", rows.heads[p]) for p in sorted(rows.heads)),
            tail_columns, tuple(_sparse_witness(rows.tails[p]) for p in tail_columns),
            tuple(place[p] for p in tail_columns),
        )
    context = WitnessContext(
        tuple(_proof_bytes(c) for c in columns),
        tuple(_proof_bytes(c) for c in parts.of_levels),
        tuple(_proof_bytes(p) for p in positions),
        dependent_columns, values, factors,
    )
    blocks: list[WitnessBlock] = []
    start = 0
    for k, column in enumerate(columns):
        levels = (() if packed and k == 0 else
                  tuple(np.flatnonzero(positions[k] < 0).tolist()))
        exceptions = tuple(
            (c if factored else frozen[c]) if (c := int(column[level])) >= 0 else None
            for level in levels
        )
        blocks.append(WitnessBlock(
            grouping.blocks[k].n_levels, start, levels, exceptions, context, k
        ))
        start += column.size - len(levels)
    retained = None
    if packed:
        try:
            retained = _retained_rows(context)
        except MemoryError:
            pass  # Optional action only; required full action remains available.
    return WitnessProof(tuple(blocks), echelon, retained)


def _retained_rows(context: WitnessContext | OperationalBasis) -> RetainedRows:
    """Compile final-tail sparse rows once, never scan eliminated head levels.

    Global E columns already order each block's levels. Per-coordinate exact
    reference sums therefore use the same nonreference-level addition order.
    First-block units contribute zero to every retained reference coefficient.
    """
    factors = context.factors
    assert factors is not None and (type(factors.heads) is PackedHeads
                                    or type(factors.heads) is HeadRows)
    minimum = factors.heads.count
    columns = tuple(np.frombuffer(raw, dtype=np.intp) for raw in context.columns)
    positions = tuple(np.frombuffer(raw, dtype=np.intp) for raw in context.positions)
    components0 = tuple(np.frombuffer(raw, dtype=np.intp) for raw in context.components)
    # This E-sized native lookup is temporary; no full-input observation table.
    count = minimum + sum(int(np.count_nonzero(col >= 0)) for col in columns[1:])
    block_of = np.empty(count - minimum, dtype=np.intp)
    level_of = np.empty(count - minimum, dtype=np.intp)
    references = ({}, *({int(components0[b][level]): int(level)
                         for level in np.flatnonzero(columns[b] < 0)}
                        for b in range(1, len(columns))))
    for block, col in enumerate(columns[1:], start=1):
        levels = np.flatnonzero(col >= 0)
        block_of[col[levels] - minimum], level_of[col[levels] - minimum] = block, levels
    actions: list[dict[int, list[tuple[int, int, int]]]] = [{} for _ in columns]
    norm = Fraction(1)
    for index, (pivot, row) in enumerate(zip(factors.tail_columns, factors.tails, strict=True)):
        norm = max(norm, sum((abs(Fraction(n, d)) for n, d in
                             zip(row.numerators, row.denominators, strict=True)), Fraction(0)))
        sums: dict[tuple[int, int], int | Fraction] = {}
        terms = [(pivot, 1, 1), *zip(row.positions, row.numerators, row.denominators, strict=True)]
        for column, numerator, denominator in sorted(terms):
            block, level = int(block_of[column - minimum]), int(level_of[column - minimum])
            component = int(components0[block][level])
            value = numerator if denominator == 1 else Fraction(numerator, denominator)
            key = (block, component)
            sums[key] = sums.get(key, 0) - value
            if column != pivot:
                actions[block].setdefault(level, []).append((index, numerator, denominator))
        for (block, component), value in sums.items():
            if value:
                level = references[block][component]
                actions[block].setdefault(level, []).append(
                    (index, value.numerator, value.denominator))
    kept_levels, kept_positions = [b""], [b""]
    for place in positions[1:]:
        levels = np.flatnonzero(place >= minimum)
        kept_levels.append(levels.tobytes())
        kept_positions.append((place[levels] - minimum).tobytes())
    ordered = tuple(tuple(sorted(block)) for block in actions)
    return RetainedRows(minimum, tuple(kept_levels), tuple(kept_positions), ordered,
                        tuple(tuple(tuple(actions[b][level]) for level in levels)
                              for b, levels in enumerate(ordered)), norm)


def _packed_coordinates(
    columns: tuple[IndexArray, ...], head_count: int,
    later: tuple[int, ...], count: int,
) -> tuple[IndexArray, tuple[IndexArray, ...]]:
    """The same dense lookup, without head-sized Python kept/place objects."""
    selected = frozen_index(np.concatenate((np.arange(head_count, dtype=np.intp),
                                            np.asarray(later, dtype=np.intp))))
    lookup = np.full(count + 1, -1, dtype=np.intp)
    lookup[selected] = np.arange(selected.size, dtype=np.intp)
    return selected, tuple(frozen_index(lookup[column]) for column in columns)


def _sparse_witness(mapping: Rational) -> SparseValues:
    """Detach canonical dependent primitives without allocating outward triples."""
    items = [(p, v) for p, v in sorted(mapping.items()) if v]
    return SparseValues(
        tuple(p for p, _ in items), tuple(v.numerator for _, v in items),
        tuple(v.denominator for _, v in items),
    )


def _proof_bytes(values: IndexArray) -> bytes:
    """Retain complete immutable value bytes, never a mutable array header."""
    if (values.flags.c_contiguous and type(values.base) is bytes
            and len(values.base) == values.nbytes):
        return values.base
    return values.tobytes()


_OPERATIONAL_BASIS_CLASS = CertificateBasis
_OPERATIONAL_DATA_CLASS = OperationalBasis
_OPERATIONAL_HEAD_CLASS = HeadRows
_OPERATIONAL_BASIS_DICT = vars(CertificateBasis).get("__dict__")
_OPERATIONAL_BASIS_INIT = CertificateBasis.__init__
_OPERATIONAL_BASIS_NEW = CertificateBasis.__new__
Echelon = tuple[tuple[tuple[int, int], ...], ...]
Replay = tuple[IndexArray, dict[int, Reduced], Echelon]


def _audit_echelon(mapped: tuple[IndexArray, ...], count: int) -> Replay | None:
    """Use the same Python row kernel, freezing primitive rows only on demand."""
    found = _operational_rows(mapped, count, None, audit=True)
    if found is None or found is False:
        return None
    first, tails, _work, echelon = found
    return first, tails, echelon


def _anchor_row(head: int, later: list[int], tails: dict[int, Reduced]) -> Reduced:
    """The head's anchor row reduced by the later pivots that exist now."""
    row: Reduced = {head: 1}
    for column in later:
        tail = tails.get(column)
        if tail is None:
            _add(row, column, 1)
        else:
            for other, value in tail.items():
                _add(row, other, -value)
    return row


_Weight = int | Fraction | list[int | Fraction]
_Negated = tuple[IndexArray, list[int | Fraction]]  # a dependent witness, negated
_Terms = list[tuple[IndexArray, _Weight]]
_AUDIT_INT64_MAX = 2**63 - 1


def _small_integers(weights: Iterable[object], limit: int) -> bool:
    return all(type(w) is int and -limit <= w <= limit for w in weights)


def _audit_sum(parts: _Terms) -> Witness:
    """The canonical witness of exact sums over position terms.

    Each part is a position array with one weight for all of them, or one weight
    per position; equal positions are summed exactly, zeros are dropped and the
    triples are ordered by position, as :func:`_canonical` orders them.  Integer
    weights use int64 only when the sum of absolute term bounds fits int64;
    every partial sum then fits too. Otherwise use arbitrary Python numbers.
    """
    terms = sum(p.size for p, _ in parts)
    limit = _AUDIT_INT64_MAX // max(terms, 1)
    if (_small_integers((w for _, w in parts if type(w) is not list), limit)
            and _small_integers((w for _, v in parts if type(v) is list for w in v), limit)):
        positions = (np.concatenate([p for p, _ in parts]) if parts
                     else np.empty(0, np.intp))
        weights = np.concatenate([
            np.asarray(w, dtype=np.int64) if type(w) is list
            else np.full(p.size, w, dtype=np.int64) for p, w in parts
        ]) if parts else np.empty(0, np.int64)
        unique, inverse = np.unique(positions, return_inverse=True)
        sums = np.zeros(unique.size, dtype=np.int64)
        np.add.at(sums, inverse, weights)
        keep = sums != 0
        return tuple((p, v, 1) for p, v in
                     zip(unique[keep].tolist(), sums[keep].tolist(), strict=True))
    total: dict[int, int | Fraction] = {}
    for positions, weight in parts:
        values = (weight if type(weight) is list
                  else [cast("int | Fraction", weight)] * positions.size)
        for position, value in zip(positions.tolist(), values, strict=True):
            if new := total.get(position, 0) + value:
                total[position] = new
            else:
                total.pop(position, None)
    return tuple((p, v.numerator, v.denominator) for p, v in sorted(total.items()) if v)


class _AnchorIndex:
    """The heads whose anchor row holds a later column, by column."""

    __slots__ = ("columns", "heads")

    def __init__(self, mapped: tuple[IndexArray, ...], first_of: IndexArray) -> None:
        count = first_of.size
        anchors = [col[first_of] for col in mapped[1:]]
        columns = (np.concatenate(anchors) if anchors
                   else np.empty(0, dtype=np.intp))
        heads = np.tile(np.arange(count, dtype=np.intp), len(anchors))
        keep = columns >= 0
        order = np.argsort(columns[keep], kind="stable")
        self.columns: IndexArray = columns[keep][order]
        self.heads: IndexArray = heads[keep][order]

    def of(self, column: int) -> IndexArray:
        lo = int(np.searchsorted(self.columns, column, side="left"))
        hi = int(np.searchsorted(self.columns, column, side="right"))
        return self.heads[lo:hi]


def _dependent_witnesses(
    columns: tuple[IndexArray, ...], tails: dict[int, Reduced],
    position: dict[int, int], anchors: _AnchorIndex,
) -> dict[int, Witness]:
    """The witness of every dependent later column from the final tails.

    ``w_p = T[p, c]`` at the later pivots and, at a head, the anchor's indicator
    of ``c`` minus the tail entries of its anchor pivots at ``c``
    (:meth:`FactoredRows.dependent`), formed through the anchor index instead of
    a pass over every head.
    """
    held: dict[int, list[tuple[int, int | Fraction]]] = {}
    for p, tail in tails.items():
        for column, value in tail.items():
            held.setdefault(column, []).append((p, value))
    found: dict[int, Witness] = {}
    for column in np.concatenate([c[c >= 0] for c in columns[1:]]).tolist():
        if column in position:
            continue
        holders = held.get(column, [])
        terms: _Terms = [(anchors.of(column), 1)]
        terms.extend((anchors.of(p), -value) for p, value in holders)
        terms.append((np.asarray([position[p] for p, _ in holders], dtype=np.intp),
                      [value for _, value in holders]))
        found[column] = _audit_sum(terms)
    return found


def _reference_witness(
    block: int, level: int, column: IndexArray, *, parts: tuple[IndexArray, ...],
    position: dict[int, int], negated: dict[int, _Negated],
) -> Witness:
    """A reference's witness: its component's heads minus the block's other levels."""
    count = parts[0].size
    component = int(parts[block][level])
    terms: _Terms = [(np.flatnonzero(parts[0] == component), 1)]
    kept: list[int] = []
    for s in np.flatnonzero(parts[block] == component).tolist():
        if s == level:
            continue
        other = int(column[s])
        if other in negated:
            terms.append(negated[other])
        else:
            kept.append(position[other] if other >= count else other)
    terms.append((np.asarray(kept, dtype=np.intp), -1))
    return _audit_sum(terms)


def _audit_operational(
    data: OperationalBasis,
) -> tuple[tuple[tuple[Witness, ...], ...], Echelon] | None:
    """The canonical witnesses and echelon from the operational replay, or ``None``.

    The replay (:func:`_audit_echelon`) reproduces the stored heads and tails from
    the immutable codes and columns; a difference declines, and the caller runs
    the original canonical kernels.  Each dependent column's witness is the unique
    rational ``w`` with ``E_c = E_J w`` (:func:`_dependent_witnesses`); a
    reference's witness is the component sum of :func:`_witnesses`.
    """
    if (type(data) is not OperationalBasis or not _fit_generated()
            or _audit_helpers_original is not _AUDIT_HELPERS_GUARD
            or not _audit_helpers_original()):
        return None
    factors = data.factors
    if type(factors) is not FactoredRows:
        return None
    heads = factors.heads
    if type(heads) is not HeadRows:
        return None
    columns = tuple(np.frombuffer(raw, dtype=np.intp) for raw in data.columns)
    codes = tuple(np.frombuffer(raw, dtype=np.intp) for raw in heads.codes)
    parts = tuple(np.frombuffer(raw, dtype=np.intp) for raw in data.components)
    count = int(columns[0].size)
    if (heads.count != count or len(codes) != len(columns)
            or any(code.size != codes[0].size for code in codes)):
        return None
    mapped = tuple(column[code] for column, code in zip(columns, codes, strict=True))
    replayed = _audit_echelon(mapped, count)
    if replayed is None:
        return None
    first_of, tails, echelon = replayed
    pivots = tuple(sorted(tails))
    if (first_of.tobytes() != heads.first or pivots != factors.tail_columns
            or tuple(_sparse_witness(tails[p]) for p in pivots) != factors.tails):
        return None
    position = {p: count + i for i, p in enumerate(pivots)}
    dependent = _dependent_witnesses(
        columns, tails, position, _AnchorIndex(mapped, first_of)
    )
    negated = {
        column: (np.asarray([p for p, _, _ in witness], dtype=np.intp),
                 [-(n if d == 1 else Fraction(n, d)) for _, n, d in witness])
        for column, witness in dependent.items()
    }
    views: list[tuple[Witness, ...]] = []
    for k, column in enumerate(columns):
        block: list[Witness] = []
        for level, c in enumerate(column.tolist()):
            if c in dependent:
                block.append(dependent[c])
            elif c >= 0:
                block.append(((c if c < count else position[c], 1, 1),))
            else:
                block.append(_reference_witness(
                    k, level, column, parts=parts, position=position, negated=negated,
                ))
        views.append(tuple(block))
    return tuple(views), echelon


_OPERATIONAL_AUDIT = (
    _reduced, _column_witnesses, _dependent, _witnesses, _canonical, _reference,
    (math.gcd, heapq.heapify, heapq.heappop, heapq.heappush, _divide, Fraction),
    np.flatnonzero, _audit_operational,
)


def _operational_data(basis: object) -> OperationalBasis | None:
    """Read immutable value data only; this issues no numerical admission."""
    if type(basis) is not _OPERATIONAL_BASIS_CLASS:
        return None
    if vars(_OPERATIONAL_BASIS_CLASS).get("__dict__") is not _OPERATIONAL_BASIS_DICT:
        return None
    data = vars(basis).get("_operational")
    return data if type(data) is _OPERATIONAL_DATA_CLASS else None


def _export_operational_basis(basis: CertificateBasis) -> CertificateBasis:
    """Demand the exact historical public view from immutable numeric originals.

    Detached result adapters may supply their own grouping/header objects with
    the same immutable data. No live fit, provider, verdict or numerical replay
    is consumed by this audit-only operation.
    """
    data = _operational_data(basis)
    if data is None:
        return basis
    witnesses, echelon = data.audit(_OPERATIONAL_AUDIT)
    values = vars(basis)
    return _OPERATIONAL_BASIS_CLASS(
        grouping=values["grouping"], route=values["route"], rank=values["rank"],
        references=values["references"], columns=values["columns"],
        selected=values["selected"], positions=values["positions"],
        witnesses=witnesses, echelon=echelon, rows=values["rows"],
    )


def _rebind_operational_basis(basis: CertificateBasis, grouping: Grouping) -> CertificateBasis:
    """A data/header view at the existing live rebind, never a capability issuer."""
    data = _operational_data(basis)
    if data is None:
        raise ValueError("the operational view needs its actual numeric data")
    values = vars(basis)
    found = _OPERATIONAL_BASIS_CLASS(
        grouping=grouping, route=values["route"], rank=values["rank"],
        references=values["references"], columns=values["columns"],
        selected=values["selected"], positions=values["positions"],
        witnesses=cast("tuple[tuple[Witness, ...], ...]", data),
        echelon=cast("tuple[tuple[tuple[int, int], ...], ...]", data), rows=values["rows"],
    )
    object.__setattr__(found, "_operational", data)
    if "_basis_backend" in values:
        object.__setattr__(found, "_basis_backend", values["_basis_backend"])
    return found


def _operational_rows(
    mapped: tuple[IndexArray, ...], count: int, budget: int | None, *, audit: bool = False,
    _first: IndexArray | None = None,
) -> tuple[IndexArray, dict[int, Reduced], int, Echelon] | None | Literal[False]:
    """One authoritative ordered Python loop; audit rows are frozen at birth."""
    first_of = np.full(count, -1, dtype=np.intp) if _first is None else _first
    tails: dict[int, Reduced] = {}
    holders: dict[int, set[int]] = {}
    work = 0
    heads: dict[int, Row] | None = {} if audit else None
    later_rows: dict[int, Row] | None = {} if audit else None
    for index, values in enumerate(zip(*mapped, strict=True)):
        head = int(values[0])
        if not 0 <= head < count:
            return False
        later = [int(column) for column in values[1:] if column >= 0]
        previous = int(first_of[head])
        work += len(later)
        if previous < 0:
            first_of[head] = index
            if audit:
                cast("dict[int, Row]", heads)[head] = _primitive(_anchor_row(head, later, tails))
        else:
            anchor = [int(col[previous]) for col in mapped[1:] if col[previous] >= 0]
            found: Reduced = dict.fromkeys(later, 1)
            for column in anchor:
                _add(found, column, -1)
            work += len(anchor) + _reduce_by(found, tails)
            if found:
                pivot = min(found)
                if audit:
                    cast("dict[int, Row]", later_rows)[pivot] = _primitive(found)
                lead = found.pop(pivot)
                tail = {column: _over(value, lead) for column, value in found.items()}
                work += len(found) + _new_tail(pivot, tail, tails, holders)
        if budget is not None and work > budget:
            return None
    if (first_of < 0).any():
        return False
    echelon = (() if not audit else
        tuple(tuple(sorted(cast("dict[int, Row]", heads)[h].items())) for h in range(count)) +
        tuple(tuple(sorted(cast("dict[int, Row]", later_rows)[p].items()))
              for p in sorted(cast("dict[int, Row]", later_rows))))
    return first_of, tails, work, echelon


def _native_tails(
    module: ModuleType, mapped: tuple[IndexArray, ...], count: int, budget: int,
    trace: list,
) -> tuple[IndexArray, dict[int, Reduced], int] | None:
    """A single native attempt; only representational declines replay Python."""
    if count > 2**63 - 1 or not 0 <= budget <= 2**63 - 1:
        trace[:] = ["python", None, None, "argument_capacity", False, False]
        return None
    trace[4], trace[5] = True, True
    try:
        found, work, reason = module.attempt_from_mapped(mapped, count, budget)
        trace[1], trace[3] = work, reason
        if found is None:
            trace[4] = False
            return None
        first_of, rows, actual_work = found
        result = np.asarray(first_of, dtype=np.intp), dict(rows), actual_work
    except MemoryError:
        trace[0], trace[2], trace[3], trace[4] = "python", None, "allocation", False
        return None
    # Other native failures propagate through the proposal boundary unchanged.
    trace[0], trace[4] = "cpp", False
    return result


def _operational_echelon(
    grouping: Grouping, columns: tuple[IndexArray, ...], first: IndexArray,
    budget: int, *, _native: ModuleType | None = None, _trace: list | None = None,
) -> FactoredRows | None | Literal[False]:
    """The shared exact residual-tail operations; no fit-time historical rows."""
    count = columns[0].size
    try:
        first_of = np.full(count, -1, dtype=np.intp)
        mapped = tuple(column[block.codes[first]] for column, block in
                       zip(columns, grouping.blocks, strict=True))
    except MemoryError:
        return False
    found = None
    if _native is not None and _trace is not None:
        found = _native_tails(_native, mapped, count, budget, _trace)
    if found is None:
        if _trace is not None and _trace[3] == "work_limit":
            return None  # Same completed-row budget crossing; canonical fallback only.
        result = _operational_rows(mapped, count, budget, _first=first_of)
        if result is None or result is False:
            return result
        first_of, tails, work, _ = result
        if _trace is not None:
            _trace[0], _trace[2] = "python", work
    else:
        first_of, tails, work = found
    # Required exact reductions have completed. Optional value freezing leaves
    # this helper frame before the old producer can be entered.
    try:
        pivots = tuple(sorted(tails))
        code_headers = tuple(_retained_header(b.codes) for b in grouping.blocks)
        column_headers = tuple(_retained_header(col) for col in columns)
        if any(header is None for header in (*code_headers, *column_headers)):
            return False
        heads = HeadRows(count, first[first_of].tobytes(),
                         tuple(header[1] for header in code_headers),
                         tuple(header[1] for header in column_headers))
        return FactoredRows(heads, pivots,
                            tuple(_sparse_witness(tails[p]) for p in pivots),
                            tuple(count + i for i in range(len(pivots))))
    except MemoryError:
        return False


def _operational_basis_prepared(
    owner: FESample, grouping: Grouping, parts: Components, rank: IndicatorRank,
    references: tuple[IndexArray, ...], columns: tuple[IndexArray, ...],
    first: IndexArray, *, _native: ModuleType | None = None, _trace: list | None = None,
) -> CertificateBasis | None | Literal[False]:
    """Produce the one fit's numerical layout, not a generic proof graph."""
    count = _count(columns)
    rows = _operational_echelon(
        grouping, columns, first, WORK * (first.size * len(columns) + count),
        _native=_native, _trace=_trace,
    )
    if rows is None or rows is False:
        return rows
    heads = cast(HeadRows, rows.heads)
    kept = heads.count + len(rows.tail_columns)
    if not rank.lower <= kept <= rank.upper:
        return None
    try:
        selected, positions = _packed_coordinates(
            columns, heads.count, rows.tail_columns, count
        )
        data = OperationalBasis(
            heads.columns,
            tuple(_proof_bytes(part) for part in parts.of_levels),
            tuple(_proof_bytes(place) for place in positions), rows,
        )
        try:
            action = _retained_rows(data)
        except MemoryError:
            action = None
        data = OperationalBasis(data.columns, data.components, data.positions, rows, action)
        basis = CertificateBasis(
            grouping, "elimination", rank, references, columns, selected, positions,
            cast("tuple[tuple[Witness, ...], ...]", data),
            cast("tuple[tuple[tuple[int, int], ...], ...]", data), int(first.size),
        )
        object.__setattr__(basis, "_operational", data)
        headers = _retained_headers(basis, grouping)
        if headers is None:
            return False
        scope = (ref(basis), ref(grouping), ref(owner))

        def construction():
            return scope, data, headers

        object.__setattr__(basis, _GENERATED_RETAINED, construction)
        return basis
    except (MemoryError, TypeError):
        return False


_OPERATIONAL_PREPARED_ISSUER = _operational_basis_prepared


_BASIS_ACCESS = CertificateBasis.__getattribute__
_BASIS_GETATTR = getattr(CertificateBasis, "__getattr__", None)
_BASIS_WITNESS_FIELD = vars(CertificateBasis).get("witnesses")
_BASIS_ECHELON_FIELD = vars(CertificateBasis).get("echelon")


def _witness_proof(basis: object) -> WitnessProof | None:
    """Read private immutable proof values only from the exact built-in record."""
    if type(basis) is not CertificateBasis:
        return None
    try:
        if (CertificateBasis.__getattribute__ is not _BASIS_ACCESS
                or getattr(CertificateBasis, "__getattr__", None) is not _BASIS_GETATTR
                or vars(CertificateBasis).get("witnesses") is not _BASIS_WITNESS_FIELD
                or vars(CertificateBasis).get("echelon") is not _BASIS_ECHELON_FIELD):
            return None
    except MemoryError:
        # Optional raw-field selection only; ordinary field reads remain required.
        return None
    found: object = vars(basis).get("witnesses")
    return found if type(found) is WitnessProof else None


def _fit_coordinates(
    columns: tuple[IndexArray, ...], kept: list[int], count: int,
) -> tuple[IndexArray, tuple[IndexArray, ...]]:
    """Materialize one dense E-to-J lookup and the complete detached field arrays."""
    selected = frozen_index(np.asarray(kept, dtype=np.intp))
    lookup = np.full(count + 1, -1, dtype=np.intp)
    lookup[selected] = np.arange(selected.size, dtype=np.intp)
    positions = tuple(frozen_index(lookup[column]) for column in columns)
    return selected, positions


_HEAD_RECIPE_CLASS = HeadRecipe
_FACTORED_ROWS_CLASS = FactoredRows
_AUDIT_ANCHOR_CLASS = _AnchorIndex
_AUDIT_HELPERS = (_operational_rows, _anchor_row, _small_integers, _audit_sum,
    _AnchorIndex, vars(_AnchorIndex).get("of"), _dependent_witnesses, _reference_witness,
    _audit_echelon, _audit_operational)


def _audit_helpers_original() -> bool:
    if _AnchorIndex is not _AUDIT_ANCHOR_CLASS:
        return False
    current = (_operational_rows, _anchor_row, _small_integers, _audit_sum,
        _AnchorIndex, vars(_AnchorIndex).get("of"), _dependent_witnesses, _reference_witness,
        _audit_echelon, _audit_operational)
    return all(a is b for a, b in zip(current, _AUDIT_HELPERS, strict=True))


_AUDIT_HELPERS_GUARD = _audit_helpers_original
_FIT_GENERATORS = (
    components, _indicator_rank_prepared, _references, _columns, _distinct,
    _proposed_echelon, _primitive, _over, _reduce_by, _new_tail, _add,
    _witness_row, _fit_maps, _fit_echelon, _fit_column_maps, _fit_dependent,
    _valid_elimination, _fit_add, _canonical,
    _echelon_rows, _fit_row_maps, _EchelonRows,
    _fit_witnesses, _fit_coordinates, _compact_witnesses,
    _fit_component_sums, _fit_reference_block, _witness_proof,
    WitnessProof, WitnessBlock, WitnessProof.witness, WitnessProof.materialize,
    WitnessProof.__new__, WitnessBlock.__new__,
    WitnessBlock.witness, WitnessBlock.materialize,
    WitnessBlock.iter_witnesses,
    _primitive_head, _first_rows, _indexed_echelon,
    _sparse_witness, _proof_bytes, SparseValues, SparseValues.__new__,
    SparseValues.items, SparseValues.canonical, SparseValues.__getattribute__,
    WitnessContext, WitnessContext.__new__, WitnessContext._add,
    WitnessContext.references, WitnessContext.__getattribute__,
    WitnessProof.materialize_echelon, WitnessBlock.iter_selected,
    CertificateBasis.__getattribute__, getattr(CertificateBasis, "__getattr__", None),
    vars(CertificateBasis).get("witnesses"), vars(CertificateBasis).get("echelon"),
    _head_work, _factored_echelon,
    HeadRecipe, HeadRecipe.__new__, HeadRecipe.__getattribute__,
    HeadRecipe.at, HeadRecipe.expanded,
    FactoredRows, FactoredRows.__new__, FactoredRows.__getattribute__,
    FactoredRows.dependent, SparseValues.at, WitnessContext.dependent,
    _witness_values.HeadRecipe, _witness_values.FactoredRows,
    getattr(HeadRecipe, "__getattr__", None),
    vars(HeadRecipe).get("pivot"), vars(HeadRecipe).get("anchor"),
    vars(HeadRecipe).get("birth"),
    getattr(FactoredRows, "__getattr__", None),
    vars(FactoredRows).get("heads"), vars(FactoredRows).get("tail_columns"),
    vars(FactoredRows).get("tails"), vars(FactoredRows).get("tail_positions"),
    _operational_echelon, _operational_basis_prepared, _operational_data,
    _operational_rows, _native_tails,
    _export_operational_basis, _rebind_operational_basis,
    HeadRows, HeadRows.__new__, HeadRows.__getattribute__, HeadRows.anchor,
    getattr(HeadRows, "__getattr__", None),
    vars(HeadRows).get("count"), vars(HeadRows).get("first"),
    vars(HeadRows).get("codes"), vars(HeadRows).get("columns"),
    OperationalBasis, OperationalBasis.__new__, OperationalBasis.__getattribute__,
    OperationalBasis.dependent, OperationalBasis._add, OperationalBasis.references,
    OperationalBasis.audit, getattr(OperationalBasis, "__getattr__", None),
    vars(OperationalBasis).get("columns"), vars(OperationalBasis).get("components"),
    vars(OperationalBasis).get("positions"), vars(OperationalBasis).get("factors"),
    vars(OperationalBasis).get("retained"),
    _witness_values.HeadRows, _witness_values.OperationalBasis,
)


def _fit_generated() -> bool:
    """Only unchanged package generators may omit replay of their own exact facts.

    This selects code, not a data validity flag or a cached verdict. A changed
    generator or validation hook retains the original checked construction.
    """
    if (HeadRecipe is not _HEAD_RECIPE_CLASS
            or FactoredRows is not _FACTORED_ROWS_CLASS
            or CertificateBasis is not _OPERATIONAL_BASIS_CLASS
            or HeadRows is not _OPERATIONAL_HEAD_CLASS
            or OperationalBasis is not _OPERATIONAL_DATA_CLASS):
        return False
    current = (
        components, _indicator_rank_prepared, _references, _columns, _distinct,
        _proposed_echelon, _primitive, _over, _reduce_by, _new_tail, _add,
        _witness_row, _fit_maps, _fit_echelon, _fit_column_maps, _fit_dependent,
        _valid_elimination, _fit_add, _canonical,
        _echelon_rows, _fit_row_maps, _EchelonRows,
        _fit_witnesses, _fit_coordinates, _compact_witnesses,
        _fit_component_sums, _fit_reference_block, _witness_proof,
        WitnessProof, WitnessBlock, WitnessProof.witness, WitnessProof.materialize,
        WitnessProof.__new__, WitnessBlock.__new__,
        WitnessBlock.witness, WitnessBlock.materialize,
        WitnessBlock.iter_witnesses,
        _primitive_head, _first_rows, _indexed_echelon,
        _sparse_witness, _proof_bytes, SparseValues, SparseValues.__new__,
        SparseValues.items, SparseValues.canonical, SparseValues.__getattribute__,
        WitnessContext, WitnessContext.__new__, WitnessContext._add,
        WitnessContext.references, WitnessContext.__getattribute__,
        WitnessProof.materialize_echelon, WitnessBlock.iter_selected,
        CertificateBasis.__getattribute__, getattr(CertificateBasis, "__getattr__", None),
        vars(CertificateBasis).get("witnesses"), vars(CertificateBasis).get("echelon"),
        _head_work, _factored_echelon,
        HeadRecipe, HeadRecipe.__new__, HeadRecipe.__getattribute__,
        HeadRecipe.at, HeadRecipe.expanded,
        FactoredRows, FactoredRows.__new__, FactoredRows.__getattribute__,
        FactoredRows.dependent, SparseValues.at, WitnessContext.dependent,
        _witness_values.HeadRecipe, _witness_values.FactoredRows,
        getattr(HeadRecipe, "__getattr__", None),
        vars(HeadRecipe).get("pivot"), vars(HeadRecipe).get("anchor"),
        vars(HeadRecipe).get("birth"),
        getattr(FactoredRows, "__getattr__", None),
        vars(FactoredRows).get("heads"), vars(FactoredRows).get("tail_columns"),
        vars(FactoredRows).get("tails"), vars(FactoredRows).get("tail_positions"),
        _operational_echelon, _operational_basis_prepared, _operational_data,
        _operational_rows, _native_tails,
        _export_operational_basis, _rebind_operational_basis,
        HeadRows, HeadRows.__new__, HeadRows.__getattribute__, HeadRows.anchor,
        getattr(HeadRows, "__getattr__", None),
        vars(HeadRows).get("count"), vars(HeadRows).get("first"),
        vars(HeadRows).get("codes"), vars(HeadRows).get("columns"),
        OperationalBasis, OperationalBasis.__new__, OperationalBasis.__getattribute__,
        OperationalBasis.dependent, OperationalBasis._add, OperationalBasis.references,
        OperationalBasis.audit, getattr(OperationalBasis, "__getattr__", None),
        vars(OperationalBasis).get("columns"), vars(OperationalBasis).get("components"),
        vars(OperationalBasis).get("positions"), vars(OperationalBasis).get("factors"),
        vars(OperationalBasis).get("retained"),
        _witness_values.HeadRows, _witness_values.OperationalBasis,
    )
    return all(a is b for a, b in zip(current, _FIT_GENERATORS, strict=True))


_PAIR_COMPONENTS = Components
_PAIR_BLOCK = Block
_PAIR_COUNT_GENERATORS = (
    _indicator_rank_counted, _connectivity._rank_from_counts,
    _connectivity._rank_from_pairs, _connectivity.components,
    Components, Components.__new__, Components.__init__, Components.__post_init__,
    Components.__getattribute__, getattr(Components, "__getattr__", None),
    Components.__dict__["n_components"],
    _connectivity._aligned, _connectivity._offsets, _connectivity._vertex_labels,
    _connectivity._vectors, _connectivity._index, _groups.frozen_index,
    Block, Block.__getattribute__, getattr(Block, "__getattr__", None),
    Block.__dict__["n_rows"], Block.__dict__["n_levels"],
)


def _fit_pair_counts_generated() -> bool:
    """Only unchanged omitted pair-record operations may use the count-only route.

    Keep _fit_generated's original scope: a changed count-only hook must run
    the old rank construction, not trigger unrelated canonical-proof replay.
    """
    if not _fit_generated():
        return False
    if (Components is not _PAIR_COMPONENTS
            or _connectivity.Components is not _PAIR_COMPONENTS
            or Block is not _PAIR_BLOCK):
        return False
    current = (
        _indicator_rank_counted, _connectivity._rank_from_counts,
        _connectivity._rank_from_pairs, _connectivity.components,
        Components, Components.__new__, Components.__init__, Components.__post_init__,
        Components.__getattribute__, getattr(Components, "__getattr__", None),
        Components.__dict__["n_components"],
        _connectivity._aligned, _connectivity._offsets, _connectivity._vertex_labels,
        _connectivity._vectors, _connectivity._index, _groups.frozen_index,
        Block, Block.__getattribute__, getattr(Block, "__getattr__", None),
        Block.__dict__["n_rows"], Block.__dict__["n_levels"],
    )
    return all(a is b for a, b in zip(current, _PAIR_COUNT_GENERATORS, strict=True))


def _fit_basis_prepared(
    grouping: Grouping, parts: Components, rank: IndicatorRank,
    references: tuple[IndexArray, ...], columns: tuple[IndexArray, ...],
    *, distinct: IndexArray | None = None, first: IndexArray | None = None,
    _owner: FESample | None = None,
    _native: ModuleType | None = None, _trace: list | None = None,
) -> CertificateBasis | None:
    """Fresh exact construction; replay only foreign generator/hook results.

    The unchanged integer generator maintains canonical primitive pivots in input
    row order. Its exact reduced rows directly give R_J^-1 R and hence all column
    maps. Re-proving those same generated facts is development verification, not
    changing numerical rank/support: the live rank bounds remain below, and
    positive-curvature support remains a separate fresh consumer check.
    """
    count = _count(columns)
    generated = _fit_generated()
    if first is not None and not generated:
        return None
    operational = False
    if (generated and first is not None and type(_owner) is FESample
            and vars(_owner).get("grouping") is grouping and rank.exact != count
            and vars(_OPERATIONAL_BASIS_CLASS).get("__dict__") is _OPERATIONAL_BASIS_DICT
            and CertificateBasis.__init__ is _OPERATIONAL_BASIS_INIT
            and CertificateBasis.__new__ is _OPERATIONAL_BASIS_NEW):
        try:
            operational = (_packed_original is _PACKED_GUARD and _packed_original())
        except MemoryError:
            pass  # Optional guard only, never exact operational reductions.
    if operational:
        actual = _operational_basis_prepared(
            _owner, grouping, parts, rank, references, columns, first,
            _native=_native, _trace=_trace,
        )
        if actual is not False:
            return actual
        if _trace is not None:
            _trace[0], _trace[2] = "python", None
            reason = _trace[3]
            _trace[3] = ("storage" if reason is None or reason == "ineligible"
                         else reason + "; storage")
    pivots: dict[int, Row] = {}
    tails: dict[int, Reduced] = {}
    rows: _EchelonRows | None = None
    packed = False
    later: tuple[int, ...] = ()
    if rank.exact == count:
        route: Route = "connectivity"
        kept, echelon = list(range(count)), ()
    elif generated:
        rows = (
            _echelon_rows(
                columns, cast("IndexArray", distinct),
                WORK * (cast("IndexArray", distinct).size + count),
                compact=True,
            ) if first is None else _indexed_echelon(
                grouping, columns, first, WORK * (first.size * len(columns) + count),
            )
        )
        if rows is None:
            return None
        packed = type(rows.heads) is PackedHeads
        if packed:
            later = tuple(sorted(rows.later))
            kept = range(rows.heads.count + len(later))
            echelon = rows.heads
        else:
            kept = sorted(rows.heads)
            kept.extend(sorted(rows.later))
            echelon = tuple(
                rows.heads[p] if p in rows.heads else rows.later[p]
                if type(rows.later[p]) is SparseValues else
                tuple(sorted(cast("Row", rows.later[p]).items()))
                for p in kept
            )
        route = "elimination"
    else:
        proposed = _proposed_echelon(
            columns, cast("IndexArray", distinct),
            WORK * (cast("IndexArray", distinct).size + count),
        )
        if proposed is None:
            return None
        pivots, tails = proposed
        parsed = _fit_echelon(pivots, count)
        if parsed is None:
            return None
        kept, echelon = parsed
        route = "elimination"
    if not rank.lower <= len(kept) <= rank.upper:
        return None
    place = ({c: rows.heads.count + i for i, c in enumerate(later)} if packed else
             {c: i for i, c in enumerate(kept)})
    of_column = (
        {} if route == "connectivity" else
        {} if packed else {} if rows is not None and all(
            type(row) is HeadRecipe for row in rows.heads.values()
        ) else
        _fit_row_maps(rows, place, count) if rows is not None else
        _fit_maps(pivots, tails, place, count)
    )
    if of_column is None:
        return None
    if not generated:
        checked = _fit_column_maps(pivots, kept, place, count, of_column)
        if checked is None:
            return None
        if route == "elimination" and not _valid_elimination(
            pivots, kept, place, columns, distinct=cast("IndexArray", distinct),
            dep_inv=checked[0], dep_scale=checked[1],
        ):
            return None
    if packed:
        selected, positions = _packed_coordinates(columns, rows.heads.count, later, count)
    elif generated:
        selected, positions = _fit_coordinates(columns, kept, count)
    else:
        selected = frozen_index(np.asarray(kept, dtype=np.intp))
        positions = tuple(
            frozen_index(np.array([place.get(int(c), -1) for c in col], dtype=np.intp))
            for col in columns
        )
    proof = (
        _compact_witnesses(
            grouping, parts, (columns, positions), place, of_column,
            echelon=echelon, rows=rows,
        ) if generated else None
    )
    witnesses = (
        cast("tuple[tuple[Witness, ...], ...]", proof) if proof is not None else
        _fit_witnesses(grouping, parts, columns, place, of_column)
    )
    basis = CertificateBasis(
        grouping=grouping, route=route, rank=rank, references=references,
        columns=columns, selected=selected,
        positions=positions, witnesses=witnesses,
        echelon=(cast("tuple[tuple[tuple[int, int], ...], ...]", proof)
                 if proof is not None else echelon),
        rows=int(first.size if first is not None
                 else cast("IndexArray", distinct).shape[0]),
    )
    if (packed and proof is not None and proof.retained is not None
            and type(_owner) is FESample and vars(_owner).get("grouping") is grouping):
        try:
            headers = _retained_headers(basis, grouping)
            scope = (ref(basis), ref(grouping), ref(_owner))

            def construction():
                return scope, proof, headers

            object.__setattr__(basis, _GENERATED_RETAINED, construction)
        except (MemoryError, TypeError):
            pass  # Optional issuance only; the proof/full action remains valid.
    return basis


_FIT_PREPARED_ISSUER = _fit_basis_prepared


def _record_basis_backend(basis: CertificateBasis, trace: list | None) -> CertificateBasis:
    """Store only scalar execution facts on this freshly produced exact basis."""
    if trace is not None and type(basis) is CertificateBasis:
        trace[4] = True  # Required truthful recording must not retry the calculation.
        object.__setattr__(basis, "_basis_backend", (*trace[:4], trace[5]))
        trace[4] = False
    return basis


def _prepare_fit_basis(
    owner: FESample, *, basis_backend: str = "python", _native: ModuleType | None = None,
) -> CertificateBasis:
    """One fresh canonical producer scoped by the unchanged NumericExecution guards.

    Component membership proves D = E M. Unchanged exact generators construct
    primitive pivots, span and rational maps in original row order; changed hooks
    keep their original checks before canonical witness emission. No caller-held
    basis/flag or serialized _Held is consumed on private success.
    Public/foreign owned_basis and the original bounded fallback remain unchanged.
    """
    native = (_native if type(_native) is ModuleType else
              _resolve_native_echelon() if type(basis_backend) is str
              and basis_backend == "cpp" else None)
    trace = None if native is None else ["python", None, None, "ineligible", False, False]
    grouping = owner.grouping
    if not _fit_grouping(grouping):
        found = owned_basis(owner)
        return found if trace is None else _record_basis_backend(found, trace)
    assert grouping is not None
    try:
        parts = components(grouping.blocks)
        of_rows = [part[block.codes] for part, block in
                   zip(parts.of_levels, grouping.blocks, strict=True)]
        if all(np.array_equal(of_rows[0], found) for found in of_rows):
            del of_rows
            rank = (
                _indicator_rank_counted(grouping.blocks, parts)
                if _fit_pair_counts_generated() else
                _indicator_rank_prepared(grouping.blocks, parts)
            )
            references = _references(grouping, parts)
            columns = _columns(grouping, references)
            if _fit_generated():
                first = _first_rows(grouping)
                if _fit_basis_prepared is _FIT_PREPARED_ISSUER:
                    basis = _fit_basis_prepared(
                        grouping, parts, rank, references, columns,
                        first=first, _owner=owner,
                        _native=native, _trace=trace,
                    )
                else:
                    basis = _fit_basis_prepared(
                        grouping, parts, rank, references, columns, first=first,
                    )
            else:
                basis = _fit_basis_prepared(
                    grouping, parts, rank, references, columns,
                    distinct=_distinct(grouping),
                )
            if basis is not None:
                return basis if trace is None else _record_basis_backend(basis, trace)
    except Exception:  # same proposal-only refusal/fallback as _admitted_proposal
        if trace is not None and trace[4]:
            raise  # An explicit native failure is never a Python fallback.
        pass
    if trace is not None:
        trace[0], trace[2] = "python", None
        if trace[3] is None:
            trace[3] = "canonical_fallback"
    found = certificate_basis(grouping)
    return found if trace is None else _record_basis_backend(found, trace)


def _admitted_proposal(grouping: Grouping, content: Content) -> CertificateBasis | None:
    """An owned basis of the proposal's values, when :func:`_valid` admits them.

    Rebuild only the checked snapshot, as on reuse: serialization can erase a
    proposal's array shape/dtype, scalar types or grouping, which are not admitted.
    ``None`` otherwise, and then :func:`certificate_basis` constructs or refuses.
    """
    try:
        proposal = _proposal(grouping)
        if proposal is not None:
            held = _held(proposal, id(grouping), content)
            if _valid(held, grouping, content):
                return _rebuilt(held, grouping)
    except Exception:  # a failed proposal has no standing at all
        return None
    return None


def owned_basis(owner: object) -> CertificateBasis:
    """``certificate_basis(owner.grouping)``, constructed once for ``owner``.

    The values held by ``owner`` are used only for the very grouping object they were
    constructed from, when fresh validation against ``owner.grouping`` proves they
    equal the exact canonical construction on every field, and then as a new basis;
    otherwise the basis is constructed, and kept when the content was unchanged by the
    construction (module docstring).  The construction is a proposal admitted by the
    same validation, or else :func:`certificate_basis`.  The refusals are those of
    :func:`certificate_basis`; a refused construction keeps nothing.
    """
    grouping: object = getattr(owner, "grouping", None)
    store = getattr(owner, "__dict__", None)
    content = _content(grouping)
    if content is None or type(store) is not dict:
        return certificate_basis(grouping)
    # Non-None content follows _content's fresh Grouping and storage checks.
    grouping = cast(Grouping, grouping)
    held = store.get(HELD)
    if type(held) is _Held and _valid(held, grouping, content):
        basis = _rebuilt(held, grouping)
        admitted = True
    else:
        basis = _admitted_proposal(grouping, content)
        admitted = basis is not None
        if basis is None:
            basis = certificate_basis(grouping)
        if _content(grouping) == content:
            held = _held(basis, id(grouping), content)
            object.__setattr__(owner, HELD, held)
        else:
            admitted = False
    # Only this function has both freshly read input bytes and the actual
    # canonical-validation result.  A caller cannot issue a reader by supplying
    # an allegedly validated _Held to a separate sealing helper.
    # _valid checks exact primitive types throughout the content, rank, maps,
    # echelon and every unit/dependent/reference witness, not just equality.
    if admitted and type(owner) is FESample and type(grouping) in (Grouping, Derived):
        construction_values = cast(_Held, held)
        scope = (ref(basis), ref(grouping), ref(owner))

        def construction() -> tuple[Scope, _Held]:
            return scope, construction_values

        object.__setattr__(basis, CONSTRUCTION, construction)
    return basis


Scope = tuple[
    ReferenceType[CertificateBasis], ReferenceType[Grouping], ReferenceType[FESample]
]

_GENERATED_RETAINED = "_generated_retained_rows"


def _retained_header(array: object):
    """A weak current header plus its actual complete immutable byte owner."""
    if (type(array) is not np.ndarray or array.dtype != np.intp or array.ndim != 1
            or array.flags.writeable or not array.flags.c_contiguous):
        return None
    root = array
    while type(root) is np.ndarray:
        root = root.base
    if type(root) is not bytes or len(root) != array.nbytes:
        return None
    return ref(array), root, array.shape, array.strides, array.dtype.str


def _retained_headers(basis: CertificateBasis, grouping: Grouping):
    """Freeze only headers/weak identities, never a sample/provider graph."""
    values = vars(basis)
    fields = tuple((name, values[name]) for name in
                   ("route", "rank", "rows", "witnesses", "echelon"))
    arrays = []
    for name in ("selected", "references", "columns", "positions"):
        given = (values[name],) if name == "selected" else values[name]
        if type(given) is not tuple:
            return None
        headers = tuple(_retained_header(a) for a in given)
        if any(h is None for h in headers):
            return None
        arrays.append((name, headers))
    group = vars(grouping)
    blocks = group.get("blocks")
    if type(blocks) is not tuple:
        return None
    block_headers = tuple((ref(b), _retained_header(vars(b).get("codes")),
                           _retained_header(vars(b).get("counts"))) for b in blocks)
    position = _retained_header(group.get("positions"))
    if position is None or any(c is None or n is None for _, c, n in block_headers):
        return None
    return fields, tuple(arrays), position, block_headers


def _matching_retained_header(array: object, header) -> bool:
    if header is None or header[0]() is not array:
        return False
    current = _retained_header(array)
    return (current is not None and current[1] is header[1]
            and current[2:] == header[2:])


def _matching_retained_headers(basis: CertificateBasis, grouping: Grouping, headers) -> bool:
    if headers is None or type(basis) is not CertificateBasis:
        return False
    if type(grouping) is not Grouping and type(grouping) is not Derived:
        return False
    fields, arrays, position, blocks = headers
    values, group = vars(basis), vars(grouping)
    if any(values.get(name) is not expected for name, expected in fields):
        return False
    for name, expected in arrays:
        given = (values.get(name),) if name == "selected" else values.get(name)
        if (type(given) is not tuple or len(given) != len(expected)
                or not all(_matching_retained_header(a, h)
                           for a, h in zip(given, expected, strict=True))):
            return False
    current = group.get("blocks")
    return (type(current) is tuple and len(current) == len(blocks)
            and _matching_retained_header(group.get("positions"), position)
            and all(scope() is b and _matching_retained_header(vars(b).get("codes"), c)
                    and _matching_retained_header(vars(b).get("counts"), n)
                    for b, (scope, c, n) in zip(current, blocks, strict=True)))


def _produced_retained_action(basis: object, owner: object):
    """Read only the actual fresh producer's literal weak-scoped reader."""
    if type(basis) is not CertificateBasis or type(owner) is not FESample:
        return None
    reader = vars(basis).get(_GENERATED_RETAINED)
    if (type(reader) is not FunctionType or not _fit_generated()
            or _packed_original is not _PACKED_GUARD or not _packed_original()
            or (reader.__code__ is not _construction_code(_FIT_PREPARED_ISSUER)
                and reader.__code__ is not _construction_code(_OPERATIONAL_PREPARED_ISSUER))):
        return None
    scope, proof, headers = reader()
    grouping = vars(owner).get("grouping")
    if (scope[0]() is not basis or scope[1]() is not grouping or scope[2]() is not owner
            or vars(basis).get("grouping") is not grouping
            or (vars(basis).get("_operational") if type(proof) is OperationalBasis
                else vars(basis).get("witnesses")) is not proof
            or not _matching_retained_headers(basis, grouping, headers)):
        return None
    return proof, headers


def _construction_code(issuer: object) -> CodeType:
    """The package's literal reader code, with no registry of live constructions."""
    return next(
        code
        for code in cast(FunctionType, issuer).__code__.co_consts
        if type(code) is CodeType and code.co_name == "construction"
    )


_PACK_TYPES = (PackedHeads, RetainedRows, Fraction, Grouping, Derived, Block, FESample)
_PACK_HELPERS = (
    _packed_echelon, _packed_head_work, _retained_rows, _packed_coordinates,
    _fit_basis_prepared, _retained_header, _retained_headers,
    _matching_retained_header, _matching_retained_headers, _produced_retained_action,
    _construction_code, np.empty, np.frombuffer, np.flatnonzero, np.count_nonzero,
    np.arange, np.asarray, np.concatenate, np.intp, np.ndarray,
)
_PACK_METHODS = tuple((kind, name, vars(kind).get(name)) for kind, names in (
    (PackedHeads, ("__new__", "__getattribute__", "__getattr__", "head",
                   "count", "offsets", "anchors", "births", "versions", "later")),
    (RetainedRows, ("__new__", "__getattribute__", "__getattr__", "minimum", "kept_levels",
                   "kept_positions", "levels", "terms", "norm")),
    (Fraction, ("__new__", "__getattribute__", "__getattr__", "__add__", "__radd__",
                "__sub__", "__rsub__", "__mul__", "__rmul__", "__neg__", "__abs__",
                "numerator", "denominator")),
    (Grouping, ("__getattribute__", "__getattr__", "__dict__")),
    (Derived, ("__getattribute__", "__getattr__", "__dict__")),
    (Block, ("__getattribute__", "__getattr__", "__dict__")),
    (FESample, ("__getattribute__", "__getattr__", "__dict__", "grouping")),
) for name in names)


def _packed_original() -> bool:
    """Only optional packed preparation; the existing generated predicate is intact."""
    kinds = (PackedHeads, RetainedRows, Fraction, Grouping, Derived, Block, FESample)
    if any(a is not b for a, b in zip(kinds, _PACK_TYPES, strict=True)):
        return False
    current = (
        _packed_echelon, _packed_head_work, _retained_rows, _packed_coordinates,
        _fit_basis_prepared, _retained_header, _retained_headers,
        _matching_retained_header, _matching_retained_headers, _produced_retained_action,
        _construction_code, np.empty, np.frombuffer, np.flatnonzero, np.count_nonzero,
        np.arange, np.asarray, np.concatenate, np.intp, np.ndarray,
    )
    return (all(a is b for a, b in zip(current, _PACK_HELPERS, strict=True))
            and all(vars(kind).get(name) is original for kind, name, original in _PACK_METHODS))


_PACKED_GUARD = _packed_original


def _construction(basis: object, owner: object) -> _Held | None:
    """Read an authentic construction scoped to this basis, grouping and sample."""
    if type(basis) is not CertificateBasis or type(owner) is not FESample:
        return None
    reader = vars(basis).get(CONSTRUCTION)
    if type(reader) is not FunctionType:
        return None
    if (
        reader.__code__ is not _construction_code(owned_basis)
        and reader.__code__ is not _construction_code(rebound_basis)
    ):
        return None
    scope, held = reader()
    if (
        scope[0]() is not basis
        or scope[1]() is not basis.grouping
        or scope[2]() is not owner
        or vars(owner).get("grouping") is not basis.grouping
    ):
        return None
    return cast(_Held, held)


def rebound_basis(
    basis: CertificateBasis,
    grouping: Grouping,
    *,
    source_owner: object,
    target_owner: object,
) -> CertificateBasis:
    """Bind a within-fit basis to the handoff's value-matched live grouping.

    Parameters
    ----------
    basis : CertificateBasis
        The certificate's original basis, after the handoff's full capture check.
    grouping : Grouping
        The live grouping at the corresponding admitted anchor.
    source_owner : object
        The original certificate sample, scoped by the authenticated reader.
    target_owner : object
        The sample at the handoff's already value-matched live anchor.

    Returns
    -------
    CertificateBasis
        The original field values with the live grouping.  An authentic
        construction is carried only if fresh complete grouping bytes also match.
        No reader is fabricated for an ineligible or changed input.
    """
    found = replace(basis, grouping=grouping)
    held = _construction(basis, source_owner)
    if (
        held is not None
        and type(target_owner) is FESample
        and vars(target_owner).get("grouping") is grouping
        and type(grouping) in (Grouping, Derived)
        and _content(grouping) == held.content
    ):
        rebound = held._replace(grouping=id(grouping))
        scope = (ref(found), ref(grouping), ref(target_owner))

        def construction() -> tuple[Scope, _Held]:
            return scope, rebound

        object.__setattr__(found, CONSTRUCTION, construction)
    return found


def retention_basis(owner: object, supplied: object) -> CertificateBasis:
    """Obtain the canonical expectation at the existing retention check site.

    Parameters
    ----------
    owner : object
        The admitted sample whose grouping is being consumed now.
    supplied : object
        The same-fit certificate basis.  Its mutable fields are not evidence.

    Returns
    -------
    CertificateBasis
        A fresh-owned expectation from authenticated immutable values, after a
        full current-content comparison, or the unchanged general construction.
        The caller must still compare every supplied basis field against it.
    """
    # Do not evaluate caller descriptors to decide eligibility.  The general
    # route retains their original evaluation/refusal order.
    if type(owner) is not FESample:
        return owned_basis(owner)
    grouping = vars(owner).get("grouping")
    if type(grouping) in (Grouping, Derived):
        held = _construction(supplied, owner)
        if (
            held is not None
            and cast(CertificateBasis, supplied).grouping is grouping
            and _content(grouping) == held.content
        ):
            return _rebuilt(held, grouping)
    return owned_basis(owner)


def positive_support(basis: CertificateBasis, positive: BoolArray) -> SupportCheck:
    """The exact rank of ``E_J`` on the rows where ``positive`` holds.

    ``ValueError`` for a mask that is not one boolean per row of the grouping.
    """
    mask = np.asarray(positive)
    if mask.dtype != np.bool_ or mask.shape != (basis.grouping.n_rows,):
        raise ValueError("positive must be one boolean per row of the grouping")
    rows, kept = int(np.count_nonzero(mask)), basis.n_kept
    if mask.all():
        return SupportCheck(rows, basis.rows, kept, kept, False)
    if not rows:
        return SupportCheck(0, 0, 0, kept, False)
    distinct = _distinct(basis.grouping, mask)
    pivots, _ = eliminate(_row_columns(row, basis.positions) for row in distinct)
    return SupportCheck(rows, int(distinct.shape[0]), len(pivots), kept, True)


# ------------------------------------------------------------------------ maps


def embed(basis: CertificateBasis, values: FloatArray) -> tuple[FloatArray, ...]:
    """The coefficients of ``J`` (one row per coordinate) at their levels."""
    found: list[FloatArray] = []
    for position in basis.positions:
        level = np.zeros((position.size, *values.shape[1:]))
        kept = position >= 0
        level[kept] = values[position[kept]]
        found.append(immutable_array(level))
    return tuple(found)


def lift(basis: CertificateBasis, embedded: tuple[FloatArray, ...]) -> FloatArray:
    """``sum_k a_k[c_k]`` by row: the gathers added in block order, checked."""
    blocks = basis.grouping.blocks
    total = embedded[0][blocks[0].codes]
    for block, values in zip(blocks[1:], embedded[1:], strict=True):
        total = checked_sum(total, values[block.codes], "lifted correction")
    return immutable_array(total)
