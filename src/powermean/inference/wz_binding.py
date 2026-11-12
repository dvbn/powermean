"""Declared three-way PPML provenance and bounded current rows (03.5.4, 04.7).

Labels are read from the actual formula's owned blocks, never recoded to satisfy
an adapter. Equal partitions cannot substitute for the requested source name.
Only exact int/str labels on balanced directed panels enter: complete N^2T
support with every self-pair (P=N^2) or off-diagonal N(N-1)T support with none
(P=N(N-1)). The layout carries that admitted actual P; partial diagonals refuse.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.inference.wz_operator import IndexArray, Reservation, Rows, _plan
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.absorbed import AbsorbedOptions

type Label = str | int


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


def unavailable(case: str) -> SpecificationError:
    """Refuse an unsupported WZ request explicitly, without fallback."""
    return SpecificationError(
        "specification.unsupported_inference",
        {"method": "WZ pair covariance", "case": case},
    )


@dataclass(frozen=True)
class WZSource:
    """Lossless declared term provenance; every call rereads current owned labels."""

    terms: tuple[tuple[str, ...], ...]
    grouping: SourceGrouping
    row_labels: tuple[object, ...]

    def components(self, name: str) -> tuple[tuple[Label, ...], ...]:
        """Recognize two interactions sharing a date from their declared terms."""
        terms, blocks = self.terms, self.grouping.blocks
        if len(terms) != 3 or tuple(b.name for b in blocks) != tuple(
            "^".join(term) for term in terms
        ):
            raise unavailable("exact declared three-way FE source required")
        matches = [i for i, b in enumerate(blocks) if b.name == name]
        if len(matches) != 1:
            raise unavailable("cluster must name the actual declared pair FE source")
        pair = matches[0]
        indices = [i for i in range(3) if i != pair]
        first, second = (terms[i] for i in indices)
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
        a, ta = _interaction(blocks[indices[0]], first.index(origin))
        b, tb = _interaction(blocks[indices[1]], second.index(destination))
        if ta != tb:
            raise unavailable("shared date source labels differ")
        _pair_source(blocks[pair], tuple(zip(a, b, strict=True)), pair_term, origin)
        return a, b, ta


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


def _pair_source(
    block: SourceBlock,
    pairs: tuple[tuple[Label, Label], ...],
    term: tuple[str, ...],
    origin: str,
) -> None:
    levels = block.levels
    forward: dict[object, tuple[Label, Label]] = {}
    reverse: dict[tuple[Label, Label], object] = {}
    for code, key in zip(block.codes, pairs, strict=True):
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


@dataclass(frozen=True)
class WZSupport:
    """One call's recognized balanced cells; never reused as admission."""

    nodes: tuple[Label, ...]
    dates: tuple[Label, ...]
    pairs: int


def support(
    origins: tuple[Label, ...],
    destinations: tuple[Label, ...],
    dates: tuple[Label, ...],
) -> WZSupport:
    """Recognize exactly balanced complete N^2T or off-diagonal N(N-1)T cells.

    The row count selects the only candidate P, N^2 or N(N-1), on the common
    node and date sets. Unique cells inside those sets fill that candidate
    exactly, so no N*N*T grid is built; off-diagonal support has no self-pair.
    A count alone never admits: partial or mixed diagonals, repeated or missing
    cells refuse. Layout and the coefficient pair-role search share this path.
    """
    nodes, times = tuple(dict.fromkeys(origins)), tuple(dict.fromkeys(dates))
    nn, tt, n = len(nodes), len(times), len(origins)
    pairs = nn * nn if n == nn * nn * tt else nn * (nn - 1)
    if nn < 3 or tt < 2 or n != pairs * tt or set(nodes) != set(destinations):
        raise unavailable(
            "balanced complete N^2T or off-diagonal N(N-1)T directed panel, "
            "N>=3 and T>=2 required"
        )
    self_pairs = pairs != nn * nn and any(
        i == j for i, j in zip(origins, destinations, strict=True)
    )
    if self_pairs or len(set(zip(origins, destinations, dates, strict=True))) != n:
        raise unavailable("partial diagonal, repeated or incomplete pair-time cells")
    return WZSupport(nodes, times, pairs)


@dataclass(frozen=True)
class WZLayout:
    """Per-call panel indices and whole-call reservation; never reused as admission.

    pairs is the admitted actual P: N^2 on complete, N(N-1) on off-diagonal support.
    """

    origins: IndexArray
    destinations: IndexArray
    dates: IndexArray
    order: IndexArray
    nodes: int
    periods: int
    pairs: int
    plan: Reservation


def reserve_validation(shape: tuple[int, int], bounds: AttemptBounds) -> None:
    """Reserve original source validation before reading/copying source arrays."""
    n, k = shape
    Reservation(8192 * n * (k + 1), 4096 * n * (k + 1), n, k, 0, 0).require(bounds)


def layout(
    source: WZSource,
    name: str,
    shape: tuple[int, int],
    absorbed: AbsorbedOptions,
    bounds: AttemptBounds,
    *,
    extra: tuple[int, int] = (0, 0),
    workspace: Callable[[int, int, int, int], tuple[int, int]] | None = None,
) -> WZLayout:
    """Reserve validation first, then the accepted full operator before snapshots."""
    n, k = shape
    reserve_validation(shape, bounds)
    a, b, dates = source.components(name)
    if len(a) != n or len(source.row_labels) != n:
        raise unavailable("source labels must align with every current row")
    tuple(map(_label, source.row_labels))
    if k < 1:
        raise unavailable("at least one current slope, K>=1, required")
    admitted = support(a, b, dates)
    nn, tt = len(admitted.nodes), len(admitted.dates)
    added = (0, 0) if workspace is None else workspace(n, k, nn, tt)
    plan = _plan(
        n,
        k,
        (2 * nn - 1) * (tt - 1),
        tt,
        absorbed.absorb_max_sweeps,
        extra_bytes=extra[0] + added[0],
        extra_work=extra[1] + added[1],
    )
    plan.require(bounds)
    ni, ti = (
        dict(zip(admitted.nodes, range(nn), strict=True)),
        dict(zip(admitted.dates, range(tt), strict=True)),
    )
    i = np.fromiter((ni[v] for v in a), dtype=np.intp, count=n)
    j = np.fromiter((ni[v] for v in b), dtype=np.intp, count=n)
    t = np.fromiter((ti[v] for v in dates), dtype=np.intp, count=n)
    return WZLayout(
        i, j, t, np.lexsort((t, j, i)).astype(np.intp), nn, tt, admitted.pairs, plan
    )


def capture_rows(
    panel: WZLayout,
    x: FloatArray,
    y: FloatArray,
    mu: FloatArray,
    names: tuple[str, ...],
    *,
    route: str,
) -> Rows:
    """Own finite current numerical inputs after the caller's complete reservation.

    Pair totals, the reshape and Rows.pairs all use the layout's admitted P.
    """
    n, k = x.shape
    if (
        y.shape != (n,)
        or mu.shape != (n,)
        or len(names) != k
        or panel.order.shape != (n,)
    ):
        raise ValueError("WZ current rows/names are not aligned")
    for array in (x, y, mu):
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise FloatingPointError("WZ requires finite float64 current arrays")
    if np.any(mu <= 0) or np.any(y < 0):
        raise unavailable(
            "strictly positive representable fitted means and nonnegative y required"
        )
    if np.any(y[panel.order].reshape(panel.pairs, panel.periods).sum(axis=1) <= 0):
        raise unavailable("all pairs must have positive outcome totals; no trimming")
    digest = hashlib.sha256()
    for bound_array in (x, y, mu, panel.origins, panel.destinations, panel.dates):
        digest.update(bound_array.tobytes())
    return Rows(
        FloatStorage.store(x),
        FloatStorage.store(y),
        FloatStorage.store(mu),
        IndexStorage.store(panel.origins),
        IndexStorage.store(panel.destinations),
        IndexStorage.store(panel.dates),
        IndexStorage.store(panel.order),
        panel.nodes,
        panel.periods,
        panel.pairs,
        names,
        digest.hexdigest(),
        route,
    )
