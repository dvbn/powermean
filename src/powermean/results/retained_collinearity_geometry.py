"""Detached snapshots of collinearity factorizations, span tests, and relations.

This module detaches ``Factorization``, ``SpanTest``, ``Relation``, and ``WithinRank``
into explicit finite typed records with fresh owned arrays on every access.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.data.collinearity_geometry import (
    Relation,
    SpanTest,
    WithinRank,
)
from powermean.data.collinearity_numeric import Factorization
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.owned import owned_array
from powermean.results.retained_input_values import _tuple_arrays

__all__ = [
    "RetainedFactorAttempt",
    "RetainedRelationCheck",
    "RetainedSpanTest",
    "RetainedWithinColumns",
    "_detach_factorization",
    "_detach_relation",
    "_detach_span_test",
    "_detach_within",
]


@dataclass(frozen=True)
class RetainedFactorAttempt:
    """Detached snapshot of ``Factorization``."""

    kind: str
    purpose: str
    shape: tuple[int, int]
    outcome: str
    _values: FloatArray
    cutoff: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        """Own stored numeric array."""
        object.__setattr__(self, "_values", owned_array(self._values))

    @classmethod
    def from_factorization(cls, f: Factorization) -> RetainedFactorAttempt:
        """Detach a ``Factorization`` record."""
        if type(f) is not Factorization:
            raise TypeError("f must be a concrete Factorization")
        return cls(
            kind=str(f.kind),
            purpose=str(f.purpose),
            shape=(int(f.shape[0]), int(f.shape[1])),
            outcome=str(f.outcome),
            _values=owned_array(f.values),
            cutoff=None if f.cutoff is None else float(f.cutoff),
            rank=None if f.rank is None else int(f.rank),
        )

    @property
    def values(self) -> FloatArray:
        """Return a fresh owned copy of ``values``."""
        return owned_array(self._values)


def _detach_factorization(f: Factorization) -> RetainedFactorAttempt:
    """Detach a ``Factorization`` into :class:`RetainedFactorAttempt`."""
    return RetainedFactorAttempt.from_factorization(f)


@dataclass(frozen=True)
class RetainedSpanTest:
    """Detached snapshot of ``SpanTest`` with owned arrays on every access."""

    basis: int
    components: int
    _scales: FloatArray
    cutoff: float
    _singular_b: FloatArray
    _singular_h: FloatArray
    rank_b: int
    rank_h: int

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_scales", owned_array(self._scales))
        object.__setattr__(self, "_singular_b", owned_array(self._singular_b))
        object.__setattr__(self, "_singular_h", owned_array(self._singular_h))

    @classmethod
    def from_span(cls, s: SpanTest) -> RetainedSpanTest:
        """Detach a ``SpanTest`` record."""
        if type(s) is not SpanTest:
            raise TypeError("s must be a concrete SpanTest")
        return cls(
            basis=int(s.basis),
            components=int(s.components),
            _scales=owned_array(s.scales),
            cutoff=float(s.cutoff),
            _singular_b=owned_array(s.singular_b),
            _singular_h=owned_array(s.singular_h),
            rank_b=int(s.rank_b),
            rank_h=int(s.rank_h),
        )

    @property
    def scales(self) -> FloatArray:
        """Return a fresh owned copy of ``scales``."""
        return owned_array(self._scales)

    @property
    def singular_b(self) -> FloatArray:
        """Return a fresh owned copy of ``singular_b``."""
        return owned_array(self._singular_b)

    @property
    def singular_h(self) -> FloatArray:
        """Return a fresh owned copy of ``singular_h``."""
        return owned_array(self._singular_h)

    @property
    def nested(self) -> bool:
        """Return whether adding ``U_j`` changed the rank by zero or one."""
        return self.rank_h - self.rank_b in (0, 1)

    @property
    def spanned(self) -> bool:
        """Return whether ``rank(H) == rank(B)``."""
        return self.rank_b == self.rank_h


def _detach_span_test(s: SpanTest | None) -> RetainedSpanTest | None:
    """Detach an optional ``SpanTest`` into :class:`RetainedSpanTest`."""
    return None if s is None else RetainedSpanTest.from_span(s)


@dataclass(frozen=True)
class RetainedRelationCheck:
    """Detached snapshot of ``Relation`` with owned arrays on every access."""

    basis: tuple[int, ...]
    _solved: FloatArray
    _mapped: FloatArray
    _components: tuple[FloatArray, ...]
    span: RetainedSpanTest

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_solved", owned_array(self._solved))
        object.__setattr__(self, "_mapped", owned_array(self._mapped))
        object.__setattr__(self, "_components", _tuple_arrays(self._components))

    @classmethod
    def from_relation(cls, r: Relation) -> RetainedRelationCheck:
        """Detach a ``Relation`` record."""
        if type(r) is not Relation:
            raise TypeError("r must be a concrete Relation")
        return cls(
            basis=tuple(int(b) for b in r.basis),
            _solved=owned_array(r.solved),
            _mapped=owned_array(r.mapped),
            _components=_tuple_arrays(r.components),
            span=RetainedSpanTest.from_span(r.span),
        )

    @property
    def solved(self) -> FloatArray:
        """Return a fresh owned copy of ``solved``."""
        return owned_array(self._solved)

    @property
    def mapped(self) -> FloatArray:
        """Return a fresh owned copy of ``mapped``."""
        return owned_array(self._mapped)

    @property
    def coefficients(self) -> FloatArray:
        """Compatibility alias for ``mapped``."""
        return self.mapped

    @property
    def components(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of ``components``."""
        return _tuple_arrays(self._components)


def _detach_relation(r: Relation | None) -> RetainedRelationCheck | None:
    """Detach an optional ``Relation`` into :class:`RetainedRelationCheck`."""
    return None if r is None else RetainedRelationCheck.from_relation(r)


@dataclass(frozen=True)
class RetainedWithinColumns:
    """Detached snapshot of ``WithinRank`` with owned arrays on every access."""

    active: tuple[int, ...]
    _scales: FloatArray
    gram_cutoff: float | None
    gram_rank: int | None
    _gram_diagonal: FloatArray
    _pivots: IndexArray
    cutoff: float | None
    direct_rank: int | None
    _singular_values: FloatArray
    visits: tuple[tuple[int, int, int], ...]
    kept: tuple[int, ...]
    _kept_values: FloatArray

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_scales", owned_array(self._scales))
        object.__setattr__(self, "_gram_diagonal", owned_array(self._gram_diagonal))
        object.__setattr__(self, "_pivots", owned_array(self._pivots))
        object.__setattr__(self, "_singular_values", owned_array(self._singular_values))
        object.__setattr__(self, "_kept_values", owned_array(self._kept_values))

    @classmethod
    def from_within(cls, w: WithinRank) -> RetainedWithinColumns:
        """Detach a ``WithinRank`` record."""
        if type(w) is not WithinRank:
            raise TypeError("w must be a concrete WithinRank")
        return cls(
            active=tuple(int(a) for a in w.active),
            _scales=owned_array(w.scales),
            gram_cutoff=None if w.gram_cutoff is None else float(w.gram_cutoff),
            gram_rank=None if w.gram_rank is None else int(w.gram_rank),
            _gram_diagonal=owned_array(w.gram_diagonal),
            _pivots=owned_array(w.pivots),
            cutoff=None if w.cutoff is None else float(w.cutoff),
            direct_rank=None if w.direct_rank is None else int(w.direct_rank),
            _singular_values=owned_array(w.singular_values),
            visits=tuple((int(v[0]), int(v[1]), int(v[2])) for v in w.visits),
            kept=tuple(int(k) for k in w.kept),
            _kept_values=owned_array(w.kept_values),
        )

    @property
    def scales(self) -> FloatArray:
        """Return a fresh owned copy of ``scales``."""
        return owned_array(self._scales)

    @property
    def gram_diagonal(self) -> FloatArray:
        """Return a fresh owned copy of ``gram_diagonal``."""
        return owned_array(self._gram_diagonal)

    @property
    def pivots(self) -> IndexArray:
        """Return a fresh owned copy of ``pivots``."""
        return owned_array(self._pivots)

    @property
    def singular_values(self) -> FloatArray:
        """Return a fresh owned copy of ``singular_values``."""
        return owned_array(self._singular_values)

    @property
    def kept_values(self) -> FloatArray:
        """Return a fresh owned copy of ``kept_values``."""
        return owned_array(self._kept_values)

    @property
    def omitted(self) -> tuple[int, ...]:
        """Return the active columns not kept, in formula order."""
        return tuple(j for j in self.active if j not in self.kept)


def _detach_within(w: WithinRank | None) -> RetainedWithinColumns | None:
    """Detach an optional ``WithinRank`` into :class:`RetainedWithinColumns`."""
    return None if w is None else RetainedWithinColumns.from_within(w)
