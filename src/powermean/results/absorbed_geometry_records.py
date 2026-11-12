"""Detached snapshots of FE basis, sparse matrices, factor records, and sparse measures.

This module defines :func:`_owned_sparse_matrix`, :class:`RetainedBasisSnapshot`,
:func:`snapshot_basis`, :class:`RetainedSparseMatrix`, :func:`snapshot_sparse_matrix`,
:func:`_owned_factor_record`, :class:`RetainedMeasuresSnapshot`, and
:func:`snapshot_measures`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

from powermean.fe.certificate_basis import (
    CertificateBasis,
    Witness,
    _export_operational_basis,
    _operational_data,
    _witness_proof,
)
from powermean.fe.certificate_witness import OperationalBasis, WitnessProof
from powermean.fe.connectivity import IndicatorRank
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_groupings import (
    RetainedGroupingSnapshot,
    detach_grouping,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_input_values import (
    PartitionKey,
    _tuple_arrays,
)
from powermean.solve.absorbed_certificate_factor import FactorRecord
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_sparse import SparseMeasures
from powermean.solve.compact_matrix import CompactMatrix

__all__ = [
    "RetainedBasisSnapshot",
    "RetainedMeasuresSnapshot",
    "RetainedSparseMatrix",
    "_owned_factor_record",
    "_owned_sparse_matrix",
    "snapshot_basis",
    "snapshot_measures",
    "snapshot_sparse_matrix",
]


def _owned_sparse_matrix(m: SparseMatrix) -> SparseMatrix:
    """Return a fresh ``SparseMatrix`` with owned CSC index and data buffers."""
    return SparseMatrix(
        m.shape, owned_array(m.indptr), owned_array(m.indices), owned_array(m.data)
    )


@dataclass(frozen=True)
class RetainedBasisSnapshot:
    """Detached snapshot of ``CertificateBasis`` (`J`, references, columns)."""

    partition_key: PartitionKey | None
    route: Literal["connectivity", "elimination"]
    rank: IndicatorRank
    rows: int
    witnesses: tuple[tuple[Witness, ...], ...]
    echelon: tuple[tuple[tuple[int, int], ...], ...]
    _grouping: RetainedGroupingSnapshot
    _references: tuple[IndexArray, ...]
    _columns: tuple[IndexArray, ...]
    _selected: IndexArray
    _positions: tuple[IndexArray, ...]

    def __getattribute__(self, name: str) -> object:
        """Materialize complete witnesses only for actual public/rich field access."""
        found: object = object.__getattribute__(self, name)
        if name == "witnesses" and type(found) is WitnessProof:
            return found.materialize()
        if name == "echelon" and type(found) is WitnessProof:
            return found.materialize_echelon()
        return found

    def __post_init__(self) -> None:
        """Own all stored index arrays."""
        object.__setattr__(self, "_references", _tuple_arrays(self._references))
        object.__setattr__(self, "_columns", _tuple_arrays(self._columns))
        object.__setattr__(self, "_selected", owned_array(self._selected))
        object.__setattr__(self, "_positions", _tuple_arrays(self._positions))

    @property
    def grouping(self) -> RetainedGroupingSnapshot:
        """Return a freshly materialized :class:`RetainedGroupingSnapshot`."""
        return self._grouping.rematerialize()

    @property
    def references(self) -> tuple[IndexArray, ...]:
        """Return fresh owned copies of the per-block reference index arrays."""
        return _tuple_arrays(self._references)

    @property
    def columns(self) -> tuple[IndexArray, ...]:
        """Return fresh owned copies of the per-block column index arrays."""
        return _tuple_arrays(self._columns)

    @property
    def selected(self) -> IndexArray:
        """Return a fresh owned copy of the selected incidence column indices."""
        return owned_array(self._selected)

    @property
    def positions(self) -> tuple[IndexArray, ...]:
        """Return fresh owned copies of the per-block level position maps."""
        return _tuple_arrays(self._positions)

    @property
    def n_incidence(self) -> int:
        """Return the number of columns of ``E``."""
        return sum(int(np.count_nonzero(c >= 0)) for c in self._columns)

    @property
    def n_kept(self) -> int:
        """Return the number of kept fixed-effect columns in ``J``."""
        return int(self._selected.size)

    @property
    def basis(self) -> CertificateBasis:
        """Return a ``CertificateBasis`` view with owned index arrays."""
        return CertificateBasis(
            cast("Any", self.grouping),
            self.route,
            self.rank,
            self.references,
            self.columns,
            self.selected,
            self.positions,
            self.witnesses,
            self.echelon,
            self.rows,
        )


def _snapshot_echelon(
    rows: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Share genuinely immutable proof tuples; normalize every other input as before.

    Exact tuples of exact integer pairs have no mutable or callback-owned part.
    No read-only array flag, marker or previously cached verdict supplies ownership.
    A subclass, list or other value takes the original conversion in original order.
    """
    if type(rows) is tuple and all(
        type(row) is tuple and all(
            type(pair) is tuple and len(pair) == 2
            and type(pair[0]) is int and type(pair[1]) is int
            for pair in row
        ) for row in rows
    ):
        return rows
    return tuple(tuple((int(p[0]), int(p[1])) for p in row) for row in rows)


def snapshot_basis(
    basis: CertificateBasis,
    *,
    partition_key: PartitionKey | None = None,
    grouping_snapshot: RetainedGroupingSnapshot | None = None,
) -> RetainedBasisSnapshot:
    """Snapshot a ``CertificateBasis`` into a :class:`RetainedBasisSnapshot`."""
    basis = _export_operational_basis(basis)
    grp = (
        grouping_snapshot
        if grouping_snapshot is not None
        else detach_grouping(basis.grouping)
    )
    pkey = partition_key if partition_key is not None else grp.partition_key
    proof = _witness_proof(basis)
    wit = (
        tuple(tuple(w) for w in basis.witnesses) if proof is None
        else cast("tuple[tuple[Witness, ...], ...]", proof)
    )
    raw_echelon = (None if proof is None
                   else object.__getattribute__(basis, "echelon"))
    ech = (cast("tuple[tuple[tuple[int, int], ...], ...]", proof)
           if proof is not None and raw_echelon is proof else
           _snapshot_echelon(basis.echelon))
    return RetainedBasisSnapshot(
        partition_key=pkey,
        route=basis.route,
        rank=basis.rank,
        rows=int(basis.rows),
        witnesses=wit,
        echelon=ech,
        _grouping=grp,
        _references=_tuple_arrays(basis.references),
        _columns=_tuple_arrays(basis.columns),
        _selected=owned_array(basis.selected),
        _positions=_tuple_arrays(basis.positions),
    )


@dataclass(frozen=True, slots=True)
class _OperationalBasisSnapshot:
    """Owned numerical headers and immutable exact actions, without proof views.

    This is detached data, not admission authority.  A rich request reconstructs
    the canonical presentation from these values, never from a live fit owner.
    """

    partition_key: PartitionKey | None
    grouping: RetainedGroupingSnapshot
    route: Literal["connectivity", "elimination"]
    rank: IndicatorRank
    rows: int
    references: tuple[IndexArray, ...]
    columns: tuple[IndexArray, ...]
    selected: IndexArray
    positions: tuple[IndexArray, ...]
    operational: OperationalBasis

    def numerical_basis(self) -> CertificateBasis:
        """Return detached numerical coordinates without creating proof views."""
        basis = CertificateBasis(
            cast("Any", self.grouping), self.route, self.rank,
            self.references, self.columns, self.selected, self.positions,
            (), (), self.rows,
        )
        object.__setattr__(basis, "_operational", self.operational)
        return basis

    def materialize(self) -> RetainedBasisSnapshot:
        """Create the canonical detached basis only for an actual rich consumer."""
        return snapshot_basis(
            self.numerical_basis(), partition_key=self.partition_key,
            grouping_snapshot=self.grouping,
        )


def _capture_operational_basis(
    basis: CertificateBasis,
    *,
    partition_key: PartitionKey | None,
    grouping_snapshot: RetainedGroupingSnapshot,
) -> _OperationalBasisSnapshot | None:
    """Detach numerical basis headers without opening canonical proof fields."""
    operational = _operational_data(basis)
    if operational is None:
        return None
    return _OperationalBasisSnapshot(
        partition_key, grouping_snapshot, basis.route, basis.rank, int(basis.rows),
        _tuple_arrays(basis.references), _tuple_arrays(basis.columns),
        owned_array(basis.selected), _tuple_arrays(basis.positions), operational,
    )


@dataclass(frozen=True)
class RetainedSparseMatrix:
    """Owned snapshot of a canonical CSC ``SparseMatrix``."""

    shape: tuple[int, int]
    _indptr: IndexArray | None
    _indices: IndexArray | None
    _data: FloatArray | None
    _compact: CompactMatrix | None = field(default=None, init=False, repr=False)

    @classmethod
    def _from_compact(cls, matrix: CompactMatrix) -> RetainedSparseMatrix:
        """Retain immutable partition bytes/scales, not three full CSC copies."""
        result = object.__new__(cls)
        for name, value in (("shape", matrix.shape), ("_indptr", None),
                            ("_indices", None), ("_data", None),
                            ("_compact", matrix.detached())):
            object.__setattr__(result, name, value)
        return result

    def __post_init__(self) -> None:
        """Own all stored CSC buffers."""
        object.__setattr__(self, "_indptr", owned_array(self._indptr))
        object.__setattr__(self, "_indices", owned_array(self._indices))
        object.__setattr__(self, "_data", owned_array(self._data))

    @property
    def indptr(self) -> IndexArray:
        """Return a fresh owned copy of ``indptr``."""
        if self._compact is not None:
            return self._compact.materialize().indptr
        assert self._indptr is not None
        return owned_array(self._indptr)

    @property
    def indices(self) -> IndexArray:
        """Return a fresh owned copy of ``indices``."""
        if self._compact is not None:
            return self._compact.materialize().indices
        assert self._indices is not None
        return owned_array(self._indices)

    @property
    def data(self) -> FloatArray:
        """Return a fresh owned copy of ``data``."""
        if self._compact is not None:
            return self._compact.materialize().data
        assert self._data is not None
        return owned_array(self._data)

    @property
    def nnz(self) -> int:
        """Return the number of stored nonzeros."""
        if self._compact is not None:
            return self._compact.nnz
        assert self._data is not None
        return int(self._data.size)

    @property
    def matrix(self) -> SparseMatrix:
        """Return a fresh ``SparseMatrix`` with owned CSC buffers."""
        if self._compact is not None:
            return self._compact.materialize()
        return SparseMatrix(
            shape=self.shape,
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
        )


def snapshot_sparse_matrix(matrix: SparseMatrix | CompactMatrix | None) -> RetainedSparseMatrix | None:
    """Snapshot a ``SparseMatrix`` into a :class:`RetainedSparseMatrix`, or ``None``."""
    if matrix is None:
        return None
    if isinstance(matrix, CompactMatrix):
        return RetainedSparseMatrix._from_compact(matrix)
    return RetainedSparseMatrix(
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        _indptr=owned_array(matrix.indptr),
        _indices=owned_array(matrix.indices),
        _data=owned_array(matrix.data),
    )


def _owned_factor_record(r: FactorRecord) -> FactorRecord:
    """Return a fresh ``FactorRecord`` with owned arrays and frozen mappings."""
    opts = MappingProxyType(dict(r.options))
    call = None if r.call is None else MappingProxyType(dict(r.call))
    head = (r.kind, r.threshold, opts, r.status, r.reason, _optional(r.perm_r))
    tail = (
        _optional(r.perm_c),
        r.l_nnz,
        r.u_nnz,
        _optional(r.diagonal),
        r.version,
        call,
    )
    return FactorRecord(*head, *tail)


@dataclass(frozen=True)
class RetainedMeasuresSnapshot:
    """Owned snapshot of ``SparseMeasures`` without live SuperLU objects."""

    estimate: float
    rcond: float
    conditioned: bool
    positive_definite: bool
    scaled_score: float
    index_correction: float
    actions: Mapping[str, int]
    errors: Mapping[str, str]
    _ordinary: FactorRecord
    _curvature: FactorRecord
    _correction: FloatArray | None
    _change: FloatArray | None
    _residual: FloatArray | None
    _slope_map: FloatArray | None
    score_metric: str = "current_symmetric_hessian"

    def __post_init__(self) -> None:
        """Own factor records, arrays, and freeze mappings."""
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))
        object.__setattr__(self, "_ordinary", _owned_factor_record(self._ordinary))
        object.__setattr__(self, "_curvature", _owned_factor_record(self._curvature))
        object.__setattr__(self, "_correction", _optional(self._correction))
        object.__setattr__(self, "_change", _optional(self._change))
        object.__setattr__(self, "_residual", _optional(self._residual))
        object.__setattr__(self, "_slope_map", _optional(self._slope_map))

    @property
    def ordinary(self) -> FactorRecord:
        """Return a fresh owned ``FactorRecord`` for the ordinary LU factor."""
        return _owned_factor_record(self._ordinary)

    @property
    def curvature(self) -> FactorRecord:
        """Return a fresh owned ``FactorRecord`` for the curvature LU factor."""
        return _owned_factor_record(self._curvature)

    @property
    def correction(self) -> FloatArray | None:
        """Return a fresh owned copy of ``correction``, or ``None``."""
        return _optional(self._correction)

    @property
    def change(self) -> FloatArray | None:
        """Return a fresh owned copy of ``change``, or ``None``."""
        return _optional(self._change)

    @property
    def residual(self) -> FloatArray | None:
        """Return a fresh owned copy of ``residual``, or ``None``."""
        return _optional(self._residual)

    @property
    def slope_map(self) -> FloatArray | None:
        """Return a fresh owned copy of ``slope_map``, or ``None``."""
        return _optional(self._slope_map)

    def without_slope_map(self) -> RetainedMeasuresSnapshot:
        """Return a copy with ``slope_map`` discarded (``None``)."""
        if self._slope_map is None:
            return self
        return replace(self, _slope_map=None)


def snapshot_measures(
    m: SparseMeasures, *, keep_slope_map: bool = True
) -> RetainedMeasuresSnapshot:
    """Snapshot ``SparseMeasures`` into a :class:`RetainedMeasuresSnapshot`."""
    s_map = m.slope_map if keep_slope_map else None
    return RetainedMeasuresSnapshot(
        estimate=float(m.estimate),
        rcond=float(m.rcond),
        conditioned=bool(m.conditioned),
        positive_definite=bool(m.positive_definite),
        scaled_score=float(m.scaled_score),
        index_correction=float(m.index_correction),
        actions=MappingProxyType(dict(m.actions)),
        errors=MappingProxyType(dict(m.errors)),
        _ordinary=_owned_factor_record(m.ordinary),
        _curvature=_owned_factor_record(m.curvature),
        _correction=_optional(m.correction),
        _change=_optional(m.change),
        _residual=_optional(m.residual),
        _slope_map=_optional(s_map),
        score_metric=str(m.score_metric),
    )
