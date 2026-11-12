"""Detached snapshots of FE rows, geometry, and expected-information assembly.

This module detaches ``FreshRows`` and ``FEGeometry`` into passive finite typed records
with owned arrays on ingress and every exposure, and owns the single lazy
expected-information assembly implementation (:class:`ExpectedInformationInputs`).
Basis and sparse matrix records live in
:mod:`powermean.results.absorbed_geometry_records`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from powermean.errors import SpecificationError
from powermean.fe.absorb_numeric import checked_product
from powermean.fe.certificate_basis import SupportCheck
from powermean.model.order_scale import OrderScale
from powermean.model.rows import evaluate_rows
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_geometry_records import (
    RetainedBasisSnapshot,
    RetainedSparseMatrix,
    _owned_sparse_matrix,
    snapshot_basis,
    snapshot_sparse_matrix,
)
from powermean.results.absorbed_points import RetainedPointSnapshot
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_inputs import (
    PointKey,
    SampleKey,
    owned_order_scale,
)
from powermean.solve.absorbed_certificate_geometry import (
    FEGeometry,
    SparseMatrix,
)
from powermean.solve.absorbed_certificate_geometry import (
    hessian as assemble_hessian,
)
from powermean.solve.absorbed_certificate_point import FreshRows

Family = PowerVariance | NB2Variance


@dataclass(frozen=True)
class RetainedRowsSnapshot:
    """Passive owned snapshot of ``FreshRows`` and same-point row inputs."""

    frequency: bool
    variance: Family | None
    _b: FloatArray
    _curvature: FloatArray
    _weights: FloatArray
    _response: FloatArray
    _q: FloatArray
    _influence_discarded: bool = False

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_b", owned_array(self._b))
        object.__setattr__(self, "_curvature", owned_array(self._curvature))
        object.__setattr__(self, "_weights", owned_array(self._weights))
        object.__setattr__(self, "_response", owned_array(self._response))
        object.__setattr__(self, "_q", owned_array(self._q))

    @property
    def b(self) -> FloatArray:
        """Return a fresh owned copy of ``b``."""
        return owned_array(self._b)

    @property
    def curvature(self) -> FloatArray:
        """Return a fresh owned copy of ``curvature``."""
        return owned_array(self._curvature)

    @property
    def weights(self) -> FloatArray:
        """Return a fresh owned copy of ``weights``."""
        return owned_array(self._weights)

    @property
    def response(self) -> FloatArray:
        """Return a fresh owned copy of ``response``."""
        return owned_array(self._response)

    @property
    def q(self) -> FloatArray:
        """Return a fresh owned copy of ``q``."""
        return owned_array(self._q)

    @property
    def fresh_rows(self) -> FreshRows:
        """Return a fresh ``FreshRows`` with owned vectors."""
        return FreshRows(
            b=self.b,
            curvature=self.curvature,
            weights=self.weights,
            frequency=self.frequency,
        )

    def as_discarded(self) -> RetainedRowsSnapshot:
        """Return a copy marked as belonging to an influence-discarded envelope."""
        if self._influence_discarded:
            return self
        return replace(self, _influence_discarded=True)


def snapshot_rows(
    rows: FreshRows,
    *,
    response: FloatArray,
    q: FloatArray,
    variance: Family | None,
) -> RetainedRowsSnapshot:
    """Snapshot ``FreshRows`` alongside passive same-point row inputs."""
    return RetainedRowsSnapshot(
        frequency=bool(rows.frequency),
        variance=variance,
        _b=owned_array(rows.b),
        _curvature=owned_array(rows.curvature),
        _weights=owned_array(rows.weights),
        _response=owned_array(response),
        _q=owned_array(q),
    )


@dataclass(frozen=True)
class RetainedGeometrySnapshot:
    """Passive detached snapshot of ``FEGeometry`` (`M`, `S`, `H`, `D`, `A`)."""

    sample_key: SampleKey
    point_key: PointKey
    point: RetainedPointSnapshot
    rows: RetainedRowsSnapshot
    basis: RetainedBasisSnapshot
    support: SupportCheck
    norm: float
    refusal: str
    _design: FloatArray
    _moment: FloatArray
    _hessian: RetainedSparseMatrix
    _scale: FloatArray | None
    _ordinary: RetainedSparseMatrix | None
    _curvature: RetainedSparseMatrix | None

    def __post_init__(self) -> None:
        """Own all stored numeric arrays."""
        object.__setattr__(self, "_design", owned_array(self._design))
        object.__setattr__(self, "_moment", owned_array(self._moment))
        object.__setattr__(self, "_scale", _optional(self._scale))

    @property
    def design(self) -> FloatArray:
        """Return a fresh owned copy of ``design``."""
        return owned_array(self._design)

    @property
    def moment(self) -> FloatArray:
        """Return a fresh owned copy of ``moment``."""
        return owned_array(self._moment)

    @property
    def scale(self) -> FloatArray | None:
        """Return a fresh owned copy of ``scale``, or ``None``."""
        return _optional(self._scale)

    @property
    def n_slopes(self) -> int:
        """Return the number of active slope columns ``K``."""
        return int(self._design.shape[1])

    @property
    def n_effects(self) -> int:
        """Return the number of kept fixed-effect columns in ``J``."""
        return self.basis.n_kept

    @property
    def dimension(self) -> int:
        """Return total joint dimension ``K + n_kept``."""
        return self.n_slopes + self.n_effects

    @property
    def hessian(self) -> SparseMatrix:
        """Return a fresh ``SparseMatrix`` for the unscaled joint Hessian ``H``."""
        return self._hessian.matrix

    @property
    def ordinary(self) -> SparseMatrix | None:
        """Return a fresh ``SparseMatrix`` for the equilibrated ``A``, or ``None``."""
        return None if self._ordinary is None else self._ordinary.matrix

    @property
    def curvature(self) -> SparseMatrix | None:
        """Return a fresh ``SparseMatrix`` for the symmetrized ``Aplus`` or ``None``."""
        return None if self._curvature is None else self._curvature.matrix

    def with_rows(self, rows: RetainedRowsSnapshot) -> RetainedGeometrySnapshot:
        """Return a copy bound to ``rows``."""
        if self.rows is rows:
            return self
        return replace(self, rows=rows)


def snapshot_geometry(
    geometry: FEGeometry,
    *,
    point_snapshot: RetainedPointSnapshot,
    rows_snapshot: RetainedRowsSnapshot,
    basis_snapshot: RetainedBasisSnapshot,
) -> RetainedGeometrySnapshot:
    """Snapshot an ``FEGeometry`` into a :class:`RetainedGeometrySnapshot`."""
    h_snap = snapshot_sparse_matrix(geometry.hessian)
    assert h_snap is not None
    return RetainedGeometrySnapshot(
        sample_key=point_snapshot.sample_key,
        point_key=point_snapshot.key,
        point=point_snapshot,
        rows=rows_snapshot,
        basis=basis_snapshot,
        support=geometry.support,
        norm=float(geometry.norm),
        refusal=str(geometry.refusal),
        _design=owned_array(geometry.design),
        _moment=owned_array(geometry.moment),
        _hessian=h_snap,
        _scale=_optional(geometry.scale),
        _ordinary=snapshot_sparse_matrix(geometry.ordinary),
        _curvature=snapshot_sparse_matrix(geometry.curvature),
    )


@dataclass(frozen=True)
class ExpectedInformationInputs:
    """Single lazy expected-information assembly implementation (F5).

    Can only be obtained from a retained envelope with ``keep_influence=True``.
    Attempting to construct or evaluate it from rows marked as discarded raises
    ``specification.influence_discarded``.
    """

    sample: RetainedSampleSnapshot
    point: RetainedPointSnapshot
    basis: RetainedBasisSnapshot
    rows: RetainedRowsSnapshot
    order: OrderScale | None
    _design: FloatArray

    def __init__(
        self,
        sample: RetainedSampleSnapshot,
        point: RetainedPointSnapshot,
        basis: RetainedBasisSnapshot,
        rows: RetainedRowsSnapshot,
        *,
        order: OrderScale | None,
        design: FloatArray,
    ) -> None:
        """Validate non-discarded row provenance and own the design and order scale."""
        if rows._influence_discarded:
            raise SpecificationError(
                "specification.influence_discarded",
                {"request": "expected_information"},
            )
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self, "order", None if order is None else owned_order_scale(order)
        )
        object.__setattr__(self, "_design", owned_array(design))

    @property
    def design(self) -> FloatArray:
        """Return a fresh owned copy of the active design matrix."""
        return owned_array(self._design)

    def expected_curvature(self) -> FloatArray:
        """Compute raw expected (Fisher) curvature ``a * c_expected`` on demand."""
        if self.rows._influence_discarded:
            raise SpecificationError(
                "specification.influence_discarded",
                {"request": "expected_information"},
            )
        w = self.rows._weights
        if self.rows.variance is None:
            return owned_array(w)
        eval_res = evaluate_rows(
            self.rows._response, self.rows._q, variance=self.rows.variance
        )
        expected = checked_product(w, eval_res.fisher_curvature, "expected curvature")
        if not np.isfinite(expected).all() or (expected < 0).any():
            raise FloatingPointError(
                "the expected curvature is negative or not representable"
            )
        return owned_array(expected)

    def expected_hessian(self) -> SparseMatrix:
        """Assemble the expected joint Hessian ``M' diag(C_expected) M`` on demand."""
        exp_c = self.expected_curvature()
        return _owned_sparse_matrix(
            assemble_hessian(self._design, exp_c, self.basis.basis)
        )


def reconstruct_expected_information(
    inputs: ExpectedInformationInputs,
) -> SparseMatrix:
    """Assemble and return an owned CSC expected joint Hessian from ``inputs``."""
    if type(inputs) is not ExpectedInformationInputs:
        raise TypeError("inputs must be an ExpectedInformationInputs instance")
    return inputs.expected_hessian()


__all__ = [
    "ExpectedInformationInputs",
    "Family",
    "RetainedBasisSnapshot",
    "RetainedGeometrySnapshot",
    "RetainedRowsSnapshot",
    "RetainedSparseMatrix",
    "_owned_sparse_matrix",
    "reconstruct_expected_information",
    "snapshot_basis",
    "snapshot_geometry",
    "snapshot_rows",
    "snapshot_sparse_matrix",
]
