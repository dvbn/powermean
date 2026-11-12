"""Detached snapshots of FE effect equations, inner evidence, & certificate.

This module detaches ``EffectEquations``, ``InnerEvidence``, ``CoefficientCheck``, and
``FEPointCertificate`` without re-encoding groupings, and re-exports the passive
geometry/basis/row/measures snapshots from :mod:`powermean.results.absorbed_geometry`
and :mod:`powermean.results.absorbed_geometry_records`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from powermean.fe.certificate_basis import SupportCheck
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.results.absorbed_geometry import (
    ExpectedInformationInputs,
    RetainedBasisSnapshot,
    RetainedGeometrySnapshot,
    RetainedRowsSnapshot,
    RetainedSparseMatrix,
    _owned_sparse_matrix,
    reconstruct_expected_information,
    snapshot_basis,
    snapshot_geometry,
    snapshot_rows,
    snapshot_sparse_matrix,
)
from powermean.results.absorbed_geometry_records import (
    RetainedMeasuresSnapshot,
    _OperationalBasisSnapshot,
    _capture_operational_basis,
    _owned_factor_record,
    snapshot_measures,
)
from powermean.results.absorbed_groupings import (
    GroupingIdentityMap,
    RetainedGroupingSnapshot,
)
from powermean.results.absorbed_history import (
    PointIdentityMap,
    RetainedAbsorptionResult,
    RetainedLinearNormalization,
    RetainedPointSnapshot,
    RetainedRecovery,
    _owned_absorption_result,
    _owned_recovery,
    snapshot_linear_normalization,
    snapshot_point_record,
)
from powermean.results.absorbed_sample import (
    RetainedSampleSnapshot,
    SampleIdentityMap,
)
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_inputs import PointKey, SampleKey, _tuple_arrays
from powermean.solve.absorbed_certificate import FEGate, FEPointCertificate, Level
from powermean.solve.absorbed_certificate_inner import (
    InnerCheck,
    InnerEvidence,
)
from powermean.solve.absorbed_certificate_point import WorkStep
from powermean.solve.absorbed_certificate_geometry import FEGeometry
from powermean.solve.assessment import CertificateControls
from powermean.solve.reported import CoefficientCheck

Family = PowerVariance | NB2Variance


def _owned_coefficient_check(c: CoefficientCheck | None) -> CoefficientCheck | None:
    """Return a fresh ``CoefficientCheck`` with owned arrays, or ``None``."""
    if c is None:
        return None
    cor, se = owned_array(c.corrections), owned_array(c.standard_errors)
    return CoefficientCheck(
        cor, se, c.coef_tol, owned_array(c.passed), owned_array(c.ratios)
    )


@dataclass(frozen=True)
class RetainedEffectEquations:
    """Owned snapshot of ``EffectEquations`` with fresh array exposure."""

    maximum: float
    _means: tuple[FloatArray, ...]

    def __init__(self, maximum: float, means: tuple[FloatArray, ...]) -> None:
        """Own all stored block mean vectors."""
        object.__setattr__(self, "maximum", float(maximum))
        object.__setattr__(self, "_means", _tuple_arrays(means))

    @property
    def means(self) -> tuple[FloatArray, ...]:
        """Return fresh owned copies of the block level mean vectors."""
        return _tuple_arrays(self._means)


@dataclass(frozen=True)
class RetainedInnerEvidence:
    """Detached snapshot of ``InnerEvidence`` without live ``FESample``.

    The semantics of :class:`~powermean.solve.absorbed_certificate_inner.
    InnerEvidence`: an inapplicable component (an exact root point's) has no value
    and never passes, and evidence without an applicable check does not pass.
    """

    checks: tuple[InnerCheck, ...]
    recovery: RetainedRecovery | None
    normalization: RetainedLinearNormalization | None
    auxiliary: RetainedAbsorptionResult | None
    steps: tuple[WorkStep, ...]

    @property
    def applicable(self) -> bool:
        """Return True if an inner check applies at the point."""
        return any(c.applicable for c in self.checks)

    @property
    def passed(self) -> bool:
        """Return True if a check applies and every applicable check passed."""
        found = [c.passed for c in self.checks if c.applicable]
        return bool(found) and all(found)

    @property
    def reason(self) -> str:
        """Return the reason of the first failed applicable check, or empty string."""
        failed = (c for c in self.checks if c.applicable and not c.passed)
        return next((c.reason for c in failed), "")

    @property
    def not_applicable(self) -> tuple[str, ...]:
        """Return the reasons of the components that do not apply at the point."""
        return tuple(c.reason for c in self.checks if not c.applicable)


def _snapshot_inner_evidence(
    inner: InnerEvidence | None,
    *,
    pt_snap: RetainedPointSnapshot,
    sample_snapshot: RetainedSampleSnapshot,
    grp_snap: RetainedGroupingSnapshot | None,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedInnerEvidence | None:
    """Detach optional ``InnerEvidence`` into a :class:`RetainedInnerEvidence`."""
    if inner is None:
        return None
    get_pt = None if point_map is None else point_map.get
    return RetainedInnerEvidence(
        checks=tuple(inner.checks),
        recovery=_owned_recovery(
            inner.recovery,
            grouping_snapshot=grp_snap,
            grouping_map=grouping_map,
        ),
        normalization=snapshot_linear_normalization(
            inner.normalization,
            point_key=pt_snap.key,
            sample_key=sample_snapshot.key,
            get_point=get_pt,
            sample_map=sample_map,
        ),
        auxiliary=_owned_absorption_result(
            inner.auxiliary,
            grouping_snapshot=grp_snap,
            grouping_map=grouping_map,
        ),
        steps=tuple(inner.steps),
    )


@dataclass(frozen=True)
class RetainedCertificateSnapshot:
    """Detached snapshot of ``FEPointCertificate`` with result-local keys."""

    sample_key: SampleKey
    point_key: PointKey
    controls: CertificateControls
    point: RetainedPointSnapshot
    basis: RetainedBasisSnapshot
    rows: RetainedRowsSnapshot | None
    effects: RetainedEffectEquations | None
    inner: RetainedInnerEvidence | None
    geometry: RetainedGeometrySnapshot | None
    measures: RetainedMeasuresSnapshot | None
    error_messages: Mapping[FEGate, str]
    work: tuple[WorkStep, ...]
    strict: tuple[FEGate, ...]
    secondary: tuple[FEGate, ...]
    _coefficients: CoefficientCheck | None
    not_applicable: tuple[FEGate, ...] = ()
    require_positive_definite: bool = True
    score_metric: str = "current_symmetric_hessian"

    def __post_init__(self) -> None:
        """Own coefficient check arrays and freeze error mapping."""
        object.__setattr__(
            self, "error_messages", MappingProxyType(dict(self.error_messages))
        )
        if self._coefficients is not None:
            object.__setattr__(
                self, "_coefficients", _owned_coefficient_check(self._coefficients)
            )

    @property
    def errors(self) -> Mapping[FEGate, str]:
        """Return an immutable mapping of gate failure reasons."""
        return self.error_messages

    @property
    def coefficients(self) -> CoefficientCheck | None:
        """Return a fresh owned ``CoefficientCheck``, or ``None``."""
        return _owned_coefficient_check(self._coefficients)

    @property
    def empty(self) -> bool:
        """Return True when the point has no active slopes (``K = 0``)."""
        return bool(self.point.n_slopes == 0)

    @property
    def level(self) -> Level | None:
        """Return ``'strict'``, ``'secondary'``, or ``None``."""
        if not self.strict:
            return "strict"
        return "secondary" if not self.secondary else None

    @property
    def accepted(self) -> bool:
        """Return True if the point passes every gate at strict or secondary level."""
        return self.level is not None

    @property
    def degenerate(self) -> bool:
        """Return True if a reported standard error is zero."""
        return self._coefficients is not None and self._coefficients.degenerate

    @property
    def scaled_score(self) -> float:
        """Return ``||Hplus^-1/2 S||``, or NaN when not formed."""
        return math.nan if self.measures is None else self.measures.scaled_score

    @property
    def index_correction(self) -> float:
        """Return ``||M d||_inf``, or NaN when not formed."""
        return math.nan if self.measures is None else self.measures.index_correction

    def without_slope_map(self) -> RetainedCertificateSnapshot:
        """Return a copy with ``slope_map`` discarded and ``rows`` marked discarded."""
        new_meas = None if self.measures is None else self.measures.without_slope_map()
        new_rows = None if self.rows is None else self.rows.as_discarded()
        new_geom = (
            None
            if self.geometry is None or new_rows is None
            else self.geometry.with_rows(new_rows)
        )
        return replace(
            self,
            rows=new_rows,
            geometry=new_geom,
            measures=new_meas,
        )


@dataclass(frozen=True, slots=True)
class _GeometryData:
    """Detached numerical geometry without its redundant presentation links."""

    support: SupportCheck
    norm: float
    refusal: str
    design: FloatArray
    moment: FloatArray
    hessian: RetainedSparseMatrix
    scale: FloatArray | None
    ordinary: RetainedSparseMatrix | None
    curvature: RetainedSparseMatrix | None

    @classmethod
    def capture(cls, geometry: FEGeometry) -> _GeometryData:
        """Own the admitted numerical buffers without retaining the source graph."""
        hessian = snapshot_sparse_matrix(geometry.hessian)
        assert hessian is not None
        return cls(
            geometry.support, float(geometry.norm), str(geometry.refusal),
            owned_array(geometry.design), owned_array(geometry.moment), hessian,
            _optional(geometry.scale), snapshot_sparse_matrix(geometry.ordinary),
            snapshot_sparse_matrix(geometry.curvature),
        )

    def materialize(
        self, point: RetainedPointSnapshot, rows: RetainedRowsSnapshot,
        basis: RetainedBasisSnapshot,
    ) -> RetainedGeometrySnapshot:
        """Assemble canonical links from already owned values, without arithmetic."""
        return RetainedGeometrySnapshot(
            point.sample_key, point.key, point, rows, basis, self.support,
            self.norm, self.refusal, self.design, self.moment, self.hessian,
            self.scale, self.ordinary, self.curvature,
        )


@dataclass(frozen=True, slots=True)
class _OperationalCertificate:
    """Owned gate values and numerical payload, not a retained rich certificate."""

    sample_key: SampleKey
    point_key: PointKey
    controls: CertificateControls
    point: RetainedPointSnapshot
    basis: RetainedBasisSnapshot | _OperationalBasisSnapshot
    rows: RetainedRowsSnapshot | None
    effects: RetainedEffectEquations | None
    inner: RetainedInnerEvidence | None
    geometry: RetainedGeometrySnapshot | _GeometryData | None
    measures: RetainedMeasuresSnapshot | None
    error_messages: Mapping[FEGate, str]
    work: tuple[WorkStep, ...]
    strict: tuple[FEGate, ...]
    secondary: tuple[FEGate, ...]
    _coefficients: CoefficientCheck | None
    not_applicable: tuple[FEGate, ...]
    require_positive_definite: bool
    score_metric: str

    @property
    def level(self) -> Level | None:
        """Read the actual gate outcome without constructing presentation records."""
        if not self.strict:
            return "strict"
        return "secondary" if not self.secondary else None

    @property
    def accepted(self) -> bool:
        """Read the selected point's existing acceptance verdict."""
        return self.level is not None

    @property
    def degenerate(self) -> bool:
        """Read the existing direct-SE zero qualification."""
        return self._coefficients is not None and self._coefficients.degenerate

    def without_slope_map(self) -> _OperationalCertificate:
        """Actually discard the stored map and guard retained diagnostic rows."""
        return replace(
            self,
            rows=None if self.rows is None else self.rows.as_discarded(),
            measures=(None if self.measures is None
                      else self.measures.without_slope_map()),
        )

    def materialize(self) -> RetainedCertificateSnapshot:
        """Create the canonical rich graph from detached values on demand."""
        basis = (self.basis.materialize()
                 if isinstance(self.basis, _OperationalBasisSnapshot) else self.basis)
        geometry = (
            None if self.geometry is None or self.rows is None else
            self.geometry.materialize(self.point, self.rows, basis)
            if isinstance(self.geometry, _GeometryData) else
            self.geometry.with_rows(self.rows)
        )
        return RetainedCertificateSnapshot(
            self.sample_key, self.point_key, self.controls, self.point, basis,
            self.rows, self.effects, self.inner, geometry, self.measures,
            self.error_messages, self.work, self.strict, self.secondary,
            self._coefficients, self.not_applicable, self.require_positive_definite,
            self.score_metric,
        )


def _capture_certificate(
    certificate: FEPointCertificate,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    selected_point: RetainedPointSnapshot | None = None,
    keep_slope_map: bool = True,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedCertificateSnapshot | _OperationalCertificate:
    """Capture admitted values, keeping native operational data free of views."""
    lin_norm = None if certificate.inner is None else certificate.inner.normalization
    pt_snap = selected_point or snapshot_point_record(
        certificate.point,
        sample_snapshot=sample_snapshot,
        linear_normalization=lin_norm,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    # The sample's internally owned grouping view and response: the basis stores the
    # view privately and rematerializes it outward; the rows own their own response.
    grp_snap = sample_snapshot._final_grouping
    operational = (
        None if grp_snap is None else _capture_operational_basis(
            certificate.basis, partition_key=sample_snapshot.partition_key,
            grouping_snapshot=grp_snap,
        )
    )
    basis_snap = (
        snapshot_basis(
            certificate.basis, partition_key=sample_snapshot.partition_key,
            grouping_snapshot=grp_snap,
        ) if operational is None else operational
    )
    rows_snap = (
        None
        if certificate.rows is None
        else snapshot_rows(
            certificate.rows,
            response=sample_snapshot._owned_response,
            q=pt_snap.q,
            variance=certificate.point.variance,
        )
    )
    if rows_snap is not None and not keep_slope_map:
        rows_snap = rows_snap.as_discarded()
    eff_snap = (
        None
        if certificate.effects is None
        else RetainedEffectEquations(
            certificate.effects.maximum, certificate.effects.means
        )
    )
    inner_snap = _snapshot_inner_evidence(
        certificate.inner,
        pt_snap=pt_snap,
        sample_snapshot=sample_snapshot,
        grp_snap=grp_snap,
        point_map=point_map,
        sample_map=sample_map,
        grouping_map=grouping_map,
    )
    geom_snap: RetainedGeometrySnapshot | _GeometryData | None = (
        None
        if certificate.geometry is None or rows_snap is None
        else _GeometryData.capture(certificate.geometry)
        if operational is not None else snapshot_geometry(
            certificate.geometry,
            point_snapshot=pt_snap,
            rows_snapshot=rows_snap,
            basis_snapshot=cast("RetainedBasisSnapshot", basis_snap),
        )
    )
    meas_snap = (
        None
        if certificate.measures is None
        else snapshot_measures(certificate.measures, keep_slope_map=keep_slope_map)
    )
    if operational is None:
        return RetainedCertificateSnapshot(
            sample_key=sample_snapshot.key,
            point_key=pt_snap.key,
            controls=certificate.controls,
            point=pt_snap,
            basis=cast("RetainedBasisSnapshot", basis_snap),
            rows=rows_snap,
            effects=eff_snap,
            inner=inner_snap,
            geometry=cast("RetainedGeometrySnapshot | None", geom_snap),
            measures=meas_snap,
            error_messages=MappingProxyType(dict(certificate.errors)),
            work=tuple(certificate.work),
            strict=tuple(certificate.strict),
            secondary=tuple(certificate.secondary),
            _coefficients=_owned_coefficient_check(certificate.coefficients),
            not_applicable=tuple(certificate.not_applicable),
            require_positive_definite=bool(certificate.require_positive_definite),
            score_metric=str(certificate.score_metric),
        )
    return _OperationalCertificate(
        sample_key=sample_snapshot.key,
        point_key=pt_snap.key,
        controls=certificate.controls,
        point=pt_snap,
        basis=basis_snap,
        rows=rows_snap,
        effects=eff_snap,
        inner=inner_snap,
        geometry=geom_snap,
        measures=meas_snap,
        error_messages=MappingProxyType(dict(certificate.errors)),
        work=tuple(certificate.work),
        strict=tuple(certificate.strict),
        secondary=tuple(certificate.secondary),
        _coefficients=_owned_coefficient_check(certificate.coefficients),
        not_applicable=tuple(certificate.not_applicable),
        require_positive_definite=bool(certificate.require_positive_definite),
        score_metric=str(certificate.score_metric),
    )


def snapshot_certificate(
    certificate: FEPointCertificate,
    *,
    sample_snapshot: RetainedSampleSnapshot,
    selected_point: RetainedPointSnapshot | None = None,
    keep_slope_map: bool = True,
    point_map: PointIdentityMap | None = None,
    sample_map: SampleIdentityMap | None = None,
    grouping_map: GroupingIdentityMap | None = None,
) -> RetainedCertificateSnapshot:
    """Return the canonical detached certificate for an explicit rich request."""
    captured = _capture_certificate(
        certificate, sample_snapshot=sample_snapshot, selected_point=selected_point,
        keep_slope_map=keep_slope_map, point_map=point_map,
        sample_map=sample_map, grouping_map=grouping_map,
    )
    return (captured.materialize() if isinstance(captured, _OperationalCertificate)
            else captured)


__all__ = [
    "ExpectedInformationInputs",
    "Family",
    "RetainedBasisSnapshot",
    "RetainedCertificateSnapshot",
    "RetainedEffectEquations",
    "RetainedGeometrySnapshot",
    "RetainedInnerEvidence",
    "RetainedMeasuresSnapshot",
    "RetainedRowsSnapshot",
    "RetainedSparseMatrix",
    "_owned_coefficient_check",
    "_owned_factor_record",
    "_owned_sparse_matrix",
    "reconstruct_expected_information",
    "snapshot_basis",
    "snapshot_certificate",
    "snapshot_geometry",
    "snapshot_measures",
    "snapshot_rows",
    "snapshot_sparse_matrix",
]
