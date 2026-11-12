"""Value binding of the new full-joint root path before retained detachment.

Only root candidates pay this admission cost.  These are the canonical start,
metric and observed-geometry routines, not another optimizer or a second metric.
The actual factor/solve work of this necessary binding is returned to the receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np

from powermean.data.sample_fe_state import same_bits
from powermean.fe.admission_records import AdmissionWork, WorkLog
from powermean.fe.certificate_basis import owned_basis
from powermean.solve.absorbed_certificate import FEPointCertificate, _geometry
from powermean.solve.absorbed_certificate_geometry import FEGeometry
from powermean.solve.absorbed_certificate_point import WorkStep
from powermean.solve.absorbed_certificate_sparse import sparse_measures
from powermean.solve.absorbed_root import (
    convex_start,
    merit_at_start,
    point_at,
    summary_of,
)
from powermean.solve.absorbed_root_records import AbsorbedRootCandidate, RootPoint

UNBOUND = "the root history, start, merit or observed evidence is not bound"


def _value(value: object) -> object:
    """Detach private numerical records for exact same-host value comparisons."""
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes(order="C")
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Mapping):
        return tuple((key, _value(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_value(item) for item in value)
    if isinstance(value, list):
        return list, tuple(_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(
            (field.name, _value(getattr(value, field.name))) for field in fields(value)
        )
    if value is None or isinstance(value, str | bool | int | bytes):
        return value
    raise TypeError("unsupported private root binding value")


def _start(candidate: AbsorbedRootCandidate) -> None:
    """Bind full coordinates to the caller slopes or the audited convex point."""
    sample, init = candidate.sample, candidate.initializer
    basis = owned_basis(sample)
    if init.audit is not None:
        if init.audit.history.candidate is not init.candidate:
            raise ValueError(UNBOUND)
        expected, gap = convex_start(sample, basis, init.audit)
    else:
        given = candidate.start
        if given is None:
            raise ValueError(UNBOUND)
        try:
            gamma = np.concatenate([given.view(), np.zeros(basis.n_kept)])
            expected = point_at(sample, basis, gamma)
        except FloatingPointError:
            expected = None
        gap = None
    if _value(gap) != _value(init.gap) or _value(expected) != _value(init.point):
        raise ValueError(UNBOUND)


def _selected(candidate: AbsorbedRootCandidate) -> None:
    """Bind selected normalized coordinates to the retained target attempt."""
    if not candidate.stages:
        return
    target = candidate.target_stage
    if target is None:
        raise ValueError(UNBOUND)
    stage = candidate.stages[target]
    gamma = stage.attempts[stage.retained].point
    try:
        expected: RootPoint = point_at(
            candidate.sample, owned_basis(candidate.sample), gamma, normalized=True
        )
    except FloatingPointError:
        if candidate.point is not None:
            raise ValueError(UNBOUND) from None
        return
    if candidate.point is None:
        raise ValueError(UNBOUND)
    if not same_bits(expected.gamma, candidate.point.gamma):
        raise ValueError(UNBOUND)
    if _value(expected) != _value(candidate.point):
        raise ValueError(UNBOUND)


def _observed(cert: FEPointCertificate, steps: list[WorkStep]) -> None:
    """Rebind full geometry, actual bread, correction and W by canonical routines."""
    rows = cert.rows
    if rows is None:
        if cert.geometry is not None or cert.measures is not None:
            raise ValueError(UNBOUND)
        return
    fresh = _geometry(cert.point, rows, cert.basis, steps)
    if isinstance(fresh, str):
        if cert.geometry is not None or cert.measures is not None:
            raise ValueError(UNBOUND)
        return
    actual = cert.geometry
    if type(actual) is not FEGeometry:
        raise ValueError(UNBOUND)
    names = (
        "support",
        "design",
        "moment",
        "hessian",
        "scale",
        "ordinary",
        "norm",
        "curvature",
        "refusal",
    )
    if any(_value(getattr(fresh, n)) != _value(getattr(actual, n)) for n in names):
        raise ValueError(UNBOUND)
    measured = sparse_measures(fresh, cert.controls.rcond_min, merit=cert.merit)
    steps.extend(measured.steps())
    if _value(measured) != _value(cert.measures):
        raise ValueError(UNBOUND)


def bind_root(
    candidate: AbsorbedRootCandidate, cert: FEPointCertificate | None
) -> tuple[AdmissionWork, tuple[WorkStep, ...]]:
    """Bind a root's actual start, fixed metric and selected numerical evidence."""
    _start(candidate)
    _selected(candidate)
    work = WorkLog()
    steps: list[WorkStep] = []
    metric, reason, text = merit_at_start(
        candidate.sample, candidate.root_options, candidate.initializer, work=work
    )
    if _value(metric) != _value(candidate.merit):
        raise ValueError(UNBOUND)
    if metric is None:
        if (
            candidate.stages
            or candidate.reason != reason
            or candidate.merit_reason != text
        ):
            raise ValueError(UNBOUND)
    elif candidate.merit_reason:
        raise ValueError(UNBOUND)
    if cert is not None:
        if cert.merit is not candidate.merit:
            raise ValueError(UNBOUND)
        _observed(cert, steps)
        target = candidate.target_stage
        if target is None or _value(summary_of(cert)) != _value(
            candidate.stages[target].summary
        ):
            raise ValueError(UNBOUND)
    return work.snapshot(), tuple(steps)
