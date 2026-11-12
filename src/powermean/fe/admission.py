"""Fresh per-fit structural admission of the unchanged numerical candidate.

No public export or cross-fit trust cache. Successful admission owns raw columns
and binds actual positional rows and ordered original FE labels/codes once.
Exact FE quotient evidence and operational floating original-slope evidence are
separate. Neither is a current-point/Hessian, convergence or inference certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np

from powermean.data.collinearity_records import DiagnosticControls
from powermean.fe._cancel_storage import BuilderRecord, FloatStorage
from powermean.fe.admission_budget import AdmissionBudget, reserve_admission
from powermean.fe.admission_incidence import QuotientRank, quotient_rank
from powermean.fe.admission_records import (
    AdmissionError,
    AdmissionWork,
    EvidenceValue,
    WorkLog,
)
from powermean.fe.admission_slopes import admit_slopes
from powermean.fe.geometry_checks import GeometryCheck, GeometryInputs, check_geometry
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_errors import GeometryAdmissionError
from powermean.fe.geometry_storage import _build_record
from powermean.fe.groups import Grouping, Labels, _positions, encode_block
from powermean.model.variance import FloatArray, _finite_array


class AdmissionInputs(NamedTuple):
    """Current raw sample, including original ordered labels of every FE block."""

    slopes: GeometryInputs
    cancelled: Labels
    retained: tuple[Labels, ...]


@dataclass(frozen=True, eq=False, init=False)
class AdmittedGeometry(BuilderRecord):
    """One immutable actual structural problem, never an externally set flag."""

    __slots__ = (
        "_raw_slopes",
        "budget",
        "check",
        "controls",
        "geometry",
        "mode",
        "quotient",
        "slope_evidence",
        "work",
    )

    geometry: NumericalGeometry
    check: GeometryCheck
    quotient: QuotientRank
    work: AdmissionWork
    budget: AdmissionBudget
    slope_evidence: EvidenceValue
    controls: DiagnosticControls
    mode: str
    _raw_slopes: FloatStorage

    @property
    def raw_slopes(self) -> FloatArray:
        """Original columns with immutable bytes and fresh outward metadata."""
        return self._raw_slopes.view()


def _snapshot(geometry: NumericalGeometry, inputs: GeometryInputs) -> FloatStorage:
    raw = inputs.raw_slopes
    if not isinstance(raw, np.ndarray) or raw.dtype.kind not in "biuf":
        raise TypeError("raw slopes must be a real numeric ndarray")
    if raw.shape != (geometry.structure.n_rows, geometry.n_slopes_full):
        raise ValueError("raw slopes must have shape (n_rows, n_slopes)")
    return FloatStorage.store(_finite_array(raw, "raw slopes"))


def _grouping(geometry: NumericalGeometry, inputs: AdmissionInputs) -> Grouping:
    layout = geometry.structure
    if not isinstance(inputs.retained, tuple) or len(inputs.retained) != len(
        layout.block_sizes
    ):
        raise ValueError("one ordered current label vector per retained block required")
    for labels in (inputs.cancelled, *inputs.retained):
        if isinstance(labels, np.ndarray):
            if labels.shape != (layout.n_rows,):
                raise ValueError("FE labels must have one value per row")
        elif not isinstance(labels, Sequence) or len(labels) != layout.n_rows:
            raise ValueError(
                "FE labels must be a sized sequence with one value per row"
            )
    cancelled = encode_block(inputs.cancelled, name="cancelled")
    retained = tuple(
        encode_block(labels, name=f"retained{j}")
        for j, labels in enumerate(inputs.retained)
    )
    if (
        cancelled.n_rows != layout.n_rows
        or not np.array_equal(cancelled.codes, layout.cancel_codes)
        or cancelled.levels != layout.cancel_levels
    ):
        raise GeometryAdmissionError(
            "cancelled_binding", "current cancelled labels/codes differ"
        )
    for block, codes, levels in zip(
        retained, layout.retained_sorted_codes, layout.retained_levels, strict=True
    ):
        if (
            block.n_rows != layout.n_rows
            or not np.array_equal(block.codes[layout.order], codes)
            or block.levels != levels
        ):
            raise GeometryAdmissionError(
                "retained_binding", "current ordered retained labels/codes differ"
            )
    return Grouping((cancelled, *retained), layout.positions)


def _agree(geometry: NumericalGeometry, exact: QuotientRank) -> None:
    annihilated = tuple(tuple(bool(x) for x in a) for a in geometry.annihilated_levels)
    if (
        exact.rank != geometry.n_retained_identified
        or exact.block_ranks != geometry.block_ranks
        or exact.annihilated_levels != annihilated
        or sum(exact.block_ranks) - exact.rank != geometry.cross_block_redundancy
    ):
        raise GeometryAdmissionError(
            "FE_quotient_rank", "exact incidence and candidate disagree"
        )


def _admit(
    geometry: NumericalGeometry,
    inputs: AdmissionInputs,
    *,
    controls: DiagnosticControls,
    mode: str,
    budget: AdmissionBudget,
    work: WorkLog,
) -> AdmittedGeometry:
    raw = work.run("raw_capture", lambda: _snapshot(geometry, inputs.slopes))
    positions = inputs.slopes.positions
    if not isinstance(positions, np.ndarray) or positions.shape != (
        geometry.structure.n_rows,
    ):
        raise ValueError("positions must be an ndarray with one value per row")
    owned_positions = work.run(
        "position_capture", lambda: _positions(positions, geometry.structure.n_rows)
    )
    bound = GeometryInputs(raw.view(), owned_positions, inputs.slopes.source_columns)
    check = work.run(
        "candidate_check",
        lambda: check_geometry(geometry, bound, max_bytes=budget.prospective_bytes),
    )
    grouping = work.run("label_binding", lambda: _grouping(geometry, inputs))
    exact = work.run("FE_quotient", lambda: quotient_rank(geometry.structure, work))
    _agree(geometry, exact)
    evidence = admit_slopes(
        geometry,
        raw,
        check.source_columns,
        grouping,
        controls,
        mode=mode,
        work=work,
    )
    # Grouping, absorption/residuals and rich analysis records die here. Shared
    # geometry retains original labels/codes; evidence retains only owned values.
    return cast(
        AdmittedGeometry,
        _build_record(
            AdmittedGeometry,
            geometry=geometry,
            check=check,
            quotient=exact,
            work=work.snapshot(),
            budget=budget,
            slope_evidence=evidence,
            _raw_slopes=raw,
            controls=controls,
            mode=mode,
        ),
    )


def admit_geometry(
    geometry: NumericalGeometry,
    inputs: AdmissionInputs,
    *,
    controls: DiagnosticControls,
    mode: str,
    max_bytes: int,
) -> AdmittedGeometry:
    """Freshly bind and admit the current external sample, once for this fit.

    All added managed workspace is reserved before raw or label copies, checks,
    exact rank or slope work. Explicit controls are unchanged. Refusals preserve
    partial work/evidence; incompatible candidates are never retuned or replaced.
    A new external call always repeats admission on current data.
    """
    if type(geometry) is not NumericalGeometry:
        raise TypeError("geometry must come from numerical_geometry")
    budget = reserve_admission(geometry.design, max_bytes)
    if type(inputs) is not AdmissionInputs or type(inputs.slopes) is not GeometryInputs:
        raise TypeError("use AdmissionInputs with a GeometryInputs request")
    if type(controls) is not DiagnosticControls or mode not in ("drop", "error"):
        raise ValueError("explicit DiagnosticControls and drop/error mode required")
    controls = DiagnosticControls(
        controls.tolerance, controls.max_sweeps, controls.acceleration
    )
    work = WorkLog()
    work.begin("structural_admission")
    try:
        with np.errstate(over="raise", invalid="raise"):
            record = _admit(
                geometry,
                inputs,
                controls=controls,
                mode=mode,
                budget=budget,
                work=work,
            )
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as error:
        cause = (
            error.cause
            if isinstance(error, GeometryAdmissionError)
            else type(error).__name__
        )
        raise AdmissionError(cause, str(error), work) from error
    work.complete("structural_admission")
    # Record only the now-completed top-level call, sharing every owned buffer.
    object.__setattr__(record, "work", work.snapshot())
    return record
