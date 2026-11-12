"""Bind the existing operational original-slope analysis to a Gram candidate."""

from __future__ import annotations

import numpy as np

from powermean.data.collinearity import analyze_collinearity
from powermean.data.collinearity_numeric import norm
from powermean.data.collinearity_records import (
    ABSORBED,
    COLLINEAR,
    COMPLETE,
    GENERAL,
    CollinearityAnalysis,
    CollinearityProblem,
    DiagnosticControls,
)
from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.absorb_numeric import checked_product, checked_quotient
from powermean.fe.admission_records import EvidenceValue, WorkLog, freeze_evidence
from powermean.fe.geometry_checks import _relative_defect
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_errors import GeometryAdmissionError
from powermean.fe.groups import Grouping


def _evidence(analysis: CollinearityAnalysis) -> EvidenceValue:
    absorption = analysis.absorption
    summary: object = None
    if absorption is not None:
        summary = (
            absorption.status,
            absorption.reason,
            absorption.tolerance,
            absorption.column_errors,
            absorption.maximum,
            absorption.sweeps,
            absorption.visits,
            absorption.operators,
            absorption.diagnostic_passes,
            absorption.acceleration,
            absorption.state.columns,
            absorption.state.coefficients,
        )
    return freeze_evidence(
        (
            analysis.status,
            analysis.cause,
            analysis.message,
            analysis.reasons,
            analysis.columns,
            analysis.targets,
            analysis.fresh,
            analysis.within,
            analysis.factorizations,
            summary,
        )
    )


def _omission_evidence(analysis: CollinearityAnalysis) -> None:
    """Require the source's positive original-column evidence, not rank alone."""
    if analysis.reasons is None or analysis.kept is None:
        raise GeometryAdmissionError("slope_evidence", "missing completed reasons")
    kept = tuple(int(j) for j in analysis.kept)
    for column, reason in zip(analysis.columns, analysis.reasons, strict=True):
        if (
            reason == ABSORBED
            and column.kind == GENERAL
            and (
                column.span is None or not column.span.nested or not column.span.spanned
            )
        ):
            raise GeometryAdmissionError("slope_evidence", "missing original FE span")
        if reason == COLLINEAR:
            relation = column.relation
            if (
                relation is None
                or relation.basis != kept
                or not relation.span.nested
                or not relation.span.spanned
            ):
                raise GeometryAdmissionError(
                    "slope_evidence", "missing final original-column relation"
                )


def _aliases(analysis: CollinearityAnalysis) -> FloatStorage:
    kept = tuple(int(j) for j in analysis.kept) if analysis.kept is not None else ()
    aliases = np.zeros((len(kept), len(analysis.columns)))
    for i, j in enumerate(kept):
        aliases[i, j] = 1.0
    for column in analysis.columns:
        relation = column.relation
        if relation is None:
            continue
        scales = np.asarray([analysis.columns[j].scale for j in kept], dtype=np.float64)
        scale = np.asarray(column.scale, dtype=np.float64)
        ratio = checked_quotient(scale, scales, "original slope alias scales")
        aliases[:, column.index] = checked_product(
            relation.mapped, ratio, "original slope aliases"
        )
    return FloatStorage.store(aliases)


def admit_slopes(
    geometry: NumericalGeometry,
    raw: FloatStorage,
    names: tuple[str, ...],
    grouping: Grouping,
    controls: DiagnosticControls,
    *,
    mode: str,
    work: WorkLog,
) -> EvidenceValue:
    """Run source classification once; refuse rank/set/reason/alias disagreement.

    The source owns the decision table, reversible scaling, frozen machine-rank
    cutoffs, reverse traversal, direct-rank veto and positive omission witnesses.
    This adapter neither duplicates those decisions nor proves an exact projector.
    """
    if geometry.n_slopes_full == 0:
        return None
    problem = CollinearityProblem(
        raw.view(), names, geometry.structure.positions, grouping
    )
    analysis = work.run(
        "original_slope_analysis",
        lambda: analyze_collinearity(problem, controls, mode=mode),
    )
    work.evidence = _evidence(analysis)
    if analysis.status != COMPLETE:
        raise GeometryAdmissionError(
            analysis.cause, f"{analysis.status}: {analysis.message}"
        )
    _omission_evidence(analysis)
    expected = tuple(int(j) for j in analysis.kept) if analysis.kept is not None else ()
    actual = tuple(int(j) for j in geometry.kept_slope_indices)
    if expected != actual:
        raise GeometryAdmissionError("slope_rank", "original-slope kept set disagrees")
    reasons = {
        "centered_constant": ABSORBED,
        "collinear_with_retained": ABSORBED,
        "collinear_with_slopes": COLLINEAR,
    }
    proposed = tuple(
        None if reason is None else reasons[reason]
        for reason in geometry.slope_omission_reasons
    )
    if analysis.reasons != proposed:
        raise GeometryAdmissionError(
            "slope_reasons", "original omission reasons disagree"
        )
    aliases = _aliases(analysis).view()
    if aliases.size == 0:
        return work.evidence
    _relative_defect(
        (geometry.slope_alias_matrix - aliases).ravel(),
        norm(aliases.ravel(), "original aliases"),
        geometry.gram_rtol,
        "slope_alias",
    )
    return work.evidence
