"""Bind the existing operational original-slope analysis to a Gram candidate."""

from __future__ import annotations

import numpy as np

from powermean.data.collinearity import analyze_collinearity, near_omitted
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
from powermean.data.design import column_scale
from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.absorb_numeric import checked_product, checked_quotient
from powermean.fe.admission_records import EvidenceValue, WorkLog, freeze_evidence
from powermean.fe.geometry_checks import _relative_defect
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_errors import GeometryAdmissionError
from powermean.fe.groups import Grouping
from powermean.model.variance import FloatArray


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


def _near(analysis: CollinearityAnalysis) -> tuple[int, ...]:
    """Relative-policy omissions, not exact original-column dependence claims.

    The dispatch zeros a relative FE omission.  A slope-pivot omission is an exact
    copy of the first kept column when that column comes before it, else zeros.
    The geometry still applies its exact rules to that explicitly reduced design.
    """
    assert analysis.reasons is not None
    return near_omitted(analysis)


def _omission_evidence(analysis: CollinearityAnalysis) -> None:
    """Require the source's positive original-column evidence, not rank alone."""
    if analysis.reasons is None or analysis.kept is None:
        raise GeometryAdmissionError("slope_evidence", "missing completed reasons")
    kept = tuple(int(j) for j in analysis.kept)
    near = _near(analysis)
    for column, reason in zip(analysis.columns, analysis.reasons, strict=True):
        if column.index in near:
            continue
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


def _aliases(analysis: CollinearityAnalysis, raw_scales: FloatArray) -> FloatStorage:
    """The aliases of the raw columns: those of the analysis in the raw unit.

    ``raw_scales`` are the power-of-two scales of the raw columns, those an analysis
    of the raw columns uses; the analysis of the same columns in another power-of-two
    unit has the same scaled columns and relations.  A near slope is the copy of the
    first kept column; a relative FE omission is zero and has no alias.
    """
    kept = tuple(int(j) for j in analysis.kept) if analysis.kept is not None else ()
    aliases = np.zeros((len(kept), len(analysis.columns)))
    for i, j in enumerate(kept):
        aliases[i, j] = 1.0
    for j in _near(analysis):
        if (
            analysis.reasons is not None
            and analysis.reasons[j] == COLLINEAR
            and kept
            and kept[0] < j
        ):
            aliases[0, j] = 1.0
    for column in analysis.columns:
        relation = column.relation
        if relation is None:
            continue
        scales = np.asarray([raw_scales[j] for j in kept], dtype=np.float64)
        scale = np.asarray(raw_scales[column.index], dtype=np.float64)
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
    analysis: CollinearityAnalysis | None = None,
) -> EvidenceValue:
    """Run source classification once; refuse rank/set/reason/alias disagreement.

    The source owns the decision table, reversible scaling, frozen machine-rank
    cutoffs, reverse traversal, direct-rank veto and positive omission witnesses.
    This adapter neither duplicates those decisions nor proves an exact projector.
    ``analysis`` is the fit's own analysis of these columns (the dispatch's, which
    decided them); without it the analysis runs here.
    """
    if geometry.n_slopes_full == 0:
        return None
    if analysis is None:
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
    near = _near(analysis)
    first = expected[0] if expected else len(analysis.columns)
    stated = tuple(
        ABSORBED if j in near and first > j else reason
        for j, reason in enumerate(analysis.reasons or ())
    )
    if stated != proposed:
        raise GeometryAdmissionError(
            "slope_reasons", "original omission reasons disagree"
        )
    aliases = _aliases(analysis, column_scale(raw.view())).view()
    if aliases.size == 0:
        return work.evidence
    _relative_defect(
        (geometry.slope_alias_matrix - aliases).ravel(),
        norm(aliases.ravel(), "original aliases"),
        geometry.gram_rtol,
        "slope_alias",
    )
    return work.evidence
