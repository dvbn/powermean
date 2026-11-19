"""``balance_table``: means by treatment arm and differences from the control arm.

One row per variable.  For each arm the mean and the number of observations; for each
arm ``a`` other than the control ``0``:

==================  ================================================================
difference          ``mean_a - mean_0``; within strata with ``fe``
std. error, p       by default the two-sample comparison with unequal variances, the
                    test of ``t.test`` in R: the variance ``s_a^2/n_a + s_0^2/n_0``,
                    which is the HC2 variance of the regression of the variable on
                    arm indicators, with the Welch degrees of freedom
std. difference     ``difference / sqrt((s_a^2 + s_0^2)/2)``; not a test
==================  ================================================================

With ``cluster``, ``fe`` or ``weights`` the comparison is the regression of the
variable on arm indicators, weighted where weights are given and within strata where
``fe`` is given, with the robust or cluster-robust sandwich and the factor and
reference of the ``ssc`` preset (:mod:`powermean.describe.regression`).  The default
two-sample comparison has its own variance and reference and takes no factor.

Each variable uses its own non-missing rows.  The printed table shows one ``N``
column per arm only where the counts differ across variables; ``to_frame()`` always
holds them.
"""

from __future__ import annotations

import math
from typing import Unpack

import numpy as np

from powermean.describe import common, frame, numeric
from powermean.describe.frame import Arms, BoolArray, Context, FloatArray, Variable
from powermean.describe.options import (
    BALANCE_KEYWORDS,
    TableKeywords,
    TableOptions,
    collect,
)
from powermean.describe.regression import regress
from powermean.results import tables
from powermean.results.tables import Cell, Row, Table

_COMPARISON = ("difference", "std_error", "statistic", "df", "p_value", "std_diff")


def _two_sample(
    moments: list[tuple[float, float, int]], position: int
) -> dict[str, float]:
    """The default comparison of the arm at ``position`` with the control."""
    result = numeric.welch(moments[position], moments[0])
    usable = result.std_error > 0
    return {
        "difference": result.difference,
        "std_error": result.std_error,
        "statistic": result.difference / result.std_error if usable else math.nan,
        "df": result.df,
        "p_value": result.p_value,
    }


def _by_regression(
    values: FloatArray, rows: BoolArray, context: Context, options: TableOptions
) -> list[dict[str, float]]:
    """The comparisons of every arm with the control by one regression."""
    assert context.arms is not None
    codes = context.arms.codes[rows]
    treated = len(context.arms.levels) - 1
    indicators = np.column_stack(
        [(codes == d + 1).astype(np.float64) for d in range(treated)]
    )
    estimates = regress(
        values[rows],
        indicators,
        None if context.weights is None else context.weights[rows],
        None if context.clusters is None else context.clusters[rows],
        None if context.strata is None else context.strata[rows],
        vcov="robust",
        ssc=options.ssc,
    )
    comparisons: list[dict[str, float]] = []
    for d in range(treated):
        variance = float(estimates.covariance[d, d])
        std_error = math.sqrt(variance) if variance >= 0 else math.nan
        difference = math.nan if estimates.deficient else float(estimates.beta[d])
        statistic = difference / std_error if std_error > 0 else math.nan
        comparisons.append(
            {
                "difference": difference,
                "std_error": std_error,
                "statistic": statistic,
                "df": estimates.reference.df,
                "p_value": numeric.two_sided_p(statistic, estimates.reference),
            }
        )
    return comparisons


def _record(
    values: FloatArray, rows: BoolArray, context: Context, options: TableOptions
) -> dict[str, float]:
    """The statistics of one variable on its rows: arms, then comparisons."""
    arms = context.arms
    assert arms is not None
    moments: list[tuple[float, float, int]] = []
    record: dict[str, float] = {}
    for d, level in enumerate(arms.labels):
        chosen = rows & (arms.codes == d)
        weights = None if context.weights is None else context.weights[chosen]
        mean, variance = numeric.mean_variance(values[chosen], weights)
        moments.append((mean, variance, int(chosen.sum())))
        record[f"n[{level}]"] = float(chosen.sum())
        record[f"mean[{level}]"] = mean
        record[f"sd[{level}]"] = math.sqrt(variance) if variance >= 0 else math.nan
    two_sample = (
        context.weights is None and context.clusters is None and context.strata is None
    )
    regressed = [] if two_sample else _by_regression(values, rows, context, options)
    for d in range(1, len(arms.levels)):
        comparison = _two_sample(moments, d) if two_sample else regressed[d - 1]
        pooled = math.sqrt((moments[d][1] + moments[0][1]) / 2.0)
        comparison["std_diff"] = (
            comparison["difference"] / pooled if pooled > 0 else math.nan
        )
        for name in _COMPARISON:
            record[f"{name}[{arms.labels[d]}]"] = comparison[name]
    return record


def _columns(arms: Arms, *, counts: bool) -> list[tuple[str, str]]:
    """The printed columns as ``(frame column, heading)`` pairs."""
    columns: list[tuple[str, str]] = []
    for d, level in enumerate(arms.labels):
        columns.append((f"mean[{level}]", "mean" + common.arm_suffix(arms, d)))
        if counts:
            columns.append((f"n[{level}]", "N" + common.arm_suffix(arms, d)))
    headings = {
        "difference": "difference",
        "std_error": "std. err.",
        "p_value": "p-value",
        "std_diff": "std. diff.",
    }
    for d in range(1, len(arms.levels)):
        suffix = common.treated_suffix(arms, d)
        columns += [
            (f"{name}[{arms.labels[d]}]", heading + suffix)
            for name, heading in headings.items()
        ]
    return columns


def _cell(name: str, value: float, digits: int) -> Cell:
    """One printed statistic, formatted by its kind."""
    if name.startswith("n["):
        return tables.integer(value)
    if name.startswith("p_value["):
        return tables.p_value(value)
    if name.startswith("std_diff["):
        return tables.number(value, max(digits, 3))
    return tables.number(value, digits)


def _notes(context: Context, options: TableOptions, count: int) -> tuple[str, ...]:
    """The notes of the table: the pointer to the joint test, then the method."""
    notes = [f"{count} variable(s). Joint test: pm.balance_regression."]
    if context.weights is None and context.clusters is None and context.strata is None:
        notes.append(
            "Differences from the control arm; standard errors and p-values from "
            "the two-sample comparison with unequal variances (HC2 variance, Welch "
            "degrees of freedom)."
        )
    else:
        notes.append(
            "Differences from the control arm by the regression of each variable "
            "on arm indicators. " + common.covariance_note(options)
        )
    notes += [note for note in (common.strata_note(options),) if note]
    notes.append(
        "Standardized difference: difference / sqrt((s_a^2 + s_0^2)/2); not a test."
    )
    if options.complete:
        notes.append("Rows complete in all variables.")
    return (*notes, *context.notes)


def _records(
    variables: tuple[Variable, ...],
    context: Context,
    complete: BoolArray,
    options: TableOptions,
) -> tuple[list[dict[str, object]], bool]:
    """The records of all subgroups and whether arm counts vary across rows."""
    arms = context.arms
    assert arms is not None
    records: list[dict[str, object]] = []
    for heading, positions in context.groups:
        mask = frame.group_mask(context, positions)
        for variable in variables:
            rows = mask & complete & ~np.isnan(variable.values)
            record = _record(variable.values, rows, context, options)
            records.append({"by": heading, "variable": variable.name, **record})
    full = [
        common.arm_counts(arms, frame.group_mask(context, positions) & complete)
        for _, positions in context.groups
    ]
    varying = any(
        record[f"n[{level}]"] != full[g][d]
        for g, (heading, _) in enumerate(context.groups)
        for record in records
        if record["by"] == heading
        for d, level in enumerate(arms.labels)
    )
    return records, varying


def _sections(
    context: Context,
    records: list[dict[str, object]],
    columns: list[tuple[str, str]],
    options: TableOptions,
) -> tuple[tuple[Row, ...], ...]:
    """The printed sections of the balance table, one per subgroup."""
    sections: list[tuple[Row, ...]] = []
    for heading, _ in context.groups:
        body = [
            Row(
                (
                    tables.text(tables.label_of(str(r["variable"]), options.labels)),
                    *(
                        _cell(name, common.real(r, name), options.digits)
                        for name, _ in columns
                    ),
                )
            )
            for r in records
            if r["by"] == heading
        ]
        sections.append(tuple([tables.group_row(heading), *body] if heading else body))
    return tuple(sections)


def balance_table(
    data: object,
    treatment: object,
    vars: object = None,  # noqa: A002 - public argument name
    **options: Unpack[TableKeywords],
) -> Table:
    """Means by treatment arm and differences from the control arm.

    Parameters
    ----------
    data : pandas.DataFrame
        The data, read and never modified.
    treatment : str
        The treatment column, with two or more values.
    vars : str or sequence of str, optional
        The variables.  ``None`` takes every numeric column except the treatment,
        weight, cluster, strata and subgroup columns.
    **options : TableKeywords
        ``control``, ``bins``, ``cluster``, ``fe``, ``weights``, ``ssc``, ``by``,
        ``complete``, ``expand``, ``sample``, ``labels`` and ``digits`` (default
        2); see :mod:`powermean.describe.options`.

    Returns
    -------
    Table
        One row per variable (module docstring).  ``to_frame()`` has, for each arm
        ``d``, the columns ``n[d]``, ``mean[d]`` and ``sd[d]`` and, for each arm
        ``a`` other than the control, ``difference[a]``, ``std_error[a]``,
        ``statistic[a]``, ``df[a]`` (the Welch degrees of freedom, those of a ``t``
        reference, or infinity under the normal reference), ``p_value[a]`` and
        ``std_diff[a]``; its ``attrs`` name the control and the arms.

    Raises
    ------
    SpecificationError
        An unknown keyword, a value outside its domain, an absent column, a named
        column that is not numeric without ``expand=True``, a ``control`` that is
        not a value of the treatment.
    SampleError
        A negative weight; fewer than two arms on the usable rows.
    """
    resolved = collect(options, "balance_table", BALANCE_KEYWORDS, digits=2)
    context = frame.context(data, resolved, treatment)
    arms = context.arms
    assert arms is not None
    complete = np.ones(len(context.frame), dtype=np.bool_)
    variables = frame.variables(
        context.frame,
        vars,
        resolved,
        exclude=(treatment,),
        support=complete if resolved.complete else None,
    )
    records, varying = _records(variables, context, complete, resolved)
    columns = _columns(arms, counts=varying)
    attrs = {
        "treatment": arms.name,
        "arms": list(arms.levels),
        "control": arms.levels[0],
    }
    return Table(
        title="Balance table",
        subtitle=common.arms_subtitle(arms, context.usable & complete),
        columns=(tables.EMPTY, *(tables.text(heading) for _, heading in columns)),
        sections=_sections(context, records, columns, resolved),
        notes=_notes(context, resolved, len(variables)),
        align=tables.alignment("l", rest="r", count=len(columns)),
        frame=common.assemble(records, ["variable"], attrs),
    )
