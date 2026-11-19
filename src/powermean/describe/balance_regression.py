"""``balance_regression``: treatment status on all covariates, with the joint test.

For each arm other than the control, on the rows of that arm and of the control: a
linear probability model of the arm indicator on all the variables, with the
coefficient table and the joint test that every coefficient on the variables is zero.
With ``continuous=True`` the single regression is that of the dose on the variables,
with the same joint test.  Rows with a missing value in any variable are dropped from
the regression and counted.

==================  ================================================================
covariance          joint test
==================  ================================================================
robust (default)    Wald test with the robust covariance and the factor of the
                    ``ssc`` preset; chi-square, or ``F`` under ``"regression"``
``cluster``         the same with the cluster-robust covariance
``vcov="classical"``  the ``F`` statistic of ordinary least squares under
                    homoskedasticity, which is the statistic of ``lm`` in R
==================  ================================================================

``fe`` absorbs one classification of strata, as in the balance table; the intercept
is then absorbed and not reported.  An expanded categorical column enters without
its first level.  A design without full column rank is refused
(``identification.table_rank``): the function does not drop a variable on its own.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Unpack

import numpy as np

from powermean.describe import common, frame, numeric
from powermean.describe.frame import BoolArray, Context, FloatArray, Variable
from powermean.describe.options import (
    REGRESSION_KEYWORDS,
    TableKeywords,
    TableOptions,
    collect,
)
from powermean.describe.regression import Estimates, regress
from powermean.errors import IdentificationError
from powermean.results import tables
from powermean.results.tables import Cell, Row, Table

INTERCEPT = "Intercept"


class Comparison(NamedTuple):
    """One regression: its heading, dependent variable and rows before missing."""

    heading: str
    target: FloatArray
    rows: BoolArray


class Fitted(NamedTuple):
    """One fitted regression with its joint test and its counts."""

    by: str
    heading: str
    estimates: Estimates
    joint: numeric.JointTest
    dropped: int


def _comparisons(
    context: Context, dose: Variable | None, mask: BoolArray
) -> list[Comparison]:
    """The regressions of one subgroup: one per treated arm, or the dose."""
    if dose is not None:
        return [Comparison(dose.name, dose.values, mask & ~np.isnan(dose.values))]
    arms = context.arms
    assert arms is not None
    return [
        Comparison(
            arms.text(d),
            (arms.codes == d).astype(np.float64),
            mask & ((arms.codes == d) | (arms.codes == 0)),
        )
        for d in range(1, len(arms.levels))
    ]


def _fit(
    comparison: Comparison,
    regressors: list[Variable],
    context: Context,
    options: TableOptions,
    by: str,
) -> Fitted:
    """Fit one regression on its complete rows and test its slopes jointly."""
    complete = comparison.rows & common.complete_mask(
        context, [v.values for v in regressors]
    )
    estimates = regress(
        comparison.target[complete],
        np.column_stack([v.values[complete] for v in regressors]),
        None if context.weights is None else context.weights[complete],
        None if context.clusters is None else context.clusters[complete],
        None if context.strata is None else context.strata[complete],
        vcov=options.vcov,
        ssc=options.ssc,
    )
    if estimates.deficient:
        raise IdentificationError(
            "identification.table_rank",
            {
                "target": comparison.heading,
                "columns": len(regressors) + (context.strata is None),
                "rows": int(complete.sum()),
            },
        )
    joint = numeric.joint_test(
        estimates.beta, estimates.covariance, estimates.reference
    )
    dropped = int(comparison.rows.sum()) - int(complete.sum())
    return Fitted(by, comparison.heading, estimates, joint, dropped)


def _coefficients(
    fitted: Fitted, names: list[str], intercept: str
) -> list[dict[str, object]]:
    """The coefficient records of one regression, the intercept last."""
    estimates = fitted.estimates
    values = [*estimates.beta.tolist()]
    variances = [*np.diag(estimates.covariance).tolist()]
    terms = list(names)
    if not math.isnan(estimates.intercept):
        terms.append(intercept)
        values.append(estimates.intercept)
        variances.append(estimates.intercept_variance)
    records: list[dict[str, object]] = []
    for term, value, variance in zip(terms, values, variances, strict=True):
        std_error = math.sqrt(variance) if variance >= 0 else math.nan
        statistic = value / std_error if std_error > 0 else math.nan
        records.append(
            {
                "by": fitted.by,
                "comparison": fitted.heading,
                "term": term,
                "estimate": value,
                "std_error": std_error,
                "statistic": statistic,
                "p_value": numeric.two_sided_p(statistic, estimates.reference),
            }
        )
    return records


def _joint_text(joint: numeric.JointTest, digits: int) -> Cell:
    """``chi2(3) = 2.41`` or ``F(3, 1996) = 0.80``."""
    if math.isnan(joint.statistic):
        return tables.EMPTY
    degrees = (
        f"{joint.df_num}"
        if joint.distribution == "chi2"
        else f"{joint.df_num}, {joint.df_denom:g}"
    )
    return tables.text(
        f"{joint.distribution}({degrees}) = {joint.statistic:.{digits}f}"
    )


def _term_rows(
    fits: list[Fitted],
    records: list[dict[str, object]],
    terms: list[str],
    options: TableOptions,
) -> tuple[Row, ...]:
    """The coefficient and standard-error rows for one subgroup."""
    lookup = {(r["comparison"], r["term"]): r for r in records if r["by"] == fits[0].by}
    body: list[Row] = [tables.group_row(fits[0].by)] if fits[0].by else []
    for term in terms:
        found = [lookup[(fit.heading, term)] for fit in fits]
        estimates = [
            tables.number(common.real(r, "estimate"), options.digits) for r in found
        ]
        errors = [
            tables.parenthesized(
                tables.number(common.real(r, "std_error"), options.digits)
            )
            for r in found
        ]
        body.append(
            Row((tables.text(tables.label_of(term, options.labels)), *estimates))
        )
        body.append(Row((tables.EMPTY, *errors)))
    return tuple(body)


def _block_rows(fits: list[Fitted], options: TableOptions) -> tuple[Row, ...]:
    """The joint-test and count rows for one subgroup."""
    block = [
        Row((tables.text("Joint test"), *(_joint_text(f.joint, 2) for f in fits))),
        Row((tables.text("p-value"), *(tables.p_value(f.joint.p_value) for f in fits))),
        Row(
            (
                tables.text("Observations"),
                *(tables.integer(f.estimates.n) for f in fits),
            )
        ),
        Row(
            (
                tables.text("Rows dropped (missing)"),
                *(tables.integer(f.dropped) for f in fits),
            )
        ),
    ]
    if options.cluster is not None:
        counts = (tables.integer(f.estimates.clusters) for f in fits)
        block.append(Row((tables.text("Clusters"), *counts)))
    if options.fe is not None:
        counts = (tables.integer(f.estimates.strata) for f in fits)
        block.append(Row((tables.text("Strata (absorbed)"), *counts)))
    return tuple(block)


def _sections(
    fits: list[Fitted],
    records: list[dict[str, object]],
    terms: list[str],
    options: TableOptions,
) -> list[tuple[Row, ...]]:
    """The coefficient rows and the block of the joint test, for one subgroup."""
    return [_term_rows(fits, records, terms, options), _block_rows(fits, options)]


def _notes(context: Context, options: TableOptions, *, continuous: bool) -> list[str]:
    """The notes of the table: the model, the covariance and the test."""
    model = (
        "Regression of the dose on all variables."
        if continuous
        else "Linear probability model of each arm against the control on all "
        "variables, on the rows of the two arms."
    )
    if options.vcov == "classical":
        inference = (
            "Covariance: classical (homoskedastic); the joint test is the F "
            "statistic of ordinary least squares."
        )
    else:
        inference = common.covariance_note(options) + (
            " Joint Wald test that every coefficient on the variables is zero."
        )
    notes = [model, inference, "Standard errors in parentheses."]
    notes += [note for note in (common.strata_note(options),) if note]
    return [*notes, *context.notes]


def _fit_all(
    context: Context,
    vars_arg: object,
    treatment: object,
    options: TableOptions,
) -> tuple[list[Fitted], list[dict[str, object]], list[str]]:
    """Fit every comparison across all subgroups and return fits, records, terms."""
    dose = (
        frame.variables(context.frame, [treatment], options, argument="treatment")[0]
        if options.continuous
        else None
    )
    found = frame.variables(context.frame, vars_arg, options, exclude=(treatment,))
    regressors = [v for v in found if not v.reference]
    fits = [
        _fit(comparison, regressors, context, options, heading)
        for heading, positions in context.groups
        for comparison in _comparisons(
            context, dose, frame.group_mask(context, positions)
        )
    ]
    names = [v.name for v in regressors]
    intercept = INTERCEPT
    if options.fe is None and intercept in names:
        intercept = frame._distinct_labels(
            [*names, intercept], lambda i: "constant" if i == len(names) else None
        )[-1]
    records = [
        record for fitted in fits for record in _coefficients(fitted, names, intercept)
    ]
    terms = [*names, intercept] if options.fe is None else names
    return fits, records, terms


def _joint_records(fits: list[Fitted]) -> list[dict[str, object]]:
    """The joint-test metadata records stored in ``frame.attrs['joint_test']``."""
    return [
        {
            "by": f.by,
            "comparison": f.heading,
            **f.joint._asdict(),
            "n": f.estimates.n,
            "dropped": f.dropped,
            "clusters": f.estimates.clusters,
            "strata": f.estimates.strata,
        }
        for f in fits
    ]


def _all_sections(
    context: Context,
    fits: list[Fitted],
    records: list[dict[str, object]],
    terms: list[str],
    options: TableOptions,
) -> tuple[tuple[Row, ...], ...]:
    """All coefficient and joint-test sections across subgroups."""
    sections: list[tuple[Row, ...]] = []
    for heading, _ in context.groups:
        group = [fitted for fitted in fits if fitted.by == heading]
        sections += _sections(group, records, terms, options)
    return tuple(sections)


def balance_regression(
    data: object,
    treatment: object,
    vars: object = None,  # noqa: A002 - public argument name
    **options: Unpack[TableKeywords],
) -> Table:
    """Regress treatment status on all covariates and test them jointly.

    Parameters
    ----------
    data : pandas.DataFrame
        The data, read and never modified.
    treatment : str
        The treatment column: two or more arms, or a dose with ``continuous=True``.
    vars : str or sequence of str, optional
        The variables.  ``None`` takes every numeric column except the treatment,
        weight, cluster, strata and subgroup columns.
    **options : TableKeywords
        ``control``, ``bins``, ``continuous``, ``cluster``, ``fe``, ``weights``,
        ``vcov`` (``"robust"`` or ``"classical"``), ``ssc``, ``by``, ``expand``,
        ``sample``, ``labels`` and ``digits`` (default 3); see
        :mod:`powermean.describe.options`.

    Returns
    -------
    Table
        One column per arm other than the control (or one for the dose): the
        coefficients with their standard errors, then the joint test, its p-value,
        the number of observations and the number of rows dropped for a missing
        value.  ``to_frame()`` is indexed by ``comparison`` and ``term`` with the
        columns ``estimate``, ``std_error``, ``statistic`` and ``p_value``; its
        ``attrs["joint_test"]`` holds one record per regression (statistic,
        distribution, degrees of freedom, p-value, observations, dropped rows,
        clusters, strata).

    Raises
    ------
    SpecificationError
        An unknown keyword, a value outside its domain, contradictory keywords
        (``vcov="classical"`` with ``cluster``; ``continuous`` with ``bins``), an
        absent column, a column that is not numeric.
    SampleError
        A negative weight; fewer than two arms on the usable rows.
    IdentificationError
        A regression whose design has not full column rank.
    """
    resolved = collect(options, "balance_regression", REGRESSION_KEYWORDS, digits=3)
    context = frame.context(data, resolved, None if resolved.continuous else treatment)
    fits, records, terms = _fit_all(context, vars, treatment, resolved)
    headings = [fitted.heading for fitted in fits if fitted.by == fits[0].by]
    control = None if context.arms is None else context.arms.levels[0]
    attrs = {
        "treatment": str(treatment),
        "control": control,
        "joint_test": _joint_records(fits),
    }
    return Table(
        title="Balance regression",
        subtitle="" if context.arms is None else f"control: {context.arms.text(0)}",
        columns=(tables.EMPTY, *(tables.text(heading) for heading in headings)),
        sections=_all_sections(context, fits, records, terms, resolved),
        notes=tuple(_notes(context, resolved, continuous=resolved.continuous)),
        align=tables.alignment("l", rest="c", count=len(headings)),
        frame=common.assemble(records, ["comparison", "term"], attrs),
    )
