"""``etable``: several fitted results side by side.

One column per result. The table reads only public result projections (``coef``,
``tidy``, ``effects``), ``sample`` and ``provenance``; an explicit ``vcov=`` uses
``with_vcov``. Every number comes from those methods; nothing is fitted, corrected
or tested again here. A result without a certificate, or one that refuses the
requested covariance, keeps its column with the relevant status and notes.

Blocks of rows, separated by a rule:

1. One pair of rows per term, in the order of first appearance across the results:
   the estimate with its stars, then the standard error in parentheses
   (``uncertainty="se"``), the confidence interval in brackets (``"ci"``) or the
   p-value in parentheses (``"p"``).
2. The model: the estimator ("PPML" and "GPML" for ``omega = 1`` with ``tau = 1`` and
   ``tau = 2`` only, "log OLS" for ``omega = 0``, the indices otherwise), the mean
   order, one row per fixed-effect term ("Estimated", "Cancelled" or "No"), the
   coefficient correction where a result has one, the covariance and its number of
   clusters, and the status of the certificate where a result is not plainly
   ``verified``.
3. Observations by default; the share of zeros when requested through ``stats``.

Scale (``scale=``).  ``"coefficient"`` prints the reported coefficients.
``"effect"`` prints ``100 (exp(b) - 1)``, a transformation of each reported
coefficient coordinate in percent, as ``result.effects()`` reports it: the
standard error is the delta-method one, and the interval transforms the endpoints of
the coefficient interval.  The intercept is left out on that
scale unless ``keep`` names it.

Stars compare the two-sided p-value of a zero coefficient, under the reference
distribution of each result, with the thresholds ``0.10``, ``0.05`` and ``0.01`` by
default; a coefficient without usable inference has none.

A named HC covariance left missing by a hat value of one (owner's ruling 74; its
record's ``diagnostics`` name ``covariance.unstable_leverage``) prints ``NA`` in its
uncertainty cells, as ``fixest::etable`` prints a missing value, ``(NA)`` or
``[NA, NA]``, and one note under the table names its columns (owner's ruling 88).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Unpack, cast

import pandas as pd

from powermean import messages
from powermean.errors import SpecificationError
from powermean.results import tables
from powermean.results.etable_models import (
    VERIFIED,
    FittedResult,
    Model,
    index_text,
    model_of,
)
from powermean.results.etable_options import (
    EtableKeywords,
    EtableOptions,
    collect,
    refuse,
)
from powermean.results.tables import Cell, Row, Table

INTERCEPT = "Intercept"
# The cell of a named HC error missing for a hat value of one (owner's ruling 88),
# as ``fixest::etable`` prints it, and the diagnostic that the record carries then.
MISSING = Cell("NA")
UNIT_LEVERAGE = "covariance.unstable_leverage"


# ------------------------------------------------------------------------- rows


def _terms(models: Sequence[Model], options: EtableOptions) -> list[str]:
    """The terms of the table: the union in order of first appearance, filtered."""
    union: list[str] = []
    for model in models:
        union += [term for term in model.terms.index if term not in union]
    for name in (*(options.keep or ()), *options.drop):
        if name not in union:
            raise SpecificationError(
                "specification.invalid_value",
                {
                    "argument": "keep or drop",
                    "value": name,
                    "valid": messages.format_keys(union),
                },
            )
    if options.keep is not None:
        return [term for term in options.keep if term not in options.drop]
    hidden = set(options.drop)
    if options.scale == "effect":
        hidden.add(INTERCEPT)
    return [term for term in union if term not in hidden]


def _threshold(value: float) -> str:
    """A threshold as printed: two decimals where they are exact, ``0.10``."""
    return f"{value:.2f}" if round(value, 2) == value else f"{value:g}"


def _stars_of(p: float, thresholds: tuple[float, ...]) -> str:
    """One star per threshold that the p-value is strictly below."""
    return "*" * sum(1 for threshold in thresholds if p < threshold)


def _unit_leverage(model: Model) -> bool:
    """Whether the model's named HC covariance is missing for a hat value of one."""
    record = model.projection.get("covariance")
    if not isinstance(record, Mapping):
        return False
    diagnostics = record.get("diagnostics") or ()
    return UNIT_LEVERAGE in tuple(cast("Sequence[str]", diagnostics))


def _term_rows(term: str, models: Sequence[Model], options: EtableOptions) -> list[Row]:
    """The estimate row and the uncertainty row of one term."""
    estimates: list[Cell] = []
    second: list[Cell] = []
    for model in models:
        if term not in model.terms.index:
            estimates.append(tables.EMPTY)
            second.append(tables.EMPTY)
            continue
        row = cast("Mapping[str, float]", model.terms.loc[term].to_dict())
        p = float(row["p_value"])
        value = tables.number(float(row["estimate"]), options.digits)
        estimates.append(tables.starred(value, _stars_of(p, options.stars)))
        if _unit_leverage(model) and not math.isfinite(float(row["std_error"])):
            # Owner's ruling 88, as ``fixest::etable`` prints a missing value.
            second.append(
                tables.bracketed(MISSING, MISSING)
                if options.uncertainty == "ci"
                else tables.parenthesized(MISSING)
            )
        elif options.uncertainty == "se":
            error = tables.number(float(row["std_error"]), options.digits)
            second.append(tables.parenthesized(error))
        elif options.uncertainty == "p":
            second.append(tables.parenthesized(tables.p_value(p)))
        else:
            low = tables.number(float(row["conf_low"]), options.digits)
            high = tables.number(float(row["conf_high"]), options.digits)
            second.append(tables.bracketed(low, high))
    label = tables.label_of(term, options.labels)
    return [Row((tables.text(label), *estimates)), Row((tables.EMPTY, *second))]


def _text_row(label: str, values: Sequence[str]) -> Row:
    """One row of text cells with its row label."""
    return Row((tables.text(label), *(tables.text(value) for value in values)))


def _effect_rows(models: Sequence[Model], options: EtableOptions) -> list[Row]:
    """One row per fixed-effect block across the models."""
    blocks: list[str] = []
    for model in models:
        blocks += [block for block in model.effects if block not in blocks]
    rows: list[Row] = []
    for block in blocks:
        label = tables.label_of(block, options.labels) + " effects"
        rows.append(_text_row(label, [m.effects.get(block, "No") for m in models]))
    return rows


def _covariance_and_status_rows(models: Sequence[Model]) -> list[Row]:
    """The covariance, cluster and status rows of the model block."""
    rows = [
        _text_row("Covariance", [m.covariance for m in models]),
        _text_row("Convention", [m.convention for m in models]),
        _text_row("Reference", [m.reference for m in models]),
    ]
    if any(m.clusters for m in models):
        rows.append(_text_row("Clusters", [m.clusters for m in models]))
    counted = [
        str(
            cast("Mapping[str, object]", m.projection.get("covariance", {})).get(
                "absorbed_count", ""
            )
        )
        for m in models
    ]
    if any(counted):
        # Fixed-effect log-OLS (ruling 80): which count of the absorbed effects.
        rows.append(_text_row("Absorbed count", counted))
    if any(m.status != VERIFIED for m in models):
        rows.append(_text_row("Status", [m.status for m in models]))
    return rows


def _model_rows(models: Sequence[Model], options: EtableOptions) -> list[Row]:
    """The block that describes each model: estimator, effects, covariance."""
    rows = [
        _text_row("Estimator", [m.estimator for m in models]),
        _text_row("Mean order (omega)", [index_text(m.omega) for m in models]),
        *_effect_rows(models, options),
    ]
    if any(m.correction for m in models):
        rows.append(
            _text_row("Coefficient correction", [m.correction or "No" for m in models])
        )
    return [*rows, *_covariance_and_status_rows(models)]


def _statistic_rows(models: Sequence[Model], options: EtableOptions) -> list[Row]:
    """The statistics that the results expose: observations and zeros."""
    rows: list[Row] = []
    for name in options.stats:
        if name == "nobs":
            cells = [tables.integer(m.nobs) for m in models]
            rows.append(Row((tables.text("Observations"), *cells)))
        else:
            shares = [
                tables.percent(m.zeros / m.nobs if m.nobs else float("nan"))
                for m in models
            ]
            rows.append(Row((tables.text("Zeros"), *shares)))
    return rows


def _scale_and_uncertainty_notes(options: EtableOptions) -> list[str]:
    """The notes describing the scale, uncertainty row and significance stars."""
    percent = f"{100 * options.level:g}%"
    notes: list[str] = []
    if options.scale == "effect":
        notes.append(
            "Entries are 100 (exp(b) - 1) for each reported coefficient "
            "coordinate; not a standalone regressor change."
        )
    if options.uncertainty == "se":
        delta = " (delta method)" if options.scale == "effect" else ""
        notes.append(f"Standard errors{delta} in parentheses.")
    elif options.uncertainty == "p":
        notes.append("Two-sided p-values of a zero coefficient in parentheses.")
    else:
        transformed = (
            ", transformed from the coefficient interval"
            if options.scale == "effect"
            else ""
        )
        notes.append(f"{percent} confidence intervals in brackets{transformed}.")
    if options.stars:
        marks = ", ".join(
            f"{'*' * (position + 1)} p<{_threshold(threshold)}"
            for position, threshold in reversed(list(enumerate(options.stars)))
        )
        notes.append(
            f"{marks}; two-sided tests of a zero coefficient under each model's "
            "reference distribution, from unrounded values."
        )
    return notes


def _saturated_notes(models: Sequence[Model]) -> list[str]:
    """The note of each saturated fit's missing covariance, once per distinct text."""
    found: list[str] = []
    for model in models:
        record = model.projection.get("covariance")
        if isinstance(record, Mapping) and record.get("saturated"):
            note = str(record.get("note"))
            if note not in found:
                found.append(note)
    return found


def _model_notes(models: Sequence[Model]) -> list[str]:
    """The notes conditioned on the models' corrections, statuses and orders."""
    notes: list[str] = []
    if any(m.correction for m in models):
        notes.append(
            "Corrected coefficients are reported with first-order inference at "
            "the certified uncorrected root."
        )
    if any(m.status.startswith("not certified") for m in models):
        notes.append(
            "A column without a certified solution prints no coefficient: the "
            "package withholds inference at such a point."
        )
    if any(m.covariance.startswith("refused") for m in models):
        notes.append(
            "A column whose result refuses the requested covariance prints its "
            "coefficients without standard errors, intervals, p-values or stars."
        )
    for note in _saturated_notes(models):
        notes.append(note)
    if len({m.omega for m in models}) > 1:
        notes.append(
            "Models with different mean orders describe different conditional "
            "power means; their coefficients do not estimate one common parameter."
        )
    return notes


def _missing_notes(models: Sequence[Model], names: Sequence[str]) -> list[str]:
    """The line on the named HC errors missing for a hat value of one (ruling 88)."""
    columns = [
        name if name == model.covariance else f"{name} ({model.covariance})"
        for model, name in zip(models, names, strict=True)
        if _unit_leverage(model) and model.terms["std_error"].isna().any()
    ]
    if not columns:
        return []
    return [
        f"NA: the standard errors of {', '.join(columns)} are missing: an "
        "observation has leverage one (a hat value of one), where HC2 to HC5 and "
        "HC4m are undefined; see provenance.covariance."
    ]


def _notes(
    models: Sequence[Model], options: EtableOptions, names: Sequence[str]
) -> tuple[str, ...]:
    """The notes: scale, uncertainty, stars, corrections, differing orders."""
    return (
        *_scale_and_uncertainty_notes(options),
        *_model_notes(models),
        *_missing_notes(models, names),
        *options.notes,
    )


def _frame(
    models: Sequence[Model], names: Sequence[str], terms: Sequence[str]
) -> pd.DataFrame:
    """The unrounded numbers: one row per model and term, model records in ``attrs``."""
    pieces = [
        model.terms.loc[[t for t in terms if t in model.terms.index]].assign(model=name)
        for model, name in zip(models, names, strict=True)
    ]
    frame = pd.concat(pieces).reset_index().set_index(["model", "term"])
    frame.attrs["models"] = [
        {
            "model": name,
            "estimator": m.estimator,
            "omega": m.omega,
            "fixed_effects": dict(m.effects),
            "correction": m.correction,
            "covariance": m.covariance,
            "clusters": m.clusters,
            "status": m.status,
            "nobs": m.nobs,
            "zeros": m.zeros,
            "convention": m.convention,
            "reference": m.reference,
            "projection": dict(m.projection),
        }
        for m, name in zip(models, names, strict=True)
    ]
    return frame


def _unpack_results(
    results: tuple[object, ...], options: EtableOptions
) -> tuple[Sequence[FittedResult], tuple[str, ...]]:
    """Validate the positional results and resolve their column names."""
    given: Sequence[object] = results
    if len(results) == 1 and isinstance(results[0], list | tuple):
        given = results[0]
    expected = "one or more fitted results"
    if not given:
        raise refuse("results", given, expected)
    for item in given:
        if not all(callable(getattr(item, name, None)) for name in ("tidy", "effects")):
            raise refuse("results", type(item).__name__, expected)
        sample = getattr(item, "sample", None)
        provenance = getattr(item, "provenance", None)
        if not all(hasattr(sample, name) for name in ("count", "zeros")) or not all(
            hasattr(provenance, name)
            for name in ("estimator", "covariance", "certified", "status")
        ):
            raise refuse("results", type(item).__name__, expected)
        if options.vcov is not None and not callable(getattr(item, "with_vcov", None)):
            raise refuse("results", type(item).__name__, expected)
    fitted = cast("Sequence[FittedResult]", given)
    names = options.names or tuple(f"({i + 1})" for i in range(len(fitted)))
    if len(names) != len(fitted):
        raise refuse("names", options.names, f"{len(fitted)} names, one per result")
    return fitted, names


def etable(*results: object, **options: Unpack[EtableKeywords]) -> Table:
    """Tabulate several fitted results, one column each.

    Parameters
    ----------
    *results : Result
        The results, or one list of them (``pm.etable([a, b])``).
    **options : EtableKeywords
        ``names``, ``scale`` (``"coefficient"`` or ``"effect"``), ``uncertainty``
        (``"se"``, ``"ci"`` or ``"p"``), ``level``, ``vcov``, ``stars``, ``keep``,
        ``drop``, ``labels``, ``digits`` (default 3), ``stats``, ``title`` and
        ``notes``; see :mod:`powermean.results.etable_options`.

    Returns
    -------
    Table
        The table (module docstring), with ``to_frame()``, ``to_markdown()``,
        ``to_latex()``, ``to_html()`` and ``to_text()``.  ``to_frame()`` holds one
        row per model and term with ``estimate``, ``std_error``, ``p_value``,
        ``conf_low`` and ``conf_high`` on the requested scale, and the model
        records in ``attrs["models"]``.

    Raises
    ------
    SpecificationError
        No result, an object that is not a fitted result, an unknown keyword, a
        value outside its domain, as many ``names`` as results are not given, or a
        ``keep`` or ``drop`` name that is no term of any result.

    Notes
    -----
    A result that is not certified, and a recognized inference refusal from
    ``vcov=``, do not raise: their column prints the
    status (module docstring).
    """
    resolved = collect(options)
    fitted, names = _unpack_results(results, resolved)
    models = [model_of(result, resolved) for result in fitted]
    terms = _terms(models, resolved)
    coefficients = [row for term in terms for row in _term_rows(term, models, resolved)]
    sections = [
        tuple(coefficients),
        tuple(_model_rows(models, resolved)),
        tuple(_statistic_rows(models, resolved)),
    ]
    frame = _frame(models, names, terms)
    if resolved.scale == "effect":
        frame.attrs["interpretation"] = (
            "100 (exp(b) - 1) for each reported coefficient coordinate; "
            "not a standalone regressor change"
        )
    return Table(
        title=resolved.title,
        subtitle="",
        columns=(tables.EMPTY, *(tables.text(name) for name in names)),
        sections=tuple(section for section in sections if section),
        notes=_notes(models, resolved, names),
        align=tables.alignment("l", rest="c", count=len(models)),
        frame=frame,
    )
