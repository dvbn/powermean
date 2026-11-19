"""``summary_statistics``: one row of descriptive statistics per variable.

For each variable: the number of non-missing values, the number missing, the mean, the
standard deviation and the requested quantiles; for a variable whose values are all
nonnegative, also the share of zeros and the mean of the positive values, which say
whether a logarithm is defined on the whole sample, how many rows a positive-only
analysis sets aside, and how the positive values differ from the full sample.

The formulas, with and without weights, are those of
:mod:`powermean.describe.numeric`: unweighted quantiles interpolate linearly between
order statistics (the default of R's ``quantile`` and of NumPy, so that the
unweighted table agrees with the R package ``RCT``); weighted quantiles invert the
weighted empirical distribution function.  ``N`` and ``missing`` count rows, with or
without weights.
"""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from typing import TYPE_CHECKING, Unpack, cast

import numpy as np

from powermean.describe import common, frame, numeric
from powermean.describe.options import (
    SUMMARY_KEYWORDS,
    TableKeywords,
    TableOptions,
    collect,
)
from powermean.errors import SampleError
from powermean.results import tables
from powermean.results.tables import Cell, Row, Table

if TYPE_CHECKING:
    import pandas as pd


def quantile_name(prob: float) -> str:
    """``p50`` for the level 0.5; ``p2.5`` for 0.025."""
    return f"p{100.0 * prob:g}"


def _quantile_names(probs: tuple[float, ...]) -> tuple[str, ...]:
    """Keep short headings unless they would merge distinct probability levels."""
    names = [quantile_name(p) for p in probs]
    counts = Counter(names)
    for i, (p, name) in enumerate(zip(probs, names, strict=True)):
        if counts[name] > 1:
            # Shift the decimal exponent of the round-trip probability. Multiplying
            # by 100 in binary64 can itself merge adjacent levels before formatting.
            parts = Decimal(repr(p)).as_tuple()
            percent = Decimal((parts.sign, parts.digits, int(parts.exponent) + 2))
            if percent != Decimal(name[1:]):
                names[i] = f"p{percent:g}"
    return tuple(names)


def _statistics(
    values: frame.FloatArray,
    w: frame.FloatArray | None,
    rows: int,
    probs: tuple[float, ...],
    quantile_names: tuple[str, ...],
) -> dict[str, object]:
    """The statistics of one variable on its non-missing values."""
    mean, variance = numeric.mean_variance(values, w)
    record: dict[str, object] = {
        "n": int(values.size),
        "missing": rows - int(values.size),
        "mean": mean,
        "sd": math.sqrt(variance) if variance >= 0 else math.nan,
    }
    quantiles = numeric.quantiles(values, probs, w)
    record.update(
        {name: float(q) for name, q in zip(quantile_names, quantiles, strict=True)}
    )
    nonnegative = values.size > 0 and bool((values >= 0).all())
    share, positive = (
        numeric.share_and_positive_mean(values, w)
        if nonnegative
        else (math.nan, math.nan)
    )
    record.update(zeros=share, mean_positive=positive)
    return record


def _cells(record: dict[str, object], names: list[str], options: TableOptions) -> Row:
    """The printed row of one variable."""
    label = tables.label_of(str(record["variable"]), options.labels)
    cells: list[Cell] = [tables.text(label)]
    for name in names:
        value = common.real(record, name)
        if name in ("n", "missing"):
            cells.append(tables.integer(value))
        elif name == "zeros":
            cells.append(tables.percent(value))
        else:
            cells.append(tables.number(value, options.digits))
    return Row(tuple(cells))


def _all_records(
    variables: tuple[frame.Variable, ...],
    context: frame.Context,
    complete: frame.BoolArray,
    probs: tuple[float, ...],
) -> list[dict[str, object]]:
    """The records of all variables across all subgroups."""
    probs = tuple(dict.fromkeys(probs))
    quantile_names = _quantile_names(probs)
    records: list[dict[str, object]] = []
    for heading, positions in context.groups:
        rows = positions[complete[positions]]
        for variable in variables:
            observed = rows[~np.isnan(variable.values[rows])]
            weights = None if context.weights is None else context.weights[observed]
            record = _statistics(
                variable.values[observed],
                weights,
                int(rows.size),
                probs,
                quantile_names,
            )
            records.append({"by": heading, "variable": variable.name, **record})
    return records


def _sections(
    context: frame.Context,
    records: list[dict[str, object]],
    names: list[str],
    options: TableOptions,
) -> tuple[tuple[Row, ...], ...]:
    """The printed sections of the summary-statistics table."""
    sections: list[tuple[Row, ...]] = []
    width = len(records) // len(context.groups)
    for i, (heading, _) in enumerate(context.groups):
        body = [_cells(r, names, options) for r in records[i * width : (i + 1) * width]]
        sections.append(tuple([tables.group_row(heading), *body] if heading else body))
    return tuple(sections)


def summary_statistics(
    data: object,
    vars: object = None,  # noqa: A002 - public argument name
    **options: Unpack[TableKeywords],
) -> Table:
    """Descriptive statistics of the variables of a frame, one row per variable.

    Parameters
    ----------
    data : pandas.DataFrame
        The data, read and never modified.
    vars : str or sequence of str, optional
        The variables.  ``None`` takes every numeric column except the weight and
        subgroup columns.
    **options : TableKeywords
        ``probs`` (quantile levels; default ``(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9,
        0.95, 1)``), ``weights``, ``by``, ``complete``, ``expand``, ``sample``,
        ``labels`` and ``digits`` (default 2); see
        :mod:`powermean.describe.options`.

    Returns
    -------
    Table
        Printed columns ``N``, ``missing``, ``mean``, ``sd``, one per quantile, and
        ``zeros`` and ``mean | >0`` when at least one variable is nonnegative.
        ``to_frame()`` has the columns ``n``, ``missing``, ``mean``, ``sd``,
        ``p0`` ..., ``zeros`` and ``mean_positive`` (NaN for a variable with a
        negative value), indexed by ``variable``, after ``by`` where it is given.
        Quantile columns follow the requested order, once per distinct level. Their
        percentage labels gain precision when distinct levels share a short label.

    Raises
    ------
    SpecificationError
        An unknown keyword, a value outside its domain, an absent column, a named
        column that is not numeric without ``expand=True``.
    SampleError
        A negative weight.
    """
    resolved = collect(options, "summary_statistics", SUMMARY_KEYWORDS, digits=2)
    context = frame.context(data, resolved)
    complete = np.ones(len(context.frame), dtype=np.bool_)
    variables = frame.variables(
        context.frame, vars, resolved, support=complete if resolved.complete else None
    )
    if not context.groups:
        # context has already checked that the input is a DataFrame.
        raise SampleError("sample.empty", {"count": len(cast("pd.DataFrame", data))})
    records = _all_records(variables, context, complete, resolved.probs)
    names = [name for name in records[0] if name not in ("by", "variable")]
    if all(math.isnan(common.real(r, "zeros")) for r in records):
        names = [name for name in names if name not in ("zeros", "mean_positive")]
    printed = {"n": "N", "mean_positive": "mean | >0"}
    notes = list(context.notes)
    if resolved.complete:
        notes.append("Rows complete in all variables.")
    total = int((context.usable & complete).sum())
    return Table(
        title="Summary statistics",
        subtitle=f"N = {total:,d} rows",
        columns=(tables.EMPTY, *(tables.text(printed.get(n, n)) for n in names)),
        sections=_sections(context, records, names, resolved),
        notes=tuple(notes),
        align=tables.alignment("l", rest="r", count=len(names)),
        frame=common.assemble(records, ["variable"], {"rows": total}),
    )
