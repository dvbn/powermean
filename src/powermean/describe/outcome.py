"""``outcome_table``: outcomes by arm and unadjusted proportional differences.

For each outcome and each arm: the number of observations, the share of zeros, the
mean and the mean of the positive values.  Then, for each requested mean order
``omega`` and each arm ``a`` other than the control ``0``, the unadjusted proportional
difference of the power means,

    theta_omega = (M_omega(a) / M_omega(0))^(1/omega) - 1,
    theta_0     = exp(mean of log Y in arm a - mean of log Y in arm 0) - 1,

where ``M_omega(d)`` is the mean of ``Y^omega`` in arm ``d``,

with its confidence interval.  No solver is involved: a model saturated in the arms
fits each arm's power moment exactly whatever the working variance, so
``theta_omega`` equals the effect of ``pm.fit("y ~ C(treat)", omega=omega)``.

Inference.  Row ``i`` of arm ``d`` contributes ``(Z_i - M(d)) / (omega n_d M(d))`` to
the log ratio (with the sign of its arm), and contributions are summed within
clusters when ``cluster`` is given.  The variance is the sum of squared
contributions times the factor of the ``ssc`` preset, counted on all the rows and
arms that the order uses, as for the saturated fit; the interval transforms the
endpoints of the interval of the log ratio, ``exp(l -/+ q se) - 1``.

Orders at or below zero need positive outcomes.  When an outcome has zeros they
require ``positive_only=True``, are computed on the positive outcomes, and are
marked ``Y > 0``; the shares of zeros by arm stand in the same block of rows.  A
different share of zeros across arms means that the positive-only comparison does
not compare like with like.

The symbols ``theta_N``, ``theta_L``, ``theta_H`` and ``theta_omega`` name the
unadjusted proportional difference in means, geometric means, harmonic means and power
means of order ``omega``.  Whether a difference is a causal effect depends on the
design.
"""

from __future__ import annotations

import math
from typing import Unpack

import numpy as np

from powermean.describe import common, frame, numeric
from powermean.describe.frame import Arms, Context, FloatArray, Variable
from powermean.describe.options import (
    OUTCOME_KEYWORDS,
    TableKeywords,
    TableOptions,
    collect,
    refuse,
)
from powermean.errors import SampleError
from powermean.results import tables
from powermean.results.tables import Cell, Row, Table

_ARM = ("n", "zeros", "mean", "mean_positive")
_DIFFERENCE = ("estimate", "conf_low", "conf_high", "log_ratio", "std_error")


def order_text(omega: float) -> str:
    """``omega`` as printed: an integer without a fraction."""
    return str(int(omega)) if omega == int(omega) else repr(omega)


def target_symbol(omega: float) -> str:
    """The symbol of the proportional difference at order ``omega``."""
    named = {1.0: "theta_N", 0.0: "theta_L", -1.0: "theta_H"}
    return named.get(omega, f"theta_{order_text(omega)}")


def target_label(omega: float) -> str:
    """The printed label of the unadjusted difference at order ``omega``."""
    means = {1.0: "means", 0.0: "geometric means", -1.0: "harmonic means"}
    kind = means.get(omega, f"power means of order {order_text(omega)}")
    return f"Proportional difference in {kind}, unadjusted ({target_symbol(omega)})"


def _transformed(log_ratio: float) -> float:
    """Refuse range loss rather than confuse it with a structural zero."""
    if math.isnan(log_ratio):
        return math.nan
    if log_ratio < math.log(np.finfo(float).tiny) or log_ratio > math.log(
        np.finfo(float).max
    ):
        raise refuse("outcomes", "numerical range", "a representable power-mean ratio")
    return math.expm1(log_ratio)


def _contrast(
    moment: numeric.LogMoment, control: numeric.LogMoment, omega: float
) -> float:
    """Difference centers before combining the scaled moment corrections."""
    if omega == 0:
        return moment.center - control.center
    return (moment.center - control.center) + (
        moment.log_relative - control.log_relative
    ) / omega


def _inference(
    scores: dict[int, float],
    control: dict[int, float],
    factor: float,
    reference: numeric.Reference,
    log_ratio: float,
    options: TableOptions,
) -> dict[str, float]:
    """Merge sparse cluster totals (or independent row scores) once per contrast."""
    result = {
        "estimate": _transformed(log_ratio),
        "log_ratio": log_ratio,
        "std_error": math.nan,
        "conf_low": math.nan,
        "conf_high": math.nan,
    }
    if not math.isfinite(factor) or not math.isfinite(log_ratio):
        return result
    keys = scores.keys() | control.keys()
    differences = (scores.get(k, 0.0) - control.get(k, 0.0) for k in keys)
    # The standard error can be representable even when its square is not.
    se = math.sqrt(factor) * math.hypot(*differences)
    if not math.isfinite(se):
        raise refuse("outcomes", "numerical range", "a representable standard error")
    result["std_error"] = se
    if se > 0:
        half = numeric.critical_value(options.level, reference) * se
        result.update(
            conf_low=_transformed(log_ratio - half),
            conf_high=_transformed(log_ratio + half),
        )
    return result


def _order_differences(
    y: FloatArray,
    positions: frame.IntArray,
    omega: float,
    context: Context,
    options: TableOptions,
) -> dict[str, float]:
    """Order-local support, cached arm moments and sparse cluster score sums."""
    arms = context.arms
    assert arms is not None
    codes = arms.codes[positions]
    observed = np.unique(codes)
    k = len(observed)
    n = positions.size
    clusters = context.clusters
    g = len(np.unique(clusters[positions])) if clusters is not None else 0
    factor, reference = numeric.convention(options.ssc, numeric.Counts(n, g, k, k))
    if clusters is not None and g < 2:
        factor = math.nan
    if reference.kind == "t" and reference.df <= 0:
        factor = math.nan
    partitions = frame._group_rows(codes, np.ones(n, dtype=np.bool_), len(arms.levels))
    moments: list[numeric.LogMoment] = []
    totals: list[dict[int, float]] = []
    for local in partitions:
        chosen = positions[local]
        weights = None if context.weights is None else context.weights[chosen]
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="raise"
            ):
                moment = numeric.log_power_moment(y[chosen], omega, weights)
        except (FloatingPointError, OverflowError, ValueError) as error:
            raise refuse(
                "outcomes", "numerical range", "representable power moments and scores"
            ) from error
        moments.append(moment)
        reduced: dict[int, float] = {}
        keys = chosen if clusters is None else clusters[chosen]
        for key, value in zip(keys, moment.influence, strict=True):
            reduced[int(key)] = reduced.get(int(key), 0.0) + float(value)
        totals.append(reduced)
    result: dict[str, float] = {}
    for d in range(1, len(arms.levels)):
        missing = dict.fromkeys(_DIFFERENCE, math.nan)
        if partitions[0].size and partitions[d].size:
            control, moment = moments[0], moments[d]
            if math.isfinite(control.center) and math.isfinite(moment.center):
                log_ratio = _contrast(moment, control, omega)
                missing = _inference(
                    totals[d], totals[0], factor, reference, log_ratio, options
                )
            elif math.isfinite(control.center) and omega > 0:
                missing["estimate"] = -1.0
        result.update(
            {f"{key}[{arms.labels[d]}]": value for key, value in missing.items()}
        )
    return result


def _arm_statistics(
    y: FloatArray, positions: frame.IntArray, context: Context
) -> dict[str, float]:
    """Pre-order descriptive support is independent of positive-only moments."""
    arms = context.arms
    assert arms is not None
    partitions = frame._group_rows(
        arms.codes[positions], np.ones(positions.size, dtype=np.bool_), len(arms.levels)
    )
    record: dict[str, float] = {}
    for d, local in enumerate(partitions):
        chosen = positions[local]
        weights = None if context.weights is None else context.weights[chosen]
        share, positive = numeric.share_and_positive_mean(y[chosen], weights)
        level = arms.labels[d]
        record.update(
            {
                f"n[{level}]": float(chosen.size),
                f"zeros[{level}]": share,
                f"mean[{level}]": numeric.mean(y[chosen], weights),
                f"mean_positive[{level}]": positive,
            }
        )
    return record


def _records(
    name: str,
    y: FloatArray,
    positions: frame.IntArray,
    context: Context,
    options: TableOptions,
) -> list[dict[str, object]]:
    """One record per requested order, with shared local descriptive support."""
    rows = positions[~np.isnan(y[positions])]
    negative = int((y[rows] < 0).sum())
    if negative:
        raise SampleError("sample.negative_outcome", {"count": negative})
    zeros = int((y[rows] == 0).sum())
    described = _arm_statistics(y, rows, context)
    records: list[dict[str, object]] = []
    for omega in options.orders:
        positive = omega <= 0 and zeros > 0
        if positive and not options.positive_only:
            raise SampleError(
                "sample.zero_outcome_at_order", {"count": zeros, "omega": omega}
            )
        used = rows[y[rows] > 0] if positive else rows
        differences = _order_differences(y, used, omega, context, options)
        records.append(
            {
                "outcome": name,
                "order": omega,
                "rows": "positive" if positive else "all",
                **described,
                **differences,
            }
        )
    return records


def _columns(arms: Arms, level: float, *, counts: bool) -> list[tuple[str, str]]:
    """The printed columns as ``(frame column, heading)`` pairs."""
    headings = {
        "n": "N",
        "zeros": "zeros",
        "mean": "mean",
        "mean_positive": "mean | >0",
    }
    columns: list[tuple[str, str]] = []
    for name in _ARM:
        if name == "n" and not counts:
            continue
        columns += [
            (f"{name}[{value}]", headings[name] + common.arm_suffix(arms, d))
            for d, value in enumerate(arms.labels)
        ]
    for d in range(1, len(arms.levels)):
        suffix = common.treated_suffix(arms, d)
        columns.append(
            (f"estimate[{arms.labels[d]}]", "difference, unadjusted" + suffix)
        )
        columns.append(
            (f"conf_low[{arms.labels[d]}]", f"[{100 * level:g}% conf. int.]" + suffix)
        )
    return columns


def _cell(name: str, record: dict[str, object], digits: int, *, first: bool) -> Cell:
    """One printed cell; the arm statistics print on the first row of an outcome."""
    statistic, _, rest = name.partition("[")
    value = common.real(record, name)
    if statistic == "estimate":
        return tables.percent(value)
    if statistic == "conf_low":
        high = common.real(record, f"conf_high[{rest}")
        return tables.bracketed(tables.percent(value), tables.percent(high))
    if not first:
        return tables.EMPTY
    if statistic == "n":
        return tables.integer(value)
    if statistic == "zeros":
        return tables.percent(value)
    return tables.number(value, digits)


def _body(
    records: list[dict[str, object]],
    columns: list[tuple[str, str]],
    options: TableOptions,
) -> list[Row]:
    """The printed rows of one subgroup: one per outcome and order."""
    body: list[Row] = []
    previous: object = None
    for record in records:
        first = record["outcome"] != previous
        previous = record["outcome"]
        label = tables.label_of(str(record["outcome"]), options.labels) if first else ""
        target = target_symbol(common.real(record, "order"))
        if record["rows"] == "positive":
            target += " (Y > 0)"
        cells = [
            _cell(name, record, options.digits, first=first) for name, _ in columns
        ]
        body.append(Row((tables.text(label), tables.text(target), *cells)))
    return body


def _notes(context: Context, options: TableOptions, positive: bool) -> list[str]:
    """The notes of the table: targets, inference, the positive-only warning."""
    notes = [target_label(omega) + "." for omega in options.orders]
    notes.append(
        "Intervals transform the endpoints of the interval of the log ratio. "
        + common.covariance_note(options)
    )
    if positive:
        notes.append(
            "Rows marked Y > 0 use positive outcomes only; a different share of "
            "zeros across arms means that this comparison does not compare like "
            "with like."
        )
    return [*notes, *context.notes]


def _all_records(
    found: tuple[Variable, ...], context: Context, options: TableOptions
) -> list[dict[str, object]]:
    """All outcome-by-order records across subgroups."""
    records: list[dict[str, object]] = []
    for heading, positions in context.groups:
        for outcome in found:
            for record in _records(
                outcome.name, outcome.values, positions, context, options
            ):
                records.append({"by": heading, **record})
    return records


def _varying_counts(
    arms: Arms, context: Context, records: list[dict[str, object]]
) -> bool:
    """Whether observation counts vary across outcomes or subgroups."""
    full = common.arm_counts(arms, context.usable)
    return len(context.groups) > 1 or any(
        record[f"n[{level}]"] != full[d]
        for record in records
        for d, level in enumerate(arms.labels)
    )


def outcome_table(
    data: object,
    outcomes: object,
    treatment: object,
    **options: Unpack[TableKeywords],
) -> Table:
    """Outcomes by treatment arm and unadjusted proportional differences.

    Parameters
    ----------
    data : pandas.DataFrame
        The data, read and never modified.
    outcomes : str or sequence of str
        The outcome columns, nonnegative.
    treatment : str
        The treatment column, with two or more values.
    **options : TableKeywords
        ``orders`` (mean orders; default ``(1,)``), ``control``, ``bins``,
        ``cluster``, ``weights``, ``ssc``, ``positive_only``, ``level`` (default
        0.95), ``by``, ``sample``, ``labels`` and ``digits`` (default 2); see
        :mod:`powermean.describe.options`.

    Returns
    -------
    Table
        One row per outcome and order (module docstring).  ``to_frame()`` is
        indexed by ``outcome`` and ``order`` with the column ``rows`` (``"all"`` or
        ``"positive"``), for each arm ``d`` the columns ``n[d]``, ``zeros[d]``,
        ``mean[d]`` and ``mean_positive[d]`` and, for each arm ``a`` other than the
        control, ``estimate[a]`` (the proportional difference, a proportion),
        ``conf_low[a]``, ``conf_high[a]``, ``log_ratio[a]`` and ``std_error[a]``
        (of the log ratio).

    Raises
    ------
    SpecificationError
        An unknown keyword, a value outside its domain, an absent column, an
        outcome that is not numeric.
    SampleError
        A negative outcome (``sample.negative_outcome``); zeros at an order at or
        below zero without ``positive_only=True``
        (``sample.zero_outcome_at_order``); a negative weight; fewer than two arms.
    """
    if outcomes is None:
        raise refuse("outcomes", outcomes, "a column name or a sequence of them")
    resolved = collect(options, "outcome_table", OUTCOME_KEYWORDS, digits=2)
    context = frame.context(data, resolved, treatment)
    arms = context.arms
    assert arms is not None
    found = frame.variables(context.frame, outcomes, resolved, argument="outcomes")
    records = _all_records(found, context, resolved)
    varying = _varying_counts(arms, context, records)
    columns = _columns(arms, resolved.level, counts=varying)
    sections = []
    for heading, _ in context.groups:
        body = _body([r for r in records if r["by"] == heading], columns, resolved)
        sections.append(tuple([tables.group_row(heading), *body] if heading else body))
    positive = any(record["rows"] == "positive" for record in records)
    headings = [tables.text("target"), *(tables.text(h) for _, h in columns)]
    attrs = {
        "treatment": arms.name,
        "arms": list(arms.levels),
        "control": arms.levels[0],
        "level": resolved.level,
        "labels": {order: target_label(order) for order in resolved.orders},
    }
    return Table(
        title="Outcome table",
        subtitle=common.arms_subtitle(arms, context.usable),
        columns=(tables.EMPTY, *headings),
        sections=tuple(sections),
        notes=tuple(_notes(context, resolved, positive)),
        align=tables.alignment("l", "l", rest="r", count=len(columns)),
        frame=common.assemble(records, ["outcome", "order"], attrs),
    )
