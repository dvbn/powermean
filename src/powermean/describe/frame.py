"""The rows, variables, arms, weights and subgroups of a table call, read from a frame.

This module is the only one of the package ``describe`` that reads a
:class:`pandas.DataFrame`; the tables receive arrays.

- Data: a DataFrame, read and never modified.  ``sample=result`` restricts the table
  to the estimation sample of a fit of the same frame, through the public
  ``result.sample`` (``positions``, ``nobs``, ``input_labels``), so that descriptive
  and regression tables describe the same rows.
- Variables: ``vars=None`` takes every numeric column except those that the call
  uses otherwise (treatment, weights, cluster, strata, subgroups).  A column that is
  not numeric is skipped under ``vars=None``, refused when named, and with
  ``expand=True`` replaced by one indicator per level, named ``"column = level"``.
- Treatment: two or more values; the control arm is ``control``, by default the
  first value in sorted order.  ``bins=k`` cuts the treatment into ``k`` quantile
  groups ``Q1`` to ``Qk`` with right-closed intervals; tied quantiles leave fewer.
- Missing values: a row whose treatment, weight, cluster or stratum is missing, or
  whose weight is zero, is outside the table, and the notes count such rows.  Each
  variable then uses its own non-missing rows.
- Weights: nonnegative; a negative weight raises ``sample.negative_weight``.
- Subgroups: ``by`` splits the rows by the levels of one column.
- Level names: treatment arms, subgroups and expanded indicators retain distinct
  labels when different levels print alike. Such levels include their type and
  a numbered suffix if needed. Treatment metadata retain the original values.
  Distinct selected column keys also receive distinct names, with literal string
  keys preserved.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from powermean.describe.options import TableOptions, refuse
from powermean.errors import SampleError, SpecificationError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Variable:
    """One numeric variable: its name and its values, NaN where missing.

    ``reference`` marks the first indicator of an expanded categorical column,
    which a regression with an intercept omits.
    """

    name: str
    values: FloatArray
    reference: bool = False


@dataclass(frozen=True)
class Arms:
    """The arms of a treatment: codes with the control at 0, and -1 where missing."""

    name: str
    codes: IntArray
    levels: tuple[object, ...]
    labels: tuple[str, ...]

    def text(self, position: int) -> str:
        """``"treatment = level"`` for the arm at ``position``."""
        return f"{self.name} = {self.labels[position]}"


@dataclass(frozen=True)
class Context:
    """What a table call reads from the frame, on the rows of the table.

    ``usable`` marks the rows inside the table (module docstring); ``groups`` holds
    one ``(heading, positions)`` pair per subgroup, a single pair with an empty heading
    without ``by``.
    """

    frame: pd.DataFrame
    usable: BoolArray
    weights: FloatArray | None
    clusters: IntArray | None
    strata: IntArray | None
    arms: Arms | None
    groups: tuple[tuple[str, IntArray], ...]
    notes: tuple[str, ...]


def column(frame: pd.DataFrame, name: object, argument: str) -> pd.Series:
    """The column ``name`` of the frame; refused where the frame has none."""
    if name not in frame.columns:
        raise SpecificationError(
            "specification.table_column", {"argument": argument, "name": name}
        )
    return frame[name]


def _numeric(series: pd.Series) -> bool:
    """Whether a column holds real numbers or booleans."""
    kind = series.dtype
    return bool(
        pd.api.types.is_bool_dtype(kind)
        or (
            pd.api.types.is_numeric_dtype(kind)
            and not pd.api.types.is_complex_dtype(kind)
        )
    )


def _floats(series: pd.Series) -> FloatArray:
    """A numeric column as floats, NaN where missing."""
    return np.asarray(
        series.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64
    )


def _codes(series: pd.Series) -> tuple[IntArray, list[object]]:
    """Integer codes of the levels in sorted order, -1 where missing."""
    codes, levels = pd.factorize(series, sort=True)
    return np.asarray(codes, dtype=np.intp), list(levels)


def _distinct_labels(
    labels: list[str], qualifier: Callable[[int], str | None]
) -> tuple[str, ...]:
    """Keep unique spellings and literal column names; qualify colliding labels.

    Reserve the complete original vocabulary before assigning names: a literal
    such as "1 (int)" owns that spelling even when integer 1 needs a qualifier.
    A None qualifier identifies a selected input column whose name is preserved.
    """
    reserved = set(labels)
    if len(reserved) == len(labels):
        return tuple(labels)
    counts = Counter(labels)
    next_suffix: dict[str, int] = {}
    for i, label in enumerate(labels):
        if counts[label] == 1:
            continue
        detail = qualifier(i)
        if detail is None:
            continue
        base = f"{label} ({detail})"
        candidate = base
        suffix = next_suffix.get(base, 2)
        while candidate in reserved:
            candidate = f"{base} #{suffix}"
            suffix += 1
        next_suffix[base] = suffix
        labels[i] = candidate
        reserved.add(candidate)
    return tuple(labels)


def _level_labels(levels: Sequence[object]) -> tuple[str, ...]:
    """Display an already factorized set of levels without changing its order."""
    return _distinct_labels(
        [str(v) for v in levels], lambda i: type(levels[i]).__name__
    )


def classification(frame: pd.DataFrame, spec: str, argument: str) -> IntArray:
    """The codes of one classification, or of the interaction ``"a^b"``."""
    combined: IntArray | None = None
    for name in (part.strip() for part in spec.split("^")):
        codes, levels = _codes(column(frame, name, argument))
        if combined is None:
            combined = codes
            continue
        joint = np.where(
            (combined < 0) | (codes < 0), -1, combined * len(levels) + codes
        )
        present = joint >= 0
        combined = np.full(joint.size, -1, dtype=np.intp)
        combined[present] = np.unique(joint[present], return_inverse=True)[1]
    assert combined is not None
    return combined


def _weights(frame: pd.DataFrame, name: str) -> FloatArray:
    """The weights as floats; a negative weight is refused."""
    series = column(frame, name, "weights")
    if not _numeric(series):
        raise SpecificationError("specification.table_not_numeric", {"name": name})
    values = _floats(series)
    negative = int((values < 0).sum())
    if negative:
        raise SampleError("sample.negative_weight", {"count": negative})
    return values


def _quantile_groups(values: FloatArray, bins: int) -> tuple[IntArray, list[object]]:
    """Quantile groups ``Q1..Qk`` of a continuous treatment, -1 where missing."""
    present = ~np.isnan(values)
    edges = np.quantile(values[present], np.linspace(0.0, 1.0, bins + 1))
    raw = np.searchsorted(edges[1:-1], values[present], side="left")
    kept, inverse = np.unique(raw, return_inverse=True)
    codes = np.full(values.size, -1, dtype=np.intp)
    codes[present] = inverse
    return codes, [f"Q{int(g) + 1}" for g in kept]


def arms_of(frame: pd.DataFrame, treatment: object, options: TableOptions) -> Arms:
    """The arms of the treatment column, the control first (module docstring)."""
    series = column(frame, treatment, "treatment")
    if options.bins is not None:
        if not _numeric(series):
            raise SpecificationError(
                "specification.table_not_numeric", {"name": treatment}
            )
        codes, levels = _quantile_groups(_floats(series), options.bins)
    else:
        codes, levels = _codes(series)
    position = 0
    if options.control is not None:
        matches = [i for i, level in enumerate(levels) if level == options.control]
        if not matches:
            raise refuse(
                "control",
                options.control,
                "a value of the treatment column: "
                + ", ".join(repr(level) for level in levels),
            )
        position = matches[0]
    order = [position, *(i for i in range(len(levels)) if i != position)]
    rank = np.empty(len(levels) + 1, dtype=np.intp)
    rank[np.array(order, dtype=np.intp)] = np.arange(len(levels))
    rank[-1] = -1
    labels = _level_labels(levels)
    return Arms(
        str(treatment),
        rank[codes],
        tuple(levels[i] for i in order),
        tuple(labels[i] for i in order),
    )


def _restricted(data: pd.DataFrame, sample: object) -> tuple[pd.DataFrame, list[str]]:
    """The rows of the estimation sample of a fit, where ``sample`` is given."""
    if sample is None:
        return data, []
    view = getattr(sample, "sample", None)
    positions = getattr(view, "positions", None)
    nobs = getattr(view, "nobs", None)
    if positions is None or not isinstance(nobs, int | np.integer):
        raise refuse(
            "sample", type(sample).__name__, "a fitted result with its sample view"
        )
    labels = getattr(view, "input_labels", None)
    if nobs != len(data):
        detail = f"the fit read {nobs} rows and the data have {len(data)}"
        raise SpecificationError("specification.table_sample", {"detail": detail})
    if not isinstance(labels, pd.Index) or not data.index.equals(labels):
        detail = "the row labels differ from those of the frame that was fitted"
        raise SpecificationError("specification.table_sample", {"detail": detail})
    if (
        not isinstance(positions, np.ndarray)
        or positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or np.any(positions < 0)
        or np.any(positions >= nobs)
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise SpecificationError(
            "specification.table_sample",
            {
                "detail": "sample positions must be increasing one-dimensional "
                "integers in range"
            },
        )
    kept = positions.astype(np.intp, copy=False)
    note = f"Rows: the estimation sample of the fit ({kept.size:,d} of {nobs:,d})."
    return data.iloc[kept], [note]


def _group_rows(codes: IntArray, usable: BoolArray, count: int) -> tuple[IntArray, ...]:
    """Stable compact slices from codes in O(N + groups) work and storage."""
    rows = np.flatnonzero(usable)
    sizes = np.bincount(codes[rows], minlength=count)
    offsets = np.r_[0, np.cumsum(sizes)]
    cursor = offsets[:-1].copy()
    grouped = np.empty(rows.size, dtype=np.intp)
    for row in rows:
        code = codes[row]
        grouped[cursor[code]] = row
        cursor[code] += 1
    return tuple(grouped[offsets[i] : offsets[i + 1]] for i in range(count))


def group_mask(context: Context, positions: IntArray) -> BoolArray:
    """One temporary mask for consumers that operate on full design arrays."""
    mask = np.zeros(len(context.frame), dtype=np.bool_)
    mask[positions] = True
    return mask


def context(data: object, options: TableOptions, treatment: object = None) -> Context:
    """Read the design columns of a table call (module docstring).

    Raises
    ------
    SpecificationError
        ``data`` is not a DataFrame, a named column is absent, the weights are not
        numeric, ``control`` is not a value of the treatment, or ``sample`` does not
        match the data.
    SampleError
        A negative weight; a treatment with fewer than two values on the usable
        rows.
    """
    if not isinstance(data, pd.DataFrame):
        raise refuse(
            "data",
            type(data).__name__,
            "a pandas DataFrame; pass sample=result to use the estimation sample "
            "of a fit",
        )
    frame, notes = _restricted(data, options.sample)
    usable = np.ones(len(frame), dtype=np.bool_)
    weights: FloatArray | None = None
    if options.weights is not None:
        weights = _weights(frame, options.weights)
        usable &= weights > 0
        notes.append(f"Weights: {options.weights!r} (analytic or probability).")
    clusters = strata = None
    if options.cluster is not None:
        clusters = classification(frame, options.cluster, "cluster")
        usable &= clusters >= 0
    if options.fe is not None:
        strata = classification(frame, options.fe, "fe")
        usable &= strata >= 0
    arms: Arms | None = None
    if treatment is not None:
        arms = arms_of(frame, treatment, options)
        usable &= arms.codes >= 0
    groups: tuple[tuple[str, IntArray], ...] = (("", np.flatnonzero(usable)),)
    if options.by is not None:
        codes, levels = _codes(column(frame, options.by, "by"))
        usable &= codes >= 0
        grouped = _group_rows(codes, usable, len(levels))
        groups = tuple(
            (f"{options.by} = {level}", grouped[i])
            for i, level in enumerate(_level_labels(levels))
        )
    excluded = len(frame) - int(usable.sum())
    if excluded:
        notes.append(
            f"{excluded:,d} row(s) excluded: missing treatment, weight, cluster, "
            "stratum or subgroup, or zero weight."
        )
    if arms is not None and np.unique(arms.codes[usable]).size < 2:
        raise SampleError(
            "sample.table_arms",
            {"name": arms.name, "count": int(np.unique(arms.codes[usable]).size)},
        )
    return Context(frame, usable, weights, clusters, strata, arms, groups, tuple(notes))


def _expanded(name: object, series: pd.Series) -> list[Variable]:
    """One indicator per level of a categorical column, NaN where it is missing."""
    codes, levels = _codes(series)
    indicators: list[Variable] = []
    for position, level in enumerate(_level_labels(levels)):
        values = np.where(codes < 0, np.nan, (codes == position).astype(np.float64))
        indicators.append(Variable(f"{name} = {level}", values, position == 0))
    return indicators


def _candidates(
    frame: pd.DataFrame,
    names: object,
    options: TableOptions,
    exclude: Sequence[object],
    argument: str,
) -> list[object]:
    """The candidate column names for ``variables``."""
    used = {options.weights, options.cluster, options.by, *exclude}
    if options.fe is not None:
        used.update(part.strip() for part in options.fe.split("^"))
    if names is None:
        return [c for c in frame.columns if c not in used]
    if isinstance(names, str):
        return [names]
    if isinstance(names, Sequence) and len(names) > 0:
        return list(names)
    raise refuse(argument, names, "a column name or a nonempty sequence of them")


def _named_variables(
    selected: list[object], columns: list[Variable | pd.Series]
) -> tuple[Variable, ...]:
    """Resolve mixed source keys and expanded terms across the selected columns."""
    # Name each native source once, then replay repeated selections in order.
    unique = dict(zip(selected, columns, strict=True))
    keys = list(unique)
    labels = _distinct_labels(
        [str(key) for key in keys],
        lambda i: None if isinstance(keys[i], str) else type(keys[i]).__name__,
    )
    found: list[Variable] = []
    indicators: set[int] = set()
    spans: dict[object, slice] = {}
    for (key, value), label in zip(unique.items(), labels, strict=True):
        start = len(found)
        if isinstance(value, Variable):
            found.append(value if value.name == label else replace(value, name=label))
        else:
            found.extend(_expanded(label, value))
            indicators.update(range(start, len(found)))
        spans[key] = slice(start, len(found))
    if indicators:
        terms = _distinct_labels(
            [v.name for v in found], lambda i: "indicator" if i in indicators else None
        )
        for i in indicators:
            if terms[i] != found[i].name:
                found[i] = replace(found[i], name=terms[i])
    if len(selected) == len(unique):
        return tuple(found)
    return tuple(v for key in selected for v in found[spans[key]])


def variables(
    frame: pd.DataFrame,
    names: object,
    options: TableOptions,
    *,
    exclude: Sequence[object] = (),
    argument: str = "vars",
    support: BoolArray | None = None,
) -> tuple[Variable, ...]:
    """The variables of a table (module docstring), in the order given.

    Raises
    ------
    SpecificationError
        ``names`` is not a column name or a sequence of them; a named column is
        absent or, without ``expand=True``, not numeric; no variable remains.
    """
    candidates = _candidates(frame, names, options, exclude, argument)
    selected: list[object] = []
    columns: list[Variable | pd.Series] = []
    first_positions: dict[object, int] = {}
    plain = True
    for name in candidates:
        series = column(frame, name, argument)
        if names is not None:
            # A requested native key may recur. Replay its first prepared input
            # without converting the same column or updating complete rows twice.
            first = first_positions.setdefault(series.name, len(columns))
            if first < len(columns):
                selected.append(series.name)
                columns.append(columns[first])
                continue
        numeric = _numeric(series)
        if not numeric and not (options.expand and argument == "vars"):
            if names is not None:
                raise SpecificationError(
                    "specification.table_not_numeric", {"name": name}
                )
            continue
        if support is not None:
            support &= series.notna().to_numpy(dtype=np.bool_)
        selected.append(series.name)
        if numeric:
            columns.append(Variable(str(series.name), _floats(series)))
        else:
            columns.append(series)
        plain = plain and numeric and isinstance(series.name, str)
    if plain:
        # These are all numeric variables with literal names. Repeated selections
        # already occupy their requested positions and need no name allocation.
        result = tuple(cast("list[Variable]", columns))
    else:
        result = _named_variables(selected, columns)
    if not result:
        raise refuse(argument, names, "at least one numeric column")
    return result
