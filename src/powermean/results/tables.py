"""The table object shared by ``etable`` and the four preliminary tables.

A table object with ``to_frame()``, ``to_latex()``, ``to_markdown()`` and
``to_html()``.  This module is the one formatting layer of the five table functions:
every number that a table prints is formatted by a function below, and every export is
written from the same grid of cells by :mod:`powermean.results.table_writers`.

A :class:`Table` is presentation only.  It holds

- a title and a subtitle (the line that names the sample or the arms);
- one header row and the body, a sequence of sections that the writers separate by a
  rule; a row is a data row or a group row, whose single cell spans the table (the
  heading of a ``by=`` subgroup);
- the notes printed under the table;
- the numbers behind the cells, a :class:`pandas.DataFrame` that ``to_frame()``
  hands out as an owned copy, unrounded.

A :class:`Cell` is the printed text of one cell.  Its ``latex`` form is given only
where it differs from the escaped text: a minus sign set in mathematics, and the
significance stars set as a superscript.

Rounding rules (``digits=`` "sets rounding"): a real number is printed with
``digits`` decimals and a comma between thousands; a count without decimals; a share
in percent with one decimal; a p-value with three decimals, ``<0.001`` below that; a
value that is not finite prints as an empty cell.  The frame of ``to_frame()`` is
never rounded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from powermean.results import table_writers

Align = Literal["l", "c", "r"]
RowKind = Literal["data", "group"]

# Decimals of a p-value and of a share in percent, fixed by the rounding rules above.
P_VALUE_DECIMALS = 3
PERCENT_DECIMALS = 1


@dataclass(frozen=True)
class Cell:
    """The printed text of one cell, and its LaTeX form where the two differ."""

    text: str
    latex: str | None = None


EMPTY = Cell("")


@dataclass(frozen=True)
class Row:
    """One row of the body: data cells, or one group heading that spans the table."""

    cells: tuple[Cell, ...]
    kind: RowKind = "data"


def group_row(heading: str) -> Row:
    """The heading of a group of rows, for example one level of ``by=``."""
    return Row((Cell(heading),), "group")


@dataclass(frozen=True)
class Table:
    """A formatted table with its exports (module docstring).

    Parameters
    ----------
    title, subtitle : str
        The name of the table and the line that describes its sample or arms;
        either may be empty.
    columns : tuple of Cell
        The header row; the first cell heads the row labels.
    sections : tuple of tuple of Row
        The body.  Every data row has as many cells as ``columns``.
    notes : tuple of str
        The notes under the table, one sentence or more each.
    align : tuple of {"l", "c", "r"}
        The alignment of each column.
    frame : pandas.DataFrame
        The unrounded numbers; read it through :meth:`to_frame`.
    """

    title: str
    subtitle: str
    columns: tuple[Cell, ...]
    sections: tuple[tuple[Row, ...], ...]
    notes: tuple[str, ...]
    align: tuple[Align, ...]
    frame: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check that the grid is rectangular; an internal error otherwise."""
        object.__setattr__(self, "frame", deepcopy(self.frame))
        width = len(self.columns)
        if len(self.align) != width:
            raise ValueError("one alignment per column is required")
        for section in self.sections:
            for row in section:
                expected = 1 if row.kind == "group" else width
                if len(row.cells) != expected:
                    raise ValueError("a row does not match the header of its table")

    def to_frame(self, *, formatted: bool = False) -> pd.DataFrame:
        """The table as a :class:`pandas.DataFrame`, an owned copy.

        Parameters
        ----------
        formatted : bool, default False
            ``False`` returns the unrounded numbers, with the metadata of the table
            in ``attrs``.  ``True`` returns the printed cells as strings, one row
            per data row and one column per printed column; a group heading fills
            the column ``group``.
        """
        if not formatted:
            return deepcopy(self.frame)
        return table_writers.to_strings(self)

    def to_text(self) -> str:
        """The table as fixed-width text."""
        return table_writers.to_text(self)

    def to_markdown(self) -> str:
        """The table as a GitHub-flavored Markdown pipe table, with its notes."""
        return table_writers.to_markdown(self)

    def to_latex(
        self, *, booktabs: bool = True, floating: bool = True, label: str | None = None
    ) -> str:
        r"""The table as LaTeX.

        Parameters
        ----------
        booktabs : bool, default True
            Rules of the ``booktabs`` package (``\toprule``, ``\midrule``,
            ``\bottomrule``), which the document must load; ``False`` writes
            ``\hline``, which needs no package.
        floating : bool, default True
            Wrap the ``tabular`` in a ``table`` environment with the title as its
            caption; ``False`` writes the ``tabular`` and the notes only.
        label : str, optional
            A ``\label`` for the floating table. Letters, digits, underscores,
            colons, hyphens and periods are kept; other keys are encoded as
            ``powermean-`` followed by their UTF-8 hexadecimal bytes.
        """
        return table_writers.to_latex(
            self, booktabs=booktabs, floating=floating, label=label
        )

    def to_html(self) -> str:
        """The table as an HTML ``table`` element, without a style sheet."""
        return table_writers.to_html(self)

    def __str__(self) -> str:
        """The fixed-width text of :meth:`to_text`."""
        return self.to_text()

    def _repr_html_(self) -> str:
        """The HTML of :meth:`to_html`, for a notebook."""
        return self.to_html()


# ------------------------------------------------------------------ formatting


def _math_minus(text: str) -> str | None:
    """The LaTeX form of a number: its minus signs in mathematics, else ``None``."""
    return text.replace("-", "$-$") if "-" in text else None


def text(value: object) -> Cell:
    """A cell of plain text."""
    return Cell(str(value))


def number(value: float, digits: int) -> Cell:
    """A real number with ``digits`` decimals and a comma between thousands."""
    if not math.isfinite(value):
        return EMPTY
    printed = f"{value:,.{digits}f}"
    return Cell(printed, _math_minus(printed))


def integer(value: float) -> Cell:
    """A count, without decimals and with a comma between thousands."""
    if not math.isfinite(value):
        return EMPTY
    return Cell(f"{round(value):,d}")


def percent(value: float, digits: int = PERCENT_DECIMALS) -> Cell:
    """A share or a proportional difference ``value`` printed as ``100 value %``."""
    if not math.isfinite(value):
        return EMPTY
    printed = f"{100.0 * value:,.{digits}f}%"
    latex = printed.replace("-", "$-$").replace("%", r"\%")
    return Cell(printed, latex)


def p_value(value: float) -> Cell:
    """A p-value with three decimals; ``<0.001`` below one thousandth."""
    if not math.isfinite(value):
        return EMPTY
    floor = 10.0**-P_VALUE_DECIMALS
    if value < floor:
        return Cell(
            f"<{floor:.{P_VALUE_DECIMALS}f}", f"$<${floor:.{P_VALUE_DECIMALS}f}"
        )
    return Cell(f"{value:.{P_VALUE_DECIMALS}f}")


def bracketed(low: Cell, high: Cell) -> Cell:
    """The interval ``[low, high]`` of two formatted endpoints; empty if either is."""
    if not low.text or not high.text:
        return EMPTY
    latex = f"{{[}}{low.latex or low.text}, {high.latex or high.text}{{]}}"
    return Cell(f"[{low.text}, {high.text}]", latex)


def parenthesized(cell: Cell) -> Cell:
    """A formatted value in parentheses; an empty cell stays empty."""
    if not cell.text:
        return EMPTY
    return Cell(f"({cell.text})", f"({cell.latex or cell.text})")


def starred(cell: Cell, stars: str) -> Cell:
    """A formatted value followed by its significance stars, a superscript in LaTeX."""
    if not stars or not cell.text:
        return cell
    return Cell(cell.text + stars, f"{cell.latex or cell.text}$^{{{stars}}}$")


def alignment(*leading: Align, rest: Align, count: int) -> tuple[Align, ...]:
    """The alignments ``leading``, then ``count`` columns aligned as ``rest``."""
    return (*leading, *(rest for _ in range(count)))


def label_of(name: str, labels: Mapping[str, str]) -> str:
    """The printed label of a column or term name: ``labels[name]`` where given."""
    return labels.get(name, name)


__all__ = [
    "EMPTY",
    "Align",
    "Cell",
    "Row",
    "Table",
    "alignment",
    "bracketed",
    "group_row",
    "integer",
    "label_of",
    "number",
    "p_value",
    "parenthesized",
    "percent",
    "starred",
    "text",
]
