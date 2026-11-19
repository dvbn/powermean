r"""The four writers of a :class:`~powermean.results.tables.Table`.

Each writer reads the same grid of cells, so that the exports of a table cannot
disagree; none computes or rounds a number.

- Text: the fixed-width layout, a title line, a rule, the header, a rule between
  sections and the notes.
- Markdown: a pipe table.  The characters that Markdown reads as markup (``\``,
  `````, ``*``, ``_``, ``|``, ``<``, ``>``) are escaped, so that significance stars
  and names with underscores print as written.  Sections are not separated, since a
  pipe table has no interior rule.
- LaTeX: a ``tabular``, by default with the rules of ``booktabs`` and inside a
  ``table`` environment; the notes follow in a ``minipage``.  The ten special
  characters of LaTeX are escaped, and a bracket is written ``{[}`` so that a cell
  that starts with one is not read as the optional argument of ``\\``.
- HTML: a ``table`` element with one ``tbody`` per section, well formed as XML.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from powermean.results.tables import Cell, Row, Table

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    "[": "{[}",
    "]": "{]}",
}
_MARKDOWN_SPECIAL = "\\`*_|<>&[]()!"
_MARKDOWN_ALIGN = {"l": ":---", "c": ":---:", "r": "---:"}
_HTML_ALIGN = {"l": "left", "c": "center", "r": "right"}


def latex_escape(text: str) -> str:
    """``text`` with the special characters of LaTeX escaped."""
    return "".join(_LATEX_SPECIAL.get(character, character) for character in text)


def markdown_escape(text: str) -> str:
    """``text`` with the characters that Markdown reads as markup escaped."""
    return "".join(
        "\\" + character if character in _MARKDOWN_SPECIAL else character
        for character in text
    )


def _caption(table: Table) -> str:
    """The title and the subtitle on one line; empty when the table has neither."""
    return ". ".join(part for part in (table.title, table.subtitle) if part)


def _latex(cell: Cell) -> str:
    """The LaTeX form of a cell."""
    return cell.latex if cell.latex is not None else latex_escape(cell.text)


# ------------------------------------------------------------------------ frame


def to_strings(table: Table) -> pd.DataFrame:
    """The printed cells as a frame of strings, with the group of each row."""
    names = [cell.text for cell in table.columns]
    records: list[list[str]] = []
    group = ""
    for section in table.sections:
        for row in section:
            if row.kind == "group":
                group = row.cells[0].text
                continue
            records.append([group, *(cell.text for cell in row.cells)])
    frame = pd.DataFrame(records, columns=["group", *names], dtype=object)
    frame.attrs["title"] = table.title
    frame.attrs["subtitle"] = table.subtitle
    frame.attrs["notes"] = list(table.notes)
    return frame


# ------------------------------------------------------------------------- text


def _padded(text: str, width: int, align: str) -> str:
    """``text`` in a field of ``width`` characters."""
    if align == "l":
        return text.ljust(width)
    if align == "c":
        return text.center(width)
    return text.rjust(width)


def to_text(table: Table) -> str:
    """The fixed-width text of a table."""
    rows = [row for section in table.sections for row in section if row.kind == "data"]
    widths = [len(cell.text) for cell in table.columns]
    for row in rows:
        widths = [
            max(w, len(cell.text)) for w, cell in zip(widths, row.cells, strict=True)
        ]
    total = sum(widths) + 2 * (len(widths) - 1)
    rule = "-" * max(total, len(table.title) + len(table.subtitle) + 4)

    def line(cells: tuple[Cell, ...]) -> str:
        fields = [
            _padded(cell.text, width, align)
            for cell, width, align in zip(cells, widths, table.align, strict=True)
        ]
        return "  ".join(fields).rstrip()

    lines: list[str] = []
    if table.title or table.subtitle:
        gap = len(rule) - len(table.title) - len(table.subtitle)
        lines.append((table.title + " " * gap + table.subtitle).rstrip())
    lines += [rule, line(table.columns), rule]
    for section in table.sections:
        for row in section:
            lines.append(row.cells[0].text if row.kind == "group" else line(row.cells))
        lines.append(rule)
    lines += list(table.notes)
    return "\n".join(lines)


# --------------------------------------------------------------------- markdown


def _markdown_row(row: Row, width: int) -> str:
    """One pipe-table row; a group heading is bold in the first column."""
    if row.kind == "group":
        cells = [f"**{markdown_escape(row.cells[0].text)}**"] + [""] * (width - 1)
    else:
        cells = [markdown_escape(cell.text) for cell in row.cells]
    return "| " + " | ".join(cells) + " |"


def to_markdown(table: Table) -> str:
    """The Markdown of a table: a caption line, the pipe table and the notes."""
    width = len(table.columns)
    lines: list[str] = []
    caption = _caption(table)
    if caption:
        lines += [f"**{markdown_escape(caption)}**", ""]
    lines.append(
        "| " + " | ".join(markdown_escape(c.text) for c in table.columns) + " |"
    )
    lines.append("| " + " | ".join(_MARKDOWN_ALIGN[a] for a in table.align) + " |")
    for section in table.sections:
        lines += [_markdown_row(row, width) for row in section]
    if table.notes:
        notes = " ".join(markdown_escape(note) for note in table.notes)
        lines += ["", f"*Notes:* {notes}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ latex


def _latex_row(row: Row, width: int) -> str:
    """One ``tabular`` row; a group heading spans the table in italics."""
    if row.kind == "group":
        heading = latex_escape(row.cells[0].text)
        return rf"\multicolumn{{{width}}}{{l}}{{\textit{{{heading}}}}} \\"
    return " & ".join(_latex(cell) for cell in row.cells) + r" \\"


def _latex_label(label: str) -> str:
    """Reference keys admit underscores, but must contain no TeX commands."""
    if all(c.isascii() and (c.isalnum() or c in "_:-.") for c in label):
        return label
    return "powermean-" + label.encode("utf-8").hex()


def _latex_header(
    table: Table, caption: str, *, floating: bool, label: str | None
) -> list[str]:
    """The opening lines of the LaTeX output before the ``tabular`` rules."""
    lines: list[str] = []
    if floating:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label is not None:
            lines.append(rf"\label{{{_latex_label(label)}}}")
    elif caption:
        lines.append(f"% {caption}")
    lines.append(rf"\begin{{tabular}}{{{''.join(table.align)}}}")
    return lines


def _latex_footer(table: Table, *, floating: bool) -> list[str]:
    """The closing lines of the LaTeX output after the ``tabular`` body."""
    lines = [r"\end{tabular}"]
    if table.notes:
        notes = " ".join(latex_escape(note) for note in table.notes)
        lines += [
            r"\par\smallskip",
            r"\begin{minipage}{\linewidth}",
            rf"\footnotesize\textit{{Notes:}} {notes}",
            r"\end{minipage}",
        ]
    if floating:
        lines.append(r"\end{table}")
    return lines


def to_latex(table: Table, *, booktabs: bool, floating: bool, label: str | None) -> str:
    """The LaTeX of a table (:meth:`powermean.results.tables.Table.to_latex`)."""
    width = len(table.columns)
    top, middle, bottom = (
        (r"\toprule", r"\midrule", r"\bottomrule")
        if booktabs
        else (r"\hline", r"\hline", r"\hline")
    )
    caption = latex_escape(_caption(table))
    lines = _latex_header(table, caption, floating=floating, label=label)
    lines += [top, " & ".join(_latex(c) for c in table.columns) + r" \\", middle]
    for position, section in enumerate(table.sections):
        if position:
            lines.append(middle)
        lines += [_latex_row(row, width) for row in section]
    lines.append(bottom)
    lines += _latex_footer(table, floating=floating)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- html


def _html_row(row: Row, width: int, align: tuple[str, ...]) -> str:
    """One ``tr``; the first cell of a data row is its row header."""
    if row.kind == "group":
        heading = html.escape(row.cells[0].text)
        return f'<tr><th colspan="{width}" scope="colgroup">{heading}</th></tr>'
    cells = [f'<th scope="row">{html.escape(row.cells[0].text)}</th>']
    cells += [
        f'<td style="text-align: {_HTML_ALIGN[a]}">{html.escape(cell.text)}</td>'
        for cell, a in zip(row.cells[1:], align[1:], strict=True)
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def to_html(table: Table) -> str:
    """The HTML of a table: caption, header, one ``tbody`` per section, notes."""
    width = len(table.columns)
    lines = ['<table class="powermean-table">']
    caption = _caption(table)
    if caption:
        lines.append(f"<caption>{html.escape(caption)}</caption>")
    header = "".join(
        f'<th scope="col">{html.escape(cell.text)}</th>' for cell in table.columns
    )
    lines.append(f"<thead><tr>{header}</tr></thead>")
    for section in table.sections:
        lines.append("<tbody>")
        lines += [_html_row(row, width, table.align) for row in section]
        lines.append("</tbody>")
    if table.notes:
        notes = " ".join(html.escape(note) for note in table.notes)
        lines.append(
            f'<tfoot><tr><td colspan="{width}">Notes: {notes}</td></tr></tfoot>'
        )
    lines.append("</table>")
    return "\n".join(lines) + "\n"
