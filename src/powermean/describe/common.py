"""The pieces of layout that the preliminary tables share.

The subtitle that names the arms with their numbers of rows, the suffix of a column
that concerns one arm, the sentence that states the covariance of a table, and the
assembly of the unrounded frame from one record per row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from powermean.describe.frame import Arms, BoolArray, Context
from powermean.describe.options import TableOptions


def real(record: Mapping[str, object], name: str) -> float:
    """The real value of the field ``name`` of a record."""
    return float(cast("float", record[name]))


def arm_counts(arms: Arms, usable: BoolArray) -> list[int]:
    """The number of usable rows of each arm, the control first."""
    return [int((usable & (arms.codes == d)).sum()) for d in range(len(arms.levels))]


def arms_subtitle(arms: Arms, usable: BoolArray) -> str:
    """``control: treat = 0 (N = 1,012); treat = 1 (N = 988)``."""
    counts = arm_counts(arms, usable)
    parts = [f"{arms.text(d)} (N = {counts[d]:,d})" for d in range(len(counts))]
    return "control: " + "; ".join(parts)


def arm_suffix(arms: Arms, position: int) -> str:
    """`` (level)``, the mark of a column that concerns the arm at ``position``."""
    return f" ({arms.labels[position]})"


def treated_suffix(arms: Arms, position: int) -> str:
    """The mark of a comparison column: empty when one arm is compared."""
    return arm_suffix(arms, position) if len(arms.levels) > 2 else ""


def covariance_note(options: TableOptions) -> str:
    """The sentence that states a robust or cluster-robust covariance."""
    kind = (
        f"cluster-robust by {options.cluster!r}"
        if options.cluster is not None
        else "robust to heteroskedasticity"
    )
    return f"Covariance: {kind}, convention ssc={options.ssc!r}."


def strata_note(options: TableOptions) -> str:
    """The sentence that states absorbed strata; empty without them."""
    if options.fe is None:
        return ""
    return f"Strata {options.fe!r} absorbed: differences are within strata."


def complete_mask(context: Context, columns: Sequence[np.ndarray]) -> BoolArray:
    """The rows on which every one of the columns is observed."""
    mask = np.ones(len(context.frame), dtype=np.bool_)
    for values in columns:
        mask &= ~np.isnan(values)
    return mask


def assemble(
    records: Sequence[Mapping[str, object]],
    index: Sequence[str],
    attrs: Mapping[str, object],
) -> pd.DataFrame:
    """The unrounded frame of a table: one record per row, indexed by ``index``.

    Every record names its subgroup under ``by``: the column is dropped when it is
    empty in every record (no subgroups), and it leads the index otherwise.
    """
    frame = pd.DataFrame.from_records(list(records))
    keys = list(index)
    if (frame["by"] == "").all():
        frame = frame.drop(columns="by")
    else:
        keys = ["by", *keys]
    frame = frame.set_index(keys)
    frame.attrs.update(attrs)
    return frame
