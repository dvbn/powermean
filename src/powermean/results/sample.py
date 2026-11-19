"""The sample view of a result: positions, labels, accounting and counts.

``design/02_computation.md``, Section 2 ("the result stores, for every input row,
either its position in the estimation sample or the reason for its exclusion;
``result.sample.table()`` prints the counts in the order above"), and Section 1, rule
4; ``design/04_inference_and_effects.md``, Section 2.3; decisions D41 and D55.

The view reads the :class:`~powermean.data.retained.OrderSample` of the fit and the
owned user labels of every input row, and hands out nothing retained: integer arrays
and labels are fresh copies on every access, so that an array made writable or a
label changed by its holder reaches no retained state.  Original positions are the
row keys; the user's labels are kept apart from them, exact (duplicates, a MultiIndex,
dates with a time zone), and never a key.  Three counts stay apart: ``nobs``, the input
rows; ``count``, the retained rows that enter the estimating equation (``N`` of the
summary of file 02); and ``effective_n``, the statistical count of the ``"stata"``
factors, the exact total of frequency counts.  A cluster count ``G`` belongs to a
covariance, not to the sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from powermean._prepare.compose import Columns, _owned_index
from powermean.data.accounting import CodeArray, frozen_codes, frozen_index
from powermean.data.retained import OrderSample, order_counts, order_table
from powermean.data.sample_fe_state import RowLedger
from powermean.data.weights import WeightKind
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray


@dataclass(frozen=True, eq=False)
class SampleView:
    """The sample of a result (module docstring).

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit.
    labels : pandas.Index
        The user's labels of every input row, owned by the result.
    columns : Columns
        The columns that the preparation read.
    subset_missing : int
        The rows whose subset condition was missing.
    design : tuple of str, optional
        The columns of the fitted design, in formula order.  Those that the sample
        does not hold are the regressors omitted because of collinearity (owner's
        rulings 109 and 110); without it, none is reported.
    """

    _sample: OrderSample
    _labels: pd.Index[Any]
    columns: Columns
    subset_missing: int
    _design: tuple[str, ...]
    _exhausted: RowLedger | None = None

    def __init__(
        self,
        sample: OrderSample,
        labels: pd.Index[Any],
        columns: Columns,
        subset_missing: int,
        design: tuple[str, ...] = (),
        *,
        exhausted: RowLedger | None = None,
    ) -> None:
        """Keep the records; the labels are owned by the caller."""
        object.__setattr__(self, "_sample", sample)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "subset_missing", subset_missing)
        object.__setattr__(self, "_design", design)
        object.__setattr__(self, "_exhausted", exhausted)

    # ---------------------------------------------------------------- identity

    @property
    def fingerprint(self) -> str:
        """The fingerprint of the sample (:mod:`powermean.data.fingerprint`)."""
        if self._exhausted is not None:
            identity = self._sample.fingerprint.encode("ascii")
            return sha256(b"powermean.sample/exhausted\0" + identity).hexdigest()
        return self._sample.fingerprint

    @property
    def omega(self) -> float:
        """The mean order of the sample."""
        return self._sample.omega

    @property
    def positive_only(self) -> bool:
        """Whether the caller asked for the positive outcomes only."""
        return self._sample.positive_only

    @property
    def conditioned(self) -> bool:
        """Whether a zero outcome was removed: the fit conditions on positive ones."""
        return self._sample.conditioned

    @property
    def column_names(self) -> tuple[str, ...]:
        """The names of the columns of the design, in order."""
        return self._sample.data.column_names

    # ------------------------------------------------------------------ counts

    @property
    def dropped_collinear(self) -> tuple[str, ...]:
        """The regressors omitted because of collinearity, in formula order.

        The columns of the fitted design that the sample does not hold, as the
        routes with fixed effects report them
        (:attr:`powermean.results.fe_sample.FESampleView.dropped_collinear`).
        """
        held = self._sample.data.column_names
        return tuple(name for name in self._design if name not in held)

    @property
    def omitted_slopes(self) -> tuple[tuple[str, str], ...]:
        """Each omitted regressor with its reason, ``"collinear"``."""
        return tuple((name, "collinear") for name in self.dropped_collinear)

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return self._sample.nobs

    @property
    def count(self) -> int:
        """The retained rows, which enter the estimating equation."""
        return self._rows.count

    @property
    def effective_n(self) -> int:
        """``N``: the exact total of frequency counts, else the retained rows."""
        return self._sample.effective_n if self._exhausted is None else 0

    @property
    def zeros(self) -> int:
        """The retained rows whose outcome is zero."""
        return (
            int(np.count_nonzero(self._sample.data.y == 0))
            if self._exhausted is None
            else 0
        )

    @property
    def weight_kind(self) -> WeightKind | None:
        """The stated kind of the weights, or ``None`` without weights."""
        weights = self._sample.data.weights
        return None if weights is None else weights.kind

    @property
    def weights(self) -> FloatArray | None:
        """The raw weights of the retained rows, a fresh copy, or ``None``."""
        weights = self._sample.data.weights
        if weights is None:
            return None
        return weights.values.copy() if self._exhausted is None else np.empty(0)

    @property
    def counts(self) -> Mapping[str, int]:
        """The rows excluded by each reason, in the order of the steps, and retained."""
        if self._exhausted is not None:
            return order_counts(self._exhausted.codes, separated=True)
        return self._sample.counts

    @property
    def _rows(self) -> OrderSample | RowLedger:
        """The final positions and first causes, including an exhausted refusal."""
        return self._sample if self._exhausted is None else self._exhausted

    @property
    def missing(self) -> Mapping[str, int]:
        """The rows that each variable group of step 2 explains; groups overlap."""
        return self._sample.missing.counts

    def table(self) -> tuple[tuple[str, int], ...]:
        """The counts in the order of file 02, Section 2 (``OrderSample.table``)."""
        return order_table(self.counts, self._rows.missing)

    def reason(self, position: int) -> str | None:
        """The reason that excluded the input row ``position``, or ``None`` if kept."""
        return self._rows.reason(position)

    # --------------------------------------------------------------- positions

    @property
    def positions(self) -> IndexArray:
        """The original positions of the retained rows, increasing; a fresh copy."""
        return frozen_index(self._rows.positions)

    @property
    def retained_index(self) -> IndexArray:
        """For each input row, its position in the sample or ``-1``; a fresh copy."""
        return frozen_index(self._rows.retained_index)

    @property
    def codes(self) -> CodeArray:
        """One code per input row, zero where retained; a fresh copy."""
        return frozen_codes(self._rows.codes)

    # ------------------------------------------------------------------ labels

    @property
    def labels(self) -> pd.Index[Any]:
        """The user's labels of the retained rows, by position; a fresh owned copy."""
        return _owned_index(self._labels.take(self._rows.positions))

    @property
    def input_labels(self) -> pd.Index[Any]:
        """The user's labels of every input row, as given; a fresh owned copy."""
        return _owned_index(self._labels)

    @property
    def cluster_names(self) -> tuple[str, ...]:
        """The classifications retained with the fit, by name."""
        return tuple(self._sample.metadata)
