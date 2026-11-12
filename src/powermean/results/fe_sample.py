"""Detached sample views for normal private FE results."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from powermean._prepare.compose import Columns, _owned_index
from powermean.data.accounting import CodeArray, frozen_codes, frozen_index
from powermean.data.retained import OrderSample
from powermean.data.weights import WeightKind
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_partitions import RetainedGroupingSnapshot
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.cancelled_retained import RetainedCancelledEnvelope
from powermean.results.fe_helpers import detached_record
from powermean.systems.cancelled import CancelledAccounting


class FESampleView:
    """Unified detached sample view for absorbed and cancelled fixed-effects results."""

    __slots__ = (
        "_cancelled_envelope",
        "_columns",
        "_dependence",
        "_fe_snapshot",
        "_labels",
        "_order_sample",
        "_outcome_snapshot",
        "_subset_missing",
    )

    def __init__(
        self,
        order_sample: OrderSample,
        labels: pd.Index[Any],
        columns: Columns,
        subset_missing: int,
        *,
        dependence: tuple[str, ...] = (),
        fe_snapshot: RetainedSampleSnapshot | None = None,
        outcome_snapshot: RetainedOutcomeSnapshot | None = None,
        cancelled_envelope: RetainedCancelledEnvelope | None = None,
    ) -> None:
        self._order_sample = order_sample
        self._labels = _owned_index(labels)
        self._columns = columns
        self._subset_missing = int(subset_missing)
        self._dependence = dependence
        self._fe_snapshot = fe_snapshot
        self._outcome_snapshot = outcome_snapshot
        self._cancelled_envelope = cancelled_envelope

    @property
    def fingerprint(self) -> str:
        """Return the fingerprint from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.fingerprint
        if (
            self._outcome_snapshot is not None
            and self._outcome_snapshot.sample is not None
        ):
            return self._outcome_snapshot.sample.fingerprint
        return self._order_sample.fingerprint

    @property
    def omega(self) -> float:
        """Return the omega from the actual retained state, detached outward."""
        return self._order_sample.omega

    @property
    def positive_only(self) -> bool:
        """Return the positive only from the actual retained state, detached outward."""
        return self._order_sample.positive_only

    @property
    def conditioned(self) -> bool:
        """Return the conditioned from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.conditioned
        if (
            self._outcome_snapshot is not None
            and self._outcome_snapshot.sample is not None
        ):
            return self._outcome_snapshot.sample.conditioned
        return self._order_sample.conditioned

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the column names from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.active_names
        if self._cancelled_envelope is not None:
            return self._cancelled_envelope.slope_names
        if (
            self._outcome_snapshot is not None
            and self._outcome_snapshot.sample is not None
        ):
            return self._outcome_snapshot.sample.active_names
        return self._order_sample.data.column_names

    @property
    def active_names(self) -> tuple[str, ...]:
        """Return the active names from the actual retained state, detached outward."""
        return self.column_names

    @property
    def nobs(self) -> int:
        """Return the nobs from the actual retained state, detached outward."""
        return self._order_sample.nobs

    @property
    def count(self) -> int:
        """Return the count from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.count
        if self._outcome_snapshot is not None:
            return self._outcome_snapshot.ledger.count
        return self._order_sample.count

    @property
    def effective_n(self) -> int:
        """Return the effective n from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.effective_n
        if (
            self._outcome_snapshot is not None
            and self._outcome_snapshot.sample is not None
        ):
            return self._outcome_snapshot.sample.effective_n
        return self._order_sample.effective_n

    @property
    def zeros(self) -> int:
        """Return the zeros from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return int(np.count_nonzero(self._fe_snapshot.data.y == 0))
        if (
            self._outcome_snapshot is not None
            and self._outcome_snapshot.sample is not None
        ):
            return int(np.count_nonzero(self._outcome_snapshot.sample.data.y == 0))
        return int(np.count_nonzero(self._order_sample.data.y == 0))

    @property
    def weight_kind(self) -> WeightKind | None:
        """Return the weight kind from the actual retained state, detached outward."""
        w = self._order_sample.data.weights
        return None if w is None else w.kind

    @property
    def weights(self) -> FloatArray | None:
        """Return the weights from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            w = self._fe_snapshot.data.weights
            return None if w is None else w.values.copy()
        w = self._order_sample.data.weights
        return None if w is None else w.values.copy()

    @property
    def dropped_singleton(self) -> int:
        """Return the dropped singleton from retained state, detached."""
        if self._fe_snapshot is not None:
            return int(self._fe_snapshot.ledger.counts["singleton"])
        if self._outcome_snapshot is not None:
            return int(self._outcome_snapshot.ledger.counts["singleton"])
        return 0

    @property
    def dropped_separated(self) -> int:
        """Return the dropped separated from retained state, detached."""
        if self._fe_snapshot is not None:
            return int(self._fe_snapshot.ledger.counts["separated"])
        if self._outcome_snapshot is not None:
            return int(self._outcome_snapshot.ledger.counts["separated"])
        return 0

    @property
    def dropped_collinear(self) -> tuple[str, ...]:
        """Return the dropped collinear from retained state, detached."""
        if self._cancelled_envelope is not None:
            return tuple(name for name, _ in self._cancelled_envelope.omitted_slopes)
        if self._fe_snapshot is not None:
            all_names = self._order_sample.data.column_names
            active_set = set(self._fe_snapshot.active_names)
            return tuple(n for n in all_names if n not in active_set)
        if (
            self._outcome_snapshot is not None
            and self._outcome_snapshot.sample is not None
        ):
            all_names = self._order_sample.data.column_names
            active_set = set(self._outcome_snapshot.sample.active_names)
            return tuple(n for n in all_names if n not in active_set)
        return ()

    @property
    def omitted_slopes(self) -> tuple[tuple[str, str], ...]:
        """Return the omitted slopes from retained state, detached."""
        if self._cancelled_envelope is not None:
            return self._cancelled_envelope.omitted_slopes
        return tuple((name, "collinear") for name in self.dropped_collinear)

    @property
    def counts(self) -> Mapping[str, int]:
        """Return the counts from the actual retained state, detached outward."""
        base_counts = dict(self._order_sample.counts)
        if self._fe_snapshot is not None or self._outcome_snapshot is not None:
            base_counts["singleton"] = self.dropped_singleton
            base_counts["separated"] = self.dropped_separated
            base_counts["retained"] = self.count
        return MappingProxyType(base_counts)

    @property
    def missing(self) -> Mapping[str, int]:
        """Return the missing from the actual retained state, detached outward."""
        return self._order_sample.missing.counts

    def table(self) -> tuple[tuple[str, int], ...]:
        """Return the table from the actual retained state, detached outward."""
        return tuple(self.counts.items())

    def reason(self, position: int) -> str | None:
        """Return the row's first exclusion reason, or ``None`` if retained.

        A retained snapshot's ``reasons`` are its collinearity reasons per original
        COLUMN; the row reasons are its ledger's, as the other branches read them.
        """
        if self._fe_snapshot is not None:
            if not 0 <= position < self.nobs:
                raise IndexError("row position out of range")
            return self._fe_snapshot._ledger.reason(position)
        if self._outcome_snapshot is not None:
            if not 0 <= position < self.nobs:
                raise IndexError("row position out of range")
            return self._outcome_snapshot.ledger.reason(position)
        return self._order_sample.reason(position)

    @property
    def reasons(self) -> tuple[str | None, ...]:
        """Return one row reason per input row, from the actual retained state."""
        if self._fe_snapshot is not None:
            ledger = self._fe_snapshot._ledger
            return tuple(ledger.reason(i) for i in range(self.nobs))
        if self._outcome_snapshot is not None:
            return tuple(
                self._outcome_snapshot.ledger.reason(i) for i in range(self.nobs)
            )
        return tuple(self._order_sample.reason(i) for i in range(self.nobs))

    @property
    def positions(self) -> IndexArray:
        """Return the positions from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.positions
        if self._outcome_snapshot is not None:
            return frozen_index(self._outcome_snapshot.ledger.positions)
        return frozen_index(self._order_sample.positions)

    @property
    def retained_index(self) -> IndexArray:
        """Return the retained index from retained state, detached."""
        pos = self.positions
        idx = np.full(self.nobs, -1, dtype=np.intp)
        if pos.size > 0:
            idx[pos] = np.arange(pos.size, dtype=np.intp)
        return frozen_index(idx)

    @property
    def codes(self) -> CodeArray:
        """Return the codes from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return frozen_codes(self._fe_snapshot.ledger.codes)
        if self._outcome_snapshot is not None:
            return frozen_codes(self._outcome_snapshot.ledger.codes)
        return frozen_codes(self._order_sample.codes)

    @property
    def labels(self) -> pd.Index[Any]:
        """Return the labels from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return self._fe_snapshot.labels
        pos = self.positions
        return _owned_index(self._labels.take(pos))

    @property
    def input_labels(self) -> pd.Index[Any]:
        """Return the input labels from the actual retained state, detached outward."""
        return _owned_index(self._labels)

    @property
    def cluster_names(self) -> tuple[str, ...]:
        """Return the cluster names from the actual retained state, detached outward."""
        if self._dependence:
            return self._dependence
        return tuple(self._order_sample.metadata.columns)

    @property
    def source(self) -> RetainedGroupingSnapshot | None:
        """Return the source from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return detached_record(self._fe_snapshot.source)
        return None

    @property
    def grouping(self) -> RetainedGroupingSnapshot | None:
        """Return the grouping from the actual retained state, detached outward."""
        if self._fe_snapshot is not None:
            return detached_record(self._fe_snapshot.grouping)
        return None

    @property
    def cancelled_name(self) -> str | None:
        """Return the cancelled name from retained state, detached."""
        if self._cancelled_envelope is not None:
            return self._cancelled_envelope.cancelled_name
        return None

    @property
    def retained_names(self) -> tuple[str, ...] | None:
        """Return the retained names from retained state, detached."""
        if self._cancelled_envelope is not None:
            return self._cancelled_envelope.retained_names
        return None

    @property
    def accounting(self) -> CancelledAccounting | None:
        """Return the accounting from the actual retained state, detached outward."""
        if self._cancelled_envelope is not None:
            return self._cancelled_envelope.accounting
        return None
