"""Temporary per-handoff input and problem identity maps and value re-exports.

This module defines :class:`InputIdentityMap` and :class:`ProblemIdentityMap` with
concrete typed caches and strong-reference anchors cleared after handoff capture, and
re-exports input value helpers from :mod:`powermean.results.retained_input_values`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean.data.collinearity import CollinearityProblem
from powermean.data.sample import SampleBase
from powermean.results.retained_diagnostic_problems import (
    RetainedCollinearityProblem,
    RetainedGammaProblem,
    RetainedSupportProblem,
)
from powermean.results.retained_input_values import (
    GroupingKey,
    InputKey,
    PartitionKey,
    PointKey,
    ProblemKey,
    RetainedInputSnapshot,
    RetainedLabelBatch,
    SampleKey,
    _copy_column,
    _copy_scalar_with_memo,
    _copy_with_memo,
    _is_missing_scalar,
    _owned_column,
    _owned_column_provenance,
    _owned_column_values,
    _owned_dependence,
    _owned_dependence_provenance,
    _owned_effect_columns,
    _owned_fixed_effects,
    _owned_missing,
    _owned_model_frame,
    _owned_prepared_frame,
    _owned_raw_inputs,
    _owned_row_ledger,
    _owned_sample_base,
    _owned_scaled_design,
    _owned_transformed_problem,
    _same_column_values,
    _stored_raw_elements,
    _tuple_arrays,
    build_label_batch,
    owned_order_scale,
)
from powermean.solve.support_gamma_problem import GammaProblem
from powermean.solve.support_problem import SupportProblem

__all__ = [
    "GroupingKey",
    "InputIdentityMap",
    "InputKey",
    "PartitionKey",
    "PointKey",
    "ProblemIdentityMap",
    "ProblemKey",
    "RetainedInputSnapshot",
    "RetainedLabelBatch",
    "SampleKey",
    "_copy_column",
    "_copy_scalar_with_memo",
    "_copy_with_memo",
    "_is_missing_scalar",
    "_owned_column",
    "_owned_column_provenance",
    "_owned_column_values",
    "_owned_dependence",
    "_owned_dependence_provenance",
    "_owned_effect_columns",
    "_owned_fixed_effects",
    "_owned_missing",
    "_owned_model_frame",
    "_owned_prepared_frame",
    "_owned_raw_inputs",
    "_owned_row_ledger",
    "_owned_sample_base",
    "_owned_scaled_design",
    "_owned_transformed_problem",
    "_same_column_values",
    "_stored_raw_elements",
    "_tuple_arrays",
    "build_label_batch",
    "owned_order_scale",
]


class InputIdentityMap:
    """Temporary per-handoff identity map from ``SampleBase`` / ``PreparedFrame``."""

    def __init__(self) -> None:
        """Initialize an empty input identity map."""
        self._by_base_id: dict[int, RetainedInputSnapshot] = {}
        self._by_prepared_id: dict[int, RetainedInputSnapshot] = {}
        self._refs: list[object] = []
        self._next_key: InputKey = 0

    def register_snapshot(
        self,
        snapshot: RetainedInputSnapshot,
        *,
        base: SampleBase | None = None,
        prepared: PreparedFrame | None = None,
    ) -> RetainedInputSnapshot:
        """Bind an existing ``RetainedInputSnapshot`` to original object identities."""
        if snapshot.key >= self._next_key:
            self._next_key = snapshot.key + 1
        if base is not None:
            self._by_base_id[id(base)] = snapshot
            self._refs.append(base)
        if prepared is not None:
            self._by_prepared_id[id(prepared)] = snapshot
            self._by_base_id[id(prepared.base)] = snapshot
            self._refs.append(prepared)
            self._refs.append(prepared.base)
        return snapshot

    def capture(
        self,
        base: SampleBase,
        *,
        prepared: PreparedFrame | None = None,
        labels: pd.Index[Any] | None = None,
    ) -> RetainedInputSnapshot:
        """Return the shared ``RetainedInputSnapshot`` for ``prepared`` or ``base``."""
        if prepared is not None and id(prepared) in self._by_prepared_id:
            return self._by_prepared_id[id(prepared)]
        if id(base) in self._by_base_id:
            snap = self._by_base_id[id(base)]
            if prepared is not None:
                self._by_prepared_id[id(prepared)] = snap
                self._refs.append(prepared)
            return snap
        key = self._next_key
        self._next_key += 1
        snap = RetainedInputSnapshot(base, key=key, prepared=prepared, labels=labels)
        return self.register_snapshot(snap, base=base, prepared=prepared)

    def register(self, prepared: PreparedFrame) -> RetainedInputSnapshot:
        """Capture or return the shared ``RetainedInputSnapshot`` for ``prepared``."""
        return self.capture(prepared.base, prepared=prepared)

    def clear_anchors(self) -> None:
        """Release strong references to original input objects after capture."""
        self._refs.clear()


class ProblemIdentityMap:
    """Temporary per-handoff map assigning ``ProblemKey`` with concrete typed caches."""

    def __init__(self) -> None:
        """Initialize an empty problem identity map."""
        self._by_id: dict[int, ProblemKey] = {}
        self._support_by_id: dict[int, RetainedSupportProblem] = {}
        self._gamma_by_id: dict[int, RetainedGammaProblem] = {}
        self._collinearity_by_id: dict[int, RetainedCollinearityProblem] = {}
        self._refs: list[object] = []
        self._next_key: ProblemKey = 0

    def key_for(self, problem: object) -> ProblemKey:
        """Return the ``ProblemKey`` for ``problem``, assigning a new one if needed."""
        pid = id(problem)
        if pid in self._by_id:
            return self._by_id[pid]
        key = self._next_key
        self._next_key += 1
        self._by_id[pid] = key
        self._refs.append(problem)
        return key

    def register_support(
        self,
        problem: SupportProblem,
        builder: Callable[[ProblemKey], RetainedSupportProblem],
    ) -> RetainedSupportProblem:
        """Return cached or freshly built :class:`RetainedSupportProblem`."""
        pid = id(problem)
        if pid in self._support_by_id:
            return self._support_by_id[pid]
        snap = builder(self.key_for(problem))
        self._support_by_id[pid] = snap
        return snap

    def register_gamma(
        self,
        problem: GammaProblem,
        builder: Callable[[ProblemKey], RetainedGammaProblem],
    ) -> RetainedGammaProblem:
        """Return cached or freshly built :class:`RetainedGammaProblem`."""
        pid = id(problem)
        if pid in self._gamma_by_id:
            return self._gamma_by_id[pid]
        snap = builder(self.key_for(problem))
        self._gamma_by_id[pid] = snap
        return snap

    def register_collinearity(
        self,
        problem: CollinearityProblem,
        builder: Callable[[ProblemKey], RetainedCollinearityProblem],
    ) -> RetainedCollinearityProblem:
        """Return cached or freshly built :class:`RetainedCollinearityProblem`."""
        pid = id(problem)
        if pid in self._collinearity_by_id:
            return self._collinearity_by_id[pid]
        snap = builder(self.key_for(problem))
        self._collinearity_by_id[pid] = snap
        return snap

    def clear_anchors(self) -> None:
        """Release strong references to original problem objects after capture."""
        self._refs.clear()
