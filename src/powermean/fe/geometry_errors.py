"""Typed refusal diagnostics for the geometry preparation."""

from __future__ import annotations


class GeometryBudgetError(ValueError):
    """Raised before allocation when required geometry bytes exceed ``max_bytes``."""

    def __init__(
        self,
        prospective_bytes: int,
        max_bytes: int,
        *,
        n_rows: int,
        n_slopes: int,
        n_retained_levels: int,
    ) -> None:
        self.prospective_bytes = int(prospective_bytes)
        self.max_bytes = int(max_bytes)
        self.n_rows = int(n_rows)
        self.n_slopes = int(n_slopes)
        self.n_retained_levels = int(n_retained_levels)
        self.n_columns_full = int(n_slopes + n_retained_levels)
        super().__init__(
            f"geometry prospective allocation {self.prospective_bytes} bytes exceeds "
            f"task-local max_bytes={self.max_bytes} for n={self.n_rows}, "
            f"K={self.n_slopes}, "
            f"L={self.n_retained_levels}, p={self.n_columns_full}"
        )


class GeometryRankError(ValueError):
    """Unresolved spectral, Schur or conditioning evidence."""

    def __init__(
        self,
        stage: str,
        gap_ratio: float,
        gap_min: float,
        *,
        gram_rtol: float,
        threshold: float,
        smallest_retained: float,
        largest_discarded: float,
    ) -> None:
        self.stage = stage
        self.gap_ratio = float(gap_ratio)
        self.gap_min = float(gap_min)
        self.gram_rtol = float(gram_rtol)
        self.threshold = float(threshold)
        self.smallest_retained = float(smallest_retained)
        self.largest_discarded = float(largest_discarded)
        super().__init__(
            f"unresolved geometry in {stage}: gap_ratio={self.gap_ratio:.6g}, "
            f"gap_min={self.gap_min:.6g} "
            f"(gram_rtol={self.gram_rtol:.3g}, threshold={self.threshold:.6g}, "
            f"smallest_retained={self.smallest_retained:.6g}, "
            f"largest_discarded={self.largest_discarded:.6g})"
        )


class GeometryAdmissionError(ValueError):
    """Fresh raw binding or original-space evidence disagrees with a candidate."""

    def __init__(self, cause: str, detail: str) -> None:
        self.cause = cause
        super().__init__(f"geometry admission refused ({cause}): {detail}")
