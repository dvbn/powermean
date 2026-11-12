"""Typed lower result boundary; implementation composition belongs to interface."""

from __future__ import annotations

from typing import Protocol, Unpack

import pandas as pd

from powermean.data.specs import Estimator
from powermean.effects.targets import ReportingContext
from powermean.errors import ConvergenceError
from powermean.formula.frame import Design
from powermean.results.absorbed_owned import RetainedCertificateSnapshot
from powermean.results.fe_helpers import PredictionOptions
from powermean.results.fe_sample import FESampleView
from powermean.results.provenance import Provenance
from powermean.solve.cancelled_records import CancelledCertificate


class FEBackend(Protocol):
    """The retained capabilities delegated by the same public Result."""

    estimator: Estimator
    design: Design

    def _reporting_context(self) -> ReportingContext:
        """Snapshot actual reported beta, active full V, reference and point once."""
        ...

    @property
    def certificate(self) -> RetainedCertificateSnapshot | CancelledCertificate | None:
        """The authentic certificate, absent if none formed."""
        ...

    @property
    def sample(self) -> FESampleView:
        """The detached sample view."""
        ...

    @property
    def provenance(self) -> Provenance:
        """The actual computation record."""
        ...

    def _names(self) -> pd.Index[str]:
        """Serve the actual retained capability."""
        ...

    def _withheld(self) -> ConvergenceError:
        """Serve the actual retained capability."""
        ...

    def coef(self) -> pd.Series[float]:
        """Serve the actual retained capability."""
        ...

    def se(self) -> pd.Series[float]:
        """Serve the actual retained capability."""
        ...

    def vcov(self, spec: object = None) -> pd.DataFrame:
        """Serve the actual retained capability."""
        ...

    def with_vcov(self, spec: object) -> FEBackend:
        """A retained covariance view with the original fit certificate."""
        ...

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """Serve the actual retained capability."""
        ...

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """Serve the actual retained capability."""
        ...

    def summary(self, level: float = 0.95) -> str:
        """Serve the actual retained capability."""
        ...

    def influence(self) -> pd.DataFrame:
        """Serve the actual retained capability."""
        ...

    def fixef(self) -> dict[str, pd.Series[float]]:
        """Serve the actual retained capability."""
        ...

    def predict(
        self, data: pd.DataFrame | None = None, **options: Unpack[PredictionOptions]
    ) -> pd.Series[float]:
        """Return a defined prediction or explicit capability refusal."""
        ...
