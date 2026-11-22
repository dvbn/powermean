"""Sample and coefficient reporting for a Jochmans local solution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Literal, Unpack, overload

import numpy as np
import pandas as pd

from powermean.errors import SpecificationError
from powermean.results import citations, prediction
from powermean.results.fe_helpers import two_sided_p
from powermean.results.prediction import Prediction, PredictionOptions, prediction_kind
from powermean.results.two_way_prediction import TwoWayPrediction
from powermean.solve.jochmans import FloatArray, LocalRoot

_CONVENTIONS = ("paper", "stata")


@dataclass(frozen=True, eq=False)
class _Sample:
    """All input rows are retained, in input order."""

    _labels: pd.Index[Any]
    zeros: int
    fingerprint: str
    column_names: tuple[str, ...]

    @property
    def count(self) -> int:
        """Number of observed cells."""
        return len(self._labels)

    @property
    def nobs(self) -> int:
        """Number of input rows."""
        return self.count

    @property
    def positions(self) -> np.ndarray:
        """Original, zero-based positions of the observed cells."""
        return np.arange(self.count, dtype=np.intp)

    @property
    def retained_index(self) -> pd.Index[Any]:
        """Original index labels, including repeated labels."""
        return self._labels.copy(deep=True)

    @property
    def counts(self) -> Mapping[str, int]:
        """Accounting for the rows of the fit."""
        return MappingProxyType({"retained": self.count})


@dataclass(frozen=True)
class _Provenance:
    """The equation, sample geometry and inference used by the fit."""

    estimator: Mapping[str, object]
    covariance: Mapping[str, object]
    status: str
    references: tuple[citations.Reference, ...]
    certified: bool = True


@dataclass(frozen=True, eq=False)
class JochmansResult:
    """A fitted set of slopes with local root checks and cell-projection inference."""

    _beta: FloatArray
    _paper: FloatArray
    _stata_factor: float
    _convention: str
    sample: _Sample
    certificate: LocalRoot
    formula: str
    method: str
    layout: str
    dimensions: tuple[int, int]
    fixed_effects: tuple[str, str]
    _prediction: TwoWayPrediction

    @property
    def provenance(self) -> _Provenance:
        """The estimator, covariance and citations used in reporting."""
        estimator = MappingProxyType(
            {
                "method": f"Jochmans {self.method.upper()}",
                "layout": self.layout,
                "mean": MappingProxyType({"omega": 1.0}),
                "fixed_effects": MappingProxyType({"differenced": self.fixed_effects}),
            }
        )
        covariance = MappingProxyType(
            {
                "kind": "cell projection",
                "convention": self._convention,
                "factor": self._factor,
                "reference": "normal",
            }
        )
        refs = (*citations.MANUSCRIPTS, citations.JOCHMANS, citations.JOCHMANS_VERARDI)
        return _Provenance(
            estimator,
            covariance,
            self.certificate.status,
            tuple(citations.REGISTRY[key] for key in refs),
        )

    @property
    def _factor(self) -> float:
        """Finite-sample multiplier of the active covariance."""
        return self._stata_factor if self._convention == "stata" else 1.0

    def _names(self) -> pd.Index[str]:
        """Coefficient labels in the package reporting convention."""
        return pd.Index(self.sample.column_names, name="term")

    def coef(self) -> pd.Series[float]:
        """Slopes in formula design order."""
        return pd.Series(self._beta.copy(), index=self._names(), name="estimate")

    def fixef(self) -> dict[str, pd.Series[float]]:
        """Recover the two log-scale effects at the fixed GMM slopes."""
        stored = self._prediction
        out = {}
        for levels, values in zip(stored.levels, stored.recovered.effects, strict=True):
            if np.isinf(values).any() or not np.isfinite(stored.common):
                raise FloatingPointError(
                    "a recovered fixed effect is not representable"
                )
            effect = pd.Series(
                values.copy(), index=levels.copy(deep=True), name=levels.name
            )
            effect.attrs.update(
                fe_recovery="fixed_slope_ppml",
                normalization="first supported column level set to zero per component",
            )
            out[str(levels.name)] = effect
        return out

    @overload
    def predict(
        self,
        data: pd.DataFrame | None = None,
        *,
        type: Literal["power_mean", "residual"] = "power_mean",
    ) -> pd.Series[float]: ...

    @overload
    def predict(
        self,
        data: pd.DataFrame | None = None,
        *,
        type: Literal["fixed_effects"],
    ) -> dict[str, pd.Series[float]]: ...

    def predict(
        self, data: pd.DataFrame | None = None, **options: Unpack[PredictionOptions]
    ) -> Prediction:
        """Fitted means and residuals on observed cells, or recovered effects."""
        kind = prediction_kind(options)
        prediction.prediction_data(data, kind)
        if kind == "fixed_effects":
            return self.fixef()
        stored, labels = self._prediction, self.sample.retained_index
        matrix = None
        if data is not None:
            prediction.check_fitted_groups(
                data,
                labels,
                (
                    ((name,), levels, codes)
                    for name, levels, codes in zip(
                        self.fixed_effects,
                        stored.levels,
                        (stored.rows, stored.columns),
                        strict=True,
                    )
                ),
            )
            matrix = prediction.replayed_matrix(stored.design, data)
        q = stored.recovered.q
        zero = np.isneginf(q)
        boundary = bool(zero.any())
        if matrix is not None:
            delta = stored.counterfactual_delta(matrix, zero if boundary else None)
            with np.errstate(over="raise", invalid="raise"):
                q = q + delta
        if kind == "residual":
            series = prediction.residual_series(
                q, stored.y, labels, 1.0, boundary=boundary
            )
        else:
            series = prediction.fitted_series(
                q, labels, 1.0, boundary=boundary, counterfactual=data is not None
            )
        series.attrs.update(
            fe_recovery="fixed_slope_ppml",
            zero_boundary_rows=tuple(np.flatnonzero(zero).tolist()),
        )
        return series

    def se(self) -> pd.Series[float]:
        """Standard errors of the active cell-projection covariance."""
        return pd.Series(
            np.sqrt(self._paper.diagonal() * self._factor),
            index=self._names(),
            name="std_error",
        )

    def vcov(self, spec: str | None = None) -> pd.DataFrame:
        """Full slope covariance under the active or requested convention."""
        view = self if spec is None else self.with_vcov(spec)
        names = self._names()
        return pd.DataFrame(view._paper * view._factor, index=names, columns=names)

    def with_vcov(self, spec: object) -> JochmansResult:
        """Return a reporting view with 'paper' or 'stata' normalization."""
        convention = covariance_convention(spec)
        return replace(self, _convention=convention)

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """One row per slope, with two-sided normal-reference inference."""
        if not isinstance(level, float) or not 0 < level < 1:
            raise ValueError("level must be a float strictly between zero and one")
        se = self.se().to_numpy()
        quantile = -NormalDist().inv_cdf((1 - level) / 2)
        with np.errstate(over="ignore"):
            z = self._beta / se
            radius = quantile * se
            return pd.DataFrame(
                {
                    "term": list(self.sample.column_names),
                    "estimate": self._beta.copy(),
                    "std_error": se,
                    "statistic": z,
                    "p_value": two_sided_p(z),
                    "conf_low": self._beta - radius,
                    "conf_high": self._beta + radius,
                    "inference": "ok",
                }
            )

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """Normal-reference intervals by slope name."""
        table = self.tidy(level)
        return pd.DataFrame(
            table[["conf_low", "conf_high"]].to_numpy(),
            index=self._names(),
            columns=["lower", "upper"],
        )

    def effects(self, *, level: float = 0.95) -> pd.DataFrame:
        """Report exp(b) - 1 for a unit change in each design column."""
        table = self.tidy(level)
        transformed = ["estimate", "conf_low", "conf_high"]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            table[transformed] = np.expm1(table[transformed])
            table["std_error"] *= np.exp(self._beta)
        if not np.isfinite(table[[*transformed, "std_error"]].to_numpy()).all():
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": "Jochmans effect transformation outside float64 range"},
            )
        return table

    def summary(self, level: float = 0.95) -> str:
        """Summarize the selected root, sample and normal-reference inference."""
        table = self.tidy(level).drop(columns="inference")
        nr, nc = self.dimensions
        return "\n".join(
            (
                f"Jochmans {self.method.upper()} exponential regression",
                f"Formula: {self.formula}",
                f"Sample: {self.layout} {nr} x {nc}, "
                f"{self.sample.count} observed cells ({self.sample.zeros} zeros)",
                f"Std. errors: cell projection ({self._convention}), normal reference",
                f"Solution: local root, "
                f"relative moment {self.certificate.relative_moment:.1e}, "
                f"scaled correction {self.certificate.correction:.1e}",
                "",
                table.to_string(index=False, float_format=lambda value: f"{value:.5g}"),
                "",
                *(reference.text() for reference in self.provenance.references),
            )
        )


def covariance_convention(spec: object) -> str:
    """Validate the fixed set of projection normalizations."""
    if not isinstance(spec, str) or spec not in _CONVENTIONS:
        raise SpecificationError(
            "specification.unsupported_inference",
            {"method": spec, "case": "Jochmans cell-projection inference"},
        )
    return spec
