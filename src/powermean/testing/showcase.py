"""One firm-year panel with a column for every feature of the package.

The showcase is not a design of the paper.  It exists so that every argument of the
package can be tried on one dataset whose parameters are known.  Its outcome laws
are those of the solver study.

The mean of ``y`` is ``exposure * exp(index)``, where the index sums a firm effect,
a year effect and the regressors with the coefficients of ``SHOWCASE_TRUTH``.  Every
regressor is independent of the others and of the effects, except ``x_normal``.
Omitting independent regressors from an exponential mean changes its intercept
only, so a model with fewer regressors still has the listed coefficients on those
it keeps.

Changing how an existing column is drawn changes every fixture built on this
dataset.  A new column draws from a new stream, after the existing ones.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from powermean.testing.synthetic import (
    CONTAMINATION,
    TREATMENT_COEFFICIENT,
    FloatArray,
    IntArray,
    Synthetic,
    checksum,
    draw_outcome,
    inflate_zeros,
    require_design,
    seeded,
)

Columns = dict[str, npt.NDArray[Any]]

SHOWCASE_LAWS: dict[str, tuple[str, Mapping[str, float]]] = {
    "poisson": ("poisson", {}),
    "gamma": ("gamma", {}),
    "nb2_mild": ("nb2", {"size": 5.0}),
    "nb2": ("nb2", {"size": 1.0}),
    "nb2_strong": ("nb2", {"size": 0.2}),
    "tweedie15": ("tweedie15", {}),
    "ig3": ("ig3", {}),
    "lognormal": ("lognormal_dmc", {"residual_sd": 1.0}),
    "lognormal_heavy": ("lognormal_dmc", {"residual_sd": 2.5}),
    "contamination": ("contamination", CONTAMINATION),
}
"""Outcome laws of the solver study, by the names used in its design ids."""

SHOWCASE_TRUTH: dict[str, float] = {
    "treat": TREATMENT_COEFFICIENT,
    "x_normal": 0.3,
    "x_skewed": -0.2,
    "x_binary": 0.25,
    "category[b]": 0.2,
    "category[c]": -0.3,
    "dose": 0.1,
    "arm[1]": math.log(1.05),
    "arm[2]": math.log(1.15),
    "policy": math.log(1.10),
    "log(exposure)": 1.0,
    "d_endog": 0.2,
}
"""True coefficients.  ``d_endog`` enters the mean of ``y_iv`` only."""

_INTERCEPT = math.log(2.0)
_FIRM_SD, _YEAR_SD, _SHOCK_SD = 0.5, 0.3, 0.3
_FIRM_LOADING = 0.6  # correlation of x_normal with the firm effect
_SD_UNTREATED, _SD_TREATED = 0.5, 0.7  # log-scale standard deviations of y_pos
_ENDOGENEITY = 0.6  # loading of the error of y_iv on the first-stage error
_FIRST_STAGE = (0.7, 0.5, 0.5)  # coefficients of z1, z2 and the independent noise
_INVALID_LOADING = 0.5  # loading of z_invalid on the first-stage error
_STREAMS = {
    "effects": 0,
    "regressors": 1,
    "outcomes": 2,
    "instruments": 3,
    "auxiliary": 4,
    "zeros": 5,
}

_ROLES = {
    "state": "cluster with few levels: five regions per state",
    "region": "cluster; firms are nested in regions",
    "firm": "unit; fixed effect in the mean",
    "year": "period; fixed effect in the mean",
    "cohort": "classification with skewed frequencies and singletons; not in the mean",
    "y": "outcome; mean exposure * exp(index)",
    "y_pos": "positive outcome with the same mean; its log variance depends on treat",
    "y_iv": "outcome whose mean also depends on the endogenous d_endog",
    "y_sep": "y set to zero where sep is one: separation when sep is a regressor",
    "treat": "binary treatment, coefficient log(1.08)",
    "arm": "three treatment arms, 0 is control; coefficients log(1.05), log(1.15)",
    "dose": "continuous treatment on [0, 2], coefficient 0.1",
    "policy": "binary, constant within region and year, coefficient log(1.10)",
    "x_normal": "normal, correlation 0.6 with the firm effect, coefficient 0.3",
    "x_skewed": "standardized Gamma(2, 1), coefficient -0.2",
    "x_binary": "binary with mean 0.2, coefficient 0.25",
    "category": "categorical a, b, c; coefficients 0.2 (b) and -0.3 (c)",
    "exposure": "exposure; unit coefficient on its logarithm",
    "d_endog": "endogenous regressor of y_iv, structural coefficient 0.2",
    "z1": "valid instrument for d_endog",
    "z2": "valid instrument for d_endog",
    "z_invalid": "relevant instrument that violates the moment condition",
    "weight": "sampling weight, independent of everything",
    "freq": "integer frequency weight from 1 to 3",
    "x_miss": "x_skewed with 2 percent of its values missing",
    "x_dup": "twice x_normal: exactly collinear",
    "firm_size": "constant within firm; not in the mean",
    "sep": "binary with mean 0.05, independent of everything",
    "true_mean": "E(y | regressors, effects) = exposure * exp(index)",
    "structural_mean_iv": "exposure * exp(index + 0.2 d_endog)",
}
"""Role of each column, in the order of the columns of the frame."""


@dataclass(frozen=True)
class _Panel:
    """Positions, starting at zero, of the rows of a balanced firm-year panel."""

    regions: int
    firms: int
    years: int
    firm: IntArray
    year: IntArray
    region: IntArray
    cell: IntArray  # region-year cell

    @property
    def rows(self) -> int:
        return len(self.firm)


def _panel(regions: int, firms_per_region: int, years: int) -> _Panel:
    firms = regions * firms_per_region
    firm = np.repeat(np.arange(firms), years)
    year = np.tile(np.arange(years), firms)
    region = firm // firms_per_region
    return _Panel(regions, firms, years, firm, year, region, region * years + year)


def _mean_one(effect: FloatArray) -> FloatArray:
    """Shift log effects so that the average of their exponentials is one."""
    return effect - float(np.log(np.mean(np.exp(effect))))


def _effects(
    panel: _Panel, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Standardized firm draws, the firm and year effects of each row, and the shock.

    The shock is common to a region and year.  Its exponential has mean one.
    """
    firm_draw = rng.normal(size=panel.firms)
    firm_effect = _mean_one(_FIRM_SD * firm_draw)
    year_effect = _mean_one(rng.normal(0, _YEAR_SD, panel.years))
    shock = rng.normal(0, _SHOCK_SD, panel.regions * panel.years) - 0.5 * _SHOCK_SD**2
    return (
        firm_draw,
        firm_effect[panel.firm] + year_effect[panel.year],
        shock[panel.cell],
    )


def _regressors(
    panel: _Panel, firm_draw: FloatArray, rng: np.random.Generator
) -> Columns:
    n = panel.rows
    treat = rng.binomial(1, 0.5, n)
    noise = math.sqrt(1 - _FIRM_LOADING**2) * rng.normal(size=n)
    return {
        "treat": treat,
        "x_normal": _FIRM_LOADING * firm_draw[panel.firm] + noise,
        "x_skewed": (rng.gamma(2.0, 1.0, n) - 2.0) / math.sqrt(2.0),
        "x_binary": rng.binomial(1, 0.2, n),
        "category": rng.choice(np.array(["a", "b", "c"]), n, p=[0.5, 0.3, 0.2]),
        "dose": rng.uniform(0.0, 2.0, n),
        "arm": rng.integers(0, 3, n),
        "policy": rng.binomial(1, 0.5, panel.regions * panel.years)[panel.cell],
        "exposure": rng.uniform(0.5, 2.0, n),
    }


def _regression_index(c: Columns) -> FloatArray:
    """Sum of the regressors times their coefficients, without effects or exposure."""
    b = SHOWCASE_TRUTH
    index: FloatArray = (
        b["treat"] * c["treat"]
        + b["x_normal"] * c["x_normal"]
        + b["x_skewed"] * c["x_skewed"]
        + b["x_binary"] * c["x_binary"]
        + b["category[b]"] * (c["category"] == "b")
        + b["category[c]"] * (c["category"] == "c")
        + b["dose"] * c["dose"]
        + b["arm[1]"] * (c["arm"] == 1)
        + b["arm[2]"] * (c["arm"] == 2)
        + b["policy"] * c["policy"]
    )
    return index


def _instruments(n: int, rng: np.random.Generator) -> tuple[Columns, FloatArray]:
    """The endogenous regressor, its instruments, and the first-stage error."""
    z1, z2, v = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    pi1, pi2, pi_noise = _FIRST_STAGE
    columns = {
        "d_endog": pi1 * z1 + pi2 * z2 + v + pi_noise * rng.normal(size=n),
        "z1": z1,
        "z2": z2,
        "z_invalid": _INVALID_LOADING * v + rng.normal(size=n),
    }
    return columns, v


def _auxiliary(panel: _Panel, c: Columns, rng: np.random.Generator) -> Columns:
    """Columns for weights, missing values, collinearity, singletons, separation."""
    n = panel.rows
    weight = rng.uniform(0.5, 1.5, n)
    freq = rng.integers(1, 4, n)
    missing = rng.random(n) < 0.02
    firm_size = np.exp(rng.normal(size=panel.firms))[panel.firm]
    cohorts = max(n // 6, 2)
    frequency = (1.0 + np.arange(cohorts)) ** -1.25
    return {
        "weight": weight,
        "freq": freq,
        "x_miss": np.where(missing, np.nan, c["x_skewed"]),
        "x_dup": 2.0 * c["x_normal"],
        "firm_size": firm_size,
        "cohort": rng.choice(cohorts, n, p=frequency / frequency.sum()) + 1,
        "sep": rng.binomial(1, 0.05, n),
    }


def _notes() -> tuple[str, ...]:
    b = SHOWCASE_TRUTH
    pi1, pi2, pi_noise = _FIRST_STAGE
    first_stage_variance = pi1**2 + pi2**2 + 1.0 + pi_noise**2
    exogenous_limit = b["d_endog"] + _ENDOGENEITY / first_stage_variance
    return (
        "Pass exposure as the exposure of the model; its logarithm has a unit "
        "coefficient.",
        "Without firm effects the coefficient of x_normal converges to "
        f"{b['x_normal'] + _FIRM_SD * _FIRM_LOADING:.2f} as the number of firms grows.",
        "Outcomes of the same region and year share a lognormal shock with mean one "
        f"and log-scale standard deviation {_SHOCK_SD}; policy varies at that level, "
        "and region-by-year effects absorb both.",
        "y_pos: the coefficient of treat in the power mean of order omega is "
        "showcase_treat_coefficient(omega); the other coefficients do not depend on "
        "the order.",
        "y_iv: with U = y_iv / structural_mean_iv, E(U | z1, z2, exogenous regressors) "
        "= 1, so the multiplicative moment holds.  The first stage is linear with an "
        "error independent of the instruments, so the additive moment gives the same "
        f"slopes and an intercept higher by {b['d_endog'] * _ENDOGENEITY:.2f}.  "
        f"Treating d_endog as exogenous gives the coefficient {exogenous_limit:.4f} "
        "in the limit.",
        "z_invalid is correlated with the structural error: E(z_invalid (U - 1)) = "
        f"{_INVALID_LOADING * _ENDOGENEITY:.2f}.",
    )


def showcase_treat_coefficient(omega: float) -> float:
    """Coefficient of ``treat`` in the power mean of order ``omega`` of ``y_pos``.

    ``y_pos`` is lognormal with log-scale standard deviation 0.5 without treatment
    and 0.7 with it, so ``E(y_pos^omega | x)^(1/omega)`` is exponential in the index
    with the treatment coefficient ``log(1.08) + (omega - 1)(0.7^2 - 0.5^2)/2``.  At
    ``omega = 0`` this is the coefficient of the regression of ``log(y_pos)``.
    """
    gap = (_SD_TREATED**2 - _SD_UNTREATED**2) / 2
    return TREATMENT_COEFFICIENT + (omega - 1.0) * gap


def showcase_data(
    *,
    law: str = "nb2",
    regions: int = 40,
    firms_per_region: int = 5,
    years: int = 6,
    zeros: float = 0.0,
    seed: int = 1,
    draw: int = 0,
) -> Synthetic:
    """Draw the showcase panel: every firm of every region, observed every year.

    Parameters
    ----------
    law : str
        Outcome law of ``y`` and ``y_iv``: a key of ``SHOWCASE_LAWS``.
    regions, firms_per_region, years : int
        The panel has ``regions * firms_per_region`` firms and ``years`` rows per
        firm.
    zeros : float
        Share of ``y`` and ``y_iv`` set to zero, with the mean kept.
    seed, draw : int
        Each stream is seeded with ``SeedSequence([seed, draw, stream])``.

    Returns
    -------
    Synthetic
        ``columns`` gives the role of each column and ``notes`` the properties of
        the design that hold by construction.
    """
    require_design(SHOWCASE_LAWS, law, "showcase law")
    if min(regions, firms_per_region, years) < 2:
        raise ValueError("regions, firms_per_region and years must be at least 2")
    kind, parameters = SHOWCASE_LAWS[law]
    panel = _panel(regions, firms_per_region, years)

    def stream(name: str) -> np.random.Generator:
        return seeded([seed, draw, _STREAMS[name]])

    firm_draw, effects, shock = _effects(panel, stream("effects"))
    columns = _regressors(panel, firm_draw, stream("regressors"))
    log_mean = (
        _INTERCEPT + effects + _regression_index(columns) + np.log(columns["exposure"])
    )

    rng = stream("outcomes")
    y = draw_outcome(kind, parameters, log_mean + shock, np.exp(log_mean + shock), rng)
    sd = np.where(columns["treat"] == 1, _SD_TREATED, _SD_UNTREATED)
    y_pos = np.exp(log_mean + shock + sd * rng.normal(size=panel.rows) - 0.5 * sd**2)

    rng = stream("instruments")
    instruments, v = _instruments(panel.rows, rng)
    log_structural = log_mean + SHOWCASE_TRUTH["d_endog"] * instruments["d_endog"]
    log_mean_iv = log_structural + shock + _ENDOGENEITY * v - 0.5 * _ENDOGENEITY**2
    y_iv = draw_outcome(kind, parameters, log_mean_iv, np.exp(log_mean_iv), rng)

    columns |= instruments | _auxiliary(panel, columns, stream("auxiliary"))
    rng = stream("zeros")
    y = inflate_zeros(y, zeros, rng)
    columns |= {
        "state": panel.region // 5 + 1,
        "region": panel.region + 1,
        "firm": panel.firm + 1,
        "year": panel.year + 2001,
        "y": y,
        "y_pos": y_pos,
        "y_iv": inflate_zeros(y_iv, zeros, rng),
        "y_sep": y * (1 - columns["sep"]),
        "true_mean": np.exp(log_mean),
        "structural_mean_iv": np.exp(log_structural),
    }
    frame = pd.DataFrame({name: columns[name] for name in _ROLES})
    numeric = frame.drop(columns=["y", "category"]).to_numpy(dtype=float)
    return Synthetic(
        design=f"showcase/{law}",
        frame=frame,
        formula=(
            "y ~ treat + x_normal + x_skewed + x_binary + C(category) + dose + C(arm) "
            "+ policy | firm + year"
        ),
        truth=dict(SHOWCASE_TRUTH),
        columns=dict(_ROLES),
        seed=(
            f"SeedSequence([{seed}, {draw}, stream]); streams: effects 0, "
            "regressors 1, outcomes 2, instruments 3, auxiliary 4, zeros 5"
        ),
        source="",
        fingerprint=checksum(numeric, y),
        notes=_notes(),
    )
