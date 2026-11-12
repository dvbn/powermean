"""The 20 designs without fixed effects of the solver study.

Every design has ``E(y | x) = exp(x'b)`` with an intercept, a binary treatment of
mean one half and coefficient ``log(1.08)``, and ``k - 2`` other regressors; the
intercept is analytical and sets the population mean of untreated observations to
two.

========  =====================================================================
Designs   Regressors and outcome law
========  =====================================================================
1 to 10   standard normal regressors; Poisson, Gamma, three negative binomials,
          compound Poisson, inverse Gaussian, heterogeneous negative binomial,
          lognormal, lognormal with half of the outcomes at zero
11 to 16  menus A (binary and normal), B (skewed) and C (A with a rare binary
          regressor of coefficient ``-log(20)``); two negative binomials
17 to 20  stress: dispersed means, heavy lognormal, rare large outcomes, nearly
          collinear regressors
========  =====================================================================

Designs 11 to 16 use the built-in ``sum``, whose floating-point algorithm changed
in Python 3.12: with 100 regressors their intercept differs in its last bits between
Python 3.10 or 3.11 and later versions.  The archive of the study was produced with
Python 3.14.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermean.testing.synthetic import (
    CONTAMINATION,
    TREATMENT_COEFFICIENT,
    FloatArray,
    IntArray,
    Synthetic,
    checksum,
    describe_law,
    draw_outcome,
    inflate_zeros,
    require_design,
    seeded,
)

SOLVER_SOURCE = (
    "solver study of Benatia, Bellego and Pape: designs without fixed effects"
)
SOLVER_SEED = 2026100301
"""Master seed of the reporting runs of the study (split ``"test"``)."""
SOLVER_SPLITS = {"development": 1, "validation": 2, "test": 3}
_STREAMS = {
    "covariates": 0,
    "outcome": 1,
    "zeros": 3,
}  # 2 orders the methods in the study


@dataclass(frozen=True)
class SolverDesign:
    """One design of the solver study."""

    number: int
    covariates: str
    law: str
    parameters: Mapping[str, float] = field(default_factory=dict)
    menu: str = ""

    def describe(self) -> str:
        """One line for listings: regressors, outcome law and parameters."""
        regressors = {
            "cp11_gaussian": "normal regressors",
            "cp8_menu": f"regressor menu {self.menu}",
            "dmc_gaussian": "normal regressors",
            "dmc_equicorrelated": "nearly collinear normal regressors",
        }[self.covariates]
        return f"{regressors}; {describe_law(self.law, self.parameters)}"


SOLVER_DESIGNS: dict[str, SolverDesign] = {
    "G_poisson": SolverDesign(1, "cp11_gaussian", "poisson"),
    "G_gamma": SolverDesign(2, "cp11_gaussian", "gamma"),
    "G_nb2_mild": SolverDesign(3, "cp11_gaussian", "nb2", {"size": 5.0}),
    "G_nb2": SolverDesign(4, "cp11_gaussian", "nb2", {"size": 1.0}),
    "G_nb2_strong": SolverDesign(5, "cp11_gaussian", "nb2", {"size": 0.2}),
    "G_tweedie15": SolverDesign(6, "cp11_gaussian", "tweedie15"),
    "G_ig3": SolverDesign(7, "cp11_gaussian", "ig3"),
    "G_heterogeneous_nb2": SolverDesign(
        8, "cp11_gaussian", "heterogeneous_nb2", {"size_h1": 1 / 3, "size_h0": 1.0}
    ),
    "G_lognormal": SolverDesign(9, "cp11_gaussian", "lognormal_cp11"),
    "G_zero_lognormal": SolverDesign(10, "cp11_gaussian", "zero_lognormal_cp11"),
    "A_nb2": SolverDesign(11, "cp8_menu", "nb2", {"size": 1.0}, menu="A"),
    "A_nb2_strong": SolverDesign(12, "cp8_menu", "nb2", {"size": 0.2}, menu="A"),
    "B_nb2": SolverDesign(13, "cp8_menu", "nb2", {"size": 1.0}, menu="B"),
    "B_nb2_strong": SolverDesign(14, "cp8_menu", "nb2", {"size": 0.2}, menu="B"),
    "C_nb2": SolverDesign(15, "cp8_menu", "nb2", {"size": 1.0}, menu="C"),
    "C_nb2_strong": SolverDesign(16, "cp8_menu", "nb2", {"size": 0.2}, menu="C"),
    "S1_dispersion25_nb2_strong": SolverDesign(
        17, "dmc_gaussian", "nb2", {"index_sd": 2.5, "size": 0.2}
    ),
    "S2_dispersion25_lognormal25": SolverDesign(
        18, "dmc_gaussian", "lognormal_dmc", {"index_sd": 2.5, "residual_sd": 2.5}
    ),
    "S3_contamination_p001_x500": SolverDesign(
        19, "dmc_gaussian", "contamination", CONTAMINATION
    ),
    "S4_equicorrelated999_lognormal": SolverDesign(
        20,
        "dmc_equicorrelated",
        "lognormal_dmc",
        {"index_sd": 1.5, "correlation": 0.999, "residual_sd": 1.0},
    ),
}


def _logaddexp(a: float, b: float) -> float:
    """``log(exp(a) + exp(b))`` (``_logaddexp``)."""
    if a == b:
        return a + math.log(2.0)
    high, low = (a, b) if a > b else (b, a)
    return high + math.log1p(math.exp(low - high))


def _menu_features(menu: str, k: int) -> tuple[tuple[str, float], ...]:
    """Kind and probability of each nuisance regressor.

    The probability is ``nan`` for the kinds that have none.
    """
    features = [("rare_bernoulli", 0.02)] if menu == "C" else []
    remaining = k - 2 - len(features)
    binary = (remaining + 1) // 2
    for j in range(remaining):
        if menu == "B":
            features.append(("std_gamma", math.nan))
        elif j < binary:
            features.append(("std_bernoulli", (0.05, 0.2, 0.5)[j % 3]))
        else:
            features.append(("normal", math.nan))
    return tuple(features)


def _menu_log_mgf(kind: str, p: float, t: float) -> float:
    """Exact ``log E exp(t W)`` of one nuisance regressor (``_cp8_log_mgf``)."""
    if kind == "normal":
        return t * t / 2
    if kind == "std_gamma":
        if t >= math.sqrt(2.0):
            raise ValueError("the standardized Gamma requires t < sqrt(2)")
        return -math.sqrt(2.0) * t - 2 * math.log1p(-t / math.sqrt(2.0))
    scale = math.sqrt(p * (1 - p)) if kind == "std_bernoulli" else 1.0
    shift = -t * p / scale if kind == "std_bernoulli" else 0.0
    return float(shift + _logaddexp(math.log1p(-p), math.log(p) + t / scale))


def _solver_coefficients(design: SolverDesign, k: int) -> list[float]:
    """Intercept, treatment coefficient and slopes (``coefficients``).

    The intercept is analytical: it sets the population mean of untreated
    observations to two.
    """
    p = design.parameters
    if design.covariates == "cp11_gaussian":
        slopes = [0.6 * (-1.0) ** j / math.sqrt(k - 2) for j in range(k - 2)]
        alpha = math.log(2) - 0.18
    elif design.covariates == "cp8_menu":
        features = _menu_features(design.menu, k)
        extra = 1 if design.menu == "C" else 0
        count = len(features) - extra
        slopes = [0.6 * (-1 if j % 2 else 1) / math.sqrt(count) for j in range(count)]
        if extra:
            slopes.insert(0, -math.log(20.0))
        log_mgf = sum(
            _menu_log_mgf(kind, q, b)
            for (kind, q), b in zip(features, slopes, strict=True)
        )
        alpha = math.log(2.0) - log_mgf
    elif design.covariates in ("dmc_gaussian", "dmc_equicorrelated"):
        sigma = p["index_sd"]
        slopes = [(-1.0) ** j for j in range(k - 2)]
        if design.covariates == "dmc_equicorrelated":
            rho = p["correlation"]
            variance = (1 - rho) * sum(b * b for b in slopes) + rho * sum(slopes) ** 2
        else:
            variance = sum(b * b for b in slopes)
        slopes = [b * (sigma / math.sqrt(variance)) for b in slopes]
        alpha = math.log(2.0) - 0.5 * sigma**2
    else:
        raise ValueError(f"unknown covariate design {design.covariates!r}")
    return [alpha, TREATMENT_COEFFICIENT, *slopes]


def _solver_covariates(
    design: SolverDesign, n: int, k: int, rng: np.random.Generator
) -> tuple[FloatArray, IntArray | None]:
    """Regressors with their intercept column, and the heterogeneity indicator."""
    if design.covariates == "cp11_gaussian":
        columns = (np.ones(n), rng.binomial(1, 0.5, n), rng.normal(size=(n, k - 2)))
        x = np.asarray(np.column_stack(columns), dtype=np.float64)
        return x, (x[:, 2] > 0).astype(np.int64)
    if design.covariates == "cp8_menu":
        x = np.empty((n, k), dtype=float)
        x[:, 0] = 1.0
        x[:, 1] = rng.binomial(1, 0.5, size=n)
        for column, (kind, q) in enumerate(_menu_features(design.menu, k), start=2):
            if kind == "rare_bernoulli":
                x[:, column] = rng.binomial(1, q, size=n)
            elif kind == "std_bernoulli":
                x[:, column] = (rng.binomial(1, q, size=n) - q) / math.sqrt(q * (1 - q))
            elif kind == "std_gamma":
                x[:, column] = (rng.gamma(2.0, 1.0, size=n) - 2.0) / math.sqrt(2.0)
            else:
                x[:, column] = rng.normal(size=n)
        return x, None
    treatment = rng.binomial(1, 0.5, n)
    z = rng.normal(size=(n, k - 2))
    if design.covariates == "dmc_equicorrelated":
        rho = design.parameters["correlation"]
        z = math.sqrt(rho) * rng.normal(size=(n, 1)) + math.sqrt(1 - rho) * z
    columns = (np.ones(n), treatment, z)
    return np.asarray(np.column_stack(columns), dtype=np.float64), None


def _solver_roles(design: SolverDesign, k: int) -> dict[str, str]:
    kinds = {
        "rare_bernoulli": "binary with mean {q}",
        "std_bernoulli": "standardized binary with mean {q}",
        "std_gamma": "standardized Gamma(2, 1)",
        "normal": "standard normal",
    }
    if design.covariates == "cp8_menu":
        features = _menu_features(design.menu, k)
        nuisances = [kinds[kind].format(q=q) for kind, q in features]
    elif design.covariates == "dmc_equicorrelated":
        rho = design.parameters["correlation"]
        nuisances = [f"standard normal, pairwise correlation {rho}"] * (k - 2)
    else:
        nuisances = ["standard normal"] * (k - 2)
    roles = {
        "y": "outcome",
        "treat": "binary treatment with mean 1/2, coefficient log(1.08)",
    }
    roles.update({f"x{j}": role for j, role in enumerate(nuisances, start=1)})
    roles["true_mean"] = "E(y | regressors)"
    return roles


def solver_data(
    design: str,
    *,
    n: int = 1000,
    k: int = 10,
    zeros: float = 0.0,
    seed: int = SOLVER_SEED,
    draw: int = 0,
    split: str = "test",
) -> Synthetic:
    """One dataset of a design of the solver study.

    The model is ``E(y | x) = exp(x'b)`` with an intercept, a binary treatment with
    coefficient ``log(1.08)`` and ``k - 2`` other regressors.  With the default
    ``seed`` and ``split``, sizes ``(n, k)`` of ``(1000, 10)`` or ``(10000, 100)``
    and ``draw`` below 1000, the result is a dataset of the study.

    Parameters
    ----------
    design : str
        A key of ``SOLVER_DESIGNS``.
    n, k : int
        Observations and regressors including the intercept; ``k`` is at least 4.
    zeros : float
        Share of outcomes set to zero, with the mean kept.
    seed, draw : int
        Master seed and replication number.
    split : str
        ``"development"``, ``"validation"`` or ``"test"``.  Each stream is seeded
        with ``SeedSequence([seed, split code, design number, n, k, draw, stream])``.

    Returns
    -------
    Synthetic
        Columns ``y``, ``treat``, ``x1`` to ``x{k-2}`` and ``true_mean``.
    """
    require_design(SOLVER_DESIGNS, design, "solver")
    if n < 1 or k < 4:
        raise ValueError("solver designs need n >= 1 and k >= 4")
    if split not in SOLVER_SPLITS:
        raise ValueError(f"split must be one of {', '.join(SOLVER_SPLITS)}")
    spec = SOLVER_DESIGNS[design]
    key = [seed, SOLVER_SPLITS[split], spec.number, n, k, draw]

    x, h = _solver_covariates(spec, n, k, seeded([*key, _STREAMS["covariates"]]))
    beta = np.asarray(_solver_coefficients(spec, k), dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        index = x @ beta
        mean = np.exp(index)
    outcome_rng = seeded([*key, _STREAMS["outcome"]])
    y = draw_outcome(spec.law, spec.parameters, index, mean, outcome_rng, h=h)
    y = inflate_zeros(y, zeros, seeded([*key, _STREAMS["zeros"]]))

    names = ["treat", *(f"x{j}" for j in range(1, k - 1))]
    frame = pd.DataFrame(x[:, 1:], columns=names)
    frame["treat"] = frame["treat"].astype(int)
    frame.insert(0, "y", y)
    frame["true_mean"] = mean
    return Synthetic(
        design=f"solver/{design}",
        frame=frame,
        formula="y ~ " + " + ".join(names),
        truth=dict(zip(["Intercept", *names], beta.tolist(), strict=True)),
        columns=_solver_roles(spec, k),
        seed=(
            f"SeedSequence([{seed}, {SOLVER_SPLITS[split]}, {spec.number}, {n}, {k}, "
            f"{draw}, stream]); streams: covariates 0, outcome 1, zeros 3"
        ),
        source=SOLVER_SOURCE,
        fingerprint=checksum(x, y),
        notes=("The population mean of untreated observations is two.",),
    )
