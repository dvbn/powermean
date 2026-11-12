"""Shared data for the tests: simulated samples with known structure."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def counts() -> pd.DataFrame:
    """Poisson outcomes with zeros, two regressors, a binary regressor and clusters."""
    rng = np.random.default_rng(20261110)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.4).astype(float)
    mean = np.exp(0.2 + 0.3 * x1 - 0.2 * x2 + 0.25 * d)
    return pd.DataFrame(
        {
            "y": rng.poisson(mean).astype(float),
            "x1": x1,
            "x2": x2,
            "d": d,
            "firm": np.repeat(np.arange(80), 5),
            "group": rng.choice(["a", "b", "c"], size=n),
        }
    )


@pytest.fixture(scope="session")
def positive(counts: pd.DataFrame) -> pd.DataFrame:
    """Strictly positive outcomes on the same regressors."""
    rng = np.random.default_rng(7)
    mean = np.exp(0.2 + 0.3 * counts["x1"] - 0.2 * counts["x2"] + 0.25 * counts["d"])
    return counts.assign(y=rng.gamma(shape=2.0, scale=mean / 2.0) + 1e-3)


def design(data: pd.DataFrame, names: list[str]) -> np.ndarray:
    """The regressor matrix with an intercept first."""
    return np.column_stack([np.ones(len(data))] + [data[name].to_numpy() for name in names])
