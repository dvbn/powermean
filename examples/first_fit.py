"""Fit three-way PPML to a reproducible synthetic panel of directed trade flows."""

import numpy as np
import pandas as pd

import powermean as pm

FORMULA = "trade ~ 0 + agreement | pair + exporter^year + importer^year"


def gravity_data():
    """Return eight years of simulated trade among 12 countries."""
    rng = np.random.default_rng(20261118)
    countries = range(12)
    years = np.arange(8)
    exporter, importer = np.array(
        [(i, j) for i in countries for j in countries if i != j]
    ).T
    n_pairs = len(exporter)

    adoption_year = rng.integers(1, 12, size=n_pairs)
    agreement = years[None, :] >= adoption_year[:, None]
    pair_effect = rng.normal(0, 0.35, size=(n_pairs, 1))
    exporter_year = rng.normal(0, 0.25, size=(len(countries), len(years)))
    importer_year = rng.normal(0, 0.25, size=(len(countries), len(years)))
    mean = np.exp(
        1.2 + pair_effect + exporter_year[exporter] + importer_year[importer]
        + 0.35 * agreement
    )

    return pd.DataFrame(
        {
            "exporter": np.repeat(exporter, len(years)),
            "importer": np.repeat(importer, len(years)),
            "year": np.tile(2001 + years, n_pairs),
            "pair": np.repeat(np.arange(n_pairs), len(years)),
            "agreement": agreement.astype(int).ravel(),
            "trade": rng.poisson(mean).ravel(),
        }
    )


def main():
    data = gravity_data()
    result = pm.fit(
        FORMULA, data, omega=1, tau=1, cancel=None, bias_correction=False,
        vcov={"cluster": "pair"},
    )
    print(result.summary())
    print(pm.etable(
        result, names=("Gravity PPML",), scale="effect", uncertainty="ci",
        stats=("nobs", "zeros"), stars=False, digits=1,
    ))


if __name__ == "__main__":
    main()
