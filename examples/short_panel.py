"""Compare joint and firm-cancelled Gamma scores in a synthetic short panel."""

import numpy as np
import pandas as pd

import powermean as pm

FORMULA = "sales ~ 0 + price_index | firm + year"


def firm_data():
    """Return four observations per firm, with persistent differences in sales."""
    rng = np.random.default_rng(20261119)
    n_firms = 96
    years = np.arange(4)
    firm_scale = rng.normal(0, 0.50, size=(n_firms, 1))
    price_index = 0.4 * firm_scale + rng.uniform(-1, 1, (n_firms, len(years)))
    mean = np.exp(1.4 + firm_scale + 0.1 * years - 0.3 * price_index)
    return pd.DataFrame(
        {
            "firm": np.repeat(np.arange(n_firms), len(years)),
            "year": np.tile(2020 + years, n_firms),
            "price_index": price_index.ravel(),
            "sales": rng.poisson(mean).ravel(),
        }
    )


def main():
    data = firm_data()
    common = {"omega": 1, "tau": 2, "cluster": "firm"}
    joint = pm.fit(FORMULA, data, **common)
    cancelled = pm.fit(FORMULA, data, cancel="firm", **common)
    corrected = pm.fit(
        FORMULA, data, cancel="firm", bias_correction="short_group", **common
    )
    print(pm.etable(
        joint, cancelled, names=("Joint", "Firm cancelled"),
        uncertainty="ci", stats=("nobs", "zeros"), stars=False, digits=4,
    ))
    print(corrected.summary())


if __name__ == "__main__":
    main()
