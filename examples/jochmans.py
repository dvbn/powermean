"""Fit Jochmans GMM1 and GMM2 on complete panels and directed trade links."""

import numpy as np
import pandas as pd

import powermean as pm


def trade_panel(n_rows: int, n_columns: int, *, directed: bool) -> pd.DataFrame:
    """Generate trade in levels with two origin/destination effects and some zeros."""
    rng = np.random.default_rng(82)
    i, j = np.indices((n_rows, n_columns))
    observed = i != j if directed else np.ones_like(i, dtype=bool)
    i, j = i[observed], j[observed]
    cost = rng.normal(size=len(i))
    agreement = rng.binomial(1, 0.4, size=len(i))
    exporter = rng.normal(0, 0.3, n_rows)
    importer = rng.normal(0, 0.3, n_columns)
    mean = np.exp(0.85 - 0.27 * cost + 0.18 * agreement + exporter[i] + importer[j])
    return pd.DataFrame(
        {
            "trade": rng.poisson(mean),
            "cost": cost,
            "agreement": agreement,
            "origin": [f"port-{k:02d}" for k in i],
            "destination": [f"port-{k:02d}" for k in j],
        }
    )


def main() -> None:
    """Fit a complete firm-by-year panel and a cross-section of trade flows."""
    firm_year = trade_panel(14, 20, directed=False).rename(
        columns={
            "trade": "patents",
            "cost": "log_rd",
            "agreement": "grant",
            "origin": "firm",
            "destination": "year",
        }
    )
    firm_year["year"] = 2000 + pd.factorize(firm_year["year"])[0]
    trade = trade_panel(20, 20, directed=True)
    designs = (
        (
            "rectangular",
            "patents ~ 0 + log_rd + grant | firm + year",
            firm_year,
            "patents",
            "grant",
        ),
        (
            "directed",
            "trade ~ 0 + cost + agreement | origin + destination",
            trade,
            "trade",
            "agreement",
        ),
    )
    for layout, formula, data, outcome, policy in designs:
        gmm1 = pm.fit(formula, data, estimator="gmm1", fe_layout=layout)
        gmm2 = pm.fit(
            formula,
            data,
            fe_layout=layout,
            estimator="gmm2",
            start=gmm1.coef(),
        )
        print(f"{layout} ({gmm1.sample.count} cells, {gmm1.sample.zeros} zeros)")
        print(pm.etable(gmm1, gmm2, uncertainty="ci"))
        retained = data.iloc[gmm2.sample.positions]
        fitted = gmm2.predict()
        residuals = gmm2.predict(type="residual")
        effects = gmm2.predict(type="fixed_effects")
        no_policy = gmm2.predict(retained.assign(**{policy: 0}))
        print("\nGMM2 predictions with slopes and recovered effects held fixed")
        print(pd.DataFrame({
            "Observed": retained[outcome],
            "Fitted": fitted,
            f"Without {policy}": no_policy,
            "Residual": residuals,
        }).head().round(3).to_string(index=False))
        print("\nFixed-effect levels recovered by dimension")
        print({name: len(levels) for name, levels in effects.items()})


if __name__ == "__main__":
    main()
