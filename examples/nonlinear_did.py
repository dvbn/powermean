"""Estimate proportional and level effects in two synthetic panel DiD designs."""

import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy.stats import norm

import powermean as pm

TWFE = "visits ~ 0 + treated | unit + period"
CELLS = {(2, 2): "g2_t2", (2, 3): "g2_t3", (2, 4): "g2_t4",
         (3, 3): "g3_t3", (3, 4): "g3_t4"}


def did_data(*, staggered=False, with_x=False):
    """Simulate nonnegative visits with common untreated growth across cohorts."""
    rng = np.random.default_rng(20261120)
    periods = np.arange(1, 5 if staggered else 3)
    cohorts = (0, 2, 3) if staggered else (0, 2)
    n_per_cohort = 80
    cohort = np.repeat(cohorts, n_per_cohort)
    unit = np.arange(len(cohort))
    pretreatment_scale = rng.uniform(0.6, 1.4, size=len(unit)) + 0.3 * cohort
    untreated_mean = 2.1 * pretreatment_scale[:, None] * 1.15 ** (periods - 1)
    treated = (cohort[:, None] > 0) & (periods >= cohort[:, None])
    if with_x:
        # Draw the entire covariate path independently of adoption and outcome shocks.
        x = np.random.default_rng(20261121).uniform(-1, 1, size=treated.shape)
        untreated_mean = untreated_mean * np.exp(0.35 * x)
    if staggered:
        multipliers = np.ones(treated.shape)
        for (g, t), effect in zip(CELLS, [1.15, 1.35, 1.50, 1.20, 1.40], strict=True):
            multipliers[(cohort == g)[:, None] & (periods == t)] = effect
    else:
        multipliers = np.where(treated, 1.40, 1.0)
    data = pd.DataFrame(
        {
            "visits": rng.poisson(untreated_mean * multipliers).ravel(),
            "unit": np.repeat(unit, len(periods)),
            "cohort": np.repeat(cohort, len(periods)),
            "period": np.tile(periods, len(unit)),
            "treated": treated.astype(int).ravel(),
        }
    )
    if with_x:
        data["x"] = x.ravel()
    if staggered:
        for (g, t), term in CELLS.items():
            data[term] = ((data["cohort"] == g) & (data["period"] == t)).astype(int)
    return data


def comparison_rows(fits, panel, terms):
    """Display each percentage effect above its unit-clustered SE."""
    first = next(iter(fits.values()))
    for fit in fits.values():
        np.testing.assert_array_equal(fit.sample.positions, first.sample.positions)
        assert fit.sample.labels.equals(panel.iloc[fit.sample.positions].index)
    table = pm.etable(
        *fits.values(), names=tuple(fits), keep=terms, scale="effect",
        uncertainty="se", stars=False, digits=2,
    )
    # Each requested term occupies two rows before the model statistics.
    rows = table.to_frame(formatted=True).head(2 * len(terms))
    return rows.rename(columns={"": "term"})[["term", *fits]]


def treated_visit_effects(pooled, panel, formula):
    """Report effects and conditional-on-design clustered delta-method inference."""
    retained = panel.iloc[pooled.sample.positions]
    assert retained.index.equals(pooled.sample.labels)
    treated = retained.loc[retained["treated"] == 1]
    no_policy = treated.copy()
    no_policy[list(CELLS.values())] = 0
    observed = pooled.predict(treated)
    untreated = pooled.predict(no_policy)
    assert observed.index.equals(treated.index) and untreated.index.equals(treated.index)
    fitted = pooled.predict()
    assert fitted.index.equals(retained.index)
    np.testing.assert_allclose(observed, fitted.loc[treated.index])

    # Replay the same RHS on both prediction frames in fitted coefficient order.
    names = pooled.coef().index
    design = model_matrix(formula.split("~", 1)[1], retained).model_spec
    x1 = design.get_model_matrix(treated).loc[:, names].to_numpy()
    x0 = design.get_model_matrix(no_policy).loc[:, names].to_numpy()
    d1 = observed.to_numpy()[:, None] * x1
    d0 = untreated.to_numpy()[:, None] * x0
    delta = d1 - d0
    extra = observed - untreated
    groups = treated.groupby(["cohort", "period"]).indices
    counts = [len(idx) for idx in groups.values()]
    cell_att = [extra.iloc[idx].mean() for idx in groups.values()]
    att = extra.mean()
    np.testing.assert_allclose(att, np.average(cell_att, weights=counts))
    ratio = observed.mean() / untreated.mean()
    gradients = np.vstack([
        delta.mean(axis=0),
        100 * (d1.mean(axis=0) - ratio * d0.mean(axis=0)) / untreated.mean(),
        *(delta[idx].mean(axis=0) for idx in groups.values()),
    ])
    covariance = pooled.vcov().loc[names, names].to_numpy()
    se = np.sqrt(np.sum((gradients @ covariance) * gradients, axis=1))
    assert np.isfinite(se).all()
    estimates = np.array([att, 100 * (ratio - 1), *cell_att])
    half_width = norm.ppf(0.975) * se
    inference = pd.DataFrame({
        "Estimate": estimates,
        "Clustered SE": se,
        "95% CI lower": estimates - half_width,
        "95% CI upper": estimates + half_width,
    })
    overall = inference.iloc[:2].copy()
    overall.index = pd.Index(["Extra visits per treated unit-period", "Increase in visits (%)"],
                             name="Effect")
    cells = inference.iloc[2:].copy()
    cells.index = pd.MultiIndex.from_tuples(groups, names=["Cohort", "Period"])
    cells.insert(0, "Treated rows", counts)
    return overall, cells


def _print_effect_table(table):
    display = table.reset_index()
    lower = display.pop("95% CI lower")
    upper = display.pop("95% CI upper")
    display["95% CI"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(lower, upper, strict=True)]
    print(display.to_string(index=False, float_format="{:.3f}".format))


def main():
    data = did_data()
    cases = (
        (1, None, "Joint PPML"),
        (1, "unit", "Cancelled PPML"),
        (1.5, "unit", "Cancelled 1.5"),
        (2, "unit", "Cancelled Gamma"),
        (2, None, "Joint Gamma"),
    )
    fits = [
        pm.fit(TWFE, data, omega=1, tau=tau, cancel=cancel, cluster="unit")
        for tau, cancel, _ in cases
    ]
    print(pm.etable(
        *fits, names=[name for _, _, name in cases],
        stats=("nobs", "zeros"), stars=False, digits=4,
    ))

    data = did_data(staggered=True)
    rhs = " + ".join(CELLS.values())
    pooled = pm.fit(
        f"visits ~ 1 + C(cohort) + C(period) + {rhs}",
        data, omega=1, tau=1, cluster="unit",
    )
    within = pm.fit(
        f"visits ~ 0 + {rhs} | unit + period",
        data, omega=1, tau=1, cluster="unit", cancel="unit",
    )
    gamma = pm.fit(
        f"visits ~ 0 + {rhs} | unit + period",
        data, omega=1, tau=2, cluster="unit", cancel="unit",
    )
    print(pm.etable(
        pooled, within, gamma,
        names=("Pooled PPML", "Cancelled PPML", "Cancelled Gamma"),
        keep=tuple(CELLS.values()),
        labels={term: f"Cohort {g}, period {t}" for (g, t), term in CELLS.items()},
        scale="effect", uncertainty="ci", stars=False, digits=1,
    ))

    methods = {"Joint PPML": (1, None), "Cancelled 1.5": (1.5, "unit"),
               "Cancelled Gamma": (2, "unit")}
    two_period = did_data(with_x=True)
    two_fits = {
        name: pm.fit("visits ~ 0 + treated + x | unit + period", two_period,
                     omega=1, tau=tau, cancel=cancel, cluster="unit")
        for name, (tau, cancel) in methods.items()
    }
    print(f"Two periods with X: {two_fits['Joint PPML'].sample.count} retained rows")
    print(comparison_rows(two_fits, two_period, ["treated"]).to_string(index=False))

    staggered = did_data(staggered=True, with_x=True)
    adjusted_formula = f"visits ~ 1 + C(cohort) + C(period) + {rhs} + x"
    adjusted = pm.fit(adjusted_formula, staggered, omega=1, tau=1, cluster="unit")
    fits_x = {"Pooled PPML": adjusted}
    for name, (tau, cancel) in methods.items():
        fits_x[name] = pm.fit(f"visits ~ 0 + {rhs} + x | unit + period", staggered,
                             omega=1, tau=tau, cancel=cancel, cluster="unit")
    print(f"Staggered with X: {adjusted.sample.count} retained rows")
    print(comparison_rows(fits_x, staggered, list(CELLS.values())).to_string(index=False))

    overall, cells = treated_visit_effects(adjusted, staggered, adjusted_formula)
    print(f"Overall effects across {cells['Treated rows'].sum()} treated unit-periods")
    _print_effect_table(overall)
    print("\nExtra visits per treated unit-period, by cohort and period")
    _print_effect_table(cells)
    print("Unit-clustered SEs and pointwise 95% intervals condition on the observed treated rows.")


if __name__ == "__main__":
    main()
