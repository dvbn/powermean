# Estimate a trade agreement slope in a gravity model

How much more do two countries trade in years when they have an agreement?
The example follows pairs of countries over time, including years with zero
trade. It compares the same pair with and without an agreement while accounting
for conditions in each country and year. The model for expected trade is

$$
E[Y_{ijt}\mid X,\alpha,\gamma,\delta]
=\exp(\beta D_{ijt}+\alpha_{ij}+\gamma_{it}+\delta_{jt}).
$$

Here $D_{ijt}$ is the agreement indicator, $X$ contains the panel regressors,
and the three fixed-effect terms represent a directed pair, an exporter-year
and an importer-year. The target is the conditional arithmetic mean of trade.
For a change in $D$ from zero to one with the fixed effects held constant,
the model gives the ratio $\exp(\beta)$.

## Create the trade panel

After [installing Powermean](install.md), run the following in a Python
session. It generates all the data used below. The same example is also
available as a [standalone script](../examples/first_fit.py).

```python
import numpy as np
import pandas as pd
import powermean as pm

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

trade = pd.DataFrame(
    {
        "exporter": np.repeat(exporter, len(years)),
        "importer": np.repeat(importer, len(years)),
        "year": np.tile(2001 + years, n_pairs),
        "pair": np.repeat(np.arange(n_pairs), len(years)),
        "agreement": agreement.astype(int).ravel(),
        "trade": rng.poisson(mean).ravel(),
    }
)
print(pm.summary_statistics(trade, vars=("trade",), probs=(0, 0.5, 1)))
```

```text
Summary statistics                                      N = 1,056 rows
----------------------------------------------------------------------
           N  missing  mean    sd    p0   p50   p100  zeros  mean | >0
----------------------------------------------------------------------
trade  1,056        0  4.08  3.25  0.00  3.00  37.00   6.0%       4.33
----------------------------------------------------------------------
```

The simulation has 12 countries, all 132 directed foreign pairs, and eight
years for each pair. There is no domestic trade in this design. Agreements
start at different dates, with some pairs untreated throughout. Adoption
dates are drawn independently by directed pair for this exercise. Real
agreements are often reciprocal, in which case the appropriate clustering
may differ. Trade is drawn in levels from a Poisson distribution with
pair-specific baselines, country-year shifts and an agreement slope of 0.35.
This makes the examples reproducible and shows how to handle genuine zeros.

## Fit with three fixed-effect blocks

```python
formula = "trade ~ 0 + agreement | pair + exporter^year + importer^year"
fit = pm.fit(
    formula, trade, omega=1, tau=1, cancel=None, bias_correction=False,
    vcov={"cluster": "pair"},
)
print(fit.summary())
```

```text
powermean
====================================================================================
PPML (omega = 1, tau = 1)
Fixed effects: pair (estimated), exporter^year (estimated), importer^year (estimated)
Observations: 1056 of 1056 input rows (none dropped)
Status: verified; certified: True
Coverage: unique_root (BBP prop:iols-exact-sample-root)
------------------------------------------------------------------------------------
                  coef.   std. err.        z    P>|z|    [95% conf. interval]
------------------------------------------------------------------------------------
agreement        0.4315      0.0563     7.66    0.000      0.3210      0.5419
------------------------------------------------------------------------------------
Fixed-effect block counts (fitted rows, active covariance)
  pair: levels=132, redundant=132 (exact), nested in=pair
  exporter^year: levels=96, redundant=1 (exact), nested in=none
  importer^year: levels=96, redundant=8 (exact), nested in=none
Nested blocks count as fully redundant in these block counts.
```

The console summary shows the estimator, the three jointly estimated blocks,
the number of usable observations and the numerical status. The agreement
row is on the log mean scale.

- `omega=1` models the arithmetic mean. `tau=1` specifies the power working
  variance $V(m)=m$ and gives PPML. The PPML moment needs a correctly specified
  mean for its usual interpretation. Its use extends to nonnegative outcomes
  measured in continuous units.
- Terms after `|` are fixed-effect blocks. `pair` identifies a *directed*
  exporter–importer pair. `^` creates one interaction block per term.
  `0 +` omits a separate intercept because the effect indicators already
  span a constant. The fit estimates the blocks jointly through absorption.
- The covariance sums score contributions within each of the 132 pairs before
  combining them. That matches independent pair histories and allows arbitrary
  dependence within a history under the pair-sandwich assumptions. Shared
  shocks across pairs require a separate sampling argument.

All 63 zero flows contribute to the fit. In other data, some patterns of
zeros can imply separation and change the estimation sample. Inspect
`fit.sample` and `fit.summary()` for the actual sample and diagnostics.
[Prepare your data](help/prepare-data.md)

## Identify the slope

Pair effects absorb permanent bilateral differences such as distance.
Exporter-year effects absorb shifts common to all partners of an exporter
in a year, and importer-year effects do the analogous job for importers.
The agreement slope uses within-pair changes in policy that remain after
both country-year blocks are accounted for. A policy that moves identically
for all destinations of an exporter in each year would be absorbed by the
exporter-year block, even if it changes within pairs.

These controls identify a coefficient of the maintained mean model when
enough variation remains. A causal trade-policy interpretation in an
observational panel also needs assumptions about policy timing and other
time-varying bilateral shocks. The model holds exporter-year and importer-year
effects fixed when computing this comparison. A counterfactual in which those
effects respond to policy requires an additional model.

## Report a percentage and interval

Ask `pm.etable` to report the coefficient as a percentage with the
transformed confidence interval.

```python
table = pm.etable(
    fit, names=("Gravity PPML",), scale="effect", uncertainty="ci",
    stats=("nobs", "zeros"), stars=False, digits=1,
)
print(table)
```

```text
------------------------------------
                        Gravity PPML
------------------------------------
agreement                   53.9
                        [37.9, 71.9]
------------------------------------
Estimator                   PPML
Mean order (omega)           1
pair effects             Estimated
exporter^year effects    Estimated
importer^year effects    Estimated
Covariance             cluster: pair
Convention                 stata
Reference                  normal
Clusters                    132
------------------------------------
Observations               1,056
Zeros                       6.0%
------------------------------------
Entries are 100 (exp(b) - 1) for each reported coefficient coordinate; not a standalone regressor change.
95% confidence intervals in brackets, transformed from the coefficient interval.
```

With the same fixed effects, the fitted conditional mean with an agreement
is 53.9% higher. The 95% interval is pointwise. In a Python notebook,
display `table` as the last expression for an HTML table. Use
`table.to_latex()` for a LaTeX table or `fit.tidy()` for the unrounded
coefficients and uncertainty as a DataFrame. For a named regressor
comparison on the proportion scale, use `fit.effect("agreement", contrast=(0, 1))`.

## Compare with the raw outcome means

```python
outcomes = pm.outcome_table(
    trade, outcomes="trade", treatment="agreement", cluster="pair",
)
print(outcomes)
```

```text
Outcome table                                                          control: agreement = 0 (N = 721); agreement = 1 (N = 335)
--------------------------------------------------------------------------------------------------------------------------------
       target   zeros (0)  zeros (1)  mean (0)  mean (1)  mean | >0 (0)  mean | >0 (1)  difference, unadjusted  [95% conf. int.]
--------------------------------------------------------------------------------------------------------------------------------
trade  theta_N       6.9%       3.9%      3.56      5.18           3.83           5.39                   45.5%    [30.1%, 62.7%]
--------------------------------------------------------------------------------------------------------------------------------
Proportional difference in means, unadjusted (theta_N).
Intervals transform the endpoints of the interval of the log ratio. Covariance: cluster-robust by 'pair', convention ssc='stata'.
```

The outcome table reports 45.5% more average trade in observed agreement
years. That comparison uses the two observed groups of rows. The
regression's 53.9% holds the three fixed-effect blocks constant. Read the
two numbers as answers to their respective comparisons. Both tables use the same simulated panel and were reproduced with Powermean 0.1.1.
[Benatia et al. (2026), §§2–3 and 5](https://arxiv.org/abs/2203.11820v4) · [Weidner and Zylkin (2021), §§3.1–3.4](https://doi.org/10.1016/j.jinteco.2021.103513)

## Continue with the same model

Read [fixed effects](methods/fixed-effects.md) to compare joint absorption
with pair cancellation on this panel. Read [coefficient bias](methods/bias-correction.md)
and [covariance refinements](methods/covariance-refinements.md) to see how
the adjustments in Weidner and Zylkin (2021) change the agreement estimate
and its interval.

<!-- guide-navigation:start -->
[Previous: Nonlinear panel DiD](examples/nonlinear-did.md) · [Guide index](README.md) · [Next: Difference out two blocks](methods/jochmans.md)
<!-- guide-navigation:end -->
