# Difference out two fixed-effect blocks

GMM1 and GMM2 estimate slopes in an exponential mean model with two fixed
effects. Both compare outcomes in groups of four observations, so neither
fixed effect enters the slope equation. Use these estimators for a complete
panel, such as firms observed in every year, or a complete set of directed
links, such as trade between countries. [Jochmans (2017), §§I–III](https://doi.org/10.1162/REST_a_00620) · [Open manuscript](https://www.repository.cam.ac.uk/handle/1810/286880)

## When is the comparison valid?

Take two firms $i,k$ observed in two years $j,\ell$. Write
$\mu_{ij}=a_i g_j\exp(x_{ij}'\beta)$ for expected sales and
$U_{ij}=Y_{ij}/\mu_{ij}$ for the multiplicative error. Let $C$ denote the
full regressor array and both sets of fixed effects. The mean model gives
$E[U_{ij}\mid C]=1$ for each observation.

GMM1 and GMM2 also compare the products on the two diagonals. For this
comparison to average to zero at the true slope, the rectangle must satisfy

$$
E[U_{ij}U_{k\ell}\mid C]=E[U_{i\ell}U_{kj}\mid C].
$$

Take a retailer and its supplier. Measure each firm's sales as a percentage
above or below what its covariates and fixed effects predict. One diagonal
pairs the retailer this year with the supplier next year. The other pairs
the supplier this year with the retailer next year. Since each error has
conditional mean one, the equation above holds exactly when the two
pairings have the same conditional covariance.

- **When it holds.** Suppose each firm's demand shocks are independent of
  the other firm's shocks, given the covariates and fixed effects. A good
  year can carry over into another good year at the same firm. Both
  cross-firm covariances are still zero. The condition also holds if the
  two covariances have the same nonzero value.
- **When it fails.** Suppose the retailer restocks after an unexpectedly
  strong year, raising the supplier's sales the following year. Suppose
  also that an unexpectedly good year for the supplier is unrelated to the
  retailer's sales the following year. The first covariance is positive
  and the second is zero.

In either case the model can correctly predict each firm's sales on
average. GMM1 and GMM2 need the extra assumption because they use products
of the firms' errors. If the covariance difference lines up with relative
price changes across firms, it can distort the estimated price elasticity
even when prices are exogenous.

PPML and Gamma PML use each observation's deviation from its own expected
sales. The same spillover can leave their estimating equations valid when
the conditional mean is correctly specified. Their standard errors still
need to account for dependence across firms and years.

Jochmans (2017) derives the difference moment with both conditional products
equal to one. The GMM standard errors reported here use the paper's cell
projections and assume conditional independence across observations, along
with its other large-array conditions. In a firm panel, both the number of
firms and the number of years must grow for this inference.
[Jochmans (2017), eqs. (1.3)–(1.4), (2.1), and Theorem 2](https://doi.org/10.1162/REST_a_00620)

## Choose the observed array

Pass one row per observed cell, with finite regressors and a finite,
nonnegative outcome. Put the two identifiers after `|` and omit the
intercept with `0 +`. Set `estimator="gmm1"` or `estimator="gmm2"` to
choose the moment equation.

### Rectangular panels

Use `fe_layout="rectangular"` (the default for GMM fits, corresponding to
`twexp`) for a complete crossing with at least two levels on each axis. A panel needs one row
per unit and period. The unit and period labels are separate, and the two
dimensions can have different lengths.
A two-by-two panel contains only one rectangle, so it cannot yield usable
cell-projection standard errors. The fit refuses with `unusable_covariance`
if it reaches a root on that layout.

The `twexp` help illustrates the method with firms' patent counts and log R&D
spending, originally stored in separate columns for each year. Reshape such
data to long format before fitting. To follow the authors' example of using
the GMM1 slopes to initialize GMM2, keep the formula and its column order fixed.

```python
import powermean as pm

formula = "patents ~ 0 + log_rd | firm + year"
gmm1 = pm.fit(formula, firm_year, estimator="gmm1")
gmm2 = pm.fit(formula, firm_year, estimator="gmm2", start=gmm1.coef())
print(pm.etable(gmm1, gmm2, uncertainty="ci"))
```

### Directed dyads

Use `fe_layout="directed"` (corresponding to `twgravity`) for a cross-section of
ordered pairs of agents. Supply every ordered non-self link among at least
four agents. Use the same labels for an agent in the origin and destination
columns. The outcome and regressors may differ in opposite directions.

The `twgravity` help fits trade flows in levels using bilateral variables
such as log distance, a shared border and a shared language.

```python
fit = pm.fit(
    "trade ~ 0 + log_distance + border + common_language | exporter + importer",
    bilateral_trade, estimator="gmm2", fe_layout="directed",
)
print(fit.summary())
```

An observed zero stays in the sample. Missing pairs, duplicate pairs, missing
or non-finite model values, and self-links in the directed layout raise
`SampleError`. To restrict the fit as you would with Stata's `if` or `in`,
filter the data frame first and pass a complete array on the selected levels.
[Runnable examples for both layouts](../../examples/jochmans.py)

## Choose the moment equation

Write $p=x_{ij}+x_{k\ell}$, $q=x_{i\ell}+x_{kj}$, $P=y_{ij}y_{k\ell}$ and
$Q=y_{i\ell}y_{kj}$. Both methods set the average of $(p-q)D(b)$ to zero over
admissible rectangles. Directed rectangles use four distinct agent labels.

| Method | Difference $D(b)$ |
| --- | --- |
| `"gmm1"` | $P\exp(-p'b)-Q\exp(-q'b)$ |
| `"gmm2"` | $P\exp(q'b)-Q\exp(p'b)$ |

GMM2 multiplies each GMM1 rectangle by $\exp((p+q)'b)$. Both target the same
identified slope under the rectangle restriction, but rectangles receive
different weights and the estimates can differ in a given sample.

PML also offers a choice of weights, applied to a different quantity.
At `omega=1`, a PML score with working variance $\mu^\tau$ weights each
observation's relative residual $Y/\mu-1$ by $\mu^{2-\tau}$. PPML gives
more weight to observations with larger means. Gamma PML weights relative
residuals equally. Changing `tau` reweights these single-observation
residuals. Choosing between GMM1 and GMM2 reweights four-observation
contrasts, and both choices require the extra product restriction above.
[Mean and working variance](model.md#set-weights-in-the-estimating-equation) · [Benatia et al. (2026), §3.2](https://arxiv.org/abs/2203.11820v4)

Each GMM equation has one moment per slope. The Newton solver starts at
zero unless you pass a finite vector in coefficient order with `start=`.
Use `max_iter=` to set the iteration limit.

An intercept, an additive row or column regressor, or dependent rectangle
contrasts raise `IdentificationError`. The solver also requires a resolved
full-rank Jacobian at the evaluated point. Both a scaled moment and a
local Newton correction must pass before inference. The returned
`fit.certificate.status` is `"local_solution"`; existence and outcome can
depend on the data and the start. [Troubleshoot a refusal](../help/troubleshooting.md#gmm1-and-gmm2-fits)

## Read uncertainty and compare with Stata

Use `fit.coef()`, `fit.se()`, `fit.vcov()`, `fit.confint()`, `fit.tidy()`
and `fit.summary()`. `fit.sample.positions` and `fit.sample.retained_index`
keep input order. `pm.etable` can display fitted slopes or their
unit-change transformations `exp(b) - 1`.

Each projection row sums the kernels of rectangles incident on that
observed cell. With $M$ rectangles and $N$ observed cells, the projection
is $h_s=(N/M)\sum_{r\ni s}(p_r-q_r)D_r(b)$. Let $G$ be the derivative of
the mean moment. The `vcov="paper"` default uses
$G^{-1}(\sum_s h_s h_s'/N)G^{-\mathsf T}/N$ and normal-reference intervals.
In the rectangle $M=\binom{I}{2}\binom{J}{2}$ and $N=IJ$. In the directed
array $M=\binom{n}{2}\binom{n-2}{2}$ and $N=n(n-1)$.

Use `vcov="stata"` to apply the `twgravity` additional multiplier
$(n-1)/n$ for directed fits. The source averages projection products
over its $n^2$ matrix slots and uses the observed $N$ for the final
coefficient covariance. The option has unit multiplier in the rectangle.
`fit.with_vcov("stata")` changes the reporting convention on an existing
fit. Both layouts use cell projections as their available inference.

Powermean subtracts the mean of the observed design columns before solving.
In the source comparisons, `twexp` GMM2 centers on a square-padded
array and `twgravity` centers by the observed sum divided by $n^2$.
The different centers change valid-cell regressors by a common shift,
scaling the equation and its projection by a coefficient-dependent
factor. At the root that factor cancels from the sandwich because the
corresponding Jacobian term proportional to the moment is zero. A
fully padded row or column contributes zero to both rectangular outcome
products. In the directed calculation the diagonal is masked in both
factors, so only rectangles made from observed links contribute.

The implementation contracts matrix products and sums without listing
rectangles. In float64 it places a conservative bound of 150 on the
absolute centered fitted index and on the log range of positive outcomes
after common normalization. Ill-resolved cancellation, failed root
checks and unusable covariance can lead to refusal. [Function index](../reference/api.md#pmfit)

## Predict trade after fitting

`fit.predict()` recovers origin and destination effects by PPML with
the GMM slopes held fixed. The fitted trade totals match the observed
origin and destination totals. Each residual is observed trade minus
fitted trade.

For the directed fit above, the following compares fitted trade with
trade predicted if every retained pair had a common language.

```python
retained = bilateral_trade.iloc[fit.sample.positions]
fitted = fit.predict()
effects = fit.predict(type="fixed_effects")
residuals = fit.predict(type="residual")
same_language = fit.predict(retained.assign(common_language=1))
```

The second prediction uses the same slopes and recovered effects.
See the [runnable GMM example](../../examples/jochmans.py) for synthetic data
and [prediction and counterfactuals](../reference/results.md#prediction-and-retention)
for the retained-row requirements.

<!-- guide-navigation:start -->
[Previous: Gravity walkthrough](../get-started.md) · [Guide index](../README.md) · [Next: Coefficient bias](bias-correction.md)
<!-- guide-navigation:end -->
