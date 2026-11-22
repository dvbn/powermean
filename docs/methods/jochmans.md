# Difference out two fixed-effect blocks

Set `estimator="gmm1"` or `"gmm2"` in `pm.fit` for an exogenous exponential mean
with two fixed-effect blocks. Jochmans's differences remove both blocks from
the estimating equation. The fitted result reports the slopes.
[Papers](../references.md#two-way-exponential-regression)

## Choose the observed array

Supply a data frame with one row per observed cell, finite regressors
and a finite nonnegative outcome. Use `0 +` for an intercept-free
Formulaic design and put exactly two identifier columns after `|`, in place of
the two identifier options in the authors' Stata commands. `estimator=` selects
the equation (`model()` in Stata). `start=` takes a one-dimensional vector in
coefficient order (`initial()` in Stata).

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

These are the two moment equations in Jochmans and Verardi (2020).
GMM2 gives each GMM1 rectangle a coefficient-dependent multiplier
$\exp((p+q)'b)$. The roots can differ. With exogenous slopes each equation
has as many moments as slopes and uses a local Newton root solve. The
default start is zero. Use `start=` to supply a finite vector in the
displayed coefficient order and `max_iter=` to change the budget.

The multiplicative errors must have unit conditional means and unit
conditional products for the opposite cell pairs used by the differences,
given the full regressor array and fixed effects. For the cell-projection
standard errors, assume conditional independence of the observed cells
and the large-array regularity of the source, including growth in both
dimensions. This scope excludes fixed-$T$ and unrestricted dyadic
dependence. Calling this estimator adopts these assumptions.

An intercept, an additive row or column regressor, or dependent rectangle
contrasts raise `IdentificationError`. The solver also requires a resolved
full-rank Jacobian at the evaluated point. Both a scaled moment and a
local Newton correction must pass before inference. The returned
`fit.certificate.status` is `"local_solution"`; existence and outcome can
depend on the data and the start. [Troubleshoot a refusal](../help/troubleshooting.md#jochmans-fits)

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

The implementation contracts matrix products and sums without listing
rectangles. In float64 it places a conservative bound of 150 on the
absolute centered fitted index and on the log range of positive outcomes
after common normalization. Ill-resolved cancellation, failed root
checks and unusable covariance can lead to refusal. [Function index](../reference/api.md#pmfit)

<!-- guide-navigation:start -->
[Previous: Cancel one block](cancellation.md) · [Guide index](../README.md) · [Next: Numerical methods](numerical-methods.md)
<!-- guide-navigation:end -->
