# Supported methods

<a id="what-powermean-01-supports"></a>

Start with the mean you want to estimate. The formula, the treatment of
fixed effects and the sampling design determine which estimators and
standard errors you can use together.

## Choose the model

| Your data and question | What to fit |
| --- | --- |
| Nonnegative outcome, including zeros | Use `pm.fit` at a positive mean order. At `omega=1`, PPML (`tau=1`), Gamma PML (`tau=2`) and NB2 (`variance=pm.NB2(kappa)`) all model the arithmetic mean. [Mean order and working variance](methods/model.md) |
| Outcome is strictly positive | You can also model the geometric mean by log OLS (`omega=0`) or use a negative mean order. `positive_only=True` restricts the population represented by a fit when the original data include zeros. [Outcome domain](methods/model.md#choose-a-conditional-mean) |
| Repeated observations with group differences | Put one or more fixed-effect blocks after the formula's vertical bar. The default fits them jointly. At a nonzero mean order, `cancel="firm"` cancels one named block while estimating the other blocks. [Joint fixed effects](methods/fixed-effects.md) · [Cancellation in a short panel](methods/cancellation.md) |
| Complete rectangular panel or directed array with two fixed-effect blocks | GMM1 and GMM2 difference out both blocks under an additional assumption on products of errors across cells. [Assumptions, eligible arrays and examples](methods/jochmans.md) |
| Trade panel or policy adoption | Fit three-way gravity or nonlinear difference-in-differences with `pm.fit`. The DiD example covers common and staggered adoption, cohort-by-period estimates and aggregation. [Gravity](get-started.md) · [Nonlinear DiD](examples/nonlinear-did.md) |

## Set up the sample and estimation

- Specify regressors and interactions in the formula. Use a subset to
  restrict rows, an offset or exposure for a known scale, and observation
  weights on supported fits without fixed effects. To reuse the same
  design, call `pm.prepare` and then `prepared.fit`.
  [Data and formulas](help/prepare-data.md) · [Sample and weight options](reference/options.md#sample-arguments)
- Choose uncertainty to match the sampling design. Ordinary fits support
  robust, named HC, one-way or multiway cluster covariance within the
  relevant model's scope. GMM1 and GMM2 use cell projections.
  [Covariance and conventions](methods/inference.md) · [Fixed-effect covariance defaults](reference/options.md#fit-time-covariance-for-fixed-effects)
- Newton is the default for nonlinear models. IOLS is available for
  certain positive-order models without fixed effects. Log OLS uses a
  linear solve. Outside the working-variance range `1 <= tau <= 2`, a
  joint power-score fit at nonzero order seeks a local root. Read the
  numerical assessment before reporting it.
  [Solver choices](methods/numerical-methods.md) · [Solver controls](reference/solver-options.md) · [IOLS installation](iols_installation.md)
- Coefficient bias corrections target distinct panel designs. Benatia et
  al. (2026) give the short-group and cell corrections for supported
  cancelled models. Weidner and Zylkin (2021) give a coefficient
  correction and a separate pair-covariance adjustment for three-way
  PPML. [Coefficient bias](methods/bias-correction.md) · [Covariance adjustment](methods/covariance-refinements.md) · [Fixed-effect controls](reference/experimental-fixed-effects.md)

## Use the fitted model

| Task | Where to look |
| --- | --- |
| Check which observations entered or why a solve stopped | [Compare estimation samples](examples/compare-samples.md) · [Diagnose unsuccessful fits](help/troubleshooting.md) |
| Calculate a percentage change, compare scenarios or test coefficients | [Proportional effects](methods/proportional-effects.md) · [Contrasts and tests](reference/results.md#proportional-effects-contrasts-and-tests) |
| Predict outcomes, get residuals or extract group effects | [Prediction and counterfactuals](reference/results.md#prediction-and-retention) · [GMM1 and GMM2 results](reference/results.md#gmm1-and-gmm2-fits) |
| Describe outcomes, check covariate balance or export a regression table | [Descriptive, balance and model tables](reference/tables.md) |
| Generate synthetic data or find an exact call | [Simulation designs](reference/api.md#generate-synthetic-data) · [Function index](reference/api.md) |

## Current scope limits

Powermean accepts a user-specified working variance. Automatic working-variance
targeting, instrumental-variable estimation, HAC and Conley covariance, and
the deletion and higher-order covariance estimators of Benatia et al. (2026)
are unavailable. [Reserved arguments](reference/options.md#reserved-requests) · [Papers and methods](references.md)

<!-- guide-navigation:start -->
[Previous: Function index](reference/api.md) · [Guide index](README.md) · [Next: Fit options](reference/options.md)
<!-- guide-navigation:end -->
