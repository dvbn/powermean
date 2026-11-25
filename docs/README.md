# Powermean user guide

<a id="powermean-01-user-guide"></a>

<img src="assets/powermean-logo.png" align="right" width="120" class="guide-logo" alt="Powermean logo" />

Start with the [first fit](../README.md#a-first-fit), then follow the same three worked examples as in the R guide: a conditional mean, three-way gravity, and nonlinear difference-in-differences. Use the chapters below to prepare data, choose a model and report the results. [Function index](reference/api.md) · [Offline guide](guide.html) · [Quarto source](guide.qmd)

## Get started

- [Install Powermean](install.md) from PyPI or a local distribution.
- [Three worked examples](quick-tour.md): fit a conditional mean, compare joint and cancelled gravity models, and estimate effects under staggered adoption.
- [Fit one mean three ways](methods/model.md). Use twelve observations to see what PPML, Gamma PML and other working variances estimate. Then choose a power mean, log OLS or an NB2 variance for your outcome.
- [Prepare your data and formula](help/prepare-data.md). Specify regressors and interactions, restrict the sample, and account for exposure, offsets or observation weights.
- [Compare estimation samples](examples/compare-samples.md). Find which input rows each fit used, why rows were excluded and how the outcome distribution changed.

## Read and report results

- [Interpret a proportional change](methods/proportional-effects.md). Distinguish a change in an individual regressor from a contrast involving several covariates.
- [Choose standard errors](methods/inference.md). Match robust or clustered covariance to the sampling design.
- [Reuse a fit](examples/reuse-a-fit.md). Prepare a design once, compare solvers, change covariance, and compute contrasts, linear combinations and hypothesis tests.
- [Predict and inspect results](reference/results.md). Find fitted means, residuals, influence, numerical diagnostics, and [predictions at other covariate values](reference/results.md#which-rows-can-i-predict-for).
- [Describe the data and export tables](reference/tables.md). Report summary statistics, balance, outcomes by group, or regression coefficients and intervals.

<a id="fit-a-model"></a>

## Fixed effects

- [Estimate fixed effects jointly](methods/fixed-effects.md). Write several blocks in one formula and check which comparisons identify the slope.
- [Cancel a fixed-effect block](methods/cancellation.md). Follow the same firms over time, recover their effects after fitting, and [predict sales after a price change](reference/results.md#predict-from-a-cancelled-gamma-fit).

<a id="panel-designs-and-inference"></a>
<a id="read-the-guide--panel-designs-and-inference"></a>

## Panel designs and applications

- [Three-way gravity](get-started.md). Fit bilateral trade in levels with pair, exporter-year and importer-year effects.
- [Nonlinear difference-in-differences](examples/nonlinear-did.md). Estimate policy effects with common or staggered adoption under multiplicative parallel trends.
- [Difference out two fixed effects](methods/jochmans.md). Check the extra assumption behind GMM1 and GMM2 before using a complete panel or directed pairs.
- [Correct coefficient bias](methods/bias-correction.md). Compare the short-panel adjustments of Benatia et al. (2026) and the three-way PPML correction of Weidner and Zylkin (2021).
- [Adjust pair covariance in gravity](methods/covariance-refinements.md). See how the covariance adjustment changes an interval for the agreement effect.
- [Compare with Stata ppmlhdfe](from-ppmlhdfe.md) on the same observations and specification.
- [Run the example scripts](examples/README.md) to reproduce the applications.

## Reference and help

| Page | What to look up |
| --- | --- |
| [Function index](reference/api.md) | Calls, returned objects, reporting tools, exceptions and synthetic data. |
| [Supported methods](availability.md) | Which means, variances, fixed effects, solvers and covariance requests work together. |
| [Fit options](reference/options.md) | Formulas, sample restrictions, weights, variance, clustering and small-sample conventions. |
| [Numerical methods](methods/numerical-methods.md) | Which equation was solved and what its certificate means. |
| [Solver options](reference/solver-options.md) | Starting values, iteration limits and accuracy controls. |
| [Advanced fixed effects](reference/experimental-fixed-effects.md) | Joint and cancelled controls, separation and correction requirements. |
| [IOLS options](reference/iols_options.md) | Select the implementation and set iteration or scaling options. |
| [IOLS installation](iols_installation.md) | Compiled and NumPy-only builds. |
| [Troubleshooting](help/troubleshooting.md) | Diagnose a warning, a sample exclusion or a failed fit. |
| [Report a problem](help/report-a-problem.md) | Gather sample, diagnostic and version information. |
| [Validation](validation.md) | Numerical and statistical checks. |
| [Papers and methods](references.md) | Sources for the estimators and their assumptions. |
| [Build the guide](BUILD_GUIDE.md) | Regenerate the standalone HTML. |
