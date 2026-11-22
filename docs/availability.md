# Supported methods

<a id="what-powermean-01-supports"></a>

Statistical interpretation
depends on the mean, sampling assumptions and the applicable fixed-effect
regime. [Methods and papers](references.md)

| Task | Domain |
| --- | --- |
| Fit without fixed effects | `pm.fit` with finite mean orders, power working variance or supported NB2, plus log OLS at order zero. Positive orders allow zeros. Nonpositive orders require positive outcomes. |
| Estimate fixed effects jointly | One or more blocks after `\|` in the formula. Predict in sample or replace covariates on retained rows. Supplied observation weights are unavailable. |
| Cancel one block | Select a named block at nonzero mean order, with every other block retained. Recover the cancelled scales with `fixef()` or use them for prediction. |
| Difference out two blocks | `pm.fit` with `estimator="gmm1"` or `"gmm2"` on complete rectangles or complete directed non-self links. Cell-projection covariance and prediction with recovered effects. [Method](methods/jochmans.md) |
| Choose a numerical procedure | Newton is the default for nonlinear fits. IOLS handles supported no-FE models with positive order and power `tau` in `[1, 2]`. Order zero uses a linear calculation. |
| Reuse prepared rows | `pm.prepare` captures the design and available cluster information. Supported fits can change mean, working variance or covariance within the prepared domain. |
| Estimate uncertainty | Robust, named HC and one- or multiway cluster methods within their fit-time and postfit domains. With cancellation, use observed bread. The fit-specific default covariance and the named HC rules are in [options](reference/options.md#fit-time-covariance-for-fixed-effects). |
| Correct coefficient bias | The `"short_group"` and `"cell"` corrections of Benatia et al. (2026) for supported cancelled fits. WZ `"wz"` for unweighted three-way PPML within its panel requirements. |
| Adjust pair covariance in gravity | `vcov={"cluster": "pair", "adjustment": "wz"}` for the covered three-way PPML design. |
| Report or export | Coefficients, intervals, conditional proportional changes, contrasts, tests, prediction where supported, and regression, outcome and balance tables. |

For joint fits with nonzero mean order, power `tau` outside `[1, 2]`
requests a selected root. Those values have narrower existence results.
[Working-variance range](reference/options.md#full-family-scope-and-the-current-entry-restriction)

The [advanced FE options](reference/experimental-fixed-effects.md)
describe experimental module-level records and correction requirements.
Powermean does not compute HAC or Conley covariance, the deletion or higher-order covariance estimators of Benatia et al. (2026), automatic working-variance targeting,
or instrumental-variable estimates. The `cancel=` option of `pm.fit` accepts
one block. The
[options](reference/options.md) and [results](reference/results.md)
references give accepted spellings and result-specific limits. Find a
function or output in the [function index](reference/api.md) and look up all
display and export options in [tables and formats](reference/tables.md).

<!-- guide-navigation:start -->
[Previous: Function index](reference/api.md) · [Guide index](README.md) · [Next: Options](reference/options.md)
<!-- guide-navigation:end -->
