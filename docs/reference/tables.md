# Tables and formats

For a first run, see the [worked sample comparison](../examples/compare-samples.md)
and the [balance example](results.md#inspect-balance-on-the-baseline-sample).
All five functions return a `Table`. Call `print(table)` for the fixed-width
text shown in the guide.

## Describe data and treatment arms

Pass a pandas DataFrame. Use `vars` to name one column or a sequence of
columns. With `vars=None`, the summary and balance calls select eligible
numeric columns and skip their grouping, treatment and weight inputs.

| Call | Output |
| --- | --- |
| `pm.summary_statistics(data, vars=None, **options)` | One row per variable with the usable and missing counts, mean, standard deviation and quantiles. For nonnegative variables it also reports the zero share and the mean among positive values. |
| `pm.balance_table(data, treatment, vars=None, **options)` | Covariate means by arm, differences from the control, standard errors, p-values and standardized differences. The numeric frame also holds arm counts and standard deviations. |
| `pm.balance_regression(data, treatment, vars=None, **options)` | Regresses treatment status on the listed covariates. It reports coefficients with standard errors, the joint test, its p-value and the fitted sample. Each noncontrol arm has its own comparison with the control. With `continuous=True`, it instead regresses the dose itself on the covariates. |
| `pm.outcome_table(data, outcomes, treatment, **options)` | One row per nonnegative outcome and mean order. Reports zero shares, means and positive-outcome means by arm and unadjusted proportional differences from the control with intervals. The numeric frame is indexed by outcome and order. |

Use the original promotion data with the price-adjusted fit to describe its
retained rows.

- Summarize visits with
  `pm.summary_statistics(data, vars=["visits"], probs=(0.5,), sample=with_price)`.
  [Saved summaries](../examples/compare-samples.md#describe-the-two-retained-samples)
- Compare observed promotion arms with
  `pm.outcome_table(data, outcomes="visits", treatment="promotion", sample=with_price)`.
  [Saved outcome table](../examples/compare-samples.md#summarize-outcomes-by-promotion-within-one-sample)

Use `baseline`, the period-1 rows of the DiD panel, to check baseline
differences by cohort.

- Compare baseline visits by cohort with
  `pm.balance_table(baseline, treatment="cohort", vars=("visits",))`.
  [Saved baseline tables](results.md#inspect-balance-on-the-baseline-sample)
- Regress cohort assignment on baseline visits and report the joint test with
  `pm.balance_regression(baseline, treatment="cohort", vars=("visits",))`.
  [Saved baseline tables](results.md#inspect-balance-on-the-baseline-sample)

All four functions accept `weights` for a weight column, `by` for a
subgroup column, `labels` for display names, and `digits` for display
precision. `sample=fit` uses a fitted result's row positions. Supply the
original frame with the same index labels in the same order. Each calculation
then handles missing values within those rows according to its own options.

| Where | Additional options |
| --- | --- |
| Summary | `probs` selects quantile levels from 0 to 1. The default is `(0, .05, .1, .25, .5, .75, .9, .95, 1)`. `complete=True` uses rows complete in every requested variable, and `expand=True` expands categorical columns into indicators. |
| Balance table | `control`, `bins`, `cluster`, `fe`, `ssc`, `complete`, `expand`. |
| Balance regression | `control`, `bins`, `continuous`, `cluster`, `fe`, `vcov`, `ssc`, `expand`. |
| Outcome table | `orders`, `control`, `bins`, `cluster`, `ssc`, `positive_only`, `level`. |

`control` selects the reference arm and defaults to the first sorted arm.
`bins` groups a numeric treatment into quantile groups. For balance
calculations, `fe` names one absorbed classification or one interaction
such as `"site^year"`. `cluster` names one clustering column. The supported
`ssc` names are `"stata"` (default), `"none"` and `"regression"`.
Balance regression uses robust covariance by default and also accepts
`vcov="classical"` without clustering. In a balance regression choose either
`bins` for arm comparisons or `continuous=True` for the dose itself.

The outcome table defaults to `orders=(1,)` and `level=0.95`. If you
request a nonpositive order in the presence of zeros, use
`positive_only=True` to report that order on the positive outcomes.
Intervals transform the log-ratio endpoints and are specific to each
comparison. These tables describe the observed groups. For an adjusted
comparison, fit a model and inspect a fitted effect or contrast.
[Proportional changes](../methods/proportional-effects.md)

To work with the recorded outcome table from the
[promotion example](../examples/compare-samples.md#summarize-outcomes-by-promotion-within-one-sample),
read the numeric output of the same `groups` table:

```python
values = groups.to_frame().reset_index()
columns = ["outcome", "order", "estimate[1]", "conf_low[1]", "conf_high[1]"]
print(values[columns].round(3).to_string(index=False))
```

```text
outcome  order  estimate[1]  conf_low[1]  conf_high[1]
 visits    1.0          2.2        0.388         6.375
```

The estimate and its endpoints are proportions in the numeric frame.
The console outcome table displays them as percentages.

## Tabulate fitted models

`pm.etable(a, b)` and `pm.etable([a, b])` give one column per fitted result.
The table reports the estimator and mean order, fixed-effect treatment,
the covariance convention and reference, and relevant counts and correction
labels. Its numeric frame has one row per model and term, with estimates,
standard errors, p-values and confidence endpoints. Model information is
available in `table.to_frame().attrs["models"]`.

In the promotion example, check the retained positions before tabulating
the fits with `pm.etable(with_price, with_stock, stats=("nobs", "zeros"))`.
The table displays each fit's own uncertainty.
[Saved model table](../examples/compare-samples.md#compare-the-estimates)

To report alternate confidence intervals for the Newton fit, pass its
alternate result view with `pm.etable(robust, uncertainty="ci")`, or request
the covariance within the table with
`pm.etable(newton, vcov="robust", uncertainty="ci")`.
Both use the fitted point without refitting.
[Worked covariance example](../examples/reuse-a-fit.md#inspect-a-fit-and-change-its-covariance)

| Options | Meaning |
| --- | --- |
| `names`, `title`, `notes` | Label the models and add a table title or notes. Supply one name per result. |
| `scale` | `"coefficient"` by default. `"effect"` reports `100 * expm1(b)` separately for each coefficient coordinate, with uncertainty on that display scale. |
| `uncertainty`, `level` | Show standard errors (`"se"`, default), intervals (`"ci"`) or p-values (`"p"`). The default confidence level is 0.95. |
| `vcov` | Request an alternate covariance available from the retained information, without refitting the models. |
| `keep`, `drop`, `labels` | Select, exclude or rename terms by their fitted names. |
| `stars`, `digits` | Give p-value cutoffs or turn stars off with `False`. Default cutoffs are .10, .05 and .01; default precision is three decimals. |
| `stats` | Show the observation count with `"nobs"` (default), the zero share with `"zeros"`, or both. |

For a change that depends on several coefficients, use `fit.contrast()` for
two complete regressor settings or `fit.nlcom()` for a coefficient expression.
[Worked contrast](../examples/reuse-a-fit.md#report-effects-and-contrasts) · [Worked coefficient expression](../examples/reuse-a-fit.md#test-or-combine-coefficients)

## Print or export

| Method | Output |
| --- | --- |
| `print(table)`, `table.to_text()` | Fixed-width console text. |
| `table.to_frame()` | An independent DataFrame with unrounded values and metadata; its columns depend on the table. |
| `table.to_frame(formatted=True)` | The displayed cells as strings, including the group heading when applicable. |
| `table.to_markdown()` | Markdown source to paste into a document or save to a file. [Rendered example](../examples/compare-samples.md#export-a-table-to-markdown) |
| `table.to_html()` | An HTML table. In a notebook, displaying the object uses its HTML representation. |
| `table.to_latex(booktabs=True, floating=True, label=None)` | A LaTeX table. Set `floating=False` for a standalone tabular and `booktabs=False` to use ordinary rules. |

The formatted exports use the same displayed cells. Use the numeric frame
for calculations and save model coefficients and covariance from the fitted
result for further inference. [Worked exports](../examples/reuse-a-fit.md#export-tables)

<!-- guide-navigation:start -->
[Previous: Results](results.md) · [Guide index](../README.md) · [Next: Solver options](solver-options.md)
<!-- guide-navigation:end -->
