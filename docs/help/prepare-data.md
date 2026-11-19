# Prepare the estimation sample

The [gravity example](../get-started.md) uses one row per directed pair and
year. An observed zero is a trade flow of zero. A missing row or a missing
outcome conveys different information. Keep those distinctions when you build
the panel.

## Check outcomes and identifiers

- Keep observed zeros for a positive mean order. Log OLS and negative orders
  require strictly positive outcomes; `positive_only=True` defines a
  different estimation sample.
- Check that the same directed pair has the same label across dates. Check
  uniqueness at the intended observation level before interpreting a slope.
- Keep the outcome, regressor, effect and cluster columns on the same rows.
  Report missingness and explicit `subset` or positive-outcome restrictions.
- Inspect `result.sample` and `result.summary()` for the rows used, fixed-effect
  sample changes and dropped regressors. Compare those rows before comparing
  estimators or software. [Worked sample comparison](../examples/compare-samples.md)

## Write the formula

The left-hand side names the outcome. The right-hand side before `|` uses
Formulaic to construct ordinary regressors. Write a square as `I(x**2)`.
After `|`, use `+` to list blocks and `^` to make one interaction block.

```text
trade ~ 0 + agreement | pair + exporter^year + importer^year
sales ~ 0 + price_index | firm + year
```

A standalone intercept is redundant with these effect blocks. A fixed
regressor within a pair may also be redundant. A single interaction
classification, such as `cluster="exporter^importer"`, differs from two
cluster dimensions, `cluster="exporter + importer"`.

## Keep known scales and sampling information explicit

Use `exposure="duration"` for a positive exposure, or
`offset="log_duration"` for an offset already on the log power-mean scale.
Supply one of them. If the specification accepts weights, name the column
and its type, such as `weights="frequency", weight_type="frequency"`.
Fixed-effect fits require unit observation weights.
Keep the clustering column at the level at which histories can be
considered independent. Supply the relevant cluster information to
`pm.prepare` if you will request it from prepared data later.

See [options](../reference/options.md) for input domains and
[troubleshooting](troubleshooting.md) for missing, separated or unidentified fits.

<!-- guide-navigation:start -->
[Previous: Gravity walkthrough](../get-started.md) · [Guide index](../README.md) · [Next: Run the examples](../examples/README.md)
<!-- guide-navigation:end -->
