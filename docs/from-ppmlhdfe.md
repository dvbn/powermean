# Compare Powermean with ppmlhdfe

Use the [gravity panel](get-started.md) for a direct translation of the
specification. Start by matching the rows and mean equation, then match
the inference convention.

## Match the gravity equation

```python
fit = pm.fit(
    "trade ~ 0 + agreement | pair + exporter^year + importer^year",
    trade, omega=1, tau=1, cluster="pair",
)
```

With the same columns loaded in Stata, the corresponding syntax is

```stata
ppmlhdfe trade agreement, absorb(pair exporter#year importer#year) vce(cluster pair)
```

After `|` in Powermean, `exporter^year` names the exporter-year group,
corresponding to the interaction in Stata's `absorb()`.
Cluster on the directed pair. In other designs, use
`offset="log_duration"` or `exposure="duration"` to enter a known scale,
when the formulation supports it. [ppmlhdfe syntax](https://scorreia.com/help/ppmlhdfe.html)

## Check the sample and uncertainty

Check used rows, singleton and separated observations, excluded regressors,
offsets and weights. The fixed-effect interface uses unit
observation multipliers. Compare coefficients and fitted values at the
same numerical accuracy. For uncertainty, also match the cluster
classification, treatment of absorbed degrees of freedom, small-sample
factor and reference distribution. [Powermean conventions](reference/options.md#small-sample-conventions)

Coefficient correction and WZ covariance adjustment are additional
requests with their own domains. Use the same adjustment, or compare the
uncorrected PPML fits. [WZ options](methods/bias-correction.md)

When comparing elapsed time, include the same work in each timer.
Capturing a formula and sample, solving, assessing the numerical point
and computing uncertainty all take time. Compare first fits with first
fits and prepared fits with prepared fits. [Validation](validation.md)

<!-- guide-navigation:start -->
[Previous: Gravity pair covariance](methods/covariance-refinements.md) · [Guide index](README.md) · [Next: Run the examples](examples/README.md)
<!-- guide-navigation:end -->
