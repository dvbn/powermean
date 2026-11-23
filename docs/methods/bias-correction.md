# Correct coefficient bias from estimated effects

With many firms and a fixed number of years, [cancelling firms](cancellation.md)
can leave a fixed number of year effects. In gravity, the number of
exporter-year and importer-year effects grows with the number of countries,
even after cancelling pairs. Their estimation can shift the distribution
of the slope by an amount comparable to its standard error. The growing-dimension result of Benatia et al. (2026) describes a leading bias of order $d/n$ when the
number of retained coefficients $d$ is at most of order $\sqrt n$ under its
information and moment conditions. A coefficient correction estimates and
subtracts that leading bias. It is an explicit analysis choice.
[Benatia et al. (2026), §5.2, Theorem 2 and Appendix C.2](../references.md)

<a id="bbp-after-cancelling-independent-short-groups"></a>
<a id="bias-correction--bbp-after-cancelling-independent-short-groups"></a>

## Correction after cancelling independent short groups

Use `bias_correction="short_group"` when the cancelled histories are the
independent sampling units under the conditions of Benatia et al. (2026). The calculation uses
their scores and derivatives at the uncorrected solution. It accounts for
the association between each history's score and derivative and for the
curvature of the total score. In the fixed-year firm example the retained
dimension is fixed, so a leading bias correction is unnecessary for a centered first-order limit. This call illustrates how to request it and
report the change.

```python
corrected = pm.fit(
    firm_formula, firms, cancel="firm", bias_correction="short_group", **common
)
print(corrected.summary())
```

```text
powermean
====================================================================================
GPML (omega = 1, tau = 2)
Fixed effects: firm (cancelled), year (estimated)
Observations: 384 of 384 input rows (none dropped)
Status: verified; certified: True
Coverage: unique_root (BBP prop:fe-fixed-design)
BBP-corrected reported coefficients; certificate and first-order inference at the uncorrected estimating root.
   uncorrected      bias  estimate
0    -0.311998 -0.001029  -0.31097
------------------------------------------------------------------------------------
                    coef.   std. err.        z    P>|z|    [95% conf. interval]
------------------------------------------------------------------------------------
price_index       -0.3110      0.0428    -7.27    0.000     -0.3948     -0.2272
------------------------------------------------------------------------------------
```

The reported coefficient is the uncorrected coefficient minus estimated
bias. The standard error here is the first-order firm sandwich evaluated
at the uncorrected root. In the covered growing-dimension gravity argument,
Benatia et al. (2026) assume independent pair histories and dense support, such as complete
off-diagonal pairs. A working variance other than PPML requires its own
cancelled equation and the corresponding conditional moment.

The separate `"cell"` option uses the cell form of the correction. Its
feasible cancelled version additionally needs growing cancellation
groups to recover scales uniformly and the stated dependence and
information conditions. A short-pair panel does not gain that justification
from software acceptance. Powermean supplies these coefficient corrections on
cancelled fits. [Benatia et al. (2026), equations (5.5)–(5.7), the three-way panel remark and Appendix C.2](../references.md)

## WZ for three-way PPML gravity

Weidner and Zylkin (2021) show that three-way PPML is consistent with a fixed number
of years under their assumptions. Estimating exporter-year and importer-year
effects can still shift the slope distribution and distort pair-clustered
standard errors. The published analytical slope and covariance adjustments
target those two sources of error.

Martin Weidner and Thomas Zylkin (2021). *Bias and consistency in three-way
gravity models*. Journal of International Economics 132, 103513. The
coefficient correction follows §3.4.2 and the expansion in Proposition 3.
[Published paper](https://doi.org/10.1016/j.jinteco.2021.103513)

Continue with `trade` and `formula` from the [gravity walkthrough](../get-started.md).
Here the explicit covariance request keeps the ordinary pair sandwich
while you inspect the coefficient adjustment.

```python
wz_slope = pm.fit(
    formula,
    trade,
    omega=1,
    tau=1,
    cancel=None,
    bias_correction="wz",
    vcov={"cluster": "pair", "adjustment": "none"},
)
print(wz_slope.summary())
```

```text
powermean
====================================================================================
PPML (omega = 1, tau = 1)
Fixed effects: pair (estimated), exporter^year (estimated), importer^year (estimated)
Observations: 1056 of 1056 input rows (none dropped)
Status: verified; certified: True
Coverage: unique_root (BBP prop:iols-exact-sample-root)
WZ-corrected reported coefficients; certificate and first-order inference at the uncorrected root.
Fixed effects and predictions remain original, uncorrected estimating-root outputs.
   uncorrected      bias  estimate
0     0.431455 -0.003962  0.435417
------------------------------------------------------------------------------------
                  coef.   std. err.        z    P>|z|    [95% conf. interval]
------------------------------------------------------------------------------------
agreement        0.4354      0.0563     7.73    0.000      0.3250      0.5459
------------------------------------------------------------------------------------
Fixed-effect block counts (fitted rows, active covariance)
  pair: levels=132, redundant=132 (exact), nested in=pair
  exporter^year: levels=96, redundant=1 (exact), nested in=none
  importer^year: levels=96, redundant=8 (exact), nested in=none
Nested blocks count as fully redundant in these block counts.
```

For this draw the slope rises by about 0.004. `tidy()`, intervals and
coefficient contrasts use the corrected point, with the requested
covariance evaluated at the original root. After a WZ correction,
`fixef()` and joint in-sample predictions remain at that root, as their
metadata indicate. The correction works on unweighted built-in PPML with
the three specified blocks, observed bread and the covered sample and
normalization. See the [exact WZ requirements](../reference/experimental-fixed-effects.md#coefficient-corrections)
before using it on a different panel.

[Covariance refinements](covariance-refinements.md) runs the separate WZ
covariance request. These adjustments address leading terms under their
respective assumptions. The single draw above illustrates the output.

<!-- guide-navigation:start -->
[Previous: Difference out two blocks](jochmans.md) · [Guide index](../README.md) · [Next: Gravity pair covariance](covariance-refinements.md)
<!-- guide-navigation:end -->
