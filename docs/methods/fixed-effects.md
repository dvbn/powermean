# Estimate and absorb fixed effects

The [gravity fit](../get-started.md) puts a separate baseline in the mean for
each directed pair and lets each exporter and importer change in every year.
The fit estimates these effects together with the agreement slope. Powermean
can absorb the indicator blocks, which avoids materializing a full dense
matrix of indicators in the solve.

## Specify blocks and check the remaining variation

With three blocks, an agreement can be collinear even if it changes over
time. A policy that varies only by exporter and year lies in the span of
the exporter-year indicators. A time-invariant bilateral feature lies in
the span of the pair indicators. Each needs variation outside the span of
the *combined* effect blocks and other regressors to identify a coefficient.

Absorption is a computational technique. Profiling solves a nuisance
equation conditional on the remaining parameters and substitutes that
solution into the joint equation. Both can produce a fit of the joint
estimator. For the arithmetic mean, `joint.predict()` gives fitted trade
in sample and `joint.fixef()` gives estimated group levels under a
normalization. Individual levels depend on the normalization.
[Result methods](../reference/results.md)

## Compare joint estimation with pair cancellation

Cancellation constructs estimating equations in which the pair scale drops
out. Exporter-year and importer-year effects are still estimated. In the
special case of PPML the concentrated joint slope equation coincides with
the cancelled one on a common usable sample. Use `trade` and `formula`
from the [gravity walkthrough](../get-started.md).

```python
joint = pm.fit(formula, trade, omega=1, tau=1, cluster="pair")
pair_cancelled = pm.fit(
    formula, trade, omega=1, tau=1, cancel="pair", cluster="pair"
)
print(pm.etable(
    joint, pair_cancelled, names=("Joint", "Pair cancelled"),
    stats=("nobs", "zeros"), stars=False, digits=4,
))
```

```text
----------------------------------------------------
                           Joint      Pair cancelled
----------------------------------------------------
agreement                  0.4315         0.4315
                          (0.0563)       (0.0563)
----------------------------------------------------
Estimator                   PPML           PPML
Mean order (omega)           1              1
pair effects             Estimated      Cancelled
exporter^year effects    Estimated      Estimated
importer^year effects    Estimated      Estimated
Covariance             cluster: pair  cluster: pair
Convention                 stata          stata
Reference                  normal         normal
Clusters                    132            132
----------------------------------------------------
Observations               1,056          1,056
Zeros                       6.0%           6.0%
----------------------------------------------------
Standard errors in parentheses.
```

For PPML, at a given slope and set of country-year effects, the fitted
pair total equals the observed total. Its fitted allocation to dates is
proportional to the exponential index with the pair effect omitted.
Subtracting those allocated flows from observed flows yields the same
within-pair equation by cancellation and by profiling. The pair-clustered
standard errors above use the same convention and sample.

At other working variances the cancellation and profiling weights generally
differ. Which estimator is appropriate depends on the group's conditional
mean restriction and how much information each nuisance effect receives as
the sample grows. Absorbing a growing block quickly still leaves its
estimation error in the joint score. [Benatia et al. (2026), §5.1 and Supplemental Appendix S3.6.1](../references.md)

## Choose what you need from the fit

Use joint estimation when the joint equation is the one justified for your
design or when you need joint in-sample predictions. Use cancellation when
the within-group equation and short-group argument fit your design.
`fixef()` in a cancelled fit can recover a scale from each group's observed
outcomes and the fitted slopes and retained effects. A short group's recovery
can remain noisy as more groups enter the sample. Cancelled results offer
slopes, uncertainty and in-sample scale recovery. Joint results supply
in-sample predictions.

[Cancellation in a short firm panel](cancellation.md) shows where switching
estimators changes the result. [Advanced FE options](../reference/experimental-fixed-effects.md)
covers sample checks and numerical controls.

<!-- guide-navigation:start -->
[Previous: Mean and variance](model.md) · [Guide index](../README.md) · [Next: Cancel one block](cancellation.md)
<!-- guide-navigation:end -->
