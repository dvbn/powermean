# Cancel one fixed-effect block

In a short firm panel, every additional firm brings a new intercept that
must be estimated from just a few years. Suppose you want to estimate how
expected sales change with a firm's price index, allowing for permanent
differences among firms and common year shifts. The cancellation method of Benatia et al. (2026)
uses a within-firm equation to estimate the slope and the year effects
without estimating a firm scale in that equation.

## Compare two equations on the same firm panel

Run this from the source root. The [standalone short-panel script](../../examples/short_panel.py)
creates 96 independent firms observed for four years, with a permanent
firm effect correlated with the price index. Prices also vary within
firms. Sales have a log-linear arithmetic mean and a simulated price
slope of −0.30.

```python
import powermean as pm

from examples.short_panel import firm_data

firms = firm_data()
firm_formula = "sales ~ 0 + price_index | firm + year"
common = {"omega": 1, "tau": 2, "cluster": "firm"}
joint_gamma = pm.fit(firm_formula, firms, **common)
cancelled_gamma = pm.fit(firm_formula, firms, cancel="firm", **common)
print(pm.etable(
    joint_gamma, cancelled_gamma, names=("Joint", "Firm cancelled"),
    uncertainty="ci", stats=("nobs", "zeros"), stars=False, digits=4,
))
```

```text
----------------------------------------------------------
                          Joint           Firm cancelled
----------------------------------------------------------
price_index              -0.3211             -0.3120
                    [-0.4377, -0.2045]  [-0.3958, -0.2282]
----------------------------------------------------------
Estimator                  GPML                GPML
Mean order (omega)          1                   1
firm effects            Estimated           Cancelled
year effects            Estimated           Estimated
Covariance            cluster: firm       cluster: firm
Convention                stata               stata
Reference                 normal              normal
Clusters                    96                  96
----------------------------------------------------------
Observations               384                 384
Zeros                      4.2%                4.2%
----------------------------------------------------------
95% confidence intervals in brackets.
```

Both fits estimate how expected sales change with prices at fixed firm and
year effects. They use the same Gamma working variance $V(m)=m^2$ and the
same cluster classification. The joint equation estimates the firm scales
and uses them in the score. The cancelled equation removes those scales.
This draw illustrates the difference in the estimates. It gives no ranking
of their accuracy or interval coverage across samples.

## Where the scale goes

For mean order one, let $B_i$ denote the within-firm deviations of the
regressors *and* the retained year indicators, after redundant columns
are removed. Let $m_i=\exp(B_i'\theta)$ and let $H_g$ include the entire
regressor history and the effect of firm $g$. The maintained model is
$E(Y_i\mid H_g)=A_g m_i$ at the true parameter. The relative outcomes
$Y_i/m_i$ thus have common conditional mean $A_g$ within the firm.

For working variance $V(m)$, center $B_i$ using weights
$w_i=m_i^2/V(m_i)$ within $g$. The group score sums
$w_i(B_i-\bar B_{g,w})Y_i/m_i$. Its conditional expectation is zero at
the true parameter because the weighted centered deviations sum to zero.
At `tau=2` those weights are one. Profiling joint GPML instead multiplies
each group score by an inverse fitted group scale, which depends on the
observed outcomes. At `tau=1` this additional group factor is one on a
common sample with finite scales, giving the [PPML identity](fixed-effects.md#compare-joint-estimation-with-pair-cancellation).
[Benatia et al. (2026), §5.1, equations (5.1)–(5.4), and Supplemental Appendix S3.6.1, equations (S3.18)–(S3.19)](../references.md)

## Choose the group and the sampling unit

Use the name in the formula, here `cancel="firm"`. The other block stays
estimated. Cancelling years would leave a growing set of short firm effects
to estimate. With many independent short firms and a fixed number of years,
cancelling firms leaves a fixed-dimensional problem. Proposition 7 of Benatia et al. (2026)
uses a mean restriction conditional on the full history, independent
groups, bounded group sizes, moments and identifying within-group
variation. It permits dependence across years within a firm. Choosing the
group with the most levels via `cancel=True` is a software convenience.
The study design determines whether that block has the required properties.

A regressor constant within a cancelled group contributes no within-group
variation. Singletons and all-zero groups add no slope information to the
cancelled equation. Check the usable sample, the remaining design rank and
the covariance classification before interpreting the fit.
[Benatia et al. (2026), §5.1 and Supplemental Appendix S3.3](../references.md)

When you cancel firm fixed effects in PPML and all outcomes are positive,
a firm observed once stays in the sample and contributes zero to the slope score.
If you cluster by firm, that firm still counts toward the number of clusters
`G` in the default `G/(G−1)` adjustment. Joint PPML drops singleton firms
by default. The two fits can consequently agree on slopes and differ in
cluster-adjusted standard errors. When a zero outcome remains after the
initial sample filters, both routes apply the joint singleton and separation rules before
fitting. Inspect `result.sample.table()` and `result.vcov().attrs["G"]` to
see the sample and cluster count used for your fit.

## Use the cancelled result

To predict sales for a year in the panel, Powermean combines the price
slope, the estimated year effect and the firm scale recovered from that
firm's sales. `cancelled_gamma.predict()` gives fitted sales for every
retained row. `cancelled_gamma.fixef()["firm"]` gives one recovered log
effect per firm. With only four years per firm, these levels can be noisy.
[Predict sales at a higher price](../reference/results.md#predict-from-a-cancelled-gamma-fit) ·
[How the firm scale is recovered](../reference/results.md#recovered-cancelled-effects)

In gravity the exporter-year and importer-year blocks can grow even after
cancelling pairs. Their estimation can still shift the slope distribution.
[Coefficient bias](bias-correction.md) explains that case and the optional
short-group correction.

<!-- guide-navigation:start -->
[Previous: Estimate fixed effects](fixed-effects.md) · [Guide index](../README.md) · [Next: Nonlinear panel DiD](../examples/nonlinear-did.md)
<!-- guide-navigation:end -->
