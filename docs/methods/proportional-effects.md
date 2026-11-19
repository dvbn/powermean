# Interpret proportional changes

The [gravity example](../get-started.md) estimates a conditional mean ratio.
A population comparison requires a denominator and a population over which
to average. Benatia et al. (2026) show how those choices determine the parameter.

## Compare population means

For finite means and a positive untreated population mean, the normalized
average treatment effect (NATE) is

$$
\theta_N=\frac{E[Y(1)]}{E[Y(0)]}-1.
$$

It also gives the proportional change in the expected total for a fixed-size
population. For example, take two units whose untreated outcomes are 1
and 9 and whose treated outcomes are 2 and 9. Their mean rises from 5 to
5.5, a 10% increase. Averaging the individual percentage changes gives
50%. When individual baselines are positive, the ratio of means weights
individual percentage effects by baseline outcomes. If one individual's
baseline is zero, the ratio of population means can still be defined.

For a nonnegative outcome the mean in each state factors into the share with
positive outcomes and the mean among those positive outcomes. The two factors
can both change. The set of positive units can also change.
[Benatia et al. (2026), §2 and Supplemental Appendix S1.1.3, equations (S1.2) and (S1.4)](../references.md)

## When an exponentiated coefficient is enough

Suppose a binary treatment enters the exponential conditional arithmetic
mean as $\mu_d(w)=\exp(\alpha+\beta_D d+w'\gamma)$. Treatment multiplies
each fitted conditional mean by $\exp(\beta_D)$. Averaging over the same
covariate distribution preserves that factor. In a causal design with
consistency, conditional mean exchangeability and overlap for the intended
population, $\exp(\beta_D)-1$ then gives the NATE. Other designs can identify
the two means by different assumptions, such as the untreated trend and no
anticipation in [panel DiD](../examples/nonlinear-did.md).

Even when the conditional mean ratio is constant, individual outcomes can
respond differently. For an uninteracted numeric regressor, `effect()`
reports the change for one unit or a specified pair of values. Use
`contrast()` for an interacted, categorical or transformed regressor.
[Benatia et al. (2026), §3.1 and Supplemental Appendix S1.1.4](../references.md)

## With interactions, standardize the fitted means

If treatment multiplies the mean by different amounts across covariates,
evaluate the fitted means for both states at each covariate value in the
target population. Average the treated predictions and the untreated
predictions separately. Form their ratio and subtract one. A level
effect uses the difference of those means instead. For nonzero power
orders, average each state's conditional power moment before taking
its power root. [Benatia et al. (2026), equations (3.1)–(3.2)](../references.md)

Uncertainty for an average of counterfactual predictions uses the derivatives
of the full calculation and, when applicable, sampling variability of the
target covariates. The result method `contrast(before, after)` reports the
ratio for two supplied regressor configurations. General standardized
effects and their uncertainty require a separate calculation.
[Result methods](../reference/results.md#proportional-effects-contrasts-and-tests)

<!-- guide-navigation:start -->
[Previous: Explore and export results](../examples/reuse-a-fit.md) · [Guide index](../README.md) · [Next: Nonlinear panel DiD](../examples/nonlinear-did.md)
<!-- guide-navigation:end -->
