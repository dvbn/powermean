# Choose uncertainty for the sampling design

In the [gravity example](../get-started.md), pair clustering accumulates
each directed pair's score contributions across years. Exporter-year and
importer-year effects control changes in the mean. Dependence in the
remaining scores determines the covariance request.

## Form the sandwich at the fitted point

The middle of the sandwich combines variability of the estimating-equation
contributions. Its outer factors account for how those equations change
with the coefficients. Observed bread evaluates the equation's derivative
at the fitted sample point. The robust calculation forms outer products
of row contributions. A clustered calculation first adds them within the
declared classification. [Benatia et al. (2026), Supplemental Appendix S1.4 and §5.1](../references.md)

The default `"stata"` convention uses $N/(N-1)$ for robust covariance
and $G/(G-1)$ for one-way clustering, where $G$ is the number of
clusters. In a frequency-weighted fit $N$ can differ from the number of
stored rows. The gravity example uses a normal reference. Match these
choices explicitly when comparing intervals across implementations.
[Small-sample conventions](../reference/options.md#small-sample-conventions)

In a cancelled fit the default classification is the cancelled groups.
If several cancelled groups form one independent history, cluster at
that history level when the requested method supports it. Cancellation
determines the equation and clustering determines how score contributions
are aggregated for uncertainty. The independent short-group and cell correction arguments in Benatia et al. (2026) each specify their own sampling units.
[Benatia et al. (2026), §5.2 and Appendix C.2](../references.md)

## Read uncertainty on the right scale

Use `tidy()`, `se()`, `vcov()` and `confint(level=0.95)` to inspect
pointwise coefficient uncertainty. For a proportional change use the
transformed interval from `effect()` or `contrast()`. To compare several
coefficients jointly, use a test that accounts for their covariance.
Undefined and degenerate entries have explicit statuses.
[Results](../reference/results.md#proportional-effects-contrasts-and-tests)

`with_vcov(spec)` makes a result view with newly requested inference at
the stored solution. `vcov(spec)` returns a matrix. To use either,
the result needs retained information for that request.
`keep_influence=False` can prevent later recomputation.
[Prepared-fit example](../examples/reuse-a-fit.md)

## Other covariance methods and their availability

- With independent rows, request a named HC method from `"HC0"` through
  `"HC5"` or `"HC4m"`. Each has its own leverage or finite-sample rule.
  Keep the same sample and rank when comparing these methods. With fixed
  effects, the row leverage calculation accounts for the effect geometry.
  Cancelled fits use observed bread.
- For crossed clustering use a spec such as `{"cluster": "firm + year"}`.
  The default signed CGM form adds and subtracts terms for classifications
  and intersections. With `"multiway": "ddg"`, Powermean adds the one-way
  components. In contrast, `"firm^year"` labels a single intersection
  classification. The sampling argument must match the method used.
- For signed multiway covariance, `psd` controls the handling of negative
  eigenvalues. The signed estimate can yield an undefined standard error
  when left unadjusted. Record any eigenvalue adjustment.
- WZ's pair covariance adjustment has its own three-way PPML domain.
  [Covariance refinements](covariance-refinements.md)
- HAC and Conley calculations are unavailable.

The [options reference](../reference/options.md#fit-time-covariance-for-fixed-effects)
describes when each request can be evaluated and the retained inputs it needs.

<!-- guide-navigation:start -->
[Previous: Proportional changes](proportional-effects.md) · [Guide index](../README.md) · [Next: Explore and export a fit](../examples/reuse-a-fit.md)
<!-- guide-navigation:end -->
