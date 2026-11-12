# Powermean

Powermean is a Python package for estimating proportional relationships and
effects in regression models. It brings exponential regression, including
Poisson pseudo-maximum likelihood (PPML) and Gamma pseudo-maximum likelihood
(GPML), and log-linear regression into a common power-mean framework. The
methods build on Benatia, Bellégo and Pape (BBP),
[*Dealing with Logs and Zeros in Regression Models*](https://arxiv.org/abs/2203.11820).

> **Status: pre-release `0.1.0a1`.** This version estimates models without
> fixed effects. A formula with fixed effects is refused with a clear message;
> fixed effects are planned for version 0.1. Interfaces may still change.

## From the question to the model

Suppose you want to know how sales change proportionally with a firm's
characteristics. A change in average sales and an average change in log sales
are different quantities. That distinction matters especially when some firms
have no sales: a model for the arithmetic mean can include those zeros,
whereas a regression of log sales requires positive outcomes.

Powermean makes this choice explicit through the **mean order**, `omega`.
Order one models the arithmetic conditional mean; order zero gives a
regression of the log outcome. Other orders describe other power means.
Choosing an order therefore starts with the quantity you want to study, rather
than with the method that happens to be easiest to compute.

Once the mean model is specified, the **working variance** determines the
weights in its estimating equation. At order one, PPML and GPML both model the
arithmetic conditional mean, but use different working variances. Under the
appropriate mean, identification and regularity conditions, they estimate the
same population coefficients; their estimates and precision can differ in a
sample. This is a choice of estimator for the model, separate from choosing
which mean to model.

In an exponential mean model without interactions, exponentiating a slope
gives the conditional mean ratio for a one unit increase in that regressor,
holding the others fixed. A causal interpretation requires an identification
argument in addition to the mean specification.

## Install

Powermean requires Python 3.12 or later. This alpha is available from source
for testing; it is not published to a package index. From the repository root:

```bash
python -m pip install .
```

The package is pure Python and depends on NumPy, pandas, SciPy and Formulaic.
The CI matrix is configured for Python 3.12–3.14 on Linux, macOS and Windows.

## Fit and interpret a first model

```python
import numpy as np
import pandas as pd
import powermean as pm

rng = np.random.default_rng(20261028)
n = 240
x1, x2 = rng.normal(size=n), rng.normal(size=n)
mean = np.exp(0.3 + 0.25 * x1 - 0.15 * x2)
data = pd.DataFrame({"y": rng.poisson(mean), "x1": x1, "x2": x2})

result = pm.ppml("y ~ 1 + x1 + x2", data, vcov="robust")
print(result.summary())
print(result.effect("x1").to_dict())
result.tidy().to_csv("coefficients.csv", index=False)
```

The fit uses all 240 observations, including 66 zeros. The coefficient on `x1`
is 0.224509 with robust standard error 0.053047. The corresponding conditional
mean ratio is approximately 1.252: a one unit increase in `x1` is associated
with a 25.2% higher mean, holding `x2` fixed.

`effect("x1")` reports the proportional change and its interval, rather than
leaving the exponential transformation to you. It is a conditional comparison,
not an automatically identified causal effect.

## What this version does

| Task | Interface |
| --- | --- |
| Fit PPML, GPML or log OLS | `pm.ppml`, `pm.gpml`, `pm.logols`; or specify the mean order and working variance with `pm.fit`. |
| Use IOLS instead of Newton | `solver="iols"`, for working variances `tau` in `[1, 2]` and positive mean orders, without weights or offsets. Newton is the default. |
| Compare specifications on one prepared sample | `pm.prepare(...).fit(...)`. |
| Change the covariance without refitting | `result.vcov(spec)` for a matrix, or `result.with_vcov(spec)` for tables, intervals and tests using that covariance: robust, HC0 to HC5, one-way and multiway clustering. |
| Report proportional changes and test coefficients | `effect`, `effects`, `contrast`, `lincom`, `nlcom`, `test` and `wald`. |
| Predict | `result.predict()`, in sample or on new data. |

At nonzero mean orders, Newton accepts any finite power `tau`. Outside `[1, 2]`
the fit reports a selected root: one root of the estimating equation, not
promised unique or available for every sample. The result records this, and
`summary()` prints it as `Coverage: selected_root`. The fit also issues a
`CoverageWarning` when `tau < 1`, and when `tau > 2` and the sample has zero
outcomes; with `tau > 2` and positive outcomes only, it issues none.

## Known limits of this version

- No fixed effects, instruments, HAC or spatial covariances.
- IOLS runs in NumPy only. This version ships no compiled IOLS kernel and
  refuses `iols_backend="cpp"`. Its IOLS results were checked against the NumPy
  kernel only, not against a compiled one, and its speed was not measured.
  Newton is the default solver.
- A regressor that is an exact linear combination of the others is not dropped:
  the fit stops without a certified solution.
- A sample in which the estimate does not exist (separation) is not diagnosed
  before the fit: the fit stops without a certified solution.

## Citation

If you use the package, please cite the paper: David Benatia, Christophe
Bellégo and Louis-Daniel Pape, *Dealing with Logs and Zeros in Regression
Models*, [arXiv:2203.11820](https://arxiv.org/abs/2203.11820).

## Help

Report a problem with a small simulated or redistributable example through
[Issues](https://github.com/dvbn/powermean/issues), without confidential data.

## License

BSD-3-Clause; see [LICENSE](LICENSE). Bundled scikit-learn components keep their
[third party notice](src/powermean/_vendor/SKLEARN_COPYING).
