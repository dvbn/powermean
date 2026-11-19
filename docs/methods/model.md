# Choose the mean order and working variance

Start with the quantity you want to compare as a regressor changes.
In the [gravity example](../get-started.md) it is expected trade in levels.
A positive mean order allows zeros, and the working variance determines how
the estimating equation uses residuals.

## Choose a conditional mean

For $\omega\ne 0$, the regression assumes
$E(Y^\omega\mid X)=\exp(\omega X'\beta)$, where $X'\beta$ can include
fixed effects or a known offset. The conditional power mean is the
$1/\omega$ power of that moment, so Powermean reports slopes on its log scale.

| Order | Fitted target | Outcome domain |
| --- | --- | --- |
| `omega=1` | Arithmetic conditional mean | Nonnegative |
| `omega=0.5` | Square of the conditional mean of $\sqrt Y$ | Nonnegative |
| `omega=0` | Mean of log outcomes, giving the geometric mean after exponentiation | Strictly positive |
| `omega=-1` | Harmonic conditional mean | Strictly positive, with finite inverse moment |

When a fit succeeds with `0 < abs(omega) <= 1e-8`, Powermean issues a numerical
warning. Raising a positive outcome to a power very close to zero can round
away differences between observations. Dividing by the requested order to
report coefficients and standard errors can magnify that rounding. The
warning threshold is advisory. Use `omega=0` if the desired model is log OLS.

A change in order generally changes the population target and the
conditional moment restriction. Recovering the arithmetic mean from a
log regression also requires the conditional retransformation factor
$E[\exp(\varepsilon)\mid X]$. With zeros, a log model needs a separately
justified positive-outcome sample. A fixed shift such as `log(Y + 1)`
changes the target and its response to outcome units.
[Benatia et al. (2026), §§2.2, 3.1 and 3.3](../references.md)

## Set weights in the estimating equation

At order one, the joint power quasi-score sums
$X_i m_i^{1-\tau}(Y_i-m_i)$, where $m_i=\exp(X_i'\beta)$ and
$V(m)=m^\tau$. The working variance specifies weights. The actual
conditional variance describes the data. The PPML and Gamma equations
can share a population root under a correctly specified mean while
responding differently to the same data. Consistency of the estimator
also depends on how the fixed effects grow. The short-panel argument for
PPML cannot be transferred directly to joint GPML.
[Benatia et al. (2026), §3.2, Proposition 3 and §5](../references.md)

| Argument in `pm.fit` | Equation at order one | Shortcut |
| --- | --- | --- |
| `omega=1, tau=1` | PPML with level residual $Y_i-m_i$ | `pm.ppml` |
| `omega=1, tau=2` | Gamma quasi-score with relative residual $Y_i/m_i-1$ | `pm.gpml` |
| `omega=1, tau=1.5` | Power quasi-score with $V(m)=m^{1.5}$ | |
| `omega=1, variance=pm.NB2(0.5)` | Working variance $m+0.5m^2$ | |

Supply `tau` or `variance`. At order zero, use `pm.fit(..., omega=0)`
or its shortcut `pm.logols`. Order zero uses least squares on log outcomes.

Both PPML and Gamma quasi-scores can include zeros. When there is just
an intercept and a single binary regressor with positive means in both
groups, PPML and GPML set each fitted mean equal to its group mean.
With additional regressors, the difference in weights can yield different
coefficients. [Benatia et al. (2026), Supplemental Appendix S1.4, especially §S1.4.3](../references.md)

Run the named estimators on the same positive-outcome data to read their
resolved working equations. The NB2 fit uses the general entry point.

```python
import pandas as pd
import powermean as pm

data = pd.DataFrame({"y": [1, 3, 2, 4, 4, 7, 8, 10], "x": list(range(8))})
models = [
    pm.ppml("y ~ x", data),
    pm.gpml("y ~ x", data),
    pm.logols("y ~ x", data),
    pm.fit("y ~ x", data, omega=1, variance=pm.NB2(kappa=0.5)),
]
print(pm.etable(models, names=("PPML", "Gamma PML", "Log OLS", "NB2"), stars=False))
```

```text
-----------------------------------------------------------------
                      PPML   Gamma PML  Log OLS        NB2
-----------------------------------------------------------------
Intercept            0.450     0.402     0.329        0.423
                    (0.199)   (0.254)   (0.235)      (0.231)
x                    0.271     0.283     0.295        0.277
                    (0.034)   (0.048)   (0.044)      (0.042)
-----------------------------------------------------------------
Estimator             PPML      GPML    log OLS  NB2, kappa = 0.5
Mean order (omega)     1         1         0            1
Covariance           robust    robust    robust       robust
Convention           stata     stata     stata        stata
Reference            normal    normal    normal       normal
-----------------------------------------------------------------
Observations           8         8         8            8
-----------------------------------------------------------------
Standard errors in parentheses.
Models with different mean orders describe different conditional power means; their coefficients do not estimate one common parameter.
```

## Hold the formulation fixed when you compare weights

Compare `tau` values on the same sample and fixed-effect specification.
For example, use `pm.fit(formula, trade, omega=1, tau=2, cluster="pair")`
on the [gravity data](../get-started.md) to obtain joint GPML. For a short
panel the [cancellation chapter](cancellation.md) compares two different
formulations at a single working variance.

For joint fits with nonzero order and a power variance outside `[1, 2]`,
Powermean uses a selected-root procedure. Check identification, existence and
the reported numerical assessment before interpreting the result.
Cancellation has its own domains and starting rules. Use explicit variance
choices: omitting both `tau` and `variance` in a cancelled `pm.fit` call
selects `tau=2`. See [options](../reference/options.md#full-family-scope-and-the-current-entry-restriction)
and [numerical methods](numerical-methods.md).

## Fit one conditional mean three ways

Suppose `y` counts orders and `x` is a strictly exogenous binary regressor. The data below include a zero in each group. Fit the log-linear arithmetic mean with an intercept, first by PPML and then with two other working variances.

```python
import pandas as pd
import powermean as pm

orders = pd.DataFrame({
    "x": [0] * 6 + [1] * 6,
    "y": [0, 1, 2, 3, 4, 2, 0, 2, 3, 5, 6, 8],
})
ppml = pm.fit("y ~ x", orders, omega=1, tau=1)
intermediate = pm.fit("y ~ x", orders, omega=1, tau=1.5)
gamma = pm.fit("y ~ x", orders, omega=1, tau=2)
change = ppml.effect("x", contrast=(0, 1))
```

The fitted means are 2 orders at `x=0` and 4 at `x=1`. With a binary regressor and an intercept, each estimating equation fits the two group means exactly. Read `ppml.tidy()` for the coefficient on the log-mean scale. Multiply `change.estimate` and each endpoint of `change.interval` by 100 for the fitted percentage change and its pointwise interval. Here the fitted increase is 100%. With more covariates the choice of working variance can affect the estimated slope.

For strictly positive outcomes, log OLS targets the conditional mean of `log(y)`. [Mean orders and outcome domains](#choose-a-conditional-mean)

<!-- guide-navigation:start -->
[Previous: Run the examples](../examples/README.md) · [Guide index](../README.md) · [Next: Fixed effects](fixed-effects.md)
<!-- guide-navigation:end -->
