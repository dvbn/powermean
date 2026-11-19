# Explore and export a fitted result

This small no-FE example holds the design and equation fixed across
Newton and IOLS. It then computes a proportional change between two
covariate configurations. The groups in the simulation let you exercise
the cluster interface; its rows are generated independently.

## Solve the same mean equation twice

Both solvers use the same input frame. Once both fits are available, check
`newton.sample.positions.tolist() == iols.sample.positions.tolist()` before
comparing their coefficients.

```python
import numpy as np
import pandas as pd
import powermean as pm

rng = np.random.default_rng(20261028)
x1, x2 = rng.normal(size=(2, 240))
mean = np.exp(0.3 + 0.25 * x1 - 0.15 * x2)
data = pd.DataFrame(
    {"y": rng.poisson(mean), "x1": x1, "x2": x2, "group": np.repeat(np.arange(60), 4)}
)
prepared = pm.prepare("y ~ 1 + x1 + x2", data, cluster="group")
newton = prepared.fit(omega=1, tau=1)
iols = prepared.fit(omega=1, tau=1, solver="iols")
print(pm.etable(
    newton, iols, names=("Newton", "IOLS"), stars=False, digits=4,
))
```

```text
--------------------------------------------------
                        Newton           IOLS
--------------------------------------------------
Intercept               0.3377          0.3377
                       (0.0621)        (0.0621)
x1                      0.2245          0.2245
                       (0.0537)        (0.0537)
x2                     -0.2494         -0.2494
                       (0.0737)        (0.0737)
--------------------------------------------------
Estimator                PPML            PPML
Mean order (omega)        1               1
Covariance          cluster: group  cluster: group
Convention              stata           stata
Reference               normal          normal
Clusters                  60              60
--------------------------------------------------
Observations             240             240
--------------------------------------------------
Standard errors in parentheses.
```

Both calls estimate the same PPML equation using the retained design and
cluster classification. This example used the compiled IOLS implementation.
An artifact with the NumPy implementation can report `numpy` instead.
[Supported IOLS domain](../methods/numerical-methods.md#iols-an-optional-no-fe-solver)

## Inspect a fit and change its covariance

```python
print(newton.tidy().to_string(index=False))
```

```text
     term  estimate  std_error  statistic      p_value  conf_low  conf_high inference
Intercept  0.337672   0.062123   5.435507 5.464090e-08  0.215913   0.459432        ok
       x1  0.224509   0.053719   4.179356 2.923359e-05  0.119223   0.329796        ok
       x2 -0.249392   0.073739  -3.382119 7.192900e-04 -0.393917  -0.104868        ok
```

The coefficient table gives the estimate, standard error, test and
confidence interval for each term. Use `newton.coef()`, `newton.se()`
and `newton.confint()` when you need one of those objects on its own.
Use `newton.vcov()` for the active covariance matrix, including off-diagonal
entries. Use `robust_matrix = newton.vcov("robust")` to get the alternate
matrix from the same fitted point, with covariance metadata in
`robust_matrix.attrs`. The [results reference](../reference/results.md) also
explains `predict()`, `fixef()`, `influence()` and the numerical certificate.

Use `with_vcov()` to carry the alternate covariance into a separate result
view. Both covariance calls reuse the fitted point without refitting.

```python
robust = newton.with_vcov("robust")
print(robust.confint().reset_index().to_string(index=False))
```

```text
     term     lower     upper
Intercept  0.215915  0.459430
       x1  0.120539  0.328480
       x2 -0.378758 -0.120027
```

The intervals above use independent-row robust covariance. Use
`robust.tidy()` for coefficients with the alternate uncertainty or
`pm.etable(robust, uncertainty="ci")` to report the intervals in a model
table. The effects and tests below also use the view's covariance.
[Post-fit covariance requests](../reference/results.md#recompute-uncertainty-from-stored-information)

See [compare estimation samples](compare-samples.md) when two models were
fitted on different rows.

## Read diagnostics and covariance

```python
print(newton.vcov().round(6).to_string(index_names=False))
print(newton.estimator.to_dict())
record = newton.provenance.to_dict()
print({key: record[key] for key in ("solver", "status", "certified")})
```

```text
           Intercept        x1        x2
Intercept   0.003859 -0.000807  0.000518
x1         -0.000807  0.002886 -0.000370
x2          0.000518 -0.000370  0.005437
{'type': 'Estimator', 'mean': {'type': 'PowerMean', 'omega': 1.0}, 'variance': {'type': 'PowerVariance', 'tau': 1.0}, 'moment': {'type': 'ExogenousMean'}, 'fixed_effects': {'type': 'NoFixedEffects'}}
{'solver': 'newton', 'status': 'verified', 'certified': True}
```

The row and column names of the covariance matrix are fitted term names.
Use its `attrs` to read the covariance metadata. The estimator record
describes the equation, while provenance records how it was solved.

## Report effects and contrasts

```python
change = robust.effect("x1", contrast=(0, 1))
print(change.estimate, change.interval)
```

```text
0.25170852542599226 (0.12810458307540534, 0.38885548036053386)
```

`effect()` compares two values of one regressor that enters directly.
Use `effects()` for a table of separate one-unit changes.

```python
print(robust.effects(["x1", "x2"]).T.to_string(header=False))
```

```text
term                                                                                  x1                                                                    x2
target              theta_N/NATE: proportional effect on the conditional arithmetic mean  theta_N/NATE: proportional effect on the conditional arithmetic mean
contrast                                                                      (0.0, 1.0)                                                            (0.0, 1.0)
estimate                                                                        0.251709                                                             -0.220726
std_error                                                                         0.0664                                                              0.051435
statistic                                                                       4.232259                                                             -3.778441
p_value                                                                         0.000023                                                              0.000158
conf_low                                                                        0.128105                                                             -0.315289
conf_high                                                                       0.388855                                                             -0.113103
inference                                                                             ok                                                                    ok
log_contrast                                                                    0.224509                                                             -0.249392
log_std_error                                                                   0.053047                                                              0.066004
reference                                                                         normal                                                                normal
reference_df                                                                        None                                                                  None
units                                                                       proportional                                                          proportional
unavailable_reason                                                                  None                                                                  None
```

Use `contrast()` for a change involving several coefficients. Supply complete
before and after regressor settings, including for a change to one variable
that enters through an interaction or transformation.

```python
comparison = robust.contrast({"x1": 0.0, "x2": 0.0}, {"x1": 1.0, "x2": -1.0})
print(comparison.to_frame().T.to_string(header=False))
```

```text
target              theta_N/NATE: proportional effect on the conditional arithmetic mean
contrast                      X(after) - X(before); fixed effects and offsets held fixed
before                                                            {'x1': 0.0, 'x2': 0.0}
after                                                            {'x1': 1.0, 'x2': -1.0}
estimate                                                                        0.606249
std_error                                                                       0.144593
statistic                                                                       5.264459
p_value                                                                              0.0
conf_low                                                                        0.346445
conf_high                                                                       0.916185
inference                                                                             ok
log_contrast                                                                    0.473902
log_std_error                                                                   0.090019
reference                                                                         normal
reference_df                                                                        None
units                                                                       proportional
unavailable_reason                                                                  None
```

The log mean ratio for these configurations is
$\widehat\beta_{x1}-\widehat\beta_{x2}$. The fit replays the formula on
both settings and holds fixed effects and offsets fixed where present.
The estimate and interval are proportions. The default interval transforms
the endpoints of the log-scale interval.
[Effect and contrast methods](../reference/results.md#proportional-effects-contrasts-and-tests)

## Test or combine coefficients

```python
print(robust.test("x1 = x2").to_frame().to_string(index=False))
print(robust.lincom("x1 - x2").to_frame().to_string(index=False))
```

```text
   term  statistic      p_value  wald_quadratic distribution  df_num df_denom  requested  reduction inference reference reference_df unavailable_reason
x1 = x2   27.71453 1.406027e-07        27.71453         chi2       1     None          1          0        ok    normal         None               None
   term  estimate  std_error  statistic      p_value  conf_low  conf_high inference reference reference_df                  units unavailable_reason
x1 - x2  0.473902   0.090019   5.264459 1.406027e-07  0.297468   0.650336        ok    normal         None coefficient expression               None
```

`test()` tests an equality using the full covariance. `lincom()` reports
a linear combination, with its standard error and interval. For a
nonlinear coefficient expression use `nlcom()`. For example,
`robust.nlcom("exp(x1 - x2) - 1")` evaluates the same point contrast here
with a delta-method interval. Use `wald(R, q)` for joint restrictions.
Here the columns are intercept, `x1` and `x2`, in the order of
`robust.coef()`. The rows below test both slopes against zero.

```python
print(robust.nlcom("exp(x1 - x2) - 1").to_frame().to_string(index=False))
print(robust.wald([[0, 1, 0], [0, 0, 1]], [0, 0]).to_frame().to_string(index=False))
```

```text
            term  estimate  std_error  statistic  p_value  conf_low  conf_high inference reference reference_df                  units unavailable_reason
exp(x1 - x2) - 1  0.606249   0.144593   4.192795 0.000028  0.322852   0.889647        ok    normal         None coefficient expression               None
term  statistic      p_value  wald_quadratic distribution  df_num df_denom  requested  reduction inference reference reference_df unavailable_reason
wald   28.43249 6.698278e-07        28.43249         chi2       2     None          2          0        ok    normal         None               None
```

[Results and tests](../reference/results.md#proportional-effects-contrasts-and-tests)

## Inspect predictions and influence contributions

```python
print(robust.predict(pd.DataFrame({"x1": [0.0], "x2": [0.0]})))
print(newton.influence().head(2).reset_index().to_string(index=False))
```

```text
0    1.401681
Name: power_mean, dtype: float64
 position  Intercept        x1        x2
        0  -0.004538  0.001877  0.001345
        1   0.001679 -0.001195 -0.000852
```

The prediction is the fitted arithmetic mean at the supplied regressor
values. The influence rows give the first two stored sampling-unit
contributions of the clustered fit. Use `fixef()` on an FE result to
inspect its normalized or recovered group levels.
[Fixed-effect results](../reference/results.md#fixed-effect-results)

## Export tables

```python
from pathlib import Path

table = pm.etable(newton, iols, names=("Newton", "IOLS"), stars=False, digits=4)
Path("models.md").write_text(table.to_markdown(), encoding="utf-8")
Path("models.tex").write_text(table.to_latex(), encoding="utf-8")
Path("models.html").write_text(table.to_html(), encoding="utf-8")
robust.tidy().to_csv("coefficients.csv", index=False)
robust.vcov().to_csv("covariance.csv")
```

In a notebook, display the table object for its HTML representation.
Use `print(table)` in a console and `table.to_frame()` for numeric values
and model metadata. See the [rendered Markdown example](compare-samples.md#export-a-table-to-markdown).
[Table export formats](../reference/tables.md#print-or-export)

<!-- guide-navigation:start -->
[Previous: Compare samples](compare-samples.md) · [Guide index](../README.md) · [Next: Proportional changes](../methods/proportional-effects.md)
<!-- guide-navigation:end -->
