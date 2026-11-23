# Results

Start with `fit.summary()` for the printed regression or `fit.tidy()` for
a table of coefficients and confidence intervals. To see what the model
predicts for each observation, go to [fitted values, residuals and
counterfactuals](#prediction-and-retention). To find out why the fit used
fewer rows than you supplied, [inspect the estimation sample](#inspect-the-estimation-sample).

| Method or property | Contents |
| --- | --- |
| `coef()` | Coefficients as a pandas Series indexed by term. |
| `se()` | Standard errors of the active covariance: the one requested at the fit, unless `with_vcov` selected another. |
| `vcov()` | Covariance DataFrame, with convention metadata in `attrs`. |
| `vcov(spec)` | A requested covariance matrix from retained information. In no-FE fits, `ssc` selects the convention and `bread` selects the bread for a named HC method. |
| `with_vcov(spec)` | A separate result view whose uncertainty uses the requested covariance. Both calls reuse the fitted point. [Recompute uncertainty](#recompute-uncertainty-from-stored-information) |
| `confint(level=0.95)` | A DataFrame with `lower` and `upper`. |
| `tidy(level=0.95)` | Coefficients and uncertainty in one table. |
| `summary(level=0.95)` | Text summary, displayed with `print(result.summary())`. |
| `predict(data=None, type="power_mean")` | Fitted conditional power means, fixed-effect levels or response residuals. Replacement covariates are supported for fitted means. |
| `influence(unit="sampling")` | Retained first-order contributions by sampling unit and coefficient. A cancelled result gives one row per cancelled group by default. Use `unit="row"` for contributions indexed by original row position. |
| `fixef()` | Joint FE: normalized estimated levels. Cancelled and two-way GMM: includes ex-post recovered levels. Available for fits with fixed effects. |
| `effect`, `effects`, `contrast` | Proportional changes on the fitted conditional power mean; see [below](#proportional-effects-contrasts-and-tests). |
| `lincom`, `nlcom`, `test`, `wald` | Coefficient combinations and restriction tests. |
| `sample.table()` | Counts by exclusion reason, in processing order. |
| `sample.positions`, `sample.labels` | Zero-based positions and original index labels of retained rows. |
| `sample.reason(position)` | First recorded exclusion reason, or `None` for a retained row. |
| `estimator` | Resolved order and working variance. |
| `certificate` | Numerical assessment, when available. |
| `provenance` | Recorded computation information. |
| `provenance.coverage` | Root-theory label and basis, such as `unique_root` or `selected_root`; unrelated to confidence interval coverage. |

On fixed-effect log OLS, the covariance metadata, `vcov().attrs` and
`provenance.covariance`, holds two counts. `rank` is the exact rank of the
design, the slopes and the absorbed effects, where the request forms it (HC2
to HC5, or `ssc=pm.ssc(..., absorbed_count="exact")`), and `None` otherwise.
`K` is the parameter count that the covariance reads, present only on a
request whose factor or reference reads one: HC1's `N/(N-K)`, the factors
`N/(N-K*)` of the `"regression"` convention and the Student t reference. It
counts the absorbed effects as `fixest` and `pyfixest` do by default, the
levels less one per further block, or by their exact rank with
`absorbed_count="exact"`. On the same requests, and only there,
`provenance.covariance["absorbed_count"]` names the count, `summary()` prints
one line on it and `etable` adds the row "Absorbed count"; a request that reads
no count, such as the default `"stata"` robust one, shows none, as the tables
of `fixest` and `pyfixest` show no count.

## Inspect the estimation sample

See [Compare estimation samples](../examples/compare-samples.md) for
`sample.table()`, `sample.reason()` and the retained row positions and
labels. For two fits on the same input frame in the same order, compare
`a.sample.positions.tolist() == b.sample.positions.tolist()` before reading
the coefficients side by side. Pass a zero-based input position to
`sample.reason(position)` to check a particular exclusion.
[Worked row comparison](../examples/compare-samples.md#find-the-rows-each-fit-used)

Use `sample=fit` with the original frame in `pm.summary_statistics`,
`pm.outcome_table`, `pm.balance_table` or `pm.balance_regression` to describe
that fit's rows. [Descriptive table calls](tables.md#describe-data-and-treatment-arms)

| Sample member | What it reports |
| --- | --- |
| `nobs`, `count`, `effective_n`, `zeros` | Input rows, retained rows, the exact total of frequency counts for frequency-weighted fits (otherwise the retained count), and retained zero outcomes. |
| `counts`, `missing` | Exclusion counts by step and missing-value counts by input role. Missing-role counts can overlap. |
| `positions`, `labels`, `input_labels` | Retained input positions, retained index labels and the full input index. |
| `retained_index`, `codes`, `reason(i)` | For each input row, its position in the retained sample or `-1`, its exclusion code (zero if retained), or its first exclusion reason. |
| `fingerprint` | A fingerprint of the fitted sample and design. Use positions to check whether two different specifications kept the same rows. |
| `omega`, `positive_only`, `conditioned` | The mean order, the positive-outcome request and whether it removed a zero. |
| `column_names`, `dropped_collinear`, `omitted_slopes` | Design names, names omitted for collinearity, and name–reason records. FE views also give `active_names`. |
| `weight_kind`, `weights`, `cluster_names` | Applied weight type and retained weights, when present, and the recorded clustering classifications. |
| `columns`, `subset_missing` | The input roles read by the no-FE fit and the number of missing subset flags. |
| `reasons`, `dropped_singleton`, `dropped_separated`, `dropped_separated_by_fixed_effect` | Fixed-effect row reasons and counts from the fitted sample. |
| `source`, `grouping`, `cancelled_name`, `retained_names`, `accounting` | Where available on FE views, source and retained grouping records, the chosen cancellation block, the estimated blocks and the cancellation accounting. |

`sample.grouping` can expose block `names` and `n_levels` on retained rows.
Use the [joint summary](#fixed-effect-block-counts-in-the-summary) for the
per-block redundancy and nesting report.

## Read the estimator and diagnostics

`fit.estimator` records `mean`, `variance`, `moment` and `fixed_effects`.
Its `route` is `"linear"` at order zero and `"power"` otherwise.
Use `estimator.to_dict()` for a structured record and
`type(fit.estimator).from_dict(record)` to reconstruct the specification.

`fit.provenance.to_dict()` returns the computation record. Its sections cover
software `versions` and `implementation`, the `solver` and its `options`
and `trace`, the `linear` calculation when applicable, `requested_start`,
`formula`, `estimator`, `sample`, `covariance`, and `references`. Read
`status`, `level`, `certified`, `degenerate`, `coverage` and `failure` to
interpret the result. The record can also include `carried_files`.
Optional entries depend on the fit type. [Worked diagnostics](../examples/reuse-a-fit.md#read-diagnostics-and-covariance)

`fit.certificate` contains the numerical assessment when the route supplies
one. Read it with the reported status and separate coverage label.
[Numerical assessment](../methods/numerical-methods.md#what-the-convergence-assessment-checks)

## Export tables

See the [table reference](tables.md) for all table options and formats.

### Inspect balance on the baseline sample

The [two-period panel](../examples/nonlinear-did.md) starts with higher
expected visits in the treated cohort. Use a printed table to check that
baseline difference before looking at changes over time. Take the period-1
rows, one per unit. The default control arm, `cohort=0`, identifies
never-treated units. `cohort=2` identifies units treated in period 2.

```python
import powermean as pm

from examples.nonlinear_did import did_data

panel = did_data()
baseline = panel.loc[panel["period"].eq(1)]
balance = pm.balance_table(baseline, treatment="cohort", vars=("visits",))
print(balance)
```

```text
Balance table        control: cohort = 0 (N = 80); cohort = 2 (N = 80)
----------------------------------------------------------------------
        mean (0)  mean (2)  difference  std. err.  p-value  std. diff.
----------------------------------------------------------------------
visits      2.17      3.19        1.01       0.29   <0.001       0.559
----------------------------------------------------------------------
1 variable(s). Joint test: pm.balance_regression.
Differences from the control arm; standard errors and p-values from the two-sample comparison with unequal variances (HC2 variance, Welch degrees of freedom).
Standardized difference: difference / sqrt((s_a^2 + s_0^2)/2); not a test.
```

The difference describes the baseline outcome. The DiD assumption concerns
how the untreated mean would have grown over time. Use `pm.balance_regression` for the
linear probability regression of assignment on baseline covariates.
Here the regression has one baseline covariate. Adding further
covariates to `vars` expands the joint test shown in the table.

```python
balance_fit = pm.balance_regression(baseline, treatment="cohort", vars=("visits",))
print(balance_fit)
```

```text
Balance regression    control: cohort = 0
-----------------------------------------
                           cohort = 2
-----------------------------------------
visits                       0.072
                            (0.018)
Intercept                    0.306
                            (0.064)
-----------------------------------------
Joint test              chi2(1) = 16.02
p-value                      <0.001
Observations                  160
Rows dropped (missing)         0
-----------------------------------------
Linear probability model of each arm against the control on all variables, on the rows of the two arms.
Covariance: robust to heteroskedasticity, convention ssc='stata'. Joint Wald test that every coefficient on the variables is zero.
Standard errors in parentheses.
```

`balance_fit` is a table. Read the joint Wald test and its sample counts with
`balance_fit.to_frame().attrs["joint_test"]`.
[Balance table options](tables.md#describe-data-and-treatment-arms)

### Save a fitted model's output

`pm.etable` returns a display-ready table. Its fixed-effect rows describe
absorbed or cancelled blocks. Indicators entered as ordinary regressors
appear among the coefficients. For examples, see the
[gravity walkthrough](../get-started.md) and [nonlinear DiD](../examples/nonlinear-did.md).

Keep coefficients and their covariance together when reporting or using a fit.
For an existing fitted `result`, choose your output paths:

```python
result.tidy().to_csv("coefficients.csv", index=False)
result.vcov().to_csv("covariance.csv")
```

`tidy` contains `term`, `estimate`, `std_error`, `statistic`, `p_value`,
`conf_low`, `conf_high` and `inference`. Status is `"ok"`, `"degenerate"` for
a zero standard error, or `"missing"`. A zero standard error does not supply a
finite normal test statistic. Keep unrounded values for further calculations.
Statistics, p-values and intervals use the reference recorded by the
convention of the active covariance: normal, or Student t with its degrees
of freedom. The default `"stata"` convention of a no-FE fit uses the normal
reference. A corrected FE result adds the columns `uncorrected` and `bias`.

## Prediction and retention

| What do you want? | Python call | What you get |
| --- | --- | --- |
| Fitted values | `fit.predict()` | One conditional power mean per retained observation, in fitting order. |
| Residuals | `fit.predict(type="residual")` | One observed minus fitted response per retained observation, on the scale used for estimation. |
| Fixed effects | `fit.fixef()` or `fit.predict(type="fixed_effects")` | A dictionary keyed by fixed-effect term. Each Series has one log-scale value per level. |
| Fitted values at other covariates | `fit.predict(new_data)` | A Series indexed like `new_data`. With fixed effects, change the covariates on the retained rows. |

At `omega=1`, fitted values and residuals are in outcome units. At
`omega=0`, the fitted value is the geometric mean and the residual is on
the log scale. At any other order, `predict()` gives the conditional power
mean $m$ and the residual is $Y^\omega-m^\omega$.

### Predict from a cancelled Gamma fit

These four firms each report sales in three years. What would the fitted
sales be if the price index rose by 0.10 in every firm and year?

```python
import pandas as pd
import powermean as pm

panel = pd.DataFrame({
    "firm": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
    "year": [2022, 2023, 2024] * 4,
    "price": [1.0, 1.2, 1.1, 1.3, 1.1, 1.0, 1.2, 1.0, 1.3, 1.0, 1.3, 1.2],
    "sales": [10, 8, 12, 5, 10, 12, 13, 17, 12, 8, 6, 9],
})
fit = pm.fit("sales ~ 0 + price | firm + year", panel, omega=1, tau=2,
             cancel="firm", cluster="firm", bias_correction=False)

rows = panel.iloc[fit.sample.positions]
fitted = fit.predict()
residuals = fit.predict(type="residual")
higher = fit.predict(rows.assign(price=rows["price"] + 0.10))

print(pd.DataFrame({"Sales": rows["sales"], "Fitted": fitted,
                    "At higher price": higher, "Residual": residuals}).head(3).round(2))
```

```text
   Sales  Fitted  At higher price  Residual
0     10   10.19             8.64     -0.19
1      8    8.26             7.01     -0.26
2     12   11.42             9.69      0.58
```

In 2022, firm A sold 10 units. The model fits 10.19 at its observed price
and 8.64 at the higher price, holding the firm's recovered effect and
the 2022 effect fixed.

The fit estimates the price slope and year effects after cancelling the
firm effects. It recovers one scale for each firm from its observed sales.
To look up a particular firm or year, index the Series by its label.

```python
effects = fit.predict(type="fixed_effects")
print(round(effects["firm"].loc["A"], 3), round(effects["year"].loc[2022], 3))
# 4.102 -0.133

firm_and_year = rows["firm"].map(effects["firm"]) + rows["year"].map(effects["year"])
```

`firm_and_year` is the combined fixed-effect contribution for each row.
Here the fitted mean is `exp(price * slope + firm_and_year)`. The separate
firm and year values depend on the normalization. With only three sales
observations per firm the recovered firm effects can be noisy.
[How each method obtains fixed effects](#fixed-effect-results) ·
[Prediction after a bias correction](#corrected-results)

### Which rows can I predict for?

**With fixed effects**, start from `data.iloc[fit.sample.positions]` and
change the covariates of interest. Keep the index and order, even if all
of your proposed group labels appeared in the fit. Powermean holds each
row's group membership and original offset or exposure fixed. You can
omit the outcome and fixed-effect identifiers from the replacement frame
unless they also enter the regressor formula. If you supply a fixed-effect
identifier, its values must match the fitted rows.

**Without fixed effects**, supply any number of new rows. Include the raw
variables used in the regressor formula, even inputs for omitted collinear
terms, and any fitted offset or exposure. Powermean applies the formula
and categorical coding from estimation. A missing category gives a missing
prediction on that row. An unseen category raises `SpecificationError`.
[Example with new rows](../examples/reuse-a-fit.md#inspect-predictions-and-influence-contributions)

### Recompute uncertainty from stored information

Covariance recomputation requires stored information. A classification never
stored cannot be added just by naming it later. `keep_influence=False` removes
contributions needed by some subsequent requests. Inference requires an accepted
numerical result.

Both calls reuse the fitted point without refitting. In a no-FE fit with
retained influence, use `fit.vcov("robust")` for a covariance DataFrame with
request metadata in `.attrs`. Use
`robust = fit.with_vcov("robust")` for a separate result view whose
`se()`, `tidy()`, `confint()`, effects and tests use the robust covariance.
The certificate still describes the covariance requested at fitting.
Report the view with `pm.etable(robust, uncertainty="ci")`.
[Worked covariance example](../examples/reuse-a-fit.md#inspect-a-fit-and-change-its-covariance)

Without fixed effects, `spec` can be `"robust"`, a named HC method, or a mapping
whose `"cluster"` names one retained classification or several retained
dimensions, with optional `"multiway"` and `"psd"` keys. `with_vcov(None)`
selects the fit's own request. `vcov()` or `vcov(None)` without per-request
keywords returns the active covariance, which may be a view's; with such
keywords, `None` uses the fit's request as the base. `ssc=` gives the request's
convention; omitted, the fit's
convention is inherited. `bread="expected"` is accepted only for a named HC
method. The fit's own request is answered from its stored record, even after
`keep_influence=False`; another request needs the stored contributions and
is refused once they are discarded. FE results refuse the `ssc=` and `bread=`
keywords. [Inference](../methods/inference.md#other-covariance-methods-and-their-availability)
shows call recipes.

## Proportional effects, contrasts and tests

These methods derive targets from one certified point and its active
covariance. They do not refit, recover fixed effects or change `tidy()`.
An uncertified result withholds them, as it withholds `se()`.

| Call | Reports |
| --- | --- |
| `effect(term, *, contrast=None, level=0.95, ci="transform")` | `expm1(beta * (d1 - d0))` for a standalone regressor entered directly, with `contrast=(d0, d1)`; one unit by default. |
| `effects(terms=None, *, level=0.95, ci="transform")` | A table. Selected `terms` give one-unit changes, as in `effect`. Without `terms`, each reported coefficient gives `exp(beta) - 1`, labeled as a coefficient coordinate rather than a regressor change. |
| `contrast(before, after, *, level=0.95, ci="transform")` | `expm1((X(after) - X(before))' beta)` between two explicit regressor configurations. |
| `lincom(expression, *, level=0.95)` | An affine combination such as `"x1 + 2*x2"`, with variance `c'Vc`. |
| `nlcom(expression, *, level=0.95)` | A smooth function using arithmetic and `exp`, such as `"exp(x1 + 2*x2) - 1"`, by the delta method with an analytic gradient. |
| `test(expression)` | One affine equality such as `"x1 = x2"`, reported as by `wald`. |
| `wald(r, q)` | The joint restriction `R beta = q`, for an explicit matrix with one column per reported coefficient, in `coef()` order, and a matching vector. |

`effect` accepts a single standalone regressor entered directly. Use
`contrast` for a change that involves several coefficients, such as a
simultaneous change in two regressors or a change in a categorical,
transformed or interacted regressor. Use `nlcom` when the target is
specified as a coefficient expression.

`contrast` takes `before` and `after` mappings. Give one scalar value for each raw
right-hand-side variable of the formula, and no other entries. The fitted
categorical coding and transformations are replayed; nothing is filled in or
averaged. A level outside the fitted encoding is refused. Inputs outside the
raw RHS names, including extra outcome, fixed effect, offset or exposure
entries, are refused. Fixed effects and offsets are held fixed even if an RHS
variable also supplies one of those components. A contrast is conditional on
the supplied values, not a population
average. Expressions name reported coefficients and are parsed, never
evaluated as Python.

Intervals are pointwise. Each method uses the reference of the active
covariance, as `tidy()` does. For a proportional change, `ci="transform"`
applies `expm1` to the endpoints of the interval for the log-scale contrast
`l`; `ci="delta"` centers an interval on the
estimate with the delta standard error `exp(l) se(l)`. The statistic and
p-value test `l` against zero. Zero standard errors are reported as
degenerate and undefined ones as missing, never floored.

Each proportional target is labeled with the fitted mean order. Only an
`omega=1` fit is labeled `theta_N/NATE`, the proportional change in the
conditional arithmetic mean. The label names a conditional comparison, not
an identified causal effect. Local semi-elasticities, elasticities and
standardized treatment effects have no public method.

`wald` counts redundant restrictions once. The effective count is the rank
of `R V R'` under NumPy's default `matrix_rank` rule, which the generalized
inverse shares. With a normal reference the statistic is referred to
chi-square with that count; with a t reference, the statistic divided by the
count is referred to F with the fit's denominator degrees of freedom. The
report includes the requested and effective counts and the reduction. If
`R beta - q` has a component outside the range of `R V R'`, the null cannot
be tested with this covariance: the report is degenerate, without a p-value.
Exact-zero covariance is degenerate; negative or undefined covariance is
missing. Empty or inconsistent requests are refused.

`effect` and `contrast` return records with `estimate`, `std_error`,
`interval`, `label`, `order`, `population`, `gradient` and `metadata`.
Read `level` and `ci` for the requested interval settings. The contrast
record also gives the aligned `before` and `after` values,
`coefficient_names`, `contrast` and `learned_columns`. Their
`influence()` multiplies the target gradient by the retained coefficient
contributions and needs those contributions. `lincom` and `nlcom` return
`estimate`, `std_error`, `statistic`, `p_value`, `interval`, `gradient` and
`metadata`, with `term`, `kind` and `level` identifying the request.
Test records expose `statistic`, `p_value`, the restriction `rows` and `rhs`,
the estimated `difference`, its `covariance`, and `metadata`. Both the
combination and test records export with `to_dict()` and `to_frame()`.
An `effect` record exports one row with `to_dict()`; a `contrast` record
has both `to_dict()` and `to_frame()`. The method `effects(terms)` returns
a DataFrame directly. The [follow-on walkthrough](../examples/reuse-a-fit.md)
shows these result types in use.
On a corrected FE result, these methods use the corrected coefficients with
the first-order covariance at the uncorrected root, and `metadata` records
both points.

See the [worked results example](../examples/reuse-a-fit.md#report-effects-and-contrasts)
for recorded effects, a contrast, coefficient tests and table exports.

## Fixed-effect results

For joint fits, `fixef()` returns a dictionary with one Series per
fixed-effect term in formula order. The reporting normalization processes
terms by their number of levels in the estimation sample, largest first,
with ties resolved by formula order.
Within each connected component, the first level of each later term is set to
zero. The remaining indicator columns are considered in that same term order
and in level order. Effects for linearly dependent columns are set to zero.
The normalization preserves the fitted index and `predict()` returns in-sample
fitted values.

For cancelled fits, `fixef()` returns the retained blocks and the recovered
cancelled block described below. `predict()` uses the same recovery across
supported working variances. A retained all-zero group gives zero fitted
means and a missing log effect. The prediction's `zero_boundary_rows` attribute
lists boundary rows by position within the retained sample.

For two-way GMM, post-estimation recovery holds the GMM slopes fixed and fits
only the two effect blocks by PPML at the slope index. The fitted means match
the observed row and column totals on observed cells. A directed fit uses
its non-self links. If a cell cannot carry positive mass at those margins,
its fitted mean is zero. An unsupported level has a missing effect. In each
component of the positive support, the first supported column effect is
normalized to zero.

### Fixed-effect block counts in the summary

For a successful joint fit, `print(fit.summary())` reports the number of
levels of each block on the fitted rows, the redundant levels, and nesting
in a cluster dimension of the active covariance. A block is nested when
each of its levels lies in one cluster of that dimension. Reported nested
blocks count as fully redundant in this display. Changing the active
covariance can change the nesting calculation. With multiple cluster dimensions,
the summary names the first matching dimension in request order.

For three or more nonnested blocks the later entries can be pairwise lower
bounds. The covariance's `K` and `rank` have their own request-specific
definitions above. In the [gravity summary](../get-started.md#fit-with-three-fixed-effect-blocks),
the three-block design has a nested pair block. This small example leaves
all three blocks nonnested and shows how the summary labels the bound.

```python
import pandas as pd
import powermean as pm

cells = [("A1", "B2", "C1"), ("A1", "B1", "C2"), ("A2", "B2", "C2")]
data = pd.DataFrame(
    [
        (a, b, c, x, y + x * (1 + (i * i + 3 * i) % 7))
        for i, (a, b, c) in enumerate(cells)
        for x in (0, 1)
        for y in (3 + i % 2, 6 + i % 3, 10 - i % 3)
    ],
    columns=["a", "b", "c", "x", "y"],
)
fit = pm.fit(
    "y ~ 0 + x | a + b + c", data, omega=1, tau=1,
    cancel=None, bias_correction=False, vcov="robust", ssc="regression",
)
print(fit.summary())
```

```text
powermean
====================================================================================
PPML (omega = 1, tau = 1)
Fixed effects: a (estimated), b (estimated), c (estimated)
Observations: 18 of 18 input rows (none dropped)
Status: verified; certified: True
Coverage: unique_root (BBP prop:iols-exact-sample-root)
------------------------------------------------------------------------------------
          coef.   std. err.        t    P>|t|    [95% conf. interval]
------------------------------------------------------------------------------------
x        0.4169      0.1781     2.34    0.035      0.0350      0.7988
------------------------------------------------------------------------------------
Fixed-effect block counts (fitted rows, active covariance)
  a: levels=2, redundant=0 (exact), nested in=none
  b: levels=2, redundant=1 (exact), nested in=none
  c: levels=2, redundant=1 (pairwise lower bound), nested in=none
```


### Recovered cancelled effects

After solving for the slopes and the remaining fixed effects, `fixef()`
recovers one value for each level of the cancelled block from the
estimation sample. Each group's value is
`log(groupmean(Z exp(-v))) / omega`, where `Z` is the outcome on the fit's
transformed scale and `v` the fitted working index without the cancelled
effect. The Series attribute `attrs["cancelled_recovery"]` holds, by group,
the arithmetic `scale`, defined as `groupmean(Z exp(-M_C v))` where `M_C v`
is `v` centered within the group, its `log_scale`, a `status` and a
`log_effect_status`.

| `status` | Meaning |
| --- | --- |
| `"positive"` | Finite recovered scale. |
| `"zero"` | Every outcome in the group is zero: the scale is zero and the effect is missing, with `log_effect_status` `"no_finite_effect"`. |
| `"overflow"`, `"underflow"` | The arithmetic scale is not representable and is missing; `log_scale` is kept. |

`log_effect_status` is `"finite"`, `"no_finite_effect"`, `"overflow"` or
`"underflow"`. The recorded `point` is the original uncorrected root or,
after a short-group or cell correction, its reporting point. The package
reports these group values without standard errors. When the number of
observations per group stays small, an individual effect can remain noisy
even when the slope is well estimated. Fitted means and counterfactuals
hold these recovered scales fixed.

Inspect the fitted levels and recovered scales after following the
[short-panel example](../methods/cancellation.md):

```python
print(joint_gamma.fixef()["firm"].head(2))
recovered = cancelled_gamma.fixef()["firm"]
record = recovered.attrs["cancelled_recovery"]
print(record["point"])
print(record["scale"][:2])
print(record["status"][:2])
```

```text
firm
0    0.603407
1    2.019176
Name: firm, dtype: float64
original_uncorrected
[2.2137495  7.94845472]
('positive', 'positive')
```

### Corrected results

When you request a coefficient bias correction, `coef()` and the `estimate`
column of `tidy()` report the corrected coefficients. `tidy()` also gives
the uncorrected coefficients and estimated bias. Confidence intervals are
centered on the corrected coefficients. The numerical certificate and the
first-order covariance use the original estimating root.

After the coefficient correction of Weidner and Zylkin (2021), both
`predict()` and `fixef()` use the original slopes and effects. After a
short-group or cell correction, a cancelled fit uses the corrected reporting
point to predict and to recover fixed effects. [Details of the corrections](experimental-fixed-effects.md#coefficient-corrections) · [Weidner and Zylkin (2021)](https://doi.org/10.1016/j.jinteco.2021.103513)

The FE retained backends have additional recomputation paths for named HC,
cluster and WZ-adjusted requests. Each requires the relevant retained
information and its own numerical conditions. These paths do not expand the
initial fit time inference requirements. See [inference](../methods/inference.md).

Returned tables are intended for inspection and export, not as mutable inputs
to a future fit. Keep the requested method, sample accounting and numerical
status with coefficients and standard errors. The
[example collection](../examples/README.md) links these outputs back to
recorded displays; [validation](../validation.md) explains what evidence
is needed to make broader claims.

<a id="jochmans-fits"></a>

## GMM1 and GMM2 fits

`pm.fit(..., estimator="gmm1")` and `estimator="gmm2"` return a
`JochmansResult`. Both methods difference out the two fixed effects when
estimating the slopes. To predict afterwards, Powermean holds those slopes
fixed and recovers both sets of effects by PPML. [Method and worked example](../methods/jochmans.md)

| Task | Available on the fitted result |
| --- | --- |
| Report estimates | `coef()`, `se()`, `confint(level)`, `tidy(level)` and `summary(level)`. Use `effects(level=...)` for `exp(b) - 1` by coefficient or `pm.etable(fit)` to build a regression table. |
| Choose the cell-projection convention | `vcov()` gives the active matrix. `vcov("paper")` or `vcov("stata")` returns the requested matrix. `with_vcov(...)` applies it to subsequent reporting. [Covariance conventions](../methods/jochmans.md#read-uncertainty-and-compare-with-stata) |
| Predict or extract effects | `predict()` gives fitted outcomes and `predict(type="residual")` gives observed minus fitted outcomes. `fixef()` and `predict(type="fixed_effects")` return both effect blocks. Use `predict(new_data)` for replacement covariates on the same cells, in fitting order. [Prediction example](../methods/jochmans.md#predict-trade-after-fitting) |
| Identify the fitted array | Read `formula`, `method`, `layout`, `dimensions` and `fixed_effects` on the result. `sample.positions` gives the original row positions and `sample.retained_index` the original **index labels**. The complete-array requirement keeps all submitted rows. |
| Check the numerical solution | `certificate.status` is `"local_solution"` on success. `relative_moment` and `correction` on the certificate give the final scaled moment and Newton correction. `tolerance` and `iterations` give the threshold and number of accepted updates. |

`fit.provenance` records the estimator, covariance convention, status and
references. For coefficient-level changes use `effects()` here. The named
`effect()`, `contrast()`, coefficient tests and observation-level `influence()`
belong to the ordinary regression result.

<!-- guide-navigation:start -->
[Previous: Explore and export a fit](../examples/reuse-a-fit.md) · [Guide index](../README.md) · [Next: Tables and exports](tables.md)
<!-- guide-navigation:end -->
