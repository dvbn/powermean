# Function index

Examples use `import powermean as pm`. Choose the task you want to do.

| Task | Start here |
| --- | --- |
| Estimate a regression or reuse a design | [Fitting functions](#fit-and-configure-a-model) and [supported methods by model](../availability.md). |
| Read an estimate, change its standard errors or test a restriction | [Result methods](#inspect-a-fitted-result). GMM1 and GMM2 have a [smaller set of result methods](results.md#gmm1-and-gmm2-fits). |
| Predict outcomes, calculate residuals or recover fixed effects | [Prediction and fixed effects](#predictions-and-fixed-effects). |
| Find which rows were used or why a fit failed | [Sample and diagnostics](#sample-and-diagnostics) and [errors and warnings](#catch-a-refusal-or-inspect-a-warning). |
| Describe data, compare fits or export a table | [Table functions and formats](#build-or-export-a-table). |
| Simulate a panel or a dataset with a known slope | [Data generators and seed controls](#generate-synthetic-data). |

## Fit and configure a model

### `pm.fit()`

`pm.fit(formula, data, **options)` returns a `Result` for a power-mean
regression or log OLS. Use the
untransformed outcome on the left of the formula. Choose the conditional
mean with `omega` and the working variance with either `tau` or `variance`.
A formula with fixed-effect blocks fits them jointly by default.
Use `cancel="block"` to cancel one named block.
[Fit options](options.md) · [Gravity fit](../get-started.md#fit-with-three-fixed-effect-blocks) · [Nonlinear DiD](../examples/nonlinear-did.md#fit-the-twfe-equation-with-different-working-variances)

Set `estimator="gmm1"` or `"gmm2"` to difference out exactly two fixed-effect
blocks using the moments derived in Jochmans (2017). Use an intercept-free formula.
The default `fe_layout="rectangular"` requires a complete crossing. Set
`fe_layout="directed"` for complete ordered non-self links. The returned
`JochmansResult` reports coefficient inference, sample identity and a local
root certificate. [Two-way difference options](options.md#two-way-difference-options) ·
[When the two-way comparison is valid](../methods/jochmans.md#when-is-the-comparison-valid)

### `pm.ppml()`

`pm.ppml(formula, data, **options)` returns a PPML result with `omega=1`
and `tau=1`. With the gravity example's formula and data, use
`pm.ppml(formula, trade, cluster="pair")`.
[Gravity walkthrough](../get-started.md) · [Working-variance comparison](../methods/model.md#set-weights-in-the-estimating-equation)

### `pm.gpml()`

`pm.gpml(formula, data, **options)` returns a Gamma PML result with
`omega=1` and `tau=2`. PPML and GPML require any explicit `omega` or
`tau` to match their fixed values. Use `pm.fit` to supply a custom
`variance`.
[Working-variance comparison](../methods/model.md#set-weights-in-the-estimating-equation) · [Short-panel example](../methods/cancellation.md)

### `pm.logols()`

`pm.logols(formula, data, **options)` returns a least-squares result on
the log outcome, with `omega=0`. Use `positive_only=True` to explicitly
restrict data with zeros to the positive-outcome sample, then check
`fit.sample.positions`. Exponentiating the fitted index gives the
conditional geometric mean.
[Mean order and outcome domain](../methods/model.md#choose-a-conditional-mean) · [Worked log OLS call](../methods/model.md#set-weights-in-the-estimating-equation)

### `pm.prepare()`

`pm.prepare(formula, data, **options)` returns a `Prepared` object with
the design and base inputs. Supply sample arguments and needed cluster
classifications during preparation. `prepared.positive_only` records the
fixed outcome-domain choice. `prepared.fit(**options)` returns a
`Result` for that fit. Compare `sample.positions` when changing the mean
order or equation.
[Sample arguments](options.md#sample-arguments) · [Reuse a fit](../examples/reuse-a-fit.md#solve-the-same-mean-equation-twice)

### `pm.NB2()`

`pm.NB2(kappa=1.0)` returns a working variance `m + kappa*m**2`.
Pass `variance=pm.NB2(kappa=0.5)` in place of `tau`. The fit holds the
finite, nonnegative `kappa` fixed. Read `.kappa` or call `.to_dict()`
to inspect the choice. Given `q = log(m)`, the variance object also
provides `log_value(q)` for the log working variance, `elasticity(q)` for
the derivative of log variance with respect to log mean, and
`elasticity_derivative(q)` for its derivative with respect to `q`.
[Working variances](../methods/model.md#set-weights-in-the-estimating-equation)

### `pm.ssc()`

`pm.ssc(*, robust, cluster, cluster_count, reference, psd, absorbed_count="fixest")`
returns an immutable convention to pass to a covariance request. The
first five keywords are required. To save and restore the convention, use
`record.to_dict()` and `type(record).from_dict(record_dict)`.
[Small-sample conventions and example](options.md#small-sample-conventions)

### `pm.solver_defaults()`

`pm.solver_defaults(solver="newton")` returns a `SolverOptions` record
for Newton or an `IolsOptions` record for `"iols"`. Use the fit's
`solver=` keyword to choose which solver runs.
[Solver options](solver-options.md) · [IOLS options](iols_options.md)

```python
newton_defaults = pm.solver_defaults().to_dict()
iols_defaults = pm.solver_defaults("iols").to_dict()
```

### `pm.SolverOptions()`

`pm.SolverOptions(max_iter=500)` constructs a Newton options record with
that iteration limit and the remaining defaults. Read its fields or call
`.to_dict()`. Pass the record or a mapping as `solver_options`.
[Newton options and keyword defaults](solver-options.md#ordinary-newton-options)

See [advanced fixed effects](experimental-fixed-effects.md) for optional
module-level sample, solver and cancellation records.

## Inspect a fitted result

The calls below apply to ordinary `Result` objects, including fixed effects
estimated jointly or with one block cancelled. For the complete panel or
dyad estimators, see [GMM1 and GMM2 results](results.md#gmm1-and-gmm2-fits).

### Coefficients and uncertainty

#### `coef()`

`fit.coef()` returns a pandas Series of estimates indexed by term.
[Results](results.md)

#### `se()`

`fit.se()` returns a Series of standard errors by term from the active
covariance. [Worked result](../examples/reuse-a-fit.md#inspect-a-fit-and-change-its-covariance)

#### `confint()`

`fit.confint(level=0.95)` returns a DataFrame with `lower` and `upper`
columns by term. [Worked intervals](../examples/reuse-a-fit.md#inspect-a-fit-and-change-its-covariance)

#### `tidy()`

`fit.tidy(level=0.95)` returns a coefficient DataFrame with estimates,
standard errors, test statistics, p-values and interval endpoints.
[Worked coefficient table](../examples/reuse-a-fit.md#inspect-a-fit-and-change-its-covariance)

#### `summary()`

`fit.summary(level=0.95)` returns a text report with estimates, sample
counts and diagnostics. Use `print(fit.summary())`.
[Gravity report](../get-started.md#fit-with-three-fixed-effect-blocks)

### Covariance and influence

#### `vcov()`

`fit.vcov(spec=None, *, ssc=..., bread=...)` returns a covariance
DataFrame with metadata in `.attrs`. `fit.vcov()` reads the active
covariance. An alternate `spec` requests a matrix at the fitted point,
using retained information.
[Post-fit uncertainty](results.md#recompute-uncertainty-from-stored-information) · [Worked covariance](../examples/reuse-a-fit.md#read-diagnostics-and-covariance)

#### `with_vcov()`

`fit.with_vcov(spec, *, ssc=..., bread=...)` returns a separate
result view whose standard errors, intervals, tables, effects and
tests use the requested covariance at the same fitted point. With
fixed effects, supply only `spec` to either method. Put any supported
`ssc` or `bread` override inside the request mapping.
[FE covariance options](options.md#fixed-effect-options) · [Worked change of covariance](../examples/reuse-a-fit.md#inspect-a-fit-and-change-its-covariance)

#### `influence()`

`fit.influence(*, unit="sampling")` returns a DataFrame of retained
coefficient contributions, with a row per retained observation even
under clustering. Cancellation gives one row per cancelled group by
default. Use `unit="row"` for contributions indexed by original row
position. [Worked influence output](../examples/reuse-a-fit.md#inspect-predictions-and-influence-contributions)

### Effects and contrasts

These calls use the active covariance and report proportions with pointwise
intervals. Choose `ci="transform"` for transformed endpoints or `ci="delta"`
for a delta-method interval.
[Effects and tests](results.md#proportional-effects-contrasts-and-tests)

#### `effect()`

`fit.effect(term, *, contrast=None, level=0.95, ci="transform")` returns
a proportional change record for one directly entered regressor. Supply
`contrast=(0, 1)` for a zero-to-one change. Read `estimate`, `std_error`,
`interval`, `gradient` or `metadata`. The record offers `to_dict()` and,
with retained contributions, `influence()`.
[Agreement effect](../get-started.md#report-a-percentage-and-interval)

#### `effects()`

`fit.effects(terms=None, *, level=0.95, ci="transform")` returns a
DataFrame of one-unit changes for named direct regressors. With
`terms=None`, it reports changes per coefficient coordinate.
[Cohort-by-date example](../examples/nonlinear-did.md#allow-different-effects-by-adoption-cohort-and-date)

#### `contrast()`

`fit.contrast(before, after, *, level=0.95, ci="transform")` returns a
proportional change record across two complete settings of the formula's
raw right-hand-side variables. It replays transformations and
interactions while holding fixed effects and offsets fixed. Read the
same estimate and interval fields as for `effect()`, or use `to_dict()`,
`to_frame()` or `influence()` with retained contributions.
[Worked contrast](../examples/reuse-a-fit.md#report-effects-and-contrasts)

### Coefficient combinations and restrictions

Use reported coefficient names in expressions. These calls use the
active covariance. Both record types provide `to_dict()` and `to_frame()`.
[Worked combinations and tests](../examples/reuse-a-fit.md#test-or-combine-coefficients)

#### `lincom()`

`fit.lincom(expression, *, level=0.95)` returns an affine-combination
record, for example from `"x1 - x2"`. It includes `estimate`,
`std_error`, `interval`, `gradient`, `statistic`, `p_value` and `metadata`.
[Worked linear combination](../examples/reuse-a-fit.md#test-or-combine-coefficients)

#### `nlcom()`

`fit.nlcom(expression, *, level=0.95)` returns the same kind of record
for a smooth coefficient expression using arithmetic and `exp`. Use it
for a coefficient-based change such as `"exp(x1 - x2) - 1"` with
delta-method uncertainty.
[Worked nonlinear combination](../examples/reuse-a-fit.md#test-or-combine-coefficients)

#### `test()`

`fit.test(expression)` returns a restriction record for one affine
equality such as `"x1 = x2"`. Read its `statistic`, `p_value` and
`metadata`. For a single restriction or a joint `wald()` test, `difference`
contains the estimated restriction residuals and `covariance` their
covariance matrix. [Worked equality test](../examples/reuse-a-fit.md#test-or-combine-coefficients)

#### `wald()`

`fit.wald(r, q)` returns a restriction record for `R beta = q`.
Supply a matrix with columns in `fit.coef().index` order and one entry
of `q` per restriction row.
[Worked joint test](../examples/reuse-a-fit.md#test-or-combine-coefficients)

### Predictions and fixed effects

#### `predict()`

`fit.predict(data=None, *, type="power_mean")` returns a Series of fitted
conditional power means. Use `type="fixed_effects"` to extract the block
Series returned by `fixef()`, or `type="residual"` for response residuals
on the fit's estimating scale. Both use the original fit sample.
For fixed-effect counterfactuals, pass replacement covariates in the
retained row order and keep the index. Slopes, offsets and fixed effects
come from the fit. Fits without fixed effects also accept compatible new
rows.
[Prediction](results.md#prediction-and-retention) · [Worked prediction](../examples/reuse-a-fit.md#inspect-predictions-and-influence-contributions)

#### `fixef()`

`fit.fixef()` returns a dictionary of block names to Series of normalized
levels for an FE fit, including recovered levels for cancelled and two-way
GMM fits. [Fixed effects](results.md#fixed-effect-results) · [Worked recovery](results.md#recovered-cancelled-effects)

### Sample and diagnostics

#### `sample`

`fit.sample` gives the input and retained counts (`nobs`, `count`),
`effective_n`, `zeros`, `counts`, `missing`, and the row ledger. Use
`table()` for exclusion counts and `reason(position)` for a specific row.
The ledger includes zero-based `positions`, index `labels`, `input_labels`,
`retained_index` and `codes`. Compare the retained positions before
comparing estimates from fits on the same input frame, for example with
`with_price.sample.positions.tolist() == with_stock.sample.positions.tolist()`.
[Sample](results.md#inspect-the-estimation-sample) · [Worked row comparison](../examples/compare-samples.md#find-the-rows-each-fit-used)

#### `estimator`

`fit.estimator` records the fitted equation, including mean order and
working variance. Use `fit.estimator.to_dict()` for the resolved settings.
[Diagnostic records](results.md#read-the-estimator-and-diagnostics)

#### `certificate`

`fit.certificate` gives the numerical assessment when the route produces
one. [Diagnostic records](results.md#read-the-estimator-and-diagnostics)

#### `provenance`

`fit.provenance` gives the computation record, including the status,
root-theory coverage label, options, trace and covariance where
applicable. Use `fit.provenance.to_dict()` for export.
[Worked diagnostics](../examples/reuse-a-fit.md#read-diagnostics-and-covariance)

## Build or export a table

All five functions return a `Table`. The four descriptive functions
accept `sample=fit` with the original frame in its original row order,
then apply their own missing-value rules.
[Summary and balance table options](tables.md#describe-data-and-treatment-arms)

### `pm.summary_statistics()`

`pm.summary_statistics(data, vars=None, **options)` reports usable and
missing counts, means, standard deviations and quantiles. For nonnegative
variables it also reports zero shares and means among positive values.
[Compare retained samples](../examples/compare-samples.md#describe-the-two-retained-samples)

### `pm.balance_table()`

`pm.balance_table(data, treatment, vars=None, **options)` reports
covariate means by arm and differences from the selected control, with
uncertainty and standardized differences.
[Baseline balance example](results.md#inspect-balance-on-the-baseline-sample)

### `pm.balance_regression()`

`pm.balance_regression(data, treatment, vars=None, **options)`
regresses assignment on the covariates and reports coefficients and a
joint test in a table. It compares each noncontrol arm with the control.
Use `continuous=True` to regress the dose itself on the covariates.
[Baseline joint test](results.md#inspect-balance-on-the-baseline-sample)

### `pm.outcome_table()`

`pm.outcome_table(data, outcomes, treatment, **options)` reports outcome
means, zero shares and unadjusted proportional comparisons by arm and
mean order. Use the fitted result's `effect()` or `contrast()` for an
adjusted conditional change.
[Outcome table by promotion](../examples/compare-samples.md#summarize-outcomes-by-promotion-within-one-sample)

### `pm.etable()`

`pm.etable(*results, **options)` displays one model per column and
reports the estimator, fixed effects, estimation sample and covariance
choices. You can also pass one list or tuple of results. Use
`uncertainty="ci"` to display intervals and `scale="effect"` to display
percentage changes for coefficient coordinates. For a change involving
several coefficients, use `fit.contrast()` for two regressor settings
or `fit.nlcom()` for a coefficient expression.
[Regression tables](tables.md#tabulate-fitted-models) · [Worked model comparison](../examples/compare-samples.md#compare-the-estimates)

### Table exports

Use `table.to_frame(*, formatted=False)` for a DataFrame with unrounded values or set
`formatted=True` for the displayed cells. `print(table)` displays
`table.to_text()`. For reports, use `table.to_markdown()`,
`table.to_html()`, or
`table.to_latex(*, booktabs=True, floating=True, label=None)`.
[Export formats](tables.md#print-or-export) · [Worked exports](../examples/reuse-a-fit.md#export-tables)

## Catch a refusal or inspect a warning

Inspect `identifier`, `fields` and `message` on the exception or warning.
Existence and convergence errors also carry `status`, `certificate` and
`last_state` when available. [Troubleshooting](../help/troubleshooting.md)

| Public exception | When to check it |
| --- | --- |
| `pm.SpecificationError` | Invalid, conflicting or unsupported request. |
| `pm.SampleError` | Invalid outcomes or no usable estimation rows. |
| `pm.IdentificationError` | The remaining design fails to identify the parameters. |
| `pm.ExistenceError` | The equation has no finite solution. |
| `pm.SolverCapabilityError` | The solver does not support the requested case. |
| `pm.ConvergenceError` | The solver failed to reach a reportable point. |

| Public warning | What it describes |
| --- | --- |
| `pm.CoverageWarning` | A limit of the available theory or a recognized change to the formula's interpretation. |
| `pm.NumericalWarning` | A numerical condition or assessment. Read `status` and `certificate`. |
| `pm.CovarianceWarning` | A covariance adjustment or unavailable uncertainty. |

## Generate synthetic data

Import these names from `powermean.testing`. The four generators return
`Synthetic` records. Their `zeros` option adds zeros in the simulation
by randomly zeroing outcomes and rescaling the remaining values to
preserve the conditional mean. Record the seed and draw to identify a
dataset. [Simulation validation](../validation.md)

### `solver_data()`

`solver_data(design, *, n=1000, k=10, zeros=0.0, seed=SOLVER_SEED, draw=0, split="test")`
draws a no-FE dataset from `SOLVER_DESIGNS` with default seed
`SOLVER_SEED`. Choose `"development"`, `"validation"` or `"test"` for the
seed split.

### `twoway_data()`

`twoway_data(design, *, n=2000, k=9, zeros=0.0, seed=TWOWAY_SEED, draw=0, split=40)`
draws a two-way FE dataset from `TWOWAY_DESIGNS` with default seed
`TWOWAY_SEED`. The seed split is an integer.

### `threeway_data()`

`threeway_data(design, *, countries=20, years=5, zeros=0.0, seed=THREEWAY_SEED, draw=0, master=(100, 10), self_pairs=True)`
draws a directed dyadic panel from `THREEWAY_DESIGNS` with default seed
`THREEWAY_SEED`. `master` gives the countries and periods of the underlying
draw. Use `self_pairs=False` to exclude within-country pairs.

```python
from powermean.testing import solver_data, threeway_data, twoway_data

no_fe = solver_data("G_poisson", draw=1, split="test")
two_way = twoway_data("d12", draw=1, split=40)
three_way = threeway_data("wz2", countries=8, years=3, draw=1)
metadata = three_way.metadata()
```

### `showcase_data()`

`showcase_data(*, law="nb2", regions=40, firms_per_region=5, years=6, zeros=0.0, seed=1, draw=0)`
draws a firm-year panel. Choose a law in `SHOWCASE_LAWS` for `y` and
`y_iv`. `SHOWCASE_TRUTH` records the showcase coefficient values.

```python
from powermean.testing import showcase_data

demo = showcase_data(regions=2, firms_per_region=2, years=3, seed=7)
print(demo.design, len(demo.frame))
print(demo.formula)
```

```text
showcase/nb2 12
y ~ treat + x_normal + x_skewed + x_binary + C(category) + dose + C(arm) + policy | firm + year
```

### `showcase_treat_coefficient()`

`showcase_treat_coefficient(omega)` returns the true coefficient of
`treat` for the conditional power mean of the positive outcome `y_pos`
at the requested order.

### `Synthetic`

The dataset record contains `design`, `frame`, `formula`, `truth`,
`columns`, `seed`, `source`, `fingerprint` and `notes`. Read
`zero_share` for the fraction of zeros in `y`, `digest()` for numeric
checks and `metadata()` for the JSON-compatible recipe and digest.

### `save()`

`save(data, path)` takes a `Synthetic` record and a `pathlib.Path`
ending in `.csv` or `.dta`. It writes the dataset and a companion
JSON file with the same stem, and returns `None`.

<!-- guide-navigation:start -->
[Previous: Run the examples](../examples/README.md) · [Guide index](../README.md) · [Next: Supported methods](../availability.md)
<!-- guide-navigation:end -->
