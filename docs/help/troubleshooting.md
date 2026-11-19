# Diagnose errors and unsuccessful fits

Start with the exception class, the full diagnostic and the formula. A single
change to the outcome transformation or retained rows can change the
quantity being estimated.

| Symptom | Inspect |
| --- | --- |
| `SpecificationError` | Unknown or conflicting options, formula syntax and method requirements. |
| `SampleError` | Negative outcomes, zeros at a nonpositive mean order, negative weights, or an empty estimation sample. |
| `ValueError` during data preparation | Check that exposure is positive, frequency weights are whole counts, and the subset has one Boolean per input row. |
| `KeyError` during data preparation | Check the names of the requested fixed-effect and weight columns against the data. |
| `IdentificationError` | Variation after accounting for all fixed effects and the remaining regressors. |
| `ExistenceError` | Separation and other obstacles to a finite solution. |
| `ConvergenceError` | The last numerical state, assessment and solver controls. |
| `SolverCapabilityError` | Whether the requested solver handles the formula, variance and data options. |

Powermean exceptions and warnings provide an `identifier` for programmatic
inspection and `fields` and `message` for details. `ExistenceError` and
`ConvergenceError` also carry `status`, `certificate` and `last_state`
when available. For example, a log fit refuses zero outcomes unless you
explicitly request a positive-outcome sample.

```python
import pandas as pd
import powermean as pm

try:
    pm.logols("y ~ x", pd.DataFrame({"y": [0, 2, 4], "x": [1, 2, 3]}))
except pm.SampleError as error:
    print(error.identifier)
    print(error.fields)
```

```text
sample.zero_outcome_at_order
{'count': 1, 'omega': 0.0}
```

## A fixed effect call is refused

Check whether one valid label per row identifies each block. In gravity,
`exporter^year` is one block. If you cancel, choose an existing block by name.
For WZ, check the actual retained sample and the three gravity blocks against
the [WZ requirements](../reference/experimental-fixed-effects.md#coefficient-corrections).
For optional module-level controls, check the record against the formula,
mean order and working variance.
[Options](../reference/options.md#fixed-effect-options)

When a trend is omitted, inspect its scale on the retained rows. By default,
the fixed-effect collinearity check compares the sum of squared values after
removing the fixed effects with the sum of squared original values. A ratio
at or below `1e-9` counts as absorbed. If a time index has a large arbitrary
origin, express it as elapsed time from a meaningful base date before fitting.
Subtracting a constant preserves the slope in a model with fixed effects.

In a joint PPML model with two fixed effects and slopes, a worker–firm
pattern can separate zero observations even when every worker and every firm
has a positive outcome. Powermean checks the proposed direction on the
original rows, records the separated observations in the sample ledger, and
checks the remaining sample again.

If a fixed-effect fit with zeros reports `separation_unresolved`, Powermean
could not certify which remaining rows support a finite fit. On small designs,
when both iterative checks are inconclusive, it uses a sparse linear program
to check whether all remaining rows can stay. Fitting continues only if the
result satisfies the checks for every slope and fixed-effect level. The program
cannot select observations for removal.

## Your outcome contains zeros

Keep recorded zero flows in a model of the arithmetic mean. For the log or
a negative order, justify a positive-outcome analysis before selecting it.
In a joint PPML fit, a pattern of zeros can leave some effects without
finite estimates. Inspect the sample ledger and any separation diagnostic.
A perfectly absorbed regressor needs a different source of variation.

A regressor can also separate zero outcomes in an ordinary PPML fit without
fixed effects. For example, if a category has only zeros, Powermean checks
which rows can be removed and fits the remaining observations, including
any other zeros they contain. Use `fit.sample.reason(i)` for the exclusion
reason at input position `i` and `fit.sample.positions` for the fitted rows.

For Gamma PML (`tau=2`), the same pattern can leave the criterion with no
finite optimum. A proved direction gives `no_finite_root`. With
`on_failure="return"`, inspect the diagnostic in `fit.certificate`.
The sample stays intact and the result has no reportable coefficients or
standard errors.

A joint fit with profiled fixed effects can return a result with a
`NumericalWarning` about a very small fitted mean at a zero outcome. In
this case, a fresh separation check proved overlap on the same retained
rows, and the final fixed-effect calculation and all strict accuracy
checks passed. The warning remains visible because a very small fitted
mean can still deserve attention in the fitted model. Inspect the warning
and `result.provenance`. If overlap remains unresolved or a final check
fails, the fit follows the usual failure handling and withholds standard
errors.

## IOLS reports a slope below its scale floor

If IOLS reports that a slope SD fell below its normalization floor, inspect the recorded scales in `result.provenance.failure` with `on_failure="return"`. Rescaling that predictor or using Newton can preserve the fitted model. [IOLS scale diagnostics](../reference/iols_options.md#when-slope-scaling-stops-the-solve)

## Standard errors are withheld

With `on_failure="return"`, inspect the status, summary and numerical
assessment. A returned last iterate with a failed assessment has no
reportable uncertainty. A successful fit can still have a degenerate or
undefined entry for a particular inference request. Look at the relevant
row in `tidy()` and the [covariance conditions](../methods/inference.md).

`CoverageWarning` refers to the theory supporting the selected numerical
route. `NumericalWarning` and `CovarianceWarning` describe the fitted
calculation and inference diagnostics. Preserve those messages along
with the result when reporting a problem.

## Results differ from another package

First match the outcome, actual fitted rows, slopes, effect blocks and
estimator. Then match the clustering, small-sample factors, reference
distribution and final numerical accuracy. [Compare with ppmlhdfe](../from-ppmlhdfe.md)

If the issue persists, keep both calls and use a small example someone else
can run. [Report a problem](report-a-problem.md)

<!-- guide-navigation:start -->
[Previous: IOLS installation](../iols_installation.md) · [Guide index](../README.md) · [Next: Report a problem](report-a-problem.md)
<!-- guide-navigation:end -->
