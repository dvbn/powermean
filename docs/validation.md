# Read validation and performance evidence

After interpreting a result, ask what supports the claim you want to make.
A small equation residual supports numerical accuracy at a computed point.
A simulation study addresses sampling behavior in its specified design.
A benchmark measures a task on a particular system. Each answers a different
question.

Use this page to read model diagnostics, compare results and time a complete
analysis.

## Begin with the claim

| Question | Relevant evidence | Limit of that evidence |
| --- | --- | --- |
| Does a calculation match the intended formula? | Independent derivation and direct calculation, including full covariance and SE. | Agreement on tested cases does not establish public integration or repeated-sample behavior. |
| Does the software detect selected invalid inputs or wrong calculations? | Perturbations and invalid-input checks. | Passing those checks does not rule out every possible defect. |
| Do two packages agree? | Reference comparison with matched data, model, covariance and numerical accuracy. | Agreement does not transfer to unmatched conventions or other designs. |
| How does an estimator behave across samples? | A specified Monte Carlo study. | Its bias and coverage findings are specific to the design. |
| How long does the analysis take? | A complete fit benchmark with stated hardware, threads and timing boundaries. | Runtime does not establish correct inference or a universal speed ranking. |
| Does installation work in an environment? | Installation and platform checks. | A tested environment does not validate every future dependency combination. |

## What the guide currently demonstrates

The [gravity example](get-started.md) illustrates a coefficient, pair-clustered
standard error and interval on one simulated PPML sample. Use
[its script](../examples/first_fit.py) to reproduce the calculation. The
[IOLS example](iols_installation.md) shows how to check whether the installed
package used the compiled or NumPy implementation.

## Read statistical summaries with their failures

To interpret a repeated experiment, identify its target, data-generating
process, sample sizes, independent units, repetitions and each method's
assumptions. Read bias, dispersion, reported standard errors and interval
coverage together.

A summary restricted to successful fits must state that denominator and also
report the overall failure rate and reasons. Replacing failed estimates by
zero or silently dropping them changes the evidence. A larger sample alone
also does not validate a covariance formula: short panels and growing numbers
of effects require the relevant information and dependence conditions.

## Compare runtime on matched tasks

For PPML, use `omega=1`, `tau=1` in Powermean and match the other package's
rows, regressors, effects and uncertainty calculation. State elapsed time,
threads, repetitions and the timing boundary. A complete fit includes
preparation, solving, final numerical assessment, covariance and storing the
result. Compare cold fits with cold fits and prepared fits with prepared fits.
Report unsuccessful fits and their timings separately.

Use the full analysis elapsed time to compare user workflows. [Coming from
ppmlhdfe](from-ppmlhdfe.md) explains how to align the inputs and settings.

The [model reference](availability.md) describes the available requests.
To share a reproducible example, see [Report a problem](help/report-a-problem.md).

<!-- guide-navigation:start -->
[Previous: Report a problem](help/report-a-problem.md) · [Guide index](README.md) · [Next: Papers and methods](references.md)
<!-- guide-navigation:end -->
