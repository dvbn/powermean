# Advanced fixed-effect options

Routine fits use the formula and defaults from the [fixed effects chapter](../methods/fixed-effects.md).
The optional module-level control records in this reference are experimental.

Use the default settings for an ordinary fixed-effect call. Supply a
`solver_options` record to choose the organization or control sample
construction for the cases below.

## Formula and call shapes

Use `pm.fit` and specify the supported equation. Write FE blocks after `|`, for example
`"y ~ 0 + x | firm + year"`. The slope intercept is omitted here because
the group indicators already span a constant.

The calls below use a frame with columns `y`, `x`, `firm`, `year` and
`unit`. The first estimates both blocks jointly. The second cancels the
firm block, estimates year effects and uses the cancelled groups for its
default covariance.

```python
import powermean as pm

joint = pm.fit(
    "y ~ 0 + x | firm + year", data,
    omega=1, tau=1, cluster="unit",
)
cancelled = pm.fit(
    "y ~ 0 + x | firm + year", data,
    omega=1, tau=1, cancel="firm",
)
```

At `omega=1, tau=1`, joint and cancelled PPML have the same uncorrected
slopes on the same retained sample. Each positive-total firm's fitted
values sum to its observed total. This equation determines its intercept
at each Newton step. When the admitted fixed effects permit that calculation,
the default zero-start cancelled fit uses the joint solver to perform it.
The solver trace records the steps it actually took. The cancelled fit
checks its own score and reports the requested covariance and bias correction.

To select an organization explicitly, pass `solver_options={"absorbed": "index"}`
or `solver_options={"absorbed": "profiled"}` to `pm.fit`. Omitting it
uses `"profiled"`. Within that mapping, a joint call can supply an
already constructed `SampleRequest` under `"sample_request"`. A cancelled
call can supply a `CancelledFitControls` record under `"controls"`.
The [resolver](../../src/powermean/interface/fe_options.py) describes
these experimental module-level records.

Any one block can be cancelled by name. Use `cancel=True` to select the
unique block with the most levels. A cancelled `pm.fit` call that omits
both `tau` and `variance` uses `tau=2`, so state the working variance
explicitly when comparing calls.

## Joint sample controls

A joint fit first constructs the sample appropriate to its mean order and
working variance. [`SampleRequest`](../../src/powermean/interface/sample_records.py) requires
all eight fields:

| Field | Requirement |
| --- | --- |
| `omega` | Current mean order; must match the fit. |
| `positive_only` | Current prepared outcome domain; must match the fit. |
| `variance` | Matching `PowerVariance` or `NB2Variance` at nonzero order; `None` at order zero. |
| `keep_singletons` | Explicit Boolean governing that sample step. |
| `support` | Actual `SupportControls` for power `1 <= tau < 2` or NB2; otherwise `None`. |
| `recession` | Actual `LPControls` for power `tau=2`; otherwise `None`. |
| `collinearity` | Actual diagnostic controls for the FE sample. |
| `mode` | `"drop"` or `"error"` for collinearity. |

For power `tau < 1` or `tau > 2` at a nonzero order, both existence controls
are `None`: sample construction runs rank checks only, without an existence
analysis.

These records govern ordered sample construction. They are not saved success
flags for another sample, order or variance. Inspect the actual sample
accounting. Keeping singletons does not disable subsequent identification
or existence checks.

## Cancellation controls

A cancelled fit has its own geometry and numerical work controls. To override
the defaults, supply a
[`CancelledFitControls`](../../src/powermean/interface/cancelled_selection.py)
record with ten fields:

| Field | Role |
| --- | --- |
| `gram_rtol` | Geometry rank threshold. |
| `gap_min` | Required geometry spectral gap. |
| `geometry_bytes` | Geometry allocation budget. |
| `admission_bytes` | Admission allocation budget. |
| `identity_atol` | Numerical identity comparison threshold. |
| `attempt` | `AttemptBounds` for the numerical attempt. |
| `certificate` | `AttemptBounds` for its assessment. |
| `inference` | `AttemptBounds` for inference work. |
| `diagnostic` | Collinearity/absorption diagnostic controls. |
| `collinear` | Explicit collinearity policy. |

`AttemptBounds(max_bytes=..., max_work=...)` requires positive Python integers.
Its byte limit reserves specified numerical buffers; it is not a hard limit
on process memory. Provider workspace, Python objects and other storage can
lie outside that reservation. See the
[constructor](../../src/powermean/solve/cancelled_operator.py).

Choose custom controls for the scale and purpose of your analysis.
A large budget does not resolve a failure of identification.

## Other numerical records

After those input requirements, the requested variance determines the
computational procedure. Cancellation at `omega=0` is refused; log OLS uses its separate additive
formulation. Execution for a supported cancelled model is recorded in
`result.provenance.options["route"]`: `"convex_tau2"` for power variance
`tau=2`, `"absorbed_tau1"` for `tau=1` or NB2 with zero `kappa`,
and `"root_newton"` otherwise. These are computational labels, not different
estimands to choose after a failure. The `absorbed_options` affect the
absorbed tau-one calculation; other branches use their own recorded controls.

`solver_options` can also contain the ordinary solver options record,
`absorbed_options` and `root_options`, a `RootOptions` record or its mapping.
Equivalent flat and nested explicit settings must agree. A cancelled call may
use `root_options`. A joint call reads it only for a power `tau` outside
`[1, 2]` at a nonzero order and refuses it otherwise; flat root keys are
unknown to FE calls. A joint call refuses cancellation controls, and a
cancelled call refuses `sample_request`.
A joint call may add `max_dense_bytes`, a positive integer ceiling for the
WZ numeric buffers of the fit and of its later covariance requests; a
cancelled call refuses it. FE fits use Newton; `solver="iols"` is refused.

On that selected-root route, a joint fit starts from a certified `tau=1` fit
(below 1) or `tau=2` fit (above 2) of the same sample, at the requested mean
order and computed by the requested organization, unless `start` is supplied.
At `omega=1` these are PPML and GPML. It then runs sparse root Newton on
the full joint equation. Its settings sit beside `"absorbed"` and
`"sample_request"`, for example `"root_options": {"merit_metric": "gram"}`.
It issues a `CoverageWarning` where the root is not characterized.
`result.provenance.options` records `operation` and the resolved
`root_options`, and `result.provenance.trace` records the initializer and the
stage path.

The [root record](../../src/powermean/solve/cancelled_records.py), shared by
cancellation and the selected-root routes, has `root_armijo=1e-4`,
`root_max_backtracks=60`, `root_max_iter=200`, `merit_metric="gamma"`,
`continuation="fallback"` and `continuation_step=0.5`.
These specify a root procedure, not permission to change the estimator after
failure. This table does not claim broad scientific validation of every
continuation path.

## Coefficient corrections

`bias_correction` requests a coefficient correction for an FE fit; ordinary
no-FE calls accept only `False`. A correction is evaluated once at the
certified uncorrected root, without refitting. A domain that is not
established is refused rather than replaced by a zero correction.

| Request | Fit | Requirements |
| --- | --- | --- |
| `False` (default) | Any | No correction work runs. |
| `True` or `"short_group"` | Cancelled | Short-group correction of Benatia et al. (2026), with cancelled groups as the sampling units. |
| `"cell"` | Cancelled | Cell correction of Benatia et al. (2026). Without `vcov`, the sampling units are the cancelled groups; a declared one-way classification that consists of complete cancelled groups is the sampling unit. |
| `"wz"` | Joint or cancelled | Weidner and Zylkin's correction for builtin PPML, `omega=1` and `tau=1`, with observed bread and no weights. |

Powermean supplies the coefficient corrections of Benatia et al. (2026) for cancelled fits. A `vcov` request that declares no sampling of the correction
(robust, named HC, several dimensions, a classification that crosses the
cancelled groups, or one coarser than them beside `True` or `"short_group"`)
is answered at the fit as `with_vcov` answers it after the fit: the correction
at the cancelled groups and the requested covariance at the uncorrected root,
with the same estimates, covariance and diagnostic notes. The
corrections use the cancelled fit's working variance and their own numerical
checks.

For three-way PPML, the WZ correction needs an identifiable directed pair block.
Name the pair cluster explicitly if the formula leaves its role ambiguous.
The correction requires the same set of at least three countries on the origin
and destination sides and at least two dates. Each retained directed pair
must appear once at every observed date. Whole pairs may be absent, and
diagonal pairs may be present or absent.

A joint fit must keep its original unconditioned sample: no positive-only
restriction, conditioning or recorded sample-construction change. A cancelled
fit must be unconditioned, cancel that pair block and keep every slope identified.
Without a `vcov` request, the covariance becomes the pair cluster with
`adjustment="wz"`. An explicit request must cluster on the pair; adding
`"adjustment": "none"` keeps the ordinary pair covariance.

`coef()`, the `estimate` column of `tidy()`, effects, contrasts and tests use
the corrected coefficients. `tidy()` adds `uncorrected` and `bias`, and
`summary()` prints the decomposition. The certificate refers to the
uncorrected root; `se()`, `vcov()` and intervals use the first-order
covariance evaluated there. `provenance.options["bias_correction"]` records
the correction. If the root is not certified, the coefficients and the
correction are withheld.

After a WZ correction, `fixef()` and `predict()` are
outputs of the original uncorrected root; corrected slopes are not combined
with original effects. Their `attrs` record
`evaluation_point="original uncorrected estimating root"` and
`nuisance_correction="not_supplied"`.
After a short-group or cell correction, cancelled predictions and effects
use the corrected reporting point.

The WZ covariance adjustment is a separate request:
`vcov={"cluster": pair, "adjustment": "wz"}`, where `pair` names the pair
block, at fit time or through `vcov(spec)` and `with_vcov(spec)`. At fit
time it requires builtin PPML, observed bread and a named `ssc` preset; with
cancellation, `cancel` must name that pair block. The returned
`vcov().attrs` record `adjustment="wz"`, `coefficient_correction=False` and
`covariance_point="uncorrected root"`.

## Current inference and result limits

Joint and cancelled calls accept robust, named HC (`"HC0"` to `"HC5"` and
`"HC4m"`) and one-way or multiway cluster requests, subject to the method's
requirements. Cancelled calls use observed bread. A named HC request on a
cancelled call is evaluated once at the accepted point, with or without
`bias_correction`.

For example, a cancelled fit can use
`vcov={"cluster": "country + family", "multiway": "cgm", "psd": "none"}`
with `ssc="none"` for the unadjusted signed two-way covariance. To request
the same covariance after a fit, retain the cluster labels and the influence
contributions. `influence(unit="row")` returns the original-row contributions
in estimation-sample order. The default `influence()` returns one contribution
per cancelled group.

Fixed-effect calls refuse supplied observation weights, even if all values
equal one. HAC and Conley computation are deferred.

Joint `fixef()` returns estimated normalized effects. Cancelled `fixef()`
returns the retained blocks and, under the cancelled block's name, effects
recovered in sample at the stored point; [results](results.md#recovered-cancelled-effects)
describes their status labels and how they are used in prediction.
A failed fit retains its actual status and diagnostics.

Preparation remains order neutral. Each fit still needs controls consistent
with its requested order, family and formulation. See
[cancellation](../methods/cancellation.md), [results](results.md) and
[availability](../availability.md).

If a request is refused, [Troubleshooting](../help/troubleshooting.md)
explains what to inspect before changing it.

<!-- guide-navigation:start -->
[Previous: Solver options](solver-options.md) · [Guide index](../README.md) · [Next: IOLS options](iols_options.md)
<!-- guide-navigation:end -->
