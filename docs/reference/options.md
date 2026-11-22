# Options

Use this page to translate a statistical choice into an exact argument.
Start with the function that matches the mean and estimating equation, supply
the sample decisions, then choose uncertainty and numerical controls.
The [walkthrough](../get-started.md) shows those choices in a complete analysis.

Fitting and preparation take a formula, a pandas DataFrame and named
keywords. Unknown or conflicting requests raise `SpecificationError`.
For reproducible results, record the installed version and the wheel or source archive used. [Function index](api.md)

| Function | Purpose |
| --- | --- |
| `fit(formula, data, **options)` | Choose mean order and working variance, or select a two-way difference estimator. |
| `ppml(formula, data, **options)` | Fix `omega=1`, `tau=1`. |
| `gpml(formula, data, **options)` | Fix `omega=1`, `tau=2`. |
| `logols(formula, data, **options)` | Fix `omega=0`; require positive outcomes. |
| `prepare(formula, data, **options)` | Capture a problem for subsequent `prepared.fit()` calls. |
| `solver_defaults(solver="newton")` | Inspect typed defaults: `SolverOptions` for Newton, the IOLS record for `"iols"`. |
| `NB2(kappa=1.0)` | Working variance `m + kappa * m**2` with a fixed, finite `kappa >= 0`. |
| `ssc(*, robust, cluster, cluster_count, reference, psd, absorbed_count="fixest")` | Declare a custom small-sample convention. The first five keywords are required. The record supports `to_dict()` and `type(record).from_dict(...)`. See [conventions](#small-sample-conventions). |

## Sample arguments

These arguments determine which observations and known scales enter the
equation. They belong to preparation when you use a prepared problem.

| Argument | Default | Meaning |
| --- | --- | --- |
| `positive_only` | `False` | Explicitly restrict to positive outcomes. |
| `weights` | Omitted | Name of the weight column. |
| `weight_type` | `"analytic"` | `"analytic"`, `"frequency"` or `"probability"`, with `weights`. |
| `offset` | Omitted | A column or numeric expression on the log power mean scale. |
| `exposure` | Omitted | A positive column or expression whose logarithm is the offset. |
| `subset` | Omitted | Boolean condition per row, such as `data["x"] > 0`; missing conditions exclude rows. |
| `retain` | Omitted | Experimental FE metadata columns for supported later requests; not an ordinary no-FE activation. |

Choose either offset or exposure. Supply sample arguments to `prepare` rather
than `prepared.fit`. A fit cannot override the prepared outcome domain.

## Fit and inference arguments

Once the sample is set, choose the mean order and working variance.
The covariance describes sampling uncertainty; the solver controls how the
equation is computed.

| Argument | Default | Meaning |
| --- | --- | --- |
| `omega` | `1` | Any finite real mean order; nonpositive orders require positive outcomes. |
| `tau` | `1`, or `2` for a cancelled `pm.fit` | User-specified power variance index. Any finite value with Newton at a nonzero `omega`; outside `[1, 2]` the estimate is a selected root. IOLS and `omega=0` accept `[1, 2]` only; see the [range note](#full-family-scope-and-the-current-entry-restriction). Cannot accompany `variance`. |
| `variance` | Omitted | A variance object, including `pm.NB2(kappa)`; NB2 is refused with IOLS. |
| `vcov` | `"robust"` for no-FE fits; see [FE defaults below](#fit-time-covariance-for-fixed-effects) | `"robust"`; a named HC method, `"HC0"` to `"HC5"` or `"HC4m"`, never beside a cluster; or a cluster mapping: `{"cluster": "unit"}` for one classification, `{"cluster": "a + b"}` for several dimensions, with optional `"multiway"` and `"psd"` keys. |
| `cluster` | Omitted | Cluster shorthand; must agree with `vcov` if both appear. |
| `ssc` | Omitted | The covariance convention, given as `"stata"`, `"none"`, `"regression"` or a `pm.ssc(...)` record. If omitted, the default `"stata"` or the prepared request's own convention applies. At fit time, pass `ssc` as a keyword alongside `vcov` or `cluster`. |
| `bread` | `"observed"` | Without fixed effects, `"expected"` is accepted only beside a named HC method; robust and cluster requests use the observed bread. |
| `solver` | `"newton"` | `"newton"`, or the optional no-FE `"iols"`; order zero uses a dedicated linear calculation and refuses IOLS. |
| `solver_options` | Canonical defaults | A mapping or `SolverOptions` record; inspect `pm.solver_defaults()`. With IOLS, a mapping or the record from `pm.solver_defaults("iols")`. A no-FE Newton fit with `tau` outside `[1, 2]` also reads the root-Newton keys of the [range note](#full-family-scope-and-the-current-entry-restriction). |
| `tol` | Canonical default | Shorthand for the numerical `tol`; contradictory explicit values are refused. |
| `start` | Automatic | One finite value per retained coefficient on the reported scale; omit entries for dropped regressors. Refused with IOLS. |
| `on_failure` | `"raise"` | `"return"` permits inspection with inference withheld. |
| `keep_influence` | `True` | Retain contributions needed for subsequent requests. |

Prepared fits use only clustering information retained during preparation.
Convenience wrappers refuse contradictory mean or variance choices.
The full numerical defaults and domains are in [solver options](solver-options.md),
including the [IOLS options](solver-options.md#iols-options).

### Two-way difference options

Select `estimator="gmm1"` or `"gmm2"` in `pm.fit` and use an intercept-free
formula with exactly two fixed-effect identifiers. These fits accept the
following options and require the input to contain every cell of the chosen
geometry. Filter the data frame first to fit on a smaller complete array.
[Two-way difference method](../methods/jochmans.md)

| Argument | Default | Meaning |
| --- | --- | --- |
| `estimator` | Required for this route | `"gmm1"` or `"gmm2"` selects the Jochmans moment equation. |
| `fe_layout` | `"rectangular"` | A complete crossing of two dimensions, or `"directed"` for all ordered non-self links among common agents. |
| `vcov` | `"paper"` | Observed-cell normalization, or `"stata"` to match the directed Stata normalization. |
| `start` | Zero slopes | Finite starting slopes in formula design order. |
| `max_iter` | `100` | Maximum number of accepted Newton updates. |

### Full-family scope and the current entry restriction

The power-score family is not limited to the PPML-to-GPML interval. With
Newton at a nonzero `omega`, an ordinary no-FE fit and a joint fit in the
[fixed-effect options](#fixed-effect-options) accept any finite `tau`. Inside
`[1, 2]` their convex Newton route is unchanged. Outside it, the estimate is
a selected root of the specified equation. Unless `start` is supplied, root
Newton starts from the certified fit of the nearest convex member on the same
sample, at the requested mean order: `tau=1` for `tau < 1`, `tau=2` for
`tau > 2` (PPML and GPML at `omega=1`). If that fit is not
certified, no root step runs and the result is not certified. A supplied
`start` is used as given and recorded; it is not certified first, and no
continuation path leaves from it. The certificate establishes a regular root;
the root is not claimed to be unique or global.

Existence of a finite root is not characterized for `tau < 1`, or for
`tau > 2` with zero outcomes. Such a fit issues a `CoverageWarning`, drops no
row for it and leaves the decision to the solver and the certificate.
A wider family does not guarantee a finite root, identification or a
certified fit for every user-specified value.

A no-FE fit outside `[1, 2]` reads six root-Newton keys from the same
`solver_options` mapping: `root_armijo`, `root_max_backtracks`,
`root_max_iter`, `merit_metric` (`"gamma"` or `"gram"`), `continuation`
(`"fallback"` or `"never"`) and `continuation_step`, as in
`solver_options={"merit_metric": "gram"}`. Omitted keys, or a
`SolverOptions` record, keep their defaults; other fits refuse these keys as
unknown. A joint fit takes them in its nested `root_options` record instead;
see [advanced FE options](experimental-fixed-effects.md#other-numerical-records).
The resolved root options are recorded with the fit's other numerical
settings.

IOLS still accepts only `[1, 2]`, and order zero still refuses powers outside
that interval. Cancellation resolves any finite `tau`, or an NB2
variance, through its own routes and starts; the nearest convex start above
applies to the no-FE and joint FE routes only.

### Small-sample conventions

Use `ssc` to choose the convention for a robust, cluster or named HC
request. Pass a preset (`"stata"`, `"none"` or `"regression"`) or an immutable
record from `pm.ssc`. All six parameters are keyword-only. The first five
are required. A value outside the table raises `SpecificationError`.
The convention takes effect when a covariance request uses it.

| Keyword | Accepted values | Role |
| --- | --- | --- |
| `robust` | `"none"`, `"N/(N-1)"`, `"N/(N-K*)"` | Factor of a robust request. |
| `cluster` | `"none"`, `"G/(G-1)"`, `"(N-1)/(N-K*)*G/(G-1)"` | Factor of a cluster request. |
| `cluster_count` | `"min"`, `"each"` | One factor with the smallest cluster count of a multiway request, or each component with its own count. |
| `reference` | `"normal"`, `"t"` | Reference distribution of statistics, p-values and intervals. |
| `psd` | `"cgm"`, `"fixest"`, `"none"` | Treatment of an indefinite signed multiway estimate: `reghdfe`'s and `ppmlhdfe`'s eigenvalue adjustment in standardized coordinates, `fixest`'s rule (eigenvalues below `1e-12` set to `1e-12` on the reported matrix), or none. |
| `absorbed_count` | `"fixest"` (default), `"exact"` | Count of absorbed effects for fixed-effect log OLS when the covariance factor or reference uses that count. |

Here `K*` is the count `K_factor` used by the factor. With fixed effects it is
the number of retained regressors plus the count of the absorbed parameters.
For fixed-effect log OLS (`omega=0`), `"fixest"` counts levels less one per
counted block after the first. `"exact"` uses the rank of the counted
indicators. For clustered log OLS, both rules exclude blocks nested in a
requested cluster dimension and subtract one from the count if any block
is nested and any counted block remains. HC1 uses the chosen count for its
factor. The default robust convention uses `N/(N-1)` and a normal reference.
Other joint fits use the indicator rank with the cluster nesting adjustment.
Only fixed-effect log OLS supports `absorbed_count="exact"` among FE fits.
For the named HC methods of cancelled fits, redundant levels are removed
exactly for one and two blocks and by a pairwise bound, taken in the order
of the formula, from the third block on.
[Covariance metadata](results.md)

A robust or cluster request applies its convention's factor once and uses
its reference. A signed multiway request without its own `"psd"` key also
uses its `psd` treatment.
A named HC method keeps its own factor and takes the convention's reference.
An omitted or inherited convention never adds its factor to that method.
An explicit convention with robust factor `"none"` adds nothing; an explicit
factor identical to the method's own, by formula and exact counts, is that
same correction applied once. Any other explicit factor is refused as a
second correction. For robust or cluster requests, an explicit `"stata"`
keeps the default factors and is recorded as requested.

For a robust fit without fixed effects, use the `data` frame from the
[worked no-FE example](../examples/reuse-a-fit.md#solve-the-same-mean-equation-twice).

```python
import powermean as pm

convention = pm.ssc(
    robust="N/(N-1)",
    cluster="G/(G-1)",
    cluster_count="min",
    reference="normal",
    psd="cgm",
)
fit = pm.fit("y ~ 1 + x1 + x2", data, vcov="robust", ssc=convention)
```

When a fit of a prepared problem gives `vcov` or `cluster`, that request and
its convention replace the prepared ones; `ssc` given alone applies to the
prepared request, as does `bread` to a prepared named HC method. After a fit
without fixed effects, `vcov(spec, ssc=...)` and `with_vcov(spec, ssc=...)`
use the request's explicit convention or inherit the fit's. Both reuse
the fitted point. See the [fixed-effect options](#fixed-effect-options)
for FE requests.

## Fixed-effect options

Write the effect blocks after `|` in the formula. To cancel one block, name it
with `cancel="block"`, or use `cancel=True` to select the unique largest block.
Ordinary fixed-effect calls use the package's sample and solver defaults.
Within `solver_options`, choose `absorbed="index"` to use the index organization
instead of the default `"profiled"`. Preparation remains order-neutral.
[Advanced FE options](experimental-fixed-effects.md) explains the optional
control records and their supported domains.

FE fits use Newton; `solver="iols"` is refused. Through `pm.fit`, a cancelled
call that omits both `tau` and `variance` uses power `tau=2`, not the
ordinary default `tau=1`.

### Fit-time covariance for fixed effects

For an ordinary joint fit with no covariance request, Powermean clusters
on the pair if the three retained fixed-effect blocks identify a unique
pair, origin-by-date and destination-by-date structure. For example, the
pair block can be `pair` in `pair + origin^year + destination^year`, provided
it groups the same endpoint pairs on the retained sample. If no such
structure is established, the ordinary default remains `"robust"`. If more
than one assignment is valid, specify the intended cluster explicitly,
for example `cluster="pair"` for a block named `pair`.

An explicit covariance request, including `vcov="robust"`, takes precedence
on this route. Omitting covariance does not request a coefficient correction
or the WZ covariance adjustment. WZ-corrected fits and cancelled fits use
their own covariance rules.

At fit time, joint fits accept robust, `"HC0"` to `"HC5"`, `"HC4m"` and
one-way or multiway cluster requests, subject to the requirements of each
covariance. Supply `ssc=` and `bread=` as fit keywords. Cancelled fits
also accept robust, named HC and one-way or multiway cluster requests
within their domains.
They use observed bread. FE fits also accept `bias_correction` (`False`, `True`,
`"short_group"`, `"cell"` or `"wz"`) within the
[correction domains](experimental-fixed-effects.md#coefficient-corrections).
Fixed-effect fits use unit multipliers. Supplied observation weights are unavailable.
Unsupported families, domains and inference requests are refused; see the
package API documentation and [availability](../availability.md).

### Post-fit covariance for fixed effects

Use `fit.vcov()` to read the active covariance. For a supported alternate
request, `fit.vcov(spec)` returns a matrix and `fit.with_vcov(spec)` returns
a result view. Both require an accepted fit and the information needed
for the request. For clustering, the classifications must be available
from the fit. [Recompute uncertainty](results.md#recompute-uncertainty-from-stored-information)

FE results accept `ssc` and `bread` overrides as keys inside the request
mapping, subject to the covariance's domain. For example, with a joint
FE fit and the convention above, use
`fit.vcov({"method": "robust", "ssc": convention})`. Pass any overrides
through `spec` for both methods.

## Covariance adjustment key

Ordinary no-FE calls refuse the `vcov` mapping key `adjustment`. In FE
requests, `{"cluster": pair, "adjustment": "wz"}` asks for Weidner and
Zylkin's three-way PPML covariance adjustment, at fit time or through
`vcov(spec)`, within its
[documented domain](experimental-fixed-effects.md#coefficient-corrections).
Under `bias_correction="wz"`, an explicit `"adjustment": "none"` keeps the
ordinary pair-cluster covariance. The key does not activate either covariance refinement of Benatia et al. (2026). See [covariance refinements](../methods/covariance-refinements.md)
for the statistical distinctions.

## Deferred HAC and Conley requests

HAC and Conley covariance are unavailable. The reserved FE request
forms below result in an unavailable or invalid request diagnostic.

| Request | Reserved mapping keys | Typed record |
| --- | --- | --- |
| HAC | `{"hac": time, "unit": unit, "lag": lag, "step": step}` | `HacRequest(time, unit=..., lag=..., step=...)` |
| Conley | `{"conley": [latitude, longitude], "cutoff": cutoff, "units": units, "distance": distance, "pool_times": True}` | `ConleyRequest(latitude, longitude, cutoff=..., units=..., distance=..., pool_times=...)` |

The records are defined in
[`powermean.data.inference_requests`](../../src/powermean/data/inference_requests.py),
not as root exports `pm.HacRequest` or `pm.ConleyRequest`.
The variable names in the table stand for column names and explicit settings;
optional fields may be omitted. `None` is a supplied value, not omission.

For HAC, `time` and an explicitly supplied `unit` are column names. `lag` is
a nonnegative integer and `step` is a positive integer. Omitted lag represents
the reserved automatic-lag request; the record's omitted step represents one
period. No lag rule, time grid validation or covariance calculation runs here.

For Conley, the two column names are latitude then longitude. `cutoff` is
a finite nonnegative number, including zero. Explicit `units` is `"km"` or
`"mi"` and requires an explicit cutoff. `distance` is `"triangular"` or
`"spherical"`. `pool_times`, when supplied, accepts only `True`.
Reserved omission metadata represents kilometers and triangular distance,
with an automatic cutoff request if cutoff is omitted. This does not validate
coordinates or repeated locations, choose a numerical cutoff or compute a
spatial covariance. It does not substitute another covariance or refit.

## Reserved requests

Automatic working-variance selection (targeted IOLS) is unavailable.
The `tau` and `variance` arguments choose a working variance you provide;
neither requests automatic targeting, and `solver="iols"` computes the
equation you specified.

Bias correction, difference moments, instruments and public thread control
remain unavailable through ordinary no-FE calls, as does expected bread beside
a robust or cluster request. `advise` and `gmm` are also reserved rather than
active features. Recognized argument names can produce an informative
unsupported-feature error without activating the feature.

Inactive `cancel=None`, `difference=False`, `bias_correction=False` and
`moment="multiplicative"` are accepted, as is `bread="observed"` beside a
robust or cluster request. Without fixed effects, `ssc` always belongs to the
covariance request; see [small-sample conventions](#small-sample-conventions).
See [availability](../availability.md).

For computational controls, continue to [Solver options](solver-options.md).
For inspecting a returned fit, use [Results](results.md).

<!-- guide-navigation:start -->
[Previous: Supported methods](../availability.md) · [Guide index](../README.md) · [Next: Results](results.md)
<!-- guide-navigation:end -->
