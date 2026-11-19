# Solver options

Use solver controls after specifying the sample, model and estimating equation.
They govern computational accuracy and effort. They do not change which
population quantity the model targets or which observations can be dependent.
[Numerical methods](../methods/numerical-methods.md) explains the distinction.

The ordinary defaults below apply to Newton, the default solver. The
optional IOLS solver has its own record, described after the Newton table.
Inspect the installed version before changing a setting:

```python
import powermean as pm

options = pm.solver_defaults()
print({key: options.to_dict()[key] for key in ("tol", "max_iter", "curvature")})
iols_options = pm.solver_defaults("iols")
print({key: iols_options.to_dict()[key] for key in ("iols_backend", "stopcrit")})
```

```text
{'tol': 1e-08, 'max_iter': 100000, 'curvature': 'observed'}
{'iols_backend': 'cpp', 'stopcrit': 1000000}
```

## Ordinary Newton options

| Option | Default | Accepted values and role |
| --- | --- | --- |
| `tol` | `1e-8` | Positive finite real; strict numerical threshold. |
| `tol_secondary` | `1e-6` | Positive finite real; secondary numerical threshold. |
| `coef_tol` | `1e-6` | Positive finite real; coefficient accuracy diagnostic threshold. |
| `rcond_min` | `1e-13` | Strictly between zero and one; conditioning safeguard. |
| `newton_provider` | `"bundled"` | `"bundled"` or compatible installed `"sklearn"`. |
| `gradient_tol` | `1e-10` | Positive finite real; iteration gradient threshold. |
| `max_iter` | `100000` | Positive integer; iteration budget. |
| `max_evaluations` | `1000000` | Positive integer; evaluation budget. |
| `armijo` | `2**-11` | Strictly between zero and one; sufficient decrease setting. |
| `backtrack_factor` | `0.5` | Strictly between zero and one; step reduction factor. |
| `max_backtracks` | `21` | Positive integer; backtracking budget. |
| `retry` | `1` | Integer zero or one; the solver's specified retry control. |
| `curvature` | `"observed"` | `"observed"` or `"expected"` for the Newton system. |

The solver's `curvature` option is not the inference argument `bread`. Expected
curvature in an iteration does not activate expected bread covariance.
`retry=1` allows at most one retry. In an eligible profiled fixed-effect fit,
if the first attempt fails to find a numerical step, complete its line search
or finish the inner absorption, the solver may restart from the original
coefficient start, opening with all the fixed effects estimated together.
The retry uses the remaining budgets and the same sample and accuracy
requirements. If it does not certify, the original refusal stands. Set
`retry=0` to disable retries.

Pass a mapping through `solver_options` or a complete `SolverOptions` record.
Unknown keys are errors. Booleans are not numeric tolerances or budgets; a
floating point value is not accepted in place of an integer budget.

The top level `tol` keyword is shorthand for the nested `tol`. If both are
explicit, they must agree exactly. A complete `SolverOptions` record counts as
explicit values, including the fields equal to defaults.

## IOLS options

`solver="iols"` selects IOLS for a supported no-FE fit. Its `solver_options`
is a mapping or the record returned by `pm.solver_defaults("iols")`, not a
`SolverOptions` record. In a mapping, `tol`, `tol_secondary`, `coef_tol`
and `rcond_min` keep the certificate roles, domains and defaults of the table
above, and the top level `tol` follows the same agreement rule. Every other
key belongs to the IOLS record. The [complete IOLS table](iols_options.md)
lists all nine controls, defaults and constructor versus execution domains.
Print `pm.solver_defaults("iols").to_dict()` to inspect your installation.

The key `iols_backend` requests an implementation: `"cpp"` for the compiled
extension or `"numpy"`. Omit it for automatic selection. An explicit `"cpp"`
request is refused when the compiled extension is unavailable.
`result.provenance.trace` records `requested_backend`, `actual_backend` and
`fallback_used`; read them rather than assuming which implementation ran.

Passing the typed default record preserves omitted implementation selection.
Passing its complete `.to_dict()` mapping instead makes `"cpp"` explicit,
because that mapping contains `iols_backend`. On a NumPy-only installation,
use the typed record or explicitly request `"numpy"`. The
[installation guide](../iols_installation.md#select-the-implementation-deliberately)
explains this distinction.

IOLS refuses settings outside its no-FE scope, listed in
[numerical methods](../methods/numerical-methods.md#iols-an-optional-no-fe-solver).
It does not select `tau` and does not change the Newton default.

## Experimental absorption options

With fixed effects, the iteration also solves weighted within-group problems.
Fixed-effect fits use the additional controls below for that work. These are not
ordinary no-FE solver keywords.

| Option | Default | Role |
| --- | --- | --- |
| `absorbed` | `"profiled"` | Choose `"index"` to request the index organization explicitly. |
| `absorb_tol_start` | `1e-4` | Positive finite starting absorption tolerance. |
| `absorb_tol_final` | `2e-11` | Positive finite final absorption tolerance. |
| `absorb_max_sweeps` | `20000` | Positive integer sweep limit. |
| `absorb_acceleration` | `True` | Boolean acceleration choice. |
| `profile_tol` | `2e-11` | Positive finite profiling tolerance. |
| `profile_max_sweeps` | `20000` | Positive integer profiling sweep limit. |

Ordinary joint and cancelled fits use the package's default sample and numerical
controls. To customize them, use `SampleRequest` for a joint fit or
`CancelledFitControls` for a cancelled fit. The
[advanced FE options](experimental-fixed-effects.md) describes these records.
FE fits use Newton; `solver="iols"` is refused with fixed effects.

Larger budgets are not a remedy for nonidentification, separation or a wrong
model. A numerical threshold does not establish coefficient parity, coverage or
a universal error bound. Preserve the actual failure and review its cause.

Use [Troubleshooting](../help/troubleshooting.md) to distinguish an unavailable
request, a sample problem and an unsuccessful numerical solve.

<!-- guide-navigation:start -->
[Previous: Tables](tables.md) · [Guide index](../README.md) · [Next: Advanced FE options](experimental-fixed-effects.md)
<!-- guide-navigation:end -->
