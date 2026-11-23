# Choose a solver and read its assessment

The mean model specifies the target and the working variance specifies the
estimating equation. The solver computes a point satisfying that equation.
For instance, the default solver in the gravity example computes joint
PPML with three absorbed blocks.

## Which calculation runs?

Nonlinear calls default to Newton. Joint and cancelled FE fits use their
corresponding Newton procedures, with the profiled organization as the
default. Order zero uses least squares on the log outcome. Keep solver
settings separate from the order, working variance and cancellation choice.

For nonzero order and a power `tau` outside `[1, 2]`, joint and no-FE
calls compute a selected root. Unless you supply `start`, the procedure
starts from a certified fit of the nearest convex family member on the same
sample. The available root need not be unique, and some combinations have
narrower existence results. Cancellation uses its own starting rules.
If continuation runs, earlier certified stages supply starting values. The
reported result and its certification always use the requested `tau`.
[Power-variance range](../reference/options.md#full-family-scope-and-the-current-entry-restriction)

## What the convergence assessment checks

The `certificate` assesses the computed point with diagnostics appropriate
to the equation, which can include score residuals, index corrections and
fixed-effect equations. Inspect the status if the achieved tolerance
differs from the request. When the assessment fails, inference is withheld.
Use `on_failure="return"` to inspect the returned state.
[Solver options](../reference/solver-options.md) · [Troubleshooting](../help/troubleshooting.md)

The exact-score completion of Benatia et al. (2026) returns to the original estimating equation
after solving approximations. Under the conditions in its numerical
argument, the remaining numerical error is negligible at the statistical
scale. A numerical assessment at one sample shows how well that point
solves the equation. Establishing a valid population interpretation requires
the assumptions of the chosen model and estimator. [Benatia et al. (2026), §4 and Supplemental Appendix S2](../references.md)

## IOLS: an optional no-FE solver

Iterated ordinary least squares uses shifted-log approximations and repeated
least-squares steps to compute a specified power-score equation. The shifts
scale with the outcome and fitted mean. A fixed shift before one log
regression solves a different problem. Powermean offers `solver="iols"`
for no-FE fits with positive order, a power `tau` in `[1, 2]`, an intercept,
at least one other regressor and more observations than coefficients. The
route excludes NB2, weights, offsets, exposure and a supplied start.

The [prepared-fit example](../examples/reuse-a-fit.md) solves one equation
by both Newton and IOLS. IOLS can use a compiled loop or a NumPy implementation.
The result records the implementation. To inspect or request one, see
[IOLS installation](../iols_installation.md) and [IOLS controls](../reference/iols_options.md).

<!-- guide-navigation:start -->
[Previous: Fit options](../reference/options.md) · [Guide index](../README.md) · [Next: Solver options](../reference/solver-options.md)
<!-- guide-navigation:end -->
