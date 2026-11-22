# Changelog

## 0.1.1

- Predict fitted power means and response residuals for cancelled and two-way
  GMM fits, with ex-post fixed effects available through `predict()` or
  `fixef()`. Fixed-effect fits accept replacement covariates on retained rows.
- Fit Jochmans GMM1 and GMM2 for complete rectangular panels and directed
  dyads using the `estimator=` option of `powermean.fit`. Report slopes, full cell-projection
  covariance, and normal-reference intervals. Select the Stata `twgravity`
  covariance convention with `vcov="stata"`.
- Follow the two-way exponential regression guide and runnable Python example.
- Correct false certification and convergence refusals in cancelled Gamma,
  joint Gamma near the support boundary, and exact-fit cancelled PPML with
  Weidner–Zylkin inference. Improve convex absorbed fits and dense PPML with an
  exposure and frequency weights.
- Improve certification at strict tolerances in profiled interior-power
  fixed-effect fits.
- Stabilize the signed fixed-effect coefficient check when the reported
  standard error is zero. Correct working covariance rank when a finite matrix
  has an eigenvalue beyond float64 range.

## 0.1.0

- Fit power-mean regressions with multiple fixed-effect blocks or cancel
  one block. Report conditional proportional effects and clustered inference.
- Apply short-group, cell, or Weidner–Zylkin corrections in supported designs.
- Use the compiled no-fixed-effects IOLS solver and the fixed-effect basis accelerator.
- Follow the installed examples for gravity, nonlinear difference-in-differences,
  fixed effects, and reporting.
- Install native wheels on Linux, macOS, and Windows for x86-64 and ARM64,
  with CPython 3.12, 3.13, and 3.14. The macOS wheels target macOS 15 or later.

## 0.1.0a1

First no-fixed-effects alpha for source and CI testing. This is not a stable
release or a package-index upload.

- Regression models without fixed effects: `fit`, `ppml`, `gpml`, `logols`, and
  `prepare` for several fits on one prepared sample.
- Two solvers: Newton (default) and IOLS (`solver="iols"`), both in pure Python.
  There is no compiled IOLS kernel in this version: `iols_backend="cpp"` is refused.
- Covariances: robust, HC0 to HC5, one-way and multiway clustering, with the
  small-sample conventions of `ssc`.
- After estimation: `tidy`, `summary`, `effect`, `effects`, `contrast`, `lincom`,
  `nlcom`, `test`, `wald`, `predict`, `influence`.
- A formula with fixed effects (a part after `|`) is refused in this version. Fixed
  effects are planned for version 0.1.
