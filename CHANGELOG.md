# Changelog

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
