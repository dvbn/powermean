# Papers and methods

Benatia, David, Christophe Bellégo and Louis-Daniel Pape (2026).
*Dealing with Logs and Zeros in Regression Models*, arXiv:2203.11820v4,
16 November 2026. The supplemental references below use the ancillary PDF
from that same revision. [Paper](https://arxiv.org/abs/2203.11820v4) ·
[Supplemental appendices](https://arxiv.org/src/2203.11820v4/anc/supplemental_appendices.pdf) ·
[Replication materials](https://github.com/dvbn/bbp-replication)

| Topic | Sections in Benatia et al. (2026) |
| --- | --- |
| [Proportional effects](methods/proportional-effects.md) | §2 and Supplemental Appendix S1.1, including covariance identity (S1.2) and positive-outcome decomposition (S1.4). |
| [Mean order and working variance](methods/model.md) | §§3.1–3.3, equation (3.4) and Propositions 2–3. |
| [IOLS computation](methods/numerical-methods.md) | §4 and Supplemental Appendix S2. See §3.2 for the estimating equation. |
| [Cancellation](methods/cancellation.md) | §5.1, Proposition 7, Supplemental Appendix S3.3 and S3.6.1. Equations (S3.18) and (S3.19) compare cancellation and joint profiling. |
| [Bias from retained effects](methods/bias-correction.md) | §5.2, Theorem 2 and Appendix C.2. Equation (5.7) is the empirical group correction. |
| [Delete-group and higher-order covariance](methods/covariance-refinements.md#delete-group-and-higher-order-covariance) | Supplemental Appendix S3.4.2, Proposition S12, and S3.4.3, Proposition S13. |

## The separate three-way PPML correction

Weidner, Martin and Thomas Zylkin (2021). *Bias and consistency in three-way
gravity models*. Journal of International Economics 132, 103513. Section 3
sets out the fixed-$T$ model and the asymptotic result. Sections 3.4.2 and 3.4.3
give the analytical coefficient correction and pair covariance adjustment.
Powermean's WZ calls use those analytical procedures.
[Published article](https://doi.org/10.1016/j.jinteco.2021.103513) ·
[Open-access copy](https://ora.ox.ac.uk/objects/uuid%3A318e5814-ce0b-4696-a619-1a3b173412af)

## Panel difference-in-differences

- Wooldridge, Jeffrey M. (1999). *Distribution-free estimation of some
  nonlinear panel data models*. Journal of Econometrics 90(1), 77–97.
  See the mean-based argument for conditional Poisson fixed effects.
  [Article](https://www.sciencedirect.com/science/article/pii/S0304407698000335)
- Wooldridge, Jeffrey M. (2023). *Simple approaches to nonlinear
  difference-in-differences with panel data*. The Econometrics Journal 26(3),
  C31–C66. Section 2 introduces the two-period case. Section 3 gives the
  staggered conditional index assumptions, pooled estimation and comparison
  with exponential unit FE. [Article](https://doi.org/10.1093/ectj/utad016)
- Wooldridge, Jeffrey M. (2025). *Two-way fixed effects, the two-way Mundlak
  regression, and difference-in-differences estimators*. Empirical Economics.
  The linear extended TWFE specification allows effects to differ across
  cohorts and post-treatment dates. [Article](https://doi.org/10.1007/s00181-025-02807-z)
- Rios-Avila, Fernando. Stata `jwdid` documentation and implementation.
  [Nonlinear models and postestimation](https://friosavila.github.io/app_metrics/app_metrics11.html) ·
  [Command source](https://github.com/friosavila/stpackages/blob/main/jwdid/jwdid.ado)

Record the exact Powermean build with your analysis. For using third-party
software, see the [Powermean license](../LICENSE),
[scikit-learn notice](../src/powermean/_vendor/SKLEARN_COPYING) and
[native extension notices](../licenses/NATIVE_NOTICE.md).

<!-- guide-navigation:start -->
[Previous: Validation](validation.md) · [Guide index](README.md) · [Next: Build the guide](BUILD_GUIDE.md)
<!-- guide-navigation:end -->
